# 02_probe_b_module.py —— 探针 B：模块档实验预算（ch2-ch6 各插槽实验的定档底数）
# 用途：在 ~24M 模块档（d=512/L=6/h=8/V=8192，直接改装 Book2 ch10 底座）下，同一训练循环跑
#       四个变体各 150 步——GPT-2 基线 / RMSNorm 版（ch2）/ SwiGLU 版（ch3）/ GQA 版（ch6）——
#       记录各变体 sec/step 与 MPS 内存峰值，回答「一个 ≤30 分钟的模块实验 = 多少步」。
#       单插槽改装原则：每个变体只换一个插槽（RMSNorm 版仍用 GELU MLP + MHA + wpe；
#       GQA 版仍用 LayerNorm + GELU MLP + wpe——RoPE 不掺入，RoPE 是 ch4 的事）。
# 所属章节：Book3 00-feasibility（ch2/ch3/ch6 模块实验定档；ch10 整合短训预算参考）。
# 运行方式：cd code/00-feasibility && python 02_probe_b_module.py [--steps 150] [--out-name NAME]
#   语料：log/book2-ch08/tokens.bin（Book2 遗产：sp-8k 词表 8192 的 306M token 流，uint16 memmap 流式读取）
#   产物：log/book3-feasibility/{out-name}.json + {out-name}_curves.csv（不 commit）
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _base in (os.path.join(HERE, "..", "..", "Book2-预训练革命", "ch10"),                              # 工作区布局
              os.path.join(HERE, "..", "..", "..", "book2-pretraining-revolution", "code", "ch10")):  # 书仓快照布局
    if os.path.isdir(_base):
        sys.path.insert(0, os.path.abspath(_base))  # Book2 ch10 GPT-2 定版底座
from model import GPT2, GPT2Config                    # noqa: E402  Book2 ch10 定版底座（改装对象）
from llama_slots import RMSNorm, SwiGLU, LLaMAConfig, repeat_kv  # noqa: E402  本册插槽组件

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book3-feasibility")
TOKENS_BIN = os.path.join(REPO_ROOT, "log", "book2-ch08", "tokens.bin")   # 306M token，V=8192
SEED = 20261002                                                          # 全书统一种子


class GQASelfAttention(nn.Module):
    """GPT-2 底座上的 GQA 插槽替换（ch6 的单插槽改装版）：分离 q/k/v/o 投影 + repeat_kv。

    与 llama_slots.GroupedQueryAttention 的区别：这里不带 RoPE（位置编码仍是底座的 wpe 查表，
    单插槽改装一次只动一个零件）；投影无 bias（对齐 LLaMA 口径，参数账在注释里报差量）。
    输入 (B,n,C) -> 输出 (B,n,C)，接口与底座 CausalSelfAttention 完全一致（插槽可换性的现场证明）。
    """

    def __init__(self, C, n_head, n_kv_head):
        super().__init__()
        assert n_head % n_kv_head == 0
        self.n_head, self.n_kv_head = n_head, n_kv_head
        self.n_rep = n_head // n_kv_head
        self.hd = C // n_head
        self.q_proj = nn.Linear(C, n_head * self.hd, bias=False)      # (B,n,C) -> (B,n,h*hd)
        self.k_proj = nn.Linear(C, n_kv_head * self.hd, bias=False)   # (B,n,C) -> (B,n,h_kv*hd)
        self.v_proj = nn.Linear(C, n_kv_head * self.hd, bias=False)
        self.o_proj = nn.Linear(n_head * self.hd, C, bias=False)      # (B,n,h*hd) -> (B,n,C)
        self.mask = nn.Buffer(torch.tril(torch.ones(2048, 2048)).view(1, 1, 2048, 2048), persistent=False)

    def forward(self, x):
        B, n, C = x.shape                                              # (B,n,C)
        h, h_kv, hd = self.n_head, self.n_kv_head, self.hd
        q = self.q_proj(x).view(B, n, h, hd).transpose(1, 2)           # (B,h,n,hd)
        k = self.k_proj(x).view(B, n, h_kv, hd).transpose(1, 2)        # (B,h_kv,n,hd)
        v = self.v_proj(x).view(B, n, h_kv, hd).transpose(1, 2)        # (B,h_kv,n,hd)
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)      # -> (B,h,n,hd)（KV 头分组共享）
        att = (q @ k.transpose(-2, -1)) / (hd ** 0.5)                  # (B,h,n,n)
        att = att.masked_fill(self.mask[:, :, :n, :n] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, n, C)       # (B,n,C)
        return self.o_proj(y)


def make_variant(kind, cfg):
    """四变体工厂：同一底座实例上做单插槽替换（RMSNorm/SwiGLU 换模块对象，GQA 换 attn 对象）。"""
    torch.manual_seed(SEED)                        # 同种子：四变体初始权重同分布同数值（替换件除外）
    m = GPT2(cfg)
    C = cfg.n_embd
    if kind == "base":
        pass
    elif kind == "rmsnorm":                        # ch2：LayerNorm -> RMSNorm（ln_1/ln_2/ln_f 三处）
        for blk in m.blocks:
            blk.ln_1 = RMSNorm(C, eps=1e-6)
            blk.ln_2 = RMSNorm(C, eps=1e-6)
        m.ln_f = RMSNorm(C, eps=1e-6)
    elif kind == "swiglu":                         # ch3：GELU MLP -> SwiGLU（d_ff = 2/3*4C 取 256 倍数 = 1408）
        for blk in m.blocks:
            blk.mlp = SwiGLU(LLaMAConfig(hidden_size=C, intermediate_size=1408,
                                         num_attention_heads=cfg.n_head))
    elif kind == "gqa":                            # ch6：MHA -> GQA（h=8 -> h_kv=2，KV 头省 4 倍）
        for blk in m.blocks:
            blk.attn = GQASelfAttention(C, cfg.n_head, n_kv_head=2)
    else:
        raise ValueError(kind)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--out-name", default="probe_b_module")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = GPT2Config(vocab_size=8192, n_positions=1024, n_layer=6, n_embd=512, n_head=8)

    tokens = np.memmap(TOKENS_BIN, dtype=np.uint16, mode="r")          # 306M token 流（只读，不进内存）
    data_rng = torch.Generator().manual_seed(SEED)                     # 独立数据 RNG：四变体喂完全相同的批序列

    os.makedirs(OUT_DIR, exist_ok=True)
    results = {"seed": SEED, "device": str(device), "steps": args.steps,
               "batch": args.batch, "block": args.block,
               "corpus": "log/book2-ch08/tokens.bin (sp-8k, V=8192, 306M tok)", "variants": {}}
    curves_path = os.path.join(OUT_DIR, f"{args.out_name}_curves.csv")
    with open(curves_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["variant", "step", "loss"])

        for kind in ["base", "rmsnorm", "swiglu", "gqa"]:
            data_rng.manual_seed(SEED)                                 # 每变体重置：数据序列一致
            m = make_variant(kind, cfg).to(device)
            n_par = m.n_params()
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
            m.train()
            base_alloc = torch.mps.current_allocated_memory() / 2**20 if device.type == "mps" else 0.0
            peak, times, losses = base_alloc, [], []
            for step in range(args.steps):
                ix = torch.randint(0, len(tokens) - args.block - 1, (args.batch,), generator=data_rng)
                x = torch.from_numpy(np.stack(tokens[ix.numpy()[:, None] + np.arange(args.block)]).astype(np.int64)).to(device)
                y = torch.from_numpy(np.stack(tokens[ix.numpy()[:, None] + 1 + np.arange(args.block)]).astype(np.int64)).to(device)
                t0 = time.perf_counter()
                _, loss = m(x, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if device.type == "mps":
                    torch.mps.synchronize()
                times.append(time.perf_counter() - t0)
                peak = max(peak, torch.mps.current_allocated_memory() / 2**20 if device.type == "mps" else 0.0)
                losses.append(loss.item())
                if step % 30 == 0 or step == args.steps - 1:
                    writer.writerow([kind, step, round(loss.item(), 4)])
                    print(f"[{kind:8s}] step {step:4d}  loss {loss.item():.4f}", flush=True)

            steady = times[20:]                                       # 丢前 20 步（编译/缓存预热）
            sec = sum(steady) / len(steady)
            rec = {"params": n_par,
                   "sec_step_steady": round(sec, 4),
                   "sec_step_std": round(float(np.std(steady)), 4),
                   "mem_peak_delta_mb": round(peak - base_alloc, 1),
                   "loss_first": round(losses[0], 4),
                   "loss_last": round(losses[-1], 4),
                   "loss_last10_mean": round(float(np.mean(losses[-10:])), 4),
                   "steps_per_30min": int(30 * 60 / sec)}
            results["variants"][kind] = rec
            print(f"[{kind:8s}] 参数={n_par:,}  sec/step={sec:.4f}  峰值增量={rec['mem_peak_delta_mb']} MB  "
                  f"loss {rec['loss_first']} -> {rec['loss_last10_mean']}(末10均值)  30分钟≈{rec['steps_per_30min']} 步")
            del m, opt
            if device.type == "mps":
                torch.mps.empty_cache()

    # 定档建议：以最慢变体为准
    worst = max(results["variants"].values(), key=lambda r: r["sec_step_steady"])
    results["sizing_advice"] = {
        "worst_sec_step": worst["sec_step_steady"],
        "full_30min_steps": int(30 * 60 / worst["sec_step_steady"]),
        "fast_5min_steps": int(5 * 60 / worst["sec_step_steady"]),
        "note": "full 档建议 ≤6000 步（留 eval 余量）；fast 档 1000 步（趋势可见）",
    }
    out = os.path.join(OUT_DIR, f"{args.out_name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"产物已写：{out} / {curves_path}")


if __name__ == "__main__":
    main()
