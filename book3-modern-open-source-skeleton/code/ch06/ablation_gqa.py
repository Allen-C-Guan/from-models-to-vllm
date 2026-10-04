# ablation_gqa.py —— Book3 ch6 组数扫描消融：MHA(h_kv=8) / GQA-4(h_kv=4) / MQA(h_kv=1) 三臂同预算短训
# 用途：统一模块档（d=512/L=6/h=8/V=8192，B=16/T=512，其余插槽=底座 LayerNorm+GELU+wpe 不动——单变量
#       = KV 头分组）下跑三臂，产出每臂 eval loss / 曲线 / 秒步速度 + KV 几何账（每 token 字节与 @2048
#       MiB，随 JSON 输出）；「质量降序 vs KV 字节降序」的反向关系 = ch6.8 的核心读数。
#       参数差如实标注（k/v 投影随 h_kv 缩窄：三臂注意力槽参数 1,048,576/786,432/589,824，模块档不做
#       FFN 补齐——等参数补齐版是 216M 三兄弟的只算账参照，见 kv_probe.py 与 notes/07）。
# 运行方式：python ablation_gqa.py [--steps 1000] [--out-name fast] [--cells mha8,gqa4,mqa1] [--eval-every 500]
#           两档位：fast=1000 步 / full=5000 步。
# 产物（log/book3-ch06/，不入库）：ablation_gqa_{out}.json + curve_{臂}_{out}.csv + ckpt_{臂}_{out}.pt。
# 协议口径（= ch2/ch3/ch4 消融同款，Book2 五件套）：AdamW(0.9/0.95, wd=0.1) + clip 1.0 + lr 1e-3 余弦
#           降至 1e-4 + warmup 100；train bf16 autocast / eval fp32；tokens.bin 顺序分块一次通过
#           （三臂按步号取同批）；eval 固定 49 窗 (16,512)=401,408 token。
# 换装细节（承探针 B 的单插槽版）：c_attn 合并式 -> q/k/v/o 分立（无 bias，对齐 LLaMA 口径），
#           repeat_kv 后注意力照常按 h=8 个头算——省在投影端不在计算端（ch6.7 的教学点）。
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "00-feasibility")))
for _base in (os.path.join(HERE, "..", "..", "Book2-预训练革命", "ch10"),                              # 工作区布局
              os.path.join(HERE, "..", "..", "..", "book2-pretraining-revolution", "code", "ch10")):  # 书仓快照布局
    if os.path.isdir(_base):
        sys.path.insert(0, os.path.abspath(_base))  # Book2 ch10 GPT-2 定版底座
from model import GPT2, GPT2Config          # noqa: E402  Book2 ch10 定版底座（改装对象）
from llama_slots import repeat_kv           # noqa: E402  本册插槽组件库（KV 头扩展正身）

# ---------------- 固定口径 ----------------
SEED = 20261002
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book3-ch06")
TOKENS_BIN = os.path.join(REPO_ROOT, "log", "book2-ch08", "tokens.bin")
EVAL_BIN = os.path.join(REPO_ROOT, "log", "book2-ch08", "eval.bin")

VOCAB = 8192
BATCH = 16
BLOCK = 512
TPB = BATCH * BLOCK
EVAL_BATCHES = 49
PEAK_LR = 1e-3
MIN_LR_RATIO = 0.1
WARMUP = 100
WDECL = 0.1
TAIL = 300
N_LAYER, N_EMBD, N_HEAD = 6, 512, 8              # 统一模块档
N_POSITIONS = 1024
ARMS = {"mha8": 8, "gqa4": 4, "mqa1": 1}         # 臂名 -> h_kv（h=8 不变，唯一变量 = KV 头分组）

# KV 几何账（手算公式 2*L*n*h_kv*d_k*bytes；统一在 kv_probe.py 对拍实测，此处随报数）
DTYPE_BYTES = 2                                   # fp16/bf16


def kv_bytes(L, n, h_kv, d_k, bytes_per=DTYPE_BYTES):
    """KV cache 字节手算：K 与 V 各一份，形状 (B,L,n,h_kv,d_k)。"""
    return 2 * L * n * h_kv * d_k * bytes_per


def kv_geometry_block():
    """随 JSON 输出的 KV 几何账（一律带几何标签，防 207M/216M 两套几何混用）。"""
    per_token = {arm: kv_bytes(N_LAYER, 1, h_kv, N_EMBD // N_HEAD)
                 for arm, h_kv in ARMS.items()}
    return {
        "formula": "bytes = 2(K+V) * L * n * h_kv * d_k * dtype_bytes",
        "module_tier": {"geometry": f"d={N_EMBD}/L={N_LAYER}/h={N_HEAD}/d_k={N_EMBD//N_HEAD}/B=1/fp16",
                        "kv_per_token_bytes": per_token,
                        "kv_per_token_kib": {k: round(v / 2**10, 2) for k, v in per_token.items()},
                        "kv_at_2048_mib": {k: round(v * 2048 / 2**20, 2) for k, v in per_token.items()}},
        "refs": {
            "207m_cand1": {"geometry": "L=12,h=16,h_kv=8,d_k=64（本册 207M 候选，V=32k）",
                           "kv_per_token_kib": 24.0, "kv_at_2048_mib": 48.0,
                           "mha_counterfactual_at_2048_mib": 96.0, "mqa_at_2048_mib": 6.0},
            "brothers_216m": {"geometry": "L=16,h=16,d_k=64（216M 等参数三兄弟，V=8k；MHA/GQA-4/MQA）",
                              "kv_per_token_kib": {"mha": 64.0, "gqa4": 16.0, "mqa": 4.0},
                              "kv_at_2048_mib": {"mha": 128.0, "gqa4": 32.0, "mqa": 8.0},
                              "note": "任务书原文「KV/token 三档 64/32/8KiB」勘正为 64/16/4 KiB"
                                      "（@2048 才是 128/32/8 MiB）——与 plan v1.1 勘正口径一致"},
        },
        "note": "模块档实测对拍与 216M 三兄弟参数账见 ch06/kv_probe.py 产物",
    }


class GQASelfAttention(nn.Module):
    """GPT-2 底座上的 GQA 插槽替换（ch6 单插槽改装版，承探针 B）：分离 q/k/v/o 投影 + repeat_kv。

    与 llama_slots.GroupedQueryAttention 的区别：不带 RoPE（位置编码仍是底座 wpe 查表——单插槽
    改装一次只动一个零件，RoPE 是 ch4 的事）；投影无 bias（对齐 LLaMA 口径）。
    k_proj/v_proj 只输出 h_kv*hd 维——KV 头省在投影端，注意力计算端 repeat_kv 后照常 h 个头。
    输入 (B,n,C) -> 输出 (B,n,C)，接口与底座 CausalSelfAttention 一致（插槽可换性）。
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
        self._mask_cache = {}                                          # n -> causal mask（动态构建）

    def _causal_mask(self, n, device):
        if n not in self._mask_cache or self._mask_cache[n].device != device:
            self._mask_cache[n] = torch.tril(torch.ones(n, n, device=device)).view(1, 1, n, n)
        return self._mask_cache[n]

    def forward(self, x):
        B, n, C = x.shape                                              # (B,n,C)
        h, h_kv, hd = self.n_head, self.n_kv_head, self.hd
        q = self.q_proj(x).view(B, n, h, hd).transpose(1, 2)           # (B,h,n,hd)
        k = self.k_proj(x).view(B, n, h_kv, hd).transpose(1, 2)        # (B,h_kv,n,hd)
        v = self.v_proj(x).view(B, n, h_kv, hd).transpose(1, 2)        # (B,h_kv,n,hd)
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)      # -> (B,h,n,hd)（分组共享）
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)                # (B,h,n,n)
        att = att.masked_fill(self._causal_mask(n, x.device) == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)                               # 行和=1
        y = (att @ v).transpose(1, 2).contiguous().view(B, n, C)       # (B,h,n,hd) -> (B,n,C)
        return self.o_proj(y)                                          # (B,n,C)


def lr_at(step, total):
    if step <= WARMUP:
        return PEAK_LR * step / WARMUP
    prog = (step - WARMUP) / max(1, total - WARMUP)
    return PEAK_LR * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * prog)))


class TrainStream:
    """tokens.bin 流首部按步号顺序切块（三臂按步号取同批；断点续跑天然安全）。"""

    def __init__(self, path):
        self.arr = np.memmap(path, dtype=np.uint16, mode="r")
        self.max_step = (len(self.arr) - 1) // TPB

    def batch(self, step, device):
        off = (step - 1) * TPB
        span = np.asarray(self.arr[off:off + TPB + 1], dtype=np.int64)       # (8193,)
        x = torch.from_numpy(span[:-1].reshape(BATCH, BLOCK)).to(device)     # (16,512)
        y = torch.from_numpy(span[1:].reshape(BATCH, BLOCK)).to(device)      # (16,512)
        return x, y


def build_eval_batches(device):
    """固定 eval 窗集：eval 池固定种子抽 49 批 (16,512)（跨臂/跨步完全同集）。"""
    ev = np.memmap(EVAL_BIN, dtype=np.uint16, mode="r")
    gen = torch.Generator().manual_seed(SEED)
    offs = torch.randint(0, len(ev) - BLOCK - 1, (EVAL_BATCHES * BATCH,), generator=gen).numpy()
    xs, ys = [], []
    for o in offs:
        xs.append(np.asarray(ev[o:o + BLOCK], dtype=np.int64))
        ys.append(np.asarray(ev[o + 1:o + BLOCK + 1], dtype=np.int64))
    x = torch.from_numpy(np.stack(xs).reshape(EVAL_BATCHES, BATCH, BLOCK)).to(device)  # (49,16,512)
    y = torch.from_numpy(np.stack(ys).reshape(EVAL_BATCHES, BATCH, BLOCK)).to(device)
    return x, y


@torch.no_grad()
def eval_loss(model, ex, ey):
    model.eval()
    losses = []
    for i in range(EVAL_BATCHES):
        _, loss = model(ex[i], ey[i])              # (16,512) -> 标量
        losses.append(loss.item())
    model.train()
    mean = sum(losses) / len(losses)
    se = (sum((l - mean) ** 2 for l in losses) / (len(losses) - 1)) ** 0.5 / math.sqrt(len(losses))
    return mean, se


def attn_slot_params(h_kv):
    """注意力槽参数手算：q/o 全宽 2C²，k/v 按 h_kv 缩窄 2*C*h_kv*hd。"""
    C, hd = N_EMBD, N_EMBD // N_HEAD
    return 2 * C * C + 2 * C * h_kv * hd


def build_arm(arm):
    """构造三臂：同种子建底座（RNG 一致），再换装注意力槽（换装件重打 GPT-2 口径初始化）。

    底座非注意力件（wte/wpe/ln/mlp/ln_f）初始权重跨臂逐位一致；换装槽以固定子种子重打：
    q_proj 跨臂逐位一致，k/v 形状即变量本身，o_proj 因抽取顺序不同而异（被换插槽内部差异，
    单变量 = KV 头分组）。o_proj 作残差流出端按 GPT-2 口径额外乘 1/sqrt(2L)（ch3 换装纪律沿用）。
    """
    h_kv = ARMS[arm]
    torch.manual_seed(SEED)
    cfg = GPT2Config(vocab_size=VOCAB, n_positions=N_POSITIONS,
                     n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD)
    m = GPT2(cfg)
    C = N_EMBD
    for i, blk in enumerate(m.blocks):             # 6 处注意力换装
        torch.manual_seed(SEED + 1 + i)            # 换装件逐块子种子：跨臂同块 q_proj 逐位一致
        new_attn = GQASelfAttention(C, N_HEAD, h_kv)
        for mod in new_attn.modules():             # GPT-2 口径初始化（N(0,0.02)）
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
        nn.init.normal_(new_attn.o_proj.weight, mean=0.0,
                        std=0.02 / math.sqrt(2 * N_LAYER))   # 残差流出端 1/sqrt(2L)
        blk.attn = new_attn
    # ---- 参数账 assert（手算 = 实测逐位）----
    n_total = sum(p.numel() for p in m.parameters())
    base_attn = (C * 3 * C + 3 * C) + (C * C + C)  # 底座 c_attn(含 bias)+c_proj(含 bias)
    hand = 23_633_920 - base_attn * N_LAYER + attn_slot_params(h_kv) * N_LAYER
    assert n_total == hand, f"参数对账失败：{n_total} != {hand}（arm={arm}）"
    return m


def run_arm(arm, steps, eval_every, device, stream, ex, ey, report, out_json):
    ckpt_path = os.path.join(OUT_DIR, f"ckpt_{arm}_{report['meta']['out_name']}.pt")
    m = build_arm(arm).to(device)
    n_total = sum(p.numel() for p in m.parameters())
    n_nonemb = n_total - m.wte.weight.numel() - m.wpe.weight.numel()
    opt = torch.optim.AdamW(m.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=WDECL)

    train_losses, lrs, times, evals = [], [], [], []
    start_step, elapsed_train, elapsed_eval = 0, 0.0, 0.0
    resumed_from = None
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        m.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        train_losses, lrs, times = ck["train_losses"], ck["lrs"], ck["times"]
        evals = ck["evals"]
        start_step, elapsed_train, elapsed_eval = ck["step"], ck["elapsed_train"], ck["elapsed_eval"]
        resumed_from = start_step
        print(f"  [{arm} 续跑] 自 step {start_step + 1}", flush=True)

    m.train()
    base_alloc = torch.mps.current_allocated_memory() / 2**20 if device.type == "mps" else 0.0
    peak = base_alloc
    diverged_at, reason = None, None
    t_arm = time.perf_counter()
    for step in range(start_step + 1, steps + 1):
        lr = lr_at(step, steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = stream.batch(step, device)           # (16,512)×2（三臂同批）
        t0 = time.perf_counter()
        with torch.autocast(device_type="mps", dtype=torch.bfloat16) if device.type == "mps" \
                else torch.autocast("cpu", enabled=False):
            _, loss = m(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if device.type == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
        elapsed_train += times[-1]
        lv = loss.item()
        train_losses.append(lv)
        lrs.append(lr)
        if device.type == "mps":
            peak = max(peak, torch.mps.current_allocated_memory() / 2**20)
        if not math.isfinite(lv) or lv > 100.0:
            diverged_at, reason = step, ("non-finite" if not math.isfinite(lv) else f"loss={lv:.1f}>100")
            print(f"  [{arm} 发散] step {step}: {reason}", flush=True)
            break
        if step % 100 == 0 or step == 1:
            print(f"  [{arm:5s}] step {step:5d}/{steps} train {lv:.4f} lr {lr:.2e} "
                  f"({elapsed_train/(step - start_step + 1):.3f}s/step)", flush=True)
        if step % eval_every == 0 or step == steps:
            t0 = time.perf_counter()
            el, se = eval_loss(m, ex, ey)
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed_eval += time.perf_counter() - t0
            evals.append({"step": step, "eval_loss": round(el, 5), "eval_se": round(se, 5),
                          "n_tokens": EVAL_BATCHES * TPB})
            print(f"  [{arm:5s}] [eval] step {step}: eval loss {el:.4f} ± {se:.4f}", flush=True)
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": step,
                        "train_losses": train_losses, "lrs": lrs, "times": times, "evals": evals,
                        "elapsed_train": elapsed_train, "elapsed_eval": elapsed_eval}, ckpt_path)

    steps_done = len(train_losses)
    steady = times[20:] if len(times) > 20 else times
    tail = train_losses[-min(TAIL, steps_done):]
    h_kv = ARMS[arm]
    rec = {
        "arm": arm, "h_kv": h_kv, "n_rep": N_HEAD // h_kv,
        "arm_def": f"GQA 插槽（h={N_HEAD}, h_kv={h_kv}）：q/k/v/o 分立无 bias + repeat_kv；其余底座件不动",
        "config": {"d": N_EMBD, "L": N_LAYER, "h": N_HEAD, "h_kv": h_kv, "V": VOCAB,
                   "B": BATCH, "T": BLOCK, "n_positions": N_POSITIONS},
        "params_total": n_total, "params_non_embedding": n_nonemb,
        "attn_slot_params_per_layer": attn_slot_params(h_kv),
        "params_diff_vs_mha8": n_total - (23_633_920 - ((N_EMBD * 3 * N_EMBD + 3 * N_EMBD)
                                                       + (N_EMBD * N_EMBD + N_EMBD)) * N_LAYER
                                         + attn_slot_params(8) * N_LAYER),
        "kv_per_token_bytes": kv_bytes(N_LAYER, 1, h_kv, N_EMBD // N_HEAD),
        "kv_at_2048_mib": round(kv_bytes(N_LAYER, 2048, h_kv, N_EMBD // N_HEAD) / 2**20, 2),
        "steps_done": steps_done, "steps_planned": steps, "resumed_from_step": resumed_from,
        "tail300_train_loss": round(sum(tail) / len(tail), 5),
        "last_step_loss": round(train_losses[-1], 5) if train_losses else None,
        "min_train_loss": round(min(train_losses), 5) if train_losses else None,
        "evals": evals,
        "eval_final": evals[-1]["eval_loss"] if evals else None,
        "eval_final_se": evals[-1]["eval_se"] if evals else None,
        "sec_per_step_steady": round(sum(steady) / len(steady), 4),
        "sec_step_std": round(float(np.std(steady)), 4),
        "mem_peak_delta_mb": round(peak - base_alloc, 1),
        "wallclock_sec": round(time.perf_counter() - t_arm, 1),
        "diverged": diverged_at is not None, "diverged_at_step": diverged_at, "divergence_reason": reason,
        "precision": "train bf16 autocast / eval fp32",
    }
    curve_path = os.path.join(OUT_DIR, f"curve_{arm}_{report['meta']['out_name']}.csv")
    with open(curve_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["step", "train_loss", "lr"])
        for i, (lv, lr) in enumerate(zip(train_losses, lrs), 1):
            w.writerow([i, f"{lv:.6g}", f"{lr:.6g}"])
    if steps_done >= steps and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    report["arms"][arm] = rec
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  [{arm} 完成] steps {steps_done} | tail300 {rec['tail300_train_loss']} | "
          f"eval_final {rec['eval_final']} | {rec['sec_per_step_steady']}s/step", flush=True)
    del m, opt
    if device.type == "mps":
        torch.mps.empty_cache()
    return rec


def add_pair_summaries(report):
    """三臂两两对照（判读方向见 notes/07 §1 预注册：质量-KV 反向关系）。"""
    arms = report["arms"]
    ps = {}
    for hi, lo in [("mha8", "gqa4"), ("gqa4", "mqa1"), ("mha8", "mqa1")]:
        a, b = arms.get(hi), arms.get(lo)
        if not (a and b and a["eval_final"] is not None and b["eval_final"] is not None):
            continue
        ps[f"{hi}_vs_{lo}"] = {
            "eval_final_diff": round(a["eval_final"] - b["eval_final"], 5),   # 正 = KV 头多者更低（质量更好）
            "kv_bytes_ratio": round(a["kv_per_token_bytes"] / b["kv_per_token_bytes"], 2),
            "sec_per_step_diff": round(a["sec_per_step_steady"] - b["sec_per_step_steady"], 4),
        }
    report["pair_summaries"] = ps


def main():
    ap = argparse.ArgumentParser(description="ch6 组数扫描消融：MHA/GQA-4/MQA 三臂（同数据同种子）")
    ap.add_argument("--steps", type=int, default=1000, help="每臂步数（fast=1000 / full=5000）")
    ap.add_argument("--out-name", type=str, default=None)
    ap.add_argument("--cells", type=str, default="mha8,gqa4,mqa1", help="臂选择（mha8 / gqa4 / mqa1）")
    ap.add_argument("--eval-every", type=int, default=500)
    args = ap.parse_args()
    out_name = args.out_name or ("fast" if args.steps <= 1500 else "full")
    arms = [c.strip() for c in args.cells.split(",")]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"未知臂 {a}（可选 {'/'.join(ARMS)}）")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    stream = TrainStream(TOKENS_BIN)
    if args.steps > stream.max_step:
        raise SystemExit(f"步数 {args.steps} 超出一次通过上限 {stream.max_step}")
    ex, ey = build_eval_batches(device)
    out_json = os.path.join(OUT_DIR, f"ablation_gqa_{out_name}.json")

    if os.path.exists(out_json):
        with open(out_json, encoding="utf-8") as f:
            report = json.load(f)
        report.setdefault("arms", {})
        print(f"[续跑] 已有报告载入：臂 {sorted(report['arms'])}", flush=True)
    else:
        report = {"arms": {}}
    report["meta"] = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "device": str(device),
        "torch": torch.__version__, "out_name": out_name,
        "experiment": "ch6 组数扫描：MHA(h_kv=8)/GQA-4(h_kv=4)/MQA(h_kv=1)——单变量 = KV 头分组",
        "module_tier": f"d={N_EMBD}/L={N_LAYER}/h={N_HEAD}/V={VOCAB}，B={BATCH}/T={BLOCK}（8192 tok/步）",
        "swap_detail": ("c_attn 合并式 -> q/k/v/o 分立（无 bias）+ llama_slots.repeat_kv；"
                        "位置编码仍 wpe、归一化仍 LN、FFN 仍 GELU（单插槽原则）；"
                        "模块档不做 FFN 参数补齐（差量如实标注），等参数补齐参照 = 216M 三兄弟只算账"),
        "corpus": "log/book2-ch08/tokens.bin（sp-8k，306M token 流）",
        "optimizer": f"AdamW(betas=0.9,0.95,wd={WDECL}) + grad_clip 1.0（Book2 五件套）",
        "schedule": f"lr {PEAK_LR} 余弦降至 {PEAK_LR*MIN_LR_RATIO:.0e}，warmup={WARMUP}；三臂统一日程",
        "data_protocol": "顺序连续分块一次通过；三臂按步号取同批",
        "eval_protocol": f"每 {args.eval_every} 步+终步：eval.bin 固定 49 窗 {EVAL_BATCHES*TPB:,} token，fp32",
        "kv_geometry": kv_geometry_block(),
        "precision": "train bf16 autocast / eval fp32",
        "ckpt": "每 eval 边界存 ckpt，臂完成即删；中断后重跑同命令自动续跑",
        "argv": " ".join(sys.argv[1:]),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    t_all = time.perf_counter()
    for arm in arms:
        done = report["arms"].get(arm)
        if done and done["steps_done"] >= args.steps:
            print(f"[跳过] 臂 {arm} 已完成（{done['steps_done']} 步）", flush=True)
            continue
        print(f"\n===== 臂 {arm}（h_kv={ARMS[arm]}，{args.steps} 步） =====", flush=True)
        run_arm(arm, args.steps, args.eval_every, device, stream, ex, ey, report, out_json)
    add_pair_summaries(report)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[全部完成] 总耗时 {(time.perf_counter()-t_all)/60:.1f} min -> {out_json}", flush=True)
    for a, r in report["arms"].items():
        print(f"  {a:5s} params={r['params_total']:,} KV/tok={r['kv_per_token_bytes']}B "
              f"eval_final={r['eval_final']} {r['sec_per_step_steady']}s/step", flush=True)


if __name__ == "__main__":
    main()
