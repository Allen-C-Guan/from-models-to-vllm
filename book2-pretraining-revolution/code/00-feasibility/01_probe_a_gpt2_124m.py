# 01_probe_a_gpt2_124m.py —— 探针(a)：手写 GPT-2 124M（12L/768d/12H，pre-LN+GELU+tied）训练可行性实测
# 实测 MPS 与 CPU 双口径 sec/step 与内存峰值；batch=8/seq=256 与 batch=4/seq=1024 两档；
# 附 bf16 autocast 算子覆盖矩阵与加速比、梯度检查点开/关对照。数值（loss）以 CPU 报数为对拍口径。
# 用法：python 01_probe_a_gpt2_124m.py [--quick]；产物落 log/book2-feasibility/probe_a_gpt2_124m.json

import argparse
import gc
import json
import math
import os
import platform
import sys
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ---------------- 固定种子 ----------------
SEED = 20261002
torch.manual_seed(SEED)

OUT_DIR = os.path.join("log", "book2-feasibility")


# ---------------- 手写 GPT-2（与官方 124M 同构：pre-LN + GELU + tied lm_head） ----------------
class CausalSelfAttention(nn.Module):
    """多头因果自注意力（合并式 QKV 投影，nanoGPT 口径）。"""

    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        # qkv 合并投影：(B,T,768) -> (B,T,2304)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape  # (B,T,768)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)          # 各 (B,T,768)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B,12,T,64)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))     # (B,12,T,T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        y = F.softmax(att, dim=-1) @ v                              # (B,12,T,64)
        y = y.transpose(1, 2).contiguous().view(B, T, C)            # (B,T,768)
        return self.c_proj(y)


class MLP(nn.Module):
    """GPT-2 前馈：768 -> 3072 -> 768，GELU 激活。"""

    def __init__(self, n_embd: int):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    """pre-LN 残差块：x = x + attn(ln(x)); x = x + mlp(ln(x))。"""

    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x, use_ckpt: bool = False):
        if use_ckpt:
            x = x + checkpoint(self.attn, self.ln_1(x), use_reentrant=False)
            x = x + checkpoint(self.mlp, self.ln_2(x), use_reentrant=False)
        else:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    """GPT-2 124M：词嵌入 50257x768 + 位置嵌入 1024x768（学习式）+ 12 个 Block + ln_f + tied lm_head。"""

    def __init__(self, n_layer=12, n_embd=768, n_head=12, vocab=50257, block_size=1024):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab, bias=False)
        self.head.weight = self.wte.weight  # tied embeddings：输出投影与词嵌入共享
        self._n_layer = n_layer
        self.apply(self._init_weights)
        # 残差流出端投影（attn.c_proj / mlp.c_proj）额外按 1/sqrt(2*n_layer) 缩放（nanoGPT 口径）
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        """GPT-2 初始化：Linear/Embedding ~ N(0, 0.02)，偏置清零。
        教学警示：若沿用 PyTorch 默认（Embedding ~ N(0,1)），tied head 的 logit 方差 ≈768，
        初始 loss 会冲到 ~399 而非 ln(50257)≈10.83 —— ch10 必须交代这一步。"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def n_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:  # Kaplan 6ND 口径：N = 非 embedding 参数（wte 与 wpe 都扣除；tied head 同属 wte）
            n -= self.wte.weight.numel() + self.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None, use_ckpt=False):
        B, T = idx.shape  # (B,T)
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)      # (B,T,768)
        for blk in self.blocks:
            x = blk(x, use_ckpt=use_ckpt)
        x = self.ln_f(x)                        # (B,T,768)
        if targets is None:
            return self.head(x), None           # (B,T,50257)
        logits = self.head(x)                   # (B,T,50257)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


# ---------------- 内存峰值采样：Apple 统一内存无独立显存，口径 = 轮询 current_allocated_memory ----------------
class MpsMemSampler(threading.Thread):
    """后台线程每 interval 秒采样 MPS 分配内存，记录峰值（current 与 driver 两种口径）。"""

    def __init__(self, interval=0.03):
        super().__init__(daemon=True)
        import torch.mps as mps
        self._mps = mps
        self.interval = interval
        self.peak_current = 0.0
        self.peak_driver = 0.0
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                self.peak_current = max(self.peak_current,
                                        self._mps.current_allocated_memory() / 1024**2)
                self.peak_driver = max(self.peak_driver,
                                       self._mps.driver_allocated_memory() / 1024**2)
            except Exception:
                pass
            self._stop_evt.wait(self.interval)

    def stop_and_report(self):
        self._stop_evt.set()
        self.join(timeout=1.0)
        return {"peak_current_mib": round(self.peak_current, 1),
                "peak_driver_mib": round(self.peak_driver, 1)}


def make_batch(batch, seq, vocab, device):
    """随机 token 批：x 与 y 错位一格（因果 LM 目标）。"""
    idx = torch.randint(0, vocab, (batch, seq + 1), device=device)
    return idx[:, :-1].contiguous(), idx[:, 1:].contiguous()


def _cpu_rss_mib():
    """CPU 口径的进程常驻内存峰值（macOS getrusage 的 ru_maxrss 单位是字节，非 Linux 的 KB）。"""
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 1)


def train_steps(model, opt, batch, seq, vocab, device, n_steps, use_ckpt=False,
                autocast_ctx=None, sampler=True):
    """跑 n_steps 个完整训练步（前向+反向+optimizer.step），返回 sec/step、内存峰值、loss 序列。"""
    rss_before = _cpu_rss_mib()
    mem = MpsMemSampler() if (sampler and device == "mps") else None
    if mem:
        mem.start()
    losses = []
    t0 = time.perf_counter()
    for _ in range(n_steps):
        x, y = make_batch(batch, seq, vocab, device)
        if autocast_ctx is not None:
            with autocast_ctx():
                _, loss = model(x, y, use_ckpt=use_ckpt)
        else:
            _, loss = model(x, y, use_ckpt=use_ckpt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cpu":
        pass
    sec_total = time.perf_counter() - t0
    mem_report = mem.stop_and_report() if mem else {}
    mem_report["cpu_rss_peak_mib"] = max(_cpu_rss_mib(), rss_before)
    return {"sec_per_step": round(sec_total / n_steps, 4),
            "sec_total": round(sec_total, 1),
            "losses": [round(l, 4) for l in losses],
            "mem": mem_report}


def bf16_op_coverage(device):
    """bf16 autocast 在 MPS 的算子覆盖矩阵：逐算子 try/except，再测整模型一步。"""
    ops = {}

    def probe(name, fn):
        try:
            out = fn()
            ops[name] = {"ok": True, "out_dtype": str(out.dtype) if torch.is_tensor(out) else "n/a"}
        except Exception as e:
            ops[name] = {"ok": False, "err": str(e)[:160]}

    lin = nn.Linear(128, 128).to(device)
    ln = nn.LayerNorm(128).to(device)
    emb = nn.Embedding(100, 128).to(device)
    idx = torch.randint(0, 100, (2, 16), device=device)
    x = torch.randn(2, 16, 128, device=device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        probe("linear", lambda: lin(x))
        probe("layernorm", lambda: ln(x))
        probe("gelu", lambda: F.gelu(x))
        probe("matmul_qk", lambda: x @ x.transpose(-2, -1))
        probe("softmax", lambda: F.softmax(x, dim=-1))
        probe("embedding", lambda: emb(idx))
        probe("cross_entropy", lambda: F.cross_entropy(x.view(-1, 128),
                                                       torch.randint(0, 128, (32,), device=device)))
    # 带反向的整模型一步
    model = GPT2().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    try:
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(*make_batch(2, 128, 50257, device))
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        ops["full_model_step"] = {"ok": True, "loss_dtype": str(loss.dtype)}
    except Exception as e:
        ops["full_model_step"] = {"ok": False, "err": str(e)[:200]}
    del model, opt
    gc.collect()
    return ops


def extrapolate(sec_per_step, tokens_per_step, targets=(5_000_000, 20_000_000, 50_000_000, 100_000_000)):
    """从实测 sec/step 外推若干 token 预算的 wall-clock。"""
    out = {}
    for tgt in targets:
        steps = tgt / tokens_per_step
        sec = steps * sec_per_step
        out[f"{tgt/1e6:.0f}M_tok"] = {"steps": int(steps), "hours": round(sec / 3600, 2)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="快速档：CPU 步数减半")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    report = {
        "meta": {
            "date": "2026-10-02", "seed": SEED,
            "chip": platform.platform(), "torch": torch.__version__,
            "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        }
    }

    # 参数量对拍：应为 124,439,808（GPT-2 124M 官方口径，含 embedding）
    model_cpu = GPT2()
    report["meta"]["n_params_total"] = model_cpu.n_params()
    report["meta"]["n_params_non_embedding"] = model_cpu.n_params(non_embedding=True)
    report["meta"]["random_init_expected_loss_ln50257"] = round(math.log(50257), 4)
    del model_cpu

    device = "mps"
    model = GPT2().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    import torch.mps as mps
    report["meta"]["mps_recommended_max_gib"] = round(mps.recommended_max_memory() / 1024**3, 2)

    # ---- 1) fp32 基线：batch=8/seq=256 ----
    model.train()
    train_steps(model, opt, 8, 256, 50257, device, 3)  # 预热（含 Metal shader 编译与优化器懒分配）
    r = train_steps(model, opt, 8, 256, 50257, device, 10)
    report["mps_fp32_b8_s256"] = r
    print("mps fp32 b8s256:", r["sec_per_step"], "s/step, mem", r["mem"], flush=True)

    # ---- 2) fp32：batch=4/seq=1024（同 token 量、长序列档） ----
    r = train_steps(model, opt, 4, 1024, 50257, device, 2)  # 预热
    r = train_steps(model, opt, 4, 1024, 50257, device, 5)
    report["mps_fp32_b4_s1024"] = r
    print("mps fp32 b4s1024:", r["sec_per_step"], "s/step, mem", r["mem"], flush=True)

    # ---- 3) bf16 autocast：覆盖矩阵 + 加速比 ----
    report["bf16_op_coverage"] = bf16_op_coverage(device)
    r = train_steps(model, opt, 8, 256, 50257, device, 3)  # 重新预热回 fp32 稳态
    r = train_steps(model, opt, 8, 256, 50257, device, 10,
                    autocast_ctx=lambda: torch.autocast(device_type="mps", dtype=torch.bfloat16))
    report["mps_bf16_b8_s256"] = r
    report["mps_bf16_speedup_vs_fp32"] = round(
        report["mps_fp32_b8_s256"]["sec_per_step"] / r["sec_per_step"], 2)
    print("mps bf16 b8s256:", r["sec_per_step"], "s/step, speedup",
          report["mps_bf16_speedup_vs_fp32"], flush=True)

    # ---- 4) 梯度检查点对照（batch=4/seq=1024，激活省内存最有感的档） ----
    r = train_steps(model, opt, 4, 1024, 50257, device, 3, use_ckpt=True)
    r = train_steps(model, opt, 4, 1024, 50257, device, 5, use_ckpt=True)
    report["mps_fp32_b4_s1024_ckpt"] = r
    print("mps ckpt b4s1024:", r["sec_per_step"], "s/step, mem", r["mem"], flush=True)

    # ---- 5) 外推（MPS fp32 基线，b8s256=2048 tok/step） ----
    report["extrapolate_mps_fp32_b8s256"] = extrapolate(report["mps_fp32_b8_s256"]["sec_per_step"], 8 * 256)

    del model, opt
    gc.collect()
    torch.mps.empty_cache()

    # ---- 6) CPU 口径（数值对拍基准 + sec/step） ----
    torch.manual_seed(SEED)  # 与 MPS 完全同种子，loss 序列可对拍
    model = GPT2().to("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    n_timed = 2 if args.quick else 3
    train_steps(model, opt, 8, 256, 50257, "cpu", 1)  # 预热
    r = train_steps(model, opt, 8, 256, 50257, "cpu", n_timed)
    report["cpu_fp32_b8_s256"] = r
    print("cpu fp32 b8s256:", r["sec_per_step"], "s/step, loss0", r["losses"][0], flush=True)
    r = train_steps(model, opt, 4, 1024, 50257, "cpu", max(1, n_timed - 1))
    report["cpu_fp32_b4_s1024"] = r
    print("cpu fp32 b4s1024:", r["sec_per_step"], "s/step", flush=True)
    report["extrapolate_cpu_fp32_b8s256"] = extrapolate(report["cpu_fp32_b8_s256"]["sec_per_step"], 8 * 256)

    out_path = os.path.join(OUT_DIR, "probe_a_gpt2_124m.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
