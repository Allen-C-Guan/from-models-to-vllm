# ablation_pe.py —— Book3 ch4 位置编码换件消融：学习式 wpe vs RoPE（同数据同种子两臂短训 + 4× 外推 eval）
# 用途：统一模块档（d=512/L=6/h=8/V=8192，B=16/T=512）下跑两臂——wpe（Book2 ch10 GPT-2 底座原样，
#       wpe 查表 n_positions=1024）vs rope（改装第三例：删 wpe、llama_slots 的 RoPE 组件作用于
#       attention 内的 Q/K——换装发生在初始化之后，两臂共享件初始权重逐位一致）。
#       训练窗 T=512（统一模块档）；eval 双口径：同窗 512（49 固定窗，判据主字段）+
#       4× 外推 2048（RoPE 臂出分桶 loss；wpe 臂在 2048 直接触发查表越界——try/except 捕获报错，
#       作为 N2「1024 位之墙」的硬墙素材记录进 JSON）。
# 运行方式：python ablation_pe.py [--steps 1000] [--out-name fast] [--cells wpe,rope] [--eval-every 500]
#           两档位：fast=1000 步 / full=5000 步（卷级大纲 ch4 契约：训 512 测 2048，与 ch7 的 4× 叙事对齐）。
# 产物（log/book3-ch04/，不入库）：ablation_pe_{out}.json（总报告）+ curve_{臂}_{out}.csv（每臂曲线）
#           + ckpt_{臂}_{out}.pt（断点续跑用，臂完成即删）。
# 协议口径（= ch2/ch3 消融同款，Book2 五件套）：AdamW(betas 0.9/0.95, wd=0.1) + grad clip 1.0
#           + lr 1e-3 余弦降至 1e-4 + warmup 100；train bf16 autocast / eval fp32；
#           tokens.bin 顺序连续分块一次通过（第 k 步消费 (k-1)*8192 起 8193 token，两臂同批）；
#           eval 固定窗：eval.bin 固定种子抽 49 个 (16,512) 窗 = 401,408 token（跨臂跨步同集）。
# 参数账（模块档，如实标注不 assert 对齐——位置编码两件的参数天然不同）：
#           wpe 臂 23,633,920（含 wpe 表 1024×512=524,288）；rope 臂 23,109,632（删 wpe，RoPE 零位置参数）。
import argparse
import csv
import json
import math
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "00-feasibility")))
for _base in (os.path.join(HERE, "..", "..", "Book2-预训练革命", "ch10"),                              # 工作区布局
              os.path.join(HERE, "..", "..", "..", "book2-pretraining-revolution", "code", "ch10")):  # 书仓快照布局
    if os.path.isdir(_base):
        sys.path.insert(0, os.path.abspath(_base))  # Book2 ch10 GPT-2 定版底座
from model import GPT2, GPT2Config                     # noqa: E402  Book2 ch10 定版底座（改装对象）
from llama_slots import apply_rope, build_rope_cache   # noqa: E402  本册插槽组件库（ch4 换装件，RoPE 正身）

# ---------------- 固定口径 ----------------
SEED = 20261002          # 全书统一种子（模型初始化；数据由步号决定，无需 RNG）
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book3-ch04")
TOKENS_BIN = os.path.join(REPO_ROOT, "log", "book2-ch08", "tokens.bin")   # 306M token 训练池（sp-8k）
EVAL_BIN = os.path.join(REPO_ROOT, "log", "book2-ch08", "eval.bin")       # 1.2M token eval 池（heldout 尾段）

VOCAB = 8192             # sp-8k 词表
BATCH = 16               # B：批大小
BLOCK = 512              # T：训练序列长（统一模块档；wpe 表 1024 行留余量）
TPB = BATCH * BLOCK      # 8192 token/步
EVAL_BATCHES = 49        # 49×8192 = 401,408 token（batch-1 同口径）
EXTRAP_N = 2048          # 4× 外推 eval 窗（512×4；wpe 臂在此触发 N2 硬墙）
EXTRAP_WINDOWS = 192     # 192×2048 = 393,216 token（与 ch7 各窗长同 token 预算）
EXTRAP_BATCH = 4         # 2048 窗 fp32 前向批大小
EXTRAP_BUCKETS = [0, 512, 1024, 1536, 2048]   # 位置分桶：首桶=训练见过的位置段
PEAK_LR = 1e-3
MIN_LR_RATIO = 0.1
WARMUP = 100
WDECL = 0.1
TAIL = 300
N_LAYER, N_EMBD, N_HEAD = 6, 512, 8              # 统一模块档（与 ch2/ch3/ch6 同档）
N_POSITIONS = 1024                              # wpe 查表行数（底座原样）
ROPE_THETA = 10000.0                            # RoPE 基底（LLaMA1/2 口径）

_MASK_CACHE = {}   # n -> causal mask（rope 臂动态构建，越过 n_positions 也能用——RoPE 无查表上限）


def causal_mask(n, device):
    """(1,1,n,n) 下三角掩码（rope 臂用；底座 mask buffer 只有 1024²，外推需动态表）。"""
    key = (n, str(device))
    if key not in _MASK_CACHE:
        _MASK_CACHE[key] = torch.tril(torch.ones(n, n, device=device)).view(1, 1, n, n)
    return _MASK_CACHE[key]


def lr_at(step, total):
    """lr 日程（Book2 五件套同式）：warmup 段线性升，之后余弦降到 min_ratio*peak。"""
    if step <= WARMUP:
        return PEAK_LR * step / WARMUP
    prog = (step - WARMUP) / max(1, total - WARMUP)
    return PEAK_LR * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * prog)))


class TrainStream:
    """训练数据：tokens.bin 流首部按步号顺序切块（一次通过，构造上无重复 epoch）。

    第 k 步（1 起）消费 tokens[(k-1)*8192 : (k-1)*8192+8193]：x=前 8192 reshape (16,512)，y=左移一格。
    两臂按步号取到完全相同的批；步号确定性使断点续跑天然安全。
    """

    def __init__(self, path):
        self.arr = np.memmap(path, dtype=np.uint16, mode="r")     # 只读 memmap
        self.max_step = (len(self.arr) - 1) // TPB

    def batch(self, step, device):
        off = (step - 1) * TPB
        span = np.asarray(self.arr[off:off + TPB + 1], dtype=np.int64)       # (8193,)
        x = torch.from_numpy(span[:-1].reshape(BATCH, BLOCK)).to(device)     # (16,512)
        y = torch.from_numpy(span[1:].reshape(BATCH, BLOCK)).to(device)      # (16,512)
        return x, y


def build_eval_batches(device):
    """固定 eval 窗集（同窗 512 口径）：eval 池内固定种子抽 49 批 (16,512)，跨臂/跨步完全同集。"""
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
    """同窗固定窗 eval loss：49 批 fp32 前向，返回 (均值, 批间 se)。"""
    model.eval()
    losses = []
    for i in range(EVAL_BATCHES):
        _, loss = model(ex[i], ey[i])              # (16,512) -> 标量
        losses.append(loss.item())
    model.train()
    mean = sum(losses) / len(losses)
    se = (sum((l - mean) ** 2 for l in losses) / (len(losses) - 1)) ** 0.5 / math.sqrt(len(losses))
    return mean, se


@torch.no_grad()
def eval_extrap(model, device):
    """RoPE 臂 4× 外推 eval：eval.bin 首部不重叠 2048 窗 ×192（393,216 token，fp32）。

    返回 (loss, ppl, 分桶 loss, n_tokens)。分桶口径：窗内相对位置 [0:512)=训练见过 / [512:1024) /
    [1024:1536) / [1536:2048)——外推段的退化形态直接可读（ch4.9 的核心产出）。
    """
    model.eval()
    ev = np.memmap(EVAL_BIN, dtype=np.uint16, mode="r")
    losses, ntok = 0.0, 0
    bsum = [0.0] * (len(EXTRAP_BUCKETS) - 1)
    bcnt = [0] * (len(EXTRAP_BUCKETS) - 1)
    for b0 in range(0, EXTRAP_WINDOWS, EXTRAP_BATCH):
        nb = min(EXTRAP_BATCH, EXTRAP_WINDOWS - b0)
        xs = [ev[i * EXTRAP_N:(i + 1) * EXTRAP_N] for i in range(b0, b0 + nb)]
        ys = [ev[i * EXTRAP_N + 1:(i + 1) * EXTRAP_N + 1] for i in range(b0, b0 + nb)]
        x = torch.from_numpy(np.stack(xs).astype(np.int64)).to(device)       # (nb,2048)
        y = torch.from_numpy(np.stack(ys).astype(np.int64)).to(device)       # (nb,2048)
        logits, loss = model(x, y)                                           # (nb,2048,V)
        k = x.numel()
        losses += loss.item() * k
        ntok += k
        pos_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   y.reshape(-1), reduction="none").view(x.shape).mean(dim=0)  # (2048,)
        for bi in range(len(bsum)):
            lo, hi = EXTRAP_BUCKETS[bi], EXTRAP_BUCKETS[bi + 1]
            bsum[bi] += pos_loss[lo:hi].sum().item() * nb
            bcnt[bi] += (hi - lo) * nb
    model.train()
    avg = losses / ntok
    buckets = {f"pos[{EXTRAP_BUCKETS[i]}:{EXTRAP_BUCKETS[i+1]}]": round(bsum[i] / bcnt[i], 4)
               for i in range(len(bsum))}
    return avg, math.exp(avg), buckets, ntok


def _rope_attn_forward(self, x):
    """rope 臂的注意力前向（换装件）：与底座 CausalSelfAttention 同一零件（c_attn/c_proj 权重不变），
    仅在 Q/K 投影后施加 RoPE 旋转。输入 (B,n,C) -> 输出 (B,n,C)。v 不旋（式 11 只约束 QK 内积）。

    cos/sin 由 llama_slots.build_rope_cache 按 n 现算：(1,n,hd)——n 无上限，这正是外推不需要改代码的原因。
    """
    B, n, C = x.shape                                              # (B,n,C)
    hd = C // self.n_head                                          # 64
    q, k, v = self.c_attn(x).split(C, dim=2)                       # (B,n,3C) -> 各 (B,n,C)
    q = q.view(B, n, self.n_head, hd).transpose(1, 2)              # (B,h,n,hd)
    k = k.view(B, n, self.n_head, hd).transpose(1, 2)              # (B,h,n,hd)
    v = v.view(B, n, self.n_head, hd).transpose(1, 2)              # (B,h,n,hd)
    cos, sin = build_rope_cache(n, hd, ROPE_THETA, x.device)       # (1,n,hd) x2（llama_slots 正身）
    q, k = apply_rope(q, k, cos, sin)                              # 只旋 q/k -> (B,h,n,hd)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)                # (B,h,n,hd)·(B,h,hd,n) -> (B,h,n,n)
    att = att.masked_fill(causal_mask(n, x.device)[:, :, :n, :n] == 0, float("-inf"))
    att = self.attn_drop(F.softmax(att, dim=-1))                   # 行和=1（pdrop=0）
    y = att @ v                                                    # (B,h,n,n)·(B,h,n,hd) -> (B,h,n,hd)
    y = y.transpose(1, 2).contiguous().view(B, n, C)               # (B,n,h*hd)=(B,n,C)
    return self.resid_drop(self.c_proj(y))                         # (B,n,C)


class GPT2RoPE(GPT2):
    """rope 臂整机：底座构造后删 wpe、六处注意力换 RoPE 前向（改装第三例的实验落地）。

    构造 RNG 与 wpe 臂完全一致（super().__init__ 先建全套含 wpe，再删），因此 wte/blocks/ln_f 初始
    权重逐位相同——唯一变量 = 位置编码件（wpe 表 524,288 参数 vs RoPE 零参数，如实标注）。
    """

    def __init__(self, cfg: GPT2Config):
        super().__init__(cfg)
        self.wpe = None                                            # 删 wpe：参数与查表一并消失
        for blk in self.blocks:                                    # 6 处注意力换 RoPE 前向（权重不动）
            blk.attn.forward = types.MethodType(_rope_attn_forward, blk.attn)

    def forward(self, idx, targets=None):
        B, n = idx.shape                                           # (B,n)——无 wpe 查表，n 无硬上限
        x = self.drop(self.wte(idx))                               # (B,n,C)（只加词嵌入，位置进 attention 旋）
        for blk in self.blocks:                                    # 6 × (B,n,C)
            x = blk(x)
        x = self.ln_f(x)                                           # (B,n,C)
        logits = self.head(x)                                      # (B,n,C)·(C,V) -> (B,n,V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                  targets.reshape(-1))
        return logits, loss

    def n_params(self, non_embedding=False):
        """参数账：rope 臂无 wpe，non_embedding 只扣 wte（tied head 同存储不计）。"""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel()
        return n


def build_arm(arm):
    """构造两臂：同种子先建底座（RNG 消耗序列一致），rope 臂再换装位置编码件。"""
    torch.manual_seed(SEED)
    cfg = GPT2Config(vocab_size=VOCAB, n_positions=N_POSITIONS,
                     n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD)
    if arm == "wpe":
        return GPT2(cfg)                                           # 底座原样（学习式位置查表）
    if arm == "rope":
        return GPT2RoPE(cfg)                                       # 删 wpe + Q/K 旋转
    raise ValueError(arm)


def probe_wpe_wall(model, device):
    """N2 硬墙素材采集：wpe 臂在 2048 的两种报错（模型前向 assert + wpe 查表裸越界），try/except 记录。"""
    rec = {"n_positions": N_POSITIONS, "probe_len": EXTRAP_N}
    x = torch.randint(0, VOCAB, (1, EXTRAP_N), device=device)      # (1,2048)
    try:
        with torch.no_grad():
            model(x)                                               # 底座 forward 的 n<=block_size assert
        rec["forward_error"] = None                                # 不应到达（assert 必触发）
    except AssertionError as e:
        rec["forward_error"] = {"type": "AssertionError", "message": str(e)}
    except Exception as e:                                         # 防御：非预期异常也如实记录
        rec["forward_error"] = {"type": type(e).__name__, "message": str(e)[:300]}
    try:
        with torch.no_grad():
            # 裸查表探测放 CPU：MPS 的 embedding 越界不保证抛错（平台差异），CPU 口径才有确定性报错
            F.embedding(torch.arange(EXTRAP_N), model.wpe.weight.detach().cpu())   # (2048,) -> 查表（行数 1024）
        rec["wpe_lookup_error"] = None
    except IndexError as e:
        rec["wpe_lookup_error"] = {"type": "IndexError", "message": str(e)[:300]}
    except Exception as e:
        rec["wpe_lookup_error"] = {"type": type(e).__name__, "message": str(e)[:300]}
    rec["note"] = ("学习式 PE 的墙是硬墙：位置索引越过表长即崩（前向 assert / CPU 裸查表 IndexError；"
                   "MPS 裸查表不抛错系平台差异，故探测取 CPU 口径）；"
                   "RoPE 无查表、位置是连续函数——墙变软墙（ch7 主场）")
    return rec


def run_arm(arm, steps, eval_every, device, stream, ex, ey, report, out_json):
    """单臂训练（含断点续跑）：训练循环 = 五件套；每 eval 边界存 ckpt，臂完成即删。"""
    ckpt_path = os.path.join(OUT_DIR, f"ckpt_{arm}_{report['meta']['out_name']}.pt")
    m = build_arm(arm).to(device)
    n_total = m.n_params()
    n_nonemb = m.n_params(non_embedding=True)
    opt = torch.optim.AdamW(m.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=WDECL)

    train_losses, lrs, times, evals = [], [], [], []
    start_step, elapsed_train, elapsed_eval = 0, 0.0, 0.0
    resumed_from = None
    if os.path.exists(ckpt_path):   # ---- 断点续跑 ----
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
        x, y = stream.batch(step, device)           # (16,512)×2（按步号定位，两臂同批）
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
        if step % eval_every == 0 or step == steps:   # ---- eval 边界 ----
            t0 = time.perf_counter()
            el, se = eval_loss(m, ex, ey)
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed_eval += time.perf_counter() - t0
            evals.append({"step": step, "eval_loss": round(el, 5), "eval_se": round(se, 5),
                          "n_tokens": EVAL_BATCHES * TPB})
            print(f"  [{arm:5s}] [eval] step {step}: eval loss {el:.4f} ± {se:.4f}（401,408 token，fp32）",
                  flush=True)
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": step,
                        "train_losses": train_losses, "lrs": lrs, "times": times, "evals": evals,
                        "elapsed_train": elapsed_train, "elapsed_eval": elapsed_eval}, ckpt_path)

    steps_done = len(train_losses)
    steady = times[20:] if len(times) > 20 else times
    tail = train_losses[-min(TAIL, steps_done):]
    rec = {
        "arm": arm,
        "arm_def": ("GPT-2 底座原样（学习式 wpe 查表，n_positions=1024）" if arm == "wpe"
                    else "删 wpe + llama_slots RoPE 旋 Q/K（theta=10000，零位置参数）"),
        "config": {"d": N_EMBD, "L": N_LAYER, "h": N_HEAD, "V": VOCAB, "B": BATCH, "T": BLOCK,
                   "n_positions": N_POSITIONS, "rope_theta": ROPE_THETA},
        "params_total": n_total, "params_non_embedding": n_nonemb,
        "pos_params": (N_POSITIONS * N_EMBD) if arm == "wpe" else 0,
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

    # ---- 终局双口径：同窗已有；4× 外推（rope 出数 / wpe 出墙）----
    if steps_done >= steps and diverged_at is None:
        if arm == "rope":
            t0 = time.perf_counter()
            xl, xp, buckets, ntok = eval_extrap(m, device)
            if device.type == "mps":
                torch.mps.synchronize()
            rec["extrap_2048"] = {"loss": round(xl, 5), "ppl": round(xp, 3), "buckets": buckets,
                                  "n_tokens": ntok, "wall_sec": round(time.perf_counter() - t0, 1),
                                  "windows": EXTRAP_WINDOWS,
                                  "note": f"训练窗 T=512 的 4× 外推（{EXTRAP_WINDOWS}×2048=393,216 token，fp32）"}
            print(f"  [rope 外推] 2048 窗 loss {xl:.4f} ppl {xp:.2f} 分桶 {buckets}", flush=True)
        else:
            rec["wall_2048"] = probe_wpe_wall(m, device)
            print(f"  [wpe 硬墙] forward: {rec['wall_2048']['forward_error']} | "
                  f"查表: {rec['wall_2048']['wpe_lookup_error']}", flush=True)

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


def add_pair_summary(report):
    """双臂对照汇总（判读规则见 notes/07 §1 预注册）。"""
    a, b = report["arms"].get("wpe"), report["arms"].get("rope")
    if not (a and b):
        return
    report["pair_summary"] = {
        "eval_final_diff_wpe_minus_rope": round(a["eval_final"] - b["eval_final"], 5)
        if a["eval_final"] is not None and b["eval_final"] is not None else None,
        "params_diff_rope_minus_wpe": b["params_total"] - a["params_total"],
        "params_note": "RoPE 臂少 wpe 表 1024×512=524,288（-2.2%，位置参数 wpe→0）——两臂参数天然不对齐，如实标注",
        "extrap_note": ("rope 臂 extrap_2048.buckets 看 4× 外推退化形态；wpe 臂 wall_2048 = N2 硬墙素材"
                        "（越界即崩）；判据方向预注册见 notes/07 §1.2"),
    }


def main():
    ap = argparse.ArgumentParser(description="ch4 位置编码换件消融：wpe vs RoPE（训 512 测 2048）")
    ap.add_argument("--steps", type=int, default=1000, help="每臂步数（fast=1000 / full=5000）")
    ap.add_argument("--out-name", type=str, default=None,
                    help="输出名（缺省按步数自动取 fast(≤1500)/full）")
    ap.add_argument("--cells", type=str, default="wpe,rope", help="臂选择（wpe / rope）")
    ap.add_argument("--eval-every", type=int, default=500)
    args = ap.parse_args()
    out_name = args.out_name or ("fast" if args.steps <= 1500 else "full")
    arms = [c.strip() for c in args.cells.split(",")]
    for a in arms:
        if a not in ("wpe", "rope"):
            raise SystemExit(f"未知臂 {a}（可选 wpe / rope）")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    stream = TrainStream(TOKENS_BIN)
    if args.steps > stream.max_step:
        raise SystemExit(f"步数 {args.steps} 超出一次通过上限 {stream.max_step}")
    ex, ey = build_eval_batches(device)
    out_json = os.path.join(OUT_DIR, f"ablation_pe_{out_name}.json")

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
        "experiment": "ch4 换件消融：学习式 wpe（底座）vs RoPE（llama_slots 组件旋 Q/K）——单插槽改装",
        "module_tier": f"d={N_EMBD}/L={N_LAYER}/h={N_HEAD}/V={VOCAB}，B={BATCH}/T={BLOCK}（8192 tok/步）",
        "corpus": "log/book2-ch08/tokens.bin（sp-8k，306M token 流）",
        "optimizer": f"AdamW(betas=0.9,0.95,wd={WDECL}) + grad_clip 1.0（Book2 五件套）",
        "schedule": f"lr {PEAK_LR} 余弦降至 {PEAK_LR*MIN_LR_RATIO:.0e}，warmup={WARMUP}；两臂统一日程",
        "data_protocol": "顺序连续分块一次通过；两臂按步号取同批",
        "eval_protocol": (f"同窗：每 {args.eval_every} 步+终步，eval.bin 固定 49 窗 (16,512)=401,408 token fp32；"
                          f"终局加 4× 外推：rope 臂 {EXTRAP_WINDOWS}×2048=393,216 token 分桶 fp32；"
                          f"wpe 臂 2048 越界报错采集（N2 硬墙素材）"),
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
        print(f"\n===== 臂 {arm}（{args.steps} 步） =====", flush=True)
        run_arm(arm, args.steps, args.eval_every, device, stream, ex, ey, report, out_json)
    add_pair_summary(report)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[全部完成] 总耗时 {(time.perf_counter()-t_all)/60:.1f} min -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
