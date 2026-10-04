# extrapolate.py —— Book3 ch7 外推核心实验：2k 窗 mini 训练 + 四干预免重训 eval（none/PI/NTK/YaRN-lite）
# 用途：(1) 训一个 2k 窗 RoPE mini（llama_slots.LLaMA：d=256/L=4/h=4/h_kv=4/V=8192，7.4M，
#           B=8/T=2048，五件套配方，tokens.bin 顺序分块一次通过）；
#       (2) 训练后**免重训**只改 RoPE 参数做四干预 eval（s=4，2k→8k 名义窗）：
#           none      原样 RoPE（theta=10000，位置原尺度）——直接外推对照
#           pi        位置缩放（PI 式 4：位置 m→m/s，等价 inv_freq/=s；HF rope_type=linear 同式）
#           ntk       base 放大（YaRN 式 15 转述 bloc97：b'=b·s^(d/(d-2))，d=head_dim=64——
#                     高频外推、低频插值的「改 base 不改尺度」）
#           yarn_lite ntk 的 base + 注意力温度（YaRN 式 22：√(1/t)=0.1·ln s+1；HF get_mscale
#                     口径 = mscale 乘 cos/sin，q/k 各乘一次 → logits×mscale²=1/t）
#           （YaRN 本体还有 NTK-by-parts 的波长分段斜坡——本实验为「免重训单参数对照」取其
#            base+温度两件的 lite 版，分段斜坡归正文讲解；命名如实带 _lite 后缀）
#       (3) eval 三窗长 2048（同窗）/8192（4× 目标窗）/16384（二次越界）× 位置分桶长文 ppl，
#           全部 fp32；同一 heldout 段、同 token 预算（各窗长均 393,216 token）。
# 运行方式：python extrapolate.py [--steps 2000] [--out-name fast] [--eval-every 500]
#           两档位：fast=2000 步（~12 min）/ full=5000 步（~31 min）；训练断点续跑。
# 产物（log/book3-ch07/，不入库）：extrapolate_{out}.json + curve_train_{out}.csv
#           + mini2k_{out}.pt（终态 ckpt：model+config，供 passkey.py 与后续复用）。
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "00-feasibility")))
from llama_slots import LLaMA, LLaMAConfig          # noqa: E402  本册插槽组件库（整机正身）

# ---------------- 固定口径 ----------------
SEED = 20261002
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book3-ch07")
TOKENS_BIN = os.path.join(REPO_ROOT, "log", "book2-ch08", "tokens.bin")
EVAL_BIN = os.path.join(REPO_ROOT, "log", "book2-ch08", "eval.bin")     # 1.2M heldout（独立文件，天然互斥）

MINI_CFG = dict(vocab_size=8192, hidden_size=256, intermediate_size=704,
                num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
                max_position_embeddings=2048, rope_theta=10000.0)       # 探针 G 同 config（7,407,872 参数）
BATCH, BLOCK = 8, 2048                # 16,384 tok/步（探针 G 实测 ~0.38 s/step）
TPB = BATCH * BLOCK
PEAK_LR, MIN_LR_RATIO, WARMUP, WDECL, TAIL = 1e-3, 0.1, 100, 0.1, 300

SCALE = 4                              # s：2k→8k 名义扩窗（四干预共用）
HEAD_DIM = 64                          # d=256/h=4
ROPE_BASE = 10000.0
NTK_BASE = ROPE_BASE * SCALE ** (HEAD_DIM / (HEAD_DIM - 2))   # YaRN 式 15：b·s^(d/(d-2))
YARN_MSCALE = 0.1 * math.log(SCALE) + 1                        # YaRN 式 22：√(1/t)=0.1·ln s+1（HF get_mscale）

# 四干预定义（免重训，只改 RoPE 参数；eval 时逐干预重建 cos/sin）
INTERVENTIONS = {
    "none":      {"pos_scale": 1.0, "base": ROPE_BASE, "mscale": 1.0,
                  "def": "原样 RoPE（直接外推对照）"},
    "pi":        {"pos_scale": 1.0 / SCALE, "base": ROPE_BASE, "mscale": 1.0,
                  "def": f"PI 位置缩放 s={SCALE}：位置 m→m/s（等价 inv_freq/=s；论文式 4/HF linear）"},
    "ntk":       {"pos_scale": 1.0, "base": NTK_BASE, "mscale": 1.0,
                  "def": f"NTK base 放大：b'=b·s^(d/(d-2))={NTK_BASE:.1f}（YaRN 式 15 转述 bloc97）"},
    "yarn_lite": {"pos_scale": 1.0, "base": NTK_BASE, "mscale": YARN_MSCALE,
                  "def": f"YaRN-lite：NTK base + 温度 mscale=0.1·ln s+1={YARN_MSCALE:.4f} 乘 cos/sin"
                         f"（式 22 √(1/t)；HF get_mscale 口径；不含 by-parts 波长分段）"},
}

# eval 窗长协议：各窗长固定批大小与窗数（token 预算一律 393,216——跨窗长可比）
EVAL_PLAN = {2048: {"windows": 192, "batch": 8, "buckets": [0, 512, 1024, 2048]},
             8192: {"windows": 48, "batch": 4, "buckets": [0, 512, 1024, 2048, 4096, 8192]},
             16384: {"windows": 24, "batch": 1, "buckets": [0, 1024, 2048, 4096, 8192, 16384]}}


def lr_at(step, total):
    if step <= WARMUP:
        return PEAK_LR * step / WARMUP
    prog = (step - WARMUP) / max(1, total - WARMUP)
    return PEAK_LR * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * prog)))


class TrainStream:
    """tokens.bin 流首部按步号顺序切块（B=8/T=2048，16,384 tok/步；断点续跑天然安全）。

    full 5000 步 = 81.9M token < 306M 流长，零 epoch 重复；eval 用独立文件 eval.bin，天然互斥。
    （Book2 的「训练只读流首部 ≤40M」是 scaling 族曲线可比性口径，本章为外推机制实验，不受该约束。）
    """

    def __init__(self, path):
        self.arr = np.memmap(path, dtype=np.uint16, mode="r")
        self.max_step = (len(self.arr) - 1) // TPB

    def batch(self, step, device):
        off = (step - 1) * TPB
        span = np.asarray(self.arr[off:off + TPB + 1], dtype=np.int64)       # (16385,)
        x = torch.from_numpy(span[:-1].reshape(BATCH, BLOCK)).to(device)     # (8,2048)
        y = torch.from_numpy(span[1:].reshape(BATCH, BLOCK)).to(device)      # (8,2048)
        return x, y


def rope_cache_variant(n, head_dim, base, device, pos_scale=1.0, mscale=1.0):
    """四干预共用的 RoPE cos/sin 查表（llama_slots.build_rope_cache 的参数化推广）。

    inv_freq_i = 1/base^(2i/head_dim)；t = pos_scale·arange(n)（PI 的位置缩放就在这一步）；
    emb = cat(freqs, freqs)（rotate_half 约定需整维表）；mscale 乘 cos/sin（YaRN 温度的
    「零开销」实现——q/k 各乘一次 → logits ×mscale²）。返回 (1,n,hd) fp32 各一。
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(n, dtype=torch.float32, device=device) * pos_scale      # (n,)
    freqs = torch.outer(t, inv_freq)                                          # (n,hd/2)
    emb = torch.cat((freqs, freqs), dim=-1)                                   # (n,hd)
    return (emb.cos() * mscale)[None], (emb.sin() * mscale)[None]             # (1,n,hd) x2


def rope_cache_for(intervention, n, device, head_dim=HEAD_DIM):
    spec = INTERVENTIONS[intervention]
    return rope_cache_variant(n, head_dim, spec["base"], device,
                              pos_scale=spec["pos_scale"], mscale=spec["mscale"])


def forward_with_rope(model, idx, cos, sin, targets=None):
    """显式 cos/sin 注入的前向（绕过 backbone 内置 rope 构建——干预只发生在 eval 侧，权重不动）。

    idx (B,n) -> logits (B,n,V)；逐层 x = layer(x, cos, sin)（llama_slots.LLaMADecoderLayer 原生接口）。
    """
    x = model.model.embed_tokens(idx)                                    # (B,n,C)
    for layer in model.model.layers:                                     # 4 × (B,n,C)
        x = layer(x, cos, sin)
    x = model.model.norm(x)                                              # (B,n,C)
    logits = model.lm_head(x)                                            # (B,n,C)·(C,V) -> (B,n,V)
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))
    return logits, loss


@torch.no_grad()
def eval_windows(model, device, n, windows, batch, buckets, intervention=None):
    """heldout 首部不重叠窗一次通过：token 加权 loss -> ppl + 位置分桶 loss（全 fp32）。

    intervention=None 用模型原生前向（训练中途 eval）；否则注入该干预的 cos/sin（免重训）。
    """
    model.eval()
    ev = np.memmap(EVAL_BIN, dtype=np.uint16, mode="r")
    losses, ntok = 0.0, 0
    bsum = [0.0] * (len(buckets) - 1)
    bcnt = [0] * (len(buckets) - 1)
    for b0 in range(0, windows, batch):
        nb = min(batch, windows - b0)
        xs = [ev[i * n:(i + 1) * n] for i in range(b0, b0 + nb)]
        ys = [ev[i * n + 1:(i + 1) * n + 1] for i in range(b0, b0 + nb)]
        x = torch.from_numpy(np.stack(xs).astype(np.int64)).to(device)      # (nb,n)
        y = torch.from_numpy(np.stack(ys).astype(np.int64)).to(device)      # (nb,n)
        if intervention is None:
            logits, loss = model(x, y)
        else:
            cos, sin = rope_cache_for(intervention, n, device)
            logits, loss = forward_with_rope(model, x, cos, sin, y)
        k = x.numel()
        losses += loss.item() * k
        ntok += k
        pos_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   y.reshape(-1), reduction="none").view(x.shape).mean(dim=0)  # (n,)
        for bi in range(len(bsum)):
            lo, hi = buckets[bi], min(buckets[bi + 1], n)
            if hi <= lo:
                continue
            bsum[bi] += pos_loss[lo:hi].sum().item() * nb
            bcnt[bi] += (hi - lo) * nb
    model.train()
    avg = losses / ntok
    out = {f"pos[{buckets[i]}:{buckets[i+1]}]": round(bsum[i] / max(1, bcnt[i]), 4) for i in range(len(bsum))}
    return avg, math.exp(avg), out, ntok


def main():
    ap = argparse.ArgumentParser(description="ch7 外推核心：2k 窗 mini 训练 + 四干预免重训 eval")
    ap.add_argument("--steps", type=int, default=2000, help="训练步数（fast=2000 / full=5000）")
    ap.add_argument("--out-name", type=str, default=None)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--init-from", type=str, default=None,
                    help="可选：从已有 ckpt 起续（如探针 G 的 probe_g_mini_2k.pt）；缺省从头训")
    args = ap.parse_args()
    out_name = args.out_name or ("fast" if args.steps <= 3000 else "full")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    cfg = LLaMAConfig(**MINI_CFG)
    model = LLaMA(cfg).to(device)
    n_params = model.n_params()
    hand = (MINI_CFG["vocab_size"] * MINI_CFG["hidden_size"] * 2                        # untied 两份
            + MINI_CFG["num_hidden_layers"] * (
                2 * MINI_CFG["hidden_size"] ** 2
                + 2 * MINI_CFG["hidden_size"] * MINI_CFG["num_key_value_heads"] * HEAD_DIM
                + 3 * MINI_CFG["hidden_size"] * MINI_CFG["intermediate_size"]
                + 2 * MINI_CFG["hidden_size"])
            + MINI_CFG["hidden_size"])
    assert n_params == hand == 7_407_872, f"参数对账失败：{n_params} != {hand}"
    print(f"[config] 2k 窗 RoPE mini：{n_params:,} 参数（d=256/L=4/h=4/V=8192），设备 {device}", flush=True)

    stream = TrainStream(TOKENS_BIN)
    if args.steps > stream.max_step:
        raise SystemExit(f"步数 {args.steps} 超出一次通过上限 {stream.max_step}")
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=WDECL)

    out_json = os.path.join(OUT_DIR, f"extrapolate_{out_name}.json")
    ckpt_path = os.path.join(OUT_DIR, f"ckpt_train_{out_name}.pt")
    curve_path = os.path.join(OUT_DIR, f"curve_train_{out_name}.csv")
    train_losses, lrs, times, evals = [], [], [], []
    start_step, elapsed_train, elapsed_eval, resumed_from, init_from = 0, 0.0, 0.0, None, None
    if os.path.exists(ckpt_path):        # ---- 断点续跑 ----
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        train_losses, lrs, times = ck["train_losses"], ck["lrs"], ck["times"]
        evals = ck["evals"]
        start_step, elapsed_train, elapsed_eval = ck["step"], ck["elapsed_train"], ck["elapsed_eval"]
        resumed_from = start_step
        print(f"[续跑] 自 step {start_step + 1}", flush=True)
    elif args.init_from:                 # ---- 可选热起（如探针 G ckpt；lr 日程重启，须如实标注） ----
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        init_from = args.init_from
        print(f"[热起] 自 {args.init_from} 载入权重（lr 日程按本轮 {args.steps} 步重启）", flush=True)

    report = {"meta": {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "device": str(device),
        "torch": torch.__version__, "out_name": out_name,
        "experiment": "ch7 外推核心：2k 窗 mini 训练 + 四干预免重训 eval（none/PI/NTK/YaRN-lite）",
        "model": {**MINI_CFG, "params": n_params, "head_dim": HEAD_DIM},
        "corpus": "log/book2-ch08/tokens.bin（sp-8k，306M 流；顺序分块一次通过，零 epoch 重复）",
        "train_protocol": (f"B={BATCH}/T={BLOCK}（{TPB} tok/步），{args.steps} 步 = "
                           f"{args.steps*TPB/1e6:.1f}M token；AdamW(0.9,0.95,wd=0.1)+clip1.0+"
                           f"lr 1e-3 余弦降至 1e-4+warmup 100（五件套）；train bf16 autocast"),
        "scale": SCALE,
        "interventions": {k: {"pos_scale": v["pos_scale"], "base": round(v["base"], 2),
                              "mscale": round(v["mscale"], 5), "def": v["def"]}
                          for k, v in INTERVENTIONS.items()},
        "intervention_note": ("免重训：干预只发生在 eval 侧的 cos/sin 重建，权重一步不动"
                             "（「训练后租窗」的实验显形）；YaRN-lite 不含 by-parts 波长分段（正文讲解件）"),
        "eval_protocol": ("heldout=eval.bin（独立文件，与训练流互斥）；各窗长不重叠窗首部一次通过，"
                          "token 预算一律 393,216（2048×192 / 8192×48 / 16384×24）；fp32；位置分桶随报"),
        "precision": "train bf16 autocast / eval fp32",
        "ckpt": f"终态 ckpt = mini2k_{out_name}.pt（供 passkey.py）；断点续跑 ckpt_train_{out_name}.pt",
        "init_from": init_from, "argv": " ".join(sys.argv[1:]),
    }}
    if os.path.exists(out_json):
        with open(out_json, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("meta", {}).get("argv") == report["meta"]["argv"]:
            report = old                                    # 同命令续跑：载入旧报告
            print(f"[续跑] 已有报告载入", flush=True)

    # ---- (1) 训练 2k 窗 ----
    model.train()
    t_train = time.perf_counter()
    for step in range(start_step + 1, args.steps + 1):
        lr = lr_at(step, args.steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = stream.batch(step, device)               # (8,2048)×2
        t0 = time.perf_counter()
        with torch.autocast(device_type="mps", dtype=torch.bfloat16) if device.type == "mps" \
                else torch.autocast("cpu", enabled=False):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device.type == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
        elapsed_train += times[-1]
        lv = loss.item()
        train_losses.append(lv)
        lrs.append(lr)
        if not math.isfinite(lv) or lv > 100.0:
            print(f"[发散] step {step}: {lv}", flush=True)
            break
        if step % 100 == 0 or step == 1:
            print(f"[train] step {step:5d}/{args.steps} loss {lv:.4f} lr {lr:.2e} "
                  f"({elapsed_train/(step - start_step + 1):.3f}s/step)", flush=True)
        if step % args.eval_every == 0 or step == args.steps:   # 中途 eval：同窗 2048（none 口径）
            t0 = time.perf_counter()
            p = EVAL_PLAN[2048]
            el, ep, _, ntok = eval_windows(model, device, 2048, p["windows"], p["batch"],
                                           p["buckets"], intervention=None)
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed_eval += time.perf_counter() - t0
            evals.append({"step": step, "eval_loss_2k": round(el, 5), "ppl_2k": round(ep, 3),
                          "n_tokens": ntok})
            print(f"[eval] step {step}: 2k 窗 loss {el:.4f} ppl {ep:.2f}（393,216 tok，fp32）",
                  flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "train_losses": train_losses, "lrs": lrs, "times": times, "evals": evals,
                        "elapsed_train": elapsed_train, "elapsed_eval": elapsed_eval}, ckpt_path)
    train_sec = time.perf_counter() - t_train
    steady = times[20:] if len(times) > 20 else times
    print(f"[train] {len(train_losses)} 步完成 {train_sec:.0f}s（稳态 {sum(steady)/len(steady):.4f}s/step）",
          flush=True)
    with open(curve_path, "w", newline="", encoding="utf-8") as cf:   # 全量曲线（含续跑段）
        w = csv.writer(cf)
        w.writerow(["step", "train_loss", "lr"])
        for i, (lv, lr) in enumerate(zip(train_losses, lrs), 1):
            w.writerow([i, f"{lv:.6g}", f"{lr:.6g}"])

    # ---- (2) 四干预 × 三窗长 eval（免重训） ----
    table = {}
    for iv_name in INTERVENTIONS:
        table[iv_name] = {}
        for n, p in EVAL_PLAN.items():
            t0 = time.perf_counter()
            el, ep, buckets, ntok = eval_windows(model, device, n, p["windows"], p["batch"],
                                                 p["buckets"], intervention=iv_name)
            if device.type == "mps":
                torch.mps.synchronize()
            wall = time.perf_counter() - t0
            table[iv_name][str(n)] = {"loss": round(el, 5), "ppl": round(ep, 3), "buckets": buckets,
                                      "n_tokens": ntok, "wall_sec": round(wall, 1)}
            print(f"[四干预] {iv_name:9s} n={n:5d}  loss {el:.4f}  ppl {ep:9.2f}  "
                  f"({wall:.0f}s, {ntok:,} tok)", flush=True)

    # ---- (3) 产物落盘 ----
    final_ckpt = os.path.join(OUT_DIR, f"mini2k_{out_name}.pt")
    torch.save({"model": model.state_dict(), "config": MINI_CFG, "steps": len(train_losses),
                "seed": SEED, "out_name": out_name}, final_ckpt)
    if len(train_losses) >= args.steps and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    report["train"] = {
        "steps_done": len(train_losses), "steps_planned": args.steps, "resumed_from_step": resumed_from,
        "tokens_seen": len(train_losses) * TPB,
        "tail300_train_loss": round(sum(train_losses[-min(TAIL, len(train_losses)):])
                                    / min(TAIL, len(train_losses)), 5),
        "first_loss": round(train_losses[0], 5) if train_losses else None,
        "last_loss": round(train_losses[-1], 5) if train_losses else None,
        "sec_per_step_steady": round(sum(steady) / len(steady), 4),
        "train_wall_sec": round(train_sec, 1), "eval_overhead_sec": round(elapsed_eval, 1),
        "mid_evals": evals,
    }
    report["intervention_eval"] = table
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[完成] 训练 {len(train_losses)} 步 + 四干预 eval -> {out_json}\n终态 ckpt -> {final_ckpt}",
          flush=True)


if __name__ == "__main__":
    main()
