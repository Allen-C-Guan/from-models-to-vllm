# run_family.py —— Book2 ch8 Kaplan 单曲线族：固定 D=40M token，变 N（非嵌入参数）5 点，一次通过不重复
# 所属章节：Book2 第 8 章 §8.3（ch08-ScalingLaws.md 表 8.2/图 8.3 数据源）
# 设计书：notes/01-实验可行性.md §4(i)。
#   * 五规模 N∈{1M,3M,7M,15M,20M}（非嵌入参数，脚本打印实数），(n_layer,n_embd) 自配、head_dim 统一 64；
#   * 每点 D=40M token 一次通过（顺序连续分块，构造上无重复 epoch——冒烟档 32-epoch 教训的正式档修正）；
#   * b32×s256、AdamW(0.9,0.95,wd=0.1)+grad-clip 1.0、lr 1e-3 余弦降至 1e-4（默认无 warmup）——
#     各规模统一日程：这是 Kaplan「固定日程跨规模」口径的缩小版（原论文各模型共用同一 LR 日程，只在
#     终段衰减；本实验为余弦全衰缩短训版），单变量 = N。注意：无 warmup 口径下深模型（9L+）早期
#     优化受损、L(N) 可倒挂（实测见 notes/07）；--warmup 200 为各规模统一加 warmup 的修正口径；
#   * 训练 bf16 autocast（MPS），eval fp32（结果数字进拟合，用全精度报数）；
#   * 每点每 500 步在 eval 池取 49×8192=401,408 token 算 eval loss（≥40 万，设计书 §4 eval 协议）；
#   * 断点续跑：每 eval 边界存 ckpt（模型+优化器+曲线），重启后跳过已完成点、续跑未完成点。
#   * 模型定义直接 import 自 ch04/warmup_ablation.py（from warmup_ablation import GPT, Block）——单轨依赖链
#     ch4 -> ch8 的族工厂（与 ch04/surgery.py 同型导入；此前为逐行同构复制，批二修订 2026-10-03 改回导入）。
# 运行方式：python run_family.py [--points all|1m,3m,...] [--steps N] [--warmup N] [--out-name NAME] [--probe-tok-std]
#       --out-name 缺省时随 warmup 自动取 family_fast（=200）/family_nowarmup（其余），防裸跑覆写正式族文件。
# 产物：log/book2-ch08/{family_fast.json, curve_<name>.csv/.npy, ckpt_<name>.pt(跑完即删), tok_std.json}

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# ---------------- 固定口径 ----------------
SEED = 20261002          # 全书统一种子（与 ch04/00-feasibility 同源）
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book2-ch08")
TOKENS_BIN = os.path.join(OUT_DIR, "tokens.bin")   # 训练池（data_prep.py 产物，流首部）
EVAL_BIN = os.path.join(OUT_DIR, "eval.bin")       # eval 池（流末尾留出，与训练段互斥）

VOCAB = 8192             # sp-8k
BLOCK = 256              # 序列长 s
BATCH = 32               # 批大小 b（b32×s256=8192 token/步）
TPB = BATCH * BLOCK      # tokens per step
D_TOKENS = 40_000_000    # Kaplan 口径固定 D（名义 40M）
EVAL_EVERY = 500
EVAL_BATCHES = 49        # 49×8192 = 401,408 token ≥ 40 万（设计书 §4 eval 协议）
PEAK_LR = 1e-3
MIN_LR_RATIO = 0.1       # 余弦降到 0.1×peak = 1e-4
WARMUP = 0               # 无 warmup（口径见文件头；ch4 §4.5 证明 pre-LN 无 warmup 不发散）
WDECL = 0.1

# 五规模：名义 N（非嵌入）→ (n_layer, n_embd, n_head)，head_dim 全族统一 64
# 实数 N = n_layer*(12C^2+13C)+2C（每块 attn 4C^2+9C? 精确口径见 GPT.n_params 实测打印）：
#   1m:  5L×128d×2h  ->   991,616
#   3m:  7L×192d×3h  -> 3,114,432
#   7m:  9L×256d×4h  -> 7,108,352
#   15m: 12L×320d×5h -> 14,796,160
#   20m: 11L×384d×6h -> 19,519,872
FAMILY = [
    ("1m",  1_000_000, 5, 128, 2),
    ("3m",  3_000_000, 7, 192, 3),
    ("7m",  7_000_000, 9, 256, 4),
    ("15m", 15_000_000, 12, 320, 5),
    ("20m", 20_000_000, 11, 384, 6),
]


# ---------------- 模型：直接复用 ch04/warmup_ablation.py 的 GPT/Block（单轨依赖链 ch4 -> ch8） ----------------
# 块实现即 ch4「手术②」的参数化 GPT 工厂（pre-LN + ln_f + tied lm_head，std=0.02 初始化，残差出端投影再乘
# 0.02/sqrt(2*n_layer)）；族内只用 ln_position="pre"（默认），三插槽 n_layer/n_embd/n_head 可配。
# ch04 侧默认 vocab=65/block_size=256 是其 shakespeare 语料口径，本族调用时显式传 vocab=8192/block_size=256。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ch04"))
from warmup_ablation import GPT, Block  # noqa: E402,F401  族模型工厂（Block 备行级引用/调试）


# ---------------- 日程与数据 ----------------
def lr_at(step, peak, warmup, total, min_ratio=MIN_LR_RATIO):
    """lr 日程（与 ch04 同式）：warmup 段线性升（本实验 warmup=0），之后余弦降到 min_ratio*peak。"""
    if warmup > 0 and step <= warmup:
        return peak * step / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


class TrainStream:
    """训练数据：tokens.bin 流首部按步顺序切块（一次通过，构造上无重复）。

    第 k 步（1 起）消费 tokens[(k-1)*8192 : (k-1)*8192+8193]：x=前 8192 reshape(32,256)，y=左移一格。
    批内 32 行是流中连续铺开的 256-token 块；每个 token 恰作一次预测目标（块首 token 只当上下文）。
    """

    def __init__(self, path):
        self.arr = np.memmap(path, dtype=np.uint16, mode="r")
        self.max_step = (len(self.arr) - 1) // TPB

    def batch(self, step, device):
        off = (step - 1) * TPB
        span = np.asarray(self.arr[off:off + TPB + 1], dtype=np.int64)      # (8193,)
        x = torch.from_numpy(span[:-1].reshape(BATCH, BLOCK)).to(device)    # (32,256)
        y = torch.from_numpy(span[1:].reshape(BATCH, BLOCK)).to(device)     # (32,256)
        return x, y


def build_eval_batches(device):
    """固定 eval 批集：eval 池内固定种子抽 49 个 (32,256) 窗（跨点/跨步完全同集，可比）。"""
    ev = np.memmap(EVAL_BIN, dtype=np.uint16, mode="r")
    gen = torch.Generator().manual_seed(SEED)      # 窗位固定（同 ch04 eval 口径：固定种子生成器）
    offs = torch.randint(0, len(ev) - BLOCK - 1, (EVAL_BATCHES * BATCH,), generator=gen).numpy()
    xs, ys = [], []
    for o in offs:
        xs.append(np.asarray(ev[o:o + BLOCK], dtype=np.int64))
        ys.append(np.asarray(ev[o + 1:o + BLOCK + 1], dtype=np.int64))
    x = torch.from_numpy(np.stack(xs).reshape(EVAL_BATCHES, BATCH, BLOCK)).to(device)  # (49,32,256)
    y = torch.from_numpy(np.stack(ys).reshape(EVAL_BATCHES, BATCH, BLOCK)).to(device)
    return x, y


@torch.no_grad()
def eval_loss(model, ex, ey):
    """eval loss（fp32 前向，无 autocast——报数进拟合用全精度）：49 批均值 = 401,408 token 池化均值。"""
    model.eval()
    losses = []
    for i in range(EVAL_BATCHES):
        _, loss = model(ex[i], ey[i])            # (32,256) -> 标量
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses), sum(losses) / len(losses) / math.sqrt(len(losses))


# ---------------- 单点训练（含断点续跑） ----------------
def run_point(name, n_nominal, n_layer, n_embd, n_head, steps, device, out_name, train_stream,
              warmup=WARMUP):
    ckpt_path = os.path.join(OUT_DIR, f"ckpt_{name}.pt")
    torch.manual_seed(SEED)   # 每点重置（init 流跨点一致，唯一变量 = 架构规模；ch04 同口径）
    model = GPT(n_layer=n_layer, n_embd=n_embd, n_head=n_head,
                vocab=VOCAB, block_size=BLOCK).to(device)
    n_total = model.n_params()
    n_nonemb = model.n_params(non_embedding=True)
    print(f"\n===== 点 {name}: {n_layer}L×{n_embd}d×{n_head}h（head_dim={n_embd//n_head}） "
          f"非嵌入参数实数 {n_nonemb:,}（名义 {n_nominal:,}，偏差 {n_nonemb/n_nominal-1:+.1%}）/ "
          f"总参数 {n_total:,} =====", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=WDECL)
    ex, ey = build_eval_batches(device)

    train_losses, lrs, evals = [], [], []
    start_step, elapsed_train, elapsed_eval = 0, 0.0, 0.0
    if os.path.exists(ckpt_path):   # ---- 断点续跑：从最近 eval 边界恢复 ----
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        train_losses, lrs, evals = ck["train_losses"], ck["lrs"], ck["evals"]
        start_step, elapsed_train, elapsed_eval = ck["step"], ck["elapsed_train"], ck["elapsed_eval"]
        print(f"  [续跑] 自 step {start_step + 1}（已完成 {start_step} 步，eval {len(evals)} 次）", flush=True)

    diverged_at, reason = None, None
    t_point = time.perf_counter()
    for step in range(start_step + 1, steps + 1):
        lr = lr_at(step, PEAK_LR, warmup, steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = train_stream.batch(step, device)
        t0 = time.perf_counter()
        with torch.autocast(device_type="mps", dtype=torch.bfloat16) if device == "mps" else torch.autocast("cpu", enabled=False):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device == "mps":
            torch.mps.synchronize()
        elapsed_train += time.perf_counter() - t0
        lv = loss.item()
        train_losses.append(lv)
        lrs.append(lr)
        if not math.isfinite(lv) or lv > 100.0:
            diverged_at, reason = step, ("non-finite" if not math.isfinite(lv) else f"loss={lv:.1f}>100")
            print(f"  [发散] step {step}: {reason}", flush=True)
            break
        if step % 100 == 0 or step == 1:
            print(f"  step {step:5d}/{steps} train {lv:.4f} lr {lr:.2e} "
                  f"({elapsed_train/step:.3f}s/step 累计)", flush=True)
        if step % EVAL_EVERY == 0 or step == steps:
            t0 = time.perf_counter()
            el, _ = eval_loss(model, ex, ey)
            if device == "mps":
                torch.mps.synchronize()
            elapsed_eval += time.perf_counter() - t0
            evals.append({"step": step, "eval_loss": round(el, 5),
                          "n_tokens": EVAL_BATCHES * TPB})
            print(f"  [eval] step {step}: eval loss {el:.4f}（401,408 token，fp32）", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "train_losses": train_losses, "lrs": lrs, "evals": evals,
                        "elapsed_train": elapsed_train, "elapsed_eval": elapsed_eval,
                        "config": {"n_layer": n_layer, "n_embd": n_embd, "n_head": n_head}},
                       ckpt_path)

    wallclock = time.perf_counter() - t_point
    steps_done = len(train_losses)
    tail = train_losses[-min(100, steps_done):]
    result = {
        "name": name, "N": n_nonemb, "N_nominal": n_nominal, "N_total": n_total,
        "n_layer": n_layer, "n_embd": n_embd, "n_head": n_head, "head_dim": n_embd // n_head,
        "steps": steps_done, "steps_planned": steps,
        "d_tokens_processed": steps_done * TPB,
        "one_pass": "顺序连续分块（第 k 步消费流首部第 (k-1)*8192 起 8192 token），构造上零重复",
        "final_train_loss": round(sum(tail) / len(tail), 5),
        "final_train_loss_def": "最后 100 步 train loss 均值",
        "last_step_loss": round(train_losses[-1], 5) if train_losses else None,
        "min_train_loss": round(min(train_losses), 5) if train_losses else None,
        "eval_losses": evals,
        "eval_final": evals[-1]["eval_loss"] if evals else None,
        "eval_tokens_per_eval": EVAL_BATCHES * TPB,
        "diverged": diverged_at is not None, "diverged_at_step": diverged_at, "divergence_reason": reason,
        "wallclock": round(wallclock, 1), "wallclock_sec_train_only": round(elapsed_train, 1),
        "sec_per_step": round(elapsed_train / max(1, steps_done), 4),
        "precision": "train bf16 autocast / eval fp32",
    }
    # 曲线落盘 + 清 ckpt（点已完成，结果自含）
    stem = os.path.join(OUT_DIR, f"curve_{name}")
    with open(stem + ".csv", "w", encoding="utf-8") as f:
        f.write("step,train_loss,lr\n")
        for i, (lv, lr) in enumerate(zip(train_losses, lrs), 1):
            f.write(f"{i},{lv:.6g},{lr:.6g}\n")
    np.save(stem + ".npy", np.column_stack([np.arange(1, steps_done + 1), train_losses, lrs]))
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"  [点完成] steps {steps_done} | final_train {result['final_train_loss']} | "
          f"eval_final {result['eval_final']} | {result['sec_per_step']}s/step | "
          f"wall {wallclock/60:.1f} min", flush=True)
    del model, opt, ex, ey
    if device == "mps":
        torch.mps.empty_cache()
    return result


# ---------------- tok_std 探针：eval 池首块逐 token loss std（定 eval token 需求） ----------------
def probe_tok_std(device, train_stream):
    """设计书 §4 eval 协议底数：最小 1M 模型 500 步后，eval 池首块逐 token loss 的 std。
    用于校验「≥40 万 eval token => se<0.005」是否在本语料成立（冒烟档 shakespeare 实测 3.14）。"""
    name, n_nominal, n_layer, n_embd, n_head = FAMILY[0]
    print(f"\n[tok_std 探针] {n_layer}L×{n_embd}d×{n_head}h 训 500 步（正式点 1m 的前 500 步同口径）", flush=True)
    torch.manual_seed(SEED)
    model = GPT(n_layer=n_layer, n_embd=n_embd, n_head=n_head,
                vocab=VOCAB, block_size=BLOCK).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=WDECL)
    model.train()
    t0 = time.perf_counter()
    for step in range(1, 501):
        lr = lr_at(step, PEAK_LR, WARMUP, 500)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = train_stream.batch(step, device)
        with torch.autocast(device_type="mps", dtype=torch.bfloat16) if device == "mps" else torch.autocast("cpu", enabled=False):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    if device == "mps":
        torch.mps.synchronize()
    sec = time.perf_counter() - t0

    @torch.no_grad()
    def per_token_losses(x, y):
        model.eval()
        logits, _ = model(x)                                        # (B,T,V) fp32
        pt = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), reduction="none")
        model.train()
        return pt                                                   # (B*T,)

    ev = np.memmap(EVAL_BIN, dtype=np.uint16, mode="r")
    first = np.asarray(ev[:TPB + 1], dtype=np.int64)
    x1 = torch.from_numpy(first[:-1].reshape(BATCH, BLOCK)).to(device)
    y1 = torch.from_numpy(first[1:].reshape(BATCH, BLOCK)).to(device)
    pt1 = per_token_losses(x1, y1)                                  # 首 8192 token（"首块"口径）
    ex, ey = build_eval_batches(device)                             # 正式 eval 集 49 窗
    pt2 = torch.cat([per_token_losses(ex[i], ey[i]) for i in range(EVAL_BATCHES)])
    std1, std2 = float(pt1.std()), float(pt2.std())
    n_eval = EVAL_BATCHES * TPB
    probe = {
        "model": f"{n_layer}L×{n_embd}d×{n_head}h（{model.n_params(non_embedding=True):,} 非嵌入）",
        "train_steps": 500, "seed": SEED, "wallclock_sec": round(sec, 1),
        "tok_std_first_block_8192": round(std1, 4),
        "tok_std_eval_set_401408": round(std2, 4),
        "mean_loss_first_block": round(float(pt1.mean()), 4),
        "se_at_401k": round(std2 / math.sqrt(n_eval), 5),
        "n_required_for_se_lt_0.005": int((std2 / 0.005) ** 2),
        "note": "逐 token loss std（fp32 前向）；冒烟档 shakespeare 首块(6912 tok)实测 3.14 的 OWT 正式档复测",
    }
    out = os.path.join(OUT_DIR, "tok_std.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(probe, f, ensure_ascii=False, indent=2)
    print(json.dumps(probe, ensure_ascii=False, indent=2), flush=True)
    print(f" -> {out}", flush=True)
    del model, opt
    if device == "mps":
        torch.mps.empty_cache()


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="ch8 Kaplan 单曲线族（固定 D=40M，N 五点，一次通过）")
    ap.add_argument("--points", type=str, default="all", help="all 或逗号分隔点名（1m,3m,7m,15m,20m）")
    ap.add_argument("--steps", type=int, default=D_TOKENS // TPB,
                    help=f"每点步数（默认 D//8192={D_TOKENS // TPB}；冒烟可改小）")
    ap.add_argument("--out-name", type=str, default=None,
                    help="输出名；缺省随 warmup 自动取 family_fast(warmup=200)/family_nowarmup(其余)，防覆写正式族文件")
    ap.add_argument("--probe-tok-std", action="store_true", help="只跑 tok_std 探针（1M×500 步）")
    ap.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--warmup", type=int, default=WARMUP,
                    help="warmup 步数（默认 0=任务原口径；ch4 §4.5 实测 pre-LN 加 warmup 亦有增益，"
                         "深度带来的 L(N) 倒挂可用统一 warmup 修正——仍满足 Kaplan 各规模统一日程口径）")
    args = ap.parse_args()
    warmup = args.warmup
    if args.out_name is None:
        args.out_name = "family_fast" if warmup == 200 else "family_nowarmup"
    os.makedirs(OUT_DIR, exist_ok=True)
    train_stream = TrainStream(TOKENS_BIN)
    device = args.device

    if args.probe_tok_std:
        probe_tok_std(device, train_stream)
        return

    names = [f[0] for f in FAMILY] if args.points == "all" else [p.strip() for p in args.points.split(",")]
    for n in names:
        assert n in [f[0] for f in FAMILY], f"未知点名 {n}"
    assert args.steps <= train_stream.max_step, \
        f"步数 {args.steps} 超出训练池一次通过上限 {train_stream.max_step}（不许重复 epoch）"

    out_json = os.path.join(OUT_DIR, f"{args.out_name}.json")
    # ---- 断点续跑：读已有 JSON，跳过已完成点 ----
    if os.path.exists(out_json):
        with open(out_json, encoding="utf-8") as f:
            report = json.load(f)
        done = {p["name"]: p for p in report["points"]}
        report["points"] = list(done.values())
        print(f"[续跑] 已有 {len(done)} 点完成：{sorted(done)}", flush=True)
    else:
        done = {}
        report = {"meta": {}, "points": []}

    data_meta = {}
    dmeta_path = os.path.join(OUT_DIR, "data_meta.json")
    if os.path.exists(dmeta_path):
        with open(dmeta_path, encoding="utf-8") as f:
            data_meta = json.load(f)
    report["meta"] = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "device": device,
        "torch": torch.__version__,
        "corpus": "OpenWebText 子采样 1.15GB（owt_docs.jsonl；分词器分片独立，见 data_meta.json）",
        "tokenizer": f"sp-8k BPE byte_fallback（只在 29MB 分词器分片上训练，与 LM 语料分离）",
        "experiment": "Kaplan 单曲线缩小版：固定 D=40M token 一次通过，N∈{1M,3M,7M,15M,20M} 非嵌入参数",
        "n_def": "N=非嵌入参数（扣 wte+wpe；tied head 与 wte 同存储天然不计）——01-实验可行性.md §4 统一口径",
        "batch_block": f"b{BATCH}×s{BLOCK}（{TPB} token/步）",
        "optimizer": f"AdamW(betas=0.9,0.95,wd={WDECL})+grad_clip1.0",
        "schedule": (f"lr {PEAK_LR} 余弦降至 {PEAK_LR*MIN_LR_RATIO:.0e}，warmup={warmup}；各规模统一日程"
                     "（Kaplan 固定日程口径的缩小版；单变量=N）"),
        "data_protocol": "顺序连续分块一次通过（零重复 epoch）；所有点共用流首部同一段 40M token（单变量=N）",
        "eval_protocol": f"每 {EVAL_EVERY} 步 + 终步：eval 池（{EVAL_BIN}，与训练段互斥）固定 49 窗 "
                         f"{EVAL_BATCHES*TPB:,} token，fp32 前向",
        "precision": "train bf16 autocast / eval fp32",
        "model_note": "块实现直接 import 自 ch04/warmup_ablation.py（from warmup_ablation import GPT, Block；pre-LN + ln_f + tied + std=0.02 初始化）",
        "data_meta": data_meta.get("encoding", {}),
        "argv": " ".join(sys.argv[1:]),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    t_all = time.perf_counter()
    for name, n_nominal, n_layer, n_embd, n_head in FAMILY:
        if name not in names or name in done:
            continue
        result = run_point(name, n_nominal, n_layer, n_embd, n_head, args.steps, device,
                           args.out_name, train_stream, warmup)
        report["points"].append(result)
        report["points"].sort(key=lambda p: p["N"])
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[进度] {len(report['points'])}/{len(names)} 点完成，累计 {(time.perf_counter()-t_all)/60:.1f} min",
              flush=True)

    print(f"\n[全部完成] {len(report['points'])} 点 / 总 {(time.perf_counter()-t_all)/60:.1f} min -> {out_json}", flush=True)
    for p in report["points"]:
        print(f"  {p['name']:>3s} N={p['N']:>9,d} steps={p['steps']} eval_final={p['eval_final']}", flush=True)


if __name__ == "__main__":
    main()
