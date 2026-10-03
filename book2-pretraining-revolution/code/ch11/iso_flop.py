# iso_flop.py —— Book2 ch11 实验一：iso-FLOP 网格（compute-optimal 前沿的缩小版复现）
# 所属章节：Book2 第 11 章 §11.2（ch11-大项目II.md 图 11.2/11.3 数据源）
# 设计书：notes/01-实验可行性.md §4(ii)；批二教训：notes/07-批二实验数字.md §3.1
#   （正式口径统一 warmup=200——无 warmup 族 L(N) 倒挂的修正，各规模统一日程仍满足 iso-FLOP 单变量口径）。
#   * 3 档算力 C∈{6e14, 2e15, 6e15}（FLOPs=6ND，N=非嵌入参数，D=已处理 token；bf16 训练/fp32 eval 照 ch08）；
#   * 每档 (N,D) 点取 D=C/(6N)（步数取整后反报 C_actual）；C1/C3 各 3 点，**中间档 C2=2e15 加密至 5 点**（N 对数等距）；
#   * N 网格选型依据：ch08 Kaplan 族在 D=40M 固定时 L(N) 单调降到 19.5M 且末段斜率仍变陡（notes/07 §2.1）
#     → 谷底 N* 在高算力档可能 ≥20M，C3 顶点放宽到 23.1M（超出设计书「N≤20M」的记录在案偏离，为兜住谷底）；
#   * 每点：流首部一次通过读 D token（顺序连续分块，构造上零重复；各点共用流首部=数据分布跨点固定），
#     b32×s256、AdamW(0.9,0.95,wd=0.1)+clip1.0、lr 1e-3 余弦降至 1e-4 + warmup 200、每 500 步 eval
#     （49 固定窗×8192=401,408 token，与 ch08 完全同窗集可比）；
#   * 每档对 (log10 N, eval_final) 抛物线拟合取谷底 N*(C)；再拟合 log10 N* ∝ log10 C 得 compute-optimal 斜率。
#     不确定度：①5 点档给 polyfit 参数协方差（delta 法）+ 逐点剔除（LOO）敏感性；②全档参数 bootstrap
#     （eval se=0.004 主口径 / 0.008 敏感性口径，notes/07 §1.3+§2.1）；③斜率 CI 由各档 N* bootstrap 联合 Monte Carlo。
#     3 点档抛物线为精确拟合（零自由度）→ 不确定度几乎全部来自假设的 eval 噪声，如实报告（不确定度大）。
#   * 模型：from warmup_ablation import GPT（ch04 族工厂，ch08 同款单轨依赖链）；分词器/数据复用 ch08/data_prep
#     产物（log/book2-ch08/tokens.bin + eval.bin，sp-8k）。
# 运行方式：python iso_flop.py [--levels c1,c2,c3] [--smoke] [--analyze-only] [--device mps]
#       --smoke：c1 三点各 300 步、每 100 步 eval，写 iso_flop_smoke.json（冒烟前置，不动正式文件）
# 产物：log/book2-ch11/{iso_flop.json, curve_<点名>.csv/.npy, ckpt_<点名>.pt(跑完即删)}

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

# ---------------- 固定口径 ----------------
SEED = 20261002          # 全书统一种子
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book2-ch11")
CH08_DIR = os.path.join(REPO_ROOT, "log", "book2-ch08")
TOKENS_BIN = os.path.join(CH08_DIR, "tokens.bin")   # 训练池（ch08/data_prep.py 产物，流首部）
EVAL_BIN = os.path.join(CH08_DIR, "eval.bin")       # eval 池（流末尾留出，与训练段互斥；与 ch08 同窗集）

VOCAB = 8192             # sp-8k（与 ch08 族同分词器）
BLOCK = 256              # 序列长 s
BATCH = 32               # 批大小 b（b32×s256=8192 token/步，与 ch08 同）
TPB = BATCH * BLOCK
EVAL_EVERY = 500
EVAL_BATCHES = 49        # 49×8192 = 401,408 token（≥40 万协议，notes/07 §1.3 实测 se≈0.0039）
PEAK_LR = 1e-3
MIN_LR_RATIO = 0.1
WARMUP = 200             # 批二教训：统一 warmup 200（notes/07 §3.1）
WDECL = 0.1
BOOT_SE = 0.004          # bootstrap 主口径：401,408 token eval se（notes/07 §1.3）
BOOT_SE_SENS = 0.008     # 敏感性口径：叠加同配置重跑漂移 ~0.004（notes/07 §2.1）后的保守值
BOOT_DRAWS = 2000

# iso-FLOP 网格：档 × 点（点名, C 名义, n_layer, n_embd, n_head）。N 实数由 GPT.n_params 报出，
# 公式 N = L·(12C²+13C)+2C（ch08 实测同式）。N 名义取对数等距；head_dim 尽量 64（ch08 族口径），
# 个别点 56/72/80（对数等距优先，运行元数据如实登记）。
# 【网格自适应记录 2026-10-03】初版按 ch08 Kaplan 曲线（D=40M 固定时 L(N) 到 19.5M 仍降）预设谷底偏大 N
#   （C1{3.1,5.5,9.9}M / C2{4.0..19.5}M / C3{9.9,14.8,23.1}M）；实测 C1 三点 eval_final 随 N 单调下降
#   （4.979/5.183/5.484）→ 谷底 ≤3.1M、本尺度为「数据富余」型——C1 补 c1d=1.50M 兜左沿，
#   C2/C3 整体下移（C2{1.86..8.69}M 五点 / C3{4.18,9.86,19.5}M）。全部 D≤~240M ≤306M 池一次通过。
#   C1=6e14:  0.60M / 0.79M / 0.99M / 1.50M / 3.11M / 5.53M / 9.86M（D≈167.8M / 125.9M / 100.7M / 66.5M / 32.1M /
#             18.1M / 10.1M；c1d/e/f/g 自适应补点——谷底持续左滑，0.33M 为一次通过池硬下限 N≥C/(6·306M)）
#   C2=2e15:  1.86M / 3.11M / 4.00M / 6.05M / 8.69M（加密档；D≈179.6M / 107.0M / 83.2M / 55.1M / 38.4M）
#   C3=6e15:  4.18M / 9.86M / 19.52M（D≈239.5M / 101.4M / 51.2M）
LEVELS = [
    ("c1", 6e14, [
        ("c1a", 7, 192, 3),
        ("c1b", 7, 256, 4),
        ("c1c", 8, 320, 5),
        ("c1d", 6, 144, 2),
        ("c1e", 5, 128, 2),
        ("c1f", 4, 128, 2),
        ("c1g", 3, 128, 2),
    ]),
    ("c2", 2e15, [
        ("c2a", 6, 160, 2),
        ("c2b", 7, 192, 3),
        ("c2c", 9, 192, 3),
        ("c2d", 10, 224, 4),
        ("c2e", 11, 256, 4),
    ]),
    ("c3", 6e15, [
        ("c3a", 8, 208, 4),
        ("c3b", 8, 320, 5),
        ("c3c", 11, 384, 6),
    ]),
]

# ---------------- 模型：复用 ch04/warmup_ablation.py 的 GPT（ch08 同款单轨依赖链） ----------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ch04"))
from warmup_ablation import GPT, Block  # noqa: E402,F401  族模型工厂（Block 备行级引用/调试）


# ---------------- 日程与数据（与 ch08/run_family.py 同式，单独成文件便于行级引用） ----------------
def lr_at(step, peak, warmup, total, min_ratio=MIN_LR_RATIO):
    """lr 日程（ch04/ch08 同式）：warmup 段线性升，之后余弦降到 min_ratio*peak。"""
    if warmup > 0 and step <= warmup:
        return peak * step / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


class TrainStream:
    """训练数据：tokens.bin 流首部按步顺序切块（一次通过，构造上无重复）。

    第 k 步消费 tokens[(k-1)*8192 : (k-1)*8192+8193]：x=前 8192 reshape(32,256)，y=左移一格。
    iso-FLOP 各点从流首部起步、长度各为自身 D——各点数据互为嵌套前缀，数据分布跨点固定（单变量=N×D 权衡）。
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
    """固定 eval 批集：与 ch08/run_family.py 完全同窗（同种子同 eval.bin）——ch8/ch11 数字可直接同图。"""
    ev = np.memmap(EVAL_BIN, dtype=np.uint16, mode="r")
    gen = torch.Generator().manual_seed(SEED)
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
    return sum(losses) / len(losses)


# ---------------- 单点训练（含断点续跑；ch08/run_family.py 同骨架） ----------------
def run_point(name, level, c_nominal, n_layer, n_embd, n_head, device, train_stream,
              steps_override=None, eval_every=EVAL_EVERY):
    ckpt_path = os.path.join(OUT_DIR, f"ckpt_{name}.pt")
    torch.manual_seed(SEED)   # 每点重置（ch08 同口径：init 流跨点一致，唯一变量 = 架构规模×算力档）
    model = GPT(n_layer=n_layer, n_embd=n_embd, n_head=n_head,
                vocab=VOCAB, block_size=BLOCK).to(device)
    n_total = model.n_params()
    n_nonemb = model.n_params(non_embedding=True)
    # iso-FLOP 步数：D=C/(6N) 取整到步；反报 D_actual 与 C_actual（与名义 C 偏差 <0.1% 量级，如实登记）
    if steps_override is not None:
        steps = steps_override
    else:
        steps = int(round(c_nominal / (6.0 * n_nonemb) / TPB))
    d_tokens = steps * TPB
    c_actual = 6.0 * n_nonemb * d_tokens
    if steps_override is None:
        assert steps >= 400, f"{name}: 步数 {steps} 过短（warmup 200 后余量不足）"
    assert steps <= train_stream.max_step, \
        f"{name}: 步数 {steps} 超出训练池一次通过上限 {train_stream.max_step}"
    print(f"\n===== 点 {name}（档 {level}, C={c_nominal:.1e}）: {n_layer}L×{n_embd}d×{n_head}h "
          f"（head_dim={n_embd // n_head}）非嵌入 N={n_nonemb:,} / 总参 {n_total:,} =====\n"
          f"      steps={steps:,} -> D={d_tokens:,} tok, C_actual=6ND={c_actual:.6e} "
          f"（名义偏差 {c_actual / c_nominal - 1:+.4%}）", flush=True)
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
        lr = lr_at(step, PEAK_LR, WARMUP, steps)
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
            print(f"  step {step:6d}/{steps} train {lv:.4f} lr {lr:.2e} "
                  f"({elapsed_train / step:.3f}s/step 累计)", flush=True)
        if step % eval_every == 0 or step == steps:
            t0 = time.perf_counter()
            el = eval_loss(model, ex, ey)
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
        "name": name, "level": level, "C_nominal": c_nominal, "C_actual": c_actual,
        "N": n_nonemb, "N_total": n_total,
        "n_layer": n_layer, "n_embd": n_embd, "n_head": n_head, "head_dim": n_embd // n_head,
        "steps": steps_done, "steps_planned": steps,
        "D_tokens": steps_done * TPB,
        "one_pass": "顺序连续分块一次通过（流首部，构造上零重复；各点数据互为嵌套前缀）",
        "final_train_loss": round(sum(tail) / len(tail), 5),
        "final_train_loss_def": "最后 100 步 train loss 均值",
        "eval_losses": evals,
        "eval_final": evals[-1]["eval_loss"] if evals else None,
        "eval_tokens_per_eval": EVAL_BATCHES * TPB,
        "diverged": diverged_at is not None, "diverged_at_step": diverged_at, "divergence_reason": reason,
        "wallclock": round(wallclock, 1), "wallclock_sec_train_only": round(elapsed_train, 1),
        "wallclock_sec_eval_only": round(elapsed_eval, 1),
        "sec_per_step": round(elapsed_train / max(1, steps_done), 4),
        "precision": "train bf16 autocast / eval fp32",
    }
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
          f"wall {wallclock / 60:.1f} min", flush=True)
    del model, opt, ex, ey
    if device == "mps":
        torch.mps.empty_cache()
    return result


# ---------------- 分析：每档抛物线谷底 N*(C) + 斜率拟合 ----------------
def _vertex(x, y):
    """抛物线 y=a x²+b x+c 的谷底 x*=-b/(2a)；a<=0（无谷）返回 None。"""
    a, b, c = np.polyfit(x, y, 2)
    return None if a <= 0 else -b / (2 * a)


def fit_level(level_name, c_nominal, pts):
    """单档拟合：精确抛物线 + 协方差(≥4点) + bootstrap(两 se 口径) + LOO(≥4点)。"""
    x = np.log10(np.array([p["N"] for p in pts], dtype=float))
    y = np.array([p["eval_final"] for p in pts], dtype=float)
    a, b, c = np.polyfit(x, y, 2)
    out = {
        "level": level_name, "C_nominal": c_nominal,
        "C_actual_mean": float(np.mean([p["C_actual"] for p in pts])),
        "n_points": len(pts), "points": [p["name"] for p in pts],
        "N_range": [float(10 ** x.min()), float(10 ** x.max())],
        "quad": {"a": float(a), "b": float(b), "c": float(c)},
        "eval_finals": [p["eval_final"] for p in pts],
        "eval_final_monotone_decreasing_in_N": bool(all(y[i] > y[i + 1] for i in range(len(y) - 1))),
    }
    if a > 0:
        xv = -b / (2 * a)
        out["log10_N_star"] = float(xv)
        out["N_star"] = float(10 ** xv)
        out["L_star_pred"] = float(a * xv * xv + b * xv + c)
        out["N_star_inside_grid"] = bool(x.min() <= xv <= x.max())
    else:
        out["log10_N_star"] = None
        out["N_star"] = None
        out["L_star_pred"] = None
        out["N_star_inside_grid"] = False
        out["note"] = "a<=0：抛物线开口向上不存在，该档 eval_final 单调（无内部谷底）"

    # ① 参数协方差（≥4 点才有自由度）：delta 法传到 x*
    if len(pts) >= 4:
        try:
            coef, cov = np.polyfit(x, y, 2, cov=True)
            a_, b_ = coef[0], coef[1]
            if a_ > 0:
                g = np.array([b_ / (2 * a_ * a_), -1.0 / (2 * a_), 0.0])   # x* 对 (a,b,c) 的梯度
                out["log10_N_star_se_cov"] = float(math.sqrt(max(0.0, g @ cov @ g)))
        except Exception as e:  # noqa: BLE001
            out["cov_error"] = str(e)

    # ② 参数 bootstrap：eval se=0.004（主）/0.008（敏感性）
    rng = np.random.default_rng(SEED)
    for se, tag in ((BOOT_SE, "main"), (BOOT_SE_SENS, "sens")):
        xs = []
        no_valley = 0
        for _ in range(BOOT_DRAWS):
            yb = y + rng.normal(0.0, se, len(y))
            xv = _vertex(x, yb)
            if xv is None:
                no_valley += 1
            else:
                xs.append(xv)
        xs = np.asarray(xs)
        entry = {"assumed_se": se, "draws": BOOT_DRAWS, "no_valley_frac": round(no_valley / BOOT_DRAWS, 4)}
        if len(xs) >= 100:
            q = np.percentile(xs, [2.5, 50, 97.5])
            entry.update({"log10_N_star_ci95": [float(v) for v in q],
                          "N_star_ci95": [float(10 ** v) for v in q],
                          "n_valid_draws": int(len(xs)),
                          "note": "3 点档抛物线为精确拟合，CI 几乎全部由假设的 eval 噪声驱动——真实不确定度更大"})
        else:
            entry["note"] = f"有效抽取仅 {len(xs)}，谷底不可定位"
        out[f"bootstrap_{tag}"] = entry

    # ③ 逐点剔除（LOO，≥4 点）
    if len(pts) >= 4:
        loo = []
        for i in range(len(pts)):
            keep = [j for j in range(len(pts)) if j != i]
            xv = _vertex(x[keep], y[keep])
            loo.append({"dropped": pts[i]["name"],
                        "log10_N_star": None if xv is None else float(xv),
                        "N_star": None if xv is None else float(10 ** xv)})
        out["leave_one_out"] = loo
    return out


def fit_slope(level_fits, points=None):
    """log10 N* ∝ log10 C。锚点分级：①谷底被网格兜住的档用抛物线顶点（可靠）；
    ②谷底滑出网格的档（N_star_inside_grid=False）顶点为外推伪影（LOO 摆幅 0.3k-250k 量级），
    只给 argmin 上界（N* ≤ 最优点 N）与一次通过池下界（N ≥ C/(6·306M)）。
    斜率报三口径：in-grid 顶点两档 / 全档 argmin / argmin₁+顶点₂₃ 混合——band 即诚实区间。"""
    out = {
        "slope_def": "d log10 N* / d log10 C（N*=各档 compute-optimal 非嵌入参数；N 非嵌入口径）",
        "lit_compare": {"kaplan_N_star_vs_C": 0.73, "chinchilla_N_star_vs_C": 0.5,
                        "note": "文献值为各自大模型/大算力口径的 N*∝C^γ 指数（03-引用复核清单 警-2/警-4 已复核："
                                "Kaplan §6.1/Table 5 p_N=0.73；Chinchilla iso-FLOP 法 0.49/0.51）；本实验是 "
                                "OWT/sp-8k/≤23M 参数缩小版，C 仅跨 1 个数量级，只做同型对照不比大小"},
    }
    # 每档锚点：vertex（in-grid 才可靠）与 argmin（实测最优点，恒为 N* 上界）
    anchors = {}
    for f in level_fits:
        a = {"level": f["level"], "log10C": math.log10(f["C_actual_mean"]),
             "vertex": f["log10_N_star"], "vertex_in_grid": f["N_star_inside_grid"]}
        if points is not None:
            pts = [p for p in points if p["level"] == f["level"]]
            if pts:
                best = min(pts, key=lambda p: p["eval_final"])
                a["argmin"] = math.log10(best["N"])
                a["argmin_name"] = best["name"]
        anchors[f["level"]] = a
    out["anchors"] = list(anchors.values())

    def ols(pairs):
        lc = np.array([p[0] for p in pairs]); ln = np.array([p[1] for p in pairs])
        s, b = np.polyfit(lc, ln, 1)
        return float(s), float(b)

    ingrid = [a for a in anchors.values() if a["vertex"] is not None and a["vertex_in_grid"] and "argmin" in a]
    out["slope_ingrid_vertices"] = None
    if len(ingrid) >= 2:
        s, b = ols([(a["log10C"], a["vertex"]) for a in ingrid])
        out["slope_ingrid_vertices"] = {
            "slope": s, "levels": [a["level"] for a in ingrid],
            "note": f"仅用谷底被网格兜住的 {len(ingrid)} 档抛物线顶点（可靠锚点）"}
        ll = [(a["log10C"], a["vertex"]) for a in ingrid]
    argmins = [a for a in anchors.values() if "argmin" in a]
    if len(argmins) >= 2:
        s, b = ols([(a["log10C"], a["argmin"]) for a in argmins])
        out["slope_argmin_all"] = {
            "slope": s, "levels": [a["level"] for a in argmins],
            "note": "argmin=各档实测最优 N（N* 的上界：真谷底 ≤ argmin）→ 此斜率是斜率的下界口径"}
        # 混合：低档 argmin（若其顶点出网格）+ 其余顶点
        mixed = [(a["log10C"], a["vertex"] if (a["vertex"] is not None and a["vertex_in_grid"]) else a["argmin"])
                 for a in argmins]
        s2, _ = ols(mixed)
        out["slope_mixed"] = {"slope": s2, "note": "低档取 argmin 上界、高档取 in-grid 顶点——斜率的上界口径"}
        if "slope_ingrid_vertices" in out and out["slope_ingrid_vertices"]:
            lo_s = min(out["slope_argmin_all"]["slope"], out["slope_ingrid_vertices"]["slope"], s2)
            hi_s = max(out["slope_argmin_all"]["slope"], out["slope_ingrid_vertices"]["slope"], s2)
        else:
            lo_s, hi_s = min(out["slope_argmin_all"]["slope"], s2), max(out["slope_argmin_all"]["slope"], s2)
        out["slope_band"] = [lo_s, hi_s]
        out["slope_band_note"] = ("诚实区间=三口径包络：argmin 口径是下界（真 N* 更小→斜率更大），"
                                  "顶点口径受近平局三点拟合摆动影响；正文引斜率必须引 band 而非点估计")
    # 外推顶点档的警戒标记
    frag = [a["level"] for a in anchors.values() if a["vertex"] is not None and not a["vertex_in_grid"]]
    if frag:
        out["extrapolated_vertex_levels"] = {
            "levels": frag,
            "warning": ("谷底滑出网格左沿（实测曲线整档单调），抛物线顶点为外推伪影：LOO 摆幅可达数量级，"
                        "且可低于一次通过池硬下界 N=C/(6·306M)——严禁把该顶点当 N* 引用；"
                        "该档只报 N*<argmin 上界 + 池下界")}
    # L*(C) 幂律（in-grid 档顶点损失）
    if len(ingrid) >= 2:
        ll = [(a["log10C"], math.log10(next(f["L_star_pred"] for f in level_fits if f["level"] == a["level"])))
              for a in ingrid]
        ls, _ = ols(ll)
        out["L_star_power_in_C_ingrid"] = float(ls)
    # D*/N* 比（token/param at optimum）——Chinchilla 20:1 对照
    if points is not None:
        ratios = {}
        for f in level_fits:
            n_star = f["N_star"] if (f["N_star"] is not None and f["N_star_inside_grid"]) else None
            if n_star:
                ratios[f["level"]] = round(f["C_actual_mean"] / (6 * n_star * n_star), 1)
        out["D_over_N_at_optimum"] = {
            "ratios_tokens_per_param": ratios,
            "chinchilla_ref": "~20 token/param（大算力口径）；本实验比值随 C 增大而降（数据富余度随规模缓解）",
        }
    return out


def analyze(report):
    """对已完成点重算全部分析（主流程末尾与 --analyze-only 共用）。"""
    pts = report["points"]
    level_fits = []
    for lname, c_nom, cfgs in LEVELS:
        lp = sorted([p for p in pts if p["level"] == lname], key=lambda p: p["N"])
        if not lp:
            continue
        fit = fit_level(lname, c_nom, lp)
        fit["complete"] = len(lp) == len(cfgs)
        if not fit["complete"]:
            fit["note_incomplete"] = f"仅 {len(lp)}/{len(cfgs)} 点完成，拟合为部分结果"
        level_fits.append(fit)
    report["level_fits"] = level_fits
    report["slope_fit"] = fit_slope(level_fits, points=pts)
    return report


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="ch11 iso-FLOP 网格（3 档算力 × N 网格，D=C/(6N)）")
    ap.add_argument("--levels", type=str, default="all", help="all 或逗号分隔档名（c1,c2,c3）")
    ap.add_argument("--smoke", action="store_true", help="冒烟：c1 三点各 300 步、每 100 步 eval，写 iso_flop_smoke.json")
    ap.add_argument("--analyze-only", action="store_true", help="只重跑分析（从已有 iso_flop.json）")
    ap.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_name = "iso_flop_smoke" if args.smoke else "iso_flop"
    out_json = os.path.join(OUT_DIR, f"{out_name}.json")

    if args.analyze_only:
        with open(out_json, encoding="utf-8") as f:
            report = json.load(f)
        analyze(report)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(json.dumps({"level_fits": report["level_fits"], "slope_fit": report["slope_fit"]},
                         ensure_ascii=False, indent=2), flush=True)
        print(f" -> {out_json}", flush=True)
        return

    level_names = [l[0] for l in LEVELS] if args.levels == "all" else \
        [s.strip() for s in args.levels.split(",")]
    for n in level_names:
        assert n in [l[0] for l in LEVELS], f"未知档名 {n}"
    if args.smoke:
        level_names = ["c1"]

    train_stream = TrainStream(TOKENS_BIN)
    device = args.device

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

    smoke_note = ("（冒烟档：300 步上限+每 100 步 eval，不构成 iso-FLOP 点，仅验证管道）" if args.smoke else "")
    report["meta"] = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "device": device,
        "torch": torch.__version__ + smoke_note,
        "corpus": "OpenWebText 子采样（ch08/data_prep.py 产物：log/book2-ch08/tokens.bin 流首部，sp-8k）",
        "tokenizer": "sp-8k BPE byte_fallback（与 ch08 族同一分词器与同一 eval 窗集，数字可直接同图）",
        "experiment": ("iso-FLOP 网格缩小版：C∈{6e14,2e15,6e15} 三档，每档 D=C/(6N)，"
                       "C2=2e15 加密 5 点（N 对数等距）；每档抛物线拟合取 N*(C)，再拟合 log N*∝log C"),
        "flops_def": "C=6ND（N=非嵌入参数，D=已处理 token；6=2 前向+4 反向；忽略 attention 分数项与 embedding 查表）",
        "batch_block": f"b{BATCH}×s{BLOCK}（{TPB} token/步）",
        "optimizer": f"AdamW(betas=0.9,0.95,wd={WDECL})+grad_clip1.0",
        "schedule": (f"lr {PEAK_LR} 余弦降至 {PEAK_LR*MIN_LR_RATIO:.0e}，warmup={WARMUP} 统一"
                     "（notes/07 §3.1 教训：无 warmup 族 L(N) 倒挂）；注意各点总步数∝D、日程形状跨点不严格一致"
                     "（余弦各自退火到自身终点——Chinchilla iso-FLOP 同型局限，如实声明）"),
        "data_protocol": ("顺序连续分块一次通过（零重复 epoch）；各点自流首部起步、长度=自身 D"
                          "（各点数据互为嵌套前缀，数据分布跨点固定，单变量=N×D 权衡）"),
        "eval_protocol": (f"每 {EVAL_EVERY} 步 + 终步：eval 池（ch08 eval.bin，与训练段互斥）固定 49 窗 "
                          f"{EVAL_BATCHES*TPB:,} token，fp32 前向，窗集与 ch08/run_family.py 逐位相同"),
        "precision": "train bf16 autocast / eval fp32",
        "model_note": "块实现 import 自 ch04/warmup_ablation.py（from warmup_ablation import GPT, Block；"
                      "pre-LN + ln_f + tied + std=0.02 初始化）",
        "grid_note": ("N 网格对数等距；head_dim 以 64 为主（1.86M=80 / 4.18M=52 / 6.05M=56 / 1.5M=72 等个别点例外）。"
                      "初版网格按 ch08 Kaplan 曲线预设谷底偏大 N，实测 C1 谷底 ≤3.1M 后自适应下移"
                      "（见文件头【网格自适应记录】）；最大 D≈239.5M ≤306M 训练池一次通过 ≤ 设计书 ~110M 预算"
                      "的超出已在案（为保 iso-FLOP 口径 D=C/(6N) 严格成立，数据池充足无重复）"),
        "argv": " ".join(sys.argv[1:]),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    t_all = time.perf_counter()
    for lname, c_nom, cfgs in LEVELS:
        if lname not in level_names:
            continue
        for name, n_layer, n_embd, n_head in cfgs:
            if name in done:
                continue
            result = run_point(name, lname, c_nom, n_layer, n_embd, n_head, device, train_stream,
                               steps_override=300 if args.smoke else None,
                               eval_every=100 if args.smoke else EVAL_EVERY)
            report["points"].append(result)
            report["points"].sort(key=lambda p: (p["level"], p["N"]))
            analyze(report)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"[进度] {len(report['points'])} 点完成，累计 {(time.perf_counter()-t_all)/60:.1f} min", flush=True)

    analyze(report)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[全部完成] {len(report['points'])} 点 / 总 {(time.perf_counter()-t_all)/60:.1f} min -> {out_json}", flush=True)
    for p in report["points"]:
        print(f"  {p['name']} C={p['C_nominal']:.0e} N={p['N']:>10,d} D={p['D_tokens']:>11,d} "
              f"steps={p['steps']:>6,d} eval_final={p['eval_final']}", flush=True)
    print("\n[每档谷底与斜率]", flush=True)
    print(json.dumps({"level_fits": report["level_fits"], "slope_fit": report["slope_fit"]},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
