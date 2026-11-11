# 用途：Book2第8章的三类核算图：幂律、预算线、已有实验终点拟合。
# 正文包含八张图；本脚本只生成上述三类计算图，不覆盖正文的机制示意或历史图片。
# 来源：Kaplan v1 Table5；Chinchilla v1 Table3/D.4；用户指定的本地族结果。
# 运行方式：激活环境后，在本册目录运行 python code/ch08/make_figures.py --plots theory
# --plots family/all需完整的族结果与拟合文件；默认theory不伪造缺失实验。
# 产物：当前目录log/book2-ch08/figures/check-8-*.png
import json
import os
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path("log/book2-ch08/figures")
LOG_DIR = Path("log/book2-ch08")
FIT_PREFIX = "fit_kaplan"
FAMILY_FILE = "family_fast.json"
NOWARMUP_FILE = "family_nowarmup.json"

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MUTED = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": TICK, "axes.labelcolor": DARK,
    "xtick.color": TICK, "ytick.color": TICK, "text.color": DARK,
    "axes.linewidth": 0.8, "font.family": "DejaVu Sans",
})


# ---------------- 核算图A：按v1 Table5常数画函数，非原文训练散点 ----------------
def fig_8_1():
    # Kaplan 2001.08361v1 Table5；显示范围不证明外推有效。
    panels = [
        ("L(N) = (Nc/N)^a_N", 8.8e13, 0.076, r"$N$ params (non-embedding)",
         1e3, 1e9, r"$N_c = 8.8\times10^{13}$, $\alpha_N = 0.076$", BLUE),
        ("L(D) = (Dc/D)^a_D", 5.4e13, 0.095, r"$D$ tokens",
         1e6, 1e11, r"$D_c = 5.4\times10^{13}$, $\alpha_D = 0.095$", TEAL),
        ("L(Cmin) = (Cc/Cmin)^a_C", 3.1e8, 0.050, r"$C_{min}$ PF-days (small-batch limit)",
         1e-4, 1e2, r"$C_c^{min} = 3.1\times10^{8}$, $\alpha_C^{min} = 0.050$", ORANGE),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8, 3.1), dpi=300)
    for ax, (title, c, alpha, xlabel, lo, hi, note, color) in zip(axes, panels):
        x = np.logspace(np.log10(lo), np.log10(hi), 200)
        y = (c / x) ** alpha
        ax.plot(x, y, color=color, lw=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=8.2)
        ax.set_title(title, fontsize=8.6)
        ax.text(0.97, 0.94, note + f"\nlog-log slope = -{alpha}", transform=ax.transAxes,
                fontsize=7.6, color=MUTED, va="top", ha="right")
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("test loss $L$ (nats/token)", fontsize=8.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "check-8-powerlaws.png"), facecolor="white")
    plt.close(fig)


# ---------------- 核算图B：两文参数建议；横轴统一为显示N/D的6ND近似 ----------------
def fig_8_2():
    # Chinchilla v1 Table3：(参数、原表FLOPs、训练token)；原表显示数不逐位满足6ND。
    t3 = np.array([
        [4e8, 1.92e19, 8.0e9], [1e9, 1.21e20, 2.02e10], [1e10, 1.23e22, 2.051e11],
        [6.7e10, 5.76e23, 1.5e12], [1.75e11, 3.85e24, 3.7e12], [2.8e11, 9.90e24, 5.9e12],
        [5.2e11, 3.43e25, 1.1e13], [1e12, 1.27e26, 2.12e13],
        [1e13, 1.30e28, 2.162e14],
    ])
    C = np.logspace(18.5, 26, 200)
    n_chinchilla = 6.7e10 * (C / (6*6.7e10*1.5e12)) ** 0.50  # 示意方向线，锚点也用6ND。
    n_kaplan = 4.68e9 * (C / 1e21) ** 0.73               # Kaplan 线：斜率 0.73，锚点=附录 D.4 按其律算出的 4.68B@1e21
    # 实配模型点用Chinchilla Table1的标称参数与token统一6ND；不冒充各原报告硬件实测。
    models = [
        ("GPT-3 175B\n300B tok (D/N=1.7)", 6*1.75e11*3e11, 1.75e11, 3.0e11, ORANGE, (2e22, 1.2e11)),
        ("Gopher 280B\n300B tok (D/N=1.1)", 6*2.8e11*3e11, 2.8e11, 3.0e11, ORANGE, (2e22, 4.5e11)),
        ("MT-NLG 530B\n270B tok", 6*5.3e11*2.7e11, 5.3e11, 2.7e11, MUTED, (2e24, 1.1e12)),
        ("Chinchilla 70B\n1.4T tok (D/N=20)", 6*7e10*1.4e12, 7.0e10, 1.4e12, TEAL, (1.5e24, 1.5e10)),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=300)
    ax.plot(C, n_kaplan, ls="--", lw=2, color=BLUE,
            label="0.73 direction, anchored to Chinchilla App.D.4")
    ax.plot(C, n_chinchilla, ls="-.", lw=2, color=TEAL,
            label="Approach 1 direction: slope 0.50 (illustrative anchor)")
    ax.plot(6*t3[:, 0]*t3[:, 2], t3[:, 0], "o", ms=7, color=TEAL, label="Chinchilla Table 3 optimal points")
    for label, c, n, d, color, off in models:
        ax.plot([c], [n], "*", ms=13, color=color, zorder=5)
        ax.annotate(label, (c,n), textcoords="data", xytext=off, fontsize=7.2, color=DARK,
                    arrowprops=dict(arrowstyle="-",color=MUTED,lw=.8),
                    bbox=dict(facecolor="white",edgecolor="none",alpha=.9,pad=1.5))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(2e18, 3e26); ax.set_ylim(1e8, 3e12)
    ax.set_xlabel("approximate FLOPs from displayed parameters and tokens: 6ND", fontsize=8.6)
    ax.set_ylabel("parameters N (source-specific convention)", fontsize=8.6)
    ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "check-8-frontiers.png"), facecolor="white")
    plt.close(fig)


# ---------------- 图 8.3：自家 L(N) 双拟合 + 无 warmup 倒挂（数据=log/book2-ch08/） ----------------
def fig_8_3():
    fit = json.loads((LOG_DIR / f"{FIT_PREFIX}.json").read_text())
    rows = {}
    with open(LOG_DIR / f"{FIT_PREFIX}_curves.csv", encoding="utf-8") as f:
        next(f)
        for line in f:
            kind, n, l = line.strip().split(",")
            rows.setdefault(kind, ([], []))[0].append(float(n))
            rows[kind][1].append(float(l))
    aB = fit["fit_fixed0"]["params"]["alpha"]
    a_lo, a_hi = fit["fit_fixed0"]["bootstrap_CI"]["alpha"]
    A_fit = fit["fit_fixed0"]["params"]["A"]
    E_fit = fit["fit_free"]["params"]["L_inf"]
    e_lo,e_hi = fit["fit_free"]["bootstrap_CI"]["L_inf"]
    grid = np.array(rows["fit_fixed0"][0])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8), dpi=300)
    a = axes[0]
    # alpha的边际区间不能固定A后冒充预测带，故只报告区间文字。
    a.plot(grid, rows["fit_fixed0"][1], color=BLUE, lw=2,
           label=f"fixed E=0, log-loss OLS: A={A_fit:.3f}, alpha={aB:.4f}")
    a.plot(grid, rows["fit_free"][1], color=ORANGE, lw=1.6, ls="--",
           label=f"free E, raw-loss residuals: E={E_fit:.3f}")
    a.plot(rows["measured"][0], rows["measured"][1], "o", ms=7, color=DARK, label=f"supplied endpoints ({len(rows['measured'][0])})")
    a.set_xscale("log")
    a.set_xlabel("parameters $N$ (input convention)", fontsize=8.4)
    a.set_ylabel("eval loss $L$ (nats/token)", fontsize=8.4)
    interval = lambda lo,hi: "not estimated" if lo is None or hi is None else f"[{lo:.4f}, {hi:.4f}]"
    a.text(.03,.04, f"conditional case intervals: alpha {interval(a_lo,a_hi)}\nE {interval(e_lo,e_hi)}; not a prediction interval",
           transform=a.transAxes, fontsize=6.8, color=MUTED, va="bottom")
    a.grid(True, color=GRID, lw=0.5); a.set_axisbelow(True)
    a.legend(frameon=False, fontsize=7.0, loc="upper center", bbox_to_anchor=(.5,-.22))
    a.set_title("(A) supplied endpoints and two fitting objectives", fontsize=8.6)
    b = axes[1]
    fast = json.loads((LOG_DIR / FAMILY_FILE).read_text())
    nw = json.loads((LOG_DIR / NOWARMUP_FILE).read_text())
    pts = lambda r: (np.array([p["N"] for p in r["points"]], dtype=float),
                     np.array([p["eval_final"] for p in r["points"]]))
    nf, lf = pts(fast)
    nn, ln = pts(nw)
    b.plot(nf, lf, "o-", color=BLUE, lw=2, ms=7, label=f"family: {Path(FAMILY_FILE).stem}")
    b.plot(nn, ln, "s--", color=ORANGE, lw=2, ms=7, label=f"comparison: {Path(NOWARMUP_FILE).stem}")
    # 两组族的终点不是另一份独立7m A/B记录；不把4.710硬标为4.715。
    if not np.isfinite(lf).all() or not np.isfinite(ln).all():
        raise ValueError("族结果包含无效损失，不能画成有效对照")
    b.set_xscale("log")
    b.set_xlabel("parameters $N$ (input convention)", fontsize=8.4)
    b.set_ylabel("final eval loss $L$", fontsize=8.4)
    b.grid(True, color=GRID, lw=0.5); b.set_axisbelow(True)
    b.legend(frameon=False, fontsize=7.2, loc="upper center", bbox_to_anchor=(.5,-.22))
    b.set_title("(B) supplied family endpoints, not training curves", fontsize=8.6)
    fig.tight_layout(rect=(0,.14,1,1))
    fig.savefig(os.path.join(FIG_DIR, "check-8-family-fit.png"), facecolor="white")
    plt.close(fig)


def main():
    global FIG_DIR, LOG_DIR, FIT_PREFIX, FAMILY_FILE, NOWARMUP_FILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", choices=["theory","family","all"], default="theory")
    ap.add_argument("--log-dir", type=Path, default=LOG_DIR)
    ap.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--fit-prefix", default=FIT_PREFIX)
    ap.add_argument("--family-file", default=FAMILY_FILE)
    ap.add_argument("--nowarmup-file", default=NOWARMUP_FILE)
    args = ap.parse_args()
    LOG_DIR, FIG_DIR = args.log_dir, args.fig_dir
    FIT_PREFIX, FAMILY_FILE, NOWARMUP_FILE = args.fit_prefix, args.family_file, args.nowarmup_file
    if args.plots in ("family","all"):
        for filename in (f"{FIT_PREFIX}.json", f"{FIT_PREFIX}_curves.csv", FAMILY_FILE, NOWARMUP_FILE):
            if not (LOG_DIR / filename).is_file():
                raise FileNotFoundError(f"缺少{filename}；仅核算理论图可使用--plots theory")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if args.plots in ("theory","all"):
        fig_8_1();fig_8_2()
    if args.plots in ("family","all"):
        fig_8_3()
    print("产物目录:", FIG_DIR)

if __name__ == "__main__":
    main()
