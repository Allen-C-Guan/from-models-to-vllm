# 用途：Book2 ch8 三张图——图 8.1 Kaplan 幂律三组（按原文 Table 4 常数重绘拟合曲线；表号勘正 2026-10-03：单变量拟合常数在 Table 4，compute-efficient 参数在 Table 5）
#   / 图 8.2 Kaplan vs Chinchilla 最优线对照（Chinchilla Table 3 数据点 + D.4 锚点）
#   / 图 8.3 自家 L(N) 双拟合与无 warmup 倒挂对照（数据全部来自 log/book2-ch08/）
# 所属章节：Book2 第 8 章 §8.2、§8.3、§8.5
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch08/make_figures.py"
# 产物：figures/fig-8-{1,2,3}-*.png（300dpi，图内文字英文、图注见正文）

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(REPO, "figures")
LOG_DIR = os.path.join(REPO, "log", "book2-ch08")
os.makedirs(FIG_DIR, exist_ok=True)

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MUTED = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": TICK, "axes.labelcolor": DARK,
    "xtick.color": TICK, "ytick.color": TICK, "text.color": DARK,
    "axes.linewidth": 0.8, "font.family": "DejaVu Sans",
})


# ---------------- 图 8.1：Kaplan 幂律三组（按 v1 Table 4 拟合常数重绘的曲线族，非原文散点） ----------------
def fig_8_1():
    # (Kaplan 2001.08361 v1 Table 4 逐字常数)
    panels = [
        ("L(N) = (Nc/N)^a_N", 8.8e13, 0.076, r"$N$ params (non-embedding)",
         1e3, 1e9, r"$N_c = 8.8\times10^{13}$, $\alpha_N = 0.076$", BLUE),
        ("L(D) = (Dc/D)^a_D", 5.4e13, 0.095, r"$D$ tokens",
         1e6, 1e11, r"$D_c = 5.4\times10^{13}$, $\alpha_D = 0.095$", TEAL),
        ("L(Cmin) = (Cc/Cmin)^a_C", 3.1e8, 0.050, r"$C_{min}$ PF-days (at $B_{crit}$)",
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
    fig.savefig(os.path.join(FIG_DIR, "fig-8-1-kaplan-three-powerlaws.png"), facecolor="white")
    plt.close(fig)


# ---------------- 图 8.2：Kaplan vs Chinchilla 最优线对照（两文官方数据自绘） ----------------
def fig_8_2():
    # Chinchilla v1 Table 3（方法一最优点）：(params, FLOPs, tokens)——FLOPs=6ND 逐行自洽
    t3 = np.array([
        [4e8, 1.92e19, 8.0e9], [1e9, 1.21e20, 2.02e10], [1e10, 1.23e22, 2.051e11],
        [6.7e10, 5.76e23, 1.5e12], [1.75e11, 3.85e24, 3.7e12], [2.8e11, 9.90e24, 5.9e12],
        [5.2e11, 3.43e25, 1.1e13], [1e12, 1.27e26, 2.12e13],
    ])
    C = np.logspace(18.5, 26, 200)
    n_chinchilla = 6.7e10 * (C / 5.76e23) ** 0.50        # 等比线：斜率 0.50（Table 2 方法一）
    n_kaplan = 4.68e9 * (C / 1e21) ** 0.73               # Kaplan 线：斜率 0.73，锚点=附录 D.4 按其律算出的 4.68B@1e21
    # 实配模型点：(FLOPs, params, tokens)——GPT-3 Table D.1；Gopher/Chinchilla/MT-NLG=Chinchilla 表 1+6ND
    models = [
        ("GPT-3 175B\n300B tok (D/N=1.7)", 3.14e23, 1.75e11, 3.0e11, ORANGE, (10, -20)),
        ("Gopher 280B\n300B tok (D/N=1.1)", 5.76e23, 2.8e11, 3.0e11, ORANGE, (6, 4)),
        ("MT-NLG 530B\n270B tok", 8.6e23, 5.3e11, 2.7e11, MUTED, (6, -14)),
        ("Chinchilla 70B\n1.4T tok (D/N=20)", 5.76e23, 7.0e10, 1.4e12, TEAL, (-10, 12)),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=300)
    ax.plot(C, n_kaplan, ls="--", lw=2, color=BLUE,
            label="Kaplan law: slope 0.73")
    ax.plot(C, n_chinchilla, ls="-.", lw=2, color=TEAL,
            label="Chinchilla App.1: slope 0.50 (D/N $\\approx$ 20)")
    ax.plot(t3[:, 1], t3[:, 0], "o", ms=7, color=TEAL, label="Chinchilla Table 3 optimal points")
    for label, c, n, d, color, off in models:
        ax.plot([c], [n], "*", ms=13, color=color, zorder=5)
        ax.annotate(label, (c, n), textcoords="offset points", xytext=off, fontsize=7.2, color=color)
    ax.annotate("", xy=(5.76e23, 7.0e10), xytext=(5.76e23, 2.8e11),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.text(3.2e23, 1.4e11, "same budget:\n280B -> 70B", fontsize=7.4, color=MUTED)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(2e18, 3e26); ax.set_ylim(1e8, 3e12)
    ax.set_xlabel("training compute $C$ (FLOPs)", fontsize=8.6)
    ax.set_ylabel("optimal non-emb. params $N^*$", fontsize=8.6)
    ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-8-2-optimal-lines.png"), facecolor="white")
    plt.close(fig)


# ---------------- 图 8.3：自家 L(N) 双拟合 + 无 warmup 倒挂（数据=log/book2-ch08/） ----------------
def fig_8_3():
    fit = json.load(open(os.path.join(LOG_DIR, "fit_kaplan.json"), encoding="utf-8"))
    rows = {}
    with open(os.path.join(LOG_DIR, "fit_kaplan_curves.csv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            kind, n, l = line.strip().split(",")
            rows.setdefault(kind, ([], []))[0].append(float(n))
            rows[kind][1].append(float(l))
    aB = fit["fit_fixed0"]["params"]["alpha"]
    a_lo, a_hi = fit["fit_fixed0"]["bootstrap_CI"]["alpha"]
    A_med = fit["fit_fixed0"]["bootstrap_CI"]["A_median"]
    grid = np.array(rows["fit_fixed0"][0])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), dpi=300)
    a = axes[0]
    a.fill_between(grid, A_med * grid ** (-a_lo), A_med * grid ** (-a_hi),
                   color=BLUE, alpha=0.18, label="95% CI envelope of $\\alpha$ (A at median)")
    a.plot(grid, rows["fit_fixed0"][1], color=BLUE, lw=2,
           label=f"fit B: $L=9.02\\,N^{{-\\alpha}}$, $\\alpha$={aB:.4f}")
    a.plot(grid, rows["fit_free"][1], color=ORANGE, lw=1.6, ls="--",
           label="fit A: $L=L_\\infty+A\\,N^{-\\alpha}$ ($L_\\infty\\to0$)")
    a.plot(rows["measured"][0], rows["measured"][1], "o", ms=7, color=DARK, label="measured (5 pts)")
    a.set_xscale("log")
    a.set_xlabel("non-embedding params $N$", fontsize=8.4)
    a.set_ylabel("eval loss $L$ (nats/token)", fontsize=8.4)
    a.text(0.05, 0.06, f"$\\alpha$ CI [{a_lo:.4f}, {a_hi:.4f}]   $L_\\infty$ CI [0.00, 3.34] (unidentifiable)",
           transform=a.transAxes, fontsize=7.2, color=MUTED, va="bottom")
    a.grid(True, color=GRID, lw=0.5); a.set_axisbelow(True)
    a.legend(frameon=False, fontsize=7.0, loc="upper right")
    a.set_title("(A) official family (warmup=200): power law", fontsize=8.6)
    b = axes[1]
    fast = json.load(open(os.path.join(LOG_DIR, "family_fast.json"), encoding="utf-8"))
    nw = json.load(open(os.path.join(LOG_DIR, "family_nowarmup.json"), encoding="utf-8"))
    pts = lambda r: (np.array([p["N"] for p in r["points"]], dtype=float),
                     np.array([p["eval_final"] for p in r["points"]]))
    nf, lf = pts(fast)
    nn, ln = pts(nw)
    b.plot(nf, lf, "o-", color=BLUE, lw=2, ms=7, label="warmup=200 (official family)")
    b.plot(nn, ln, "s--", color=ORANGE, lw=2, ms=7, label="no warmup (schedule-inadequate)")
    i7 = list(nf).index(7108352)
    b.annotate("", xy=(nf[i7], lf[i7]), xytext=(nn[i7], ln[i7]),
               arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    b.text(7.4e6, 4.99, "A/B at 7M:\nwarmup fixes 0.55 nat", fontsize=7.2, color=MUTED)
    b.set_xscale("log")
    b.set_xlabel("non-embedding params $N$", fontsize=8.4)
    b.set_ylabel("final eval loss $L$", fontsize=8.4)
    b.grid(True, color=GRID, lw=0.5); b.set_axisbelow(True)
    b.legend(frameon=False, fontsize=7.2, loc="upper right")
    b.set_title("(B) without warmup, L(N) inverts", fontsize=8.6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-8-3-family-fit.png"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_8_1()
    fig_8_2()
    fig_8_3()
    for f in sorted(os.listdir(FIG_DIR)):
        if f.startswith("fig-8-"):
            print("产物:", os.path.join(FIG_DIR, f))
