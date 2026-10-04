# make_figures.py —— Book3 ch4 六张图的绘制脚本（图 4.1 二维旋转 / 图 4.2 块对角+整机 / 图 4.3 波长谱 /
#                    图 4.4 衰减曲线 / 图 4.5 实现三型 / 图 4.6 换件消融+外推）
# 用途：产出 figures/fig-4-{1..6}-*.png（300dpi 印刷规格，图内英文、
#       图注中文在正文）；图 4.6 依赖 log/book3-ch04/ablation_pe_{fast,full}.json（消融产物），
#       其余五张纯数学自算、不依赖实验数据。图 4.4 的衰减量 = 论文式 (37) 的数值化（本机复算）。
# 所属章节：Book3 ch4（4.3 图 4.1 / 4.4 图 4.2-4.3 / 4.5 图 4.4 / 4.7 图 4.5 / 4.9 图 4.6）。
# 运行方式：cd code/ch04 && python make_figures.py [--only 1,2,3,4,5] [--data auto]
#       （--data auto：优先 full、缺 full 自动落 fast 并在终端注明；图 4.6 不在 --only 里时跳过数据检查）
# 规格：写作规范 §4——蓝 #2a78d6 / 橙 #eb6834 / 青 #1baf7a / 黄 #eda100；线宽 2；网格 hairline
#       #e1e0d9 置底层；刻度 #898781；标注 #0b0b0b / #52514e；禁双 y 轴；tight_layout。
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DIR = os.path.join(REPO, "figures")
LOG_DIR = os.path.join(REPO, "log", "book3-ch04")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MID = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
SEQ_LIGHT, SEQ_DARK = "#cde2fb", "#0d366b"     # 顺序型蓝色单色渐变（块深浅用）
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9.5,
    "axes.edgecolor": TICK, "axes.labelcolor": DARK, "xtick.color": TICK,
    "ytick.color": TICK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "font.family": "sans-serif",
})


def style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def inv_freqs(head_dim, theta=10000.0):
    """θ_i = theta^{-2i/d}，i=0..d/2-1（0 起口径，HF 同款）。"""
    return theta ** (-2.0 * np.arange(head_dim // 2) / head_dim)


# ---------------- 图 4.1：二维旋转直觉（同一向量按位置旋转 + 内积只依赖 m−n） ----------------
def fig_4_1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # (a) 同一单位向量在 m=0..8、三档角速度下的旋转轨迹
    th = np.linspace(0, 2 * np.pi, 256)
    ax1.plot(np.cos(th), np.sin(th), color=GRID, lw=1.0)
    z = np.exp(0.35j)                                   # 固定的「内容」：模长 1、初始辐角 0.35
    for c, w, lab in ((BLUE, 1.0, r"$\theta=1.0$"), (ORANGE, 0.5, r"$\theta=0.5$"), (TEAL, 0.25, r"$\theta=0.25$")):
        m = np.arange(9)
        pts = z * np.exp(1j * w * m)
        ax1.plot(pts.real, pts.imag, "-o", color=c, lw=1.6, ms=5, label=lab)
        ax1.annotate("", xy=(pts.real[-1], pts.imag[-1]), xytext=(pts.real[-2], pts.imag[-2]),
                     arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4))
    ax1.plot([z.real], [z.imag], "k*", ms=9)
    ax1.text(z.real + 0.06, z.imag + 0.05, "$z$ (pos 0)", fontsize=8, color=DARK)
    ax1.set_xlim(-1.25, 1.35); ax1.set_ylim(-1.3, 1.35)
    ax1.set_aspect("equal"); ax1.grid(False)
    ax1.set_title("(a) one vector, rotated by position", fontsize=9.5, color=DARK)
    ax1.set_xlabel("dim 1"); ax1.set_ylabel("dim 2")
    ax1.legend(loc="lower left", fontsize=7.5, framealpha=0.9)

    # (b) 单位向量的内积只依赖 m−n（移动整段文本曲线不动）
    d = np.arange(0, 21)
    for c, w, lab in ((BLUE, 1.0, r"$\theta=1.0$"), (ORANGE, 0.5, r"$\theta=0.5$"), (TEAL, 0.25, r"$\theta=0.25$")):
        ax2.plot(d, np.cos(w * d), "-o", color=c, lw=2, ms=4.5, label=lab)
    ax2.axhline(0, color=MID, lw=0.8)
    ax2.annotate("shift the whole text by +1000:\ncurve unchanged (only $m{-}n$ enters)",
                 xy=(3, np.cos(3.0)), xytext=(6.5, 0.72), fontsize=7.5, color=MID,
                 arrowprops=dict(arrowstyle="->", color=MID, lw=1.0))
    ax2.set_xlabel("relative distance $m-n$"); ax2.set_ylabel(r"$\langle q_m, k_n\rangle$")
    ax2.set_title("(b) inner product depends only on $m{-}n$", fontsize=9.5, color=DARK)
    ax2.legend(fontsize=7.5)
    style_axis(ax2)

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)   # 给含 mathtext 的两行标题留高度，防顶裁
    fig.savefig(os.path.join(FIG_DIR, "fig-4-1-2d-rotation-intuition.png"))
    plt.close(fig)


# ---------------- 图 4.2：块对角旋转矩阵 + 整机动刀位置（形状流转） ----------------
def fig_4_2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.4), width_ratios=[1, 1.35])

    # (a) R^d_{Theta,m} 的块对角结构（示意 d_k=8：4 个 2x2 块；实际 d_k=64 有 32 块）
    ax1.set_xlim(-0.6, 8.6); ax1.set_ylim(-1.0, 8.2)
    ax1.axis("off"); ax1.grid(False); ax1.set_aspect("equal")
    ax1.set_title("(a) $R^d_{\\Theta,m}$: block-diagonal", fontsize=9.5, color=DARK)
    ax1.add_patch(plt.Rectangle((0, 0), 8, 8, fc="#f7f6f2", ec="#e9e8e2", lw=0.6))   # 零区域底色
    shades = [SEQ_DARK, "#3564a8", "#7ba3d9", SEQ_LIGHT]      # 快块深、慢块浅
    for b in range(4):                                        # 对角线上第 b 个 2x2 旋转块
        x0, y0 = 2 * b, 6 - 2 * b                             # 左下角（y 自下而上）
        ax1.add_patch(plt.Rectangle((x0, y0), 2, 2, fc=shades[b], ec="white", lw=1.6))
        ax1.text(x0 + 1, y0 + 1, f"$m\\theta_{b}$", ha="center", va="center",
                 fontsize=8, color="white" if b < 2 else DARK)
    ax1.text(4.0, -0.55, r"$\theta_0=1$ (fast) $\cdots$ $\theta_{d/2-1}\!\approx\!10^{-4}$ (slow)",
             ha="center", fontsize=7.5, color=MID)

    # (b) 整机动刀位置：入口删 wpe，位置改在每层每头 Q/K 上现做
    ax2.set_xlim(0, 100); ax2.set_ylim(0, 58)
    ax2.axis("off"); ax2.grid(False)
    ax2.set_title("(b) where RoPE sits in the machine", fontsize=9.5, color=DARK)

    def box(x, y, w, h, label, sub="", ec=MID, fc="white", ls="-", fs=8.5, lw=1.4):
        ax2.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=2))
        ax2.text(x + w / 2, y + h / 2 + (2.0 if sub else 0), label, ha="center", va="center",
                 fontsize=fs, color=DARK, zorder=3)
        if sub:
            ax2.text(x + w / 2, y + h / 2 - 2.2, sub, ha="center", va="center", fontsize=7, color=MID, zorder=3)

    def arr(x1, y1, x2, y2, label="", ls="-", color=MID):
        ax2.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, linestyle=ls))
        if label:
            ax2.text((x1 + x2) / 2, max(y1, y2) + 1.0, label, ha="center", fontsize=7, color=MID)

    # 入口
    box(2, 24, 16, 10, "$w_{te}$ lookup", "$(B,n){\\to}(B,n,C)$")
    box(2, 8, 16, 9, "$w_{pe}$ removed", "$(1024,C)$, dashed", ec="#b9b7af", ls=(0, (3, 2)), fc="#f4f3ef")
    ax2.text(10, 4.6, "no position added\nat the input", ha="center", fontsize=7, color=MID)
    arr(10, 24, 10, 39)
    # Block 放大
    box(24, 20, 46, 30, "", "")
    ax2.text(47, 46.5, "Block $\\times L$   (pre-RMSNorm + attn + SwiGLU, ch2-3)", ha="center", fontsize=8, color=DARK)
    box(27, 36, 12, 7, "$q,k$ proj", "$(B,n,C)$")
    box(27, 26, 12, 8, "RoPE", "$(B,h,n,d_k)$", ec=ORANGE, lw=2.0)
    box(43, 26, 11, 8, "$v$ proj", "$(B,n,C)$\nno rotation", ec="#b9b7af")
    box(58, 26, 9, 8, "$q k^\\top$", "$(B,h,n,n)$")
    arr(33, 36, 33, 34)
    arr(39, 30, 43, 30)
    arr(54, 30, 58, 30)
    ax2.text(33, 22.5, "position enters here:\nrotate $q,k$ per 2-D block", ha="center", fontsize=7, color=ORANGE)
    # 出口
    arr(70, 35, 78, 35)
    box(78, 30, 18, 10, "ln_f + head", "$(B,n,C){\\to}(B,n,V)$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-4-2-block-diagonal-and-machine.png"))
    plt.close(fig)


# ---------------- 图 4.3：波长谱（d=128 档逐对波长；快慢指针） ----------------
def fig_4_3():
    d = 128
    lam = 2 * np.pi / inv_freqs(d)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    i = np.arange(d // 2)
    ax.plot(i, lam, "-o", color=BLUE, lw=2, ms=3.5, label=r"$\lambda_i = 2\pi/\theta_i$  ($d_k$=128)")
    ax.set_yscale("log")
    for y, lab, c in ((512, "module tier $T$=512 (this book)", MID), (2048, "LLaMA1 train window 2048", TEAL)):
        ax.axhline(y, color=c, lw=1.1, ls=(0, (4, 2)))
        ax.text(1.0, y * 1.12, lab, fontsize=7.5, color=c)
    for xi, yi, txt in ((0, lam[0], "$\\lambda_0$=6.3"), (32, lam[32], "$\\lambda_{32}$=628"),
                        (63, lam[63], "$\\lambda_{63}$=54,410")):
        ax.annotate(txt, xy=(xi, yi), xytext=(xi - 1.5, yi * 1.55), fontsize=7.5, color=DARK,
                    arrowprops=dict(arrowstyle="->", color=MID, lw=0.9))
    ax.text(33, 9.2, "fast hands:\nmany cycles inside\nthe train window", fontsize=7.5, color=MID)
    ax.text(14, 2.6e4, "slow hands:\nnever finish one cycle\n($\\lambda\\approx 2\\pi\\cdot$base)", fontsize=7.5, color=MID)
    ax.set_xlabel("2-D block index $i$ (0 = fastest)")
    ax.set_ylabel("wavelength $\\lambda_i$ (tokens per cycle)")
    ax.legend(loc="center left", fontsize=7.5)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-4-3-wavelength-spectrum.png"))
    plt.close(fig)


# ---------------- 图 4.4：长程衰减上界（论文式 37 的数值化，d=64/128 双臂） ----------------
def decay_curve(d, dists, theta=10000.0):
    """(2/d) * sum_j |S_j|，S_j = sum_{i<j} exp(i*delta*theta_i)——论文 Figure 2 的复算。"""
    th = inv_freqs(d, theta)
    out = []
    for delta in dists:
        phasors = np.exp(1j * delta * th)              # (d/2,)
        S = np.cumsum(phasors)                         # S_j, j=1..d/2
        out.append(np.abs(S).sum() / (d / 2))
    return np.array(out)


def fig_4_4():
    dists = np.arange(0, 251)
    y64, y128 = decay_curve(64, dists), decay_curve(128, dists)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(dists, y128, color=BLUE, lw=2, label="$d$=128 (LLaMA head dim)")
    ax.plot(dists, y64, color=ORANGE, lw=2, label="$d$=64 (book tiers)")
    i50, i100 = 50, 100
    ax.plot([50], [y64[i50]], "o", color=ORANGE, ms=6)
    ax.plot([100], [y64[i100]], "o", color=ORANGE, ms=6)
    ax.annotate(f"non-monotonic: {y64[i100]:.1f} at dist 100\n> {y64[i50]:.1f} at dist 50",
                xy=(100, y64[i100]), xytext=(128, y64[i100] + 3.2), fontsize=7.5, color=MID,
                arrowprops=dict(arrowstyle="->", color=MID, lw=0.9))
    ax.annotate(f"32.5 $\\to$ 6.5", xy=(250, y128[-1]), xytext=(196, 14.5), fontsize=8, color=DARK,
                arrowprops=dict(arrowstyle="->", color=MID, lw=0.9))
    ax.set_xlabel("relative distance $m-n$")
    ax.set_ylabel(r"upper-bound scale  $\frac{2}{d}\sum_j |S_j|$")
    ax.set_ylim(0, 34)
    ax.legend(fontsize=7.5)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-4-4-decay-upper-bound.png"))
    plt.close(fig)
    print(f"  [fig 4.4] d=128: {y128[0]:.1f} (dist 0) -> {y128[-1]:.1f} (dist 250)；"
          f"d=64: {y64[0]:.1f} -> {y64[-1]:.1f}；d=64 dist50={y64[i50]:.2f} dist100={y64[i100]:.2f}（回勾）")


# ---------------- 图 4.5：实现三型对照（相邻配对复数/相邻配对实数/两半式 rotate_half） ----------------
def fig_4_5():
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 4.6))
    titles = ("(a) complex interleaved  —  Meta llama repo (view_as_complex)",
              "(b) real interleaved  —  llama2.c",
              "(c) half-split rotate_half  —  HF LLaMA / GPT-NeoX")
    pairs = {0: [(0, 1), (2, 3), (4, 5), (6, 7)],
             1: [(0, 1), (2, 3), (4, 5), (6, 7)],
             2: [(0, 4), (1, 5), (2, 6), (3, 7)]}
    for r, ax in enumerate(axes):
        ax.set_xlim(-0.7, 8.2); ax.set_ylim(-0.1, 2.5)
        ax.axis("off"); ax.grid(False)
        ax.set_title(titles[r], fontsize=8.5, color=DARK, loc="left")
        for j in range(8):
            ax.add_patch(plt.Rectangle((j, 0.5), 0.92, 0.9, fc="#f4f3ef", ec=MID, lw=1.0))
            ax.text(j + 0.46, 0.95, f"$x_{j + 1}$", ha="center", va="center", fontsize=8, color=DARK)
        for a, b in pairs[r]:
            ax.plot([a + 0.46, b + 0.46], [1.75, 1.75], color=ORANGE if r == 2 else BLUE, lw=1.8)
            ax.plot([a + 0.46, a + 0.46], [1.45, 1.75], color=ORANGE if r == 2 else BLUE, lw=1.2)
            ax.plot([b + 0.46, b + 0.46], [1.45, 1.75], color=ORANGE if r == 2 else BLUE, lw=1.2)
        if r == 2:                                   # 两半式的 rotate_half 说明
            ax.text(4.45, 0.18, r"pair $(x_j,\,x_{j+d/2})$;  rotate: $x\cos + \mathrm{rot}(x)\sin$",
                    fontsize=7, color=MID, ha="center")
        else:
            ax.text(4.45, 0.18, r"pair $(x_{2j},\,x_{2j+1})$ as complex $x_{2j}+i\,x_{2j+1}$",
                    fontsize=7, color=MID, ha="center")
    axes[2].text(8.35, 0.95, "", fontsize=7)
    fig.text(0.5, 0.005, "mathematically equal up to a fixed permutation $\\tau$ (max|Δ| ≤ 2.4e-7);"
                          "  weights NOT interchangeable ($W_q, W_k$ need permute)",
             ha="center", fontsize=7.5, color=DARK)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(FIG_DIR, "fig-4-5-three-impl-styles.png"))
    plt.close(fig)


# ---------------- 图 4.6：换件消融曲线 + 4× 外推分桶（依赖 ablation_pe 产物） ----------------
def fig_4_6(which="auto"):
    path = None
    if which in ("auto", "full"):
        p = os.path.join(LOG_DIR, "ablation_pe_full.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                rep = json.load(f)
            if rep.get("arms", {}).get("rope", {}).get("extrap_2048"):
                path, tag = p, "full"
    if path is None:
        p = os.path.join(LOG_DIR, "ablation_pe_fast.json")
        with open(p, encoding="utf-8") as f:
            rep = json.load(f)
        path, tag = p, "fast"
    print(f"  [fig 4.6] 数据源：{os.path.basename(path)}（{tag} 档）")

    wpe, rope = rep["arms"]["wpe"], rep["arms"]["rope"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    for arm, c, lab in ((wpe, BLUE, "wpe (learned, 1024-table)"), (rope, ORANGE, "RoPE (no table)")):
        steps = [e["step"] for e in arm["evals"]]
        losses = [e["eval_loss"] for e in arm["evals"]]
        ses = [e["eval_se"] for e in arm["evals"]]
        ax1.errorbar(steps, losses, yerr=ses, fmt="-o", color=c, lw=2, ms=4, capsize=2.5, label=lab)
    ax1.set_xlabel("step"); ax1.set_ylabel("fixed-window eval loss (512, fp32)")
    ax1.set_title(f"(a) swap ablation, {tag} tier (T=512)", fontsize=9, color=DARK)
    ax1.legend(fontsize=7.5)
    style_axis(ax1)

    buckets = rope["extrap_2048"]["buckets"]
    labels = ["[0,512)", "[512,1024)", "[1024,1536)", "[1536,2048)"]
    vals = [buckets[f"pos[{a}:{b}]"] for a, b in ((0, 512), (512, 1024), (1024, 1536), (1536, 2048))]
    bars = ax2.bar(range(4), vals, color=ORANGE, width=0.62)
    bars[0].set_color("#f0a583")
    ax2.axhline(rope["eval_final"], color=MID, lw=1.2, ls=(0, (4, 2)))
    ax2.text(2.6, rope["eval_final"] - 0.028, f"in-window eval {rope['eval_final']:.4f}",
             fontsize=7.5, color=MID, ha="center")
    ax2.text(1.5, max(vals) + 0.012, f"4x extrapolation loss {rope['extrap_2048']['loss']:.4f}"
             f"  (ppl {rope['extrap_2048']['ppl']:.1f})", fontsize=7.5, color=DARK, ha="center")
    ax2.text(1.5, min(vals) - 0.055, "wpe arm at 2048: table-overflow assert (hard wall)",
             fontsize=7.5, color=BLUE, ha="center")
    ax2.set_xticks(range(4)); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylim(min(vals) - 0.08, max(vals) + 0.05)
    ax2.set_xlabel("position bucket (train window = first bucket)")
    ax2.set_ylabel("bucket eval loss (fp32)")
    ax2.set_title("(b) RoPE 4x extrapolation buckets", fontsize=9, color=DARK)
    style_axis(ax2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-4-6-swap-ablation-extrapolation.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="1,2,3,4,5,6")
    ap.add_argument("--data", type=str, default="auto", choices=["auto", "fast", "full"])
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    todo = {int(x) for x in args.only.split(",")}
    if 1 in todo:
        fig_4_1(); print("  [fig 4.1] done")
    if 2 in todo:
        fig_4_2(); print("  [fig 4.2] done")
    if 3 in todo:
        fig_4_3(); print("  [fig 4.3] done")
    if 4 in todo:
        fig_4_4()
    if 5 in todo:
        fig_4_5(); print("  [fig 4.5] done")
    if 6 in todo:
        fig_4_6(args.data)
    print(f"make_figures.py 完成 -> {FIG_DIR}/fig-4-*.png")


if __name__ == "__main__":
    main()
