# fig_ch02.py —— Book5 ch2 制图：图 2.1 banded mask 与 sink 数据流 / 图 2.2 三家窗口层带 /
#                图 2.3 needle 位置扫描两臂曲线（编号=正文出现顺序；slug=swa-sink 防跨册冲撞）
# 用途：图 2.1 = 2.2/2.3 机制总图（SlidingWindowSinkAttention 数据流，全部张量框标形状、
#         mask 分块 + sink 标量进分母两处高亮；配例 2.1 的 4x4 手画矩阵为小图）；
#       图 2.2 = 2.2 三家 config 事实的层带对比（Gemma2 1:1 窗 4096 / Gemma3 5:1 窗 1024 /
#         gpt-oss 1:1 窗 128——窗口越开越窄、全局层越留越稀）；
#       图 2.3 = 2.6 needle 位置扫描（喂 log/book5-ch02/needle_scan.csv：full vs swa_sink 两臂、
#         20k 正档实线+8k 参照虚线、复合视野边界 p=31/32 与单层悬崖 p=46 两条竖线）。
# 所属章节：Book5 第 2 章（图 2.1-2.3）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch02/fig_ch02.py"
#           [--which all|dataflow|bands|needle]
# 产物：figures/fig-2-{1,2,3}-swa-sink.png
# 图规：300dpi 白底；系列色 蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；线宽 2；图内英文、图注中文。
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "book5-frontier-hybrid-architectures", "figures")
SCAN_CSV = os.path.join(REPO_ROOT, "log", "book5-ch02", "needle_scan.csv")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e1e0d9"
BAND_FC = "#dcebfc"          # 进 mask/分母的量用浅蓝底


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=8.0, lw=1.5, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc, zorder=4)
    return x + w / 2, y + h / 2


def arrow(ax, x0, y0, x1, y1, label=None, color=MUTED, fs=7.0, dx=0.0, dy=0.06, lw=1.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=lw, zorder=2))
    if label:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, label, ha="center", va="bottom",
                fontsize=fs, color=color, zorder=4)


# ---------------- 图 2.1：banded mask 与 sink 数据流 ----------------
def fig_dataflow():
    fig, ax = plt.subplots(figsize=(10.4, 5.6), dpi=300)
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    ax.text(6.2, 7.15, "SlidingWindowSinkAttention: banded mask + per-head learnable sink",
            ha="center", fontsize=11, color=INK, weight="bold")

    # 主链：x → qkv → scores → mask → cat sink → softmax → drop → ·V → o_proj
    box(ax, 0.2, 5.6, 1.5, 0.9, "x\n(B,n,d)", ec=MUTED)
    box(ax, 2.2, 5.6, 1.7, 0.9, "q/k/v proj\n(d→h·d_k)", ec=TEAL)
    box(ax, 4.4, 5.6, 2.3, 0.9, "S = q·kᵀ/√d_k\n(B,h,n,n)", ec=BLUE)
    box(ax, 7.2, 5.6, 2.1, 0.9, "⊕ banded mask\n(n,n)  -inf out", ec=ORANGE, fc=BAND_FC)
    box(ax, 9.7, 5.6, 1.9, 0.9, "cat sink b_h\n(B,h,n,n+1)", ec=ORANGE, fc=BAND_FC)
    arrow(ax, 1.7, 6.05, 2.2, 6.05)
    arrow(ax, 3.9, 6.05, 4.4, 6.05)
    arrow(ax, 6.7, 6.05, 7.2, 6.05)
    arrow(ax, 9.3, 6.05, 9.7, 6.05)
    box(ax, 9.7, 4.2, 1.9, 0.9, "softmax → drop\n(B,h,n,n)", ec=BLUE)
    arrow(ax, 10.65, 5.6, 10.65, 5.1)
    box(ax, 6.9, 4.2, 2.2, 0.9, "p·V\n(B,h,n,d_k)", ec=BLUE)
    arrow(ax, 9.7, 4.65, 9.1, 4.65)
    box(ax, 4.3, 4.2, 2.1, 0.9, "o_proj\n(B,n,d)", ec=TEAL)
    arrow(ax, 6.9, 4.65, 6.4, 4.65)
    ax.text(6.2, 3.7, "weights over window sum to (1 − α_sink): relative ratios inside window unchanged",
            ha="center", fontsize=7.5, color=MUTED)

    # 下方左：4x4 banded mask 小图（例 2.1 同源，w=3）
    ax.text(1.75, 3.35, "banded mask M, w=3, n=4", fontsize=8.5, color=INK, ha="center", weight="bold")
    grid = [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1]]
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((0.55 + j * 0.6, 1.35 + i * 0.5), 0.58, 0.48,
                                   fc=BAND_FC if grid[i][j] else "white", ec=GRIDC, lw=0.8, zorder=3))
        ax.text(0.32, 1.35 + i * 0.5 + 0.24, f"q{i}", fontsize=7, color=MUTED, ha="right", va="center")
    for j in range(4):
        ax.text(0.84 + j * 0.6, 1.12, f"k{j}", fontsize=7, color=MUTED, ha="center")
    ax.text(1.75, 0.55, "row qi: only last W=3 keys visible", fontsize=7.5, color=MUTED, ha="center")

    # 下方右：sink 进分母的公式条
    box(ax, 3.6, 0.9, 8.4, 2.1,
        r"$\alpha_j = e^{s_j}\,/\,(e^{b_h} + \sum_j e^{s_j})$" + "\n"
        "b_h: one learnable scalar per head (zero-init)\n"
        r"$b_h \equiv 0 \to$ softmax$_1$ (off-by-one);   sink mass"
        r" $\alpha_{\rm sink} = e^{b_h}/(e^{b_h}+\sum_j e^{s_j})$",
        ec=ORANGE, fc=BAND_FC, fs=9)
    # b_h 从 cat sink 框沿右侧通道下到公式条（避免穿过 softmax 框）
    ax.plot([11.6, 12.15, 12.15, 10.6], [6.05, 6.05, 3.35, 3.35], color=ORANGE, lw=1.2, zorder=2)
    ax.add_patch(FancyArrowPatch((10.6, 3.35), (10.6, 3.0), arrowstyle="-|>", mutation_scale=10,
                                 color=ORANGE, lw=1.2, zorder=2))
    ax.text(12.15, 4.7, "b_h", fontsize=7.5, color=ORANGE, ha="left", va="center")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-2-1-swa-sink.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 2.1] {path}")


# ---------------- 图 2.2：三家窗口层带对比 ----------------
def fig_bands():
    rows = [  # (名称, 层数, 全局层集合规则, 窗宽, 上下文, 注记)
        ("Gemma2-9B (2024-06)\n1:1, W=4096, ctx 8192", 42, lambda i: (i + 1) % 2 == 0, "W/ctx = 50%"),
        ("Gemma3-27B (2025-03)\n5:1, W=1024, ctx 128K", 62, lambda i: (i + 1) % 6 == 0, "W/ctx = 0.78%"),
        ("gpt-oss-20b (2025-08)\n1:1, W=128, ctx 128K", 24, lambda i: (i + 1) % 2 == 0, "W/ctx = 0.098%"),
    ]
    fig, ax = plt.subplots(figsize=(11.0, 4.0), dpi=300)
    ax.set_xlim(-0.5, 92)
    ax.set_ylim(-0.9, 3.4)
    ax.axis("off")
    ax.text(45.75, 3.15, "Layer bands 2024→2025: window narrows, global layers thin out (never to zero)",
            ha="center", fontsize=10.5, color=INK, weight="bold")
    for r, (name, L, is_global, note) in enumerate(rows):
        y = 2.1 - r * 1.05
        for i in range(L):
            ax.add_patch(Rectangle((i, y), 0.92, 0.62, fc=ORANGE if is_global(i) else BLUE,
                                   ec="white", lw=0.4, zorder=3))
        ax.text(-1.2, y + 0.31, name, fontsize=7.6, color=INK, ha="left", va="center")
        ax.text(64.8, y + 0.31, f"{note} | global {sum(is_global(i) for i in range(L))}/{L}",
                fontsize=7.6, color=MUTED, ha="left", va="center")
    ax.add_patch(Rectangle((0.0, -0.72), 0.7, 0.34, fc=BLUE, ec="white", lw=0.4))
    ax.text(0.9, -0.55, "sliding (window W)", fontsize=7.6, color=INK, va="center")
    ax.add_patch(Rectangle((8.5, -0.72), 0.7, 0.34, fc=ORANGE, ec="white", lw=0.4))
    ax.text(9.4, -0.55, "global (full attention)", fontsize=7.6, color=INK, va="center")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-2-2-swa-sink.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 2.2] {path}")


# ---------------- 图 2.3：needle 位置扫描两臂曲线（喂 needle_scan.csv） ----------------
def fig_needle():
    rows = []
    with open(SCAN_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["arm"], int(r["steps"]), int(r["position"]), float(r["acc"])))
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=300)
    style = {("full", 20000): (BLUE, "-", "o", 2.2, "full attention, 20k steps"),
             ("swa_sink", 20000): (ORANGE, "-", "s", 2.2, "SWA + sink (w=16, L=2), 20k steps"),
             ("full", 8000): (BLUE, "--", "o", 1.1, "full attention, 8k steps"),
             ("swa_sink", 8000): (ORANGE, "--", "s", 1.1, "SWA + sink, 8k steps")}
    for (arm, steps), pts in {(a, s): sorted([(p, v) for (aa, ss, p, v) in rows if aa == a and ss == s])
                              for (a, s) in style}.items():
        if not pts:
            continue
        c, ls, mk, lw, lb = style[(arm, steps)]
        ax.plot([p for p, _ in pts], [v for _, v in pts], color=c, ls=ls, marker=mk,
                lw=lw, ms=6 if lw > 1.5 else 4, label=lb, zorder=4)
    ax.axhline(1 / 64, color=MUTED, ls=":", lw=1.0, zorder=2)
    ax.text(16.4, 0.035, "random baseline 1/64", fontsize=7.5, color=MUTED, va="bottom")
    ax.axvline(32, color=TEAL, ls="--", lw=1.2, zorder=2)
    ax.text(32.4, 0.47, "composed reach\nvalue pos ≥ 62−2(w−1)=32", fontsize=7.5, color=TEAL, va="center")
    ax.axvline(46, color=YELLOW, ls="--", lw=1.2, zorder=2)
    ax.text(45.6, 0.13, "single-layer cliff p=46\n(preregistered)", fontsize=7.5, color="#b07c00",
            va="center", ha="right")
    ax.set_xlabel("needle position p  (T=64, query at position 62)", fontsize=9, color=MUTED)
    ax.set_ylabel("recall accuracy (argmax)", fontsize=9, color=MUTED)
    ax.set_xlim(14, 48)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xticks([16, 22, 28, 34, 40, 46])
    ax.tick_params(colors="#898781", labelsize=8)
    ax.grid(color=GRIDC, lw=0.6, zorder=0)
    ax.legend(fontsize=7.8, frameon=False, loc="center left")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-2-3-swa-sink.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 2.3] {path}（数据源 needle_scan.csv）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch2 制图（图 2.1-2.3）")
    ap.add_argument("--which", default="all", choices=["all", "dataflow", "bands", "needle"])
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    if args.which in ("all", "dataflow"):
        fig_dataflow()
    if args.which in ("all", "bands"):
        fig_bands()
    if args.which in ("all", "needle"):
        fig_needle()
