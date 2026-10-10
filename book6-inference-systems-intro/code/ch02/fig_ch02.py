# fig_ch02.py —— Book6 ch2 制图：图 2.1 三层对拍示意图 / 图 2.2 下限-基线-引擎三档条形（slug=arithmetic）
# 用途：图 2.1 = 2.1 节方法论总纲的机制图（手算→mini 引擎→真实引擎三层各答各的问题、层间差距=待解释对象）；
#       图 2.2 = 2.4 节「没有引擎的价格」的视觉锚（0.348 ms 带宽下限 / 25.15 ms 裸跑基线 / 8.9 ms vllm 运行栈
#         ——第 7 章预告档；log 轴+倍率标注 72×/26×）。
# 所属章节：Book6 第 2 章（图 2.1/2.2）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/Book6-推理系统导论/ch02/fig_ch02.py"
# 产物：drafts/Book6-推理系统导论/figures/fig-2-1-three-lens.png / fig-2-2-price-bars.png
# 图规：300dpi 白底；系列色 蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；线宽 2；图内英文、图注中文。
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "drafts", "Book6-推理系统导论", "figures")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e1e0d9"


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=7.2, lw=1.3, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc, zorder=4)


# ---------------- 图 2.1：三层对拍 ----------------
def fig_three_lens():
    fig, ax = plt.subplots(figsize=(11.6, 4.4), dpi=300)
    ax.set_xlim(0, 15); ax.set_ylim(0, 6.0)
    ax.axis("off")
    ax.text(7.5, 5.75, "Three-layer cross-checking: hand calc → mini-engine → real engine (gaps between layers are the object to explain)",
            ha="center", fontsize=9.4, color=INK, weight="bold")
    box(ax, 0.4, 1.6, 4.0, 2.6, "LAYER 1  hand calc\n(from config alone)\n\n2N FLOPs · weight bytes\nroofline floors\n\nanswers: what is the\nPHYSICAL limit?", ec=BLUE, fc="#dcebfc", fs=7.4)
    box(ax, 5.4, 1.6, 4.0, 2.6, "LAYER 2  mini-engine\n(ch4-7, our four versions)\n\ncontinuous batching ·\npaged KV · prefix cache\n\nanswers: how far does\nENGINEERING get?", ec=ORANGE, fc="#fde3d6", fs=7.4)
    box(ax, 10.4, 1.6, 4.0, 2.6, "LAYER 3  real engine\n(vllm runtime, black box)\n\nQwen3-0.6B — trend-level\ncomparison (ch7 caveat)\n\nanswers: what does\nINDUSTRY achieve?", ec=TEAL, fc="#d9f2e7", fs=7.2)
    # 层间差距标注
    for x0, label in ((4.6, "gap = engineering\noverhead to explain"), (9.6, "gap = remaining\nheadroom to close")):
        ax.annotate("", xy=(x0 + 0.75, 2.9), xytext=(x0, 2.9), arrowprops=dict(arrowstyle="<|-|>", color=MUTED, lw=1.4))
        ax.text(x0 + 0.38, 3.55, label, fontsize=6.6, color=MUTED, ha="center")
    ax.text(7.5, 0.7, "example thread: decode floor 0.348 ms  →  bare loop 25.15 ms (72×, 207M, launch-bound → ch12)  →  vllm 8.9 ms (Qwen3-0.6B, trend-level, ch7)",
            ha="center", fontsize=7.4, color=INK)
    ax.plot([0.4, 14.4], [1.25, 1.25], color=GRIDC, lw=0.8, zorder=1)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-2-1-three-lens.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 2.1] {path}")


# ---------------- 图 2.2：下限-基线-引擎三档 ----------------
def fig_price_bars():
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=300)
    items = [("bandwidth floor\n(hand calc, 207M bf16)", 0.348, MUTED),
             ("bare loop\n(ch1 baseline, 207M bf16)", 25.15, ORANGE),
             ("vllm runtime\n(Qwen3-0.6B, trend-level)", 8.9, TEAL)]
    xs = range(len(items))
    ax.bar(xs, [v for _, v, _ in items], color=[c for _, _, c in items], width=0.52, zorder=3)
    for i, (_, v, _) in enumerate(items):
        ax.text(i, v * 1.12, f"{v} ms", ha="center", fontsize=9, color=INK, weight="bold")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 90)
    ax.set_xticks(list(xs)); ax.set_xticklabels([k for k, _, _ in items], fontsize=8.2)
    ax.set_ylabel("ms per decode token (log)", fontsize=9, color=MUTED)
    # 倍率标注
    ax.annotate("", xy=(1, 25.15), xytext=(0, 0.348),
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.1, ls="--", connectionstyle="arc3,rad=-0.2"))
    ax.text(0.75, 2.6, "72× (this ch:\nthe price of no engine,\nsame 207M model)", fontsize=7.2, color=BLUE, ha="center")
    ax.text(1.9, 16.5, "~2.8× recovered (ch4-7+engine,\ncross-model trend-level);\n~26× still to floor (ch12)", fontsize=7.0, color=MUTED, ha="center")
    ax.tick_params(colors="#898781", labelsize=8)
    ax.grid(axis="y", color=GRIDC, lw=0.5, zorder=0)
    ax.set_title("Decode latency, Atlas 910B3: floor vs no-engine (207M) vs real engine (Qwen3-0.6B, trend-level)", fontsize=8.8, color=INK)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-2-2-price-bars.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 2.2] {path}（数据源 arithmetic_base.json + ch7 vllm 读数预告）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch2 制图（图 2.1/2.2）")
    ap.add_argument("--which", default="all", choices=["all", "three_lens", "price_bars"])
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    if args.which in ("all", "three_lens"):
        fig_three_lens()
    if args.which in ("all", "price_bars"):
        fig_price_bars()
