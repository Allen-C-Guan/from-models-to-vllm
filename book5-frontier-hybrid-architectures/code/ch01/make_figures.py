# make_figures.py —— Book5 ch1 两张图：图 1.1 三族增长曲线（log-log）/ 图 1.2 三条路线「改动多少」阶梯图
# 用途：产出 figures/fig-1-{1,2}-<ch01 专属 slug>.png（300dpi 印刷规格，
#       图内英文、图注中文在正文）。两图均纯数学自算（图 1.1 数据与 ch01/context_account.py 同源：
#       打分账 n^2、Gemma3-27B 全满 KV 斜率 253,952/token、Qwen3-Next 36 层 GDN 固定状态 18,874,368）。
# 所属章节：Book5 第 1 章（1.2 图 1.1 / 1.3 图 1.2）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch01/make_figures.py" [--only 1,2]
# 规格：写作规范 §4——蓝 #2a78d6 / 橙 #eb6834 / 青 #1baf7a / 黄 #eda100；线宽 2；网格 hairline #e1e0d9
#       置底层；刻度 #898781；标注 #0b0b0b / #52514e；禁双 y 轴；tight_layout（不用 bbox_inches='tight'）。
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIG_DIR = os.path.join(REPO, "book5-frontier-hybrid-architectures", "figures")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MID = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9.5,
    "axes.edgecolor": TICK, "axes.labelcolor": DARK, "xtick.color": TICK,
    "ytick.color": TICK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True,
})


def fig_1_1():
    """三族增长曲线：平方（打分账）/ 线性（全满 KV）/ 常数（线性层状态）——log-log 下斜率 2/1/0。"""
    fig, ax = plt.subplots(figsize=(6.4, 4))
    n = np.logspace(3, np.log10(1.2e6), 200)
    score = n ** 2                                             # 每层每头打分元素（整矩阵口径）
    kv = 253_952 * n                                           # Gemma3-27B 全满口径：2·62·16·128=253,952 元素/token
    state = np.full_like(n, 18_874_368)                        # Qwen3-Next 36 层 GDN 递归状态 36×524,288

    ax.plot(n, score, color=BLUE, lw=2, label="score matrix, per layer-head:  $n^2$")
    ax.plot(n, kv, color=ORANGE, lw=2, label="full-machine KV (Gemma3-27B, all-full):  $253{,}952\\,n$")
    ax.plot(n, state, color=TEAL, lw=2, label="fixed state (Qwen3-Next, 36 GDN layers):  18,874,368")

    for x0, lab in ((8.19e3, "8K era"), (1e6, "1M era")):
        ax.axvline(x0, color=GRID, lw=1.0, ls="--", zorder=0)
        ax.text(x0, 2.2e6, lab, rotation=90, fontsize=8, color=MID, va="bottom", ha="right")
    ax.annotate("slope 2", xy=(1.1e4, 1.1e4 ** 2 / 1.9), fontsize=9, color=BLUE, rotation=38)
    ax.annotate("slope 1", xy=(1.6e5, 253_952 * 1.6e5 * 1.9), fontsize=9, color=ORANGE, rotation=14)
    ax.annotate("slope 0", xy=(1.5e5, 18_874_368 * 1.45), fontsize=9, color=TEAL)
    ax.annotate("", xy=(1e6, 1.5e12), xytext=(8.19e3, 1.5e12),
                arrowprops=dict(arrowstyle="<->", color=MID, lw=1.2))
    ax.text(9.2e4, 5.5e11, "8K→1M:  ×14,901 on the score account", fontsize=9, color=DARK, ha="center")

    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(1e3, 1.5e6), ax.set_ylim(1e6, 4e13)
    ax.set_xlabel("context length  n  (tokens)"), ax.set_ylabel("elements (log scale)")
    ax.legend(loc="upper left", fontsize=8.2, framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-1-1-three-growth-classes.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[产物] {out}")


def _step(ax, x, y, w, h, color, title, sub, models, ch, kvnote):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                         fc="white", ec=color, lw=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.075, title, ha="center", va="top", fontsize=10,
            color=color, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.26, sub, ha="center", va="top", fontsize=8.6, color=DARK)
    ax.text(x + w / 2, y + h - 0.47, models, ha="center", va="top", fontsize=8.2, color=MID)
    ax.text(x + w / 2, y + 0.16, ch, ha="center", va="bottom", fontsize=9, color=color)
    ax.text(x + w / 2, y + 0.045, kvnote, ha="center", va="bottom", fontsize=7.8, color=MID)


def fig_1_2():
    """三条路线「改动多少」阶梯：改视野 → 改参与集合 → 换算子（动到注意力的部件越来越多）。"""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")

    base = FancyBboxPatch((0.03, 0.02), 0.94, 0.13, boxstyle="round,pad=0.008,rounding_size=0.015",
                          fc="#f4f3ef", ec=MID, lw=1.6)
    ax.add_patch(base)
    ax.text(0.5, 0.085, "full attention:  every query scores every key  —  $n^2$ per layer-head,"
            "  KV grows with $n$", ha="center", va="center", fontsize=9.5, color=DARK)

    steps = [
        (0.035, 0.22, BLUE, "Route 1 · restrict the view",
         "sliding window $W$ (static banded mask)\noperator untouched, horizon shrinks",
         "Gemma3 5:1 W=1024 · gpt-oss 1:1 W=128", "ch2",
         "window layers: KV capped at min(n, W)"),
        (0.355, 0.45, ORANGE, "Route 2 · learn the participant set",
         "top-k block selection (learned scorer)\nexact dot-product on selected blocks",
         "NSA → DSA · GLM-5: 78× indexed_attention", "ch3-4",
         "scoring bounded by k, KV still per-token"),
        (0.675, 0.68, TEAL, "Route 3 · replace the operator",
         "kernel $\\varphi(q)^{\\top}\\varphi(k)$ + fixed state $S$\nsoftmax dot-product retired",
         "GDN · KDA · Qwen3-Next / K3 (3:1 hybrid)", "ch5",
         "unbounded KV → constant-size state"),
    ]
    for x, y, color, title, sub, models, ch, kvnote in steps:
        _step(ax, x, y, 0.29, 0.29, color, title, sub, models, ch, kvnote)

    for (x0, y0), (x1, y1) in (((0.335, 0.36), (0.365, 0.585)), ((0.655, 0.59), (0.685, 0.815))):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                                     color=MID, lw=1.6))
    ax.text(0.5, 0.985, "more surgery on the attention operator  →", ha="center", va="top",
            fontsize=9.5, color=MID, style="italic")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-1-2-escape-routes-ladder.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[产物] {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="1,2", help="只绘制指定图（逗号分隔编号）")
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    only = {s.strip() for s in args.only.split(",")}
    if "1" in only:
        fig_1_1()
    if "2" in only:
        fig_1_2()
