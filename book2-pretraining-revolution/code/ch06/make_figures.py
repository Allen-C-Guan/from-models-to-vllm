# -*- coding: utf-8 -*-
# 用途：Book2 第 6 章插图生成——图 6.1 同一架机器的四个配方旋钮（BERT vs RoBERTa 量级对照）
# 所属章节：ch06-RoBERTa.md（原理章无教学代码，本脚本仅出图）
# 运行方式：source env.sh && python "code/ch06/make_figures.py"
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 全书统一分类色（固定序）：蓝 / 橙 —— BERT=蓝(slot1)，RoBERTa=橙(slot2)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, SUB, AXIS, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 9,
})

FIG_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)


def panel(ax, title, unit, bert_v, rob_v, fmt, ratio):
    """单格：两根细条（BERT vs RoBERTa），顶端直标数值，格内标倍率。

    无返回值；每格自成一张小图（各自的量纲与坐标，互不共用 y 轴）。
    """
    xs = [0.0, 1.0]
    vals = [bert_v, rob_v]
    ax.bar(xs, vals, width=0.52, color=[BLUE, ORANGE], zorder=3)
    for x, v in zip(xs, vals):
        ax.text(x, v + max(vals) * 0.035, fmt.format(v), ha="center",
                va="bottom", fontsize=9, color=INK, zorder=4)
    ax.text(0.5, max(vals) * 0.55, ratio, ha="center", va="center",
            fontsize=11, color=SUB, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(["BERT", "RoBERTa"], fontsize=9, color=AXIS)
    ax.set_title(title, fontsize=9.5, color=INK, pad=6)
    ax.set_ylabel(unit, fontsize=8.5, color=AXIS)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.tick_params(colors=AXIS, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def main():
    fig, axes = plt.subplots(1, 4, figsize=(8, 3.4), dpi=300)

    # 数据源：1907.11692 v1 §3.2 / Table 3 / §4.4；BERT 侧 A.2（1810.04805 v2）
    # 与 §3（词表 30,000）。token 通过量为本书换算（批量×512×步数）。
    panel(axes[0], "Pretraining text", "GB", 16, 160, "{:.0f}", "x10")
    panel(axes[1], "Batch size", "seqs / step", 256, 8192, "{:,.0f}", "x32")
    panel(axes[2], "Token passes", "billion tokens", 131, 2100, "{:,.0f}", "x16")
    panel(axes[3], "BPE vocab", "thousand units", 30, 50, "{:.0f}", "x1.7")

    fig.suptitle("Same architecture, four recipe knobs (2018 vs 2019)",
                 fontsize=10.5, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(FIG_DIR, "fig-6-1-recipe-knobs.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
