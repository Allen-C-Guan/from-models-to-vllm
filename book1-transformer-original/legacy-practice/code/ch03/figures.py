# 用途：第 3 章插图——fig-3-1 多头机制框图；fig-3-2 多头分工热图（构造演示）+ ConvS2S 层间分工示意
# 所属章节：Book1《Transformer 原典》第 3 章（绘图脚本；规格见 _写作规范.md 第 4 节）
# 运行方式：source env.sh && python code/ch03/figures.py
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrow, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mha import run_demo  # noqa: E402

BLUE, ORANGE, INK, SUB, GRID, LIGHT = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e1e0d9", "#cde2fb"
CMAP = LinearSegmentedColormap.from_list("book_blue", ["#cde2fb", "#0d366b"])  # 蓝色单色渐变，禁 jet
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))), "drafts", "Book1-Transformer原典", "figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": SUB,
                     "axes.labelcolor": SUB, "xtick.color": "#898781", "ytick.color": "#898781"})


def box(ax, x, y, w, h, text, fill=LIGHT, edge=BLUE, fs=9, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=fill, edgecolor=edge, linewidth=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrow(x0, y0, x1 - x0, y1 - y0, width=0.012, head_width=0.07,
                            head_length=0.09, length_includes_head=True, color=SUB))


def fig_3_1():
    """多头机制图：自绘重制 Vaswani et al. 2017 Fig.2（右）。

    形状标注（右侧注记与框内小字，原图无）按整批口径：B=批大小、h=8、d_k=d_v=64；
    每个注意力框内标注其内部的打分/权重矩阵 (B,h,n,n)——与正文表 3.1 第 4-5 行对应。
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    # 顶部三输入 + 各自的 Linear 投影（投影符号写在框内，避免与箭头混淆）
    for cx, name, sup in ((2.0, "V", "$W_i^{V}$"), (5.0, "K", "$W_i^{K}$"), (8.0, "Q", "$W_i^{Q}$")):
        box(ax, cx - 0.85, 8.9, 1.7, 0.75, name, fs=10)
        arrow(ax, cx, 8.9, cx, 8.0)
        box(ax, cx - 0.85, 7.25, 1.7, 0.75, f"Linear {sup}", fill="white", fs=8.5)
    # 中部：h 份并行的缩放点积注意力（画三份 + 省略号），三路投影扇入堆顶
    heads_y = [5.7, 5.0, 4.3]
    for k, yy in enumerate(heads_y):
        box(ax, 2.2, yy, 4.6, 0.62, "Scaled Dot-Product Attention\n(scores (B, h, n, n))"
            if k == 1 else "", fill=LIGHT, fs=8)
    for cx, tx in ((2.0, 3.4), (5.0, 4.7), (8.0, 6.0)):
        arrow(ax, cx, 7.25, tx, heads_y[0] + 0.62)
    ax.text(7.6, 5.0, "$\\vdots$\nh copies\n(h = 8, base;\n$d_k$=$d_v$=64)", ha="left", va="center",
            fontsize=8, color=SUB)
    # 下部：Concat -> Linear(W^O) -> 输出
    arrow(ax, 5.0, 4.3, 5.0, 3.5)
    box(ax, 3.6, 2.75, 2.8, 0.75, "Concat", fill="white", fs=9.5)
    arrow(ax, 5.0, 2.75, 5.0, 2.0)
    box(ax, 3.6, 1.25, 2.8, 0.75, "Linear (W$^O$)", fill="white", fs=9.5)
    arrow(ax, 5.0, 1.25, 5.0, 0.55)
    ax.text(5.0, 0.38, "output  (B, n, 512)", ha="center", fontsize=8.5, color=SUB)
    # 右侧维度注记（整批口径）
    for y, t in ((9.28, "(B, n, 512) each"), (7.6, "W: 512 x 64 per head"),
                 (5.0, "(B, n, 64) per head"), (2.99, "(B, n, 512)"), (1.49, "(B, n, 512)")):
        ax.text(9.9, y, t, ha="right", va="center", fontsize=7.5, color=SUB)
    ax.set_title("Multi-Head Attention: h parallel subspaces, then Concat + W$^O$",
                 fontsize=10, color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-3-1-mha-mechanism.png"), facecolor="white")
    plt.close(fig)


def fig_3_2():
    """上排：本书构造演示的四个头；下排：ConvS2S 层间分工示意（凭调研文字记录重制）。"""
    torch.manual_seed(3)
    A = run_demo().numpy()  # (4, 12, 12)：复用 mha.py 的构造演示
    tokens = ["<s>", "the", "old", "hse", "that", "stood", "here", "was", "built", "in", "nine", "."]
    fig = plt.figure(figsize=(8, 5.4), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.55, top=0.90, bottom=0.08,
                          left=0.07, right=0.93)
    top = gs[0].subgridspec(1, 4, wspace=0.30)
    titles = ["head 1: verb anchor\n(content subspace)", "head 2: previous token\n(position subspace)",
              "head 3: sentence start\n(position subspace)", "head 4: same word class\n(content subspace)"]
    for i in range(4):
        ax = fig.add_subplot(top[i])
        ax.imshow(A[i], cmap=CMAP, vmin=0, vmax=1, origin="upper")
        ax.set_title(titles[i], fontsize=7.5, color=INK)
        ax.set_xticks(range(12))
        ax.set_xticklabels(tokens, rotation=90, fontsize=5.5)
        ax.set_yticks(range(12))
        ax.set_yticklabels(tokens if i == 0 else [], fontsize=5.5)
        ax.set_xlabel("key position", fontsize=7)
        if i == 0:
            ax.set_ylabel("query position", fontsize=7)
        ax.tick_params(length=1.5)
    fig.text(0.5, 0.955, "(a) Constructed demo (not trained): four heads of one MHA layer on a "
             "12-token synthetic input", ha="center", fontsize=8.5, color=INK)
    # 下排：ConvS2S 附录 C 层间分工示意（8 个解码层；层 1/3/6 近线性，层 5/7 盯住动词 built）
    rng = np.random.default_rng(3)
    n_t, n_s, verb = 8, 10, 7
    bottom = gs[1].subgridspec(1, 8, wspace=0.42)
    for layer in range(1, 9):
        ax = fig.add_subplot(bottom[layer - 1])
        logit = rng.normal(0, 0.35, (n_t, n_s))
        jj = np.arange(n_s)[None, :]
        diag = jj * (n_s / n_t) * np.ones((n_t, 1))  # 目标行 i 的线性对齐位置
        rows = np.arange(n_t)[:, None]
        if layer in (1, 3, 6):
            logit += 2.2 * np.exp(-((jj - diag) ** 2) / 1.8)          # 近线性对齐带
        elif layer in (5, 7):
            logit += 2.8 * np.exp(-((jj - verb) ** 2) / 0.9)          # 全体目标行盯住动词列
        else:
            logit += 1.1 * np.exp(-((jj - diag) ** 2) / 6.0)          # 宽对角 + 漫散
        ax.imshow(np.exp(logit) / np.exp(logit).sum(-1, keepdims=True), cmap=CMAP,
                  vmin=0, vmax=1, origin="upper")
        ax.set_title(f"layer {layer}", fontsize=7, color=INK)
        ax.plot(verb, -0.9, marker="v", markersize=3, color=ORANGE, clip_on=False)  # 动词列标记
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(length=0)
    fig.text(0.5, 0.045, "(b) Schematic redraw of ConvS2S (Gehring et al. 2017, App. C Fig. 3): "
             "attention of each decoder layer; layers 1/3/6 near-linear, layers 5/7 fixate on the "
             "reordered verb 'built' (orange marker)", ha="center", fontsize=8, color=SUB)
    fig.text(0.985, 0.5, "attention weight", rotation=270, va="center", fontsize=7, color=SUB)
    cax = fig.add_axes([0.95, 0.25, 0.012, 0.55])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1)), cax=cax)
    cb.ax.tick_params(labelsize=6, color=SUB)
    cb.outline.set_edgecolor(GRID)
    fig.savefig(os.path.join(FIG_DIR, "fig-3-2-head-diversity.png"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_3_1()
    fig_3_2()
    print("figures saved to", FIG_DIR)
