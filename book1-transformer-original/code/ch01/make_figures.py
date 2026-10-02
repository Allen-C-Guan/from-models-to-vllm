# -*- coding: utf-8 -*-
"""Book1 第 1 章插图生成：图 1.1 seq2seq 完整架构图 / 图 1.2 长句衰减双证 / 图 1.3 对齐热图。

用途：按 _写作规范.md 第 4 节印刷规格，自绘重制第 1 章三张示意图（图 1.4 由
      bottleneck_demo.py 随实验自产）。图 1.2 / 1.3 为论文图的「趋势/形态示意重制」，
      非逐值数据提取——图注已声明，正文引用须带证据等级标注。
所属章节：drafts/Book1-Transformer原典/ch01-前史动机.md
运行方式：source env.sh && python code/ch01/make_figures.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

BLUE, ORANGE, TEAL = "#2a78d6", "#eb6834", "#1baf7a"
INK, SUB, AXIS = "#0b0b0b", "#52514e", "#898781"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": AXIS,
    "axes.labelcolor": SUB, "xtick.color": AXIS, "ytick.color": AXIS,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "font.family": "DejaVu Sans", "font.size": 9,
})

FIG_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "figures"))


def box(ax, x0, y0, w, h, text, fc, ec, fs=8.5, lw=1.2, tc=INK):
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))
    ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=4)


def arrow(ax, x0, y0, x1, y1, lw=1.4, color=SUB):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                                 linewidth=lw, color=color, zorder=2))


def fig_1_1():
    """图 1.1：seq2seq 完整架构图（两塔＋唯一桥 c＋回喂回路；自绘，中文标签）。

    完整要素：词嵌入、循环单元逐步更新的隐状态链（串行依赖）、c 的唯一桥地位
    （橙色高亮，桥宽口径「走例 d=1000；Sutskever 实机 8000，见考据框」）、解码器
    三步走例 X/Y/<eos>（softmax 概率框）、起始符 <bos> 与上一步输出回喂回路
    （青色）、训练/推理两种模式与两堵墙（底部三栏标注）。
    形状标注（走例口径 d=1000）：首个「嵌入」框内注 d=1000；两塔底部更新式旁注
    h_j/x_j/s_i ∈ R^1000；c 框内注 1000/8000 两口径；softmax 出 R^{|V|}。
    """
    import matplotlib as mpl
    _cjk = {"font.family": "sans-serif",
            "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "DejaVu Sans"]}
    with mpl.rc_context(_cjk):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_xlim(0, 16)
        ax.set_ylim(1.3, 7.85)
        ax.axis("off")

        # —— 两塔背景 ——
        ax.add_patch(Rectangle((0.35, 3.0), 5.85, 4.6, facecolor="#eaf2fc", edgecolor=BLUE, linewidth=1.4, zorder=1))
        ax.text(3.27, 7.32, "LSTM 编码器（4 层 × 1000 单元）", ha="center", fontsize=9, color=INK)
        ax.add_patch(Rectangle((8.3, 3.0), 5.75, 4.6, facecolor="#eaf2fc", edgecolor=BLUE, linewidth=1.4, zorder=1))
        ax.text(11.17, 7.32, "LSTM 解码器", ha="center", fontsize=9, color=INK)
        ax.text(3.27, 3.30, "$h_j = \\mathrm{LSTM}(h_{j-1},\\,x_j)$\n$h_j,\\,x_j\\in\\mathbb{R}^{1000}$（走例）",
                ha="center", va="center", fontsize=8, color=SUB)
        ax.text(11.17, 3.30, "$s_i = \\mathrm{LSTM}(s_{i-1},\\,y_{i-1},\\,c)$\n$s_i\\in\\mathbb{R}^{1000}$（走例；softmax 出 $\\mathbb{R}^{|V|}$）",
                ha="center", va="center", fontsize=8, color=SUB)

        # —— 编码塔：源词 → 嵌入 → 循环单元链 ——
        enc_x = [1.4, 2.8, 4.2, 5.6]
        for j, (x, tok) in enumerate(zip(enc_x, ("A", "B", "C", "<eos>"))):
            box(ax, x - 0.45, 6.3, 0.9, 0.6, tok, "#f3f7fd", BLUE, fs=8)
            arrow(ax, x, 6.3, x, 5.92)
            box(ax, x - 0.45, 5.35, 0.9, 0.57, "嵌入\n$d{=}1000$" if j == 0 else "嵌入",
                "white", BLUE, fs=6 if j == 0 else 7)
            arrow(ax, x, 5.35, x, 5.02)
            box(ax, x - 0.5, 4.0, 1.0, 1.0, "LSTM", "white", BLUE, fs=8)
            if j < 3:  # 隐状态链：h_j 是第 j 格的输出、第 j+1 格的输入
                arrow(ax, x + 0.5, 4.5, x + 1.4 - 0.5, 4.5, lw=1.8, color=BLUE)
                ax.text(x + 0.7, 4.78, f"$h_{j + 1}$", ha="center", fontsize=7, color=SUB)
        ax.add_patch(plt.Circle((0.62, 4.5), 0.09, facecolor="white", edgecolor=BLUE, linewidth=1.2, zorder=3))
        ax.text(0.62, 4.82, "$h_0$", ha="center", fontsize=7, color=SUB)
        arrow(ax, 0.71, 4.5, 0.9, 4.5, lw=1.8, color=BLUE)
        ax.text(0.62, 6.6, "源句", ha="center", fontsize=7.5, color=SUB, rotation=90, va="center")

        # —— 唯一的桥：c（橙色高亮） ——
        arrow(ax, 6.1, 4.5, 6.32, 4.5, lw=2.2, color=ORANGE)
        ax.text(6.21, 4.78, "$h_{T_x}$", ha="center", fontsize=7, color=SUB)
        box(ax, 6.32, 3.85, 1.93, 1.35, "", "#fdeee7", ORANGE, fs=8, lw=2.2)
        ax.text(7.28, 4.92, "$c = h_{T_x}$", ha="center", fontsize=8.5, color=INK)
        ax.text(7.28, 4.66, "固定 $d$ 维向量", ha="center", fontsize=6.8, color=INK)
        ax.text(7.28, 4.44, "走例 $d{=}1000$；", ha="center", fontsize=5.6, color=SUB)
        ax.text(7.28, 4.26, "Sutskever 实机 8000", ha="center", fontsize=5.6, color=SUB)
        ax.text(7.28, 4.08, "（见 1.1 考据框）", ha="center", fontsize=5.6, color=SUB)
        ax.text(7.28, 5.48, "唯一的桥", ha="center", fontsize=8, color=INK, fontweight="bold")
        arrow(ax, 8.25, 4.5, 8.78, 4.5, lw=2.2, color=ORANGE)
        ax.text(8.52, 4.78, "$s_0 = c$", ha="center", fontsize=7.5, color=SUB)

        # —— 解码塔：三步走例 X / Y / <eos>，softmax 概率框，<bos> 起步与回喂 ——
        dec_x = [10.3, 12.0, 13.5]
        for i, x in enumerate(dec_x):
            box(ax, x - 0.5, 4.0, 1.0, 1.0, "LSTM", "white", BLUE, fs=8)
            if i < 2:
                arrow(ax, x + 0.5, 4.5, x + 1.7 - 0.5, 4.5, lw=1.8, color=BLUE)
                ax.text(x + 0.85, 4.78, f"$s_{i + 1}$", ha="center", fontsize=7, color=SUB)
            # softmax 概率框：LSTM 状态 → 概率分布 → 词
            box(ax, x - 0.34, 5.28, 0.68, 0.46, "", "white", SUB, fs=5, lw=0.8)
            for b, hgt in enumerate((0.16, 0.30, 0.44, 0.24)):  # 框内小概率条
                ax.add_patch(Rectangle((x - 0.24 + b * 0.13, 5.33), 0.07, hgt * 0.9,
                                       facecolor="#e1e0d9", edgecolor="none", zorder=4))
            arrow(ax, x, 5.0, x, 5.28, lw=1.0)
            arrow(ax, x, 5.74, x, 6.3, lw=1.0)
            tok = ("X", "Y", "<eos>")[i]
            fc, ec = ("#e6f7f1", TEAL) if tok == "<eos>" else ("#f3f7fd", BLUE)
            box(ax, x - 0.45, 6.3, 0.9, 0.6, tok, fc, ec, fs=8)
        ax.text(9.62, 5.51, "softmax", ha="center", fontsize=6, color=SUB, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
        # <bos>：第 1 步的输入——与 <eos> 对称的固定起始符（青色）
        box(ax, 8.32, 6.3, 0.9, 0.6, "<bos>", "#e6f7f1", TEAL, fs=8)
        arrow(ax, 9.1, 6.28, 9.92, 5.03, lw=1.4, color=TEAL)
        # 回喂：上一步输出（嵌入后）成为下一步输入
        arrow(ax, 10.72, 6.28, 11.72, 5.03, lw=1.4, color=TEAL)
        arrow(ax, 12.42, 6.28, 13.12, 5.03, lw=1.4, color=TEAL)
        ax.text(11.55, 5.92, "上一步输出 $y_{i-1}$（嵌入后）回喂", ha="center", fontsize=6, color=SUB,
                zorder=5, bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
        ax.text(15.35, 6.6, "译文", ha="center", fontsize=7.5, color=SUB, rotation=90, va="center")

        # —— 底部三栏机制标注 ——
        ax.annotate("", xy=(6.1, 2.42), xytext=(0.9, 2.42),
                    arrowprops=dict(arrowstyle="<|-|>", color=SUB, linewidth=1.0))
        ax.text(3.5, 1.95, "串行墙：第 $j$ 步要等 $h_{j-1}$——隐状态链\n只能逐步向前，$O(n)$ 步",
                ha="center", fontsize=7.5, color=SUB)
        ax.annotate("", xy=(7.28, 3.82), xytext=(7.28, 2.5),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, linewidth=1.0))
        ax.text(7.28, 1.95, "容量墙：整句必须\n装进一个固定向量",
                ha="center", fontsize=7.5, color=INK, fontweight="bold")
        ax.text(11.7, 1.95, "训练＝教师强制（喂标准答案的上一个词 $y_{i-1}$）\n推理＝自回归（喂自己的输出）；抽到 <eos> 为止",
                ha="center", fontsize=7.5, color=SUB)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig-1-1-seq2seq-framework.png"), dpi=300)


def fig_1_2():
    """图 1.2：长句衰减双证（左：Sutskever Fig.3 左重制趋势；右：Bahdanau Fig.2 重制趋势）。"""
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 4))
    x1 = np.linspace(0, 60, 240)
    y1 = 33.5 + 2.2 * (1 - np.exp(-x1 / 12)) - 0.028 * np.maximum(0, x1 - 35)
    a.plot(x1, y1, color=BLUE, linewidth=2)
    a.axvline(35, color=SUB, linestyle="--", linewidth=1)
    a.annotate("35 words:\n\"no degradation\nbelow\" (paper)", xy=(35, 34.2), xytext=(40, 32.6),
               fontsize=7, color=SUB)
    a.set_title("(a) deep LSTM ensemble, WMT'14 En-Fr\nredrawn from Sutskever et al. 2014, Fig.3 (left)",
                fontsize=8, color=INK)
    a.set_xlabel("source sentence length (words)", fontsize=8)
    a.set_ylabel("BLEU", fontsize=8)
    a.set_ylim(31.5, 36.8)
    a.text(2, 32.0, "original x-axis: test sentences sorted by length", fontsize=6.5, color=AXIS)

    x2 = np.linspace(0, 80, 240)
    enc = 17.5 * np.exp(-((x2 - 10) / 24) ** 2) + 1.2
    s50 = 26.0 + 1.2 * (1 - np.exp(-x2 / 9))
    s30 = 20.0 + 0.9 * (1 - np.exp(-x2 / 9))
    b.plot(x2, enc, color=ORANGE, linewidth=2, label="RNNencdec-50 (no attention)")
    b.plot(x2, s50, color=BLUE, linewidth=2, label="RNNsearch-50 (attention)")
    b.plot(x2, s30, color=TEAL, linewidth=2, label="RNNsearch-30 (attention)")
    b.axvline(50, color=SUB, linestyle="--", linewidth=1)
    b.annotate("\"no performance deterioration beyond\n50 words\" (paper)", xy=(50, 27.1), xytext=(13, 28.4),
               fontsize=7, color=SUB, arrowprops=dict(arrowstyle="-|>", color=SUB, linewidth=0.9))
    b.set_title("(b) RNNencdec vs RNNsearch, WMT'14 En-Fr\nredrawn from Bahdanau et al. 2015, Fig.2",
                fontsize=8, color=INK)
    b.set_xlabel("source sentence length (words)", fontsize=8)
    b.set_ylabel("BLEU", fontsize=8)
    b.set_ylim(0, 30)
    b.legend(frameon=True, facecolor="white", framealpha=0.9, edgecolor="#e1e0d9",
            fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-1-2-length-degradation-two-evidences.png"), dpi=300)


def fig_1_3():
    """图 1.3：Bahdanau 软对齐热图（形态示意重制，改绘自 1409.0473 v7 Fig.3(a)；数值为示意）。"""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("book_blue", ["#cde2fb", "#0d366b"])
    alpha = np.array([[0.18, 0.22, 0.60],      # zone ←– Area 为主
                      [0.08, 0.78, 0.14],      # économique ←– Economic 为主
                      [0.72, 0.18, 0.10]])     # européenne ←– European 为主（与首行交叉=非单调）
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    im = ax.imshow(alpha, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    src = ["European", "Economic", "Area"]
    tgt = ["zone", "économique", "européenne"]
    ax.set_xticks(range(3), src, fontsize=9)
    ax.set_yticks(range(3), tgt, fontsize=9)
    ax.set_xlabel("source (English)", fontsize=9)
    ax.set_ylabel("target (French)", fontsize=9)
    for i in range(3):
        for j in range(3):
            v = alpha[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.45 else INK)
    ax.set_ylim(3.15, -0.5)  # 矩阵上方留一行空白给标注
    ax.annotate("crossing = non-monotonic (word order differs)", xy=(2.02, 0.0), xytext=(-0.45, 2.72),
                fontsize=7.5, color=SUB,
                arrowprops=dict(arrowstyle="-|>", color=SUB, linewidth=1.0,
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_title("Soft alignment: European Economic Area -> zone économique européenne\n"
                 "(redrawn schematic, after Bahdanau et al. 2015, Fig.3(a))", fontsize=8, color=INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("alignment weight", fontsize=8, color=SUB)
    cb.ax.tick_params(labelsize=7, colors=AXIS)
    cb.outline.set_edgecolor(AXIS)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-1-3-bahdanau-alignment-heatmap.png"), dpi=300)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_1_1()
    fig_1_2()
    fig_1_3()
    print(f"三张图已保存至 {FIG_DIR}")
