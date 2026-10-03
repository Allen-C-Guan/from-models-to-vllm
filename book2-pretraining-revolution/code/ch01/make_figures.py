# -*- coding: utf-8 -*-
# 用途：Book2 第 1 章插图生成——图 1.1 五环论证链 / 图 1.2 ELMo bi-LM 结构（自绘重制，引 1802.05365）
# 所属章节：ch01-范式转移.md（原理章无教学代码，本脚本仅出图）
# 运行方式：source env.sh && python "code/ch01/make_figures.py"
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, SUB, AXIS = "#0b0b0b", "#52514e", "#898781"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 9,
})

FIG_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)


def box(ax, x0, y0, w, h, text, fc, ec, fs=8.0, lw=1.2, tc=INK):
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor=fc, edgecolor=ec,
                           linewidth=lw, zorder=3))
    ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=4, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, lw=1.5, color=SUB, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=12, linewidth=lw, color=color,
                                 linestyle=ls, zorder=2))


def fig_1_1():
    """五环论证链：监督稀缺(问题) → 静态词向量 → 上下文向量 → 微调配方 → 工业化。

    问题环用灰（非解），四个解答环按规范分类色序 蓝/橙/青/黄；箭头上标
    「上一环被击破的局限」，即逼出下一环的压力。
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    rings = [
        (0.25, "#f0efec", SUB, "Ring 0 (the problem)\nsupervised data\nruns out",
         "labeled: $10^4$–$10^5$ pairs\nunlabeled: $\\sim10^9$ words"),
        (3.55, "#eaf2fc", BLUE, "Ring 1\nstatic embeddings\nword2vec · GloVe\n(2013–14)",
         "signal from raw text:\npredict nearby words"),
        (6.85, "#fdeee7", ORANGE, "Ring 2\ncontextual vectors\nELMo bi-LM\n(2018)",
         "one vector per token,\nnot per word type"),
        (10.15, "#e7f6f0", TEAL, "Ring 3\nfine-tuning recipe\nULMFiT\n(2018)",
         "pre-train once,\nadapt per task"),
        (13.45, "#fdf3e0", YELLOW, "Ring 4\nindustrialization\nGPT-1 · BERT\n(2018)",
         "signal + representation\n+ recipe on one\nTransformer backbone"),
    ]
    for x0, fc, ec, title, sub in rings:
        box(ax, x0, 3.1, 2.9, 3.3, title, fc, ec, fs=8.5)
        ax.text(x0 + 1.45, 2.15, sub, ha="center", va="top", fontsize=7.0,
                color=SUB, linespacing=1.4)

    gaps = [
        (3.15, "need a training signal\nfrom unlabeled text"),
        (6.45, "one vector per word type:\nno disambiguation"),
        (9.75, "feature-based: a new task\narchitecture each time"),
        (13.05, "LSTM backbone:\nshort range, classification"),
    ]
    for x, label in gaps:
        arrow(ax, x, 4.75, x + 0.4, 4.75, lw=2.0, color=INK)
        ax.text(x + 0.2, 6.75, label, ha="center", va="center", fontsize=6.8,
                color=SUB, linespacing=1.35)

    ax.text(8.0, 0.55, "each arrow names the limitation that forces the next ring "
                       "(detailed in Table 1.1)", ha="center", fontsize=7.2,
            color=AXIS, style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-1-1-five-ring-chain.png"), dpi=300)
    plt.close(fig)


def fig_1_2():
    """ELMo bi-LM 结构（自绘重制，引 1802.05365 Fig.3 风格；许可核不到故自绘）。

    左侧主图：token 层（字级 CNN）→ 两个 biLSTM 层带（层带内前向蓝 / 后向橙
    两行，层间残差）；右侧面板：每个 token 的 2L+1 层表示与任务可学的层加权。
    形状标注（序列口径，n 为 token 数）：token 层 (n,512)；每个方向 (n,512)；
    拼接层 h_j (n,1024)。
    """
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.4)
    ax.axis("off")

    xs = [2.3, 4.1, 5.9, 7.7]          # 四个 token 列中心
    toks = ["$t_1$", "$t_2$", "$t_3$", "$t_4$"]

    # —— 底部：token 串 ——
    for x, t in zip(xs, toks):
        box(ax, x - 0.62, 0.42, 1.24, 0.55, t, "#f3f7fd", BLUE, fs=8.5)
    ax.text(1.0, 0.70, "chars", ha="center", va="center", fontsize=7, color=SUB)

    # —— token 层：字级 CNN（h_0） ——
    for x in xs:
        box(ax, x - 0.72, 1.42, 1.44, 0.8,
            "char CNN\n+highway+proj", "white", BLUE, fs=6.0)
        arrow(ax, x, 0.97, x, 1.42)
    ax.text(8.65, 1.82, "$h_{k,0}\\in\\mathbb{R}^{512}$\n$(n,512)$",
            ha="left", va="center", fontsize=6.8, color=SUB)

    # —— biLSTM 层带（前向蓝 / 后向橙） ——
    def bilstm_band(y0, layer_idx):
        ax.add_patch(Rectangle((1.15, y0), 7.15, 1.62, facecolor="#f7f8fa",
                               edgecolor=AXIS, linewidth=0.9, zorder=1))
        ax.text(1.02, y0 + 0.81, "biLSTM\nlayer %d" % layer_idx, ha="right",
                va="center", fontsize=6.8, color=SUB)
        for x in xs:
            box(ax, x - 0.62, y0 + 0.88, 1.24, 0.60, "fwd", "white",
                BLUE, fs=6.6, tc=BLUE)
            box(ax, x - 0.62, y0 + 0.10, 1.24, 0.60, "bwd", "white",
                ORANGE, fs=6.6, tc=ORANGE)
        for a, b in zip(xs[:-1], xs[1:]):
            arrow(ax, a + 0.64, y0 + 1.18, b - 0.64, y0 + 1.18, color=BLUE)
            arrow(ax, b - 0.64, y0 + 0.40, a + 0.64, y0 + 0.40, color=ORANGE)
        ax.text(8.65, y0 + 1.18, "proj $4096\\!\\to\\!512$, each dir $(n,512)$",
                ha="left", va="center", fontsize=6.4, color=SUB)
        ax.text(8.65, y0 + 0.40,
                "$h_{k,%d}=[\\overrightarrow{h};\\overleftarrow{h}]$ $(n,1024)$"
                % layer_idx,
                ha="left", va="center", fontsize=6.6, color=INK)

    bilstm_band(2.52, 1)
    bilstm_band(4.86, 2)
    # 纵向数据流：h0 → 层带1 → 层带2；左侧虚线=层间残差
    for x in xs:
        arrow(ax, x, 2.22, x, 2.52, lw=1.2)
        arrow(ax, x + 0.28, 4.14, x + 0.28, 4.86, lw=1.2)
    arrow(ax, 1.55, 4.14, 1.55, 4.86, color=SUB, ls="--", lw=1.1)
    ax.text(1.45, 4.50, "residual", ha="right", va="center",
            fontsize=6.4, color=SUB)

    # —— 训练目标：两端的 softmax ——
    arrow(ax, 8.32, 6.94, 8.62, 6.94, color=BLUE, lw=1.3)
    ax.text(8.70, 6.94, "softmax $\\hat{t}_{k+1}$", ha="left", va="center",
            fontsize=6.6, color=BLUE)
    arrow(ax, 1.12, 3.02, 0.82, 3.02, color=ORANGE, lw=1.3)
    ax.text(0.75, 3.02, "$\\hat{t}_{k-1}$", ha="right", va="center",
            fontsize=6.6, color=ORANGE)
    ax.text(4.9, 8.05, "training loss = forward + backward log-likelihood\n"
                       "(token embedding $\\Theta_x$ and softmax $\\Theta_s$ "
                       "shared by both directions)",
            ha="center", va="center", fontsize=7.0, color=SUB, linespacing=1.5)
    ax.text(4.9, 0.28, "shape flow: ids $(n,)$ $\\to$ $h_0$ $(n,512)$ "
            "$\\to$ [fwd; bwd] $(n,1024)$ per layer",
            ha="center", va="center", fontsize=6.6, color=AXIS)

    # —— 右侧面板：ELMo 加权 ——
    px = 10.6
    ax.add_patch(Rectangle((px, 0.55), 5.1, 7.7, facecolor="#e7f6f0",
                           edgecolor=TEAL, linewidth=1.4, zorder=1))
    ax.text(px + 2.55, 7.85, "ELMo: learned layer mixing", ha="center",
            fontsize=8.2, color=INK)
    rows = [
        ("layer 0   $h_{k,0}$", "(512,)"),
        ("layer 1   $h_{k,1}$", "(1024,)"),
        ("layer 2   $h_{k,2}$", "(1024,)"),
    ]
    for i, (name, dim) in enumerate(rows):
        y = 6.85 - i * 0.80
        box(ax, px + 0.35, y - 0.29, 2.75, 0.58, name, "white", TEAL, fs=7.2)
        ax.text(px + 3.35, y, dim, ha="center", va="center", fontsize=7.0,
                color=SUB)
        arrow(ax, px + 3.85, y - 0.18, px + 3.05, 2.90, lw=1.0, color=TEAL)
    ax.text(px + 0.35, 4.28, "$2L+1=5$ representations per token\n"
            "$s_j$: softmax weights (per task)\n$\\gamma$: learned task scale",
            ha="left", va="top", fontsize=6.8, color=SUB, linespacing=1.5)
    box(ax, px + 1.05, 2.42, 3.0, 0.95,
        "$\\mathrm{ELMo}_k = \\gamma\\sum_{j=0}^{L} s_j\\, h_{k,j}$",
        "#ffffff", TEAL, fs=7.6)
    ax.text(px + 2.55, 1.80, "usage: $[\\,x_k;\\,\\mathrm{ELMo}_k\\,]$ fed to a "
            "task model\n(biLM frozen — feature-based)", ha="center",
            va="center", fontsize=6.8, color=SUB, linespacing=1.5)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-1-2-elmo-bilm.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_1_1()
    fig_1_2()
    print("figures written to", FIG_DIR)
    for f in sorted(os.listdir(FIG_DIR)):
        if f.startswith("fig-1-"):
            print(" -", f)
