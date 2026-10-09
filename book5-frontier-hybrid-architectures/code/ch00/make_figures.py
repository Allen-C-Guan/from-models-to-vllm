# 用途：生成图 0.1/0.2 —— 导览两图（0.1 三条突围路线总览：一台满注意力机器+三把刀各自动哪里；0.2 K3 玩具粗图 vs Book4 两刀机：注意力格从「MLA 一色」变「3:1 交替」高亮，示意级）
# 所属章节：Book5 第 0 章导览（ch00-导览.md 图 0.1/图 0.2；数字为探针/大纲定版路标，正文引用以各章正式脚本为准）
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch00/make_figures.py" [--only 1|2]
# 图规：_写作规范.md §4（300dpi / 固定分类色蓝橙青黄 / 图内英文图注中文 / 结构图张量框标形状）；输出文件名带本章专属 slug（fig-0-*-，跨册冲撞教训）

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from pathlib import Path

# ---- 分类色（写作规范固定序）----
BLUE = "#2a78d6"     # 满注意力正身 / 不动件
TEAL = "#1baf7a"     # 路线一（窗口）/ KDA 线性层
YELLOW = "#eda100"   # 路线二（稀疏）/ MoE 槽
ORANGE = "#eb6834"   # 路线三（换算子）/ 保留全局层高亮
INK = "#0b0b0b"
SUB = "#52514e"
GRAY = "#898781"
GHOST = "#b8b6ae"
FILL_BLUE = "#cde2fb"
FILL_TEAL = "#c9f0e2"
FILL_YELLOW = "#fdeec9"
FILL_ORANGE = "#fbd9cb"

FIG_DIR = Path(__file__).resolve().parents[3] / "book5-frontier-hybrid-architectures" / "figures"


def new_ax(w, h, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w, h), dpi=300)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def box(ax, x, y, w, h, text, ec=BLUE, fc="white", lw=1.6, fs=7.0, tc=INK, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.05,rounding_size=0.10",
                                linewidth=lw, edgecolor=ec, facecolor=fc,
                                linestyle=ls, zorder=3))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=4, linespacing=1.45)


def arrow(ax, p1, p2, color=SUB, ls="-", lw=1.5):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=10,
                                 linewidth=lw, color=color, linestyle=ls,
                                 zorder=2, shrinkA=1, shrinkB=1))


def badge(ax, x, y, n, color=ORANGE):
    ax.add_patch(Circle((x, y), 0.24, facecolor=color, edgecolor="white",
                        linewidth=1.1, zorder=6))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=8,
            color="white", fontweight="bold", zorder=7)


def score_grid(ax, x0, y0, cs, n, filled, ec, fc, ghost=False):
    """画 n×n 打分矩阵示意格：filled(i,j)=True 的格子上色；ghost=True 只画虚线轮廓（路线三：表被废）。"""
    for i in range(n):
        for j in range(n):
            if ghost:
                face, edge, ls = "white", GHOST, (0, (2, 2))
            else:
                face, edge, ls = (fc if filled(i, j) else "white"), (ec if filled(i, j) else "#e1e0d9"), "-"
            ax.add_patch(Rectangle((x0 + j * cs, y0 + (n - 1 - i) * cs), cs, cs,
                                   facecolor=face, edgecolor=edge, linewidth=0.7,
                                   linestyle=ls, zorder=3))


# =====================================================================
# 图 0.1 三条突围路线总览：一台满注意力机器 + 三把刀各自动哪里
# =====================================================================
def fig_1():
    fig, ax = new_ax(8.8, 4.7, 17.6, 9.4)
    n = 8

    ax.text(8.8, 9.1, "One object, three escape routes — the score matrix $(h,\\,n,\\,n)$ of full attention on the operating table",
            ha="center", va="center", fontsize=9.5, color=INK)
    ax.text(8.8, 8.74, "the dot-product softmax operator survives until Ch.5; what gets cut first is who is allowed into the matrix",
            ha="center", va="center", fontsize=7.5, color=SUB)

    # ---- 左：手术台上的本体（满注意力）----
    box(ax, 0.35, 0.85, 5.55, 7.45, "", lw=1.8, fc="#f5f9fe")
    ax.text(3.12, 7.98, "The patient: full attention (2017, Book1 ch2)",
            ha="center", va="center", fontsize=8.4, color=INK, zorder=4)
    ax.text(3.12, 7.58, "scores $QK^{\\top}$: $(B,h,n,d_k)\\cdot(B,h,d_k,n)\\to(B,h,n,n)$",
            ha="center", va="center", fontsize=6.8, color=SUB, zorder=4)
    score_grid(ax, 0.90, 3.05, 0.51, n, lambda i, j: j <= i, BLUE, FILL_BLUE)
    ax.text(2.95, 2.70, "keys $j\\ \\to$", ha="center", fontsize=6.8, color=GRAY)
    ax.text(0.66, 5.10, "queries $i$", rotation=90, ha="center", va="center",
            fontsize=6.8, color=GRAY)
    ax.text(3.12, 2.20, "all causal pairs kept — $n(n{+}1)/2$ of $n^2$ cells,\ngrows with $n$ (the O$(n^2)$ account)",
            ha="center", fontsize=7.0, color=SUB, linespacing=1.5)
    box(ax, 0.85, 1.05, 4.6, 0.62, "softmax per row $\\to\\ \\cdot V$ $(B,h,n,d_v)$ $\\to$ output",
        ec=BLUE, fs=6.6, fc="white")

    # ---- 右：三把刀纵排（同一张表，三种动法；箭头各自水平入面板，互不穿越）----
    panels = [
        (6.20, 6.05, TEAL, FILL_TEAL, "Route 1 $\\cdot$ Ch.2 — window + sink",
         "trim the view: each row keeps only its last $W$ keys ($W$=3 here);\nthe operator is untouched — a banded mask does the surgery",
         "per-layer KV: $\\min(n, W)$ — bounded   (Gemma3, gpt-oss)"),
        (6.20, 3.53, YELLOW, FILL_YELLOW, "Route 2 $\\cdot$ Ch.3-4 — learnable sparse",
         "pick the participants: top-$k$ over contiguous blocks, learned\nend-to-end (NSA $\\to$ DSA $\\to$ CSA); each row is still an exact dot product,\nbut only on the chosen blocks",
         "the participant set becomes a parameter, not a hyper-parameter"),
        (6.20, 1.01, ORANGE, FILL_ORANGE, "Route 3 $\\cdot$ Ch.5 — linear operator",
         "replace the operator: no $n\\times n$ matrix is ever formed —\na fixed-size state accumulates every token:  write $\\to$ forget $\\to$ read",
         "state $S_t$: $(B,h,d_k,d_v)$ fixed — never grows   (GDN, KDA)"),
    ]
    filled_r1 = lambda i, j: 0 <= i - j <= 2
    blocks = [{0}, {0}, {0, 1}, {0, 1}, {0, 2}, {0, 2}, {1, 3}, {2, 3}]
    filled_r2 = lambda i, j: (j // 2 in blocks[i]) and (j <= i)
    for k, (px, py, ec, fc, title, mech, acct) in enumerate(panels):
        ph = 2.30
        box(ax, px, py, 11.15, ph, "", ec=ec, fc="#fbfcfd" if k < 2 else "#fffaf8", lw=1.8)
        badge(ax, px + 0.34, py + ph - 0.30, k + 1, color=ec)
        ax.text(px + 0.75, py + ph - 0.30, title, ha="left", va="center", fontsize=7.8,
                color=INK, zorder=4)
        if k < 2:
            filled = filled_r1 if k == 0 else filled_r2
            score_grid(ax, px + 0.30, py + 0.30, 0.212, n, filled, ec, fc)
        else:
            score_grid(ax, px + 0.30, py + 0.30, 0.212, n, lambda i, j: False, ec, fc, ghost=True)
            ax.text(px + 0.30 + 4 * 0.212, py + 0.30 + 4 * 0.212, "$\\times$",
                    ha="center", va="center", fontsize=13, color=ORANGE, zorder=6)
            box(ax, px + 2.35, py + 0.42, 2.10, 1.35,
                "fixed state $S_t$\n$(B,\\,h,\\,d_k,\\,d_v)$",
                ec=ORANGE, fc=FILL_ORANGE, fs=6.8)
        tx = px + 4.75
        ax.text(tx, py + ph - 0.62, mech, ha="left", va="center", fontsize=6.6,
                color=SUB, zorder=4, linespacing=1.55)
        ax.text(tx, py + 0.42, acct, ha="left", va="center", fontsize=6.6,
                color=ec, zorder=4, linespacing=1.5)
        arrow(ax, (5.97, py + ph / 2), (px - 0.04, py + ph / 2), color=ec, lw=1.7)

    ax.text(8.8, 0.40, "same matrix, three surgeries — ordered by how much they change:  cut the view (cheapest)  $\\to$  cut the participant set  $\\to$  replace the operator (deepest)",
            ha="center", fontsize=7.2, color=INK)

    out = FIG_DIR / "fig-0-1-attn-three-routes.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


# =====================================================================
# 图 0.2 K3 玩具粗图 vs Book4 两刀机（注意力格「MLA 一色」→「3:1 交替」）
# =====================================================================
def fig_2():
    fig, ax = new_ax(8.8, 4.7, 17.6, 9.4)

    ax.text(8.8, 9.12, "The machine of this book: a K3 toy (43.9M) vs the Book4 two-cut machine (409M)",
            ha="center", va="center", fontsize=9.5, color=INK)
    ax.text(8.8, 8.76, "same skeleton discipline — only the attention slot changes:  'all MLA'  $\\to$  a 3:1 alternation (3 linear : 1 global)",
            ha="center", va="center", fontsize=7.5, color=SUB)

    # ---- 左：Book4 两刀机（12 层，注意力格 MLA 一色）----
    box(ax, 0.35, 1.95, 4.75, 6.45, "", lw=1.8)
    ax.text(2.72, 8.10, "Book4 ch9 $\\cdot$ two-cut machine  409,115,648",
            ha="center", va="center", fontsize=8.0, color=INK, zorder=4)
    ax.text(2.72, 7.74, "$d$=1024, $L$=12, $V$=32000 (sp-32k); MoE $E$=8 top-2 +1s",
            ha="center", va="center", fontsize=6.4, color=GRAY, zorder=4)
    for li in range(12):
        y = 7.30 - li * 0.445
        ax.text(0.62, y + 0.155, str(li + 1), ha="right", va="center",
                fontsize=5.4, color=GRAY)
        box(ax, 0.70, y, 1.55, 0.33, "MLA", ec=BLUE, fc=FILL_BLUE, fs=5.6, lw=1.0)
        box(ax, 2.31, y, 1.55, 0.33, "MoE", ec=YELLOW, fc=FILL_YELLOW, fs=5.6, lw=1.0)
    ax.text(4.42, 4.9, "attention slot:\nMLA $\\times$ 12\n— one color", rotation=90,
            ha="center", va="center", fontsize=6.6, color=BLUE, linespacing=1.5)
    ax.text(3.08, 2.12, "residual river $(B,\\,n,\\,1024)$; RMSNorm/RoPE unchanged (Book3)",
            ha="center", va="center", fontsize=5.8, color=GRAY, zorder=4)

    # ---- 中：动刀箭头 ----
    arrow(ax, (5.25, 5.0), (9.30, 5.0), color=ORANGE, lw=2.2)
    ax.text(7.28, 5.35, "attention slot re-cut\n(this book's only architectural cut)",
            ha="center", va="center", fontsize=6.6, color=ORANGE, linespacing=1.4)

    # ---- 右：K3 玩具（8 层，[KDA×3, MLA]×2 交替高亮）----
    box(ax, 9.45, 1.95, 7.85, 6.45, "", ec=ORANGE, lw=1.8, fc="#fffdfa")
    ax.text(13.38, 8.10, "Book5 ch6 $\\cdot$ K3 toy  43,905,888 / 11,662,176 active",
            ha="center", va="center", fontsize=8.0, color=INK, zorder=4)
    ax.text(13.38, 7.74, "$d$=512, $L$=8, $V$=8192 (sp-8k) — pattern $[$KDA$\\times$3, MLA$]\\times$2",
            ha="center", va="center", fontsize=6.4, color=GRAY, zorder=4)
    pattern = ["KDA", "KDA", "KDA", "MLA", "KDA", "KDA", "KDA", "MLA"]
    for li, kind in enumerate(pattern):
        y = 7.30 - li * 0.545
        ax.text(9.72, y + 0.235, str(li + 1), ha="right", va="center",
                fontsize=5.8, color=GRAY)
        if kind == "KDA":
            box(ax, 9.80, y, 1.80, 0.45, "KDA", ec=TEAL, fc=FILL_TEAL, fs=6.4, lw=1.2)
        else:
            box(ax, 9.80, y, 1.80, 0.45, "MLA", ec=ORANGE, fc=FILL_ORANGE, fs=6.4, lw=1.6)
        box(ax, 11.66, y, 1.80, 0.45, "MoE", ec=YELLOW, fc=FILL_YELLOW, fs=6.4, lw=1.0)
    # 3:1 循环括注（第 1-4 层）
    ax.plot([13.62, 13.80, 13.80, 13.62],
            [7.75 - 3 * 0.545, 7.75 - 3 * 0.545, 7.75, 7.75],
            color=INK, lw=1.3)
    ax.text(13.95, 6.36, "3:1 cycle — 3 linear : 1 global;\nlast layer global (K3 convention)",
            ha="left", va="center", fontsize=6.4, color=INK, linespacing=1.5)
    ax.text(13.95, 4.42, "KDA: fixed state\n32,768 elem / layer", ha="left", va="center",
            fontsize=6.2, color=TEAL, linespacing=1.5)
    ax.text(13.95, 3.20, "MLA: KV unbounded\n(latent 128 + rope 32)", ha="left", va="center",
            fontsize=6.2, color=ORANGE, linespacing=1.5)
    ax.text(12.62, 2.12, "data river $(B,\\,n,\\,512)$; MoE: $E$=64, top-4 + 1 shared, $w$=32",
            ha="center", va="center", fontsize=5.8, color=GRAY, zorder=4)

    # ---- 底部：血统写在 import 语句里 ----
    box(ax, 0.35, 0.30, 16.95, 1.35, "", ec=SUB, fc="#f7f7f5", lw=1.2)
    ax.text(0.65, 1.38, "import lineage — the bloodline written in import statements:",
            ha="left", va="center", fontsize=7.4, color=INK)
    ax.text(0.65, 0.98, "RMSNorm / RoPE $\\leftarrow$ Book3 $\\cdot$ MLA, DeepSeekMoE $\\leftarrow$ Book4 $\\cdot$ KDA $\\leftarrow$ Book5 ch6   |   golden truth: Kimi K3 tech report (arXiv:2607.24653)",
            ha="left", va="center", fontsize=6.8, color=SUB)
    ax.text(0.65, 0.60, "toy = schema-faithful miniature of the 3:1 recipe, not K3 behavior at 43.9M   |   ch9 (optional): $[$GDN$\\times$3, MLA$]\\times$3 $\\approx$409M — 207M/409M/409M iso-param chain",
            ha="left", va="center", fontsize=6.8, color=SUB)

    out = FIG_DIR / "fig-0-2-k3-toy-3to1.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, choices=[1, 2], default=None,
                    help="只渲染指定图（默认两张都出；输出文件名带专属 slug，不覆写他图）")
    args = ap.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if args.only in (None, 1):
        fig_1()
    if args.only in (None, 2):
        fig_2()


if __name__ == "__main__":
    main()
