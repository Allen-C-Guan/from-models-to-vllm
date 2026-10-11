# fig_ch13.py —— Book7 ch13 制图：图 13.1 八册地图（第七站）
# 定位：丛书八册在「模型→vLLM」主线上的位置图，本册（Book7）高亮。
# 数据源：plan/00-系列总览.md 八册书单（架构册 1-5 蓝系/推理册 6-8 橙系；Book7 高亮描边）。
# 所属章节：Book7 ch13（§13.5）。运行：source env.sh && python code/Book7-vLLM源码剖析上/ch13/fig_ch13.py
# 产物：drafts/Book7-vLLM源码剖析上/figures/fig-13-1-eight-books.png
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..",
                                      "drafts", "Book7-vLLM源码剖析上", "figures"))
BLUE, ORANGE, TEAL, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"

BOOKS = [
    ("B1", "Transformer\noriginal\n(2017)", BLUE),
    ("B2", "pretraining\nrevolution", BLUE),
    ("B3", "modern open\nskeleton\n(LLaMA-style)", BLUE),
    ("B4", "scale &\nefficiency\n(MoE/MLA)", BLUE),
    ("B5", "frontier\nhybrid\narchitectures", BLUE),
    ("B6", "inference\nsystems\nintro (physics)", ORANGE),
    ("B7", "vLLM V1\nsource code\nvol.1 = THIS BOOK", ORANGE),
    ("B8", "vLLM V1\nsource code\nvol.2 (topics)", TEAL),
]


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.6, 3.6), dpi=300)
    ax.set_xlim(0, 16.4)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    w, h, gap = 1.72, 1.5, 0.24
    for i, (tag, name, c) in enumerate(BOOKS):
        x = 0.3 + i * (w + gap)
        hot = tag == "B7"
        ax.add_patch(FancyBboxPatch((x, 1.0), w, h,
                                    boxstyle="round,pad=0.055,rounding_size=0.12",
                                    facecolor=c, alpha=0.55 if hot else 0.30,
                                    edgecolor=INK if hot else c,
                                    lw=2.0 if hot else 0.8, zorder=3))
        ax.text(x + w / 2, 2.24, tag, ha="center", fontsize=9, color=c if not hot else INK,
                weight="bold", zorder=4)
        ax.text(x + w / 2, 1.55, name, ha="center", va="center", fontsize=6.3,
                color=INK, zorder=4, linespacing=1.15)
        if i < len(BOOKS) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, 1.75), (x + w + gap - 0.02, 1.75),
                                         arrowstyle="->", mutation_scale=9,
                                         color=MUTED, lw=1.0, zorder=2))
    ax.text(4.6, 0.52, "the model side: five books of architecture evolution",
            fontsize=7.4, color=BLUE, ha="center", style="italic")
    ax.text(12.3, 0.52, "the serving side: physics -> engine source (you are here)",
            fontsize=7.4, color=ORANGE, ha="center", style="italic")
    ax.annotate("", xy=(8.0, 0.86), xytext=(1.2, 0.86),
                arrowprops=dict(arrowstyle="-", color="#e1e0d9", lw=1.2))
    ax.annotate("", xy=(15.6, 0.86), xytext=(8.6, 0.86),
                arrowprops=dict(arrowstyle="-", color="#e1e0d9", lw=1.2))
    ax.text(8.1, 3.05, "Eight-book map: from the model to vLLM — station 7 of 8",
            ha="center", fontsize=8.6, color=INK, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-13-1-eight-books.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print("[图 13.1] eight books map")


if __name__ == "__main__":
    main()
