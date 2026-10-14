# fig_ch13.py —— Book8 ch13 制图：图 13.1 八册地图终点（收官图）
# 所属章节：Book8 ch13（§13.3）
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "drafts", "Book8-vLLM源码剖析下", "figures"))
BLUE, ORANGE, TEAL = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"

BOOKS = [
    ("B1 Transformer\n原典", BLUE), ("B2 预训练\n革命", BLUE), ("B3 现代\n开源骨架", BLUE),
    ("B4 规模\n与效率", BLUE), ("B5 前沿\n混合架构", BLUE),
    ("B6 推理\n系统导论", TEAL), ("B7 vLLM V1\n源码·上", ORANGE), ("B8 vLLM V1\n源码·下", ORANGE),
]

def fig_end():
    fig, ax = plt.subplots(figsize=(12.4, 4.4), dpi=300)
    ax.set_xlim(0, 16); ax.set_ylim(0, 6); ax.axis("off")
    ax.text(8, 5.6, "The eight-book journey: model side (B1-B5) -> systems side (B6-B8) -> the end of the map",
            ha="center", fontsize=9, color=INK, weight="bold")
    x = 0.6
    for i, (name, color) in enumerate(BOOKS):
        w = 1.75
        ax.add_patch(FancyBboxPatch((x, 2.4), w, 1.7, boxstyle="round,pad=0.03",
                                    fc=color, ec="white", lw=1.2, alpha=0.88, zorder=3))
        ax.text(x + w/2, 3.25, name, ha="center", va="center",
                fontsize=6.6, color="white", zorder=4)
        if i < 7:
            ax.annotate("", xy=(x + w + 0.15, 3.25), xytext=(x + w, 3.25),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
        x += w + 0.15
    ax.text(8, 1.7, "two turns: B5->B6 (designing models -> serving models), B7->B8 (vertical dissection -> horizontal topics)",
            ha="center", fontsize=7, color=MUTED, style="italic")
    ax.text(8, 1.0, "one hidden thread: the O(n^2) & autoregressive-serial 'not enough' from 2017 -> paged KV / speculation / hybrid / sparse",
            ha="center", fontsize=7, color=MUTED)
    ax.text(0.6, 0.3, "fin.", fontsize=8, color=INK, style="italic")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-13-1-eight-books-end.png")
    fig.savefig(out, facecolor="white"); plt.close(fig)
    print("[fig]", out)

if __name__ == "__main__":
    fig_end()
