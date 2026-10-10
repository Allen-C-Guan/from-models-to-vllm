# fig_ch13.py —— Book6 ch13 制图：图 13.1 八册地图（you are here=第六站）（slug=finale）
# 用途：收束章的系统视角门载体——八站旅程地图：站 6 高亮（本册），左侧五册旧账已清（F/B 钩回收），
#       右侧 Book7/8 两站（B6 系新钩流向）；沿 Book4/5 收束章「站点画法+you are here 标记」图式。
# 所属章节：Book6 第 13 章（图 13.1）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/Book6-推理系统导论/ch13/fig_ch13.py"
# 产物：drafts/Book6-推理系统导论/figures/fig-13-1-eight-books.png
# 图规：300dpi 白底；系列色 蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；图内英文、图注中文。
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


def fig_map():
    fig, ax = plt.subplots(figsize=(12.6, 4.2), dpi=300)
    ax.set_xlim(0, 16.4); ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.text(8.2, 5.35, "Eight books, one journey: from the 2017 Transformer to the vLLM source — you are HERE (station 6)",
            ha="center", fontsize=9.8, color=INK, weight="bold")
    books = [
        (0.4, "B1", "Transformer\n原典", MUTED, "white"),
        (2.4, "B2", "预训练\n革命", MUTED, "white"),
        (4.4, "B3", "现代开源\n骨架", MUTED, "white"),
        (6.4, "B4", "规模与\n效率", MUTED, "white"),
        (8.4, "B5", "前沿混合\n架构", MUTED, "white"),
        (10.4, "B6", "推理系统\n导论", ORANGE, "#fde3d6"),
        (12.4, "B7", "vLLM V1\n源码·上", BLUE, "#dcebfc"),
        (14.4, "B8", "vLLM V1\n源码·下", BLUE, "#dcebfc"),
    ]
    for x, tag, name, ec, fc in books:
        hot = tag == "B6"
        ax.add_patch(FancyBboxPatch((x, 2.0), 1.6, 1.7, boxstyle="round,pad=0.03",
                                    ec=ORANGE if hot else ec, fc=fc, lw=2.6 if hot else 1.2, zorder=3))
        ax.text(x + 0.8, 3.15, tag, ha="center", fontsize=9.5, color=INK, weight="bold", zorder=4)
        ax.text(x + 0.8, 2.45, name, ha="center", fontsize=7.2, color=INK, zorder=4)
        if hot:
            ax.annotate("YOU ARE HERE", xy=(x + 0.8, 4.15), ha="center", fontsize=8.2, color=ORANGE, weight="bold")
            ax.annotate("", xy=(x + 0.8, 3.9), xytext=(x + 0.8, 4.05),
                        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4))
    for i in range(len(books) - 1):
        x0 = books[i][0] + 1.6
        ax.annotate("", xy=(x0 + 0.4, 2.85), xytext=(x0, 2.85),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0))
    # 左侧旧账 / 右侧新钩
    ax.text(4.6, 1.15, "five books' IOUs settled here:\nF1-F18 / N5-N7 / B3-B5 hooks closed in ch1-12",
            ha="center", fontsize=7.6, color=TEAL)
    ax.plot([7.6, 8.2], [1.4, 1.9], color=TEAL, lw=1.0)
    ax.text(12.9, 1.15, "new IOUs (B6-1..9) ride on:\nBook7 = source mainline\nBook8 = specials & production",
            ha="center", fontsize=7.6, color=BLUE)
    ax.plot([12.0, 11.6], [1.4, 1.9], color=BLUE, lw=1.0)
    ax.text(8.2, 0.35, "what you carry: five books of architecture depth + one book of inference physics — the last two books wire the physics into engineering",
            ha="center", fontsize=7.4, color=MUTED)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-13-1-eight-books.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 13.1] {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch13 制图（图 13.1 八册地图）")
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_map()
