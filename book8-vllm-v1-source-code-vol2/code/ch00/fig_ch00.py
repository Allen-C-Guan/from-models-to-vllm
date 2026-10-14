# fig_ch00.py —— Book8 ch00 制图：图 0.1 专题地图（母图——17 站×专题挂站）
# 所属章节：Book8 ch0（§0.6）；底图=Book7 图 0.1 的 17 站横轴（改标专题）
# 运行：cd <workspace> && source env.sh && python code/Book8-vLLM源码剖析下/ch00/fig_ch00.py
# 产物：drafts/Book8-vLLM源码剖析下/figures/fig-0-1-topic-map.png
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "drafts", "Book8-vLLM源码剖析下", "figures"))
BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#0b0b0b", "#52514e"

GROUPS = ["1-3\nprotocol", "4-6\nfrontend", "7-8\nsched", "9-10\nKV", "11\nfwd", "12\nops", "13\nsample", "14-17\noutput"]
TOPICS = [
    ("ch1 hybrid KV", [0, 0, 0, 3, 0, 0, 0, 0], BLUE),
    ("ch2 spec", [0, 0, 3, 0, 3, 0, 3, 0], ORANGE),
    ("ch3 structured", [3, 0, 3, 0, 3, 3, 0, 0], TEAL),
    ("ch4 mm/LoRA", [3, 3, 0, 3, 3, 0, 0, 3], YELLOW),
    ("ch5 overlap", [0, 0, 3, 0, 0, 0, 0, 0], BLUE),
    ("ch6-7 dist/PD", [0, 0, 3, 3, 3, 3, 0, 0], ORANGE),
    ("ch8 quant", [3, 0, 0, 0, 3, 0, 0, 0], TEAL),
    ("ch9 observ", [3, 0, 0, 0, 0, 0, 0, 3], YELLOW),
    ("ch10-12 prod", [3, 0, 0, 0, 0, 0, 0, 0], "#9b9a94"),
]


def fig_topics():
    fig, ax = plt.subplots(figsize=(12.2, 6.6), dpi=300)
    ax.set_xlim(0, 12); ax.set_ylim(0, 10.4); ax.axis("off")
    ax.text(6, 10.1, "Topic map: nine special topics on the 17-station spine (Book7's map, re-labeled)",
            ha="center", fontsize=9.2, color=INK, weight="bold")
    x0, y0, cw, chh = 2.2, 0.9, 1.12, 0.9
    for j, g in enumerate(GROUPS):
        ax.text(x0 + j * cw + cw / 2, y0 + 9 * chh + 0.18, g, ha="center", va="bottom", fontsize=6.2, color=MUTED)
    ax.text(x0 - 0.2, y0 + 9 * chh + 0.62, "stations →", fontsize=7, color=MUTED, style="italic")
    for i, (name, marks, color) in enumerate(TOPICS):
        y = y0 + (8 - i) * chh
        ax.text(x0 - 0.15, y + chh / 2, name, ha="right", va="center", fontsize=7.2, color=INK)
        for j, m in enumerate(marks):
            if m:
                ax.add_patch(FancyBboxPatch((x0 + j * cw + 0.04, y + 0.06), cw - 0.12, chh - 0.16,
                                            boxstyle="round,pad=0.02", fc=color, ec="white",
                                            lw=0.8, alpha=0.85, zorder=3))
    ax.text(x0, 0.35, "solid = main anchoring stations of the topic; ch10-12 row = production line (cross-cutting, anchored at station 0/edge)",
            fontsize=6.6, color=MUTED, style="italic")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-0-1-topic-map.png")
    fig.savefig(out, facecolor="white"); plt.close(fig)
    print("[fig]", out)


if __name__ == "__main__":
    fig_topics()
