# fig_ch12.py —— Book8 ch12 制图：图 12.1 八册×17 站总地图（压轴总图）
# 所属章节：Book8 ch12（§12.3）；数据=表 12.2 投影矩阵
# 运行：cd <workspace> && source env.sh && python code/Book8-vLLM源码剖析下/ch12/fig_ch12.py
# 产物：drafts/Book8-vLLM源码剖析下/figures/fig-12-1-eight-books-map.png
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "drafts", "Book8-vLLM源码剖析下", "figures"))
BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#0b0b0b", "#52514e"

# 站分组（17 站折成 8 组列）与册×组贡献密度 0-3
GROUPS = ["1-3\nprotocol", "4-6\nfrontend", "7-8\nschedule", "9-10\nKV blocks", "11\nfwd", "12\nattn/ops", "13\nsample", "14-17\noutput"]
BOOKS = ["B1 原典", "B2 预训练", "B3 骨架", "B4 规模", "B5 前沿", "B6 推理", "B7 源码上", "B8 本册"]
# 贡献密度：0=无、1=指回、2=有账、3=主讲
D = np.array([
    [1,0,0,0,1,0,0,0],   # B1 tokenizer/attention math
    [0,0,0,0,1,0,0,1],   # B2 weights provenance
    [0,0,1,3,2,2,0,0],   # B3 KV ledger/GQA/RoPE
    [0,0,1,1,2,3,0,0],   # B4 MoE/quant math
    [0,0,0,1,1,2,0,0],   # B5 hybrid/sparse needs
    [0,0,2,2,2,1,1,0],   # B6 engine-agnostic physics
    [2,3,3,3,3,3,3,3],   # B7 main-line read
    [2,2,2,2,2,2,2,0],   # B8 topics (no output-station chapter)
])
COLORS = {0: "#f5f4f0", 1: "#dcebfc", 2: "#9dc3ec", 3: "#2a78d6"}


def fig_map():
    fig, ax = plt.subplots(figsize=(11.8, 6.2), dpi=300)
    ax.set_xlim(0, 10); ax.set_ylim(0, 9.6); ax.axis("off")
    ax.text(5, 9.3, "Eight books x 17 stations: the full-series knowledge map", ha="center",
            fontsize=9.5, color=INK, weight="bold")
    x0, y0, cw, chh = 1.6, 0.8, 0.95, 0.92
    for j, g in enumerate(GROUPS):
        ax.text(x0 + j * cw + cw / 2, y0 + 8 * chh + 0.25, g, ha="center", va="bottom",
                fontsize=6.2, color=MUTED)
    for i, b in enumerate(BOOKS):
        y = y0 + (7 - i) * chh
        ax.text(x0 - 0.15, y + chh / 2, b, ha="right", va="center", fontsize=7, color=INK)
        for j in range(len(GROUPS)):
            v = D[i, j]
            ax.add_patch(Rectangle((x0 + j * cw, y), cw * 0.92, chh * 0.9,
                                   fc=COLORS[v], ec="white", lw=0.8, zorder=3))
    # 图例
    for k, lab in enumerate(["none", "refers back", "has ledger", "main line"]):
        ax.add_patch(Rectangle((x0 + 8 * cw - 3.4 + k * 0.85, 0.25), 0.5, 0.32,
                               fc=COLORS[k], ec="white", lw=0.8))
        ax.text(x0 + 8 * cw - 3.4 + k * 0.85 + 0.25, 0.12, lab, ha="center", fontsize=5.8, color=MUTED)
    ax.text(x0, 0.12, "B7 row = vertical dissection; B8 row = horizontal topic needle; together they stitch both sides",
            fontsize=6.4, color=MUTED, style="italic")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-12-1-eight-books-map.png")
    fig.savefig(out, facecolor="white"); plt.close(fig)
    print("[fig]", out)


if __name__ == "__main__":
    fig_map()
