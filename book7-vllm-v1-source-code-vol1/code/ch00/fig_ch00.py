# fig_ch00.py —— Book7 ch00 制图：图 0.1 本册读法地图（章 × 站 矩阵）
# 定位：导览级粗图——横轴 17 站（按进程三色）、纵轴章（按三段旅程分组），格子=该章覆盖的站段。
#   与 ch02 图 2.2（文件级细节版）分工：本图管「哪章在哪、按什么顺序读」，不标文件锚。
# 数据源：卷级大纲 §ch2 图 2.2 的站-章对应（ch3=1-6/ch4=6-8+14/ch5=8-9/ch6=9/ch7=10-12/
#   ch8+9=12 内部/ch11=13+15-17）+ 进程三色（1-6、15-16-17 前端/7-10、14 引擎核/11-13 worker）。
# 所属章节：Book7 ch00（§0.6）。运行：source env.sh && python code/Book7-vLLM源码剖析上/ch00/fig_ch00.py
# 产物：drafts/Book7-vLLM源码剖析上/figures/fig-0-1-reading-map.png
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..",
                                      "drafts", "Book7-vLLM源码剖析上", "figures"))
BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e1e0d9"

STATIONS = ["1\nprotocol", "2\nvalidate", "3\ntemplate", "4\ntokenize",
            "5\nqueue", "6\nZMQ out", "7\nadd_req", "8\nschedule",
            "9\nkv alloc", "10\ninputs", "11\nforward", "12\nattention",
            "13\nsample", "14\nZMQ back", "15\noutputs", "16\ndetok", "17\nstream"]
PROC = [BLUE] * 6 + [ORANGE] * 4 + [TEAL] * 3 + [ORANGE, BLUE, BLUE, BLUE]
# (label, [(start, end), ...], journey_color, dashed_hint)
ROWS = [
    ("ch1  prologue: ecology & 3 generations", [], MUTED),
    ("ch2  terrain map & reading method", [], MUTED),
    ("ch3  entry layer", [(1, 6)], BLUE),
    ("ch4  EngineCore process", [(6, 8), (14, 14)], BLUE),
    ("ch5  scheduler", [(8, 9)], ORANGE),
    ("ch6  KV & prefix cache", [(9, 9)], ORANGE),
    ("ch7  execution / model runner", [(10, 12)], ORANGE),
    ("ch8  compile & graph replay", [(12, 12)], ORANGE),
    ("ch9  attention backends", [(12, 12)], TEAL),
    ("ch10 weights & registry (startup)", [], YELLOW),
    ("ch11 sampling + output chain", [(13, 13), (15, 17)], TEAL),
    ("ch12 experiment methodology (all)", [(1, 17)], TEAL),
]


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.6, 6.4), dpi=300)
    n = len(STATIONS)
    # 站表头（进程三色）
    for i, (s, c) in enumerate(zip(STATIONS, PROC)):
        ax.add_patch(Rectangle((i + 0.5, len(ROWS) + 0.42), 0.94, 0.9,
                               facecolor=c, alpha=0.13, edgecolor=c, lw=0.6))
        ax.text(i + 0.97, len(ROWS) + 1.02, s, ha="center", va="center",
                fontsize=5.6, color=c, linespacing=0.95)
    # 进程带说明
    ax.text(3.2, len(ROWS) + 1.72, "frontend proc", color=BLUE, fontsize=7.5, weight="bold")
    ax.text(8.4, len(ROWS) + 1.72, "EngineCore proc", color=ORANGE, fontsize=7.5, weight="bold")
    ax.text(12.3, len(ROWS) + 1.72, "worker", color=TEAL, fontsize=7.5, weight="bold")
    ax.text(15.0, len(ROWS) + 1.72, "frontend", color=BLUE, fontsize=7.5, weight="bold")
    # 章行
    for r, (label, spans, c) in enumerate(ROWS):
        y = len(ROWS) - r - 0.5
        ax.text(0.42, y, label, ha="right", va="center", fontsize=6.6,
                color=c if spans else MUTED)
        if not spans:
            hint = "not on the request path" if "ch10" in label else (
                "orientation, no stations" if "ch1" in label or "ch2" in label else "")
            ax.text(9.0, y, hint, ha="center", va="center", fontsize=5.8,
                    color=MUTED, style="italic")
        for s, e in spans:
            ax.add_patch(Rectangle((s + 0.5, y - 0.30), e - s + 0.94, 0.60,
                                   facecolor=c, alpha=0.42, edgecolor=c, lw=0.7))
        ax.plot([0.5, n + 0.5], [y - 0.5, y - 0.5], color=GRIDC, lw=0.4, zorder=0)
    # 三段旅程括注（右侧）
    for y0, y1, name, c in [(8.0, 11.0, "journey I\nentry &\nprocesses", BLUE),
                            (4.5, 7.6, "journey II\nscheduling\n& execution", ORANGE),
                            (0.6, 3.6, "journey III\nbackends &\nperiphery", TEAL)]:
        ax.annotate("", xy=(n + 0.75, y0), xytext=(n + 0.75, y1),
                    arrowprops=dict(arrowstyle="-", color=c, lw=1.6))
        ax.text(n + 0.95, (y0 + y1) / 2, name, fontsize=6.2, color=c,
                va="center", linespacing=1.1)
    ax.text(9.5, -0.55,
            "one request travels left to right; two crossings of the process boundary (stn 6 out / 14 back) are the two serialization points\n"
            "ch10 = startup phase (weights/registry, before any request); ch12 = the methodology layer over every station; ch13 = the full walk again",
            ha="center", va="center", fontsize=6.0, color=MUTED, style="italic")
    ax.set_xlim(-4.6, n + 2.4)
    ax.set_ylim(-1.0, len(ROWS) + 2.0)
    ax.axis("off")
    ax.set_title("Reading map: chapters x 17 stations (coarse view; file-level detail = fig 2.2 in ch2)",
                 fontsize=8.4, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-0-1-reading-map.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print("[图 0.1] reading map")


if __name__ == "__main__":
    main()
