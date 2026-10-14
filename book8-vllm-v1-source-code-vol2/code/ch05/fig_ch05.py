# fig_ch05.py —— Book8 ch05 制图：图 5.1 重叠时序对比（同步步进的空转 vs async 重叠）
# 所属章节：Book8 ch5（§5.1）；素材底单=notes/15 §三 例 5.1（T_cpu=3/T_npu=8）
# 运行：cd <workspace> && source env.sh && python code/Book8-vLLM源码剖析下/ch05/fig_ch05.py
# 产物：drafts/Book8-vLLM源码剖析下/figures/fig-5-1-overlap-timeline.png
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "drafts", "Book8-vLLM源码剖析下", "figures"))
BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GAP = "#0b0b0b", "#52514e", "#f0efec"


def lane(ax, x, y, w, h, label, color):
    ax.add_patch(Rectangle((x, y), w, h, fc=color, ec="none", alpha=0.85, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7.5,
            color="white" if color != GAP else MUTED, zorder=4)


def fig_overlap():
    fig, ax = plt.subplots(figsize=(11.6, 5.2), dpi=300)
    ax.set_xlim(0, 24); ax.set_ylim(0, 10.5); ax.axis("off")
    ax.text(12, 10.1, "Sync vs async step timeline: the overlap gap and who hides in whose shadow",
            ha="center", fontsize=9, color=INK, weight="bold")

    T_CPU, T_NPU = 3, 8
    # 同步：两步周期 11ms（NPU 空 3ms）
    ax.text(0.4, 9.0, "sync:  T_step = T_cpu + T_npu = 11 ms   (NPU idles 3 ms every step)",
            fontsize=8, color=INK)
    y = 7.2
    x = 0.6
    for step in range(2):
        lane(ax, x, y + 0.9, T_CPU, 0.85, "cpu", ORANGE)
        lane(ax, x + T_CPU, y + 0.9, T_NPU, 0.85, "npu", BLUE)
        ax.annotate("", xy=(x + T_CPU + T_NPU + 0.4, y + 1.3), xytext=(x, y + 1.3),
                    arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1))
        ax.text(x + (T_CPU + T_NPU) / 2, y + 1.75, f"step {step}: 11 ms", ha="center",
                fontsize=7, color=MUTED)
        x += T_CPU + T_NPU + 2.2
    ax.text(0.6 + T_CPU + 11 + 2.2 + 11, 0, "")  # noop spacer
    # 异步：一步 8ms（cpu 藏进 npu 影子）
    ax.text(0.4, 5.6, "async: T_step = max(T_cpu, T_npu) = 8 ms   (cpu of step N+1 hides under npu of step N)",
            fontsize=8, color=INK)
    y = 3.2
    x = 0.6
    for step in range(3):
        lane(ax, x, y + 0.9, T_NPU, 0.85, f"npu {step}", BLUE)
        lane(ax, x + 1, y, T_CPU, 0.7, f"cpu {step + 1}", ORANGE)
        x += T_NPU + 1.2
    ax.annotate("", xy=(0.6 + T_NPU + 1.2 + T_NPU + 0.3, 5.05), xytext=(0.6, 5.05),
                arrowprops=dict(arrowstyle="<->", color=TEAL, lw=1))
    ax.text(0.6 + T_NPU, 5.3, "step period 8 ms  (upper bound 11/8 = 1.375x)", ha="center",
            fontsize=7, color=TEAL)

    ax.text(0.4, 1.6, "reverse branch: if T_cpu = 12 > T_npu = 8, async step stays 12 ms - overlap rescues the faster side, never the bottleneck side",
            fontsize=7.5, color=MUTED)
    ax.text(0.4, 0.7, "sketched from vLLM v0.30.0 async scheduling design; numbers = ex.5.1 toy (ms)",
            fontsize=7, color=MUTED, style="italic")

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-5-1-overlap-timeline.png")
    fig.savefig(out, facecolor="white"); plt.close(fig)
    print("[fig]", out)


if __name__ == "__main__":
    fig_overlap()
