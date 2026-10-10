# fig_ch12.py —— Book6 ch12 制图：图 12.1 图执行机制 / 图 12.2 TP 两次 all-reduce / 图 12.3 PD 分离（slug=graph-parallel）
# 用途：图 12.1 = 12.2-12.3 节「录一次放一次」机制图（eager 逐算子发射 vs capture/replay 一次提交，
#         标三静态约束与本机 NPUGraph 11.4× 读数锚）；
#       图 12.2 = 12.6 节 TP 一层的通信时序（列切 QKV→头本地注意力→行切 o_proj→AR①→列切 FFN→行切 down→AR②
#         ——两次 all-reduce 都在部分和处，注意力在头切分下本地完成）；
#       图 12.3 = 12.7 节 PD 分离部署形态（prefill 组/decode 组两组流转+KV 传输+SLO 分治标注）。
# 所属章节：Book6 第 12 章（图 12.1/12.2/12.3）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/Book6-推理系统导论/ch12/fig_ch12.py"
# 产物：drafts/Book6-推理系统导论/figures/fig-12-{1,2,3}-graph-parallel.png
# 图规：300dpi 白底；系列色 蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；线宽 2；图内英文、图注中文。
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


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=7.2, lw=1.3, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc, zorder=4)


def arr(ax, x0, y0, x1, y1, color=MUTED, lw=1.1, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle=style, color=color, lw=lw))


# ---------------- 图 12.1：capture/replay 机制 ----------------
def fig_graph_exec():
    fig, ax = plt.subplots(figsize=(11.8, 4.6), dpi=300)
    ax.set_xlim(0, 15.4); ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.text(7.7, 5.95, "Graph execution: record the kernel chain once, then replay it with ONE submission (static shapes pay the rent)",
            ha="center", fontsize=9.4, color=INK, weight="bold")
    # eager 行
    box(ax, 0.4, 3.6, 1.7, 1.1, "eager:\n200 launches\nper step", ec=ORANGE, fc="#fde3d6", fs=7.2)
    for i in range(6):
        box(ax, 2.5 + i * 1.05, 3.7, 0.75, 0.9, f"k{i+1}", ec=MUTED, fc="white", fs=7.2)
        arr(ax, 2.25 + i * 1.05, 4.15, 2.5 + i * 1.05, 4.15, color=ORANGE, lw=0.9)
    ax.text(5.6, 3.25, "each kernel pays ~11 µs launch tax  →  0.34 ms for an 8-op chain (this run)",
            fontsize=7.2, color=ORANGE, ha="center")
    # capture 行
    box(ax, 0.4, 1.3, 1.7, 1.1, "capture:\nrun once,\nrecord the chain", ec=BLUE, fc="#dcebfc", fs=7.2)
    ax.add_patch(FancyBboxPatch((2.5, 1.4), 6.3, 0.9, boxstyle="round,pad=0.02", ec=BLUE, fc="#dcebfc", lw=1.6, zorder=3))
    ax.text(5.65, 1.85, "GRAPH (one object: k1→k2→…→k8, static buffers)", ha="center", fontsize=7.6, color=BLUE, zorder=4)
    arr(ax, 2.1, 1.85, 2.5, 1.85, color=BLUE)
    box(ax, 9.4, 1.3, 2.5, 1.1, "replay:\nONE submission\n0.030 ms (11.4×)", ec=TEAL, fc="#d9f2e7", fs=7.4)
    arr(ax, 8.8, 1.85, 9.4, 1.85, color=TEAL, lw=1.4)
    box(ax, 12.4, 1.0, 2.7, 1.7, "three statics:\nshape / buffer addr\n/ control flow\n(re-record on change)", ec=MUTED, fc="#f2f1ec", fs=7.0)
    arr(ax, 11.9, 1.85, 12.4, 1.85)
    ax.text(7.7, 0.45, "decode fits: every step is the same (B,1,d) chain — bucketing handles the B axis (vLLM: 1,2,4; ×8…; ×16…)",
            ha="center", fontsize=7.2, color=MUTED)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-12-1-graph-parallel.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 12.1] {path}（读数锚 launch_probe_base.json graph_vs_eager）")


# ---------------- 图 12.2：TP 一层两次 all-reduce ----------------
def fig_tp_comm():
    fig, ax = plt.subplots(figsize=(12.2, 4.4), dpi=300)
    ax.set_xlim(0, 15.8); ax.set_ylim(0, 5.8)
    ax.axis("off")
    ax.text(7.9, 5.55, "TP within one layer (N cards, head-sharded): attention is LOCAL — the two all-reduces sit at partial-sum joints",
            ha="center", fontsize=9.2, color=INK, weight="bold")
    # 主体流程
    box(ax, 0.3, 2.2, 2.1, 1.3, "x (B,n,d)\ncolumn-shard\nQKV per card", ec=BLUE, fc="#dcebfc", fs=7.0)
    box(ax, 3.0, 2.2, 2.6, 1.3, "attention on\nown heads (LOCAL —\nhead i's K/V all local)", ec=TEAL, fc="#d9f2e7", fs=7.0)
    box(ax, 6.2, 2.2, 2.3, 1.3, "W_o row-shard:\npartial outputs\n(B,n,d/N) each", ec=BLUE, fc="#dcebfc", fs=7.0)
    box(ax, 9.1, 2.2, 1.9, 1.3, "all-reduce ①\n(sum partials)\n≈ B·n·d elems", ec=ORANGE, fc="#fde3d6", fs=7.0)
    box(ax, 11.5, 2.2, 1.9, 1.3, "FFN up/gate\ncol-shard →\ndown row-shard", ec=BLUE, fc="#dcebfc", fs=7.0)
    box(ax, 13.9, 2.2, 1.7, 1.3, "all-reduce ②\n(B,n,d)", ec=ORANGE, fc="#fde3d6", fs=7.0)
    for x0 in (2.4, 5.6, 8.5, 11.0, 13.4):
        arr(ax, x0, 2.85, x0 + 0.6, 2.85)
    ax.text(7.9, 1.55, "per layer × 2 all-reduces; per step × L layers — the latency of synchronization (not just bytes) is the small-model decode killer",
            ha="center", fontsize=7.4, color=INK)
    ax.text(7.9, 0.85, "例 12.1：B=8, n=1, d=1024, bf16 → 每次 16 KiB，一层两次 32 KiB，12 层 384 KiB/步（数据小、同步多）",
            ha="center", fontsize=7.2, color=MUTED)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-12-2-graph-parallel.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 12.2] {path}")


# ---------------- 图 12.3：PD 分离 ----------------
def fig_pd():
    fig, ax = plt.subplots(figsize=(11.8, 4.4), dpi=300)
    ax.set_xlim(0, 15.2); ax.set_ylim(0, 5.8)
    ax.axis("off")
    ax.text(7.6, 5.55, "PD disaggregation: two pools, two SLOs — prefill chews compute, decode sips bandwidth, KV rides between",
            ha="center", fontsize=9.2, color=INK, weight="bold")
    box(ax, 0.4, 2.0, 3.4, 2.3, "PREFILL pool\n(compute-strong cards)\n\nSLO: TTFT\nbig matmuls — MFU-friendly", ec=BLUE, fc="#dcebfc", fs=7.4)
    box(ax, 11.4, 2.0, 3.4, 2.3, "DECODE pool\n(bandwidth-rich cards)\n\nSLO: TPOT\nmemory-bound — batch deep", ec=TEAL, fc="#d9f2e7", fs=7.4)
    box(ax, 5.6, 2.5, 3.9, 1.3, "KV transfer\n(512-token req on OPT-66B\n≈ 1.13 GB; 10 rps → 11 GB/s)", ec=ORANGE, fc="#fde3d6", fs=7.0)
    arr(ax, 3.8, 3.15, 5.6, 3.15, color=ORANGE, lw=1.4)
    arr(ax, 9.5, 3.15, 11.4, 3.15, color=ORANGE, lw=1.4)
    arr(ax, 10.4, 2.4, 4.6, 2.4, color=MUTED, lw=1.0, style="-|>")
    ax.text(7.5, 1.95, "responses stream back to users", fontsize=6.8, color=MUTED, ha="center")
    ax.text(7.6, 1.0, "colocation 的三笔代价：两段互相干扰 / 资源配置耦合 / 调度目标打架——分池逐一消除（DistServe, OSDI'24）",
            ha="center", fontsize=7.4, color=INK)
    ax.text(7.6, 0.45, "收益读数：双 SLO 下 7.4× 更多请求 或 12.6× 更紧 SLO（>90% 守约）；工程生态（connector）→ Book8",
            ha="center", fontsize=7.2, color=MUTED)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-12-3-graph-parallel.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 12.3] {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch12 制图（图 12.1/12.2/12.3）")
    ap.add_argument("--which", default="all", choices=["all", "graph_exec", "tp_comm", "pd"])
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    if args.which in ("all", "graph_exec"):
        fig_graph_exec()
    if args.which in ("all", "tp_comm"):
        fig_tp_comm()
    if args.which in ("all", "pd"):
        fig_pd()
