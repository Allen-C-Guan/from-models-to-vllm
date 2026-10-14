# fig_ch06.py —— Book8 ch06 制图：图 6.1 TP2 并行拓扑（双卡 4 进程框线图）
# 所属章节：Book8 ch6（§6.2）；素材底单=notes/16 §二 6.2 组织建议（规范 §4：标每卡实例-角色与卡间通道）
# 运行：cd <workspace> && source env.sh && python code/Book8-vLLM源码剖析下/ch06/fig_ch06.py
# 产物：drafts/Book8-vLLM源码剖析下/figures/fig-6-1-tp2-topology.png
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "drafts", "Book8-vLLM源码剖析下", "figures"))
BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#0b0b0b", "#52514e"


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=6.4, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


def fig_tp2():
    fig, ax = plt.subplots(figsize=(12.6, 6.2), dpi=300)
    ax.set_xlim(0, 22); ax.set_ylim(0, 10.6); ax.axis("off")
    ax.text(11, 10.25, "TP2 deployment topology: 2 cards, 4 processes (A + DP + N = 1 + 1 + 2)",
            ha="center", fontsize=9, color=INK, weight="bold")

    # 前端/API 进程（A）
    box(ax, 6.4, 8.6, 9.2, 1.25,
        "API server process (A=1)\nHTTP /v1/*  -  tokenizer  -  SSE\n(port 18xxx, 127.0.0.1)", ec=ORANGE, fc="#fde3d6", fs=6.6)
    # EngineCore（DP=1）
    box(ax, 6.4, 6.5, 9.2, 1.35,
        "EngineCore process (DP=1)\nscheduler (station 8) -> scheduler_output broadcast\nMultiprocExecutor: shared-mem MQ -> both workers", ec=BLUE, fc="#dcebfc", fs=6.4)

    # 卡框
    for i, (x0, card) in enumerate(((0.6, "NPU card 0"), (12.2, "NPU card 1"))):
        ax.add_patch(Rectangle((x0, 0.7), 9.2, 5.2, fc="#faf9f6", ec=MUTED, lw=1.0, ls="--", zorder=1))
        ax.text(x0 + 4.6, 5.55, card, ha="center", fontsize=7.5, color=MUTED)
        box(ax, x0 + 0.5, 3.0, 8.2, 2.1,
            f"VllmWorker-{i}  (Worker_TP{i} prefix)\nmodel runner: shard {i}/2 of every TP layer\nqkv/gate_up rows split; block table shared", ec=TEAL, fc="#d9f2e7", fs=6.2)
        box(ax, x0 + 0.5, 1.1, 8.2, 1.5,
            "HCCL device group member\nworld_size=2 rank=" + str(i) + f" local_rank={i} backend=hccl\nall_reduce after each TP layer (x" + "N layers)", ec=YELLOW, fc="#fdf0d2", fs=6.0)

    # 连线
    ax.annotate("", xy=(11.0, 8.6), xytext=(11.0, 7.85), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4))
    ax.annotate("", xy=(5.2, 6.5), xytext=(8.0, 6.5), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0))  # spacer
    ax.annotate("", xy=(5.2, 6.5), xytext=(7.0, 6.5), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2))
    ax.annotate("", xy=(16.8, 6.5), xytext=(15.0, 6.5), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2))
    ax.annotate("", xy=(5.2, 5.1), xytext=(7.0, 6.5), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.0, ls="--"))
    ax.annotate("", xy=(16.8, 5.1), xytext=(15.0, 6.5), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.0, ls="--"))
    # 卡间 HCCL 双向箭头
    ax.annotate("", xy=(12.2, 1.85), xytext=(9.8, 1.85), arrowprops=dict(arrowstyle="<|-|>", color="#c0392b", lw=2.0))
    ax.text(11.0, 2.25, "HCCL\nall_reduce / broadcast", ha="center", fontsize=6.4, color="#c0392b")

    ax.text(0.6, 0.25, "sketched from vLLM v0.30.0 + vllm-ascend 0.27.1rc1 (TP2 first-run evidence: 4 processes in ps, world_size lines x2, deterministic output)",
            fontsize=6.8, color=MUTED, style="italic")

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-6-1-tp2-topology.png")
    fig.savefig(out, facecolor="white"); plt.close(fig)
    print("[fig]", out)


if __name__ == "__main__":
    fig_tp2()
