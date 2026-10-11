# fig_ch04.py —— Book7 ch04 制图：图 4.1 进程拓扑（ZMQ socket 逐跳）/ 图 4.2 启动时序
# 用途：ch04《EngineCore 进程模型》两图——图 4.1=三列进程框+四条 ZMQ 通道（ROUTER↔DEALER 输入/PUSH→PULL
#       输出/握手/核→Worker 广播，uniproc 时 Worker 虚线=住在 EngineCore 内）；图 4.2=启动时序三泳道八段
#       （数据源=冒烟日志时间戳：0→55.1 s 全程，eager 模式图捕获为空条带）。
# 所属章节：Book7 ch04（§4.2/§4.5）。
# 运行：cd <workspace> && source env.sh && python code/Book7-vLLM源码剖析上/ch04/fig_ch04.py
# 产物：drafts/Book7-vLLM源码剖析上/figures/fig-4-1-process-topology.png / fig-4-2-startup-timeline.png
# 图规：300dpi 白底；蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；图内英文图注中文；自绘+双轨标注。
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "drafts", "Book7-vLLM源码剖析上", "figures"))
BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e1e0d9"


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=6.8, lw=1.3, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw, ls=ls,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


def arr(ax, x0, y0, x1, y1, color=MUTED, lw=1.2, ls="-", style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, color=color, lw=lw, ls=ls,
                                 mutation_scale=11, zorder=2))


# ---------------- 图 4.1：进程拓扑 ----------------
def fig_topology():
    fig, ax = plt.subplots(figsize=(12.4, 6.6), dpi=300)
    ax.set_xlim(0, 17); ax.set_ylim(0, 10.2); ax.axis("off")
    ax.text(8.5, 9.85, "Process topology and ZMQ sockets (single-card serve: A + DP + N = 3 roles, 2 physical procs — worker absorbed by uniproc)",
            ha="center", fontsize=8.8, color=INK, weight="bold")
    # 三列
    box(ax, 0.7, 3.0, 4.0, 5.4,
        "API Server  (frontend proc)\n\n- tokenize / multimodal\n- output_handler\n- metrics, admission*\n- ZMQ: bind ROUTER / bind PULL",
        ec=BLUE, fc="#dcebfc", fs=7.2)
    box(ax, 6.4, 3.0, 4.2, 5.4,
        "EngineCore  (engine proc)\n\nbusy loop  core.py #L1403\ninput_queue -> schedule() ->\nexecute_model(non_block) ->\nupdate_from_output\nZMQ: connect DEALER / PUSH\n(input & output threads #L1113)",
        ec=ORANGE, fc="#fde3d6", fs=7.2)
    box(ax, 12.2, 3.0, 4.2, 5.4,
        "Worker  (per card)\n\n- NPUWorker worker.py #L114\n- model runner\n- KV cache tensors live here\nuniproc (ws==1): lives INSIDE\nEngineCore proc (parallel.py\n#L1010-1011)",
        ec=TEAL, fc="#d9f2e7", fs=7.2)
    # 通道
    arr(ax, 4.7, 7.35, 6.4, 7.35, color=BLUE, lw=1.6)
    ax.text(5.55, 7.62, "inputs: ROUTER <-> DEALER\n(identity=engine idx)", ha="center", fontsize=6.6, color=BLUE)
    arr(ax, 6.4, 5.7, 4.7, 5.7, color=ORANGE, lw=1.6)
    ax.text(5.55, 5.97, "outputs: PUSH -> PULL\n(linger=4000: dead msg survives)", ha="center", fontsize=6.6, color=ORANGE)
    arr(ax, 4.7, 4.35, 6.4, 4.35, color=MUTED, lw=1.1, ls="--", style="<|-|>")
    ax.text(5.55, 4.62, "handshake: HELLO / INIT / READY", ha="center", fontsize=6.4, color=MUTED)
    arr(ax, 10.6, 5.7, 12.2, 5.7, color=TEAL, lw=1.6)
    ax.text(11.4, 5.97, "rpc_broadcast_mq\n(multiproc) or direct\ncall (uniproc)", ha="center", fontsize=6.6, color=TEAL)
    ax.text(8.5, 2.35, "formula: Total = A + DP + N (+1 DP coordinator if DP>1), N = DP x PP x TP  — counts ROLES; ps counts PROCS\n"
                       "example: tp4 => 1+1+4 = 6 procs;  tp2 dp4 (8 cards) => 4+4+8+1 = 17 procs   [arch_overview.md #L107-113]",
            ha="center", fontsize=7.0, color=INK)
    ax.text(2.7, 2.72, "* admission control: v0.30.0 only — runtime 0.27.1 has no such layer", fontsize=6.2, color=MUTED, style="italic")
    ax.text(8.5, 0.35, "sketched from v0.30.0 (ced6857); socket pairs per core.py / core_client.py anchors in text; runtime observed on 0.27.1 + vllm-ascend",
            ha="center", fontsize=6.2, color=MUTED, style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-4-1-process-topology.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print("[图 4.1] topology")


# ---------------- 图 4.2：启动时序 ----------------
def fig_startup():
    fig, ax = plt.subplots(figsize=(12.4, 5.6), dpi=300)
    ax.set_xlim(0, 62); ax.set_ylim(0, 9.4)
    ax.axis("off")
    ax.text(31, 9.05, "Startup timeline of a cold boot (207M, enforce_eager, single card; wall-clock from smoke log, total 55.1 s)",
            ha="center", fontsize=8.8, color=INK, weight="bold")
    lanes = [("frontend proc", 6.2, BLUE), ("EngineCore proc", 3.6, ORANGE), ("worker side", 1.0, TEAL)]
    for name, y, c in lanes:
        ax.text(0.4, y + 1.15, name, fontsize=7.4, color=c, weight="bold")
        ax.plot([3.2, 60], [y + 0.75, y + 0.75], color=GRIDC, lw=0.8, zorder=1)
    SEGS = [
        ("import & plugin load\n(both procs)", 0, 36, BLUE, 6.2),
        ("spawn EngineCore\n+ handshake", 30, 12, MUTED, 4.9),
        ("load weights\n0.39 GB", 37, 5, TEAL, 1.0),
        ("profile_run\n(determine mem)", 42, 7, ORANGE, 3.6),
        ("init KV cache\n8110 blocks", 49, 3, TEAL, 1.0),
        ("init engine total 9.7 s", 37, 15, ORANGE, 2.3),
        ("graph capture: SKIPPED (enforce_eager; ch8 covers non-eager)", 52, 3, MUTED, 3.6),
        ("READY (55.1 s)", 55, 1.5, INK, 6.2),
    ]
    for label, x0, w, c, y in SEGS:
        h = 0.85 if y != 2.3 and y != 4.9 else 0.6
        ax.add_patch(Rectangle((x0, y), w, h, fc=c, ec="none", alpha=0.35, zorder=2))
        ax.text(x0 + w / 2, y + h / 2, label, ha="center", va="center", fontsize=5.9, color=INK, zorder=3)
    ax.set_xlim(0, 62)
    ax.text(31, 0.25, "seconds (approx, from log timestamps of log/book7-00/vllm_smoke_eager1.log; NPU card, runtime 0.27.1); with StartPlan a matching fingerprint skips profile_run (exp ch04)",
            ha="center", fontsize=6.2, color=MUTED, style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-4-2-startup-timeline.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print("[图 4.2] startup-timeline")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_topology()
    fig_startup()
