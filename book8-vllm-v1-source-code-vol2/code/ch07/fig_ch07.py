# fig_ch07.py —— Book8 ch07 制图：图 7.1 PD 拓扑与时序（P/D 双实例+proxy+KV 块流向+握手消息）
# 所属章节：Book8 ch7（§7.2）；素材底单=notes/17 §二 7.2 图规格（nixl_kv_push_connector.md 时序为蓝本）
# 运行：cd <workspace> && source env.sh && python code/Book8-vLLM源码剖析下/ch07/fig_ch07.py
# 产物：drafts/Book8-vLLM源码剖析下/figures/fig-7-1-pd-topology.png
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "drafts", "Book8-vLLM源码剖析下", "figures"))
BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#0b0b0b", "#52514e"


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=6.2, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


def arr(ax, x0, y0, x1, y1, color=MUTED, lw=1.2, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                 mutation_scale=10, zorder=2))


def fig_pd():
    fig, ax = plt.subplots(figsize=(13.0, 7.0), dpi=300)
    ax.set_xlim(0, 20); ax.set_ylim(0, 11.6); ax.axis("off")
    ax.text(10, 11.25, "PD disaggregation topology and handoff sequence (sketched from vLLM v0.30.0 + vllm-ascend kv_pool.md)",
            ha="center", fontsize=8.8, color=INK, weight="bold")

    # proxy
    box(ax, 6.6, 9.5, 6.8, 1.15,
        "router / proxy (station 1-3)\ninput router: prefill leg -> P, decode leg -> D\n(load_balance_proxy_server_example.py)", ec=ORANGE, fc="#fde3d6", fs=6.2)

    # P instance
    box(ax, 0.4, 5.6, 6.2, 3.2,
        "P instance (kv_role=kv_producer)\nscheduler + SCHEDULER-role connector\nworker + WORKER-role connector\nruns prefill only\n\nsave_kv_layer per layer -> staging buffer\n(P-side KV written during forward)", ec=BLUE, fc="#dcebfc", fs=6.0)
    # D instance
    box(ax, 13.4, 5.6, 6.2, 3.2,
        "D instance (kv_role=kv_consumer)\nscheduler + SCHEDULER-role connector\nworker + WORKER-role connector\nruns decode only\n\nget_num_new_matched_tokens -> alloc\nwait_for_layer_load per layer (28x)", ec=TEAL, fc="#d9f2e7", fs=6.0)

    # KV transfer band
    box(ax, 3.2, 2.5, 13.6, 1.9,
        "KV transfer band = connector (one class, two roles)\nNIXL pull: D issues READ after prefill completes | push: P writes into D's pre-allocated memory (WRITE)\nmooncake / ascend_store / offloading-family: the same contract, different transports\nblock flow: 128-token blocks, e.g. Qwen3-0.6B bf16 = 14 MiB per block", ec=YELLOW, fc="#fdf0d2", fs=6.0)

    # handshake messages
    arr(ax, 7.2, 9.5, 3.5, 8.8, color=ORANGE); ax.text(4.6, 9.35, "prefill leg", fontsize=6.4, color=ORANGE)
    arr(ax, 12.8, 9.5, 16.5, 8.8, color=ORANGE); ax.text(13.9, 9.35, "decode leg", fontsize=6.4, color=ORANGE)
    arr(ax, 6.6, 6.6, 13.4, 6.6, color="#c0392b", lw=2.0)
    ax.text(10.0, 6.85, "D registers blocks (block hashes + ids)  ->  P stages KV  ->  transfer (pull READ / push WRITE)",
            ha="center", fontsize=6.6, color="#c0392b")
    arr(ax, 3.5, 5.6, 4.6, 4.4, color=MUTED, ls="--")
    arr(ax, 16.5, 5.6, 15.4, 4.4, color=MUTED, ls="--")
    arr(ax, 10.0, 4.4, 10.0, 2.5, color=YELLOW)

    # timeline strip
    box(ax, 0.4, 0.4, 19.2, 1.5,
        "handoff account: D's get_num_new_matched_tokens says how many blocks hit externally | update_state_after_alloc maps them to local block ids |\n"
        "build_connector_meta carries {req, remote ids, local ids} across processes | per-layer wait gates the decode forward on layer-by-layer readiness\n"
        "failure policy: recompute (roll back to last legal prefix) vs fail (default) - a lost save is a future miss, a wrong handoff is dirty data",
        ec=INK, fs=6.2)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-7-1-pd-topology.png")
    fig.savefig(out, facecolor="white"); plt.close(fig)
    print("[fig]", out)


if __name__ == "__main__":
    fig_pd()
