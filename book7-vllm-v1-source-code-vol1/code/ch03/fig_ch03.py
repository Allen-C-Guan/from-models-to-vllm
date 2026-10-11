# fig_ch03.py —— Book7 ch03 制图：图 3.1 双入口总图（离线链/在线链在 EngineCoreRequest 处汇流）
# 用途：ch03《入口层》唯一正文图——两条泳道（左=LLM 离线类，右=vllm serve 在线链）各自走到
#       InputProcessor 产出 EngineCoreRequest 消息，汇成一支箭头进 EngineCore（虚线框=ch4 地界）；
#       make_client 处画进程开关门（VLLM_ENABLE_V1_MULTIPROCESSING）；每跳标 文件::符号#L行（v0.30.0）。
# 所属章节：Book7 ch03（§3.1）。
# 运行：cd <workspace> && source env.sh && python code/Book7-vLLM源码剖析上/ch03/fig_ch03.py
# 产物：drafts/Book7-vLLM源码剖析上/figures/fig-3-1-two-entries.png
# 图规：300dpi 白底；蓝#2a78d6 橙#eb6834 青#1baf7a；图内英文图注中文；自绘+版本锚。
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "drafts", "Book7-vLLM源码剖析上", "figures"))
BLUE, ORANGE, TEAL = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=6.6, lw=1.2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw, ls=ls,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


def arr(ax, x0, y0, x1, y1, color=MUTED, lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                 mutation_scale=10, zorder=2))


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.6, 6.6), dpi=300)
    ax.set_xlim(0, 17); ax.set_ylim(0, 10.6); ax.axis("off")
    ax.text(8.5, 10.25, "Two entries, one core: offline LLM class and vllm serve converge on the EngineCoreRequest message (sketched from v0.30.0)",
            ha="center", fontsize=9.0, color=INK, weight="bold")
    # 左泳道：离线
    ax.text(0.6, 9.35, "OFFLINE: LLM class", fontsize=8.0, color=BLUE, weight="bold")
    L = [("LLM.generate\nllm.py #L418", 8.2), ("offline_utils\n_add_request #L560\nFINAL_ONLY + id", 8.2),
         ("LLMEngine\n#L48 (legacy sync)", 8.2)]
    y = 7.7
    for label, w in L:
        box(ax, 0.6, y, 3.9, 1.05, label, ec=BLUE, fc="#dcebfc")
        arr(ax, 2.55, y, 2.55, y - 0.42)
        y -= 1.47
    # 右泳道：在线
    ax.text(12.5, 9.35, "ONLINE: vllm serve", fontsize=8.0, color=ORANGE, weight="bold")
    R = [("CLI -> API server\nlaunchers/api_server/\nentry.py #L94", 8.2), ("chat route\nserving.py #L118\ntemplate here", 8.2),
         ("AsyncLLM\n#L78 (asyncio)\nadd_request #L361", 8.2)]
    y = 7.7
    for label, w in R:
        box(ax, 12.5, y, 3.9, 1.05, label, ec=ORANGE, fc="#fde3d6")
        arr(ax, 14.45, y, 14.45, y - 0.42)
        y -= 1.47
    # 汇流：InputProcessor（两链各一）
    box(ax, 4.9, 2.35, 7.2, 1.5,
        "InputProcessor  input_processor.py #L38\ntokenize / multimodal / params -> EngineCoreRequest (msgspec, engine/__init__.py #L107)\noffline: process_inputs (main thread)   online: process_inputs_async (renderer thread pool)",
        ec=TEAL, fc="#d9f2e7", fs=6.6)
    arr(ax, 2.55, 3.29, 4.9, 3.1, color=BLUE)
    arr(ax, 14.45, 3.29, 12.1, 3.1, color=ORANGE)
    # make_client 开关门
    box(ax, 4.9, 0.9, 7.2, 1.0,
        "make_client  core_client.py #L91  (four-way dispatch)\nVLLM_ENABLE_V1_MULTIPROCESSING=0 -> InprocClient (same proc) | default=1 -> Sync/AsyncMPClient (ZMQ)",
        ec=MUTED, fc="white", fs=6.4)
    arr(ax, 8.5, 2.35, 8.5, 1.9, color=TEAL)
    # EngineCore 地界
    box(ax, 6.3, -0.75, 4.4, 1.35, "EngineCore  (ch4 territory)\ncore.py #L105", ec=INK, fc="white", fs=7.0, ls="--")
    arr(ax, 7.6, 0.9, 7.6, 0.6, color=MUTED, ls="--")
    arr(ax, 9.4, 0.9, 9.4, 0.6, color=MUTED, ls="--")
    ax.text(8.5, 0.28, "ZMQ + msgpack (dashed)", ha="center", fontsize=6.0, color=MUTED, style="italic")
    ax.text(2.55, 2.0, "default: SyncMPClient\n(EngineCoreProc child)", ha="center", fontsize=6.2, color=BLUE)
    ax.text(14.45, 2.0, "always AsyncMPClient\n(ZMQ)", ha="center", fontsize=6.2, color=ORANGE)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-3-1-two-entries.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print("[图 3.1] two-entries")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    main()
