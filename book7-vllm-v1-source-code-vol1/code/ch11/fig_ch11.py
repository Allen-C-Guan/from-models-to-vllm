# fig_ch11.py —— Book7 ch11 制图：图 11.1 输出处理链（token 的返程）
# 用途：ch11 唯一正文图——从 EngineCoreOutput 信封到 SSE chunk 的完整返程。
# 所属章节：Book7 ch11（§11.2/11.5）。运行：source env.sh && python code/Book7-vLLM源码剖析上/ch11/fig_ch11.py
# 产物：drafts/Book7-vLLM源码剖析上/figures/fig-11-1-output-chain.png
# 图规：300dpi 白底；蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；图内英文图注中文。
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "drafts", "Book7-vLLM源码剖析上", "figures"))
BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#0b0b0b", "#52514e"

def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=6.2, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=INK, zorder=4)

def arr(ax, x0, y0, x1, y1, color=MUTED, lw=1.1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", color=color, lw=lw,
                                 mutation_scale=10, zorder=2))

def main() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 6.6), dpi=300)
    ax.set_xlim(0, 18); ax.set_ylim(0, 10.2); ax.axis("off")
    ax.text(9, 9.85, "The return trip: from EngineCoreOutputs envelope to SSE chunks (v0.30.0; sketched from source, X = toy Example 11.1)",
            ha="center", fontsize=8.6, color=INK, weight="bold")
    box(ax, 0.5, 6.9, 4.0, 2.2,
        "EngineCore proc (st.13-14)\n\nSamplerOutput ->\nEngineCoreOutput{request_id,\nnew_token_ids=[X], finish}\n-> EngineCoreOutputs envelope",
        ec=ORANGE, fc="#fde3d6", fs=6.0)
    box(ax, 5.2, 7.5, 2.6, 1.0, "ZMQ PUSH->PULL\n(msgpack frames)", ec=MUTED, fc="white", fs=6.0)
    arr(ax, 4.5, 8.0, 5.2, 8.0)
    box(ax, 8.4, 6.4, 4.4, 3.0,
        "frontend proc (st.15)\n\noutput_handler (chunked loop\n  CHUNK_SIZE=128, sleep(0))\n-> process_outputs #L629:\n  stats | detokenize+stop\n  | logprobs | make_output",
        ec=BLUE, fc="#dcebfc", fs=5.9)
    arr(ax, 7.8, 8.0, 8.4, 8.0)
    box(ax, 13.6, 7.0, 4.0, 1.7,
        "RequestOutputCollector\n(per-request mailbox) #L51\nDELTA merge if producer\ngets ahead (aggregate=True)",
        ec=TEAL, fc="#d9f2e7", fs=6.0)
    arr(ax, 12.8, 7.85, 13.6, 7.85)
    box(ax, 8.4, 4.4, 4.4, 1.6,
        "IncrementalDetokenizer (st.16)\nFast: DecodeStream(ids=prompt) #L184\nSlow: prefix/read dual cursor #L263\nstop buffer: max(len(s))-1 #L87",
        ec=YELLOW, fc="#fdf0d2", fs=5.9)
    arr(ax, 10.6, 6.4, 10.6, 6.0, color=YELLOW)
    box(ax, 13.6, 4.4, 4.0, 2.2,
        "generate() consumer #L705\nq.get_nowait() or await q.get()\n-> SSE stream 5 event kinds:\nrole | delta | finish | usage\n| data: [DONE]",
        ec=BLUE, fc="#dcebfc", fs=5.9)
    arr(ax, 15.6, 7.0, 15.6, 6.6)
    ax.text(9, 3.2, "each step names the data structure; process boundary at ZMQ (orange|blue);\n"
                    "offline path: no collector - process_outputs appends to a list (branch at #L726-731)",
            ha="center", fontsize=6.2, color=MUTED, style="italic")
    ax.text(9, 2.2, "observed live: 12-token request = 15 SSE events (1 role + 12 delta + 1 usage + 1 DONE) - log/book7-ch11/sse_stream_raw.txt",
            ha="center", fontsize=6.4, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-11-1-output-chain.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print("[图 11.1] output-chain")

if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    main()
