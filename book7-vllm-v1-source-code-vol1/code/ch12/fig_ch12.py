# fig_ch12.py —— Book7 ch12 制图：图 12.1/12.2 双轴参数扫描 / 图 12.3 端到端时序示例
# 用途：图 12.1=max-num-batched-tokens 三档（log/book7-ch12/lat_mnbt*.json 实测直读）；
#       图 12.2=max-num-seqs 三档（lat_mns*.json 实测直读）；图 12.3=17 站三层观测时序拼装示例。
# 纪律：曲线数据一律从 log/ JSON 直读（不手抄数字——实物制度）；底档缺失即报错退出。
# 所属章节：Book7 ch12（§12.6/12.7）。运行：python code/Book7-vLLM源码剖析上/ch12/fig_ch12.py
# 产物：drafts/Book7-vLLM源码剖析上/figures/fig-12-1-sweep.png / fig-12-2-mns.png / fig-12-3-timeline.png
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIG_DIR = os.path.join(WS, "drafts", "Book7-vLLM源码剖析上", "figures")
LOG = os.path.join(WS, "log", "book7-ch12")
BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e1e0d9"


def _load(axis, points):
    xs, ys = [], []
    for p in points:
        f = os.path.join(LOG, f"lat_{axis}{p}.json")
        with open(f, encoding="utf-8") as fh:
            xs.append(p)
            ys.append(round(json.load(fh)["avg_latency"], 4))
    return xs, ys


def fig_sweep():
    xs, ys = _load("mnbt", [2048, 4096, 8192])
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=300)
    ax.plot(xs, ys, "o-", color=BLUE, lw=2, ms=7, zorder=3)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}s", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("max-num-batched-tokens  (batch=16, in=512, out=64, eager)", fontsize=9, color=MUTED)
    ax.set_ylabel("avg batch latency (s)", fontsize=9, color=MUTED)
    ax.grid(color=GRIDC, lw=0.5, zorder=0)
    ax.tick_params(colors="#898781", labelsize=8)
    ax.set_title("Token-budget sweep: 16x512=8192 tok fits one step at 8192 (2048 needs 4 chunked steps)",
                 fontsize=8.2, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-12-1-sweep.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 12.1] mnbt sweep {list(zip(xs, ys))}")


def fig_mns():
    xs, ys = _load("mns", [8, 16, 32])
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=300)
    ax.plot(xs, ys, "s-", color=ORANGE, lw=2, ms=7, zorder=3)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}s", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("max-num-seqs  (batch=16 requested, in=512, out=64, eager)", fontsize=9, color=MUTED)
    ax.set_ylabel("avg batch latency (s)", fontsize=9, color=MUTED)
    ax.grid(color=GRIDC, lw=0.5, zorder=0)
    ax.tick_params(colors="#898781", labelsize=8)
    ax.set_title("Sequence-budget sweep: latency changes only while the cap binds the batch",
                 fontsize=8.2, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-12-2-mns.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 12.2] mns sweep {list(zip(xs, ys))}")


def fig_timeline():
    fig, ax = plt.subplots(figsize=(13.0, 6.0), dpi=300)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9.2)
    ax.axis("off")
    ax.text(9, 8.85, "End-to-end timeline of one request: three observation layers, each with a blind spot (sketch; assignment asks you to assemble it)",
            ha="center", fontsize=8.4, color=INK, weight="bold")
    lanes = [("client (bench --plot-timeline)", 6.6, BLUE, ["send t0", "first byte", "last byte"]),
             ("frontend proc (arrival_time, SSE chunks)", 4.6, ORANGE, ["arrival", "chunk x N", "[DONE]"]),
             ("EngineCore proc (event timestamps)", 2.6, TEAL, ["QUEUED", "SCHEDULED", "NEW_TOKENS x N (batch-level single ts)"])]
    for name, y, c, marks in lanes:
        ax.text(0.4, y + 1.05, name, fontsize=7.6, color=c, weight="bold")
        ax.plot([3.4, 17.2], [y + 0.62, y + 0.62], color=GRIDC, lw=0.8, zorder=1)
        for i, m in enumerate(marks):
            x = 4.6 + i * 4.2
            ax.plot([x], [y + 0.62], "o", color=c, ms=6, zorder=3)
            ax.text(x, y + 0.28, m, fontsize=6.2, color=c, ha="center")
    for pair in [(7.0, 5.0), (7.0, 3.0), (5.0, 3.0)]:
        ax.annotate("", xy=(pair[1], 4.0), xytext=(pair[0], 4.0),
                    arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.9, ls="--"))
    ax.text(9, 1.2, "blind spots: client never sees the queue; engine core never sees HTTP; inproc hooks remove the bridge -\n"
                    "checkpoints: monotonic clocks are not comparable across processes; NEW_TOKENS is one ts per batch",
            ha="center", fontsize=6.4, color=MUTED, style="italic")
    ax.text(9, 0.35, "station 13/14 land in ch11/ch04 respectively (see those chapters); full 17-station map = ch02 fig 2.2",
            ha="center", fontsize=6.4, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-12-3-timeline.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print("[图 12.3] timeline")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_sweep()
    fig_mns()
    fig_timeline()
