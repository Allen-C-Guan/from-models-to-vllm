# fig_ch00.py —— Book6 ch0 制图：图 0.1 全书旅程地图 / 图 0.2 两阶段时间线（slug=overview）
# 用途：图 0.1 = 0.4 节三段旅程（立账/造引擎/深化）+ 四版引擎进化时间线的视觉承载；
#       图 0.2 = 例 0.1 的 3+2 token 时间线（prefill 并行块 (1,3,d) vs decode 串行点 (1,1,d)——
#         标张量形状，与第 1 章图 1.1 的资源画像版互指不重复）。
# 所属章节：Book6 第 0 章（图 0.1/0.2）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/Book6-推理系统导论/ch00/fig_ch00.py"
# 产物：drafts/Book6-推理系统导论/figures/fig-0-1-journey.png / fig-0-2-two-phase.png
# 图规：300dpi 白底；系列色 蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；图内英文、图注中文。
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


def arr(ax, x0, y0, x1, y1, color=MUTED, lw=1.1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))


# ---------------- 图 0.1：旅程地图 ----------------
def fig_journey():
    fig, ax = plt.subplots(figsize=(12.4, 4.6), dpi=300)
    ax.set_xlim(0, 16.2); ax.set_ylim(0, 6.0)
    ax.axis("off")
    ax.text(8.1, 5.75, "Book6 journey: measure first (ch1-3), then build the engine (ch4-7), then deepen single items (ch8-13)",
            ha="center", fontsize=9.6, color=INK, weight="bold")
    # 三段
    box(ax, 0.4, 2.2, 4.6, 2.5, "PART 1  立账  (ch1-3)\n\nservice metrics & roofline (ch1)\narithmetic of inference (ch2)\nKV compression 3 lines (ch3)", ec=BLUE, fc="#dcebfc", fs=7.6)
    box(ax, 5.6, 2.2, 4.6, 2.5, "PART 2  造引擎  (ch4-7)\n\nv1 naive → v2 continuous batching\n→ v3 paged KV → v4 prefix cache\n→ chunked prefill & preemption", ec=ORANGE, fc="#fde3d6", fs=7.6)
    box(ax, 10.8, 2.2, 5.0, 2.5, "PART 3  单项深化  (ch8-12)\n\nattention kernel / sampling /\nspeculative decoding / quantization /\ngraph exec & parallel → ch13 finale", ec=TEAL, fc="#d9f2e7", fs=7.6)
    arr(ax, 5.0, 3.45, 5.6, 3.45, lw=1.6); arr(ax, 10.2, 3.45, 10.8, 3.45, lw=1.6)
    # 代码线：四版时间线
    labels = [("baseline\nch1", 0.8), ("v1/v2\nch4", 5.2), ("v3\nch5", 7.4), ("v4\nch6", 9.4), ("finale\ncurves ch7", 11.6)]
    ax.plot([0.9, 12.4], [1.35, 1.35], color=YELLOW, lw=1.6, zorder=2)
    for name, x in labels:
        ax.scatter([x], [1.35], color=YELLOW, s=36, zorder=4)
        ax.text(x, 0.85, name, fontsize=6.8, color=INK, ha="center")
    ax.text(13.1, 1.35, "code line:\nno-engine →\nfour engine versions", fontsize=7.0, color=MUTED, va="center")
    ax.text(8.1, 0.25, "three threads run through: the code line (yellow), the IOU line (18 old hooks in, B6-1..9 out), and one ledger (hand-calc → engine → real-engine cross-check)",
            ha="center", fontsize=7.2, color=MUTED)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-0-1-journey.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 0.1] {path}")


# ---------------- 图 0.2：两阶段时间线（例 0.1 同款） ----------------
def fig_two_phase():
    fig, ax = plt.subplots(figsize=(10.6, 4.0), dpi=300)
    ax.set_xlim(0, 13.6); ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.text(6.8, 4.95, "One request, two phases (3-token prompt + 2 generated): parallel block vs serial steps",
            ha="center", fontsize=9.4, color=INK, weight="bold")
    box(ax, 0.5, 2.3, 4.4, 1.7, "PREFILL\n(1, 3, d) one forward\nall 3 tokens in parallel\n→ token 4's distribution", ec=BLUE, fc="#dcebfc", fs=7.8)
    ax.annotate("TTFT starts", xy=(2.7, 4.0), fontsize=8, color=BLUE, ha="center", weight="bold")
    arr(ax, 4.9, 3.15, 5.6, 3.15, lw=1.6)
    box(ax, 5.6, 2.3, 2.6, 1.7, "decode step 1\ninput (1, 4, d)\nsees ALL history\n→ token 5", ec=ORANGE, fc="#fde3d6", fs=7.2)
    arr(ax, 8.2, 3.15, 8.9, 3.15, lw=1.6)
    box(ax, 8.9, 2.3, 2.6, 1.7, "decode step 2\ninput (1, 5, d)\nsees ALL history\n→ token 6", ec=ORANGE, fc="#fde3d6", fs=7.2)
    ax.text(9.6, 4.15, "serial: one token per step,\neach step drags all history", fontsize=7.4, color=ORANGE, ha="center")
    box(ax, 12.0, 2.3, 1.4, 1.7, "...", ec=MUTED, fc="white", fs=10)
    ax.text(6.8, 1.15, "without a KV cache every step recomputes the whole prefix (the 49 ms/token baseline); with one, each step only mints ONE new K/V row (ch3)",
            ha="center", fontsize=7.4, color=MUTED)
    ax.text(6.8, 0.5, "shapes: prefill works a fat tensor (1,n₀,d); decode works a thin one (1,1,d) — the whole book starts from this shape difference (ch1)",
            ha="center", fontsize=7.4, color=INK)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-0-2-two-phase.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 0.2] {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch0 制图（图 0.1/0.2）")
    ap.add_argument("--which", default="all", choices=["all", "journey", "two_phase"])
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    if args.which in ("all", "journey"):
        fig_journey()
    if args.which in ("all", "two_phase"):
        fig_two_phase()
