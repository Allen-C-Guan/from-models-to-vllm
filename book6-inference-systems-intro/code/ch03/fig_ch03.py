# fig_ch03.py —— Book6 ch3 制图：图 3.1 三线路线图 / 图 3.2 KIVI 非对称分组 / 图 3.3 淘汰时机时间线（slug=kv-3lines）
# 用途：图 3.1 = 3.1 节末路线图（一份 KV 账单+三把刀各动哪一维；兼承三类生命周期同池小面板）；
#       图 3.2 = 3.4 节 KIVI 非对称量化（K 按 d_k 通道列分组 (h,n,d_k)→每列一 scale；V 按 token 行分组
#         (h,n,d_k)→每行一 scale——形状标注+离群结构对齐）；
#       图 3.3 = 3.5 节 H2O/SnapKV 时机对比（prefill/decode 时间轴+TTFT/TPOT+各家付账时点）。
# 所属章节：Book6 第 3 章（图 3.1/3.2/3.3）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/Book6-推理系统导论/ch03/fig_ch03.py"
# 产物：drafts/Book6-推理系统导论/figures/fig-3-{1,2,3}-kv-3lines.png
# 图规：300dpi 白底；系列色 蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；图内英文、图注中文。
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

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


# ---------------- 图 3.1：三线路线图 ----------------
def fig_routes():
    fig, ax = plt.subplots(figsize=(12.2, 4.8), dpi=300)
    ax.set_xlim(0, 15.8); ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.text(7.9, 5.95, "One KV bill, three knives: design it small (architecture) / store it low-bit (quantization) / drop it (eviction)",
            ha="center", fontsize=9.5, color=INK, weight="bold")
    box(ax, 5.5, 4.55, 4.8, 1.15, "KV bill: 2·L·h_kv·d_k elems/token\nrent (per request) · caps concurrency · read every step", ec=MUTED, fc="#f2f1ec", fs=7.4)
    arr(ax, 6.4, 4.5, 2.6, 3.75); arr(ax, 7.9, 4.5, 7.9, 3.75); arr(ax, 9.5, 4.5, 13.3, 3.75)
    box(ax, 0.3, 1.7, 4.6, 1.95, "LINE 1  architecture (ch3.3)\ncut h_kv: MHA→MQA→GQA 12,288\nlow-rank: MLA 3,840 (−68.75%)\nno KV at all: hybrid layers (fixed state)\nPAY at training time", ec=BLUE, fc="#dcebfc", fs=7.0)
    box(ax, 5.5, 1.7, 4.6, 1.95, "LINE 2  quantization (ch3.4)\nK per-channel / V per-token (KIVI 2bit)\nnon-uniform + offline calib (KVQuant)\nPAY at deploy time + dequant op", ec=ORANGE, fc="#fde3d6", fs=7.0)
    box(ax, 10.7, 1.7, 4.8, 1.95, "LINE 3  eviction (ch3.5)\nH2O: online bookkeeping\ncum-score, heavy hitters\nSnapKV: prefill-time one-shot pick\nPAY at serving time", ec=TEAL, fc="#d9f2e7", fs=7.0)
    # 生命周期小面板
    box(ax, 0.3, 0.25, 15.2, 1.05, "same pool, three lifetimes (hybrid models, Book5 ch1):  global-layer KV grows unboundedly · sliding-window KV bounded by W · linear-layer state fixed — one paged pool must host all three (ch5)", ec=YELLOW, fc="#fbf3dc", fs=7.2)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-3-1-kv-3lines.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 3.1] {path}")


# ---------------- 图 3.2：KIVI 非对称分组 ----------------
def fig_kivi_axes():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), dpi=300)
    rng = np.random.RandomState(7)
    for ax, name, mode in ((axes[0], "K: outliers pinned to CHANNELS", "K"), (axes[1], "V: outliers scattered on TOKENS", "V")):
        M = rng.randn(12, 8) * 0.4
        if mode == "K":
            M[:, 2] *= 60                       # 离群通道（列）
            M[:, 5] *= 40
        else:
            M[3, :] *= 55                       # 离群 token（行）
            M[9, :] *= 45
        im = ax.imshow(M, aspect="auto", cmap="Blues" if mode == "K" else "Oranges")
        ax.set_xticks(range(8)); ax.set_yticks(range(12))
        ax.set_xticklabels([f"c{j}" for j in range(8)], fontsize=6.5)
        ax.set_yticklabels([f"t{i}" for i in range(12)], fontsize=6.5)
        ax.set_xlabel("channel (d_k) →  one scale per COLUMN", fontsize=7.6, color=BLUE if mode == "K" else MUTED)
        ax.set_ylabel("token (n) →  one scale per ROW", fontsize=7.6, color=ORANGE if mode == "V" else MUTED)
        ax.set_title(f"{name}\nshape (h, n, d_k) per head", fontsize=8.2, color=INK)
        if mode == "K":
            for j in (2, 5):
                ax.add_patch(Rectangle((j - 0.5, -0.5), 1, 12, fill=False, ec=BLUE, lw=2.0))
        else:
            for i in (3, 9):
                ax.add_patch(Rectangle((-0.5, i - 0.5), 8, 1, fill=False, ec=ORANGE, lw=2.0))
    fig.suptitle("KIVI's asymmetry: quantize K per-channel (columns) and V per-token (rows) — the scale follows the outlier structure",
                 fontsize=9.0, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path = os.path.join(FIG_DIR, "fig-3-2-kv-3lines.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 3.2] {path}（KIVI 2402.02750 Fig 1/2 语义自绘，数字为示意）")


# ---------------- 图 3.3：淘汰时机时间线 ----------------
def fig_evict_timeline():
    fig, ax = plt.subplots(figsize=(11.8, 4.2), dpi=300)
    ax.set_xlim(0, 15.4); ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.text(7.7, 5.35, "Two clocks for eviction: H2O pays every decode step; SnapKV pays once at the end of prefill",
            ha="center", fontsize=9.4, color=INK, weight="bold")
    # 时间轴
    ax.annotate("", xy=(14.9, 2.6), xytext=(0.5, 2.6), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.text(3.0, 2.25, "PREFILL  (parallel, TTFT clock)", fontsize=7.8, color=BLUE, ha="center")
    ax.text(10.4, 2.25, "DECODE  (serial, TPOT clock)", fontsize=7.8, color=ORANGE, ha="center")
    ax.plot([6.8, 6.8], [2.25, 2.95], color=MUTED, ls="--", lw=1.0)
    # H2O
    box(ax, 0.8, 3.3, 2.6, 0.95, "H2O:\nbookkeeping starts\nfrom step 1", ec=TEAL, fc="#d9f2e7", fs=6.8)
    for i in range(5):
        ax.scatter([7.6 + i * 1.4], [3.7], marker="$§$", s=90, color=TEAL)
    ax.text(10.5, 4.1, "every step: update Fscore, evict min — online, adaptive, O(1) bookkeeping per step", fontsize=7.0, color=TEAL)
    # SnapKV
    box(ax, 3.9, 1.0, 2.6, 0.95, "SnapKV:\nobserve-window scores\n→ one-shot pick @prefill end", ec=YELLOW, fc="#fbf3dc", fs=6.6)
    ax.annotate("", xy=(6.8, 1.5), xytext=(6.5, 1.5), arrowprops=dict(arrowstyle="-|>", color=YELLOW, lw=1.6))
    box(ax, 7.6, 1.0, 6.9, 0.95, "decode: fixed cache list, zero eviction cost — pays once, adapts never", ec=YELLOW, fc="#fbf3dc", fs=7.0)
    ax.text(7.7, 0.45, "H2O = defensive (adapts to generation-time attention) · SnapKV = offensive (prompt-end attention predicts generation needs) — two time-views of the same budget",
            ha="center", fontsize=7.2, color=MUTED)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-3-3-kv-3lines.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 3.3] {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch3 制图（图 3.1/3.2/3.3）")
    ap.add_argument("--which", default="all", choices=["all", "routes", "kivi", "timeline"])
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    if args.which in ("all", "routes"):
        fig_routes()
    if args.which in ("all", "kivi"):
        fig_kivi_axes()
    if args.which in ("all", "timeline"):
        fig_evict_timeline()
