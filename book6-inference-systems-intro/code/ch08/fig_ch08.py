# fig_ch08.py —— Book6 ch8 制图：图 8.1 铺表 vs 分块搬运路径 / 图 8.2 两状态更新流水 /
#                图 8.3 三实现时延曲线（slug=fa-tile）
# 用途：图 8.1 = 8.1 交通账的视觉载体（依据 FA1 Fig 1 机制自绘——版权档：FA 论文不贴原图）；
#       图 8.2 = 8.2 例 8.1 的两块 m/l 修正流水（依据 FA2 Fig 1 机制自绘——两块版手算可跟走）；
#       图 8.3 = 8.3 时延扫描（喂 log/book6-ch08/fa1_base.json 四档，log 轴——教学版慢两个量级）。
# 所属章节：Book6 第 8 章（图 8.1-8.3）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/Book6-推理系统导论/ch08/fig_ch08.py"
#           [--which all|path|online|latency]
# 产物：drafts/Book6-推理系统导论/figures/fig-8-{1,2,3}-fa-tile.png
# 图规：300dpi 白底；系列色 蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；线宽 2；图内英文、图注中文。
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "drafts", "Book6-推理系统导论", "figures")
FA1_JSON = os.path.join(REPO_ROOT, "log", "book6-ch08", "fa1_base.json")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e1e0d9"


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=7.2, lw=1.4, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc, zorder=4)


def arr(ax, x0, y0, x1, y1, color=MUTED, lw=1.1, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


# ---------------- 图 8.1：铺表 vs 分块的搬运路径（依据 FA1 Fig 1 机制自绘） ----------------
def fig_path():
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=300)
    for ax in axes:
        ax.axis("off")
    # 左：naive 四趟
    a = axes[0]
    a.set_xlim(0, 10); a.set_ylim(0, 6.4)
    a.set_title("naive: the (B,h,n,n) table lives in HBM — 4 crossings", fontsize=9, color=INK)
    box(a, 0.3, 4.6, 2.2, 1.0, "Q,K,V\n(B,h,n,d_k)", ec=MUTED)
    box(a, 3.6, 4.6, 2.6, 1.0, "S = QK^T\n(B,h,n,n)", ec=ORANGE, fc="#fde3d6")
    box(a, 3.6, 2.6, 2.6, 1.0, "P = softmax(S)\n(B,h,n,n)", ec=ORANGE, fc="#fde3d6")
    box(a, 7.0, 4.6, 2.4, 1.0, "O = P·V\n(B,h,n,d_k)", ec=TEAL)
    for y0, y1, lbl in ((5.1, 5.1, "write"), (5.1, 3.1, "read"), (3.1, 3.1, "write"), (3.1, 5.1, "read")):
        pass
    arr(a, 2.5, 5.1, 3.6, 5.1)
    arr(a, 4.9, 4.6, 4.9, 3.6, color=ORANGE)
    arr(a, 6.2, 3.1, 7.0, 4.9, color=ORANGE)
    a.text(4.9, 1.9, "HBM round-trips ≈ 4×(B,h,n,n)·2B\nn=2048,h=16: ~512 MiB per attention\n≈ one full pass over 207M weights",
           ha="center", fontsize=7.4, color=ORANGE)
    # 右：分块双循环
    b = axes[1]
    b.set_xlim(0, 10); b.set_ylim(0, 6.4)
    b.set_title("tiled: blocks live on-chip — table never materialized", fontsize=9, color=INK)
    box(b, 0.3, 4.8, 2.4, 0.9, "HBM: Q,K,V,O\n(B,h,n,d_k)", ec=MUTED)
    box(b, 3.7, 4.8, 2.6, 0.9, "SRAM tile:\nK_j,V_j (B_c,d_k)", ec=TEAL, fc="#d9f2e7")
    box(b, 3.7, 2.9, 2.6, 1.0, "on-chip compute\nS_ij (B_r,B_c) — discarded", ec=BLUE, fc="#dcebfc")
    box(b, 7.1, 2.9, 2.2, 1.0, "running\nm, l, acc", ec=YELLOW, fc="#fdf3d7")
    arr(b, 2.7, 5.25, 3.7, 5.25)
    arr(b, 5.0, 4.8, 5.0, 3.9, color=TEAL)
    arr(b, 6.3, 3.4, 7.1, 3.4, color=BLUE)
    b.text(5.0, 1.9, "outer loop: stream K/V blocks to SRAM (A100: ~20MB @ ~19TB/s est)\ninner: Q block × K block on-chip; (B_r,B_c) score tile never reaches HBM",
           ha="center", fontsize=7.4, color=MUTED)
    b.add_patch(Rectangle((4.6, 0.7), 4.8, 0.62, fc="none", ec=ORANGE, lw=1.0, ls="--"))
    b.text(7.0, 1.01, "the n×n matrix (dotted) never exists", fontsize=7.2, color=ORANGE, ha="center")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-8-1-fa-tile.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 8.1] {path}")


# ---------------- 图 8.2：两状态更新流水（依据 FA2 Fig 1 机制自绘；例 8.1 同数） ----------------
def fig_online():
    fig, ax = plt.subplots(figsize=(10.6, 4.4), dpi=300)
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.0)
    ax.axis("off")
    ax.text(6.5, 5.75, "Online softmax: two-block merge with exchange rate α (example 8.1 numbers)",
            ha="center", fontsize=10, color=INK, weight="bold")
    # 块 A
    box(ax, 0.3, 3.2, 2.6, 1.4, "block A  s=[1,3,2]\nm_A=3\nl_A=1.503  acc_A=2.135",
        ec=BLUE, fc="#dcebfc", fs=7.4)
    # 块 B
    box(ax, 0.3, 1.0, 2.6, 1.4, "block B  s=[5,1,2]\nm_B=5\nl_B=1.068  wsum=1.150",
        ec=BLUE, fc="#dcebfc", fs=7.4)
    # 合并
    box(ax, 4.0, 2.1, 3.4, 1.6, "merge @ m_new=5\nα=e^{3−5}=0.135\nl = α·1.503 + 1.068 = 1.270\nacc = α·2.135 + 1.150 = 1.438",
        ec=ORANGE, fc="#fde3d6", fs=7.6)
    arr(ax, 2.9, 3.9, 4.0, 3.2, color=BLUE)
    arr(ax, 2.9, 1.7, 4.0, 2.4, color=BLUE)
    ax.text(3.45, 3.75, "α ×", fontsize=8, color=ORANGE)
    ax.text(3.45, 1.55, "1 ×", fontsize=8, color=ORANGE)
    # 输出
    box(ax, 8.2, 2.4, 2.9, 1.0, "out = acc / l\n= 1.438/1.270 = 1.132", ec=TEAL, fc="#d9f2e7", fs=7.8)
    arr(ax, 7.4, 2.9, 8.2, 2.9, color=ORANGE)
    # 全量对照
    box(ax, 11.3, 2.4, 1.5, 1.0, "full-pass\n= 1.132\n(bitwise)", ec=MUTED, fs=7.0)
    arr(ax, 11.1, 2.9, 11.3, 2.9, color=MUTED, style="<|-|>")
    ax.text(6.5, 0.55, "no step ever goes back: old ledger converted by α, new block booked at β — final division once",
            ha="center", fontsize=7.6, color=MUTED)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-8-2-fa-tile.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 8.2] {path}")


# ---------------- 图 8.3：三实现时延（log 轴） ----------------
def fig_latency():
    d = json.load(open(FA1_JSON))["npu_bench"]
    ns = [r["n"] for r in d]
    fig, ax = plt.subplots(figsize=(7.8, 4.4), dpi=300)
    for key, color, mk, lb in (("naive_ms", ORANGE, "o", "naive (materialize)"),
                               ("sdpa_ms", TEAL, "s", "sdpa (industrial)"),
                               ("flash1_ms", BLUE, "^", "flash1 (teaching, python loop)")):
        ax.plot(ns, [r[key] for r in d], color=color, marker=mk, lw=2, ms=6, label=lb, zorder=4)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length n  (shape (1,16,n,64), bf16)", fontsize=9, color=MUTED)
    ax.set_ylabel("latency (ms, log)", fontsize=9, color=MUTED)
    for r in d:
        if r["n"] == 4096:
            ax.annotate(f"sdpa/naive = {r['naive_ms']/r['sdpa_ms']:.1f}×",
                        xy=(r["n"], r["sdpa_ms"]), xytext=(r["n"] * 0.45, r["sdpa_ms"] * 0.4),
                        fontsize=8, color=TEAL, arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.0))
    ax.tick_params(colors="#898781", labelsize=8)
    ax.grid(color=GRIDC, lw=0.6, zorder=0)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Same FLOPs, different homes for the intermediate: naive n² vs sdpa near-linear\n(teaching flash1: correct math, python-loop tax — two orders slower)",
                 fontsize=8.6, color=INK)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-8-3-fa-tile.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 8.3] {path}（数据源 fa1_base.json）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch8 制图（图 8.1-8.3）")
    ap.add_argument("--which", default="all", choices=["all", "path", "online", "latency"])
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    if args.which in ("all", "path"):
        fig_path()
    if args.which in ("all", "online"):
        fig_online()
    if args.which in ("all", "latency"):
        fig_latency()
