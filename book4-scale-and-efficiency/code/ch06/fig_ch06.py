# fig_ch06.py —— Book4 ch6 制图：图 6.1 d=8→r=2 手算例 / 图 6.2 解耦 RoPE 双车道 /
#                图 6.3 MLA 双形态数据流 / 图 6.4 KV 账本多几何节省（编号=正文出现顺序）
# 用途：图 6.1 = 正文 6.2 手算例（h=[1,2,...] → c=[4,8] → K/V 两读法，缓存 −81.25%）；
#       图 6.2 = 6.5 解耦 RoPE 的内容/位置双车道（k_R 全头共享，式 (6.9)）；
#       图 6.3 = 6.6 的机构总图（左显式上投影 / 右权重吸收，全部张量框标形状、箱子高亮）；
#       图 6.4 = 6.7/6.8 的多几何 KV 账本（喂 log/book4-ch06/kv_probe_mla_*.json，--src 自动）。
# 所属章节：Book4 第 6 章（图 6.1-6.4，按正文出现顺序编号）。
# 运行方式：cd 工作区根目录 && source env.sh && python code/ch06/fig_ch06.py
#           [--which all|diagram|lanes|bars|hand] [--src auto|run1]
# 产物：figures/fig-6-{1,2,3,4}-*.png
# 图规：印刷规格 300dpi 白底；系列色 蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；线宽 2；
#       刻度 #898781、标注 #0b0b0b；图内英文、图注中文（图注在正文 markdown，不入图）。
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "figures")
LOG_DIR = os.path.join(REPO_ROOT, "log", "book4-ch06")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e1e0d9"
CACHE_FC = "#dcebfc"          # 箱子/缓存项的浅蓝底（进入 KV cache 的量）


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=8.0, lw=1.5, tc=INK, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw,
                                linestyle=ls, mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc, zorder=4)
    return x + w / 2, y + h / 2


def arrow(ax, x0, y0, x1, y1, label=None, color=MUTED, fs=7.0, dx=0.0, dy=0.06, lw=1.2, ha="center"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=lw, zorder=2))
    if label:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, label, ha=ha, va="bottom",
                fontsize=fs, color=MUTED, zorder=4)


def lroute(ax, pts, label=None, color=MUTED, fs=7.0, at=1, dx=0.0, dy=0.0):
    """L 形折线路由：pts 诸拐点连线，仅末段带头；label 贴第 at 段中点（防穿越框线）。"""
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]], color=color, lw=1.2,
                zorder=2, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=1.2, zorder=2))
    if label:
        a, b = pts[at], pts[at + 1]
        ax.text((a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2 + dy, label, ha="center", va="bottom",
                fontsize=fs, color=MUTED, zorder=4)


# ---------------- 图 6.3：MLA 数据流双形态 ----------------
def fig_dual_form():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9), dpi=300)
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8.6)
        ax.axis("off")

    # —— 左：显式上投影（训练 / HF eager）——
    ax = axes[0]
    ax.text(5.0, 8.25, "Explicit up-projection (training / HF eager)", ha="center",
            fontsize=10.5, color=INK, weight="bold")
    box(ax, 0.2, 6.9, 1.7, 0.8, "h\n(B,n,5120)", ec=MUTED)
    box(ax, 2.5, 6.9, 2.1, 0.8, "kv_a_proj\n[W_DKV;W_KR]\n(576,5120)")
    box(ax, 5.4, 7.25, 2.0, 0.75, "c_KV\n(B,n,512)", ec=BLUE, fc=CACHE_FC)
    box(ax, 5.4, 6.1, 2.0, 0.75, "k_R→RoPE\n(B,1,n,64)", ec=BLUE, fc=CACHE_FC)
    arrow(ax, 1.9, 7.3, 2.5, 7.3)
    arrow(ax, 4.6, 7.35, 5.4, 7.55, "split", dx=-0.15)
    arrow(ax, 4.6, 7.15, 5.4, 6.45)
    ax.text(7.55, 7.62, "→ KV cache", fontsize=8.5, color=BLUE, va="center")
    ax.text(7.55, 6.48, "→ KV cache", fontsize=8.5, color=BLUE, va="center")
    box(ax, 2.5, 4.9, 2.1, 0.8, "kv_b_proj\n[W_UK;W_UV]\n(32768,512)", ec=TEAL)
    arrow(ax, 5.4, 7.35, 3.6, 5.7, "read-out", dx=0.62, dy=-0.02)
    box(ax, 0.2, 4.9, 2.0, 0.8, "K, V\n(B,h,n,128)\nmaterialized", ec=TEAL)
    arrow(ax, 2.5, 5.3, 2.2, 5.3, "per head", dy=0.12)
    box(ax, 0.2, 3.5, 1.8, 0.8, "q_C\n(B,h,n,128)", ec=MUTED)
    box(ax, 2.7, 3.5, 2.1, 0.8, "RoPE(q_R)\n(B,h,n,64)", ec=ORANGE)
    arrow(ax, 2.0, 3.9, 2.7, 3.9)
    box(ax, 0.4, 1.9, 4.7, 1.0, "scores = (q_C·K + q_R·k_R)/√192\n(B,h,n,n)   192-dim logical dot",
        ec=BLUE)
    arrow(ax, 1.1, 3.5, 1.4, 2.9)
    arrow(ax, 3.75, 3.5, 3.5, 2.9)
    lroute(ax, [(2.2, 5.3), (2.35, 5.3), (2.35, 2.9)])                    # K,V 下行（走窄缝防穿框）
    arrow(ax, 5.4, 6.45, 5.1, 2.9, "expand to h heads", dx=0.35, dy=0.5)  # k_R 下行
    box(ax, 0.4, 0.5, 4.7, 0.9, "softmax → ·V → W_O → u\n(B,h,n,128) → (B,n,5120)", ec=BLUE)
    arrow(ax, 2.6, 1.9, 2.6, 1.4)
    ax.text(5.0, 0.15, "cache write in compressed state; read-out up-projects every step",
            fontsize=7.5, color=MUTED, ha="center")

    # —— 右：权重吸收（推理）——
    ax = axes[1]
    ax.text(5.0, 8.25, "Absorbed (inference: attention in latent space)", ha="center",
            fontsize=10.5, color=INK, weight="bold")
    box(ax, 0.2, 6.9, 2.2, 0.8, "KV cache\n(c_KV, k_R)\n576 / token / layer", ec=BLUE, fc=CACHE_FC)
    box(ax, 3.4, 6.9, 1.8, 0.8, "q_C\n(B,h,n,128)", ec=MUTED)
    box(ax, 3.4, 5.7, 1.8, 0.8, "RoPE(q_R)\n(B,h,n,64)", ec=ORANGE)
    box(ax, 5.8, 6.9, 2.4, 0.8, "q_hat = q_C·W_UK\n(B,h,n,512)", ec=TEAL)
    arrow(ax, 5.2, 7.3, 5.8, 7.3, "absorb W_UK", dy=0.12)
    box(ax, 4.6, 3.9, 3.9, 0.9, "scores = (q_hat·c_KV + q_R·k_R)/√192\n(B,h,n,n)   512-dim latent dot",
        ec=BLUE)
    arrow(ax, 7.0, 6.9, 7.0, 4.8)                                          # q_hat 下行
    arrow(ax, 4.3, 5.7, 5.3, 4.8)                                          # RoPE(q_R) 下行
    arrow(ax, 1.3, 6.9, 4.6, 4.62, "c_KV (B,n,512)", dx=-0.35, dy=0.05)    # cache→scores
    lroute(ax, [(2.1, 6.9), (2.1, 4.35), (4.6, 4.35)], "k_R (B,1,n,64)", at=0, dx=0.45, dy=0.0)
    box(ax, 0.3, 2.2, 3.3, 1.0, "softmax → c_mix = A·c_KV\n(B,h,n,512)  mix inside box", ec=BLUE)
    arrow(ax, 5.3, 3.9, 2.8, 3.2)
    box(ax, 4.2, 2.2, 2.7, 1.0, "W_UV absorbed\nu = c_mix·W_UVᵀ\n(B,h,n,128)", ec=TEAL)
    arrow(ax, 3.6, 2.7, 4.2, 2.7, "absorb W_UV", dy=0.14)
    box(ax, 3.0, 0.5, 3.6, 0.9, "W_O → u\n(B,n,5120)", ec=BLUE)
    arrow(ax, 5.5, 2.2, 5.0, 1.4)
    ax.text(5.0, 0.15, "prefix side never up-projected — associativity A(BC)=(AB)C",
            fontsize=7.5, color=MUTED, ha="center")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-6-3-mla-dual-form-dataflow.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 6.3] {path}")


# ---------------- 图 6.2：解耦 RoPE 双车道 ----------------
def fig_two_lanes():
    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    ax.text(5.0, 7.3, "Decoupled RoPE: content lane (boxed) + position lane (hand-carried)",
            ha="center", fontsize=10.5, color=INK, weight="bold")

    # 内容车道（上路，蓝）
    box(ax, 0.3, 5.5, 2.0, 0.9, "c_KV\n(B,n,512)\nposition-free", ec=BLUE, fc=CACHE_FC)
    box(ax, 3.1, 5.5, 2.4, 0.9, "k_C = W_UK·c_KV\nper head (B,h,n,128)", ec=TEAL)
    box(ax, 6.3, 5.5, 1.9, 0.9, "content score\nq_C · k_C\n(B,h,n,n)", ec=BLUE)
    arrow(ax, 2.3, 5.95, 3.1, 5.95)
    arrow(ax, 5.5, 5.95, 6.3, 5.95)
    ax.text(2.9, 4.95, "content lane: compressible & absorbable (this lane lives in the box)",
            fontsize=8.0, color=BLUE, ha="center")

    # 位置车道（下路，橙）
    box(ax, 0.3, 3.2, 2.0, 0.9, "k_R = W_KR·h\n(B,n,64)\nshared by ALL heads", ec=ORANGE, fc="white")
    box(ax, 3.1, 3.2, 2.2, 0.9, "RoPE(k_R)\nat write time\n(B,1,n,64)", ec=ORANGE)
    box(ax, 6.2, 3.2, 1.9, 0.9, "position score\nRoPE(q_R)·k_R\n(B,h,n,n)", ec=ORANGE)
    arrow(ax, 2.3, 3.65, 3.1, 3.65)
    arrow(ax, 5.3, 3.65, 6.2, 3.65)
    ax.text(1.55, 2.65, "position lane: 64-dim, MQA-style, never compressed", fontsize=8.0,
            color=ORANGE, ha="center")

    # 汇合
    box(ax, 4.0, 1.1, 3.6, 1.1, "score = (content + position) / √(128+64)=√192\n→ causal softmax → output",
        ec=INK)
    arrow(ax, 7.15, 5.5, 6.4, 2.2, dx=0.5)
    arrow(ax, 7.15, 3.2, 6.4, 2.2, dx=0.5)
    ax.text(8.5, 4.4, "Eq (6.9):\nq·k splits into\ntwo independent dots", fontsize=7.5,
            color=MUTED, ha="center")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-6-2-decoupled-rope-two-lanes.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 6.2] {path}")


# ---------------- 图 6.4：KV 账本多几何节省（喂 kv_probe JSON） ----------------
def fig_bars(src):
    cands = sorted(glob.glob(os.path.join(LOG_DIR, f"kv_probe_mla_{src}.json"))) or \
        sorted(glob.glob(os.path.join(LOG_DIR, "kv_probe_mla_*.json")))
    with open(cands[-1], "r", encoding="utf-8") as f:
        rows = json.load(f)["part2_geometry_table"]["vs_baseline"]
    labels = ["probe\n(L=6)", "207M two-cut\n(vs GQA-8)", "207M\n(vs MHA)", "DSV2-Lite\n(L=27)",
              "DSV2-236B\n(L=60)"]
    base = [r["baseline_elems"] for r in rows]
    mla = [r["mla_elems"] for r in rows]
    sav = [r["saving_pct"] for r in rows]
    y = range(len(rows))
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=300)
    ax.barh([i + 0.21 for i in y], base, height=0.38, color=MUTED, label="MHA/GQA baseline", lw=0)
    ax.barh([i - 0.21 for i in y], mla, height=0.38, color=BLUE, label="MLA", lw=0)
    ax.set_xscale("log")
    for i, (b, m, s) in enumerate(zip(base, mla, sav)):
        ax.text(b * 1.12, i + 0.21, f"{b:,}", va="center", fontsize=7.5, color=MUTED)
        ax.text(m * 1.12, i - 0.21, f"{m:,}", va="center", fontsize=7.5, color=BLUE)
        ax.text(3.4e6, i, f"−{s}%", va="center", fontsize=9, color=INK, weight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.set_xlabel("KV cache elements per token", fontsize=9, color=MUTED)
    ax.tick_params(colors="#898781", labelsize=8)
    ax.set_xlim(1e3, 1.1e7)
    ax.grid(axis="x", color=GRIDC, lw=0.6, zorder=0)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-6-4-kv-multi-geometry.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 6.4] {path}（数据源 {os.path.basename(cands[-1])}）")


# ---------------- 图 6.1：d=8→r=2 手算例 ----------------
def fig_hand_example():
    fig, ax = plt.subplots(figsize=(9.2, 4.0), dpi=300)
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 7.0)
    ax.axis("off")
    ax.text(5.7, 6.75, "Hand example: d=8 → rank 2 — one box, two readings, −81.25% cache",
            ha="center", fontsize=10.5, color=INK, weight="bold")

    box(ax, 0.2, 4.6, 1.9, 1.2, "h\n[1,2,1,2,\n1,2,1,2]", ec=MUTED)
    box(ax, 2.7, 4.6, 2.0, 1.2, "W_DKV\n(2,8)", ec=TEAL)
    box(ax, 5.3, 4.6, 1.9, 1.2, "c = [4, 8]\n(2,)  ← cache", ec=BLUE, fc=CACHE_FC)
    box(ax, 5.3, 2.6, 1.9, 1.0, "k_R = 3\n(1,)  ← cache", ec=ORANGE, fc=CACHE_FC)
    box(ax, 2.7, 2.6, 2.0, 1.0, "W_KR\n(1,8)", ec=ORANGE)
    arrow(ax, 2.1, 5.2, 2.7, 5.2)
    arrow(ax, 4.7, 5.2, 5.3, 5.2)
    arrow(ax, 2.1, 4.8, 2.7, 3.2, dx=-0.3)
    arrow(ax, 4.7, 3.1, 5.3, 3.1)

    box(ax, 8.0, 5.4, 2.6, 0.9, "W_UK lens → k\n[1,1,1,1,1,1,1,1]", ec=TEAL)
    box(ax, 8.0, 3.9, 2.6, 0.9, "W_UV lens → v\n[4,−8,4,−8,4,−8,4,−8]", ec=TEAL)
    arrow(ax, 7.2, 5.4, 8.0, 5.75, "K reading", dy=0.12)
    arrow(ax, 7.2, 4.9, 8.0, 4.35, "V reading", dy=-0.12)
    ax.text(9.3, 3.35, "same box c, two lenses:\nk=W_UK·c,  v=W_UV·c\nidentical to full-rank M·h",
            fontsize=7.8, color=MUTED, ha="center", va="top")
    box(ax, 0.6, 0.5, 5.4, 1.3, "cache: (2 + 1) / 16 = 3/16 = 18.75%\n→ −81.25%  (miniature of the"
        " real −98.2%)", ec=BLUE, fc=CACHE_FC)
    arrow(ax, 6.2, 2.6, 4.4, 1.8, dx=0.4)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-6-1-hand-example-d8-r2.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 6.1] {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch6 制图（图 6.1-6.4）")
    ap.add_argument("--which", default="all", choices=["all", "diagram", "lanes", "bars", "hand"])
    ap.add_argument("--src", default="run1", help="kv_probe_mla JSON 后缀（图 6.3 数据源）")
    args = ap.parse_args()
    if args.which in ("all", "diagram"):
        fig_dual_form()
    if args.which in ("all", "lanes"):
        fig_two_lanes()
    if args.which in ("all", "bars"):
        fig_bars(args.src)
    if args.which in ("all", "hand"):
        fig_hand_example()
