# fig_ch01.py —— Book6 ch1 制图：图 1.1 两阶段时间线 / 图 1.2 roofline（既有件不在此） / 图 1.3 饱和曲线（slug=ttft-tpot）
# 用途：图 1.1 = 1.3 节章名级机制图（自绘示意级：prefill 并行块 (1,n0,d) vs decode 串行点 (1,1,d)——
#         规范 §4「关键机制必配图」+ batch 维形状标注）；
#       图 1.3 = 1.4 节 matmul 尺寸→饱和度五点曲线（喂 notes/04 P2 底档：
#         2048/3072/4096/6144/8192 → 140.6/214.4/250.6/269.3/269.1——含 8192 平台段的完整故事；
#         本脚本为既有 PNG 的归档重渲，原图已在 drafts/figures/）。
# 所属章节：Book6 第 1 章（图 1.1/1.3；图 1.2 为既有件）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/Book6-推理系统导论/ch01/fig_ch01.py"
# 产物：drafts/Book6-推理系统导论/figures/fig-1-1-ttft-tpot.png / fig-1-3-saturation.png
# 图规：300dpi 白底；系列色 蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；线宽 2；图内英文、图注中文。
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "drafts", "Book6-推理系统导论", "figures")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e1e0d9"
# notes/04 P2 底档五点（bf16 方阵 matmul，TFLOPS，中位）
SAT = [(2048, 140.6), (3072, 214.4), (4096, 250.6), (6144, 269.3), (8192, 269.1)]


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=7.4, lw=1.3, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc, zorder=4)


# ---------------- 图 1.1：两阶段时间线 ----------------
def fig_timeline():
    fig, ax = plt.subplots(figsize=(11.0, 4.2), dpi=300)
    ax.set_xlim(0, 14); ax.set_ylim(0, 5.8)
    ax.axis("off")
    ax.text(7, 5.55, "One request, two regimes: parallel prefill block vs serial decode steps (207M, batch=1)",
            ha="center", fontsize=9.6, color=INK, weight="bold")
    # prefill 大块
    box(ax, 0.3, 3.0, 4.6, 1.5, "PREFILL\ninput (1, n₀, d) → (1, n₀, V)\nn₀ tokens in ONE forward\nbig matmuls — compute-bound",
        ec=BLUE, fc="#dcebfc", fs=8.0)
    ax.annotate("TTFT", xy=(4.95, 3.75), fontsize=9, color=BLUE, ha="left", weight="bold")
    ax.plot([4.9, 4.9], [2.3, 4.7], color=BLUE, lw=1.0, ls="--", zorder=2)
    # decode 串行点
    for i in range(7):
        box(ax, 5.6 + i * 1.15, 3.25, 0.85, 1.0, f"$t_{{{i+1}}}$\n(1,1,d)", ec=ORANGE, fc="#fde3d6", fs=6.8)
    ax.text(9.6, 4.75, "decode: one token per step — memory-bound (weights re-read each step)",
            fontsize=7.8, color=ORANGE, ha="center")
    ax.annotate("TPOT = step time", xy=(9.6, 2.6), fontsize=8.6, color=ORANGE, ha="center", weight="bold")
    # 底部资源画像条
    box(ax, 0.3, 0.6, 4.6, 1.3, "weights amortized over n₀ tokens\narithmetic intensity ≈ n₀/3 → high\n→ roofline plateau (compute)",
        ec=BLUE, fs=7.2)
    box(ax, 5.6, 0.6, 7.85, 1.3, "weights carried by 1 token\narithmetic intensity ≈ 1 → low\n→ roofline slope (bandwidth); without engine: launch-bound (§1.5)",
        ec=ORANGE, fs=7.2)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-1-1-ttft-tpot.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 1.1] {path}")


# ---------------- 图 1.3：尺寸→饱和度五点 ----------------
def fig_saturation():
    xs = [p[0] for p in SAT]; ys = [p[1] for p in SAT]
    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=300)
    ax.plot(xs, ys, color=BLUE, marker="o", lw=2, ms=6, zorder=4)
    ax.axhline(269.3, color=MUTED, ls="--", lw=1.0, zorder=2)
    ax.text(2070, 272, "saturated plateau ≈ 269.3 TFLOPS (n=6144 median)", fontsize=7.4, color=MUTED)
    for x, y in SAT:
        ax.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, -14),
                    fontsize=7.2, color=INK, ha="center")
    ax.axhline(313, color=ORANGE, ls=":", lw=1.0, zorder=2)
    ax.text(2070, 316, "official converted 313 (Atlas 800T A2 ÷ 8, vendor-derived)", fontsize=7.4, color=ORANGE)
    ax.set_xlabel("matmul size n (bf16 square, Atlas 910B3)", fontsize=9, color=MUTED)
    ax.set_ylabel("achieved TFLOPS", fontsize=9, color=MUTED)
    ax.set_ylim(100, 345)
    ax.tick_params(colors="#898781", labelsize=8)
    ax.grid(color=GRIDC, lw=0.6, zorder=0)
    ax.set_title("Peak is a big-matrix privilege: n=2048 reaches only 52% of plateau\n(5-point probe, notes/04 P2)",
                 fontsize=9, color=INK)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-1-3-saturation.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 1.3] {path}（数据源 notes/04 P2 五点底档）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch1 制图（图 1.1/1.3）")
    ap.add_argument("--which", default="all", choices=["all", "timeline", "saturation"])
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    if args.which in ("all", "timeline"):
        fig_timeline()
    if args.which in ("all", "saturation"):
        fig_saturation()
