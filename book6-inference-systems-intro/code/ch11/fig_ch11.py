# fig_ch11.py —— Book6 ch11 制图：图 11.1 误差摊派三路线 / 图 11.2 Marlin 加速-批量曲线（slug=quant）
# 用途：图 11.1 = 11.1 节「四种方法答同一个问题：误差往哪摊」的视觉总纲（RTN 均摊崩于离群/
#         GPTQ 按 Hessian 二阶摊派/缩放迁移搬去不收税的房间）；
#       图 11.2 = 11.5 节 Marlin（2408.11743）图 1 语义的示意级重绘：Marlin 把 bs=1 的近 4× 平台
#         「维持」到 bs 16-32、bs=128 仍有 ~1.5×；其他 4bit kernel bs=1 也快但随批崩落；
#         b_opt=I*/4≈50 竖线（脊点 208.3 FLOP/B÷4 FLOP/B per token）。
# 所属章节：Book6 第 11 章（图 11.1/11.2）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/Book6-推理系统导论/ch11/fig_ch11.py"
# 产物：drafts/Book6-推理系统导论/figures/fig-11-1-error-alloc.png / fig-11-2-marlin-batch.png
# 图规：300dpi 白底；系列色 蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；线宽 2；图内英文、图注中文。
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


# ---------------- 图 11.1：误差摊派三路线 ----------------
def fig_error_alloc():
    fig, ax = plt.subplots(figsize=(12.0, 4.6), dpi=300)
    ax.set_xlim(0, 15.6); ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.text(7.8, 5.95, "One question, four answers: where does the quantization error go?",
            ha="center", fontsize=9.6, color=INK, weight="bold")
    box(ax, 6.3, 4.5, 3.0, 1.05, "ROUND to grid\n(RTN: split evenly)", ec=MUTED, fc="#f2f1ec", fs=7.4)
    box(ax, 0.3, 1.0, 4.4, 2.6, "RTN (no calibration)\nsplit error evenly per grid\n\n→ outlier channel eats the\nglobal scale; normal channels\nfall into zero-bits\n(GPTQ T3: OPT 3-bit PPL 1e3+)", ec=ORANGE, fc="#fde3d6", fs=7.0)
    box(ax, 5.6, 1.0, 4.4, 2.6, "GPTQ (calibration set C)\nminimize ||(W'−W)X||\nper-column quant + compensate\nthe rest via H⁻¹=(2XXᵀ+λI)⁻¹\n\n→ push error where the model\ncares least (2nd-order info)", ec=BLUE, fc="#dcebfc", fs=7.0)
    box(ax, 10.9, 1.0, 4.4, 2.6, "Scaling migration (AWQ / SQ)\nY=(X·diag(s)⁻¹)(diag(s)·W)\nexact identity, then quantize\n\n→ move error into the room\nthat pays no tax: fp16 acts\n(AWQ) or tolerant weights (SQ)", ec=TEAL, fc="#d9f2e7", fs=7.0)
    for x in (4.9, 10.2):
        arr(ax, x, 2.3, x + 0.6, 2.3)
    arr(ax, 7.8, 4.4, 2.5, 3.7); arr(ax, 7.8, 4.4, 7.8, 3.7); arr(ax, 7.8, 4.4, 13.1, 3.7)
    ax.text(7.8, 0.45, "execution side (11.5): only the bandwidth region pays out — 4-bit cuts weight bytes ×4, FLOPs unchanged (Marlin)",
            ha="center", fontsize=7.4, color=MUTED)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-11-1-error-alloc.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 11.1] {path}")


# ---------------- 图 11.2：Marlin 加速-批量曲线 ----------------
def fig_marlin():
    bs = [1, 2, 4, 8, 16, 32, 64, 128]
    marlin = [3.9, 3.9, 3.9, 3.85, 3.8, 3.1, 2.2, 1.5]      # 平台保到 16-32、128 档 1.5（论文 Fig 1 读数锚）
    ideal = [3.87, 3.87, 3.87, 3.87, 3.87, 3.5, 2.4, 1.5]   # 理想线：斜坡区平、过脊点随算力区衰减
    others = [3.7, 3.3, 2.6, 1.9, 1.3, 0.9, 0.7, 0.55]      # 其他 4bit kernel（bs=1 也快、随批崩落）
    fig, ax = plt.subplots(figsize=(7.8, 4.4), dpi=300)
    ax.plot(bs, ideal, color=MUTED, lw=1.4, ls="--", label="ideal (bandwidth → compute)", zorder=3)
    ax.plot(bs, marlin, color=BLUE, lw=2.2, marker="o", ms=4.5, label="MARLIN (W4A16, A10)", zorder=5)
    ax.plot(bs, others, color=ORANGE, lw=1.8, marker="s", ms=4, label="other W4A16 kernels (bs=1-optimized)", zorder=4)
    ax.axhline(3.87, color=GRIDC, lw=0.8, zorder=1)
    ax.text(1.05, 4.0, "theoretical 4× (3.87 measured cap)", fontsize=7.2, color=MUTED)
    ax.axvline(50, color=TEAL, lw=1.2, ls=":", zorder=2)
    ax.text(52, 2.9, "b_opt = I*/4 ≈ 50\n(I* ≈ 208.3 FLOP/B:\n125 TFLOP/s ÷ 600 GB/s)", fontsize=7.2, color=TEAL)
    ax.annotate("platform held to bs 16-32\n(base-clock ridge 108.8 → b≈27)", xy=(24, 3.35), xytext=(38, 3.95),
                fontsize=7.0, color=INK, arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9))
    ax.set_xscale("log", base=2)
    ax.set_xticks(bs); ax.set_xticklabels([str(b) for b in bs], fontsize=8)
    ax.set_xlabel("batch size (decode → prefill as it grows)", fontsize=9, color=MUTED)
    ax.set_ylabel("speedup vs FP16 PyTorch (CUTLASS)", fontsize=9, color=MUTED)
    ax.set_ylim(0, 4.6)
    ax.tick_params(colors="#898781", labelsize=8)
    ax.grid(color=GRIDC, lw=0.5, zorder=0)
    ax.legend(fontsize=7.6, frameon=False, loc="center left")
    ax.set_title("Quantization pays in the bandwidth region only (A10, 72k×18k, g128)", fontsize=9.2, color=INK)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-11-2-marlin-batch.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 11.2] {path}（示意级重绘，锚点读数=Marlin 论文 Fig 1/摘要）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch11 制图（图 11.1/11.2）")
    ap.add_argument("--which", default="all", choices=["all", "error_alloc", "marlin"])
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    if args.which in ("all", "error_alloc"):
        fig_error_alloc()
    if args.which in ("all", "marlin"):
        fig_marlin()
