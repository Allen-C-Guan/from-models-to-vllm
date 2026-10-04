# make_figures.py —— Book3 ch02 三张图：图 2.1 计算图并排 / 图 2.2 动刀位置 / 图 2.3 消融曲线
# 用途：图 2.1 自绘示意级结构图（张量框与关键箭头标形状；橙色虚线 = RMSNorm 砍掉的 LN 部件）；
#       图 2.2 自绘示意级（Book2 三插槽底座整机 + Block 放大，13 处 REPLACE 高亮）；
#       图 2.3 读 log/book3-ch02/ablation_norm_{tier}.json 的曲线与 eval 点（fast=1000 / full=5000 步）。
# 所属章节：Book3 第 2 章 §2.3（图 2.1）/§2.5（图 2.2）/§2.6（图 2.3）
# 运行方式：cd 工作区根目录 && source env.sh && python code/ch02/make_figures.py \
#           [--tier auto|fast|full]      （auto=full 的 JSON 完整跑毕（各臂 steps_done==steps_planned）则用 full，否则 fast）
# 产物：figures/fig-2-1-norm-compute-graph.png 、fig-2-2-swap-position.png
#       与 fig-2-3-norm-ablation.png（300dpi；图内文字英文、图注中文见正文）
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(REPO, "figures")
LOG_DIR = os.path.join(REPO, "log", "book3-ch02")
os.makedirs(FIG_DIR, exist_ok=True)

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MUTED = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": TICK, "axes.labelcolor": DARK,
    "xtick.color": TICK, "ytick.color": TICK, "text.color": DARK,
    "axes.linewidth": 0.8, "font.family": "DejaVu Sans",
})


def box(ax, x, y, w, h, label, fc="white", ec=DARK, lw=1.1, fs=7.6, tc=None, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc or DARK)


def arrow(ax, x1, y1, x2, y2, label=None, color=DARK, lw=1.1, ls="-", fs=6.8, dx=0.1, dy=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                 color=color, lw=lw, linestyle=ls))
    if label:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, label, fontsize=fs, color=MUTED,
                ha="left", va="center")


# ---------------- 图 2.1：LayerNorm 与 RMSNorm 计算图并排（自绘示意级，标形状） ----------------
def fig_2_1():
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.6), dpi=300)
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10.6)
        ax.axis("off")

    # ---- 左：LayerNorm（两件事：重中心化 + 重缩放）----
    ax = axes[0]
    ax.text(5.0, 10.25, "LayerNorm  (2 affine params per site: 2C)", ha="center",
            fontsize=9, color=BLUE, fontweight="bold")
    box(ax, 3.0, 8.9, 4.0, 0.85, "input x\n(B, n, C)", fc="#f2f7fd", ec=BLUE, lw=1.3)
    # 统计两路：mu（RMSNorm 砍掉）/ sigma（保留）
    box(ax, 0.4, 6.9, 4.1, 1.0, "mu = mean(x)\n(B, n, 1)", fc="#fdf3ef", ec=ORANGE, lw=1.3, ls="--")
    box(ax, 5.5, 6.9, 4.1, 1.0, "sigma = sqrt(var(x))\n(B, n, 1)", ec=BLUE)
    arrow(ax, 4.2, 8.9, 2.6, 7.95, color=ORANGE, ls="--")
    arrow(ax, 5.8, 8.9, 7.4, 7.95)
    # 中心化 + 除以 sigma
    box(ax, 2.4, 4.9, 5.2, 0.95, "( x - mu ) / sigma\n(B, n, C)", ec=BLUE)
    arrow(ax, 2.6, 6.9, 3.9, 5.9, "(x-mu)" if False else None, color=ORANGE, ls="--")
    arrow(ax, 7.4, 6.9, 6.2, 5.9)
    # 仿射
    box(ax, 1.4, 2.9, 7.2, 0.95, "gamma * ( . ) + beta\n(C,)  (C,)", ec=BLUE)
    box(ax, 0.4, 1.7, 3.2, 0.8, "beta (C,)", fc="#fdf3ef", ec=ORANGE, lw=1.2, ls="--", fs=7.2)
    arrow(ax, 5.0, 4.9, 5.0, 3.9)
    arrow(ax, 2.0, 2.5, 3.4, 2.9, color=ORANGE, ls="--")
    box(ax, 3.0, 0.9, 4.0, 0.85, "output\n(B, n, C)", fc="#f2f7fd", ec=BLUE, lw=1.3)
    arrow(ax, 5.0, 2.9, 5.0, 1.8)
    ax.text(9.75, 5.35, "re-centering\n+ re-scaling", fontsize=7.6, color=MUTED, ha="right")

    # ---- 右：RMSNorm（只做重缩放；橙色虚线部件全部消失）----
    ax = axes[1]
    ax.text(5.0, 10.25, "RMSNorm  (1 affine param per site: C)", ha="center",
            fontsize=9, color=ORANGE, fontweight="bold")
    box(ax, 3.0, 8.9, 4.0, 0.85, "input x\n(B, n, C)", fc="#fdf6f2", ec=ORANGE, lw=1.3)
    box(ax, 2.7, 6.9, 4.6, 1.0, "RMS(x) = sqrt( mean( x^2 ) )\n(B, n, 1)", ec=ORANGE)
    arrow(ax, 5.0, 8.9, 5.0, 7.95, color=ORANGE)
    box(ax, 2.4, 4.9, 5.2, 0.95, "x * rsqrt( mean(x^2) + eps )\n(B, n, C)", ec=ORANGE)
    arrow(ax, 5.0, 6.9, 5.0, 5.9, color=ORANGE)
    box(ax, 2.7, 2.9, 4.6, 0.95, "g * ( . )     g in (C,)\nno bias", ec=ORANGE)
    arrow(ax, 5.0, 4.9, 5.0, 3.9, color=ORANGE)
    box(ax, 3.0, 0.9, 4.0, 0.85, "output\n(B, n, C)", fc="#fdf6f2", ec=ORANGE, lw=1.3)
    arrow(ax, 5.0, 2.9, 5.0, 1.8, color=ORANGE)
    ax.text(9.75, 5.35, "re-scaling only\n(no mean, no bias)", fontsize=7.6, color=MUTED, ha="right")
    ax.text(5.0, 0.25, "dashed components on the left (mu path, beta) are exactly what RMSNorm removes",
            fontsize=7.0, color=MUTED, ha="center")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-2-1-norm-compute-graph.png"), facecolor="white")
    plt.close(fig)


# ---------------- 图 2.2：改装第一例的动刀位置（Book2 三插槽底座整机 + Block 放大，示意级） ----------------
def fig_2_2_machine():
    fig, ax = plt.subplots(figsize=(8, 5.6), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11.4)
    ax.axis("off")
    ax.text(7.0, 11.05, "GPT-2 base (Book2 ch10) — where the knife goes: all 2L+1 = 13 norm slots",
            ha="center", fontsize=9.2, color=DARK, fontweight="bold")

    # ---- 顶层整机链（左->右）----
    y0 = 9.2
    box(ax, 0.3, y0, 2.3, 0.9, "token ids\n(B, n)")
    box(ax, 3.0, y0, 2.9, 0.9, "wte + wpe\n-> (B, n, C)")
    box(ax, 6.3, y0, 2.6, 0.9, "Block x L=6\n(B, n, C)", fc="#f7fafe", ec=BLUE, lw=1.4)
    box(ax, 9.6, y0, 2.1, 0.9, "ln_f\n(B, n, C)", fc="#fdf3ef", ec=ORANGE, lw=1.6)
    box(ax, 12.0, y0, 1.7, 0.9, "head\n(B, n, V)")
    arrow(ax, 2.6, y0 + 0.45, 3.0, y0 + 0.45)
    arrow(ax, 5.9, y0 + 0.45, 6.3, y0 + 0.45)
    arrow(ax, 8.9, y0 + 0.45, 9.6, y0 + 0.45)
    arrow(ax, 11.7, y0 + 0.45, 12.0, y0 + 0.45)

    # ---- Block 放大：双行蛇形（行 1 = attn 支路，行 2 = mlp 支路）----
    ax.plot([7.6, 7.6], [y0, 8.25], ls=":", color=TICK, lw=1.0)
    ax.text(7.6, 8.0, "zoom into one Block  (pre-LN topology: untouched)", fontsize=8.4,
            color=MUTED, ha="center")

    def branch(y, in_label, norm_name, sub_label, out_label):
        box(ax, 0.4, y, 2.0, 1.0, in_label, fc="#f7fafe", ec=BLUE)
        box(ax, 3.0, y, 2.3, 1.0, f"{norm_name}  (C,)\nREPLACE", fc="#fdf3ef", ec=ORANGE, lw=1.6)
        box(ax, 6.0, y, 2.5, 1.0, sub_label, ec=BLUE)
        box(ax, 9.2, y, 0.7, 1.0, "+", ec=BLUE, fs=10)
        box(ax, 10.5, y, 3.2, 1.0, out_label, fc="#f7fafe", ec=BLUE)
        arrow(ax, 2.4, y + 0.5, 3.0, y + 0.5)
        arrow(ax, 5.3, y + 0.5, 6.0, y + 0.5)
        arrow(ax, 8.5, y + 0.5, 9.2, y + 0.5)
        arrow(ax, 9.9, y + 0.5, 10.5, y + 0.5)
        # 残差跨接：输入框底部绕到 + 框底部
        yb = y - 0.45
        ax.plot([1.4, 1.4], [y, yb], color=TICK, lw=1.0)
        ax.plot([1.4, 9.55], [yb, yb], color=TICK, lw=1.0)
        arrow(ax, 9.55, yb, 9.55, y - 0.02, color=TICK, lw=1.0)
        ax.text(5.5, yb - 0.28, "residual (identity)", fontsize=7.0, color=MUTED, ha="center")

    branch(6.3, "x\n(B, n, C)", "ln_1", "attn\nc_attn / c_proj", "x1 = x + attn(ln_1 x)\n(B, n, C)")
    branch(3.0, "x1\n(B, n, C)", "ln_2", "mlp\nc_fc / c_proj", "x2 = x1 + mlp(ln_2 x1)\n(B, n, C)")
    # 蛇形回卷：行 1 输出右侧下行，沿底边回到行 2 输入
    ax.plot([13.7, 13.7], [6.8, 1.6], color=BLUE, lw=1.0)
    ax.plot([13.7, 1.4], [1.6, 1.6], color=BLUE, lw=1.0)
    arrow(ax, 1.4, 1.6, 1.4, 2.95, color=BLUE, lw=1.0)
    ax.text(7.5, 1.32, "next block (or ln_f -> head after the last one)", fontsize=7.0,
            color=MUTED, ha="center")

    # ---- 底注：一刀的内容 ----
    box(ax, 0.3, 0.15, 13.4, 0.85,
        "the swap (13 sites = ln_1, ln_2 in every block + ln_f):"
        "  nn.LayerNorm(C, eps=1e-5, weight+bias)  ->  RMSNorm(C, eps=1e-5, g only)\n"
        "params per site 2C -> C   |   topology / attn / mlp / head: untouched",
        fc="#fdf8f4", ec=ORANGE, lw=1.2, fs=7.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-2-2-swap-position.png"), facecolor="white")
    plt.close(fig)


# ---------------- 图 2.3：LN vs RMSNorm 消融曲线（自产：train EMA + 固定窗 eval 点） ----------------
def ema(vals, alpha=0.01):
    out, v = [], None
    for x in vals:
        v = x if v is None else alpha * x + (1 - alpha) * v
        out.append(v)
    return out


def fig_2_3(tier):
    report_path = os.path.join(LOG_DIR, f"ablation_norm_{tier}.json")
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    arms = [("base", BLUE, "LayerNorm (base arm)"), ("rmsnorm", ORANGE, "RMSNorm arm")]
    for arm, color, label in arms:
        curve_path = os.path.join(LOG_DIR, f"curve_{arm}_{tier}.csv")
        steps, losses = [], []
        with open(curve_path, newline="") as f:
            for row in csv.DictReader(f):
                steps.append(int(row["step"]))
                losses.append(float(row["train_loss"]))
        ax.plot(steps, ema(losses), color=color, lw=1.6, alpha=0.85, label=f"{label}, train (EMA)")
        ev = report["arms"][arm]["evals"]
        ax.plot([e["step"] for e in ev], [e["eval_loss"] for e in ev], "o", color=color,
                ms=5.5, mec="white", mew=0.6, label=f"{label}, fixed-window eval")
    a, b = report["arms"]["base"], report["arms"]["rmsnorm"]
    diff = abs(a["eval_final"] - b["eval_final"])
    band = 0.02
    verdict = (f"|diff| = {diff:.4f} nat  <=  pre-registered band {band} nat\n"
               f"-> equivalence holds at this tier" if diff <= band else
               f"|diff| = {diff:.4f} nat  >  pre-registered band {band} nat\n"
               f"-> outside band: single seed, no anomaly; qualified reading (see text)")
    txt = (f"final eval: LN {a['eval_final']:.4f} / RMS {b['eval_final']:.4f}\n"
           f"{verdict} ({tier}, {a['steps_done']} steps)")
    ax.text(0.985, 0.965, txt, transform=ax.transAxes, fontsize=8.0, color=DARK,
            ha="right", va="top",
            bbox=dict(fc="white", ec=GRID, lw=0.8, boxstyle="round,pad=0.4"))
    ax.set_xlabel("training step", fontsize=9)
    ax.set_ylabel("loss (nat)", fontsize=9)
    ax.set_xlim(0, a["steps_done"])
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.0, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-2-3-norm-ablation.png"), facecolor="white")
    plt.close(fig)
    print(f"[fig 2.3] tier={tier}: LN {a['eval_final']} / RMS {b['eval_final']} "
          f"|diff|={diff:.4f} nat; sec/step {a['sec_per_step_steady']} vs {b['sec_per_step_steady']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="auto", choices=["auto", "fast", "full"])
    args = ap.parse_args()
    tier = args.tier
    if tier == "auto":
        # full 档判据=各臂 steps_done==steps_planned（完整跑毕），不用字节阈值——完整但体积小的 full JSON 不得误判降档
        try:
            with open(os.path.join(LOG_DIR, "ablation_norm_full.json"), encoding="utf-8") as f:
                arms = json.load(f).get("arms", {})
            tier = "full" if arms and all(
                a.get("steps_done") == a.get("steps_planned") for a in arms.values()) else "fast"
        except (OSError, ValueError):
            tier = "fast"
    fig_2_1()
    fig_2_2_machine()
    fig_2_3(tier)
