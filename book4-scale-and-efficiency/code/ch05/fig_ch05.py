# fig_ch05.py —— Book4 ch5 制图：图 5.1 细粒度组合空间 + 图 5.2 共享/路由专家分工 + 图 5.3 四臂消融曲线
# 用途：图 5.1 = 正文 5.2 的组合数论证可视化（C(16,2)=120 vs C(64,8)=4,426,165,368——同激活预算下
#       可表达知识组合爆炸式增长，自绘示意级）；图 5.2 = 5.3 的机构图（路由专家 top-k + 共享专家
#       常驻，DSV2 路由口径：softmax 打分不归一×rsf，全框标形状）；图 5.3 = 5.7 四臂消融读数可视化
#       （②③ 实跑曲线喂 log/book4-ch05/，①④ 引用臂自批一 JSON 的 eval 点）。
# 所属章节：Book4 第 5 章（图 5.1 / 图 5.2 / 图 5.3）。
# 运行方式：cd 工作区根目录 && source env.sh && python code/ch05/fig_ch05.py
#           [--which all|comb|diagram|curve] [--src auto|fast|full]
# 产物：figures/fig-5-1-granularity-combination-space.png /
#       fig-5-2-shared-routed-experts.png / fig-5-3-granularity-ablation.png
# 图规：印刷规格 300dpi 白底；系列色 蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；线宽 2；
#       刻度 #898781、标注 #0b0b0b；图内英文、图注中文（图注在正文 markdown，不入图）。
import argparse
import csv
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "figures")
LOG_DIR = os.path.join(REPO_ROOT, "log", "book4-ch05")
CH02_JSON = os.path.join(REPO_ROOT, "log", "book4-ch02", "ablation_moe_full.json")
CH03_JSON = os.path.join(REPO_ROOT, "log", "book4-ch03", "ablation_balance_full.json")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e1e0d9"


def _expert_grid(ax, x0, y0, n_macro, split, cell, gap_g, gap_e, hit_set, ec, lw=1.0):
    """画专家群：n_macro×n_macro 个「原专家」位，每位内 split×split 个细分小格。

    hit_set = 高亮（被选中）的小格线性编号（按 macro 行优先、格内行优先）。
    返回 (总宽, 总高)。"""
    inner = split * cell + (split - 1) * gap_e
    for mi in range(n_macro * n_macro):
        mx = x0 + (mi % n_macro) * (inner + gap_g)
        my = y0 + (mi // n_macro) * (inner + gap_g)
        ax.add_patch(Rectangle((mx - 0.6, my - 0.6), inner + 1.2, inner + 1.2,
                               fill=False, ec=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=2))
        for si in range(split * split):
            sx = mx + (si % split) * (cell + gap_e)
            sy = my + (si // split) * (cell + gap_e)
            gid = mi * split * split + si
            hit = gid in hit_set
            ax.add_patch(Rectangle((sx, sy), cell, cell, fc=(ec if hit else "white"),
                                   ec=ec, lw=lw, alpha=(0.85 if hit else 0.9), zorder=3))
    return n_macro * inner + (n_macro - 1) * gap_g, n_macro * inner + (n_macro - 1) * gap_g


def fig_combination():
    """图 5.1：细粒度 vs 粗粒度组合空间（自绘示意级；激活预算两侧同为 2×FFN）。"""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), dpi=300)
    for ax in axes:
        ax.axis("off")

    # 左：粗粒度——16 个整 FFN 专家（split=1），top-2 任选 2 个高亮
    ax = axes[0]
    _expert_grid(ax, x0=4.0, y0=3.2, n_macro=4, split=1, cell=3.4, gap_g=1.7, gap_e=0.0,
                 hit_set={5, 10}, ec=ORANGE)
    ax.text(11.7, 20.9, "coarse: 16 whole-FFN experts, top-2", ha="center", fontsize=9.5, color=INK)
    ax.text(11.7, 1.1, r"$\binom{16}{2}=120$ combinations" + "\nactive budget = 2 × FFN",
            ha="center", va="bottom", fontsize=9, color=MUTED)
    ax.set_xlim(0, 23.4)
    ax.set_ylim(0, 22.6)

    # 右：细粒度——每个原专家切 4 份（m=4）→ 64 个 0.25×FFN 小专家，top-8
    ax = axes[1]
    _expert_grid(ax, x0=3.2, y0=3.2, n_macro=4, split=2, cell=1.25, gap_g=1.5, gap_e=0.5,
                 hit_set={5 * 4 + 1, 5 * 4 + 2, 10 * 4 + 0, 10 * 4 + 3,
                          2 * 4 + 2, 6 * 4 + 0, 9 * 4 + 3, 15 * 4 + 1}, ec=TEAL)
    ax.text(10.9, 20.9, "fine: split each ×4 ($m{=}4$) $\\to$ 64 experts $\\times\\,0.25$, top-8",
            ha="center", fontsize=9.5, color=INK)
    ax.text(10.9, 1.1, r"$\binom{64}{8}=4{,}426{,}165{,}368$ combinations" + "\nactive budget = 8 × 0.25 = 2 × FFN (same)",
            ha="center", va="bottom", fontsize=9, color=MUTED)
    ax.set_xlim(0, 21.8)
    ax.set_ylim(0, 22.6)

    fig.suptitle("Same activation budget, exponentially more knowledge combinations "
                 "(dashed = original expert, filled = activated)", fontsize=10, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-5-1-granularity-combination-space.png")
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 5.1] {out}")


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=8.0, lw=1.6, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc, zorder=4)
    return x + w / 2, y + h / 2


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=lw, zorder=2))


def fig_diagram():
    """图 5.2：DeepSeekMoE 块——路由专家（细而多，top-k）+ 共享专家（常驻），全框标形状。"""
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    # 输入与展平
    _, cy_in = box(ax, 0.5, 21, 10.5, 8, "input  x\n$(B, n, d)$", ec=BLUE)
    _, cy_flat = box(ax, 14.5, 21, 9.5, 8, "flatten\n$(N,\\ d)$", ec=BLUE, fc="#f3f8fe")
    arrow(ax, 11, cy_in, 14.5, cy_flat)
    ax.text(19.2, 17.6, "$N = B{\\cdot}n$", ha="center", va="top", fontsize=7.5, color=MUTED, style="italic")

    # 上路（橙）：DSV2 路由三步——fp32 打分 / softmax / top-k（不归一 ×rsf）
    _, cy_r = box(ax, 29, 38.5, 16.5, 8, "router  $W_g$ (fp32)\nscore $(N, E)$", ec=ORANGE, fc="#fdf1ec", fs=7.5)
    arrow(ax, 24, cy_flat + 3.4, 29.5, cy_r - 1.4)
    _, cy_sm = box(ax, 49, 38.5, 14.5, 8, "softmax fp32\n$(N, E)$", ec=ORANGE, fs=7.5)
    arrow(ax, 45.5, cy_r, 49, cy_sm)
    _, cy_tk = box(ax, 67.5, 38.5, 19, 8, "top-k $\\to$ $w{\\times}\\mathrm{rsf}$\n$w,\\ idx$ $(N, k)$  (no renorm)",
                   ec=ORANGE, fc="#fdf1ec", fs=7.0)
    arrow(ax, 63.5, cy_sm, 67.5, cy_tk, color=ORANGE)

    # 中路（青）：路由专家库（细而多）
    ax.add_patch(FancyBboxPatch((29, 8.5), 34, 21.5, boxstyle="round,pad=0.02", ec=TEAL,
                                fc="#f2faf7", lw=1.4, ls="--", zorder=1))
    ax.text(46, 27.7, "E routed experts × narrow SwiGLU (fused 3D)", ha="center", va="center",
            fontsize=8, color=TEAL)
    arrow(ax, 24, cy_flat - 3.4, 31, 20.5)
    _, cy_g = box(ax, 31, 14.5, 11.5, 7.5, "gather\n$(n_e, d)$", ec=TEAL, fs=7.5)
    ax.text(36.7, 13.2, "$n_e \\approx N{\\cdot}k/E$", ha="center", va="top", fontsize=7.0,
            color=MUTED, style="italic")
    _, cy_ffn = box(ax, 45.5, 14.5, 15.5, 7.5, "SwiGLU$_e$ (w)\n$(n_e{,}d){\\to}(n_e{,}d)$",
                    ec=TEAL, fc="#f2faf7", fs=7.5)
    arrow(ax, 42.5, cy_g, 45.5, cy_ffn)
    _, cy_mg = box(ax, 46.5, 6.0, 14.5, 6.5, "$y_e{\\cdot}w_e$, scatter\n$(N, d)$", ec=TEAL, fs=7.0)
    arrow(ax, 53, cy_ffn - 3.7, 53.7, cy_mg + 3.2)
    ax.add_patch(FancyArrowPatch((77, 38.5), (61.5, cy_mg + 1.5), arrowstyle="-|>", mutation_scale=11,
                                 color=YELLOW, lw=1.8, zorder=2))
    ax.text(73.5, 25.5, "gate weights $(N, k)$", ha="center", va="center", fontsize=7.5,
            color="#a06b00", rotation=27)

    # 下路（黄底蓝边）：共享专家——每 token 无条件激活（吃未展平原输入，HF residuals 路径同构）
    _, cy_sh = box(ax, 29, 0.5, 26, 6.5, "shared expert (always-on)\nSwiGLU width $n_s{\\cdot}w$: $(B,n,d){\\to}(B,n,d)$",
                   ec=BLUE, fc="#fdf8ea", fs=7.0)
    ax.add_patch(FancyArrowPatch((5.75, 21), (5.75, 3.75), arrowstyle="-", color=MUTED, lw=1.4, zorder=2))
    arrow(ax, 5.75, 3.75, 29, 3.75)
    ax.text(16.5, 5.0, "every token,\nunconditionally", ha="center", va="bottom", fontsize=6.8,
            color=MUTED, style="italic")

    # 汇合：路由输出 + 共享输出
    _, cy_sum = box(ax, 78, 12.5, 9.5, 8, "sum\n$(B, n, d)$", ec=BLUE)
    arrow(ax, 61, cy_mg + 3.0, 78, cy_sum - 1.2)
    ax.add_patch(FancyArrowPatch((55.5, 0.5), (83.0, 0.5), arrowstyle="-", color=MUTED, lw=1.4, zorder=2))
    arrow(ax, 83.0, 0.5, 83.0, 12.5)
    _, cy_out = box(ax, 89.5, 21, 10, 8, "output  y\n$(B, n, d)$", ec=BLUE)
    arrow(ax, 85, cy_sum + 3.4, 89.5, cy_out - 1.4)

    ax.set_title("DeepSeekMoE block: fine-grained routed experts (top-k of E) + always-on shared expert, "
                 "DSV2 routing (softmax, no renorm)", fontsize=9.5, color=INK, pad=10)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-5-2-shared-routed-experts.png")
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 5.2] {out}")


def fig_curve(src):
    """图 5.3：四臂消融（②③ 实跑 train+eval；①④ 引用臂 eval 点）——等激活模块档 2000 步。"""
    src_used = src
    if src == "auto":
        src_used = "full" if os.path.exists(os.path.join(LOG_DIR, "curve_fine_aux_full.csv")) else "fast"
        if src_used == "fast":
            print("[提示] full 曲线未落地，回落 fast 档（正文引用需标注档位）")
    with open(os.path.join(LOG_DIR, f"ablation_granularity_{src_used}.json"), encoding="utf-8") as f:
        rep = json.load(f)
    with open(CH02_JSON, encoding="utf-8") as f:
        dense_evals = json.load(f)["arms"]["dense"]["evals"]
    with open(CH03_JSON, encoding="utf-8") as f:
        coarse_evals = json.load(f)["arms"]["aux"]["evals"]

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=300)
    # 实跑两臂：train 细线 + eval 点（线与点各一条图例，ch2 图 2.2 同款）
    for arm, label, color in (("fine_aux", "② fine E16/top4 +shared +aux", TEAL),
                              ("fine_noaux", "③ fine + noaux", YELLOW)):
        path = os.path.join(LOG_DIR, f"curve_{arm}_{src_used}.csv")
        steps, losses = [], []
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                steps.append(int(row["step"]))
                losses.append(float(row["train_loss"]))
        ax.plot(steps, losses, color=color, lw=1.0, alpha=0.55, label=f"{label} (train)",
                solid_capstyle="round", zorder=2)
        ev = rep["arms"][arm]["evals"]
        ax.errorbar([e["step"] for e in ev], [e["eval_loss"] for e in ev],
                    yerr=[e.get("eval_se", 0.0) for e in ev], color=color, marker="o", ms=6,
                    lw=0, elinewidth=1.0, capsize=2.5, label=f"{label} (eval)", zorder=5)
    # 引用两臂：仅 eval 点（批一 full 产物）
    for ev, label, color, mk in ((coarse_evals, "① coarse E8/top2 +aux (ref ch3)", ORANGE, "s"),
                                 (dense_evals, "④ dense d_ff=1408 (ref ch2)", BLUE, "D")):
        ax.errorbar([e["step"] for e in ev], [e["eval_loss"] for e in ev],
                    yerr=[e.get("eval_se", 0.0) for e in ev], color=color, marker=mk, ms=6,
                    lw=1.6, elinewidth=1.0, capsize=2.5, label=label, zorder=4)

    ax.set_xlabel("step", fontsize=9, color=MUTED)
    ax.set_ylabel("eval loss (nats, 49 fixed windows)", fontsize=9, color=MUTED)
    ax.tick_params(colors="#898781", labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#898781")
    ax.grid(True, color=GRIDC, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(f"four arms, equal-activation tier (single seed, {src_used} run)",
                 fontsize=9, color=INK, pad=8)
    ax.legend(fontsize=7.0, frameon=False, loc="upper right", ncols=2)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-5-3-granularity-ablation.png")
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 5.3] {out}（档位 {src_used}）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Book4 ch5 制图（图 5.1 组合空间 / 图 5.2 机构图 / 图 5.3 消融曲线）")
    ap.add_argument("--which", default="all", choices=["all", "comb", "diagram", "curve"])
    ap.add_argument("--src", default="auto", choices=["auto", "fast", "full"],
                    help="图 5.3 数据档位（auto=full 优先回落 fast）")
    args = ap.parse_args()
    if args.which in ("all", "comb"):
        fig_combination()
    if args.which in ("all", "diagram"):
        fig_diagram()
    if args.which in ("all", "curve"):
        fig_curve(args.src)
