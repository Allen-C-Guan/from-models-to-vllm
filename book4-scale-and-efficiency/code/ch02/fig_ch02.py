# fig_ch02.py —— Book4 ch2 制图：图 2.1 MoE 块数据流（自绘示意级）+ 图 2.2 dense vs MoE 基线曲线
# 用途：图 2.1 = 正文 2.2/2.7 的机构总图（router 打分 (N,E)、top-k 掩码 (N,k)、gather-scatter
#       全标形状——matplotlib 框线示意级）；图 2.2 = 2.7 消融读数可视化（曲线喂
#       log/book4-ch02/curve_{dense,moe}_{src}.csv + ablation_moe_{src}.json 的 eval 点，
#       --src 自动：full 优先、缺则回落 fast 并在命令行提示）。
# 所属章节：Book4 第 2 章（图 2.1 / 图 2.2）。
# 运行方式：cd 工作区根目录 && source env.sh && python code/ch02/fig_ch02.py
#           [--which all|diagram|curve] [--src auto|fast|full]
# 产物：figures/fig-2-1-moe-block-dataflow.png / fig-2-2-dense-vs-moe.png
# 图规：印刷规格 300dpi 白底；系列色 蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；线宽 2；
#       刻度 #898781、标注 #0b0b0b；图内英文、图注中文（图注在正文 markdown，不入图）。
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "figures")
LOG_DIR = os.path.join(REPO_ROOT, "log", "book4-ch02")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e1e0d9"


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=8.5, lw=1.6, style="round,pad=0.02", tc=INK):
    """圆角框 + 居中文字（label 支持 \n）。返回 (cx, cy) 中心坐标。"""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, ec=ec, fc=fc, lw=lw,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc, zorder=4)
    return x + w / 2, y + h / 2


def arrow(ax, x0, y0, x1, y1, label=None, color=MUTED, fs=7.5, dx=0.0, dy=0.06, lw=1.4):
    """细箭头 + 可选标签（标签放在中点偏移处）。"""
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=lw, zorder=2))
    if label:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, label, ha="center", va="bottom",
                fontsize=fs, color=MUTED, zorder=4)


def fig_dataflow():
    """图 2.1：MoE 块数据流（自绘示意级；通用形状 B,n,d / N=B·n / E / k / w 全标）。"""
    fig, ax = plt.subplots(figsize=(9.0, 4.3), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    # 主干：输入 → 展平 →（路由 / 专家两路）→ 加权合并 → 输出
    _, cy_in = box(ax, 0.5, 19, 11, 8, "input  x\n$(B, n, d)$", ec=BLUE)
    _, cy_flat = box(ax, 15.5, 19, 9.5, 8, "flatten\n$(N,\\ d)$", ec=BLUE, fc="#f3f8fe")
    arrow(ax, 11.5, cy_in, 15.5, cy_flat, "N = B·n", dx=0, dy=0.5)
    ax.text(20.2, 15.4, "routing decides\nper token", ha="center", va="top", fontsize=7.5,
            color=MUTED, style="italic")

    # 上路（橙）：router 打分 → softmax → top-k
    _, cy_r = box(ax, 30, 33, 15, 8, "router  $W_g$\nscore $(N, E)$", ec=ORANGE, fc="#fdf1ec")
    arrow(ax, 25, cy_flat + 3.2, 30, cy_r - 1.2, None)
    _, cy_sm = box(ax, 49.5, 33, 12.5, 8, "softmax\nfp32 $(N, E)$", ec=ORANGE)
    arrow(ax, 45, cy_r, 49.5, cy_sm, None)
    _, cy_tk = box(ax, 66, 33, 17.5, 8, "top-k mask $\\to$ renorm\n$w_{tk},\\ idx_{tk}$ $(N, k)$",
                   ec=ORANGE, fc="#fdf1ec")
    arrow(ax, 62, cy_sm, 66, cy_tk, "k of E kept", dx=0, dy=0.5)

    # 下路（青）：专家库容器（gather → SwiGLU）
    ax.add_patch(FancyBboxPatch((30, 4), 35, 22, boxstyle="round,pad=0.02", ec=TEAL,
                                fc="#f2faf7", lw=1.4, ls="--", zorder=1))
    ax.text(47.5, 24.0, "expert bank  (E × SwiGLU, fused 3D weights)", ha="center", va="center",
            fontsize=8, color=TEAL)
    arrow(ax, 25, cy_flat - 3.2, 32, 16.5, None)
    _, cy_g = box(ax, 32, 10, 12, 8, "gather\n$(n_e, d)$", ec=TEAL)
    ax.text(38, 8.6, "$n_e \\approx N{\\cdot}k/E$", ha="center", va="top", fontsize=7.5,
            color=MUTED, style="italic")
    _, cy_ffn = box(ax, 47, 10, 16.5, 8, "SwiGLU$_e$\n$(n_e{,}2w){\\to}(n_e{,}w){\\to}(n_e{,}d)$",
                    ec=TEAL, fc="#f2faf7", fs=8)
    arrow(ax, 44, cy_g, 47, cy_ffn, None)

    # 加权合并（青框）：专家输出 × 门控权重 → index_add 汇回 token 维
    _, cy_mg = box(ax, 67, 10, 16, 9, "merge\n$y_e \\cdot g_e$, index_add\n$(N, d)$", ec=TEAL, fs=7.5)
    arrow(ax, 63.5, cy_ffn, 67, cy_mg, None)
    ax.add_patch(FancyArrowPatch((75, 33), (75, 19), arrowstyle="-|>", mutation_scale=11,
                                 color=YELLOW, lw=1.8, zorder=2))
    ax.text(76, 26.3, "gate weights $(N, k)$", ha="left", va="center", fontsize=7.5, color="#a06b00")

    # 输出
    _, cy_out = box(ax, 87.5, 19, 11.5, 8, "output  y\n$(B, n, d)$", ec=BLUE)
    arrow(ax, 83, cy_mg + 2.0, 87.5, cy_out - 1.6, "reshape", dx=1.4, dy=-0.2)

    ax.set_title("Sparse MoE block: router scores $(N,E)$, top-k selects k of E, "
                 "gather-scatter dispatches tokens", fontsize=10, color=INK, pad=10)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-2-1-moe-block-dataflow.png")
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 2.1] {out}")


def fig_curve(src):
    """图 2.2：dense vs MoE 基线曲线（train 曲线 + eval 点；两臂同数据同种子）。"""
    src_used = src
    if src == "auto":
        src_used = "full" if os.path.exists(os.path.join(LOG_DIR, "curve_dense_full.csv")) else "fast"
        if src_used == "fast":
            print("[提示] full 曲线未落地，回落 fast 档（正文引用需标注档位）")
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=300)
    ev_points = {}
    with open(os.path.join(LOG_DIR, "ablation_moe_%s.json" % src_used), encoding="utf-8") as f:
        rep = json.load(f)
    for arm, color in (("dense", BLUE), ("moe", ORANGE)):
        path = os.path.join(LOG_DIR, f"curve_{arm}_{src_used}.csv")
        steps, losses = [], []
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                steps.append(int(row["step"]))
                losses.append(float(row["train_loss"]))
        ax.plot(steps, losses, color=color, lw=2, label=f"{arm} (train)",
                solid_capstyle="round")
        ev = rep["arms"][arm].get("evals") or []
        ev_points[arm] = ([e["step"] for e in ev], [e["eval_loss"] for e in ev],
                          [e.get("eval_se", 0.0) for e in ev], color)

    for arm, (xs, ys, es, color) in ev_points.items():
        ax.errorbar(xs, ys, yerr=es, color=color, marker="o", ms=6, lw=0, elinewidth=1.0,
                    capsize=2.5, label=f"{arm} (eval, 49 windows)", zorder=5)

    ax.set_xlabel("step", fontsize=9, color=MUTED)
    ax.set_ylabel("cross-entropy loss (nats)", fontsize=9, color=MUTED)
    ax.tick_params(colors="#898781", labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#898781")
    ax.grid(True, color=GRIDC, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ps = rep["arms"]["dense"].get("params_total")
    pa = rep["arms"]["dense"].get("params_active")
    pm = rep["arms"]["moe"].get("params_total")
    ax.set_title(f"dense ($N$={ps/1e6:.1f}M) vs MoE (total {pm/1e6:.1f}M, "
                 f"active {pa/1e6:.1f}M) — equal-activation, single seed, {src_used} run",
                 fontsize=9, color=INK, pad=8)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-2-2-dense-vs-moe.png")
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 2.2] {out}（档位 {src_used}）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Book4 ch2 制图（图 2.1 数据流 / 图 2.2 消融曲线）")
    ap.add_argument("--which", default="all", choices=["all", "diagram", "curve"])
    ap.add_argument("--src", default="auto", choices=["auto", "fast", "full"],
                    help="图 2.2 数据档位（auto=full 优先回落 fast）")
    args = ap.parse_args()
    if args.which in ("all", "diagram"):
        fig_dataflow()
    if args.which in ("all", "curve"):
        fig_curve(args.src)
