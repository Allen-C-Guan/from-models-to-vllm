# fig_ch04.py —— Book4 ch4 制图：图 4.1 Mixtral 整机结构（自绘重制，示意级）+ 图 4.2 自家版路由统计
# 用途：图 4.1 = 正文 4.1/4.4 的整机图（MoE 块放大 + 四插槽标注——「MoE 只动 FFN 插槽」的图证；
#       Mixtral 论文无官方整机架构图，本图按 checkpoint config + 论文 §2 自绘重制）；
#       图 4.2 = 4.7 节路由统计三图（分工矩阵热图 / 位置局部性曲线 / 层间重复率），
#       喂 log/book4-ch04/route_analysis_{src}.json（--src 缺省 run1）。
# 所属章节：Book4 第 4 章（图 4.1 / 图 4.2）。
# 运行方式：cd 工作区根目录 && source env.sh && python code/ch04/fig_ch04.py
#           [--which all|machine|stats] [--src run1]
# 产物：figures/fig-4-1-mixtral-machine.png / fig-4-2-routing-stats.png
# 图规：印刷规格 300dpi 白底；系列色 蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；
#       顺序型热图蓝单色渐变 #cde2fb→#0d366b；线宽 2、标记 ≥6pt；刻度 #898781、标注 #0b0b0b；
#       图内英文、图注中文（图注在正文 markdown，不入图）。
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "figures")
LOG_DIR = os.path.join(REPO_ROOT, "log", "book4-ch04")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e1e0d9"
GREY = "#898781"
HEAT_LO, HEAT_HI = "#cde2fb", "#0d366b"      # 顺序型蓝渐变端点


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=8.0, lw=1.5, ls="-", tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc,
                                lw=lw, ls=ls, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc, zorder=4)
    return x + w / 2, y + h / 2


def arrow(ax, x0, y0, x1, y1, label=None, color=MUTED, fs=7.0, dx=0.0, dy=0.06, lw=1.3,
          style="italic"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=lw, zorder=2))
    if label:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, label, ha="center", va="bottom",
                fontsize=fs, color=MUTED, style=style, zorder=4)


def fig_machine():
    """图 4.1：Mixtral 整机结构（自绘重制，示意级）——整机 | Block 放大 | MoE 块再放大。"""
    fig, ax = plt.subplots(figsize=(10.0, 4.9), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 54)
    ax.axis("off")

    # ---- 左列：整机（embed → ×32 Block → norm → lm_head） ----
    ax.text(9, 52.6, "Mixtral 8x7B (L=32)", ha="center", fontsize=9, color=INK, weight="bold")
    _, cy_tok = box(ax, 2.5, 45.5, 13, 5, "tokens $(B,n)$", ec=BLUE, fs=8)
    _, cy_emb = box(ax, 2.5, 37.5, 13, 5.5, "embed\n$(B,n,4096)$", ec=BLUE, fc="#f3f8fe", fs=8)
    arrow(ax, 9, cy_tok - 2.5, 9, 43, "V=32000 lookup")
    box(ax, 2.5, 26.5, 13, 8.5, "decoder block\n$\\times$ 32", ec=GREY, fc="white", fs=8.5, lw=1.3)
    arrow(ax, 9, cy_emb - 2.75, 9, 35)
    arrow(ax, 9, 26.5, 9, 23.5)
    _, cy_nm = box(ax, 2.5, 18, 13, 5, "RMSNorm", ec=BLUE, fs=8)
    arrow(ax, 9, cy_nm - 2.5, 9, 13.5)
    box(ax, 2.5, 5.5, 13, 7, "lm_head untied\n$(B,n,4096){\\to}(B,n,32000)$", ec=BLUE, fs=7.5)
    ax.text(9, 3.4, "E+U: 2×Vd = 262M", ha="center", fontsize=7, color=MUTED, style="italic")

    # ---- 中列：Block 放大（残差河横向） ----
    ax.text(41, 52.6, "one decoder block  (zoom 1)", ha="center", fontsize=9, color=INK,
            weight="bold")
    ax.add_patch(FancyBboxPatch((19.5, 14), 43, 34, boxstyle="round,pad=0.02", ec=GREY,
                                fc="#fcfcfb", lw=1.1, ls=":", zorder=1))
    _, cy_x = box(ax, 21, 41.5, 8.5, 5, "$x$\n$(B,n,4096)$", ec=BLUE, fs=7.5)
    _, cy_n1 = box(ax, 33, 41.5, 8.5, 5, "RMSNorm\n(slot ①)", ec=BLUE, fc="#f3f8fe", fs=7.5)
    arrow(ax, 29.5, cy_x, 33, cy_n1)
    _, cy_at = box(ax, 45, 40.5, 16.5, 7,
                   "GQA attn (slot ④)\n32 Q / 8 KV × 128 = Llama-2-70B", ec=GREY, fc="white", fs=7)
    arrow(ax, 41.5, cy_n1, 45, cy_at)
    ax.text(53.2, 36.6, "RoPE $\\theta$=1e6 (slot ③) — unchanged", ha="center", fontsize=6.6,
            color=MUTED, style="italic")
    box(ax, 45, 26.5, 7, 4.5, "$\\oplus$", ec=MUTED, fc="white", fs=9)
    arrow(ax, 53.2, 40.5, 48.5, 31, None)
    arrow(ax, 25.2, cy_x - 2.5, 25.2, 28.7, "residual", dx=-1.2, dy=0.0)
    ax.add_patch(FancyArrowPatch((25.2, 28.7), (45, 28.7), arrowstyle="-", color=MUTED,
                                 lw=1.1, zorder=2))
    _, cy_n2 = box(ax, 33, 21, 8.5, 5, "RMSNorm\n(slot ①)", ec=BLUE, fc="#f3f8fe", fs=7.5)
    arrow(ax, 48.5, 26.5, 41, 24, None)
    _, cy_moe = box(ax, 45, 19.5, 16.5, 6.5, "MoE block  (slot ②)\n$E$=8 experts, top-2",
                    ec=ORANGE, fc="#fdf1ec", fs=7.5, lw=1.8)
    arrow(ax, 41.5, cy_n2, 45, cy_moe)
    box(ax, 45, 16.2, 7, 3.6, "$\\oplus$", ec=MUTED, fc="white", fs=9)
    arrow(ax, 53.2, 19.5, 48.5, 17.9, None)
    ax.add_patch(FancyArrowPatch((25.2, 28.7), (25.2, 18), arrowstyle="-", color=MUTED,
                                 lw=1.1, zorder=2))
    ax.add_patch(FancyArrowPatch((25.2, 18), (45, 18), arrowstyle="-", color=MUTED, lw=1.1, zorder=2))
    _, cy_o = box(ax, 55.5, 15.5, 7, 5, "$x$\n$(B,n,4096)$", ec=BLUE, fs=7.5)
    arrow(ax, 52, 17.9, 55.5, 17.9, None)
    # 被换下的 dense FFN 虚影
    box(ax, 21, 43.2, 0.01, 0.01, "", ec="white")  # noqa: F841 keep zorder stable
    ax.add_patch(FancyBboxPatch((33, 8.5), 28.5, 5.5, boxstyle="round,pad=0.02", ec=GREY,
                                fc="white", lw=1.1, ls="--", zorder=1))
    ax.text(47.2, 11.2, "dense SwiGLU  4096→14336→4096\n(Mistral 7B part, replaced)",
            ha="center", va="center", fontsize=7, color=GREY, style="italic")
    arrow(ax, 53.2, cy_moe + 3.25, 47.2, 14.0, "replaced", color=GREY, fs=6.6, dx=4.2, dy=0.2)

    # ---- 右列：MoE 块再放大 ----
    ax.text(84.5, 52.6, "MoE block (zoom 2)", ha="center", fontsize=9, color=INK, weight="bold")
    ax.add_patch(FancyBboxPatch((66, 8.5), 32.5, 40.5, boxstyle="round,pad=0.02", ec=ORANGE,
                                fc="#fefaf8", lw=1.2, ls=":", zorder=1))
    _, cy_h = box(ax, 67.5, 41.5, 10.5, 5, "$h$\n$(N,4096)$", ec=BLUE, fs=7.5)
    ax.text(72.7, 39.4, "$N = B{\\cdot}n$ (flatten)", ha="center", fontsize=6.6, color=MUTED,
            style="italic")
    _, cy_rt = box(ax, 82, 41.5, 14.5, 6.5, "router $W_g\\,(8,4096)$\n$(N,4096){\\to}(N,8)$",
                   ec=ORANGE, fc="#fdf1ec", fs=7)
    arrow(ax, 78, cy_h, 82, cy_rt)
    _, cy_tk = box(ax, 82, 31, 14.5, 6.5, "softmax fp32\n$\\to$ top-2 + renorm\n$(N,2)$",
                   ec=ORANGE, fs=6.8)
    arrow(ax, 89.2, 41.5, 89.2, 37.5, None)
    _, cy_ga = box(ax, 67.5, 28, 10.5, 5, "gather\n$(n_e,4096)$", ec=TEAL, fc="#f2faf7", fs=7.5)
    arrow(ax, 72.7, 39.4 - 0.8, 72.7, 33, None)
    _, cy_ex = box(ax, 67.5, 16.5, 29, 8,
                   "8 × expert SwiGLU (slot ②)\n$(n_e,4096){\\to}(n_e,14336){\\to}(n_e,4096)$",
                   ec=TEAL, fc="#f2faf7", fs=7.5)
    arrow(ax, 72.7, 28, 72.7, 24.5, None)
    ax.add_patch(FancyArrowPatch((89.2, 31), (89.2, 24.5), arrowstyle="-|>", mutation_scale=10,
                                 color=YELLOW, lw=1.6, zorder=2))
    ax.text(89.9, 27.6, "gate $(N,2)$", ha="left", va="center", fontsize=6.6, color="#a06b00")
    _, cy_mg = box(ax, 67.5, 10.2, 29, 4.6,
                   "merge  $y=\\sum_e g_e E_e(x)$, scatter $(N,4096)$", ec=TEAL, fs=7)
    arrow(ax, 82, 16.5, 82, 14.8, None)

    # ---- 放大镜连线：左列 block ↔ 中列 zoom1 ↔ 右列 zoom2 ----
    ax.add_patch(FancyArrowPatch((15.5, 30.7), (19.5, 30.7), arrowstyle="-", color=GREY,
                                 lw=1.0, ls=(0, (2, 2)), zorder=2))
    ax.add_patch(FancyArrowPatch((61.5, 22.7), (66, 28), arrowstyle="-", color=GREY,
                                 lw=1.0, ls=(0, (2, 2)), zorder=2, connectionstyle="arc3,rad=0.15"))
    ax.text(63.9, 24.3, "zoom", fontsize=6.4, color=GREY, style="italic")

    # ---- 插槽徽标注记 ----
    ax.text(50, 1.8, "slots: ① RMSNorm  ② SwiGLU (lives inside each expert)  ③ RoPE ④ GQA "
                     "— only the FFN slot was swapped", ha="center", fontsize=7.6, color=INK)
    ax.set_title("Mixtral 8x7B: LLaMA-style skeleton with the FFN slot replaced by an 8-expert MoE "
                 "(per layer)", fontsize=10, color=INK, pad=10)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-4-1-mixtral-machine.png")
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 4.1] {out}")


def fig_stats(src):
    """图 4.2：自家版路由统计三图（喂 route_analysis_{src}.json）。"""
    path = os.path.join(LOG_DIR, f"route_analysis_{src}.json")
    with open(path, encoding="utf-8") as f:
        rep = json.load(f)
    fin, init = rep["final"], rep.get("init") or {}
    L = len(fin["repetition_per_layer"])
    base = fin["baseline_uniform"]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.5), dpi=300)

    # (a) 分工矩阵热图（顺序蓝渐变，行归一化）
    ax = axes[0]
    frac = np.array(fin["assignment_matrix"]["frac"])
    names = fin["assignment_matrix"]["type_names"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seqblue", [HEAT_LO, HEAT_HI])
    im = ax.imshow(frac, cmap=cmap, vmin=0.0, aspect="auto",
                   vmax=max(0.16, float(frac.max()) * 1.05))
    for i in range(frac.shape[0]):
        for j in range(frac.shape[1]):
            ax.text(j, i, f"{frac[i, j]:.2f}", ha="center", va="center", fontsize=6.2,
                    color="white" if frac[i, j] > float(frac.max()) * 0.62 else INK)
    ax.set_xticks(range(frac.shape[1]), [f"e{j}" for j in range(frac.shape[1])])
    ax.set_yticks(range(frac.shape[0]), names)
    ax.tick_params(colors=GREY, labelsize=7)
    ax.set_xlabel("expert (first choice)", fontsize=8, color=MUTED)
    ax.set_title("(a) token type × expert\nrow-normalized, all layers", fontsize=8, color=INK)

    # (b) 位置局部性：逐层连续 token 首选项重复率
    ax = axes[1]
    xs = list(range(L))
    ax.axhline(base, color=ORANGE, lw=1.6, ls="--", label=f"uniform baseline 1/E = {base:.3f}")
    if init:
        ax.plot(xs, init["repetition_per_layer"], color=GREY, lw=1.6, ls=":", marker="o", ms=4,
                label="init (before training)")
    ax.plot(xs, fin["repetition_per_layer"], color=BLUE, lw=2, marker="o", ms=6,
            label=f"final ({rep['steps']} steps)")
    for x, y in zip(xs, fin["repetition_per_layer"]):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=6.4, color=INK)
    ax.set_xlabel("layer", fontsize=8, color=MUTED)
    ax.set_ylabel("consecutive-token repetition", fontsize=8, color=MUTED)
    ax.set_title("(b) positional locality\n(Mixtral Table-5 metric)", fontsize=8, color=INK)
    ax.legend(fontsize=6.2, frameon=False, loc="lower right")

    # (c) 层间重复率（本书补充口径）
    ax = axes[2]
    xs2 = [l + 0.5 for l in range(L - 1)]
    ax.axhline(base, color=ORANGE, lw=1.6, ls="--", label=f"baseline 1/E = {base:.3f}")
    if init:
        ax.plot(xs2, init["cross_layer_repetition"], color=GREY, lw=1.6, ls=":", marker="s",
                ms=4, label="init")
    ax.plot(xs2, fin["cross_layer_repetition"], color=TEAL, lw=2, marker="s", ms=6,
            label=f"final ({rep['steps']} steps)")
    for x, y in zip(xs2, fin["cross_layer_repetition"]):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=6.4, color=INK)
    ax.set_xlabel("layer boundary (l → l+1)", fontsize=8, color=MUTED)
    ax.set_ylabel("cross-layer repetition", fontsize=8, color=MUTED)
    ax.set_title("(c) same token, same expert?\n(this-book metric, not in paper)", fontsize=8,
                 color=INK)
    ax.legend(fontsize=6.2, frameon=False, loc="lower right")

    for ax in axes[1:]:
        ax.tick_params(colors=GREY, labelsize=7)
        ax.grid(True, color=GRIDC, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GREY)
        ax.set_ylim(0, max(0.30, max(fin["repetition_per_layer"]) * 1.25))
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-4-2-routing-stats.png")
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 4.2] {out}（档位 {src}）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Book4 ch4 制图（图 4.1 整机 / 图 4.2 路由统计）")
    ap.add_argument("--which", default="all", choices=["all", "machine", "stats"])
    ap.add_argument("--src", default="run1", help="route_analysis 产物后缀")
    args = ap.parse_args()
    if args.which in ("all", "machine"):
        fig_machine()
    if args.which in ("all", "stats"):
        fig_stats(args.src)
