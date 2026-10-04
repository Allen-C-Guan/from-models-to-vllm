# fig_ch08.py —— Book4 ch8 制图：图 8.1 MTP 模块结构 + off-by-one 三视图 / 图 8.2 双支路 loss 曲线
# 用途：图 8.1 = 正文 8.4 的 DSV3 式 MTP 模块（D=1）数据流（左半，全框标形状——共享件高亮）
#       与 8.6 的 off-by-one 三视图（右半，防坑主图：三个视图等长 T-2、互相恰错一位）；
#       图 8.2 = 8.6 的 λ=0.3/λ=0.0 双臂双支路 loss 曲线（读 log/book4-ch08/mtp_fast.json
#       与 curve_{lam30,lam00}_fast.csv——批二 P11 前置产物，正文数字同源）。
# 所属章节：Book4 第 8 章（图 8.1 / 图 8.2）。
# 运行方式：cd 工作区根目录 && source env.sh && python code/ch08/fig_ch08.py
#           [--which all|diagram|curve] [--src auto|fast|full]
# 产物：figures/fig-8-1-mtp-module-off-by-one.png / fig-8-2-mtp-dual-branch-loss.png
# 图规：印刷规格 300dpi 白底；系列色 蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；线宽 2；
#       刻度 #898781、标注 #0b0b0b；图内英文、图注中文（图注在正文 markdown，不入图）。
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "figures")
LOG_DIR = os.path.join(REPO_ROOT, "log", "book4-ch08")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e1e0d9"


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=7.0, lw=1.5, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc,
                                lw=lw, ls=ls, mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs,
            color=INK, zorder=4)
    return x + w / 2, y + h / 2


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.3, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=lw, linestyle=ls, zorder=2))


def fig_diagram():
    """图 8.1：左=MTP 模块数据流（DSV3 Eq 21-23，D=1）；右=off-by-one 三视图。"""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 4.9), dpi=300,
                                   gridspec_kw={"width_ratios": [1.35, 1.0]})
    for ax in (axL, axR):
        ax.axis("off")

    # ---------------- 左：模块数据流 ----------------
    axL.set_xlim(0, 118)
    axL.set_ylim(-2, 58)
    # 主干三件
    _, cy_tok = box(axL, 1, 47, 15, 7.5, "tokens $x_{in}$\n$(B, n)$ · $n{=}T{-}2$", fs=7.0)
    _, cy_bb = box(axL, 20, 47, 21, 7.5, "shared trunk ($L$ layers)\n$(B,n,d)\\to(B,n,d)$", fs=7.0)
    _, cy_h = box(axL, 45, 47, 9, 7.5, "$h$\n$(B,n,d)$", fs=7.0)
    arrow(axL, 16, cy_tok, 20, cy_bb)
    arrow(axL, 41, cy_bb, 45, cy_h)
    # 共享头（黄底=两支路共用同一件）
    _, cy_head = box(axL, 66, 38, 17, 7, "shared head\n$(B,n,d){\\to}(B,n,V)$",
                     fc="#fdf8ea", lw=1.8, fs=7.0)
    arrow(axL, 54, cy_h, 66, cy_head + 1.6)
    _, cy_lmain = box(axL, 92, 47, 25, 7.5,
                      "logits$_{main}$ $(B,n,V)$\nCE vs $y_{main}$ ($t_{i+1}$)", fs=7.0)
    arrow(axL, 83, cy_head + 3.2, 92, cy_lmain - 1.2)
    axL.text(101, 44.0, "CE$_{main}$", ha="center", fontsize=7.5, color=INK)

    # MTP 模块容器
    axL.add_patch(FancyBboxPatch((14, 3), 76, 30, boxstyle="round,pad=0.02", ec=TEAL,
                                 fc="none", lw=1.3, ls=(0, (4, 3)), zorder=1))
    axL.text(52, 30.9, "MTP module (D=1) — DSV3 Eq (21)-(23)", ha="center", fontsize=7.8,
             color=TEAL)
    _, cy_hn = box(axL, 17, 21.5, 17, 6.5, "RMSNorm($h$)\n$(B,n,d)$", ec=TEAL, fs=7.0)
    _, cy_emb = box(axL, 17, 5.5, 17, 8, "$Emb(y_{main})$ + RMSNorm\n$(B,n)\\to(B,n,d)$",
                    ec=TEAL, fc="#f2faf7", fs=6.6)
    axL.text(25.5, 18.6, "teacher forcing:\nground truth $t_{i+1}$", ha="center", va="center",
             fontsize=6.2, color=MUTED, style="italic")
    _, cy_cat = box(axL, 40, 13.5, 11, 6.5, "concat\n$(B,n,2d)$", ec=TEAL, fs=6.8)
    _, cy_m1 = box(axL, 55, 13.5, 9, 6.5, "$M_1$\n$\\to(B,n,d)$", ec=TEAL, fs=6.8)
    _, cy_trm = box(axL, 68, 13.5, 20, 6.5, "TRM$_1$ block\n(GQA+SwiGLU) $(B,n,d)$",
                    ec=TEAL, fc="#f2faf7", fs=6.6)
    arrow(axL, 49.5, 47, 25.5, cy_hn + 3.4)                           # h -> hnorm（自 h 框底边）
    arrow(axL, 34, cy_hn, 40, cy_cat + 2.2)
    arrow(axL, 34, cy_emb, 40, cy_cat - 2.2)
    arrow(axL, 51, cy_cat, 55, cy_m1)
    arrow(axL, 64, cy_m1, 68, cy_trm)
    arrow(axL, 88, cy_trm + 3.0, 74.5, cy_head - 3.6)
    axL.text(87.5, 27.5, "RMSNorm", ha="center", fontsize=6.2, color=MUTED, style="italic")
    _, cy_lmtp = box(axL, 92, 3, 25, 7.5,
                     "logits$_{mtp}$ $(B,n,V)$\nCE vs $y_{mtp}$ ($t_{i+2}$)", ec=ORANGE, fs=7.0)
    arrow(axL, 83.5, cy_head - 3.4, 92, cy_lmtp + 1.4)
    axL.text(101, 13.6, "$\\lambda\\cdot$CE$_{mtp}$", ha="center", fontsize=7.5, color=INK)
    axL.text(59, -1.2, "joint loss:  $L = \\mathrm{CE}_{main} + \\lambda\\cdot\\mathrm{CE}_{mtp}$"
                        "   (Eq 24-25, D=1)", ha="center", fontsize=7.5, color=INK)
    axL.set_title("MTP module (D=1) on a shared trunk — shapes on every tensor",
                  fontsize=9.0, color=INK)

    # ---------------- 右：off-by-one 三视图 ----------------
    axR.set_xlim(0, 100)
    axR.set_ylim(-4, 58)
    T = 8
    cw, ch, x0 = 9.6, 7.2, 21.5

    def row(y, toks, ec, label, sub, hl=None):
        for j, tk in enumerate(toks):
            hit = (hl == j)
            axR.add_patch(Rectangle((x0 + j * cw, y), cw, ch, fc=("#fdf3d6" if hit else "white"),
                                    ec=ec, lw=1.4, zorder=3))
            axR.text(x0 + j * cw + cw / 2, y + ch / 2, tk, ha="center", va="center",
                     fontsize=7.6, color=INK, zorder=4)
        axR.text(x0 - 1.6, y + ch / 2, label, ha="right", va="center", fontsize=7.4, color=INK)
        axR.text(x0 - 1.6, y + ch / 2 - 3.1, sub, ha="right", va="center", fontsize=6.2,
                 color=MUTED, style="italic")

    # 顶行：原始序列
    row(46.5, [f"$t_{{{k}}}$" for k in range(T)], MUTED, "tokens", "$(B, T)$, $T{=}8$", hl=2)
    # 三个视图（各 T-2=6 格）
    row(33.0, [f"$t_{{{k}}}$" for k in range(6)], BLUE, "$x_{in}$", "$t[:,:-2]$ input", hl=2)
    row(20.5, [f"$t_{{{k}}}$" for k in range(1, 7)], TEAL, "$y_{main}$", "$t[:,1:-1]$ target $t{+}1$", hl=1)
    row(8.0, [f"$t_{{{k}}}$" for k in range(2, 8)], ORANGE, "$y_{mtp}$", "$t[:,2:]$ target $t{+}2$", hl=0)
    # 错位引导线：顶行 t2 → 各视图中的 t2（黄格）
    for (xa, ya, xb, yb) in ((x0 + 2 * cw + cw / 2, 46.5, x0 + 2 * cw + cw / 2, 40.2),
                             (x0 + 2 * cw + cw / 2, 46.5, x0 + 1 * cw + cw / 2, 27.7),
                             (x0 + 2 * cw + cw / 2, 46.5, x0 + 0 * cw + cw / 2, 15.2)):
        axR.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-|>", mutation_scale=9,
                                      color="#a06b00", lw=1.1, ls=(0, (3, 2)), zorder=2))
    axR.text(50, 43.6, "same token $t_2$: one column earlier in each lower view",
             ha="center", fontsize=6.6, color="#a06b00", style="italic")
    # 支路输入提示：y_main -> Emb
    axR.annotate("", xy=(x0 + 5.5 * cw, 28.0), xytext=(x0 + 5.5 * cw, 20.5),
                 arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1, ls=(0, (3, 2))))
    axR.text(x0 + 5.7 * cw, 24.6, "$Emb(y_{main})$ feeds the MTP branch\n"
             "(NOT $y_{mtp}$ — that leaks the answer)", ha="left", va="center", fontsize=6.2,
             color=MUTED)
    axR.text(50, 2.2, "three views, all length $T{-}2$; each pair off by exactly one",
             ha="center", fontsize=7.0, color=INK)
    axR.text(50, -2.2, "asserts: equal length / shift-by-one / single-point check (run every step)",
             ha="center", fontsize=6.2, color=MUTED, style="italic")
    axR.set_title("off-by-one: three views of one token stream", fontsize=9.0, color=INK)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-8-1-mtp-module-off-by-one.png")
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 8.1] {out}")


def fig_curve(src):
    """图 8.2：λ=0.3 主臂双支路（train 细线+eval 点）与 λ=0 对照臂（main eval + 不训的 mtp 支路平走）。"""
    src_used = src
    if src == "auto":
        src_used = "full" if os.path.exists(os.path.join(LOG_DIR, "mtp_full.json")) else "fast"
        if src_used == "fast":
            print("[提示] full 档未落地，用 fast 档（正文口径即 fast 500 步）")
    with open(os.path.join(LOG_DIR, f"mtp_{src_used}.json"), encoding="utf-8") as f:
        rep = json.load(f)

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=300)
    # λ=0.3 臂：双支路 train 细线 + eval 点
    arm = "lam30"
    for key, label, color in (("main", "main branch, $\\lambda$=0.3", BLUE),
                              ("mtp", "MTP branch, $\\lambda$=0.3", TEAL)):
        path = os.path.join(LOG_DIR, f"curve_{arm}_{src_used}.csv")
        steps, losses = [], []
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                steps.append(int(row["step"]))
                losses.append(float(row[f"{key}_loss"]))
        ax.plot(steps, losses, color=color, lw=1.0, alpha=0.55, label=f"{label} (train)",
                solid_capstyle="round", zorder=2)
        ev = rep["arms"][arm]["evals"]
        ax.errorbar([e["step"] for e in ev], [e[f"eval_{key}"] for e in ev],
                    yerr=[e.get(f"eval_{key}_se", 0.0) for e in ev], color=color, marker="o",
                    ms=6, lw=0, elinewidth=1.0, capsize=2.5, label=f"{label} (eval)", zorder=5)
    # λ=0 臂：main eval（对照终值）+ 不训的 mtp 支路（平走）
    ev = rep["arms"]["lam00"]["evals"]
    ax.errorbar([e["step"] for e in ev], [e["eval_main"] for e in ev],
                yerr=[e.get("eval_main_se", 0.0) for e in ev], color=ORANGE, marker="s",
                ms=6, lw=1.4, elinewidth=1.0, capsize=2.5,
                label="main branch, $\\lambda$=0 (control, eval)", zorder=4)
    ax.errorbar([e["step"] for e in ev], [e["eval_mtp"] for e in ev],
                yerr=[e.get("eval_mtp_se", 0.0) for e in ev], color=YELLOW, marker="D",
                ms=6, lw=1.2, elinewidth=1.0, capsize=2.5,
                label="MTP branch, $\\lambda$=0 (untrained, flat)", zorder=4)
    # 末态标注：反优读数
    a30 = rep["arms"]["lam30"]["eval_main_final"]
    a00 = rep["arms"]["lam00"]["eval_main_final"]
    ax.annotate(f"main @500: {a30:.3f} vs {a00:.3f}\n($\\lambda$=0.3 lower by "
                f"{a00 - a30:.3f} nats)", xy=(500, a30), xytext=(315, 8.35),
                fontsize=7.0, color=INK,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0))

    ax.set_xlabel("step", fontsize=9, color=MUTED)
    ax.set_ylabel("loss (nats, 49 fixed eval windows / train batches)",
                  fontsize=8.5, color=MUTED)
    ax.tick_params(colors="#898781", labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#898781")
    ax.grid(True, color=GRIDC, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(f"two branches under joint MTP loss vs $\\lambda$=0 control "
                 f"(single seed, {src_used} run)", fontsize=9, color=INK, pad=8)
    ax.legend(fontsize=6.8, frameon=False, loc="center right")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-8-2-mtp-dual-branch-loss.png")
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 8.2] {out}（档位 {src_used}）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Book4 ch8 制图（图 8.1 模块+三视图 / 图 8.2 双支路曲线）")
    ap.add_argument("--which", default="all", choices=["all", "diagram", "curve"])
    ap.add_argument("--src", default="auto", choices=["auto", "fast", "full"],
                    help="图 8.2 数据档位（auto=full 优先回落 fast；正文口径=fast 500 步）")
    args = ap.parse_args()
    if args.which in ("all", "diagram"):
        fig_diagram()
    if args.which in ("all", "curve"):
        fig_curve(args.src)
