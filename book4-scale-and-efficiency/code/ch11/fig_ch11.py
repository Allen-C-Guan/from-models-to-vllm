# 用途：生成图 11.1（LatentMoE 块数据流，自绘示意级，张量框逐个标形状）与图 11.2（经典 MoE vs LatentMoE 的 diff 对照）——Book4 第 11 章配图（本章无代码交付物，本脚本为图 0.1 先例同款的配图基础设施）
# 所属章节：Book4 第 11 章 LatentMoE 支线（ch11-LatentMoE支线.md 图 11.1/图 11.2）
# 运行方式：cd 工作区根目录 && source env.sh && python code/ch11/fig_ch11.py --which all
#           （可选 --which dataflow|diff 只画单图；秒级、CPU、无随机性）
# 图规：_写作规范.md §4（300dpi / 固定分类色 / 图内英文图注中文 / 结构图张量框标形状）
# 机构依据：Nemotron 3 §2.2 原文（arXiv:2512.20856 v1）+ 其 Figure 3b 的描述；数值取论文表 1 右列（d=4096, ℓ=1024, E'=512, k'=22）

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from pathlib import Path

# ---- 分类色（写作规范固定序）----
BLUE = "#2a78d6"    # 全宽结构（输入/投影电梯/共享专家/残差河）
TEAL = "#1baf7a"    # 路由门（全宽、不搬家）
ORANGE = "#eb6834"  # 路由专家与潜空间（本章主角色）
YELLOW = "#eda100"  # 门控权重施加点（黄虚线）
INK = "#0b0b0b"
SUB = "#52514e"
GRAY = "#898781"
LATENT_FILL = "#fdf1ea"   # 潜空间区浅底
ACTIVE_FILL = "#fde3d5"   # 被选中专家的浅填充


def box(ax, x, y, w, h, text, ec=BLUE, fc="white", lw=1.8, fs=7.2, tc=INK, ls="-"):
    """圆角框 + 居中文字（结构图张量框）"""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.05,rounding_size=0.10",
                                linewidth=lw, edgecolor=ec, facecolor=fc,
                                linestyle=ls, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=4, linespacing=1.35)


def arrow(ax, p1, p2, color=SUB, ls="-", lw=1.6):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=lw, color=color, linestyle=ls,
                                 zorder=2, shrinkA=1, shrinkB=1))


def polyline(ax, points, color=SUB, lw=1.4):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=lw, zorder=2, solid_capstyle="round")


def junction(ax, x, y):
    ax.add_patch(Circle((x, y), 0.07, facecolor=SUB, edgecolor=SUB, zorder=5))


def plus(ax, x, y):
    ax.add_patch(Circle((x, y), 0.20, facecolor="white", edgecolor=SUB, lw=1.5, zorder=3))
    ax.text(x, y, "+", ha="center", va="center", fontsize=11, color=SUB, zorder=4)


# =====================================================================
# 图 11.1 LatentMoE 块数据流
# =====================================================================
def make_dataflow(out_dir: Path):
    fig, ax = plt.subplots(figsize=(10.6, 5.6), dpi=300)
    ax.set_xlim(0, 21.2)
    ax.set_ylim(0, 11.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # 顶部标题（图内英文）
    ax.text(10.6, 10.85, "LatentMoE block (Nemotron 3 $\\S$2.2): routed experts live at latent width $\\ell$;"
            " gate & shared expert stay at full width $d$",
            ha="center", va="center", fontsize=9.0, color=INK)
    ax.text(10.6, 10.45, "numbers = paper Table 1 config: $d$=4096, $\\ell$=1024 ($d/\\ell$=4), "
            "$E'$=512 experts, $k'$=22 active per token",
            ha="center", va="center", fontsize=7.3, color=SUB)

    Y_MID = 6.1  # 主车道中心

    # ---- 输入与分叉 ----
    box(ax, 0.25, Y_MID - 0.55, 1.85, 1.1, "input $x$\n$(B,\\ n,\\ d)$\n$d$=4096")
    junction(ax, 2.45, Y_MID)

    # ---- 上车道：路由门（全宽、不搬家）----
    polyline(ax, [(2.45, Y_MID), (2.45, 8.65)])
    arrow(ax, (2.45, 8.65), (2.95, 8.65))
    box(ax, 3.0, 8.0, 4.5, 1.3,
        "router gate — stays in $d$\n$W_g\\in\\mathbb{R}^{E'\\times d}$: "
        "$(N,\\ d)\\to(N,\\ E')$\ntop-$k'$ mask $\\to$ gates $(N,\\ k')$", ec=TEAL)
    ax.text(5.25, 7.72, "not moved (off the bottleneck bill)", ha="center",
            fontsize=6.4, color=GRAY, style="italic")

    # ---- 中车道：下投影电梯 ----
    arrow(ax, (2.55, Y_MID), (3.15, Y_MID))
    box(ax, 3.2, Y_MID - 0.62, 2.6, 1.24,
        "$W^{\\downarrow}\\in\\mathbb{R}^{\\ell\\times d}$\n$(N,\\ d)\\cdot(d,\\ \\ell)\\to(N,\\ \\ell)$")
    ax.text(4.5, Y_MID - 0.86, "down-projection (elevator)", ha="center",
            fontsize=6.4, color=GRAY, style="italic")

    # ---- 下车道：共享专家（全宽、不搬家）----
    polyline(ax, [(2.45, Y_MID), (2.45, 2.9)])
    arrow(ax, (2.45, 2.9), (2.95, 2.9))
    box(ax, 3.0, 2.35, 3.9, 1.1,
        "shared expert — stays in $d$\n$(B,\\ n,\\ d)\\to(B,\\ n,\\ d)$  (ch5)", ec=BLUE)
    ax.text(4.95, 2.1, "not moved; every token, always on", ha="center",
            fontsize=6.4, color=GRAY, style="italic")

    # ---- 潜空间区：扩大了的专家群 ----
    arrow(ax, (5.8, Y_MID), (6.6, Y_MID))
    ax.text(6.2, Y_MID + 0.24, "$(N,\\ \\ell)$", ha="center", fontsize=6.8, color=SUB)

    LX, LY, LW, LH = 6.6, 4.3, 8.0, 4.3
    ax.add_patch(FancyBboxPatch((LX, LY), LW, LH,
                                boxstyle="round,pad=0.06,rounding_size=0.14",
                                linewidth=1.8, edgecolor=ORANGE,
                                facecolor=LATENT_FILL, zorder=1))
    ax.text(LX + LW / 2, LY + LH - 0.26, "latent space, width $\\ell$=1024  —  "
            "expanded bank: $E'$=512 experts, discrete top-$k'$=22 per token",
            ha="center", va="center", fontsize=7.4, color=INK, zorder=4)
    ax.text(LX + LW / 2, LY + LH - 0.56, "experts work entirely here: SwiGLU "
            "$(n_i,\\ \\ell)\\to(n_i,\\ \\ell)$, $\\ell\\times m$ each ($m$ untouched)",
            ha="center", va="center", fontsize=6.6, color=SUB, zorder=4)

    # 专家格：2 行 × 4 个（部分高亮=被选中），加省略号
    ew, eh = 1.55, 0.92
    for r in range(2):
        for c in range(4):
            ex = LX + 0.55 + c * 1.85
            ey = LY + 0.95 + (1 - r) * 1.28
            active = (r + c) % 3 == 0  # 少数高亮：离散选中的少数
            box(ax, ex, ey, ew, eh, f"expert $E_{{{r * 4 + c + 1}}}$",
                ec=ORANGE, fc=ACTIVE_FILL if active else "white",
                lw=1.5, fs=6.8)
            if active:
                ax.text(ex + ew / 2, ey - 0.18, "selected", ha="center",
                        fontsize=5.6, color=ORANGE)
        ax.text(LX + 0.55 + 4 * 1.85 - 0.08, LY + 0.95 + (1 - r) * 1.28 + eh / 2,
                "$\\cdots$", ha="center", va="center", fontsize=10, color=SUB)
    ax.text(LX + LW / 2, LY + 0.42,
            "dispatch / combine (gather-scatter as ch2): all-to-all payload $\\propto k'\\cdot\\ell$",
            ha="center", va="center", fontsize=6.4, color=SUB, style="italic")

    # ---- 加权合并节点 ----
    arrow(ax, (LX + LW, Y_MID), (15.0, Y_MID))
    ax.add_patch(Circle((15.25, Y_MID), 0.22, facecolor="white",
                        edgecolor=YELLOW, lw=1.8, zorder=3))
    ax.text(15.25, Y_MID, "$\\Sigma$", ha="center", va="center", fontsize=8,
            color=INK, zorder=4)
    ax.text(15.25, Y_MID - 0.5, "$\\sum g_iE_i \\to (N,\\ \\ell)$", ha="center",
            fontsize=6.6, color=SUB)
    # 门控虚线：路由门 → 加权节点
    polyline(ax, [(7.5, 8.0), (7.5, 9.35), (15.25, 9.35)], color=YELLOW, lw=1.3)
    arrow(ax, (15.25, 9.35), (15.25, Y_MID + 0.42), color=YELLOW, ls="--", lw=1.3)
    ax.text(11.4, 9.52, "gates $g_i$ $(N,\\ k')$ — weight the selected few",
            ha="center", fontsize=6.4, color=YELLOW)

    # ---- 上投影电梯 ----
    arrow(ax, (15.55, Y_MID), (16.15, Y_MID))
    box(ax, 16.2, Y_MID - 0.62, 2.6, 1.24,
        "$W^{\\uparrow}\\in\\mathbb{R}^{d\\times\\ell}$\n$(N,\\ \\ell)\\cdot(\\ell,\\ d)\\to(N,\\ d)$")
    ax.text(17.5, Y_MID - 0.86, "up-projection (elevator)", ha="center",
            fontsize=6.4, color=GRAY, style="italic")

    # ---- 合并 + 出口 ----
    arrow(ax, (18.8, Y_MID), (19.3, Y_MID))
    plus(ax, 19.5, Y_MID)
    polyline(ax, [(4.95, 2.9), (19.5, 2.9)], color=SUB)
    arrow(ax, (19.5, 2.9), (19.5, Y_MID - 0.35))
    ax.text(10.5, 3.1, "shared path (full width, no elevator)", ha="center",
            fontsize=6.4, color=SUB, style="italic")
    arrow(ax, (19.75, Y_MID), (20.1, Y_MID))
    ax.text(20.13, Y_MID, "out $y$\n$(B,\\ n,\\ d)$\nresidual", ha="left",
            va="center", fontsize=7.0, color=INK, linespacing=1.4)

    # ---- 底部账本条 ----
    ax.add_patch(FancyBboxPatch((0.4, 0.25), 20.4, 1.35,
                                boxstyle="round,pad=0.05,rounding_size=0.10",
                                linewidth=1.2, edgecolor=SUB, facecolor="#f7f7f5", zorder=1))
    ax.text(0.75, 1.26, "Per-token ledger ($d/\\ell$=4, reinvested: $E'=E\\cdot d/\\ell$, $k'\\approx k\\cdot d/\\ell$):",
            ha="left", va="center", fontsize=7.2, color=INK, zorder=4)
    ax.text(0.75, 0.88, "traffic $\\propto k'\\ell$ = 22$\\times$1024 = 22,528  $\\approx$  baseline $k\\,d$ = 6$\\times$4096 = 24,576"
            "   |   depot $E'\\ell$ = 512$\\times$1024 = $E\\,d$ = 524,288 (exact)",
            ha="left", va="center", fontsize=6.8, color=SUB, zorder=4)
    ax.text(0.75, 0.52, "weight loads per token $\\propto k'\\ell m\\approx k\\,d\\,m$ (flat)"
            "   |   nonlinear budget $k'\\times m\\approx3.7\\times k\\times m$ ($\\uparrow$)"
            "   |   still discrete top-$k'$",
            ha="left", va="center", fontsize=6.8, color=SUB, zorder=4)

    out = out_dir / "fig-11-1-latentmoe-dataflow.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


# =====================================================================
# 图 11.2 经典 MoE vs LatentMoE 的 diff 对照
# =====================================================================
def _diff_panel(ax, title, sub, note_lines, xmin=0.0, xmax=10.0):
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 10.6)
    ax.axis("off")
    ax.text((xmin + xmax) / 2, 10.25, title, ha="center", va="center",
            fontsize=8.6, color=INK)
    ax.text((xmin + xmax) / 2, 9.88, sub, ha="center", va="center",
            fontsize=6.9, color=SUB)
    ax.add_patch(FancyBboxPatch((xmin + 0.25, 0.25), xmax - xmin - 0.5, 1.7,
                                boxstyle="round,pad=0.05,rounding_size=0.10",
                                linewidth=1.2, edgecolor=SUB, facecolor="#f7f7f5", zorder=1))
    for i, line in enumerate(note_lines):
        ax.text(xmin + 0.55, 1.55 - i * 0.4, line, ha="left", va="center",
                fontsize=6.6, color=SUB, zorder=4)


def make_diff(out_dir: Path):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 5.4), dpi=300)
    fig.patch.set_facecolor("white")

    # ================= 左：经典 MoE（第 2 章机构）=================
    _diff_panel(axL, "Standard MoE (ch2): experts at full width $d$",
                "cut the capacity $m$ (fine-grained), façade stays $d$",
                ["per-expert params $\\propto d\\,m$   |   per-token traffic $\\propto k\\,d$",
                 "nonlinear budget $k\\times m$   |   depot $E$ experts $\\times\\ d\\,m$"])
    Y = 5.1
    box(axL, 0.35, Y - 0.5, 1.5, 1.0, "input\n$(N,\\ d)$", fs=6.8)
    arrow(axL, (1.85, Y), (2.25, Y))
    box(axL, 2.3, Y - 0.55, 1.7, 1.1, "router\n$(N,\\ E)$", ec=TEAL, fs=6.8)
    axL.text(3.15, Y - 0.82, "top-$k$", ha="center", fontsize=6.2, color=TEAL)
    arrow(axL, (4.0, Y), (4.4, Y))
    # 三个全宽专家（宽盒子）
    for i in range(3):
        ey = Y + 1.15 - i * 1.15
        box(axL, 4.45, ey - 0.42, 2.5, 0.84,
            f"expert$_{{{i + 1}}}$  $(n_i,\\ d)\\!\\to\\!(n_i,\\ d)$", ec=ORANGE,
            fc=ACTIVE_FILL if i < 2 else "white", fs=6.2)
    axL.text(7.15, Y + 0.35, "$\\cdots$", ha="center", fontsize=10, color=SUB)
    axL.text(5.7, Y + 1.85, "$E$=128 total, $k$=6 active (d=4096)", ha="center",
             fontsize=6.4, color=SUB, style="italic")
    for i in range(3):
        ey = Y + 1.15 - i * 1.15
        arrow(axL, (6.95, ey), (7.55, Y + (0.18 if i == 0 else 0)))
    axL.add_patch(Circle((7.75, Y), 0.2, facecolor="white", edgecolor=YELLOW, lw=1.6, zorder=3))
    axL.text(7.75, Y, "$\\Sigma$", ha="center", va="center", fontsize=7, color=INK, zorder=4)
    arrow(axL, (8.0, Y), (8.4, Y))
    axL.text(8.45, Y, "out\n$(N,\\ d)$", ha="left", va="center", fontsize=7.0,
             color=INK, linespacing=1.4)
    axL.text(5.0, 2.55, "façade $d$ bills: traffic + weight reads\n"
             "capacity $m$ bills: bandwidth + expressiveness",
             ha="center", va="center", fontsize=6.6, color=SUB, linespacing=1.5)

    # ================= 右：LatentMoE =================
    _diff_panel(axR, "LatentMoE: experts at latent width $\\ell$",
                "cut the façade $d\\to\\ell$ ($\\times$4), capacity $m$ untouched",
                ["per-expert $\\propto \\ell\\,m$ ($\\div$4)   |   traffic $\\propto k'\\ell\\approx k\\,d$ (flat)",
                 "budget $k'\\times m\\approx3.7\\times$ ($\\uparrow$)   |   depot $E'\\ell=E\\,d$=524,288 (exact)"])
    box(axR, 0.35, Y - 0.5, 1.5, 1.0, "input\n$(N,\\ d)$", fs=6.8)
    arrow(axR, (1.85, Y), (2.25, Y))
    box(axR, 2.3, Y - 0.55, 1.6, 1.1, "$W^{\\downarrow}$\n$(N,\\ d)\\!\\to\\!(N,\\ \\ell)$", fs=6.4)
    arrow(axR, (3.9, Y), (4.3, Y))
    # 六个潜宽专家（窄盒、两列）+ 省略号
    for i in range(6):
        cx = 4.35 + (i % 2) * 1.32
        cy = Y + 1.5 - (i // 2) * 1.05
        box(axR, cx, cy - 0.36, 1.18, 0.72, "expert\n$(n_i,\\ \\ell)$", ec=ORANGE,
            fc=ACTIVE_FILL if i in (0, 1, 3) else "white", fs=5.8)
    axR.text(4.35 + 2 * 1.32 + 0.35, Y + 0.45, "$\\vdots$", ha="center", fontsize=10, color=SUB)
    axR.text(5.0, Y + 2.15, "$E'$=512 total, $k'$=22 active ($\\ell$=1024)", ha="center",
             fontsize=6.4, color=SUB, style="italic")
    # 潜空间浅底标注
    axR.text(5.0, Y - 1.15, "all experts live in latent space", ha="center",
             fontsize=6.4, color=ORANGE, style="italic")
    for i in range(6):
        cx = 4.35 + (i % 2) * 1.32 + 0.59
        cy = Y + 1.5 - (i // 2) * 1.05
        arrow(axR, (cx, cy), (7.05, Y + 0.1))
    axR.add_patch(Circle((7.25, Y), 0.2, facecolor="white", edgecolor=YELLOW, lw=1.6, zorder=3))
    axR.text(7.25, Y, "$\\Sigma$", ha="center", va="center", fontsize=7, color=INK, zorder=4)
    arrow(axR, (7.45, Y), (7.75, Y))
    box(axR, 7.8, Y - 0.55, 1.6, 1.1, "$W^{\\uparrow}$\n$(N,\\ \\ell)\\!\\to\\!(N,\\ d)$", fs=6.4)
    arrow(axR, (9.4, Y), (9.6, Y))
    axR.text(9.63, Y, "out\n$(N,\\ d)$", ha="left", va="center", fontsize=7.0,
             color=INK, linespacing=1.4)
    axR.text(5.0, 2.55, "same depot, same toll: $E'\\ell=E\\,d$;  $k'\\ell\\approx k\\,d$\n"
             "what changes: 4$\\times$ expert diversity, 3.7$\\times$ budget",
             ha="center", va="center", fontsize=6.6, color=SUB, linespacing=1.5)

    # 两panel之间的中央注释（抬高避开右panel汇聚箭头）
    fig.text(0.5, 0.585, "shrink the façade,\nnot the capacity",
             ha="center", va="center", fontsize=7.5, color=ORANGE,
             bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=ORANGE, lw=1.4))

    out = out_dir / "fig-11-2-classic-vs-latent-diff.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


def main():
    ap = argparse.ArgumentParser(description="Book4 ch11 figures (diagrams, no randomness)")
    ap.add_argument("--which", default="all", choices=["all", "dataflow", "diff"])
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parents[2] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.which in ("all", "dataflow"):
        make_dataflow(out_dir)
    if args.which in ("all", "diff"):
        make_diff(out_dir)


if __name__ == "__main__":
    main()
