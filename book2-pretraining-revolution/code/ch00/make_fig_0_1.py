# 用途：生成图 0.1 —— GPT-2 124M 整机粗图（自绘示意级：一条数据河 + 四项差异插座标注）
# 所属章节：Book2 第 0 章导览（ch00-导览.md 图 0.1）
# 运行方式：cd /Users/allen/Code/model_analysis && source env.sh && python code/ch00/make_fig_0_1.py
# 图规：_写作规范.md §4（300dpi / 固定分类色 / 图内英文、图注中文）

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from pathlib import Path

# ---- 分类色（写作规范固定序）----
BLUE = "#2a78d6"    # 主结构框
TEAL = "#1baf7a"    # 嵌入侧
YELLOW = "#eda100"  # 输出侧
ORANGE = "#eb6834"  # 四项手术插座
INK = "#0b0b0b"
SUB = "#52514e"
GRAY = "#898781"
GHOST = "#b8b6ae"   # Book1 已拆除部分（虚影）

fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
ax.set_xlim(0, 16.2)
ax.set_ylim(0, 8)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, text, ec=BLUE, fc="white", lw=1.8, fs=8.0, tc=INK, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.05,rounding_size=0.10",
                                linewidth=lw, edgecolor=ec, facecolor=fc,
                                linestyle=ls, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=4, linespacing=1.4)


def arrow(p1, p2, color=SUB, ls="-", lw=1.6, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=lw, color=color, linestyle=ls,
                                 zorder=2, shrinkA=1, shrinkB=1,
                                 connectionstyle=f"arc3,rad={rad}"))


def polyline(points, color=SUB, lw=1.4):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=lw, zorder=2, solid_capstyle="round")


def badge(x, y, n):
    ax.add_patch(Circle((x, y), 0.27, facecolor=ORANGE, edgecolor="white",
                        linewidth=1.2, zorder=6))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=9,
            color="white", fontweight="bold", zorder=7)


def junction(x, y):
    ax.add_patch(Circle((x, y), 0.06, facecolor=SUB, edgecolor=SUB, zorder=5))


# ---- 顶部标题（图内英文）----
ax.text(8.1, 7.75, "GPT-2 124M: one tower, four sockets  (vs. Book1 2017 two-tower machine)",
        ha="center", va="center", fontsize=9.5, color=INK)

# ---- 主数据河：token → 嵌入 → Block×12 → ln_f → lm_head → probs ----
Y_RIVER = 5.42  # 主河道 y 坐标

box(0.25, Y_RIVER - 0.5, 1.7, 1.0, "token ids\n$(B,\\ n)$")
arrow((1.95, Y_RIVER), (2.4, Y_RIVER))

box(2.45, Y_RIVER + 0.16, 2.3, 0.72, "token embed wte\n$(50257,\\ 768)$", ec=TEAL)
box(2.45, Y_RIVER - 0.88, 2.3, 0.72, "pos embed wpe\n$(1024,\\ 768)$", ec=TEAL)
ax.text(3.6, Y_RIVER - 1.16, "learned lookup, $n \\leq 1024$",
        ha="center", va="center", fontsize=7.0, color=SUB)
badge(4.55, Y_RIVER - 0.1, 3)  # 手术③：学习式位置编码

arrow((4.75, Y_RIVER + 0.52), (5.42, Y_RIVER + 0.12))
arrow((4.75, Y_RIVER - 0.52), (5.42, Y_RIVER - 0.12))
ax.add_patch(Circle((5.55, Y_RIVER), 0.20, facecolor="white", edgecolor=SUB, lw=1.5, zorder=3))
ax.text(5.55, Y_RIVER, "+", ha="center", va="center", fontsize=10, color=SUB, zorder=4)
arrow((5.75, Y_RIVER), (6.1, Y_RIVER))
ax.text(5.93, Y_RIVER + 0.30, "$(B,\\ n,\\ 768)$", ha="center", fontsize=6.8, color=SUB)

# ---- Block 容器（×12，pre-LN 内部结构）----
CX, CY, CW, CH = 6.1, 3.95, 6.3, 3.0
ax.add_patch(FancyBboxPatch((CX, CY), CW, CH,
                            boxstyle="round,pad=0.05,rounding_size=0.12",
                            linewidth=2.0, edgecolor=BLUE, facecolor="#f5f9fe", zorder=1))
ax.text(CX + CW / 2, CY + CH - 0.22, "Block $\\times$ 12   ($d_{model}$=768)",
        ha="center", va="center", fontsize=8.5, color=INK, zorder=4)

Y_LANE = 5.35  # 组件车道 y 中心
junction(6.35, Y_RIVER)
box(6.5, Y_LANE - 0.31, 0.72, 0.62, "ln$_1$", fs=8.0)
badge(6.86, Y_LANE + 0.48, 2)  # 手术②：post→pre-LN
box(7.5, Y_LANE - 0.41, 1.6, 0.82, "causal\nself-attn\n12 heads", fs=7.5)
ax.text(8.85, Y_LANE - 0.62, "$(B,\\ 12,\\ n,\\ 64)$", ha="center", fontsize=6.5, color=SUB)
arrow((7.22, Y_LANE), (7.5, Y_LANE))
arrow((9.1, Y_LANE), (9.35, Y_LANE))
ax.add_patch(Circle((9.47, Y_LANE), 0.18, facecolor="white", edgecolor=SUB, lw=1.5, zorder=3))
ax.text(9.47, Y_LANE, "+", ha="center", va="center", fontsize=10, color=SUB, zorder=4)
box(9.85, Y_LANE - 0.31, 0.72, 0.62, "ln$_2$", fs=8.0)
arrow((9.65, Y_LANE), (9.85, Y_LANE))
arrow((10.57, Y_LANE), (10.75, Y_LANE))
box(10.75, Y_LANE - 0.36, 1.3, 0.72, "MLP\n768-3072-768", fs=7.2)
arrow((12.05, Y_LANE), (12.18, Y_LANE))
ax.add_patch(Circle((12.3, Y_LANE), 0.18, facecolor="white", edgecolor=SUB, lw=1.5, zorder=3))
ax.text(12.3, Y_LANE, "+", ha="center", va="center", fontsize=10, color=SUB, zorder=4)

# 残差旁路 1：入口 → attn 后相加
polyline([(6.35, Y_RIVER + 0.1), (6.35, 6.5), (9.47, 6.5)])
arrow((9.47, 6.5), (9.47, Y_LANE + 0.24))
# 残差旁路 2：⊕1 → MLP 后相加
junction(9.47, Y_LANE - 0.2)
polyline([(9.47, Y_LANE - 0.26), (9.47, 4.45), (12.3, 4.45)])
arrow((12.3, 4.45), (12.3, Y_LANE - 0.24))
ax.text(9.1, 4.62, "residual", fontsize=6.5, color=SUB, style="italic")
ax.text(6.5, 6.62, "residual", fontsize=6.5, color=SUB, style="italic")

# 手术①：交叉注意力岗位已拆（decoder-only），在 attn 框下标记
badge(7.82, Y_LANE - 0.62, 1)

# ---- 出口：ln_f → lm_head（tied）→ probs ----
arrow((12.4, Y_LANE), (12.72, Y_LANE))
box(12.72, Y_LANE - 0.31, 0.72, 0.62, "ln$_f$", fs=8.0)
arrow((13.44, Y_LANE), (13.66, Y_LANE))
box(13.66, Y_LANE - 0.41, 1.66, 0.82, "lm_head $=\\, W_{te}^{\\top}$\n(same matrix)", ec=YELLOW)
badge(15.12, Y_LANE + 0.55, 4)  # 手术④：tied embeddings
arrow((15.32, Y_LANE), (15.62, Y_LANE))
ax.text(15.66, Y_LANE, "probs\n$(B,\\ n,\\ 50257)$", ha="left", va="center",
        fontsize=7.5, color=INK, linespacing=1.4)

# ---- 底部：Book1 编码塔虚影（已拆除）+ 插座图例 ----
box(0.7, 0.95, 5.2, 1.55,
    "Book1 2017 machine\nencoder tower + cross-attn\n(dashed = removed)",
    ec=GHOST, fc="white", lw=1.6, ls=(0, (4, 3)), fs=7.8, tc=GRAY)
arrow((5.9, 2.3), (7.80, Y_LANE - 0.68), color=ORANGE, ls=(0, (4, 3)), lw=1.6)
ax.text(7.06, 3.35, "$\\times$", ha="center", va="center", fontsize=15,
        color=ORANGE, fontweight="bold")
ax.text(6.0, 2.62, "surgery 1: folded away\n(decoder-only)", ha="left", va="bottom",
        fontsize=7.2, color=ORANGE, linespacing=1.4)

box(8.9, 0.95, 6.6, 1.55, "", ec=SUB, fc="#f7f7f5", lw=1.2)
ax.text(9.15, 2.22, "Four sockets = surgeries from the Book1 machine (ch. 4):",
        ha="left", va="center", fontsize=7.8, color=INK)
legend_lines = [
    (1, "single tower: cross-attn folded away"),
    (2, "post-LN $\\to$ pre-LN (drop warmup)"),
    (3, "sinusoidal $\\to$ learned PE table"),
    (4, "tied embeddings (wte reused as head)"),
]
for i, (n, txt) in enumerate(legend_lines):
    yy = 1.92 - i * 0.30
    badge(9.32, yy, n)
    ax.text(9.66, yy, txt, ha="left", va="center", fontsize=7.4, color=SUB)

# ---- 输出 ----
out_dir = Path(__file__).resolve().parents[2] / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "fig-0-1-gpt2-overview.png"
fig.tight_layout()
fig.savefig(out, dpi=300, facecolor="white")
print(f"saved: {out}")
