# 用途：生成图 0.1 —— LLaMA 式 207M 整机粗图（自绘示意级：一条 (B,n,1024) 数据河 + 四插座高亮 + untied 出入口 + 底部 GPT-2 换下零件虚影）
# 所属章节：Book3 第 0 章导览（ch00-导览.md 图 0.1）
# 运行方式：cd 工作区根目录 && source env.sh && python code/ch00/make_fig_0_1.py
# 图规：_写作规范.md §4（300dpi / 固定分类色 / 图内英文、图注中文 / 结构图张量框标形状）

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from pathlib import Path

# ---- 分类色（写作规范固定序）----
BLUE = "#2a78d6"    # 主结构框
TEAL = "#1baf7a"    # 嵌入侧
YELLOW = "#eda100"  # 输出侧
ORANGE = "#eb6834"  # 四插座改装标记
INK = "#0b0b0b"
SUB = "#52514e"
GRAY = "#898781"
GHOST = "#b8b6ae"   # GPT-2 换下零件（虚影）

fig, ax = plt.subplots(figsize=(8.8, 4.4), dpi=300)
ax.set_xlim(0, 17.8)
ax.set_ylim(0, 8.7)
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
    # 四插座实心橙圆（编号=本册改装次序）
    ax.add_patch(Circle((x, y), 0.27, facecolor=ORANGE, edgecolor="white",
                        linewidth=1.2, zorder=6))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=9,
            color="white", fontweight="bold", zorder=7)


def badge_u(x, y):
    # untied 附加一刀：橙色空心圆（区别于四插座实心圆）
    ax.add_patch(Circle((x, y), 0.27, facecolor="white", edgecolor=ORANGE,
                        linewidth=1.8, zorder=6))
    ax.text(x, y, "U", ha="center", va="center", fontsize=8.5,
            color=ORANGE, fontweight="bold", zorder=7)


def junction(x, y):
    ax.add_patch(Circle((x, y), 0.06, facecolor=SUB, edgecolor=SUB, zorder=5))


# ---- 顶部标题（图内英文）----
ax.text(8.9, 8.45, "LLaMA-style 207M: four sockets + one untied exit  (vs. Book2 GPT-2 124M)",
        ha="center", va="center", fontsize=9.5, color=INK)

# ---- 主数据河：token → wte → Block×12 → final RMSNorm → lm_head(untied) → probs ----
Y_RIVER = 5.92  # 主河道 y 坐标

box(0.25, Y_RIVER - 0.5, 1.7, 1.0, "token ids\n$(B,\\ n)$")
arrow((1.95, Y_RIVER), (2.4, Y_RIVER))

box(2.45, Y_RIVER - 0.37, 2.5, 0.74, "token embed wte\n$(32000,\\ 1024)$", ec=TEAL)
ax.text(3.7, Y_RIVER - 0.70, "no position table added here",
        ha="center", va="center", fontsize=7.0, color=SUB)
arrow((4.95, Y_RIVER), (5.75, Y_RIVER))
ax.text(5.35, Y_RIVER + 0.22, "$(B,\\ n,\\ 1024)$", ha="center", fontsize=6.8, color=SUB)

# 插座③虚影：GPT-2 的 wpe 查表已拆除（位置改在注意力内部注入）
box(2.45, Y_RIVER - 1.66, 2.5, 0.74, "pos embed wpe\n$(1024,\\ 768)$  [GPT-2]",
    ec=GHOST, fc="white", lw=1.6, ls=(0, (4, 3)), fs=7.2, tc=GRAY)
arrow((4.7, Y_RIVER - 1.29), (5.4, Y_RIVER - 0.18), color=ORANGE, ls=(0, (4, 3)), lw=1.6)
ax.text(5.08, Y_RIVER - 0.98, "$\\times$", ha="center", va="center", fontsize=14,
        color=ORANGE, fontweight="bold")
ax.text(2.45, Y_RIVER - 1.92, "socket 3: lookup removed $\\to$ RoPE inside attn",
        ha="left", va="center", fontsize=7.0, color=ORANGE)

# ---- Block 容器（×12，pre-RMSNorm 内部结构）----
CX, CY, CW, CH = 5.75, 3.9, 7.25, 3.35
ax.add_patch(FancyBboxPatch((CX, CY), CW, CH,
                            boxstyle="round,pad=0.05,rounding_size=0.12",
                            linewidth=2.0, edgecolor=BLUE, facecolor="#f5f9fe", zorder=1))
ax.text(CX + CW / 2, CY + CH - 0.22, "Block $\\times$ 12   ($d_{model}$=1024)",
        ha="center", va="center", fontsize=8.5, color=INK, zorder=4)

Y_LANE = 5.5  # 组件车道 y 中心
junction(6.0, Y_RIVER)
box(6.15, Y_LANE - 0.31, 0.9, 0.62, "RMSNorm", fs=7.6)
badge(6.62, Y_LANE + 0.50, 1)  # 插座①：LayerNorm→RMSNorm
box(7.35, Y_LANE - 0.48, 1.8, 0.96, "causal self-attn\nGQA: 16 Q / 8 KV heads\nRoPE on $q,k$", fs=7.0)
ax.text(8.25, Y_LANE - 0.74, "$(B,\\,16,\\,n,\\,64)$ / $(B,\\,8,\\,n,\\,64)$",
        ha="center", fontsize=6.5, color=SUB)
badge(7.70, Y_LANE + 0.62, 4)  # 插座④：MHA→GQA（与③同宿注意力框——同一格挨两刀）
badge(8.62, Y_LANE + 0.62, 3)  # 插座③：wpe→RoPE（作用点在注意力内）
arrow((7.05, Y_LANE), (7.35, Y_LANE))
arrow((9.15, Y_LANE), (9.28, Y_LANE))
ax.add_patch(Circle((9.42, Y_LANE), 0.18, facecolor="white", edgecolor=SUB, lw=1.5, zorder=3))
ax.text(9.42, Y_LANE, "+", ha="center", va="center", fontsize=10, color=SUB, zorder=4)
box(9.75, Y_LANE - 0.31, 0.9, 0.62, "RMSNorm", fs=7.6)
arrow((10.65, Y_LANE), (10.85, Y_LANE))
box(10.85, Y_LANE - 0.40, 1.5, 0.80, "SwiGLU MLP\n1024-2816-1024", fs=7.2)
badge(11.95, Y_LANE + 0.56, 2)  # 插座②：GELU-4C→SwiGLU-8/3C
arrow((12.35, Y_LANE), (12.48, Y_LANE))
ax.add_patch(Circle((12.62, Y_LANE), 0.18, facecolor="white", edgecolor=SUB, lw=1.5, zorder=3))
ax.text(12.62, Y_LANE, "+", ha="center", va="center", fontsize=10, color=SUB, zorder=4)

# 残差旁路 1：入口 → attn 后相加
polyline([(6.0, Y_RIVER + 0.1), (6.0, 6.85), (9.42, 6.85)])
arrow((9.42, 6.85), (9.42, Y_LANE + 0.26))
# 残差旁路 2：⊕1 → MLP 后相加
junction(9.42, Y_LANE - 0.2)
polyline([(9.42, Y_LANE - 0.26), (9.42, 4.42), (12.62, 4.42)])
arrow((12.62, 4.42), (12.62, Y_LANE - 0.24))
ax.text(9.05, 4.6, "residual", fontsize=6.5, color=SUB, style="italic")
ax.text(6.12, 6.97, "residual", fontsize=6.5, color=SUB, style="italic")

# ---- 出口：final RMSNorm → lm_head（untied 独立矩阵）→ probs ----
arrow((12.8, Y_LANE), (13.1, Y_LANE))
box(13.1, Y_LANE - 0.31, 0.9, 0.62, "RMSNorm", fs=7.4)
arrow((14.0, Y_LANE), (14.2, Y_LANE))
box(14.2, Y_LANE - 0.41, 1.7, 0.82, "lm_head\n$(32000,\\ 1024)$ own\nweights", ec=YELLOW, fs=7.0)
badge_u(15.5, Y_LANE + 0.58)  # 附加一刀：tied→untied
arrow((15.9, Y_LANE), (16.1, Y_LANE))
ax.text(16.18, Y_LANE, "probs\n$(B,\\ n,\\ 32000)$", ha="left", va="center",
        fontsize=7.5, color=INK, linespacing=1.4)

# ---- 底部左：GPT-2 换下零件虚影清单 ----
box(0.7, 0.85, 6.5, 1.75, "", ec=GHOST, fc="white", lw=1.6)
ax.text(0.95, 2.32, "GPT-2 parts swapped out (dashed):",
        ha="left", va="center", fontsize=7.8, color=GRAY)
ghost_lines = [
    "$w_{pe}$ $(1024,\\,768)$ removed $\\to$ RoPE (ch. 4)",
    "LayerNorm $\\to$ RMSNorm (ch. 2)",
    "GELU MLP 768-3072-768 $\\to$ SwiGLU 1024-2816-1024 (ch. 3)",
    "MHA: 12 heads, 12 KV $\\to$ GQA: 16 Q, 8 KV (ch. 6)",
    "tied head $=w_{te}^{\\top}$ $\\to$ untied lm\\_head (ch. 5)",
]
for i, txt in enumerate(ghost_lines):
    yy = 2.02 - i * 0.27
    ax.text(1.15, yy, txt, ha="left", va="center", fontsize=7.0, color=GRAY)

# ---- 底部右：插座图例 ----
box(7.6, 0.85, 9.2, 1.75, "", ec=SUB, fc="#f7f7f5", lw=1.2)
ax.text(7.85, 2.32, "Four sockets + one extra cut (swap order of this book):",
        ha="left", va="center", fontsize=7.8, color=INK)
legend_lines = [
    (1, "LayerNorm $\\to$ pre-RMSNorm (ch. 2)"),
    (2, "GELU-4$d$ MLP $\\to$ SwiGLU-$\\frac{8}{3}d$ (ch. 3)"),
    (3, "learned $w_{pe}$ table $\\to$ RoPE on $q,k$ (ch. 4)"),
    (4, "MHA $\\to$ GQA, $h$=16 / $h_{kv}$=8 (ch. 6)"),
]
for i, (n, txt) in enumerate(legend_lines):
    yy = 2.0 - i * 0.29
    badge(8.05, yy, n)
    ax.text(8.42, yy, txt, ha="left", va="center", fontsize=7.4, color=SUB)
badge_u(8.05, 2.0 - 4 * 0.29)
ax.text(8.42, 2.0 - 4 * 0.29, "untied: tied $\\to$ independent lm\\_head (ch. 5, extra cut)",
        ha="left", va="center", fontsize=7.4, color=SUB)

# ---- 输出 ----
out_dir = Path(__file__).resolve().parents[2] / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "fig-0-1-llama-207m-overview.png"
fig.tight_layout()
fig.savefig(out, dpi=300, facecolor="white")
print(f"saved: {out}")
