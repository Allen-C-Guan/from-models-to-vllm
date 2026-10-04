# 用途：生成图 0.1 —— 409M 两刀机整机粗图（自绘示意级：一条 (B,n,1024) 数据河 + 前馈/注意力两格动刀高亮 + 底部 207M 换下零件虚影，对照 Book3 图 0.1 四插座图式）
# 所属章节：Book4 第 0 章导览（ch00-导览.md 图 0.1）
# 运行方式：cd 工作区根目录 && source env.sh && python code/ch00/make_fig0.py
# 图规：_写作规范.md §4（300dpi / 固定分类色 / 图内英文、图注中文 / 结构图张量框标形状）

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from pathlib import Path

# ---- 分类色（写作规范固定序）----
BLUE = "#2a78d6"    # 主结构框（承 Book3 不动件）
TEAL = "#1baf7a"    # 嵌入侧
YELLOW = "#eda100"  # 输出侧
ORANGE = "#eb6834"  # 两刀标记
INK = "#0b0b0b"
SUB = "#52514e"
GRAY = "#898781"
GHOST = "#b8b6ae"   # 207M 换下零件（虚影）

fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=300)
ax.set_xlim(0, 18.2)
ax.set_ylim(0, 9.0)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, text, ec=BLUE, fc="white", lw=1.8, fs=8.0, tc=INK, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.05,rounding_size=0.10",
                                linewidth=lw, edgecolor=ec, facecolor=fc,
                                linestyle=ls, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=4, linespacing=1.4)


def arrow(p1, p2, color=SUB, ls="-", lw=1.6):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=lw, color=color, linestyle=ls,
                                 zorder=2, shrinkA=1, shrinkB=1))


def polyline(points, color=SUB, lw=1.4):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=lw, zorder=2, solid_capstyle="round")


def badge(x, y, n):
    # 两刀实心橙圆（编号=本册动刀次序：1=FFN→MoE，2=GQA→MLA）
    ax.add_patch(Circle((x, y), 0.27, facecolor=ORANGE, edgecolor="white",
                        linewidth=1.2, zorder=6))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=9,
            color="white", fontweight="bold", zorder=7)


def junction(x, y):
    ax.add_patch(Circle((x, y), 0.06, facecolor=SUB, edgecolor=SUB, zorder=5))


def plus(x, y):
    ax.add_patch(Circle((x, y), 0.18, facecolor="white", edgecolor=SUB, lw=1.5, zorder=3))
    ax.text(x, y, "+", ha="center", va="center", fontsize=10, color=SUB, zorder=4)


# ---- 顶部标题（图内英文）----
ax.text(9.1, 8.65, "Two-cut machine (cand2): 409M total / 168.9M active (incl. lm_head) — FFN slot $\\to$ MoE, attn slot $\\to$ MLA",
        ha="center", va="center", fontsize=9.5, color=INK)
ax.text(9.1, 8.32, "base: the Book3 207M skeleton (four sockets, two re-cut)",
        ha="center", va="center", fontsize=7.5, color=SUB)

# ---- 主数据河：token → wte → Block×12 → final RMSNorm → lm_head(untied) → probs ----
Y_RIVER = 6.0

box(0.25, Y_RIVER - 0.5, 1.7, 1.0, "token ids\n$(B,\\ n)$")
arrow((1.95, Y_RIVER), (2.35, Y_RIVER))

box(2.4, Y_RIVER - 0.37, 2.45, 0.74, "token embed wte\n$(32000,\\ 1024)$", ec=TEAL)
ax.text(3.62, Y_RIVER - 0.68, "unchanged (Book3 ch5, untied)",
        ha="center", va="center", fontsize=6.6, color=GRAY)
arrow((4.85, Y_RIVER), (5.6, Y_RIVER))
ax.text(5.22, Y_RIVER + 0.22, "$(B,\\ n,\\ 1024)$", ha="center", fontsize=6.8, color=SUB)

# ---- Block 容器（×12，两刀都在 Block 内的两格插槽）----
CX, CY, CW, CH = 5.6, 4.0, 8.05, 3.35
ax.add_patch(FancyBboxPatch((CX, CY), CW, CH,
                            boxstyle="round,pad=0.05,rounding_size=0.12",
                            linewidth=2.0, edgecolor=BLUE, facecolor="#f5f9fe", zorder=1))
ax.text(CX + CW / 2, CY + CH - 0.22, "Block $\\times$ 12   ($d_{model}$=1024; 12/12 layers MoE, first$_k$dense$_{replace}$=0)",
        ha="center", va="center", fontsize=8.3, color=INK, zorder=4)

Y_LANE = 5.6  # 组件车道 y 中心
junction(5.85, Y_RIVER)
box(6.0, Y_LANE - 0.31, 0.9, 0.62, "RMSNorm", fs=7.4)
ax.text(6.45, Y_LANE - 0.52, "unchanged", ha="center", fontsize=6.2, color=GRAY)
arrow((6.9, Y_LANE), (7.1, Y_LANE))

# 第二刀：注意力格 GQA → MLA（KV 装箱 + 解耦位置键）
box(7.1, Y_LANE - 0.50, 2.35, 1.00,
    "MLA attention\n16 Q heads; KV latent $c^{KV}$ $(B,n,256)$\n+ decoupled $k^{R}$ $(B,n,64)$", fs=6.8)
badge(7.45, Y_LANE + 0.64, 2)
arrow((9.45, Y_LANE), (9.6, Y_LANE))
plus(9.76, Y_LANE)
arrow((9.94, Y_LANE), (10.1, Y_LANE))

box(10.1, Y_LANE - 0.31, 0.9, 0.62, "RMSNorm", fs=7.4)
ax.text(10.55, Y_LANE - 0.52, "unchanged", ha="center", fontsize=6.2, color=GRAY)
arrow((11.0, Y_LANE), (11.2, Y_LANE))

# 第一刀：前馈格 dense SwiGLU → MoE（router 打分 + top-2 + 共享专家）
box(11.2, Y_LANE - 0.54, 2.05, 1.08,
    "MoE FFN\nrouter $(B{\\cdot}n,\\ 8)$ $\\to$ top-2\nE=8 experts $w$=938 + 1 shared", fs=6.7)
badge(11.55, Y_LANE + 0.68, 1)
arrow((13.25, Y_LANE), (13.4, Y_LANE))
plus(13.52, Y_LANE)

# 残差旁路 1：入口 → attn 后相加
polyline([(5.85, Y_RIVER + 0.1), (5.85, 6.95), (9.76, 6.95)])
arrow((9.76, 6.95), (9.76, Y_LANE + 0.26))
# 残差旁路 2：⊕1 → MoE 后相加
junction(9.76, Y_LANE - 0.2)
polyline([(9.76, Y_LANE - 0.26), (9.76, 4.52), (13.52, 4.52)])
arrow((13.52, 4.52), (13.52, Y_LANE - 0.24))
ax.text(9.2, 4.7, "residual", fontsize=6.5, color=SUB, style="italic")
ax.text(5.98, 7.08, "residual", fontsize=6.5, color=SUB, style="italic")

# ---- 出口：final RMSNorm → lm_head（untied，承 Book3）→ probs ----
arrow((13.72, Y_LANE), (13.95, Y_LANE))
box(13.98, Y_LANE - 0.31, 0.9, 0.62, "RMSNorm", fs=7.2)
arrow((14.88, Y_LANE), (15.05, Y_LANE))
box(15.08, Y_LANE - 0.41, 1.75, 0.82, "lm_head\n$(32000,\\ 1024)$ own\nweights (untied)", ec=YELLOW, fs=6.8)
arrow((16.83, Y_LANE), (17.0, Y_LANE))
ax.text(17.06, Y_LANE, "probs\n$(B,\\ n,\\ 32000)$", ha="left", va="center",
        fontsize=7.4, color=INK, linespacing=1.4)

# ---- 底部左：207M 换下零件虚影清单 ----
box(0.55, 0.75, 6.7, 1.8, "", ec=GHOST, fc="white", lw=1.6)
ax.text(0.8, 2.28, "Parts retired from the 207M (dashed):",
        ha="left", va="center", fontsize=7.8, color=GRAY)
ghost_lines = [
    "dense SwiGLU FFN 1024-2816-1024 (Book3 ch3) $\\to$ MoE",
    "GQA: 16 Q / 8 KV heads, repeat$_{kv}$ (Book3 ch6) $\\to$ MLA",
]
for i, txt in enumerate(ghost_lines):
    ax.text(1.0, 1.88 - i * 0.42, txt, ha="left", va="center", fontsize=7.2, color=GRAY)
ax.text(0.8, 0.98, "(swapped out by Cut 1 / Cut 2 below)",
        ha="left", va="center", fontsize=6.6, color=GRAY, style="italic")

# ---- 底部右：两刀图例 ----
box(7.55, 0.75, 10.1, 1.8, "", ec=SUB, fc="#f7f7f5", lw=1.2)
ax.text(7.8, 2.28, "Two cuts of this book (order of the journey):",
        ha="left", va="center", fontsize=7.8, color=INK)
badge(8.0, 1.95, 1)
ax.text(8.38, 1.95, "Cut 1 (ch2-5): dense FFN $\\to$ MoE — router + top-2 of E=8 routed experts + 1 shared, $w$=938",
        ha="left", va="center", fontsize=7.1, color=SUB)
badge(8.0, 1.63, 2)
ax.text(8.38, 1.63, "Cut 2 (ch6): GQA $\\to$ MLA — joint KV latent $d_c$=256 + rope key 64; KV 3,840 elems/token ($-$68.8%)",
        ha="left", va="center", fontsize=7.1, color=SUB)
ax.text(8.38, 1.31, "Unchanged from Book3: RMSNorm (ch2) $\\cdot$ SwiGLU inside each expert (ch3) $\\cdot$ RoPE (ch4) $\\cdot$ untied vocab (ch5)",
        ha="left", va="center", fontsize=6.6, color=GRAY)
ax.text(8.38, 1.01, "12/12 layers are MoE (first$_k$dense$_{replace}$=0; dense first layer = restart variant, DESIGN §3)",
        ha="left", va="center", fontsize=6.6, color=GRAY)

# ---- 输出 ----
out_dir = Path(__file__).resolve().parents[2] / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "fig-0-1-two-cuts-409m-overview.png"
fig.tight_layout()
fig.savefig(out, dpi=300, facecolor="white")
print(f"saved: {out}")
