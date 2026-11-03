# 用途：Book1 第 4 章两图生成——图 4.1 论文 Fig 1 架构总图自绘重制（数据流自底向上，同原图）；
#       图 4.2 三种注意力角色权重矩阵对照（权重来自与 roles_demo.py 相同种子与手写 attention 的真实计算）。
#       两图均含形状标注（图 4.1 主干五站箭头旁灰/青小标、图 4.2 各面板末行）——本书叠加层，非论文原图内容。
# 所属章节：Book1《Transformer 原典》第 4 章（图内标签经 C-I6 前置任务对照 log/1706.03762v7.pdf 第 3 页位图逐区放大核对）
# 运行方式：source env.sh && python code/ch04/figures.py
#           （固定种子 42；两图输出到 figures/，秒级）
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

from roles_demo import attention  # 复用第 4 章手写注意力，保证与正文数字同源

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "drafts" / "Book1-Transformer原典" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, SUB, EDGE = "#0b0b0b", "#52514e", "#c9c7c0"
FILL = "#f7f9fd"
NBOX = "#eef0f4"


# ---------- 图 4.1：论文 Fig 1 架构总图自绘重制 ----------
# 左塔 Inputs → Input Embedding → (⊕ Positional Encoding) → N×{Multi-Head Self-Attention,
# Feed Forward}（各带 Add & Norm）；右塔 Outputs (shifted right) → Output Embedding →
# (⊕ Positional Encoding) → N×{Masked Multi-Head Self-Attention, Multi-Head Attention
# (收编码器输出), Feed Forward} → Linear → Softmax → Output Probabilities
# 灰色/青色形状小标（(B, n_src, 512) 等）为本书叠加层，非论文原图标签——只标主干五站，保持克制。
def make_fig_4_1():
    fig, ax = plt.subplots(figsize=(8, 6.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 92)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, tc="white", fs=7.0, va="center"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=1.0, zorder=3))
        ty = y if va == "center" else y + h / 2 - 1.4
        ax.text(x, ty, text, ha="center", va=va if va != "top" else "top",
                fontsize=fs, color=tc, zorder=4)

    def arrow(x0, y0, x1, y1, color=SUB, lw=1.2):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                     color=color, lw=lw, zorder=2))

    def line(pts, color=SUB, lw=1.2):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, lw=lw, zorder=2, solid_capstyle="round")

    def plus(x, y):
        ax.plot([x - 1.2, x + 1.2], [y, y], color=SUB, lw=1.1, zorder=5)
        ax.plot([x, x], [y - 1.2, y + 1.2], color=SUB, lw=1.1, zorder=5)

    def sine(x0, y0, w=8.0, h=2.2):
        t = np.linspace(0, 3 * np.pi, 120)
        ax.plot(x0 + t / (3 * np.pi) * w, y0 + np.sin(t) * h / 2, color=BLUE, lw=1.0, zorder=4)

    def add_norm(x, y):
        box(x, y, 6.4, 3.6, "Add & Norm", "white", EDGE, tc=SUB, fs=4.8)

    # ================= 左塔：encoder（数据流自底向上） =================
    EX = 26.5
    ax.text(EX, 3.0, "Inputs", ha="center", fontsize=7.5, color=INK)
    ax.text(EX, 1.0, "(B, n_src)", ha="center", fontsize=5.5, color=SUB)   # 形状标注（本书叠加）
    arrow(EX, 4.5, EX, 8.2)
    box(EX, 11.0, 20, 4.6, "Input Embedding", FILL, BLUE, tc=INK)
    arrow(EX, 13.4, EX, 16.9)
    ax.text(EX + 2.2, 15.2, "(B, n_src, 512)", ha="left", fontsize=5.5, color=SUB)
    box(EX - 15.5, 18.5, 14.5, 7.5, "Positional\nEncoding", FILL, BLUE, tc=INK, fs=6.0, va="top")
    sine(EX - 21.3, 16.0)
    arrow(EX - 8.25, 18.5, EX - 1.7, 18.5)
    plus(EX, 18.5)
    ax.add_patch(FancyBboxPatch((EX - 18, 24), 36, 32, boxstyle="round,pad=0.4,rounding_size=1.5",
                                fc=NBOX, ec=EDGE, lw=1.0, zorder=1))
    ax.text(EX + 15.0, 54.2, "N×", ha="center", fontsize=8, color=SUB, zorder=2)
    arrow(EX, 19.8, EX, 26.4)
    box(EX, 29.5, 20, 4.8, "Multi-Head\nSelf-Attention", BLUE, BLUE, fs=6.3)
    add_norm(EX + 13.8, 29.5)
    line([(EX + 10.1, 29.5), (EX + 10.6, 29.5)])
    arrow(EX, 32.2, EX, 42.4)
    box(EX, 45.5, 20, 4.8, "Feed Forward", ORANGE, ORANGE)
    add_norm(EX + 13.8, 45.5)
    line([(EX + 10.1, 45.5), (EX + 10.6, 45.5)])
    arrow(EX, 48.2, EX, 56.5)

    # ================= 右塔：decoder（数据流自底向上） =================
    DX = 73.5
    ax.text(DX - 3.2, 3.0, "Outputs", ha="center", fontsize=7.5, color=INK)
    ax.text(DX + 4.2, 3.0, "(shifted right)", ha="left", fontsize=6.0, color=SUB)
    ax.text(DX - 3.2, 1.0, "(B, n_tgt)", ha="center", fontsize=5.5, color=SUB)  # 形状标注（本书叠加）
    arrow(DX, 4.5, DX, 8.2)
    box(DX, 11.0, 20, 4.6, "Output Embedding", FILL, BLUE, tc=INK)
    arrow(DX, 13.4, DX, 16.9)
    ax.text(DX - 1.8, 15.2, "(B, n_tgt, 512)", ha="right", fontsize=5.5, color=SUB)
    box(DX + 15.5, 18.5, 14.5, 7.5, "Positional\nEncoding", FILL, BLUE, tc=INK, fs=6.0, va="top")
    sine(DX + 13.3, 16.0)
    arrow(DX + 8.25, 18.5, DX + 1.7, 18.5)
    plus(DX, 18.5)
    ax.add_patch(FancyBboxPatch((DX - 18, 24), 36, 36, boxstyle="round,pad=0.4,rounding_size=1.5",
                                fc=NBOX, ec=EDGE, lw=1.0, zorder=1))
    ax.text(DX + 15.0, 58.2, "N×", ha="center", fontsize=8, color=SUB, zorder=2)
    arrow(DX, 19.8, DX, 26.4)
    box(DX, 29.5, 20, 4.8, "Masked Multi-Head\nSelf-Attention", BLUE, BLUE, fs=5.8)
    add_norm(DX + 13.8, 29.5)
    line([(DX + 10.1, 29.5), (DX + 10.6, 29.5)])
    arrow(DX, 32.2, DX, 38.4)
    box(DX, 41.5, 20, 4.8, "Multi-Head\nAttention", BLUE, BLUE, fs=6.3)
    add_norm(DX + 13.8, 41.5)
    line([(DX + 10.1, 41.5), (DX + 10.6, 41.5)])
    arrow(DX, 44.2, DX, 50.4)
    box(DX, 53.5, 20, 4.8, "Feed Forward", ORANGE, ORANGE)
    add_norm(DX + 13.8, 53.5)
    line([(DX + 10.1, 53.5), (DX + 10.6, 53.5)])
    arrow(DX, 56.2, DX, 62.4)

    # 编码器输出 → 解码器 cross-attn（跨塔记忆流，青色高亮）
    line([(EX, 56.5), (EX, 63.5), (50, 63.5), (50, 41.5)], color=TEAL, lw=1.7)
    arrow(50, 41.5, DX - 10.6, 41.5, color=TEAL, lw=1.7)
    ax.text(48.4, 52.5, "memory (K = V)", fontsize=6.2, color=TEAL, rotation=90,
            ha="center", va="center")
    ax.text(37.0, 65.6, "(B, n_src, 512)", ha="center", fontsize=5.5, color=TEAL)  # 形状标注（本书叠加）

    # ================= 输出头（decoder 顶） =================
    box(DX, 65.5, 14, 4.4, "Linear", "white", EDGE, tc=INK, fs=7)
    arrow(DX, 67.8, DX, 71.4)
    box(DX, 74.0, 14, 4.4, "Softmax", YELLOW, YELLOW, tc=INK, fs=7)
    arrow(DX, 76.3, DX, 79.9)
    ax.text(DX, 84.0, "Output\nProbabilities", ha="center", fontsize=7.5, color=INK)
    ax.text(DX, 88.8, "(B, n_tgt, V)", ha="center", fontsize=5.5, color=SUB)  # 形状标注（本书叠加）

    fig.tight_layout()
    out = FIG_DIR / "fig-4-1-transformer-architecture.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print("saved", out)


# ---------- 图 4.2：三种注意力角色权重矩阵对照（真实计算，非示意） ----------
def make_fig_4_2(n=8, d=16):
    torch.manual_seed(42)  # 与 roles_demo.demo_roles 同种子、同手写 attention，数字与正文同源
    x_enc = torch.randn(n, d)
    y_dec = torch.randn(n, d)
    _, A_enc = attention(x_enc, x_enc, x_enc)
    _, A_dec = attention(y_dec, y_dec, y_dec, causal=True)
    _, A_cross = attention(y_dec, x_enc, x_enc)

    cmap = LinearSegmentedColormap.from_list("bookblue", ["#ffffff", "#cde2fb", "#0d366b"])
    vmax = max(A_enc.max(), A_dec.max(), A_cross.max()).item()   # 三联共用同一色标（诚实可比）
    fig, axes = plt.subplots(1, 3, figsize=(8, 3.2), dpi=300)
    panels = [
        (A_enc, "Encoder self-attention",
         "queries = keys = values = encoder states\n(no mask: all n × n connections visible)\n"
         "A: 8×8 here; (B, h, n_src, n_src) batched"),
        (A_dec, "Decoder self-attention",
         "same mechanism + causal mask\n(only j ≤ i visible)\n"
         "A: 8×8 strictly lower-tri; (B, h, n_tgt, n_tgt)"),
        (A_cross, "Cross-attention",
         "Q = decoder states, K = V = encoder output\n(shape n_tgt × n_src, no mask)\n"
         "A: 8×8 here; (B, h, n_tgt, n_src) batched"),
    ]
    for ax, (A, title, sub) in zip(axes, panels):
        im = ax.imshow(A.numpy(), cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
        ax.set_title(title, fontsize=9, color="#0b0b0b", pad=8)
        ax.text(0.5, -0.235, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=6.5, color="#52514e", linespacing=1.25)
        ax.set_xticks(range(n), [f"k{j}" for j in range(n)], fontsize=5.5, color="#898781")
        ax.set_yticks(range(n), [f"q{i}" for i in range(n)], fontsize=5.5, color="#898781")
        ax.set_xlabel("keys (attended positions)", fontsize=7, color="#898781")
        ax.set_ylabel("queries (attending positions)", fontsize=7, color="#898781")
        ax.set_xticks([j - 0.5 for j in range(1, n)], minor=True)
        ax.set_yticks([i - 0.5 for i in range(1, n)], minor=True)
        ax.grid(which="minor", color="#e1e0d9", linewidth=0.4)
        ax.tick_params(which="minor", length=0)
        for sp in ax.spines.values():
            sp.set_color("#e1e0d9")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.ax.tick_params(labelsize=6, colors="#898781")
    cb.set_label("attention weight", fontsize=7, color="#898781")
    fig.tight_layout()
    out = FIG_DIR / "fig-4-2-three-roles.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    make_fig_4_1()
    make_fig_4_2()
