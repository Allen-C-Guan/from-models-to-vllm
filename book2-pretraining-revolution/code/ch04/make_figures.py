# 用途：Book2 ch4 三张图——图 4.1 GPT-2 整机示意 / 图 4.2 四项手术高亮 / 图 4.3 停药实验 loss 曲线（自产数据）
# 所属章节：Book2 第 4 章 §4.3、§4.5
# 运行方式：cd /Users/allen/Code/model_analysis && source env.sh && python code/ch04/make_figures.py
# 产物：figures/fig-4-{1,2,3}-*.png（300dpi，图内文字英文、图注见正文）

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(REPO, "figures")
CSV_DIR = os.path.join(REPO, "log", "book2-ch04")
os.makedirs(FIG_DIR, exist_ok=True)

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MUTED = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": TICK, "axes.labelcolor": DARK,
    "xtick.color": TICK, "ytick.color": TICK, "text.color": DARK,
    "axes.linewidth": 0.8, "font.family": "DejaVu Sans",
})


def box(ax, x, y, w, h, label, fc="white", ec=DARK, lw=1.2, fs=8.5, style="round,pad=0.02", tc=None, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, ls=ls, mutation_scale=1))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc or DARK)


def arrow(ax, x1, y1, x2, y2, label=None, color=DARK, lw=1.2, dx=0.12, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=lw, linestyle=ls))
    if label:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2, label, fontsize=7.2, color=MUTED, ha="left", va="center")


# ---------------- 图 4.1：GPT-2 整机（自绘示意级，框线 + 形状标注） ----------------
def fig_4_1():
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11.2)
    ax.axis("off")
    # 输入两路
    box(ax, 0.6, 10.0, 3.4, 0.8, "token ids  (B, n)")
    box(ax, 6.0, 10.0, 3.4, 0.8, "position ids  (n,)")
    box(ax, 0.6, 8.6, 3.4, 0.9, "wte lookup\n(50257, 768)")
    box(ax, 6.0, 8.6, 3.4, 0.9, "wpe lookup\n(1024, 768)")
    arrow(ax, 2.3, 10.0, 2.3, 9.5, "(B,n,768)")
    arrow(ax, 7.7, 10.0, 7.7, 9.5, "(n,768)")
    box(ax, 3.4, 7.35, 3.2, 0.8, "add  (broadcast)")
    arrow(ax, 2.3, 8.6, 4.0, 8.15)
    arrow(ax, 7.7, 8.6, 6.0, 8.15)
    arrow(ax, 5.0, 7.35, 5.0, 6.7, "(B,n,768)")
    # 12 × Block（内部两行公式式写法，pre-LN 结构一目了然；细粒度形状见表 4.4）
    ax.add_patch(FancyBboxPatch((1.1, 3.0), 7.8, 3.7, boxstyle="round,pad=0.06",
                                fc="#f7fafe", ec=BLUE, lw=1.6))
    ax.text(5.0, 6.28, "Block × 12   (n_layer=12)", ha="center", fontsize=9, color=BLUE)
    box(ax, 1.7, 4.95, 6.6, 0.85, "x ← x + Attn( ln_1(x) )\ncausal mask, scores (B,h,n,n)")
    box(ax, 1.7, 3.5, 6.6, 0.85, "x ← x + MLP( ln_2(x) )\n768 → 3072 → 768, GELU")
    arrow(ax, 5.0, 4.95, 5.0, 4.35)
    arrow(ax, 5.0, 3.0, 5.0, 2.6, "(B,n,768)")
    # 出口
    box(ax, 3.4, 1.85, 3.2, 0.75, "ln_f  (final LN)")
    box(ax, 2.6, 0.75, 4.8, 0.8, "lm_head = wteᵀ  (tied)", fc="#fdf3ef", ec=ORANGE, lw=1.6)
    arrow(ax, 5.0, 1.85, 5.0, 1.55, "h (B,n,768)")
    box(ax, 2.6, 0.0, 4.8, 0.6, "logits (B, n, 50257) → softmax → next token")
    arrow(ax, 5.0, 0.75, 5.0, 0.6, "")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-4-1-gpt2-architecture.png"), facecolor="white")
    plt.close(fig)


# ---------------- 图 4.2：差异四格高亮（Book1 双塔 → GPT-2 单塔，四刀标号） ----------------
def fig_4_2():
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=300)
    ax.set_xlim(0, 19.6)
    ax.set_ylim(0, 9.4)
    ax.axis("off")
    # 左：Book1 双塔剪影（灰）
    ax.text(3.6, 9.0, "Book1 (2017): two towers", ha="center", fontsize=9, color=MUTED)
    ax.add_patch(FancyBboxPatch((0.5, 1.2), 2.6, 6.9, boxstyle="round,pad=0.05",
                                fc="white", ec=MUTED, lw=1.0, linestyle="--"))
    ax.text(1.8, 7.65, "Encoder ×6", ha="center", fontsize=8, color=MUTED)
    ax.text(1.8, 4.4, "self-attn\n(bidirectional)\nFFN ×6", ha="center", fontsize=7.5, color=MUTED)
    ax.add_patch(FancyBboxPatch((4.9, 1.2), 2.6, 6.9, boxstyle="round,pad=0.05",
                                fc="white", ec=MUTED, lw=1.0, linestyle="--"))
    ax.text(6.2, 7.65, "Decoder ×6", ha="center", fontsize=8, color=MUTED)
    ax.text(6.2, 4.4, "masked\nself-attn\ncross-attn\nFFN ×6", ha="center", fontsize=7.5, color=MUTED)
    arrow(ax, 3.1, 4.6, 4.9, 4.6, "cross", color=MUTED, lw=1.0)
    # 右：GPT-2 单塔（右边缘 13.4）
    ax.text(11.6, 9.0, "GPT-2 (2019): single tower", ha="center", fontsize=9, color=DARK)
    ax.add_patch(FancyBboxPatch((9.8, 2.15), 3.6, 5.95, boxstyle="round,pad=0.05",
                                fc="#f7fafe", ec=BLUE, lw=1.6))
    ax.text(11.6, 7.65, "Block × 12 (decoder-only)", ha="center", fontsize=8.5, color=BLUE)
    ax.text(11.6, 5.1, "x + Attn(ln_1 x)\nx + MLP(ln_2 x)\n(pre-LN)", ha="center", fontsize=8)
    box(ax, 10.3, 2.75, 2.6, 0.6, "wpe (1024, 768)", fs=7.5)
    # 塔底出口（tied head）
    box(ax, 10.15, 1.15, 2.9, 0.6, "lm_head = wteᵀ", fs=7.5)
    arrow(ax, 11.6, 2.15, 11.6, 1.75)
    # 高亮四刀（右侧标签列 x≥14.0，短箭头不穿塔）
    hl = dict(fc="none", ec=ORANGE, lw=1.8, linestyle="--")
    # 刀1：encoder 塔 + cross 被删（圈左侧整体）
    ax.add_patch(FancyBboxPatch((0.25, 1.0), 7.55, 7.3, boxstyle="round,pad=0.06", **hl))
    ax.text(4.0, 0.45, "① tower + cross-attn removed", ha="center", fontsize=8.5, color=ORANGE)
    # 刀2：块内 LN 位置（高亮两行公式）
    ax.add_patch(FancyBboxPatch((10.05, 4.15), 3.1, 2.1, boxstyle="round,pad=0.05", **hl))
    ax.text(14.3, 6.3, "② post-LN → pre-LN\n    + ln_f after block 12", fontsize=8.5,
            color=ORANGE, ha="left", va="center")
    arrow(ax, 14.25, 6.15, 13.2, 5.6, color=ORANGE, lw=1.0)
    # 刀3：wpe
    ax.add_patch(FancyBboxPatch((10.1, 2.6), 3.0, 0.9, boxstyle="round,pad=0.05", **hl))
    ax.text(14.3, 3.3, "③ sinusoidal → learned PE\n    (wpe, hard cap 1024)", fontsize=8.5,
            color=ORANGE, ha="left", va="center")
    arrow(ax, 14.25, 3.3, 13.15, 3.05, color=ORANGE, lw=1.0)
    # 刀4：tied head
    ax.add_patch(FancyBboxPatch((10.0, 1.05), 3.2, 0.8, boxstyle="round,pad=0.05", **hl))
    ax.text(14.3, 1.45, "④ separate output matrix\n    → tied lm_head = wteᵀ", fontsize=8.5,
            color=ORANGE, ha="left", va="center")
    arrow(ax, 14.25, 1.45, 13.25, 1.45, color=ORANGE, lw=1.0)
    ax.text(8.6, 4.9, "→", fontsize=20, ha="center", color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-4-2-four-surgeries.png"), facecolor="white")
    plt.close(fig)


# ---------------- 图 4.3：停药实验 loss 曲线（warmup_full.csv 的 ema 列，自产） ----------------
def fig_4_3():
    cells = [
        ("post_ln__w0", BLUE, "post-LN, no warmup"),
        ("post_ln__w200", ORANGE, "post-LN, warmup 200"),
        ("pre_ln__w0", TEAL, "pre-LN, no warmup"),
        ("pre_ln__w200", YELLOW, "pre-LN, warmup 200"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    for name, color, label in cells:
        path = os.path.join(CSV_DIR, f"warmup_full__{name}.csv")
        steps, ema = [], []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                steps.append(int(row["step"]))
                ema.append(float(row["ema"]))
        ax.plot(steps, ema, color=color, lw=2, label=label)
    # pre 两格的第二种子（虚线，方向复核）
    for name, color in [("pre_ln__w0", TEAL), ("pre_ln__w200", YELLOW)]:
        path = os.path.join(CSV_DIR, f"warmup_full_seed2__{name}.csv")
        if os.path.exists(path):
            steps, ema = [], []
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    steps.append(int(row["step"]))
                    ema.append(float(row["ema"]))
            ax.plot(steps, ema, color=color, lw=1.0, ls=":", alpha=0.8)
    ax.text(1500, 3.42, "frozen at ≈3.3 nat", fontsize=8.5, color=BLUE, ha="center")
    ax.set_xlabel("training step", fontsize=9)
    ax.set_ylabel("train loss (EMA, α=0.01)", fontsize=9)
    ax.set_xlim(0, 3000)
    ax.set_ylim(0, 4.4)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-4-3-warmup-ablation.png"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_4_1()
    fig_4_2()
    fig_4_3()
    for f in sorted(os.listdir(FIG_DIR)):
        if f.startswith("fig-4-"):
            print("产物:", os.path.join(FIG_DIR, f))
