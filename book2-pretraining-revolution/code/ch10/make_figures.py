# 用途：Book2 ch10 三张图——图 10.1 三插槽 Block（自绘标形状）/ 图 10.2 热身曲线 / 图 10.3 124M 冒烟
# 所属章节：Book2 第 10 章 §10.2、§10.4、§10.5
# 运行方式：cd /Users/allen/Code/model_analysis && source env.sh && python code/ch10/make_figures.py
#          （图 10.2/10.3 依赖 log/book2-ch10/{shakespeare-full,smoke124_bf16,smoke124}.json，先跑 shakespeare.py / train.py；
#            smoke124.json 的 corpus_stats 字段=语料一元账：28.2 万 train token / 5271 活跃类型 / 词频熵 6.51 nat）
# 产物：figures/fig-10-{1,2,3}-*.png（300dpi，图内文字英文、图注见正文）

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(REPO, "figures")
LOG_DIR = os.path.join(REPO, "log", "book2-ch10")
os.makedirs(FIG_DIR, exist_ok=True)

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MUTED = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": TICK, "axes.labelcolor": DARK,
    "xtick.color": TICK, "ytick.color": TICK, "text.color": DARK,
    "axes.linewidth": 0.8, "font.family": "DejaVu Sans",
})


def box(ax, x, y, w, h, label, fc="white", ec=DARK, lw=1.2, fs=8.5, ls="-", tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc or DARK)


def arrow(ax, x1, y1, x2, y2, label=None, color=DARK, lw=1.2, ls="-", lx=0.1, ly=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=lw, linestyle=ls))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=7.2, color=MUTED,
                ha="left", va="center")


def plus(ax, x, y):
    ax.add_patch(plt.Circle((x, y), 0.27, fc="white", ec=DARK, lw=1.2))
    ax.text(x, y, "+", ha="center", va="center", fontsize=11)


# ---------------- 图 10.1：三插槽 Block（自绘示意级，框线 + 形状标注；整机位置左侧小图） ----------------
def fig_10_1():
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis("off")
    XC = 7.7  # 主干中轴

    # ---- 左：整机位置小图（标注「当前部件在整机中的位置」）----
    ax.text(1.5, 10.6, "where we are", fontsize=8.5, color=MUTED, ha="center", style="italic")
    box(ax, 0.4, 9.4, 2.2, 0.8, "wte + wpe\n(B,n,768)", fs=7.2)
    box(ax, 0.4, 7.8, 2.2, 1.0, "Block x 12\n(this one)", fs=7.2, fc="#f7fafe", ec=BLUE, lw=2.0)
    box(ax, 0.4, 6.5, 2.2, 0.65, "ln_f", fs=7.2)
    box(ax, 0.4, 5.2, 2.2, 0.85, "head (= wte)\n(B,n,50257)", fs=7.2)
    arrow(ax, 1.5, 9.4, 1.5, 8.8, lw=1.0)
    arrow(ax, 1.5, 7.8, 1.5, 7.15, lw=1.0)
    arrow(ax, 1.5, 6.5, 1.5, 6.05, lw=1.0)
    ax.text(1.5, 4.6, "dashed frame = slot\n(swap target);\nsolid = fixed topology\n(pre-LN residual)",
            fontsize=7.0, color=MUTED, ha="center", va="top")

    # ---- 右：Block 内部（插槽 attn / mlp 各占一条残差支路）----
    box(ax, XC - 1.9, 10.05, 3.8, 0.62, "x in   (B,n,768)")
    box(ax, XC - 1.1, 8.9, 2.2, 0.55, "ln_1   (768)")
    arrow(ax, XC, 10.05, XC, 9.45, "(B,n,768)", lx=-0.62, ly=0.18)
    # attention 插槽（虚线框 = 改装位）
    ax.add_patch(FancyBboxPatch((4.3, 5.9), 6.8, 2.5, boxstyle="round,pad=0.05",
                                fc="#f2fbf8", ec=TEAL, lw=1.6, ls=(0, (5, 3))))
    ax.text(XC, 8.16, "attention slot", fontsize=8.5, color=TEAL, ha="center")
    box(ax, 4.6, 7.2, 2.9, 0.7, "c_attn  768->2304\n(B,n,768)->(B,n,2304)", fs=6.4)
    box(ax, 8.1, 7.2, 2.7, 0.7, "split Q|K|V + heads\n-> (B,12,n,64) x 3", fs=6.4)
    box(ax, 4.6, 6.15, 2.9, 0.8, "QK^T/sqrt(64) + mask\n-> softmax (B,12,n,n)", fs=6.4)
    box(ax, 8.1, 6.15, 2.7, 0.8, "scores @ V -> merge\nc_proj -> (B,n,768)", fs=6.4)
    arrow(ax, XC, 8.9, XC, 8.4, "(B,n,768)", lx=-0.62, ly=0.1)
    arrow(ax, 7.5, 7.55, 8.1, 7.55, lw=1.0)
    arrow(ax, 8.1, 7.2, 6.6, 6.95, lw=1.0)
    arrow(ax, 7.5, 6.55, 8.1, 6.55, lw=1.0)
    plus(ax, XC, 5.25)
    arrow(ax, XC, 5.9, XC, 5.53, "(B,n,768)", lx=-0.62, ly=0.12)
    ax.plot([XC + 1.9, 11.9, 11.9], [10.05, 10.05, 5.25], color=MUTED, lw=1.0, ls=(0, (3, 2)))
    arrow(ax, 11.9, 5.25, XC + 0.29, 5.25, color=MUTED, lw=1.0, ls=(0, (3, 2)))
    ax.text(12.05, 7.7, "residual", fontsize=7.0, color=MUTED, rotation=90, va="center")
    # 支路二
    box(ax, XC - 1.1, 4.25, 2.2, 0.55, "ln_2   (768)")
    arrow(ax, XC, 4.98, XC, 4.8, "(B,n,768)", lx=-0.62, ly=0.1)
    # MLP 插槽
    ax.add_patch(FancyBboxPatch((4.3, 1.1), 6.8, 2.6, boxstyle="round,pad=0.05",
                                fc="#fdf6f1", ec=ORANGE, lw=1.6, ls=(0, (5, 3))))
    ax.text(XC, 3.4, "MLP slot", fontsize=8.5, color=ORANGE, ha="center")
    box(ax, 4.6, 1.95, 2.2, 0.95, "c_fc  768->3072\n(B,n,768)\n-> (B,n,3072)", fs=6.4)
    box(ax, 7.05, 1.95, 1.85, 0.95, "GELU (tanh approx\n= gelu_new)", fs=6.4)
    box(ax, 9.1, 1.95, 1.85, 0.95, "c_proj  3072->768\n-> (B,n,768)", fs=6.4)
    arrow(ax, 6.8, 2.42, 7.05, 2.42, lw=1.0)
    arrow(ax, 8.9, 2.42, 9.1, 2.42, lw=1.0)
    arrow(ax, XC, 4.25, XC, 3.7, "(B,n,768)", lx=-0.62, ly=0.1)
    plus(ax, XC, 0.55)
    arrow(ax, XC, 1.1, XC, 0.83, "(B,n,768)", lx=-0.62, ly=0.1)
    ax.plot([XC, 13.4, 13.4], [4.8, 4.8, 0.55], color=MUTED, lw=1.0, ls=(0, (3, 2)))
    arrow(ax, 13.4, 0.55, XC + 0.29, 0.55, color=MUTED, lw=1.0, ls=(0, (3, 2)))
    ax.text(13.55, 2.7, "residual", fontsize=7.0, color=MUTED, rotation=90, va="center")
    box(ax, 10.9, 0.24, 3.0, 0.62, "x out  (B,n,768)", ec=BLUE)
    arrow(ax, XC + 0.29, 0.55, 10.9, 0.55, color=BLUE, lw=1.2)
    ax.text(5.9, 0.15, "pre-LN block:   x + attn(ln_1(x));   x + mlp(ln_2(x))",
            fontsize=8.0, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-10-1-block-slots.png"))
    plt.close(fig)


# ---------------- 图 10.2：Shakespeare 热身曲线（自产数据） ----------------
def fig_10_2():
    with open(os.path.join(LOG_DIR, "shakespeare-full.json"), encoding="utf-8") as f:
        d = json.load(f)
    steps = [c["step"] for c in d["curve"]]
    train = [c["train_loss"] for c in d["curve"]]
    val = [c["val_loss"] for c in d["curve"]]
    fig, ax = plt.subplots(figsize=(6.4, 4), dpi=300)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.plot(steps, train, color=BLUE, lw=2, marker="o", ms=4, label="train (running mean)")
    ax.plot(steps, val, color=ORANGE, lw=2, marker="s", ms=4, label="val (one-pass, non-overlap)")
    ax.axhline(1.4697, color=YELLOW, lw=1.6, ls="--")
    ax.text(60, 1.4697 - 0.17, "nanoGPT A100 baseline: best val 1.4697", fontsize=7.5, color=MUTED)
    ax.axvline(1000, color=TICK, lw=0.8, ls=":")
    ax.text(1030, max(val) * 0.42 + min(val) * 0.58, "fast tier\n(1000 steps)", fontsize=7.2, color=MUTED)
    best_i = min(range(len(val)), key=lambda i: val[i])
    ax.annotate(f"best val {val[best_i]:.4f} @ step {steps[best_i]}\n(same best-val metric as nanoGPT's 1.4697)",
                xy=(steps[best_i], val[best_i]), xytext=(1750, 1.58),
                fontsize=7.8, color=DARK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate("val turns up: memorization\n(~80 epochs over 1M chars;\nch4's 47-epoch lesson, larger)",
                xy=(4900, val[-1]), xytext=(1650, 4.35),
                fontsize=7.4, color=DARK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss (nats / char)")
    ax.set_xlim(0, steps[-1] * 1.02)
    ax.set_ylim(0, 5.0)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-10-2-shakespeare-warmup.png"))
    plt.close(fig)


# ---------------- 图 10.3：124M 冒烟（自产数据；log 轴 + 初始化两口径标注） ----------------
def fig_10_3():
    with open(os.path.join(LOG_DIR, "smoke124_bf16.json"), encoding="utf-8") as f:
        d = json.load(f)
    steps = [c["step"] for c in d["curve"]]
    train = [c["train_loss"] for c in d["curve"]]
    val = [c["val_loss"] for c in d["curve"]]
    init = d["meta"]["init_loss_check"]            # 随机初始化实测（std=0.02）
    init_bad = 480.04                              # PyTorch 默认初始化实测（model.py 自测，同种子同输入）
    fig, ax = plt.subplots(figsize=(6.4, 4), dpi=300)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.plot([0] + steps, [init] + val, color=ORANGE, lw=2, marker="s", ms=5, label="val (one-pass)")
    ax.plot(steps, train, color=BLUE, lw=2, marker="o", ms=5, label="train (running mean)")
    ax.plot([0], [init_bad], marker="*", ms=13, color=ORANGE)
    ax.annotate("PyTorch default init: 480\n(embedding N(0,1) -> identity copier)",
                xy=(0, init_bad), xytext=(34, 260), fontsize=7.8, color=DARK,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))
    ax.axhline(math.log(50257), color=TICK, lw=1.2, ls="--")
    ax.text(115, math.log(50257) * 1.1, "ln 50257 = 10.82\n(uniform over vocab)", fontsize=7.5,
            color=MUTED, ha="center")
    ax.axhline(6.508, color=TEAL, lw=1.2, ls="--")
    stats_path = os.path.join(LOG_DIR, "smoke124.json")   # 语料一元账（corpus_stats 字段，批三落盘）
    if os.path.exists(stats_path):
        with open(stats_path, encoding="utf-8") as f:
            cs = json.load(f)["corpus_stats"]
        uni_note = f"unigram entropy {cs['train_unigram_entropy_nat']:.2f}\n({cs['train_active_types']} active types)"
    else:
        uni_note = "unigram entropy 6.51\n(5271 active types)"
    ax.text(112, 6.9, uni_note, fontsize=7.5,
            color=MUTED, ha="center")
    ax.annotate(f"init {init:.2f} = 10.83 + 0.15", xy=(2, init), xytext=(52, 12.8),
                fontsize=7.8, color=DARK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_yscale("log")
    ax.set_xlabel("training step (batch 8 x 256 = 2048 tokens/step)")
    ax.set_ylabel("cross-entropy loss (nats / token, log scale)")
    ax.set_xlim(-8, steps[-1] * 1.05)
    ax.set_ylim(4.5, 900)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-10-3-smoke-124m.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_10_1()
    print("fig 10.1 done")
    if os.path.exists(os.path.join(LOG_DIR, "shakespeare-full.json")):
        fig_10_2()
        print("fig 10.2 done")
    else:
        print("skip fig 10.2: log/book2-ch10/shakespeare-full.json 不存在（先跑 shakespeare.py --steps 5000）")
    if os.path.exists(os.path.join(LOG_DIR, "smoke124_bf16.json")):
        fig_10_3()
        print("fig 10.3 done")
