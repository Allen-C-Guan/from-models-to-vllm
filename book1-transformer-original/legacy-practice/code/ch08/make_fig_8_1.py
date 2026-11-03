# -*- coding: utf-8 -*-
# 用途：生成 Book1 第 8 章图 8.1「Noam 学习率调度曲线」（式 (8.1)：base/big 两档，标注 warmup 段与峰值）
# 所属章节：《Transformer 原典》第 8 章 8.1 节；数值与 recipe.py 的 noam_lr 同源
# 运行：source env.sh && python "code/ch08/make_fig_8_1.py"
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, INK, SUB, AXIS, HAIR = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": AXIS,
    "axes.labelcolor": SUB, "xtick.color": AXIS, "ytick.color": AXIS,
    "axes.grid": True, "grid.color": HAIR, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "font.family": "DejaVu Sans", "font.size": 9,
})

FIG_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "figures"))


def noam_lr(step, d_model, warmup=4000):
    """式 (8.1)（原论文式 (3)）。形状：标量或 ndarray step → 同形 lrate（向量化画曲线）。"""
    return d_model ** -0.5 * np.minimum(step ** -0.5, step * warmup ** -1.5)


def main():
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    # 曲线：对数步轴 1→300000（big 训练全程），峰值点 4000 显式入网格
    steps = np.unique(np.concatenate([np.logspace(0, np.log10(300000), 800), [4000.0]]))
    lr_base = noam_lr(steps, 512) * 1e4   # 纵轴按 ×10⁻⁴ 标度
    lr_big = noam_lr(steps, 1024) * 1e4

    ax.axvspan(1, 4000, color="#f3f7fd", zorder=0)            # warmup 段底纹
    ax.axvline(4000, color=AXIS, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.plot(steps, lr_base, color=BLUE, lw=2,
            label="base ($d_{model}$=512): peak $6.99\\times10^{-4}$ at step 4000")
    ax.plot(steps, lr_big, color=ORANGE, lw=2,
            label="big ($d_{model}$=1024): peak $4.94\\times10^{-4}$ at step 4000")

    peak_b, peak_g = noam_lr(4000, 512) * 1e4, noam_lr(4000, 1024) * 1e4
    ax.plot([4000], [peak_b], "o", color=BLUE, ms=6, zorder=5)
    ax.plot([4000], [peak_g], "o", color=ORANGE, ms=6, zorder=5)
    ax.text(63, 6.15, "warmup:\nlinear ramp", ha="center", color=SUB, fontsize=8.5)
    ax.text(90000, 3.35, r"decay $\propto$ step$^{-0.5}$", ha="center", color=SUB, fontsize=8.5)
    ax.text(4000, 7.45, "warmup_steps = 4000", ha="center", color=SUB, fontsize=8)

    ax.set_xscale("log")
    ax.set_xlim(1, 300000)
    ax.set_ylim(0, 7.8)
    ax.set_xticks([1, 10, 100, 1000, 4000, 10000, 100000, 300000])
    ax.set_xticklabels(["1", "10", "100", "1000", "4000", "10K", "100K", "300K"])
    ax.set_yticks(range(0, 8))
    ax.set_xlabel("training step (log scale)", color=SUB)
    ax.set_ylabel(r"lrate  ($\times 10^{-4}$)", color=SUB)
    ax.set_title("Noam schedule (Eq. 8.1): warmup 4000 steps, then inverse-sqrt decay", color=INK)
    ax.tick_params(colors=AXIS)
    for s in ax.spines.values():
        s.set_color(AXIS)
    ax.legend(frameon=False, loc="upper right", fontsize=8, labelcolor=SUB)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-8-1-noam-lr-curve.png")
    fig.savefig(out, facecolor="white")
    print(f"[图 8.1] → {out}（base 峰值 {peak_b:.4f}×10⁻⁴，big 峰值 {peak_g:.4f}×10⁻⁴）")


if __name__ == "__main__":
    main()
