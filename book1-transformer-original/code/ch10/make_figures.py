# -*- coding: utf-8 -*-
# 用途：生成 Book1 第 10 章图 10.1「消融 ΔBLEU 与种子噪声地板」（双联：(a) base 三种子 + σ/阈值带 + ch9 交叉核验点；
#       (b) 10 行消融的 ΔBLEU 横条按 |Δ| 排序，叠加 max(2σ,0.5)=0.90 显著带与 0.5 下限线）。数据只读 log/Book1-ch10/grid_results.json。
# 所属章节：《Transformer 原典》第 10 章（效果实证：趋势级复现与引用对照）10.2/10.4 节
# 运行：source env.sh && python "code/ch10/make_figures.py"
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, SUB, AXIS, HAIR = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": AXIS,
    "axes.labelcolor": SUB, "xtick.color": AXIS, "ytick.color": AXIS,
    "axes.grid": True, "grid.color": HAIR, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "font.family": "DejaVu Sans", "font.size": 9,
})

HERE = os.path.dirname(__file__)
RESULTS = os.path.normpath(os.path.join(HERE, "..", "..", "..", "log", "Book1-ch10", "grid_results.json"))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "figures"))
CH9_EVAL = os.path.normpath(os.path.join(HERE, "..", "..", "..", "log", "Book1-ch09", "eval_transformer.json"))

# 消融行的图面标签（英文，图内文字英文、图注中文的规范口径）
LABELS = {"h=1": "h=1", "h=4": "h=4", "h=16": "h=16", "d_k=16": "d_k=16", "N=2": "N=2",
          "Pdrop=0": "Pdrop=0", "eps_ls=0": "eps_ls=0", "learned_pe": "learned PE",
          "no_pe": "no PE", "no_scale": "no 1/sqrt(d_k)"}


def main():
    with open(RESULTS, encoding="utf-8") as f:
        res = json.load(f)
    s = res["summary"]
    mean, sigma, thr = s["base_mean_bleu"], s["base_sigma_bleu"], s["threshold"]
    seeds = [r["bleu_13a"] for r in res["runs"] if r["row"] == "base"]
    rows = sorted(s["rows"], key=lambda r: abs(r["delta_bleu"]))  # |Δ| 小者画在下端，大者置顶
    with open(CH9_EVAL, encoding="utf-8") as f:
        ch9 = json.load(f)["bleu_13a"]
    print(f"[数据] base 种子 {seeds} → 均值 {mean}，σ {sigma}，阈值 max(2σ,0.5)={thr}；ch9 交叉核验点 {ch9}")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8, 4), dpi=300, gridspec_kw={"width_ratios": [1, 1.55]})

    # ── (a) 三种子 + σ 带 + 阈值带 + ch9 交叉核验点 ─────────────────────────
    axa.axvspan(mean - sigma, mean + sigma, color="#dceafd", zorder=0)           # ±σ 带
    axa.axvspan(mean - 2 * sigma, mean - sigma, color="#f3f2ef", zorder=0)       # 阈值带外缘（中性灰）
    axa.axvspan(mean + sigma, mean + 2 * sigma, color="#f3f2ef", zorder=0)
    axa.axvline(mean, color=BLUE, lw=2, zorder=2)
    axa.axvline(mean - thr, color=AXIS, lw=1, ls=(0, (4, 3)), zorder=1)
    axa.axvline(mean + thr, color=AXIS, lw=1, ls=(0, (4, 3)), zorder=1)
    for i, b in enumerate(seeds):  # 三种子：蓝圆，标签只标最值与中间（选择性直标）
        axa.plot([b], [0.72 + 0.06 * i], "o", color=BLUE, ms=6, zorder=4)
    axa.plot([ch9], [0.30], "D", color=ORANGE, ms=6, zorder=4)
    axa.text(seeds[0], 0.90, f"{seeds[0]:.2f}", ha="center", va="bottom", color=SUB, fontsize=8)
    axa.text(seeds[1], 0.90, f"{seeds[1]:.2f}", ha="center", va="bottom", color=SUB, fontsize=8)
    axa.text(seeds[2], 0.90, f"{seeds[2]:.2f}", ha="center", va="bottom", color=SUB, fontsize=8)
    axa.text(ch9, 0.14, f"ch9 run {ch9:.2f}", ha="center", va="top", color=SUB, fontsize=8)
    axa.text(mean, 1.06, f"mean {mean:.2f}", ha="center", va="bottom", color=INK, fontsize=8.5)
    axa.text(mean + sigma + 0.012, 0.06, r"$\pm\sigma$ = %.2f" % sigma, ha="left", va="bottom",
             color=SUB, fontsize=8)
    axa.text(mean + thr + 0.012, 0.06, r"$\pm2\sigma$ = %.2f" % thr, ha="left", va="bottom",
             color=SUB, fontsize=8)
    axa.set_xlim(21.0, 23.1)
    axa.set_ylim(0, 1.25)
    axa.set_yticks([])
    axa.set_xlabel("test BLEU (sacreBLEU 13a), base x 3 seeds", color=SUB)
    axa.set_title("(a) seed noise floor", color=INK, fontsize=10)
    axa.plot([], [], "o", color=BLUE, ms=6, label="base, seeds 0/1/2")
    axa.plot([], [], "D", color=ORANGE, ms=6, label="ch9 run (seed 2017)")
    axa.legend(frameon=False, loc="upper right", fontsize=8, labelcolor=SUB)

    # ── (b) ΔBLEU 横条：按 |Δ| 排序；超阈值橙、未超蓝；阈值带 + 0.5 下限线 ────
    names = [LABELS[r["row"]] for r in rows]
    deltas = [r["delta_bleu"] for r in rows]
    colors = [ORANGE if r["beyond_seed_noise"] else BLUE for r in rows]
    ys = range(len(rows))
    axb.axvspan(-thr, thr, color="#f3f2ef", zorder=0)  # 未显著区（=阈值带）
    axb.axvline(0, color=AXIS, lw=1, zorder=1)
    axb.barh(ys, deltas, height=0.62, color=colors, zorder=2)
    for y, d in zip(ys, deltas):  # 条端选择性直标（每条一值，量少可全标）
        axb.text(d + (0.05 if d >= 0 else -0.05), y, f"{d:+.2f}",
                 ha="left" if d >= 0 else "right", va="center", color=SUB, fontsize=8)
    for x in (0.5, -0.5):
        axb.axvline(x, color=AXIS, lw=0.8, ls=(0, (1, 2)), zorder=1)
    axb.text(-thr, -0.68, f"not significant: |delta| < max(2sigma, 0.5) = {thr:.2f}",
             ha="left", va="top", color=SUB, fontsize=8)
    axb.text(0.5, len(rows) - 0.45, "0.5 floor", ha="left", va="center", color=SUB, fontsize=7.5)
    axb.set_yticks(list(ys))
    axb.set_yticklabels(names)
    axb.set_xlim(-2.95, 1.05)
    axb.set_ylim(-0.9, len(rows) - 0.35)
    axb.set_xlabel(r"$\Delta$BLEU vs base mean (%.2f), single seed" % mean, color=SUB)
    axb.set_title("(b) ablation deltas vs Table 3 directions", color=INK, fontsize=10)
    axb.plot([], [], "s", color=ORANGE, ms=6, label="beyond noise")
    axb.plot([], [], "s", color=BLUE, ms=6, label="within noise")
    axb.legend(frameon=False, loc="lower left", fontsize=8, labelcolor=SUB)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-10-1-ablation-delta-bleu.png")
    fig.savefig(out, facecolor="white")
    print(f"[图 10.1] → {out}")


if __name__ == "__main__":
    main()
