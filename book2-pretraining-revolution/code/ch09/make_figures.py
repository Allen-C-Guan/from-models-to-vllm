# 用途：Book2 ch9 两张图——图 9.1 显存构成（理论记账 vs 本机实测峰值，124M，b4×n1024）
#   / 图 9.2 训练不稳定时间线（BERT 2018 -> OPT logbook 2021.11-2022.01 -> PaLM 2022.4，附 spike-回滚小示意图）
# 所属章节：Book2 第 9 章 §9.2、§9.6
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch09/make_figures.py"
# 产物：figures/fig-9-{1,2}-*.png（300dpi，图内文字英文、图注见正文）

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(REPO, "figures")
LOG_DIR = os.path.join(REPO, "log", "book2-ch09")
os.makedirs(FIG_DIR, exist_ok=True)

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MUTED = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": TICK, "axes.labelcolor": DARK,
    "xtick.color": TICK, "ytick.color": TICK, "text.color": DARK,
    "axes.linewidth": 0.8, "font.family": "DejaVu Sans",
})


# ---------------- 图 9.1：显存构成（理论记账 vs 实测峰值；数据=memory_probe.json） ----------------
def fig_9_1():
    probe = json.load(open(os.path.join(LOG_DIR, "memory_probe.json")))
    th = probe["theory"]
    segs = [  # (label, GiB, color) —— 理论记账四件（fp32 口径，B=4,n=1024）
        ("model state\n16 B/param", th["state_gib"], BLUE),
        ("linear acts\n12·B·n·d·L", th["act_linear_gib"], TEAL),
        ("attn scores\nB·h·n²·L", th["act_attn_gib"], ORANGE),
        ("logits side\n2·B·n·V", th["logits_gib"], YELLOW),
    ]
    fp32_peak = probe["fp32"]["peak_mib"] / 1024
    ckpt_peak = probe["ckpt_fp32"]["peak_mib"] / 1024

    fig, ax = plt.subplots(figsize=(6.4, 4), dpi=300)
    xs = [0, 1, 2]
    # 柱 1：理论记账（保留集口径，分四段堆叠）
    bot = 0.0
    for label, v, c in segs:
        ax.bar(0, v, bottom=bot, width=0.55, color=c, lw=0,
               label=label.replace("\n", "  "))
        if v > 0.55:
            ax.text(0, bot + v / 2, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.8, color="white")
        bot += v
    ax.text(0, bot + 0.18, f"{bot:.2f} GiB", ha="center", fontsize=8.4, color=DARK)
    # 柱 2/3：实测峰值（池口径，不可分解 —— 单色+影线）
    ax.bar(1, fp32_peak, width=0.55, color="#b5b4ad", hatch="//", lw=0,
           edgecolor="#898781")
    ax.text(1, fp32_peak + 0.18, f"{fp32_peak:.2f} GiB", ha="center", fontsize=8.4, color=DARK)
    ax.bar(2, ckpt_peak, width=0.55, color="#d8d7d0", hatch="//", lw=0,
           edgecolor="#898781")
    ax.text(2, ckpt_peak + 0.18, f"{ckpt_peak:.2f} GiB", ha="center", fontsize=8.4, color=DARK)
    # 检查点增益箭头（柱 2 -> 柱 3；数字直读 JSON，与产物保持一致）
    d_gib = ckpt_peak - fp32_peak
    pct = d_gib / fp32_peak * 100
    t_pct = probe["ckpt_fp32"]["time_overhead_pct"]
    ax.annotate("", xy=(2.32, ckpt_peak + 1.2), xytext=(1.28, fp32_peak - 0.9),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.text(1.8, 7.3, f"gradient ckpt:\n{d_gib:+.2f} GiB ({pct:.0f}%)\n+{t_pct:.0f}% step time",
            ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels(["theory ledger\n(retained set)", "measured peak\n(fp32)",
                        "measured peak\n(fp32 + ckpt)"], fontsize=8.4)
    ax.set_ylabel("GiB (MPS pool)", fontsize=8.6)
    ax.set_ylim(0, 12.4)
    ax.set_title("Training-memory ledger vs measured peaks — GPT-2 124M, B=4, n=1024",
                 fontsize=9)
    ax.legend(loc="upper right", fontsize=7.4, framealpha=0.95)
    ax.grid(True, axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-9-1-memory-stack.png"), facecolor="white")
    plt.close(fig)


# ---------------- 图 9.2：训练不稳定时间线（三站 + spike-回滚小示意图） ----------------
def fig_9_2():
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.set_xlim(2017.6, 2022.9)
    ax.set_ylim(-1.25, 1.35)

    # 时间轴
    ax.axhline(0, color=TICK, lw=1.2, zorder=1)
    for yr in range(2018, 2023):
        ax.plot([yr], [0], "o", ms=4, color=TICK, zorder=2)
        ax.text(yr, -0.13, str(yr), ha="center", fontsize=8, color=TICK)

    def station(x, name, lines, up, color):
        y0 = 0.42 if up else -0.42
        ax.plot([x], [0], "o", ms=8, color=color, zorder=3)
        ax.annotate("", xy=(x, y0), xytext=(x, 0.06 if up else -0.06),
                    arrowprops=dict(arrowstyle="-", color=color, lw=1.0))
        va = "bottom" if up else "top"
        ax.text(x, y0 + (0.03 if up else -0.03), name, ha="center", va=va,
                fontsize=8.6, color=DARK, fontweight="bold")
        ax.text(x, y0 + (0.22 if up else -0.22), "\n".join(lines), ha="center", va=va,
                fontsize=7.3, color=MUTED, linespacing=1.45)

    station(2018.45, "BERT-LARGE finetune (2018)", [
        "small-data finetune \"sometimes unstable\"",
        "fix: multiple random restarts,",
        "pick best dev  (1810.04805 §4.1, A.3)",
    ], up=False, color=TEAL)

    station(2022.3, "PaLM 540B (2022)", [
        "~20 loss spikes despite grad clipping",
        "(global norm 1.0); none in smaller runs",
        "restart −100 steps, skip 200–500 batches;",
        "cause: specific batches × param state",
        "(2204.02311 §5.1)",
    ], up=False, color=BLUE)

    # OPT：一段持续区间（2021-11-05 -> 2022-01-06）
    x0, x1 = 2021.85, 2022.02
    ax.plot([x0, x1], [0, 0], "-", lw=6, color=ORANGE, solid_capstyle="butt", zorder=2)
    ax.annotate("", xy=(x1 + 0.04, 0.14), xytext=(x0 - 0.04, 0.14),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.0))
    ax.text((x0 + x1) / 2, 0.2,
            "OPT-175B logbook: 2021-11-05 “Run 11.0: LETS GO” → done 2022-01-06\n"
            "≥35 manual + 70+ auto restarts · dynamic loss scale → 0 · clip 2.5→…→0.3\n"
            "SGD trial failed → back to AdamW  (2205.01068; logbook PDF)",
            ha="center", va="bottom", fontsize=7.3, color=MUTED, linespacing=1.45)
    ax.text((x0 + x1) / 2, 0.02, "", fontsize=7)

    # 内嵌小图：loss spike 与「回跳 checkpoint + 跳批」处置示意（置于左上空白区，避让 OPT 文块与 PaLM 站）
    axin = fig.add_axes([0.06, 0.52, 0.235, 0.36])
    rng = np.random.default_rng(3)
    s = np.arange(1, 261)
    loss = 3.4 * s**-0.18 + 1.2 + rng.normal(0, 0.012, s.size)
    spike_i = 190
    loss[spike_i] += 1.9
    loss[spike_i + 1] += 1.1
    loss[spike_i + 2] += 0.4
    axin.plot(s, loss, lw=1.6, color=BLUE)
    s2 = s[spike_i - 40:]
    loss2 = 3.4 * s2**-0.18 + 1.2 + rng.normal(0, 0.012, s2.size)
    axin.plot(s2, loss2, lw=1.6, color=TEAL)
    axin.annotate("", xy=(spike_i - 40, loss2[0]), xytext=(spike_i + 14, 2.6),
                  arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))
    axin.text(spike_i - 36, 2.75, "restart −100 steps,\nskip 200–500 batches",
              fontsize=6.6, color=DARK)
    axin.set_title("loss spike & rollback (schematic)", fontsize=7)
    axin.set_xticks([])
    axin.set_yticks([])
    for sp in ("top", "right"):
        axin.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axin.spines[sp].set_color(TICK)

    ax.set_title("A phenomenology of training instability at scale (2018–2022)",
                 fontsize=9.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-9-2-instability-timeline.png"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_9_1()
    fig_9_2()
    print("[产物] fig-9-1-memory-stack.png / fig-9-2-instability-timeline.png ->", FIG_DIR)
