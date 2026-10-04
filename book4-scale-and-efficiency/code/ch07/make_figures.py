# make_figures.py —— Book4 ch7 四张图：位布局 / bulk 断崖 / 分组收益条件性 / 三路线对照
# 用途：图 7.1（自绘示意级：E4M3/E5M2 位分配与动态范围——次正规区与「E4M3 无 Inf」的取舍）；
#       图 7.2（自产：读 fp8_sim JSON 的 exp2——bulk SNR 断崖与 per-group 挽救，含「全体 SNR 撒谎」对照）；
#       图 7.3（自产：读 exp3——分组收益的条件性：平滑分布 ≈1.0× vs 通道离群 3.6-3.8×）；
#       图 7.4（自绘示意级：三路线稳定化对照表图——同一病灶三种药，QK-Norm/QK-Clip/sink+clamp）。
# 所属章节：Book4 第 7 章 §7.2（图 7.1）/§7.3（图 7.2、图 7.3）/§7.8（图 7.4）
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch07/make_figures.py" \
#           [--json run1]      （先跑 fp8_sim.py 生成 log/book4-ch07/fp8_sim_{json}.json）
# 产物：figures/fig-7-1-fp8-bit-layout.png、fig-7-2-bulk-snr-cliff.png、
#       fig-7-3-group-gain.png、fig-7-4-three-routes.png（300dpi；图内文字英文、图注中文见正文）
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DIR = os.path.join(REPO, "figures")
LOG_DIR = os.path.join(REPO, "log", "book4-ch07")
os.makedirs(FIG_DIR, exist_ok=True)

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MUTED = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": TICK, "axes.labelcolor": DARK,
    "xtick.color": TICK, "ytick.color": TICK, "text.color": DARK,
    "axes.linewidth": 0.8, "font.family": "DejaVu Sans",
})


# ---------------- 图 7.1：E4M3/E5M2 位布局与动态范围（自绘示意级） ----------------
def fig_7_1():
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.4), dpi=300)
    specs = [
        ("E4M3  (forward: weights & activations)", 4, 3,
         "max normal 448 = 1.75*2^8   |   min normal 2^-6   |   min subnormal 2^-9",
         "no Inf, single NaN (S.1111.111)  ->  7 extra octave steps 256..448"),
        ("E5M2  (gradients)", 5, 2,
         "max normal 57,344 = 1.75*2^15   |   min normal 2^-14   |   min subnormal 2^-16",
         "follows IEEE 754 custom (±Inf / NaN)  ->  long ruler, coarse ticks"),
    ]
    for ax, (title, e, m, rng, note) in zip(axes, specs):
        ax.set_xlim(0, 8.6)
        ax.set_ylim(0, 3.1)
        ax.axis("off")
        ax.text(0.05, 2.78, title, fontsize=9.5, color=DARK, fontweight="bold")
        # 位单元：1 符号 + e 指数 + m 尾数
        cells = [("S", 1, MUTED)] + [("E", e, BLUE)] + [("M", m, ORANGE)]
        x = 0.3
        for lab, n, c in cells:
            for _ in range(n):
                ax.add_patch(FancyBboxPatch((x, 1.45), 0.62, 0.85,
                                            boxstyle="round,pad=0.02", fc="white", ec=c, lw=1.4))
                ax.text(x + 0.31, 1.875, lab, ha="center", va="center", fontsize=8.5, color=c)
                x += 0.68
        ax.text(0.05, 1.05, rng, fontsize=7.8, color=DARK)
        ax.text(0.05, 0.55, note, fontsize=7.8, color=MUTED)
        if e == 4:  # 量程对比小图：两把尺子
            ax.text(6.55, 2.35, "dynamic range (normal):\nE4M3  2^14.8 ~ 2.9e4\nE5M2  2^29  ~ 5.4e8",
                    fontsize=7.2, color=DARK,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#f2f7fd", ec=BLUE, lw=0.9))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-7-1-fp8-bit-layout.png"))
    plt.close(fig)


# ---------------- 图 7.2：bulk SNR 断崖与分组挽救（自产，读 exp2） ----------------
def fig_7_2(rep):
    exp2 = rep["exp2_bulk_cliff"]
    tags = [("mild_x30", "outlier x30\n(max/med ~1.9e2)"),
            ("strong_x300", "outlier x300\n(~1.8e3)"),
            ("extreme_x30000", "outlier x30000\n(~1.7e5)")]
    series = [("E4M3_per_tensor", "overall_db", "per-tensor overall SNR", BLUE, "//"),
              ("E4M3_per_tensor", "bulk_only_db", "per-tensor bulk SNR", ORANGE, None),
              ("E4M3_per_group128", "bulk_only_db", "per-group(128) bulk SNR", TEAL, None)]
    fig, ax = plt.subplots(figsize=(6.4, 4), dpi=300)
    xs = range(len(tags))
    w = 0.26
    for i, (key, metric, lab, c, hat) in enumerate(series):
        vals = [exp2[t][key][metric] for t, _ in tags]
        ax.bar([x + (i - 1) * w for x in xs], vals, w, label=lab, color=c,
               hatch=hat, edgecolor="white", linewidth=0.5)
    # 断崖标注：x30000 档 per-tensor bulk
    cliff = exp2["extreme_x30000"]["E4M3_per_tensor"]["bulk_only_db"]
    top = exp2["extreme_x30000"]["E4M3_per_tensor"]["overall_db"]
    ax.annotate(f"cliff {top:.1f} -> {cliff:.1f} dB\n(bulk median after scale\n"
                f"{exp2['extreme_x30000']['bulk_median_after_pertensor_scale']:.4f}"
                " < 2^-6 subnormal)",
                xy=(2 - w, cliff + 0.4), xytext=(1.02, 20.5), fontsize=7.6, color=DARK,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax.axhline(31.4, color=MUTED, lw=0.8, ls="--")
    ax.text(-0.42, 31.7, "healthy bulk SNR ~31.5 dB", fontsize=7.2, color=MUTED)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([lab for _, lab in tags], fontsize=8)
    ax.set_ylabel("SNR (dB)")
    ax.set_ylim(0, 36)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-7-2-bulk-snr-cliff.png"))
    plt.close(fig)


# ---------------- 图 7.3：分组收益的条件性（自产，读 exp3） ----------------
def fig_7_3(rep):
    g = rep["exp3_group_gain"]
    rows = [("gauss 4096x512\n(smooth)", "E4M3/gauss_4096x512"),
            ("real residual stream\n(207M layer-6, smooth)", "E4M3/real_act_512x1024"),
            ("channel outliers x100\n(5/512 cols)", "E4M3/outlier_x100_4096x512"),
            ("channel outliers x1000\n(5/512 cols)", "E4M3/outlier_x1000_4096x512")]
    labs = [r[0] for r in rows]
    vals = [g[r[1]]["group_gain_x"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4), dpi=300)
    cs = [BLUE, BLUE, ORANGE, ORANGE]
    bars = ax.bar(range(len(vals)), vals, 0.55, color=cs, edgecolor="white", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}x", ha="center",
                fontsize=8.5, color=DARK)
    ax.axhline(1.0, color=MUTED, lw=1.0, ls="--")
    ax.text(2.98, 1.1, "1.0x = no gain", fontsize=7.4, color=MUTED, ha="right")
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, fontsize=7.8)
    ax.set_ylabel("per-group(128) error reduction (x)")
    ax.set_ylim(0, 4.4)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-7-3-group-gain.png"))
    plt.close(fig)


# ---------------- 图 7.4：三路线稳定化对照（自绘示意级表格图） ----------------
def fig_7_4():
    fig, ax = plt.subplots(figsize=(8.4, 3.7), dpi=300)
    ax.set_xlim(0, 12.7)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    xs = {"route": 0.18, "where": 2.9, "mech": 5.75, "who": 9.55, "mla": 11.75}
    hdr = {"route": "route", "where": "where it acts", "mech": "mechanism (one line)",
           "who": "adopters", "mla": "works on\nMLA?"}
    for k, x in xs.items():
        ax.text(x, 6.35, hdr[k], fontsize=8.2, color=DARK, fontweight="bold",
                ha="left" if k != "mla" else "center", va="top")
    ax.plot([0.08, 12.62], [5.62, 5.62], color=DARK, lw=1.0)
    rows = [
        ("QK-Norm\n(Henry 2020)",
         "before softmax: normalize\nq, k per head (L2 / RMS)",
         "q·k^T becomes cosine,\nbounded in [-1,1]; scale g",
         "Qwen3 / Gemma3\n(source-level)",
         "no (K not\nmaterialized)"),
        ("QK-Clip\n(K2 MuonClip)",
         "after the step: rescale\nW_q, W_k weights per head",
         "head max logit S>tau=100\ntriggers weight rescale",
         "Kimi K2 (15.5T,\nzero spike)",
         "yes (4-part\nvariant)"),
        ("sink + clamp\n(gpt-oss)",
         "on logits: 1 learnable scalar\nper head + training clamp",
         "extra column in softmax\ndenominator; clamp max",
         "gpt-oss (+ SwiGLU\nlimit 7.0)",
         "yes"),
    ]
    ys = [4.62, 2.8, 0.98]
    cs = [BLUE, ORANGE, TEAL]
    for (r, y, c) in zip(rows, ys, cs):
        ax.add_patch(FancyBboxPatch((0.08, y - 0.8), 12.54, 1.66,
                                    boxstyle="round,pad=0.02", fc="white", ec=c, lw=1.2))
        ax.text(xs["route"], y, r[0], fontsize=8.0, color=c, fontweight="bold", va="center")
        ax.text(xs["where"], y, r[1], fontsize=7.2, color=DARK, va="center")
        ax.text(xs["mech"], y, r[2], fontsize=7.2, color=DARK, va="center")
        ax.text(xs["who"], y, r[3], fontsize=7.2, color=DARK, va="center")
        ax.text(xs["mla"], y, r[4], fontsize=7.2, color=DARK, va="center", ha="center")
    ax.text(0.08, 0.1, "same lesion (exploding attention logits), three prescriptions — "
                       "they differ in where the hands are put", fontsize=7.6, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-7-4-three-routes.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="run1", help="fp8_sim.py 产物后缀（log/book4-ch07/fp8_sim_*.json）")
    args = ap.parse_args()
    with open(os.path.join(LOG_DIR, f"fp8_sim_{args.json}.json"), encoding="utf-8") as f:
        rep = json.load(f)
    fig_7_1()
    fig_7_2(rep)
    fig_7_3(rep)
    fig_7_4()
    print("[产物] fig-7-1..fig-7-4 ->", FIG_DIR)


if __name__ == "__main__":
    main()
