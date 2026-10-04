# make_figures.py —— Book3 ch7 五张图的绘制脚本（图 7.1-7.5，按正文出现顺序编号：
#       7.1 直接外推 / 7.2 位置分桶 / 7.3 passkey / 7.4 机制示意 / 7.5 四干预招牌图）
# 用途：产出 figures/fig-7-{1..5}-*.png（300dpi 印刷规格，图内英文、
#       图注中文在正文）。图 7.1/7.2/7.5 依赖 extrapolate_{out}.json、图 7.3 依赖 passkey_{out}.json
#       （均在 log/book3-ch07/，full 档缺时可用 --ablation fast 先出形态）；图 7.4 机制示意不依赖实验。
# 所属章节：Book3 ch7（7.2 图 7.1/7.2/7.3 / 7.6 图 7.4 / 7.7 图 7.5）。
# 运行方式：cd code/ch07 && python make_figures.py [--only 4] [--ablation full]
# 规格：写作规范 §4——分类色固定序 蓝 #2a78d6 / 橙 #eb6834 / 青 #1baf7a / 黄 #eda100；
#       四干预配色全书固定 none=蓝 / pi=橙 / ntk=青 / yarn_lite=黄（色随实体，跨图不换）；
#       线宽 2、标记 ≥6pt、网格 hairline #e1e0d9 置底层、刻度 #898781、标注 #0b0b0b/#52514e；
#       文字不用系列色；≥2 系列有图例；禁双 y 轴；tight_layout。
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DIR = os.path.join(REPO, "figures")
LOG_DIR = os.path.join(REPO, "log", "book3-ch07")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MID = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
IV_COLOR = {"none": BLUE, "pi": ORANGE, "ntk": TEAL, "yarn_lite": YELLOW}
IV_MARKER = {"none": "o", "pi": "s", "ntk": "^", "yarn_lite": "D"}
IV_LABEL = {"none": "none (direct extrapolation)", "pi": "PI (m$\\to$m/s)",
            "ntk": "NTK (base 10k$\\to$41.8k)", "yarn_lite": "YaRN-lite (base+temp)"}
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9.5,
    "axes.edgecolor": TICK, "axes.labelcolor": DARK, "xtick.color": TICK,
    "ytick.color": TICK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "font.family": "sans-serif",
})

WINDOWS = [2048, 8192, 16384]
# 探针 G 600 步档（notes/01 探针 G：2k 5.2387 / 8k 5.3296 / 16k 5.3922）——「弱模型只见渐进」对照
PROBE_G_600 = {2048: 5.2387, 8192: 5.3296, 16384: 5.3922}


def style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def load_extrapolate(out_name):
    path = os.path.join(LOG_DIR, f"extrapolate_{out_name}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["intervention_eval"]


def merge_buckets(buckets, target):
    """把分桶字典按目标边界重箱（加权平均）。target=[0,1024,2048,4096,8192(,16384)]。"""
    # 原边界可能是 [0,512,1024,2048,...]——按 token 计数精确合并
    edges = sorted(int(k.strip("pos[]").split(":")[0]) for k in buckets)
    edges.append(max(int(k.strip("pos[]").split(":")[1]) for k in buckets))
    vals = np.array(list(buckets.values()), dtype=float)
    widths = np.diff(np.array(edges))
    out = []
    for lo, hi in zip(target[:-1], target[1:]):
        m = (np.array(edges[:-1]) >= lo) & (np.array(edges[1:]) <= hi)
        out.append(float((vals[m] * widths[m]).sum() / widths[m].sum()) if m.any() else np.nan)
    return out


# ---------------- 图 7.1：直接外推退化曲线（none：full 档 vs 600 步弱模型对照） ----------------
def fig_7_1(out_name):
    ev = load_extrapolate(out_name)
    y_full = [ev["none"][str(n)]["loss"] for n in WINDOWS]
    y_weak = [PROBE_G_600[n] for n in WINDOWS]
    base = y_full[0]

    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(WINDOWS, y_full, color=BLUE, lw=2, marker="o", ms=7,
            label=f"none, {out_name} ({5000 if out_name == 'full' else 2000} steps)")
    ax.plot(WINDOWS, y_weak, color=MID, lw=2, ls="--", marker="o", ms=6, mfc="white",
            label="none, 600-step probe (weak model)")
    for n, v in zip(WINDOWS, y_full):
        ax.annotate(f"+{v - base:.2f} nat", xy=(n, v), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8, color=DARK)
    ax.set_xticks(WINDOWS)
    ax.set_xticklabels(["2048\n(train window)", "8192 (4x)", "16384 (8x)"])
    ax.set_xlabel("evaluation window (tokens)", fontsize=9)
    ax.set_ylabel("eval loss (nat, fp32)", fontsize=9)
    ax.set_title("Direct extrapolation degrades — softly", fontsize=9.5, color=DARK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style_axis(ax)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-7-1-direct-extrapolation-ppl.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig 7.1] {out}（{out_name} 档）")


# ---------------- 图 7.2：位置分桶（none@8k / none@16k，训练窗底纹；折线不用截断条形） ----------------
def fig_7_2(out_name):
    ev = load_extrapolate(out_name)
    b8 = merge_buckets(ev["none"]["8192"]["buckets"], [0, 1024, 2048, 4096, 8192])
    b16 = merge_buckets(ev["none"]["16384"]["buckets"], [0, 1024, 2048, 4096, 8192, 16384])
    edges = [1024, 2048, 4096, 8192, 16384]          # 各桶上缘（桶=[上缘/2, 上缘]）

    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(edges[:4], b8[:4], color=BLUE, lw=2, marker="o", ms=7, label="window 8192")
    ax.plot(edges, b16, color=ORANGE, lw=2, marker="s", ms=6, label="window 16384")
    ax.axvspan(500, 2048, color="#f4f3ef", zorder=0)
    ax.text(1260, b8[0] - 0.06, "inside train window", ha="center", fontsize=7.5, color=MID)
    for xx, v in list(zip(edges[:4], b8[:4])) + list(zip(edges, b16)):
        ax.annotate(f"{v:.2f}", xy=(xx, v), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=7.5, color=DARK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(edges)
    ax.set_xticklabels(["0-1k", "1k-2k", "2k-4k", "4k-8k", "8k-16k"])
    ax.set_xlabel("position bucket (tokens from sequence start)", fontsize=9)
    ax.set_ylabel("bucket eval loss (nat)", fontsize=9)
    ax.set_title("none: position buckets — cliff at the train window edge", fontsize=9.5, color=DARK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style_axis(ax)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-7-2-position-buckets.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig 7.2] {out}（{out_name} 档）")


# ---------------- 图 7.5：四干预对照（招牌图） ----------------
def fig_7_5(out_name):
    ev = load_extrapolate(out_name)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for iv in ("none", "pi", "ntk", "yarn_lite"):
        y = [ev[iv][str(n)]["loss"] for n in WINDOWS]
        ax.plot(WINDOWS, y, color=IV_COLOR[iv], lw=2, marker=IV_MARKER[iv], ms=7, label=IV_LABEL[iv])
        ax.annotate(f"{y[-1]:.2f}", xy=(WINDOWS[-1], y[-1]), xytext=(6, -3),
                    textcoords="offset points", fontsize=8, color=MID)
    ax.axvline(2048, color=TICK, lw=1, ls=":")
    ax.text(2048, ax.get_ylim()[0] + 0.02, "train window", fontsize=7.5, color=MID,
            ha="right", rotation=90, va="bottom")
    ax.set_xticks(WINDOWS)
    ax.set_xticklabels(["2048", "8192 (4x)", "16384 (8x)"])
    ax.set_xlim(1500, 18500)
    ax.set_xlabel("evaluation window (tokens)", fontsize=9)
    ax.set_ylabel("eval loss (nat, fp32)", fontsize=9)
    ax.set_title("Four RoPE interventions, one checkpoint, zero retraining (s=4)", fontsize=9.5, color=DARK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style_axis(ax)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-7-5-four-interventions.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig 7.5] {out}（{out_name} 档）")


# ---------------- 图 7.3：passkey 准确率 vs 上下文（四干预） ----------------
def fig_7_3(out_name):
    path = os.path.join(LOG_DIR, f"passkey_{out_name}.json")
    with open(path, encoding="utf-8") as f:
        pk = json.load(f)
    lengths = sorted({int(k) for k in pk["by_length_summary"][next(iter(pk["by_length_summary"]))]})
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for iv in ("none", "pi", "ntk", "yarn_lite"):
        y = [pk["by_length_summary"][iv].get(str(n), 0.0) for n in lengths]
        ax.plot(lengths, y, color=IV_COLOR[iv], lw=2, marker=IV_MARKER[iv], ms=7, label=IV_LABEL[iv])
    ax.axvline(2048, color=TICK, lw=1, ls=":")
    ax.text(2048, ax.get_ylim()[0] + 0.02, "train window", fontsize=7.5, color=MID,
            ha="right", rotation=90, va="bottom")
    ax.set_xticks(lengths)
    ax.set_xticklabels([f"{n}\n({'in-window' if n < 2048 else f'{n // 2048}x over'})" for n in lengths])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("context length (tokens)", fontsize=9)
    ax.set_ylabel("passkey accuracy (exact match)", fontsize=9)
    n_tr = pk["meta"]["trials"]
    all_zero = all(v == 0.0 for iv in pk["by_length_summary"] for v in pk["by_length_summary"][iv].values())
    if all_zero:
        ax.text(0.5, 0.5, "all cells 0/{} hits — weak-model clause engaged:\nprotocol validation only, not a quality read"
                .format(n_tr), transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color=MID)
    ax.set_title(f"Passkey retrieval vs context ({n_tr} trials/point, greedy 12 tokens)",
                 fontsize=9.5, color=DARK)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    style_axis(ax)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-7-3-passkey-accuracy.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig 7.3] {out}（{out_name} 档）")


# ---------------- 图 7.4：三代配方的干预点（频率谱上的改写系数，机制示意） ----------------
def fig_7_4():
    D, L, s, base = 64, 2048, 4, 1e4          # 本机 mini 档：head_dim=64 / 训练窗 2048 / s=4
    i = np.arange(D // 2)                      # 频率对下标：0=最快（θ=1），31=最慢
    lam = 2 * np.pi * base ** (2 * i / D)      # 各维波长（token 数）

    pi_ratio = np.full_like(i, 1.0 / s, dtype=float)                       # PI：全部 1/s
    ntk_ratio = s ** (-2 * i / (D - 2))                                    # NTK：i=0 处 1 → 最慢处 ~1/s
    # YaRN by-parts（式 17-20）：r=L/λ，γ 斜坡 α=1/β=32 → ratio=(1-γ)/s+γ
    r = L / lam
    gamma = np.clip((r - 1) / (32 - 1), 0, 1)
    yarn_ratio = (1 - gamma) / s + gamma

    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.axhline(1.0, color=TICK, lw=1, ls=":")
    ax.plot(i, pi_ratio, color=ORANGE, lw=2, label="PI: uniform $\\theta_i/s$")
    ax.plot(i, ntk_ratio, color=TEAL, lw=2, label="NTK-aware: $s^{-2i/(D-2)}$")
    ax.plot(i, yarn_ratio, color=YELLOW, lw=2, label="YaRN by-parts: ramp $\\gamma$")
    i_slow = int(np.argmax(lam >= L))                                       # λ≥L：训练窗内没转完一圈
    ax.axvspan(i_slow - 0.5, D / 2 - 0.5, color="#f4f3ef", zorder=0)
    ax.text((i_slow + D / 2) / 2, 0.44, "$\\lambda_i \\geq L$: never a full\ncycle in training",
            ha="center", fontsize=7.5, color=MID)
    ax.annotate("interp. $\\to$ 1/s", xy=(D / 2 - 1, pi_ratio[-1]), xytext=(-8, 8),
                textcoords="offset points", fontsize=7.5, color=MID, ha="right")
    ax.annotate("extrap. $\\to$ 1 (untouched)", xy=(0, 1.0), xytext=(6, -12),
                textcoords="offset points", fontsize=7.5, color=MID)
    ax.set_xlim(-1, D / 2)
    ax.set_ylim(0.15, 1.12)
    ax.set_xlabel("frequency-pair index $i$  (0 = fastest, $\\lambda_0\\approx$6.3 toks;"
                  "  31 = slowest, $\\lambda_{31}\\approx$47k toks)", fontsize=8.5)
    ax.set_ylabel("rewritten $\\theta'_i/\\theta_i$  (log)", fontsize=9)
    ax.set_yscale("log")
    ax.set_yticks([0.25, 0.5, 1.0])
    ax.set_yticklabels(["1/s", "0.5", "1"])
    ax.set_title("Where each recipe intervenes on the wavelength spectrum (d=64, L=2048, s=4)",
                 fontsize=9.5, color=DARK)
    ax.legend(frameon=False, fontsize=8, loc="center left")
    style_axis(ax)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-7-4-intervention-mechanism.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig 7.4] {out}")


def main():
    ap = argparse.ArgumentParser(description="ch7 五张图（图 7.4 机制示意不依赖实验数据，立即可画）")
    ap.add_argument("--only", type=str, default=None, help="只画部分图，如 --only 4")
    ap.add_argument("--ablation", type=str, default="full", help="实验图用的档位名（默认 full）")
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    todo = set(args.only.split(",")) if args.only else {"1", "2", "3", "4", "5"}
    if "4" in todo:
        fig_7_4()
    if "1" in todo:
        fig_7_1(args.ablation)
    if "2" in todo:
        fig_7_2(args.ablation)
    if "3" in todo:
        fig_7_3(args.ablation)
    if "5" in todo:
        fig_7_5(args.ablation)


if __name__ == "__main__":
    main()
