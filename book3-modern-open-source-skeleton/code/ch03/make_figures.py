# make_figures.py —— Book3 ch3 三张图的绘制脚本（图 3.1 权路 / 图 3.2 消融曲线 / 图 3.3 激活四景）
# 用途：产出 figures/fig-3-{1,2,3}-*.png（300dpi 印刷规格，图内英文、
#       图注中文在正文）；图 3.2 依赖 log/book3-ch03/ 的消融产物（curve_*.csv + ablation_ffn_*.json），
#       full 档未跑完时可用 --only 1,3 先出不依赖消融数据的两张。
# 所属章节：Book3 ch3（3.5 图 3.1 / 3.6 图 3.2 / 3.7 图 3.3）。
# 运行方式：cd code/ch03 && python make_figures.py [--only 1,3]
# 规格：写作规范 §4——蓝 #2a78d6 / 橙 #eb6834 / 青 #1baf7a / 黄 #eda100；线宽 2；网格 hairline
#       #e1e0d9 置底层；刻度 #898781；标注 #0b0b0b / #52514e；禁双 y 轴；tight_layout。
import argparse
import csv
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
LOG_DIR = os.path.join(REPO, "log", "book3-ch03")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MID = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9.5,
    "axes.edgecolor": TICK, "axes.labelcolor": DARK, "xtick.color": TICK,
    "ytick.color": TICK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "font.family": "sans-serif",
})


def style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------- 图 3.1：门控前馈权路（GPT-2 两矩阵 vs SwiGLU 三矩阵，标形状） ----------------
def fig_3_1():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")
    ax.grid(False)

    def tensor(x, y, w, h, label, sub=""):
        """张量框：浅灰圆角框 + 形状标注。"""
        r = plt.Rectangle((x, y), w, h, fc="#f4f3ef", ec=MID, lw=1.0, zorder=2)
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2 + (2.2 if sub else 0), label, ha="center", va="center",
                fontsize=8.5, color=DARK, zorder=3)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.4, sub, ha="center", va="center",
                    fontsize=7.5, color=MID, zorder=3)

    def mat(x, y, w, h, label, sub, color=BLUE):
        """矩阵/算子框：分类色描边。"""
        r = plt.Rectangle((x, y), w, h, fc="white", ec=color, lw=1.6, zorder=2)
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2 + (2.0 if sub else 0), label, ha="center", va="center",
                fontsize=8.5, color=DARK, zorder=3)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.2, sub, ha="center", va="center",
                    fontsize=7.5, color=MID, zorder=3)

    def arrow(x1, y1, x2, y2, label="", color=MID, dy=1.6, ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, linestyle=ls), zorder=1)
        if label:
            ax.text((x1 + x2) / 2, max(y1, y2) + dy, label, ha="center", va="bottom",
                    fontsize=7.5, color=MID)

    # ---- 上排：GPT-2 MLP（两矩阵，d_ff = 4C）----
    y1 = 36
    ax.text(2, y1 + 13.5, "GPT-2 MLP — 2 matrices, $d_{ff}=4C$  (Book2 ch10)", fontsize=10, color=DARK)
    tensor(1, y1, 9, 8, "x", "(B,n,C)")
    mat(13.5, y1, 10, 8, "c_fc", "C$\\to$4C")
    tensor(26, y1, 11, 8, "h", "(B,n,4C)")
    mat(39.5, y1, 10, 8, "GELU", "(tanh)", color=ORANGE)
    tensor(52, y1, 11, 8, "a", "(B,n,4C)")
    mat(65.5, y1, 10, 8, "c_proj", "4C$\\to$C")
    tensor(78, y1, 9, 8, "y", "(B,n,C)")
    arrow(10, y1 + 4, 13.5, y1 + 4)
    arrow(23.5, y1 + 4, 26, y1 + 4)
    arrow(37, y1 + 4, 39.5, y1 + 4)
    arrow(49.5, y1 + 4, 52, y1 + 4)
    arrow(63, y1 + 4, 65.5, y1 + 4)
    arrow(75.5, y1 + 4, 78, y1 + 4)
    ax.text(89, y1 + 4, "params $2C{\\cdot}4C=8C^2$", fontsize=8, color=MID, va="center")

    # ---- 下排：SwiGLU MLP（三矩阵，d_ff = 8C/3）----
    y2 = 6
    yg, yu = y2 + 14, y2      # gate 支路在上、up 支路在下
    ax.text(2, y2 + 26.5, "SwiGLU MLP — 3 matrices, $d_{ff}=\\frac{8}{3}C$  (this chapter)",
            fontsize=10, color=DARK)
    tensor(1, y2 + 5, 9, 8, "x", "(B,n,C)")
    mat(13.5, yg, 11, 8, "gate_proj", "C$\\to\\frac{8}{3}C$")
    mat(13.5, yu, 11, 8, "up_proj", "C$\\to\\frac{8}{3}C$")
    mat(27, yg, 8, 8, "SiLU", "", color=ORANGE)
    circ = plt.Circle((35.5, y2 + 9), 2.6, fc="white", ec=TEAL, lw=1.8, zorder=2)
    ax.add_patch(circ)
    ax.text(35.5, y2 + 9, "$\\odot$", ha="center", va="center", fontsize=11, color=DARK, zorder=3)
    tensor(41, y2 + 5, 12, 8, "gated", "(B,n,$\\frac{8}{3}$C)")
    mat(56.5, y2 + 5, 11, 8, "down_proj", "$\\frac{8}{3}C\\to$C")
    tensor(71, y2 + 5, 9, 8, "y", "(B,n,C)")
    arrow(10, y2 + 11, 13.5, yg + 4)            # x -> gate 支路
    arrow(10, y2 + 7, 13.5, yu + 4)             # x -> up 支路
    arrow(24.5, yg + 4, 27, yg + 4)
    arrow(35, yg + 4, 35.0, y2 + 10.5)
    ax.annotate("", xy=(33.4, y2 + 10.0), xytext=(24.5, yu + 4),
                arrowprops=dict(arrowstyle="-|>", color=MID, lw=1.3))  # up 支路汇入 ⊙
    ax.text(32.5, y2 + 22.5, "gate path: (B,n,8C/3)", fontsize=7.5, color=MID, ha="center")
    ax.text(28.0, yu - 1.9, "value path: (B,n,8C/3)", fontsize=7.5, color=MID, ha="center")
    arrow(37.6, y2 + 9.6, 41, y2 + 9)
    arrow(53, y2 + 9, 56.5, y2 + 9)
    arrow(67.5, y2 + 9, 71, y2 + 9)
    ax.text(82, y2 + 9, "params $3C{\\cdot}\\frac{8}{3}C=8C^2$\n(same budget)", fontsize=8,
            color=MID, va="center")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-3-1-gated-ffn-path.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig 3.1] {out}")


# ---------------- 图 3.3：激活函数四景（ReLU / GELU / SiLU / 门控乘积） ----------------
def fig_3_3():
    x = np.linspace(-4, 4, 801)

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    fig, axs = plt.subplots(2, 2, figsize=(8, 4.6), sharex=True, sharey=True)
    panels = [
        ("(a) ReLU: hard binary gate", np.maximum(0, x), None, BLUE),
        ("(b) GELU: x·Φ(x)", x * 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2))), None, BLUE),
        ("(c) SiLU = Swish (β=1): x·σ(x)", x * sigmoid(x), None, BLUE),
        ("(d) Gating: value × gate (scalar view)", x * sigmoid(x), sigmoid(x), BLUE),
    ]
    for ax, (title, main, gate, color) in zip(axs.flat, panels):
        ax.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        if gate is not None:
            ax.plot(x, x, color="#b9b7b0", lw=1.2, label="value x", zorder=1)
            ax.plot(x, gate, color=ORANGE, lw=2, label="gate silu(x)", zorder=3)
            ax.plot(x, main, color=color, lw=2, label="x * silu(x)", zorder=2)
            ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        else:
            ax.plot(x, main, color=color, lw=2)
        ax.axhline(0, color=TICK, lw=0.8)
        ax.axvline(0, color=TICK, lw=0.8)
        ax.set_title(title, fontsize=9, color=DARK)
        ax.set_ylim(-2.6, 4.2)
        style_axis(ax)
    for ax in axs[1]:
        ax.set_xlabel("x", fontsize=9)
    for ax in axs[:, 0]:
        ax.set_ylabel("f(x)", fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-3-3-activation-four-panels.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig 3.3] {out}")


# ---------------- 图 3.2：GELU-4C vs SwiGLU-8/3C 消融曲线（参数对齐档，自产） ----------------
def load_curve(arm, out_name):
    path = os.path.join(LOG_DIR, f"curve_{arm}_{out_name}.csv")
    if not os.path.exists(path):
        return None, None
    steps, losses = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            losses.append(float(row["train_loss"]))
    return np.array(steps), np.array(losses)


def load_evals(out_name):
    path = os.path.join(LOG_DIR, f"ablation_ffn_{out_name}.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {arm: rec["evals"] for arm, rec in json.load(f)["arms"].items()}


def fig_3_2(out_name="full"):
    t_g = load_curve("gelu4c", out_name)
    t_s = load_curve("swiglu83", out_name)
    e = load_evals(out_name)
    if t_g is None or t_s is None:
        raise SystemExit(f"[fig 3.2] 缺 full 档曲线（log/book3-ch03/curve_*_{out_name}.csv）——先跑 ablation_ffn.py --steps 5000")

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    ax = axs[0]
    ax.plot(t_g[0], t_g[1], color=BLUE, lw=2, label="GELU-4C (d_ff=3072)")
    ax.plot(t_s[0], t_s[1], color=ORANGE, lw=2, label="SwiGLU-8/3C (d_ff=2048)")
    ax.set_xlabel("step", fontsize=9)
    ax.set_ylabel("train loss (bf16)", fontsize=9)
    ax.set_title("(a) Train loss", fontsize=9.5, color=DARK)
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)

    ax = axs[1]
    for arm, color in (("gelu4c", BLUE), ("swiglu83", ORANGE)):
        ev = e.get(arm, [])
        if not ev:
            continue
        s = np.array([d["step"] for d in ev])
        v = np.array([d["eval_loss"] for d in ev])
        se = np.array([d["eval_se"] for d in ev])
        ax.errorbar(s, v, yerr=se, color=color, lw=2, marker="o", ms=5, capsize=2,
                    label=f"{arm} (d_ff={'3072' if arm == 'gelu4c' else '2048'})")
    ax.set_xlabel("step", fontsize=9)
    ax.set_ylabel("eval loss (fp32, fixed 401k tokens)", fontsize=9)
    ax.set_title("(b) Fixed-window eval loss", fontsize=9.5, color=DARK)
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-3-2-ablation-curves.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig 3.2] {out}（{out_name} 档）")


def main():
    ap = argparse.ArgumentParser(description="ch3 三张图（图 3.1/3.3 立即可画；图 3.2 需消融产物）")
    ap.add_argument("--only", type=str, default=None, help="只画部分图，如 --only 1,3")
    ap.add_argument("--ablation", type=str, default="full", help="图 3.2 用的消融档位名（默认 full）")
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    todo = set(args.only.split(",")) if args.only else {"1", "2", "3"}
    if "1" in todo:
        fig_3_1()
    if "2" in todo:
        fig_3_2(args.ablation)
    if "3" in todo:
        fig_3_3()


if __name__ == "__main__":
    main()
