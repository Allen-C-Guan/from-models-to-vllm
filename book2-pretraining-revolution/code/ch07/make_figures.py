# 用途：Book2 ch7 三张图——图 7.1 四设定示意 / 图 7.2 涌现阈值重排（引 Wei v2 Table 1 数据自绘）/ 图 7.3 指标整流器（教学算例）
# 所属章节：Book2 第 7 章 §7.2、§7.3、§7.5
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch07/make_figures.py"
# 产物：figures/fig-7-{1,2,3}-*.png（300dpi，图内文字英文、图注见正文）

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(REPO, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MUTED = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": TICK, "axes.labelcolor": DARK,
    "xtick.color": TICK, "ytick.color": TICK, "text.color": DARK,
    "axes.linewidth": 0.8, "font.family": "DejaVu Sans",
})


def box(ax, x, y, w, h, label, fc="white", ec=DARK, lw=1.1, fs=8.0, tc=None, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc or DARK)


def arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=lw))


# ---------------- 图 7.1：四设定示意（fine-tuning / few-shot / one-shot / zero-shot） ----------------
def fig_7_1():
    fig, ax = plt.subplots(figsize=(8, 5.4), dpi=300)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 13.6)
    ax.axis("off")
    rows = [
        ("Fine-tuning", 10.6, ORANGE, "weights UPDATED\nby gradient descent",
         "task training set\n(thousands of labelled\nexamples)"),
        ("Few-shot", 7.3, BLUE, "weights FROZEN\nno gradient updates",
         "task description +\nK demonstrations (K ~ 10-100)"),
        ("One-shot", 4.0, BLUE, "weights FROZEN\nno gradient updates",
         "task description +\n1 demonstration"),
        ("Zero-shot", 0.7, BLUE, "weights FROZEN\nno gradient updates",
         "task description only\n(natural language)"),
    ]
    for name, y, c, right, left in rows:
        ax.text(0.15, y + 1.35, name, fontsize=9.5, color=c, fontweight="bold")
        box(ax, 0.15, y, 6.6, 1.15, left, fs=7.6)
        box(ax, 7.35, y, 4.5, 1.15,
            "sea otter -> loutre de mer\ncheese -> fromage\n...\nplush girafe -> girafe peluche" if name == "Few-shot"
            else ("sea otter -> loutre de mer\n[query] cheese -> ?" if name == "One-shot"
                  else ("[labelled pairs]\nEng -> Fr ...\n(loss on weights)" if name == "Fine-tuning"
                        else "Translate English to French:\n[query] cheese -> ?")),
            fc="#f7fafe" if name != "Fine-tuning" else "#fdf3ef", ec=c, lw=1.3, fs=7.0)
        box(ax, 12.5, y, 2.5, 1.15, "GPT-3\npredict", ec=c, lw=1.3, fs=8.0)
        arrow(ax, 6.75, y + 0.58, 7.35, y + 0.58, color=c)
        arrow(ax, 11.85, y + 0.58, 12.5, y + 0.58, color=c)
        arrow(ax, 15.0, y + 0.58, 15.35, y + 0.58, color=c)
        ax.text(15.5, y + 0.75, right, fontsize=7.4, color=MUTED, va="center")
    ax.text(3.45, 13.25, "conditioning (prompt, frozen weights)", ha="center", fontsize=8.2, color=MUTED)
    ax.text(9.6, 13.25, "what the model sees (n <= n_ctx = 2048)", ha="center", fontsize=8.2, color=MUTED)
    ax.text(13.75, 13.25, "forward pass", ha="center", fontsize=8.2, color=MUTED)
    ax.text(0.15, 0.05, "example strings from GPT-3 Figure 2.1 (English-to-French); "
                        "few-shot K bounded by the 2048-token context window",
            fontsize=7.2, color=TICK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-7-1-four-settings.png"), facecolor="white")
    plt.close(fig)


# ---------------- 图 7.2：涌现阈值重排（Wei v2 Table 1 few-shot panel 数据自绘） ----------------
def fig_7_2():
    # (ability, FLOPs threshold, params, family)
    data = [
        ("3-digit add/sub", 2.3e22, "13B", "GPT-3"),
        ("CivilComments toxicity", 1.3e22, "7.1B", "Gopher"),
        ("4-5-digit add/sub", 3.1e23, "175B", "GPT-3"),
        ("MMLU 57-topic avg.", 3.1e23, "175B", "GPT-3"),
        ("grounded conceptual maps", 3.1e23, "175B", "GPT-3"),
        ("TruthfulQA", 5.0e23, "280B", "Gopher"),
        ("MMLU 26-topic subset", 5.0e23, "280B", "Gopher"),
        ("MMLU 30-topic subset", 5.0e23, "70B", "Chinchilla"),
        ("Word in Context", 2.5e24, "540B", "PaLM"),
    ]
    order = sorted(data, key=lambda r: r[1])
    colors = {"GPT-3": BLUE, "Gopher": ORANGE, "Chinchilla": TEAL, "PaLM": YELLOW}
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=300)
    ax.axvspan(1.3e22, 2.5e24, color="#f4f3ee", zorder=0)  # 跨度下限=全表最小阈值 CivilComments 1.3e22
    for i, (name, fl, params, fam) in enumerate(order):
        y = len(order) - 1 - i
        ax.plot([fl], [y], "o", color=colors[fam], ms=8, zorder=3)
        ax.text(fl * 1.25, y, f"  {fam} {params}", fontsize=7.4, va="center", color=MUTED)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([r[0] for r in order], fontsize=7.8)
    ax.set_xscale("log")
    ax.set_xlim(8e21, 4e25)
    ax.set_xlabel("training FLOPs at which the ability first appears (Wei et al. 2022, Table 1)", fontsize=8.6)
    ax.grid(True, axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.text(1.55e22, len(order) - 0.4, "span ≈ 190x\n(1.3e22 – 2.5e24)", fontsize=8.0, color=DARK)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k, ms=7) for k, c in colors.items()]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-7-2-emergence-thresholds.png"), facecolor="white")
    plt.close(fig)


# ---------------- 图 7.3：指标整流器（教学算例，非论文数据） ----------------
def fig_7_3():
    import numpy as np
    xs = np.logspace(8, 12, 3000)                     # model parameters (toy scale axis)
    p = 0.14 + 0.78 / (1 + np.exp(-(np.log10(xs) - 9.2) / 0.85))  # smooth per-token accuracy
    # 四个教学档位：在平滑曲线上取 p 恰为 0.30/0.50/0.65/0.80 的位置（与 handcalc.py [4] 同源）
    targets = [0.30, 0.50, 0.65, 0.80]
    pts_x = np.array([10 ** (np.interp(t, p, np.log10(xs))) for t in targets])
    pts_p = np.array(targets)
    L = 4
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300, sharex=True)
    a = axes[0]
    a.plot(xs, p, color=BLUE, lw=2)
    a.plot(pts_x, pts_p, "o", color=BLUE, ms=7)
    for x0, p0 in zip(pts_x, pts_p):
        a.annotate(f"p={p0:.2f}", (x0, p0), textcoords="offset points", xytext=(4, -12), fontsize=7.6, color=MUTED)
    a.set_xscale("log")
    a.set_xlabel("model scale (params, toy axis)", fontsize=8.6)
    a.set_ylabel("per-token accuracy p (smooth)", fontsize=8.6)
    a.set_ylim(0, 1.0)
    a.grid(True, color=GRID, lw=0.5)
    a.set_axisbelow(True)
    a.set_title("(A) underlying skill: smooth", fontsize=9, color=DARK)
    b = axes[1]
    b.plot(xs, p ** L, color=ORANGE, lw=2, label="nonlinear metric: exact match = p^4")
    b.plot(xs, p, color=TEAL, lw=2, ls="--", label="linear metric (e.g. token edit distance)")
    b.axhline(1e-4, color=MUTED, lw=1.0, ls=":")
    b.text(1.2e8, 2.2e-4, "random chance 10^-4 (4 digits)", fontsize=7.2, color=MUTED)
    b.plot(pts_x, pts_p ** L, "o", color=ORANGE, ms=7)
    for x0, p0 in zip(pts_x, pts_p):
        b.annotate(f"{p0 ** L:.3f}", (x0, p0 ** L), textcoords="offset points", xytext=(4, -12),
                   fontsize=7.6, color=MUTED)
    b.set_xscale("log")
    b.set_ylim(0, 1.0)
    b.set_xlabel("model scale (params, toy axis)", fontsize=8.6)
    b.set_ylabel("score on a 4-digit target", fontsize=8.6)
    b.grid(True, color=GRID, lw=0.5)
    b.set_axisbelow(True)
    b.legend(frameon=False, fontsize=7.6, loc="upper left")
    b.set_title("(B) same models, metric-rectified: step-like", fontsize=9, color=DARK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-7-3-metric-rectifier.png"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_7_1()
    fig_7_2()
    fig_7_3()
    for f in sorted(os.listdir(FIG_DIR)):
        if f.startswith("fig-7-"):
            print("产物:", os.path.join(FIG_DIR, f))
