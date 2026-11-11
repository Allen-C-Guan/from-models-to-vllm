# 用途：Book2 ch7 三张图——图 7.1 四设定示意 / 图 7.2 涌现阈值重排（引 Wei v2 Table 1 数据自绘）/ 图 7.3 指标变换与统计分辨率（教学算例）
# 所属章节：Book2 第 7 章 §7.2、§7.3、§7.5
# 运行方式：激活配套Python环境后，在本册目录运行 python "code/ch07/make_figures.py"
# 产物：log/book2-ch07/figures/fig-7-{1,2,3}-*.png；不覆盖正文图片

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG_DIR = Path("log/book2-ch07/figures")

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
         "task description +\nK demonstrations (2 shown)"),
        ("One-shot", 4.0, BLUE, "weights FROZEN\nno gradient updates",
         "task description +\n1 demonstration"),
        ("Zero-shot", 0.7, BLUE, "weights FROZEN\nno gradient updates",
         "task description only\n(natural language)"),
    ]
    for name, y, c, right, left in rows:
        ax.text(0.15, y + 1.35, name, fontsize=9.5, color=c, fontweight="bold")
        box(ax, 0.15, y, 6.6, 1.15, left, fs=7.6)
        box(ax, 7.35, y, 4.5, 1.15,
            "sea otter -> loutre de mer\ncheese -> fromage\n[query] apple -> ?" if name == "Few-shot"
            else ("sea otter -> loutre de mer\n[query] cheese -> ?" if name == "One-shot"
                  else ("[labelled pairs]\nEng -> Fr ...\n(training inputs and labels)" if name == "Fine-tuning"
                        else "Translate English to French:\n[query] cheese -> ?")),
            fc="#f7fafe" if name != "Fine-tuning" else "#fdf3ef", ec=c, lw=1.3, fs=7.0)
        box(ax, 12.5, y, 2.5, 1.15, "GPT-3\nforward + loss" if name == "Fine-tuning" else "GPT-3\npredict", ec=c, lw=1.3, fs=8.0)
        arrow(ax, 6.75, y + 0.58, 7.35, y + 0.58, color=c)
        arrow(ax, 11.85, y + 0.58, 12.5, y + 0.58, color=c)
        arrow(ax, 15.0, y + 0.58, 15.35, y + 0.58, color=c)
        ax.text(15.5, y + 0.75, right, fontsize=7.4, color=MUTED, va="center")
    ax.text(3.45, 13.25, "task examples / prompt", ha="center", fontsize=8.2, color=MUTED)
    ax.text(9.6, 13.25, "input sequence", ha="center", fontsize=8.2, color=MUTED)
    ax.text(13.75, 13.25, "model computation", ha="center", fontsize=8.2, color=MUTED)
    ax.text(0.15, 0.05, "translation example adapted from GPT-3 Figure 2.1; "
                        "K shares the 2048-token budget with the query and generated answer",
            fontsize=7.2, color=TICK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-7-1-four-settings.png"), facecolor="white")
    plt.close(fig)


# ---------------- 图 7.2：涌现阈值重排（Wei v2 Table 1 few-shot panel 数据自绘） ----------------
def fig_7_2():
    # (ability, reported training FLOPs, params, family); numeric rows of few-shot panel only
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
    ax.axvspan(1.3e22, 2.5e24, color="#f4f3ee", zorder=0)  # 下限仅指few-shot九项数值记录：CivilComments 1.3e22
    for i, (name, fl, params, fam) in enumerate(order):
        y = i
        ax.plot([fl], [y], "o", color=colors[fam], ms=8, zorder=3)
        ax.text(fl * 1.25, y, f"  {fam} {params}", fontsize=7.4, va="center", color=MUTED)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([r[0] for r in order], fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(8e21, 4e25)
    ax.set_xlabel("reported training FLOPs (Wei et al. 2022, Table 1, few-shot numeric rows)", fontsize=8.6)
    ax.grid(True, axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Reported scale spans about 192x; these are not universal thresholds",
                 fontsize=9, color=DARK, pad=12)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k, ms=7) for k, c in colors.items()]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-7-2-emergence-thresholds.png"), facecolor="white")
    fig.savefig(FIG_DIR / "fig-7-2-emergence-thresholds.svg", facecolor="white")
    plt.close(fig)


# ---------------- 图7.3：非线性期望与有限测试量（教学构造） ----------------
def fig_7_3():
    import numpy as np
    p = np.linspace(0., 1., 400)
    m = np.arange(0, 501)
    zero_success = (1 - .3**4)**m
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=220)
    axes[0].plot(p, p, color=BLUE, lw=2, label="mean position accuracy p")
    axes[0].plot(p, p**4, color=ORANGE, lw=2, label="four independent positions: p^4")
    axes[0].set(xlabel="per-position success p (toy)", ylabel="expected score",
                title="Both curves are continuous", xlim=(0,1), ylim=(0,1.05))
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(m, zero_success, color=BLUE, lw=2)
    axes[1].set(xlabel="number of independent test questions m",
                ylabel="probability of observing no successes",
                title="Fixed p=0.3; more tests reveal rare success",
                xlim=(0,500), ylim=(0,1.05))
    for ax in axes:
        ax.grid(True, color=GRID, lw=.5)
        ax.set_axisbelow(True)
    fig.suptitle("7.3 Nonlinear scoring and finite test sets", fontsize=12)
    fig.tight_layout(rect=(0,.06,1,.92))
    fig.text(.08,.025,"Toy construction: fixed length 4; independent positions and questions. Not measured model outputs.",
             fontsize=8, color=MUTED)
    fig.savefig(FIG_DIR / "fig-7-3-metric-rectifier.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_7_1()
    fig_7_2()
    fig_7_3()
    for f in sorted(os.listdir(FIG_DIR)):
        if f.startswith("fig-7-"):
            print("产物:", os.path.join(FIG_DIR, f))
