# make_figures.py —— Book4 ch1 图 1.1（效率悖论散点：参数 vs 每 token 计算量，双对数）
# 用途：产出 figures/fig-1-1-paradox-scatter.png（300dpi 印刷规格，
#       图内英文、图注中文在正文）；数据源 = Switch Transformers (2101.03961 v3) Table 9 官方数字。
# 所属章节：Book4 第 1 章（1.2 效率悖论的机制拆解）。
# 运行方式：cd 工作区根目录 && source env.sh && python code/ch01/make_figures.py
# 规格：写作规范 §4——蓝 #2a78d6（dense）/ 橙 #eb6834（MoE）；线宽 2；网格 hairline #e1e0d9
#       置底层；刻度 #898781；标注 #0b0b0b / #52514e；禁双 y 轴；tight_layout。
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DIR = os.path.join(REPO, "figures")

BLUE, ORANGE = "#2a78d6", "#eb6834"
GRID, TICK, DARK, MID = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9.5,
    "axes.edgecolor": TICK, "axes.labelcolor": DARK, "xtick.color": TICK,
    "ytick.color": TICK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "font.family": "sans-serif",
})


def fig_1_1():
    # Switch Transformers Table 9（官方论文 2101.03961 v3）：参数 vs FLOPs/seq
    dense = {"T5-Base": (0.2, 124), "T5-Large": (0.7, 425), "T5-XXL": (11, 6300)}
    moe = {"Switch-Base": (7, 124), "Switch-Large": (26, 425),
           "Switch-XXL": (395, 6300), "Switch-C": (1571, 890)}

    fig, ax = plt.subplots(figsize=(6.4, 4))
    # dense 参考线：FLOPs ∝ N（T5 三点在对角线上，取 T5-Base 点作斜率 1 延长）
    (x0, y0), _ = dense["T5-Base"], dense["T5-XXL"]
    xs = [0.12, 2600]
    ax.plot(xs, [y0 * (xs[0] / x0) ** 1.0, y0 * (xs[1] / x0) ** 1.0], color=BLUE,
            lw=1.2, ls="--", alpha=0.55, zorder=1, label="dense guide: FLOPs $\\propto$ N")

    # 同 FLOPs 配对：水平虚线连接 dense↔MoE
    for (dx, dy), mn in zip(dense.values(), ["Switch-Base", "Switch-Large", "Switch-XXL"]):
        mx = moe[mn][0]
        ax.plot([dx, mx], [dy, dy], color=MID, lw=0.9, ls=":", zorder=1)
        ax.text(dx * (mx / dx) ** 0.38, dy * 1.30, f"$\\times${mx / dx:.0f} params,\nsame FLOPs",
                ha="center", va="bottom", fontsize=7.5, color=MID)

    ax.scatter(*zip(*dense.values()), s=46, color=BLUE, zorder=3, label="dense (T5)")
    ax.scatter(*zip(*moe.values()), s=46, color=ORANGE, marker="s", zorder=3,
               label="MoE (Switch, top-1)")
    name_off = {"T5-Base": (8, -13), "T5-Large": (8, -13), "T5-XXL": (8, -13),
                "Switch-Base": (9, 4), "Switch-Large": (0, -15), "Switch-XXL": (-10, 9),
                "Switch-C": (9, 4)}
    for name, (x, y) in {**dense, **moe}.items():
        ax.annotate(name, (x, y), textcoords="offset points", xytext=name_off[name],
                    fontsize=7.5, color=DARK)

    # Switch-C 极点标注
    ax.annotate("143$\\times$ T5-XXL params,\nonly 1/7 its FLOPs",
                xy=moe["Switch-C"], xytext=(-116, -34), textcoords="offset points",
                fontsize=8, color=DARK,
                arrowprops=dict(arrowstyle="-|>", color=MID, lw=1.1))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total parameters (B)", fontsize=9.5)
    ax.set_ylabel("FLOPs per sequence (B)", fontsize=9.5)
    ax.set_xlim(0.12, 3000)
    ax.set_ylim(60, 20000)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-1-1-paradox-scatter.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig 1.1] {out}")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_1_1()
