# make_fig_1_1.py —— 图 1.1 组件-组装双轴时间线（自绘示意级：上轴组件论文节点、下轴 LLaMA 组装节点）
# 用途：ch1 1.5节已有组件与LLaMA整机的关系；横轴时间，连线标所选采纳关系；
#       日期全部官方锚（arXiv v1 日期 / 许可日期），出处见 research papers/06 §六双轴时间线表。
# 所属章节：Book3 第 1 章 1.5 节（图 1.1；纯 matplotlib，CPU 秒级）。
# 运行方式：激活既有环境后按本书路径调用；--out-dir可指定工作目录log中的输出。
# 产物：log/book3-ch01/figures/fig-1-1-component-assembly-timeline.png（300dpi）；不改正文图。
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, INK, SUB = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"

Y_COMP, Y_ASM = 0.85, -0.75            # 上轴（组件论文）/ 下轴（LLaMA 组装）纵坐标
# (名称, 年份小数, 标签层级)：层级 0=贴轴上方 1=更高 2=轴下方（错峰防碰撞）
COMPONENTS = [                          # 上轴：组件论文（蓝点；Chinchilla 为配方，黄方）
    ("GLU\n2016", 2016.98, 0), ("RMSNorm\n2019", 2019.79, 0), ("MQA\n2019", 2019.84, 2),
    ("SwiGLU\n2020", 2020.11, 1), ("RoPE\n2021", 2021.31, 0),
    ("Chinchilla\n(recipe) 2022", 2022.22, 1), ("GQA\n2023", 2023.40, 0),
]
ASSEMBLIES = [                          # 下轴：LLaMA 代际（橙方；日期=arXiv/许可锚）
    ("LLaMA 1\n2023-02", 2023.16, 0), ("LLaMA 2\n34B/70B: GQA\n2023-07", 2023.55, 1),
    ("Llama 3\n2024-04", 2024.30, 0), ("Llama 3.1 405B\n2024-07", 2024.56, 1),
]
ADOPT = [("RMSNorm", 2023.16, TEAL, "-"), ("SwiGLU", 2023.16, TEAL, "-"),
         ("RoPE", 2023.16, TEAL, "-"), ("Chinchilla", 2023.16, YELLOW, "--"),
         ("GQA", 2023.55, TEAL, "-")]   # 采纳连线：三改+配方→LLaMA1；GQA→LLaMA2
LINEAGE = [(2016.98, 2020.11), (2019.84, 2023.40)]   # 前史连线：GLU→SwiGLU、MQA→GQA
POS = {n.split("\n")[0]: x for n, x, _ in COMPONENTS}


def main():
    ap=argparse.ArgumentParser(description="选定组件与LLaMA采纳关系；不是首次发明/采用证明")
    ap.add_argument('--out-dir',type=Path,default=Path.cwd()/'log/book3-ch01/figures')
    args=ap.parse_args();out=args.out_dir/'fig-1-1-component-assembly-timeline.png'
    if out.exists():raise FileExistsError(out)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    for yr in range(2016, 2025):                      # 年度竖网格（hairline 置底层）
        ax.axvline(yr, color=GRID, lw=0.6, zorder=0)
    for name, x, lv in COMPONENTS:                    # 上轴节点与标签
        is_recipe = name.startswith("Chinchilla")
        ax.scatter([x], [Y_COMP], s=80 if is_recipe else 70, marker="s" if is_recipe else "o",
                   color=YELLOW if is_recipe else BLUE, zorder=3)
        dy = {0: 0.30, 1: 0.62, 2: -0.36}[lv]
        ax.text(x, Y_COMP + dy, name, ha="center", va="bottom" if dy > 0 else "top",
                fontsize=8, color=INK if not is_recipe else SUB)
    for name, x, lv in ASSEMBLIES:                    # 下轴节点与标签（两级错峰）
        ax.scatter([x], [Y_ASM], s=80, marker="s", color=ORANGE, zorder=3)
        ax.text(x, Y_ASM - {0: 0.30, 1: 0.58}[lv], name, ha="center", va="top", fontsize=8, color=INK)
    for src, dst, color, style in ADOPT:              # 采纳连线（青实线=组件、黄虚线=配方）
        ax.annotate("", xy=(dst, Y_ASM + 0.10), xytext=(POS[src], Y_COMP - 0.10),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, ls=style,
                                    connectionstyle="arc3,rad=0.10"), zorder=2)
    for a, b in LINEAGE:                              # 前史连线（灰，弧线绕行轴下方）
        ax.annotate("", xy=(b, Y_COMP - 0.10), xytext=(a, Y_COMP - 0.10),
                    arrowprops=dict(arrowstyle="-|>", color=TICK, lw=1.0, alpha=0.85,
                                    connectionstyle="arc3,rad=-0.30"), zorder=1)
    ax.annotate("", xy=(2019.79, 2.08), xytext=(2023.16, 2.08),          # 3.4 年跨度标注
                arrowprops=dict(arrowstyle="<|-|>", color=SUB, lw=1.0))
    ax.text(2021.48, 2.16, "~3.4 years: RMSNorm paper to LLaMA 1", ha="center",
            fontsize=8, color=SUB)
    ax.text(2015.95, Y_COMP, "component\npapers", ha="left", va="center", fontsize=8,
            color=SUB, style="italic")
    ax.text(2015.95, Y_ASM, "LLaMA\nreleases", ha="left", va="center", fontsize=8,
            color=SUB, style="italic")
    handles = [plt.Line2D([], [], marker="o", ls="", color=BLUE, label="component paper"),
               plt.Line2D([], [], marker="s", ls="", color=YELLOW, label="training recipe"),
               plt.Line2D([], [], marker="s", ls="", color=ORANGE, label="LLaMA release"),
               plt.Line2D([], [], color=TEAL, lw=1.8, label="adopted at assembly"),
               plt.Line2D([], [], color=TICK, lw=1.0, label="lineage (earlier idea)")]
    ax.legend(handles=handles, loc="lower left", fontsize=7.5, frameon=False,
              bbox_to_anchor=(0.002, 0.02))
    ax.set_xlim(2015.8, 2025.15)
    ax.set_ylim(-2.10, 2.48)
    ax.set_xticks(range(2016, 2025))
    ax.set_yticks([])
    ax.tick_params(colors=TICK, labelsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title("Selected components and their adoption in LLaMA models", fontsize=11, color=INK)
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)                                  # 规范：savefig 不用 bbox_inches='tight'
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
