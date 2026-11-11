# -*- coding: utf-8 -*-
# 用途：Book2 第 6 章插图生成——图 6.1 主干相近时的五项设置对照（位置预算按实际长度日程换算）
# 所属章节：drafts/Book2-预训练革命/ch06-RoBERTa.md（原理章无教学代码，本脚本仅出图）
# 运行方式：source env.sh && python "code/ch06/make_figures.py"
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#263b50"
GRAY, GREEN = "#edf0f2", "#def0e8"
# 在工作区根目录运行时，全部产物保存在工作区log；导入本模块不创建目录。
FIG_DIR = Path("log/book2-ch06/figures")


def comparison_rows():
    """由论文配置换算标量位置预算；返回五行文字及核算账本。"""
    bert_positions = 256 * (900_000 * 128 + 100_000 * 512)
    roberta_positions = 8192 * 500_000 * 512
    uniform_positions = 256 * 1_000_000 * 512
    ratio = roberta_positions / bert_positions
    rows = [
        ("数据", "语料约16GB → 超过160GB；来源与清洗也会改变"),
        ("批量", "每步256 → 8192条序列；学习率与更新次数同时配套"),
        ("累计位置",
         f"BERT约{bert_positions / 1e9:.1f}B → RoBERTa约{roberta_positions / 1e9:.1f}B，约{ratio:.2f}倍"),
        ("掩码与句段", "动态掩码、去掉NSP、改变打包；须分别辨认各项作用"),
        ("词表", "WordPiece约30K → 字节BPE约50K；词表参数也会增加"),
    ]
    return rows, {
        "bert_scheduled_positions": bert_positions,
        "roberta_positions": roberta_positions,
        "scheduled_ratio": ratio,
        "uniform_512_ratio": roberta_positions / uniform_positions,
    }


def main():
    # 中文布局与正文图6.1对应；不调用会重画全册的组装脚本。
    plt.rcParams.update({
        "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
        "font.size": 12, "axes.unicode_minus": False,
        "figure.facecolor": "white",
    })
    rows, budget = comparison_rows()
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=200)
    fig.subplots_adjust(top=.84, bottom=.15, left=.04, right=.97)
    ax.set(xlim=(0, 14), ylim=(0, 8))
    ax.axis("off")
    for i, (label, detail) in enumerate(rows):
        y = 6.6 - i * 1.4
        for x, width, text, color in [
            (.15, 3.0, label, GRAY), (3.8, 9.9, detail, GREEN)
        ]:
            ax.add_patch(FancyBboxPatch(
                (x, y), width, 1, boxstyle="round,pad=.025",
                facecolor=color, edgecolor="#7794a5"))
            ax.text(x + width / 2, y + .5, text,
                    ha="center", va="center", color=INK, fontsize=12)
        ax.add_patch(FancyArrowPatch(
            (3.2, y + .5), (3.75, y + .5),
            arrowstyle="-|>", mutation_scale=13, color=INK))
    fig.suptitle("6.1 RoBERTa：主干相近，配方仍有多个变量",
                 fontsize=19, color=INK, y=.95)
    fig.text(.04, .075,
             "BERT v2 A.2；RoBERTa v1 §§3–5。累计输入位置含填充，不是独立语料或有效token的实测量。",
             fontsize=10, color=INK)
    fig.text(.04, .035,
             "若统一假设BERT也全程使用512长度，倍率为16；此假设与实际长度日程须分开。",
             fontsize=10, color=INK)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig-6-1-recipe-knobs.png"
    fig.savefig(out)
    plt.close(fig)
    print(budget)
    print("saved:", out)


if __name__ == "__main__":
    main()
