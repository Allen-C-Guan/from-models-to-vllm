# -*- coding: utf-8 -*-
# 用途：生成 Book2 第 12 章两张收束图——图 12.1 八册地图（Book2 高亮，钩子流向）与
#       图 12.2 本册代码资产总览（ch01-ch11 各章代码件与单轨依赖链）；图内文字英文，中文在书稿图注。
# 所属章节：《预训练革命》第 12 章（收束章，无新实验代码；本脚本仅作图）
# 运行方式：source env.sh && python "code/ch12/make_figures.py"
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# 本书印刷图规格：固定分类色序 蓝/橙/青/黄；线宽、灰阶、白底与 _写作规范.md §4 一致
BLUE, ORANGE, CYAN, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUE_F, ORNG_F, CYAN_F, YLW_F = "#cde2fb", "#fde8db", "#d8f3e8", "#fdf3d7"
INK, NOTE, HAIR = "#0b0b0b", "#52514e", "#898781"

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"


def box(ax, x, y, w, h, text, ec=BLUE, fc=BLUE_F, fs=6.4, lw=1.3, tc=INK, ls="-"):
    """画一个框内多行文字的资产框；x/y/w/h 均为 axes 坐标（0-1）。"""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.01",
                                ec=ec, fc=fc, lw=lw, ls=ls, mutation_scale=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.55)


def arrow(ax, p, q, color=HAIR, lw=1.3, rad=0.0, ls="-"):
    """从点 p 到点 q 画一条带箭头的连线（rad 控制弧度）。"""
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


# ============================ 图 12.1 八册地图 ============================
fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=300)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

books = [
    ("Book 1\nTransformer\nOriginal", "the 2017 machine"),
    ("Book 2\nPretraining\nRevolution", "data · objective · laws"),
    ("Book 3\nModern Open\nSkeletons", "slot-by-slot upgrades"),
    ("Book 4\nScale &\nEfficiency", "the price of scale"),
    ("Book 5\nFrontier\nHybrids", "attention at its limits"),
    ("Book 6\nInference\nSystems", "serving physics"),
    ("Book 7\nvLLM V1\nSource I", "one request's life"),
    ("Book 8\nvLLM V1\nSource II", "topics & production"),
]
xs = [0.085 + 0.118 * i for i in range(8)]
BW, BH, BY = 0.104, 0.22, 0.74
BOOK_BOT = BY - BH / 2          # 0.629：书框底边
for i, ((title, tag), x) in enumerate(zip(books, xs)):
    hot = (i == 1)
    box(ax, x, BY, BW, BH, title,
        ec=ORANGE if hot else HAIR, fc=ORNG_F if hot else "white",
        fs=6.3, lw=2.0 if hot else 1.1)
    ax.text(x, BOOK_BOT - 0.030, tag, ha="center", va="center",
            fontsize=5.4, color=NOTE, style="italic")
# 阅读路径箭头
for i in range(7):
    arrow(ax, (xs[i] + BW / 2, BY), (xs[i + 1] - BW / 2, BY), color=HAIR, lw=1.1)
# 高亮注记
ax.annotate("you are here", xy=(xs[1], BY + BH / 2), xytext=(xs[1] - 0.118, BY + BH / 2 + 0.085),
            fontsize=6.2, color=ORANGE, ha="center",
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.3))

# F 钩子回收（Book1 → Book2，蓝色虚线，浅弧走标签带之下）
arrow(ax, (xs[0], BOOK_BOT), (xs[1], BOOK_BOT), color=BLUE, lw=1.5, ls="--", rad=0.28)
ax.text((xs[0] + xs[1]) / 2 - 0.006, 0.455, "F3 F5 F6 F7\nF9 F10 F15\nsettled here",
        ha="center", va="center", fontsize=5.6, color=BLUE, linespacing=1.4)

# N 钩子埋设（Book2 → 后册）：一条橙色总线 + 四根上行支线（地铁图式，避免弧线交叉）
BUS_Y = 0.28
ax.plot([xs[1], xs[5]], [BUS_Y, BUS_Y], color=ORANGE, lw=1.6)
ax.plot([xs[1], xs[1]], [BOOK_BOT, BUS_Y], color=ORANGE, lw=1.6)
ax.text(xs[1] + 0.012, 0.415, "planted\nhere", fontsize=5.6, color=ORANGE,
        ha="left", va="center", linespacing=1.4)
for dst, label in [(2, "N1–N4, N7"), (3, "N5"), (4, "N6, N7"), (5, "N6, N7")]:
    x = xs[dst]
    arrow(ax, (x, BUS_Y), (x, BOOK_BOT - 0.062), color=ORANGE, lw=1.4)
    ax.text(x, BUS_Y - 0.036, label, ha="center", va="center", fontsize=5.8, color=ORANGE)

ax.text(0.5, 0.085, "F = Book1 hooks settled in Book2;  N = new hooks planted by Book2 (Table 12.2)",
        ha="center", va="center", fontsize=6.0, color=NOTE, style="italic")

fig.tight_layout()
fig.savefig(FIG_DIR / "fig-12-1-eight-book-map.png", facecolor="white")
plt.close(fig)
print(f"图 12.1 已保存：{FIG_DIR / 'fig-12-1-eight-book-map.png'}")

# ======================= 图 12.2 本册代码资产总览 =======================
fig, ax = plt.subplots(figsize=(9.6, 5.8), dpi=300)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# 四段旅程表头（对应导览 0.4 的旅程地图；第 12 章自身无代码）
def stage(x, label, color):
    ax.text(x, 0.965, label, ha="center", va="center", fontsize=7.4, color=INK)
    ax.plot([x - 0.105, x + 0.105], [0.935, 0.935], color=color, lw=2.0)

stage(0.095, "Why  ch01", HAIR)
stage(0.335, "Three paradigms  ch02-06", BLUE)
stage(0.595, "Scaling  ch07-09", CYAN)
stage(0.860, "Build & prove  ch10-11", YELLOW)

# ---- 第一列：为什么（ch01，仅图脚本） ----
box(ax, 0.095, 0.62, 0.150, 0.100,
    "ch01/make_figures.py\nfive-link chain · bi-LM\n(paper diagrams only)",
    ec=HAIR, fc="white", fs=5.9, ls="--")

# ---- 第二列：范式三物种（ch02-06） ----
box(ax, 0.335, 0.800, 0.190, 0.095,
    "ch02/bpe.py  (317)\nmini-BPE + family probe\ntiktoken ≡ HF parity", ec=BLUE, fc=BLUE_F, fs=5.9)
box(ax, 0.335, 0.665, 0.190, 0.095,
    "ch03/glue_finetune.py  (164)\nBERT tiny/base · SST-2 · MRPC\nmajority baseline + variance", ec=BLUE, fc=BLUE_F, fs=5.9)
box(ax, 0.335, 0.530, 0.190, 0.095,
    "ch04/surgery.py  (240)\nfour ops on Book1 block\nshape-flow ledger", ec=ORANGE, fc=ORNG_F, fs=5.9)
box(ax, 0.335, 0.360, 0.190, 0.120,
    "ch04/warmup_ablation.py  (347)\nparam'd GPT factory\nwarmup-off experiment\nsingle-track lineage starts",
    ec=ORANGE, fc=ORNG_F, fs=5.9, lw=2.0)
box(ax, 0.335, 0.200, 0.190, 0.082,
    "ch05·ch06/make_figures.py\n(figure scripts only)",
    ec=HAIR, fc="white", fs=5.9, ls="--")

# ---- 第三列：规模化（ch07-09） ----
box(ax, 0.595, 0.815, 0.200, 0.088,
    "ch07/handcalc.py  (80)\n6ND · 175B hand ledger", ec=CYAN, fc=CYAN_F, fs=5.9)
box(ax, 0.595, 0.680, 0.200, 0.095,
    "ch08/data_prep.py  (195)\nOWT subsample → sp-8k stream\ntokenizer/corpus split", ec=CYAN, fc=CYAN_F, fs=5.9)
box(ax, 0.595, 0.545, 0.200, 0.095,
    "ch08/run_family.py  (385)\nKaplan family\n(imports ch04 GPT)", ec=CYAN, fc=CYAN_F, fs=5.9)
box(ax, 0.595, 0.415, 0.200, 0.088,
    "ch08/fit_scaling.py  (180)\nlog-log fit · CI · E-term", ec=CYAN, fc=CYAN_F, fs=5.9)
box(ax, 0.595, 0.265, 0.200, 0.098,
    "ch09/memory_probe.py  (135)\n16B/param + activation split\n(imports ch04 GPT @ 124M)", ec=CYAN, fc=CYAN_F, fs=5.9)

# ---- 第四列：大项目（ch10-11） ----
box(ax, 0.860, 0.815, 0.225, 0.100,
    "ch10/model.py  (197)\nthree-slot GPT-2 · assert 124,439,808\n(Book3 slot contract)",
    ec=YELLOW, fc=YLW_F, fs=5.9, lw=2.0)
box(ax, 0.860, 0.675, 0.225, 0.092,
    "ch10/train.py (298) + shakespeare.py (30)\ndefinitive loop · warmup run", ec=YELLOW, fc=YLW_F, fs=5.9)
box(ax, 0.860, 0.545, 0.225, 0.092,
    "ch11/data.py  (275)\nOWT pipeline · tok↔word box\nWT2 counting baselines", ec=YELLOW, fc=YLW_F, fs=5.9)
box(ax, 0.860, 0.415, 0.225, 0.092,
    "ch11/official_gpt2.py  (271)\nofficial-weights parity ×4\n(loads ch10 model)", ec=YELLOW, fc=YLW_F, fs=5.9)
box(ax, 0.860, 0.280, 0.225, 0.098,
    "ch11/sample.py  (359)\ngreedy / T / top-k · ICL probe\n(both model lines)", ec=YELLOW, fc=YLW_F, fs=5.9)
box(ax, 0.860, 0.135, 0.225, 0.115,
    "ch11/iso_flop.py (554) + train_124m.py (470)\n15-point iso-FLOP grid + 124M fast\n(ch04 GPT, on ch08 token stream)\nch11/scaling_curve.py (271) → paper coords",
    ec=YELLOW, fc=YLW_F, fs=5.6)

# ---- 依赖箭头：ch04 工厂线（橙，单轨主线） ----
arrow(ax, (0.430, 0.545), (0.495, 0.545), color=ORANGE, lw=1.6)                 # 工厂 → run_family
arrow(ax, (0.430, 0.360), (0.495, 0.290), color=ORANGE, lw=1.3, rad=-0.15)      # 工厂 → memory_probe
# 工厂 → ch11 iso/124m：沿列间空当的肘形长线，自左下进入 iso 框
ax.plot([0.470, 0.470], [0.360, 0.045], color=ORANGE, lw=1.6)
ax.plot([0.470, 0.990], [0.045, 0.045], color=ORANGE, lw=1.6)
arrow(ax, (0.800, 0.045), (0.800, 0.0755), color=ORANGE, lw=1.6)
ax.plot([0.990, 0.990], [0.045, 0.280], color=ORANGE, lw=1.2)   # 第五户：sample.py 自训线同用 ch04 工厂
arrow(ax, (0.990, 0.280), (0.975, 0.280), color=ORANGE, lw=1.2)
ax.text(0.510, 0.068, "one factory, five consumers (ch08 · ch09 · ch11×3)",
        fontsize=5.6, color=ORANGE, style="italic")

# ---- 依赖箭头：ch10 定版线（黄） ----
arrow(ax, (0.860, 0.765), (0.860, 0.723), color=HAIR, lw=1.1)                   # model.py → train.py（真实 import）
ax.plot([0.7475, 0.732], [0.815, 0.815], color=HAIR, lw=1.1)                    # model.py → official_gpt2（左缘肘线）
ax.plot([0.732, 0.732], [0.815, 0.415], color=HAIR, lw=1.1)
arrow(ax, (0.732, 0.415), (0.7475, 0.415), color=HAIR, lw=1.1)

# ---- 数据流箭头（青） ----
arrow(ax, (0.595, 0.632), (0.595, 0.594), color=CYAN, lw=1.2)                   # data_prep → run_family

ax.text(0.5, 0.012, "18 content files · 4,768 lines  (+ 8 feasibility probes · 1,371 lines as reference base; figure scripts not counted)",
        ha="center", va="center", fontsize=6.0, color=NOTE, style="italic")

fig.tight_layout()
fig.savefig(FIG_DIR / "fig-12-2-code-assets.png", facecolor="white")
plt.close(fig)
print(f"图 12.2 已保存：{FIG_DIR / 'fig-12-2-code-assets.png'}")
