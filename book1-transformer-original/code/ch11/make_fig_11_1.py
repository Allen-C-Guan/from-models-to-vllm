# -*- coding: utf-8 -*-
# 用途：生成 Book1 第 11 章图 11.1「全书代码资产总览图」——ch01-ch10 各章交付的代码
#       如何组成整机与管线（兼作全书源码地图）；图内文字用英文，中文解释在书稿图注。
#       形状标注：ch07 整机框与 ch09 model 框各标数据河主形状 (B,S,512)/(B,S,256)
#       （B=批大小，S=序列长度；第三维即 d_model，512 为论文 base、256 为第 9 章缩尺寸）。
# 所属章节：《Transformer 原典》第 11 章
# 运行：source env.sh && python "code/ch11/make_fig_11_1.py"
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# 本书印刷图规格：固定分类色序（注意力=蓝 / FFN·配方=橙 / 嵌入·PE·整机=青 / 点火·实验=黄）
BLUE, ORANGE, CYAN, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUE_F, ORNG_F, CYAN_F, YLW_F = "#cde2fb", "#fde8db", "#d8f3e8", "#fdf3d7"
INK, NOTE, HAIR = "#0b0b0b", "#52514e", "#898781"

FIG_DIR = Path(__file__).resolve().parents[3] / "drafts" / "Book1-Transformer原典" / "figures"

fig, ax = plt.subplots(figsize=(9.0, 5.6), dpi=300)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def box(x, y, w, h, text, ec=BLUE, fc=BLUE_F, fs=6.4, lw=1.3, tc=INK, ls="-"):
    """画一个框内多行文字的资产框；x/y/w/h 均为 axes 坐标（0-1），返回框的几何参数。"""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.01",
                                ec=ec, fc=fc, lw=lw, ls=ls, mutation_scale=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.55)
    return (x, y, w, h)


def arrow(p, q, color=HAIR, lw=1.3, rad=0.0, ls="-"):
    """从点 p 到点 q 画一条带箭头的连线（rad 控制弧度）。"""
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def stage(x, label, color):
    """在顶部画旅程段的表头（段名 + 色条）。"""
    ax.text(x, 0.965, label, ha="center", va="center", fontsize=7.6, color=INK)
    ax.plot([x - 0.105, x + 0.105], [0.935, 0.935], color=color, lw=2.0)


# ---- 四段旅程的表头（对应导览 0.4 的五段旅程图，收官段即本章、无代码） ----
stage(0.105, "Prologue\nch01", HAIR)
stage(0.375, "Parts shop  ch02-06", BLUE)
stage(0.645, "Assembly + recipe  ch07-08", ORANGE)
stage(0.885, "Ignition + proof  ch09-10", YELLOW)

# ---- 第一列：前史（ch01） ----
box(0.105, 0.60, 0.165, 0.115,
    "ch01/bottleneck_demo.py\nGRU fixed-vector demo\n(+ make_figures.py)",
    ec=HAIR, fc="white", fs=6.2)

# ---- 第二列：零件铺（ch02-06，按部件家族着色） ----
parts = [
    (0.795, 0.205, 0.105, "ch02/scaled_dot_product.py\nscaled dot-product attn\n$\\sqrt{d_k}$ · $O(n^2)$ ledger", BLUE, BLUE_F),
    (0.660, 0.205, 0.105, "ch03/mha.py\nmulti-head attention\n$h\\times d_k{=}64$ subspaces", BLUE, BLUE_F),
    (0.525, 0.205, 0.105, "ch04/roles_demo.py\nthree roles + masks\nalignment heatmap", BLUE, BLUE_F),
    (0.390, 0.205, 0.105, "ch05/pe.py\nsinusoidal PE\npermutation test", CYAN, CYAN_F),
    (0.255, 0.205, 0.105, "ch06/blocks.py\nFFN + residual\n+ post-LN", ORANGE, ORNG_F),
]
for y, w, h, text, ec, fc in parts:
    box(0.375, y, w, h, text, ec=ec, fc=fc, fs=6.1)

# 零件总线：五份零件汇成一条 import 线，进入 ch07 整机
BUS_X = 0.512
ax.plot([BUS_X, BUS_X], [0.255, 0.795], color=HAIR, lw=1.3)
for y, w, h, *_ in parts:
    arrow((0.375 + w / 2, y), (BUS_X, y), color=HAIR, lw=1.0)

# ---- 第三列：组装（ch07 整机）与配方（ch08） ----
box(0.645, 0.62, 0.195, 0.175,
    "ch07/transformer.py + masks.py\nfull encoder-decoder\n(imports ch02-06, zero rewrite)\nmask grammar · 63.1M ledger\ndata river (B,S,512)",   # 末行 = 数据河主形状，d_model=512（论文 base）
    ec=BLUE, fc=BLUE_F, fs=6.2)
box(0.645, 0.255, 0.195, 0.135,
    "ch08/recipe.py\nNoamLR · label smoothing\nbeam search + length penalty",
    ec=ORANGE, fc=ORNG_F, fs=6.2)
arrow((BUS_X, 0.62), (0.645 - 0.0975, 0.62), color=INK, lw=1.6)

# ---- 第四列：点火与验证（ch09 管线 + ch10 网格） ----
pipe = [
    (0.825, "ch09/data.py\nMulti30K + BPE-8k\nvocab & batching"),
    (0.680, "ch09/model.py\nreuses ch07 machine\n(scaled: N4 / d256)\nriver (B,S,256)"),   # 末行 = 缩尺寸后的数据河主形状，d_model=256
    (0.535, "ch09/train.py\nruns on ch08\ninstruments"),
    (0.390, "ch09/decode.py + eval.py\nbeam · BLEU\nlength buckets"),
]
for y, text in pipe:
    box(0.885, y, 0.175, 0.105, text, ec=YELLOW, fc=YLW_F, fs=6.0)
box(0.885, 0.235, 0.175, 0.105, "ch09/gru_baseline.py\n2014 baselines\n(echo of ch01)",
    ec=HAIR, fc="white", fs=6.0, ls="--")
box(0.885, 0.085, 0.175, 0.098, "ch10/ablation.py\n10-row grid on the\nch09 pipeline",
    ec=YELLOW, fc=YLW_F, fs=6.0)
for y0, y1 in [(0.772, 0.737), (0.627, 0.592), (0.482, 0.447), (0.337, 0.292)]:
    arrow((0.885, y0), (0.885, y1), color=HAIR, lw=1.1)
arrow((0.885, 0.180), (0.885, 0.136), color=HAIR, lw=1.1)

# ---- 跨列箭头 ----
arrow((0.645 + 0.0975, 0.62), (0.885 - 0.0875, 0.680), color=INK, lw=1.5, rad=0.12)     # 整机 → ch09 model
ax.text(0.802, 0.700, "the machine", fontsize=6.0, color=INK, ha="center", style="italic")
# 配方 → ch09 train：走两列之间的空当（肘形路径，避免穿过 ch07 整机框）
ax.plot([0.7425, 0.770], [0.255, 0.255], color=ORANGE, lw=1.4)
ax.plot([0.770, 0.770], [0.255, 0.535], color=ORANGE, lw=1.4)
arrow((0.770, 0.535), (0.885 - 0.0875, 0.535), color=ORANGE, lw=1.4)
ax.text(0.735, 0.163, "instruments on board\n(allclose-checked)",
        fontsize=5.6, color=ORANGE, ha="center", style="italic", linespacing=1.4)
arrow((0.105 + 0.0825, 0.60), (0.885 - 0.0875, 0.235), color=HAIR, lw=1.1, ls="--", rad=-0.22)  # ch01 → GRU baseline
ax.text(0.475, 0.115, "the 2014 machine returns as baseline", fontsize=6.0, color=NOTE, ha="center", style="italic")

fig.tight_layout()
fig.savefig(FIG_DIR / "fig-11-1-code-asset-map.png", facecolor="white")
plt.close(fig)
print(f"图 11.1 已保存：{FIG_DIR / 'fig-11-1-code-asset-map.png'}")
