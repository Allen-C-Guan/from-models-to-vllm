# make_figures.py —— 生成 Book3 第 11 章两张收束图：图 11.1 八册地图（Book3 高亮，F/N/B3 钩子流向）
#                   与图 11.2 本册代码资产总览（GPT-2 底座→llama_slots 对拍锚→四插槽→207M 整机→设计文档）。
#                   图内文字英文、图注中文在正文（写作规范 §4）；收束章无新实验代码，本脚本仅作图。
# 所属章节：Book3 ch11（11.3 图 11.1 / 11.5 图 11.2；行数与件数按 2026-10 wc -l 实测登记，见正文 11.5 节）。
# 运行方式：cd 工作区根目录 && source env.sh && \
#           python code/ch11/make_figures.py [--only 1,2]
# 规格：蓝 #2a78d6 / 橙 #eb6834 / 青 #1baf7a / 黄 #eda100；线宽 2 级；白底；tight_layout；图式沿用 Book2 ch12。
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUE_F, ORNG_F, CYAN_F, YLW_F = "#cde2fb", "#fde8db", "#d8f3e8", "#fdf3d7"
INK, NOTE, HAIR = "#0b0b0b", "#52514e", "#898781"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DIR = os.path.join(REPO, "figures")


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


# ---------------- 图 11.1：八册地图（Book3 高亮） ----------------
def fig_11_1():
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
    BOOK_BOT = BY - BH / 2                       # 0.629：书框底边
    for i, ((title, tag), x) in enumerate(zip(books, xs)):
        hot = (i == 2)                            # Book3 高亮
        box(ax, x, BY, BW, BH, title,
            ec=ORANGE if hot else HAIR, fc=ORNG_F if hot else "white",
            fs=6.3, lw=2.0 if hot else 1.1)
        ax.text(x, BOOK_BOT - 0.030, tag, ha="center", va="center",
                fontsize=5.4, color=NOTE, style="italic")
    for i in range(7):                            # 阅读路径箭头
        arrow(ax, (xs[i] + BW / 2, BY), (xs[i + 1] - BW / 2, BY), color=HAIR, lw=1.1)
    ax.annotate("you are here", xy=(xs[2], BY + BH / 2), xytext=(xs[2] - 0.118, BY + BH / 2 + 0.085),
                fontsize=6.2, color=ORANGE, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.3))

    # F 钩子回收（Book1 → Book3，蓝色虚线大弧，从 Book2 底下走过）
    arrow(ax, (xs[0], BOOK_BOT), (xs[2] - BW / 2 - 0.004, BOOK_BOT - 0.062),
          color=BLUE, lw=1.5, ls="--", rad=0.30)
    ax.text((xs[0] + xs[1]) / 2 + 0.008, 0.395, "F4 F6 F14 F15 F16\nsettled here",
            ha="center", va="center", fontsize=5.6, color=BLUE, linespacing=1.4)
    # N 钩子回收（Book2 → Book3，蓝色虚线小弧）
    arrow(ax, (xs[1], BOOK_BOT), (xs[2] - 0.012, BOOK_BOT - 0.020),
          color=BLUE, lw=1.5, ls="--", rad=0.14)
    ax.text((xs[1] + xs[2]) / 2 + 0.010, 0.560, "N1 N2 N3 N4 N7\nsettled here",
            ha="center", va="center", fontsize=5.6, color=BLUE, linespacing=1.4)

    # B3 新钩埋设（Book3 → 后册）：橙色总线 + 三根上行支线（地铁图式）
    BUS_Y = 0.235
    ax.plot([xs[2], xs[5]], [BUS_Y, BUS_Y], color=ORANGE, lw=1.6)
    ax.plot([xs[2], xs[2]], [BOOK_BOT, BUS_Y], color=ORANGE, lw=1.6)
    ax.text(xs[2] + 0.012, 0.335, "planted\nhere", fontsize=5.6, color=ORANGE,
            ha="left", va="center", linespacing=1.4)
    for dst, label in [(3, "B3-2 5 8 9"), (4, "B3-3 4 7"), (5, "B3-1 6")]:
        x = xs[dst]
        arrow(ax, (x, BUS_Y), (x, BOOK_BOT - 0.062), color=ORANGE, lw=1.4)
        ax.text(x, BUS_Y - 0.034, label, ha="center", va="center", fontsize=5.8, color=ORANGE)

    ax.text(0.5, 0.085,
            "F = Book1 hooks · N = Book2 hooks (both settled in Book3);  "
            "B3 = new hooks planted by Book3 (Table 11.2)",
            ha="center", va="center", fontsize=6.0, color=NOTE, style="italic")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-11-1-eight-book-map.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}")


# ---------------- 图 11.2：本册代码资产总览 ----------------
def fig_11_2():
    fig, ax = plt.subplots(figsize=(9.6, 5.8), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 四段旅程表头（与导览 0.4 同口径；ch11 自身仅图脚本）
    def stage(x, label, color):
        ax.text(x, 0.965, label, ha="center", va="center", fontsize=7.4, color=INK)
        ax.plot([x - 0.105, x + 0.105], [0.935, 0.935], color=color, lw=2.0)

    stage(0.095, "Why  ch01", HAIR)
    stage(0.335, "Slot swaps  ch02-06", BLUE)
    stage(0.595, "Eng & genealogy  ch07-09", TEAL)
    stage(0.860, "Integrate & wrap  ch10-11", YELLOW)

    # ---- 第一列：为什么（ch01 账本；底座虚影=单轨主线的入口） ----
    box(ax, 0.095, 0.800, 0.150, 0.100,
        "ch01/handcalc.py  (63)\n6ND · GPU-hours · PF-days\nthree-account triangle",
        ec=HAIR, fc="white", fs=5.9)
    box(ax, 0.095, 0.420, 0.150, 0.115,
        "Book2 ch10/model.py  (198)\nthree-slot GPT-2 base\n(single-track lineage\nenters here)",
        ec=HAIR, fc="white", fs=5.9, ls="--")

    # ---- 第二列：逐插槽改装（ch02-06；橙框=被 ch10 import 的四插槽正件，蓝框=出入口插曲） ----
    box(ax, 0.335, 0.815, 0.190, 0.085,
        "ch02/rmsnorm.py  (136)\nablation_norm.py  (338)", ec=ORANGE, fc=ORNG_F, fs=5.9, lw=1.8)
    box(ax, 0.335, 0.700, 0.190, 0.085,
        "ch03/swiglu.py  (135)\nablation_ffn.py  (375)", ec=ORANGE, fc=ORNG_F, fs=5.9, lw=1.8)
    box(ax, 0.335, 0.585, 0.190, 0.085,
        "ch04/rope.py  (215)\nablation_pe.py  (470)", ec=ORANGE, fc=ORNG_F, fs=5.9, lw=1.8)
    box(ax, 0.335, 0.455, 0.190, 0.085,
        "ch05/vocab_scan.py  (501)\nconfig_infer.py  (427)", ec=BLUE, fc=BLUE_F, fs=5.9)
    box(ax, 0.335, 0.325, 0.190, 0.100,
        "ch06/gqa.py (241) · kv_probe.py (183)\nablation_gqa.py  (414)", ec=ORANGE, fc=ORNG_F,
        fs=5.9, lw=1.8)
    # 对拍锚：00-feasibility 组件库正身（青虚线，各章教学件与其逐位一致）
    box(ax, 0.215, 0.155, 0.130, 0.105,
        "00-feas/llama_slots.py  (282)\ncomponent reference\n(parity anchor)", ec=TEAL, fc="white",
        fs=5.7, ls="--")

    # ---- 第三列：工程与谱系（ch07-09） ----
    box(ax, 0.595, 0.815, 0.200, 0.095,
        "ch07/extrapolate.py  (349)\npasskey.py  (250)\nfour interventions · no retrain", ec=TEAL,
        fc=CYAN_F, fs=5.9)
    box(ax, 0.595, 0.665, 0.200, 0.080,
        "ch08 (no code)\nrecomputed via ch05\nconfig_infer.py", ec=HAIR, fc="white", fs=5.9,
        ls="--")
    box(ax, 0.595, 0.530, 0.200, 0.085,
        "ch09/config_reverse.py  (292)\ndual mirrors · 3-way assert", ec=TEAL, fc=CYAN_F, fs=5.9)

    # ---- 第四列：整合与收束（ch10-11） ----
    box(ax, 0.860, 0.815, 0.225, 0.105,
        "ch10/llama215.py  (322)\nimports ch02/03/04/06 slots\nassert 207,119,360 · HF parity",
        ec=YELLOW, fc=YLW_F, fs=5.9, lw=2.0)
    box(ax, 0.860, 0.675, 0.225, 0.092,
        "ch10/train_215.py  (316)\nprep_sp32k.py  (276)\nsmoke · integrated 1k steps", ec=YELLOW,
        fc=YLW_F, fs=5.9)
    box(ax, 0.860, 0.530, 0.225, 0.095,
        "ch10/DESIGN-215M.md  (156)\nrestart manual\n'ready to re-execute' standard", ec=TEAL,
        fc=CYAN_F, fs=5.9)
    box(ax, 0.860, 0.395, 0.225, 0.075,
        "ch11 (this figure script only)", ec=HAIR, fc="white", fs=5.9, ls="--")

    # ---- 依赖链：底座 → 对拍锚（青虚线：对拍关系） ----
    ax.plot([0.170, 0.170], [0.3625, 0.155], color=TEAL, lw=1.2, ls="--")
    arrow(ax, (0.170, 0.155), (0.150, 0.155), color=TEAL, lw=1.2, ls="--")
    ax.plot([0.215, 0.215], [0.2075, 0.815], color=TEAL, lw=1.2, ls="--")
    for yy in (0.815, 0.700, 0.585, 0.455, 0.325):
        arrow(ax, (0.215, yy), (0.240, yy), color=TEAL, lw=1.0, ls="--")
    ax.text(0.203, 0.620, "parity anchor", fontsize=5.4, color=TEAL, rotation=90, ha="center",
            va="center", style="italic")

    # ---- 依赖链：四插槽 → 整机（橙，单轨主线；沿底部肘形总线走，避开第三列） ----
    for yy in (0.815, 0.700, 0.585, 0.325):
        ax.plot([0.430, 0.470], [yy, yy], color=ORANGE, lw=1.4)
    ax.plot([0.470, 0.470], [0.815, 0.065], color=ORANGE, lw=1.6)
    ax.plot([0.470, 0.985], [0.065, 0.065], color=ORANGE, lw=1.6)
    ax.plot([0.985, 0.985], [0.065, 0.815], color=ORANGE, lw=1.6)
    arrow(ax, (0.985, 0.815), (0.9725, 0.815), color=ORANGE, lw=1.6)
    ax.text(0.560, 0.090, "single-track lineage: four slots imported by ch10/llama215.py",
            fontsize=5.6, color=ORANGE, style="italic")

    # ---- 整机 → 训练 → 设计文档（灰，章内供给） ----
    arrow(ax, (0.860, 0.7625), (0.860, 0.721), color=HAIR, lw=1.1)
    arrow(ax, (0.860, 0.629), (0.860, 0.5775), color=HAIR, lw=1.1)
    arrow(ax, (0.860, 0.4825), (0.860, 0.4325), color=HAIR, lw=1.1)

    ax.text(0.5, 0.012,
            "18 content files · 5,303 lines  "
            "(+ 8 feasibility probes · 1,298 lines as reference base; figure scripts not counted)",
            ha="center", va="center", fontsize=6.0, color=NOTE, style="italic")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-11-2-code-assets.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="1,2", help="只画指定图（如 --only 1）")
    which = set(int(x) for x in ap.parse_args().only.split(",") if x.strip())
    os.makedirs(FIG_DIR, exist_ok=True)
    if 1 in which:
        fig_11_1()
    if 2 in which:
        fig_11_2()


if __name__ == "__main__":
    main()
