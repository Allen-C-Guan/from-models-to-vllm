# -*- coding: utf-8 -*-
# 用途：Book2 第 2 章插图——图 2.1 BPE 合并树（Sennrich 手算例的测试时切分推导）与
#       图 2.2 词表大小-压缩率曲线（自写 mini-BPE 扫描，训练 vs held-out）
# 所属章节：《预训练革命》第 2 章；数据源 log/book2-ch02/bpe_report.json（bpe.py 产出）
# 运行方式：source env.sh && python "code/ch02/make_figs.py"
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE, INK, SUB, AXIS, HAIR = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": AXIS,
    "axes.labelcolor": SUB, "xtick.color": AXIS, "ytick.color": AXIS,
    "axes.grid": True, "grid.color": HAIR, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "font.family": "DejaVu Sans", "font.size": 9,
})

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(ROOT, "figures")
REPORT = json.load(open(os.path.join(ROOT, "log", "book2-ch02", "bpe_report.json"), encoding="utf-8"))


def fig_merge_tree():
    """图 2.1：OOV 词 'lower' 的测试时切分推导（合并树）。
    学习到的合并序（图注词典 {low:5, lowest:2, newer:6, wider:3}）：
    ①(e,r)→er  ②(er,</w>)→er·  ③(l,o)→lo  ④(lo,w)→low  ⇒ lower → [low][er·]"""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, 4.5)
    ax.axis("off")

    def node(x, y, text, kind):
        w, h = 0.92, 0.55
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                             boxstyle="round,pad=0.02",
                             fc="#f3f7fd" if kind == "char" else ("#dceafb" if kind == "mid" else BLUE),
                             ec=BLUE, lw=1.2)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=10,
                color=INK if kind != "final" else "white", family="DejaVu Sans")

    def arrow(x1, y1, x2, y2, label=None, lx=None, ly=None):
        ax.add_patch(FancyArrowPatch((x1, y1 + 0.28), (x2, y2 - 0.28),
                                     arrowstyle="-|>", mutation_scale=10,
                                     color=AXIS, lw=1.0, shrinkA=0, shrinkB=0))
        if label:
            ax.text(lx if lx is not None else (x1 + x2) / 2 + 0.14,
                    ly if ly is not None else (y1 + y2) / 2, label,
                    fontsize=8.5, color=ORANGE, ha="left", va="center")

    # 底层字符叶（'lower' 字符起步 + 词尾符 </w>，论文记号 ·）
    leaves = [("l", 2.0), ("o", 3.2), ("w", 4.4), ("e", 5.6), ("r", 6.8), ("·</w>", 8.0)]
    for t, x in leaves:
        node(x, 0, t, "char")
    # 合并节点（编号 = 学习到的合并序）
    node(6.2, 1, "er", "mid");   arrow(5.6, 0, 6.2, 1, "#1")     # ① (e,r)→er
    node(7.1, 2, "er·", "mid");  arrow(6.2, 1, 7.1, 2, "#2")     # ② (er,</w>)→er·
    arrow(8.0, 0, 7.1, 2)
    node(2.6, 1, "lo", "mid");   arrow(2.0, 0, 2.6, 1, "#3")     # ③ (l,o)→lo
    arrow(3.2, 0, 2.6, 1)
    node(3.5, 2, "low", "mid");  arrow(2.6, 1, 3.5, 2, "#4", lx=2.72, ly=1.62)  # ④ (lo,w)→low
    arrow(4.4, 0, 3.5, 2)
    # 最终两片
    node(3.5, 3.6, "low", "final")
    node(7.1, 3.6, "er·", "final")
    arrow(3.5, 2, 3.5, 3.6)
    arrow(7.1, 2, 7.1, 3.6)
    ax.text(5.3, 4.25, "OOV  'lower'  →  [ low ][ er· ]  (2 pieces)",
            ha="center", fontsize=11, color=INK)
    ax.text(0.25, 0.9, "start from\ncharacters", fontsize=8.5, color=SUB, va="center")
    ax.text(9.75, 1.9, "merge rules apply in\nlearned order #1…#4;\nunused rules skip",
            fontsize=8.5, color=SUB, ha="right", va="center")
    ax.set_title("BPE test-time merges for the OOV word 'lower' "
                 "(dict: low x5, lowest x2, newer x6, wider x3)", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-2-1-merge-tree.png"))
    plt.close(fig)


def fig_vocab_compression():
    """图 2.2：词表大小-压缩率曲线（口径 A 非空白字符/piece；训练语料 vs held-out 同分布切片）。"""
    sweep = REPORT["mini_bpe"]["sweep"]
    vocab = [s["vocab"] for s in sweep]
    train = [s["train"] for s in sweep]
    held = [s["heldout"] for s in sweep]
    fig, ax = plt.subplots(figsize=(6.4, 4), dpi=300)
    ax.plot(vocab, train, "o-", color=BLUE, lw=2, ms=6, label="train slice (400K chars)")
    ax.plot(vocab, held, "s-", color=ORANGE, lw=2, ms=6, label="held-out slice (100K chars)")
    # 词表=98 基础字符+合并数 的注记与泛化损失注记
    ax.annotate("vocab = 98 base chars\n+ #merges (char level = 1.0)",
                xy=(98, 1.0), xytext=(180, 1.55), fontsize=8.5, color=SUB,
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    ax.annotate(f"generalization gap\n{train[3]:.2f} vs {held[3]:.2f} at 1,098\n"
                f"({train[-1] - held[-1]:.2f} at 4,098: {train[-1]:.2f} vs {held[-1]:.2f})",
                xy=(1098, 2.322), xytext=(1450, 1.28), fontsize=8.5, color=SUB,
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    ax.set_xscale("log")
    ax.set_xticks(vocab)
    ax.set_xticklabels([f"{v:,}" for v in vocab], rotation=0)
    ax.set_xlabel("vocabulary size (= base characters + merge operations)", color=SUB)
    ax.set_ylabel("chars per piece (non-space)", color=SUB)
    ax.set_ylim(0.8, 3.8)
    ax.set_title("Compression saturates as vocabulary grows (hand-written BPE, 1MB corpus)",
                 color=INK)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-2-2-vocab-compression.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_merge_tree()
    fig_vocab_compression()
    print("saved ->", os.path.join(FIG_DIR, "fig-2-1-merge-tree.png"))
    print("saved ->", os.path.join(FIG_DIR, "fig-2-2-vocab-compression.png"))
