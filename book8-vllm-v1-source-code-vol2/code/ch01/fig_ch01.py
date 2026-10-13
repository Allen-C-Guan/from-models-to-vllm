# fig_ch01.py —— Book8 ch01 制图：图 1.1 混合组块布局（一个池子、两种集装箱）
# 所属章节：Book8 ch1（§1.2）；设计书=notes/13 §二图规格（例 1.1 的图像版）
# 运行：cd <workspace> && source env.sh && python code/Book8-vLLM源码剖析下/ch01/fig_ch01.py
# 产物：drafts/Book8-vLLM源码剖析下/figures/fig-1-1-hybrid-blocks.png
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "drafts", "Book8-vLLM源码剖析下", "figures"))
BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#0b0b0b", "#52514e"


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=6.4, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


def blocks(ax, x, y, n, color, label_below=None):
    for i in range(n):
        ax.add_patch(Rectangle((x + i * 0.55, y), 0.5, 0.5, ec=INK, fc=color, lw=0.8, zorder=3))
    if label_below:
        ax.text(x + n * 0.55 / 2, y - 0.42, label_below, ha="center", fontsize=6.2, color=MUTED)


def fig_hybrid_blocks():
    fig, ax = plt.subplots(figsize=(12.6, 6.4), dpi=300)
    ax.set_xlim(0, 18); ax.set_ylim(0, 9.4); ax.axis("off")
    ax.text(9, 9.05, "One pool, two container types: FA group grows with n, SWA group rolls in place (b=128, W=512; sketched from vLLM v0.30.0 source)",
            ha="center", fontsize=8.6, color=INK, weight="bold")
    box(ax, 0.5, 7.3, 17.0, 1.1, "shared block pool (page-unified: unify_kv_cache_spec_page_size #L1365; budget: num_blocks = mem // page_bytes #L1648)",
        ec=INK, fs=6.8)
    # FA 组（n=4096 → 32 块展示 12 块省略）
    box(ax, 0.7, 4.6, 7.8, 1.9,
        "FA group (2 layers)\nblocks grow with sequence length\nn=4096 -> 32 blocks/layer\ncdiv(n,b) * L_fa", ec=BLUE, fc="#dcebfc", fs=6.6)
    blocks(ax, 1.1, 5.0, 12, "#9cc4f2")
    # SWA 组（W=512 → 5 块滚动）
    box(ax, 9.2, 4.6, 8.0, 1.9,
        "SWA group (6 layers)\nrolling window, ~fixed footprint\nW=512 -> 5 blocks/layer (cdiv(W,b)+1)\nold tail freed as window slides", ec=ORANGE, fc="#fde3d6", fs=6.4)
    blocks(ax, 9.6, 5.0, 5, "#f6b394")
    ax.add_patch(Rectangle((9.6, 5.0), 0.5, 0.5, ec=YELLOW, lw=2.2, fill=False, zorder=5))
    ax.text(9.85, 4.82, "+1 rounding (phase)", ha="center", fontsize=5.6, color="#8a6d1a")
    # 账板
    box(ax, 0.7, 2.4, 7.8, 1.6,
        "n=512: hybrid 38 blocks vs pure-FA 32  (LOSS +6)\nn=4096: hybrid 94 vs pure-FA 256  (SAVE 63.3%)\nmixed pays off only when n >> W", ec=TEAL, fc="#d9f2e7", fs=6.4)
    box(ax, 9.2, 2.4, 8.0, 1.6,
        "linear/SSM group: fixed state as whole pages\nMambaSpec 3-state + checkpoints\n(NPU: no generic backend - read-only here)", ec=BLUE, fc="#dcebfc", fs=6.2)
    box(ax, 0.7, 0.5, 16.5, 1.4,
        "coordinator: HybridKVCacheCoordinator #L615 - fixed-point hit merge (accept-or-reduce per type, monotone convergent)\nNPU rewrite: AscendHybridKVCacheCoordinator (2-group max, FA must be one of them)",
        ec=INK, fs=6.4)
    for x in (4.6, 13.2):
        arr = FancyArrowPatch((x, 7.3), (x, 6.5), arrowstyle="-|>", color=MUTED, lw=1.1, mutation_scale=10)
        ax.add_patch(arr)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-1-1-hybrid-blocks.png")
    fig.savefig(out, facecolor="white"); plt.close(fig)
    print("[fig]", out)


if __name__ == "__main__":
    fig_hybrid_blocks()
