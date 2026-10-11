# fig_ch06.py —— Book7 ch06 制图：图 6.1 KV 管理五层结构 / 图 6.2 哈希链与命中（含 CoW 分叉）
# 用途：ch06《KV cache 管理全景》两机制图——图 6.1=五层纵向栈+两外部角色+三态分派边（谁持有写权）；
#       图 6.2=三块链式哈希+新请求查询路径（命中/断链/未满尾块/CoW 私块分叉），小数字与例 6.1' 同构。
# 所属章节：Book7 ch06（§6.2/§6.3）。
# 运行：cd <workspace> && source env.sh && python code/Book7-vLLM源码剖析上/ch06/fig_ch06.py
# 产物：drafts/Book7-vLLM源码剖析上/figures/fig-6-1-five-layers.png / fig-6-2-hash-chain.png
# 图规：300dpi 白底；蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；线宽2；图内英文图注中文；
#       标注「按 v0.30.0 源码自绘（示意级）」；块内标内容小数字（尺寸流转标注制度）。
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "drafts", "Book7-vLLM源码剖析上", "figures"))
BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC, RED = "#0b0b0b", "#52514e", "#e1e0d9", "#c0392b"


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=7.0, lw=1.3, tc=INK, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw, ls=ls,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc, zorder=4)


def arr(ax, x0, y0, x1, y1, label="", color=MUTED, lw=1.3, fs=6.2, style="-|>", rad=0.0, off=(0, 0.25)):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, color=color, lw=lw,
                                 mutation_scale=12, connectionstyle=f"arc3,rad={rad}", zorder=2))
    if label:
        ax.text((x0 + x1) / 2 + off[0], (y0 + y1) / 2 + off[1], label, fontsize=fs, color=color,
                ha="center", zorder=5)


# ---------------- 图 6.1：五层结构 ----------------
def fig_five_layers():
    fig, ax = plt.subplots(figsize=(11.6, 6.2), dpi=300)
    ax.set_xlim(0, 16); ax.set_ylim(0, 9.6); ax.axis("off")
    ax.text(8, 9.25, "KV cache management: five layers between Scheduler and worker (vLLM V1, sketched from v0.30.0 source)",
            ha="center", fontsize=9.2, color=INK, weight="bold")
    LAYERS = [
        ("L5  facade   kv_cache_manager.py::KVCacheManager #L134",
         "get_computed_blocks / allocate_slots / free / cache_blocks\nholds: coordinator, empty_kv_cache_blocks (GC-free reuse)", BLUE, "#dcebfc"),
        ("L4  coordinator   kv_cache_coordinator.py::KVCacheCoordinator #L66",
         "cross-group dispatch: NoPrefixCache / Unitary / Hybrid\nholds: single_type_managers, block_pool (its constructor)", ORANGE, "#fde3d6"),
        ("L3  per-group ledger   single_type_kv_cache_manager.py #L51",
         "one per KV-cache group: FullAttention #L737 / SlidingWindow #L944 / Mamba #L1444 ...\nholds: req_to_blocks, num_cached_block (type-specific hit rules)", TEAL, "#d9f2e7"),
        ("L2  pool   block_pool.py::BlockPool #L143 (+ BlockHashToBlockMap #L33)",
         "all blocks + hash map; primitives: get_new_blocks / touch / free_blocks / evict / cache_full_blocks\nholds: blocks list, cached_block_hash_to_block", YELLOW, "#fdf0d2"),
        ("L1  data structures   kv_cache_utils.py::KVCacheBlock #L176 / FreeKVCacheBlockQueue #L246",
         "block_id / ref_cnt / block_hash / two list pointers; queue is the only writer of pointers", MUTED, "#f0efec"),
    ]
    y = 7.7
    for title, body, ec, fc in LAYERS:
        box(ax, 0.6, y - 1.25, 10.6, 1.28, "", ec=ec, fc=fc, lw=1.5)
        ax.text(0.85, y - 0.32, title, fontsize=7.6, color=INK, weight="bold", va="center")
        ax.text(0.85, y - 0.88, body, fontsize=6.6, color=MUTED, va="center")
        y -= 1.52
    for yy in (7.7 - 1.52 * k for k in range(4)):
        arr(ax, 5.9, yy + 0.06, 5.9, yy - 0.20, "", color=INK, lw=1.2)
    # 外部角色
    box(ax, 12.0, 7.4, 3.6, 1.5, "Scheduler  (ch5)\nschedule() calls\nallocate_slots / free", ec=INK, fs=7.0)
    box(ax, 12.0, 1.1, 3.6, 1.5, "worker  (ch7)\nblock_table executes;\nCoW copies run here", ec=INK, fs=7.0)
    arr(ax, 12.0, 8.15, 11.2, 8.15, "get_computed_blocks / allocate_slots", color=BLUE, fs=6.4, off=(0, 0.3))
    arr(ax, 11.2, 1.85, 12.0, 1.85, "block ids in SchedulerOutput;\nKVCacheBlockCopy (CoW)", color=TEAL, fs=6.4, off=(0, 0.42))
    # 三态分派边
    box(ax, 3.2, 0.15, 9.6, 0.75, "factory get_kv_cache_coordinator #L1014:  no caching -> NoPrefixCache   |   1 group -> Unitary   |   else -> Hybrid",
        ec=ORANGE, fc="white", fs=6.6)
    ax.text(8, 8.95, "who writes what: each layer names its own containers — the facade hides all internals from the Scheduler (docstring #L36-38)",
            ha="center", fontsize=6.8, color=MUTED, style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-6-1-five-layers.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print("[图 6.1] five-layers")


# ---------------- 图 6.2：哈希链与断链 ----------------
def fig_hash_chain():
    fig, ax = plt.subplots(figsize=(11.6, 5.6), dpi=300)
    ax.set_xlim(0, 16); ax.set_ylim(0, 8.6); ax.axis("off")
    ax.text(8, 8.25, "Chained block hashes: hit run, chain break, own new block (h = H(parent hash, block tokens, extra keys); numbers = toy Example 6.1)",
            ha="center", fontsize=8.8, color=INK, weight="bold")
    # 已缓存链（请求 A 遗产）：3 满块 + 尾块
    bx = 0.8
    ax.text(0.8, 7.35, "cached prefix left by request A (block_size = 4):", fontsize=7.4, color=MUTED)
    for i, (toks, h) in enumerate([("t0-t3", "h1"), ("t4-t7", "h2"), ("t8-t11", "h3"), ("t12", "—")]):
        full = i < 3
        box(ax, bx, 5.7, 2.3, 1.35, f"block {i+1}\ntokens {toks}\nhash {h}" if full else f"tail block 4\ntokens {toks}\n(1/4, no hash)",
            ec=TEAL if full else MUTED, fc="#d9f2e7" if full else "white", fs=6.8, lw=1.5, ls="-" if full else "--")
        bx += 2.6
    for x0 in (3.1, 5.7, 8.3):
        arr(ax, x0, 6.35, x0 + 0.25, 6.35, "", color=TEAL, lw=1.4)
    ax.text(4.5, 7.0, "parent -> child", fontsize=6.4, color=TEAL, ha="center")
    ax.text(1.95, 5.35, "h_i = H(h_(i-1), block tokens, extra)", fontsize=6.6, color=TEAL, ha="center")
    # 新请求 B 的查询路径
    ax.text(0.8, 4.35, "new request B (prompt 15 tokens, first 10 same as A):", fontsize=7.4, color=MUTED)
    box(ax, 0.8, 2.9, 2.3, 1.1, "query h1\nmatch", ec=BLUE, fc="#dcebfc", fs=6.8)
    box(ax, 3.4, 2.9, 2.3, 1.1, "query h2\nmatch", ec=BLUE, fc="#dcebfc", fs=6.8)
    box(ax, 6.0, 2.9, 2.9, 1.1, "B's 3rd full block hash h3'\n= H([t8,t9,b10,b11])\nMISS vs h3 -> chain breaks", ec=RED, fc="white", fs=6.4)
    box(ax, 9.2, 2.9, 2.9, 1.1, "B's own new block 5\nrecompute t8..b11\n(fresh allocation,\nno redirect)", ec=ORANGE, fc="#fde3d6", fs=6.4)
    arr(ax, 3.1, 3.45, 3.4, 3.45, "", color=BLUE); arr(ax, 5.7, 3.45, 6.0, 3.45, "", color=BLUE)
    arr(ax, 8.9, 3.45, 9.2, 3.45, "recompute from break", color=ORANGE, fs=6.0, off=(0, 0.28))
    # 命中账
    box(ax, 12.4, 2.9, 3.1, 2.3, "hit ledger\nhit = 2 full blocks\n= 8 tokens only\n(full-block principle;\nalignment loss = 2 tok)",
        ec=INK, fc="white", fs=6.6)
    arr(ax, 2.6, 5.7, 1.95, 4.0, "touch: remove from free queue, ref_cnt +1", color=BLUE, fs=6.0, rad=-0.15, off=(1.3, 0.5))
    ax.text(8, 0.55, "chained-hash property: one missing block implies every later block misses too (single_type #L790-791); full blocks only (design doc L22);\nCoW is NOT triggered here - it fires only when the hit length ends inside a block (parallel-sampling fork, sec 6.5)",
            ha="center", fontsize=6.8, color=MUTED, style="italic")
    ax.text(8, 0.15, "sketched from v0.30.0 source + docs/design/prefix_caching.md", ha="center", fontsize=6.2, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-6-2-hash-chain.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print("[图 6.2] hash-chain")


# ---------------- 图 6.3：hybrid 两类住客总览 ----------------
def fig_hybrid_overview():
    fig, ax = plt.subplots(figsize=(10.4, 4.2), dpi=300)
    ax.set_xlim(0, 14); ax.set_ylim(0, 6.4); ax.axis("off")
    ax.text(7, 6.05, "Two kinds of state in one pool (hybrid KV, overview only — deep dive in Book 8 ch.1)",
            ha="center", fontsize=9.0, color=INK, weight="bold")
    # full 泳道
    ax.text(0.6, 5.1, "full-attention group: reserve slots for ALL tokens", fontsize=7.4, color=BLUE)
    for i in range(7):
        box(ax, 0.6 + i * 1.55, 3.7, 1.4, 1.1, f"blk {i}", ec=BLUE, fc="#dcebfc", fs=6.8)
    ax.text(11.6, 4.25, "grows without bound\n(slots per token)", fontsize=6.6, color=MUTED)
    # SW 泳道
    ax.text(0.6, 2.6, "sliding-window group: only the most recent W tokens", fontsize=7.4, color=ORANGE)
    for i in range(7):
        keep = i >= 4
        box(ax, 0.6 + i * 1.55, 1.2, 1.4, 1.1, f"blk {i}" if keep else "freed",
            ec=ORANGE if keep else MUTED, fc="#fde3d6" if keep else "white",
            fs=6.8, lw=1.5 if keep else 1.0, ls="-" if keep else "--")
    ax.annotate("window W: older blocks freed as decode advances", xy=(6.7, 1.15), xytext=(7.2, 0.35),
                fontsize=6.6, color=MUTED, arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0))
    # Mamba/SSM 行
    ax.text(0.6, 0.55, "linear/SSM group (e.g. Mamba, GDN): fixed-size state, no per-token slots — the third kind managed by the same coordinator",
            fontsize=6.8, color=TEAL)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-6-3-hybrid-overview.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print("[图 6.3] hybrid-overview")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_five_layers()
    fig_hash_chain()
    fig_hybrid_overview()
