# fig_ch05.py —— Book6 ch5 制图：图 5.1 逻辑-物理映射与 block table / 图 5.3 浪费位置对比 /
#                图 5.4 三类生命周期时间线 / 图 5.5 准入政策对照（图 5.2=vLLM 论文 Fig 2 原图引用，不在本脚本；slug=paged-kv）
# 用途：图 5.1 = 5.3 机制总图（两请求逻辑块→块表→物理块池，标块大小/slot 数/映射箭头/块张量形状）；
#       图 5.3 = 5.2/5.4 浪费位置对比（v2 买断+外部碎片 vs v3 内部碎片封顶 b−1）；
#       图 5.4 = 5.6 三类住客时间线（全局无界/滑窗环形/线性固定）；
#       图 5.5 = 5.7 准入对照（喂 log/book6-ch05/engine_v3_admission.json：同一池容量下
#         连续买断三档 n_cap vs 分页按需的驻留请求数，条上标 KV 利用率）。
# 所属章节：Book6 第 5 章（图 5.1/5.3/5.4/5.5——图 5.2 为论文 Fig 2 原图引用）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/Book6-推理系统导论/ch05/fig_ch05.py"
#           [--which all|map|frag|life|admit]
# 产物：drafts/Book6-推理系统导论/figures/fig-5-{1,3,4,5}-paged-kv.png
# 图规：300dpi 白底；系列色 蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；线宽 2；图内英文、图注中文。
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "drafts", "Book6-推理系统导论", "figures")
ADMIT_JSON = os.path.join(REPO_ROOT, "log", "book6-ch05", "engine_v3_admission.json")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e1e0d9"
USED_A, USED_B = "#dcebfc", "#fde3d6"     # A/B 请求占用底的浅色（蓝/橙系）


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=8.0, lw=1.5, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc, zorder=4)


def arrow(ax, x0, y0, x1, y1, label=None, color=MUTED, fs=7.0, dx=0.0, dy=0.06, lw=1.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=lw, zorder=2))
    if label:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, label, ha="center", va="bottom",
                fontsize=fs, color=color, zorder=4)


# ---------------- 图 5.1：逻辑-物理映射与 block table ----------------
def fig_map():
    fig, ax = plt.subplots(figsize=(11.0, 5.8), dpi=300)
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    ax.text(6.8, 7.35, "Block table: logical tokens -> physical blocks (block size b=4)",
            ha="center", fontsize=11, color=INK, weight="bold")

    # 左：两请求的逻辑序列（A 长 7 / B 长 5，b=4）
    ax.text(2.3, 6.7, "logical token sequence", fontsize=8.5, color=INK, ha="center", weight="bold")
    for row, (name, n_tok, y) in enumerate([("request A (n=7)", 7, 5.2), ("request B (n=5)", 5, 3.4)]):
        ax.text(0.1, y + 0.62, name, fontsize=8, color=INK)
        for t in range(n_tok):
            fc = USED_A if row == 0 else USED_B
            ax.add_patch(Rectangle((0.2 + t * 0.52, y), 0.5, 0.5, fc=fc, ec=GRIDC, lw=0.8, zorder=3))
            ax.text(0.45 + t * 0.52, y + 0.25, f"t{t}", ha="center", va="center", fontsize=6.5,
                    color=INK, zorder=4)
        # 逻辑块边界括注
        for lb in range(0, n_tok, 4):
            x0 = 0.2 + lb * 0.52
            x1 = 0.2 + min(lb + 4, n_tok) * 0.52
            ax.plot([x0, x1], [y - 0.12, y - 0.12], color=MUTED, lw=1.4, zorder=3)
            ax.text((x0 + x1) / 2, y - 0.34, f"LB{lb // 4}", ha="center", fontsize=7, color=MUTED)

    # 中：块表（逻辑块 -> 物理块号；与例 5.1 的走查数字一致）
    ax.text(6.1, 6.7, "block table (per request)", fontsize=8.5, color=INK, ha="center", weight="bold")
    rows = [("A", 0, 9), ("A", 1, 8), ("B", 0, 7), ("B", 1, 6)]
    for i, (req, lb, phys) in enumerate(rows):
        y = 5.9 - i * 0.62
        box(ax, 4.9, y, 0.9, 0.48, f"LB{lb}", ec=MUTED, fs=7.5)
        box(ax, 6.4, y, 0.9, 0.48, f"blk {phys}", ec=BLUE, fs=7.5)
        arrow(ax, 5.85, y + 0.24, 6.35, y + 0.24, lw=1.0)
        ax.text(4.6, y + 0.24, req, fontsize=7.5, color=MUTED, ha="right", va="center")
    ax.text(6.1, 3.15, "locate(p) = (p ÷ b, p mod b)\nlogical pos -> (block, slot)",
            ha="center", fontsize=7.5, color=MUTED)

    # 右：物理块池（6/7/8/9 被占，槽位示占用与空槽）
    ax.text(10.9, 6.7, "physical block pool", fontsize=8.5, color=INK, ha="center", weight="bold")
    ax.text(10.9, 6.35, "block tensor: (L, 2, h_kv, b, d_k) = (12, 2, 8, 4, 64)  [207M, bf16: 96 KiB/block]",
            ha="center", fontsize=6.8, color=MUTED)
    owners = {9: ("A", 4), 8: ("A", 3), 7: ("B", 4), 6: ("B", 1)}
    for bi, bn in enumerate([9, 8, 7, 6]):
        x = 8.5 + bi * 1.2
        own, used = owners[bn]
        fc_block = USED_A if own == "A" else USED_B
        for s in range(4):
            occ = s < used
            ax.add_patch(Rectangle((x + s * 0.26, 5.2), 0.24, 0.9,
                                   fc=fc_block if occ else "white", ec=GRIDC, lw=0.8, zorder=3))
        ax.text(x + 0.52, 4.95, f"blk {bn} ({own}, {used}/4 slots)",
                ha="center", fontsize=6.8, color=INK)
    # 映射箭头：块表每行 -> 对应物理块（A 蓝 / B 橙）
    for i, (req, lb, phys) in enumerate(rows):
        y = 5.9 - i * 0.62
        x_t = 8.5 + [9, 8, 7, 6].index(phys) * 1.2 + 0.52
        c = BLUE if req == "A" else ORANGE
        ax.add_patch(FancyArrowPatch((7.35, y + 0.24), (x_t, 5.12), arrowstyle="-|>",
                                     mutation_scale=9, color=c, lw=0.9, zorder=2))
    ax.annotate("empty slots in last block\n= internal frag (< 1 block per request)",
                xy=(12.36, 5.5), xytext=(10.9, 4.25), fontsize=7.2, color=ORANGE, ha="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0))
    ax.text(2.3, 2.5, "any free block fits any request\n-> no external fragmentation",
            ha="center", fontsize=7.5, color=TEAL)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-5-1-paged-kv.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 5.1] {path}")


# ---------------- 图 5.3：两类碎片（v2 买断 vs v3 分页） ----------------
def fig_frag():
    fig, ax = plt.subplots(figsize=(10.6, 4.8), dpi=300)
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    ax.text(6.4, 6.15, "Where the waste lives: v2 contiguous buyout vs v3 paged blocks",
            ha="center", fontsize=10.5, color=INK, weight="bold")

    def unit(x, y, w_units, fc, ec=GRIDC, hatch=None, label=None, label_color=INK):
        ax.add_patch(Rectangle((x, y), w_units * 0.4, 0.8, fc=fc, ec=ec, lw=0.8,
                               hatch=hatch, zorder=3))
        if label:
            ax.text(x + w_units * 0.2, y + 0.4, label, ha="center", va="center",
                    fontsize=6.8, color=label_color, zorder=4)

    # 上排：v2 连续买断
    ax.text(0.15, 5.0, "v2: contiguous, bought at n_cap", fontsize=8.5, color=INK, weight="bold")
    unit(0.3, 3.9, 5, USED_A, label="req1 uses 5")
    unit(0.3 + 5 * 0.4, 3.9, 3, "white", hatch="///", label="reserved\n(3 wasted)")
    unit(0.3 + 8 * 0.4, 3.9, 6, USED_B, label="req2 uses 6")
    unit(0.3 + 14 * 0.4, 3.9, 2, "white", hatch="///", label="res.")
    unit(0.3 + 16 * 0.4, 3.9, 4, "#f4f3ef", hatch="xx", label="hole (4)")
    unit(0.3 + 20 * 0.4, 3.9, 8, USED_A, label="req3 buys 8, uses 8")
    ax.annotate("external frag: 4 free units, but no contiguous 8 for req4",
                xy=(0.3 + 18 * 0.4, 3.85), xytext=(5.6, 4.85), fontsize=7.2, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0))

    # 下排：v3 分页
    ax.text(0.15, 2.9, "v3: paged, block size b=4 (any free block fits)", fontsize=8.5,
            color=INK, weight="bold")
    x0 = 0.3
    for bi in range(7):
        x = x0 + bi * 1.68
        own = ["A", "A", "B", "B", "C", "F", "F"][bi]
        used = {"A": 4, "B": 4, "C": 2, "F": 0}[own]
        fc = {"A": USED_A, "B": USED_B, "C": "#d9f2e7", "F": "white"}[own]
        for s in range(4):
            occ = s < used
            ax.add_patch(Rectangle((x + s * 0.4, 1.5), 0.38, 0.8,
                                   fc=fc if occ else "white", ec=GRIDC, lw=0.8, zorder=3))
        tag = {"A": "A full", "B": "B full", "C": "C 2/4 +2 empty", "F": "free"}[own]
        ax.text(x + 0.8, 1.25, tag, ha="center", fontsize=6.8, color=INK)
    ax.annotate("internal frag: only last block, <= b-1 = 3 slots per request",
                xy=(0.3 + 4 * 1.68 + 0.9, 1.45), xytext=(6.2, 0.5), fontsize=7.2, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0))
    ax.text(11.1, 2.3, "free blocks are\ninterchangeable:\nno holes, no shapes",
            fontsize=7.4, color=TEAL, ha="center")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-5-3-paged-kv.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 5.3] {path}")


# ---------------- 图 5.4：三类生命周期时间线 ----------------
def fig_life():
    fig, ax = plt.subplots(figsize=(10.6, 4.0), dpi=300)
    ax.set_xlim(-0.4, 34)
    ax.set_ylim(-0.5, 3.9)
    ax.axis("off")
    ax.text(16.8, 3.7, "Three tenants, one pool: KV lifetimes by layer type (n_now = 28, W = 8, b = 4)",
            ha="center", fontsize=10, color=INK, weight="bold")

    def slot_row(y, lo, hi, fc, ec=GRIDC, dashed=False):
        for u in range(lo, hi):
            ax.add_patch(Rectangle((u, y), 0.92, 0.52, fc=fc, ec=ec, lw=0.7,
                                   ls="--" if dashed else "-", zorder=3))

    # 行 1：全局层——0..28 全占
    slot_row(2.6, 0, 28, USED_A)
    ax.text(-0.4, 3.12, "global layers", fontsize=8.2, color=INK, weight="bold")
    ax.text(29.3, 2.86, "grows with n -> ⌈n/b⌉ = 7 blocks alive\n(lives to end of request)", fontsize=7.2, color=MUTED)

    # 行 2：滑窗层——仅 [20,28) 占，之前虚线（已还块）
    slot_row(1.5, 0, 20, "white", dashed=True)
    slot_row(1.5, 20, 28, "#fde3d6")
    ax.text(-0.4, 2.02, "sliding layers (W=8)", fontsize=8.2, color=INK, weight="bold")
    ax.text(29.3, 1.76, "capped at ⌈W/b⌉ = 2 blocks\n(whole block returned once out of window)",
            fontsize=7.2, color=MUTED)
    ax.annotate("blocks 0-4 returned to pool", xy=(8, 1.75), xytext=(4.5, 0.72),
                fontsize=7.2, color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0))

    # 行 3：线性层——固定状态，不进 KV 池
    ax.add_patch(FancyBboxPatch((20, 0.35), 8, 0.52, boxstyle="round,pad=0.02",
                                fc="#d9f2e7", ec=TEAL, lw=1.4, zorder=3))
    ax.text(24, 0.61, "fixed state S", ha="center", va="center", fontsize=7.4, color=INK, zorder=4)
    ax.text(-0.4, 0.61, "linear layers", fontsize=8.2, color=INK, weight="bold", va="center")
    ax.text(29.3, 0.61, "constant size, activated/destroyed\nwith request — never in KV pool",
            fontsize=7.2, color=MUTED, va="center")

    ax.plot([0, 28.6], [0.12, 0.12], color=GRIDC, lw=0.8, zorder=1)
    for u in range(0, 29, 4):
        ax.plot([u, u], [0.04, 0.2], color=MUTED, lw=0.8, zorder=2)
        ax.text(u, -0.32, f"{u}", ha="center", fontsize=6.8, color="#898781")
    ax.text(31.2, -0.32, "token position", ha="center", fontsize=7.2, color="#898781")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-5-4-paged-kv.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 5.4] {path}")


# ---------------- 图 5.5：准入政策对照（喂 admission JSON） ----------------
def fig_admit():
    with open(ADMIT_JSON, encoding="utf-8") as f:
        d = json.load(f)
    rows = {r["policy"]: r for r in d["rows"]}
    order = ["contig_n198", "contig_n512", "contig_n2048", "paged"]
    labels = ["buyout\nn_cap=198", "buyout\nn_cap=512", "buyout\nn_cap=2048", "paged\n(actual, rounded to b)"]
    admitted = [rows[k]["admitted"] for k in order]
    utils = [rows[k]["utilization_pct"] for k in order]
    colors = [BLUE, BLUE, BLUE, TEAL]

    fig, ax = plt.subplots(figsize=(7.8, 4.4), dpi=300)
    bars = ax.bar(range(4), admitted, width=0.58, color=colors, zorder=3)
    for i, (b, a, u) in enumerate(zip(bars, admitted, utils)):
        ax.text(b.get_x() + b.get_width() / 2, a + 12, f"{a}", ha="center", fontsize=9,
                color=INK, weight="bold")
        ax.text(b.get_x() + b.get_width() / 2, a / 2 if a > 120 else a + 46, f"KV util\n{u}%",
                ha="center", fontsize=7.4, color="white" if a > 120 else MUTED)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("resident requests (same 4096-block pool)", fontsize=9, color=MUTED)
    ax.set_ylim(0, 820)
    ax.tick_params(colors="#898781", labelsize=8)
    ax.grid(axis="y", color=GRIDC, lw=0.6, zorder=0)
    ax.set_title("Concurrency is what fragmentation was eating:\nsame pool, admission policy decides capacity",
                 fontsize=10, color=INK)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-5-5-paged-kv.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 5.5] {path}（数据源 engine_v3_admission.json）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch5 制图（图 5.1/5.3/5.4/5.5）")
    ap.add_argument("--which", default="all", choices=["all", "map", "frag", "life", "admit"])
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    if args.which in ("all", "map"):
        fig_map()
    if args.which in ("all", "frag"):
        fig_frag()
    if args.which in ("all", "life"):
        fig_life()
    if args.which in ("all", "admit"):
        fig_admit()
