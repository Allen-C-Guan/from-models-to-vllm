# fig_ch07.py —— Book6 ch7 制图：图 7.1 generation stall 机制 / 图 7.2 piggybacking 拼批 /
#                图 7.3 干扰复现 A/B 的 gap 序列 / 图 7.4 四版引擎吞吐对照（收官图，喂 sweep JSON）
# 用途：图 7.1 = 7.1 机制图（依据 Sarathi-Serve Fig 7 机制自绘——版权档：Sarathi 系图需改绘；
#         上轨 prefill-prioritizing：长 prefill 期间 decode 停摆；下轨 stall-free：块+decode 拼批）；
#       图 7.2 = 7.2 piggybacking 的拼批构造（512-token prefill 块 + 16 条 decode 合成一次前向，
#         标形状与两段资源画像——算力段/带宽段互补）；
#       图 7.3 = 7.5 干扰复现 A/B（喂 log/book6-ch07/sched_v4_base.json：none vs chunked 的每迭代
#         decode 间隔序列——none 出一个 4.1s 尖峰、chunked 摊成 8 个加宽间隔）。
# 所属章节：Book6 第 7 章（图 7.1/7.2/7.3/7.4）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/Book6-推理系统导论/ch07/fig_ch07.py"
#           [--which all|stall|piggy|ab]
# 产物：fig-7-{1,2,3}-chunked-prefill.png + fig-7-4-engines.png
# 图规：300dpi 白底；系列色 蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；线宽 2；图内英文、图注中文。
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "drafts", "Book6-推理系统导论", "figures")
AB_JSON = os.path.join(REPO_ROOT, "log", "book6-ch07", "sched_v4_base.json")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e1e0d9"


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=7.6, lw=1.4, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc, zorder=4)


# ---------------- 图 7.1：generation stall 机制（依据 Sarathi-Serve Fig 7 自绘） ----------------
def fig_stall():
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 4.4), dpi=300)
    for ax, title in zip(axes, ["prefill-prioritizing (vLLM / Orca-style): full prefill blocks decodes",
                                "stall-free (chunked): one chunk per iteration, decodes ride along"]):
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 2.6)
        ax.axis("off")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    # 上轨：decode 小步 → 长 prefill 大块（期间 decode 停摆）→ decode 恢复
    a = axes[0]
    for i in range(4):
        a.add_patch(Rectangle((0.4 + i * 0.75, 1.4), 0.62, 0.55, fc=BLUE, ec="white", lw=0.5, zorder=3))
    a.text(1.7, 2.15, "decode steps (A, B)", fontsize=7.4, color=BLUE, ha="center")
    a.add_patch(Rectangle((3.5, 1.4), 5.4, 0.55, fc=ORANGE, ec="white", lw=0.5, zorder=3))
    a.text(6.2, 2.15, "full prefill of C (8k tok) — seconds", fontsize=7.6, color=ORANGE, ha="center")
    a.add_patch(Rectangle((4.0, 0.55), 4.4, 0.5, fc="none", ec=ORANGE, lw=1.0, ls="--", zorder=3))
    a.text(6.2, 0.80, "decodes for A, B STALLED (TBT spike)", fontsize=7.4, color=ORANGE, ha="center")
    for i in range(3):
        a.add_patch(Rectangle((9.2 + i * 0.75, 1.4), 0.62, 0.55, fc=BLUE, ec="white", lw=0.5, zorder=3))
    a.text(10.3, 2.15, "decodes resume", fontsize=7.4, color=BLUE, ha="center")
    a.annotate("", xy=(12.6, 1.65), xytext=(9.2, 1.65),
               arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.0))
    a.text(10.9, 1.25, "TTFT of C ok,\nTBT of A/B broken", fontsize=7.0, color=MUTED, ha="center")
    # 下轨：每迭代 = decode 步 + 一块 prefill
    b = axes[1]
    for i in range(8):
        b.add_patch(Rectangle((0.4 + i * 1.62, 1.4), 0.62, 0.55, fc=BLUE, ec="white", lw=0.5, zorder=3))
        b.add_patch(Rectangle((1.06 + i * 1.62, 1.4), 0.5, 0.55, fc=TEAL, ec="white", lw=0.5, zorder=3))
    b.text(6.9, 2.15, "every iteration: decodes (blue) + one 1k-token chunk (teal) — no stall",
           fontsize=7.6, color=INK, ha="center")
    b.text(6.9, 0.85, "C's prefill finishes after 8 iterations; TTFT of C slightly later, TBT of A/B flat",
           fontsize=7.4, color=MUTED, ha="center")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-7-1-chunked-prefill.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 7.1] {path}")


# ---------------- 图 7.2：piggybacking 拼批构造（512+16，标形状） ----------------
def fig_piggy():
    fig, ax = plt.subplots(figsize=(10.6, 4.6), dpi=300)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.text(6.5, 5.95, "Piggybacking: one hybrid batch = 1 prefill chunk + 16 decodes, one forward",
            ha="center", fontsize=10, color=INK, weight="bold")
    # 输入两路
    box(ax, 0.3, 3.6, 2.5, 1.0, "prefill chunk\n(512, C) rows", ec=ORANGE, fc="#fde3d6")
    box(ax, 0.3, 1.8, 2.5, 1.0, "16 decodes\n(16, 1, C) rows", ec=BLUE, fc="#dcebfc")
    box(ax, 3.6, 2.7, 2.2, 1.0, "concat\n(528, C)", ec=MUTED)
    ax.annotate("", xy=(3.6, 3.2), xytext=(2.8, 4.1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    ax.annotate("", xy=(3.6, 3.2), xytext=(2.8, 2.3), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    # 一次前向（线性段共享权重）
    box(ax, 6.3, 2.7, 2.6, 1.0, "linear/FFN\none big matmul\nweights read once", ec=TEAL, fc="#d9f2e7")
    ax.annotate("", xy=(6.3, 3.2), xytext=(5.8, 3.2), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    box(ax, 9.4, 3.5, 2.6, 0.9, "attention: chunk block\nvs all seen KV", ec=ORANGE)
    box(ax, 9.4, 2.0, 2.6, 0.9, "attention: decodes\nvs own history (B,h,1,n)", ec=BLUE)
    ax.annotate("", xy=(9.4, 3.9), xytext=(8.9, 3.5), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.0))
    ax.annotate("", xy=(9.4, 2.4), xytext=(8.9, 3.0), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.0))
    ax.text(6.5, 1.1, "resource profiles complement: chunk pays compute (AI≈500), decodes pay bandwidth "
                      "(AI≈16) — one weight traversal serves both",
            ha="center", fontsize=7.8, color=MUTED)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-7-2-chunked-prefill.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 7.2] {path}")


# ---------------- 图 7.4：干扰复现 A/B 的 gap 序列 ----------------
def fig_ab():
    d = json.load(open(AB_JSON))
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 5.2), dpi=300, sharex=False)
    for ax, mode, color in zip(axes, ("none", "chunked"), (ORANGE, TEAL)):
        m = d["modes"][mode]
        gaps, kinds = m["gaps_ms"], m["kinds"]
        xs = list(range(1, len(gaps) + 1))
        ax.bar(xs, gaps, color=[ORANGE if k != "decode" else BLUE for k in kinds], zorder=3)
        mx = m["max_gap_ms"]
        ax.axhline(m["median_gap_ms"], color=MUTED, ls="--", lw=1.0, zorder=2)
        ax.text(len(gaps) - 0.5, m["median_gap_ms"] + 40,
                f"median {m['median_gap_ms']} ms", fontsize=7.4, color=MUTED, ha="right")
        if mode == "none":
            idx = gaps.index(mx)
            ax.annotate(f"STALL: {mx} ms\n(full 2048-tok prefill)", xy=(idx + 1, mx),
                        xytext=(idx - 9, mx * 0.82), fontsize=8, color=ORANGE,
                        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
            ax.set_title(f"mode = none: one giant prefill iteration, decodes stall", fontsize=9.5, color=INK)
        else:
            ax.set_title(f"mode = chunked (256/iter): gaps widened but bounded, no cliff", fontsize=9.5, color=INK)
        ax.set_ylabel("decode iteration gap (ms)", fontsize=8.5, color=MUTED)
        ax.set_ylim(0, max(4400, max(gaps) * 1.15))
        ax.tick_params(colors="#898781", labelsize=7.5)
        ax.grid(axis="y", color=GRIDC, lw=0.6, zorder=0)
        ax.text(0.99, 0.92, f"max {mx} ms | long req TTFT {m['long_ttft_ms']} ms",
                transform=ax.transAxes, fontsize=7.6, color=MUTED, ha="right")
    axes[1].set_xlabel("decode iteration index", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-7-3-chunked-prefill.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 7.3] {path}（数据源 sched_v4_base.json）")




# ---------------- 图 7.4：四版引擎吞吐对照（收官图，喂 sweep JSON——此前无脚本的既有件并入） ----------------
def fig_engines():
    pts = {}
    for tag in ("v1", "v2"):
        lst = []
        for B in (1, 2, 4, 8):
            p_ = os.path.join(REPO_ROOT, "log", "book6-ch04", f"engine_{tag}_sweep_b{B}.json")
            if os.path.isfile(p_):
                d = json.load(open(p_))
                lst.append((B, d["throughput_tok_s"]))
        pts[tag] = lst
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=300)
    for tag, color in (("v1", BLUE), ("v2", ORANGE)):
        lst = pts[tag]
        ax.plot([x for x, _ in lst], [y for _, y in lst], color=color, marker="o" if tag == "v1" else "s",
                lw=2.2, ms=6, label=f"{tag}: " + ("naive full-refwd" if tag == "v1" else "continuous + KV"), zorder=4)
    ax.set_xlabel("concurrent requests B", fontsize=9, color=MUTED)
    ax.set_ylabel("throughput (tok/s)", fontsize=9, color=MUTED)
    ax.set_xticks([1, 2, 4, 8])
    ax.tick_params(colors="#898781", labelsize=8)
    ax.grid(color=GRIDC, lw=0.6, zorder=0)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("Four engines, one ledger each: batch scaling (v3/v4 gains live in other ledgers)",
                 fontsize=9.5, color=INK)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-7-4-engines.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 7.4] {path}（数据源 engine_v1/v2_sweep_b*.json）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch7 制图（图 7.1/7.2/7.4）")
    ap.add_argument("--which", default="all", choices=["all", "stall", "piggy", "ab", "engines"])
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    if args.which in ("all", "stall"):
        fig_stall()
    if args.which in ("all", "piggy"):
        fig_piggy()
    if args.which in ("all", "ab"):
        fig_ab()
    if args.which in ("all", "engines"):
        fig_engines()
