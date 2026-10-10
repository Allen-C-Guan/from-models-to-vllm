# fig_ch06.py —— Book6 ch6 制图：图 6.1 基数树演化 / 图 6.2 三场景命中率 vs 天花板 /
#                图 6.3 多轮 TTFT 明细与 dec 平坦线（编号=正文出现顺序；slug=prefix-cache）
# 用途：图 6.1 = 6.2 RadixAttention 主图（依据 SGLang 论文 Fig 3 裁 4 时间点自绘——版权档：SGLang
#         图需改绘；节点字母/三色编码（新增绿/命中蓝/驱逐红）沿论文例，边标 token 段）；
#       图 6.2 = 6.1/6.3 三场景命中率（自产：v4 三 workload 实测 vs 手算天花板——fewshot 的
#         86.6% vs 87.3% 差=块对齐损耗）；
#       图 6.3 = 6.4 多轮 TTFT 逐轮明细（喂 engine_v4_base.json rounds_detail：命中/未命中着色、
#         冷启动首轮单列）+ 下幅 dec_ms 平坦线（TPOT 一分不省的实证）。
# 所属章节：Book6 第 6 章（图 6.1-6.3）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/Book6-推理系统导论/ch06/fig_ch06.py"
#           [--which all|tree|scenes|ttft]
# 产物：drafts/Book6-推理系统导论/figures/fig-6-{1,2,3}-prefix-cache.png
# 图规：300dpi 白底；系列色 蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；线宽 2；图内英文、图注中文。
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "drafts", "Book6-推理系统导论", "figures")
V4_BASE = os.path.join(REPO_ROOT, "log", "book6-ch06", "engine_v4_base.json")
V4_SCENES = {"multiturn": os.path.join(REPO_ROOT, "log", "book6-ch06", "engine_v4_multiturn-check.json"),
             "fewshot": os.path.join(REPO_ROOT, "log", "book6-ch06", "engine_v4_fewshot.json"),
             "agent": os.path.join(REPO_ROOT, "log", "book6-ch06", "engine_v4_agent.json")}

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e1e0d9"
NEW_C, HIT_C, EVICT_C = TEAL, BLUE, ORANGE


def node(ax, x, y, ch, color=INK, fc="white"):
    ax.add_patch(Circle((x, y), 0.09, fc=fc, ec=color, lw=1.6, zorder=4))
    ax.text(x, y, ch, ha="center", va="center", fontsize=7.5, color=color, zorder=5)


def edge(ax, x0, y0, x1, y1, label, color=MUTED, lw=1.4):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, zorder=2)
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.10, label, fontsize=6.4, color=INK,
                ha="center", va="bottom", zorder=3)


# ---------------- 图 6.1：基数树四时间点演化（依据 SGLang Fig 3 机制自绘） ----------------
def fig_tree():
    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.2), dpi=300)
    titles = ["(1) empty tree", "(2) first chat inserted",
              "(3) round 2: walk to fork, extend", "(4) memory tight: LRU evicts a true leaf"]
    for ax, t in zip(axes, titles):
        ax.set_xlim(-0.15, 1.25)
        ax.set_ylim(-0.75, 0.85)
        ax.axis("off")
        ax.set_title(t, fontsize=8.2, color=INK, pad=4)
        node(ax, 0, 0, "")
        ax.text(0, -0.22, "root", fontsize=6.5, color=MUTED, ha="center")

    a = axes[1]
    edge(a, 0, 0, 0.55, 0.35, "Sys [0:8]")
    edge(a, 0.55, 0.35, 1.05, 0.35, "Q1/A1 [8:12]")
    node(a, 0.55, 0.35, "a", NEW_C)
    node(a, 1.05, 0.35, "b", NEW_C)

    a = axes[2]
    edge(a, 0, 0, 0.55, 0.35, "Sys [0:8]")
    edge(a, 0.55, 0.35, 1.05, 0.35, "Q1/A1 [8:12]")
    edge(a, 0.55, 0.35, 1.05, -0.30, "Q2/A2 [12:16]")
    node(a, 0.55, 0.35, "a", HIT_C, fc="#dcebfc")
    node(a, 1.05, 0.35, "b", HIT_C, fc="#dcebfc")
    node(a, 1.05, -0.30, "c", NEW_C)
    a.annotate("match_prefix walks here\n(hit 12 tok, 0 recompute)",
               xy=(0.72, 0.35), xytext=(0.24, -0.62), fontsize=6.6, color=HIT_C,
               arrowprops=dict(arrowstyle="->", color=HIT_C, lw=0.9))

    a = axes[3]
    edge(a, 0, 0, 0.55, 0.35, "Sys [0:8]")
    edge(a, 0.55, 0.35, 1.05, -0.30, "Q2/A2 (running)")
    edge(a, 1.05, -0.30, 1.18, -0.30, "Q3...")
    edge(a, 0.55, 0.35, 1.02, 0.62, "old chat [8:12']")
    node(a, 0.55, 0.35, "a", HIT_C, fc="#dcebfc")
    node(a, 1.05, -0.30, "c", HIT_C, fc="#dcebfc")
    node(a, 1.18, -0.30, "d", NEW_C)
    a.plot([1.02, 1.02], [0.62, 0.70], color=EVICT_C, lw=1.0, ls=":")
    a.add_patch(Circle((1.02, 0.62), 0.10, fc="white", ec=EVICT_C, lw=1.2, zorder=3))
    a.text(1.02, 0.62, "×", fontsize=8, color=EVICT_C, ha="center", va="center", zorder=5)
    a.text(1.05, 0.42, "leaf of a FINISHED chat evicted (LRU);\nancestor stays: active child path + ref>0",
           fontsize=6.2, color=EVICT_C, ha="center")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-6-1-prefix-cache.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 6.1] {path}")


# ---------------- 图 6.2：三场景命中率 vs 手算天花板 ----------------
def fig_scenes():
    labels = ["multiturn\n(3 dialogs × 4 rounds)", "few-shot\n(12 req, shared 1000-tok shots)",
              "agent\n(2 sessions × 6 rounds, sys 512)"]
    measured, ceiling = [], []
    for k in ("multiturn", "fewshot", "agent"):
        d = json.load(open(V4_SCENES[k]))
        measured.append(d["hit_rate_pct"])
    ceiling = [71.4, 87.3, 82.2]          # 手算未对齐天花板（multiturn/fewshot 的损耗=块对齐截断）
    x = range(3)
    fig, ax = plt.subplots(figsize=(7.8, 4.2), dpi=300)
    ax.bar([i - 0.17 for i in x], ceiling, width=0.32, color=BLUE, label="hand-calculated ceiling", zorder=3)
    ax.bar([i + 0.17 for i in x], measured, width=0.32, color=TEAL, label="v4 measured", zorder=3)
    for i in x:
        ax.text(i + 0.17, measured[i] + 1.5, f"{measured[i]}", ha="center", fontsize=8.5, color=INK)
        ax.text(i - 0.17, ceiling[i] + 1.5, f"{ceiling[i]}", ha="center", fontsize=8.5, color=MUTED)
    ax.annotate("gap = block-alignment loss\n(multiturn 4.7pp / fewshot 0.7pp)",
                xy=(1 + 0.0, 87.0), xytext=(0.42, 60), fontsize=7.2, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0))
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylabel("prefix hit rate (%)", fontsize=9, color=MUTED)
    ax.set_ylim(0, 100)
    ax.tick_params(colors="#898781", labelsize=8)
    ax.grid(axis="y", color=GRIDC, lw=0.6, zorder=0)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-6-2-prefix-cache.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 6.2] {path}（数据源 v4 三 workload JSON）")


# ---------------- 图 6.3：多轮 TTFT 明细 + dec 平坦线 ----------------
def fig_ttft():
    d = json.load(open(V4_BASE))
    rounds = d["rounds_detail"]
    tags = [f"d{r['dialog']}r{r['round']}" for r in rounds]
    ttft = [r["ttft_ms"] for r in rounds]
    hitpct = [r["hit_len"] / r["prompt_len"] * 100 for r in rounds]
    dec = [r["dec_ms"] for r in rounds]
    x = range(len(rounds))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.4, 5.6), dpi=300, sharex=True,
                                   gridspec_kw=dict(height_ratios=[3, 1.4]))
    cols = [YELLOW if h == 0 else BLUE for h in hitpct]
    ax1.bar(x, ttft, color=cols, zorder=3)
    for i, (t, h) in enumerate(zip(ttft, hitpct)):
        ax1.text(i, t + 12, f"{t:.0f}", ha="center", fontsize=6.8, color=INK)
        if h:
            ax1.text(i, t - 74, f"hit\n{h:.0f}%", ha="center", fontsize=6.0, color="white")
    ax1.axhline(348.5, color=MUTED, ls="--", lw=1.1, zorder=2)
    ax1.text(11.6, 362, "steady-state miss baseline ≈348.5 ms", fontsize=7.2, color=MUTED, ha="right")
    ax1.set_ylabel("TTFT (ms)", fontsize=9, color=MUTED)
    ax1.set_ylim(0, 950)
    ax1.set_title("Round-by-round TTFT (bars: yellow = full miss / blue = prefix hit) & flat decode time",
                  fontsize=9.5, color=INK)
    ax1.tick_params(colors="#898781", labelsize=7.5)
    ax1.grid(axis="y", color=GRIDC, lw=0.6, zorder=0)
    ax2.plot(list(x), dec, color=ORANGE, marker="o", lw=2, ms=5, zorder=4)
    ax2.set_ylabel("decode 8 tok\n(ms)", fontsize=8, color=MUTED)
    ax2.set_ylim(250, 340)
    ax2.tick_params(colors="#898781", labelsize=7.5)
    ax2.grid(axis="y", color=GRIDC, lw=0.6, zorder=0)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(tags, fontsize=7)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-6-3-prefix-cache.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 6.3] {path}（数据源 engine_v4_base.json）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch6 制图（图 6.1-6.3）")
    ap.add_argument("--which", default="all", choices=["all", "tree", "scenes", "ttft"])
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    if args.which in ("all", "tree"):
        fig_tree()
    if args.which in ("all", "scenes"):
        fig_scenes()
    if args.which in ("all", "ttft"):
        fig_ttft()
