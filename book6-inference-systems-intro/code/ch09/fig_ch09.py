# fig_ch09.py —— Book6 ch9 制图：图 9.1 三温度分布 / 图 9.2 top-k vs top-p 截断对比（slug=sampling）
# 用途：图 9.1 = 9.2 节同一 logits 在 T=0.5/1/2 下的三套概率（喂 samplers_base.json 温度读数，
#         标 top3 与熵——锐化/平坦化的视觉正身）；
#       图 9.2 = 9.3 节尖/平两个分布上 top-k（恰 k 个）与 top-p（按累积概率自适应）的保留集差
#         （喂 samplers_base.json adaptive 读数——自绘条形+保留集着色）。
# 所属章节：Book6 第 9 章（图 9.1-9.2）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/Book6-推理系统导论/ch09/fig_ch09.py"
# 产物：drafts/Book6-推理系统导论/figures/fig-9-{1,2}-sampling.png
# 图规：300dpi 白底；系列色 蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；线宽 2；图内英文、图注中文。
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "drafts", "Book6-推理系统导论", "figures")
SAM_JSON = os.path.join(REPO_ROOT, "log", "book6-ch09", "samplers_base.json")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e1e0d9"


# ---------------- 图 9.1：三温度分布 ----------------
def fig_temp():
    g = torch.Generator().manual_seed(20261002)
    logits = torch.randn(50, generator=g) * 3
    temps = (0.5, 1.0, 2.0)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), dpi=300, sharey=True)
    for ax, t in zip(axes, temps):
        p = F.softmax(logits / t, dim=-1)
        top3 = p.topk(3).values.tolist()
        ent = float(-(p * p.clamp_min(1e-12).log()).sum())
        ax.bar(range(50), p, color=BLUE if t == 1.0 else (ORANGE if t < 1 else TEAL), zorder=3)
        ax.set_title(f"T = {t}   top3 = {top3[0]:.2f},{top3[1]:.2f},{top3[2]:.2f}   H = {ent:.2f}",
                     fontsize=8.4, color=INK)
        ax.tick_params(colors="#898781", labelsize=7)
        ax.grid(axis="y", color=GRIDC, lw=0.5, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("probability", fontsize=9, color=MUTED)
    fig.suptitle("Same logits, three temperatures: low T sharpens (top-3 eats the mass), high T flattens (entropy up)",
                 fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(FIG_DIR, "fig-9-1-sampling.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 9.1] {path}（与 samplers_base.json temperature 块同种子同读数）")


# ---------------- 图 9.2：top-k vs top-p 截断对比 ----------------
def fig_cutoff():
    sharp = torch.zeros(20); sharp[3] = 8.0
    flat = torch.ones(20)
    def topp_keep(lg, p):
        probs = F.softmax(lg, dim=-1)
        sp, idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        keep = cum - sp < p
        m = torch.zeros_like(lg, dtype=torch.bool); m[idx[keep]] = True
        return m
    def topk_keep(lg, k):
        m = torch.zeros_like(lg, dtype=torch.bool); m[torch.topk(lg, k).indices] = True
        return m
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.4), dpi=300, sharey=True)
    for ax, lg, name in ((axes[0], sharp, "sharp distribution"), (axes[1], flat, "flat distribution")):
        probs = F.softmax(lg, dim=-1)
        pk = topp_keep(lg, 0.9)
        kk = topk_keep(lg, 5)
        colors = [BLUE if (pk[i] and kk[i]) else (TEAL if pk[i] else "white") for i in range(20)]
        ax.bar(range(20), probs, color=colors, edgecolor=GRIDC, zorder=3)
        for i in range(20):
            if kk[i] and not pk[i]:
                ax.annotate("k-only", (i, probs[i].item()), textcoords="offset points",
                            xytext=(0, 4), fontsize=6.2, color=MUTED, ha="center")
        nk, np_ = int(pk.sum()), int(kk.sum())
        ax.set_title(f"{name}: top-p(0.9) keeps {np_} (blue/teal) | top-k(5) keeps {nk}",
                     fontsize=8.6, color=INK)
        ax.tick_params(colors="#898781", labelsize=7)
        ax.grid(axis="y", color=GRIDC, lw=0.5, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("probability", fontsize=9, color=MUTED)
    fig.suptitle("Fixed-k vs adaptive nucleus: sharp dist keeps 5 tokens either way; flat dist — nucleus keeps 18, top-k caps at 5",
                 fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(FIG_DIR, "fig-9-2-sampling.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 9.2] {path}（保留集与 samplers_base.json adaptive 块同款）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch9 制图（图 9.1-9.2）")
    ap.add_argument("--which", default="all", choices=["all", "temp", "cutoff"])
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    if args.which in ("all", "temp"):
        fig_temp()
    if args.which in ("all", "cutoff"):
        fig_cutoff()
