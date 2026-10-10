# fig_ch04.py —— Book6 ch4 制图：图 4.1 静态 vs 连续批处理时间线 / 图 4.2 selective batching 分段 /
#                图 4.3 v1 vs v2 并发扫描曲线（编号=正文出现顺序；slug=continuous-batch）
# 用途：图 4.1 = 4.2/4.3 机制对照总图（依据 Orca Fig 2/3 机制自绘——版权⑦：Orca 图不拷贝只自绘；
#         左联静态批：先完成的请求空转陪跑到最长者结束；右联连续批：完成即出、腾位即入）；
#       图 4.2 = 4.3 selective batching 的分段视觉载体（对齐 Orca Fig 5 的算子分段：投影/FFN 段拼批、
#         attention 段各自算（Orca 正身）或 pad+mask 向量化（本册 v2 降档）——逐段标形状）；
#       图 4.3 = 4.6 的并发扫描证据（喂 log/book6-ch04/engine_v{1,2}_sweep_b{1,2,4,8}.json：
#         v1 吞吐随 B 近线性、v2 斜率缓——手写执行层 python 税随 B 涨；v1 各点标浪费位）。
# 所属章节：Book6 第 4 章（图 4.1-4.3）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/Book6-推理系统导论/ch04/fig_ch04.py"
#           [--which all|timeline|selective|sweep]
# 产物：drafts/Book6-推理系统导论/figures/fig-4-{1,2,3}-continuous-batch.png
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
SWEEP_DIR = os.path.join(REPO_ROOT, "log", "book6-ch04")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC, WASTEC = "#0b0b0b", "#52514e", "#e1e0d9", "#f3d3c8"


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=8.0, lw=1.5, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc, zorder=4)


# ---------------- 图 4.1：静态 vs 连续批处理时间线（双联，依据 Orca Fig 2/3 机制自绘） ----------------
def fig_timeline():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), dpi=300)
    lens = {"A": 4, "B": 6, "C": 8, "D": 10}          # 输出长度（prefill 同长）
    n_step = 10

    for col, (ax, title) in enumerate(zip(axes, ["static batching (request-level)",
                                                 "continuous batching (iteration-level)"])):
        ax.set_xlim(-1.6, n_step + 2.8)
        ax.set_ylim(-0.6, 5.4)
        ax.axis("off")
        ax.set_title(title, fontsize=10.5, color=INK, weight="bold")
        # prefill 块（step 0）
        for i, r in enumerate("ABCD"):
            ax.add_patch(Rectangle((-1.4, 4.2 - i * 0.85), 1.2, 0.6, fc="#dcebfc", ec=GRIDC, lw=0.8))
        ax.text(-0.8, 4.95, "prefill", fontsize=7.5, color=MUTED, ha="center")
        for i, r in enumerate("ABCD"):
            y = 4.2 - i * 0.85
            ax.text(-1.8, y + 0.3, f"req {r} (out={lens[r]})", fontsize=7.4, color=INK, ha="right", va="center")
            for s in range(1, n_step + 1):
                x = -0.1 + s * 1.0
                if col == 0:  # 静态：完成后空转（浪费位）
                    if s <= lens[r]:
                        ax.add_patch(Rectangle((x, y), 0.8, 0.6, fc="#dcebfc", ec=GRIDC, lw=0.8))
                    else:
                        ax.add_patch(Rectangle((x, y), 0.8, 0.6, fc=WASTEC, ec=GRIDC, lw=0.6, hatch="//"))
                else:         # 连续：完成后退出，新请求 E/F 腾位即入
                    if s <= lens[r]:
                        ax.add_patch(Rectangle((x, y), 0.8, 0.6, fc="#dcebfc", ec=GRIDC, lw=0.8))
            if col == 1:      # 新请求填空位：E 从 step5、F 从 step7（各生成到步末）
                y_e, y_f = 4.2 - 0 * 0.85, 4.2 - 1 * 0.85
                for s in range(5, n_step + 1):
                    ax.add_patch(Rectangle((-0.1 + s * 1.0, y_e), 0.8, 0.6, fc="#d9f2e7", ec=GRIDC, lw=0.8))
                for s in range(7, n_step + 1):
                    ax.add_patch(Rectangle((-0.1 + s * 1.0, y_f), 0.8, 0.6, fc="#d9f2e7", ec=GRIDC, lw=0.8))
                ax.text(-1.8, y_e + 0.3, "req E (arrives t=5)", fontsize=7.4, color=TEAL, ha="right", va="center")
                ax.text(-1.8, y_f + 0.3, "req F (arrives t=7)", fontsize=7.4, color=TEAL, ha="right", va="center")
        for s in range(1, n_step + 1):
            ax.text(-0.1 + s * 1.0 + 0.4, -0.35, f"t{s}", fontsize=6.8, color="#898781", ha="center")
        # 批组成注记
        if col == 0:
            ax.text(5.4, -0.05, "batch = ABCD for all 10 steps\nwasted slots = 4+2+... after each finishes",
                    fontsize=7.2, color=ORANGE, ha="center")
        else:
            ax.text(5.4, -0.05, "B per step: 4,4,4,4,4,4,3,3,3,3\nfinished exits, arrivals take slots",
                    fontsize=7.2, color=TEAL, ha="center")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-4-1-continuous-batch.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 4.1] {path}")


# ---------------- 图 4.2：selective batching 分段（对齐 Orca Fig 5 的算子分段；标形状） ----------------
def fig_selective():
    fig, ax = plt.subplots(figsize=(11.2, 4.6), dpi=300)
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    ax.text(6.6, 6.35, "Selective batching: one decode iteration, B=3 requests (KV lengths 4/2/3)",
            ha="center", fontsize=10.5, color=INK, weight="bold")

    # 主链（上排）：embed→qkv 投影（拼批）→attention（各自）→o_proj+FFN（拼批）
    box(ax, 0.2, 4.6, 1.6, 0.9, "input tokens\n(B,1)", ec=MUTED)
    box(ax, 2.3, 4.6, 1.7, 0.9, "embed\n(B,1,C)", ec=BLUE, fc="#dcebfc")
    box(ax, 4.5, 4.6, 2.2, 0.9, "q/k/v proj (batched)\n(B,1,C)→(B,h,1,d_k)", ec=BLUE, fc="#dcebfc")
    box(ax, 7.2, 4.6, 2.5, 0.9, "attention (per-request)\n(h,1,n_i)·(h,n_i,d_k)", ec=ORANGE, fc="#fde3d6")
    box(ax, 10.2, 4.6, 2.6, 0.9, "o_proj + FFN (batched)\n(B,1,C)→(B,1,C)", ec=BLUE, fc="#dcebfc")
    for x0, x1 in ((1.8, 2.3), (4.0, 4.5), (6.7, 7.2), (9.7, 10.2)):
        ax.annotate("", xy=(x1, 5.05), xytext=(x0, 5.05),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    ax.text(6.6, 4.15, "batched segments share one big matmul; attention reads each request's own KV",
            ha="center", fontsize=7.6, color=MUTED)

    # 下排左：Orca 正身（attention 逐请求循环）
    box(ax, 0.6, 1.6, 5.2, 2.0,
        "Orca original: loop over requests\nfor b in B:  att_b = softmax(q_b · K_b^T/√d_k) · V_b\n"
        "no padding, no mask — each (h,1,n_b) exact", ec=ORANGE, fc="white", fs=7.8)
    # 下排右：v2 教学降档（pad+mask 向量化）
    box(ax, 6.4, 1.6, 6.3, 2.0,
        "this book's v2 (vectorized): pad K to n_max\nkb (B,h,n_max,d_k); attn_mask (B,1,1,n_max)\n"
        "softmax masks pad positions — one big matmul", ec=TEAL, fc="white", fs=7.8)
    ax.annotate("", xy=(3.2, 3.65), xytext=(8.2, 4.55),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.0, linestyle="--"))
    ax.annotate("", xy=(9.5, 3.65), xytext=(8.6, 4.55),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.0, linestyle="--"))

    # 右下小嵌：pad+mask 的 3×n_max 现场小图
    ax.text(12.6, 1.15, "mask rows (n_max=4):\n1111 / 1100 / 1110", fontsize=7.0, color=MUTED, ha="center")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-4-2-continuous-batch.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 4.2] {path}")


# ---------------- 图 4.3：v1 vs v2 并发扫描（喂 sweep JSON） ----------------
def fig_sweep():
    data = {}
    for tag in ("v1", "v2"):
        pts = []
        for B in (1, 2, 4, 8):
            p = os.path.join(SWEEP_DIR, f"engine_{tag}_sweep_b{B}.json")
            if os.path.isfile(p):
                d = json.load(open(p))
                pts.append((B, d["throughput_tok_s"], d.get("wasted_slots", 0)))
        data[tag] = pts
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=300)
    for tag, color in (("v1", BLUE), ("v2", ORANGE)):
        pts = data[tag]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, marker="o" if tag == "v1" else "s",
                lw=2.2, ms=6, label=f"{tag}: " + ("naive full-refwd" if tag == "v1" else "continuous + KV"), zorder=4)
        if tag == "v1":
            for B, tp, w in pts:
                if w:
                    ax.annotate(f"waste {w}", (B, tp), textcoords="offset points", xytext=(6, -12),
                                fontsize=7.2, color=MUTED)
    ax.set_xlabel("concurrent requests B", fontsize=9, color=MUTED)
    ax.set_ylabel("throughput (tok/s)", fontsize=9, color=MUTED)
    ax.set_xticks([1, 2, 4, 8])
    ax.tick_params(colors="#898781", labelsize=8)
    ax.grid(color=GRIDC, lw=0.6, zorder=0)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("Batch scaling: v1 near-linear, v2 flatter (hand-written stepper's python tax)",
                 fontsize=9.5, color=INK)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-4-3-continuous-batch.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 4.3] {path}（数据源 engine_v{1,2}_sweep_b*.json）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch4 制图（图 4.1-4.3）")
    ap.add_argument("--which", default="all", choices=["all", "timeline", "selective", "sweep"])
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    if args.which in ("all", "timeline"):
        fig_timeline()
    if args.which in ("all", "selective"):
        fig_selective()
    if args.which in ("all", "sweep"):
        fig_sweep()
