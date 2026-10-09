# fig_ch07.py —— Book5 ch7 制图：图 7.1 Stable LatentMoE 三处方分工 / 图 7.2 SiTU vs SwiGLU 限幅 /
#                图 7.3 三处方 ablation 三臂曲线（编号=正文出现顺序；slug=k3-lmoe 防跨册冲撞）
# 用途：图 7.1 = 7.1 节机制总图（Stable LatentMoE 数据流——路由支路 W↓→潜专家→聚合→RMSNorm→W↑
#         与全宽共享支路并排；两张失效模式处方笺挂在病灶上；全部张量框标形状；数值取 K3 Table 1
#         d=7168/ℓ=3584/E=896/k=16/m=3072/Ns=2）；
#       图 7.2 = 7.1 节处方二的标量响应（gate 支 SiTU softcap(x,4)·σ(x) vs SwiGLU silu(x)；
#         up 支 softcap(x,25) vs 线性——远端封顶 ±4/±25、|f|≤β1β2=100、近原点一阶贴合）；
#       图 7.3 = 7.6 节三臂 ablation（喂 log/book5-ch07/train_ablate_{full,nositu,noqb}_fast.json：
#         左=loss 三臂；右=load_max/min 轨迹 log 轴——noqb 642 vs full 2.7 的「治塌缩」读数）。
# 所属章节：Book5 第 7 章（图 7.1-7.3）。机制主图从零原创（K3 报告图 CC BY-NC-ND 禁用）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch07/fig_ch07.py"
#           [--which all|rx|situ|ablate]
# 产物：figures/fig-7-{1,2,3}-k3-lmoe.png
# 图规：300dpi 白底；系列色 蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；线宽 2；图内英文、图注中文。
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "book5-frontier-hybrid-architectures", "figures")
LOG_DIR = os.path.join(REPO_ROOT, "log", "book5-ch07")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e1e0d9"
LMOE_FC = "#e9f8f1"     # 潜空间区浅青底
FAIL_C = "#b3261e"      # 失效模式标注（深红，与系列色区分）


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=8.0, lw=1.5, tc=INK, style="round"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"{style},pad=0.02", ec=ec, fc=fc,
                                lw=lw, mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc,
            zorder=4, linespacing=1.35)
    return x + w / 2, y + h / 2


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.2, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=lw, linestyle=ls, zorder=2))


def plain(ax, x0, y0, x1, y1, color=MUTED, lw=1.2, ls="-"):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=ls, zorder=2)


# ---------------- 图 7.1：Stable LatentMoE 三处方分工（失效模式↔处方对应） ----------------
def fig_rx():
    fig, ax = plt.subplots(figsize=(12.8, 7.0), dpi=300)
    ax.set_xlim(0, 14.6)
    ax.set_ylim(0, 8.9)
    ax.axis("off")
    ax.text(7.3, 8.62, "Stable LatentMoE (one of 92 MoE layers): two failure modes, three prescriptions",
            ha="center", fontsize=11, color=INK, weight="bold")
    ax.text(7.3, 8.3, "K3 Table 1: d=7168, ℓ=3584 (0.5×d), E=896 routed, k=16 active, m=3072, Ns=2 shared",
            ha="center", fontsize=7.5, color=MUTED)

    # 潜空间浅底区（路由支路中段）
    ax.add_patch(FancyBboxPatch((4.15, 2.35), 4.75, 3.0, boxstyle="round,pad=0.06",
                                ec=LMOE_FC, fc="#f6fdfa", lw=1.0, zorder=1))
    ax.text(6.5, 5.08, "latent space ℓ=3584", ha="center", fontsize=7.5, color=TEAL)

    # 路由车道（y≈4.0）：x → W↓ → E experts → u → RMSNorm → W↑ → ⊕ → y
    box(ax, 0.25, 3.5, 1.5, 1.0, "token x\n(d,)=(7168,)", ec=MUTED)
    box(ax, 2.3, 3.5, 1.6, 1.0, "W$^\\downarrow$\n(d,)→(ℓ,)", ec=BLUE)
    box(ax, 4.4, 2.55, 2.3, 2.35, "E=896 latent experts\n(ℓ,)→(m,)→(ℓ,)\nSiTU-GLU inside\n×16 active / token",
        ec=BLUE, fc="white", fs=7.8)
    box(ax, 7.1, 3.5, 1.55, 0.95, "aggregate u\nΣ p$_i$E$_i$(·), (ℓ,)", ec=BLUE, fs=7.5)
    box(ax, 9.05, 3.5, 1.6, 0.95, "RMSNorm(u)\n(ℓ,)→(ℓ,)", ec=TEAL, fs=7.8)
    box(ax, 11.0, 3.5, 1.35, 0.95, "W$^\\uparrow$\n(ℓ,)→(d,)", ec=BLUE)
    # ⊕ 与 y（右侧汇合）
    ax.add_patch(plt.Circle((13.0, 3.97), 0.14, ec=INK, fc="white", lw=1.4, zorder=3))
    ax.text(13.0, 3.97, "+", ha="center", va="center", fontsize=10, color=INK, zorder=4)
    box(ax, 13.35, 3.55, 1.1, 0.85, "y\n(d,)", ec=MUTED)
    # 共享车道（上方）：x 的旁路 → shared ×2 → ⊕
    box(ax, 10.7, 6.3, 3.0, 1.0, "shared experts ×2  E$^{sh}$(x)\n(d,)→(m$_s$,)→(d,)  full width",
        ec=MUTED, fs=7.8)
    plain(ax, 1.6, 4.5, 1.6, 5.5, ls=(0, (3, 2)))            # x → 共享车道（旁路，虚线）
    plain(ax, 1.6, 5.5, 10.7, 5.5, ls=(0, (3, 2)))
    arrow(ax, 10.7, 5.5, 10.7, 6.3, ls=(0, (3, 2)))
    arrow(ax, 12.2, 6.3, 12.2, 5.55)                         # shared → ⊕（下折、右行、再下）
    plain(ax, 12.2, 5.55, 13.0, 5.55)
    arrow(ax, 13.0, 5.55, 13.0, 4.13)
    arrow(ax, 12.35, 3.97, 12.86, 3.97)                      # W↑ → ⊕
    arrow(ax, 13.14, 3.97, 13.35, 3.97)                      # ⊕ → y

    arrow(ax, 1.75, 4.0, 2.3, 4.0)
    arrow(ax, 3.9, 4.0, 4.4, 3.9)
    arrow(ax, 6.7, 3.9, 7.1, 3.95)
    arrow(ax, 8.65, 3.95, 9.05, 3.95)
    arrow(ax, 10.65, 3.95, 11.0, 3.95)

    # 路由器 + 处方③（下方）
    box(ax, 0.25, 1.15, 2.6, 1.5, "router  s=σ(W$_r$x)\n(N,E)=(N,896)\nTop-k on s+b",
        ec=ORANGE, fs=7.8)
    arrow(ax, 1.0, 3.5, 1.0, 2.65)
    box(ax, 3.15, 1.15, 2.7, 1.5, "Rx③ Quantile Balancing\nb ← −quantile$_{1-k/n}$(s−α)\n(E,) per step",
        ec=ORANGE, fc="#fdf1ec", fs=7.5)
    arrow(ax, 2.85, 1.9, 3.15, 1.9, color=ORANGE)
    plain(ax, 5.85, 1.9, 5.85, 2.9, color=ORANGE, ls=(0, (3, 2)))
    arrow(ax, 5.85, 2.9, 5.2, 2.9, color=ORANGE, ls=(0, (3, 2)))

    # 失效模式处方笺①②（上方）
    box(ax, 2.1, 7.3, 6.6, 1.05,
        "Failure 1 — exploding internal activations:\n"
        "W$^\\downarrow$ → expert FFN → W$^\\uparrow$ = chain of ~4 matrix multiplies at 2.8T scale",
        ec=FAIL_C, fs=7.5)
    box(ax, 3.0, 5.85, 3.3, 1.0, "Rx② SiTU-GLU  |f|≤β$_1$β$_2$=100\n(β$_1$=4, β$_2$=25, soft caps)",
        ec=TEAL, fc="#e9f8f1", fs=7.5)
    box(ax, 7.3, 5.85, 3.1, 1.0, "Rx① RMSNorm between agg & W$^\\uparrow$\n(kill scale drift)",
        ec=TEAL, fc="#e9f8f1", fs=7.5)
    arrow(ax, 4.6, 7.3, 4.6, 6.85, color=FAIL_C)             # Failure1 → Rx②
    plain(ax, 7.1, 7.3, 7.1, 7.05, color=FAIL_C)
    arrow(ax, 7.1, 7.05, 8.4, 6.85, color=FAIL_C)            # Failure1 → Rx①
    arrow(ax, 4.6, 5.85, 5.0, 4.9, color=TEAL)               # Rx② → 专家群
    arrow(ax, 8.85, 5.85, 9.85, 4.45, color=TEAL)            # Rx① → RMSNorm
    box(ax, 0.25, 6.1, 1.6, 1.45, "Failure 2\nload balancing\nbreaks at ~10$^3$\nexperts",
        ec=FAIL_C, fs=7.5)
    plain(ax, 0.9, 6.1, 0.9, 2.65, color=FAIL_C, ls=(0, (3, 2)))

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-7-1-k3-lmoe-rx.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[图 7.1] {out}")


# ---------------- 图 7.2：SiTU-GLU vs SwiGLU 标量响应（处方二的限幅） ----------------
def fig_situ():
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    x = np.linspace(-10, 30, 801)

    # gate 支：silu(x)（无界） vs softcap(x,4)·sigmoid(x)（界 ±4）
    ax = axes[0]
    ax.plot(x, x / (1 + np.exp(-x)), color=ORANGE, lw=2, label="SwiGLU gate: silu(x)  (unbounded)")
    ax.plot(x, 4.0 * np.tanh(x / 4.0) / (1 + np.exp(-x)), color=BLUE, lw=2,
            label="SiTU gate: 4·tanh(x/4)·σ(x)")
    ax.axhline(4, color=BLUE, lw=1, ls=(0, (4, 3)))
    ax.axhline(0, color=GRIDC, lw=0.8)
    ax.set_title("gate branch (β$_1$=4)", fontsize=10, color=INK)
    ax.set_xlabel("pre-activation z", fontsize=9, color=MUTED)
    ax.set_ylabel("gate factor", fontsize=9, color=MUTED)

    # up 支：z（无界线性） vs softcap(x,25)（界 ±25）
    ax = axes[1]
    ax.plot(x, x, color=ORANGE, lw=2, label="SwiGLU up: z  (unbounded)")
    ax.plot(x, 25.0 * np.tanh(x / 25.0), color=BLUE, lw=2, label="SiTU up: 25·tanh(z/25)")
    ax.axhline(25, color=BLUE, lw=1, ls=(0, (4, 3)))
    ax.axhline(0, color=GRIDC, lw=0.8)
    ax.set_title("up branch (β$_2$=25)", fontsize=10, color=INK)
    ax.set_xlabel("pre-activation z", fontsize=9, color=MUTED)
    ax.set_ylabel("up factor", fontsize=9, color=MUTED)

    for ax in axes:
        ax.grid(True, color=GRIDC, lw=0.5, zorder=0)
        ax.tick_params(colors="#898781", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRIDC)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("SiTU-GLU vs SwiGLU: first-order match near 0, |SiTU-GLU| ≤ β$_1$·β$_2$ = 100",
                 fontsize=10.5, color=INK)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-7-2-k3-lmoe-situ.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[图 7.2] {out}")


# ---------------- 图 7.3：三处方 ablation 三臂（loss + 负载不均衡轨迹） ----------------
def fig_ablate():
    arms = {"full": (BLUE, "full (Rx①+②+③)"), "nositu": (ORANGE, "nositu (w/o Rx② SiTU)"),
            "noqb": (TEAL, "noqb (w/o Rx③ QB)")}
    data = {}
    for arm in arms:
        with open(os.path.join(LOG_DIR, f"train_ablate_{arm}_fast.json")) as f:
            data[arm] = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300)

    ax = axes[0]
    for arm, (c, lab) in arms.items():
        ax.plot(range(1, len(data[arm]["losses"]) + 1), data[arm]["losses"], color=c, lw=2, label=lab)
    ax.set_title("training loss (300 steps)", fontsize=10, color=INK)
    ax.set_xlabel("step", fontsize=9, color=MUTED)
    ax.set_ylabel("loss (nat)", fontsize=9, color=MUTED)

    ax = axes[1]
    for arm, (c, lab) in arms.items():
        steps = [s["step"] for s in data[arm]["stats_trace"]]
        ratio = [s["load_max_over_min"] for s in data[arm]["stats_trace"]]
        ax.plot(steps, ratio, color=c, lw=2, label=lab)
    ax.set_yscale("log")
    ax.set_title("load imbalance: max/min load per expert", fontsize=10, color=INK)
    ax.set_xlabel("step", fontsize=9, color=MUTED)
    ax.set_ylabel("max/min (log scale)", fontsize=9, color=MUTED)
    ax.annotate("642 vs 2.7\n(2nd-half mean)", xy=(255, 642), xytext=(120, 120),
                fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0))

    for ax in axes:
        ax.grid(True, color=GRIDC, lw=0.5, zorder=0)
        ax.tick_params(colors="#898781", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRIDC)
        ax.legend(fontsize=7.5, frameon=False)
    fig.suptitle("Three-prescription ablation on the K3 toy (44M, 300 steps, seed 20261002)",
                 fontsize=10.5, color=INK)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-7-3-k3-lmoe-ablate.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[图 7.3] {out}")


def main():
    ap = argparse.ArgumentParser(description="Book5 ch7 制图（图 7.1-7.3）")
    ap.add_argument("--which", choices=["all", "rx", "situ", "ablate"], default="all")
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    if args.which in ("all", "rx"):
        fig_rx()
    if args.which in ("all", "situ"):
        fig_situ()
    if args.which in ("all", "ablate"):
        fig_ablate()


if __name__ == "__main__":
    main()
