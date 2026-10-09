# make_figures.py —— Book5 ch3 三张图：图 3.1 NSA 三分支数据流 / 图 3.2 NSA→DSA 减法 / 图 3.3 gap 阶梯
# 用途：图 3.1/3.2 为自绘示意级（框线图，张量框与关键箭头标形状——单 token 查询视角，(t,d_k) 记
#       「前文 t 个 token × d_k 维」）；图 3.3 读 log/book5-ch03/nsa_block_fast150.json 的
#       cpu_parity.approx_gap_rel_l2_by_ksel 画块选择预算 k_sel 的近似代价阶梯（log-y，含 k_sel=16
#       选全块=对拍地板一点）。
# 所属章节：Book5 第 3 章 §3.1（图 3.1）/§3.3（图 3.2）/§3.5（图 3.3）
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch03/make_figures.py"
# 产物：figures/fig-3-1-nsa-three-branch-dataflow.png、
#       fig-3-2-nsa-to-dsa-subtraction.png、fig-3-3-block-sparse-gap-ladder.png（300dpi，本章专属 slug）
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIG_DIR = os.path.join(REPO, "book5-frontier-hybrid-architectures", "figures")
LOG = os.path.join(REPO, "log", "book5-ch03", "nsa_block_fast150.json")
os.makedirs(FIG_DIR, exist_ok=True)

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MUTED = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": TICK, "axes.labelcolor": DARK,
    "xtick.color": TICK, "ytick.color": TICK, "text.color": DARK,
    "axes.linewidth": 0.8, "font.family": "DejaVu Sans",
})


def box(ax, x, y, w, h, label, fc="white", ec=DARK, lw=1.1, fs=7.4, tc=None, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc or DARK)


def note(ax, x, y, s, fs=6.8, color=MUTED, ha="left"):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va="center")


def arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.0, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


# ---------------- 图 3.1 NSA 三分支数据流（自绘示意级，标形状） ----------------
def fig_3_1():
    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 64)
    ax.axis("off")

    # 输入（单 token 查询视角）
    box(ax, 1, 18, 12.5, 30, "$q_t$  $(d_k,)$\n\n$k_{:t}$  $(t,d_k)$\n$v_{:t}$  $(t,d_v)$", fs=7.2)
    note(ax, 1.2, 15.6, "per-token view:\n$t$ = cached length", fs=6.6)

    # 三条泳道（标题在泳道左上，盒子从 x=18 起）
    lanes = [
        ("cmp  (compress)", 50, BLUE,
         [("block split $l$, stride $d$\nMLP $\\varphi$: $(t,d_k)\\to(\\lfloor t/d\\rfloor,\\,d_k)$", 24.0),
          ("Attn$(q_t,\\tilde K^{cmp},\\tilde V^{cmp})$\n$\\to (d_v,)$", 21.0)]),
        ("slc  (select)", 31, ORANGE,
         [("scores reused: $p^{cmp}_t$\n$(\\lfloor t/d\\rfloor,)$", 20.0),
          ("top-$n$ blocks $\\mathcal{I}_t$\n$(n,)$", 13.0),
          ("gather whole blocks\n$(n{\\cdot}l',\\,d_k)\\!\\to\\!$Attn$\\to(d_v,)$", 24.0)]),
        ("win  (window)", 12, TEAL,
         [("slice last $w$ tokens\n$(w,\\,d_k)$", 18.0),
          ("Attn$(q_t,k_{t-w:t},v_{t-w:t})$\n$\\to (d_v,)$", 21.0)]),
    ]
    for name, yc, color, boxes in lanes:
        ax.text(18, yc + 7.6, name, fontsize=8.4, color=color, ha="left", weight="bold")
        x = 18
        for label, w in boxes:
            box(ax, x, yc - 4.8, w, 9.6, label, ec=color, fs=6.4)
            x += w
            arrow(ax, x + 0.2, yc, x + 1.6, yc, color=color)
            x += 1.8
        # 汇聚到各自门控
        gx = 82.5
        arrow(ax, x - 1.6, yc, gx - 0.4, yc, color=color)
        box(ax, gx, yc - 3.6, 7.0, 7.2, "$g^c_t$\n$\\in[0,1]$", ec=color, fc="#f7f9fc", fs=7.0)
        arrow(ax, gx + 7.0, yc, 91.5, yc, color=color)
    arrow(ax, 91.5, 50, 92.8, 34.5, color=DARK)
    arrow(ax, 91.5, 31, 92.8, 31, color=DARK)
    arrow(ax, 91.5, 12, 92.8, 27.5, color=DARK)
    box(ax, 92.8, 26.0, 6.6, 10.0, "$o_t$\n$(d_v,)$", fc="#f7f9fc", fs=7.6)

    # 底部两行注
    ax.text(50, 4.6, r"gated sum:  $o_t=\sum_{c\in\{\mathrm{cmp,slc,win}\}} g_t^c\cdot\mathrm{Attn}(q_t,\tilde K_t^c,\tilde V_t^c)$   (Eq. 5)",
            fontsize=8.2, color=DARK, ha="center")
    note(ax, 50, 1.2, "decode-load account:  floor(t/d) + n*l' + w  tokens per step  (NSA Table 4)",
         fs=7.0, ha="center")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-3-1-nsa-three-branch-dataflow.png"))
    plt.close(fig)


# ---------------- 图 3.2 NSA→DSA 减法图（自绘示意级） ----------------
def fig_3_2():
    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58)
    ax.axis("off")

    # NSA 侧（左）
    box(ax, 2, 4, 35, 45, "", ec=BLUE, lw=1.6)
    ax.text(19.5, 52.5, "NSA  (2025-02 · research)", fontsize=8.8, color=BLUE, ha="center", weight="bold")
    for i, s in enumerate([
        "cmp:  MLP $\\varphi$ compression\nblock split, $d<l$ overlap",
        "slc:  block-level top-$n$\nscores reused from cmp",
        "win:  window branch $w{=}512$",
        "gated sum $g^c_t$, 3 branches",
        "GQA base · 64 heads / 4 groups",
        "pretrain from scratch · 270B tk",
    ]):
        box(ax, 3.5, 41.5 - i * 6.3, 32, 5.4, s, ec=BLUE, fs=6.5)

    # DSA 侧（右）
    box(ax, 63, 4, 35, 45, "", ec=ORANGE, lw=1.6)
    ax.text(80.5, 52.5, "DSA  (2025-12 · production)", fontsize=8.8, color=ORANGE, ha="center", weight="bold")
    for i, s in enumerate([
        "lightning indexer (Eq. 1)\n$I_{t,s}=\\sum_j w^I_{t,j}\\,\\mathrm{ReLU}(q^I_{t,j}\\cdot k^I_s)$",
        "token-level top-$k$ = 2048\nfine-grained selection",
        "KL-distilled indexer\n2.1B + 943.7B tokens",
        "MLA in MQA mode\nlatent = shared KV entry",
        "continued training on V3.1\n(no from-scratch cost)",
        "indexer still $O(L^2)$, FP8-cheap",
    ]):
        box(ax, 64.5, 41.5 - i * 6.3, 32, 5.4, s, ec=ORANGE, fs=6.5)

    # 中间减法区
    ax.text(50, 52.5, "subtraction,\nnot upgrade", fontsize=8.4, color=DARK, ha="center", weight="bold")
    for i, s in enumerate(["compression branch", "window branch", "gating $g^c_t$", "block granularity"]):
        box(ax, 39, 44.5 - i * 6.6, 22, 5.2, "$\\times$  " + s, ec=MUTED, fc="#f4f3ef", fs=6.9, tc=MUTED)
    arrow(ax, 37.6, 14.5, 62.4, 14.5, color=DARK, lw=1.7)
    ax.text(50, 10.8, "keep:  learned selection", fontsize=7.4, color=DARK, ha="center")
    note(ax, 50, 7.0, "hardware alignment shifts:\nblock-shared (GQA group) $\\to$ entry-shared (MQA)",
         fs=6.7, ha="center")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-3-2-nsa-to-dsa-subtraction.png"))
    plt.close(fig)


# ---------------- 图 3.3 块稀疏 gap 阶梯（自产，读正式 JSON） ----------------
def fig_3_3():
    with open(LOG, encoding="utf-8") as f:
        res = json.load(f)
    gaps = res["cpu_parity"]["approx_gap_rel_l2_by_ksel"]
    nb = res["cpu_parity"]["n_blocks"]
    parity = res["cpu_parity"]["sel_all_max_abs"]
    ks = sorted(int(k) for k in gaps)
    vals = [gaps[str(k)] for k in ks]

    fig, ax = plt.subplots(figsize=(6.4, 4), dpi=300)
    ax.plot(ks, vals, "-o", color=BLUE, lw=2, ms=6, label="rel-L2 gap vs full attention")
    ax.set_yscale("log")
    ax.set_xticks(ks)
    ax.set_xlabel("blocks selected per query block  ($k_{sel}$, of $n_b$=16)", fontsize=9)
    ax.set_ylabel("relative L2 gap  (log scale)", fontsize=9)
    ax.grid(True, which="both", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.annotate(f"select all {nb} blocks\n= parity floor (rel-L2 {vals[-1]:.1e}; max|Δ| {parity:.1e})",
                xy=(ks[-1], vals[-1]), xytext=(ks[-2] + 0.1, vals[-1] * 6),
                fontsize=7.4, color=MUTED,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))
    ax.text(0.035, 0.05, "CPU fp32, seed 20261002, n=1024, b=64 (16 blocks)", transform=ax.transAxes,
            fontsize=7.0, color=MUTED)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-3-3-block-sparse-gap-ladder.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_3_1()
    fig_3_2()
    fig_3_3()
    print("make_figures 完成：fig-3-1/3-2/3-3（slug=nsa-three-branch-dataflow / nsa-to-dsa-subtraction / block-sparse-gap-ladder）")
