# make_figures.py —— Book5 ch6 四张图：图 6.1 K3 注意力侧结构 / 图 6.2 KDA 递归三步 /
#                     图 6.3 Gated MLA=Book4 MLA+两件新零件 / 图 6.4 玩具遗忘门分布
# 用途：图 6.1/6.2/6.3 为自绘示意级框线图（张量框与关键箭头标形状——K3 报告图 CC BY-NC-ND
#       禁用，机制主图一律从零原创，不描摹报告 Figure 2/3）；图 6.4 读
#       log/book5-ch06/train_k3toy_kda_{fast,full}.json 的 g_stats 画 per-channel 遗忘门
#       g∈(−5,0) 的逐层分布直方图（800 步）与 std-across-channels 的 300/800 步对照
#       （「每通道学出了不同的遗忘速度」的第一手读数）。
# 所属章节：Book5 第 6 章 §6.1（图 6.1）/§6.2（图 6.2）/§6.3（图 6.3）/§6.7（图 6.4）
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch06/make_figures.py"
# 产物：figures/fig-6-{1,2,3,4}-k3-attn.png（300dpi，本章专属 slug k3-attn）
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIG_DIR = os.path.join(REPO, "book5-frontier-hybrid-architectures", "figures")
LOG_FULL = os.path.join(REPO, "log", "book5-ch06", "train_k3toy_kda_full.json")
LOG_FAST = os.path.join(REPO, "log", "book5-ch06", "train_k3toy_kda_fast.json")
os.makedirs(FIG_DIR, exist_ok=True)

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MUTED = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
TEAL_L, ORANGE_L, GRAY_L = "#d8efe7", "#fde4d8", "#efeeea"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": TICK, "axes.labelcolor": DARK,
    "xtick.color": TICK, "ytick.color": TICK, "text.color": DARK,
    "axes.linewidth": 0.8, "font.family": "DejaVu Sans",
})


def box(ax, x, y, w, h, label, fc="white", ec=DARK, lw=1.0, fs=7.0, tc=None, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc or DARK)


def note(ax, x, y, s, fs=6.4, color=MUTED, ha="left", weight="normal"):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va="center", weight=weight)


def arrow(ax, x1, y1, x2, y2, color=DARK, lw=0.9, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                  color=color, lw=lw, linestyle=ls,
                                  connectionstyle=f"arc3,rad={rad}"))


# ---------------- 图 6.1 K3 注意力侧结构（3:1 层带 + KDA/Gated MLA 插槽放大；自绘示意级） ----------------
def fig_6_1():
    fig, ax = plt.subplots(figsize=(9.0, 5.8), dpi=300)
    ax.set_xlim(0, 132)
    ax.set_ylim(0, 86)
    ax.axis("off")

    # ---- 左：93 层层带（1-indexed：层 4,8,...,92 为 MLA；93 为末位额外 MLA；其余 KDA） ----
    x0, y0, w = 2.0, 6.0, 10.0
    L, ih = 93, 0.73
    for i in range(1, L + 1):
        is_mla = (i % 4 == 0) or (i == L)
        yy = y0 + (L - i) * ih
        ax.add_patch(Rectangle((x0, yy), w, ih * 0.84, fc=ORANGE_L if is_mla else TEAL_L,
                               ec=ORANGE if is_mla else TEAL, lw=0.4))
    ax.add_patch(Rectangle((x0 - 0.3, y0 - 0.5), w + 0.6, L * ih + 1.0, fill=False, ec=MUTED, lw=0.6))
    for i in (1, 5, 89, 93):
        yy = y0 + (L - i) * ih + ih * 0.42
        ax.text(x0 - 0.5, yy, str(i), ha="right", va="center", fontsize=6.2, color=MUTED)
    note(ax, x0 + w / 2, y0 + L * ih + 2.4, "93 layers", fs=7.4, ha="center", weight="bold")
    note(ax, x0 + w / 2, y0 + L * ih + 1.2, "(1-indexed)", fs=6.0, ha="center")
    # block 框（最末 4 层：KDA×3+MLA）与末层标注
    ax.add_patch(Rectangle((x0 - 0.55, y0 - 0.15), w + 1.1, 4 * ih + 0.15,
                           fill=False, ec=DARK, lw=0.9, ls=(0, (3, 2))))
    note(ax, x0 - 0.55, y0 - 1.3, "one block = KDA×3 + MLA (3:1)", fs=6.2, color=DARK, weight="bold")
    # AttnRes 12 层节律（层带右侧蓝刻度）
    for b in range(1, 9):
        bot = 12 * (b - 1) + 1 if b > 1 else 1
        top = 12 * b if b < 8 else 93
        ax.add_patch(Rectangle((x0 + w + 0.35, y0 + (L - top) * ih - 0.1), 0.55,
                               (top - bot + 1) * ih + 0.1, fill=False, ec=BLUE, lw=0.5))
    note(ax, x0 + w + 1.1, y0 + 0.4, "AttnRes\n12-layer\nblocks (x8)", fs=6.0, color=BLUE)
    note(ax, x0 - 0.3, 3.0, "two rhythms, orthogonal:\n3:1 attention mix / 12-layer AttnRes", fs=6.0)

    # ---- 中：KDA 插槽放大（纵向数据流） ----
    kx, ky, kw = 22, 8.5, 51.0
    ax.add_patch(FancyBboxPatch((kx, ky), kw, 72.5, boxstyle="round,pad=0.02",
                                fc="white", ec=TEAL, lw=1.3))
    note(ax, kx + kw / 2, ky + 69.6, "KDA layer (x69) — fixed state", fs=7.8, weight="bold", ha="center")
    cx = kx + kw / 2
    box(ax, cx - 8, ky + 62.5, 16, 5.2, "$x_t$  $(B,n,7168)$", fs=6.6)
    # 第二层：qkv 通路 + 遗忘门（并排）
    box(ax, kx + 2.5, ky + 48.5, 26, 9.5, "$W_{q/k/v}$+ShortConv(4)+Swish\n"
        "+L2Norm(q,k)\n$q,k\\,(B,n,96,128)\\;v\\,(B,n,96,128)$", fs=5.8)
    box(ax, kx + 30.5, ky + 48.5, 18, 9.5, "forget gate (low-rank)\n$g=g_{\\min}\\mathrm{Sigmoid}(e^{A^h}W^{\\uparrow}W^{\\downarrow}x)$\n"
        "$(B,n,96,128)\\in(-5,0)$ per-channel", fs=5.6, ec=ORANGE)
    # 状态三步
    box(ax, kx + 7.0, ky + 33.0, 37, 11.0, "recurrent state  $S_t\\;(B,96,128,128)$\n"
        "① forget $\\mathrm{Diag}(\\alpha_t)S$   ② retrieve $k_t^{\\top}S$\n"
        "③ write $k_t\\!\\otimes\\!\\beta_t(v_t\\!-\\!k_t^{\\top}S)$", fs=6.0, ec=TEAL, fc=TEAL_L)
    # 输出
    box(ax, kx + 7.0, ky + 22.5, 37, 7.0, "read $\\tilde o_t=S_t^{\\top}q_t$ + RMSNorm\n"
        "$\\to\\,(B,n,96,128)$", fs=6.2)
    box(ax, kx + 7.0, ky + 13.0, 37, 6.6, "output gate  $\\mathrm{Sigmoid}(W_gx_t)\\odot\\tilde o_t$\n"
        "$W_g$ full-rank (Eq. 6)", fs=6.2, ec=ORANGE)
    box(ax, kx + 13.0, ky + 4.5, 25, 5.4, "$W_o\\,(12288{\\times}7168)\\to y_t\\,(B,n,7168)$", fs=6.4)
    arrow(ax, cx, ky + 62.5, cx, ky + 58.0)
    arrow(ax, kx + 15.5, ky + 48.5, kx + 15.5, ky + 44.0)
    arrow(ax, kx + 39.5, ky + 48.5, kx + 39.5, ky + 44.0, color=ORANGE)
    arrow(ax, cx, ky + 33.0, cx, ky + 29.5)
    arrow(ax, cx, ky + 22.5, cx, ky + 19.6)
    arrow(ax, cx, ky + 13.0, cx, ky + 9.9)
    note(ax, kx + 1.5, ky + 1.3, "state constant in $n$: memory replaces the KV cache", fs=6.2, color=MUTED)

    # ---- 右：Gated MLA 插槽放大（纵向数据流；灰=Book4 第 6 章已教，橙=两处新零件） ----
    mx, my, mw = 76, 8.5, 51.0
    ax.add_patch(FancyBboxPatch((mx, my), mw, 72.5, boxstyle="round,pad=0.02",
                                fc="white", ec=ORANGE, lw=1.3))
    note(ax, mx + mw / 2, my + 69.6, "Gated MLA layer (x24) = Book4 MLA + 2 new parts", fs=7.4, weight="bold", ha="center")
    mcx = mx + mw / 2
    box(ax, mcx - 8, my + 62.5, 16, 5.2, "$x_t$  $(B,n,7168)$", fs=6.6)
    box(ax, mx + 13.0, my + 53.5, 25, 6.4, "$W^{DKV}$  down-proj\n$(B,n,7168)\\to(B,n,576)$", fs=6.2, fc=GRAY_L)
    box(ax, mx + 8.0, my + 43.5, 35, 7.4, "cache  576 elem/token/layer\n$c^{KV}_t\\,(512)$ + shared key slot $(64)$", fs=6.2, fc=GRAY_L)
    note(ax, mx + 44.5, my + 45.2, "NEW ① NoPE:\nslot kept,\nnever rotated", fs=5.8, color=ORANGE, weight="bold")
    box(ax, mx + 13.0, my + 34.0, 25, 6.4, "$W^{UK}\\!/W^{UV}$  two readouts\n$\\to(B,n,96,128)$", fs=6.0, fc=GRAY_L)
    box(ax, mx + 8.0, my + 24.5, 35, 6.6, "softmax attention (global,\nfull context $\\leq n$)", fs=6.2, fc=GRAY_L)
    box(ax, mx + 8.0, my + 14.5, 35, 7.0, "NEW ②  $\\mathrm{Sigmoid}(W_gx_t)\\odot\\tilde o_t$\n"
        "$W_g\\,(7168{\\times}7168)$=51.4M  (Eq. 7)", fs=6.0, ec=ORANGE, fc=ORANGE_L)
    box(ax, mx + 13.0, my + 5.5, 25, 5.8, "$W_o\\,(7168,12288)\\to y_t\\,(B,n,7168)$", fs=6.2)
    arrow(ax, mcx, my + 62.5, mcx, my + 59.9)
    arrow(ax, mcx, my + 53.5, mcx, my + 50.9)
    arrow(ax, mcx, my + 43.5, mcx, my + 40.4)
    arrow(ax, mcx, my + 34.0, mcx, my + 31.1)
    arrow(ax, mcx, my + 24.5, mcx, my + 21.5)
    arrow(ax, mcx, my + 14.5, mcx, my + 11.3)
    note(ax, mx + 1.5, my + 1.3, "gray = Book4 ch6 unchanged;  cache grows $\\propto n$", fs=6.2, color=MUTED)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-6-1-k3-attn.png"), dpi=300)
    plt.close(fig)


# ---------------- 图 6.2 KDA 递归三步（遗忘→对齐检索→写差值；自绘示意级，标形状） ----------------
def fig_6_2():
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=300)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 64)
    ax.axis("off")

    note(ax, 2, 61.5, "single head, token $t$:  per-channel gate $g^h_t\\in(g_{\\min},0)^{d_k}$,  "
          "$\\alpha_t=\\exp(g_t)\\in(e^{-5},1)^{d_k}$,  $S\\in\\mathbb{R}^{d_k\\times d_v}$", fs=7.6, weight="bold")

    # 列 1：状态矩阵 S（d_k 行 = 键通道 × d_v 列）
    sx, sy, sw, sh = 6, 17, 24, 30
    ax.add_patch(Rectangle((sx, sy), sw, sh, fc=TEAL_L, ec=TEAL, lw=1.1))
    for r in range(6):
        ax.add_patch(Rectangle((sx, sy + sh - (r + 1) * sh / 6), sw, sh / 6, fill=False, ec=TEAL, lw=0.3))
    ax.text(sx + sw / 2, sy + sh / 2, "$S_{t-1}$", ha="center", va="center", fontsize=11)
    note(ax, sx + sw / 2, sy + sh + 3.4, "state  $(B,h,d_k,d_v)$", fs=6.8, ha="center", weight="bold")
    note(ax, sx + sw / 2, sy + sh + 1.6, "e.g. $(1,1,128,128)$", fs=6.2, ha="center")
    note(ax, sx + sw / 2, sy - 2.2, "rows = key channels $c$", fs=6.2, ha="center")

    # 列 2：遗忘（每行一条 α）
    a1x = 34
    note(ax, a1x, 52.0, "step 1  forget", fs=7.2, weight="bold")
    note(ax, a1x, 50.2, "$S\\!\\leftarrow\\!\\mathrm{Diag}(\\alpha_t)S$", fs=6.6)
    alphas = ["0.90", "0.98", "0.10", "0.50", "0.95", "0.20"]
    for r, a in enumerate(alphas):
        yy = sy + sh - (r + 0.5) * sh / 6
        ax.add_patch(Rectangle((a1x, yy - 0.9), 5.0, 1.8, fc=ORANGE_L, ec=ORANGE, lw=0.5))
        ax.text(a1x + 6.0, yy, a, ha="left", va="center", fontsize=5.6, color=DARK)
        arrow(ax, sx + sw, yy, a1x, yy, color=ORANGE, lw=0.6)
    note(ax, a1x, 13.6, "one decay rate per key channel\n(GDN: one scalar per head)", fs=6.0, color=ORANGE)

    # 列 3：对齐检索
    b2x = 54
    note(ax, b2x, 52.0, "step 2  retrieve", fs=7.2, weight="bold")
    note(ax, b2x, 50.2, "$kv^{mem}_t=S^{\\top}k_t$", fs=6.6)
    box(ax, b2x, 38.5, 18.0, 6.2, "$k_t\\,(d_k)$\nL2-normalized", fs=6.0)
    box(ax, b2x, 27.0, 18.0, 6.2, "$kv^{mem}_t\\,(d_v)$", fs=6.6)
    arrow(ax, b2x + 9.0, 38.5, b2x + 9.0, 33.2)
    note(ax, b2x, 21.5, "what memory already\nholds at key $k_t$", fs=6.0, color=MUTED)

    # 列 4：写差值
    b3x = 78
    note(ax, b3x, 52.0, "step 3  write $\\delta$", fs=7.2, weight="bold")
    note(ax, b3x, 50.2, "$S\\!\\leftarrow\\!S+k_t\\!\\otimes\\!\\delta_t$", fs=6.6)
    box(ax, b3x, 38.5, 21.0, 6.2, "$\\delta_t=\\beta_t(v_t-kv^{mem}_t)$\n$(d_v)$,  $\\beta_t\\!\\in\\!(0,1)$", fs=5.8)
    box(ax, b3x + 2.5, 27.0, 16.0, 6.2, "$S_t$ updated", fs=6.8, fc=TEAL_L, ec=TEAL)
    arrow(ax, b3x + 10.5, 38.5, b3x + 10.5, 33.2)
    note(ax, b3x, 21.5, "write the residual,\nnot the value", fs=6.0, color=MUTED)

    # 列 5：读出
    r1x = 105
    note(ax, r1x, 52.0, "read out", fs=7.2, weight="bold")
    note(ax, r1x, 50.2, "$\\tilde o_t=S_t^{\\top}q_t$", fs=6.6)
    box(ax, r1x, 38.5, 18.0, 6.2, "$q_t\\,(d_k)$\nL2-normalized", fs=6.0)
    box(ax, r1x, 27.0, 18.0, 6.2, "$\\tilde o_t\\,(d_v)$", fs=6.6)
    arrow(ax, r1x + 9.0, 38.5, r1x + 9.0, 33.2)
    note(ax, r1x, 21.5, "then RMSNorm +\n$\\mathrm{Sigmoid}(W_gx_t)\\odot$ + $W_o$", fs=6.0, color=MUTED)

    # 列间主流向箭头
    arrow(ax, a1x + 12.0, 30.0, b2x, 30.0, color=DARK, lw=0.9)
    arrow(ax, b2x + 18.0, 30.0, b3x, 30.0, color=DARK, lw=0.9)
    arrow(ax, b3x + 21.0, 30.0, r1x, 30.0, color=DARK, lw=0.9)
    note(ax, 6, 9.0, "delta rule = overwrite-what-you-re-read (ch5);  KDA adds: each key channel forgets at its own rate", fs=6.8, color=DARK)
    note(ax, 6, 5.8, "$H=96$ heads run in parallel, each with its own $(128,128)$ state;  batch dim $B$ on all tensors", fs=6.2, color=MUTED)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-6-2-k3-attn.png"), dpi=300)
    plt.close(fig)


# ---------------- 图 6.3 Gated MLA = Book4 MLA + 两件新零件（diff 图；自绘示意级） ----------------
def fig_6_3():
    fig, ax = plt.subplots(figsize=(8.6, 4.0), dpi=300)
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 46)
    ax.axis("off")

    y1, h1 = 26, 10.0
    box(ax, 2, y1, 12.0, h1, "$x_t$\n$(B,n,7168)$", fs=6.6)
    box(ax, 17, y1, 16.5, h1, "$W^{DKV}$\ndown-proj\n$(576,7168)$", fs=6.2, fc=GRAY_L)
    box(ax, 37, y1, 20.0, h1, "latent box $c^{KV}_t$\n$(B,n,512)$ + 64-dim\nshared key slot", fs=6.0, fc=GRAY_L)
    box(ax, 61, y1, 19.0, h1, "$W^{UK}\\!/W^{UV}$\ntwo readouts\n$(B,n,96,128)$", fs=6.0, fc=GRAY_L)
    box(ax, 84, y1, 19.0, h1, "softmax attention\n(global over $\\leq n$ tokens)", fs=6.0, fc=GRAY_L)
    y2, h2 = 8.0, 10.0
    box(ax, 84, y2, 19.0, h2, "$\\tilde o_t$\n$(B,n,12288)$", fs=6.4, fc=GRAY_L)
    box(ax, 61, y2, 19.0, h2, "$\\odot\\,\\mathrm{Sigmoid}(W_gx_t)$\ninput-dep. channel gate\n$W_g$: $7168{\\times}7168$=51.4M", fs=5.8, fc=ORANGE_L, ec=ORANGE)
    box(ax, 37, y2, 20.0, h2, "$W_o$\n$(7168,12288)$", fs=6.2, fc=GRAY_L)
    box(ax, 17, y2, 16.5, h2, "$y_t$\n$(B,n,7168)$", fs=6.6)
    for a, b in ((14.0, 17.0), (33.5, 37.0), (57.0, 61.0), (80.0, 84.0)):
        arrow(ax, a, y1 + h1 / 2, b, y1 + h1 / 2)
    arrow(ax, 93.5, y1, 93.5, y2 + h2)
    arrow(ax, 84, y2 + h2 / 2, 80.0, y2 + h2 / 2)
    arrow(ax, 61, y2 + h2 / 2, 57.0, y2 + h2 / 2)
    arrow(ax, 37, y2 + h2 / 2, 33.5, y2 + h2 / 2)
    arrow(ax, 17, y2 + h2 / 2, 14.0, y2 + h2 / 2)

    note(ax, 2, 41.5, "Gated MLA = Book4 ch6 MLA (gray, unchanged) + two new parts (orange)", fs=7.6, weight="bold")
    # diff ①：NoPE
    ax.add_patch(Rectangle((43.5, y1 - 4.6), 14.0, 3.6, fc=ORANGE_L, ec=ORANGE, lw=0.8))
    note(ax, 44.2, y1 - 2.8, "NEW ① NoPE: no rotation", fs=6.2, color=DARK, weight="bold")
    note(ax, 43.5, y1 - 8.2, "64-dim slot stays in the cache but is\nnever rotated ($mla\\_use\\_nope{=}true$):\nthe rope lane of Book4 Fig is torn out", fs=6.0)
    arrow(ax, 50.5, y1 - 4.6, 47.0, y1, color=ORANGE)
    # diff ②：输出门
    note(ax, 59.5, y2 + h2 + 2.2, "NEW ② full-rank output gate (Eq. 7)", fs=6.2, weight="bold", color=DARK)
    note(ax, 2, 3.0, "KDA output gate (Eq. 6) is the same form - one gate recipe, both slots; gray path = Book4 ch6 6.3/6.5", fs=6.2, color=MUTED)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-6-3-k3-attn.png"), dpi=300)
    plt.close(fig)


# ---------------- 图 6.4 玩具遗忘门分布（自产：train_k3toy_kda_{fast,full}.json 的 g_stats） ----------------
def fig_6_4():
    full = json.load(open(LOG_FULL))
    fast = json.load(open(LOG_FAST))
    gs_f = full["g_stats"]
    std_800 = [s["per_channel_std_across_channels"] for s in gs_f]
    std_300 = [s["per_channel_std_across_channels"] for s in fast["g_stats"]]
    layers = [s["layer"] for s in gs_f]

    fig = plt.figure(figsize=(8.0, 4.4), dpi=300)
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.25], hspace=0.55, wspace=0.45)

    # (a) 800 步逐层直方图（6 个小面板）
    for i, s in enumerate(gs_f):
        axh = fig.add_subplot(gs[i // 3, i % 3])
        centers = [(a + b) / 2 for a, b in zip(s["bin_edges"][:-1], s["bin_edges"][1:])]
        widths = s["bin_edges"][1] - s["bin_edges"][0]
        dens = [c / max(1, sum(s["hist_counts"])) for c in s["hist_counts"]]
        axh.bar(centers, dens, width=widths * 0.95, color=TEAL, ec="none")
        axh.axvline(-2.5, color=MUTED, lw=0.8, ls=(0, (2, 2)))
        axh.set_xlim(-5, 0)
        axh.set_ylim(0, max(dens) * 1.25)
        axh.set_title(f"KDA layer {s['layer']}  std={std_800[i]:.2f}", fontsize=6.4, color=DARK)
        axh.tick_params(labelsize=6)
        axh.grid(True, color=GRID, lw=0.5)
        axh.set_axisbelow(True)
        if i // 3 == 1:
            axh.set_xlabel("$g$ (log-decay, init $\\approx-2.5$)", fontsize=6.2, color=MUTED)
        if i % 3 == 0:
            axh.set_ylabel("channel share", fontsize=6.2, color=MUTED)

    # (b) std-across-channels：300 vs 800 步
    axb = fig.add_subplot(gs[:, 3])
    xs = range(len(layers))
    w = 0.38
    axb.bar([x - w / 2 for x in xs], std_300, width=w, color=BLUE, ec="none", label="300 steps")
    axb.bar([x + w / 2 for x in xs], std_800, width=w, color=TEAL, ec="none", label="800 steps")
    axb.set_xticks(list(xs))
    axb.set_xticklabels([str(l) for l in layers], fontsize=6.2)
    axb.set_xlabel("KDA layer (toy: 6 of 8, 0-indexed)", fontsize=6.6)
    axb.set_ylabel("std of $g$ across channels", fontsize=6.6)
    axb.tick_params(labelsize=6.2)
    axb.grid(True, color=GRID, lw=0.5, axis="y")
    axb.set_axisbelow(True)
    axb.legend(fontsize=6.2, frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-6-4-k3-attn.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_6_1()
    fig_6_2()
    fig_6_3()
    fig_6_4()
    print("[完成] fig-6-1/6-2/6-3/6-4-k3-attn.png ->", FIG_DIR)
