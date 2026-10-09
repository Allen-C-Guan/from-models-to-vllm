# make_figures.py —— Book5 ch8 两张图：图 8.1 两条哲学对决图 / 图 8.2 partial RoPE 头维切分
# 用途：产出 figures/fig-8-{1,2}-<ch08 专属 slug longctx>.png（300dpi 印刷规格，
#       图内英文、图注中文在正文）。两图均为自绘示意级（本章无数据图——收束论章的图是概念图）：
#       图 8.1 左=外推派（波长谱拉伸：快慢正弦波隐喻承 Book3 ch7 表针图）/右=状态压缩派（固定状态
#             写入-遗忘-读出）/底=iRoPE 按层分工条带（Llama 4 的 [1,1,1,0] 层带+DSV4 按 rope 组分工）；
#       图 8.2 真机 256 维头（25%=64 维旋、75%=192 维恒等）+ 玩具 8 维小例（例 8.1 同构）+
#             源码两实现（rope_utils 零频率补齐 L246-261 / qwen3_next 截断式 L111,L186-188）。
# 所属章节：Book5 第 8 章（8.1 图 8.1 / 8.5 图 8.2；数字锚=各机 config 亲核，见 ch08 正文表 8.1）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch08/make_figures.py" [--only 1|2]
# 规格：写作规范 §4——蓝 #2a78d6 / 橙 #eb6834 / 青 #1baf7a / 黄 #eda100；线宽 2；网格 hairline #e1e0d9
#       置底层；刻度 #898781；标注 #0b0b0b / #52514e；禁双 y 轴；tight_layout（不用 bbox_inches='tight'）。
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIG_DIR = os.path.join(REPO, "book5-frontier-hybrid-architectures", "figures")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MID = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
FILL_BLUE, FILL_TEAL, FILL_ORANGE, FILL_YELLOW = "#cde2fb", "#c9f0e2", "#fbd9cb", "#fdeec9"
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9.5,
    "axes.edgecolor": MID, "text.color": DARK,
})


def new_ax(w, h, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w, h), dpi=300)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def box(ax, x, y, w, h, text, ec=BLUE, fc="white", lw=1.5, fs=7.0, tc=DARK, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.05,rounding_size=0.10",
                                linewidth=lw, edgecolor=ec, facecolor=fc,
                                linestyle=ls, zorder=3))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=4, linespacing=1.4)


def arrow(ax, p1, p2, color=MID, ls="-", lw=1.5):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=10,
                                 linewidth=lw, color=color, linestyle=ls,
                                 zorder=2, shrinkA=1, shrinkB=1))


# =====================================================================
# 图 8.1 两条哲学对决：外推拉宽坐标系 vs 状态自己学（+ iRoPE 分工底带）
# =====================================================================
def fig_1():
    fig, ax = new_ax(8.8, 5.0, 17.6, 10.0)

    ax.text(8.8, 9.72, "Two philosophies of long context: where does position live?",
            ha="center", va="center", fontsize=10, color=DARK)
    ax.text(8.8, 9.38, "one question asked of every 2026 flagship — is distance written into the coordinates, or carried by the state?",
            ha="center", va="center", fontsize=7.4, color=MID)

    # ---------------- 左：外推派（拉宽坐标系）----------------
    box(ax, 0.35, 2.45, 8.25, 6.55, "", ec=BLUE, lw=1.8, fc="#f5f9fe")
    ax.text(4.47, 8.66, "Extrapolation camp — position lives in the architecture",
            ha="center", va="center", fontsize=8.2, color=DARK, zorder=4)
    ax.text(4.47, 8.32, "operator untouched; re-sign the lease on the coordinate system (PI / NTK / YaRN, Book3 ch7)",
            ha="center", va="center", fontsize=6.5, color=MID, zorder=4)

    # 波长谱：快→慢四支正弦波（Book3 ch7「一排表针」的平面化——快针管近邻、慢针管远方）
    for k, cycles in enumerate((5.0, 2.6, 1.2, 0.55)):  # 每支波内周期数递减 = 波长变长
        xc = 1.15 + k * 1.80
        xs = np.linspace(xc, xc + 1.55, 240)
        ax.plot(xs, 7.15 + 0.42 * np.sin(2 * np.pi * cycles * (xs - xc) / 1.55),
                color=BLUE, lw=1.5, zorder=4)
    ax.text(1.15, 7.85, "fast hand (neighbors)", ha="left", fontsize=6.2, color=BLUE)
    ax.text(6.55, 7.85, "slow hand (far range)", ha="right", fontsize=6.2, color=BLUE)
    ax.text(4.47, 6.52, "the wavelength spectrum: training length decides which hands never finish a turn",
            ha="center", va="center", fontsize=6.4, color=MID)

    arrow(ax, (1.10, 6.10), (7.85, 6.10), color=BLUE, lw=2.0)
    ax.text(4.47, 6.26, "stretch $\\times s$ — rescale frequencies so slow hands cover the new range",
            ha="center", va="center", fontsize=6.5, color=BLUE)
    chips = [
        "Gemma3  PI$\\times$8 on global layers only $\\to$ 128K",
        "gpt-oss  YaRN 32$\\times$ on dense layers $\\to$ 131K",
        "Qwen3-Next  YaRN 4$\\times$ $\\to$ 1.01M (serving-side)",
        "DSV4  YaRN 16$\\times$ on compress rope only, 64K$\\to$1M",
    ]
    for k, txt in enumerate(chips):
        box(ax, 0.85, 5.62 - k * 0.42, 7.25, 0.36, txt, ec=BLUE, fc=FILL_BLUE, lw=0.9, fs=6.4)
    ax.text(4.47, 3.62, "rents: original-window resolution, short-task dips, per-window re-signing (Book3 ch7 four-intervention control)",
            ha="center", va="center", fontsize=6.2, color=MID)
    ax.text(4.47, 3.05, "position is explicit, inspectable, and stretchable — but it must be stretched",
            ha="center", va="center", fontsize=6.4, color=BLUE)

    # ---------------- 右：状态压缩派（位置留给数据与状态）----------------
    box(ax, 9.00, 2.45, 8.25, 6.55, "", ec=TEAL, lw=1.8, fc="#f6fcfa")
    ax.text(13.12, 8.66, "State-compression camp — position lives in data & state",
            ha="center", va="center", fontsize=8.2, color=DARK, zorder=4)
    ax.text(13.12, 8.32, "no positional parameters at all (NoPE); order emerges from the update itself",
            ha="center", va="center", fontsize=6.5, color=MID, zorder=4)

    # token 流入固定状态（写入→遗忘→读出）
    for k, lab in enumerate(("$t{-}2$", "$t{-}1$", "$t$")):
        box(ax, 9.55 + k * 0.95, 7.42, 0.80, 0.50, lab, ec=MID, fc="white", lw=1.0, fs=6.6)
        if k < 2:
            arrow(ax, (10.35 + k * 0.95, 7.67), (10.50 + k * 0.95, 7.67), color=MID, lw=1.1)
    arrow(ax, (11.90, 7.42), (11.90, 6.86), color=TEAL, lw=1.8)
    box(ax, 10.55, 5.78, 2.70, 1.02,
        "fixed state $S_t$\n$(B,\\,h,\\,d_k,\\,d_v)$ — never grows",
        ec=TEAL, fc=FILL_TEAL, fs=6.8)
    ax.text(13.50, 6.68, "write $\\to$ forget (per-channel decay)\n$\\to$ read — the update order IS the position",
            ha="left", va="center", fontsize=6.3, color=TEAL, linespacing=1.5)

    chips2 = [
        "K3: all-NoPE, zero rope fields in config, 1M direct",
        "Kimi Linear 48B: NoPE beats RoPE on RULER, 84.3 vs 78.8",
        "recursion order + short conv (4) + data-dependent decay",
        "bonus: NoPE frees MLA to absorb into pure MQA (ch6)",
    ]
    for k, txt in enumerate(chips2):
        box(ax, 9.50, 5.62 - k * 0.42, 7.25, 0.36, txt, ec=TEAL, fc=FILL_TEAL, lw=0.9, fs=6.4)
    ax.text(13.12, 3.62, "the curriculum still walks 8K$\\to$64K / 256K$\\to$1M — what is saved is the extrapolation chain, not long-context training",
            ha="center", va="center", fontsize=6.2, color=MID)
    ax.text(13.12, 3.05, "position is implicit and free of rescaling — but it must be learned, and probed to be seen",
            ha="center", va="center", fontsize=6.4, color=TEAL)

    # ---------------- 底带：iRoPE 的调和（按层分工）----------------
    box(ax, 0.35, 0.30, 16.90, 1.95, "", ec=ORANGE, lw=1.6, fc="#fffaf8")
    ax.text(0.70, 1.95, "iRoPE's reconciliation — not right vs wrong, but division of labor (Book3 ch9: Llama 4 config)",
            ha="left", va="center", fontsize=7.6, color=DARK, zorder=4)
    # 层带：[chunked, chunked, chunked, global] × 2（no_rope_layers=[1,1,1,0]）
    for k in range(8):
        x = 0.85 + k * 1.06
        glob = (k % 4 == 3)
        box(ax, x, 1.05, 0.96, 0.62,
            "global\nRoPE+ext." if glob else "chunked\nno rope",
            ec=ORANGE if glob else MID, fc=FILL_ORANGE if glob else "white",
            lw=1.4 if glob else 1.0, fs=5.8)
    ax.text(9.60, 1.36, "local layers need no absolute position (window / chunk 8192);\nglobal layers carry the stretched long-range coordinates —\nrotary coverage 131,072 vs declared 10M (Scout)",
            ha="left", va="center", fontsize=6.2, color=MID, linespacing=1.5)
    ax.text(0.85, 0.62, "same philosophy at finer grain in DSV4: sliding-window path keeps $\\theta{=}10^4$ unscaled; only the compress rope group carries YaRN16 (ch4, C-60)",
            ha="left", va="center", fontsize=6.2, color=MID)

    out = os.path.join(FIG_DIR, "fig-8-1-longctx-two-philosophies.png")
    fig.tight_layout()
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


# =====================================================================
# 图 8.2 partial RoPE 头维切分：25% 旋转 + 75% 恒等（+源码两实现对应）
# =====================================================================
def fig_2():
    fig, ax = new_ax(8.8, 4.4, 17.6, 8.8)

    ax.text(8.8, 8.52, "Partial RoPE: rotate the first 25% of the head, leave the rest a content-only pure land",
            ha="center", va="center", fontsize=9.6, color=DARK)
    ax.text(8.8, 8.18, "Qwen3-Next / Qwen3.5 global layers: head$_{dim}$=256, partial$_{rotary\\_factor}$=0.25 $\\to$ 64 dims rotated (32 freq pairs), 192 dims untouched",
            ha="center", va="center", fontsize=6.9, color=MID)

    # ---- 上：真机 256 维头切分条 ----
    x0, y0, wtot, hb = 1.30, 6.10, 9.60, 0.95
    w_rot = wtot * 64 / 256
    ax.add_patch(Rectangle((x0, y0), w_rot, hb, facecolor=FILL_BLUE, edgecolor=BLUE, lw=1.6, zorder=3))
    ax.add_patch(Rectangle((x0 + w_rot, y0), wtot - w_rot, hb, facecolor="white", edgecolor=BLUE, lw=1.6, zorder=3))
    ax.plot([x0 + w_rot, x0 + w_rot], [y0 - 0.06, y0 + hb + 0.06], color=BLUE, lw=1.6, zorder=4)
    ax.text(x0 + w_rot / 2, y0 + hb / 2, "rotated 64 dims\n(32 pairs, $\\theta{=}10^7$)",
            ha="center", va="center", fontsize=6.8, color=DARK, linespacing=1.4, zorder=5)
    ax.text(x0 + w_rot + (wtot - w_rot) / 2, y0 + hb / 2, "NoPE 192 dims — identity (content-only)",
            ha="center", va="center", fontsize=6.8, color=MID, zorder=5)
    ax.text(x0 + w_rot / 2, y0 - 0.32, "25%", ha="center", fontsize=6.4, color=BLUE)
    ax.text(x0 + w_rot + (wtot - w_rot) / 2, y0 - 0.32, "75%", ha="center", fontsize=6.4, color=MID)
    ax.text(1.30, y0 + hb + 0.30, "one query head $q\\in(B,\\,h,\\,n,\\,256)$",
            ha="left", fontsize=6.6, color=DARK)
    ax.text(11.35, y0 + hb + 0.30, "split: rotate $(B,h,n,64)$, pass through $(B,h,n,192)$",
            ha="left", fontsize=6.6, color=DARK)

    # ---- 中：玩具 8 维小例（例 8.1 同构）----
    xt, yt, wt_t, ht = 1.30, 4.55, 9.60, 0.75
    w2 = wt_t * 2 / 8
    ax.add_patch(Rectangle((xt, yt), w2, ht, facecolor=FILL_TEAL, edgecolor=TEAL, lw=1.4, zorder=3))
    ax.add_patch(Rectangle((xt + w2, yt), wt_t - w2, ht, facecolor="white", edgecolor=TEAL, lw=1.4, zorder=3))
    ax.plot([xt + w2, xt + w2], [yt - 0.05, yt + ht + 0.05], color=TEAL, lw=1.4, zorder=4)
    ax.text(xt + w2 / 2, yt + ht / 2, "2 rotated", ha="center", va="center", fontsize=6.6, color=DARK, zorder=5)
    ax.text(xt + w2 + (wt_t - w2) / 2, yt + ht / 2, "6 identity dims (toy of Ex. 8.1: $d_k{=}8$, $\\rho{=}0.25$)",
            ha="center", va="center", fontsize=6.6, color=MID, zorder=5)
    ax.text(11.35, yt + ht / 2, "same cut at heart-scale: pair 1 turns by $m\\omega$;\npairs 2-4 turn by 0 rad $\\equiv$ not rotated",
            ha="left", va="center", fontsize=6.4, color=MID, linespacing=1.5)

    # ---- 下：源码两实现（不旋=旋 0）----
    yb = 2.35
    box(ax, 1.30, yb - 0.15, 7.35, 1.55, "", ec=BLUE, lw=1.3, fc="#f5f9fe")
    ax.text(1.50, yb + 1.18, "implementation A — zero-frequency padding (rope_utils)",
            ha="left", va="center", fontsize=7.0, color=DARK, zorder=4)
    ax.text(1.50, yb + 0.70,
            "nope_angles = head_dim // 2 - rope_angles\ninv_freq = cat(inv_freq_rotated,\n                zeros(nope_angles))   # L246-261",
            ha="left", va="center", fontsize=6.1, color=MID, linespacing=1.3, zorder=4,
            family="monospace")
    ax.text(1.50, yb + 0.10, "$\\omega{=}0\\ \\Rightarrow$ angle $=0\\ \\Rightarrow\\ \\cos{=}1,\\ \\sin{=}0$: identity rotation",
            ha="left", va="center", fontsize=6.4, color=BLUE, zorder=4)

    box(ax, 9.55, yb - 0.15, 6.70, 1.55, "", ec=ORANGE, lw=1.3, fc="#fffaf8")
    ax.text(9.75, yb + 1.18, "implementation B — truncation (qwen3_next)",
            ha="left", va="center", fontsize=7.0, color=DARK, zorder=4)
    ax.text(9.75, yb + 0.70,
            "dim = int(head_dim * partial_rotary_factor)   # L111\nq_rot, q_pass = q[..., :rotary_dim], \\\n                q[..., rotary_dim:]          # L186-188\nQwen3-Next values: factor = 0.25 $\\to$ rotary_dim = 64",
            ha="left", va="center", fontsize=6.1, color=MID, linespacing=1.3, zorder=4,
            family="monospace")
    ax.text(9.75, yb - 0.05, "same math: not rotating $\\equiv$ rotating by 0 radians",
            ha="left", va="center", fontsize=6.4, color=ORANGE, zorder=4)

    ax.text(8.8, 1.62, "why rotate less? rotated dims spend their bandwidth on relative distance; content-only dims keep pure similarity —\nretrieval wants the latter (Kimi Linear 48B: NoPE wins RULER 84.3 vs 78.8)",
            ha="center", va="center", fontsize=6.6, color=DARK, linespacing=1.6)
    ax.text(8.8, 0.88, "and a smaller rotated spectrum is cheaper to stretch: YaRN 4$\\times$ on Qwen3-Next only has to slow 32 pairs",
            ha="center", va="center", fontsize=6.6, color=MID)

    out = os.path.join(FIG_DIR, "fig-8-2-longctx-partial-rope.png")
    fig.tight_layout()
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, choices=[1, 2], default=None,
                    help="只渲染指定图（默认两张都出；输出文件名带本章专属 slug，不覆写他图）")
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    if args.only in (None, 1):
        fig_1()
    if args.only in (None, 2):
        fig_2()


if __name__ == "__main__":
    main()
