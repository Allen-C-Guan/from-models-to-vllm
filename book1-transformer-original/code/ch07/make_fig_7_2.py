# -*- coding: utf-8 -*-
# 用途：生成 Book1 第 7 章图 7.2「整机数据流与参数流（组装 + 账本）」——
#       与第 4 章图 4.1（机制视角）、第 6 章图 6.1（单 token 通路）区分：本图主打组装与账本。
#       图内文字按本书印刷规格用英文术语，中文解释全部放书稿图注。
# 所属章节：《Transformer 原典》第 7 章
# 运行：source env.sh && python "code/ch07/make_fig_7_2.py"
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# 本书印刷图规格：固定分类色序（注意力=蓝 / FFN=橙 / 嵌入=青 / LN=黄），文字用墨色不用系列色
BLUE, ORANGE, CYAN, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUE_F, ORNG_F, CYAN_F, YLW_F = "#cde2fb", "#fde8db", "#d8f3e8", "#fdf3d7"
INK, NOTE, HAIR = "#0b0b0b", "#52514e", "#898781"

FIG_DIR = Path(__file__).resolve().parents[3] / "drafts" / "Book1-Transformer原典" / "figures"

# 账本数字来自 transformer.py 的 numel 实数（2026-10 实测；公式口径见第 6 章表 6.1）
ATTN, FFN, LN = 1_050_624, 2_099_712, 1_024
ENC_L, DEC_L = 3_152_384, 4_204_032
EMB, TOTAL = 18_944_000, 63_082_496
FAM = [("FFN family 12x 2.10M", 12 * FFN, ORANGE),
       ("attention family 18x 1.05M", 18 * ATTN, BLUE),
       ("shared embedding 18.94M", EMB, CYAN)]

fig, (ax, bar) = plt.subplots(2, 1, figsize=(8.6, 5.2), dpi=300,
                              gridspec_kw={"height_ratios": [4.0, 1]})
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
bar.set_xlim(0, 1); bar.set_ylim(-1.0, 1.6); bar.axis("off")


def box(x, y, w, h, text, ec=BLUE, fc=BLUE_F, fs=7.2, lw=1.3, tc=INK):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.01",
                                ec=ec, fc=fc, lw=lw, mutation_scale=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.5)
    return (x, y, w, h)


def arrow(p, q, color=HAIR, lw=1.3, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=lw, connectionstyle=f"arc3,rad={rad}"))


def note(x, y, text, fs=6.3, color=NOTE, ha="left"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color, linespacing=1.5)


# ---- 顶行：源侧（src → 编码器） ----
box(.075, .80, .09, .056, "src ids\n(B,S)", ec=HAIR, fc="white", fs=6.6)
box(.205, .80, .15, .105, "shared embedding $E$\n$37{,}000\\times512$ = 18.94M\n$\\times\\sqrt{512}$",
    ec=CYAN, fc=CYAN_F, fs=6.8)
box(.365, .80, .082, .056, "+ sinusoidal PE\n+ dropout", ec=HAIR, fc="white", fs=6.2)
box(.545, .80, .21, .175, "Encoder $\\times N{=}6$\n\nself-attn  1.05M\nFFN  2.10M\n"
                          "2 x LN  1024\n\n3.15M x 6 = 18.91M", ec=BLUE, fc=BLUE_F, fs=6.9)
arrow((.12, .80), (.13, .80)); arrow((.28, .80), (.324, .80)); arrow((.406, .80), (.44, .80))
note(.365, .752, "(B,S,512)", fs=6.3, ha="center")            # 查表+PE 后的主干形状（=编码器入口）
note(.545, .687, "2 sub-layers per layer; bidirectional self-attention (no causal mask)\n"
                 "self-attn scores (B,h,S,S) + padding mask (B,1,1,S)",
     fs=6.1, ha="center")

# ---- 中行：目标侧（tgt → 解码器 → 输出） ----
box(.075, .45, .09, .062, "tgt ids (B,T)\nshifted right", ec=HAIR, fc="white", fs=6.0)
box(.365, .45, .082, .056, "+ sinusoidal PE\n+ dropout", ec=HAIR, fc="white", fs=6.2)
box(.545, .45, .21, .21, "Decoder $\\times N{=}6$\n\nmasked self-attn  1.05M\ncross-attn  1.05M\n"
                         "FFN  2.10M\n3 x LN  1536\n\n4.20M x 6 = 25.22M", ec=BLUE, fc=BLUE_F, fs=6.9)
arrow((.093, .481), (.163, .742), rad=-0.25)                  # tgt 查同一张 E（§3.4 三处共享）
note(.058, .60, "lookup in the same $E$\n(3-way weight tying, S3.4)", fs=6.1)
arrow((.406, .45), (.44, .45))
note(.365, .408, "(B,T,512)", fs=6.3, ha="center")            # 查表+PE 后的主干形状（=解码器入口）
box(.72, .615, .13, .08, "pre-softmax linear\n$=E^{\\top}$ (tied, 0 extra params)", ec=CYAN, fc=CYAN_F, fs=6.6)
box(.72, .44, .082, .05, "softmax", ec=HAIR, fc="white", fs=6.8)
box(.885, .44, .105, .07, "P(next token)\n(B,T,V)", ec=HAIR, fc="white", fs=6.6)
arrow((.635, .555), (.665, .59))                              # 解码器塔顶 → pre-softmax
arrow((.72, .575), (.72, .468))                               # pre-softmax → softmax（垂直）
arrow((.761, .44), (.833, .44))                               # softmax → 概率
note(.545, .327, "3 sub-layers per layer; cross-attn reads K=V=memory\n"
                 "masked self-attn scores (B,h,T,T); cross-attn scores (B,h,T,S)",
     fs=6.1, ha="center")

# ---- memory：编码器 → 解码器每层 cross 注意力 ----
arrow((.655, .73), (.655, .565), color=INK, lw=1.5)
ax.text(.668, .648, "memory (B,S,512)", fontsize=6.5, color=INK, rotation=90, va="center")

# ---- 出口：论文口径（Fig 1 无栈顶 LN；learned linear + softmax） ----
note(.035, .19, "Weight tying (S3.4): src embedding = tgt embedding = pre-softmax linear, one matrix $E$\n"
                "(enabled by the shared 37k BPE vocab; \"similar to [30]\" = Press & Wolf, arXiv:1608.05859).\n"
                "Untied counterfactual: 63.1M + 2 x 18.9M = 101.0M; tying saves 37.9M (> 1/3 of untied,\n"
                "about two encoder towers). Paper-canonical exits: no stack-top LayerNorm (Fig 1); output = learned linear + softmax (S3.4).",
     fs=6.3)

# ---- 底部：参数账本堆叠条（2px 白缝分隔，直接标注；数据 = numel 实数） ----
left = 0.0
for name, val, color in FAM:
    w = val / TOTAL
    bar.barh(0, w, left=left, height=0.6, color=color, edgecolor="white", lw=1.6)
    bar.text(left + w / 2, 0, f"{name}\n{val/1e6:.2f}M  ({val/TOTAL:.1%})",
             ha="center", va="center", fontsize=6.3, color="white")
    left += w
ln_w = 30 * LN / TOTAL
bar.barh(0, ln_w, left=left, height=0.6, color=YELLOW, edgecolor="white", lw=1.6)
bar.annotate("LN 30x1024\n0.03M (0.05%)", xy=(left + ln_w / 2, 0.31), xytext=(0.93, 1.05),
             fontsize=6.0, color=INK, ha="center", va="bottom",
             arrowprops=dict(arrowstyle="-", color=HAIR, lw=0.8))
bar.text(0.0, 0.85, "base parameter ledger (by component family)", fontsize=7.6, color=INK, va="bottom")
bar.text(0.0, -0.42, "total 63.08M counted by numel; Table 3 reports 65M "
                     "(gap 1.9M ~ 3%: vocab approximation and accounting conventions)",
         fontsize=6.4, color=NOTE, va="top")

fig.tight_layout()
fig.savefig(FIG_DIR / "fig-7-2-assembly-parameter-ledger.png", facecolor="white")
plt.close(fig)
print(f"图 7.2 已保存：{FIG_DIR / 'fig-7-2-assembly-parameter-ledger.png'}")
