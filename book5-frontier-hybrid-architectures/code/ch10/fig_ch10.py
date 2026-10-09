# fig_ch10.py —— Book5 ch10 制图：图 10.1 多模态三步曲 / 图 10.3 projector 形状流转 /
#                图 10.4 后接与原生两路线对照（图 10.2 为 LLaVA 原图引用，非本脚本产物）
# 用途：图 10.1 = 10.2 节演进图（ViT→CLIP→LLaVA 各自「对齐什么」——三联面板，逐年一格）；
#       图 10.4 = 10.3 节两路线对照（上轨=LLaVA 式后接：双塔预训练+两阶段对齐；下轨=原生联训：
#         从头联合优化、图文 token 同一目标；中条=「projector 两条路线都在」的判词）；
#       （编号 2026-10-09 按成稿首现顺序重排：projector=10.3、两路线=10.4——正文同步）
#       图 10.3 = 10.2 节 mini 实操形状流转（真实代码路径逐格标形状：(3,224,224)→(50,768)→
#         (768,)→W(1024,768)→(1024,)→占位替换→(B,L,1024)；底轨=标准几何/LLaVA-1 真机/单 token 简化三档）。
# 所属章节：Book5 第 10 章（图 10.1/10.3/10.4；图 10.2=arXiv:2304.08485 v2 Figure 1 原图，CC BY 4.0）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch10/fig_ch10.py"
#           [--which all|steps|routes|proj]
# 产物：figures/fig-10-{1,3,4}-mm-<slug>.png（slug=threesteps/native-vs-posthoc/projector）
# 图规：300dpi 白底；系列色 蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；线宽 2；图内英文、图注中文；
#       虚线框=冻结（frozen）、实线=参与训练；结构图每个张量框与关键箭头标注形状。
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "book5-frontier-hybrid-architectures", "figures")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e1e0d9"
FROZEN_C = "#898781"     # 冻结件边色（灰）


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=8.0, lw=1.5, tc=INK, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc,
                                lw=lw, linestyle=ls, mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc,
            zorder=4, linespacing=1.35)
    return x + w / 2, y + h / 2


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.2, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=lw, linestyle=ls, zorder=2))


def shape_note(ax, x, y, s, color=ORANGE, fs=7.2):
    ax.text(x, y, s, ha="center", va="center", fontsize=fs, color=color, zorder=4)


# ---------------- 图 10.1：多模态三步曲（ViT→CLIP→LLaVA 各自对齐什么） ----------------
def fig_steps():
    fig, ax = plt.subplots(figsize=(12.8, 5.0), dpi=300)
    ax.set_xlim(0, 15.0)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    ax.text(7.5, 6.35, "Three steps to a multimodal LM: what each aligns", ha="center",
            fontsize=11.5, color=INK, weight="bold")

    # ---- 面板 A：ViT（2020）——图像进 Transformer：切成词 ----
    ax.add_patch(FancyBboxPatch((0.25, 0.35), 4.3, 5.4, boxstyle="round,pad=0.04",
                                ec=GRIDC, fc="#fcfcfa", lw=1.0, zorder=1))
    ax.text(2.4, 5.42, "ViT · 2020 · arXiv:2010.11929", ha="center", fontsize=8.6,
            color=INK, weight="bold")
    c = box(ax, 0.5, 4.35, 1.5, 0.75, "image\n(3,224,224)", ec=MUTED, fs=7.4)
    c = box(ax, 2.45, 4.35, 1.9, 0.75, "16x16 patches\n196 tokens", ec=BLUE, fs=7.4)
    arrow(ax, 2.0, 4.72, 2.45, 4.72)
    c = box(ax, 0.5, 3.0, 3.85, 0.9, "standard Transformer encoder\n(learned patch embedding)", ec=BLUE, fs=7.4)
    arrow(ax, 3.4, 4.35, 3.4, 3.9)
    c = box(ax, 0.5, 1.85, 3.85, 0.75, "classification head -> label", ec=MUTED, fs=7.4)
    arrow(ax, 2.4, 3.0, 2.4, 2.6)
    ax.text(2.4, 1.35, "aligns: image <-> label words (supervised)", ha="center",
            fontsize=7.6, color=TEAL, weight="bold")
    ax.text(2.4, 0.85, "\"An image is worth 16x16 words\":\nthe image enters the Transformer as a token sequence",
            ha="center", fontsize=7.0, color=MUTED)

    # ---- 面板 B：CLIP（2021）——两个编码器、一个共享空间 ----
    ax.add_patch(FancyBboxPatch((5.0, 0.35), 4.3, 5.4, boxstyle="round,pad=0.04",
                                ec=GRIDC, fc="#fcfcfa", lw=1.0, zorder=1))
    ax.text(7.15, 5.42, "CLIP · 2021 · arXiv:2103.00020", ha="center", fontsize=8.6,
            color=INK, weight="bold")
    c = box(ax, 5.25, 4.35, 1.7, 0.75, "image\n(3,224,224)", ec=MUTED, fs=7.4)
    c = box(ax, 5.25, 3.1, 1.7, 0.75, "caption text", ec=MUTED, fs=7.4)
    c = box(ax, 7.35, 4.35, 1.75, 0.75, "image encoder", ec=BLUE, fs=7.4)
    c = box(ax, 7.35, 3.1, 1.75, 0.75, "text encoder", ec=BLUE, fs=7.4)
    arrow(ax, 6.95, 4.72, 7.35, 4.72)
    arrow(ax, 6.95, 3.47, 7.35, 3.47)
    # 共享嵌入空间（虚线椭圆）
    el = FancyBboxPatch((7.75, 1.75), 1.1, 1.05, boxstyle="round,pad=0.03", ec=ORANGE,
                        fc="white", lw=1.2, linestyle="--", zorder=3)
    ax.add_patch(el)
    ax.text(8.3, 2.28, "shared\nembedding\nspace", ha="center", va="center", fontsize=6.8,
            color=INK, zorder=4)
    arrow(ax, 8.22, 4.35, 8.3, 2.85, color=MUTED)
    arrow(ax, 8.22, 3.1, 8.28, 2.85, color=MUTED)
    ax.text(7.15, 1.35, "aligns: image <-> text embeddings (contrastive)", ha="center",
            fontsize=7.6, color=TEAL, weight="bold")
    ax.text(7.15, 0.85, "400M (image, text) pairs; matched pairs pulled together,\nmismatched pushed apart -> zero-shot transfer",
            ha="center", fontsize=7.0, color=MUTED)

    # ---- 面板 C：LLaVA（2023）——缝合两个已训世界 ----
    ax.add_patch(FancyBboxPatch((9.75, 0.35), 5.15, 5.4, boxstyle="round,pad=0.04",
                                ec=GRIDC, fc="#fcfcfa", lw=1.0, zorder=1))
    ax.text(12.32, 5.42, "LLaVA · 2023 · arXiv:2304.08485", ha="center", fontsize=8.6,
            color=INK, weight="bold")
    c = box(ax, 10.0, 4.35, 1.5, 0.75, "image\n(3,224,224)", ec=MUTED, fs=7.4)
    c = box(ax, 11.95, 4.35, 1.85, 0.75, "CLIP ViT-L/14\nFROZEN", ec=FROZEN_C, fc="#f7f7f5",
            fs=7.4, ls="--")
    arrow(ax, 11.5, 4.72, 11.95, 4.72)
    shape_note(ax, 13.42, 4.72, "(256,1024)")                       # Z_v 形状
    c = box(ax, 11.95, 3.42, 1.85, 0.62, "W: one linear layer\n(TRAINABLE)", ec=TEAL, fs=7.2)
    arrow(ax, 12.87, 4.35, 12.87, 4.04, color=TEAL, lw=1.6)
    shape_note(ax, 13.42, 3.73, "(256,5120)")                       # H_v 形状
    c = box(ax, 10.0, 2.4, 1.5, 0.75, "question\ntext", ec=MUTED, fs=7.4)
    c = box(ax, 14.0, 3.15, 0.75, 1.75, "LM\nVicuna\nFROZEN", ec=FROZEN_C, fc="#f7f7f5",
            fs=7.2, ls="--")
    arrow(ax, 13.8, 3.73, 14.0, 3.85, color=MUTED)
    arrow(ax, 11.5, 2.77, 14.0, 3.35, color=MUTED)
    c = box(ax, 14.0, 1.95, 0.75, 0.55, "answer", ec=MUTED, fs=7.2)
    arrow(ax, 14.37, 3.15, 14.37, 2.5, color=MUTED)
    ax.text(12.32, 1.45, "aligns: vision features <-> LM word-embedding space", ha="center",
            fontsize=7.6, color=TEAL, weight="bold")
    ax.text(12.32, 0.9, "stitch two pretrained worlds with ONE trainable\nprojection: H_v = W · Z_v (paper Fig. 1 = our Fig. 10.2)",
            ha="center", fontsize=7.0, color=MUTED)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-10-1-mm-threesteps.png"), dpi=300,
                facecolor="white")
    plt.close(fig)


# ---------------- 图 10.4：后接 vs 原生联训两路线对照 ----------------
def fig_routes():
    fig, ax = plt.subplots(figsize=(12.8, 7.0), dpi=300)
    ax.set_xlim(0, 15.0)
    ax.set_ylim(0, 8.6)
    ax.axis("off")
    ax.text(7.5, 8.35, "Two routes to multimodality: graft afterwards, or train jointly from the start",
            ha="center", fontsize=11.5, color=INK, weight="bold")
    ax.text(1.0, 7.85, "dashed = frozen / pretrained separately;  solid = in training", fontsize=7.4, color=MUTED)

    # ---- 上轨：后接式（LLaVA）----
    ax.add_patch(FancyBboxPatch((0.25, 4.55), 14.5, 3.05, boxstyle="round,pad=0.04",
                                ec=GRIDC, fc="#fcfcfa", lw=1.0, zorder=1))
    ax.text(7.5, 7.28, "Route A · post-hoc grafting  (LLaVA 2023; our llava_mini.py)", ha="center",
            fontsize=9.2, color=INK, weight="bold")
    c = box(ax, 0.6, 5.6, 2.5, 1.0, "vision tower: CLIP\npretrained contrastively\nFROZEN", ec=FROZEN_C,
            fc="#f7f7f5", fs=7.3, ls="--")
    c = box(ax, 3.6, 5.6, 2.5, 1.0, "language model\npretrained on text\nFROZEN in stage 1", ec=FROZEN_C,
            fc="#f7f7f5", fs=7.3, ls="--")
    c = box(ax, 6.6, 5.5, 3.5, 1.2, "Stage 1 · feature alignment\ntrain W only (595K image-text pairs)\n\"a compatible visual tokenizer\"",
            ec=TEAL, fs=7.2)
    c = box(ax, 10.5, 5.5, 3.9, 1.2, "Stage 2 · visual instruction tuning\nW + LM both trained (158K GPT-4 data)\nvision tower stays frozen",
            ec=BLUE, fs=7.2)
    arrow(ax, 3.1, 6.1, 3.6, 6.1)
    arrow(ax, 6.1, 6.1, 6.6, 6.1)
    arrow(ax, 10.1, 6.1, 10.5, 6.1)
    ax.text(7.5, 4.85, "cheap & modular: reuse two pretrained worlds; alignment is one short stage -- representation not shaped by LM loss",
            ha="center", fontsize=7.3, color=MUTED)

    # ---- 下轨：原生联训（K3 / Qwen3.5）----
    ax.add_patch(FancyBboxPatch((0.25, 0.55), 14.5, 3.05, boxstyle="round,pad=0.04",
                                ec=GRIDC, fc="#f6fdfa", lw=1.0, zorder=1))
    ax.text(7.5, 3.28, "Route B · native joint training  (K3 MoonViT-V2 2026; Qwen3.5 vision tower)", ha="center",
            fontsize=9.2, color=INK, weight="bold")
    c = box(ax, 0.6, 1.6, 2.6, 1.0, "vision tower trained\nFROM SCRATCH with LM\n(MoonViT-V2: 27L, 0.4B)", ec=BLUE, fs=7.3)
    c = box(ax, 3.7, 1.6, 2.3, 1.0, "lightweight MLP\nprojector\n(still there!)", ec=TEAL, fs=7.3)
    c = box(ax, 6.5, 1.6, 2.6, 1.0, "LM backbone\ntrained jointly\nfrom the start", ec=BLUE, fs=7.3)
    c = box(ax, 9.6, 1.45, 4.8, 1.3, "single next-token prediction objective\nover interleaved image & text tokens\nimage = 16,384 visual tokens (K3)",
            ec=BLUE, fs=7.4)
    arrow(ax, 3.2, 2.1, 3.7, 2.1)
    arrow(ax, 6.0, 2.1, 6.5, 2.1)
    arrow(ax, 9.1, 2.1, 9.6, 2.1)
    ax.text(7.5, 0.85, "representations shaped directly by the language objective; needs data + stability work\n(from-scratch vs SigLIP-init: gradient spikes vs stable, K3 report Fig. 6)",
            ha="center", fontsize=7.3, color=MUTED)

    # ---- 中条判词 ----
    ax.add_patch(FancyBboxPatch((1.6, 3.85), 11.8, 0.52, boxstyle="round,pad=0.03", ec=YELLOW,
                                fc="#fdf9ef", lw=1.2, zorder=3))
    ax.text(7.5, 4.11, "the projector is still there in BOTH routes -- what differs is WHEN alignment happens: one stage vs the whole training",
            ha="center", va="center", fontsize=8.0, color=INK, zorder=4)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-10-4-mm-native-vs-posthoc.png"), dpi=300,
                facecolor="white")
    plt.close(fig)


# ---------------- 图 10.3：projector 形状流转（mini 实机真实几何） ----------------
def fig_proj():
    fig, ax = plt.subplots(figsize=(12.8, 5.8), dpi=300)
    ax.set_xlim(0, 15.0)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.text(7.5, 6.95, "Shape flow of llava_mini.py: one image becomes one word (real code path)",
            ha="center", fontsize=11.5, color=INK, weight="bold")
    ax.text(7.5, 6.55, "frozen towers dashed; the ONLY trainable part is W (768->1024, no bias, 786,432 params)",
            ha="center", fontsize=7.6, color=MUTED)

    # 主轨（真实代码路径）
    c = box(ax, 0.3, 4.6, 1.6, 0.95, "image\n(3,224,224)", ec=MUTED, fs=7.6)
    c = box(ax, 2.35, 4.35, 2.1, 1.45, "CLIP ViT-B/32\nvision tower\nFROZEN\npatch conv + 12 enc. layers",
            ec=FROZEN_C, fc="#f7f7f5", fs=7.2, ls="--")
    arrow(ax, 1.9, 5.07, 2.35, 5.07)
    c = box(ax, 4.95, 4.6, 2.0, 0.95, "last_hidden_state\n(50,768)\n1 CLS + 49 patches", ec=MUTED, fs=7.2)
    arrow(ax, 4.45, 5.07, 4.95, 5.07)
    c = box(ax, 7.45, 4.6, 1.85, 0.95, "drop CLS,\nmean-pool 49\n-> (768,)", ec=MUTED, fs=7.2)
    arrow(ax, 6.95, 5.07, 7.45, 5.07)
    c = box(ax, 9.8, 4.35, 1.95, 1.45, "projector W\n(1024,768)\nTRAINABLE\none linear layer",
            ec=TEAL, fc="#e9f8f1", fs=7.4)
    arrow(ax, 9.3, 5.07, 9.8, 5.07, color=TEAL, lw=1.6)
    c = box(ax, 12.15, 4.6, 2.5, 0.95, "replace <|image_pad|>\nembedding: (1024,)\ninto text stream (B,L,1024)",
            ec=BLUE, fs=7.2)
    arrow(ax, 11.75, 5.07, 12.15, 5.07, color=TEAL, lw=1.6)
    c = box(ax, 4.6, 2.7, 4.6, 0.95, "Qwen3-0.6B (d=1024, bf16) FROZEN\ninputs_embeds -> logits (B,L,V=151936)", ec=FROZEN_C,
            fc="#f7f7f5", fs=7.3, ls="--")
    arrow(ax, 13.4, 4.6, 13.4, 3.65, color=MUTED)
    c = box(ax, 10.1, 2.7, 3.3, 0.95, "greedy generation\n(64 tokens, prompt-only)", ec=MUTED, fs=7.3)
    arrow(ax, 9.2, 3.18, 10.1, 3.18)
    shape_note(ax, 2.15, 3.95, "visual side", color=MUTED)
    shape_note(ax, 11.0, 3.95, "language side", color=MUTED)

    # 底轨：三档「一张图=多少个词」
    ax.add_patch(FancyBboxPatch((0.3, 0.35), 14.35, 1.75, boxstyle="round,pad=0.04",
                                ec=YELLOW, fc="#fdf9ef", lw=1.0, zorder=1))
    ax.text(7.5, 1.78, "How many words is one image? (visual token count per image)", ha="center",
            fontsize=8.6, color=INK, weight="bold")
    ax.text(2.9, 1.02, "standard geometry (ViT-B/16):\n(196,768) ->W-> (196,d)\n196 words", ha="center",
            fontsize=7.4, color=INK)
    ax.text(7.3, 1.02, "LLaVA-1 (ViT-L/14-224):\n(256,1024) ->W-> (256,5120)\n256 words", ha="center",
            fontsize=7.4, color=INK)
    ax.text(11.9, 1.02, "our mini (ViT-B/32 + mean-pool):\n(49,768) -> 1 word, (768,)\npre-registered simplification", ha="center",
            fontsize=7.4, color=INK)
    ax.plot([5.35, 5.35], [0.55, 1.55], color=GRIDC, lw=1.0)
    ax.plot([9.65, 9.65], [0.55, 1.55], color=GRIDC, lw=1.0)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-10-3-mm-projector.png"), dpi=300,
                facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Book5 ch10 figures (10.1/10.3/10.4)")
    ap.add_argument("--which", choices=["all", "steps", "routes", "proj"], default="all")
    args = ap.parse_args()
    if args.which in ("all", "steps"):
        fig_steps()
    if args.which in ("all", "routes"):
        fig_routes()
    if args.which in ("all", "proj"):
        fig_proj()
    print("[fig_ch10] done ->", FIG_DIR)


if __name__ == "__main__":
    main()
