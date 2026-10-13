# fig_ch03.py —— Book8 ch03 制图：图 3.1 schema→掩码管线（两泳道、沿站号）
# 所属章节：Book8 ch3（§3.2）；设计书=notes/12 §二图 3.1 规格
# 运行：cd <workspace> && source env.sh && python code/Book8-vLLM源码剖析下/ch03/fig_ch03.py
# 产物：drafts/Book8-vLLM源码剖析下/figures/fig-3-1-schema-bitmask-pipeline.png
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "drafts", "Book8-vLLM源码剖析下", "figures"))
BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e1e0d9"


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=6.2, lw=1.2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw, ls=ls,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


def arr(ax, x0, y0, x1, y1, color=MUTED, lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                 mutation_scale=10, zorder=2))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13.0, 6.6), dpi=300)
    ax.set_xlim(0, 18); ax.set_ylim(0, 9.8); ax.axis("off")
    ax.text(9, 9.45, "From schema to token bitmask: the structured-output pipeline (sketched from vLLM v0.30.0 source)",
            ha="center", fontsize=8.8, color=INK, weight="bold")
    box(ax, 0.4, 7.6, 17.2, 0.9, "frontend process: response_format / structured_outputs (SamplingParams #L88, station 3)", ec=INK, fs=6.8)
    box(ax, 0.4, 5.9, 17.2, 1.2,
        "EngineCore process\ngrammar_init (input thread, core.py #L995) -> async compile pool (#L168-172) -> request WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR (scheduler #L2979-88)",
        ec=ORANGE, fc="#fde3d6", fs=6.2)
    box(ax, 0.4, 4.0, 5.4, 1.5, "GrammarCompiler\nbackend_xgrammar.py #L66\n512MB in-proc cache\n(key: request.py #L76)", ec=TEAL, fc="#d9f2e7", fs=6.2)
    box(ax, 6.2, 4.0, 5.4, 1.5, "per-step bitmask fill\n_fill_bitmasks #L202\n>128 reqs parallel, batch 16\n(#L62-69)", ec=TEAL, fc="#d9f2e7", fs=6.2)
    box(ax, 12.0, 4.0, 5.6, 1.5, "apply on logits (station 12)\nutils.py #L100 apply_grammar_bitmask\nMRV2: on-device worker #L38/#L125", ec=TEAL, fc="#d9f2e7", fs=6.0)
    box(ax, 0.4, 2.2, 8.0, 1.3, "mask tensor: (B_s, ceil(V/32)) int32 bit-packed\nB_s = structured reqs in batch", ec=YELLOW, fc="#fdf0d2", fs=6.4)
    box(ax, 8.8, 2.2, 8.8, 1.3, "token feedback (station 13): accept_tokens #L438 -> grammar state advance;\ndraft path: validate_tokens #L294 longest valid prefix (+ch2)", ec=BLUE, fc="#dcebfc", fs=6.0)
    box(ax, 0.4, 0.5, 17.2, 1.2,
        "observed account: compile (one-shot, in queue_time histogram) vs per-step mask (recurring, in ITL) - two ledgers, two experiments",
        ec=INK, fs=6.4)
    for x in (3.1, 8.9, 14.8):
        arr(ax, x, 7.6, x, 7.1)
    arr(ax, 3.1, 5.9, 3.1, 5.5); arr(ax, 8.9, 5.9, 8.9, 5.5); arr(ax, 14.8, 5.9, 14.8, 5.5)
    arr(ax, 5.8, 4.75, 6.2, 4.75); arr(ax, 11.6, 4.75, 12.0, 4.75)
    arr(ax, 14.8, 4.0, 14.8, 3.5, color=YELLOW)
    arr(ax, 4.4, 3.5, 4.4, 4.0, color=TEAL, ls="--")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-3-1-schema-bitmask-pipeline.png")
    fig.savefig(out, facecolor="white"); plt.close(fig)
    print("[fig]", out)


if __name__ == "__main__":
    fig_pipeline()
