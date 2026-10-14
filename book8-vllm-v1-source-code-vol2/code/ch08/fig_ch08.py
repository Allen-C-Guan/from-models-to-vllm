# fig_ch08.py —— Book8 ch08 制图：图 8.1 C8 int8 KV 五步链数据流（启动带+每步带）
# 所属章节：Book8 ch8（§8.6）；素材底单=notes/18 §二 C8 行锚（全部 sed 核字）
# 运行：cd <workspace> && source env.sh && python code/Book8-vLLM源码剖析下/ch08/fig_ch08.py
# 产物：drafts/Book8-vLLM源码剖析下/figures/fig-8-1-c8-five-step-chain.png
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


def fig_c8_chain():
    fig, ax = plt.subplots(figsize=(13.0, 6.8), dpi=300)
    ax.set_xlim(0, 18); ax.set_ylim(0, 10.2); ax.axis("off")
    ax.text(9, 9.85, "C8 INT8 KV: from checkpoint metadata to in-kernel dequant (vllm-ascend 0.27.1rc1, sketched)",
            ha="center", fontsize=8.8, color=INK, weight="bold")

    # 启动带（站 0）：①→③→④ 与检查点 scale 装载
    box(ax, 0.4, 8.0, 17.2, 0.85,
        "startup (station 0): checkpoint door - quant_description[\"kv_cache_type\"] == \"C8\"  (modelslim_config.py #L1107-1108)",
        ec=INK, fs=6.6)
    box(ax, 0.4, 6.1, 5.4, 1.5,
        "step 1-2 gate per layer\nenable_c8_quant -> c8_quant_layers\n(#L1108, #L897-903)", ec=BLUE, fc="#dcebfc", fs=6.2)
    box(ax, 6.2, 6.1, 5.4, 1.5,
        "step 3 attach method\nAscendC8KVCacheAttentionMethod\n(#L804-808)", ec=BLUE, fc="#dcebfc", fs=6.2)
    box(ax, 12.0, 6.1, 5.6, 1.5,
        "step 4 class surgery\nlayer.impl.__class__ = AscendC8AttentionBackendImpl\n(kv_c8.py #L126-130)", ec=BLUE, fc="#dcebfc", fs=5.9)
    box(ax, 0.4, 4.6, 17.2, 1.1,
        "checkpoint tensors: .k_proj.kv_cache_scale/offset -> .attn.k_cache_scale/offset  (#L664-665)\nstatic per-channel scale+offset, ones/zeros init then loaded from ModelSlim checkpoint",
        ec=BLUE, fc="white", ls="--", fs=6.0)

    # 每步带（站 12）：写入侧→int8 块→读取侧
    box(ax, 0.4, 2.6, 5.4, 1.6,
        "write side (every step)\nnew K/V bf16 (8,128) per layer\nclamp(round(x*inv_scale+offset),-128,127)\n-> int8  (attention_v1.py #L1965-1986)", ec=ORANGE, fc="#fde3d6", fs=5.8)
    box(ax, 6.2, 2.6, 5.4, 1.6,
        "KV pool, int8 1B/value\nblock layout (2, num_blocks, 128, h_kv, d)\nblock_size % 32 == 0 asserted\nledger: 2B -> 1B per value, x2 capacity", ec=YELLOW, fc="#fdf0d2", fs=6.0)
    box(ax, 12.0, 2.6, 5.6, 1.6,
        "read side (in-kernel dequant)\nnpu_fused_infer_attention_score(\nkey/value_antiquant_scale=..., BNSD,\nantiquant_mode=0)  (#L1995-2020)", ec=TEAL, fc="#d9f2e7", fs=5.8)
    box(ax, 0.4, 0.6, 17.2, 1.4,
        "decode loop: quantize on write (station 12, orange) - store at 1B (yellow) - dequant inside FIA prologue (teal)\nattention output stays bf16; the quantized bytes never leave the kernel as int8",
        ec=INK, fs=6.4)

    for x in (3.1, 8.9, 14.8):
        arr(ax, x, 8.0, x, 7.6)
    arr(ax, 5.8, 6.85, 6.2, 6.85); arr(ax, 11.6, 6.85, 12.0, 6.85)
    arr(ax, 3.1, 6.1, 3.1, 5.7, color=BLUE, ls="--")
    arr(ax, 14.8, 6.1, 14.8, 5.7, color=BLUE, ls="--")
    arr(ax, 5.8, 3.4, 6.2, 3.4, color=ORANGE); arr(ax, 11.6, 3.4, 12.0, 3.4, color=TEAL)
    arr(ax, 8.9, 4.6, 8.9, 4.2, color=BLUE, ls="--")
    arr(ax, 14.8, 2.6, 14.8, 2.0, color=TEAL)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-8-1-c8-five-step-chain.png")
    fig.savefig(out, facecolor="white"); plt.close(fig)
    print("[fig]", out)


if __name__ == "__main__":
    fig_c8_chain()
