# fig_ch04.py —— Book8 ch04 制图：图 4.1 mm 输入管线（两个「请求变胖」的行进路线）
# 所属章节：Book8 ch4（§4.2）；素材底单=notes/14 §二（行锚已核）
# 运行：cd <workspace> && source env.sh && python code/Book8-vLLM源码剖析下/ch04/fig_ch04.py
# 产物：drafts/Book8-vLLM源码剖析下/figures/fig-4-1-mm-input-pipeline.png
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "drafts", "Book8-vLLM源码剖析下", "figures"))
BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#0b0b0b", "#52514e"


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=6.2, lw=1.2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw, ls=ls,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


def arr(ax, x0, y0, x1, y1, color=MUTED, lw=1.1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", color=color, lw=lw,
                                 mutation_scale=10, zorder=2))


def fig_mm_pipeline():
    fig, ax = plt.subplots(figsize=(13.0, 6.4), dpi=300)
    ax.set_xlim(0, 18); ax.set_ylim(0, 9.6); ax.axis("off")
    ax.text(9, 9.3, "Multimodal input pipeline: how a request gets fat (sketched from vLLM v0.30.0 source)",
            ha="center", fontsize=8.8, color=INK, weight="bold")

    # 上带：请求进门（站 2/3）
    box(ax, 0.4, 7.5, 7.6, 1.2,
        "station 2/3: HTTP request\nimage_url (base64) + text -> chat protocol\nOpenAI protocol.py image_url branch", ec=BLUE, fc="#dcebfc", fs=6.2)
    box(ax, 8.4, 7.5, 4.6, 1.2, "LoRA line (same door):\nrequest carries LoRARequest\n(model=lora_name)", ec=ORANGE, fc="#fde3d6", fs=6.2)
    box(ax, 13.3, 7.5, 4.3, 1.2, "station 3: structured_outputs\n(two fat lines share\nthe SamplingParams exit)", ec=MUTED, fc="white", fs=5.8, ls="--")

    # 中带：输入处理（站 5）
    box(ax, 0.4, 5.2, 5.4, 1.7,
        "station 5: input processor\ninput_processor.py#L19-22 imports\nMULTIMODAL_REGISTRY + MultiModalBudget\nmm processor: image -> pixel values", ec=BLUE, fc="#dcebfc", fs=6.0)
    box(ax, 6.2, 5.2, 5.6, 1.7,
        "placeholder mechanics (processing/)\n<image_pad> x N_v expanded by processor\nPromptReplacement #L381 / PlaceholderFeaturesInfo #L588\nN_v = floor(H/16)*floor(W/16)/4  (patch16, merge2x2)", ec=TEAL, fc="#d9f2e7", fs=5.7)
    box(ax, 12.1, 5.2, 5.5, 1.7,
        "budget + hash\nencoder_budget.py#L45 (per-step coding budget)\nhasher.py: mm input hash -> prefix-cache key\n(Book6 ch06 attach-hash: LoRA id rides along)", ec=YELLOW, fc="#fdf0d2", fs=5.8)

    # 下带：调度/编码/执行（站 9/12）
    box(ax, 0.4, 2.8, 5.4, 1.7,
        "station 9: mm hash joins prefix chain\nscheduler sees fat prompt\n(N_v visual tokens = prefill bill:\n640x640 image -> 400 tokens, measured 402)", ec=BLUE, fc="#dcebfc", fs=5.9)
    box(ax, 6.2, 2.8, 5.6, 1.7,
        "station 12: encoder runner + cache (MRV2)\nworker/gpu/mm/ (encoder_cache, encoder_runner, lora)\nNPU: AscendMMEncoderAttention (FIA + graph_task_group)\ntmp/CPU two-layer encoder cache (model_runner_v1#L587)", ec=TEAL, fc="#d9f2e7", fs=5.4)
    box(ax, 12.1, 2.8, 5.5, 1.7,
        "station 12: LoRA punica path\nPunicaWrapperNPU -> AscendC bgmv/sgmv\nhot-swap set_active_loras every step\nlora state = graph dispatch key", ec=ORANGE, fc="#fde3d6", fs=5.8)

    box(ax, 0.4, 0.5, 17.2, 1.4,
        "two fat lines converge: visual tokens inflate the prefill bill (station 9) and ride the encoder runner (station 12);\n"
        "LoRA inflates nothing visible - it swaps behavior via a low-rank side path B*A on q/v_proj, dispatched per request (station 12)",
        ec=INK, fs=6.4)

    arr(ax, 4.2, 7.5, 3.1, 6.9, color=BLUE); arr(ax, 9.0, 8.1, 6.2, 8.1, color=ORANGE)
    arr(ax, 5.8, 6.0, 6.2, 6.0, color=BLUE); arr(ax, 11.8, 6.0, 12.1, 6.0, color=TEAL)
    arr(ax, 3.1, 5.2, 3.1, 4.5, color=BLUE); arr(ax, 9.0, 5.2, 9.0, 4.5, color=TEAL)
    arr(ax, 14.8, 5.2, 14.8, 4.5, color=ORANGE)
    arr(ax, 3.1, 2.8, 3.1, 1.9, color=BLUE)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-4-1-mm-input-pipeline.png")
    fig.savefig(out, facecolor="white"); plt.close(fig)
    print("[fig]", out)


if __name__ == "__main__":
    fig_mm_pipeline()
