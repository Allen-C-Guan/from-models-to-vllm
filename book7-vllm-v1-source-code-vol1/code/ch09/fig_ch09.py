# fig_ch09.py —— Book7 ch09 制图：图 9.1 attention 三件套接口 / 图 9.2 后端族×架构对应（热图式）
# 用途：ch09 两图——图 9.1=AttentionBackend(身份证)/MetadataBuilder(账本厂)/Impl(执行器) 三件套
#       与模型侧 Attention 层的关系（接口即契约）；图 9.2=后端族×架构矩阵（FA 族/FlashInfer/MLA 系/
#       linear-gdn-mamba 族 × GQA/MLA/SWA/线性/混合 的支持关系+书册对应）。
# 所属章节：Book7 ch09（§9.2/§9.4）。
# 运行：cd <workspace> && source env.sh && python code/Book7-vLLM源码剖析上/ch09/fig_ch09.py
# 产物：drafts/Book7-vLLM源码剖析上/figures/fig-9-1-trio.png / fig-9-2-family-matrix.png
# 图规：300dpi 白底；蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；图内英文图注中文；自绘+版本锚。
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "drafts", "Book7-vLLM源码剖析上", "figures"))
BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e1e0d9"


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=6.4, lw=1.2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw, ls=ls,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


def arr(ax, x0, y0, x1, y1, color=MUTED, lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                 mutation_scale=10, zorder=2))


def fig_trio():
    fig, ax = plt.subplots(figsize=(12.6, 7.0), dpi=300)
    ax.set_xlim(0, 18); ax.set_ylim(0, 10.8); ax.axis("off")
    ax.text(9, 10.45, "The attention trio: model code sees one interface, the platform picks the implementation (v0.30.0)",
            ha="center", fontsize=8.8, color=INK, weight="bold")
    # 模型侧
    box(ax, 0.7, 6.4, 5.4, 2.9,
        "MODEL SIDE (per layer)\n\nllama.py #L212-232:\nself.attn = Attention(num_heads,\n    num_kv_heads, head_size, ...)\n...\nout = self.attn(q, k, v)\n\n(only 3 promises - see Q5)",
        ec=INK, fc="#f0efec", fs=6.8)
    # 中间接口
    box(ax, 7.1, 6.4, 4.2, 2.9,
        "Attention layer\nlayers/attention/attention.py#L223\n\n- stores K/V into cache\n- runs (MHA/MQA/GQA) attn\n- returns output tensor\nbackend chosen at __init__\n(L346-359) via selector",
        ec=BLUE, fc="#dcebfc", fs=6.4)
    arr(ax, 6.1, 7.85, 7.1, 7.85)
    # 三件套
    box(ax, 12.1, 8.2, 5.4, 1.35,
        "AttentionBackend  backend.py#L77\nget_name / get_impl_cls / get_builder_cls\n(+ ~20 classmethod capability checks)",
        ec=ORANGE, fc="#fde3d6", fs=6.2)
    box(ax, 12.1, 6.4, 5.4, 1.35,
        "MetadataBuilder  (one per group)\nbuilds attn metadata each step\ngpu_model_runner.py#L2568",
        ec=TEAL, fc="#d9f2e7", fs=6.2)
    box(ax, 12.1, 4.6, 5.4, 1.35,
        "Impl  (one per layer)\nforward(q, k, v, output_shape)\nflash_attn.py#L1051 / attention_v1.py#L460",
        ec=YELLOW, fc="#fdf0d2", fs=6.2)
    arr(ax, 11.3, 7.85, 12.1, 8.85); arr(ax, 11.3, 7.85, 12.1, 7.05); arr(ax, 11.3, 7.85, 12.1, 5.25)
    # 下层：家族
    box(ax, 0.7, 2.4, 16.8, 1.4,
        "backends/ family (43 enum entries, registry.py#L44-151):  FA family | FlashInfer | MLA line (mla/, 23 files) | ROCm | CPU/XPU |\n"
        "model-driven (FLASHMLA_SPARSE_DSV4 lives in vllm.models.*) | composite (runtime-synthesized, composite.py#L147) | Mamba enum separate (registry.py#L197)",
        ec=MUTED, fc="white", fs=6.2)
    for x in (3.4, 7.0, 10.6, 14.2):
        arr(ax, x, 4.6, x, 3.8, color=MUTED)
    ax.text(9, 1.5, "NPU column: @register_backend(CUSTOM, \"ASCEND\") attention_v1.py#L78  ->  AscendAttentionBackend #L79 / Metadata #L161 / Builder #L216 / Impl #L460\n"
                    "(get_name() returns \"CUSTOM\" - the enum's honest footnote); platform 4-tuple lookup platform.py#L244-249",
            ha="center", fontsize=6.4, color=MUTED, style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-9-1-trio.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print("[图 9.1] trio")


def fig_matrix():
    fig, ax = plt.subplots(figsize=(12.6, 6.4), dpi=300)
    ax.set_xlim(0, 18); ax.set_ylim(0, 9.8); ax.axis("off")
    ax.text(9, 9.45, "Backend families x architecture kinds (from Book4-5 columns; boxes = where each book taught it)",
            ha="center", fontsize=8.8, color=INK, weight="bold")
    cols = ["GQA dense\n(Book3)", "SWA window\n(Book5 ch2)", "MLA latent\n(Book4 ch6)", "linear/GDN\n(Book5 ch5-6)", "hybrid mix\n(Book5 ch9)"]
    rows = [
        ("FA family\n(flash_attn.py)", [1, 1, 0, 0, 0]),
        ("FlashInfer\n(2501.01005)", [1, 1, 1, 0, 0]),
        ("MLA line\n(mla/, 23 files)", [0, 0, 1, 0, 0]),
        ("linear/gdn/mamba\n(separate enum)", [0, 0, 0, 1, 0]),
        ("composite\n(runtime-synth)", [0, 0, 0, 0, 1]),
        ("ASCEND backends\n(vllm-ascend/attention/)", [1, 0, 1, 0, 0]),
    ]
    x0, y0, cw, ch = 4.6, 0.8, 2.4, 1.1
    for j, c in enumerate(cols):
        ax.text(x0 + j * cw + cw / 2, y0 + len(rows) * ch + 0.35, c, ha="center", fontsize=6.6, color=INK)
    for i, (rname, vals) in enumerate(rows):
        y = y0 + (len(rows) - 1 - i) * ch
        ax.text(x0 - 0.3, y + ch / 2, rname, ha="right", va="center", fontsize=6.6, color=INK)
        for j, v in enumerate(vals):
            fc = {1: "#a8cff0", 0: "white"}[v]
            ec = BLUE if v else GRIDC
            ax.add_patch(Rectangle((x0 + j * cw, y), cw - 0.12, ch - 0.12, fc=fc, ec=ec, lw=1.0, zorder=2))
            if v:
                ax.text(x0 + j * cw + (cw - 0.12) / 2, y + (ch - 0.12) / 2, "supported", ha="center", va="center",
                        fontsize=5.8, color=INK, zorder=3)
    ax.text(9, 0.35, "reading: every architecture kind from Books 3-5 has a backend family here; the mapping is the registry's raison d'etre\n"
                     "(matrix simplified - actual support also gates on head_size/dtype/capability checks; see validate_configuration's 20 questions)",
            ha="center", fontsize=6.4, color=MUTED, style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-9-2-family-matrix.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print("[图 9.2] family-matrix")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_trio()
    fig_matrix()
