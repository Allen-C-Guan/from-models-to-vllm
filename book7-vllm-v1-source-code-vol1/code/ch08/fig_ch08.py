# fig_ch08.py —— Book7 ch08 制图：图 8.1 编译-图执行管线 / 图 8.2 分派决策树
# 用途：ch08 两图——图 8.1=pytorch 模型→Dynamo 捕获→IR→分段子图（attention 留 eager）→piecewise 图
#       捕获→运行期 dispatch 五段管线（两端对照 NPU：AscendCompiler+ACLGraph+npugraph_ex）；
#       图 8.2=CudagraphDispatcher 的运行期决策树（BatchDescriptor→搜索序 FULL>PIECEWISE>None）。
# 所属章节：Book7 ch08（§8.2/§8.3）。
# 运行：cd <workspace> && source env.sh && python code/Book7-vLLM源码剖析上/ch08/fig_ch08.py
# 产物：drafts/Book7-vLLM源码剖析上/figures/fig-8-1-compile-pipeline.png / fig-8-2-dispatch-tree.png
# 图规：300dpi 白底；蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；图内英文图注中文；自绘+版本锚。
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "drafts", "Book7-vLLM源码剖析上", "figures"))
BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#0b0b0b", "#52514e"


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=6.4, lw=1.2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw, ls=ls,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


def arr(ax, x0, y0, x1, y1, color=MUTED, lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                 mutation_scale=10, zorder=2))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13.0, 6.8), dpi=300)
    ax.set_xlim(0, 18); ax.set_ylim(0, 10.4); ax.axis("off")
    ax.text(9, 10.05, "Compile-and-capture pipeline (upstream v0.30.0 top, NPU mirror bottom; sketched from source)",
            ha="center", fontsize=8.8, color=INK, weight="bold")
    STAGES = [
        ("PyTorch model\n(RMSNorm/SwiGLU/Attn...)", BLUE, "#dcebfc"),
        ("Dynamo trace\n(split at attention:\nunified_attention_with_output)", ORANGE, "#fde3d6"),
        ("IR / subgraphs\n(first / middle x(L-1) / last)\n3 unique shapes", TEAL, "#d9f2e7"),
        ("Piecewise capture\n(non-attn segments)\nattn stays eager", ORANGE, "#fde3d6"),
        ("Runtime dispatch\nper step: BatchDescriptor\n-> pick graph / eager", BLUE, "#dcebfc"),
    ]
    x = 0.5
    for label, ec, fc in STAGES:
        box(ax, x, 5.6, 3.1, 2.6, label, ec=ec, fc=fc, fs=6.2)
        if x < 13.5:
            arr(ax, x + 3.1, 6.9, x + 3.6, 6.9)
        x += 3.6
    ax.text(9, 4.9, "upstream: torch.compile backend = vllm/compilation/backends.py::CompilerManager #L124 + PiecewiseCompileInterpreter #L687   |   CUDAGraphWrapper cuda_graph.py#L145", ha="center", fontsize=6.4, color=MUTED)
    # NPU 镜像层
    NPU = [
        ("same stages...\nbut inductor DISABLED\nplatform.py#L91/#L1204", TEAL, "#d9f2e7"),
        ("AscendCompiler\ncompiler_interface.py#L226\n+ GraphFusionPassManager", ORANGE, "#fde3d6"),
        ("npugraph_ex FX layer\n(default ON in\nFULL/FULL_DECODE_ONLY)", YELLOW, "#fdf0d2"),
        ("ACLGraph capture\nacl_graph.py#L165\ntorch.npu.NPUGraph()", ORANGE, "#fde3d6"),
        ("same dispatcher\ncudagraph_dispatcher.py#L15\n(both tracks identical)", BLUE, "#dcebfc"),
    ]
    x = 0.5
    for label, ec, fc in NPU:
        box(ax, x, 1.2, 3.1, 2.5, label, ec=ec, fc=fc, fs=6.0)
        if x < 13.5:
            arr(ax, x + 3.1, 2.45, x + 3.6, 2.45)
        x += 3.6
    ax.text(9, 0.6, "key mirror: torch.cuda.CUDAGraph() cuda_graph.py#L285  <->  torch.npu.NPUGraph() acl_graph.py#L165   |   attn param update: acl_graph.py#L279 update_full_graph_params",
            ha="center", fontsize=6.2, color=MUTED, style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-8-1-compile-pipeline.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print("[图 8.1] compile-pipeline")


def fig_dispatch():
    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300)
    ax.set_xlim(0, 16); ax.set_ylim(0, 9.6); ax.axis("off")
    ax.text(8, 9.25, "Runtime dispatch decision (CudagraphDispatcher, search priority FULL > PIECEWISE > None; per cuda_graphs.md#L120)",
            ha="center", fontsize=8.4, color=INK, weight="bold")
    box(ax, 5.2, 7.6, 5.6, 1.0, "per-step BatchDescriptor\n(num_tokens, num_reqs, uniform, has_lora)\nforward_context.py", ec=INK, fs=6.6)
    box(ax, 5.2, 5.9, 5.6, 0.95, "dispatch(key)  #L307-324\nsearch existing keys by priority", ec=ORANGE, fc="#fde3d6", fs=6.6)
    # 三分支
    box(ax, 0.6, 3.4, 4.4, 1.7, "FULL hit?\nreplay full graph\n(whole step one replay;\nuniform decode only)", ec=BLUE, fc="#dcebfc", fs=6.4)
    box(ax, 5.8, 3.4, 4.4, 1.7, "PIECEWISE hit?\nrun compiled subgraphs;\nattn segment eager\n(+ params update)", ec=TEAL, fc="#d9f2e7", fs=6.4)
    box(ax, 11.0, 3.4, 4.4, 1.7, "neither -> None\neager execution\n(no graph this step)", ec=MUTED, fc="white", fs=6.4)
    arr(ax, 6.5, 5.9, 2.8, 5.1); arr(ax, 8.0, 5.9, 8.0, 5.1); arr(ax, 9.5, 5.9, 13.2, 5.1)
    box(ax, 0.6, 1.2, 14.8, 1.5,
        "NPU notes: default FULL_AND_PIECEWISE (O2 preset -> vllm/config/vllm.py#L318/#L349, both tracks identical);\n"
        "Npugraph_ex ON in FULL modes (graph_mode.md#L132-139); attn param update keeps seq_lens fresh without re-capture\n"
        "(acl_graph.py#L279/#L316-336 -> attention_v1.py#L520); ASCEND_LAUNCH_BLOCKING=1 is mutually exclusive (platform.py#L1246-1253)",
        ec=YELLOW, fc="#fdf0d2", fs=6.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-8-2-dispatch-tree.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print("[图 8.2] dispatch-tree")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_pipeline()
    fig_dispatch()
