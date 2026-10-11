# fig_ch10.py —— Book7 ch10 制图：图 10.1 注册链路四门全景
# 用途：ch10 唯一正文图——「引擎怎么认出模型」的四条门路：①内置注册表（_VLLM_MODELS）
#       ②插件注册（general_plugins——vllm-ascend 16 项实例）③运行时外部注册（register_model/
#       model_class_overrides——大项目 A 路正身）④trust_remote_code→transformers backend 兼容层。
# 所属章节：Book7 ch10（§10.3）。
# 运行：cd <workspace> && source env.sh && python code/Book7-vLLM源码剖析上/ch10/fig_ch10.py
# 产物：drafts/Book7-vLLM源码剖析上/figures/fig-10-1-registration.png
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


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=6.2, lw=1.2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw, ls=ls,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


def arr(ax, x0, y0, x1, y1, color=MUTED, lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                 mutation_scale=10, zorder=2))


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.4, 7.4), dpi=300)
    ax.set_xlim(0, 19); ax.set_ylim(0, 11.4); ax.axis("off")
    ax.text(9.5, 11.05, "Four doors of model registration: how the engine recognizes a model (v0.30.0 anchors; door 3 = our project's route)",
            ha="center", fontsize=8.6, color=INK, weight="bold")
    # 入口
    box(ax, 6.6, 9.5, 5.8, 0.9, "config.json: \"architectures\": [\"...\"]\n(the key the engine looks up)", ec=INK, fs=6.8)
    # 四门
    doors = [
        ("DOOR 1  built-in table\n\n_VLLM_MODELS dict\nregistry.py #L723-733\n~396 entries (arch -> class)\nlazily imported", BLUE, "#dcebfc", 0.4),
        ("DOOR 2  plugin registration\n\nentry_points group\nvllm.general_plugins\nvllm-ascend registers 17 archs\n(models/__init__.py)", TEAL, "#d9f2e7", 5.2),
        ("DOOR 3  runtime external\n\nModelRegistry.register_model #L1057\nmodel_class_overrides (CLI)\nchokepoint: every worker re-registers\n-> OUR PROJECT'S ROUTE", ORANGE, "#fde3d6", 10.0),
        ("DOOR 4  transformers backend\n\nmodel_impl=auto falls back silently\nwhen arch not found anywhere\n-> TransformersForCausalLM wrapper\n(the trap of door 4)", YELLOW, "#fdf0d2", 14.8),
    ]
    for label, ec, fc, x in doors:
        box(ax, x, 5.4, 4.2, 3.2, label, ec=ec, fc=fc, fs=5.9)
        arr(ax, 9.5, 9.5, x + 2.1, 8.6)
    # 底部：权重加载链
    box(ax, 2.2, 2.6, 14.6, 1.9,
        "weight loading chain (all four doors converge here):\nDefaultModelLoader.load_model -> model.load_weights(weights)   default_loader.py #L427\n"
        "AutoWeightsLoader groups by prefix; WeightsMapper merges .q/.k/.v -> qkv_proj, .gate/.up -> gate_up_proj   llama.py #L345-354\n"
        "strict accounting: track_weights_loading raises on any missed key   #L447-471 (verified: RT=AN two-track identical)",
        ec=MUTED, fc="white", fs=6.0)
    for x in (2.5, 7.3, 12.1, 16.9):
        arr(ax, x, 5.4, x, 4.5)
    ax.text(9.5, 1.7, "big project map: C = door 1 reuse (hf207m dir, architectures=LlamaForCausalLM -> built-in class loads same weights)\n"
                      "A = door 3 (llama215_vllm.Llama215ForCausalLM via model_class_overrides; cross-route parity: identical outputs)",
            ha="center", fontsize=6.4, color=INK)
    ax.text(9.5, 0.55, "vllm-ascend's own 17 registrations walk door 2 - the official precedent for our door-3 route",
            ha="center", fontsize=6.2, color=MUTED, style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-10-1-registration.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print("[图 10.1] registration")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    main()
