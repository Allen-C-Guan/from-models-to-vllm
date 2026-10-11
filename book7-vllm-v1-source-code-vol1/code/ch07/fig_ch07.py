# fig_ch07.py —— Book7 ch07 制图：图 7.1 执行层三层 / 图 7.2 InputBatch 行视图
# 用途：ch07 两图——图 7.1=executor/worker/runner 三层×上游/NPU 双列（三态、七函数、双 runner 分派）；
#       图 7.2=持久态槽位表+逐步视图+InputBuffers（例 7.1 的 A/B/C 剧情标注）。
# 所属章节：Book7 ch07（§7.2/§7.4）。
# 运行：cd <workspace> && source env.sh && python code/Book7-vLLM源码剖析上/ch07/fig_ch07.py
# 产物：drafts/Book7-vLLM源码剖析上/figures/fig-7-1-exec-layers.png / fig-7-2-input-batch.png
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


# ---------------- 图 7.1：执行层三层 ----------------
def fig_exec_layers():
    fig, ax = plt.subplots(figsize=(13.0, 7.2), dpi=300)
    ax.set_xlim(0, 18); ax.set_ylim(0, 11.2); ax.axis("off")
    ax.text(9, 10.85, "Execution layer: executor -> worker -> model runner (upstream v0.30.0 left, NPU right; sketched from source)",
            ha="center", fontsize=8.8, color=INK, weight="bold")
    # 顶：输入输出
    box(ax, 0.6, 9.3, 7.2, 0.8, "SchedulerOutput  (from EngineCore.step, sched/output.py #L225)", ec=INK, fs=6.8)
    box(ax, 10.2, 9.3, 7.2, 0.8, "ModelRunnerOutput -> scheduler.update_from_output", ec=INK, fs=6.8, ls="--")
    # 层1 executor
    box(ax, 0.6, 6.9, 7.2, 2.0,
        "L1  executor  abstract.py::Executor #L38\nget_class #L49: 'uni' | 'mp' | 'ray' | class path\n\nuni: driver_worker in-proc direct call #L103\nmp: rpc_broadcast_mq shm broadcast #L420\n     worker_busy_loop #L1029 (3 lines)\nray: -> Book 8 ch.6 (name only)", ec=BLUE, fc="#dcebfc", fs=6.2)
    box(ax, 10.2, 6.9, 7.2, 2.0,
        "L1  (platform-agnostic, same code)\n\nNPU inherits nothing here - executor\nis the same layer on every backend", ec=BLUE, fc="#dcebfc", fs=6.4)
    # 层2 worker
    box(ax, 0.6, 4.2, 7.2, 2.3,
        "L2  worker  gpu_worker.py::Worker #L182\nminimal face (plugin_system L109-117):\ninit_device #L360 (property)\nload_model #L500 | determine_mem #L528\nget_kv_cache_spec #L719 | init_from_cfg #L735\ncompile_or_warm_up #L773 | execute_model #L1149", ec=ORANGE, fc="#fde3d6", fs=6.0)
    box(ax, 10.2, 4.2, 7.2, 2.3,
        "L2  NPUWorker  worker.py #L114\nsame 7 functions:\ninit_device #L482 | load_model #L730\nget_kv_cache_spec #L994 | det_mem #L540\nexecute_model #L649 | ...\n+ kernel_warmup #L774 (NPU extra)", ec=ORANGE, fc="#fde3d6", fs=6.0)
    # 层3 runner
    box(ax, 0.6, 1.3, 7.2, 2.5,
        "L3  model runner\ngpu_worker #L469: use_v2_model_runner ?\n  MRV2 gpu/model_runner.py #L187 (main read)\n  execute_model #L1631 | sample #L1512\n  MRV1 gpu_model_runner.py #L498 (contrast)\n  execute_model #L4187 | _sample #L3658", ec=TEAL, fc="#d9f2e7", fs=6.0)
    box(ax, 10.2, 1.3, 7.2, 2.5,
        "L3  NPU dual runner\nworker.py #L491 dispatch:\n  v2/model_runner.py #L105 (inherits MRV2)\n  model_runner_v1.py #L304 (inherits MRV1)\nDEFAULT = MRV1 (patch env-only\npatch_use_v2_model_runner.py #L11-23)", ec=TEAL, fc="#d9f2e7", fs=6.0)
    for x in (4.2, 13.8):
        arr(ax, x, 9.3, x, 8.9); arr(ax, x, 6.9, x, 6.5); arr(ax, x, 4.2, x, 3.8)
        arr(ax, x + 1.0, 3.8, x + 1.0, 4.2, color=TEAL, ls="--")
    box(ax, 10.2, 0.3, 7.2, 0.75, "observed: [model_runner_v1.py:4009] in smoke log = cheapest runner probe", ec=YELLOW, fc="#fdf0d2", fs=6.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-7-1-exec-layers.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print("[图 7.1] exec-layers")


# ---------------- 图 7.2：InputBatch 行视图 ----------------
def fig_input_batch():
    fig, ax = plt.subplots(figsize=(12.0, 6.6), dpi=300)
    ax.set_xlim(0, 17); ax.set_ylim(0, 10.0); ax.axis("off")
    ax.text(8.5, 9.65, "Persistent slots + per-step view (MRV2 RequestState + InputBatch; numbers = toy Example 7.1, max_num_reqs=4)",
            ha="center", fontsize=8.6, color=INK, weight="bold")
    # 左：槽位表
    ax.text(1.0, 8.7, "persistent state  (states.py::RequestState #L12)", fontsize=7.6, color=TEAL, weight="bold")
    for i, (slot, who) in enumerate([("slot 3", "A (then C: LIFO reuse)"), ("slot 2", "B"), ("slot 1", ""), ("slot 0", "")]):
        y = 6.9 - i * 1.05
        occ = who != ""
        box(ax, 1.0, y, 6.4, 0.9, f"{slot}: {who}" if occ else f"{slot}: free", ec=TEAL if occ else MUTED,
            fc="#d9f2e7" if occ else "white", fs=6.6, lw=1.3 if occ else 1.0, ls="-" if occ else "--")
    ax.text(4.2, 2.55, "free_indices = [0, 1] after A,B  ->  pop() from tail: C gets slot 3\nadd #L100 pop | remove #L132 append | no compaction ever", ha="center", fontsize=6.4, color=MUTED)
    # 右：视图
    ax.text(9.0, 8.7, "per-step view  (gpu/input_batch.py::InputBatch #L44)", fontsize=7.6, color=ORANGE, weight="bold")
    box(ax, 9.0, 6.5, 7.4, 1.6,
        "idx_mapping = [2, 3]   (B row 0, A row 1)\nnum_scheduled_tokens = [1, 5]  (B decode, A chunk-5)\nquery_start_loc = [0, 1, 6]  (CSR offsets)\nlogits_indices = [0, 5]  (B decode row + A chunk-end row)", ec=ORANGE, fc="#fde3d6", fs=6.4)
    box(ax, 9.0, 4.2, 7.4, 1.6,
        "input_ids / positions  [num_tokens=6]\nGPU buffers pre-allocated (InputBuffers #L19):\ninput_ids, positions, is_padding,\nquery_start_loc, seq_lens  (fixed shapes)", ec=BLUE, fc="#dcebfc", fs=6.4)
    arr(ax, 9.0, 6.0, 7.4, 5.6, color=ORANGE)
    ax.text(8.1, 6.15, "view rows point at slots", fontsize=6.2, color=ORANGE, ha="center")
    box(ax, 9.0, 1.6, 7.4, 1.9,
        "hidden_states [num_tokens, d]\n  -> hidden_states[logits_indices] = [num_logits, d]\n  -> compute_logits -> [num_logits, V] -> sampler\n(prefill's 5 tokens computed, only 1 logit row read)", ec=TEAL, fc="#d9f2e7", fs=6.2)
    arr(ax, 12.7, 4.2, 12.7, 3.5, color=TEAL)
    ax.text(8.5, 0.6, "MRV1 contrast: gpu_input_batch InputBatch IS the input; remove_request must condense (slide rows, #L526 docstring) - the disease MRV2's slot+view split cures",
            ha="center", fontsize=6.4, color=MUTED, style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-7-2-input-batch.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print("[图 7.2] input-batch")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_exec_layers()
    fig_input_batch()
