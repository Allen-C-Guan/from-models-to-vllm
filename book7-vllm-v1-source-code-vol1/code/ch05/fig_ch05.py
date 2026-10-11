# fig_ch05.py —— Book7 ch05 制图：图 5.1 schedule() 决策循环流程图
# 用途：ch05《调度器》唯一正文图——schedule() 的两级预算+双队列循环+抢占内环+出账与乐观记账：
#       RUNNING 循环（蓝，#L612 起，内嵌 allocate_slots→None 的抢占内环红系）→ not-preempted 闸 →
#       WAITING 循环（橙，#L858 起，三道闸+前缀命中查找侧挂——ch6 埋线）→ SchedulerOutput（青）→
#       乐观记账回箭头指回 Request 台账。
# 所属章节：Book7 ch05（§5.3）。
# 运行：cd <workspace> && source env.sh && python code/Book7-vLLM源码剖析上/ch05/fig_ch05.py
# 产物：drafts/Book7-vLLM源码剖析上/figures/fig-5-1-schedule-loop.png
# 图规：300dpi 白底；蓝#2a78d6 橙#eb6834 青#1baf7a 红#c0392b；图内英文图注中文；自绘+版本锚。
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "drafts", "Book7-vLLM源码剖析上", "figures"))
BLUE, ORANGE, TEAL, RED = "#2a78d6", "#eb6834", "#1baf7a", "#c0392b"
INK, MUTED = "#0b0b0b", "#52514e"


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=6.6, lw=1.2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw, ls=ls,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


def dia(ax, x, y, w, h, label, ec=RED, fs=6.0):
    ax.plot([x + w / 2, x + w, x + w / 2, x, x + w / 2], [y + h, y + h / 2, y, y + h / 2, y + h],
            color=ec, lw=1.2, zorder=3)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


def arr(ax, x0, y0, x1, y1, color=MUTED, lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                 mutation_scale=10, zorder=2))


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 7.4), dpi=300)
    ax.set_xlim(0, 20); ax.set_ylim(0, 11.6); ax.axis("off")
    ax.text(10, 11.25, "The schedule() decision loop: two budgets, two queues, fill-until-full (sketched from v0.30.0, scheduler.py #L561-1305)",
            ha="center", fontsize=9.0, color=INK, weight="bold")
    # 入口
    box(ax, 7.9, 10.0, 4.2, 0.9, "schedule(throttle_prefills)  #L561", ec=INK, fs=7.2)
    # 出纳台
    box(ax, 7.9, 8.55, 4.2, 1.05, "budget desk  #L579-584\ntoken_budget | draft_slots | input_budget", ec=YELLOW if (YELLOW := "#eda100") else YELLOW, fc="#fdf0d2", fs=6.6)
    arr(ax, 10.0, 10.0, 10.0, 9.6)
    # RUNNING 循环
    box(ax, 0.6, 5.7, 8.6, 2.5,
        "RUNNING loop  #L612-852 (first)\nwhile req_index < len(running) and token_budget > 0:\n  num_new_tokens = num_tokens_with_spec\n                - num_computed_tokens   (the gap)\n  min(gap, token_budget, input_budget-draft_slots)\n  blocks = kv_cache.allocate_slots(...)  #L729",
        ec=BLUE, fc="#dcebfc", fs=6.6)
    # 抢占内环
    dia(ax, 1.0, 3.6, 3.0, 1.5, "blocks is None?\n(KV full)", ec=RED)
    box(ax, 4.6, 3.5, 4.4, 1.7,
        "preempt inner loop  #L745-799\nvictim = running[-1] (FCFS)  #L753\n_preempt_request: free blocks /\nnum_computed=0 / prepend to waiting\n#L1493/#L1496/#L1517",
        ec=RED, fc="white", fs=6.0)
    arr(ax, 3.2, 5.7, 2.5, 5.1, color=RED)
    arr(ax, 4.0, 4.35, 4.6, 4.35, color=RED)
    arr(ax, 6.8, 5.2, 5.8, 5.7, color=RED, ls="--")
    ax.text(7.4, 5.35, "retry allocate", fontsize=5.8, color=RED)
    # 闸
    dia(ax, 9.7, 6.15, 3.1, 1.4, "preempted_reqs\nempty?  #L855", ec=MUTED)
    arr(ax, 9.2, 6.95, 9.7, 6.85)
    ax.text(11.3, 5.75, "no new requests\nafter a preemption step", fontsize=5.8, color=MUTED, ha="center")
    # WAITING 循环
    box(ax, 13.0, 4.6, 6.4, 3.6,
        "WAITING loop  #L858-1305 (second)\nwhile (waiting or skipped) and token_budget>0:\n  gate1: input_budget > draft_slots  #L861\n  gate2: num_running < max_num_seqs  #L863\n  prefix lookup: get_computed_blocks\n        (ch6 hook)  #L1159\n  blocks = allocate_slots(new_tokens,\n        num_new_computed, blocks)\n  None -> break (waiting never preempts)",
        ec=ORANGE, fc="#fde3d6", fs=6.0)
    arr(ax, 12.8, 6.85, 13.0, 6.6)
    # 出账
    box(ax, 5.2, 1.6, 7.0, 1.5,
        "SchedulerOutput  #L225 (output.py)\nnum_scheduled_tokens: {req_id: n}  <- the V1-alpha 'dict'\nscheduled_new/cached_reqs | preempted_req_ids",
        ec=TEAL, fc="#d9f2e7", fs=6.4)
    arr(ax, 4.0, 3.5, 5.8, 3.1, color=TEAL)
    arr(ax, 15.5, 4.6, 12.2, 3.1, color=TEAL)
    # 乐观记账回箭头
    box(ax, 13.6, 0.4, 5.6, 0.95, "optimistic bookkeeping  #L1520-1534\nnum_computed_tokens += scheduled (corrected later)", ec=MUTED, fc="white", fs=6.0)
    arr(ax, 12.2, 2.1, 13.6, 1.0, color=MUTED, ls="--")
    ax.text(10.0, 0.9, "requests ledger: each row's num_computed_tokens chases num_tokens_with_spec (NOTE #L563-572)",
            ha="center", fontsize=6.4, color=MUTED, style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-5-1-schedule-loop.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print("[图 5.1] schedule-loop")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    main()
