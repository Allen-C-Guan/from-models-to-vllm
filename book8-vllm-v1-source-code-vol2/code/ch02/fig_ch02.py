# fig_ch02.py —— Book8 ch02 制图：图 2.1 draft-verify 时序 / 图 2.2 ngram 扫描（数据到位后渲染）
# 用途：图 2.1=一步 draft-propose-verify-accept/reject 的三泳道时序（站 8/12/13 坐标+形状标注+回滚边界+bonus）；
#       图 2.2=实验 A 的 k×窗口→τ/加速比 读数面（读 log/book8-ch02 底档）。
# 所属章节：Book8 ch2（§2.2/§2.6）；设计书=notes/11 §三 例 2.1/§四 实验 A（勘误 1 修正口径）。
# 运行：cd <workspace> && source env.sh && python code/Book8-vLLM源码剖析下/ch02/fig_ch02.py [--only fig1|fig2]
# 产物：drafts/Book8-vLLM源码剖析下/figures/fig-2-1-draft-verify.png / fig-2-2-ngram-sweep.png
# 图规：300dpi 白底；蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；图内英文图注中文；自绘+版本锚。
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIG_DIR = os.path.join(WS, "drafts", "Book8-vLLM源码剖析下", "figures")
LOG = os.path.join(WS, "log/book8-ch02")
BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e1e0d9"


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=6.4, lw=1.2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw, ls=ls,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


def arr(ax, x0, y0, x1, y1, color=MUTED, lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                 mutation_scale=10, zorder=2))


# ---------------- 图 2.1：draft-verify 时序（三泳道） ----------------
def fig_draft_verify():
    fig, ax = plt.subplots(figsize=(12.6, 7.0), dpi=300)
    ax.set_xlim(0, 18); ax.set_ylim(0, 10.6); ax.axis("off")
    ax.text(9, 10.25, "One speculative decode step: propose -> verify -> accept/reject (sketched from vLLM v0.30.0 source)",
            ha="center", fontsize=8.8, color=INK, weight="bold")
    lanes = [(1.2, "Scheduler (station 8)\nlookahead slots k=3\nrollback bookkeeping", BLUE, "#dcebfc"),
             (7.0, "Drafter / proposer (station 12)\nngram | draft_model | eagle ...", TEAL, "#d9f2e7"),
             (12.8, "Target verify forward +\nrejection sampler (station 12-13)\n(B,n,d) -> (B,n+k,d)", ORANGE, "#fde3d6")]
    for x, t, ec, fc in lanes:
        box(ax, x, 8.6, 4.6, 1.3, t, ec=ec, fc=fc, fs=6.6)
    # 时间轴上的五拍
    steps = [(1.2, "① reserve slots\n1 real + k draft\nper request", BLUE),
             (4.35, "② propose k candidates\n+ draft probs", TEAL),
             (7.5, "③ verify forward:\nappend drafts, one pass", ORANGE),
             (10.65, "④ per-position verdict:\naccept prefix / reject rest\n+ bonus if all-hit", ORANGE),
             (13.8, "⑤ bookkeeping:\nnum_accepted = out-1\nrollback if rejected", BLUE)]
    for x, t, ec in steps:
        box(ax, x, 5.4, 2.9, 2.0, t, ec=ec, fs=6.0)
    for x0, x1 in ((4.1, 4.35), (7.25, 7.5), (10.4, 10.65), (13.55, 13.8)):
        arr(ax, x0, 6.4, x1, 6.4)
    arr(ax, 3.5, 8.6, 3.5, 7.4, color=BLUE)          # scheduler -> ②? 视觉主干
    arr(ax, 9.3, 8.6, 9.3, 7.4, color=TEAL)
    arr(ax, 15.1, 8.6, 15.1, 7.4, color=ORANGE)
    # 回滚箭头（⑤→① 虚线）
    arr(ax, 15.25, 5.4, 2.65, 5.4, color=YELLOW, lw=1.4, ls="--")
    ax.text(8.95, 5.05, "reject -> num_computed_tokens -= num_rejected  (scheduler.py #L1989-1996, recompute not delete)",
            ha="center", fontsize=6.2, color="#8a6d1a")
    # 底部读法行
    box(ax, 1.2, 2.6, 15.5, 1.9,
        "workload of one step (example 2.1): B=2, n=10, k=3\n"
        "req A: accept pos 1-2, reject pos 3 -> 3 tokens out (2 accepted + 1 resampled)\n"
        "req B: all 3 accepted + bonus -> 4 tokens out (cap k+1)\n"
        "metrics: drafts=2, draft_tokens=6, accepted=5 -> mean acc len = 1+5/2 = 3.5 (bonus-inclusive), per-pos=[1.0,1.0,0.5]",
        ec=INK, fs=6.2)
    box(ax, 1.2, 0.6, 7.4, 1.5, "bonus token: sampled from target dist\nafter a full hit; NOT in num_accepted\n(it is the +1 in mean acceptance length)", ec=YELLOW, fc="#fdf0d2", fs=6.0)
    box(ax, 9.3, 0.6, 7.4, 1.5, "NPU fact: both runnable routes (ngram, draft_model)\nare MRV1-only at v0.30.0 (config/vllm.py #L2839-2849);\nverify-side math identical across runners", ec=TEAL, fc="#d9f2e7", fs=6.0)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-2-1-draft-verify.png")
    fig.savefig(out, facecolor="white"); plt.close(fig)
    print("[fig]", out)


# ---------------- 图 2.2：ngram 扫描（读实验 A 底档） ----------------
def fig_ngram_sweep():
    def counters(arm):
        p = os.path.join(LOG, f"a_{arm}_metrics.txt")
        if not os.path.exists(p):
            return None
        d = {}
        for line in open(p, encoding="utf-8"):
            parts = line.split()
            if parts and parts[0].startswith("vllm:spec_decode_") and parts[0].split("{")[0].endswith("_total"):
                d[parts[0].split("{")[0].split(":")[1].replace("_total", "")] = float(parts[1])
        return d or None

    def thr(arm):
        p = os.path.join(LOG, f"a_{arm}.json")
        if not os.path.exists(p):
            return None
        try:
            return json.load(open(p, encoding="utf-8")).get("request_throughput")
        except Exception:
            return None

    arms = [("k1_w24_echo", 1, (2, 4)), ("k1_w48_echo", 1, (4, 8)),
            ("k3_w24_echo", 3, (2, 4)), ("k3_w48_echo", 3, (4, 8)), ("k3_w28_echo", 3, (2, 8))]
    rows = []
    for arm, k, w in arms:
        c = counters(arm)
        t = thr(arm)
        if c and c.get("spec_decode_num_drafts"):
            tau = 1 + c["spec_decode_num_accepted_tokens"] / c["spec_decode_num_drafts"]
            rows.append((arm, k, w, tau, t))
    if not rows:
        print("[fig2] 无数据（实验 A 未完成）——跳过")
        return
    base = thr("off_echo")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=300)
    ks = sorted({r[1] for r in rows})
    for ax, wi, wlabel in ((axes[0], 0, "window (2,4)"), (axes[1], 1, "window (4,8)")):
        xs, taus, spds = [], [], []
        for arm, k, w, tau, t in rows:
            if w[wi] == (2 if wi == 0 else 4) or w == ((2, 4) if wi == 0 else (4, 8)) or arm == "k3_w28_echo":
                pass
        # 简化：按窗口分两组画 τ 与加速比
        sel = [r for r in rows if r[2] == ((2, 4) if wi == 0 else (4, 8))]
        if wi == 1:
            sel += [r for r in rows if r[2] == (2, 8)]
        xs = [f"k={r[1]}" for r in sel]
        taus = [r[3] for r in sel]
        ax.bar([i - 0.18 for i in range(len(sel))], taus, width=0.36, color=BLUE, label="mean acc length")
        if base:
            spds = [(r[4] / base if r[4] and base else 0) for r in sel]
            ax.bar([i + 0.18 for i in range(len(sel))], spds, width=0.36, color=ORANGE, label="speedup vs off")
        ax.set_xticks(range(len(sel))); ax.set_xticklabels(xs, fontsize=7)
        ax.set_title(wlabel + (" +default (2,8)" if wi == 1 else ""), fontsize=8)
        ax.grid(axis="y", color=GRIDC, lw=0.5, zorder=0)
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.legend(fontsize=6.5, frameon=False)
    fig.suptitle("ngram sweep on hf207m (echo workload, 32 reqs, greedy; runtime stack 0.27.1 + vllm-ascend, Atlas 910B3)",
                 fontsize=8.6, color=INK)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-2-2-ngram-sweep.png")
    fig.savefig(out, facecolor="white"); plt.close(fig)
    print("[fig]", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["fig1", "fig2"], default=None)
    a = ap.parse_args()
    if a.only in (None, "fig1"):
        fig_draft_verify()
    if a.only in (None, "fig2"):
        fig_ngram_sweep()
