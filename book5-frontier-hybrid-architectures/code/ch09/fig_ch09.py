# fig_ch09.py —— Book5 ch9 制图：图 9.1 两路线谱系 / 图 9.2 三刀机结构与插槽来源 /
#                图 9.3 三刀机冒烟+短训曲线 / 图 9.4 recall 等参三臂（编号=正文出现顺序；slug=hybrid3to1）
# 用途：图 9.1 = 9.2 节谱系总图（2025-09→2026-09 两条注意力路线的时间轴与合流；每框带证据等级标签；
#         GLM-5.3-Flash 虚线=冻结线后）；
#       图 9.2 = 9.5 节三刀机整机结构（12 层 [gdn,gdn,gdn,mla]×3；每格右标插槽供货方=「三刀改装史
#         写在 import 语句里」；GDN/MLA 两类层的 KV/状态面与形状标注；层号双口径 0 起/人读 1 起）；
#       图 9.3 = 9.5 节冒烟+整合短训（喂 log/book5-ch09/curve_{smoke_s1,integrated_run1}.csv；
#         左=smoke 20 步、右=integrated 100 步；ln V=10.3735 与预言 ln V+dσ²/2=10.5783 两基准线）；
#       图 9.4 = 9.6 节 recall 消融（喂 log/book5-ch09/recall3_run1.json + recall_ckpt_*.pt 尾窗曲线：
#         左=续跑段 6001-10000 三臂 loss+ln(64)=4.159 值先验平台线；右=@10000 五位置 acc 三臂条形）。
# 所属章节：Book5 第 9 章（图 9.1-9.4）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch09/fig_ch09.py" [--which all]
# 产物：figures/fig-9-{1..4}-hybrid3to1.png
# 图规：300dpi 白底；系列色 蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；线宽 2；图内英文、图注中文；禁双 y 轴。
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "book5-frontier-hybrid-architectures", "figures")
LOG_DIR = os.path.join(REPO_ROOT, "log", "book5-ch09")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MUTED = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
EV_TAG = {"paper": "paper", "cfg": "config+card", "prev": "preview paper", "rep": "tech report"}


def _box(ax, x, y, w, h, text, fc, ec, fontsize=7.2, dashed=False, tc=DARK, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=lw, linestyle="--" if dashed else "-",
                                mutation_aspect=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=tc, linespacing=1.35)


def fig91():
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 7.3)
    ax.axis("off")
    ax.text(5.4, 7.05, "Two escape routes, then convergence: 2024-2026", ha="center",
            fontsize=10, color=DARK, weight="bold")
    # 两条泳道标牌
    ax.plot([0.10, 0.10], [3.55, 6.60], color=TEAL, lw=2.4)
    ax.plot([0.10, 0.10], [0.35, 3.30], color=BLUE, lw=2.4)
    ax.text(0.24, 6.35, "Route 1  linear : global = 3:1\n(swap the operator)", fontsize=8.2,
            color=TEAL, weight="bold", va="top")
    ax.text(0.24, 3.18, "Route 2  sparse indexing\n(learn the participant set)", fontsize=8.2,
            color=BLUE, weight="bold", va="top")
    W, H = 2.75, 0.98
    r1 = [  # (x, y, text, tag)
        (1.05, 5.42, "Jamba  2024-03\nTransformer:Mamba = 1:7 + MoE\nfirst large-scale hybrid\n(52B/12B, ratio cited 2nd-hand)", "paper"),
        (1.05, 4.10, "Qwen3-Next-80B-A3B  2025-09\nGDN + GQA full attn\n3:1 (36:12 / 48 layers)\nMoE 512e top-10 (51:1) | 262K", "config"),
        (4.15, 4.10, "Kimi-Linear-48B-A3B  2025-10\nKDA + MLA\n3:1 (full every 4 + last)\nMoE 256e top-8 (32:1) | 1M", "paper"),
        (4.15, 5.42, "Qwen3.5  2026\nGDN(32V/16QK) + GQA 16:2\n3:1 (30:10 / 40) | 262K native\n/ 1.01M hosted | MoE 32:1", "config+card"),
        (7.30, 5.42, "Kimi K3  2026\nKDA 69 + Gated MLA 24 / 93\nMoE 896e top-16 (56:1)\n1M | MXFP4 native", "tech report"),
    ]
    r2 = [
        (1.05, 2.24, "DeepSeek-V3.2  2025-09\nMLA + DSA debut (topk 2048)\nall layers indexed\n160K | MoE of Book4", "paper"),
        (4.15, 1.06, "GLM-5  2026-02\nMLA + DSA, all 78 layers\nMoE 256e top-8 (32:1) | 198K\n744B / 40B active (paper)", "paper"),
        (7.30, 2.24, "DSV4 Flash/Pro  2026-06\nCSA(m=4)+HCA(m=128)+SWA128\nMoE 256e/384e top-6 (43-64:1)\n1M | FP4 experts + FP8", "preview"),
        (7.30, 0.62, "GLM-5.2  2026-06\n+ IndexShare (indexer 1:4)\n1M | IndexCache 2603.12201", "config+card"),
    ]
    for boxes, c in ((r1, TEAL), (r2, BLUE)):
        fc = "#f2f9f6" if c == TEAL else "#f3f7fd"
        for (x, y, txt, tag) in boxes:
            _box(ax, x, y, W, H, txt, fc, c, fontsize=6.1)
            ax.text(x + W - 0.03, y + H - 0.05, tag, ha="right", va="top", fontsize=5.6,
                    color=MUTED, style="italic")
    arrows = [((2.42, 5.42), (2.42, 5.08)),          # Jamba -> Qwen3-Next
              ((3.80, 4.59), (4.15, 4.59)),          # Qwen3-Next -> Kimi-Linear
              ((3.62, 5.08), (4.15, 5.60)),          # Qwen3-Next -> Qwen3.5
              ((6.90, 4.59), (7.30, 5.60)),          # Kimi-Linear -> K3
              ((2.42, 2.72), (7.30, 2.72)),          # DSV3.2 -> DSV4（掠过 GLM-5 上方）
              ((3.45, 2.24), (4.40, 2.04)),          # DSV3.2 -> GLM-5
              ((6.90, 1.55), (7.30, 1.11))]          # GLM-5 -> GLM-5.2
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0))
    # 合流：GLM-5.3-Flash（冻结线后，虚线黄框）+ 双向来箭头
    _box(ax, 7.30, 3.48, W, 0.80,
         "GLM-5.3-Flash  2026-09  (post-freeze)\nKDA x3 + DSA x1 in ONE model (34:11 / 45)\nboth routes merged | 1M | FP8",
         "#fdfaf2", YELLOW, fontsize=6.1, dashed=True)
    ax.annotate("", xy=(8.67, 4.28), xytext=(8.67, 5.42),
                arrowprops=dict(arrowstyle="-|>", color=YELLOW, lw=1.3, linestyle="--"))
    ax.annotate("", xy=(8.67, 3.48), xytext=(8.67, 3.22),
                arrowprops=dict(arrowstyle="-|>", color=YELLOW, lw=1.3, linestyle="--"))
    ax.text(8.86, 3.88, "routes\nmeet", fontsize=6.2, color=YELLOW, va="center")
    ax.text(0.24, 0.32, "recipe = attention route + ultra-sparse MoE + long context + low precision;"
                        "  box tag = evidence level; dashed = post-freeze",
            fontsize=6.2, color=MUTED, style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-9-1-hybrid3to1-genealogy.png"), facecolor="white")
    plt.close(fig)


def fig92():
    fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=300)
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    ax.text(5.4, 7.38, "hybrid409: the three-knife machine  (pattern [gdn, gdn, gdn, mla] x 3)",
            ha="center", fontsize=10, color=DARK, weight="bold")

    def tbox(x, y, text, fc, ec, fs=6.4, dashed=False, ha="left"):
        return ax.text(x, y, text, fontsize=fs, color=DARK, ha=ha, va="top", linespacing=1.5,
                       bbox=dict(boxstyle="round,pad=0.45", fc=fc, ec=ec, lw=1.4,
                                 linestyle="--" if dashed else "-"))

    # 左：12 层栈（层 0 在底、残差河纵贯）
    x0, w = 0.55, 3.30
    for i in range(12):
        y = 0.85 + i * 0.52
        kind = "mla" if i % 4 == 3 else "gdn"
        fc = "#fdeee6" if kind == "mla" else "#eafaf4"
        c = ORANGE if kind == "mla" else TEAL
        ax.add_patch(FancyBboxPatch((x0, y), w, 0.42, boxstyle="round,pad=0.01",
                                    fc=fc, ec=c, lw=1.4))
        ax.text(x0 + 0.10, y + 0.21, f"{i:>2} | {i + 1:>2}", fontsize=6.2, color=MUTED,
                va="center", ha="left")
        ax.text(x0 + w - 0.10, y + 0.21, kind.upper() + "  (B,n,1024) -> (B,n,1024)",
                fontsize=6.6, color=c, va="center", ha="right", weight="bold")
    ax.text(x0 + w / 2, 0.42, "residual river x: (B, n, 1024);  every layer: MoE w=810",
            ha="center", fontsize=6.8, color=DARK)
    ax.text(x0 - 0.08, 0.85, "layer idx\n0-based |\nhuman 1-based", fontsize=5.8, color=TICK,
            va="bottom", ha="right", linespacing=1.3)
    ax.annotate("", xy=(0.32, 7.05), xytext=(0.32, 0.85),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    ax.text(0.18, 3.95, "depth", rotation=90, fontsize=6.4, color=TICK, va="center")

    # 右：四张信息卡（bbox 自动包字）
    tbox(4.35, 7.05,
         "slot suppliers  =  the import chain\n"
         "RMSNorm / SwiGLU / RoPE  <-  Book3 ch2/ch3/ch4  (via llama409)\n"
         "DeepSeekMoE  w 938->810  <-  Book4 ch5   (1st knife)\n"
         "MLA_LITE (d_c 256 + rope 64 / token)  <-  Book4 ch6   (2nd knife)\n"
         "GatedDeltaNet  h_k=h_v=16, d_k=128, d_v=64  <-  Book5 ch5   (3rd knife)",
         "#faf9f6", MUTED, fs=6.3)
    tbox(4.35, 5.62,
         "MLA layers x3  (0-idx 3/7/11 = human 4/8/12; last-global = K3 convention)\n"
         "KV = (d_c 256 + d_h^R 64) = 320 elems / token / layer -> 960 / token:  UNBOUNDED in n\n"
         "GDN layers x9: fixed  h_v*d_k*d_v = 131,072 (state S: (16, 128, 64))\n"
         "               + (k_conv-1) * conv_dim = 3 * 5,120 = 15,360 (causal conv tail)\n"
         "               = 146,432 / layer  ->  9 layers = 1,317,888:  INDEPENDENT of n",
         "#fdeee6", ORANGE, fs=6.3)
    tbox(4.35, 4.05,
         "parameter account  (assert: enumeration = hand account = constant)\n"
         "GDN attn 7,393,312 x 9   +   MLA attn 2,687,488 x 3   +   MoE(w=810) 22,403,072 x 12\n"
         "+ 2x RMSNorm 2,048 x 12  +  final norm 1,024  +  embed & untied head 65,537,024\n"
         "TOTAL 409,000,736    vs two-knife cand2 409,115,648   (gap -114,912 = -0.028%)",
         "#f2f9f6", TEAL, fs=6.3)
    tbox(4.35, 2.72,
         "crossover  n* = 1,317,888 / 960  ~  1,373\n"
         "for n > n*, all 9 GDN states stay flat while the 3 MLA layers keep growing linearly",
         "#fdfaf2", YELLOW, fs=6.5)
    tbox(4.35, 1.90,
         "config in one line:\n"
         "d=1024, L=12, h=16, V=32000 untied, E8 top-2 + 1 shared, w=810,\n"
         "GDN {h_k=16, h_v=16, d_k=128, d_v=64, conv 4, chunk 64}, theta=1e4",
         "#f7f7f7", TICK, fs=6.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-9-2-hybrid3to1-threeknife.png"), facecolor="white")
    plt.close(fig)


def _read_curve(name):
    rows = []
    with open(os.path.join(LOG_DIR, name), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("train_loss"):
                rows.append((int(r["step"]), float(r["train_loss"]),
                             float(r["eval_loss"]) if r.get("eval_loss") else None))
    evals = [(int(r["step"]), float(r["eval_loss"])) for r in csv.DictReader(
        open(os.path.join(LOG_DIR, name), newline="", encoding="utf-8")) if r.get("eval_loss")]
    return rows, evals


def fig93():
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    for ax, name, title in ((axes[0], "curve_smoke_s1.csv", "smoke: 20 steps"),
                            (axes[1], "curve_integrated_run1.csv", "integrated: 100 steps")):
        rows, evals = _read_curve(name)
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        ax.plot(xs, ys, "o-", color=BLUE, lw=2, ms=5, label="train loss")
        ex, ey = zip(*evals)
        ax.plot(ex, ey, "s", color=ORANGE, ms=8, label="eval (fp32, 49 windows)")
        for x, y in evals:
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(6, 8),
                        fontsize=7.5, color=ORANGE)
        ax.axhline(10.5783, color=MUTED, lw=1.2, ls="--")
        ax.text(0.97, 10.5783 + 0.16, "ln V + d$\\sigma^2$/2 = 10.5783", ha="right",
                fontsize=7, color=MUTED)
        ax.axhline(10.3735, color=TICK, lw=1.0, ls=":")
        ax.text(0.03, 10.3735 - 0.42, "ln V = 10.3735", fontsize=7, color=TICK)
        ax.set_title(title, fontsize=9, color=DARK)
        ax.set_xlabel("step", fontsize=8, color=TICK)
        ax.set_ylabel("cross-entropy (nats/token)", fontsize=8, color=TICK)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TICK, labelsize=7.5)
        ax.set_ylim(6.8, 11.1)
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-9-3-hybrid3to1-train.png"), facecolor="white")
    plt.close(fig)


def fig94():
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    # 左：续跑段 6001-10000 三臂 loss + ln(64) 平台线
    ax = axes[0]
    cols = {"all_global": BLUE, "hybrid31": TEAL, "all_linear": YELLOW}
    for arm in ("all_global", "hybrid31", "all_linear"):
        ck = torch.load(os.path.join(LOG_DIR, f"recall_ckpt_{arm}.pt"),
                        map_location="cpu", weights_only=False)
        ls = np.asarray(ck["losses"], dtype=float)
        xs = np.arange(6001, 6001 + len(ls))
        ax.plot(xs, ls, color=cols[arm], lw=2, label=f"{arm} ({ck['step']} steps)")
    ax.axhline(np.log(64), color=ORANGE, lw=1.4, ls="--")
    ax.text(0.03, np.log(64) + 0.10, "value-prior floor ln 64 = 4.159", ha="left",
            fontsize=7.5, color=ORANGE)
    ax.set_xlabel("step (resumed segment; 6000-step run precedes)", fontsize=8, color=TICK)
    ax.set_ylabel("MQAR needle loss", fontsize=8, color=TICK)
    ax.set_ylim(1.8, 4.6)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TICK, labelsize=7.5)
    ax.legend(fontsize=7.2, loc="upper right", framealpha=0.95)
    ax.set_title("loss @ 10000 steps (B3 budget branch)", fontsize=9, color=DARK)
    # 右：五位置 acc 三臂
    ax = axes[1]
    d = {}
    for arm in ("all_global", "hybrid31", "all_linear"):
        d[arm] = json_load(os.path.join(LOG_DIR, f"recall3_{arm}_run1.json"))["recall_acc_by_pos"]
    positions = [18, 42, 58, 62, 74]
    w = 0.26
    for k, arm in enumerate(("all_global", "hybrid31", "all_linear")):
        vals = [d[arm][str(p)] for p in positions]
        ax.bar(np.arange(5) + (k - 1) * w, vals, w, color=cols[arm],
               label=arm, edgecolor="white", lw=0.5)
    ax.axhline(0.7, color=MUTED, lw=1.2, ls="--")
    ax.text(4.45, 0.72, "J-threshold 0.7", fontsize=7, color=MUTED, ha="right")
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"pos {p}" for p in positions], fontsize=7.5)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("recall accuracy", fontsize=8, color=TICK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TICK, labelsize=7.5)
    ax.legend(fontsize=7.5, framealpha=0.95)
    ax.set_title("5-position recall acc @ 10000 steps", fontsize=9, color=DARK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-9-4-hybrid3to1-recall.png"), facecolor="white")
    plt.close(fig)


def json_load(path):
    import json
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all",
                    choices=["all", "1", "2", "3", "4"], help="单图复绘")
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    todo = {"1": [fig91], "2": [fig92], "3": [fig93], "4": [fig94],
            "all": [fig91, fig92, fig93, fig94]}[args.which]
    for fn in todo:
        fn()
        print(f"[fig] {fn.__name__} done")
    print(f"[产物] {FIG_DIR}/fig-9-*.png")
