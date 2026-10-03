# -*- coding: utf-8 -*-
# 用途：生成 Book2 第 3 章三张图——
#   fig 3.1 BERT 整机（自绘重制，引 arXiv:1810.04805v2 Fig.1/Fig.2；标注张量形状）
#   fig 3.2 MLM 80-10-10 流程（自绘；选词-替换-预测三段，标注形状）
#   fig 3.3 mini-GLUE 训练曲线与准确率（自产：读 log/book2-ch03/*.json）
# 所属章节：《预训练革命》第 3 章
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch03/make_figs.py" [--only 3]
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# 本书印刷图规格（写作规范 §4）：分类色固定序 蓝/橙/青/黄；线宽 2；网格 hairline 底层；图内英文、图注中文
BLUE, ORANGE, CYAN, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUE_F, ORNG_F, CYAN_F, YLW_F = "#cde2fb", "#fde8db", "#d8f3e8", "#fdf3d7"
INK, NOTE, HAIR, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
LOG_DIR = Path("log") / "book2-ch03"


def box(ax, x, y, w, h, text, ec=BLUE, fc=BLUE_F, fs=7.0, lw=1.3, tc=INK):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.01",
                                ec=ec, fc=fc, lw=lw, mutation_scale=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.55)
    return (x, y, w, h)


def arrow(ax, p, q, color=HAIR, lw=1.3, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=lw, connectionstyle=f"arc3,rad={rad}"))


def note(ax, x, y, text, fs=6.2, color=NOTE, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color, linespacing=1.5)


def fig_3_1():
    """BERT 整机：三路嵌入相加 → L 层双向 encoder → 双头输出（MLM/NSP）。引 v2 Fig.1/Fig.2 自绘重制。"""
    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=300)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # 左：输入句对拼包（NSP 段对格式；单句退化为 B 段缺省）
    box(ax, .115, .80, .19, .10, "input ids\n[CLS] A [SEP] B [SEP]\n(B,n)", ec=HAIR, fc="white", fs=6.6)
    # 中左：三路嵌入（纵排）
    box(ax, .415, .925, .25, .072, "token embedding $V{\\times}H$\n$30522{\\times}768$, lookup\n(B,n)$\\to$(B,n,H)", ec=CYAN, fc=CYAN_F, fs=6.0)
    box(ax, .415, .80, .25, .072, "segment embedding\n$2{\\times}H$, A/B two tables\n+(B,n,H)", ec=CYAN, fc=CYAN_F, fs=6.0)
    box(ax, .415, .675, .25, .072, "position embedding\n$512{\\times}H$ (learned table)\n+(B,n,H)", ec=CYAN, fc=CYAN_F, fs=6.0)
    # 中右：求和与 LN
    box(ax, .665, .80, .10, .085, "sum\n+ LN\ndropout", ec=YELLOW, fc=YLW_F, fs=6.2)
    note(ax, .665, .735, "(B,n,H)", fs=6.0)
    # 右：encoder 堆叠
    box(ax, .885, .80, .195, .36,
        "bidirectional\nTransformer encoder\n$\\times L$\n\nself-attn + FFN\n(post-LN)\n\nBASE 12/768/12\nLARGE 24/1024/16",
        ec=BLUE, fc=BLUE_F, fs=6.4)
    note(ax, .885, .545, "scores (B,A,n,n)\nno causal mask", fs=6.0)
    # 下：双头输出
    box(ax, .30, .24, .44, .13,
        "MLM head: gather masked $T_i$\n(B,n,H)$\\to$(B,20,H) dense+GELU+LN\n$\\to$ tied decoder $E^{\\top}$+bias $\\to$ (B,20,V)",
        ec=ORANGE, fc=ORNG_F, fs=6.2)
    box(ax, .80, .24, .36, .13,
        "NSP head: pooler on [CLS]\n$C=T_0$: (B,n,H)$\\to$(B,H)\ndense+tanh $\\to$ (B,2)",
        ec=ORANGE, fc=ORNG_F, fs=6.2)
    note(ax, .30, .135, "cross-entropy vs original tokens", fs=6.0)
    note(ax, .80, .135, "IsNext / NotNext", fs=6.0)
    # 箭头
    arrow(ax, (.21, .82), (.29, .925))
    arrow(ax, (.21, .80), (.29, .80))
    arrow(ax, (.21, .78), (.29, .675))
    arrow(ax, (.54, .925), (.615, .825))
    arrow(ax, (.54, .80), (.615, .80))
    arrow(ax, (.54, .675), (.615, .775))
    arrow(ax, (.715, .80), (.785, .80))
    # encoder 出口分两路（左下 MLM / 右下 NSP）
    arrow(ax, (.86, .62), (.52, .325), rad=0.22)
    arrow(ax, (.93, .62), (.80, .325), rad=0.0)
    note(ax, .60, .475, "$T_i$: all positions (B,n,H)", fs=6.2, ha="left")
    note(ax, .955, .475, "$C=T_0$\n(B,H)", fs=6.2, ha="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig-3-1-bert-architecture.png")
    plt.close(fig)


def fig_3_2():
    """MLM 80-10-10：选中 15% → 三段式替换 → encoder → 仅在被选位置预测原词。"""
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=300)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # 第一行：原序列与选词
    box(ax, .19, .84, .30, .105, "original sequence (WordPiece)\nmy  dog  is  hair ##ry\n(B,n) token ids", ec=HAIR, fc="white", fs=6.6)
    box(ax, .635, .84, .315, .105, "select 15% at random\n(hair ##ry both eligible:\nsubword-level, no whole-word rule)\n$\\to$ index set $\\mathcal{M}$, $|\\mathcal{M}|\\leq 20$", ec=CYAN, fc=CYAN_F, fs=6.2)
    arrow(ax, (.34, .84), (.478, .84))
    note(ax, .50, .777, "here: position of \"hairy\" selected", fs=6.2, ha="center")

    # 第二行：三段式替换
    box(ax, .175, .52, .27, .14, "80%: replace by [MASK]\nmy dog is [MASK] [MASK]\ntrain/infer mismatch\nnever seen at fine-tune", ec=ORANGE, fc=ORNG_F, fs=6.2)
    box(ax, .50, .52, .24, .14, "10%: random token\nmy dog is apple [MASK]\nbiases representation\nto observed word", ec=BLUE, fc=BLUE_F, fs=6.2)
    box(ax, .79, .52, .27, .14, "10%: keep original\nmy dog is hair ##ry\nanchors identity\nat the position", ec=BLUE, fc=BLUE_F, fs=6.2)
    # 选中 → 三段式：先汇聚到 junction，再分发三路
    arrow(ax, (.635, .7875), (.50, .685))
    for xx in (.175, .50, .79):
        arrow(ax, (.50, .685), (xx, .592))
    note(ax, .655, .685, "one of three, per selected token", fs=6.2, ha="left")

    # 第三行：编码与预测
    box(ax, .235, .235, .40, .115, "bidirectional encoder (no causal mask)\n(B,n) ids $\\to$ (B,n,H) hidden states $T_i$\nevery token sees both sides", ec=BLUE, fc=BLUE_F, fs=6.4)
    box(ax, .72, .235, .40, .115, "predict ONLY at $\\mathcal{M}$:\ngather (B,20,H) $\\to$ tied decoder\n$\\to$ softmax over V=30522 (B,20,V)\ncross-entropy vs original token", ec=ORANGE, fc=ORNG_F, fs=6.2)
    arrow(ax, (.435, .235), (.52, .235))
    for x0, x1 in ((.175, .16), (.50, .30), (.79, .40)):
        arrow(ax, (x0, .45), (x1, .293))
    note(ax, .66, .385, "corrupted input feeds the encoder", fs=6.2, ha="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig-3-2-mlm-flow.png")
    plt.close(fig)


def fig_3_3():
    """mini-GLUE 实测：左=训练 loss 曲线（EMA 平滑），右=验证准确率 vs 多数类基线。"""
    runs = [("tiny", "sst2"), ("base", "sst2"), ("base", "mrpc")]
    curves = {}
    for m, t in runs:
        p = LOG_DIR / f"finetune_{m}_{t}.json"
        if p.exists():
            curves[(m, t)] = json.load(open(p))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.9), dpi=300)
    style = {("tiny", "sst2"): (ORANGE, "bert-tiny / SST-2 (3 ep)"),
             ("base", "sst2"): (BLUE, "bert-base / SST-2 (1 ep)"),
             ("base", "mrpc"): (CYAN, "bert-base / MRPC (3 ep)")}
    for key, r in curves.items():
        c, lab = style[key]
        raw = r["train"]["loss_curve"]
        ema, acc = [], 0.0
        for v in raw:  # EMA 平滑（α=0.02）
            acc = v if not ema else 0.02 * v + 0.98 * acc
            ema.append(acc)
        a1.plot(range(len(ema)), ema, color=c, lw=2, label=f"{lab}")
    a1.set_xlabel("fine-tune step", fontsize=8, color=NOTE)
    a1.set_ylabel("train loss (EMA)", fontsize=8, color=NOTE)
    a1.set_title("mini-GLUE fine-tuning loss", fontsize=9, color=INK)
    a1.grid(True, color=GRID, lw=0.5)
    a1.tick_params(colors=HAIR, labelsize=7.5)
    a1.legend(fontsize=7.2, frameon=False)

    # 右：验证准确率柱状（按任务分组）+ 多数类基线虚线（缺的运行跳过不画）
    def _final(mkey, tkey):
        r = curves.get((mkey, tkey))
        return r["val_acc_per_epoch"][-1] if r else None

    def _maj(mkey, tkey):
        r = curves.get((mkey, tkey))
        return r["majority_baseline"]["val_pct"] if r else None

    cells = [("tiny", "sst2", 0.5), ("base", "sst2", 1.4), ("tiny", "mrpc", 2.5), ("base", "mrpc", 3.4)]
    bars = [(x, _final(m, t), ORANGE if m == "tiny" else BLUE) for m, t, x in cells if _final(m, t) is not None]
    a2.bar([b[0] for b in bars], [b[1] for b in bars], width=0.72, color=[b[2] for b in bars], lw=0)
    for x, v, _ in bars:
        a2.text(x, v + 1.2, f"{v:.1f}", ha="center", fontsize=7.4, color=INK)
    # bert-base/MRPC 同种子两次运行（MPS 非确定性）：柱取第二次，标注两次终点
    if _final("base", "mrpc") is not None:
        a2.text(3.4, _final("base", "mrpc") + 5.6, "2 runs:\n85.0 / 80.4", ha="center", fontsize=6.4, color=NOTE)
    for x0, x1, m in [(-0.45, 1.65, _maj("tiny", "sst2")), (1.95, 4.05, _maj("tiny", "mrpc"))]:
        if m is not None:
            a2.hlines(m, x0, x1, colors=HAIR, linestyles="--", lw=1.4)
            a2.text(x1, m - 4.2, f"majority {m:.1f}", ha="right", fontsize=6.6, color=NOTE)
    a2.set_xticks([0.5, 1.4, 2.5, 3.4])
    a2.set_xticklabels(["bert-tiny", "bert-base"] * 2, fontsize=7.5, color=HAIR)
    a2.text(0.5, 40, "SST-2 (n=872)", ha="center", fontsize=8, color=INK)
    a2.text(3.1, 40, "MRPC (n=408)", ha="center", fontsize=8, color=INK)
    a2.set_ylim(40, 100)
    a2.set_ylabel("validation accuracy (%)", fontsize=8, color=NOTE)
    a2.set_title("val accuracy vs majority baseline", fontsize=9, color=INK)
    a2.grid(True, axis="y", color=GRID, lw=0.5)
    a2.tick_params(colors=HAIR, labelsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig-3-3-glue-curves.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, choices=[1, 2, 3])
    args = ap.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if args.only in (None, 1):
        fig_3_1()
    if args.only in (None, 2):
        fig_3_2()
    if args.only in (None, 3):
        fig_3_3()
    print("figures saved to", FIG_DIR)
