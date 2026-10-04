# make_figures.py —— Book3 ch10 两张图：图 10.1 整机结构与插槽来源 / 图 10.2 冒烟+整合短训曲线
# 用途：产出 figures/fig-10-{1,2}-*.png（300dpi 印刷规格；图内英文、
#       图注中文在正文）。图 10.1 为自绘示意级（不依赖实验数据）；图 10.2 读
#       log/book3-ch10/curve_smoke_s1.csv 与 curve_integrated_run1.csv（train_215.py 产物——
#       续跑窗口已由脚本自动合并进同一文件，无需手工缝合）。
# 所属章节：Book3 ch10（10.2 图 10.1 / 10.3 图 10.2；判读见 ch10-大项目.md）。
# 运行方式：cd 工作区根目录 && source env.sh && \
#           python code/ch10/make_figures.py [--only 1,2]
# 规格：写作规范 §4——蓝 #2a78d6 / 橙 #eb6834 / 青 #1baf7a / 黄 #eda100；线宽 2；网格 hairline
#       #e1e0d9 置底层；刻度 #898781；标注 #0b0b0b / #52514e；禁双 y 轴；tight_layout；禁 jet/rainbow。
import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DIR = os.path.join(REPO, "figures")
LOG_DIR = os.path.join(REPO, "log", "book3-ch10")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MID = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
GHOST = "#f4f3ef"
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9.5,
    "axes.labelcolor": DARK, "xtick.color": TICK, "ytick.color": TICK,
    "font.family": "sans-serif",
})

LN_V = math.log(32000)          # 10.3735：均匀分布熵地板
DS2 = 1024 * 0.02 ** 2          # 0.4096：初始 logit 方差（untied 无泄漏项）
H_UNI = 7.6743                  # sp-32k 训练段（前 8.2M token）一元分布熵（2026-10 实算）


# ---------------- 图 10.1：207M 整机结构与插槽来源（自绘示意级） ----------------
def fig_10_1():
    fig, ax = plt.subplots(figsize=(8, 6.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 80)
    ax.axis("off")

    def box(x, y, w, h, label, sub="", ec=MID, fc="white", lw=1.3, ls="-", fs=8.0, sub_fs=6.2):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
        ax.text(x + w / 2, y + h / 2 + (1.5 if sub else 0), label, ha="center", va="center",
                fontsize=fs, color=DARK, zorder=3)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 1.7, sub, ha="center", va="center",
                    fontsize=sub_fs, color=MID, zorder=3)

    def arrow(x1, y1, x2, y2, color=MID, lw=1.4):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))

    # ---- 数据河主干（左列，自上而下）----
    XC, W = 27, 30                                                 # 主干列中心与框宽
    XL, XR = XC - W / 2, XC + W / 2                                # 左右沿
    box(XL, 74, W, 4.6, "token ids", "(B,n) = (8,1024)")
    box(XL, 66, W, 5.2, "embed_tokens", "(32000,1024) -> (8,1024,1024)   [ch.5: V=32000]")
    arrow(XC, 74, XC, 71.2)
    # 12 层 Block 大框
    ax.add_patch(plt.Rectangle((XL - 1.5, 27), W + 3, 35.5, fc="#fbfaf7", ec=MID, lw=1.5, zorder=1))
    ax.text(XC, 60.2, "12 x Llama215DecoderLayer", ha="center", fontsize=8.4, color=DARK)
    ax.text(XC, 57.9, "(residual stream, width 1024)", ha="center", fontsize=6.4, color=MID)
    box(XL, 51.5, W, 4.6, "RMSNorm   slot (1)", "from ch.2  |  eps 1e-5, no bias", fc="#d7e6f9")
    box(XL, 43.5, W, 6.2, "GQA attention   slot (4)", "from ch.6: h=16, h_kv=8, d_k=64 | q/k rotated in-place",
        fc="#d3efe3")
    box(XL, 36.5, W, 4.6, r"RoPE on Q/K   slot (3)", r"from ch.4: $\theta$=10^4, no wpe table",
        fc="#d3efe3")
    box(XL, 28.5, W, 5.4, "SwiGLU MLP   slot (2)", "from ch.3: d_ff=2816 = 8C/3 | (8,1024,2816)x2",
        fc="#fde2d3")
    arrow(XC, 51.5, XC, 49.7)
    arrow(XC, 43.5, XC, 41.1)
    arrow(XC, 36.5, XC, 33.9)
    box(XL, 20, W, 4.6, "final RMSNorm", "(8,1024,1024)", fc="#d7e6f9")
    arrow(XC, 28.5, XC, 24.6)
    box(XL, 11.5, W, 5.4, "lm_head   untied, ch.5 (U)", "(1024 -> 32000)  ->  (8,1024,32000)",
        fc="#fde2d3")
    arrow(XC, 20, XC, 16.9)
    ax.text(XC, 8.6, "cross-entropy (next token, fp32)", ha="center", fontsize=7.4, color=MID)
    arrow(XC, 11.5, XC, 9.5)

    # ---- 右栏：插槽来源（import 契约）----
    ax.text(71.5, 76.8, "slot provenance  (imports in llama215.py)", ha="center",
            fontsize=8.2, color=DARK)
    rows = [("(1) RMSNorm", "ch02/rmsnorm.py", "#d7e6f9", "LayerNorm -> RMSNorm"),
            ("(2) SwiGLU", "ch03/swiglu.py", "#fde2d3", "GELU-4C -> gated 8C/3"),
            ("(3) RoPE", "ch04/rope.py", "#d3efe3", "wpe table -> rotation"),
            ("(4) GQA", "ch06/gqa.py", "#d3efe3", "MHA -> h_kv=8"),
            ("(U) untied", "inline nn.Linear (ch.5)", "#fde2d3", "tied -> two matrices")]
    y = 66.5
    for tag, path, fc, note in rows:
        box(50, y, 43, 6.6, tag + " :  " + note, path, fc=fc, fs=7.5, sub_fs=6.2)
        y -= 8.6

    # ---- 右下：从 GPT-2 换下的零件（虚影）----
    ax.text(71.5, 25.0, "removed / replaced  (GPT-2 124M base)", ha="center",
            fontsize=7.6, color=TICK)
    for i, part in enumerate(["LayerNorm x2", "GELU 4C MLP", "wpe (1024,C)",
                              "MHA h=h_kv=12", "tied head"]):
        box(50 + (i % 3) * 14.8, 17.5 - (i // 3) * 6.2, 13.8, 4.6, part, ec=TICK, fc=GHOST,
            ls="--", fs=6.6)

    ax.text(50, 1.2, "geometry: d=1024, L=12, h=16, h_kv=8, d_ff=2816, V=32000"
            "   |   params 207,119,360 (asserted)", ha="center", fontsize=7.0, color=TICK)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-10-1-llama215-machine-slots.png")
    fig.savefig(out)
    plt.close(fig)
    print("written", out)


# ---------------- 图 10.2：冒烟 + 整合短训曲线（自产） ----------------
def _read_curve(name):
    rows = []
    with open(os.path.join(LOG_DIR, name)) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    steps = [int(r["step"]) for r in rows if r["train_loss"]]
    train = [float(r["train_loss"]) for r in rows if r["train_loss"]]
    ev = [(int(r["step"]), float(r["eval_loss"])) for r in rows if r.get("eval_loss")]
    return steps, train, ev


def fig_10_2():
    s1, tr1, ev1 = _read_curve("curve_smoke_s1.csv")
    s2, tr2, ev2 = _read_curve("curve_integrated_run1.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    ax = axes[0]
    ax.plot(s1, tr1, color=BLUE, lw=2, marker="o", ms=4, label="train loss (20 steps)")
    if ev1:
        ax.plot([e[0] for e in ev1], [e[1] for e in ev1], "o", color=ORANGE, ms=6,
                label="eval (49 win, fp32)")
    ax.axhline(LN_V, color=TICK, lw=1.2, ls="--")
    ax.axhline(LN_V + DS2 / 2, color=MID, lw=1.0, ls=":")
    ax.text(10.2, LN_V + 0.06, "ln V = 10.373", fontsize=6.8, color=TICK, ha="center")
    ax.text(10.2, LN_V + DS2 / 2 + 0.08, "ln V + d$s^2$/2 = 10.578", fontsize=6.8, color=MID,
            ha="center")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (nat / token)")
    ax.set_title("smoke: init loss 10.558 vs prediction 10.578", fontsize=8.6)
    ax.legend(fontsize=6.8, frameon=False)
    ax.grid(color=GRID, lw=0.5, zorder=0)

    ax = axes[1]
    ax.plot(s2, tr2, color=BLUE, lw=2, label="train loss")
    if ev2:
        ax.plot([e[0] for e in ev2], [e[1] for e in ev2], "o", color=ORANGE, ms=6,
                label="eval (49 win, fp32)")
    ax.axhline(H_UNI, color=TEAL, lw=1.2, ls="--")
    ax.text(500, H_UNI + 0.12, "unigram entropy = 7.674", fontsize=6.8, color=TEAL, ha="center")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (nat / token)")
    ax.set_title("integrated run: 1000 steps, 8.2M tokens, eval 5.837", fontsize=8.6)
    ax.legend(fontsize=6.8, frameon=False)
    ax.grid(color=GRID, lw=0.5, zorder=0)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-10-2-smoke-integrated-curve.png")
    fig.savefig(out)
    plt.close(fig)
    print("written", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="1,2")
    args = ap.parse_args()
    if "1" in args.only.split(","):
        fig_10_1()
    if "2" in args.only.split(","):
        fig_10_2()
