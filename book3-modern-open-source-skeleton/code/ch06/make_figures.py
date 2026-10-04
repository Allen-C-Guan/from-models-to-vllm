# make_figures.py —— Book3 ch6 四张图的绘制脚本（图 6.1 数据流 / 6.2 剪刀差 / 6.3 头分组 / 6.4 消融）
# 用途：产出 figures/fig-6-{1,2,3,4}-*.png（300dpi 印刷规格，图内英文、
#       图注中文在正文）；图 6.4 依赖 log/book3-ch06/ 的消融产物（ablation_gqa_*.json），
#       自动优先 full 档、缺则回退 fast——full 未跑完时可 --only 1,2,3 先出不依赖数据的两张。
# 所属章节：Book3 ch6（6.2 图 6.1 / 6.3 图 6.2 / 6.5 图 6.3 / 6.8 图 6.4）。
# 运行方式：cd code/ch06 && python make_figures.py [--only 1,2,3]
# 规格：写作规范 §4——蓝 #2a78d6 / 橙 #eb6834 / 青 #1baf7a / 黄 #eda100；线宽 2；网格 hairline
#       #e1e0d9 置底层；刻度 #898781；标注 #0b0b0b / #52514e；禁双 y 轴；tight_layout；禁 jet/rainbow。
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DIR = os.path.join(REPO, "figures")
LOG_DIR = os.path.join(REPO, "log", "book3-ch06")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MID = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9.5,
    "axes.edgecolor": TICK, "axes.labelcolor": DARK, "xtick.color": TICK,
    "ytick.color": TICK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "font.family": "sans-serif",
})

KIB, MIB, GIB = 2**10, 2**20, 2**30


def style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------- 图 6.1：KV cache 数据流与形状（两阶段：一次读入 / 逐 token 生成） ----------------
def fig_6_1():
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")
    ax.grid(False)

    def tbox(x, y, w, h, label, sub="", ec=MID, fc="#f4f3ef"):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.0, zorder=2))
        ax.text(x + w / 2, y + h / 2 + (2.6 if sub else 0), label, ha="center", va="center",
                fontsize=8.5, color=DARK, zorder=3)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.8, sub, ha="center", va="center",
                    fontsize=7.5, color=MID, zorder=3)

    def opbox(x, y, w, h, label, color=BLUE):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc="white", ec=color, lw=1.6, zorder=2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=8.5, color=DARK, zorder=3)

    def arrow(x1, y1, x2, y2, text="", color=MID, dy=1.6, style="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, linestyle=style))
        if text:
            ax.text((x1 + x2) / 2, max(y1, y2) + dy, text, ha="center", va="bottom",
                    fontsize=7.2, color=MID)

    # ---- 左：一次读入（整段 prompt 一把算完 K/V，写入缓存）----
    ax.text(25, 54.5, "(a) read-in: one pass over the prompt", ha="center",
            fontsize=10, color=DARK, weight="bold")
    tbox(8, 44, 34, 7, "prompt tokens", "x  (B, n, d)")
    opbox(8, 33, 15, 7, "k_proj", color=ORANGE)
    opbox(27, 33, 15, 7, "v_proj", color=ORANGE)
    arrow(18, 44, 15.5, 40)
    arrow(32, 44, 34.5, 40)
    ax.add_patch(plt.Rectangle((4, 12), 42, 15, fc="#fdf6ef", ec=ORANGE, lw=1.8, zorder=1))
    ax.text(25, 24.2, "KV cache  (one pair per layer, $L$ pairs)", ha="center",
            fontsize=9, color=DARK, weight="bold", zorder=3)
    ax.text(25, 19.6, "K  (B, $h_{kv}$, n, $d_k$)      V  (B, $h_{kv}$, n, $d_k$)",
            ha="center", fontsize=8.2, color=MID, zorder=3)
    ax.text(25, 15.4, "write once, never rewrite", ha="center", fontsize=7.4,
            color=MID, zorder=3, style="italic")
    arrow(15.5, 33, 14, 27, color=ORANGE)
    arrow(34.5, 33, 36, 27, color=ORANGE)
    ax.text(25, 6.5, "causal mask $\\Rightarrow$ old K/V never change\n$\\Rightarrow$ they can be stored",
            ha="center", fontsize=7.8, color=MID)

    # ---- 右：逐 token 生成（每步只算新 token，读缓存）----
    ax.text(76, 54.5, "(b) generation: one token at a time", ha="center",
            fontsize=10, color=DARK, weight="bold")
    tbox(56, 44, 40, 7, "new token (step t)", "$x_t$  (B, 1, d)")
    opbox(56, 33, 11, 7, "q_proj")
    opbox(69, 33, 11, 7, "k_proj", color=ORANGE)
    opbox(82, 33, 12, 7, "v_proj", color=ORANGE)
    arrow(68, 44, 61.5, 40)
    arrow(76, 44, 74.5, 40)
    arrow(86, 44, 88, 40)
    tbox(52, 22, 20, 7, "query", "$q_t$  (B, h, 1, $d_k$)", ec=BLUE, fc="white")
    ax.add_patch(plt.Rectangle((74, 12), 26, 15, fc="#fdf6ef", ec=ORANGE, lw=1.8, zorder=1))
    ax.text(87, 22.6, "cache + 1 slot", ha="center", fontsize=8.4, color=DARK,
            weight="bold", zorder=3)
    ax.text(87, 18.2, "append $k_t, v_t$\n(B, $h_{kv}$, 1, $d_k$)", ha="center",
            fontsize=7.6, color=MID, zorder=3)
    arrow(61.5, 33, 62, 29, color=BLUE)
    arrow(74.5, 33, 80, 27, color=ORANGE)
    arrow(88, 33, 92, 27, color=ORANGE)
    opbox(52, 8, 20, 8, "attention\n(B,h,1,n+1)", color=BLUE)
    arrow(62, 22, 62, 16, color=BLUE)
    ax.annotate("", xy=(68, 12), xytext=(86, 18), arrowprops=dict(arrowstyle="-|>",
                color=ORANGE, lw=1.2, linestyle="--"))
    tbox(78, 3, 22, 8, "output", "$y_t$  (B, 1, d)", ec=BLUE, fc="white")
    arrow(72, 12, 78, 7, color=BLUE)
    ax.text(62.5, 0.2, "scores need no mask: keys = visible prefix", ha="center",
            fontsize=7.4, color=MID, style="italic")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-6-1-kvcache-dataflow.png"), facecolor="white")
    plt.close(fig)


# ---------------- 图 6.2：剪刀差（权重恒定线 vs KV 线性线；B=1 / B=8 两族） ----------------
def fig_6_2():
    n_max = 8192
    ns = np.linspace(0, n_max, 200)
    W7B_BYTES = 6_738_415_616 * 2                      # 7B 档权重 fp16 字节（12.55 GiB；canonical 手算参数 6,738,415,616
                                                       #  = ch5 式 (5.2) 与表 5.1 同源，正文按 12.55 GiB 口径）
    series = [  # (label, weights bytes, KV bytes/token, color)
        ("GPT-2 124M (MHA)", 0.23 * GIB, 2 * 12 * 12 * 64 * 2, BLUE),
        ("7B (MHA)", W7B_BYTES, 2 * 32 * 32 * 128 * 2, ORANGE),
        ("70B (GQA-8)", 128.5 * GIB, 2 * 80 * 8 * 128 * 2, TEAL),
        ("70B MHA (counterfactual)", 128.5 * GIB, 2 * 80 * 64 * 128 * 2, YELLOW),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    for ax, batch, tag in [(axes[0], 1, "B = 1"), (axes[1], 8, "B = 8")]:
        for label, w_bytes, bpt, color in series:
            kv = ns * bpt * batch / GIB
            ax.plot(ns, kv, color=color, lw=2, zorder=3)
            ax.axhline(w_bytes / GIB, color=color, lw=1.4, ls="--", zorder=2)
        ax.set_xlim(0, n_max)
        ax.set_xlabel("context length n (tokens)")
        ax.set_ylabel("memory (GiB)")
        ax.set_title(tag, fontsize=10)
        style_axis(ax)
    # 7B B=8 追平点：权重精确字节 / (512 KiB/token x 8) —— {本书复算} n=3,213
    n_cross = W7B_BYTES / (2 * 32 * 32 * 128 * 2 * 8)
    axes[1].plot([n_cross], [W7B_BYTES / GIB], marker="o", ms=7, color=ORANGE, zorder=5)
    axes[1].annotate(f"KV catches weights\nat n = {n_cross:,.0f}",
                     xy=(n_cross, W7B_BYTES / GIB), xytext=(n_cross + 700, 16.5), fontsize=7.6,
                     color=DARK, arrowprops=dict(arrowstyle="-|>", color=MID, lw=1.0))
    axes[0].text(4100, 14.0, "70B weights 128.5 GiB (off scale)", fontsize=7.4, color=MID)
    axes[0].text(4100, 1.2, "GPT-2 weights 0.23 GiB", fontsize=7.4, color=MID)
    axes[0].set_ylim(0, 21)
    axes[1].set_ylim(0, 42)
    from matplotlib.lines import Line2D
    handles = ([Line2D([0], [0], color=c, lw=2) for _, _, _, c in series]
               + [Line2D([0], [0], color=MID, lw=1.4, ls="--")])
    labels = [s[0] + " KV cache" for s in series] + ["weights (fp16, constant)"]
    fig.legend(handles, labels, fontsize=7.0, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=1.2)
    fig.subplots_adjust(bottom=0.3, wspace=0.25)
    fig.savefig(os.path.join(FIG_DIR, "fig-6-2-scissors-gap.png"), facecolor="white")
    plt.close(fig)


# ---------------- 图 6.3：MHA / GQA / MQA 头分组示意（h 与 h_kv 双标） ----------------
def fig_6_3():
    h = 8
    panels = [("MHA  ($h_{kv}=h=8$)", 8, "12 KiB / token"),
              ("GQA-4  ($h_{kv}=4$)", 4, "6 KiB / token"),
              ("MQA  ($h_{kv}=1$)", 1, "1.5 KiB / token")]
    fig, axes = plt.subplots(1, 3, figsize=(8, 3.6))
    for ax, (title, h_kv, kv_note) in zip(axes, panels):
        ax.set_xlim(-0.3, 8.3)
        ax.set_ylim(-2.6, 3.4)
        ax.axis("off")
        ax.grid(False)
        ax.set_title(title, fontsize=9.5)
        g = h // h_kv
        xs = np.arange(h)
        for i, xq in enumerate(xs):                      # Q 头（上行，蓝）
            ax.add_patch(plt.Rectangle((xq - 0.32, 2.0), 0.64, 0.8, fc="white", ec=BLUE, lw=1.6))
            ax.text(xq, 2.4, str(i + 1), ha="center", va="center", fontsize=7.5, color=DARK)
        for j in range(h_kv):                            # KV 头（下行，橙）
            xc = (j + 0.5) * g - 0.5                     # 组内连号：Q i 配 KV i//g
            ax.add_patch(plt.Rectangle((xc - 0.36, -0.6), 0.72, 0.8, fc="white", ec=ORANGE, lw=1.8))
            ax.text(xc, -0.2, str(j + 1), ha="center", va="center", fontsize=7.5, color=DARK)
            for r in range(g):                           # 连线：组内 g 个 Q 头
                xq = j * g + r
                ax.plot([xq, xc], [2.0, 0.2], color=MID, lw=0.7, zorder=1)
        ax.text(4.0, 3.15, f"query heads  h = {h}   Q (B,h,n,$d_k$)",
                ha="center", fontsize=7.8, color=DARK)
        ax.text(4.0, -1.35, f"KV heads  $h_{{kv}}$ = {h_kv}   K,V (B,$h_{{kv}}$,n,$d_k$)",
                ha="center", fontsize=7.8, color=DARK)
        ax.text(4.0, -2.15, f"module tier (L=6, $d_k$=64, fp16):\n{kv_note}",
                ha="center", fontsize=7.2, color=MID)
        if g > 1:
            ax.text(4.0, 1.05, f"group size g = h/$h_{{kv}}$ = {g}", ha="center",
                    fontsize=7.2, color=MID)
        else:
            ax.text(4.0, 1.05, "one KV per query head", ha="center", fontsize=7.2, color=MID)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-6-3-mha-gqa-mqa-groups.png"), facecolor="white")
    plt.close(fig)


# ---------------- 图 6.4：组数扫描消融（eval 曲线 + 质量-KV 反向读数） ----------------
def fig_6_4():
    tier = "full" if os.path.exists(os.path.join(LOG_DIR, "ablation_gqa_full.json")) else "fast"
    with open(os.path.join(LOG_DIR, f"ablation_gqa_{tier}.json"), encoding="utf-8") as f:
        rep = json.load(f)
    arms = [("mha8", "MHA ($h_{kv}$=8)", BLUE), ("gqa4", "GQA-4 ($h_{kv}$=4)", TEAL),
            ("mqa1", "MQA ($h_{kv}$=1)", ORANGE)]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    ax = axes[0]
    for key, label, color in arms:
        ev = rep["arms"][key]["evals"]
        xs = [e["step"] for e in ev]
        ys = [e["eval_loss"] for e in ev]
        es = [e["eval_se"] for e in ev]
        ax.errorbar(xs, ys, yerr=es, color=color, lw=2, ms=5, marker="o",
                    capsize=2, elinewidth=0.8, label=label, zorder=3)
    ax.set_xlabel("training step")
    ax.set_ylabel("fixed-window eval loss (fp32)")
    ax.set_title(f"(a) same budget, {tier} tier", fontsize=9.5)
    style_axis(ax)
    ax.legend(fontsize=7.6, loc="upper right")

    ax = axes[1]
    xs, ys, colors, labels = [], [], [], []
    for key, label, color in arms:
        xs.append(rep["arms"][key]["kv_per_token_bytes"] / KIB)
        ys.append(rep["arms"][key]["eval_final"])
        colors.append(color)
        labels.append(label.split(" (")[0])
    ax.plot(xs, ys, color=MID, lw=1.2, ls=":", zorder=2)
    for x, y, c, lab in zip(xs, ys, colors, labels):
        ax.plot([x], [y], marker="o", ms=8, color=c, zorder=4)
        ax.annotate(f"{lab}\n{y:.4f}", xy=(x, y), xytext=(x + 0.7, y),
                    fontsize=7.8, color=DARK, va="center")
    ax.set_xlabel("KV cache bytes per token (KiB, module tier)")
    ax.set_ylabel("final eval loss (lower = better)")
    ax.set_title(f"(b) quality vs KV ledger ({tier} tier)", fontsize=9.5)
    ax.set_xlim(0, 16.5)
    style_axis(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-6-4-gqa-ablation.png"), facecolor="white")
    plt.close(fig)
    print(f"[fig 6.4] 用 {tier} 档数据绘制完成")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="1,2,3,4", help="只画部分图（如 1,2,3——full 未跑完时）")
    args = ap.parse_args()
    todo = {s.strip() for s in args.only.split(",")}
    if "1" in todo:
        fig_6_1()
        print("[fig 6.1] done")
    if "2" in todo:
        fig_6_2()
        print("[fig 6.2] done")
    if "3" in todo:
        fig_6_3()
        print("[fig 6.3] done")
    if "4" in todo:
        fig_6_4()
