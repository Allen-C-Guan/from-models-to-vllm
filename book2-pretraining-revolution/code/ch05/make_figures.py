# 用途：Book2 ch5 两张图——图 5.1 text-to-text 统一接口四算例 / 图 5.2 三种架构对照（塔排布 + 「谁看见谁」mask）
# 所属章节：Book2 第 5 章 §5.1、§5.3-5.4
# 运行方式：cd 工作区根目录 && source env.sh && python code/ch05/make_figures.py
# 产物：figures/fig-5-{1,2}-*.png（300dpi，图内文字英文、图注见正文）
# 数据来源：全部数字/字符串取自 T5 v4（= JMLR 21(2020) 20-074）Figure 1 p.3、Table 2、§3.2、附录 D.15（本地缓存 log/research-cache/t5/t5_fulltext.txt 逐字核对，2026-10）

import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(REPO, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MUTED = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
LBLUE, GREY, LORANGE = "#cde2fb", "#f0efec", "#fde9df"  # mask 可见/遮挡/前缀-目标条
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": TICK, "axes.labelcolor": DARK,
    "xtick.color": TICK, "ytick.color": TICK, "text.color": DARK,
    "axes.linewidth": 0.8, "font.family": "DejaVu Sans",
})


def box(ax, x, y, w, h, label, fc="white", ec=DARK, lw=1.2, fs=8.5, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc or DARK)


def arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=lw, linestyle=ls))


def two_color_text(ax, x, y, s1, s2, fs=6.0, c1=BLUE, c2=DARK):
    """s1（任务前缀）着蓝、s2 着深色，宽度实测后拼接——保证逐字串两种颜色无缝衔接。"""
    t1 = ax.text(x, y, s1, fontsize=fs, color=c1, ha="left", va="center",
                 family="DejaVu Sans Mono")
    if not s2:
        return
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    w1 = t1.get_window_extent(renderer=r).transformed(ax.transData.inverted()).width
    ax.text(x + w1, y, s2, fontsize=fs, color=c2, ha="left", va="center",
            family="DejaVu Sans Mono")


# ---------------- 图 5.1：text-to-text 四算例（自绘；翻译/分类/回归逐字取自 v4 Figure 1，QA 取自 v4 附录 D.15） ----------------
def fig_5_1():
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=300)
    ax.set_xlim(0, 17.2)
    ax.set_ylim(0, 10.6)
    ax.axis("off")

    ax.text(8.6, 10.15, "Text-to-Text: every NLP task is text in → text out",
            ha="center", fontsize=10.5, weight="bold")
    ax.text(8.6, 9.62, "one model, one loss, one decoding procedure (after Raffel et al. 2020, Fig.1 + App. D.15)",
            ha="center", fontsize=6.5, color=MUTED)

    rows = [
        ("Translation", TEAL, "translate English to German: ", "That is good.", "Das ist gut."),
        ("Classification (CoLA)", BLUE, "cola sentence: ", "The course is jumping well.", "not acceptable"),
        ("Question answering (SQuAD)", ORANGE, "",
         "question: What does increased oxygen concentrations in the patient’s lungs displace? "
         "context: Hyperbaric (high-pressure) medicine uses special oxygen chambers to …",
         "carbon monoxide"),
        ("Regression (STS-B)", YELLOW, "stsb ",
         "sentence1: The rhino grazed on the grass. sentence2: A rhino is grazing in a field.", "3.8"),
    ]
    centers = [8.15, 6.05, 3.95, 1.85]
    h = 1.7

    ax.add_patch(FancyBboxPatch((9.1, 4.3), 2.5, 2.5, boxstyle="round,pad=0.04",
                                fc="#f7fafe", ec=BLUE, lw=1.6))
    ax.text(10.35, 5.55, "T5", ha="center", fontsize=11, weight="bold", color=BLUE)
    ax.text(10.35, 4.95, "one model\none loss\none decoding", ha="center", fontsize=6.8, color=MUTED)

    left_ys = [6.55, 5.9, 5.25, 4.6]
    for i, ((task, color, prefix, body, out), cy) in enumerate(zip(rows, centers)):
        y = cy - h / 2
        ax.add_patch(FancyBboxPatch((0.15, y), 8.0, h, boxstyle="round,pad=0.02",
                                    fc="white", ec=DARK, lw=1.0))
        ax.add_patch(Rectangle((0.15, y), 0.16, h, fc=color, ec="none"))
        ax.text(0.48, y + h - 0.32, task, fontsize=7.2, weight="bold")
        full = prefix + body
        lines = textwrap.wrap(full, width=68)[:3]
        for j, ln in enumerate(lines):
            if j == 0 and prefix:
                two_color_text(ax, 0.48, y + 0.82 - 0.37 * j, prefix,
                               ln[len(prefix):] if ln.startswith(prefix) else "")
            else:
                ax.text(0.48, y + 0.82 - 0.37 * j, ln, fontsize=6.0, va="center",
                        family="DejaVu Sans Mono")
        ax.add_patch(FancyBboxPatch((12.9, y), 4.1, h, boxstyle="round,pad=0.02",
                                    fc="#fdf8ef", ec=DARK, lw=1.0))
        ax.text(14.95, cy, out, ha="center", va="center", fontsize=7.4, weight="bold",
                family="DejaVu Sans Mono")
        arrow(ax, 8.15, cy, 9.1, left_ys[i])
        arrow(ax, 11.6, left_ys[i], 12.9, cy)
    ax.text(8.58, 7.62, "token ids (n_in,)", fontsize=6.0, color=MUTED)
    ax.text(11.72, 2.98, "token ids (n_out,)", fontsize=6.0, color=MUTED)

    ax.text(8.6, 0.32, "blue = task prefix (routes the task) · SQuAD carries no prefix — "
                       "its fields question: / context: are self-explanatory · batch dim B omitted",
            ha="center", fontsize=6.0, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-5-1-text-to-text.png"), facecolor="white")
    plt.close(fig)


# ---------------- 图 5.2：三种架构对照（塔排布 + 8×8「谁看见谁」mask；GLUE/参数/算力引 v4 Table 2 去噪行） ----------------
def mask_ok(kind, r, c):
    """r=查询行（0-7），c=被看见列；前 4 位为前缀/输入段、后 4 位为目标段。"""
    if kind == "c":                      # decoder-only LM：纯因果下三角
        return c <= r
    if kind == "ab":                     # enc-dec 与 prefix LM：可见性矩阵相同
        return c < 4 if r < 4 else (c < 4 or c <= r)
    raise ValueError(kind)


def draw_mask(ax, x0, y0, kind, cell=0.34, ticks=("prefix", "target")):
    n = 8
    for r in range(n):
        for c in range(n):
            fc = LBLUE if mask_ok(kind, r, c) else GREY
            ax.add_patch(Rectangle((x0 + c * cell, y0 + (n - 1 - r) * cell), cell, cell,
                                   fc=fc, ec="white", lw=0.4))
    # 前缀/目标分界（橙色）
    ax.plot([x0 + 4 * cell, x0 + 4 * cell], [y0 - 0.02, y0 + n * cell + 0.02], color=ORANGE, lw=1.2)
    ax.plot([x0 - 0.02, x0 + n * cell + 0.02], [y0 + 4 * cell, y0 + 4 * cell], color=ORANGE, lw=1.2)
    ax.text(x0 + 2 * cell, y0 + n * cell + 0.18, ticks[0], ha="center", fontsize=5.8, color=MUTED)
    ax.text(x0 + 6 * cell, y0 + n * cell + 0.18, ticks[1], ha="center", fontsize=5.8, color=MUTED)
    ax.text(x0 + n * cell + 0.55, y0 + n * cell / 2, "queries\n(rows)", fontsize=5.8,
            color=MUTED, va="center")
    ax.text(x0 + n * cell / 2, y0 - 0.30, "keys (columns) →", ha="center", fontsize=5.8, color=MUTED)
    # 图例
    ax.add_patch(Rectangle((x0 + n * cell + 0.5, y0 + 6.4 * cell), 0.42, 0.26, fc=LBLUE, ec="white", lw=0.4))
    ax.text(x0 + n * cell + 1.0, y0 + 6.53 * cell, "visible", fontsize=5.8, va="center", color=MUTED)
    ax.add_patch(Rectangle((x0 + n * cell + 0.5, y0 + 5.6 * cell), 0.42, 0.26, fc=GREY, ec="white", lw=0.4))
    ax.text(x0 + n * cell + 1.0, y0 + 5.73 * cell, "masked", fontsize=5.8, va="center", color=MUTED)


def fig_5_2():
    fig, ax = plt.subplots(figsize=(8.6, 4.9), dpi=300)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10.2)
    ax.axis("off")

    panels = [(0.25, "(a) Encoder-Decoder", "two stacks + explicit cross-attn bridge (T5 baseline)",
               "input", "target", "GLUE 83.28", "2P params · M FLOPs", "ab"),
              (6.45, "(b) Prefix LM", "one shared stack — bridge replaced by full attention",
               "prefix", "target", "GLUE 81.82", "P params · M FLOPs", "ab"),
              (12.65, "(c) Decoder-only LM", "one stack — pure causal, no prefix visibility",
               "pos 1-4", "pos 5-8", "GLUE 74.70", "P params · M FLOPs", "c")]

    for px, title, sub, t0, t1, score, pm, kind in panels:
        ax.text(px + 2.55, 9.72, title, ha="center", fontsize=9.2, weight="bold")
        ax.text(px + 2.55, 9.28, sub, ha="center", fontsize=6.4, color=MUTED)

        if title.startswith("(a)"):
            box(ax, px + 0.30, 6.0, 1.7, 2.4, "Encoder\n× 12\n(B, n_in, 768)", fc="#f7fafe", ec=BLUE, lw=1.5, fs=7.2)
            box(ax, px + 3.10, 6.0, 1.7, 2.4, "Decoder\n× 12\n(B, n_out, 768)", fc="#f7fafe", ec=BLUE, lw=1.5, fs=7.2)
            arrow(ax, px + 2.0, 7.2, px + 3.1, 7.2, color=ORANGE, lw=1.8)
            ax.text(px + 2.55, 7.62, "cross-attn", ha="center", fontsize=6.2, color=ORANGE)
            ax.text(px + 2.55, 6.42, "K,V ← enc out\n(B, n_in, 768)", ha="center", fontsize=5.2, color=MUTED)
            arrow(ax, px + 1.15, 5.1, px + 1.15, 6.0)
            ax.text(px + 1.15, 4.82, "input ids (n_in,)", ha="center", fontsize=5.8, color=MUTED)
            arrow(ax, px + 3.95, 5.1, px + 3.95, 6.0)
            ax.text(px + 3.95, 4.82, "target ids (n_out,)", ha="center", fontsize=5.8, color=MUTED)
            arrow(ax, px + 3.95, 8.4, px + 3.95, 8.95)
            ax.text(px + 3.95, 9.02, "logits (B, n_out, |V|)", ha="center", fontsize=5.6, color=MUTED)
        elif title.startswith("(b)"):
            box(ax, px + 1.0, 6.0, 3.0, 2.4, "Shared stack\n× 12\n(B, n, 768)", fc="#f7fafe", ec=BLUE, lw=1.5, fs=7.2)
            ax.add_patch(FancyBboxPatch((px + 1.0, 5.05), 1.5, 0.6, boxstyle="round,pad=0.02", fc=LBLUE, ec=DARK, lw=0.8))
            ax.text(px + 1.75, 5.35, "prefix x", ha="center", va="center", fontsize=6.4)
            ax.add_patch(FancyBboxPatch((px + 2.5, 5.05), 1.5, 0.6, boxstyle="round,pad=0.02", fc=LORANGE, ec=DARK, lw=0.8))
            ax.text(px + 3.25, 5.35, "target y", ha="center", va="center", fontsize=6.4)
            arrow(ax, px + 2.5, 5.65, px + 2.5, 6.0)
            ax.text(px + 0.9, 5.35, "ids (n,)", ha="right", fontsize=5.8, color=MUTED)
            arrow(ax, px + 2.5, 8.4, px + 2.5, 8.95)
            ax.text(px + 2.5, 9.02, "logits (B, n, |V|)", ha="center", fontsize=5.6, color=MUTED)
        else:
            box(ax, px + 1.0, 6.0, 3.0, 2.4, "Stack\n× 12\n(B, n, 768)", fc="#f7fafe", ec=BLUE, lw=1.5, fs=7.2)
            ax.add_patch(FancyBboxPatch((px + 1.0, 5.05), 3.0, 0.6, boxstyle="round,pad=0.02", fc=GREY, ec=DARK, lw=0.8))
            ax.text(px + 2.5, 5.35, "one causal stream:  x ‖ y", ha="center", va="center", fontsize=6.4)
            arrow(ax, px + 2.5, 5.65, px + 2.5, 6.0)
            ax.text(px + 0.9, 5.35, "ids (n,)", ha="right", fontsize=5.8, color=MUTED)
            arrow(ax, px + 2.5, 8.4, px + 2.5, 8.95)
            ax.text(px + 2.5, 9.02, "logits (B, n, |V|)", ha="center", fontsize=5.6, color=MUTED)

        ax.text(px + 1.9, 4.35, "who-sees-whom mask", fontsize=6.2, color=MUTED)
        draw_mask(ax, px + 0.55, 1.45, kind, ticks=(t0, t1))
        ax.text(px + 0.55, 0.62, score, fontsize=8.2, weight="bold")
        ax.text(px + 0.55 + 1.05, 0.62, "·  " + pm, fontsize=6.6, color=MUTED)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-5-2-three-architectures.png"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_5_1()
    fig_5_2()
    print("figures written to", FIG_DIR)
