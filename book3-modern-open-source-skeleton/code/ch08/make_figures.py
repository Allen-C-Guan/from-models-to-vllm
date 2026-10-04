# make_figures.py —— Book3 ch8 三张图的绘制脚本（图 8.1 整机架构 / 8.2 代际点亮 / 8.3 插槽消融收拢）
# 用途：产出 figures/fig-8-{1,2,3}-*.png（300dpi 印刷规格，图内英文、
#       图注中文在正文）。图 8.1/8.2 为自绘示意级（不依赖实验数据）；图 8.3 汇总第 2/3/4/6 章
#       消融产物（log/book3-ch0N/ablation_*_full.json，缺 full 自动回退 fast）。
# 所属章节：Book3 ch8（8.1 图 8.1 / 8.5 图 8.2 / 8.6 图 8.3；正文数字与判读见 ch08-LLaMA谱系.md）。
# 运行方式：cd 工作区根目录 && source env.sh && \
#           python code/ch08/make_figures.py [--only 1,2,3]
# 规格：写作规范 §4——蓝 #2a78d6 / 橙 #eb6834 / 青 #1baf7a / 黄 #eda100；线宽 2；网格 hairline
#       #e1e0d9 置底层；刻度 #898781；标注 #0b0b0b / #52514e；禁双 y 轴；tight_layout；禁 jet/rainbow。
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DIR = os.path.join(REPO, "figures")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MID = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
LIT = {1: "#d7e6f9", 2: "#fde2d3", 3: "#d3efe3"}     # 三代点亮底色（蓝/橙/青浅档）
GRAY = "#f4f3ef"                                      # 沿用不变底色
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9.5,
    "axes.labelcolor": DARK, "xtick.color": TICK, "ytick.color": TICK,
    "font.family": "sans-serif",
})


# ---------------- 图 8.1：LLaMA 整机架构（自绘重制，LLaMA1-7B 几何） ----------------
def fig_8_1():
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")

    def box(x, y, w, h, label, sub="", ec=MID, fc="white", lw=1.3, ls="-", fs=8.2, sub_fs=6.6,
             sub_dy=2.8):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
        cy = y + h / 2
        ax.text(x + w / 2, cy + (sub_dy + 0.6 if sub else 0), label, ha="center", va="center",
                fontsize=fs, color=DARK, zorder=3)
        if sub:
            ax.text(x + w / 2, cy - sub_dy + (0.6 if "\n" not in sub else 0), sub, ha="center",
                    va="center", fontsize=sub_fs, color=MID, zorder=3, linespacing=1.3)

    def arrow(x1, y1, x2, y2, shape_label="", color=MID):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4), zorder=1)
        if shape_label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.0, shape_label, ha="center",
                    va="bottom", fontsize=6.5, color=MID)

    # 入口与嵌入（出入口两件用灰边；四件套彩色边与顶部出处行对应）
    box(1.0, 25.5, 10.0, 9, "token ids", "(B, n)")
    arrow(11.0, 30.0, 13.5, 30.0)
    box(13.5, 25.5, 11.5, 9, "wte lookup", "32000 x 4096")
    arrow(25.0, 30.0, 27.5, 30.0, "(B, n, 4096)")
    # 被拆除的 wpe（虚影）
    box(13.5, 12.0, 11.5, 7.5, "wpe 1024x768", "removed (GPT-2)", ec=TICK,
        lw=1.0, ls=(0, (3, 3)), fs=7.4)
    ax.annotate("", xy=(19.25, 25.5), xytext=(19.25, 19.5),
                arrowprops=dict(arrowstyle="-|>", color=TICK, lw=1.0, ls=(0, (3, 3))))
    ax.text(19.25, 6.8, "no positional table at the input:\nposition enters inside attention",
            fontsize=6.6, color=MID, ha="center", va="center")

    # Block 区域（x32）
    ax.add_patch(plt.Rectangle((27.5, 0.5), 34, 55.5, fc="#fbfaf7", ec=MID, lw=1.3, zorder=0))
    ax.text(28.5, 56.6, "x L = 32 blocks", ha="left", va="bottom", fontsize=9.0, color=DARK)
    box(31.0, 47.5, 20.0, 5.5, "RMSNorm (pre)", ec=TEAL, lw=1.7)
    box(31.0, 33.5, 20.0, 9.0, "Attention: MHA  h=32", "RoPE on Q/K: (B,32,n,128)",
        ec=BLUE, lw=1.7, sub_dy=2.2)
    box(33.5, 28.7, 15.5, 3.0, "GQA h_kv=8  (L2-70B, L3)", ec=YELLOW, lw=1.1,
        ls=(0, (3, 2)), fs=6.0)
    box(40.5, 22.5, 4.0, 4.0, "+", fs=9.5)
    box(31.0, 16.0, 20.0, 5.5, "RMSNorm (pre)", ec=TEAL, lw=1.7)
    box(31.0, 4.5, 20.0, 9.5, "SwiGLU FFN",
        "(B,n,4096) -> (B,n,11008) x2\n-> mul -> down -> (B,n,4096)", ec=ORANGE, lw=1.7,
        sub_fs=6.2, sub_dy=2.6)
    box(40.5, 1.0, 3.4, 3.0, "+", fs=9.5)
    # 残差河（左侧两条跨子层弧线）
    for y0, y1 in ((49.5, 24.5), (16.8, 2.6)):
        ax.annotate("", xy=(42.5, y1), xytext=(29.5, y0),
                    arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.3, alpha=0.85,
                                    connectionstyle="arc3,rad=0.35"))
    ax.text(23.5, 38.5, "residual", fontsize=7.0, color=TEAL, rotation=90, ha="center", va="center")

    # 出口
    arrow(61.5, 30.0, 64.5, 30.0, "(B, n, 4096)")
    box(64.5, 25.5, 10.5, 9, "RMSNorm", "(final)", ec=TEAL, lw=1.7)
    arrow(75.0, 30.0, 78.0, 30.0)
    box(78.0, 25.5, 9.5, 9, "lm_head", "4096x32000\n(untied)")
    arrow(87.5, 30.0, 89.8, 30.0)
    ax.text(92.5, 30.0, "logits\n(B, n, 32000)", fontsize=7.2, color=DARK, ha="center", va="center")

    # 四件套出处行（顶部，颜色与部件边框对应）
    prov = [("RMSNorm 2019\n1910.07467", 16.0, TEAL), ("RoPE 2021\n2104.09864", 38.0, BLUE),
            ("SwiGLU 2020\n2002.05202", 60.0, ORANGE), ("GQA 2023\n2305.13245", 82.0, YELLOW)]
    for label, x, c in prov:
        ax.text(x, 63.3, label, fontsize=7.0, color=c, ha="center", va="center", linespacing=1.3)
    ax.plot([8.0, 90.0], [60.8, 60.8], color=GRID, lw=0.8)

    ax.set_title("LLaMA decoder block (7B geometry): four slots, four papers", fontsize=11, color=DARK)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-8-1-llama1-7b-machine.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[saved] {out}")


# ---------------- 图 8.2：代际组件点亮图（骨架 vs 配方） ----------------
ROWS = [  # (行名, [(单元格文字, 是否该代改动)])
    ("norm slot",        [("pre-RMSNorm\n(2019)", 1), ("unchanged", 0), ("unchanged", 0)]),
    ("FFN slot",         [("SwiGLU 8/3 d\n(2020)", 1), ("same slot\n(70B: f=3.5d)", 0), ("unchanged", 0)]),
    ("position slot",    [("RoPE, base 1e4\n(2021)", 1), ("base 1e4", 0), ("base 5e5", 1)]),
    ("attention slot",   [("MHA\n(h=h_kv)", 1), ("GQA-8\n(34B/70B only)", 1), ("GQA-8\nall sizes", 1)]),
    ("vocabulary",       [("32k sp-BPE", 1), ("32k", 0), ("128k tiktoken\n(128,256)", 1)]),
    ("context window",   [("2,048", 1), ("4,096", 1), ("8,192 -> 128K\n(3.1, 6 stages)", 1)]),
    ("training tokens",  [("1.0T / 1.4T", 1), ("2.0T", 1), ("~15T\n(flagship 15.6T)", 1)]),
    ("license",          [("research\nonly", 1), ("commercial\n(700M MAU)", 1), ("community\n(2024-04/07)", 1)]),
]
COLS = ["LLaMA 1\n2023-02", "LLaMA 2\n2023-07", "Llama 3\n2024-04/07"]


def fig_8_2():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")
    x0, cw = 24, 24            # 网格起点与列宽
    ytop, rh = 54.0, 6.2       # 网格顶与行高
    # 列头
    for j, name in enumerate(COLS):
        ax.text(x0 + cw * j + cw / 2, ytop + 2.2, name, ha="center", va="center",
                fontsize=8.6, color=DARK)
    # 行与格
    for i, (row, cells) in enumerate(ROWS):
        y = ytop - (i + 1) * rh
        ax.text(x0 - 1.5, y + rh / 2, row, ha="right", va="center", fontsize=8.2, color=DARK)
        for j, (txt, lit) in enumerate(cells):
            fc = LIT[j + 1] if lit else GRAY
            ax.add_patch(plt.Rectangle((x0 + cw * j + 0.5, y + 0.4), cw - 1.0, rh - 0.8,
                                       fc=fc, ec="#d9d7cf", lw=0.7, zorder=2))
            ax.text(x0 + cw * j + cw / 2, y + rh / 2, txt, ha="center", va="center",
                    fontsize=7.0, color=DARK, zorder=3)
    # 架构区分组框（上四行）
    y_box_top, y_box_bot = ytop - 0.2, ytop - 4 * rh - 0.4
    ax.add_patch(plt.Rectangle((x0 - 0.2, y_box_bot), 3 * cw + 0.4, y_box_top - y_box_bot,
                               fc="none", ec=MID, lw=1.1, ls=(0, (4, 3)), zorder=4))
    ax.text(x0 + 3 * cw + 0.6, (y_box_top + y_box_bot) / 2, "architecture\nslots", fontsize=7.2,
            color=MID, ha="left", va="center", rotation=90)
    ax.text(x0 - 1.5, y_box_top + 1.0, "frozen after 2023-02", fontsize=7.0, color=MID,
            ha="right", va="bottom", style="italic")
    ax.text(x0 - 1.5, y_box_bot - 1.2, "recipe & periphery: rewritten every generation",
            fontsize=7.0, color=MID, ha="right", va="top", style="italic")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=LIT[k], ec="none") for k in (1, 2, 3)]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc=GRAY, ec="none"))
    ax.legend(handles, ["introduced / changed in L1", "in L2", "in L3", "carried over unchanged"],
              loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=7.2, frameon=False)
    ax.set_title("The skeleton froze in 2023; every generation rewrote the recipe", fontsize=11, color=DARK)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-8-2-generational-slots.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[saved] {out}")


# ---------------- 图 8.3：四插槽换件消融收拢（自产数据） ----------------
def _load(path_full, path_fast, *keys):
    for p in (path_full, path_fast):
        if os.path.exists(p):
            d = json.load(open(p))
            for k in keys:
                d = d[k]
            return d, ("full" if "full" in p else "fast")
    raise FileNotFoundError(f"missing {path_full} and {path_fast}")


def fig_8_3():
    lg = os.path.join(REPO, "log")
    arms = {}  # (组名, config 注记, 判词) -> [(臂名, eval_final)]
    d, _ = _load(f"{lg}/book3-ch02/ablation_norm_full.json", f"{lg}/book3-ch02/ablation_norm_fast.json", "arms")
    arms[("norm slot  (ch2, d=512)", "equivalence check",
          "new pays +0.043\n(out of +/-0.02 band)")] = [("LayerNorm", d["base"]["eval_final"]),
                                                        ("RMSNorm", d["rmsnorm"]["eval_final"])]
    d, _ = _load(f"{lg}/book3-ch03/ablation_ffn_full.json", f"{lg}/book3-ch03/ablation_ffn_fast.json", "arms")
    arms[("FFN slot  (ch3, d=768, param-aligned)", "same budget",
          "new is 0.195 lower")] = [("GELU-4C", d["gelu4c"]["eval_final"]),
                                    ("SwiGLU-8/3C", d["swiglu83"]["eval_final"])]
    d, _ = _load(f"{lg}/book3-ch04/ablation_pe_full.json", f"{lg}/book3-ch04/ablation_pe_fast.json", "arms")
    arms[("PE slot  (ch4, d=512)", "hard wall removed",
          "new is 0.411 lower")] = [("wpe", d["wpe"]["eval_final"]),
                                    ("RoPE", d["rope"]["eval_final"])]
    d, _ = _load(f"{lg}/book3-ch06/ablation_gqa_full.json", f"{lg}/book3-ch06/ablation_gqa_fast.json", "arms")
    arms[("attention slot  (ch6, d=512)", "quality-for-KV trade",
          "KV/token 8:4:1 cheaper;\nquality pays +0.109 / +0.158")] = [
        ("MHA (h_kv=8)", d["mha8"]["eval_final"]),
        ("GQA-4", d["gqa4"]["eval_final"]),
        ("MQA", d["mqa1"]["eval_final"])]
    groups = list(arms.keys())
    ys, labels, vals, colors = [], [], [], []
    y = 0
    for g in groups:
        for name, v in arms[g]:
            ys.append(y)
            labels.append(name)
            vals.append(v)
            colors.append(BLUE if name in ("LayerNorm", "GELU-4C", "wpe", "MHA (h_kv=8)") else ORANGE)
            y += 1
        y += 0.9                              # 组间距
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x_left = 4.0
    ax.barh(ys, [v - x_left for v in vals], left=x_left, height=0.62, color=colors, edgecolor="none", zorder=3)
    for yy, v in zip(ys, vals):
        ax.text(v + 0.006, yy, f"{v:.3f}", va="center", ha="left", fontsize=7.6, color=DARK, zorder=4)
    # 组名与判词注记
    y = 0
    for g in groups:
        n = len(arms[g])
        ax.text(x_left - 0.04, y + (n - 1) / 2, f"{g[0]}\n[{g[1]}]", ha="right", va="center",
                fontsize=7.3, color=MID)
        ax.text(4.90, y + (n - 1) / 2, g[2], ha="right", va="center", fontsize=7.3, color=MID)
        y += n + 0.9
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8.0)
    ax.invert_yaxis()                          # 自上而下按组序排列
    ax.set_xlim(x_left, 4.90)
    ax.set_ylim(len(ys) + len(groups) * 0.9 - 0.6, -1.4)
    ax.set_xlabel("fixed-window eval loss (nat, 401,408 tokens, fp32)", fontsize=8.6)
    ax.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
    ax.grid(axis="y", visible=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(TICK)
    ax.spines["bottom"].set_color(TICK)
    ax.tick_params(labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=BLUE), plt.Rectangle((0, 0), 1, 1, fc=ORANGE)]
    ax.legend(handles, ["stock part (GPT-2)", "LLaMA-style part"], loc="upper center",
              bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=7.8, frameon=False)
    ax.set_title("Four slots, four controlled swaps: 20-50M module tier", fontsize=11, color=DARK)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-8-3-slot-contribution.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[saved] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="1,2,3", help="只画指定图（如 --only 1,2）")
    which = set(int(x) for x in ap.parse_args().only.split(",") if x.strip())
    os.makedirs(FIG_DIR, exist_ok=True)
    if 1 in which:
        fig_8_1()
    if 2 in which:
        fig_8_2()
    if 3 in which:
        fig_8_3()


if __name__ == "__main__":
    main()
