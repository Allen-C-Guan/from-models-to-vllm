# fig_ch11.py —— Book5 ch11 制图件（图 11.1/11.2，本章专属 slug：selfreport/swev 前缀防跨册冲撞）
# 用途：ch11 两张图——
#   图 11.2 读自报数字五步法流程图（raw → 口径 → 第三方交叉 → 复算 → 悬案登记，自绘示意级）；
#   图 11.1 SWE-V 双数字口径解剖图（77.8 厂商自报 vs 72.8 第三方榜——五维对照，自绘示意级；
#         数字正档=log/book5-ch11/swe_archive/（2026-10-08 一手存档））。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch11/fig_ch11.py"
# 产物：figures/fig-11-{1,2}-<slug>.png（figures 随书；log/ 不入 git）
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "book5-frontier-hybrid-architectures", "figures")

# 规范 §4 印刷规格常量（与 ch01-10 制图件同款）
BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
DARK, GREY, TICK, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
FILL_L, FILL_O = "#eaf2fc", "#fdf1ea"   # 浅蓝/浅橙底（ch04 同款）


def fig_11_2():
    """图 11.2：读自报数字五步法——raw → 口径 → 第三方交叉 → 复算可算的 → 悬案登记。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.1), dpi=300, layout="tight")
    ax.axis("off"); ax.set_xlim(0, 16); ax.set_ylim(0, 9.6)

    def rbox(x, y, w, h, text, fc, ec, fs=7.0, lw=1.3):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw,
                                   joinstyle="round"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=DARK)

    def arrow(x0, y0, x1, y1, color=DARK, lw=1.4):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))

    ax.text(8.0, 9.25, "How to read a vendor-reported number: a five-step checklist",
            ha="center", fontsize=10.5, color=DARK, weight="bold")

    # 输入
    rbox(0.2, 6.1, 2.5, 1.7, "a number on a\nmodel card / README /\npaper table", "white", TICK, fs=7.0)
    # 五步
    steps = [
        ("step 1\nfind the RAW\nnumber",
         "read the table row\nAND its footnotes;\nno blog relay", FILL_O),
        ("step 2\nread the caliber",
         "benchmark & version /\nharness & prompt /\nsampling, context, tier", FILL_L),
        ("step 3\ncross-check\n3rd party",
         "swebench.com / vals.ai /\neval-results / lmarena\n(each has limits too)", FILL_L),
        ("step 4\nrecompute\nwhat you can",
         "param accounts from\nconfig & shard headers\n(params_audit.py)", FILL_O),
        ("step 5\nregister\nthe rest",
         "undisclosed items\nare also information:\nkeep an open-case list", FILL_L),
    ]
    x = 3.15
    for title, sub, fc in steps:
        rbox(x, 5.7, 2.4, 1.15, title, fc, DARK, fs=7.6)
        rbox(x, 6.95, 2.4, 1.5, sub, "white", TICK, fs=6.4, lw=0.9)
        arrow(x - 0.22, 6.28, x - 0.02, 6.28)
        x += 2.55
    arrow(2.7, 6.95, 3.15, 6.95)

    # 反馈箭头：第 3 步与第 4 步都会反哺第 2 步（口径的发现路径）
    ax.annotate("", xy=(5.9, 6.9), xytext=(11.9, 5.6),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.1,
                                connectionstyle="arc3,rad=0.25"))
    ax.text(9.3, 4.55, "steps 3 & 4 keep feeding caliber facts back into step 2",
            ha="center", fontsize=7.0, color=GREY)

    # 底部两条判词
    rbox(0.2, 2.9, 15.5, 1.0,
         "every cited number carries its caliber tag  {vendor-reported / third-party / config / checkpoint}  —  a number without a caliber is not a fact, it is a slogan",
         FILL_L, BLUE, fs=7.6)
    rbox(0.2, 1.4, 15.5, 1.0,
         "sibling of the config-reverse five steps (Book3 ch.9):  family -> blueprint -> account ->\nperiphery -> grading & open cases   —   same discipline, applied to benchmark numbers",
         "white", TICK, fs=7.0, lw=1.0)
    ax.text(8.0, 0.55, "checklist walked in sec 11.2; the SWE-V case (fig. 11.1) is its anatomy specimen",
            ha="center", fontsize=7.0, color=GREY)

    path = os.path.join(FIG_DIR, "fig-11-2-selfreport-five-steps.png")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def fig_11_1():
    """图 11.1：GLM-5 SWE-bench Verified 双数字——五维口径解剖（数字=swe_archive 一手存档）。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300, layout="tight")
    ax.axis("off"); ax.set_xlim(0, 16); ax.set_ylim(0, 10.4)

    ax.text(8.0, 10.05, "one model, one exam — SWE-bench Verified (500 real GitHub tasks) — two published scores",
            ha="center", fontsize=9.8, color=DARK, weight="bold")

    def panel(x, score, who, items, ec, fc):
        ax.add_patch(plt.Rectangle((x, 3.7), 5.1, 5.6, facecolor="white", edgecolor=ec, lw=1.8))
        ax.text(x + 2.55, 8.75, score, ha="center", fontsize=21, color=ec, weight="bold")
        ax.text(x + 2.55, 8.05, who, ha="center", fontsize=8.2, color=DARK, weight="bold")
        for i, it in enumerate(items):
            ax.text(x + 0.25, 7.45 - i * 0.62, "· " + it, fontsize=7.2, color=DARK)

    panel(0.3, "77.8", "vendor · GLM-5 model card", [
        "OpenHands framework (full scaffold)",
        "tailored instruction prompt",
        "temperature 0.7, top_p 0.95",
        "max_new_tokens 16,384; 200K ctx",
        "reasoning tier: high",
    ], ORANGE, FILL_O)
    panel(10.6, "72.8", "3rd party · swebench.com leaderboard", [
        "mini-SWE-agent (~100-line bash-only,",
        "   no tools, no scaffold)",
        "leaderboard standard settings",
        "agent 2.0.0  (1.x declared incomparable)",
        "$0.53/instance · 2026-02-17 · 364/500 self-check",
    ], BLUE, FILL_L)

    # 中间五维
    dims = [
        ("exam & tasks", "SAME 500 tasks", TEAL),
        ("harness & scaffold", "differs  <- the bulk of the gap", ORANGE),
        ("sampling & context", "differs (vendor sets its own)", ORANGE),
        ("tier & versions", "differs (high tier; agent 2.x vs 1.x)", ORANGE),
        ("cost & compute", "leaderboard only ($ column)", BLUE),
    ]
    ax.text(8.0, 9.15, "five dimensions", ha="center", fontsize=8.6, color=DARK, weight="bold")
    for i, (d, v, c) in enumerate(dims):
        y = 8.35 - i * 1.02
        ax.text(8.0, y, d, ha="center", fontsize=7.4, color=DARK, weight="bold")
        ax.text(8.0, y - 0.34, v, ha="center", fontsize=6.6, color=c)

    # 判词条
    ax.add_patch(plt.Rectangle((0.3, 1.5), 15.4, 1.55, facecolor="#f4f3f0", edgecolor=GREY, lw=1.0))
    ax.text(8.0, 2.72, "verdict: nobody lied — they answer two different questions", ha="center",
            fontsize=8.6, color=DARK, weight="bold")
    ax.text(8.0, 2.02, "72.8 = \"bare model handed a bash shell\"   ·   77.8 = \"vendor's best assembly\"   —   "
                       "first question for any number: which layer was it assembled to?",
            ha="center", fontsize=7.4, color=GREY)
    ax.text(8.0, 0.75, "sources: GLM-5 HF model card (table row + footnote) & swebench.com Verified leaderboard row,\n"
                       "archived 2026-10-08 (log/book5-ch11/swe_archive/); peer rows on the card vs the leaderboard differ by 3-7 pts",
            ha="center", fontsize=6.8, color=TICK)

    path = os.path.join(FIG_DIR, "fig-11-1-swev-dual-anatomy.png")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    for fn in (fig_11_1, fig_11_2):
        print(f"[图] {fn.__name__} -> {fn()}")
    print("ch11 fig_ch11.py 完成。")
