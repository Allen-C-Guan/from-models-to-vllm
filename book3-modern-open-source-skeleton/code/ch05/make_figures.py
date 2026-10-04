# make_figures.py —— Book3 ch5 四张图：图 5.1 tie 四拨时间线 / 图 5.2 词表-嵌入占比 / 图 5.3 中英压缩对拍 / 图 5.4 五步流程
# 用途：图 5.1 自绘示意级（tie 开关四拨 + 词表账占比，数字源 = config_infer 的 tie 四拨账）；
#       图 5.2 自产（左：d=768 玩具骨架公式曲线，数字源 = config_infer 固定骨架换词表；右：本册 512 模块档四臂实测，
#            数字源 = log/book3-ch05/vocab_scan_fast.json）；图 5.3 自产（固定文本压缩对拍，数字源 = 调研探针
#            probe_ch5_vocab_config_run2.json 的固定文本实测）；图 5.4 自绘示意级（config 反推五步流程）。
# 所属章节：Book3 第 5 章 §5.2（图 5.1）/§5.4（图 5.2）/§5.4（图 5.3）/§5.6（图 5.4）
# 运行方式：cd 工作区根目录 && source env.sh && python code/ch05/make_figures.py
#           （依赖 config_infer.py 与 vocab_scan.py 已各自跑过一次产出 JSON；缺失时报错并提示先跑）
# 产物：figures/fig-5-{1,2,3,4}-*.png（300dpi；图内英文、图注中文见正文）
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(REPO, "figures")
CH05_LOG = os.path.join(REPO, "log", "book3-ch05")
PROBE_JSON = os.path.join(REPO, "log", "book3-feasibility", "probe_ch5_vocab_config_run2.json")
os.makedirs(FIG_DIR, exist_ok=True)

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MUTED = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": TICK, "axes.labelcolor": DARK,
    "xtick.color": TICK, "ytick.color": TICK, "text.color": DARK,
    "axes.linewidth": 0.8, "font.family": "DejaVu Sans",
})


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def latest_config_infer():
    """取最新的 config_infer_*.json（数字源：tie 四拨账 + 固定骨架换词表）。"""
    cands = sorted(f for f in os.listdir(CH05_LOG) if f.startswith("config_infer_") and f.endswith(".json"))
    if not cands:
        raise SystemExit("未找到 config_infer_*.json——先跑 code/ch05/config_infer.py")
    return load_json(os.path.join(CH05_LOG, cands[-1]))


def box(ax, x, y, w, h, label, fc="white", ec=DARK, lw=1.1, fs=7.6, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc or DARK)


# ---------------- 图 5.1：tie 四拨时间线（自绘示意级） ----------------
def fig_5_1():
    ci = latest_config_infer()
    rows = ci["step3_tie_switch"]          # 四拨账（config_infer 实算）
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(7.0, 9.6, "One switch, four flips: tie_word_embeddings across generations",
            ha="center", fontsize=9.5, color=DARK, fontweight="bold")

    xs = [1.9, 5.2, 8.5, 11.8]
    eras = ["GPT-2 124M\n2019", "LLaMA1/2 7B\n2023", "Llama 3 8B\n2024-07", "Llama 3.2 1B\n2024-09"]
    vocabs = ["V = 50,257", "V = 32,000", "V = 128,256", "V = 128,256"]
    states = ["tied", "untied", "untied", "tied again"]
    shares = [r["share"] for r in rows]
    vocab_accts = [r["vocab_acct"] for r in rows]
    BAR0, SCALE = 2.15, 11.0               # 条基线与缩放：31.02% -> 3.41 高

    for x, era, vv, st, sh, vac in zip(xs, eras, vocabs, states, shares, vocab_accts):
        is_tied = "untied" not in st
        col = BLUE if is_tied else ORANGE
        # ---- 顶部：模型与词表 ----
        ax.text(x, 8.95, era.replace("\n", "  "), ha="center", fontsize=8.0, color=DARK, fontweight="bold")
        ax.text(x, 8.5, vv, ha="center", fontsize=7.4, color=MUTED)
        # ---- 开关图：E 与 H 两个矩阵框，tied=连线共享存储；untied=断开两份 ----
        ys = 6.9
        box(ax, x - 1.12, ys, 0.92, 0.92, "E\n(V,d)", fc="#f2f7fd", ec=BLUE, lw=1.2, fs=7.0)
        box(ax, x + 0.20, ys, 0.92, 0.92, "H\n(V,d)", fc="#f2f7fd", ec=BLUE, lw=1.2, fs=7.0)
        if is_tied:                        # 连接的链环：一条粗线把两框接通
            ax.plot([x - 0.20, x + 0.20], [ys + 0.46, ys + 0.46], color=col, lw=2.4)
            ax.plot([x, x], [ys + 0.20, ys + 0.72], color=col, lw=2.4)
        else:                              # 断开的链环：两段短线中间留缺口
            ax.plot([x - 0.20, x - 0.04], [ys + 0.46, ys + 0.46], color=col, lw=2.4)
            ax.plot([x + 0.04, x + 0.20], [ys + 0.46, ys + 0.46], color=col, lw=2.4)
        ax.text(x, ys - 0.42, st, ha="center", fontsize=7.8, color=col, fontweight="bold")
        # ---- 占比条（词表账占参数总量比）----
        bh = sh * SCALE
        ax.add_patch(FancyBboxPatch((x - 0.52, BAR0), 1.04, bh, boxstyle="round,pad=0.01",
                                    fc=col, ec=col, alpha=0.88))
        ax.text(x, BAR0 + bh + 0.30, f"{sh*100:.1f}%", ha="center", fontsize=9.4,
                color=DARK, fontweight="bold")
        ax.text(x, BAR0 + bh + 0.86, f"{vac/1e6:,.0f}M", ha="center", fontsize=7.2, color=MUTED)

    # 时间轴
    ax.annotate("", xy=(13.5, 1.35), xytext=(0.5, 1.35),
                arrowprops=dict(arrowstyle="-|>", color=TICK, lw=1.2))
    for x, era in zip(xs, eras):
        ax.plot([x], [1.35], marker="o", ms=4.5, color=TICK)
    ax.text(7.0, 0.55, "bar = vocab account (E + H) as share of total params, hand-computed from config; "
                       "E = embed_tokens, H = lm_head", ha="center", fontsize=7.2, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-5-1-tie-four-flips.png"), facecolor="white")
    plt.close(fig)
    print("[fig 5.1] tie 四拨：", [f"{s['share']*100:.2f}%" for s in rows])


# ---------------- 图 5.2：词表-嵌入占比曲线（自产：左=768 玩具骨架公式；右=512 模块档实测） ----------------
def fig_5_2():
    ci = latest_config_infer()
    toy = ci["step3_skeleton_vocab_table"]
    scan = load_json(os.path.join(CH05_LOG, "vocab_scan_fast.json"))
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300)

    # 左：固定骨架换词表（d=768/L=12/f=2048，公式账）
    ax = axes[0]
    Vs = [r["V"] for r in toy]
    tied = [r["embed_share"] * 100 for r in toy]
    untied = [2 * r["embed"] / (r["tied_total"] + r["embed"]) * 100 for r in toy]
    ax.plot(Vs, tied, "o-", color=BLUE, lw=2, ms=6, mec="white", mew=0.7, label="tied (1 copy)")
    ax.plot(Vs, untied, "s--", color=ORANGE, lw=2, ms=6, mec="white", mew=0.7, label="untied (2 copies)")
    ax.axhline(50, color=TICK, lw=0.9, ls=":")
    ax.text(8600, 51.5, "half the budget", fontsize=7.4, color=MUTED)
    for r, t in zip(toy, tied):
        ax.annotate(f"{t:.1f}%", (r["V"], t), textcoords="offset points", xytext=(-2, -13),
                    fontsize=7.2, color=BLUE, ha="center")
    ax.set_xscale("log")
    ax.set_xticks(Vs)
    ax.set_xticklabels([f"{v//1000}k" if v >= 1000 else str(v) for v in Vs])
    ax.set_xlabel("vocab size V (d=768, L=12 skeleton)", fontsize=9)
    ax.set_ylabel("embedding share of params (%)", fontsize=9)
    ax.set_ylim(0, 80)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.0, loc="upper left")
    ax.set_title("formula: same skeleton, swap V", fontsize=8.6, color=MUTED)

    # 右：本册模块档四臂实测（d=512/L=6，tied）
    ax = axes[1]
    tt = scan["tradeoff_table"]
    Vm = [r["V"] for r in tt]
    sm = [r["embedding_share"] * 100 for r in tt]
    ax.plot(Vm, sm, "o-", color=TEAL, lw=2, ms=6, mec="white", mew=0.7)
    ax.axhline(50, color=TICK, lw=0.9, ls=":")
    ax.text(8600, 51.5, "half the budget", fontsize=7.4, color=MUTED)
    for r, s in zip(tt, sm):
        ax.annotate(f"{s:.1f}%", (r["V"], s), textcoords="offset points", xytext=(0, -14),
                    fontsize=7.2, color=TEAL, ha="center")
    ax.set_xscale("log")
    ax.set_xticks(Vm)
    ax.set_xticklabels([f"{v//1000}k" for v in Vm])
    ax.set_xlabel("vocab size V (d=512, L=6 module tier, tied)", fontsize=9)
    ax.set_ylabel("embedding share of params (%)", fontsize=9)
    ax.set_ylim(0, 80)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title("measured: ch5 vocab scan, 1000 steps", fontsize=8.6, color=MUTED)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-5-2-vocab-embedding-share.png"), facecolor="white")
    plt.close(fig)
    print("[fig 5.2] 玩具骨架 tied:", [f"{t:.2f}" for t in tied], " 实测:", [f"{s:.2f}" for s in sm])


# ---------------- 图 5.3：中英文压缩对拍（自产，固定文本实测） ----------------
def fig_5_3():
    probe = load_json(PROBE_JSON)
    comp, ex = probe["compression"], probe["piece_examples"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300)

    # 左：pieces per hanzi（三段固定中文文本，越低越好）
    ax = axes[0]
    cats = ["news", "tech", "mixed"]
    labels = ["zh news (51 hz)", "zh tech (37 hz)", "zh mixed (30 hz)"]
    p32 = [comp[f"zh_{c}"]["llama2_pieces_per_hanzi"] for c in cats]
    p128 = [comp[f"zh_{c}"]["llama3_pieces_per_hanzi"] for c in cats]
    ratio = [comp[f"zh_{c}"]["ratio_32k_over_128k"] for c in cats]
    xpos = range(len(cats))
    w = 0.34
    ax.bar([x - w / 2 for x in xpos], p32, w, color=BLUE, label="LLaMA2 32k sp-BPE")
    ax.bar([x + w / 2 for x in xpos], p128, w, color=ORANGE, label="Llama3 128k tiktoken")
    ax.axhline(1.0, color=TICK, lw=1.0, ls="--")
    ax.text(2.42, 1.04, "1 piece / hanzi", fontsize=7.2, color=MUTED, ha="right")
    ax.axhline(2.17, color=MUTED, lw=0.9, ls=":")
    ax.text(2.42, 2.21, "GPT-2 ~2.0-2.3 (Book2 ch2)", fontsize=7.2, color=MUTED, ha="right")
    for i, (a, b, r) in enumerate(zip(p32, p128, ratio)):
        ax.text(i + w / 2, b + 0.05, f"{r:.2f}x", ha="center", fontsize=8.0, color=DARK, fontweight="bold")
    ax.set_xticks(list(xpos))
    ax.set_xticklabels(labels, fontsize=8.0)
    ax.set_ylabel("pieces per hanzi (lower = better)", fontsize=9)
    ax.set_ylim(0, 2.6)
    ax.grid(True, axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.0, loc="upper right")

    # 右：具体串的 piece 数（32k vs 128k；GPT-2 的两个 Book2 实测值以文字标注）
    ax = axes[1]
    probes = [("Beijing\n(2 hanzi)", ex["北京"]["n2"], ex["北京"]["n3"], 4),
              ("Beijing+Tsinghua\nUniv. (6 hanzi)", ex["北京清华大学"]["n2"], ex["北京清华大学"]["n3"], 12),
              ("' Transformer'\n+ jiagou (2 hz)", ex[" Transformer架构"]["n2"], ex[" Transformer架构"]["n3"], None),
              ("digit string\n(32 chars)", comp["digits"]["llama2_32k_pieces"],
               comp["digits"]["llama3_128k_pieces"], None)]
    xpos = range(len(probes))
    n2 = [p[1] for p in probes]
    n3 = [p[2] for p in probes]
    ax.bar([x - w / 2 for x in xpos], n2, w, color=BLUE, label="LLaMA2 32k")
    ax.bar([x + w / 2 for x in xpos], n3, w, color=ORANGE, label="Llama3 128k")
    for i, p in enumerate(probes):
        ax.text(i - w / 2, p[1] + 0.4, str(p[1]), ha="center", fontsize=7.6, color=BLUE)
        ax.text(i + w / 2, p[2] + 0.4, str(p[2]), ha="center", fontsize=7.6, color=ORANGE)
        if p[3] is not None:
            ax.text(i, max(p[1], p[2]) + 2.2, f"GPT-2: {p[3]}", ha="center", fontsize=7.2, color=MUTED)
    ax.set_xticks(list(xpos))
    ax.set_xticklabels([p[0] for p in probes], fontsize=7.4)
    ax.set_ylabel("pieces (fixed strings)", fontsize=9)
    ax.set_ylim(0, 40)
    ax.grid(True, axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.0, loc="upper right")
    ax.text(0.02, 0.965, "3rd probe: ' Transformer' + 2 hanzi -- the 32k vocab lacks one hanzi piece,\n"
                          "so it degrades to raw UTF-8 bytes (<0xE6><0x9E><0xB6>) = byte fallback in action",
            transform=ax.transAxes, fontsize=6.8, color=MUTED, va="top")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-5-3-compression-zh-en.png"), facecolor="white")
    plt.close(fig)
    print("[fig 5.3] pieces/hanzi 32k:", p32, "128k:", p128)


# ---------------- 图 5.4：config 反推五步流程（自绘示意级） ----------------
def fig_5_4():
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.4)
    ax.axis("off")
    ax.text(8.0, 9.0, "Reading an unfamiliar model: the five-step config autopsy",
            ha="center", fontsize=9.5, color=DARK, fontweight="bold")

    steps = [
        ("STEP 1\nidentify family",
         "model_type / architectures\nversion = era;\nabsent fields = defaults",
         "4.31 has no rope_theta\n-> theta = 10000 (MHA)"),
        ("STEP 2\nrebuild blueprint",
         "field -> component map:\nredraw it with shapes\nembed | L x block | head",
         "head_dim = d / h\ndoes it divide evenly?"),
        ("STEP 3\naudit the account",
         "N = Vd(2-tie)\n+L(2d^2+2d*h_kv*d_k\n+3df+2d)+d",
         "Llama3-8B: 8,030,261,248\nfour-way exact match"),
        ("STEP 4\nread peripherals",
         "tokenizer_config\ngeneration_config\ntokenizer file itself",
         "32k = 3+256+31,741\n128,256 = 128,000+256"),
        ("STEP 5\ngrade evidence",
         "config fact > paper table\n> mirror > remote header\n> community inference",
         "output = an architecture\ndiagram WITH evidence"),
    ]
    x0, y0, bw, bh, gap = 0.35, 3.4, 2.85, 4.6, 0.32
    colors = [BLUE, TEAL, ORANGE, YELLOW, BLUE]
    for i, (title, body, note) in enumerate(steps):
        x = x0 + i * (bw + gap)
        box(ax, x, y0, bw, bh, "", fc="white", ec=colors[i], lw=1.4)
        ax.text(x + bw / 2, y0 + bh - 0.62, title, ha="center", va="center", fontsize=8.2,
                color=colors[i], fontweight="bold")
        ax.text(x + bw / 2, y0 + bh - 2.05, body, ha="center", va="center", fontsize=6.5, color=DARK)
        ax.add_patch(FancyBboxPatch((x + 0.18, y0 + 0.35), bw - 0.36, 1.28,
                                    boxstyle="round,pad=0.02", fc="#f7f6f3", ec=GRID, lw=0.7))
        ax.text(x + bw / 2, y0 + 0.99, note, ha="center", va="center", fontsize=6.6, color=MUTED)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + bw + gap - 0.06, y0 + bh / 2), xytext=(x + bw + 0.06, y0 + bh / 2),
                        arrowprops=dict(arrowstyle="-|>", color=TICK, lw=1.3))

    ax.text(8.0, 2.55, "loop back: any mismatch in STEP 3 must be attributed before you trust the blueprint",
            ha="center", fontsize=7.6, color=MUTED)
    ax.add_patch(FancyBboxPatch((3.1, 0.55), 9.8, 1.35, boxstyle="round,pad=0.03",
                                fc="#fdf8f4", ec=ORANGE, lw=1.1))
    ax.text(8.0, 1.22, "worked example: Llama-2-7b total_size/2 exceeds the true architecture\n"
                       "by 4,096 phantom params = 32 inv_freq buffers [64] F32 = 8,192 bytes / 2",
            ha="center", va="center", fontsize=7.0, color=DARK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-5-4-config-autopsy-steps.png"), facecolor="white")
    plt.close(fig)
    print("[fig 5.4] 五步流程图完成")


if __name__ == "__main__":
    fig_5_1()
    fig_5_2()
    fig_5_3()
    fig_5_4()
