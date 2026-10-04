# make_figures.py —— 生成 Book4 第 12 章两张收束图：图 12.1 八册地图（Book4 高亮，F/N/B3 十钩流入、B4 新钩流出）
#                   与图 12.2 本册代码资产总览（Book3 底座→moe_mla_slots 对拍锚→两刀教学件→ch09 两刀整机→设计文档）。
#                   图内文字英文、图注中文在正文（写作规范 §4）；收束章无新实验，本脚本仅作图。
# 所属章节：Book4 ch12（12.3 图 12.1 / 12.5 图 12.2）。框内行数为渲染时实时读取（wc -l 口径），
#           重跑脚本即定版——正文的自指总数（件数/总行数/术语条目数）已按最终渲染计数填定（见 12.5）。
# 运行方式：cd 工作区根目录 && source env.sh && \
#           python code/ch12/make_figures.py [--only 1,2]
# 规格：蓝 #2a78d6 / 橙 #eb6834 / 青 #1baf7a / 黄 #eda100；线宽 2 级；白底；tight_layout；图式沿用 Book3 ch11。
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUE_F, ORNG_F, CYAN_F, YLW_F = "#cde2fb", "#fde8db", "#d8f3e8", "#fdf3d7"
INK, NOTE, HAIR = "#0b0b0b", "#52514e", "#898781"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))          # 工作区=model_analysis 根（快照内被组装规则改为册根）
FIG_DIR = os.path.join(ROOT, "figures")
B4 = os.path.abspath(os.path.join(HERE, ".."))                        # 一级上溯：工作区=本册 code/、快照=册内 code/（两布局同址）
# Book3 底座：双候选 bootstrap（工作区布局优先，书仓布局兜底——与正文跨册 import 契约同款）
B3_CANDS = [os.path.join(B4, "..", "code", "Book3-现代开源骨架"),     # 工作区：本册 code/ 的兄弟目录
            os.path.join(B4, "..", "..", "from-models-to-vllm", "book3-modern-open-source-skeleton", "code"),
            os.path.join(B4, "..", "..", "book3-modern-open-source-skeleton", "code")]   # 书仓：册根上一级=书仓根
B3 = next((p for p in B3_CANDS if os.path.isdir(p)), B3_CANDS[0])


def lc_b4(rel):
    """Book4 代码文件行数（wc -l 口径）；缺失返回 '—'（ch09/ch10 交付前占位，终校重跑补齐）。"""
    p = os.path.join(B4, rel)
    if os.path.isfile(p):
        with open(p, encoding="utf-8", errors="ignore") as f:
            return str(sum(1 for _ in f))
    return "—"


def lc_b3(rel):
    """Book3 遗产行数（双候选目录解析）。"""
    p = os.path.join(B3, rel)
    if os.path.isfile(p):
        with open(p, encoding="utf-8", errors="ignore") as f:
            return str(sum(1 for _ in f))
    return "—"


def box(ax, x, y, w, h, text, ec=BLUE, fc=BLUE_F, fs=6.4, lw=1.3, tc=INK, ls="-"):
    """画一个框内多行文字的资产框；x/y/w/h 均为 axes 坐标（0-1）。"""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.01",
                                ec=ec, fc=fc, lw=lw, ls=ls, mutation_scale=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.55)


def arrow(ax, p, q, color=HAIR, lw=1.3, rad=0.0, ls="-"):
    """从点 p 到点 q 画一条带箭头的连线（rad 控制弧度）。"""
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


# ---------------- 图 12.1：八册地图（Book4 高亮） ----------------
def fig_12_1():
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    books = [
        ("Book 1\nTransformer\nOriginal", "the 2017 machine"),
        ("Book 2\nPretraining\nRevolution", "data · objective · laws"),
        ("Book 3\nModern Open\nSkeletons", "slot-by-slot upgrades"),
        ("Book 4\nScale &\nEfficiency", "two cuts, one machine"),
        ("Book 5\nFrontier\nHybrids", "attention at its limits"),
        ("Book 6\nInference\nSystems", "serving physics"),
        ("Book 7\nvLLM V1\nSource I", "one request's life"),
        ("Book 8\nvLLM V1\nSource II", "topics & production"),
    ]
    xs = [0.085 + 0.118 * i for i in range(8)]
    BW, BH, BY = 0.104, 0.22, 0.74
    BOOK_BOT = BY - BH / 2                       # 0.629：书框底边
    for i, ((title, tag), x) in enumerate(zip(books, xs)):
        hot = (i == 3)                            # Book4 高亮
        box(ax, x, BY, BW, BH, title,
            ec=ORANGE if hot else HAIR, fc=ORNG_F if hot else "white",
            fs=6.3, lw=2.0 if hot else 1.1)
        ax.text(x, BOOK_BOT - 0.030, tag, ha="center", va="center",
                fontsize=5.4, color=NOTE, style="italic")
    for i in range(7):                            # 阅读路径箭头
        arrow(ax, (xs[i] + BW / 2, BY), (xs[i + 1] - BW / 2, BY), color=HAIR, lw=1.1)
    ax.annotate("you are here", xy=(xs[3], BY + BH / 2), xytext=(xs[3] - 0.115, BY + BH / 2 + 0.085),
                fontsize=6.2, color=ORANGE, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.3))

    # 旧钩回收（Book1/Book2/Book3 → Book4，蓝色虚线弧；弧深随距离加大，三层嵌套汇聚到 Book4 底边）
    arrow(ax, (xs[2], BOOK_BOT), (xs[3] - 0.012, BOOK_BOT - 0.020),
          color=BLUE, lw=1.5, ls="--", rad=0.14)
    ax.text((xs[2] + xs[3]) / 2 + 0.010, 0.575, "B3-2 5 8 9\nsettled here",
            ha="center", va="center", fontsize=5.6, color=BLUE, linespacing=1.4)
    arrow(ax, (xs[1], BOOK_BOT), (xs[3] - 0.006, BOOK_BOT - 0.062),
          color=BLUE, lw=1.5, ls="--", rad=0.24)
    ax.text((xs[1] + xs[2]) / 2 + 0.015, 0.445, "N5 settled here",
            ha="center", va="center", fontsize=5.6, color=BLUE, linespacing=1.4)
    arrow(ax, (xs[0], BOOK_BOT), (xs[3] - BW / 2 - 0.004, BOOK_BOT - 0.088),
          color=BLUE, lw=1.5, ls="--", rad=0.32)
    ax.text((xs[0] + xs[1]) / 2 + 0.012, 0.300, "F2 F13 F14 F16 F18\nsettled here",
            ha="center", va="center", fontsize=5.6, color=BLUE, linespacing=1.4)

    # B4 新钩埋设（Book4 → 后册）：橙色总线 + 四根上行支线（地铁图式）
    BUS_Y = 0.205
    ax.plot([xs[3], xs[7]], [BUS_Y, BUS_Y], color=ORANGE, lw=1.6)
    ax.plot([xs[3], xs[3]], [BOOK_BOT, BUS_Y], color=ORANGE, lw=1.6)
    ax.text(xs[3] + 0.012, 0.310, "planted\nhere", fontsize=5.6, color=ORANGE,
            ha="left", va="center", linespacing=1.4)
    for dst, label in [(4, "B4-2 5 6"), (5, "B4-3 4 7*"), (6, "B4-2"), (7, "B4-1 3")]:
        x = xs[dst]
        arrow(ax, (x, BUS_Y), (x, BOOK_BOT - 0.062), color=ORANGE, lw=1.4)
        ax.text(x, BUS_Y - 0.034, label, ha="center", va="center", fontsize=5.8, color=ORANGE)

    ax.text(0.5, 0.082,
            "F = Book1 hooks · N = Book2 hooks · B3 = Book3 hooks (ten settled in Book4);  "
            "B4 = new hooks planted by Book4 (Table 12.2)\n"
            "*B4-7 = in-book doc anchor (ch09/DESIGN-409M.md); its Book6 ledger half-address rides the Book6 riser",
            ha="center", va="center", fontsize=6.0, color=NOTE, style="italic", linespacing=1.6)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-12-1-eight-book-map.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}")


# ---------------- 图 12.2：本册代码资产总览（207M→两刀机 import 链） ----------------
def fig_12_2():
    fig, ax = plt.subplots(figsize=(9.6, 5.8), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 四段旅程表头（与导览 0.4 同口径）
    def stage(x, label, color):
        ax.text(x, 0.965, label, ha="center", va="center", fontsize=7.4, color=INK)
        ax.plot([x - 0.105, x + 0.105], [0.935, 0.935], color=color, lw=2.0)

    stage(0.095, "Why  ch01", HAIR)
    stage(0.335, "First cut  ch02-05", BLUE)
    stage(0.595, "2nd cut & parts  ch06-08", TEAL)
    stage(0.860, "Dissect & integrate  ch09-12", YELLOW)

    # ---- 第一列：为什么（ch01 账本；Book3 底座虚影=单轨主线的入口） ----
    box(ax, 0.095, 0.800, 0.160, 0.100,
        f"ch01/account.py  ({lc_b4('ch01/account.py')})\nthree-account ledger\nfour-model recount",
        ec=HAIR, fc="white", fs=5.9)
    box(ax, 0.095, 0.470, 0.160, 0.105,
        f"Book3 ch10/llama215.py  ({lc_b3('ch10/llama215.py')})\n207M dense base\n(single-track lineage\nenters here)",
        ec=HAIR, fc="white", fs=5.7, ls="--")
    box(ax, 0.095, 0.295, 0.160, 0.095,
        f"Book3 00-feas/llama_slots.py\n({lc_b3('00-feasibility/llama_slots.py')})\nRMSNorm · SwiGLU · RoPE · GQA",
        ec=TEAL, fc="white", fs=5.7, ls="--")

    # ---- 第二列：第一刀（ch02-05；橙框=被 ch09 import 的两刀正件之一） ----
    box(ax, 0.335, 0.830, 0.195, 0.085,
        f"ch02/moe.py  ({lc_b4('ch02/moe.py')})\nablation_moe.py  ({lc_b4('ch02/ablation_moe.py')})\n"
        "Mixtral-style part · dense-vs-MoE",
        ec=BLUE, fc=BLUE_F, fs=5.7)
    box(ax, 0.335, 0.715, 0.195, 0.085,
        f"ch03/ablation_balance.py  ({lc_b4('ch03/ablation_balance.py')})\nnone vs aux · load histograms",
        ec=BLUE, fc=BLUE_F, fs=5.7)
    box(ax, 0.335, 0.600, 0.195, 0.085,
        f"ch04/route_analysis.py  ({lc_b4('ch04/route_analysis.py')})\n"
        f"mixtral_infer.py  ({lc_b4('ch04/mixtral_infer.py')})\nrouting stats · config reverse",
        ec=BLUE, fc=BLUE_F, fs=5.7)
    box(ax, 0.335, 0.470, 0.195, 0.090,
        f"ch05/deepseek_moe.py  ({lc_b4('ch05/deepseek_moe.py')})\n"
        f"ablation_granularity.py  ({lc_b4('ch05/ablation_granularity.py')})\nfine+shared+noaux · 4-arm",
        ec=ORANGE, fc=ORNG_F, fs=5.7, lw=1.8)

    # ---- 第三列：第二刀与训练侧补件（ch06-08） ----
    box(ax, 0.595, 0.830, 0.200, 0.090,
        f"ch06/mla.py  ({lc_b4('ch06/mla.py')})\nkv_probe_mla.py  ({lc_b4('ch06/kv_probe_mla.py')})\n"
        "dual-path MLA · KV ledger",
        ec=ORANGE, fc=ORNG_F, fs=5.7, lw=1.8)
    box(ax, 0.595, 0.700, 0.200, 0.080,
        f"ch07/fp8_sim.py  ({lc_b4('ch07/fp8_sim.py')})\nE4M3/E5M2 · group scaling sim",
        ec=TEAL, fc=CYAN_F, fs=5.9)
    box(ax, 0.595, 0.580, 0.200, 0.080,
        f"ch08/mtp.py  ({lc_b4('ch08/mtp.py')})\nsmoke_mtp.py  ({lc_b4('ch08/smoke_mtp.py')})\n"
        "MTP module · off-by-one",
        ec=TEAL, fc=CYAN_F, fs=5.7)

    # ---- 第四列：拆机与整合（ch09-12） ----
    box(ax, 0.860, 0.830, 0.230, 0.100,
        f"ch09/llama409.py  ({lc_b4('ch09/llama409.py')})\nimports B3-02 rms · B3-04 rope\n"
        "+ ch05 MoE + ch06 MLA\nassert 409,115,648",
        ec=YELLOW, fc=YLW_F, fs=5.7, lw=2.0)
    box(ax, 0.860, 0.690, 0.230, 0.090,
        f"ch09/train_409.py  ({lc_b4('ch09/train_409.py')})\n"
        f"DESIGN-409M.md  ({lc_b4('ch09/DESIGN-409M.md')} lines)\nsmoke · integrated run · restart",
        ec=YELLOW, fc=YLW_F, fs=5.7)
    box(ax, 0.860, 0.560, 0.230, 0.080,
        f"ch10/gptoss_reverse.py  ({lc_b4('ch10/gptoss_reverse.py')})\n3-way assert · flagship digest",
        ec=TEAL, fc=CYAN_F, fs=5.9)
    box(ax, 0.860, 0.430, 0.230, 0.075,
        "ch11 (no code — ledgers in text)\nch12 (this figure script only)",
        ec=HAIR, fc="white", fs=5.9, ls="--")

    # 对拍锚：00-feasibility 组件库正身 + 聚合对拍（青虚框，各章教学件与其逐位一致 0.00e+00）
    box(ax, 0.200, 0.150, 0.165, 0.105,
        f"00-feas/moe_mla_slots.py\n({lc_b4('00-feasibility/moe_mla_slots.py')}) · "
        f"parity_all.py ({lc_b4('00-feasibility/parity_all.py')})\n+ 7 probes · component & parity anchor",
        ec=TEAL, fc="white", fs=5.5, ls="--")

    # ---- 依赖链：Book3 底座 → 对拍锚 → 第一刀各章教学件（青虚线） ----
    ax.plot([0.175, 0.215], [0.340, 0.203], color=TEAL, lw=1.2, ls="--")
    ax.plot([0.175, 0.215], [0.470, 0.215], color=TEAL, lw=1.2, ls="--")
    ax.plot([0.215, 0.215], [0.203, 0.830], color=TEAL, lw=1.2, ls="--")
    for yy in (0.830, 0.715, 0.600, 0.470):
        arrow(ax, (0.215, yy), (0.2375, yy), color=TEAL, lw=1.0, ls="--")
    # 对拍锚 → ch06 教学件（青虚肘形线：沿底部走第二、三列之间的空隙上行，与橙色总线仅一点相交）
    ax.plot([0.2825, 0.455], [0.150, 0.150], color=TEAL, lw=1.1, ls="--")
    ax.plot([0.455, 0.455], [0.150, 0.830], color=TEAL, lw=1.1, ls="--")
    arrow(ax, (0.455, 0.830), (0.495, 0.830), color=TEAL, lw=1.1, ls="--")

    # ---- 两刀组装总线（橙：ch05+ch06 正件 → ch09 整机；沿底部肘形总线走，竖线走列间空隙） ----
    ax.plot([0.4325, 0.470], [0.470, 0.470], color=ORANGE, lw=1.4)
    ax.plot([0.695, 0.700], [0.830, 0.830], color=ORANGE, lw=1.4)
    ax.plot([0.470, 0.470], [0.470, 0.065], color=ORANGE, lw=1.6)
    ax.plot([0.700, 0.700], [0.830, 0.065], color=ORANGE, lw=1.6)
    ax.plot([0.470, 0.985], [0.065, 0.065], color=ORANGE, lw=1.6)
    ax.plot([0.985, 0.985], [0.065, 0.830], color=ORANGE, lw=1.6)
    arrow(ax, (0.985, 0.830), (0.975, 0.830), color=ORANGE, lw=1.6)
    ax.text(0.560, 0.088, "single-track lineage: two-cut parts imported by ch09/llama409.py",
            fontsize=5.6, color=ORANGE, style="italic")

    # 整机 → 训练/设计文档 → 拆机（灰，章内供给）；设计文档 → Book6 账本算例（橙虚短线，B4-7 消费线）
    arrow(ax, (0.860, 0.780), (0.860, 0.735), color=HAIR, lw=1.1)
    arrow(ax, (0.860, 0.645), (0.860, 0.600), color=HAIR, lw=1.1)
    arrow(ax, (0.975, 0.690), (0.997, 0.690), color=ORANGE, lw=1.2, ls="--")
    ax.text(0.998, 0.662, "to Book6\nledger", fontsize=5.2, color=ORANGE, ha="right",
            va="top", linespacing=1.4, style="italic")

    # 底部汇总：渲染时实时计数（终校定版；正文自指总数按纪律以【终校终刷】占位）
    total_n, total_l = 0, 0
    for rel in ["ch01/account.py", "ch02/moe.py", "ch02/ablation_moe.py", "ch03/ablation_balance.py",
                "ch04/route_analysis.py", "ch04/mixtral_infer.py", "ch05/deepseek_moe.py",
                "ch05/ablation_granularity.py", "ch06/mla.py", "ch06/kv_probe_mla.py",
                "ch07/fp8_sim.py", "ch08/mtp.py", "ch08/smoke_mtp.py", "ch09/llama409.py",
                "ch09/train_409.py", "ch10/gptoss_reverse.py"]:
        n = lc_b4(rel)
        if n != "—":
            total_n += 1
            total_l += int(n)
    ax.text(0.5, 0.012,
            f"{total_n}/16 content files · {total_l} lines (live count at render; totals fixed at final "
            "proof; figure scripts not counted)",
            ha="center", va="center", fontsize=6.0, color=NOTE, style="italic")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-12-2-code-assets.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="1,2", help="只画指定图（如 --only 1）")
    which = set(int(x) for x in ap.parse_args().only.split(",") if x.strip())
    os.makedirs(FIG_DIR, exist_ok=True)
    if 1 in which:
        fig_12_1()
    if 2 in which:
        fig_12_2()


if __name__ == "__main__":
    main()
