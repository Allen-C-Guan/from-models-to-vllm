# 用途：生成 Book5 第 12 章两张收束图——图 12.1 八册地图（Book5 高亮：F/N/B3/B4 十一钩+次级六钩流入、B5-1~8 新钩流出）
#                   与图 12.2 本册代码资产总览（import 血统图：Book3 四插槽底座→Book4 两刀件→本册 K3 玩具/三刀机两台整机）。
#                   图内文字英文、图注中文在正文（写作规范 §4）；收束章无新实验，本脚本仅作图。
# 所属章节：Book5 ch12（12.3 图 12.1 / 12.5 图 12.2）。框内行数为渲染时实时读取（wc -l 口径），
#           重跑脚本即定版——正文自指总数（件数/行数）按渲染计数填定（见 12.5；批三若有增删以终校终刷为准）。
# 运行方式：cd 工作区根目录 && source env.sh && \
#           python "code/ch12/make_figures.py" [--only 1,2]
# 图规：写作规范 §4（300dpi / 固定分类色蓝橙青黄 / 图内英文图注中文）；输出文件名带本章专属 slug
#       （fig-12-1-b5- / fig-12-2-b5-，Book2/Book4 期 fig-12-2 跨册冲撞教训）；八册地图图式沿用 Book1-4（Book4 ch12 同款）。
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
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))          # 工作区=model_analysis 根（快照内被组装规则改为册根）
FIG_DIR = os.path.join(ROOT, "book5-frontier-hybrid-architectures", "figures")
B5 = os.path.abspath(os.path.join(HERE, ".."))                        # 一级上溯：工作区=本册 code/、快照=册内 code/（两布局同址）
# 跨册底座：三候选 bootstrap（工作区 / 书仓嵌套 / 书仓裸布局——与正文跨册 import 契约同款，llama409 _B3_CANDS 已验证模式）
_LEG_CANDS = [os.path.join(B5, "..", "code"),                          # 工作区：本册 code/ 的兄弟目录（Book3-/Book4- 在此）
              os.path.join(B5, "..", "..", "from-models-to-vllm"),
              os.path.join(B5, "..", "..")]
LEGACY = next((p for p in _LEG_CANDS if os.path.isdir(os.path.join(p, "Book3-现代开源骨架"))), _LEG_CANDS[0])


def _lc(path):
    """代码文件行数（wc -l 口径）；缺失返回 '—'（批三交付前占位，终校重跑补齐）。"""
    if os.path.isfile(path):
        with open(path, encoding="utf-8", errors="ignore") as f:
            return str(sum(1 for _ in f))
    return "—"


def lc_b5(rel):
    return _lc(os.path.join(B5, rel))


def lc_b3(rel):
    return _lc(os.path.join(LEGACY, "Book3-现代开源骨架", rel))


def lc_b4(rel):
    return _lc(os.path.join(LEGACY, "Book4-规模与效率", rel))


def box(ax, x, y, w, h, text, ec=BLUE, fc=BLUE_F, fs=6.4, lw=1.3, tc=INK, ls="-"):
    """画一个框内多行文字的资产框；x/y/w/h 均为 axes 坐标（0-1）。"""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.01",
                                ec=ec, fc=fc, lw=lw, ls=ls, mutation_scale=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.55)


def arrow(ax, p, q, color=HAIR, lw=1.3, rad=0.0, ls="-"):
    """从点 p 到 q 画一条带箭头的连线（rad 控制弧度）。"""
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


# ---------------- 图 12.1：八册地图（Book5 高亮） ----------------
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
        ("Book 5\nFrontier\nHybrids", "attention, three ways out"),
        ("Book 6\nInference\nSystems", "serving physics"),
        ("Book 7\nvLLM V1\nSource I", "one request's life"),
        ("Book 8\nvLLM V1\nSource II", "topics & production"),
    ]
    xs = [0.085 + 0.118 * i for i in range(8)]
    BW, BH, BY = 0.104, 0.22, 0.72
    BOOK_BOT = BY - BH / 2                       # 0.61：书框底边
    for i, ((title, tag), x) in enumerate(zip(books, xs)):
        hot = (i == 4)                            # Book5 高亮
        box(ax, x, BY, BW, BH, title,
            ec=ORANGE if hot else HAIR, fc=ORNG_F if hot else "white",
            fs=6.3, lw=2.0 if hot else 1.1)
        ax.text(x, BOOK_BOT - 0.030, tag, ha="center", va="center",
                fontsize=5.4, color=NOTE, style="italic")
    for i in range(7):                            # 阅读路径箭头
        arrow(ax, (xs[i] + BW / 2, BY), (xs[i + 1] - BW / 2, BY), color=HAIR, lw=1.1)
    ax.annotate("you are here", xy=(xs[4], BY + BH / 2), xytext=(xs[4] - 0.115, BY + BH / 2 + 0.085),
                fontsize=6.2, color=ORANGE, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.3))

    # 旧钩回收（Book1/Book2/Book3/Book4 → Book5，蓝色虚线弧；弧深随距离加大，四层嵌套汇聚到 Book5 底边）
    arrow(ax, (xs[3], BOOK_BOT), (xs[4] - 0.012, BOOK_BOT - 0.018),
          color=BLUE, lw=1.5, ls="--", rad=0.12)
    ax.text((xs[3] + xs[4]) / 2 + 0.012, 0.545, "B4-2 3 5 6\nsettled here",
            ha="center", va="center", fontsize=5.6, color=BLUE, linespacing=1.4)
    arrow(ax, (xs[2], BOOK_BOT), (xs[4] - 0.006, BOOK_BOT - 0.058),
          color=BLUE, lw=1.5, ls="--", rad=0.20)
    ax.text((xs[2] + xs[3]) / 2 + 0.016, 0.445, "B3-3 4 5 7\nsettled here",
            ha="center", va="center", fontsize=5.6, color=BLUE, linespacing=1.4)
    arrow(ax, (xs[1], BOOK_BOT), (xs[4] - BW / 2 - 0.002, BOOK_BOT - 0.088),
          color=BLUE, lw=1.5, ls="--", rad=0.26)
    ax.text((xs[1] + xs[2]) / 2 + 0.020, 0.325, "N6 N7 settled here",
            ha="center", va="center", fontsize=5.6, color=BLUE, linespacing=1.4)
    arrow(ax, (xs[0], BOOK_BOT), (xs[4] - BW / 2 - 0.004, BOOK_BOT - 0.118),
          color=BLUE, lw=1.5, ls="--", rad=0.32)
    ax.text((xs[0] + xs[1]) / 2 + 0.026, 0.215, "F1 F6 F9 F10\nF11 F12 F13\nsettled here",
            ha="center", va="center", fontsize=5.6, color=BLUE, linespacing=1.4)

    # B5 新钩埋设（Book5 → 后册）：橙色总线 + 三根上行支线（地铁图式）
    BUS_Y = 0.185
    ax.plot([xs[4], xs[7]], [BUS_Y, BUS_Y], color=ORANGE, lw=1.6)
    ax.plot([xs[4], xs[4]], [BOOK_BOT, BUS_Y], color=ORANGE, lw=1.6)
    ax.text(xs[4] + 0.012, 0.290, "planted\nhere", fontsize=5.6, color=ORANGE,
            ha="left", va="center", linespacing=1.4)
    for dst, label in [(5, "B5-1 2"), (6, "B5-5 · B5-8*"), (7, "B5-3 4 6 7 · B5-8*")]:
        x = xs[dst]
        arrow(ax, (x, BUS_Y), (x, BOOK_BOT - 0.062), color=ORANGE, lw=1.4)
        ax.text(x, BUS_Y - 0.036, label, ha="center", va="center", fontsize=5.8, color=ORANGE)

    ax.text(0.5, 0.075,
            "F = Book1 hooks · N = Book2 hooks · B3/B4 = Book3/4 hooks (eleven primary + six secondary settled in Book5, Table 0.2);  "
            "B5 = new hooks planted by Book5 (Table 12.1)\n"
            "*B5-8 = in-book doc anchor (ch09/DESIGN-3TO1.md, not in prose); B5-7 = weak candidate embedded in ch11",
            ha="center", va="center", fontsize=6.0, color=NOTE, style="italic", linespacing=1.6)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-12-1-b5-eight-book-map.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}")


# ---------------- 图 12.2：本册代码资产总览（import 血统图） ----------------
def fig_12_2():
    fig, ax = plt.subplots(figsize=(9.6, 6.0), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 五段旅程表头（与导览 0.4 同口径）
    def stage(x, label, color):
        ax.text(x, 0.968, label, ha="center", va="center", fontsize=7.2, color=INK)
        ax.plot([x - 0.090, x + 0.090], [0.940, 0.940], color=color, lw=2.0)

    stage(0.095, "Why  ch01", HAIR)
    stage(0.300, "Three routes  ch02-05", BLUE)
    stage(0.520, "K3 close-read  ch06-07", TEAL)
    stage(0.740, "Synthesis  ch08-09", YELLOW)
    stage(0.925, "Beyond arch  ch10-11", ORANGE)

    # ---- 第一列：为什么（ch01 新账立式件） ----
    box(ax, 0.095, 0.800, 0.165, 0.100,
        f"ch01/context_account.py  ({lc_b5('ch01/context_account.py')})\nscore-matrix account\n+ typed KV/state ledger\nfive-model speed table",
        ec=HAIR, fc="white", fs=5.6)

    # ---- 第二列：三条路线（ch02-05 教学件；橙框=被 ch09 整机 import 的 gdn 正件） ----
    box(ax, 0.300, 0.845, 0.190, 0.085,
        f"ch02/swa_sink.py  ({lc_b5('ch02/swa_sink.py')})\nneedle_probe.py  ({lc_b5('ch02/needle_probe.py')})\n"
        f"gptoss_sinks_dump.py  ({lc_b5('ch02/gptoss_sinks_dump.py')})\nbanded mask + learnable sink",
        ec=BLUE, fc=BLUE_F, fs=5.5)
    box(ax, 0.300, 0.730, 0.190, 0.075,
        f"ch03/nsa_block.py  ({lc_b5('ch03/nsa_block.py')})\nblock top-k, gap ladder, Eq.7 ledger",
        ec=BLUE, fc=BLUE_F, fs=5.6)
    box(ax, 0.300, 0.635, 0.190, 0.075,
        f"ch04/dsv4_glm5_reverse.py  ({lc_b5('ch04/dsv4_glm5_reverse.py')})\nconfig reverse + 1M ledger + parity",
        ec=BLUE, fc=BLUE_F, fs=5.6)
    box(ax, 0.300, 0.520, 0.190, 0.090,
        f"ch05/gdn.py  ({lc_b5('ch05/gdn.py')})\nqwen3next_mini.py  ({lc_b5('ch05/qwen3next_mini.py')})\n"
        "GDN dual-form parity + scaling\n+ MiniQwen3Next strict port",
        ec=ORANGE, fc=ORNG_F, fs=5.5, lw=1.8)

    # ---- 第三列：K3 精读（ch06-07；橙框=进 K3 玩具的正件） ----
    box(ax, 0.520, 0.845, 0.195, 0.088,
        f"ch06/kda.py  ({lc_b5('ch06/kda.py')})\nk3_toy.py  ({lc_b5('ch06/k3_toy.py')}) · "
        f"train_k3toy.py  ({lc_b5('ch06/train_k3toy.py')})\nKDA part + 43.9M toy, lineage in imports",
        ec=ORANGE, fc=ORNG_F, fs=5.5, lw=1.8)
    box(ax, 0.520, 0.720, 0.195, 0.082,
        f"ch07/latentmoe_parts.py  ({lc_b5('ch07/latentmoe_parts.py')})\nqat_sim.py  ({lc_b5('ch07/qat_sim.py')})\n"
        "three Rx + QB + MXFP4 STE",
        ec=TEAL, fc=CYAN_F, fs=5.5)

    # ---- 第四列：收拢（ch08 无代码 + ch09 三刀机四件套；黄框=单轨整合交付） ----
    box(ax, 0.740, 0.845, 0.185, 0.070,
        "ch08 (no code — ledger reuse\nfrom ch01, parameterized)",
        ec=HAIR, fc="white", fs=5.7, ls="--")
    box(ax, 0.740, 0.690, 0.200, 0.150,
        f"ch09/hybrid409.py  ({lc_b5('ch09/hybrid409.py')})\ntrain_hybrid.py  ({lc_b5('ch09/train_hybrid.py')})\n"
        f"hybrid_recall.py  ({lc_b5('ch09/hybrid_recall.py')})\nqwen35_consume.py  ({lc_b5('ch09/qwen35_consume.py')})\n"
        f"DESIGN-3TO1.md  ({lc_b5('ch09/DESIGN-3TO1.md')} lines)",
        ec=YELLOW, fc=YLW_F, fs=5.5, lw=2.0)

    # ---- 第五列：能力与方法论（ch10-11） ----
    box(ax, 0.925, 0.845, 0.140, 0.085,
        f"ch10/llava_mini.py\n({lc_b5('ch10/llava_mini.py')})\nfrozen twin towers\n+ one projector",
        ec=ORANGE, fc=ORNG_F, fs=5.6)
    box(ax, 0.925, 0.720, 0.140, 0.085,
        f"ch11/params_audit.py\n({lc_b5('ch11/params_audit.py')})\nthree-scope params\n+ self-report recount",
        ec=TEAL, fc=CYAN_F, fs=5.6)
    box(ax, 0.925, 0.590, 0.140, 0.070,
        "ch12 (this figure script only)",
        ec=HAIR, fc="white", fs=5.7, ls="--")

    # ---- 底部血统层：Book3 四插槽底座 → Book4 两刀件与整机（青虚框=跨册遗产，实时读行数） ----
    box(ax, 0.150, 0.300, 0.215, 0.115,
        f"Book3 00-feas/llama_slots.py  ({lc_b3('00-feasibility/llama_slots.py')})\n"
        f"ch10/llama215.py  ({lc_b3('ch10/llama215.py')})\nfour-slot base · RMSNorm/SwiGLU/RoPE\n207,119,360 asserted",
        ec=TEAL, fc="white", fs=5.5, ls="--")
    box(ax, 0.435, 0.300, 0.240, 0.115,
        f"Book4 moe_mla_slots.py  ({lc_b4('00-feasibility/moe_mla_slots.py')})\n"
        f"ch05/deepseek_moe.py  ({lc_b4('ch05/deepseek_moe.py')}) · ch06/mla.py  ({lc_b4('ch06/mla.py')})\n"
        f"ch09/llama409.py  ({lc_b4('ch09/llama409.py')})\ntwo-cut machine 409,115,648 asserted",
        ec=TEAL, fc="white", fs=5.5, ls="--")

    # Book3 → Book4（演化线上一段）
    arrow(ax, (0.258, 0.300), (0.314, 0.300), color=TEAL, lw=1.3, ls="--")

    # ---- 单轨组装总线（橙，地铁图式）：三股下行（Book3 底座/Book4 两刀件/本册 gdn 件）→ 一条总线 → 两根上行
    #      （→ ch06 K3 玩具 43.9M；→ ch09 三刀机 409M）——「丛书的代码史写在 import 语句里」
    BUS_Y = 0.155
    ax.plot([0.150, 0.740], [BUS_Y, BUS_Y], color=ORANGE, lw=1.6)
    ax.plot([0.150, 0.150], [0.2425, BUS_Y], color=ORANGE, lw=1.4)          # Book3 底座下行
    ax.plot([0.435, 0.435], [0.2425, BUS_Y], color=ORANGE, lw=1.4)          # Book4 两刀件下行
    ax.plot([0.285, 0.285], [0.475, BUS_Y], color=ORANGE, lw=1.4)           # 本册 ch05/gdn 件下行（自 ch05 框底）
    ax.plot([0.629, 0.629], [BUS_Y, 0.870], color=ORANGE, lw=1.5)           # 上行 A：→ K3 玩具（ch06 框右缘）
    arrow(ax, (0.629, 0.870), (0.618, 0.865), color=ORANGE, lw=1.5)
    ax.plot([0.740, 0.740], [BUS_Y, 0.615], color=ORANGE, lw=1.5)           # 上行 B：→ 三刀机（ch09 框底）
    arrow(ax, (0.740, 0.615), (0.740, 0.628), color=ORANGE, lw=1.5)
    ax.text(0.400, 0.200, "single-track assembly: imports, never copies\nk3_toy 43,905,888 · hybrid409 409,000,736",
            fontsize=5.6, color=ORANGE, style="italic", ha="center", va="center", linespacing=1.5)

    # 底部汇总：渲染时实时计数（终校定版）
    files = ["ch01/context_account.py", "ch02/swa_sink.py", "ch02/needle_probe.py",
             "ch02/gptoss_sinks_dump.py", "ch03/nsa_block.py", "ch04/dsv4_glm5_reverse.py",
             "ch05/gdn.py", "ch05/qwen3next_mini.py", "ch06/kda.py", "ch06/k3_toy.py",
             "ch06/train_k3toy.py", "ch07/latentmoe_parts.py", "ch07/qat_sim.py",
             "ch09/hybrid409.py", "ch09/train_hybrid.py", "ch09/hybrid_recall.py",
             "ch09/qwen35_consume.py", "ch10/llava_mini.py", "ch11/params_audit.py"]
    total_n, total_l = 0, 0
    for rel in files:
        n = lc_b5(rel)
        if n != "—":
            total_n += 1
            total_l += int(n)
    ax.text(0.5, 0.030,
            f"{total_n}/{len(files)} content .py files · {total_l:,} lines (live count at render; "
            "figure scripts & 00-feasibility probes not counted)\n"
            "every teaching part parity-checked vs Book4 moe_mla_slots anchor + HF kernels (CPU fp32, 1e-8 级); "
            "Book5 00-feasibility = 8 design probes",
            ha="center", va="center", fontsize=6.0, color=NOTE, style="italic", linespacing=1.6)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-12-2-b5-import-lineage.png")
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
