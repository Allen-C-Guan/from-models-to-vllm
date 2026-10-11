# fig_ch02.py —— Book7 ch02 制图：图 2.1 仓库地形分层 / 图 2.2 一条请求的 17 站穿越（全书导航母图）
# 用途：ch02《仓库地形图与阅读路径》两图——图 2.1=vllm/v1 十五子包按「主线/Book8 地界/外围」三泳道分层
#       （数据=wc 实测 390 文件/169,658 行，主线占 80.6%）；图 2.2=生命周期 17 站文件链（notes/02 §2.1），
#       按进程三色（前端/EngineCore/worker）、章色带标注、离线链分岔——全书每章开篇「本章在哪」的母图。
# 所属章节：Book7 ch02（§2.1/§2.2）。
# 运行：cd <workspace> && source env.sh && python code/Book7-vLLM源码剖析上/ch02/fig_ch02.py
# 产物：drafts/Book7-vLLM源码剖析上/figures/fig-2-1-terrain.png / fig-2-2-lifecycle.png
# 图规：300dpi 白底；蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；图内英文图注中文；标注自绘+版本锚。
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "drafts", "Book7-vLLM源码剖析上", "figures"))
BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e1e0d9"


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=6.4, lw=1.2, tc=INK, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw, ls=ls,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc, zorder=4)


def arr(ax, x0, y0, x1, y1, color=INK, lw=1.1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", color=color, lw=lw,
                                 mutation_scale=10, zorder=2))


# ---------------- 图 2.1：地形三泳道 ----------------
def fig_terrain():
    fig, ax = plt.subplots(figsize=(12.4, 6.4), dpi=300)
    ax.set_xlim(0, 17); ax.set_ylim(0, 9.8); ax.axis("off")
    ax.text(8.5, 9.45, "Terrain of vllm/v1 (v0.30.0, wc-measured: 390 .py / 169,658 lines): mainline 80.6% | Book-8 territory 15.7% | periphery 3.7%",
            ha="center", fontsize=9.0, color=INK, weight="bold")
    lanes = [
        ("MAINLINE (ch3-ch11)", 6.9, BLUE, "#dcebfc", [
            ("engine/ 15f\n11.1k lines\nch3/4/11", 2.6), ("core/ 15f\n13.3k\nch5/6", 2.6),
            ("executor/ 9f\n3.9k\nch7", 2.2), ("worker/ 115f\n41.9k\nch7/8", 2.9),
            ("attention/ 97f\n56.3k\nch9", 2.9), ("sample/ 15f\n5.6k\nch11", 2.2),
            ("top-level 8 mods\n4.5k\nch5/6/8", 2.4)]),
        ("BOOK-8 TERRITORY (grayed)", 4.15, MUTED, "#f0efec", [
            ("spec_decode/ 19f\n6.3k", 2.5), ("structured_output/ 8f\n2.6k", 2.9),
            ("kv_offload/ 48f\n12.4k", 2.5), ("simple_kv_offload+hisparse 14f\n5.3k", 3.7)]),
        ("PERIPHERY", 1.4, YELLOW, "#fdf0d2", [
            ("metrics/ 9f\n4.6k", 2.5), ("watermarking/ 10f\n1.1k", 2.6),
            ("pool/ 4f 0.5k", 2.0), ("fault_tolerance/ 3f\n0.2k", 2.5)]),
    ]
    for title, y, ec, fc, pkgs in lanes:
        ax.add_patch(Rectangle((0.5, y - 0.25), 12.6, 2.25, fc="white", ec=GRIDC, lw=0.8, zorder=1))
        ax.text(0.62, y + 1.78, title, fontsize=7.6, color=ec if ec != MUTED else MUTED, weight="bold")
        x = 0.7
        for label, w in pkgs:
            box(ax, x, y, w, 1.35, label, ec=ec, fc=fc, fs=6.2)
            x += w + 0.12
    # 占比堆叠条
    y0 = 0.25
    ax.add_patch(Rectangle((0.5, y0), 12.6 * 0.806, 0.8, fc=BLUE, ec="none", zorder=2))
    ax.add_patch(Rectangle((0.5 + 12.6 * 0.806, y0), 12.6 * 0.157, 0.8, fc=MUTED, ec="none", zorder=2))
    ax.add_patch(Rectangle((0.5 + 12.6 * 0.963, y0), 12.6 * 0.037, 0.8, fc=YELLOW, ec="none", zorder=2))
    ax.text(0.5 + 12.6 * 0.4, y0 + 0.4, "mainline 80.6%", ha="center", va="center", fontsize=7.0, color="white", weight="bold", zorder=3)
    ax.text(0.5 + 12.6 * 0.885, y0 + 0.4, "15.7%", ha="center", va="center", fontsize=6.6, color="white", zorder=3)
    ax.text(0.5 + 12.6 * 0.982, y0 + 0.4, "3.7%", ha="center", va="center", fontsize=6.0, color=INK, zorder=3)
    # 右侧 vllm-ascend 姊妹城
    box(ax, 13.4, 3.2, 3.3, 5.6,
        "vllm_ascend/\n587 .py\n170,346 lines\n\n≈ same size as\nvllm/v1 itself\n\nops/ 25.3k\ndistributed/ 27.3k\nattention/ 14.6k\nworker/ 15.4k\npatch/ 10.1k\n(60 files!)",
        ec=TEAL, fc="#d9f2e7", fs=6.4)
    ax.text(8, 0.06, "whole vllm/ package: 2,479 .py / 972,641 lines — v1 is 17.4% of it.  Sketched from v0.30.0 (ced6857), wc reproducible.",
            ha="center", fontsize=6.6, color=MUTED, style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-2-1-terrain.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print("[图 2.1] terrain")


# ---------------- 图 2.2：17 站穿越 ----------------
def fig_lifecycle():
    fig, ax = plt.subplots(figsize=(15.2, 6.8), dpi=300)
    ax.set_xlim(0, 34); ax.set_ylim(0, 12.4); ax.axis("off")
    ax.text(17, 12.0, "Life of a request: 17 stations across three processes (vLLM V1, sketched from v0.30.0) — this is the map every chapter opens on",
            ha="center", fontsize=9.4, color=INK, weight="bold")
    ST = [
        ("1 CLI\nmain.py", BLUE), ("2 API server\nlaunchers/", BLUE), ("3 chat route\nserving.py #L118", BLUE),
        ("4 AsyncLLM\ngenerate #L638", BLUE), ("5 InputProc\n#L38", BLUE), ("6 ZMQ client\nAsyncMPClient", BLUE),
        ("7 EngineCore\nbusy loop #L1283", ORANGE), ("8 Scheduler\nschedule #L561", ORANGE),
        ("9 KVCacheMgr\nallocate_slots #L370", ORANGE), ("10 executor\nbroadcast", ORANGE),
        ("11 worker\nNPUWorker", TEAL), ("12 ModelRunner\nexecute #L4187", TEAL),
        ("13 Sampler\nsample #L244", TEAL), ("14 step packs\noutputs", ORANGE),
        ("15 OutputProc\n#L629", BLUE), ("16 Detokenizer\n#L31", BLUE), ("17 stream out\nSSE", BLUE),
    ]
    x0, y0, w, h, gap = 0.6, 6.4, 1.72, 1.5, 0.24
    for i, (label, color) in enumerate(ST):
        x = x0 + i * (w + gap)
        fc = {"#2a78d6": "#dcebfc", "#eb6834": "#fde3d6", "#1baf7a": "#d9f2e7"}[color]
        box(ax, x, y0, w, h, label, ec=color, fc=fc, fs=5.9)
        if i < len(ST) - 1:
            arr(ax, x + w, y0 + h / 2, x + w + gap, y0 + h / 2, color=MUTED, lw=1.0)
    # 离线链分岔
    box(ax, 3.2, 9.0, 4.6, 1.0, "offline fork: LLM.generate #L418 -> LLMEngine -> InprocClient (no ZMQ) joins at st.7",
        ec=MUTED, fc="white", fs=6.2, ls="--")
    arr(ax, 5.5, 9.0, 11.6, 7.95, color=MUTED, lw=1.0)
    # 图例
    ax.text(26.2, 9.35, "process swimlane:", fontsize=6.8, color=INK)
    box(ax, 29.0, 9.15, 0.9, 0.45, "FE", ec=BLUE, fc="#dcebfc", fs=5.8)
    box(ax, 30.1, 9.15, 0.9, 0.45, "Core", ec=ORANGE, fc="#fde3d6", fs=5.8)
    box(ax, 31.2, 9.15, 0.9, 0.45, "worker", ec=TEAL, fc="#d9f2e7", fs=5.8)
    # 章色带
    bands = [(0, 6, "ch3 entries", BLUE), (5, 8, "ch4 EngineCore", ORANGE), (7, 9, "ch5 sched", ORANGE),
             (8, 9, "ch6 KV", TEAL), (9, 12, "ch7 exec", TEAL), (13, 14, "ch4 back", ORANGE),
             (12, 17, "ch8/9/11 backends+outputs", BLUE)]
    yb = 5.3
    for a, b, label, color in bands:
        x1 = x0 + a * (w + gap)
        x2 = x0 + (b - 1) * (w + gap) + w
        ax.plot([x1, x2], [yb, yb], color=color, lw=3.5, solid_capstyle="butt", zorder=2)
        ax.text((x1 + x2) / 2, yb - 0.42, label, ha="center", fontsize=5.8, color=color)
        yb -= 0.95
    ax.text(0.6, 0.9, "station 10 is also the startup-phase door: ch10 (weights & registration) lives at model load time, before any request walks this chain;\n"
            "ch12 (methodology) is the meta-layer that instruments every station.",
            fontsize=6.6, color=MUTED)
    ax.text(0.6, 0.3, "anchors cited per station are v0.30.0 (ced6857); the runtime stack observed on NPU is 0.27.1 + vllm-ascend (differences boxed in ch4/7).",
            fontsize=6.2, color=MUTED, style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-2-2-lifecycle.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print("[图 2.2] lifecycle")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_terrain()
    fig_lifecycle()
