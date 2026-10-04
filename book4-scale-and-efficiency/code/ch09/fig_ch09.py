# fig_ch09.py —— Book4 ch9 三图复绘：图 9.1 DSV2/V3 整机对照 / 图 9.2 两刀机插槽来源 / 图 9.3 冒烟+短训曲线
# 用途：正文第 9 章三张图的绘制正身——
#   图 9.1：DSV2-236B 与 DSV3-671B 整机结构对照（自绘示意级，框线图；蓝=两代共有骨架、橙=V3 改动件，
#           报告结构依据 DSV2 §2.2-§2.3/DSV3 §2.1-§2.2 + 两家官方 config，逐字段演进账见正文表 9.1）；
#   图 9.2：409M 两刀机结构与插槽来源（自绘示意级；四插槽标注供货章——「两刀改装史写在 import 语句里」，
#           底部虚影=退役的 dense FFN 与 GQA）；
#   图 9.3：冒烟（双臂前 20 步 train loss）与整合短训（cand2 540 步 / cand3 512 步 eval 曲线，
#           与 Book3 207M 稠密同协议曲线对照）——数据源为 log/book4-ch09/*.csv 与 log/book3-ch10 曲线。
# 所属章节：Book4 第 9 章（图 9.1/9.2/9.3）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch09/fig_ch09.py"
#   （秒级；曲线图需先跑 train_409.py 冒烟与整合短训——缺失的曲线自动降级跳过并打印提示）
# 产物：figures/fig-9-{1,2,3}-*.png（dpi=300，白底）
# 图规：本册统一印刷规格——分类色固定序 蓝 #2a78d6/橙 #eb6834/青 #1baf7a/黄 #eda100；线宽 2、
#   标记 ≥6pt；网格 hairline #e1e0d9 置底层；刻度文字 #898781；图内文字英文、图注中文；禁双 y 轴。
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "figures")
LOG9 = os.path.join(REPO_ROOT, "log", "book4-ch09")
LOG3 = os.path.join(REPO_ROOT, "log", "book3-ch10")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, INK, INK2 = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": TICK, "axes.labelcolor": INK2,
                     "xtick.color": TICK, "ytick.color": TICK, "figure.dpi": 300})


def box(ax, x, y, w, h, text, fc, ec=None, fs=8.0, tc="white", lw=1.1, ls="-", bold=False):
    """圆角框 + 居中文字（框线示意级）。"""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.012",
                                fc=fc, ec=ec or fc, lw=lw, ls=ls, mutation_aspect=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            linespacing=1.35, fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, color=INK2, ls="-", lw=1.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=9))


# ---------------- 图 9.1：DSV2 / DSV3 整机对照（自绘示意级） ----------------
def fig_9_1():
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    def machine(x0, title, sub, rows, footer):
        ax.text(x0 + 2.0, 7.42, title, ha="center", fontsize=10, fontweight="bold", color=INK)
        ax.text(x0 + 2.0, 7.14, sub, ha="center", fontsize=7.0, color=INK2)
        y = 6.92
        for kind, lines in rows:                 # kind: emb/dense/attn/moe/head/new
            color = {"emb": BLUE, "dense": "#9db9dd", "attn": BLUE, "moe": BLUE,
                     "head": BLUE, "new": ORANGE}[kind]
            h = 0.72 if "\n" in lines else 0.42
            box(ax, x0, y - h, 4.0, h, lines, color, fs=7.1)
            arrow(ax, x0 + 2.0, y - h - 0.06, x0 + 2.0, y - h - 0.20)
            y -= h + 0.26
        ax.text(x0 + 2.0, y - 0.10, footer, ha="center", va="top", fontsize=6.7,
                color=INK2, linespacing=1.5)

    machine(0.20, "DeepSeek-V2 (236B)",
            "d=5120, L=60, h=128 | V=102,400 | total 235.74B, act ~21B",
            [("emb", "Embedding  (V x 5120)"),
             ("dense", "layer 0: dense FFN 12288   (first_k_dense = 1)"),
             ("attn", "MLA x59   cache c_KV (B,n,512) + k_R (B,n,64)\nd_c=512, d_c'=1536, nope/rope = 128/64"),
             ("moe", "MoE x59   160 routed top-6 + 2 shared, w=1536\nrouter: softmax, no re-norm, rsf x16.0"),
             ("head", "RMSNorm -> lm_head (untied)"),
             ("new", "balancing: 3-level aux loss (0.003/0.05/0.02)\n+ token dropping (cf=1.0) | BF16 training")],
            "KV/token = 60 x (512+64) = 34,560 elements | ctx 163,840\n(YaRN s=40 applied on decoupled k_R)")

    machine(5.65, "DeepSeek-V3 (671B)",
            "d=7168, L=61, h=128 | V=129,280 | total 671.03B, act 36.6B",
            [("emb", "Embedding  (V x 7168)"),
             ("dense", "layers 0-2: dense FFN 18432   (first_k_dense = 3)"),
             ("attn", "MLA x58   geometry unchanged vs V2\nd_c=512, d_c'=1536, nope/rope = 128/64"),
             ("moe", "MoE x58   256 routed top-8 + 1 shared, w=2048\nrouter: sigmoid, re-norm, rsf x2.5, groups 8>4"),
             ("head", "RMSNorm -> lm_head (untied)  + MTP module (D=1)"),
             ("new", "noaux: per-step bias (g 0.001->0) + SeqBal 1e-4\n+ FP8 training (all-E4M3, group-scaled)")],
            "KV/token = 61 x (512+64) = 35,136 elements | ctx 163,840\n(MTP 13.46B extra; checkpoint 685B = 671 + 14)")

    ax.text(5.12, 4.05, "blue: unchanged lineage\norange: V3 changes", ha="center",
            va="center", fontsize=6.9, color=INK2, rotation=90, linespacing=1.4)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-9-1-dsv2-v3-machine-diff.png")
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[产物] {out}")


# ---------------- 图 9.2：409M 两刀机结构与插槽来源（自绘示意级） ----------------
def fig_9_2():
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    x0, w = 0.55, 4.9

    ax.text(x0 + w / 2, 7.42, "Two-Knife 409M (cand2, main arm)", ha="center", fontsize=10,
            fontweight="bold", color=INK)
    ax.text(x0 + w / 2, 7.14, "d=1024, L=12, h=16, V=32,000 untied | total 409,115,648",
            ha="center", fontsize=7.0, color=INK2)

    box(ax, x0, 6.42, w, 0.44, "Embedding   (B,n) -> (B,n,1024)", BLUE, fs=7.6)
    arrow(ax, x0 + w / 2, 6.42, x0 + w / 2, 6.20)

    # Block x12：外框 + 两个并排插槽（注意力格 | 前馈格）
    ax.add_patch(FancyBboxPatch((x0, 4.30), w, 1.88, boxstyle="round,pad=0.012,rounding_size=0.012",
                                fc="#f7f6f2", ec=INK2, lw=1.0))
    ax.text(x0 + w / 2, 6.02, "Block x12   (residual stream stays (B,n,1024))", ha="center",
            fontsize=7.4, color=INK, fontweight="bold")
    box(ax, x0 + 0.18, 4.98, w / 2 - 0.30, 0.86,
        "MLA  (cut 2)\ncache c_KV (B,n,256) + k_R (B,n,64)\nd_c=256, d_c'=256, 64/64", ORANGE, fs=6.8)
    box(ax, x0 + w / 2 + 0.12, 4.98, w / 2 - 0.30, 0.86,
        "MoE   (cut 1)\n8 routed top-2 + 1 shared, w=938\ngate (N,E)=(8192,8)", ORANGE, fs=6.8)
    ax.text(x0 + w / 2, 4.46, "each slot: pre-RMSNorm (Book3 ch2) -> slot module | "
                              "RoPE theta=1e4 on k_R/q_R lanes (Book3 ch4, inside MLA)",
            ha="center", fontsize=6.0, color=INK2)
    arrow(ax, x0 + w / 2, 4.30, x0 + w / 2, 4.08)

    box(ax, x0, 3.62, w, 0.44, "RMSNorm -> lm_head   (B,n,1024) -> (B,n,32000)", BLUE, fs=7.4)

    # 供货方（右侧清单：色点 + 文字，不引线——插槽与供货方的对应由 cut1/cut2 标签承载）
    src = [(6.62, "Book3 ch2   RMSNorm   (unchanged)", BLUE),
           (5.60, "Book4 ch6   MLA   (cut 2, new slot)", ORANGE),
           (5.02, "Book4 ch5   DeepSeekMoE   (cut 1, new slot)\n            SwiGLU expert = Book3 ch3", ORANGE),
           (4.44, "Book3 ch4   rope   (unchanged)", BLUE),
           (3.84, "Book3 base   embed / lm_head / recipe", BLUE)]
    for y_txt, text, color in src:
        ax.plot([x0 + w + 0.72], [y_txt], marker="s", ms=5, color=color)
        ax.text(x0 + w + 0.86, y_txt, text, ha="left", va="center", fontsize=7.0,
                color=INK2, linespacing=1.35)

    # 退役虚影（灰虚线框；箭头指回各自插槽：GQA->注意力格（左）、dense FFN->前馈格（右））
    box(ax, 0.55, 2.30, 2.45, 0.86, "retired: GQA\nh=16, h_kv=8, d_k=64\nKV 12,288 elem/token", "white",
        ec=TICK, ls=(0, (3, 2)), tc=INK2, fs=6.8)
    box(ax, 3.10, 2.30, 2.45, 0.86, "retired: dense SwiGLU FFN\nd_ff = 2816", "white",
        ec=TICK, ls=(0, (3, 2)), tc=INK2, fs=6.8)
    arrow(ax, 1.78, 3.16, 1.78, 4.94, color=TICK, ls=(0, (3, 2)))        # GQA 虚影 -> MLA 槽（左）
    arrow(ax, 4.33, 3.16, 4.33, 4.94, color=TICK, ls=(0, (3, 2)))        # FFN 虚影 -> MoE 槽（右）

    ax.text(5.35, 1.30, "KV/token: 12 x (256+64) = 3,840 elements  vs  retired GQA 12,288  (-68.75%)\n"
                        "active params/step: 168.9M (with lm_head)  vs  total 409.1M\n"
                        "cand3 control arm (w=329): total 207,064,064 = Book3 207M + 0.03%",
            ha="center", va="center", fontsize=7.2, color=INK, linespacing=1.7)
    ax.text(5.35, 0.32, "the mod history lives in the imports:  RMSNorm/rope from Book3;  MLA/DeepSeekMoE from Book4",
            ha="center", fontsize=6.9, color=INK2)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-9-2-two-knife-409m-slots.png")
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[产物] {out}")


# ---------------- 图 9.3：冒烟 + 整合短训曲线（自产） ----------------
def _read_curve(path):
    if not os.path.exists(path):
        return None
    steps, train, ev = [], [], {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("step") and r.get("train_loss"):
                steps.append(int(r["step"]))
                train.append(float(r["train_loss"]))
                if r.get("eval_loss"):
                    ev[int(r["step"])] = float(r["eval_loss"])
    return steps, train, ev


def fig_9_3():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 4))
    series = [
        ("cand2 409M (smoke)", os.path.join(LOG9, "curve_smoke_cand2_s2.csv"), BLUE),
        ("cand3 207M (smoke)", os.path.join(LOG9, "curve_smoke_cand3_s3.csv"), ORANGE),
    ]
    for name, path, color in series:                                   # 左：冒烟前 20 步 train loss
        c = _read_curve(path)
        if c is None:
            print(f"[缺] {path}——先跑 train_409.py --task smoke（该曲线自动跳过）")
            continue
        steps, train, _ = c
        a1.plot(steps, train, color=color, lw=2, marker="o", ms=3.5, label=name)
    a1.axhline(10.578, color=TICK, lw=1.0, ls=(0, (4, 2)))
    a1.text(10.2, 10.578 + 0.12, "ln V + d$\\sigma^2$/2 = 10.578", fontsize=6.8, color=INK2)
    a1.set_xlabel("step"), a1.set_ylabel("train loss (nats)")
    a1.set_title("smoke: first 20 steps", fontsize=9, color=INK)
    a1.legend(frameon=False, fontsize=7), a1.grid(color=GRID, lw=0.5), a1.set_axisbelow(True)

    evals = [                                                          # 右：整合短训 eval 曲线
        ("cand2 409M two-knife", os.path.join(LOG9, "curve_integrated_cand2_run1.csv"), BLUE, "o"),
        ("cand3 207M two-knife", os.path.join(LOG9, "curve_integrated_cand3_run1.csv"), ORANGE, "s"),
        ("Book3 207M dense", os.path.join(LOG3, "curve_integrated_run1.csv"), TEAL, "^"),
    ]
    for name, path, color, mk in evals:
        c = _read_curve(path)
        if c is None:
            print(f"[缺] {path}——跳过")
            continue
        _, _, ev = c
        if not ev:
            continue
        items = sorted(ev.items())
        xs = [s for s, _ in items]
        ys = [v for _, v in items]
        a2.plot(xs, ys, color=color, lw=2, marker=mk, ms=4, label=name)
    a2.set_xlabel("step"), a2.set_ylabel("eval loss (nats, 49 windows fp32)")
    a2.set_title("integrated short runs (same corpus & recipe)", fontsize=9, color=INK)
    a2.legend(frameon=False, fontsize=7), a2.grid(color=GRID, lw=0.5), a2.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-9-3-smoke-integrated-curves.png")
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[产物] {out}")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_9_1()
    fig_9_2()
    fig_9_3()
