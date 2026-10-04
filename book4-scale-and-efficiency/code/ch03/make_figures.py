# make_figures.py —— Book4 ch3 三张图：图 3.1 赢者通吃回路 / 图 3.2 稳定性三级台阶 / 图 3.3 负载前后对照热图
# 用途：图 3.1 与图 3.2 为自绘示意级（框线图/台阶图，张量框与关键箭头标形状）；
#       图 3.3 读 log/book4-ch03/routing_{arm}_{tier}.csv 的 init 与 final 快照（长表：phase/step/layer/expert/count/frac），
#       画 6×8（层×专家）硬计数热图三联：init / none-final / aux-final——本机负载崩塌现场与干预的招牌对照图。
# 所属章节：Book4 第 3 章 §3.2（图 3.1）/§3.4（图 3.2）/§3.5（图 3.3）
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch03/make_figures.py" \
#           [--tier auto|fast|full]      （auto=full 的 JSON 完整跑毕则用 full，否则 fast）
# 产物：figures/fig-3-1-winner-take-all-loop.png、fig-3-2-stability-steps.png、
#       fig-3-3-balance-heatmap.png（300dpi；图内文字英文、图注中文见正文）
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DIR = os.path.join(REPO, "figures")
LOG_DIR = os.path.join(REPO, "log", "book4-ch03")
os.makedirs(FIG_DIR, exist_ok=True)

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MUTED = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
SEQ_CMAP = LinearSegmentedColormap.from_list("book_blue", ["#cde2fb", "#0d366b"])  # 顺序型蓝渐变（图规）
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": TICK, "axes.labelcolor": DARK,
    "xtick.color": TICK, "ytick.color": TICK, "text.color": DARK,
    "axes.linewidth": 0.8, "font.family": "DejaVu Sans",
})

N_LAYER, N_EXPERT = 6, 8


def box(ax, x, y, w, h, label, fc="white", ec=DARK, lw=1.1, fs=7.6, tc=None, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc or DARK)


def arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.1, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def pick_tier():
    """auto：full 的 JSON 两臂 steps_done==steps_planned 则 full，否则 fast。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="auto", choices=["auto", "fast", "full"])
    args = ap.parse_args()
    if args.tier != "auto":
        return args.tier
    jp = os.path.join(LOG_DIR, "ablation_balance_full.json")
    if os.path.exists(jp):
        with open(jp, encoding="utf-8") as f:
            r = json.load(f)
        arms = r.get("arms", {})
        if all(a in arms and arms[a].get("steps_done") == arms[a].get("steps_planned")
               for a in ("none", "aux")):
            return "full"
    return "fast"


# ---------------- 图 3.1：赢者通吃自增强回路（自绘示意级，标形状） ----------------
def fig_3_1():
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=300)
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    # 环形六站：router 打分 → top-k → 得宠专家收 token → 梯度只经选中门 → 专家变强 → 打分更高 → 回到 router
    box(ax, 4.9, 7.3, 2.9, 0.95, "token batch x\n(B*n, d)", fc="#f2f7fd", ec=BLUE, lw=1.3)
    box(ax, 9.2, 5.6, 3.1, 1.05, "router scores\nx @ W_g^T : (B*n, d)x(d, E)\n-> (B*n, E)", ec=BLUE)
    box(ax, 9.2, 3.2, 3.1, 1.05, "softmax + top-k\ngate (B*n, k)\nindex (B*n, k)", ec=BLUE)
    box(ax, 8.6, 0.7, 3.4, 1.05, "favored expert E_i\nreceives most tokens\n(loads >> mean)", fc="#fdf3ef",
        ec=ORANGE, lw=1.4)
    box(ax, 3.4, 0.7, 4.2, 1.05, "backward: gradient reaches E_i\nonly through its gate value g_i\n(chosen ones train)", fc="#fdf3ef",
        ec=ORANGE, lw=1.4)
    box(ax, 0.3, 3.2, 3.6, 1.05, "E_i improves on its slice\n-> stronger on those tokens", fc="#fdf3ef",
        ec=ORANGE, lw=1.4)
    box(ax, 0.3, 5.6, 3.6, 1.05, "next batch: score of E_i\nranks even higher", fc="#fdf3ef",
        ec=ORANGE, lw=1.4)

    # 顺时针箭头（右侧蓝链 = 一次前向；下方与左侧橙链 = 自增强回路）
    arrow(ax, 7.1, 7.3, 10.6, 6.7, color=BLUE)                # x -> scores
    arrow(ax, 10.75, 5.6, 10.75, 4.3, color=BLUE)             # scores -> top-k
    arrow(ax, 10.3, 3.2, 10.3, 1.8, color=ORANGE, lw=1.4)     # top-k -> 得宠专家
    arrow(ax, 8.6, 1.22, 7.6, 1.22, color=ORANGE, lw=1.4)     # 得宠专家 -> 反向
    arrow(ax, 3.4, 1.22, 2.1, 3.2, color=ORANGE, lw=1.4)      # 反向 -> 变强
    arrow(ax, 2.1, 4.25, 2.1, 5.6, color=ORANGE, lw=1.4)      # 变强 -> 打分更高
    arrow(ax, 3.9, 6.4, 10.45, 6.68, color=ORANGE, lw=1.6, rad=-0.22)  # 打分更高 -> scores（闭合）

    # 中央标注 + 落选专家支线
    ax.text(6.35, 4.15, "winner-take-all\nself-reinforcing loop", ha="center", va="center",
            fontsize=10.5, color=ORANGE, fontweight="bold")
    box(ax, 4.55, 2.5, 3.6, 1.0, "unchosen experts:\nzero gradient, frozen\n(dead experts never recover)",
        ec=MUTED, ls="--", fs=7.2)
    arrow(ax, 9.35, 3.5, 8.2, 3.0, color=MUTED, ls="--")
    ax.text(6.3, 0.18, "orange = the loop that destroys balance;  blue = one forward pass",
            ha="center", fontsize=7.4, color=MUTED)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-3-1-winner-take-all-loop.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("[fig-3-1]", out)


# ---------------- 图 3.3：负载前后对照热图（本机现场；routing 长表 → L×E 计数矩阵三联） ----------------
def load_snapshots(tier):
    """读 routing_{arm}_{tier}.csv，返回 {arm: {step: 6x8 计数矩阵}}；final = 最大 step 的快照。"""
    snaps = {}
    for arm in ("none", "aux"):
        path = os.path.join(LOG_DIR, f"routing_{arm}_{tier}.csv")
        rows = {}
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (r["phase"], int(r["step"]))
                rows.setdefault(key, [[0] * N_EXPERT for _ in range(N_LAYER)])
                rows[key][int(r["layer"])][int(r["expert"])] = int(r["count"])
        snaps[arm] = rows
    init_key = ("init", 0)
    return {
        "init": rows_lookup(snaps, "none", init_key),
        "none_final": final_lookup(snaps, "none"),
        "aux_final": final_lookup(snaps, "aux"),
    }


def rows_lookup(snaps, arm, key):
    m = snaps[arm][key]
    return [row[:] for row in m]


def final_lookup(snaps, arm):
    keys = [k for k in snaps[arm] if k[0] == "train"]
    return rows_lookup(snaps, arm, max(keys, key=lambda k: k[1]))


def fig_3_3(tier):
    data = load_snapshots(tier)
    panels = [
        ("Init (random weights)", data["init"], "none"),
        ("No aux loss", data["none_final"], "none"),
        (f"Load loss w=0.1", data["aux_final"], "aux"),
    ]
    step_final = max(k[1] for k in [k for k in csv_steps(tier, "none")] if k[0] == "train")
    vmax = max(max(max(row) for row in m) for _, m, _ in panels)

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.9), dpi=300)
    im = None
    for ax, (title, mat, _) in zip(axes, panels):
        im = ax.imshow(mat, cmap=SEQ_CMAP, vmin=0, vmax=vmax, aspect="auto")
        # 均匀期望 = 16384*2/8 = 4096/格；逐格标计数，死格（0）橙框点名
        for li in range(N_LAYER):
            for ei in range(N_EXPERT):
                c = mat[li][ei]
                ax.text(ei, li, str(c), ha="center", va="center", fontsize=5.4,
                        color="white" if c > vmax * 0.55 else DARK)
                if c == 0:
                    ax.add_patch(Rectangle((ei - 0.5, li - 0.5), 1, 1, fill=False,
                                           edgecolor=ORANGE, lw=1.4))
        mmp = maxmin_layer_mean(mat)
        ax.set_title(f"{title}\nlayer-mean max/min = {mmp}", fontsize=8.2, color=DARK)
        ax.set_xticks(range(N_EXPERT))
        ax.set_xticklabels([f"E{i}" for i in range(N_EXPERT)], fontsize=6.5)
        ax.set_yticks(range(N_LAYER))
        ax.set_yticklabels([f"L{i}" for i in range(N_LAYER)], fontsize=6.5)
        ax.set_xlabel("expert", fontsize=7.5, color=MUTED)
        if ax is axes[0]:
            ax.set_ylabel("layer", fontsize=7.5, color=MUTED)
        for s in ax.spines.values():
            s.set_color(TICK)
    axes[1].set_title(f"No aux loss  (step {step_final})\nlayer-mean max/min = "
                      f"{maxmin_layer_mean(data['none_final'])}", fontsize=8.2, color=DARK)
    axes[2].set_title(f"Load loss w=0.1  (step {step_final})\nlayer-mean max/min = "
                      f"{maxmin_layer_mean(data['aux_final'])}", fontsize=8.2, color=DARK)

    cb = fig.colorbar(im, ax=axes, fraction=0.026, pad=0.02)
    cb.set_label("expert selections per 16,384 (= n_tok x k)", fontsize=7.5, color=MUTED)
    cb.ax.tick_params(labelsize=6.5, color=TICK)
    cb.outline.set_edgecolor(TICK)
    fig.text(0.42, 0.008, "orange box = dead expert (0 selections);  uniform expectation = 2,048 per cell",
             ha="center", fontsize=7.2, color=MUTED)
    out = os.path.join(FIG_DIR, "fig-3-3-balance-heatmap.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("[fig-3-3]", out, f"(tier={tier})")


def csv_steps(tier, arm):
    """返回 [(phase, step)] 去重列表（画 final 步号用）。"""
    with open(os.path.join(LOG_DIR, f"routing_{arm}_{tier}.csv"), encoding="utf-8") as f:
        seen = []
        for r in csv.DictReader(f):
            k = (r["phase"], int(r["step"]))
            if k not in seen:
                seen.append(k)
        return seen


def maxmin_layer_mean(mat):
    """层均 max/min（死格下限取 1——与 ablation_balance.py 度量口径一致）。"""
    import numpy as np
    vals = [max(row) / max(min(row), 1) for row in mat]
    return round(float(np.mean(vals)), 2)


# ---------------- 图 3.2：稳定性三级台阶时间线（自绘示意级） ----------------
def fig_3_2():
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=300)
    ax.set_xlim(2019.0, 2027.4)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    # 三级台阶（相邻不重叠）：fp32 硬扛 → 精度分区 → 损失侧控制（低精度化与稳定化同一条线，通往 ch7）
    steps = [
        (2019.7, 1.15, 1.9, BLUE, "GShard 2020\nfp32 everywhere",
         "600B trained in full fp32;\n1T bf16 attempt abandoned\nfor numerical stability"),
        (2021.75, 2.1, 1.9, ORANGE, "Switch 2021\nselective precision",
         "router in fp32, experts bf16;\nbf16-only diverges (-3.780)\nvs selective -1.716; 0.1x init"),
        (2023.8, 3.05, 1.9, TEAL, "ST-MoE 2022\nrouter z-loss",
         "penalize (log sum exp)^2;\nc_z = 0.001: 3/6 -> 3/3 stable,\nquality -1.755 -> -1.741"),
    ]
    for x0, h, w, color, head, detail in steps:
        ax.add_patch(Rectangle((x0, 0.55), w, h, fc=color, ec="none", alpha=0.92))
        ax.text(x0 + w / 2, 0.55 + h - 0.12, head, ha="center", va="top", fontsize=8.0,
                color="white", fontweight="bold", linespacing=1.35)
        ax.text(x0 + w / 2, 0.55 + h - 0.52, detail, ha="center", va="top", fontsize=6.6,
                color="white", linespacing=1.45)
        ax.plot([x0, x0], [0.55, 0.55 + h], color=DARK, lw=0.8)   # 台阶立沿
    ax.plot([2019.4, 2025.9], [0.55, 0.55], color=DARK, lw=1.2)
    for yr in range(2020, 2026):
        ax.plot([yr, yr], [0.5, 0.6], color=DARK, lw=1.0)
        ax.text(yr, 0.28, str(yr), ha="center", fontsize=7.4, color=TICK)

    # 通往 ch7 的虚线（下一级台阶：训练侧低精度）
    ax.annotate("", xy=(2026.35, 4.28), xytext=(2025.75, 3.72),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3, linestyle="--"))
    ax.text(2026.2, 4.5, "FP8 / MXFP4\ntraining\n(Ch.7)", ha="center", va="center",
            fontsize=7.4, color=MUTED, linespacing=1.4)
    ax.text(2022.6, 0.06, "stabilization toolkit grows as precision drops: full-fp32 crutch -> precision zoning -> loss-side control",
            ha="center", fontsize=7.2, color=MUTED)
    ax.text(2019.15, 4.55, "stability under\nlower precision", fontsize=8.0, color=DARK,
            ha="left", va="center", linespacing=1.4)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-3-2-stability-steps.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("[fig-3-2]", out)


if __name__ == "__main__":
    tier = pick_tier()
    fig_3_1()
    fig_3_2()
    fig_3_3(tier)
    print(f"[done] tier={tier}")
