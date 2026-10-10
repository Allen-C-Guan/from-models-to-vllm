# fig_ch10.py —— Book6 ch10 制图：图 10.1 起草-验证-回滚时间线 / 图 10.2 验证器两分支 /
#                图 10.3 EAGLE 特征级 draft 结构图 / 图 10.4 本机实测对理论（slug=spec-decode）
# 用途：图 10.1 = 10.2 节机制全景（规范 §4 强制项：token 级对齐+回滚边界标注——自绘）；
#       图 10.2 = 10.2 节验证器接受/拒绝两分支（自绘标修正概率）；
#       图 10.4 = 10.6 节实测对理论（喂 spec_decode_base.json：E(α) 理论曲线族+本机点+两臂加速比条）
# 所属章节：Book6 第 10 章（图 10.1-10.4）
# 运行方式：cd 工作区根目录 && source env.sh && python "code/Book6-推理系统导论/ch10/fig_ch10.py"
# 产物：drafts/Book6-推理系统导论/figures/fig-10-{1,2,4}-spec-decode.png + fig-10-3-eagle-draft.png
# 图规：300dpi 白底；系列色 蓝#2a78d6 橙#eb6834 青#1baf7a 黄#eda100；线宽 2；图内英文、图注中文。
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "drafts", "Book6-推理系统导论", "figures")
SPEC_JSON = os.path.join(REPO_ROOT, "log", "book6-ch10", "spec_decode_base.json")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e1e0d9"


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=7.2, lw=1.3, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=lw,
                                mutation_scale=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc, zorder=4)


def arr(ax, x0, y0, x1, y1, color=MUTED, lw=1.1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))


# ---------------- 图 10.1：起草-验证-回滚时间线 ----------------
def fig_timeline():
    fig, ax = plt.subplots(figsize=(11.4, 4.4), dpi=300)
    ax.set_xlim(0, 14); ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.text(7, 5.95, "Draft → verify → rollback: one speculative round (γ=4, prefix n tokens)",
            ha="center", fontsize=10, color=INK, weight="bold")
    # prefix
    box(ax, 0.3, 2.6, 2.0, 1.0, "prefix\n(n, d)", ec=MUTED)
    # draft 链
    for i in range(4):
        box(ax, 2.9 + i * 1.35, 2.6, 1.15, 1.0, f"$x_{{{i+1}}}$\n(γ,)", ec=BLUE, fc="#dcebfc")
    ax.text(4.95, 4.05, "draft M_q: γ autoregressive steps — cheap", fontsize=7.6, color=BLUE, ha="center")
    # verify 一次
    box(ax, 8.6, 2.6, 2.6, 1.0, "verify M_p\n(1, n+γ) → (n+γ, V)\none forward", ec=TEAL, fc="#d9f2e7")
    arr(ax, 8.35, 3.1, 8.6, 3.1, color=TEAL)
    # 接受/拒绝
    box(ax, 12.0, 4.3, 1.7, 0.9, "accept ≤ n\n+ bonus", ec=TEAL, fc="#d9f2e7", fs=7.0)
    box(ax, 12.0, 1.4, 1.7, 0.9, "reject @ i:\nrollback + fix\nsample residual", ec=ORANGE, fc="#fde3d6", fs=6.8)
    arr(ax, 11.2, 3.4, 12.0, 4.55, color=TEAL)
    arr(ax, 11.2, 2.9, 12.0, 2.0, color=ORANGE)
    ax.text(12.85, 0.85, "outputs 1..γ+1 tokens per round", fontsize=7.4, color=MUTED, ha="center")
    # 回滚边界标注
    ax.plot([3.5 + 2 * 1.35, 3.5 + 2 * 1.35], [2.35, 5.3], color=ORANGE, lw=1.0, ls="--", zorder=2)
    ax.text(3.5 + 2 * 1.35, 5.45, "rollback boundary (first rejection: keep ≤ here,\nresample fix token from residual p−q)",
            fontsize=7.2, color=ORANGE, ha="center")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-10-1-spec-decode.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 10.1] {path}")


# ---------------- 图 10.2：验证器两分支 ----------------
def fig_branch():
    fig, ax = plt.subplots(figsize=(10.2, 4.2), dpi=300)
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.text(6.5, 5.35, "Verifier per position: accept with prob min(1, p/q); reject → fix from residual",
            ha="center", fontsize=9.6, color=INK, weight="bold")
    box(ax, 0.3, 2.3, 2.2, 1.0, "draft token x ~ q\nlogits (V,)", ec=BLUE, fc="#dcebfc")
    box(ax, 3.0, 2.3, 2.2, 1.0, "target p(x), q(x)\nscalars at x", ec=MUTED)
    arr(ax, 2.5, 2.8, 3.0, 2.8)
    # accept 分支
    box(ax, 6.0, 3.6, 2.9, 0.95, "r < p(x)/q(x)  → accept x\nP = q(x)·min(1, p/q) = min(p, q)", ec=TEAL, fc="#d9f2e7", fs=7.0)
    # reject 分支
    box(ax, 6.0, 1.2, 2.9, 0.95, "r ≥ p/q → reject; sample fix from\np′ = norm(max(0, p−q)) = (p−min(p,q))/(1−β)", ec=ORANGE, fc="#fde3d6", fs=6.8)
    arr(ax, 5.2, 3.1, 6.0, 3.95, color=TEAL)
    arr(ax, 5.2, 2.5, 6.0, 1.75, color=ORANGE)
    box(ax, 9.6, 2.4, 3.0, 1.1, "merge: min(q,p) + (1−β)·p′\n= p(x)  — identical output dist.", ec=INK, fs=7.4)
    arr(ax, 8.9, 4.05, 9.6, 3.2, color=TEAL)
    arr(ax, 8.9, 1.7, 9.6, 2.7, color=ORANGE)
    ax.text(6.5, 0.45, "two paths add up to exactly p — losslessness is a theorem, not a hope (self-test: max|freq−p| = 0.0025)",
            ha="center", fontsize=7.6, color=MUTED)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-10-2-spec-decode.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 10.2] {path}")


# ---------------- 图 10.3：EAGLE 特征级 draft 结构图 ----------------
def fig_eagle():
    fig, ax = plt.subplots(figsize=(11.6, 5.0), dpi=300)
    ax.set_xlim(0, 15); ax.set_ylim(0, 6.6)
    ax.axis("off")
    ax.text(7.5, 6.35, "EAGLE: autoregressive draft at the FEATURE level — reuse, one FC, one decoder layer, shifted-token 2nd input",
            ha="center", fontsize=9.6, color=INK, weight="bold")
    # 左：target 一次前向（冻结）
    box(ax, 0.3, 3.3, 3.4, 1.9, "TARGET M_p (frozen)\ntokens (n,d_e) → layers\n→ features f (n,d)\n→ LM head → (n,V)", ec=MUTED, fc="#f2f1ec", fs=7.2)
    ax.text(2.0, 2.95, "one forward per round (verify)", fontsize=6.8, color=MUTED, ha="center")
    # 中上：draft 单步（三件套）
    box(ax, 4.6, 4.55, 2.6, 1.35, "input ①: feature f_t  (d,)\ninput ②: token y_{t−1}  (d_e,)\n(shifted one extra step)", ec=BLUE, fc="#dcebfc", fs=6.8)
    box(ax, 7.7, 4.55, 2.2, 1.35, "FC: concat → project\n(2d→d) — trainable", ec=BLUE, fc="#dcebfc", fs=7.0)
    box(ax, 10.4, 4.55, 2.5, 1.35, "one decoder layer\n(0.24-0.99B, trainable)\n→ feature f̂_{t+1} (d,)", ec=BLUE, fc="#dcebfc", fs=6.8)
    box(ax, 13.3, 4.55, 1.5, 1.35, "reused\nLM head\n→ token ŷ_t", ec=MUTED, fc="#f2f1ec", fs=6.8)
    arr(ax, 7.2, 5.2, 7.7, 5.2); arr(ax, 9.9, 5.2, 10.4, 5.2); arr(ax, 12.9, 5.2, 13.3, 5.2)
    # 回路：ŷ_t 作为下一步的 input ②
    ax.annotate("", xy=(5.9, 5.9), xytext=(13.5, 6.15), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.0, ls="--",
               connectionstyle="arc3,rad=-0.12"))
    ax.text(9.7, 6.12, "autoregressive loop: ŷ_t feeds the next draft step", fontsize=6.6, color=BLUE, ha="center")
    ax.text(6.4, 4.25, "reused Embedding & LM Head (free)  +  FC  +  1 decoder layer  = the whole draft M_q", fontsize=7.0, color=INK, ha="center")
    # 下方：静态树展开
    box(ax, 4.6, 0.5, 3.0, 1.3, "static draft tree:\nexpand top-K candidates\nper level (fixed shape)", ec=TEAL, fc="#d9f2e7", fs=7.0)
    box(ax, 8.2, 0.5, 3.2, 1.3, "tree attention verify:\nsame-branch tokens count\nas history (mask generalizes\ncausal chain→tree)", ec=TEAL, fc="#d9f2e7", fs=6.4)
    box(ax, 12.0, 0.5, 2.6, 1.3, "τ = 3.2-4.5 tokens/forward\n(LLaMA2-Chat 70B,\n2.7-3.5× latency)", ec=TEAL, fc="#d9f2e7", fs=6.6)
    arr(ax, 7.6, 1.15, 8.2, 1.15, color=TEAL); arr(ax, 11.4, 1.15, 12.0, 1.15, color=TEAL)
    arr(ax, 3.7, 3.3, 5.6, 1.8, color=MUTED)  # target features feed draft
    ax.text(4.3, 2.7, "f (features)\nfeed draft input ①", fontsize=6.4, color=MUTED, ha="center")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-10-3-eagle-draft.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 10.3] {path}（EAGLE 2401.15077 三件套+提前一步 token 输入+静态树——自绘示意级）")


# ---------------- 图 10.4：本机实测对理论 ----------------
def fig_measured():
    d = json.load(open(SPEC_JSON))
    rep = next(w for w in d["workloads"] if w["label"] == "repeat")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0), dpi=300)
    # 左：E(α) 理论曲线族 + 本机点
    alphas = [i / 200 for i in range(4, 96)]
    for g, c in ((4, BLUE), (2, TEAL)):
        ax1.plot(alphas, [(1 - a ** (g + 1)) / (1 - a) for a in alphas], color=c, lw=2,
                 label=f"theory E(α), γ={g}", zorder=4)
    ax1.scatter([rep["accept_rate"]], [rep["accepted_per_step"]], color=ORANGE, s=52, zorder=5,
                label=f"this run: ratio {rep['accept_rate']:.2f} → {rep['accepted_per_step']:.1f} tok/round")
    ax1.set_xlabel("per-position acceptance α", fontsize=8.5, color=MUTED)
    ax1.set_ylabel("tokens per round E", fontsize=8.5, color=MUTED)
    ax1.set_ylim(0.8, 5.2)
    ax1.tick_params(colors="#898781", labelsize=7.5)
    ax1.grid(color=GRIDC, lw=0.5, zorder=0)
    ax1.legend(fontsize=7.4, frameon=False, loc="upper left")
    ax1.set_title("α decides everything: this run sits at the valley floor (E≈1.0, implied iid α≈0.05)", fontsize=8.8, color=INK)
    # 右：两臂成本
    labels = [f"spec arm\n({rep['t_spec_per_step_ms']:.1f} ms/round,\nyields {rep['accepted_per_step']:.1f} tok)",
              f"baseline arm\n({rep['t_base_per_token_ms']:.1f} ms/tok,\nyields 1 tok)"]
    vals = [rep["t_spec_per_step_ms"] / rep["accepted_per_step"],
            rep["t_base_per_token_ms"]]
    ax2.bar([0, 1], vals, color=[ORANGE, TEAL], width=0.5, zorder=3)
    for i, v in enumerate(vals):
        ax2.text(i, v + 1, f"{v:.1f} ms/tok", ha="center", fontsize=8.6, color=INK)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(labels, fontsize=7.6)
    ax2.set_ylabel("ms per produced token", fontsize=8.5, color=MUTED)
    ax2.set_ylim(0, 60)
    ax2.tick_params(colors="#898781", labelsize=7.5)
    ax2.grid(axis="y", color=GRIDC, lw=0.5, zorder=0)
    ax2.set_title(f"cost side: speedup = {rep['speedup']}× (preregistered negative result)", fontsize=8.8, color=INK)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig-10-4-spec-decode.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 10.4] {path}（数据源 spec_decode_base.json）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch10 制图（图 10.1/10.2/10.4）")
    ap.add_argument("--which", default="all", choices=["all", "timeline", "branch", "eagle", "measured"])
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    if args.which in ("all", "timeline"):
        fig_timeline()
    if args.which in ("all", "branch"):
        fig_branch()
    if args.which in ("all", "eagle"):
        fig_eagle()
    if args.which in ("all", "measured"):
        fig_measured()
