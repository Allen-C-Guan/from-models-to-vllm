# 用途：第 2 章「缩放点积注意力」全部数值实验——手写 matmul+softmax 注意力、√d_k 的
#       分布/softmax 饱和/梯度实验、mini 批次显存与 FLOPs 账本、n 二次增长 wall-clock
#       曲线，以及与 F.scaled_dot_product_attention 的对拍（实现对照，锁 MATH 后端）。
# 所属章节：Book1《Transformer 原典》第 2 章
# 运行方式：source env.sh && python code/ch02/scaled_dot_product.py
#           （CPU 上秒级-分钟级；三张图输出到 figures/）
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import FancyBboxPatch
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.manual_seed(1706)  # 固定种子（取论文编号，便于复现）

# 本书印刷图规格：蓝 #2a78d6 主色、橙 #eb6834 次色、hairline 网格、灰系坐标文字
BLUE, ORANGE, INK, GRAY = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "savefig.dpi": 300, "figure.dpi": 300,
    "axes.edgecolor": "#898781", "axes.labelcolor": GRAY,
    "xtick.color": "#898781", "ytick.color": "#898781",
})

from pathlib import Path
FIG_DIR = Path(__file__).resolve().parents[3] / "drafts" / "Book1-Transformer原典" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def style_ax(ax, title=""):
    """统一坐标轴样式：网格置底、灰色刻度、黑色标题。"""
    ax.set_axisbelow(True)
    ax.grid(True, color="#e1e0d9", linewidth=0.6)
    for s in ax.spines.values():
        s.set_color("#898781")
    if title:
        ax.set_title(title, color=INK, fontsize=11)


# ---------- 1. 手写缩放点积注意力（正文实现：matmul + softmax，不用 SDPA） ----------
def attention(q, k, v, mask=None):
    """式 (2.1)：Attention(Q,K,V)=softmax(QK^T/√d_k)V。
    q:(b,h,n,d_k)  k:(b,h,m,d_k)  v:(b,h,m,d_v)  ->  (b,h,n,d_v)"""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    # (b,h,n,d_k)×(b,h,d_k,m) -> (b,h,n,m) 打分矩阵 QK^T/√d_k
    if mask is not None:  # 可选屏蔽发生在 SoftMax 之前（对应论文 Fig.2 的 Mask (opt.)）
        scores = scores.masked_fill(mask, float("-inf"))  # (b,h,n,m) 形状不变
    w = torch.softmax(scores, dim=-1)  # (b,h,n,m) 每行归一化成对 m 个 value 的权重
    return w @ v  # (b,h,n,m)×(b,h,m,d_v) -> (b,h,n,d_v)


# ---------- 2. 实现对照：与 SDPA 对拍（锁 MATH 后端，fp32、固定种子、多组随机输入） ----------
def compare_sdpa(trials=5, b=2, h=8, n=12, d=64):
    """实现对照：手写 attention() vs F.scaled_dot_product_attention（锁 MATH 后端）。
    输入：trials 组随机 q,k,v:(b,h,n,d)；返回：(allclose 是否全过, 最大绝对差)。"""
    ok_all, worst = True, 0.0
    for t in range(trials):
        g = torch.Generator().manual_seed(1706 + t)
        q, k, v = (torch.randn(b, h, n, d, generator=g) for _ in range(3))
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            ref = F.scaled_dot_product_attention(q, k, v)  # 默认 scale=1/√d_k，与论文一致
        mine = attention(q, k, v)
        ok_all &= bool(torch.allclose(mine, ref, rtol=0, atol=1e-5))
        worst = max(worst, (mine - ref).abs().max().item())
    print(f"[对拍] 手写 vs SDPA(MATH)：{trials} 组随机输入 allclose={ok_all}（rtol=0, atol=1e-5），max|Δ|={worst:.2e}")
    return ok_all


# ---------- 3. √d_k 实验：点积分布（脚注 4 的方差论断，实测对照） ----------
def exp_dot_distribution(d_k=64, n_pairs=200_000):
    """实验一：去缩放的点积分布（脚注 4 方差论断的实测对照）。
    q,k:(n_pairs,d_k) iid N(0,1) -> 逐对点积 dots:(n_pairs,) 与 scaled:(n_pairs,)。"""
    q, k = torch.randn(n_pairs, d_k), torch.randn(n_pairs, d_k)
    dots = (q * k).sum(-1)               # (n_pairs,d_k)x2 -> (n_pairs,) 不缩放的 q·k
    scaled = dots / math.sqrt(d_k)       # (n_pairs,) 缩放后
    print(f"[分布] d_k={d_k}，{n_pairs:,} 个样本（q、k 分量 iid N(0,1)）")
    print(f"[分布] 不缩放  mean={dots.mean():+.4f}  std={dots.std():.4f}（理论 std=√d_k={math.sqrt(d_k):.1f}）")
    print(f"[分布] 缩放后  mean={scaled.mean():+.4f}  std={scaled.std():.4f}（理论 std=1.0）")
    return dots, scaled


# ---------- 4. √d_k 实验：softmax 饱和与梯度消失（m=30 个 key，万行取中位数） ----------
def exp_softmax_saturation(d_ks=(16, 64, 256), m=30, rows=10_000):
    """实验二/三：去缩放的 softmax 饱和与梯度消失扫描。
    q:(rows,d_k)、K:(rows,m,d_k) -> einsum 打分 scores:(rows,m) -> softmax p:(rows,m)。
    对每个 d_k 取 rows 个随机 (q,K) 行，报告中位数；
    ‖J‖_F 由闭式 ‖diag(p)−pp^T‖_F²=Σp²−2Σp³+(Σp²)² 计算（J 为 softmax 的 Jacobian）。"""
    out = {}
    for d_k in d_ks:
        q = torch.randn(rows, d_k)
        K = torch.randn(rows, m, d_k)
        scores = torch.einsum("rd,rmd->rm", q, K)  # (rows,d_k)x(rows,m,d_k) -> (rows,m)
        for name, z in (("不缩放", scores), ("缩放后", scores / math.sqrt(d_k))):
            p = torch.softmax(z, dim=-1)
            s2, s3 = (p * p).sum(-1), (p ** 3).sum(-1)
            jac = (s2 - 2 * s3 + s2 * s2).clamp_min(0).sqrt()  # ‖J‖_F（clamp 防浮点下溢出负）
            out[(d_k, name)] = dict(
                pmax=p.max(-1).values.median().item(),
                H=-(p * (p + 1e-12).log()).sum(-1).median().item(),
                jac=jac.median().item())
            r = out[(d_k, name)]
            print(f"[饱和] d_k={d_k:3d} {name}  p_max中位={r['pmax']:.4f}  "
                  f"熵中位={r['H']:.3f} nat  ‖J‖_F中位={r['jac']:.4f}")
        ratio = out[(d_k, "缩放后")]["jac"] / out[(d_k, "不缩放")]["jac"]
        print(f"[梯度] d_k={d_k:3d}：Jacobian 范数中位数之比 缩放/不缩放 = {ratio:.1f} 倍")
    return out


# ---------- 5. mini 批次账本：score 矩阵字节数与 matmul FLOPs（手算口径的机器复核） ----------
def ledger(batch=64, n=30, h=8, d_k=64, sites=18):
    """mini 批次账本：打分矩阵 (batch,h,n,n) 的字节数与两处 matmul 的 FLOPs。
    输入全为标量（base 配置默认值）；返回：(元素数, 字节数)。"""
    elems = batch * h * n * n
    nbytes = elems * 4  # fp32
    flops = 2 * batch * h * n * n * d_k  # 一次 (n×d)@(d×n) 矩阵乘 = 2·n²·d FLOPs
    print(f"[账本] score 张量：{batch}×{h}×{n}×{n} = {elems:,} 个元素")
    print(f"[账本] 字节数(fp32) = {elems:,}×4 = {nbytes:,} B = {nbytes / 1024 / 1024:.2f} MiB")
    print(f"[账本] 每处注意力两个 matmul ≈ 2×{flops:,} = {2 * flops / 1e6:.1f} MFLOPs（d_k=d_v=64）")
    print(f"[账本] 整机 {sites} 处注意力：score 本体 ≈ {sites * nbytes / 1024 / 1024:.1f} MiB，"
          f"matmul ≈ {sites * 2 * flops / 1e9:.2f} GFLOPs（按源/目标句同长 30 词计）")
    return elems, nbytes


# ---------- 6. wall-clock：手写注意力前向耗时随 n 的二次增长 ----------
def wallclock(ns=(64, 128, 256, 512), b=8, h=8, d=64):
    """计时：手写 attention() 前向随 n 的二次增长。
    每档随机 q,k,v:(b,h,n,d)；返回：(ns 列表, 各档中位耗时 ms 列表, 各档 score 字节数列表)。"""
    times_ms, nbytes = [], []
    for n in ns:
        q, k, v = (torch.randn(b, h, n, d) for _ in range(3))
        inner = max(4, 4096 // n)  # 小 n 多跑几轮，摊薄派发开销
        with torch.no_grad():
            for _ in range(5):
                attention(q, k, v)  # 预热
            blocks = []
            for _ in range(5):  # 取五轮的中位数
                t0 = time.perf_counter()
                for _ in range(inner):
                    attention(q, k, v)
                blocks.append((time.perf_counter() - t0) / inner * 1e3)
        times_ms.append(sorted(blocks)[2])
        nbytes.append(b * h * n * n * 4)
        print(f"[计时] n={n:3d}  score={nbytes[-1] / 1024 / 1024:6.2f} MiB  "
              f"wall-clock={times_ms[-1]:8.3f} ms（前向中位数）")
    ratio = times_ms[-1] / times_ms[0]
    print(f"[计时] t(n={ns[-1]})/t(n={ns[0]}) = {ratio:.1f} 倍；纯二次律预期 = "
          f"{(ns[-1] / ns[0]) ** 2:.0f} 倍")
    return ns, times_ms, nbytes


# ---------- 7. 三张图 ----------
def fig_flow():
    """图 2.1：注意力计算流程框图。框内标签与 v7 Fig.2 左逐字一致（C-I6 核对），
    Q/K/V/output 标注与全部形状标注（(n,d_k) 等，按式 (2.1) 二维数学视图）为本书补充
    （原图为无文字标注的箭头）。
    画幅规格：dpi=300 全画幅输出（不用 bbox_inches='tight' 裁边），保证印刷档像素。"""
    fig, ax = plt.subplots(figsize=(5.4, 6.9))
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    labels, ys = ["MatMul", "Scale", "Mask (opt.)", "SoftMax", "MatMul"], \
        [0.13, 0.30, 0.47, 0.64, 0.81]  # 自下而上，与论文原图同向
    bw, bh, cx = 0.32, 0.085, 0.56
    for text, yc in zip(labels, ys):
        ax.add_patch(FancyBboxPatch((cx - bw / 2, yc - bh / 2), bw, bh,
                                    boxstyle="round,pad=0.008", fc="#cde2fb", alpha=0.5,
                                    ec=BLUE, lw=1.6, zorder=3))
        ax.text(cx, yc, text, ha="center", va="center", fontsize=12.5, color=INK, zorder=4)
    arrow = dict(arrowstyle="-|>", color=BLUE, lw=1.8, mutation_scale=16)
    for y_low, y_high in zip(ys[:-1], ys[1:]):  # 框间主流向箭头
        ax.annotate("", xy=(cx, y_high - bh / 2 - 0.004), xytext=(cx, y_low + bh / 2 + 0.004),
                    arrowprops=arrow)
    for x, name in ((0.485, "Q"), (0.635, "K")):  # 底部两路输入（打分用）
        ax.annotate("", xy=(x, ys[0] - bh / 2 - 0.004), xytext=(x, 0.02), arrowprops=arrow)
        ax.text(x, 0.028, name, ha="center", va="bottom", fontsize=13, style="italic", color=GRAY)
    # 形状标注（本书补充）：输入 Q/K/V 与输出，式 (2.1) 的二维数学视图
    for x, shape in ((0.485, r"$(n,\,d_k)$"), (0.635, r"$(m,\,d_k)$")):
        ax.text(x - 0.015, 0.075, shape, ha="right", va="center", fontsize=8.5, color=GRAY)
    ax.plot([0.20, 0.20], [0.05, ys[-1]], color=BLUE, lw=1.8, zorder=2)  # V 沿左侧上行
    ax.annotate("", xy=(cx - bw / 2 - 0.004, ys[-1]), xytext=(0.20, ys[-1]), arrowprops=arrow)
    ax.text(0.20, 0.028, "V", ha="center", va="bottom", fontsize=13, style="italic", color=GRAY)
    ax.text(0.185, 0.075, r"$(m,\,d_v)$", ha="right", va="center", fontsize=8.5, color=GRAY)
    # 形状标注：主流向上的两个中间量（打分矩阵与权重矩阵，形状同 (n,m)）
    for y_lo, y_hi, txt in ((ys[0], ys[1], "scores\n" + r"$(n,\,m)$"),
                            (ys[3], ys[4], "weights\n" + r"$(n,\,m)$")):
        ax.text(cx + 0.19, (y_lo + y_hi) / 2, txt, ha="left", va="center",
                fontsize=8.5, color=GRAY, linespacing=1.4)
    ax.annotate("", xy=(cx, 0.96), xytext=(cx, ys[-1] + bh / 2 + 0.004), arrowprops=arrow)
    ax.text(cx, 0.968, "output " + r"$(n,\,d_v)$", ha="center", va="bottom",
            fontsize=11, color=GRAY)
    fig.savefig(FIG_DIR / "fig-2-1-attention-flow.png")
    plt.close(fig)


def fig_hist(dots, scaled):
    """图 2.3：有/无缩放的点积分布直方图（真实数据；编号按正文出现顺序，
    二次增长图在前为图 2.2、本图在 §2.5 为图 2.3）。"""
    fig, ax = plt.subplots(figsize=(6.4, 4))
    bins = np.linspace(-30, 30, 121)
    ax.hist(scaled.numpy(), bins=bins, density=True, color=BLUE, alpha=0.65,
            label="scaled (÷ √$d_k$)", zorder=3)
    ax.hist(dots.numpy(), bins=bins, density=True, color=ORANGE, alpha=0.5,
            label="unscaled", zorder=3)
    ax.text(-28, 0.36, f"unscaled: std = {dots.std():.2f}\n(theory $\\sqrt{{d_k}}$ = 8.0)",
            fontsize=9.5, color=GRAY)
    ax.text(10.5, 0.36, f"scaled: std = {scaled.std():.2f}\n(theory 1.0)",
            fontsize=9.5, color=GRAY)
    style_ax(ax, "Dot-product distribution, $d_k$=64 (200k samples)")
    ax.set_xlabel("dot product q·k"), ax.set_ylabel("probability density")
    ax.legend(frameon=False, labelcolor=GRAY)
    ax.set_ylim(0, 0.42)
    fig.savefig(FIG_DIR / "fig-2-3-dot-product-histogram.png")
    plt.close(fig)


def fig_growth(ns, times_ms, nbytes):
    """图 2.2：score 显存（理论）与前向 wall-clock（实测）随 n 的二次增长。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    mib = [b / 1024 / 1024 for b in nbytes]
    ax1.plot(ns, mib, color=BLUE, lw=2, marker="o", ms=6, zorder=3)
    for n, v in zip(ns, mib):
        ax1.annotate(f"{v:.1f}", (n, v), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=9, color=GRAY)
    style_ax(ax1, "Score matrix size (b=8, h=8, fp32)")
    ax1.set_xlabel("sequence length n"), ax1.set_ylabel("MiB")
    ns_arr = np.asarray(ns, dtype=float)
    c = times_ms[-1] / ns_arr[-1] ** 2  # 用最大 n 定标二次参考线
    ax2.loglog(ns, times_ms, color=BLUE, lw=2, marker="o", ms=6, zorder=3, label="measured (CPU)")
    ax2.loglog(ns, c * ns_arr ** 2, color=GRAY, lw=1.5, ls="--", label="$\\propto n^2$ reference")
    ratio = times_ms[-1] / times_ms[0]
    ax2.annotate(f"t({ns[-1]})/t({ns[0]}) = {ratio:.0f}×\n(ideal {ns[-1] // ns[0]}² = "
                 f"{(ns[-1] / ns[0]) ** 2:.0f}×)", (0.97, 0.05), xycoords="axes fraction",
                 ha="right", fontsize=9.5, color=GRAY)
    style_ax(ax2, "Hand-written attention, wall-clock")
    ax2.set_xlabel("sequence length n"), ax2.set_ylabel("forward time (ms)")
    ax2.legend(frameon=False, labelcolor=GRAY)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig-2-2-quadratic-growth.png")
    plt.close(fig)


if __name__ == "__main__":
    print(f"torch {torch.__version__}")
    compare_sdpa()
    dots, scaled = exp_dot_distribution()
    exp_softmax_saturation()
    ledger()
    ns, times_ms, nbytes = wallclock()
    fig_flow(), fig_hist(dots, scaled), fig_growth(ns, times_ms, nbytes)
    print(f"[图] 三张图已写入 {FIG_DIR}")
