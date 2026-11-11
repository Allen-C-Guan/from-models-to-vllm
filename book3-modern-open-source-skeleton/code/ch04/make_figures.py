# Book3第4章：位置旋转、频谱、布局与真实实验报告的教学图，默认只画1—5，另存工作目录log。
# 图1/4是数学构造；原正文前轮已修图保留。图6不推造数据、不默默回退实验档位。
import argparse
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR=str(Path.cwd()/'log/book3-ch04/figures')

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MID = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
SEQ_LIGHT, SEQ_DARK = "#cde2fb", "#0d366b"     # 顺序型蓝色单色渐变（块深浅用）
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9.5,
    "axes.edgecolor": TICK, "axes.labelcolor": DARK, "xtick.color": TICK,
    "ytick.color": TICK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
})


def style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def inv_freqs(head_dim, theta=10000.0):
    """θ_i = theta^{-2i/d}，i=0..d/2-1（0 起口径，HF 同款）。"""
    return theta ** (-2.0 * np.arange(head_dim // 2) / head_dim)


# ---------------- 图 4.1：二维旋转直觉（同一向量按位置旋转 + 内积只依赖 m−n） ----------------
def fig_4_1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # (a) 同一单位向量在 m=0..8、三档角速度下的旋转轨迹
    th = np.linspace(0, 2 * np.pi, 256)
    ax1.plot(np.cos(th), np.sin(th), color=GRID, lw=1.0)
    z = np.exp(0.35j)                                   # 固定的「内容」：模长 1、初始辐角 0.35
    for c, w, lab in ((BLUE, 1.0, r"$\theta=1.0$"), (ORANGE, 0.5, r"$\theta=0.5$"), (TEAL, 0.25, r"$\theta=0.25$")):
        m = np.arange(9)
        pts = z * np.exp(1j * w * m)
        ax1.plot(pts.real, pts.imag, "-o", color=c, lw=1.6, ms=5, label=lab)
        ax1.annotate("", xy=(pts.real[-1], pts.imag[-1]), xytext=(pts.real[-2], pts.imag[-2]),
                     arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4))
    ax1.plot([z.real], [z.imag], "k*", ms=9)
    ax1.text(z.real + 0.06, z.imag + 0.05, "$z$ (pos 0)", fontsize=8, color=DARK)
    ax1.set_xlim(-1.25, 1.35); ax1.set_ylim(-1.3, 1.35)
    ax1.set_aspect("equal"); ax1.grid(False)
    ax1.set_title("(a) one vector, rotated by position", fontsize=9.5, color=DARK)
    ax1.set_xlabel("dim 1"); ax1.set_ylabel("dim 2")
    ax1.legend(loc="lower left", fontsize=7.5, framealpha=0.9)

    # (b) 单位向量的内积只依赖 m−n（移动整段文本曲线不动）
    d = np.arange(0, 21)
    for c, w, lab in ((BLUE, 1.0, r"$\theta=1.0$"), (ORANGE, 0.5, r"$\theta=0.5$"), (TEAL, 0.25, r"$\theta=0.25$")):
        ax2.plot(d, np.cos(w * d), "-o", color=c, lw=2, ms=4.5, label=lab)
    ax2.axhline(0, color=MID, lw=0.8)
    ax2.annotate("fixed q/k, shift both indices +1000:\ncurve unchanged (only $m{-}n$ enters)",
                 xy=(3, np.cos(3.0)), xytext=(6.5, 0.72), fontsize=7.5, color=MID,
                 arrowprops=dict(arrowstyle="->", color=MID, lw=1.0))
    ax2.set_xlabel("relative distance $m-n$"); ax2.set_ylabel(r"$\langle q_m, k_n\rangle$")
    ax2.set_title("(b) fixed q/k: explicit rotation uses $m{-}n$", fontsize=9.5, color=DARK)
    ax2.legend(fontsize=7.5)
    style_axis(ax2)

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)   # 给含 mathtext 的两行标题留高度，防顶裁
    fig.savefig(os.path.join(FIG_DIR, "fig-4-1-2d-rotation-intuition.png"))
    plt.close(fig)


# ---------------- 图 4.2：块对角旋转矩阵 + 整机动刀位置（形状流转） ----------------
def fig_4_2():
    from matplotlib.patches import FancyBboxPatch
    fig=plt.figure(figsize=(14,9),dpi=300)
    grid=fig.add_gridspec(2,2,height_ratios=[.7,1.7],width_ratios=[1,2.6])
    top=fig.add_subplot(grid[0,:]);left=fig.add_subplot(grid[1,0]);ax=fig.add_subplot(grid[1,1])
    top.set(xlim=(0,20),ylim=(0,4));top.axis('off')
    def box(a,x,y,w,h,text,color=BLUE,fs=10):
        a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.03',ec=color,fc='white',lw=1.6))
        a.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,linespacing=1.5)
    def arrow(a,x1,y1,x2,y2):
        a.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=1.4,color=MID,shrinkA=0,shrinkB=0))
    top.text(.1,3.5,'本章独立模块实验：保留LayerNorm与GELU，仅改变位置方案',fontsize=17)
    nodes=[(.1,2.1,'ID (B,n)'),(2.8,3.1,'词嵌入\n(B,n,512)'),(6.6,6.1,'6层Block：LN → 注意力 / LN → GELU\n两条残差旁路保持，输出(B,n,512)'),(13.4,2.2,'末端LN\n(B,n,512)'),(16.3,3.4,'共享词表读出\n(B,n,8192)')]
    for x,w,t in nodes:box(top,x,1.2,w,1.2,t,fs=10)
    for (x,w,_),(nx,_,_) in zip(nodes,nodes[1:]):arrow(top,x+w,1.8,nx,1.8)
    top.text(3.,.35,'入口不再相加wpe；不是将前两章实验检查点继续训练。',fontsize=11)
    left.set(xlim=(-.3,8.3),ylim=(-2,9));left.axis('off');left.set_aspect('equal')
    left.set_title('相邻坐标顺序下的块对角矩阵\n示意头维d=8',fontsize=12)
    left.add_patch(plt.Rectangle((0,0),8,8,fc='#f4f3ef',ec='#b9b7af'))
    for i,color in enumerate([SEQ_DARK,'#3564a8','#7ba3d9',SEQ_LIGHT]):
        left.add_patch(plt.Rectangle((2*i,6-2*i),2,2,fc=color,ec='white'))
        left.text(2*i+1,7-2*i,f'R(mθ{i})',ha='center',va='center',fontsize=9,color='white' if i<2 else DARK)
    left.text(4,-1,'块外为0；每块2×2\n本书两半式经坐标置换得到同一构造',ha='center',va='center',fontsize=9)
    ax.set(xlim=(0,20),ylim=(0,12));ax.axis('off');ax.set_title('注意力内部：Q/K旋转后打分，V单独进入加权和',fontsize=14)
    box(ax,5.2,10.2,8.2,1.,'归一化后的输入 (B,n,512)')
    for x,label in [(0.4,'Q'),(6.4,'K'),(13.3,'V')]:
        box(ax,x,7.9,4.9,1.15,label+'投影并拆8个头\n(B,8,n,64)')
        arrow(ax,9.3,10.2,x+2.45,9.05)
    for x in (.4,6.4):
        box(ax,x,5.9,4.9,1.05,'按自身位置旋转\n(B,8,n,64)',ORANGE)
        arrow(ax,x+2.45,7.9,x+2.45,6.95)
    box(ax,1.7,3.3,8.3,1.35,'Q′K′ᵀ / √64 → 因果mask → softmax\n分数及权重 (B,8,n,n)',fs=10)
    arrow(ax,2.85,5.9,4.3,4.65);arrow(ax,8.85,5.9,7.2,4.65)
    box(ax,12.0,3.3,7.6,1.35,'权重 @ V\n(B,8,n,n) × (B,8,n,64)\n→ (B,8,n,64)',fs=10)
    arrow(ax,10.,3.975,12.,3.975);arrow(ax,15.75,7.9,15.75,4.65)
    ax.text(16.15,6.1,'V不旋转',fontsize=10)
    box(ax,10.5,.65,9.1,1.3,'拼接头 → 输出投影\n(B,n,512) → 加回残差输入',fs=11)
    arrow(ax,15.8,3.3,15.05,1.95)
    fig.text(.04,.012,'据随书GPT2RoPE与式(4.6)/(4.8)自绘，示意级。掩码的因果规则保持，存储按当前序列长度生成。',fontsize=11)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(os.path.join(FIG_DIR,'fig-4-2-block-diagonal-and-machine.png'));plt.close(fig)


# ---------------- 图 4.3：波长谱（d=128 档逐对波长；快慢指针） ----------------
def fig_4_3():
    d = 128
    lam = 2 * np.pi / inv_freqs(d)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    i = np.arange(d // 2)
    ax.plot(i, lam, "-o", color=BLUE, lw=2, ms=3.5, label=r"$\lambda_i = 2\pi/\theta_i$  ($d_k$=128)")
    ax.set_yscale("log")
    for y, lab, c in ((512, "module tier $T$=512 (this book)", MID), (2048, "LLaMA1 train window 2048", TEAL)):
        ax.axhline(y, color=c, lw=1.1, ls=(0, (4, 2)))
        ax.text(1.0, y * 1.12, lab, fontsize=7.5, color=c)
    for xi, yi, txt in ((0, lam[0], "$\\lambda_0$=6.3"), (32, lam[32], "$\\lambda_{32}$=628"),
                        (63, lam[63], "$\\lambda_{63}$=54,410")):
        ax.annotate(txt, xy=(xi, yi), xytext=(xi - 1.5, yi * 1.55), fontsize=7.5, color=DARK,
                    arrowprops=dict(arrowstyle="->", color=MID, lw=0.9))
    ax.text(33, 9.2, "fast hands:\nmany cycles inside\nthe train window", fontsize=7.5, color=MID)
    ax.text(14, 2.6e4, "slow frequencies: fewer cycles\ninside the shown windows", fontsize=7.5, color=MID)
    ax.set_xlabel("2-D block index $i$ (0 = fastest)")
    ax.set_ylabel("wavelength $\\lambda_i$ (tokens per cycle)")
    ax.legend(loc="lower right", fontsize=7.5)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-4-3-wavelength-spectrum.png"))
    plt.close(fig)


# ---------------- 图 4.4：长程衰减上界（论文式 37 的数值化，d=64/128 双臂） ----------------
def decay_curve(d, dists, theta=10000.0):
    """(2/d) * sum_j |S_j|，S_j = sum_{i<j} exp(i*delta*theta_i)——论文 Figure 2 的复算。"""
    th = inv_freqs(d, theta)
    out = []
    for delta in dists:
        phasors = np.exp(1j * delta * th)              # (d/2,)
        S = np.cumsum(phasors)                         # S_j, j=1..d/2
        out.append(np.abs(S).sum() / (d / 2))
    return np.array(out)


def fig_4_4():
    dists = np.arange(0, 251)
    y64, y128 = decay_curve(64, dists), decay_curve(128, dists)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(dists, y128, color=BLUE, lw=2, label="$d$=128 (LLaMA head dim)")
    ax.plot(dists, y64, color=ORANGE, lw=2, label="$d$=64 (book tiers)")
    i50, i100 = 50, 100
    ax.plot([50], [y64[i50]], "o", color=ORANGE, ms=6)
    ax.plot([100], [y64[i100]], "o", color=ORANGE, ms=6)
    ax.annotate(f"non-monotonic: {y64[i100]:.1f} at dist 100\n> {y64[i50]:.1f} at dist 50",
                xy=(100, y64[i100]), xytext=(128, y64[i100] + 3.2), fontsize=7.5, color=MID,
                arrowprops=dict(arrowstyle="->", color=MID, lw=0.9))
    ax.annotate(f"32.5 $\\to$ 6.5", xy=(250, y128[-1]), xytext=(196, 14.5), fontsize=8, color=DARK,
                arrowprops=dict(arrowstyle="->", color=MID, lw=0.9))
    ax.set_xlabel("relative distance $m-n$")
    ax.set_ylabel(r"phase-sum factor  $\frac{2}{d}\sum_j |S_j|$")
    ax.set_ylim(0, 34)
    ax.legend(fontsize=7.5)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-4-4-decay-upper-bound.png"))
    plt.close(fig)
    print(f"  [fig 4.4] d=128: {y128[0]:.1f} (dist 0) -> {y128[-1]:.1f} (dist 250)；"
          f"d=64: {y64[0]:.1f} -> {y64[-1]:.1f}；d=64 dist50={y64[i50]:.2f} dist100={y64[i100]:.2f}（回勾）")


# ---------------- 图 4.5：实现三型对照（相邻配对复数/相邻配对实数/两半式 rotate_half） ----------------
def fig_4_5():
    fig,axes=plt.subplots(3,1,figsize=(9,7),dpi=300)
    titles=['(a) Complex adjacent pairs: (0,1), (2,3), (4,5), (6,7)',
            '(b) Real adjacent pairs: the SAME coordinate layout as (a)',
            '(c) Half-split pairs: (0,4), (1,5), (2,6), (3,7)']
    for row,ax in enumerate(axes):
        ax.set(xlim=(-.3,8.1),ylim=(0,3.6));ax.axis('off');ax.set_title(titles[row],loc='left',fontsize=11)
        for j in range(8):
            ax.add_patch(plt.Rectangle((j,.45),.8,.8,fc='#f4f3ef',ec=MID))
            ax.text(j+.4,.85,f'$x_{j}$',ha='center',va='center',fontsize=12)
        pairs=[(0,1),(2,3),(4,5),(6,7)] if row<2 else [(0,4),(1,5),(2,6),(3,7)]
        for i,(a,b) in enumerate(pairs):
            height=1.6 if row<2 else 1.6+.45*i
            ax.plot([a+.4,a+.4,b+.4,b+.4],[1.25,height,height,1.25],color=BLUE if row<2 else ORANGE,lw=1.6)
        note='Same pairing and frequency order: no Q/K row permutation needed.' if row<2 else 'Half → adjacent: [0,4,1,5,2,6,3,7]; inverse: [0,2,4,6,1,3,5,7].'
        ax.text(0,.05,note,fontsize=10,color=MID)
    fig.text(.5,.015,'Adjacent ↔ half-split: reorder Q/K output rows within each head; V rows stay unchanged.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(os.path.join(FIG_DIR,'fig-4-5-three-impl-styles.png'));plt.close(fig)


# ---------------- 图 4.6：换件消融曲线 + 4× 外推分桶（依赖 ablation_pe 产物） ----------------
def load_report(path,historical=False):
    report=json.loads(path.read_text());arms=report.get('arms',{})
    for name in ('wpe','rope'):
        row=arms.get(name,{})
        if row.get('completed') is False:raise ValueError('明确未完成的报告不画图')
        if row.get('completed') is not True and not historical:raise ValueError('旧报告须显式--historical')
        n=row.get('steps_done',0)
        if n<=0 or n!=row.get('steps_planned') or row.get('diverged'):raise ValueError('训练未完整/已失败')
        if not row.get('evals') or row['evals'][-1]['step']!=n:raise ValueError('缺最终评估')
        if not math.isfinite(row['eval_final']):raise ValueError('非有限评估')
        if any(not math.isfinite(e['eval_loss']) for e in row['evals']):raise ValueError('非有限曲线')
    extra=arms['rope'].get('extrap_2048',{})
    if not extra.get('buckets') or not all(math.isfinite(v) for v in extra['buckets'].values()):raise ValueError('缺有效分桶')
    if not math.isfinite(extra.get('loss',float('nan'))):raise ValueError('缺外推损失')
    if not arms['wpe'].get('wall_2048',{}).get('wpe_lookup_error'):raise ValueError('缺位置表边界记录')
    return report


def fig_4_6(rep):
    tag=rep.get('meta',{}).get('out_name','supplied run')
    wpe, rope = rep["arms"]["wpe"], rep["arms"]["rope"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    for arm, c, lab in ((wpe, BLUE, "wpe (learned, 1024-table)"), (rope, ORANGE, "RoPE (no table)")):
        steps = [e["step"] for e in arm["evals"]]
        losses = [e["eval_loss"] for e in arm["evals"]]
        ses = [e["eval_se"] for e in arm["evals"]]
        ax1.errorbar(steps, losses, yerr=ses, fmt="-o", color=c, lw=2, ms=4, capsize=2.5, label=lab)
    ax1.set_xlabel("step"); ax1.set_ylabel("fixed-window eval loss (512, fp32)")
    ax1.set_title(f"(a) swap ablation, {tag} tier (T=512)", fontsize=9, color=DARK)
    ax1.legend(fontsize=7.5)
    style_axis(ax1)

    buckets = rope["extrap_2048"]["buckets"]
    labels = ["[0,512)", "[512,1024)", "[1024,1536)", "[1536,2048)"]
    vals = [buckets[f"pos[{a}:{b}]"] for a, b in ((0, 512), (512, 1024), (1024, 1536), (1536, 2048))]
    bars = ax2.bar(range(4), vals, color=ORANGE, width=0.62)
    bars[0].set_color("#f0a583")
    ax2.axhline(rope["eval_final"], color=MID, lw=1.2, ls=(0, (4, 2)))
    ax2.text(2.6, rope["eval_final"] - 0.028, f"in-window eval {rope['eval_final']:.4f}",
             fontsize=7.5, color=MID, ha="center")
    ax2.text(1.5, max(vals) + 0.012, f"4x extrapolation loss {rope['extrap_2048']['loss']:.4f}"
             f"  (ppl {rope['extrap_2048']['ppl']:.1f})", fontsize=7.5, color=DARK, ha="center")
    ax2.text(1.5, min(vals) - 0.055, "wpe: position-table boundary reached",
             fontsize=7.5, color=BLUE, ha="center")
    ax2.set_xticks(range(4)); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylim(min(vals) - 0.08, max(vals) + 0.05)
    ax2.set_xlabel("position bucket (first bucket: in training position range)")
    ax2.set_ylabel("bucket eval loss (fp32)")
    ax2.set_title("(b) RoPE 4x extrapolation buckets", fontsize=9, color=DARK)
    style_axis(ax2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-4-6-swap-ablation-extrapolation.png"))
    plt.close(fig)


def main():
    global FIG_DIR
    ap=argparse.ArgumentParser(description='数学/结构图另存log；图6须真实完整实验报告')
    ap.add_argument('--only',default='1,2,3,4,5');ap.add_argument('--report',type=Path)
    ap.add_argument('--historical',action='store_true');ap.add_argument('--out-dir',type=Path,default=Path(FIG_DIR));a=ap.parse_args()
    todo=a.only.split(',')
    if not todo or len(set(todo))!=len(todo) or any(v not in {'1','2','3','4','5','6'} for v in todo):raise ValueError('无效图号')
    names=['2d-rotation-intuition','block-diagonal-and-machine','wavelength-spectrum','decay-upper-bound','three-impl-styles','swap-ablation-extrapolation']
    for n in todo:
        if (a.out_dir/f'fig-4-{n}-{names[int(n)-1]}.png').exists():raise FileExistsError('不覆盖已有图')
    report=None
    if '6' in todo:
        if a.report is None:raise ValueError('图6须显式--report，不自动从full降为fast')
        report=load_report(a.report,a.historical)
        config=report['arms']['rope'].get('config',{})
        if config.get('T')!=512 or report['arms']['wpe'].get('config',{}).get('n_positions')!=1024 or set(report['arms']['rope']['extrap_2048']['buckets'])!={'pos[0:512]','pos[512:1024]','pos[1024:1536]','pos[1536:2048]'}:
            raise ValueError('本图6模板只支持原512/2048几何；不同实验请先修改图注与布局')
    FIG_DIR=str(a.out_dir);a.out_dir.mkdir(parents=True,exist_ok=True)
    for n in todo:
        if n=='6':fig_4_6(report)
        else:globals()['fig_4_'+n]()
    print('已另存图：',FIG_DIR)


if __name__=='__main__':main()
