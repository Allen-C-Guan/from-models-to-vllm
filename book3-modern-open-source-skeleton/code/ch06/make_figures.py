# Book3第6章：机制/公式图与真实消融报告绘图；不编造缺失曲线。
# python make_figures.py --only 1,2,3 --out-dir log/book3-ch06/figures
import argparse
import json
import os
from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR=str(Path.cwd()/'log/book3-ch06/figures')
REPORT=None
HISTORICAL=False

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MID = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9.5,
    "axes.edgecolor": TICK, "axes.labelcolor": DARK, "xtick.color": TICK,
    "ytick.color": TICK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "font.family": "sans-serif",
})

KIB, MIB, GIB = 2**10, 2**20, 2**30


def style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------- 图 6.1：KV cache 数据流与形状（两阶段：一次读入 / 逐 token 生成） ----------------
def fig_6_1():
    fig,axes=plt.subplots(1,2,figsize=(8,4.8))
    for ax in axes:ax.set_xlim(0,10.8);ax.set_ylim(0,12);ax.axis('off');ax.grid(False)
    def box(ax,x,y,w,h,text,color=MID):
        ax.add_patch(plt.Rectangle((x,y),w,h,fc='white',ec=color,lw=1.3,zorder=2))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8,color=DARK,zorder=3)
    def arrow(ax,a,b,color=MID):
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',color=color,lw=1.2))
    left,right=axes
    left.set_title('(a) Prompt: cache writes in one layer',fontsize=10)
    box(left,1,9.8,8,1.1,'layer input hidden states  (B,P,d)')
    box(left,.5,7.3,4.2,1.4,'K projection + RoPE\n(B,$h_{kv}$,P,$d_k$)',ORANGE)
    box(left,5.3,7.3,4.2,1.4,'V projection\n(B,$h_{kv}$,P,$d_k$)',ORANGE)
    arrow(left,(3,9.8),(2.6,8.7));arrow(left,(7,9.8),(7.4,8.7))
    box(left,.5,4.8,9,1.4,'cache: rotated K, unrotated V\neach (B,$h_{kv}$,P,$d_k$)',ORANGE)
    arrow(left,(2.6,7.3),(2.6,6.2),ORANGE);arrow(left,(7.4,7.3),(7.4,6.2),ORANGE)
    left.text(5,3.6,'One independent K/V pair per layer.\nPrompt attention and FFN still run;\nonly cache writes are shown here.',ha='center',va='center',fontsize=8,color=MID)
    left.text(5,1.2,'Reuse requires fixed parameters,\nhistory and position rules.',ha='center',fontsize=8,color=MID)
    right.set_title('(b) Append one token at position P',fontsize=10)
    box(right,1,9.8,8,1.1,'new layer input  (B,1,d)')
    box(right,.1,7.4,3.8,1.4,'Q projection + RoPE\n(B,h,1,$d_k$)',BLUE)
    box(right,4.6,7.4,5.3,1.4,'K projection + RoPE; V projection\nappend each (B,$h_{kv}$,1,$d_k$)',ORANGE)
    arrow(right,(3,9.8),(2,8.8));arrow(right,(7,9.8),(7.2,8.8))
    box(right,4.6,5.2,5.3,1.2,'updated K/V cache\neach (B,$h_{kv}$,P+1,$d_k$)',ORANGE)
    arrow(right,(7.2,7.4),(7.2,6.4),ORANGE)
    box(right,.1,3,9.8,1.3,'grouped QK scores + softmax: (B,h,1,P+1)\nall valid keys visible for this one query',BLUE)
    arrow(right,(2,7.4),(2,4.3),BLUE);arrow(right,(7.2,5.2),(7.2,4.3),ORANGE)
    box(right,.1,.6,9.8,1.3,'weighted V: (B,h,1,$d_k$)\nconcat + output projection: (B,1,d)',BLUE)
    arrow(right,(5,3),(5,1.9),BLUE)
    right.text(7.45,4.7,'K',fontsize=8,color=MID)
    right.plot([9.9,10.3,10.3],[5.8,5.8,1.25],color=ORANGE,lw=1.2)
    arrow(right,(10.3,1.25),(9.9,1.25),ORANGE)
    right.text(10.4,4.6,'V',fontsize=8,color=MID)
    fig.tight_layout();fig.savefig(os.path.join(FIG_DIR,'fig-6-1-kvcache-dataflow.png'),facecolor='white');plt.close(fig)


# ---------------- 图 6.2：剪刀差（权重恒定线 vs KV 线性线；B=1 / B=8 两族） ----------------
def fig_6_2():
    n_max = 8192
    ns = np.linspace(0, n_max, 200)
    W7B_BYTES = 6_738_415_616 * 2                      # 7B 档权重 fp16 字节（12.55 GiB；canonical 手算参数 6,738,415,616
                                                       #  = ch5 式 (5.2) 与表 5.1 同源，正文按 12.55 GiB 口径）
    series = [  # (label, weights bytes, KV bytes/token, color)
        ("GPT-2 124M (MHA)", 124_439_808 * 2, 2 * 12 * 12 * 64 * 2, BLUE),
        ("7B (MHA)", W7B_BYTES, 2 * 32 * 32 * 128 * 2, ORANGE),
        ("70B (GQA-8)", 68_976_648_192 * 2, 2 * 80 * 8 * 128 * 2, TEAL),
        ("70B MHA (counterfactual)", 68_976_648_192 * 2, 2 * 80 * 64 * 128 * 2, YELLOW),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    for ax, batch, tag in [(axes[0], 1, "B = 1"), (axes[1], 8, "B = 8")]:
        for label, w_bytes, bpt, color in series:
            kv = ns * bpt * batch / GIB
            ax.plot(ns, kv, color=color, lw=2, zorder=3)
            ax.axhline(w_bytes / GIB, color=color, lw=1.4, ls="--", zorder=2)
        ax.set_xlim(0, n_max)
        ax.set_xlabel("valid positions per sequence n")
        ax.set_ylabel("memory (GiB)")
        ax.set_title(tag+"  (total batch KV)", fontsize=10)
        style_axis(ax)
    # 7B B=8 追平点：权重精确字节 / (512 KiB/token x 8) —— {本书复算} n=3,213
    n_cross = W7B_BYTES / (2 * 32 * 32 * 128 * 2 * 8)
    axes[1].plot([n_cross], [W7B_BYTES / GIB], marker="o", ms=7, color=ORANGE, zorder=5)
    axes[1].annotate(f"batch KV = model weights\nat n = {n_cross:,.0f}",
                     xy=(n_cross, W7B_BYTES / GIB), xytext=(2900, 25), fontsize=7.6,
                     color=DARK, bbox=dict(facecolor="white",edgecolor="none",pad=1), arrowprops=dict(arrowstyle="-|>", color=MID, lw=1.0))
    axes[0].text(250, 19.5, "70B weights: 128.5 GiB (off scale)", fontsize=7.4, color=MID)
    axes[0].text(250, 1.8, "GPT-2 weights: 0.23 GiB", fontsize=7.4, color=MID,bbox=dict(facecolor="white",edgecolor="none",pad=1))
    axes[0].set_ylim(0, 21)
    axes[1].set_ylim(0, 42)
    from matplotlib.lines import Line2D
    handles = ([Line2D([0], [0], color=c, lw=2) for _, _, _, c in series]
               + [Line2D([0], [0], color=MID, lw=1.4, ls="--")])
    labels = [s[0] + " KV cache" for s in series] + ["weights (fp16, constant)"]
    fig.legend(handles, labels, fontsize=7.0, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=1.2)
    fig.subplots_adjust(bottom=0.3, wspace=0.25)
    fig.savefig(os.path.join(FIG_DIR, "fig-6-2-scissors-gap.png"), facecolor="white")
    plt.close(fig)


# ---------------- 图 6.3：MHA / GQA / MQA 头分组示意（h 与 h_kv 双标） ----------------
def fig_6_3():
    h = 8
    panels = [("MHA  ($h_{kv}=h=8$)", 8, "12 KiB / token"),
              ("GQA-4  ($h_{kv}=4$)", 4, "6 KiB / token"),
              ("MQA  ($h_{kv}=1$)", 1, "1.5 KiB / token")]
    fig, axes = plt.subplots(1, 3, figsize=(8, 3.9))
    for ax, (title, h_kv, kv_note) in zip(axes, panels):
        ax.set_xlim(-0.6, 7.6)
        ax.set_ylim(-2.6, 3.4)
        ax.axis("off")
        ax.grid(False)
        ax.set_title(title, fontsize=9.5)
        g = h // h_kv
        xs = np.arange(h)
        for i, xq in enumerate(xs):                      # Q 头（上行，蓝）
            ax.add_patch(plt.Rectangle((xq - 0.32, 2.0), 0.64, 0.8, fc="white", ec=BLUE, lw=1.6))
            ax.text(xq, 2.4, str(i), ha="center", va="center", fontsize=7.5, color=DARK)
        for j in range(h_kv):                            # KV 头（下行，橙）
            xc = (j + 0.5) * g - 0.5                     # 组内连号：Q i 配 KV i//g
            ax.add_patch(plt.Rectangle((xc - 0.36, -0.6), 0.72, 0.8, fc="white", ec=ORANGE, lw=1.8))
            ax.text(xc, -0.2, str(j), ha="center", va="center", fontsize=7.5, color=DARK)
            for r in range(g):                           # 连线：组内 g 个 Q 头
                xq = j * g + r
                ax.plot([xq, xc], [2.0, 0.2], color=MID, lw=0.7, zorder=1)
        ax.text(3.5, 3.15, f"query heads  h = {h}   Q (B,h,n,$d_k$)",
                ha="center", fontsize=7.8, color=DARK)
        ax.text(3.5, -1.35, f"KV heads  $h_{{kv}}$ = {h_kv}   K,V (B,$h_{{kv}}$,n,$d_k$)",
                ha="center", fontsize=7.8, color=DARK)
        ax.text(3.5, -2.15, f"module tier (L=6, $d_k$=64, fp16):\n{kv_note}",
                ha="center", fontsize=7.2, color=MID)
        if g > 1:
            ax.text(3.5, 1.05, f"group size g = h/$h_{{kv}}$ = {g}", ha="center",
                    fontsize=7.2, color=MID,bbox=dict(facecolor="white",edgecolor="none",pad=1),zorder=4)
        else:
            ax.text(3.5, 1.05, "one KV per query head", ha="center", fontsize=7.2, color=MID,bbox=dict(facecolor="white",edgecolor="none",pad=1),zorder=4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-6-3-mha-gqa-mqa-groups.png"), facecolor="white")
    plt.close(fig)


# ---------------- 图 6.4：组数扫描消融（eval 曲线 + 质量-KV 反向读数） ----------------
def load_report(path,historical=False):
    report=json.loads(Path(path).read_text());arms=report.get('arms',{})
    for name in ('mha8','gqa4','mqa1'):
        a=arms.get(name,{})
        if a.get('completed') is False or a.get('diverged') or a.get('failure'):
            raise ValueError('拒绝未完成/失败实验臂')
        if not historical and a.get('completed') is not True:raise ValueError('旧报告须显式--historical')
        done,planned=a.get('steps_done'),a.get('steps_planned')
        if not isinstance(done,int) or done<=0 or done!=planned:raise ValueError('步数不完整')
        ev=a.get('evals',[])
        if not ev or ev[-1]['step']!=done or any(x['step']>=y['step'] for x,y in zip(ev,ev[1:])):raise ValueError('评估序列不完整/非递增')
        if not math.isfinite(a['eval_final']) or a['kv_per_token_bytes']<=0:raise ValueError('无效终点评估/字节')
        if not math.isclose(a['eval_final'],ev[-1]['eval_loss'],abs_tol=1e-6):raise ValueError('终点与曲线不符')
        for point in ev:
            if not math.isfinite(point['eval_loss']):raise ValueError('非有限评估')
            se=point.get('eval_se')
            if se is not None and (not math.isfinite(se) or se<0):raise ValueError('无效SE')
    if len({arms[k]['steps_planned'] for k in arms})!=1:raise ValueError('三臂预算不一致')
    if not historical:
        identities=[{k:v for k,v in arms[n]['run_identity'].items() if k!='arm'} for n in ('mha8','gqa4','mqa1')]
        if any(i!=identities[0] for i in identities):raise ValueError('三臂身份不一致')
    return report


def fig_6_4():
    if REPORT is None:raise ValueError('图6.4必须显式传入--report；没有原始报告就保留历史图')
    rep=load_report(REPORT,HISTORICAL)
    tier=str(rep['arms']['mha8']['steps_planned'])+' steps'
    arms = [("mha8", "MHA ($h_{kv}$=8)", BLUE), ("gqa4", "GQA-4 ($h_{kv}$=4)", TEAL),
            ("mqa1", "MQA ($h_{kv}$=1)", ORANGE)]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    ax = axes[0]
    for key, label, color in arms:
        ev = rep["arms"][key]["evals"]
        xs = [e["step"] for e in ev]
        ys = [e["eval_loss"] for e in ev]
        values=[e.get("eval_se") for e in ev]
        es=values if all(v is not None for v in values) else None
        ax.errorbar(xs, ys, yerr=es, color=color, lw=2, ms=5, marker="o",
                    capsize=2, elinewidth=0.8, label=label, zorder=3)
    ax.set_xlabel("training step")
    ax.set_ylabel("fixed-window eval loss (fp32)")
    ax.set_title(f"(a) same budget, {tier} tier", fontsize=9.5)
    style_axis(ax)
    ax.legend(fontsize=7.6, loc="upper right")

    ax = axes[1]
    xs, ys, colors, labels = [], [], [], []
    for key, label, color in arms:
        xs.append(rep["arms"][key]["kv_per_token_bytes"] / KIB)
        ys.append(rep["arms"][key]["eval_final"])
        colors.append(color)
        labels.append(label.split(" (")[0])
    ax.plot(xs, ys, color=MID, lw=1.2, ls=":", zorder=2)
    for x, y, c, lab in zip(xs, ys, colors, labels):
        ax.plot([x], [y], marker="o", ms=8, color=c, zorder=4)
        ax.annotate(f"{lab}\n{y:.4f}", xy=(x, y), xytext=(x + 0.7, y),
                    fontsize=7.8, color=DARK, va="center")
    ax.set_xlabel("KV cache bytes per token (KiB, module tier)")
    ax.set_ylabel("final eval loss (lower = better)")
    ax.set_title(f"(b) quality vs KV ledger ({tier} tier)", fontsize=9.5)
    ax.set_xlim(0, 16.5)
    style_axis(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-6-4-gqa-ablation.png"), facecolor="white")
    plt.close(fig)
    print(f"[fig 6.4] 用 {tier} 档数据绘制完成")


def main():
    global FIG_DIR,REPORT,HISTORICAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="1,2,3,4", help="只画部分图（如 1,2,3——full 未跑完时）")
    ap.add_argument("--out-dir",type=Path,default=Path(FIG_DIR))
    ap.add_argument("--report",type=Path)
    ap.add_argument("--historical",action="store_true")
    args = ap.parse_args()
    FIG_DIR=str(args.out_dir);REPORT=args.report;HISTORICAL=args.historical
    os.makedirs(FIG_DIR,exist_ok=True)
    todo = {s.strip() for s in args.only.split(",")}
    if not todo or not todo<=set("1234"):raise ValueError("only可选1,2,3,4")
    if "4" in todo:load_report(REPORT,HISTORICAL) if REPORT else (_ for _ in ()).throw(ValueError("缺少--report"))
    for k in todo:
        target=list(Path(FIG_DIR).glob(f"fig-6-{k}-*.png"))
        if target:raise FileExistsError("目标图已存在，选择新的输出目录")
    if "1" in todo:
        fig_6_1()
        print("[fig 6.1] done")
    if "2" in todo:
        fig_6_2()
        print("[fig 6.2] done")
    if "3" in todo:
        fig_6_3()
        print("[fig 6.3] done")
    if "4" in todo:
        fig_6_4()


if __name__=="__main__":main()
