# fig_ch04.py —— Book4第4章图4.1、4.2、4.3的教学绘图源码
# 运行：工作区根目录 source ./env.sh 后运行本文件 --which machine|summary|stats。
# machine绘整机示意；summary仅使用正文历史汇总；stats须有真实路由JSON。
# 默认输出log/book4-ch04/figures；可用--out-dir另选目录，不覆盖已有图。
# 旧图4.2保留原始历史图，本脚本修正纵轴裁切后用于新数据的绘图。
# 图内英文、图注中文，白底300dpi。
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.getcwd()
FIG_DIR = os.path.join(REPO_ROOT, "log", "book4-ch04", "figures")
LOG_DIR = os.path.join(REPO_ROOT, "log", "book4-ch04")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e1e0d9"
GREY = "#898781"
HEAT_LO, HEAT_HI = "#cde2fb", "#0d366b"      # 顺序型蓝渐变端点


def box(ax, x, y, w, h, label, ec=BLUE, fc="white", fs=8.0, lw=1.5, ls="-", tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=ec, fc=fc,
                                lw=lw, ls=ls, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc, zorder=4)
    return x + w / 2, y + h / 2


def arrow(ax, x0, y0, x1, y1, label=None, color=MUTED, fs=7.0, dx=0.0, dy=0.06, lw=1.3,
          style="italic"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=lw, zorder=2))
    if label:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, label, ha="center", va="bottom",
                fontsize=fs, color=MUTED, style=style, zorder=4)


def draw_machine(out):
    fig,(ax,bx)=plt.subplots(1,2,figsize=(8,4),gridspec_kw={"width_ratios":[1,1.45]})
    for a in (ax,bx):
        a.set_xlim(0,10);a.set_ylim(0,10);a.axis("off")
    blue,orange="#2a78d6","#eb6834"
    def box(a,x,y,w,h,label,color=blue,size=8):
        a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08",
                    facecolor="white",edgecolor=color,linewidth=1.5))
        a.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=size,color="#0b0b0b")
    def arr(a,x,y,xx,yy):
        a.annotate("",(xx,yy),(x,y),arrowprops=dict(arrowstyle="->",color="#52514e",lw=1.1))
    ax.set_title("Mixtral 8x7B",fontsize=10)
    labels=[("Token IDs  (B, n)",9),("Embedding  (B, n, 4096)",7.4),
            ("32 Blocks  (B, n, 4096)",5.8),("RMSNorm  (B, n, 4096)",4.2),
            ("Untied LM head",2.6),("Logits  (B, n, 32000)",1)]
    for i,(label,y) in enumerate(labels):
        box(ax,.6,y-.3,8.8,.9,label)
        if i:arr(ax,5,labels[i-1][1]-.3,5,y+.6)
    bx.set_title("One Block: two residual additions",fontsize=10)
    for label,y,color in [("x  (B, n, 4096)",9.3,blue),("RMSNorm",8.2,blue),
             ("GQA: 32 Q / 8 KV, head dim 128",6.9,blue),
             ("+  (B, n, 4096)",5.6,blue),("RMSNorm",4.3,blue),
             ("MoE: 8 experts / top-2, no shared\nSwiGLU width 14336",2.8,orange),
             ("+  (B, n, 4096)",1.1,blue)]:
        box(bx,1.8,y-.25,7.8,.8 if y!=2.8 else 1.05,label,color,7.8)
    for y1,y2 in [(9.05,8.75),(7.95,7.45),(6.65,6.15),(5.35,4.85),(4.05,3.6),(2.55,1.65)]:
        arr(bx,5.7,y1,5.7,y2)
    bx.plot([1.8,.6,.6],[9.4,9.4,5.75],c="#52514e",lw=1.1)
    arr(bx,.6,5.75,1.8,5.75)
    bx.plot([5.7,1.1,1.1],[5.3,5.3,1.25],c="#52514e",lw=1.1)
    arr(bx,1.1,1.25,1.8,1.25)
    fig.text(.5,.01,"d=4096, L=32, V=32000; RoPE theta=1e6; both sublayers return (B,n,4096)",
             ha="center",fontsize=7)
    fig.tight_layout(rect=(0,.04,1,1))
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise FileExistsError(out)
    fig.savefig(out,dpi=300,facecolor="white");plt.close(fig)

def fig_machine():
    draw_machine(Path(FIG_DIR)/"fig-4-1-mixtral-machine.png")


def fig_stats(src):
    """图 4.2：自家版路由统计三图（喂 route_analysis_{src}.json）。"""
    path = os.path.join(LOG_DIR, f"route_analysis_{src}.json")
    with open(path, encoding="utf-8") as f:
        rep = json.load(f)
    fin, init = rep["final"], rep.get("init") or {}
    L = len(fin["repetition_per_layer"])
    base = fin["baseline_uniform"]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.5), dpi=300)

    # (a) 分工矩阵热图（顺序蓝渐变，行归一化）
    ax = axes[0]
    frac = np.array(fin["assignment_matrix"]["frac"], dtype=float)
    names = fin["assignment_matrix"]["type_names"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seqblue", [HEAT_LO, HEAT_HI])
    im = ax.imshow(frac, cmap=cmap, vmin=0.0, aspect="auto",
                   vmax=max(0.16, float(np.nanmax(frac)) * 1.05))
    for i in range(frac.shape[0]):
        for j in range(frac.shape[1]):
            ax.text(j, i, (f"{frac[i, j]:.2f}" if np.isfinite(frac[i,j]) else "n/a"), ha="center", va="center", fontsize=6.2,
                    color="white" if frac[i, j] > float(np.nanmax(frac)) * 0.62 else INK)
    ax.set_xticks(range(frac.shape[1]), [f"e{j}" for j in range(frac.shape[1])])
    ax.set_yticks(range(frac.shape[0]), names)
    ax.tick_params(colors=GREY, labelsize=7)
    ax.set_xlabel("expert (first choice)", fontsize=8, color=MUTED)
    ax.set_title("(a) type × expert index\npooled across layers (descriptive)", fontsize=8, color=INK)

    # (b) 位置局部性：逐层连续 token 首选项重复率
    ax = axes[1]
    xs = list(range(L))
    ax.axhline(base, color=ORANGE, lw=1.6, ls="--", label=f"uniform baseline 1/E = {base:.3f}")
    if init:
        ax.plot(xs, init["repetition_per_layer"], color=GREY, lw=1.6, ls=":", marker="o", ms=6,
                label="init (before training)")
    ax.plot(xs, fin["repetition_per_layer"], color=BLUE, lw=2, marker="o", ms=6,
            label=f"final ({rep['steps']} steps)")
    for x, y in zip(xs, fin["repetition_per_layer"]):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=6.4, color=INK)
    ax.set_xlabel("layer", fontsize=8, color=MUTED)
    ax.set_ylabel("consecutive-token repetition", fontsize=8, color=MUTED)
    ax.set_title("(b) positional locality\n(Mixtral Table-5 metric)", fontsize=8, color=INK)
    ax.legend(fontsize=6.2, frameon=False, loc="lower right")

    # (c) 层间重复率（本书补充口径）
    ax = axes[2]
    xs2 = [l + 0.5 for l in range(L - 1)]
    ax.axhline(base, color=ORANGE, lw=1.6, ls="--", label=f"uniform independent ref. = {base:.3f}")
    if init:
        ax.plot(xs2, init["cross_layer_repetition"], color=GREY, lw=1.6, ls=":", marker="s",
                ms=6, label="init")
    ax.plot(xs2, fin["cross_layer_repetition"], color=TEAL, lw=2, marker="s", ms=6,
            label=f"final ({rep['steps']} steps)")
    for x, y in zip(xs2, fin["cross_layer_repetition"]):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=6.4, color=INK)
    ax.set_xlabel("layer boundary (l → l+1)", fontsize=8, color=MUTED)
    ax.set_ylabel("cross-layer repetition", fontsize=8, color=MUTED)
    ax.set_title("(c) same index across layers\n(not semantic correspondence)", fontsize=8,
                 color=INK)
    ax.legend(fontsize=6.2, frameon=False, loc="lower right")

    if "baseline_independent_per_layer" in fin:
        axes[1].plot(range(L),fin["baseline_independent_per_layer"],ls="--",color=MUTED,
                     label="independent marginals")
        axes[1].legend(fontsize=6.2,frameon=False)
    for i, ax in enumerate(axes[1:]):
        ax.tick_params(colors=GREY, labelsize=7)
        ax.grid(True, color=GRIDC, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GREY)
        key="repetition_per_layer" if i==0 else "cross_layer_repetition"
        values=fin.get(key,[])+init.get(key,[])
        ax.set_ylim(0,min(1.02,max(.30,max(values,default=0)*1.12)))
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-4-2-routing-stats.png")
    if os.path.exists(out): raise FileExistsError(out)
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 4.2] {out}（档位 {src}）")


def fig_summary():
    """图4.3：仅以正文三位小数历史汇总作图，不恢复未提供的原始JSON。"""
    before=[.271,.571,.525,.705,.803,.856]
    after=[.149,.139,.121,.141,.190,.177]
    fig,ax=plt.subplots(figsize=(6.4,4))
    ax.plot(range(6),before,"o:",color=GREY,lw=2,ms=6,label="before training")
    ax.plot(range(6),after,"o-",color=BLUE,lw=2,ms=6,label="after 500 steps")
    ax.axhline(.125,color=ORANGE,ls="--",label="uniform independent reference")
    ax.set(xlabel="Layer",ylabel="Consecutive first-choice repetition",ylim=(0,1),xticks=range(6))
    ax.grid(color=GRIDC,lw=.5);ax.set_axisbelow(True);ax.legend(frameon=False,fontsize=8)
    ax.set_title("Published three-decimal summaries; full 0–1 range",fontsize=10)
    fig.tight_layout();os.makedirs(FIG_DIR,exist_ok=True)
    out=os.path.join(FIG_DIR,"fig-4-3-init-final-repetition.png")
    if os.path.exists(out):raise FileExistsError(out)
    fig.savefig(out,dpi=300,facecolor="white");plt.close(fig)
    print(out)


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--which",default="machine",choices=["machine","stats","summary","all"])
    ap.add_argument("--src",default="run1")
    ap.add_argument("--log-dir",default=LOG_DIR)
    ap.add_argument("--out-dir",default=FIG_DIR)
    args=ap.parse_args()
    LOG_DIR,FIG_DIR=args.log_dir,args.out_dir
    if args.which in ("machine","all"):fig_machine()
    if args.which in ("summary","all"):fig_summary()
    if args.which in ("stats","all"):fig_stats(args.src)
