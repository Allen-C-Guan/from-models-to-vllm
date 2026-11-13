"""Book4 ch11：LatentMoE数据流与条件性成本算例，不运行训练。
运行：source env.sh; python 路径/make_figures.py --out-dir log/book4-ch11/figures
依据：Nemotron3 2512.20856v1 §2.2；图为教学示意。
"""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
BLUE,ORANGE,INK="#2a78d6","#eb6834","#202020"
def box(ax,x,y,w,h,text,color=BLUE,fs=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.008",
                 facecolor="white",edgecolor=color,lw=1.6))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,color=INK)
def arrow(ax,x,y,xx,yy,dashed=False):
    ax.annotate("",xy=(xx,yy),xytext=(x,y),arrowprops={
        "arrowstyle":"->","color":"#52514e","lw":1.5,"linestyle":"--" if dashed else "-"})
def save(fig,out,name):
    path=out/name
    if path.exists():raise FileExistsError(path)
    fig.tight_layout();fig.savefig(path,dpi=300,facecolor="white");plt.close(fig)
def flow(out):
    fig,ax=plt.subplots(figsize=(9,6))
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    ax.text(.5,.98,"LatentMoE sublayer: full-width control, latent expert work",
            ha="center",fontsize=12,fontweight="bold")
    box(ax,.32,.82,.36,.09,"Input X: (N,d), N = Bn\nd = 4096")
    box(ax,.02,.62,.27,.12,"Full-width router\n(N,d) -> (N,512)\ntop-22 indices + gates")
    box(ax,.35,.62,.30,.12,"Shared down projection\nP: (1024,4096)\nZ: (N,1024)",ORANGE)
    box(ax,.73,.62,.25,.12,"Full-width shared FFN\n(N,d) -> (N,d)")
    arrow(ax,.32,.865,.155,.74);arrow(ax,.5,.82,.5,.74);arrow(ax,.68,.865,.855,.74)
    box(ax,.35,.42,.30,.12,"Dispatch to selected experts\n512 total / top-22 per token\neach: (n_i,1024) -> (n_i,1024)",ORANGE,8)
    arrow(ax,.5,.62,.5,.54);arrow(ax,.29,.665,.35,.48,True)
    box(ax,.35,.25,.30,.10,"Weighted merge in latent space\n(N,1024)",ORANGE)
    arrow(ax,.5,.42,.5,.35);arrow(ax,.155,.62,.35,.30,True)
    box(ax,.35,.08,.30,.10,"Shared up projection U\n(N,1024) -> (N,4096)",ORANGE)
    arrow(ax,.5,.25,.5,.18)
    box(ax,.73,.08,.25,.10,"Add shared + routed\nOutput: (N,4096)")
    arrow(ax,.65,.13,.73,.13);arrow(ax,.855,.62,.855,.18)
    ax.text(.02,.18,"Solid: feature values\nDashed: routing control\nOuter residual add is not shown",
            fontsize=8,va="top",linespacing=1.6,color=INK)
    save(fig,out,"fig-11-1-latentmoe-dataflow.png")
def costs(out):
    fig,ax=plt.subplots(figsize=(8,4.5))
    labels=["All routed matrices","Selected routed matrices","One-way dispatch elements","Active intermediate units"]
    vals=[1,11/12,11/12,11/3]
    pos=list(range(4))
    ax.barh(pos,vals,color=[BLUE,BLUE,BLUE,ORANGE],height=.52)
    ax.set_yticks(pos,labels,fontsize=9);ax.invert_yaxis()
    ax.set_xlim(0,4.35);ax.axvline(1,color="#52514e",ls="--",lw=1)
    for y,v,t in zip(pos,vals,["1","11/12 = 0.917","11/12 = 0.917","11/3 = 3.667"]):
        ax.text(v+.06,y,t,va="center",fontsize=9,color=INK)
    ax.set_xlabel("Latent / Standard ratio (baseline = 1)",fontsize=9)
    ax.set_title("Same intermediate width m and matrix coefficient c",fontsize=12)
    ax.grid(axis="x",color="#e1e0d9",lw=.5);ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
    fig.text(.5,.09,"d=4096, latent=1024; experts 128 -> 512; active 6 -> 22",
             ha="center",fontsize=9,color=INK)
    fig.text(.5,.035,"Extra per layer: 8,388,608 projection + 1,572,864 router parameters\n"
             "Dispatch assumes one copy per expert; these ratios are not speedups.",
             ha="center",fontsize=8,color=INK)
    fig.tight_layout(rect=(0,.15,1,1))
    path=out/"fig-11-2-classic-vs-latent-diff.png"
    if path.exists():raise FileExistsError(path)
    fig.savefig(path,dpi=300,facecolor="white");plt.close(fig)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out-dir",type=Path,required=True)
    args=ap.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    flow(args.out_dir);costs(args.out_dir)
