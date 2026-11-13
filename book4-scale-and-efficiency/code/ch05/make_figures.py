"""Book4第5章图5.1/5.2教学示意。根目录激活env.sh后运行；仅写log，不覆盖。
图5.3为历史训练曲线，未提供原CSV时不重造。
"""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch
BLUE,ORANGE,TEAL="#2a78d6","#eb6834","#1baf7a"
def save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    fig.tight_layout();fig.savefig(path,dpi=300,facecolor="white");plt.close(fig)
def budget(out):
    fig,axes=plt.subplots(1,2,figsize=(8,4))
    for ax,num,cols,selected in [(axes[0],16,8,{1,10}),(axes[1],64,16,{1,5,12,21,25,39,40,62})]:
        ax.set_xlim(-.5,cols+.5);ax.set_ylim(-2.5,5);ax.axis("off")
        for i in range(num):
            x,y=i%cols,3-i//cols
            ax.add_patch(Rectangle((x,y),.8,.8,facecolor=ORANGE if i in selected else "#cde2fb",edgecolor="#52514e",lw=.7))
        fine=num==64
        ax.set_title("64 experts, select 8" if fine else "16 experts, select 2",fontsize=11)
        ax.text(cols/2,-.7,"Width per expert: d_ff / 4" if fine else "Width per expert: d_ff",ha="center",fontsize=9)
        ax.text(cols/2,-1.4,"C(64,8) = 4,426,165,368" if fine else "C(16,2) = 120",ha="center",fontsize=9)
    fig.text(.5,.04,"Both: total expert weights = 16 x 3d d_ff; active = 2 x 3d d_ff\nRouter weights and execution costs are additional.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.12,1,1))
    path=out/"fig-5-1-granularity-combination-space.png"
    if path.exists():raise FileExistsError(path)
    out.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=300,facecolor="white");plt.close(fig)
def block(out):
    fig,ax=plt.subplots(figsize=(8,4));ax.set_xlim(0,12);ax.set_ylim(0,7);ax.axis("off")
    def box(x,y,w,h,label,color=BLUE):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.04",ec=color,fc="white",lw=1.4))
        ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=8,color="#0b0b0b")
    def arr(x,y,xx,yy,color="#52514e"):
        ax.annotate("",(xx,yy),(x,y),arrowprops=dict(arrowstyle="->",color=color,lw=1.2))
    box(.2,3.5,1.5,1,"x\n(B,n,d)")
    box(2.1,3.5,1.6,1,"Flatten\n(N,d)")
    box(4.2,5.4,5.1,1.1,"Router: logits / softmax (N,E)\nTop-k: indices and weights (N,k)",ORANGE)
    box(4.7,3.5,2.3,1,"Gather by index\n(n_e,d)",TEAL)
    box(7.5,3.5,2.3,1,"SwiGLU expert\n(n_e,d)",TEAL)
    box(7.4,1.6,2.8,1.1,"Weight, scatter-add\n(N,d) -> (B,n,d)",TEAL)
    box(2.1,.5,3.1,1.1,"Shared SwiGLU, width n_s w\n(B,n,d)",BLUE)
    box(5.8,.5,1.1,1.1,"Sum\n(B,n,d)")
    box(10.5,.5,1.3,1.1,"y\n(B,n,d)")
    arr(1.7,4,2.1,4);arr(3.7,4,4.7,4);arr(3.1,4.5,4.2,5.8)
    arr(5.4,5.4,5.4,4.5,ORANGE);ax.text(5.5,4.9,"indices",fontsize=7)
    arr(7,4,7.5,4);arr(8.6,3.5,8.6,2.7)
    # 权重从右侧绕行，避免穿过专家框。
    ax.plot([9.3,10.25,10.25],[5.9,5.9,2.1],color=ORANGE,lw=1.1)
    arr(10.25,2.1,10.2,2.1,ORANGE);ax.text(10.3,4.7,"weights",fontsize=7,rotation=90)
    ax.plot([.9,.9],[3.5,1.05],color="#52514e",lw=1.1);arr(.9,1.05,2.1,1.05)
    arr(5.2,1.05,5.8,1.05);arr(7.4,2.1,6.35,1.6);arr(6.9,1.05,10.5,1.05)
    ax.set_title("DeepSeekMoE block: routed branch + always-on shared branch",fontsize=10)
    ax.text(.2,6.6,"N = B x n; n_e is the actual routed count, not necessarily Nk/E",fontsize=8)
    save(fig,out/"fig-5-2-shared-routed-experts.png")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out-dir",type=Path,default=Path.cwd()/"log/book4-ch05/figures")
    a=p.parse_args();budget(a.out_dir);block(a.out_dir)
