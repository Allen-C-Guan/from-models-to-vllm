"""Book4导览：整机与一个Block。自绘教学示意，运行后输出到日志目录。"""
from pathlib import Path
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def draw(out):
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
    ax.set_title("Whole model",fontsize=10)
    labels=[("Token IDs  (B, n)",9),("Embedding  (B, n, 1024)",7.4),
            ("12 Blocks  (B, n, 1024)",5.8),("RMSNorm  (B, n, 1024)",4.2),
            ("Untied LM head",2.6),("Logits  (B, n, 32000)",1)]
    for i,(label,y) in enumerate(labels):
        box(ax,.6,y-.3,8.8,.9,label)
        if i:arr(ax,5,labels[i-1][1]-.3,5,y+.6)
    bx.set_title("One Block: two residual additions",fontsize=10)
    for label,y,color in [("x  (B, n, 1024)",9.3,blue),("RMSNorm",8.2,blue),
             ("MLA: latent KV 256 + RoPE key 64",6.9,orange),
             ("+  (B, n, 1024)",5.6,blue),("RMSNorm",4.3,blue),
             ("MoE: 8 routed / top-2 + 1 shared\nSwiGLU width 938",2.8,orange),
             ("+  (B, n, 1024)",1.1,blue)]:
        box(bx,1.8,y-.25,7.8,.8 if y!=2.8 else 1.05,label,color,7.8)
    for y1,y2 in [(9.05,8.75),(7.95,7.45),(6.65,6.15),(5.35,4.85),(4.05,3.6),(2.55,1.65)]:
        arr(bx,5.7,y1,5.7,y2)
    bx.plot([1.8,.6,.6],[9.4,9.4,5.75],c="#52514e",lw=1.1)
    arr(bx,.6,5.75,1.8,5.75)
    bx.plot([5.7,1.1,1.1],[5.3,5.3,1.25],c="#52514e",lw=1.1)
    arr(bx,1.1,1.25,1.8,1.25)
    fig.text(.5,.01,"d=1024, L=12, h=16, V=32000; internal attention and FFN return (B,n,1024)",
             ha="center",fontsize=7)
    fig.tight_layout(rect=(0,.04,1,1))
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=300,facecolor="white");plt.close(fig)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=Path(os.getenv("MODEL_ANALYSIS_PROJECT_ROOT",Path.cwd()))/
                    "log/Book4-round2-verification/fig-0-1-two-cuts-409m-overview.png")
    args=ap.parse_args();draw(args.out)
