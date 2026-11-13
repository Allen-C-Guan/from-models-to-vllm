"""Book4 ch9两张结构示意；只画公开几何，不重建缺失历史曲线。"""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
BLUE,ORANGE,INK="#2a78d6","#eb6834","#202020"
def box(ax,x,y,w,h,text,color=BLUE,fs=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.008",facecolor="white",edgecolor=color,lw=1.6))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,color=INK,linespacing=1.45)
def arrow(ax,x,y,xx,yy):
    ax.annotate("",xy=(xx,yy),xytext=(x,y),arrowprops={"arrowstyle":"->","color":"#52514e","lw":1.5})
def save(fig,out,name):
    path=out/name
    if path.exists():raise FileExistsError(path)
    fig.tight_layout();fig.savefig(path,dpi=300,facecolor="white");plt.close(fig)
def machines(out):
    fig,axes=plt.subplots(1,2,figsize=(10,7))
    rows=[
      ("DeepSeek-V2","5120","102400",1,59,12288,"160 routed / top-6 + 2 shared\nexpert width 1536","softmax; no selected re-normalization\nscale 16; groups 8 -> 3",
       "Training: 3 auxiliary losses + dropping\nBF16 framework; no MTP in this report"),
      ("DeepSeek-V3","7168","129280",3,58,18432,"256 routed / top-8 + 1 shared\nexpert width 2048","sigmoid; selected weights normalized\nscale 2.5; groups 8 -> 4",
       "Training: bias updates + small SeqBal\nFP8 mixed precision; MTP D=1 is optional at inference")]
    for ax,row in zip(axes,rows):
        title,d,V,nd,nm,ff,moe,route,training=row
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
        ax.text(.5,.98,title,ha="center",fontsize=15,fontweight="bold")
        box(ax,.08,.86,.84,.075,f"Embedding: (B,n) -> (B,n,{d})")
        box(ax,.08,.65,.84,.16,f"First {nd} block(s): (B,n,{d})\nNorm -> MLA -> add residual\nNorm -> dense FFN ({ff}) -> add residual")
        box(ax,.08,.405,.84,.19,f"Next {nm} blocks: (B,n,{d})\nNorm -> MLA -> add residual\nNorm -> MoE -> add residual\n{moe}",ORANGE,8.6)
        box(ax,.08,.28,.84,.075,f"Final norm -> vocabulary head\n(B,n,{d}) -> (B,n,{V})",fs=8.7)
        for y,yy in [(.86,.81),(.65,.595),(.405,.355)]:arrow(ax,.5,y,.5,yy)
        ax.text(.5,.235,f"MLA in ALL {nd+nm} blocks: 128 query heads\nKV rank 512, query rank 1536; content/RoPE/value = 128/64/128",
                ha="center",va="top",fontsize=7.6,linespacing=1.5)
        ax.text(.5,.15,route,ha="center",va="top",fontsize=8,color=INK)
        ax.text(.5,.075,training,ha="center",va="top",fontsize=7.7,color=INK,linespacing=1.4)
    fig.text(.5,.01,"MTP branches from the final backbone state and next-token embedding; it is not after main logits.",
             ha="center",fontsize=8,color=INK)
    save(fig,out,"fig-9-1-dsv2-v3-machine-diff.png")
def two_knife(out):
    fig,ax=plt.subplots(figsize=(9,6.5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    ax.text(.5,.98,"Two-knife model: cand2 (409,115,648 parameters)",ha="center",fontsize=13,fontweight="bold")
    box(ax,.08,.84,.48,.085,"Embedding\n(B,n) -> (B,n,1024)")
    box(ax,.08,.665,.48,.13,"RMSNorm -> MLA -> add residual\n(B,n,1024) -> (B,n,1024)\nKV/query rank 256; content/RoPE/value 64/64/64",ORANGE,8.6)
    box(ax,.08,.49,.48,.13,"RMSNorm -> MoE -> add residual\n8 routed / top-2 + 1 shared; width 938\nrouter: (Bn,1024) -> (Bn,8)",ORANGE,8.8)
    box(ax,.08,.325,.48,.10,"Final RMSNorm -> vocabulary head\n(B,n,1024) -> (B,n,32000)")
    for y,yy in [(.84,.795),(.665,.62),(.49,.425)]:arrow(ax,.32,y,.32,yy)
    ax.plot([.025,.025],[.49,.795],color=INK,lw=1.3)
    ax.text(.015,.642,"repeat 12 blocks",rotation=90,ha="right",va="center",fontsize=8)
    ax.text(.65,.87,"Component sources",fontsize=11,fontweight="bold")
    ax.text(.65,.79,"RMSNorm: Book3 ch2\nMLA: Book4 ch6\nMoE: Book4 ch5\nRoPE: inside MLA",fontsize=10,va="top",linespacing=1.8)
    ax.text(.65,.55,"Replaced components",fontsize=11,fontweight="bold")
    ax.text(.65,.48,"GQA: 16 Q / 8 KV heads\nhead dimension 64\nDense SwiGLU width 2816",fontsize=10,va="top",linespacing=1.8)
    ax.text(.08,.24,"Designed latent payload: 12 x (256+64) = 3,840 elements/token\n"
            "Retired GQA: 12,288 elements/token; reduction 68.75%\n"
            "Teaching forward has no persistent incremental KV cache.",fontsize=10,va="top",linespacing=1.6)
    ax.text(.08,.08,"Active ledger (with output head): 168,877,056\n"
            "cand3 control: 207,064,064 total; 55,296 fewer than Book3 baseline",fontsize=9,va="top",linespacing=1.5)
    save(fig,out,"fig-9-2-two-knife-409m-slots.png")
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out-dir",type=Path,required=True)
    args=ap.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    machines(args.out_dir);two_knife(args.out_dir)
