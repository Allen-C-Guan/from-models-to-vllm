"""Book4第7章：FP8格式图与历史最强离群档汇总。
工作区根目录激活env.sh后运行；只写log/book4-ch07/figures，不覆盖。
图7.2只用正文已发表一位小数；不将本轮不同输入的结果伪装成原始历史数据。
"""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
BLUE,ORANGE,TEAL="#2a78d6","#eb6834","#1baf7a"
def save(fig,out):
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    fig.tight_layout();fig.savefig(out,dpi=300,facecolor="white");plt.close(fig)
def formats(out):
    fig,axes=plt.subplots(2,1,figsize=(8,4))
    for ax,name,exponents,mantissa,details in [
        (axes[0],"E4M3",4,3,"Finite max 448; normal min 2^-6; subnormal min 2^-9\nNormal range ratio: 28,672; no Inf, one NaN pattern per sign"),
        (axes[1],"E5M2",5,2,"Finite max 57,344; normal min 2^-14; subnormal min 2^-16\nNormal range ratio: 939,524,096; signed Inf and NaNs")]:
        ax.set_xlim(-.2,8.2);ax.set_ylim(-1.3,1.6);ax.axis("off")
        labels=["S"]+["E"]*exponents+["M"]*mantissa
        for i,label in enumerate(labels):
            color="#52514e" if label=="S" else BLUE if label=="E" else ORANGE
            ax.add_patch(Rectangle((i,.15),.9,.7,ec=color,fc="white",lw=1.5))
            ax.text(i+.45,.5,label,ha="center",va="center",fontsize=11)
        ax.text(0,1.1,name,fontsize=12,weight="bold")
        ax.text(0,-.4,details,fontsize=9,va="top")
    save(fig,out/"fig-7-1-fp8-bit-layout.png")
def bulk_summary(out):
    labels=["Per-tensor\nall values","Per-tensor\nbulk only","Group-128\nbulk only"]
    values=[31.7,16.9,31.4]
    fig,ax=plt.subplots(figsize=(6.4,4))
    bars=ax.bar(labels,values,color=[BLUE,ORANGE,TEAL],width=.65)
    ax.bar_label(bars,labels=[f"{v:.1f}" for v in values],padding=4,fontsize=10)
    ax.axhspan(31.5,31.6,color="#898781",alpha=.25)
    ax.text(.04,.97,"Earlier two settings: per-tensor bulk ~31.5–31.6 dB",
            transform=ax.transAxes,fontsize=8)
    ax.set(ylabel="SNR (dB)",ylim=(0,37),title="Historical extreme-outlier setting (x30,000)")
    ax.grid(axis="y",color="#e1e0d9",lw=.5);ax.set_axisbelow(True)
    fig.text(.5,.01,"Published one-decimal summaries; reconstruction simulation, not training.",ha="center",fontsize=7)
    fig.tight_layout(rect=(0,.04,1,1))
    path=out/"fig-7-2-bulk-snr-cliff.png";path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    fig.savefig(path,dpi=300,facecolor="white");plt.close(fig)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out-dir",type=Path,default=Path.cwd()/"log/book4-ch07/figures")
    a=p.parse_args();formats(a.out_dir);bulk_summary(a.out_dir)
