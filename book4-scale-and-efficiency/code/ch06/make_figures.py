"""Book4图6.4：配置公式的有效载荷对照；非运行时峰值测量。
根目录激活env.sh后运行本文件。保留前三幅既有机制图；新图写入log。
"""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def main(out):
    names=["Probe: GQA-4","12-layer: GQA-8","12-layer: MHA","V2-Lite: MHA","V2-236B: MHA"]
    baselines=[2*6*4*64,2*12*8*64,2*12*16*64,2*27*16*128,2*60*128*128]
    latent=[6*(256+64),12*(256+64),12*(256+64),27*(512+64),60*(512+64)]
    fig,ax=plt.subplots(figsize=(8,4))
    for i,(b,m) in enumerate(zip(baselines,latent)):
        ax.plot([m,b],[i,i],color="#898781",lw=2)
        ax.plot(m,i,"o",color="#2a78d6",ms=7,label="MLA latent payload" if i==0 else None)
        ax.plot(b,i,"s",color="#eb6834",ms=7,label="Specified baseline" if i==0 else None)
        ax.annotate(f"{m:,}",(m,i),xytext=(0,-14),textcoords="offset points",ha="center",fontsize=7)
        ax.annotate(f"{b:,}",(b,i),xytext=(0,9),textcoords="offset points",ha="center",fontsize=7)
        ax.text(.99,i,f"{100*(1-m/b):.1f}% less",transform=ax.get_yaxis_transform(),
                ha="right",va="center",fontsize=8,color="#0b0b0b")
    ax.set(xscale="log",xlim=(1100,15000000),yticks=range(5),yticklabels=names,
           ylim=(-.6,4.6),xlabel="Elements per token, all layers")
    ax.grid(axis="x",color="#e1e0d9",lw=.5);ax.legend(loc="upper left",bbox_to_anchor=(0,1.15),ncols=2,frameon=False,fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    fig.savefig(out,dpi=300,facecolor="white");plt.close(fig)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=Path.cwd()/"log/book4-ch06/fig-6-4-kv-multi-geometry.png")
    main(p.parse_args().out)
