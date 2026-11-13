"""Book4第12章代码资产地图；不加载模型、语料或设计文档正文。
运行：source env.sh; python 路径/make_figures.py --out-dir log/book4-ch12/figures
"""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
CORE=[
    "ch01/account.py","ch02/moe.py","ch02/ablation_moe.py",
    "ch03/ablation_balance.py","ch04/mixtral_infer.py","ch04/route_analysis.py",
    "ch05/deepseek_moe.py","ch05/ablation_granularity.py","ch06/mla.py",
    "ch06/kv_probe_mla.py","ch07/fp8_sim.py","ch08/mtp.py","ch08/smoke_mtp.py",
    "ch09/llama409.py","ch09/train_409.py","ch10/gptoss_reverse.py"]
BLUE,ORANGE,TEAL,INK="#2a78d6","#eb6834","#1baf7a","#202020"
def box(ax,x,y,w,h,label,color=BLUE,fs=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.008",
        facecolor="white",edgecolor=color,lw=1.6))
    ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=fs,color=INK,linespacing=1.5)
def arrow(ax,x,y,xx,yy):
    ax.annotate("",xy=(xx,yy),xytext=(x,y),arrowprops={
        "arrowstyle":"->","color":"#52514e","lw":1.5})
def main(out):
    root=Path(__file__).resolve().parents[1]
    counts={name:len((root/name).read_bytes().splitlines()) for name in CORE}
    assert len(counts)==16
    out.mkdir(parents=True,exist_ok=True)
    path=out/"fig-12-2-code-assets.png"
    report=out/"core-files.json"
    if path.exists() or report.exists():raise FileExistsError(out)
    fig,ax=plt.subplots(figsize=(10,7))
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    ax.text(.5,.98,"Book4 code: components, assembly and independent exercises",
            ha="center",fontsize=13,fontweight="bold")
    box(ax,.025,.75,.28,.15,"Book3 components\nRMSNorm + RoPE\nSwiGLU for optional dense layers")
    box(ax,.025,.53,.28,.15,"ch05/deepseek_moe.py\nRouted + shared experts\nch06/mla.py\nExplicit / absorbed attention",ORANGE)
    box(ax,.385,.64,.26,.17,"ch09/llama409.py\nTwo-knife model\n409,115,648 parameters\n12 MoE + MLA blocks",ORANGE)
    arrow(ax,.305,.825,.385,.765);arrow(ax,.305,.605,.385,.68)
    box(ax,.385,.42,.26,.12,"ch09/train_409.py\nSmoke / integrated / resume",ORANGE)
    arrow(ax,.515,.64,.515,.54)
    box(ax,.735,.73,.24,.16,"Independent exercises\nch07 FP8 / MXFP4 simulation\nch08 MTP module + smoke\nch10 config / header accounting",TEAL,8)
    box(ax,.735,.48,.24,.15,"Chapter-level comparisons\nch01 parameter ledger\nch02-05 MoE experiments\nch06 KV shape / byte probes",TEAL,8.4)
    box(ax,.025,.19,.62,.13,"00-feasibility/moe_mla_slots.py: component reference\n"
        "parity_all.py: teaching / reference / optional HF checks\n"
        "7 early probes: historical configurations and experiment entry points",fs=9)
    box(ax,.735,.19,.24,.13,"ch11-12: figures only\nNo new model training\nDESIGN-409M.md retained",TEAL,8.5)
    ax.text(.025,.10,"Arrows: listed component imports or model use.\n"
        "Independent exercises are not a sequence of resumed checkpoints.",
        fontsize=9,color=INK,linespacing=1.6)
    ax.text(.5,.025,f"Core list: 16 Python files / {sum(counts.values()):,} lines"
        " (figure scripts, probes and design documents excluded)",
        ha="center",fontsize=9,color=INK)
    fig.tight_layout();fig.savefig(path,dpi=300,facecolor="white");plt.close(fig)
    report.write_text(json.dumps(dict(core_files=counts,total_lines=sum(counts.values())),
                                 ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(dict(files=len(counts),lines=sum(counts.values())),ensure_ascii=False))
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out-dir",type=Path,required=True)
    main(ap.parse_args().out_dir)
