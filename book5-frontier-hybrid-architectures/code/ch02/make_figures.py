# Book5 第2章：教学数据流、真实层排布、历史检索表重绘。
# 运行：source env.sh && python code/ch02/make_figures.py --out-dir log/book5-ch02/figures-run1
# 图3只转绘正文历史表2.3，不产生训练数据；连线不代表未测位置。
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.font_manager import FontProperties
import numpy as np

FONT=FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family":FONT.get_name(),"axes.unicode_minus":False,"font.size":11})
BLUE="#2678c8"; ORANGE="#dc6b31"; GREEN="#25926d"

def box(ax,x,y,w,h,text,color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.015",fc="#f4f8fc",ec=color,lw=1.6))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=11)
def arrow(ax,a,b):
    ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",lw=1.6,color="#555"))

def draw_flow(out):
    fig,ax=plt.subplots(figsize=(12,7))
    ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
    ax.set_title("滑窗与零值 sink：限制参与位置，再调整读取幅度",pad=20)
    rows=[(.82,"输入 X\n(B,n,d)","线性投影 Q / K / V\nQ:(B,h,n,$d_k$)\nK,V:(B,$h_{kv}$,n,$d_k$)"),
          (.57,"按 KV 分组展开\nK,V:(B,h,n,$d_k$)","$QK^T/\\sqrt{d_k}$，施加带状掩码\n(B,h,n,n)\n每行最多 W 个位置可见"),
          (.32,"拼接每头 sink 标量 $b_h$\n(B,h,n,n+1)","softmax 后丢弃 sink 那一列\nP:(B,h,n,n)\n每行权重和为 1−αsink"),
          (.07,"P V\n(B,h,n,$d_k$)","合并头并作输出投影\n(B,n,h·$d_k$) → (B,n,d)")]
    for y,left,right in rows:
        box(ax,.04,y,.38,.13,left)
        box(ax,.56,y,.40,.13,right,ORANGE if y==.32 else BLUE)
        arrow(ax,(.43,y+.065),(.55,y+.065))
    for y in [.82,.57,.32]:
        arrow(ax,(.76,y),(.23,y-.12))
    fig.tight_layout();fig.savefig(out/"fig-2-1-swa-sink.png",dpi=170);plt.close(fig)

def draw_layers(out):
    fig,ax=plt.subplots(figsize=(12,4.6))
    models=[("Gemma2-9B",42,2,"W=4096；全局 21/42"),
            ("Gemma3-27B",62,6,"W=1024；全局 10/62"),
            ("gpt-oss-20b",24,2,"W=128；全局 12/24")]
    for y,(name,n,period,note) in enumerate(models):
        for i in range(n):
            ax.barh(y, .9, left=i+.05, height=.42,color=ORANGE if (i+1)%period==0 else BLUE)
        ax.text(64,y,note,va="center",fontsize=10)
    ax.set_yticks(range(3),[m[0] for m in models]);ax.invert_yaxis()
    ax.set(xlim=(0,90),xlabel="层号从 1 开始；每格一层",title="窗口长度与全局层比例是两个独立配置")
    ax.set_xticks([0,9,19,29,39,49,59],[1,10,20,30,40,50,60])
    ax.legend(handles=[Patch(color=BLUE,label="滑窗层"),Patch(color=ORANGE,label="全局层")],
              loc="lower right")
    for s in ["top","right"]:ax.spines[s].set_visible(False)
    fig.tight_layout();fig.savefig(out/"fig-2-2-swa-sink.png",dpi=170);plt.close(fig)

def draw_recall(out):
    p=[16,22,28,34,40,46]
    rows=[("满注意力，20k",[1]*6,BLUE,"-","o"),
          ("滑窗 + sink，20k",[0,.05,.02,1,1,1],ORANGE,"-","s"),
          ("满注意力，8k",[.25,.10,.08,.15,.10,.07],BLUE,"--","o"),
          ("滑窗 + sink，8k",[.02,0,0,0,0,.02],ORANGE,"--","s")]
    fig,ax=plt.subplots(figsize=(10,5.8))
    for label,acc,c,ls,m in rows:ax.plot(p,acc,label=label,color=c,ls=ls,marker=m)
    ax.axhline(1/64,color="#777",ls=":",label="均匀猜测 1/64")
    ax.axvline(31,color=GREEN,ls="--")
    ax.text(31.4,.63,"值可达的必要边界 p=31\n值位置 p+1=32",color=GREEN)
    ax.axvline(46,color="#ad7a00",ls="--")
    ax.text(45.5,.27,"旧单层判据 p=46\n不适用于整个两层模型",ha="right",color="#ad7a00")
    ax.set(xticks=p,ylim=(-.04,1.08),xlabel="目标键位置 p（查询位置 62；T=64，W=16，L=2）",
           ylabel="目标召回率（每位置 60 条）",title="历史实验表 2.3：29—33 未测，连线仅辅助辨认")
    ax.grid(alpha=.2);ax.legend(loc="center left",fontsize=9)
    fig.tight_layout();fig.savefig(out/"fig-2-3-swa-sink.png",dpi=170);plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out-dir",type=Path,required=True);a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=False)
    draw_flow(a.out_dir);draw_layers(a.out_dir);draw_recall(a.out_dir)

if __name__=="__main__":main()
