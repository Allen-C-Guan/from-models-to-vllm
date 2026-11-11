# Book3第10章：整机形状与双残差；历史曲线必须来自显式CSV，不插值反造缺失数据。
# python make_figures.py --only 1 --out-dir log/book3-ch10/figures
from pathlib import Path
import argparse,csv,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BLUE,ORANGE,TEAL='#2a78d6','#eb6834','#1baf7a'
MID,GRID='#52514e','#e1e0d9'
plt.rcParams.update({'figure.dpi':300,'savefig.dpi':300,'font.size':8,'axes.axisbelow':True})
def box(ax,x,y,w,h,text,color=MID):
    ax.add_patch(plt.Rectangle((x,y),w,h,facecolor='white',edgecolor=color,lw=1.3,zorder=2))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=7.7,color='#0b0b0b',zorder=3)
def arrow(ax,a,b,color=MID):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',lw=1.3,color=color),zorder=1)
def finish(fig,path):
    if path.exists():raise FileExistsError(path)
    fig.tight_layout();fig.savefig(path,facecolor='white');plt.close(fig)
def fig_10_1(folder):
    fig,(left,right)=plt.subplots(1,2,figsize=(9.2,6.4))
    for ax in (left,right):ax.set(xlim=(0,12),ylim=(0,25));ax.axis('off')
    left.set_title('207M: whole language model',fontsize=10)
    nodes=[(22,1.4,'token ids (B,n)',MID),(18.6,2.1,'embedding weight (32000,1024)\noutput (B,n,1024)',MID),
           (14.5,2.4,'12 decoder blocks\n(B,n,1024) -> (B,n,1024)',BLUE),(10.7,1.8,'final RMSNorm [ch02]\n(B,n,1024)',TEAL),
           (6.7,2.2,'untied lm_head [ch05]\nweight (32000,1024)',MID),(2.8,1.8,'logits (B,n,32000)',MID)]
    for y,h,text,color in nodes:box(left,1.2,y,9.6,h,text,color)
    for (y,_,_,_),(ny,nh,_,_) in zip(nodes,nodes[1:]):arrow(left,(6,y),(6,ny+nh))
    left.text(6,.5,'Historical training: B=8, n=1024\nNo input position table; 207,119,360 parameters',ha='center',fontsize=7.3,color=MID)
    right.set_title('One block: component sources',fontsize=10)
    box(right,3,22,7,1.4,'x (B,n,1024)')
    box(right,3,19,7,1.7,'RMSNorm [ch02]\n(B,n,1024)',TEAL)
    box(right,3,15.4,7,2.2,'GQA [ch06], RoPE on Q/K [ch04]\nQ (B,16,n,64); K/V (B,8,n,64)\noutput (B,n,1024)',BLUE)
    box(right,5.5,13.2,2,1.3,'+')
    box(right,3,10.4,7,1.7,'RMSNorm [ch02]\n(B,n,1024)',TEAL)
    box(right,3,6.7,7,2.4,'SwiGLU [ch03]: gate/up (B,n,2816)\nSiLU(gate) x up -> down\noutput (B,n,1024)',ORANGE)
    box(right,5.5,4.5,2,1.3,'+')
    box(right,3,1.6,7,1.6,'output (B,n,1024)')
    for y,ny in [(22,20.7),(19,17.6),(15.4,14.5),(13.2,12.1),(10.4,9.1),(6.7,5.8),(4.5,3.2)]:
        arrow(right,(6.5,y),(6.5,ny))
    for top,bottom in [(21.35,13.85),(12.55,5.15)]:
        right.plot([6.5,1.4,1.4],[top,top,bottom],color=TEAL,lw=1.3,zorder=1)
        arrow(right,(1.4,bottom),(5.5,bottom),TEAL)
    right.text(.9,17.8,'identity',rotation=90,ha='center',fontsize=7,color=MID)
    right.text(.9,8.8,'identity',rotation=90,ha='center',fontsize=7,color=MID)
    right.text(6,.2,'FFN width: round (8d/3) up to a multiple of 256',ha='center',fontsize=7,color=MID)
    finish(fig,folder/'fig-10-1-llama215-machine-slots.png')
def read_curve(path):
    with Path(path).open() as f:rows=list(csv.DictReader(f))
    train={};evaluation={}
    for r in rows:
        step=int(r['step'])
        if step<1:raise ValueError('step必须正')
        if r.get('train_loss'):
            value=float(r['train_loss'])
            if step in train or not math.isfinite(value):raise ValueError('重复训练步或非有限loss')
            train[step]=value
        if r.get('eval_loss'):
            value=float(r['eval_loss'])
            if not math.isfinite(value) or (step in evaluation and evaluation[step]!=value):
                raise ValueError('同一步评估冲突或非有限')
            evaluation[step]=value
    if not train or not evaluation:raise ValueError('需要真实train和eval列')
    if list(train)!=sorted(train):raise ValueError('训练步乱序，须先核对恢复轨迹')
    if not set(evaluation).issubset(train):raise ValueError('评估步没有对应训练步')
    return train,evaluation
def fig_10_2(folder,smoke,integrated):
    a,b=read_curve(smoke),read_curve(integrated)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,(tr,ev),title in zip(axes,(a,b),('Smoke: supplied CSV','Integrated: supplied CSV')):
        ax.plot(list(tr),list(tr.values()),color=BLUE,lw=2,label='train')
        ax.plot(list(ev),list(ev.values()),'o',color=ORANGE,ms=6,label='eval')
        ax.set(xlabel='step',ylabel='loss (nat/token)',title=title);ax.grid(color=GRID,lw=.5);ax.legend(frameon=False)
    # 仅绘CSV记录；可选小配置不能静默叠加207M的lnV/方差基线。
    # 7.674仅属于旧扩大前缀；新CSV未提供同一统计样本，不自动套用这条线。
    finish(fig,folder/'fig-10-2-smoke-integrated-curve.png')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--only',choices=['1','2','1,2'],default='1')
    ap.add_argument('--out-dir',type=Path,default=Path('log/book3-ch10/figures'))
    ap.add_argument('--smoke',type=Path);ap.add_argument('--integrated',type=Path)
    a=ap.parse_args()
    if '2' in a.only and (a.smoke is None or a.integrated is None):raise ValueError('曲线必须显式提供两个CSV')
    a.out_dir.mkdir(parents=True,exist_ok=True)
    if '1' in a.only:fig_10_1(a.out_dir)
    if '2' in a.only:fig_10_2(a.out_dir,a.smoke,a.integrated)
if __name__=='__main__':main()
