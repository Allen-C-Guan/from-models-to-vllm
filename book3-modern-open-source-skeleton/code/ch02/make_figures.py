# Book3第2章：归一化计算、换件位置与真实日志曲线；输出到工作目录log，不覆盖正文图。
# structure无需训练数据；curves必须提供报告及对应CSV，旧报告另加--historical。
import argparse
import csv
import json
import math
from pathlib import Path


def configure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts={f.name for f in font_manager.fontManager.ttflist}
    for name in ('Noto Sans CJK SC','Noto Sans CJK JP','WenQuanYi Zen Hei'):
        if name in fonts:plt.rcParams['font.family']=name;break
    plt.rcParams.update({'font.size':10,'axes.unicode_minus':False,'svg.fonttype':'none'})
    return plt


def box(ax,x,y,w,h,text,changed=False,fontsize=10):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.03',
        edgecolor='#eb6834' if changed else '#2a78d6',facecolor='#fdf3ef' if changed else '#f2f7fd',lw=1.5))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize,linespacing=1.5)


def arrow(ax,start,end):
    ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',lw=1.4,color='#52514e',shrinkA=0,shrinkB=0))


def norm_graph(plt):
    fig,axes=plt.subplots(1,2,figsize=(12,8),dpi=300)
    specs=[('LayerNorm：增益与偏置各C个',r'$\mu=\mathrm{mean}_C(x),\quad\sigma^2=\mathrm{mean}_C((x-\mu)^2)$',
             r'$z=(x-\mu)/\sqrt{\sigma^2+\varepsilon}$',r'$y=\gamma\odot z+\beta$'+'\n'+r'$\gamma,\beta:(C,)$'),
           ('LLaMA式RMSNorm：增益C个',r'$q=\mathrm{mean}_C(x^2)$',r'$z=x/\sqrt{q+\varepsilon}$',
             r'$y=g\odot z$'+'\n'+r'$g:(C,)$，本实现无偏置')]
    for ax,(title,statistics,normalize,affine) in zip(axes,specs):
        ax.set(xlim=(0,10),ylim=(0,11.2));ax.axis('off');ax.set_title(title,fontsize=13,pad=15)
        box(ax,3,9.5,4,.9,'输入x\n(B,n,C)')
        box(ax,1,7.2,7.6,1.25,statistics+'\n统计量均为(B,n,1)',fontsize=10)
        box(ax,1.7,4.9,6.6,1.25,normalize+'\n(B,n,C)')
        box(ax,1.7,2.6,6.6,1.25,affine,changed=True)
        box(ax,3,.5,4,.9,'输出y\n(B,n,C)')
        for start,end in [((5,9.5),(5,8.45)),((5,7.2),(5,6.15)),((5,4.9),(5,3.85)),((5,2.6),(5,1.4))]:arrow(ax,start,end)
        ax.plot([7,9.3,9.3],[9.95,9.95,5.525],color='#52514e',lw=1.4);arrow(ax,(9.3,5.525),(8.3,5.525))
        ax.text(9.55,7.0,'保留原x',rotation=90,ha='center',va='center',fontsize=9,color='#52514e')
    fig.text(.04,.012,'两侧都显式包含epsilon。无偏置是此处LLaMA式实现的选择；RMS归一化之后也可以另加偏置。',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,1));return fig


def swap_graph(plt):
    from matplotlib.patches import Circle
    fig,(top,ax)=plt.subplots(2,1,figsize=(14,8),dpi=300,gridspec_kw={'height_ratios':[.8,1.2]})
    for a in (top,ax):a.set(xlim=(0,20),ylim=(0,5));a.axis('off')
    top.text(0,4.4,'图2.2　模块档L=6、C=512：只替换13处归一化',fontsize=17,weight='bold')
    nodes=[(.2,2.,'ID\n(B,n)'),(2.8,4.0,'词嵌入＋位置嵌入\n(B,n,512)'),(7.4,3.5,'6层Block\n每层两处Norm'),(11.5,3.0,'末端ln_f\n(B,n,512)'),(15.1,4.0,'共享词表读出\nlogits(B,n,8192)')]
    for x,width,text in nodes:box(top,x,2,width,1.2,text,changed=x==11.5)
    for (x,width,_),(nx,_,_) in zip(nodes,nodes[1:]):arrow(top,(x+width,2.6),(nx,2.6))
    top.text(.2,.8,'6×2＋1＝13处；每处去掉512个beta参数，共少6656个。注意力、前馈与读出头保持原对象。',fontsize=11)
    ax.text(0,4.5,'一层Block：pre位置与两次残差相加保持',fontsize=15,weight='bold');yy=2.1
    box(ax,1.2,yy-.5,2.0,1.,'ln_1\n(B,n,512)',True)
    box(ax,3.9,yy-.7,3.,1.4,'注意力\n头(B,8,n,64)\n输出(B,n,512)')
    box(ax,8.7,yy-.5,2.0,1.,'ln_2\n(B,n,512)',True)
    box(ax,11.5,yy-.7,3.,1.4,'GELU前馈\n中间(B,n,2048)\n输出(B,n,512)')
    for x in (7.7,15.3):
        ax.add_patch(Circle((x,yy),.24,edgecolor='#52514e',facecolor='white',lw=1.5));ax.text(x,yy,'+',ha='center',va='center',fontsize=14)
    for x1,x2 in [(.2,1.2),(3.2,3.9),(6.9,7.46),(7.94,8.7),(10.7,11.5),(14.5,15.06),(15.54,16.8)]:arrow(ax,(x1,yy),(x2,yy))
    ax.plot([.6,.6,7.7],[yy,3.7,3.7],color='#52514e',lw=1.4);arrow(ax,(7.7,3.7),(7.7,yy+.24))
    ax.text(4,3.9,'第一条旁路保留输入x',ha='center',fontsize=10)
    ax.plot([8.3,8.3,15.3],[yy,.6,.6],color='#52514e',lw=1.4);arrow(ax,(15.3,.6),(15.3,yy-.24))
    ax.text(11.8,.2,'第二条旁路保留第一次相加后的x1',ha='center',fontsize=10)
    ax.text(17.,yy,'输出到下一层\n(B,n,512)',ha='left',va='center',fontsize=11)
    fig.text(.04,.01,'橙框：LayerNorm（gamma、beta）→LLaMA式RMSNorm（g）；增益参数为(C,)，归一化输出始终为(B,n,C)。',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,1));return fig


def load_curves(path,historical=False):
    report=json.loads(path.read_text());name=report.get('meta',{}).get('out_name')
    if not name or Path(name).name!=name:raise ValueError('报告缺有效out_name')
    data=[]
    for arm in ('base','rmsnorm'):
        rec=report.get('arms',{}).get(arm,{})
        if rec.get('completed') is False:raise ValueError('明确未完成的报告不绘图')
        if rec.get('completed') is not True and not historical:raise ValueError('旧报告需显式--historical')
        steps=rec.get('steps_done',0)
        if steps<=0 or steps!=rec.get('steps_planned') or rec.get('diverged'):raise ValueError('未完成或失败实验不能画成功曲线')
        values=[]
        with (path.parent/f'curve_{arm}_{name}.csv').open() as f:
            for row in csv.DictReader(f):
                if int(row['step'])!=len(values)+1:raise ValueError('CSV步数不连续')
                value=float(row['train_loss'])
                if not math.isfinite(value):raise ValueError('CSV非有限损失')
                values.append(value)
        if len(values)!=steps or not rec.get('evals') or rec['evals'][-1]['step']!=steps:raise ValueError('曲线/报告终点不完整')
        if not isinstance(rec.get('eval_final'),(int,float)) or not math.isfinite(rec['eval_final']):raise ValueError('缺有效最终eval')
        for event in rec['evals']:
            if not math.isfinite(event['eval_loss']):raise ValueError('eval损失非有限')
        data.append((arm,rec,values))
    return data


def curve_graph(plt,data):
    fig,ax=plt.subplots(figsize=(8,4),dpi=300)
    for (arm,rec,values),color in zip(data,('#2a78d6','#eb6834')):
        smoothed=[];value=None
        for x in values:value=x if value is None else .01*x+.99*value;smoothed.append(value)
        ax.plot(range(1,len(values)+1),smoothed,color=color,label=arm+' train EMA')
        ax.plot([x['step'] for x in rec['evals']],[x['eval_loss'] for x in rec['evals']],'o',color=color,label=arm+' eval',ms=5)
    difference=abs(data[0][1]['eval_final']-data[1][1]['eval_final'])
    ax.set_title(f'Observed final difference: {difference:.5f} nat; historical descriptive band: 0.02',fontsize=9)
    ax.set(xlabel='training step',ylabel='loss (nat)');ax.grid(color='#e1e0d9',lw=.5);ax.legend(frameon=False,fontsize=8)
    fig.tight_layout();return fig


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plots',choices=['structure','curves','all'],default='structure')
    ap.add_argument('--report',type=Path);ap.add_argument('--historical',action='store_true')
    ap.add_argument('--out-dir',type=Path,default=Path.cwd()/'log/book3-ch02/figures');a=ap.parse_args()
    names=[]
    if a.plots in ('structure','all'):names+=['fig-2-1-norm-compute-graph','fig-2-2-swap-position']
    if a.plots in ('curves','all'):names+=['fig-2-3-norm-ablation']
    for name in names:
        if (a.out_dir/(name+'.png')).exists():raise FileExistsError('不覆盖已有图片：'+name)
    data=None
    if a.plots in ('curves','all'):
        if a.report is None:raise ValueError('曲线须给--report及相邻真实CSV')
        data=load_curves(a.report,a.historical)
    plt=configure();a.out_dir.mkdir(parents=True,exist_ok=True)
    if a.plots in ('structure','all'):
        for name,make in zip(names[:2],(norm_graph,swap_graph)):
            fig=make(plt);fig.savefig(a.out_dir/(name+'.png'),facecolor='white');plt.close(fig)
    if data is not None:
        fig=curve_graph(plt,data);fig.savefig(a.out_dir/'fig-2-3-norm-ablation.png',facecolor='white');plt.close(fig)
    print('图源已另存：',a.out_dir)


if __name__=='__main__':main()
