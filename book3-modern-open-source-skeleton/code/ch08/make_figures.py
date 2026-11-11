# Book3第8章：配置/数据流示意，以及各章已刊历史终点的转绘。
# 不训练，不把第2/3/4/6章的不同模型拼成累计改装轨迹。
# python make_figures.py --only 1,2,3 --out-dir log/book3-ch08/figures
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BLUE,ORANGE,TEAL,YELLOW='#2a78d6','#eb6834','#1baf7a','#eda100'
DARK,MID,GRID='#0b0b0b','#52514e','#e1e0d9'
plt.rcParams.update({'figure.dpi':300,'savefig.dpi':300,'font.size':9,'axes.axisbelow':True})


def finish(fig,folder,name):
    fig.tight_layout();fig.savefig(folder/name,facecolor='white');plt.close(fig)


def box(ax,x,y,w,h,text,color=MID):
    ax.add_patch(plt.Rectangle((x,y),w,h,fc='white',ec=color,lw=1.3,zorder=2))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8.1,color=DARK,zorder=3)


def arrow(ax,a,b,color=MID):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',lw=1.3,color=color),zorder=1)


def fig_8_1(folder):
    fig,(left,right)=plt.subplots(1,2,figsize=(8,6.3))
    for ax in (left,right):ax.set_xlim(0,12);ax.set_ylim(0,25);ax.axis('off')
    left.set_title('LLaMA 1 7B: whole decoder',fontsize=10)
    nodes=[(22,1.4,'token ids  (B,n)',MID),(18.6,2.1,'token embedding\\nweight (32000,4096)\\noutput (B,n,4096)',MID),
           (14.5,2.4,'32 decoder blocks\\n(B,n,4096) -> (B,n,4096)',BLUE),(10.7,1.8,'final RMSNorm\\n(B,n,4096)',TEAL),
           (6.7,2.2,'untied output projection\\n4096 x 32000',MID),(2.8,1.8,'logits  (B,n,32000)',MID)]
    for y,h,text,color in nodes:box(left,1.2,y,9.6,h,text.replace('\\n','\n'),color)
    for (y,_,_,_),(ny,nh,_,_) in zip(nodes,nodes[1:]):arrow(left,(6,y),(6,ny+nh))
    left.text(6,.8,'No learned position table at the input.\\nGQA later replaces MHA in selected models.'.replace('\\n','\n'),ha='center',fontsize=8,color=MID)
    right.set_title('One block: two residual additions',fontsize=10)
    box(right,3,22,7,1.4,'x  (B,n,4096)')
    box(right,3,19,7,1.7,'RMSNorm  (B,n,4096)',TEAL)
    box(right,3,15.6,7,2,'causal MHA, h = 32\\nQ/K RoPE: (B,32,n,128)\\noutput: (B,n,4096)'.replace('\\n','\n'),BLUE)
    box(right,5.5,13.4,2,1.3,'+')
    box(right,3,10.6,7,1.7,'RMSNorm  (B,n,4096)',TEAL)
    box(right,3,6.9,7,2.4,'SwiGLU: gate/up (B,n,11008)\\nSiLU(gate) x up -> down\\noutput: (B,n,4096)'.replace('\\n','\n'),ORANGE)
    box(right,5.5,4.7,2,1.3,'+')
    box(right,3,1.8,7,1.6,'output  (B,n,4096)')
    for y,ny in [(22,20.7),(19,17.6),(15.6,14.7),(13.4,12.3),(10.6,9.3),(6.9,6),(4.7,3.4)]:arrow(right,(6.5,y),(6.5,ny))
    for top,bottom in [(21.35,14.05),(12.85,5.35)]:
        right.plot([6.5,1.4,1.4],[top,top,bottom],color=TEAL,lw=1.3,zorder=1);arrow(right,(1.4,bottom),(5.5,bottom),TEAL)
    right.text(.9,17.8,'identity',rotation=90,ha='center',fontsize=7,color=MID)
    right.text(.9,9,'identity',rotation=90,ha='center',fontsize=7,color=MID)
    finish(fig,folder,'fig-8-1-llama1-7b-machine.png')


ROWS=[
 ('normalization',['pre-RMSNorm; eps by size','pre-RMSNorm; eps=1e-5','pre-RMSNorm; eps=1e-5']),
 ('FFN',['SwiGLU; rounded 8d/3','70B: f=28672 (3.5d)','f/d: 3.5 / 3.5 / 3.25']),
 ('positions',['RoPE base 10000','RoPE base 10000','base 500000; extension rules']),
 ('attention',['MHA throughout','GQA-8: 34B / 70B','GQA-8: 8B / 70B / 405B']),
 ('document mask',['not specified here','not specified here','isolate packed documents']),
 ('vocabulary',['32000 sp-BPE','32000 sp-BPE','128256 including specials']),
 ('context',['2048','4096','8192 -> 131072 (3.1)']),
 ('training tokens',['1.0T / 1.4T','2.0T reported','about 15T; 405B: 15.6T']),
 ('release license',['research use','community; conditions apply','community; conditions apply']),
]


def fig_8_2(folder):
    fig,ax=plt.subplots(figsize=(8,5.6));ax.axis('off')
    table=ax.table(cellText=[v for _,v in ROWS],rowLabels=[n for n,_ in ROWS],colLabels=['LLaMA 1','LLaMA 2','Llama 3 / 3.1'],cellLoc='center',bbox=[.18,.06,.82,.87])
    table.auto_set_font_size(False);table.set_fontsize(7.2)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#d9d7cf');cell.set_linewidth(.6)
        cell.set_facecolor('#d7e6f9' if row==0 else '#f4f3ef' if col==-1 else 'white')
    ax.set_title('Same component families, changing configurations',fontsize=11,pad=8)
    fig.text(.5,.015,'Comparison of reported settings; not a decomposition of performance gains.',ha='center',fontsize=8,color=MID)
    finish(fig,folder,'fig-8-2-generational-slots.png')


# 已刊精度（非新训练）：ch2表2.2、ch3 full段、ch4表4.3、ch6表6.4。
HISTORICAL=[
 ('Norm: d=512, L=6',[('LayerNorm',4.4945),('RMSNorm',4.5375)]),
 ('FFN: d=768, L=6\nmain matrix budget matched',[('GELU-4C',4.7866),('SwiGLU',4.5914)]),
 ('Position: d=512, L=6',[('wpe',4.4923),('RoPE',4.0813)]),
 ('Attention: d=512, L=6\nparameters differ',[('MHA-8',4.4263),('GQA-4',4.5351),('MQA-1',4.6933)]),
]


def fig_8_3(folder):
    fig,axes=plt.subplots(2,2,figsize=(8,5.5))
    for ax,(title,data) in zip(axes.flat,HISTORICAL):
        for i,(name,value) in enumerate(data):
            ax.plot(value,i,'o',ms=7,color=BLUE if i==0 else ORANGE)
            ax.annotate(f'{value:.4f}',(value,i),xytext=(5,5),textcoords='offset points',fontsize=8,color=DARK)
        ax.set_yticks(range(len(data)),[x[0] for x in data]);ax.set_xlim(4,4.95);ax.set_ylim(len(data)-.3,-.65);ax.set_title(title,fontsize=8.5)
        ax.grid(axis='x',color=GRID,lw=.5);ax.set_xlabel('eval loss (nat)',fontsize=8)
        for side in ['top','right']:ax.spines[side].set_visible(False)
    fig.suptitle('Independent module experiments: historical full-tier endpoints',fontsize=11)
    fig.tight_layout(rect=(0,.05,1,.95))
    fig.text(.5,.012,'5000 steps, one seed per experiment; controls differ by panel (see chapters 2/3/4/6).',ha='center',fontsize=8,color=MID)
    fig.savefig(folder/'fig-8-3-slot-contribution.png',facecolor='white');plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--only',default='1,2,3');ap.add_argument('--out-dir',type=Path,default=Path.cwd()/'log/book3-ch08/figures');args=ap.parse_args()
    choices=set(args.only.split(','))
    if not choices or not choices<=set('123'):raise ValueError('only可选1,2,3')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    for k in choices:
        if list(args.out_dir.glob(f'fig-8-{k}-*')):raise FileExistsError('输出图已存在，请用新目录')
    for key,fn in [('1',fig_8_1),('2',fig_8_2),('3',fig_8_3)]:
        if key in choices:fn(args.out_dir)


if __name__=='__main__':main()
