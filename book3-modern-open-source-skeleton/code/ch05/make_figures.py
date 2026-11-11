# Book3第5章：参数账/配置核对图与历史分词结果的重新排版。另存log，不覆盖正文图。
# 图3长文本比率转录自本章历史正文/原图，原三段完整文本缺失，不标成本次重新分词。
import argparse
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import NullLocator,NullFormatter
from config_infer import llama_params,gpt2_params
plt.rcParams.update({'font.family':['Noto Sans CJK SC','DejaVu Sans'],'axes.unicode_minus':False,'svg.fonttype':'none','font.size':10})
BLUE,ORANGE,TEAL='#2a78d6','#eb6834','#1baf7a'
NAMES={1:'fig-5-1-tie-four-flips',2:'fig-5-2-vocab-embedding-share',3:'fig-5-3-compression-zh-en',4:'fig-5-4-config-autopsy-steps'}
HISTORICAL_COMPRESSION={'source':'本章历史正文5.4与原图5.3，2026-10；原三段文本缺失',
    'labels':['新闻体','技术体','中英混排'],'pieces_per_hanzi_32k':[1.667,1.297,2.100],
    'pieces_per_hanzi_128k':[.863,.811,1.233],
    'strings':['北京','北京清华大学',' Transformer架构','20240721 3.14159265358979 128256'],
    'string_counts_32k':[3,7,7,33],'string_counts_128k':[1,4,3,14]}


def rows(title,items,note):
    fig,ax=plt.subplots(figsize=(13,8),dpi=300);ax.set(xlim=(-.25,14.2),ylim=(0,8));ax.axis('off');ax.set_title(title,fontsize=19,pad=18)
    for i,(label,body) in enumerate(items):
        y=6.5-i*1.35
        for x,width,text,color in [(0,3.3,label,'#f2f7fd'),(3.9,10.0,body,'#f7f6f2')]:
            ax.add_patch(FancyBboxPatch((x,y),width,1.05,boxstyle='round,pad=.02',fc=color,ec='#7794a5',lw=1.1))
            ax.text(x+width/2,y+.525,text,ha='center',va='center',fontsize=11,linespacing=1.5)
        ax.annotate('',xy=(3.85,y+.525),xytext=(3.35,y+.525),arrowprops=dict(arrowstyle='->',color='#52514e'))
    fig.text(.04,.02,note,fontsize=10);fig.tight_layout(rect=(0,.05,1,.98));return fig


def fig_5_1():
    return rows('词表共享：四个配置案例，两次状态切换',[
        ('GPT-2 124M','绑定一表：38,597,376参数，占绑定模型31.0%'),
        ('LLaMA1/2 7B','两表独立：262,144,000参数，占未绑定模型3.89%'),
        ('Llama3 8B','两表独立：1,050,673,152参数，占未绑定模型13.1%'),
        ('Llama3.2 1B','绑定一表：262,668,288参数，占绑定模型21.3%'),
        ('读法','解绑新增Vd；两表总量为2Vd。参数占比不能证明唯一决策动机。')],
        '依据本章表5.1的配置与参数算术。模型来自不同团队/规格，行次不代表同一个团队的决策过程。')


def fig_5_2():
    Vs=[8192,32000,50257,128256];Vm=[8192,16384,32768,49152]
    tied=[100*llama_params(V,768,2048,12,12,12,True)['vocab_share'] for V in Vs]
    untied=[100*llama_params(V,768,2048,12,12,12,False)['vocab_share'] for V in Vs]
    module=[100*gpt2_params(V,512,6,8,1024)['wte_share_tied'] for V in Vm]
    fig,axes=plt.subplots(1,2,figsize=(12,5),dpi=300)
    axes[0].plot(Vs,tied,'o-',lw=2,color=BLUE,label='绑定：共享一表')
    axes[0].plot(Vs,untied,'s--',lw=2,color=ORANGE,label='解绑：两表总量')
    axes[1].plot(Vm,module,'o-',lw=2,color=TEAL,label='绑定：共享一表')
    for ax,vocab,values,title in [(axes[0],Vs,tied,'LLaMA式：d=768、L=12、f=2048'),(axes[1],Vm,module,'GPT-2模块：d=512、L=6')]:
        ax.set_xscale('log');ax.set_xticks(vocab,labels=[f'{v:,}' for v in vocab]);ax.xaxis.set_minor_locator(NullLocator());ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set(ylim=(0,80),xlabel='词表行数V',ylabel='输入/输出词表参数占总量（%）',title=title)
        ax.axhline(50,color='#898781',ls=':',lw=1);ax.grid(axis='y',color='#e1e0d9',lw=.5);ax.legend(frameon=False,loc='upper left')
        for V,value in zip(vocab,values):ax.annotate(f'{value:.2f}%',(V,value),xytext=(0,-15),textcoords='offset points',ha='center',fontsize=9)
    fig.text(.04,.02,'按本章两种结构的参数公式计算；右图几何与历史扫描相同。约49k时过半只适用于右侧模块，左侧50k绑定模型约31%。',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,1));return fig


def fig_5_3():
    history=HISTORICAL_COMPRESSION;fig,axes=plt.subplots(1,2,figsize=(13,5.7),dpi=300,gridspec_kw={'width_ratios':[1,1.25]})
    left,right=axes;locations=list(range(3));width=.33
    p32=history['pieces_per_hanzi_32k'];p128=history['pieces_per_hanzi_128k']
    left.bar([i-width/2 for i in locations],p32,width,color=BLUE,label='LLaMA2 32k')
    left.bar([i+width/2 for i in locations],p128,width,color=ORANGE,label='早期Llama3 128k')
    left.set_xticks(locations,history['labels']);left.set(ylim=(0,2.65),ylabel='历史记录的pieces/汉字',title='三段历史样本比率（未重新分词）')
    left.axhline(1,color='#898781',ls='--',lw=.8);left.legend(frameon=False,loc='upper left',fontsize=9)
    for i,(a,b) in enumerate(zip(p32,p128)):left.text(i,b+.08,f'{a/b:.2f}倍',ha='center',fontsize=9)
    ns32=history['string_counts_32k'];ns128=history['string_counts_128k'];loc=list(range(4))
    right.bar([i-width/2 for i in loc],ns32,width,color=BLUE,label='LLaMA2 32k')
    right.bar([i+width/2 for i in loc],ns128,width,color=ORANGE,label='早期Llama3 128k')
    right.set_xticks(loc,['北京\n2字','北京清华大学\n6字','Transformer架构\n含一个前导空格','固定数字串\n32字符'])
    right.set(ylim=(0,39),ylabel='piece数（无额外BOS/EOS）',title='正文中的四个精确字符串')
    right.legend(frameon=False,loc='upper left',fontsize=9)
    for i,(a,b) in enumerate(zip(ns32,ns128)):
        right.text(i-width/2,a+.5,str(a),ha='center',fontsize=9);right.text(i+width/2,b+.5,str(b),ha='center',fontsize=9)
    for ax in axes:ax.set_axisbelow(True);ax.grid(axis='y',color='#e1e0d9',lw=.5)
    fig.text(.04,.075,'左：转录原正文/原图，原三段完整文本缺失；混排比率不能等同纯汉字分词率。右：文本见正文，第三串保留前导空格。',fontsize=10)
    fig.text(.04,.025,'32k的“架”回退为UTF-8字节E6 9E B6。GPT-2的4/12片属于Book2对应短串；不同长文本的比率不混作同一基准线。',fontsize=10)
    fig.tight_layout(rect=(0,.12,1,1));return fig


def fig_5_4():
    return rows('从配置开始核对模型：五个步骤',[
        ('1　固定来源与实现','记录模型来源、revision、config与代码版本\n字段缺席查对应默认；保存版本不等于训练年代'),
        ('2　逐层画形状','区分残差宽d、查询头h、KV头与显式头维\n字段只给线索，投影与连接要读实际实现'),
        ('3　算参数并对账','先算每个矩阵，再计偏置、Norm、共享与缓冲\n总量相等不是全部运算等价的证明'),
        ('4　核外围约定','tokenizer本体、特殊词元、generation_config\n词表大小、表行数、填充与mask分别核对'),
        ('5　标明证据边界','配置/源码/权重/论文互证；镜像与推断注明来源\n尚缺的字段或作者动机，明确保留未知')],
        '混合dtype案例：Llama2-7B有32×64个F32频率缓冲；8,192额外字节直接除2会被误当4,096个两字节参数。')


def main():
    ap=argparse.ArgumentParser(description='参数公式图、历史读数重排与五步核对；不运行训练/分词')
    ap.add_argument('--only',default='1,2,3,4');ap.add_argument('--out-dir',type=Path,default=Path.cwd()/'log/book3-ch05/figures');a=ap.parse_args()
    todo=a.only.split(',')
    if not todo or len(set(todo))!=len(todo) or any(x not in {'1','2','3','4'} for x in todo):raise ValueError('无效图号')
    for n in todo:
        targets=[a.out_dir/(NAMES[int(n)]+'.png')]
        if n=='1':targets.append(a.out_dir/(NAMES[1]+'.svg'))
        if any(p.exists() for p in targets):raise FileExistsError('不覆盖旧图')
    a.out_dir.mkdir(parents=True,exist_ok=True)
    for n in todo:
        fig=globals()['fig_5_'+n]();fig.savefig(a.out_dir/(NAMES[int(n)]+'.png'),facecolor='white')
        if n=='1':fig.savefig(a.out_dir/(NAMES[int(n)]+'.svg'),facecolor='white')
        plt.close(fig)
    print(a.out_dir)


if __name__=='__main__':main()
