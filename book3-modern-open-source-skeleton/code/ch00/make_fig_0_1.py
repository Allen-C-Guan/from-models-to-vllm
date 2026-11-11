# Book3导览：整机与单层Block示意，按207M配置标形状；不运行训练。
# 工作区激活env.sh后运行本文件；输出到当前工作目录log，拒绝覆盖已有图。
import argparse
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description='生成207M整机与一层Block的教学示意')
    ap.add_argument('--out-dir',type=Path,default=Path.cwd()/'log/book3-ch00/figures')
    a=ap.parse_args();name='fig-0-1-llama-207m-overview'
    targets=[a.out_dir/(name+'.'+ext) for ext in ('png','svg')]
    for path in targets:
        if path.exists():raise FileExistsError(path)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch,Circle
    from matplotlib import font_manager
    names={f.name for f in font_manager.fontManager.ttflist}
    for name in ('Noto Sans CJK SC','Noto Sans CJK JP','WenQuanYi Zen Hei'):
        if name in names:plt.rcParams['font.family']=name;break
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','axes.unicode_minus':False})
    fig,(top,block)=plt.subplots(2,1,figsize=(14,8),dpi=300,gridspec_kw={'height_ratios':[.9,1.1]})
    for ax in (top,block):ax.set(xlim=(0,20.5),ylim=(0,4.6));ax.axis('off')
    def box(ax,x,y,w,h,text,fill='#eef4fb'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.035',edgecolor='#2a78d6',facecolor=fill,lw=1.5))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10,linespacing=1.5)
    def arrow(ax,x1,y1,x2,y2):
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops={'arrowstyle':'->','lw':1.5,'color':'#52514e','shrinkA':0,'shrinkB':0})
    top.text(0,4.2,'图0.1　从输入到分布：本册LLaMA式207M模型',fontsize=18,weight='bold')
    y=2.05;h=1.25
    nodes=[(.3,2.0,'token ID\n(B,n)'),(2.95,3.0,'输入嵌入\n权重(32000,1024)\n输出(B,n,1024)'),(6.6,2.65,'12层Block\n(B,n,1024)\n单层见下图'),(9.9,2.45,'末端RMSNorm\n(B,n,1024)'),(13.,3.05,'U：独立输出头\n权重(32000,1024)\nlogits(B,n,32000)'),(16.7,3.05,'softmax\n沿词表轴归一化\n概率(B,n,32000)')]
    for x,width,text in nodes:box(top,x,y,width,h,text)
    for (x,width,_),(nx,_,_) in zip(nodes,nodes[1:]):arrow(top,x+width,y+h/2,nx,y+h/2)
    top.text(.3,1.35,'入口不加位置查表；位置信息在每层注意力的Q/K上引入。输出头与输入嵌入不共享参数。',fontsize=11)
    top.text(.3,.65,'配置：L=12，d=1024，h=16，hkv=8，每头64维；FFN隐藏维2816，词表32000。',fontsize=11)
    block.text(0,4.2,'一层Block：两条pre-norm残差分支',fontsize=15,weight='bold')
    yy=2.0
    box(block,1.25,yy-.48,2.0,.96,'① RMSNorm\n(B,n,1024)')
    box(block,4.,yy-.7,3.1,1.4,'③ RoPE / ④ GQA\nQ：(B,16,n,64)\nK/V：(B,8,n,64)\n输出：(B,n,1024)')
    box(block,9.0,yy-.48,2.0,.96,'① RMSNorm\n(B,n,1024)')
    box(block,11.8,yy-.7,3.0,1.4,'② SwiGLU\n两路(B,n,2816)\n逐元素门控后投影\n输出：(B,n,1024)')
    for x in (7.9,15.6):
        block.add_patch(Circle((x,yy),.24,edgecolor='#52514e',facecolor='white',lw=1.5))
        block.text(x,yy,'+',ha='center',va='center',fontsize=15)
    for x1,x2 in [(0.3,1.25),(3.25,4.),(7.1,7.66),(8.14,9.),(11.,11.8),(14.8,15.36),(15.84,17.)]:arrow(block,x1,yy,x2,yy)
    block.plot([.65,.65,7.9],[yy,3.45,3.45],color='#52514e',lw=1.5);arrow(block,7.9,3.45,7.9,yy+.24)
    block.plot([8.5,8.5,15.6],[yy,.65,.65],color='#52514e',lw=1.5);arrow(block,15.6,.65,15.6,yy-.24)
    block.text(4.0,3.65,'保留分支输入，逐元素相加',ha='center',fontsize=10,color='#52514e')
    block.text(12.,.25,'第二条旁路使用第一次相加后的表示',ha='center',fontsize=10,color='#52514e')
    block.text(17.15,yy,'进入下一层\n(B,n,1024)',ha='left',va='center',fontsize=11)
    fig.text(.04,.015,'①函数替换保留pre-norm位置；②隐藏维按预算规则取整；③④同在注意力内。形状兼容不保证效果等价。',fontsize=10,color='#52514e')
    fig.tight_layout(rect=(0,.05,1,1));a.out_dir.mkdir(parents=True,exist_ok=True)
    for path in targets:
        fig.savefig(path,dpi=300,facecolor='white')
        if path.suffix=='.svg':path.write_text('\n'.join(s.rstrip() for s in path.read_text().splitlines())+'\n')
    plt.close(fig);print('已生成：',targets[0])


if __name__=='__main__':main()
