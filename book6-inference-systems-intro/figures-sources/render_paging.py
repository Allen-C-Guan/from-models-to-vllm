"""第一组中文理论图。source ./env.sh 后运行；CPU SVG/PNG，可编辑图源。"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'
DATA=json.loads((ROOT/'verification/results/cases-verified.json').read_text())
FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
INK='#243449';BLUE='#236599';ORANGE='#a85c1b';GREEN='#24745b';PURPLE='#74509b';GRAY='#667587'
PALE={BLUE:'#edf5fc',ORANGE:'#fff2e4',GREEN:'#eaf6ef',PURPLE:'#f3edf9',GRAY:'#f1f3f6'}
plt.rcParams.update({'svg.fonttype':'none','axes.unicode_minus':False})

def txt(ax,x,y,s,size=15,color=INK,ha='center',bg=False):
    ax.text(x,y,s,fontproperties=FONT,fontsize=size,color=color,ha=ha,va='center',linespacing=1.6,zorder=6,
            bbox={'fc':'white','ec':'none','pad':2} if bg else None)

def canvas(title,sub,height=820):
    fig,ax=plt.subplots(figsize=(14,height/100));fig.subplots_adjust(0,0,1,1)
    ax.set(xlim=(0,1400),ylim=(height,0));ax.axis('off');fig.patch.set_facecolor('white')
    txt(ax,45,38,title,23,ha='left');txt(ax,45,84,sub,14,GRAY,ha='left')
    txt(ax,45,height-32,'中文教学示意 · 数字为构造例 · 算法依据与简化条件见正文 · 不表示 vLLM 实际布局或硬件实测',12,GRAY,ha='left')
    return fig,ax

def box(ax,x,y,w,h,title,body='',color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=8',fc=PALE[color],ec=color,lw=1.6,zorder=2))
    txt(ax,x+w/2,y+h*(.3 if body else .5),title,17,color)
    if body:txt(ax,x+w/2,y+h*.72,body,14)

def arrow(ax,a,b,label='',at=None,color=GRAY):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,color=color,lw=1.8,zorder=3))
    if label:
        x,y=at or ((a[0]+b[0])/2,(a[1]+b[1])/2-17)
        txt(ax,x,y,label,13,color,bg=True)

def route(ax,points,color=GRAY):
    for a,b in zip(points[:-2],points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=1.8,zorder=3)
    arrow(ax,points[-2],points[-1],color=color)

def save(fig,name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT/f'{name}.svg',metadata={'Date':None})
    p=OUT/f'{name}.svg';p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')
    fig.savefig(OUT/f'{name}.png',dpi=140);plt.close(fig)

def cells(ax,x,y,labels,valid=None,width=100,color=GREEN):
    for i,label in enumerate(labels):
        c=color if valid is None or i<valid else GRAY
        ax.add_patch(Rectangle((x+i*width,y),width,60,fc=PALE[c],ec=c,lw=1.5,zorder=2))
        txt(ax,x+(i+.5)*width,y+30,label,14,c)

def maps():
    groups=[('工作负载与成本','01 服务指标\n02 存储层次 · 03 成本估算'),('状态与存储','04 KV状态 · 05 地址分页\n06 分页KV · 07 前缀复用'),('批处理与调度','08 批处理 · 09 排队\n10 分块预填充与抢占'),('计算与表示','11 Attention IO\n12 低精度 · 13 权重 · 14 KV压缩'),('采样与投机','15 概率语义\n16 投机正确性 · 17 收益与候选'),('执行与系统判断','18 主机设备协同\n19 并行拓扑 · 20 综合判断')]
    for chapter in [5,6]:
        f,a=canvas(f'图 {chapter}.1  全书理论地图：本章展开状态与存储','同一位置、同一坐标；灰色为全书背景，橙色为本章范围。',850)
        for i,(t,b) in enumerate(groups):
            x=60+(i%3)*445;y=145+(i//3)*190
            box(a,x,y,390,140,t,b,ORANGE if i==1 else GRAY)
            if i%3<2:arrow(a,(x+390,y+70),(x+445,y+70))
        txt(a,700,525,f'本章展开：第{chapter}章'+('地址与分配' if chapter==5 else 'KV映射与生命周期'),17,ORANGE)
        arrow(a,(700,552),(700,582),color=ORANGE)
        if chapter==5:
            labels=['连续分配的困难','地址 → 页表 → 页框','共享、写隔离与成本']
        else:labels=['token → 块表 → 槽位','Attention与有效范围','增长、共享、完成、回收']
        for i,label in enumerate(labels):
            x=60+i*445;box(a,x,590,390,95,label,color=ORANGE)
            if i<2:arrow(a,(x+390,637),(x+445,637))
        txt(a,700,750,'箭头表示理解关系；不是请求调用顺序，也不是软件组件或进程拓扑。',15,GRAY)
        save(f,f'fig-{chapter:02d}-01-map')

def fragmentation():
    f,a=canvas('图 5.2  空闲总量足够，不代表连续空间足够','上：保持连续24字节要求；下：改成两个16字节单位及一份映射。',720)
    for i in range(5):
        x=100+i*220;c=GREEN if i%2==0 else GRAY
        box(a,x,170,210,100,'空闲16字节' if i%2==0 else '仍被占用',f'[{i*16},{(i+1)*16})',c)
    txt(a,700,325,'空闲总量 48 字节；最大连续空闲段 16 字节 → 连续 24 字节分配失败',18,ORANGE)
    box(a,100,420,280,105,'对象的逻辑内容','24字节，分为16＋8',BLUE)
    box(a,520,420,300,105,'第一个单位','16字节有效',GREEN)
    box(a,950,420,300,105,'第二个单位','8字节有效＋8字节余量',ORANGE)
    arrow(a,(380,470),(520,470),'映射0')
    route(a,[(380,510),(420,510),(420,570),(1100,570),(1100,525)])
    txt(a,680,552,'映射1',13,GRAY)
    txt(a,700,620,'两个单位可来自不相邻的空闲段；只改变存储分配方式，不改变逻辑内容顺序。',16)
    save(f,'fig-05-02-fragmentation')

def translation():
    f,a=canvas('图 5.3  地址翻译保留偏移，替换页号','构造：64字节虚拟空间，16字节/页；页表 [5,1,6,3]。',780)
    box(a,70,180,300,120,'虚拟地址 37','页号2 · 偏移5',BLUE)
    box(a,540,150,310,270,'页表 T','0 → 5\n1 → 1\n2 → 6\n3 → 3',ORANGE)
    box(a,1000,180,300,120,'物理地址 101','帧6起点96＋偏移5',GREEN)
    arrow(a,(370,225),(540,225),'查页2');arrow(a,(850,225),(1000,225),'得到帧6')
    txt(a,700,465,'二进制：10 | 0101  →  查表  →  110 | 0101；低4位偏移不变',17)
    box(a,70,540,540,110,'连续的虚拟地址','30 → 31 → 32 → 33',BLUE)
    box(a,790,540,540,110,'跨过页边界后的物理地址','30 → 31 → 96 → 97',GREEN)
    arrow(a,(610,595),(790,595),'分别翻译')
    save(f,'fig-05-03-translation')

def os_cow():
    f,a=canvas('图 5.4  同一逻辑内容可以先共享，私有写入前再分离','概念示意：原本可写的私有页被标记为COW；真正只读页不能因此随意写入。')
    box(a,80,170,250,100,'A页表','页2 → 帧6',BLUE)
    box(a,80,360,250,100,'C页表','页2 → 帧6',BLUE)
    box(a,450,245,250,150,'共享帧6','[10,20,30,40,…]\n引用数2',PURPLE)
    arrow(a,(330,220),(450,280));arrow(a,(330,410),(450,355))
    arrow(a,(730,320),(850,320),'C写入前',(790,285),ORANGE)
    box(a,900,170,390,120,'A仍使用帧6','偏移1仍为20 · 引用数1',PURPLE)
    box(a,900,360,390,120,'C改用新帧4','先复制，再将偏移1写为99',GREEN)
    txt(a,700,570,'C的虚拟地址不变；只有C的映射改变。A的私有内容不受C写入影响。',18)
    txt(a,700,650,'写保护触发 → 确认COW → 准备新帧并复制 → 更新映射/翻译 → 重试写入',16,GRAY)
    save(f,'fig-05-04-cow')

def block_table():
    f,a=canvas('图 6.2  逻辑顺序由序列给出，物理位置由块表决定','块大小4；A已计算6个位置，B已计算3个位置。其他9个物理块空闲。',830)
    box(a,60,180,300,110,'A：位置0–5','T_A = [7,2] · S_A = 6',BLUE)
    box(a,60,500,300,110,'B：位置0–2','T_B = [5] · S_B = 3',BLUE)
    for y,p,labels,valid in [(170,7,['a','b','c','d'],4),(350,2,['e','f','无效','无效'],2),(530,5,['u','v','w','无效'],3)]:
        txt(a,1030,y-25,f'物理块{p}',17,ORANGE)
        cells(a,820,y,labels,valid,105)
        txt(a,1030,y+90,'槽位0              1              2              3',12,GRAY)
    arrow(a,(360,210),(800,200),'逻辑块0 → 7',(540,175))
    arrow(a,(360,265),(800,380),'逻辑块1 → 2',(530,350))
    arrow(a,(360,555),(800,560),'逻辑块0 → 5',(530,525))
    txt(a,700,705,'位置5：5÷4商1余1 → T_A[1]=2 → 物理块2、槽位1',18)
    txt(a,700,750,'位置6虽可算出地址，但当前 S_A=6，尚未成为可读的有效历史。',15,GRAY)
    save(f,'fig-06-02-block-table')

def attention():
    f,a=canvas('图 6.3  分块存放，不分割归一化语义','独立变体：b=2，单层单头，q形状(1,1)，仅3个有效位置。',850)
    box(a,70,155,330,100,'逻辑块表 [4,1]','块0→物理4；块1→物理1',BLUE)
    box(a,540,140,345,145,'物理块4 · 2项有效','K=[0, ln2]  V=[10,20]\nK/V形状(2,1)',GREEN)
    box(a,970,140,345,145,'物理块1 · 仅首槽有效','K=[ln6, ×]  V=[40, ×]\n有效K/V形状(1,1)',GREEN)
    arrow(a,(400,195),(540,195))
    route(a,[(400,175),(450,175),(450,115),(1140,115),(1140,140)])
    box(a,540,395,345,120,'块0的归约','分子 u0=50 · 分母 t0=3',ORANGE)
    box(a,970,395,345,120,'块1的归约','分子 u1=240 · 分母 t1=6',ORANGE)
    arrow(a,(710,285),(710,395),'有效项得分',(620,340))
    arrow(a,(1140,285),(1140,395),'屏蔽无效槽',(1240,340))
    box(a,540,620,775,100,'合并后才完成同一归一化','o=(50+240)/(3+6)=290/9 ≈ 32.2222；输出形状(1,1)',GREEN)
    arrow(a,(710,515),(800,620));arrow(a,(1140,515),(1080,620))
    box(a,65,420,340,190,'错误反例','每块先独立softmax\n然后等权平均\n(50/3+40)/2 ≈ 28.3333',GRAY)
    save(f,'fig-06-03-attention')

def kv_cow():
    f,a=canvas('图 6.4  分歧写入只分离共享尾块','从A长度6的初态重置；B始终独占物理块5，图中省略。',850)
    for x,t in [(65,'分离之前'),(770,'A计算位置6之后')]:txt(a,x,145,t,19,ha='left')
    box(a,60,205,230,85,'A：[7,2]',color=BLUE);box(a,60,415,230,85,'C：[7,2]',color=BLUE)
    box(a,400,190,250,120,'满块7','a b c d · 引用2',PURPLE)
    box(a,400,400,250,120,'尾块2','e f □ □ · 引用2',PURPLE)
    arrow(a,(290,230),(400,240));arrow(a,(290,270),(400,430));arrow(a,(290,435),(400,285));arrow(a,(290,475),(400,475))
    box(a,770,200,230,85,'A：[7,0]',color=BLUE);box(a,770,490,230,85,'C：[7,2]',color=BLUE)
    box(a,1110,145,245,105,'满块7','a b c d · 引用2',PURPLE)
    box(a,1110,325,245,105,'新尾块0','e f x □ · 引用1',GREEN)
    box(a,1110,510,245,105,'旧尾块2','e f □ □ · 引用1',PURPLE)
    arrow(a,(1000,220),(1110,200));arrow(a,(1000,260),(1110,365));arrow(a,(1000,515),(1110,230));arrow(a,(1000,555),(1110,555))
    txt(a,700,700,'先为A复制尾块中的有效KV，再写入x；C的下一次写入可使用已经独占的块2。',16)
    txt(a,700,755,'方格是一个位置的全层KV身份，不是把token字符本身存进KV。',14,GRAY)
    save(f,'fig-06-04-cow')

def lifecycle():
    f,a=canvas('图 6.5  可定位、可读取与可回收，是不同状态','串行教学视图；实际系统可用等待、依赖或延迟回收保证相同约束。',840)
    box(a,60,170,260,110,'空闲','可重新分配',GRAY)
    box(a,520,170,300,110,'已预留','有物理位置；新数据尚无效',ORANGE)
    box(a,1020,170,300,110,'有效且独占','已完成所需写入',GREEN)
    arrow(a,(320,225),(520,225),'取得资源');arrow(a,(820,225),(1020,225),'计算完成')
    box(a,1020,440,300,110,'有效且共享','多个序列持有引用',PURPLE)
    arrow(a,(1110,280),(1110,440),'增加\n共享引用',(1045,360))
    arrow(a,(1250,440),(1250,280),'其他引用\n退出',(1320,360))
    box(a,520,440,300,110,'无引用但访问仍在途','暂不能复用',ORANGE)
    arrow(a,(1020,485),(820,485),'持有者全部退出',(920,455))
    arrow(a,(1120,280),(780,440),'最后引用退出\n仍有访问在途',(870,350))
    arrow(a,(520,490),(200,280),'访问结束后归还',(320,410))
    txt(a,700,645,'若没有在途访问，最后一个引用退出后即可回收；空闲不意味着内容已清零。',16)
    txt(a,700,710,'本章不保留闲置前缀缓存；第7章将增加“无活跃引用但仍有复用价值”的状态。',15,GRAY)
    save(f,'fig-06-05-lifecycle')

if __name__=='__main__':
    # 固定教学标注若与算例变更不一致，停止渲染，防止旧图悄悄复用。
    assert DATA['growth'][0]['tables']=={'A':[7,2],'B':[5]}
    assert [s['free_count'] for s in DATA['shared_branch']]==[9,8,8,9,11]
    assert abs(DATA['attention']['paged']-290/9)<1e-10
    assert DATA['os_addresses']['cross_page']==[30,31,96,97]
    maps();fragmentation();translation();os_cow();block_table();attention();kv_cow();lifecycle()
    print('9 figures: editable SVG and PNG')
