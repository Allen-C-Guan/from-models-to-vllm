"""第 3–5 章软件图；source env.sh 后运行，仅 CPU，图源输出 SVG+PNG。"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'drafts/Book8-vLLM源码剖析/figures'
FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
BLUE,ORANGE,GREEN,PURPLE,GRAY='#24659B','#B4611B','#20775F','#7055A1','#586575'
PALE={BLUE:'#EDF5FC',ORANGE:'#FFF4E7',GREEN:'#EAF6F0',PURPLE:'#F3EFFA',GRAY:'#F4F6F9'}
plt.rcParams.update({'svg.fonttype':'none','axes.unicode_minus':False})


def txt(ax,x,y,s,size=16,color='#1E293B',ha='center',bg=False):
    ax.text(x,y,s,fontproperties=FONT,fontsize=size,color=color,ha=ha,va='center',
        linespacing=1.5,bbox={'facecolor':'white','edgecolor':'none','pad':3} if bg else None,zorder=5)


def canvas(title,sub,h=850,footer=None):
    fig,ax=plt.subplots(figsize=(14,h/100));fig.subplots_adjust(0,0,1,1)
    ax.set_xlim(0,1400);ax.set_ylim(h,0);ax.axis('off');fig.patch.set_facecolor('white')
    txt(ax,40,35,title,23,ha='left');txt(ax,40,76,sub,14,GRAY,ha='left')
    txt(ax,40,h-30,footer or '依据 vLLM v0.30.0 源码；中文教学图。颜色：蓝＝前端，橙＝核心，紫＝执行，绿＝结果／资源。',12,GRAY,ha='left')
    return fig,ax


def box(ax,x,y,w,h,title,body='',c=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=9',
        facecolor=PALE[c],edgecolor=c,lw=1.6,zorder=2))
    txt(ax,x+w/2,y+h*(.3 if body else .5),title,17,c)
    if body:txt(ax,x+w/2,y+h*.7,body,14)


def arrow(ax,pts,label='',at=None,c=GRAY,dashed=False):
    for a,b in zip(pts[:-2],pts[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color=c,lw=1.7,ls='--' if dashed else '-',zorder=1)
    ax.add_patch(FancyArrowPatch(pts[-2],pts[-1],arrowstyle='-|>',mutation_scale=14,
        lw=1.7,color=c,linestyle='--' if dashed else '-',zorder=3))
    if label:
        if at is None:at=((pts[-1][0]+pts[-2][0])/2,(pts[-1][1]+pts[-2][1])/2-16)
        txt(ax,*at,label,13,c,bg=True)


def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/f'{name}.svg',metadata={'Date':None})
    svg=OUT/f'{name}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(OUT/f'{name}.png',dpi=150);plt.close(fig)


def lanes(ax,names,xs,bottom=735):
    for (name,c),x in zip(names,xs):
        box(ax,x-110,130,220,65,name,c=c)
        ax.plot([x,x],[195,bottom],ls='--',lw=1,color=c,zorder=0)


def main():
    fig,ax=canvas('图 3.1  本章把哪条边界打开','整机定位：打开输入与请求登记；执行层暂时保持折叠。')
    box(ax,70,180,250,100,'用户程序','文本 + SamplingParams',GRAY)
    box(ax,420,180,300,100,'前端输入','Renderer → InputProcessor',BLUE)
    box(ax,860,180,420,100,'EngineCore','Request → Scheduler → KV 管理',ORANGE)
    box(ax,860,450,420,100,'执行层','Executor → Worker → Runner',PURPLE)
    box(ax,420,450,300,100,'前端输出','OutputProcessor',GREEN)
    box(ax,70,450,250,100,'用户结果','RequestOutput',GRAY)
    arrow(ax,[(320,230),(420,230)],'输入')
    arrow(ax,[(720,230),(860,230)],'请求描述')
    arrow(ax,[(1010,280),(1010,450)],'执行计划',(950,365),ORANGE)
    arrow(ax,[(1150,450),(1150,280)],'token 结果',(1220,365),PURPLE)
    arrow(ax,[(860,255),(790,255),(790,500),(720,500)],'核心结果',(790,385),GREEN)
    arrow(ax,[(420,500),(320,500)],'交付')
    txt(ax,700,670,'图中是职责关系，不是进程划分；第 4 章再把这些对象放进实际部署。',17,GRAY)
    save(fig,'fig-3-1-location')

    fig,ax=canvas('图 3.2  一个请求，三种职责视图','实线箭头＝构造或交付；虚线＝通过 request_id 关联，不表示共享同一对象。')
    box(ax,400,140,600,135,'EngineCoreRequest','准备进入核心的请求描述\nprompt_token_ids · sampling_params · request_id',BLUE)
    box(ax,70,430,560,160,'核心 Request','Scheduler 持有可变进度\nstatus · num_computed_tokens · output_token_ids',ORANGE)
    box(ax,770,430,560,160,'输出侧 RequestState','OutputProcessor 持有交付状态\nexternal_req_id · detokenizer · output_kind',GREEN)
    arrow(ax,[(560,275),(560,340),(350,340),(350,430)],'核心侧构造',(360,330),ORANGE)
    arrow(ax,[(840,275),(840,340),(1050,340),(1050,430)],'前端登记时构造',(1050,330),GREEN)
    arrow(ax,[(630,515),(770,515)],'ID 关联',(700,485),GRAY,True)
    txt(ax,700,730,'三者不是继承链。它们保存与自己职责有关的信息，生命周期也不完全相同。',16,GRAY)
    save(fig,'fig-3-2-objects')

    fig,ax=canvas('图 3.3  接纳请求：先登记输出侧，再提交核心','单候选 n=1；省略参数验证内部细节。实线＝调用，虚线＝返回。')
    xs=[170,500,840,1190]
    lanes(ax,[('LLMEngine',BLUE),('InputProcessor',BLUE),('OutputProcessor',GREEN),('核心客户端／核心',ORANGE)],xs,610)
    arrow(ax,[(170,260),(500,260)],'process_inputs → 请求描述')
    arrow(ax,[(500,320),(170,320)],'EngineCoreRequest',dashed=True)
    arrow(ax,[(170,385),(500,385)],'assign_request_id')
    arrow(ax,[(170,455),(840,455)],'add_request → 输出侧 RequestState')
    arrow(ax,[(170,530),(1190,530)],'add_request → 核心 Request → 等待队列')
    txt(ax,700,665,'登记结束时还没有执行模型；输出侧已经知道未来收到这个 ID 时该交付给谁。',16,GRAY)
    save(fig,'fig-3-3-admission')

    fig,ax=canvas('图 4.2  EngineCore 协调一轮，不替每个组件完成工作','同步调度、无投机；折叠 Worker 内部调用与返回。纵向表示因果顺序，不表达耗时。',970)
    xs=[160,530,940,1240]
    lanes(ax,[('EngineCore',ORANGE),('Scheduler',ORANGE),('Executor',PURPLE),('Worker／Runner',PURPLE)],xs,870)
    for y,a,b,label,ret in [
        (250,0,1,'schedule()',False),(310,1,0,'SchedulerOutput',True),
        (380,0,2,'execute_model(plan, non_block=True)',False),
        (440,2,3,'准备输入并执行',False),(500,2,0,'Future',True),
        (640,0,2,'若结果为 None：sample_tokens(grammar)',False),
        (710,2,0,'ModelRunnerOutput',True),(790,0,1,'update_from_output(plan, result)',False),
        (850,1,0,'EngineCoreOutputs',True)]:
        arrow(ax,[(xs[a],y),(xs[b],y)],label,dashed=ret)
    arrow(ax,[(160,550),(260,550),(260,590),(160,590)],c=ORANGE)
    txt(ax,570,570,'本地 Future.result() → None（本例）',13,ORANGE,bg=True)
    save(fig,'fig-4-2-iteration')

    fig,ax=canvas('图 4.3  同一个进度字段，要连同时点一起读','基础运行第一轮，请求 A：5 个 prompt token；无前缀命中、无投机。',footer='依据 v0.30.0 源码与实际观测；本图颜色区分调度前、调度后、回填后三个时点。')
    box(ax,70,190,370,240,'① 调度之前','computed = 0\nin_flight = 0\n输出 token = []',BLUE)
    box(ax,515,190,370,240,'② schedule 返回','computed = 5\nin_flight = 5\n输出 token = []',ORANGE)
    box(ax,960,190,370,240,'③ 结果回填之后','computed = 5\nin_flight = 0\n输出 token = [12095]',GREEN)
    arrow(ax,[(440,310),(515,310)],'预记进度',(478,285),ORANGE)
    arrow(ax,[(885,310),(960,310)],'计算／回填',(923,285),GREEN)
    box(ax,300,530,800,130,'下一轮为什么还要计算 1 个 token？','已知序列长度变成 6，而 computed 仍是 5；新采样的 token 将成为下一轮输入。',PURPLE)
    txt(ax,700,730,'这里的 computed 是核心 Request.num_computed_tokens 的缩写；日志保留真实字段名。',14,GRAY)
    save(fig,'fig-4-3-progress')

    fig,ax=canvas('图 4.4  职责不变，调用边界会随部署变化','单设备配置按执行器选择源码映射；多进程对照只观测前端收发边界。',920)
    box(ax,60,170,1280,235,'教学配置：InprocClient，同一个 Python 进程','前端 LLMEngine  →  核心 EngineCore  →  UniProcExecutor / NPUWorker\nget_output() 直接推进 step_fn；因此可以在每轮之间提交 C。',BLUE)
    box(ax,60,495,440,210,'多进程配置：前端进程','LLMEngine + SyncMPClient\n提交请求／读取输出队列',BLUE)
    box(ax,760,495,580,210,'多进程配置：EngineCore 进程','EngineCoreProc + Scheduler\nUniProcExecutor + NPUWorker / Runner',ORANGE)
    arrow(ax,[(500,560),(760,560)],'请求消息（ZMQ）',c=ORANGE)
    arrow(ax,[(760,645),(500,645)],'核心输出消息',c=GREEN)
    txt(ax,700,790,'两种配置的 NPU 都执行模型算子；Worker 对象不是 NPU 设备本身。',17,PURPLE)
    save(fig,'fig-4-4-deployment')

    fig,ax=canvas('图 5.2  调度先形成可兑现的计划，再把工作交给执行层','只画样稿文本路径：无投机、无远端 KV、无多模态；其他等待原因另有分支。',970)
    box(ax,460,145,480,90,'本轮预算','token 数、活跃请求数、可分配 KV',ORANGE)
    box(ax,70,330,540,100,'先考察 running 请求','读取进度差；裁剪到本轮剩余 token 预算',ORANGE)
    box(ax,790,330,540,100,'再考察 waiting 请求','满足接纳条件后申请所需计算与状态空间',ORANGE)
    arrow(ax,[(580,235),(580,280),(340,280),(340,330)],'先处理',c=ORANGE)
    arrow(ax,[(610,380),(790,380)],'有余量再接纳',c=ORANGE)
    box(ax,460,540,480,110,'与 KV 管理器协作','allocate_slots：确认状态空间',GREEN)
    arrow(ax,[(340,430),(340,595),(460,595)],c=GREEN)
    arrow(ax,[(1060,430),(1060,595),(940,595)],c=GREEN)
    box(ax,70,740,540,100,'分配成功','登记本轮 token 数、块信息与新旧请求数据',PURPLE)
    box(ax,790,740,540,100,'分配失败','按所在队列处理：可能抢占，或等待后续轮次',GRAY)
    arrow(ax,[(580,650),(580,690),(340,690),(340,740)],'成功',c=GREEN)
    arrow(ax,[(820,650),(820,690),(1060,690),(1060,740)],'失败',c=GRAY)
    save(fig,'fig-5-2-decisions')

    fig,ax=canvas('图 5.3  SchedulerOutput 是一份执行协议','缓存请求表示执行侧已经认识它；这与 prompt 是否命中前缀缓存是两件事。')
    box(ax,70,170,390,260,'scheduled_new_reqs','首次接纳的 NewRequestData\nprompt_token_ids\nsampling_params\nblock_ids · num_computed_tokens',BLUE)
    box(ax,505,170,390,260,'scheduled_cached_reqs','已知请求的 CachedRequestData\nreq_ids · new_block_ids\nnum_computed_tokens\nnum_output_tokens',ORANGE)
    box(ax,940,170,390,260,'本轮工作与生命周期','num_scheduled_tokens\ntotal_num_scheduled_tokens\nfinished_req_ids\n以及启用功能所需的其他字段',GREEN)
    box(ax,350,600,700,100,'执行层据此更新本地状态，再构造当步输入','请求状态可跨轮保留；计算批次每轮变化。',PURPLE)
    for x in [265,700,1135]:
        arrow(ax,[(x,430),(x,510),(700,510),(700,600)] if x!=700 else [(700,430),(700,600)],c=PURPLE)
    save(fig,'fig-5-3-plan')

    path=ROOT/'log/book8-source-rewrite/chunked-inproc-traced/events.jsonl'
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    plans=[r for r in rows if r['event']=='schedule.after']
    h=360+len(plans)*76
    fig,ax=canvas('图 5.4  同一组请求，预算变小后怎样逐轮前进','真实运行：每轮最多 4 个输入 token；第 2 轮之后提交 C。数字表示本轮调度量。',h,footer='依据实际调度记录；本图颜色仅区分请求 A、B、C，不表示组件职责。')
    xs=[370,650,930]
    for x,rid,c in zip(xs,['A','B','C'],[BLUE,ORANGE,GREEN]):box(ax,x-110,135,220,60,rid,c=c)
    for i,r in enumerate(plans):
        y=240+i*76;txt(ax,130,y,f'第 {i+1} 轮',16,GRAY)
        nums={k.split('-')[0]:v for k,v in r['plan']['num_scheduled_tokens'].items()}
        for x,rid,c in zip(xs,['A','B','C'],[BLUE,ORANGE,GREEN]):
            n=nums.get(rid,0)
            if n:box(ax,x-110,y-25,220,50,str(n),c=c)
            else:txt(ax,x,y,'—',17,GRAY)
        txt(ax,1200,y,f'合计 {sum(nums.values())}',15,GRAY)
    txt(ax,700,h-95,'“—”只表示这一轮没有调度；可能尚未提交、仍在等待，或已经结束，需结合请求状态判断。',14,GRAY)
    save(fig,'fig-5-4-rounds')


if __name__=='__main__':
    main()
    from render_architecture import main as architecture_figures
    architecture_figures()
