"""第11章投机解码与PD详细教学图；CPU制图，无硬件实验。
在工作区 source ./env.sh 后运行。图的实现依据与边界见正文图注。
"""
from pathlib import Path
from render_figures import canvas, group, text, save, BLUE, ORANGE, GREEN, PURPLE, MUTED
from render_ch10 import card, edge


def speculation():
    fig,ax=canvas('图 11.2  单卡投机：原闭环里增加提议、验证与校正',
        '同步调度＋n-gram提议；实线为调用／交接，绿色虚线为下一轮候选。CPU职责框不等于进程。',1580,1700)
    group(ax,35,125,440,1280,'普通解码 · 一套引擎',MUTED)
    rows=[('EngineCore','驱动本地闭环',ORANGE),('Scheduler ↔ KV 管理','本例每轮安排 1 个输入',ORANGE),('Executor → Worker','调用本地 Runner',PURPLE),('Runner','准备输入、前向、普通采样',PURPLE),('模型／普通 Sampler','1 行 logits → 1 个输出',PURPLE)]
    for i,(a,b,c) in enumerate(rows):
        y=210+i*190
        card(ax,70,y,370,115,a,b,c)
        if i:edge(ax,[(255,y-75),(255,y)],color=ORANGE if i==1 else PURPLE)
    card(ax,70,1225,370,110,'NPU 0','目标权重／KV；执行算子',PURPLE)
    edge(ax,[(255,1085),(255,1225)],'提交计算',(255,1160))
    group(ax,535,125,1120,1075,'开启投机 · 仍为一套引擎／一张卡 · CPU 控制对象',MUTED)
    card(ax,680,185,760,90,'EngineCore','step 驱动执行；post_step 取回候选并交调度器',ORANGE)
    card(ax,680,330,760,105,'Scheduler ↔ KVCacheManager','Request 分存已确认 token／候选；验证输入 4；安排槽位',ORANGE)
    edge(ax,[(1000,275),(1000,330)],'调用 schedule',(900,302),ORANGE)
    card(ax,680,500,760,65,'Executor','传 SchedulerOutput；取回结果与候选',ORANGE)
    card(ax,680,610,760,65,'Worker','管理本地设备角色；调用 Runner',PURPLE)
    edge(ax,[(1000,565),(1000,610)])
    edge(ax,[(1000,435),(1000,500)],'A:4；候选 [41,42,43]；块表',(1050,469),ORANGE)
    group(ax,600,750,970,410,'Runner · 本地输入与结果状态',PURPLE)
    card(ax,635,815,450,100,'目标模型的输入／前向组织','[40,41,42,43] → logits (4,V)',PURPLE)
    card(ax,1140,815,390,100,'n-gram 提议器 · CPU','查已接受历史 → 下一轮候选',GREEN)
    card(ax,635,1000,450,100,'拒绝采样与本地状态更新','接受 41、42；拒绝 43；纠正 99',PURPLE)
    card(ax,1140,1000,390,100,'区分两份返回内容','已确认 [41,42,99]\n候选另经 take_draft_token_ids',BLUE)
    edge(ax,[(860,675),(860,815)],'计划 → 批次／验证索引',(860,715))
    edge(ax,[(860,915),(860,1000)],'多行预测',(860,955))
    edge(ax,[(1085,1050),(1140,1050)],color=BLUE)
    edge(ax,[(1335,1000),(1335,915)],'已确认历史',(1400,955),GREEN)
    edge(ax,[(1530,865),(1615,865),(1615,230),(1440,230)],'候选经 Worker／Executor\n供下一轮调度',(1500,320),GREEN,True)
    edge(ax,[(1140,1080),(555,1080),(555,230),(680,230)],'已确认结果\n经 Worker／Executor\n回核心更新进度',(610,515),BLUE)
    card(ax,680,1280,760,110,'同一张 NPU 0','目标权重／KV；目标前向与拒绝采样张量运算',PURPLE)
    edge(ax,[(950,1160),(950,1280)],'Runner 提交设备操作',(1080,1230))
    text(ax,850,1460,'本例 n-gram 无草稿模型；换成模型提议器时，可在 Runner 内增加草稿权重、计算与状态。',17,MUTED)
    text(ax,850,1515,'候选不是用户输出；Executor／Worker 不负责逐 token 判断候选正确与否。',17,BLUE)
    save(fig,'fig-11-2-speculation')


def verification():
    fig,ax=canvas('图 11.3  一次验证：四行输入，三个已确认输出',
        '提示词 [40,41,42,43] 已处理；首个输出 40 已确认但未处理。贪心模式，编号为教学值。',1240,1500)
    card(ax,70,125,1360,100,'Runner 组织目标输入','input_ids=[40,41,42,43] (4,)；positions=[4,5,6,7] (4,)',ORANGE)
    edge(ax,[(750,225),(750,280)])
    card(ax,70,280,1360,100,'目标模型一次因果前向','嵌入 (4,H) → 各层 (4,H) → 输出头 logits (4,V)',PURPLE)
    headers=['输入行／位置','本行输入','预测对象','构造的目标选择／用途']
    xs=[70,350,630,910]; widths=[245,245,245,520]
    for x,w,t in zip(xs,widths,headers):card(ax,x,440,w,60,t,color=MUTED)
    data=[['行 0／位置 4','40（已确认）','候选 41','选择 41 → 接受'],['行 1／位置 5','41（候选）','候选 42','选择 42 → 接受'],['行 2／位置 6','42（候选）','候选 43','选择 99 → 拒绝 43，纠正'],['行 3／位置 7','43（候选）','bonus token','含错误前缀 43 → 本例弃用']]
    for i,row in enumerate(data):
        for x,w,t in zip(xs,widths,row):card(ax,x,520+i*80,w,60,t,color=GREEN if i<2 else ORANGE)
    edge(ax,[(750,380),(750,440)])
    card(ax,70,900,640,110,'向用户提交 [41,42,99]','原有输出 [40] → 已生成长度 4',BLUE)
    card(ax,790,900,640,110,'校正已处理长度：4 → 8 → 7','安排4输入后登记8；拒绝1候选，减回7',ORANGE)
    edge(ax,[(370,820),(370,900)],color=BLUE)
    edge(ax,[(1110,820),(1110,900)],color=ORANGE)
    card(ax,70,1060,1360,100,'最终已知长度 8，只有前 7 个位置具有有效目标 KV','位置 7 曾写 43 的 KV；下一轮在位置 7 输入 99 并覆盖。无需把整份缓存删除。',GREEN)
    text(ax,750,1200,'因果掩码保证前面的预测看不到未来候选；首次拒绝后的预测因条件前缀不成立而弃用。',16,MUTED)
    save(fig,'fig-11-3-verification')


def pd():
    fig,ax=canvas('图 11.4  PD 完整拓扑：两套引擎，交接同一条请求的 KV',
        'Mooncake 拉取式；P、D 各单卡同布局。框是职责容器；UniProcExecutor 下本地对象可同进程。',2010,1840)
    card(ax,540,125,760,105,'用户 → PD 代理 → 用户','先请求 P；再把原请求＋传输定位信息交 D；转发 D 输出',BLUE)
    for side,x in [('P',40),('D',1040)]:
        is_p=side=='P'
        group(ax,x,310,760,1130,('P：预填充实例' if is_p else 'D：解码实例')+' · CPU 控制对象',MUTED)
        card(ax,x+35,370,690,80,'前端／核心客户端','请求→核心；结果／传输参数→代理' if is_p else '原请求＋远端定位→核心；已确认结果→代理',BLUE)
        group(ax,x+25,490,710,495,'EngineCore · 驱动本地计算／状态推进',ORANGE)
        card(ax,x+55,550,650,100,'Scheduler ↔ KVCacheManager','安排提示词；保留源块 [7,2]' if is_p else '分配目标块 [4,9]；等待远端 KV 就绪',ORANGE)
        card(ax,x+55,700,650,100,'调度侧 Connector','结束时延迟释放；生成 kv_transfer_params' if is_p else '匹配远端长度；组合远端源块／本地目标块',ORANGE)
        card(ax,x+55,870,650,75,'Executor','调用本实例 Worker；回传模型／传输结果',ORANGE)
        edge(ax,[(x+300,450),(x+300,550)],'核心请求',(x+300,475),BLUE)
        edge(ax,[(x+380,650),(x+380,700)],color=GREEN)
        edge(ax,[(x+380,800),(x+380,870)],'SchedulerOutput＋连接器元数据',(x+380,837),ORANGE)
        card(ax,x+55,1035,650,60,'Worker','本地设备角色；调用 Runner',PURPLE)
        card(ax,x+55,1140,650,60,'Runner','组织模型计算；绑定传输元数据并收集完成状态',PURPLE)
        edge(ax,[(x+380,1095),(x+380,1140)])
        edge(ax,[(x+380,945),(x+380,1035)],'execute_model',(x+380,1000))
        card(ax,x+55,1260,300,145,'模型执行','完整模型的 prefill\n写提示词 KV' if is_p else '加载后接续模型\n写新增 KV',PURPLE)
        card(ax,x+395,1260,310,145,'执行侧 Connector','注册 KV／握手\n等待读取完成' if is_p else 'start_load_kv\n接收线程／完成轮询',GREEN)
        edge(ax,[(x+205,1200),(x+205,1260)],color=PURPLE)
        edge(ax,[(x+550,1200),(x+550,1260)],color=GREEN)
        edge(ax,[(x+705,1380),(x+745,1380),(x+745,600),(x+705,600)],color=GREEN)
        card(ax,x+55,1530,650,170,('NPU P' if is_p else 'NPU D')+' · 设备内存与计算','完整权重；KV 块 [7,2]' if is_p else '完整权重；接收 KV 至块 [4,9]',PURPLE)
        edge(ax,[(x+205,1405),(x+205,1530)],'模型算子',(x+205,1490))
        edge(ax,[(x+550,1405),(x+550,1530)],'注册／传输任务',(x+550,1490),GREEN)
    edge(ax,[(620,230),(620,270),(350,270),(350,370)],'① 提示词；P 只生成1步',(300,245),BLUE)
    edge(ax,[(500,370),(500,300),(790,300),(790,230)],'② KV 定位参数',(665,290),BLUE)
    edge(ax,[(1140,230),(1140,270),(1430,270),(1430,370)],'③ 原请求＋定位参数',(1510,245),BLUE)
    edge(ax,[(1570,370),(1570,295),(1240,295),(1240,230)],'⑥ D 的输出',(1470,308),BLUE)
    edge(ax,[(1590,1405),(1590,1440),(920,1440),(920,1330),(745,1330)],'④ 执行侧握手／\n读取完成通知',(920,1250),GREEN,True)
    edge(ax,[(745,1615),(1095,1615)],'⑤ KV 数值：P → D',(920,1570),GREEN)
    text(ax,920,1765,'Mooncake TransferEngine／Ascend 后端：按已注册地址搬运 KV；不经过代理或 EngineCore 的消息体。',16,GREEN)
    text(ax,920,1840,'完成状态经 Runner → Worker → Executor → Core／Scheduler 回传；D 接收完成后放行，P 发送完成后解除延迟释放。',16,MUTED)
    text(ax,920,1925,'P 与 D 各有完整模型；这不是 PP 的模型前后段。两侧有独立块池；这也不是只路由不同请求的普通 DP。',16,MUTED)
    save(fig,'fig-11-4-pd')


def pd_sequence():
    fig,ax=canvas('图 11.5  一条请求的 PD 交接：安排、搬运、就绪、接续',
        '主线是 Ascend Mooncake 拉取式；列为逻辑参与方，箭头表示因果顺序，不表示耗时。',1790,1760)
    xs=[160,510,860,1210,1560]
    titles=['代理／前端','P Core／Scheduler','P Runner／连接器','D Core／Scheduler','D Runner／连接器']
    for x,title in zip(xs,titles):
        card(ax,x-140,130,280,85,title,color=ORANGE if 'Core' in title else BLUE if x==160 else GREEN)
        ax.plot([x,x],[220,1670],color=MUTED,ls='--',lw=1,zorder=0)
    events=[(0,1,'① 原提示词；P max_tokens=1',280,BLUE),(1,2,'② 计划 → 前向；写提示词 KV',385,ORANGE),(2,1,'③ 模型结果；源块暂缓释放',490,BLUE),(1,0,'④ 返回远端身份／源块／端点',595,BLUE),(0,3,'⑤ 原请求＋kv_transfer_params',700,BLUE),(3,4,'⑥ 分配目标块；等待远端 KV；下发加载元数据',805,ORANGE),(4,2,'⑦ 握手并发起拉取',910,GREEN),(2,4,'⑧ KV 数值 P 设备 → D 设备（传输后端）',1015,GREEN),(4,2,'⑨ 读取结束通知：源数据不再被本次读取使用',1120,GREEN),(4,3,'⑩ finished_recving：恢复可调度',1225,GREEN),(2,1,'finished_sending：解除延迟释放',1310,GREEN),(3,4,'⑪ 重算末提示词 token → 采样 → 后续解码',1400,ORANGE)]
    for a,b,label,y,color in events:edge(ax,[(xs[a],y),(xs[b],y)],label,((xs[a]+xs[b])/2,y-27),color)
    edge(ax,[(xs[4],1500),(xs[3],1500)],'⑫ 已确认 token；更新 D 请求进度',(1385,1473),BLUE)
    edge(ax,[(xs[3],1600),(xs[0],1600)],'⑬ 经前端返回代理，流式交付用户；结束后回收 D 状态',(685,1573),BLUE)
    text(ax,880,1720,'D 在⑥到⑩之间不能让该请求读取未就绪的 KV；期间仍可处理其他就绪请求。',17,MUTED)
    save(fig,'fig-11-5-pd-sequence')


def pd_blocks():
    fig,ax=canvas('图 11.6  传的是 KV 数值：源块号与目标块号可以不同',
        '一个 KV 组，块长2，提示词 [11,12,21,22]；槽位＝块号×块长＋偏移。均为教学构造。',1210,1600)
    card(ax,60,140,610,100,'P 的块表：[7,2]','本地槽位 [14,15,4,5]',ORANGE)
    card(ax,930,140,610,100,'D 的块表：[4,9]','本地槽位 [8,9,18,19]',ORANGE)
    rows=[('位置0：token 11','块7／偏移0／槽位14','块4／偏移0／槽位8'),('位置1：token 12','块7／偏移1／槽位15','块4／偏移1／槽位9'),('位置2：token 21','块2／偏移0／槽位4','块9／偏移0／槽位18'),('位置3：token 22','块2／偏移1／槽位5','块9／偏移1／槽位19')]
    for i,(lab,a,b) in enumerate(rows):
        y=300+i*115
        card(ax,60,y,610,80,a,lab,PURPLE)
        card(ax,930,y,610,80,b,lab,PURPLE)
        edge(ax,[(670,y+40),(930,y+40)],'各层 K、V 数值',(800,y+10),GREEN)
    card(ax,60,820,1480,110,'每层逻辑张量：K (1,2,4,4)，V (1,2,4,4)','维度：请求数1 × KV头数2 × 历史位置4 × 头维4；实际分页布局依后端而定',GREEN)
    card(ax,60,980,1480,100,'两层、FP16 有效数值合计 256 字节','2层 × K/V两份 × 4位置 × 2头 × 4维 × 2字节；不含协议、填充、额外元数据',GREEN)
    text(ax,800,1150,'权重两端各自加载；Q重新计算；KV不包含末位置的词表logits。D仍使用自己的块表。',17,MUTED)
    save(fig,'fig-11-6-pd-blocks')


def main():
    if Path(__file__).resolve().parent.name == 'code':
        import render_figures
        render_figures.OUT = Path(__file__).resolve().parent.parent / 'figures'
    for draw in [speculation, verification, pd, pd_sequence, pd_blocks]:
        draw()


if __name__ == '__main__':
    main()
