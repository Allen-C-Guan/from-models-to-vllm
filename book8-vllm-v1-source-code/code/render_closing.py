"""跨边界协议、诊断与全册收束图。"""
from render_figures import canvas,box,txt,arrow,save,BLUE,ORANGE,GREEN,PURPLE,GRAY
from render_architecture import maps,klass,relation

def connector():
    maps((26,))
    fig,ax=canvas('图 26.1  相同接口，两侧实例与消息契约','UML类型与关联视图；SCHEDULER/WORKER是进程职责，producer/consumer是传输方向。',1440)
    klass(ax,420,150,560,'KVConnectorBase_V1','role · 当前轮metadata','查询 / 分配通知 / load-save / 完成',GREEN,True)
    klass(ax,60,540,540,'具体Connector：调度侧','请求状态 / 传输任务','get matched / build meta',ORANGE)
    klass(ax,800,540,540,'具体Connector：执行侧','缓存视图 / 在途传输','start load / save / get results',PURPLE)
    relation(ax,[(330,540),(330,430),(560,430),(560,360)],'继承接口',(365,430),'generalization')
    relation(ax,[(1070,540),(1070,430),(840,430),(840,360)],'继承接口',(1035,430),'generalization')
    box(ax,60,1000,540,150,'Scheduler持有调度侧实例','计划→KVConnectorMetadata\n接收←KVConnectorOutput',ORANGE)
    box(ax,800,1000,540,150,'ActiveKVConnector持有执行侧实例','pre_forward / post_forward\n无前向也推进传输',PURPLE)
    relation(ax,[(330,1000),(330,750)],'关联 0..1',(425,875))
    relation(ax,[(1070,1000),(1070,750)],'关联 1',(1165,875))
    relation(ax,[(600,630),(800,630)],'任务 / 完成消息',(700,590),'dependency')
    txt(ax,700,1290,'Example / NIXL / Offloading是具体实现选择；两侧不共享一个Python实例。',15,GRAY)
    save(fig,'fig-26-1-classes')
    fig,ax=canvas('图 26.2  接收就绪与源块回收是不同确认','协议时序示意：竖线为参与者；虚线任务/完成，粗线数据；省略握手与分片细节。',1380)
    xs=[150,510,900,1250];names=['P核心/源状态','传输执行','D Worker','D Scheduler']
    for x,n in zip(xs,names):
        txt(ax,x,165,n,15,GRAY);ax.plot([x,x],[220,1220],ls='--',c='#b5bec9',lw=1)
    arrow(ax,[(1250,300),(900,300)],'分配目标块 / load任务',(1060,270),dashed=True)
    txt(ax,1250,400,'等待远端',13,ORANGE)
    arrow(ax,[(150,450),(510,450)],'源数据已可用',(330,420))
    arrow(ax,[(510,580),(900,580)],'状态张量：源slot → 目标slot',(700,545))
    ax.plot([510,900],[580,580],c=GREEN,lw=3)
    txt(ax,150,650,'生成可先结束\n源块继续保留',13,ORANGE)
    arrow(ax,[(900,770),(1250,770)],'finished_recving',(1070,735),dashed=True)
    txt(ax,1250,875,'提交可用状态\n恢复调度',13,GREEN)
    arrow(ax,[(510,990),(150,990)],'finished_sending',(330,955),dashed=True)
    txt(ax,150,1090,'解除传输引用\n允许回收',13,GREEN)
    txt(ax,700,1290,'取消用户请求也不能跳过尚未完成的读写依赖；先结束请求，再安全释放资源。',15,GRAY)
    save(fig,'fig-26-2-handoff')



def metrics():
    maps((27,))
    fig,ax=canvas('图 27.1  请求级状态、当次聚合与记录出口','UML关联与依赖：不同生命周期的统计对象不承担调度决策。',1430)
    klass(ax,60,150,540,'RequestStateStats','arrival / 首末token / 抢占计数','跨多次输出持有',BLUE)
    klass(ax,800,150,540,'IterationStats','当次token / 间隔 / 结束列表','update_from_output / events',BLUE)
    relation(ax,[(800,255),(600,255)],'更新依赖',(700,220),'dependency')
    klass(ax,800,530,540,'FinishedRequestStats','e2e / queue / prefill / decode','请求完成时的汇总',GREEN)
    relation(ax,[(1070,360),(1070,530)],'生成并持有 0..*',(1190,445))
    klass(ax,60,530,540,'StatLoggerManager','stat_loggers','record / log',ORANGE)
    klass(ax,60,960,540,'StatLoggerBase','统计记录接口','record()',ORANGE,True)
    klass(ax,800,960,540,'Logging / Prometheus实现','聚合 / per-engine输出','record / log',GREEN)
    relation(ax,[(330,740),(330,960)],'持有实现列表',(440,850))
    relation(ax,[(800,1065),(600,1065)],'接口继承链（折叠）',(700,1025),'generalization')
    relation(ax,[(600,635),(800,635)],'消费汇总',(700,595),'dependency')
    txt(ax,700,1310,'核心产出事件；前端按请求合并，再把统计对象交给可替换logger。',15,GRAY)
    save(fig,'fig-27-1-classes')
    fig,ax=canvas('图 27.2  时间区间先确认时钟域','全部数字为CPU定义实验的构造值，不是性能实测。上下两行的原点不同。',1100)
    box(ax,80,160,1240,180,'前端墙钟','arrival 100.0 → 首结果100.5 → 最终结果100.8\nTTFT = 0.5；E2E = 0.8',BLUE)
    box(ax,80,480,1240,210,'核心单调时钟','QUEUED 10.0 → 首SCHEDULED 10.1 → 首token10.4 → 末token10.7\nqueue = 0.1；prefill = 0.3；decode = 0.3；inference = 0.6',ORANGE)
    txt(ax,700,405,'只在各自时钟域内作差；不计算 10.4 − 100.0。',18,GRAY)
    box(ax,80,830,1240,130,'抢占不重置首次调度起点','中途PREEMPTED / 再SCHEDULED进入经过时间，不能将区间当kernel净耗时。',GREEN)
    save(fig,'fig-27-2-clocks')


def closing():
    maps((28,))
    fig,ax=canvas('图 28.1  重新看见主干对象的状态所有权','UML关联局部图；执行层次合并展示；部署与消息实现按Client/Executor条件展开。',1430)
    klass(ax,60,140,540,'EngineCore','scheduler · model_executor','step / step_with_batch_queue',ORANGE)
    klass(ax,800,140,540,'Executor → Worker → Runner','设备上下文 / 持久行 / 当步输入','execute / sample',PURPLE)
    relation(ax,[(600,245),(800,245)],'执行契约',(700,205))
    klass(ax,60,550,540,'Scheduler','Request队列 / 进度 / manager','schedule / update_from_output',ORANGE)
    relation(ax,[(330,350),(330,550)],'持有 1',(430,450))
    klass(ax,800,550,540,'KVCacheManager','Coordinator / 各组 / BlockPool','容量 / 块归属 / 复用',GREEN)
    relation(ax,[(600,655),(800,655)],'持有 1',(700,615))
    klass(ax,60,1020,540,'Request','身份 / token历史 / 进度 / 状态','请求级不变量',BLUE)
    relation(ax,[(330,760),(330,1020)],'维护 0..*',(450,875))
    klass(ax,800,1020,540,'OutputProcessor','输出侧RequestState / Collector','反分词 / 停止 / 交付',GREEN)
    relation(ax,[(60,245),(25,245),(25,950),(1070,950),(1070,1020)],'核心输出消息',(700,950),'dependency')
    txt(ax,700,1340,'同一个请求有多种状态投影；身份、进度和清理协议把它们维系为一个闭环。',15,GRAY)
    save(fig,'fig-28-1-ownership')

if __name__=='__main__':
    import sys
    mode=sys.argv[1] if len(sys.argv)>1 else '26'
    {'26':connector,'27':metrics,'28':closing}[mode]()
