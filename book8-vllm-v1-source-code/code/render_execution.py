"""第 8–9 章：执行对象、部署与索引空间。"""
from matplotlib.patches import Rectangle
from render_figures import canvas,box,txt,arrow,save,BLUE,ORANGE,GREEN,PURPLE,GRAY
from render_architecture import maps,klass,relation


def main():
    maps((8,9))
    fig,ax=canvas('图 8.1  执行组织、设备入口与模型调用分层','UML：实线＝关联，空心三角＝继承；所示 Runner 为上游 MRV2 选择。',1210)
    klass(ax,80,145,530,'Executor','vllm_config','execute_model() · sample_tokens()',PURPLE,True)
    klass(ax,790,145,530,'WorkerBase','设备／模型执行契约','execute_model()',PURPLE,True)
    klass(ax,80,510,530,'UniProcExecutor','driver_worker: WorkerWrapperBase','collective_rpc()',PURPLE)
    klass(ax,790,510,530,'Worker 〔gpu_worker.py〕','device · model_runner','execute_model() · sample_tokens()',PURPLE)
    klass(ax,80,875,530,'WorkerWrapperBase','worker: WorkerBase','execute_model() · __getattr__()',GRAY)
    klass(ax,790,875,530,'GPUModelRunner 〔MRV2〕','req_states · block_tables · model','prepare_inputs() · execute_model()',PURPLE)
    relation(ax,[(345,510),(345,355)],'继承',(405,430),'generalization')
    relation(ax,[(1055,510),(1055,355)],'继承',(1115,430),'generalization')
    relation(ax,[(345,720),(345,875)],'driver_worker  1',(475,800))
    relation(ax,[(1055,720),(1055,875)],'model_runner  1',(1190,800))
    relation(ax,[(610,980),(700,980),(700,250),(790,250)],'worker  1',(700,420))
    save(fig,'fig-8-1-classes')

    fig,ax=canvas('图 8.2  同一执行契约，两种工作者组织','进程框与对象框分开；下方设备通信不等于上方控制消息。省略前端进程。',1050)
    for x,w,title in [(50,560,'单进程执行：核心与 Worker 同进程'),(730,620,'多进程执行：核心进程')]:
        ax.add_patch(Rectangle((x,125),w,245,fill=False,ec=GRAY,lw=2));txt(ax,x+w/2,150,title,16,GRAY)
    box(ax,80,210,220,100,'EngineCore','UniProcExecutor',ORANGE)
    box(ax,360,210,220,100,'Worker','Runner',PURPLE)
    arrow(ax,[(300,260),(360,260)],'调用')
    box(ax,780,210,520,100,'EngineCore → MultiprocExecutor','方法名 + SchedulerOutput + 回复要求',ORANGE)
    for x,label in [(730,'工作者进程 0'),(1060,'工作者进程 1')]:
        ax.add_patch(Rectangle((x,490),290,230,fill=False,ec=GRAY,lw=2));txt(ax,x+145,520,label,16,GRAY)
        box(ax,x+25,565,240,110,'Worker → Runner','各自设备与 rank',PURPLE)
    arrow(ax,[(930,310),(930,420),(875,420),(875,490)],'广播计划',(850,405),dashed=True)
    arrow(ax,[(1150,310),(1150,420),(1205,420),(1205,490)],'广播计划',(1210,405),dashed=True)
    box(ax,80,565,500,110,'设备 0','模型计算与 KV 张量',PURPLE)
    arrow(ax,[(470,310),(470,565)],'提交设备工作',(390,440))
    box(ax,730,840,290,100,'设备 0',c=PURPLE);box(ax,1060,840,290,100,'设备 1',c=PURPLE)
    arrow(ax,[(875,720),(875,840)],'计算')
    arrow(ax,[(1205,720),(1205,840)],'计算')
    arrow(ax,[(1020,890),(1060,890)],'',c=PURPLE)
    txt(ax,1040,790,'TP／PP 等设备通信：第 16 章',13,PURPLE)
    txt(ax,330,845,'进程拓扑是执行器选择；\n同一 Python 接口不保证同样的等待语义。',15,GRAY)
    save(fig,'fig-8-2-deployment')

    fig,ax=canvas('图 9.1  持久状态与当步输入分离','UML：实线＝持有／引用；虚线＝构造／使用依赖。InputBatch 中的张量可引用复用缓冲区。',1180)
    klass(ax,450,130,500,'GPUModelRunner 〔MRV2〕','req_states · block_tables\ninput_buffers','execute_model() · prepare_inputs()',PURPLE)
    klass(ax,50,480,400,'RequestState 〔worker/gpu〕','req_id_to_index · free_indices\n长期 token 与进度状态','add_request() · remove_request()',ORANGE)
    klass(ax,500,480,400,'BlockTables','持久块表与暂存更新','append_block_ids()\ngather_block_tables()',GREEN)
    klass(ax,950,480,400,'InputBuffers','input_ids · positions\nquery_start_loc · seq_lens','—',PURPLE)
    klass(ax,450,880,500,'InputBatch','req_ids · idx_mapping\n当步输入与元数据','—',BLUE)
    relation(ax,[(520,340),(520,405),(250,405),(250,480)],'req_states  1',(280,400))
    relation(ax,[(700,340),(700,480)],'block_tables  1',(810,420))
    relation(ax,[(880,340),(880,405),(1150,405),(1150,480)],'input_buffers  1',(1120,400))
    relation(ax,[(950,250),(1380,250),(1380,815),(700,815),(700,880)],'«create»',(950,815),'dependency')
    relation(ax,[(950,990),(1150,990),(1150,690)],'张量视图',(1220,845),'dependency')
    save(fig,'fig-9-1-classes')

    fig,ax=canvas('图 9.2  持久行、batch 行、token 偏移不是同一坐标','真实 NPU 运行 runner-inproc-traced；A/B 简写保留身份语义，完整内部 ID 见正文记录。',1030)
    box(ax,70,160,340,150,'持久状态','A → 行 3\nB → 行 2',ORANGE)
    box(ax,530,160,340,150,'首轮 batch 顺序','batch 0 → A\nbatch 1 → B',BLUE)
    box(ax,990,160,340,150,'拼接 token 区间','A → [0,5)\nB → [5,11)',PURPLE)
    arrow(ax,[(530,240),(410,240)],'idx_mapping\n[3,2]',(470,230))
    arrow(ax,[(870,240),(990,240)],'边界\n[0,5,11]',(930,230))
    box(ax,70,570,340,150,'第三轮持久状态','A 已移除\nB 仍在行 2',ORANGE)
    box(ax,530,570,340,150,'第三轮 batch 顺序','batch 0 → B',BLUE)
    box(ax,990,570,340,150,'拼接 token 区间','B → [0,1)',PURPLE)
    arrow(ax,[(530,645),(410,645)],'idx_mapping\n[2]',(470,635))
    arrow(ax,[(870,645),(990,645)],'边界\n[0,1]',(930,635))
    arrow(ax,[(240,310),(240,570)],'A 结束；B 不搬行',(340,440))
    txt(ax,700,880,'读取状态时先按 ID 找持久行；解释张量时再按当轮 batch 与 token 偏移定位。',16,GRAY)
    save(fig,'fig-9-2-indices')


if __name__=='__main__':main()
