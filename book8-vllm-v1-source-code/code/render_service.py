"""第15章：在线前端对象与任务协作。"""
from render_figures import canvas,box,txt,arrow,save,lanes,BLUE,ORANGE,GREEN,PURPLE,GRAY
from render_architecture import maps,klass,relation


def main():
    maps((15,))
    fig,ax=canvas('图 15.1  共享前端与每请求收集器','UML：实线＝引用；虚线＝使用依赖；空心三角＝继承。Collector不是普通asyncio.Queue。',1210)
    klass(ax,440,130,520,'EngineClient','生成与服务控制契约','generate() · abort()',GRAY,True)
    klass(ax,60,510,540,'AsyncLLM','output_processor · engine_core','generate() · output_handler任务',BLUE)
    klass(ax,800,510,540,'EngineCoreClient','本例：AsyncMPClient','add_request_async()\nget_output_async()',ORANGE,True)
    klass(ax,40,910,410,'OutputProcessor','request_states: dict','process_outputs()',GREEN)
    klass(ax,495,910,410,'RequestState','内部 ID · detokenizer · queue','make_request_output()',GREEN)
    klass(ax,950,910,410,'RequestOutputCollector','output · ready: Event','put() · get()',GREEN)
    relation(ax,[(330,510),(330,415),(700,415),(700,340)],'继承',(470,405),'generalization')
    relation(ax,[(600,615),(800,615)],'engine_core',(700,590))
    relation(ax,[(250,720),(250,910)],'output_processor',(360,815))
    relation(ax,[(450,1015),(495,1015)],'',None)
    relation(ax,[(905,1015),(950,1015)],'',None)
    txt(ax,470,875,'states 0..*',12,GREEN);txt(ax,930,875,'queue 1',12,GREEN)
    relation(ax,[(500,720),(500,790),(1155,790),(1155,910)],'generate 使用同一收集器',(825,785),'dependency')
    save(fig,'fig-15-1-classes')

    fig,ax=canvas('图 15.2  共享后台分发结果，单请求任务消费','任务／消息时序：虚线跨核心边界传消息；实线为本地调用或数据交付。C为取消变体。',1180)
    xs=[160,510,870,1230]
    lanes(ax,[('C 的 generate 任务',BLUE),('C 的前端状态',GREEN),('后台输出任务',GREEN),('核心进程',ORANGE)],xs,1040)
    arrow(ax,[(160,255),(1230,255)],'登记本地状态后：ADD 消息',dashed=True)
    arrow(ax,[(1230,360),(870,360)],'核心输出',dashed=True)
    arrow(ax,[(870,445),(510,445)],'OutputProcessor → put')
    arrow(ax,[(510,530),(160,530)],'get：Tokyo / finished=False')
    box(ax,35,605,250,95,'消费者关闭生成器','aclose → GeneratorExit',BLUE)
    arrow(ax,[(160,755),(510,755)],'abort：移除前端状态')
    arrow(ax,[(160,860),(1230,860)],'ABORT：内部请求 ID',dashed=True)
    box(ax,1090,920,275,110,'核心处理终止','在途工作按契约收尾\n释放请求资源引用',ORANGE)
    txt(ax,620,1010,'取消控制消息与设备操作不是同一层事件；\n不能将abort返回理解为强行撤销已发射内核。',15,GRAY)
    save(fig,'fig-15-2-stream')


if __name__=='__main__':main()
