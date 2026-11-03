"""扩展章节：对象契约与状态提交。"""
from render_figures import canvas,box,txt,arrow,save,BLUE,ORANGE,GREEN,PURPLE,GRAY
from render_architecture import maps,klass,relation


def spec():
    maps((20,))
    fig,ax=canvas('图 20.1  提议实现、验证器与请求状态','UML：实线关联；虚线依赖；空心三角指向父类。设备与CPU状态不画成同一对象。',1510)
    klass(ax,60,150,540,'GPUModelRunner','speculator · rejection_sampler','sample_tokens()',PURPLE)
    klass(ax,800,150,540,'RejectionSampler','目标分布 / 候选信息','确认前缀 + 目标结果',GREEN)
    relation(ax,[(600,255),(800,255)],'持有 0..1',(700,220))
    names=[('BaseSpeculator','propose(...) 契约'),('DraftModelSpeculator','模型 / 草稿缓冲'),('AutoRegressiveSpeculator','首步 / 多步提议组织'),('MTPSpeculator','模型装载 / 特定钩子')]
    for i,(name,body) in enumerate(names):
        y=460+i*245
        klass(ax,60,y,540,name,body,'propose / 继承实现',PURPLE)
        if i:relation(ax,[(330,y),(330,y-35)],'继承',(395,y-18),'generalization')
    relation(ax,[(220,360),(220,460)],'持有 0..1',(130,415))
    klass(ax,800,530,540,'Scheduler','requests: Request','schedule / update_from_output',ORANGE)
    klass(ax,800,900,540,'Request','spec_token_ids · num_computed_tokens','确认输出与请求进度',ORANGE)
    relation(ax,[(1070,740),(1070,900)],'持有 0..*',(1180,815))
    box(ax,800,1230,540,150,'跨边界产物','SchedulerOutput → 执行侧\n确认输出 / 草稿 → 调度侧',GRAY)
    txt(ax,700,1450,'候选生成、目标验证、请求记账各有所有者；经产物对齐，不共享全部内部状态。',14,GRAY)
    save(fig,'fig-20-1-classes')
    fig,ax=canvas('图 20.2  一次拒绝为什么必须丢弃后缀','构造例：已确认10但尚无其KV；草稿[20,30,40]，正常贪心验证。不是硬件记录。',1130)
    rows=[('候选输入','[10, 20, 30, 40] → hidden [4,H] → logits [4,V]',PURPLE),('目标选择','[20, 99, 40, 50]：只需处理到首次不匹配',PURPLE),('验证结果','20接受；30拒绝→改为99；40与bonus50不提交',ORANGE),('有效状态','计算进度 5 → 暂记9 → 回退2 → 7；确认返回[20,99]',GREEN)]
    for i,(title,body,c) in enumerate(rows):
        y=160+i*215;box(ax,80,y,1240,150,title,body,c)
        if i:arrow(ax,[(700,y-65),(700,y)],'')
    txt(ax,700,1045,'下一轮从确认前缀继续：99尚无KV；被拒绝位置的旧字节不是有效历史。',15,GRAY)
    save(fig,'fig-20-2-verify')


def main():
    import sys
    if len(sys.argv)>1 and sys.argv[1]=="21":grammar()
    elif len(sys.argv)>1 and sys.argv[1]=="22":multimodal()
    elif len(sys.argv)>1 and sys.argv[1]=="23":lora()
    else:spec()



def grammar():
    maps((21,))
    fig,ax=canvas('图 21.1  规则编译与请求匹配状态分离','UML：实线关联；虚线创建依赖；空心三角指向父类。只画代表XGrammar实现。',1430)
    klass(ax,60,150,540,'StructuredOutputManager','backend · 编译线程池','grammar_init / grammar_bitmask',ORANGE)
    klass(ax,800,150,540,'StructuredOutputBackend','编译能力 / tokenizer','compile_grammar()',ORANGE,True)
    relation(ax,[(600,255),(800,255)],'backend 0..1',(700,220))
    klass(ax,800,510,540,'XgrammarBackend','compiler','compile_grammar()',ORANGE)
    relation(ax,[(1070,510),(1070,360)],'继承',(1130,440),'generalization')
    klass(ax,60,510,540,'StructuredOutputRequest','grammar / Future / Exception','检查就绪结果',BLUE)
    klass(ax,60,900,540,'StructuredOutputGrammar','请求级进度','accept / validate / rollback',GREEN,True)
    klass(ax,800,900,540,'XgrammarGrammar','matcher · ctx · 已处理计数','fill_bitmask / accept_tokens',GREEN)
    relation(ax,[(330,720),(330,900)],'成功时持有 1',(440,810))
    relation(ax,[(800,1005),(600,1005)],'继承',(700,970),'generalization')
    relation(ax,[(1070,720),(1070,900)],'«create»',(1150,810),'dependency')
    box(ax,60,1230,1280,110,'执行端 StructuredOutputsWorker','GrammarOutput携带mask与请求顺序 → 映射到logits位置 → 不合法项置为负无穷',PURPLE)
    save(fig,'fig-21-1-classes')
    fig,ax=canvas('图 21.2  检查候选与提交结果使用不同状态语义','状态示意：同一请求的Matcher；本图P/a等是教学字符token。',1080)
    for y,title,c in [(190,'候选检查：validate / 构造mask',ORANGE),(600,'结果提交：accept_tokens',GREEN)]:
        txt(ax,700,y-65,title,18,c)
        for x,label in [(80,'初始：允许 P / T'),(520,'已见P：只许 a'),(960,'已见Pa：只许 r')]:box(ax,x,y,360,160,label,'请求级语法游标',c)
        arrow(ax,[(440,y+80),(520,y+80)],'P')
        arrow(ax,[(880,y+80),(960,y+80)],'a')
    arrow(ax,[(1140,350),(1140,440),(260,440),(260,350)],'rollback：撤销临时推进',(700,440))
    txt(ax,700,880,'上方试走结束，持久位置仍是初始；下方确认提交后，下一轮从Pa继续。',16,GRAY)
    txt(ax,700,950,'语法合法不等于目标模型接受；投机验证之后才提交真正返回的token。',14,GRAY)
    save(fig,'fig-21-2-state')





def multimodal():
    maps((22,))
    fig,ax=canvas('图 22.1  Encoder状态的两侧所有权','UML关联图：左侧核心CPU对象，右侧执行对象；虚线表示产物依赖，不是共享对象引用。',1360)
    klass(ax,60,160,540,'Scheduler','encoder_cache_manager','schedule()',ORANGE)
    klass(ax,800,160,540,'EncoderRunner','encoder_cache · model','execute_mm_encoder()',PURPLE)
    klass(ax,60,530,540,'EncoderCacheManager','cached引用 · freeable · freed','can_allocate / free',ORANGE)
    klass(ax,800,530,540,'EncoderCache','mm_features · encoder_outputs','free_encoder_cache()',GREEN)
    relation(ax,[(330,370),(330,530)],'持有 1',(430,450))
    relation(ax,[(1070,370),(1070,530)],'持有 1',(1170,450))
    relation(ax,[(600,635),(800,635)],'驱逐hash通知',(700,590),'dependency')
    klass(ax,60,920,540,'MultiModalFeatureSpec','data · identifier · mm_position','每媒体项一个描述',BLUE)
    box(ax,800,940,540,170,'实际特征张量','identifier → Tensor [M,Hlm]\n由执行端创建与删除',GREEN)
    relation(ax,[(1060,740),(1060,940)],'保存 0..*',(1170,845))
    relation(ax,[(820,740),(820,820),(330,820),(330,920)],'mm_features按请求关联',(510,820))
    txt(ax,700,1250,'CPU容量账不包含特征张量；释放引用与发送驱逐通知是不同时间点。',15,GRAY)
    save(fig,'fig-22-1-classes')
    fig,ax=canvas('图 22.2  全局位置、媒体内部位置与当步行','构造例：图片占[2,6)，本轮query为[3,5)，此段所有位置都承载特征。',1030)
    box(ax,80,160,1240,160,'全序列','文本0 · 文本1 · 图0 · 图1 · 图2 · 图3 · 文本6',BLUE)
    box(ax,80,430,560,160,'图片内部','全局[3,5) − offset2\n得到特征切片[1,3)',ORANGE)
    box(ax,800,430,520,160,'当步输入','只有两行 [2,Hlm]\nis_mm_embed=[True,True]',PURPLE)
    arrow(ax,[(360,320),(360,430)],'窗口交集')
    arrow(ax,[(640,510),(800,510)],'两行特征')
    box(ax,80,770,1240,130,'融合后进入语言模型','特征行顺序对应当步位置；不将全局下标3、4直接当作本轮行号。',GREEN)
    arrow(ax,[(1060,590),(1060,770)],'inputs_embeds')
    save(fig,'fig-22-2-window')



def lora():
    maps((23,))
    fig,ax=canvas('图 23.1  请求身份、管理器与层增量','UML：空心三角指向父类；实线为持有或关联。LRU实现遵守相同管理职责。',1450)
    klass(ax,60,150,540,'GPUModelRunner','lora_state · lora_manager','execute_model()',PURPLE)
    klass(ax,800,150,540,'LoRAModelRunnerMixin','管理功能复用','load_lora_model / _set_active_loras',GRAY)
    relation(ax,[(600,255),(800,255)],'继承',(700,220),'generalization')
    klass(ax,60,500,540,'LoraState','持久行adapter ID / 请求描述','make_lora_inputs()',BLUE)
    klass(ax,800,500,540,'WorkerLoRAManager','_adapter_manager','set_active_adapters()',PURPLE)
    relation(ax,[(330,360),(330,500)],'持有 1',(420,435))
    relation(ax,[(570,360),(570,415),(1070,415),(1070,500)],'持有 0..1',(930,415))
    klass(ax,800,870,540,'LoRAModelManager','registered / active / slot_to_id','activate / set_adapter_mapping',PURPLE)
    relation(ax,[(1070,710),(1070,870)],'持有 1',(1160,790))
    klass(ax,60,870,540,'BaseLinearLayerWithLoRA','base_layer · A/B槽 · wrapper','_apply_sync()',GREEN)
    relation(ax,[(800,975),(600,975)],'管理各层槽',(700,940))
    box(ax,60,1210,1280,140,'参数权重与请求描述分开','LoRARequest标识要用谁；LoRAModel携带权重；层槽保存当前可执行参数。',GRAY)
    save(fig,'fig-23-1-classes')
    fig,ax=canvas('图 23.2  同一批次经过三种编号空间','CPU映射实验对应视图；无adapter权重或设备内核执行。',1070)
    box(ax,60,180,360,220,'请求行 → 当步顺序','B / A / C\n持久行 [2,3,1]\n本轮token数 [1,2,1]',BLUE)
    box(ax,520,180,360,220,'逻辑adapter ID','请求 [0,11,22]\ntoken [0,11,11,22]\n0 = 基础模型',ORANGE)
    box(ax,980,180,360,220,'当前物理槽','slot_to_id=[22,11,None]\ntoken槽 [-1,1,1,0]\n-1 = 无增量',PURPLE)
    arrow(ax,[(420,290),(520,290)],'取ID / 展开')
    arrow(ax,[(880,290),(980,290)],'反向查槽')
    box(ax,60,660,1280,180,'槽位交换，但逻辑请求不变','slot_to_id=[11,22,None] → token槽必须改为[-1,0,0,1]\n缓存键 = 逻辑映射 + 当前槽布局',GREEN)
    arrow(ax,[(1160,400),(1160,660)],'换入换出')
    txt(ax,700,970,'adapter ID是身份；槽位是可复用存储。按错一种编号，形状正确也会算错。',15,GRAY)
    save(fig,'fig-23-2-mapping')

if __name__=='__main__':main()
