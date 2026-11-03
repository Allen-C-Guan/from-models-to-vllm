"""统一章节地图与 UML 静态结构图；由 render_figures.py 调用。"""
from matplotlib.patches import FancyArrowPatch, Rectangle
from render_figures import canvas,box,txt,arrow,save,BLUE,ORANGE,GREEN,PURPLE,GRAY,PALE


def relation(ax,points,label='',at=None,kind='association',c=GRAY):
    for a,b in zip(points[:-2],points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],c=c,lw=1.5,ls='--' if kind=='dependency' else '-')
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],
        arrowstyle='-|>' if kind=='generalization' else '->',mutation_scale=23 if kind=='generalization' else 15,
        lw=1.5,linestyle='--' if kind=='dependency' else '-',edgecolor=c,facecolor='white',zorder=3))
    if label:txt(ax,*at,label,12,c,bg=True)


def klass(ax,x,y,w,name,attrs,ops,c=GRAY,abstract=False):
    h=210
    ax.add_patch(Rectangle((x,y),w,h,facecolor=PALE[c],edgecolor=c,lw=1.5,zorder=2))
    for yy in [y+66,y+137]:ax.plot([x,x+w],[yy,yy],c=c,lw=1,zorder=3)
    txt(ax,x+w/2,y+25,name,17,c)
    if abstract:txt(ax,x+w/2,y+51,'{abstract}',11,c)
    txt(ax,x+w/2,y+100,attrs or '—',12)
    txt(ax,x+w/2,y+171,ops or '—',12)


def maps(chapters=(3,4,5)):
    for ch in chapters:
        fig,ax=canvas(f'图 {ch}.0  全书主干地图：当前第 {ch} 章',
            '固定坐标：输入在左，协调在中，执行在右；下方保留输出、调度与 KV。',1080,
            footer='依据 v0.30.0 源码；职责／协作地图，不是 UML 类图或进程部署图。粗边框＝本章展开。')
        active={1:{'input','core','execution','output','scheduler','kv'},2:{'input','core','execution','output','scheduler','kv'},3:{'input','core','output'},4:{'core'},5:{'scheduler','kv'},6:{'kv'},7:{'scheduler','kv'},8:{'execution'},9:{'execution'},10:{'execution'},11:{'execution','kv'},12:{'execution','core','scheduler','output'},13:{'core','scheduler','execution'},14:{'execution'},15:{'input','core','output'},16:{'execution'},17:{'input','core','output'},18:{'execution'},19:{'execution','scheduler'},20:{'execution','scheduler','kv'},21:{'input','scheduler','execution'},22:{'input','scheduler','execution'},23:{'input','scheduler','execution','kv'},24:{'execution'},25:{'kv','scheduler','execution'},26:{'core','scheduler','execution','kv'},27:{'input','core','scheduler','execution','output','kv'},28:{'input','core','scheduler','execution','output','kv'}}[ch]
        nodes=[('input',60,165,BLUE,'① 输入与接纳',
                'Renderer → InputProcessor\nEngineCoreRequest + 内部 ID' if ch==3 else '用户文本 → 请求描述\n第 3 章'),
            ('core',520,165,ORANGE,'② 核心协调 EngineCore',
                {1:'迭代协调与资源边界\n第 4 章',2:'创建执行器、准备 KV\n再创建调度器',3:'preprocess_add_request\n构造 Request，再交给调度器',4:'step：schedule → execute\n条件采样 → update_from_output',5:'调用 schedule，取得计划\n第 4 章',6:'迭代协调\n第 4 章',7:'迭代协调\n第 4 章',8:'计划进入执行契约\n第 4 章',9:'计划进入执行契约\n第 4 章',10:'模型调用由执行侧组织\n第 8–9 章',11:'模型调用由执行侧组织\n第 8–9 章',12:'取得结果 → 请求回填\n整理核心输出',13:'在途队列：计划 + Future\n提交与回填分离',14:'迭代协调\n第 4、13 章',15:'AsyncMPClient 消息边界\n共享引擎继续调度与执行',16:'一个核心 → 一个模型副本\n多 Worker 合作执行',17:'多个核心各自调度\n归属与协调保持一致',18:'请求计划进入模型计算\nMoE 另有跨 rank 协作',19:'契约继续复用\n启动期选择具体实现',20:'目标结果与候选分开交接\n执行与回填仍由核心协调',21:'GrammarOutput → 执行边界\n确认结果 → 语法推进',22:'计划携带encoder输入项\n主干继续执行与返回',23:'请求身份随计划传递\n执行结果沿原路径返回',24:'请求计划保持主干契约\n参数表示在执行层适配',25:'一份请求进度\n多组状态共同支撑',26:'任务元数据 → 执行侧\n完成 / 失败 → 核心',27:'请求事件 / 执行结果\n异常与生存状态',28:'计划 → 执行 → 事实回填\n取消与状态提交边界'}[ch]),
            ('execution',980,165,PURPLE,'⑤ 执行与模型',
                'Executor → Worker → Runner\n模型／Attention／采样：第 8–14 章'),
            ('output',60,550,GREEN,'⑥ 结果交付',
                'OutputProcessor\n先建立 RequestState' if ch==3 else 'OutputProcessor → 用户结果\n第 3、4、12 章'),
            ('scheduler',520,550,ORANGE,'③ 调度 Scheduler',
                'running → waiting → 计划\nRequest 进度与 token 预算' if ch==5 else '请求队列、进度、执行计划\n第 5 章'),
            ('kv',980,550,GREEN,'④ KV 状态管理',
                'allocate_slots：申请状态空间\n内部块管理在第 6–7 章展开' if ch==5 else '分配、复用与回收\n第 6–7 章')]
        for key,x,y,c,title,body in nodes:
            if ch==27 and key=='input':body='外部 / 内部ID关联\n前端arrival_time'
            if ch==27 and key=='output':body='请求统计 / Collector\n结果或异常唤醒等待者'
            if ch==27 and key=='scheduler':body='队列 / 调度 / 抢占事件\n计划与状态快照'
            if ch==27 and key=='execution':body='rank / 行映射 / 张量\n最后完成的边界'
            if ch==26 and key=='kv':body='目标块 / 源块寿命\n完成通知后释放'
            if ch==26 and key=='scheduler':body='查询远端 → 分配 → 等待\n命中不等于已可读'
            if ch==26 and key=='execution':body='Connector注册缓存视图\nload / save / 汇报完成'
            if ch==25 and key=='kv':body='Spec / Group / 类型管理器\n不同历史保留规则'
            if ch==25 and key=='scheduler':body='各组共同可恢复边界\n再决定剩余计算'
            if ch==25 and key=='execution':body='按组准备元数据\n状态复制 / 前向 / 有效状态保存'
            if ch==24 and key=='execution':body='配置 → 层方法 → 参数布局\nscale与内核契约 → 计算'
            if ch==23 and key=='input':body='LoRARequest：名称 / ID / 路径\n描述不是权重张量'
            if ch==23 and key=='scheduler':body='每批不同adapter数约束\n请求仍共享token与KV预算'
            if ch==23 and key=='execution':body='加载 → 激活槽位 → 行映射\n基础计算 + 低秩增量'
            if ch==23 and key=='kv':body='参数身份影响缓存身份\n同文本不保证同KV'
            if ch==22 and key=='input':body='媒体处理 → 特征描述\n占位范围 / 输入身份'
            if ch==22 and key=='scheduler':body='encoder预算 / 缓存引用\n媒体依赖约束decoder进度'
            if ch==22 and key=='execution':body='encoder特征 → 窗口切片\n融合embedding → 语言前向'
            if ch==21 and key=='input':body='请求携带语法约束\nStructuredOutputsParams'
            if ch==21 and key=='scheduler':body='等待编译 → 逐位置mask\n确认结果后推进Grammar'
            if ch==21 and key=='execution':body='请求 / 位置映射\nmask logits → 采样 / 验证'
            if ch==20 and key=='execution':body='提议 → 目标验证 → 接受 / 拒绝\n下一批候选独立保存'
            if ch==20 and key=='scheduler':body='草稿预算 → 计划\n回填接受结果 / 修正进度'
            if ch==20 and key=='kv':body='已写字节 ≠ 有效前缀\n长度与位置约束后续读写'
            if ch==19 and key=='execution':body='Platform → Worker / Runner\nBackend / 注册层 / 算子'
            if ch==19 and key=='scheduler':body='实际类 + 有效条件\n补丁与上游委托边界'
            if ch==16 and key=='execution':body='TP：层内分片与合并\nPP：层段与中间状态'
            if ch==17 and key=='input':body='DP 选择目标核心\n记录请求与核心的归属'
            if ch==17 and key=='output':body='各核心返回 → 原前端\n完成 / 取消清理路由表'
            if ch==18 and key=='execution':body='路由 → 专家分发 → 合并\n逻辑专家 / 本地物理专家'
            if ch==15 and key=='input':body='HTTP 适配 → AsyncLLM\n每请求生成器 + 接纳'
            if ch==15 and key=='output':body='后台分发 → 请求收集器\n流式返回 / 取消清理'
            if ch==13 and key=='scheduler':body='AsyncScheduler：占位与回填\n确认历史 / 旧结果辨别'
            if ch==13 and key=='execution':body='有序设备工作与延迟输出\n持久状态 → 当步输入'
            if ch==14 and key=='execution':body='动态批次 → 执行描述\n编译 / 捕获 / 重放 / 回退'
            if ch==10 and key=='execution':body='模型类 → 模块与参数\n前向 → hidden states'
            if ch==11 and key=='execution':body='Attention → Backend / Impl\n元数据 + KV → 计算'
            if ch==12 and key=='execution':body='hidden → logits → 采样\nModelRunnerOutput'
            if ch==12 and key=='scheduler':body='追加确认 token → 停止检查\nEngineCoreOutput'
            if ch==12 and key=='output':body='反分词 → 文本停止检查\nRequestOutput'
            if ch==8 and key=='execution':body='Executor：组织 Worker\nWorker：设备上下文 → Runner'
            if ch==9 and key=='execution':body='Runner：持久行 → 当轮顺序\nInputBatch：token 与位置'
            if ch==6 and key=='kv':body='Manager → 协调器 → 组管理器\nBlockPool：位置、引用、空闲候选'
            if ch==7 and key=='kv':body='前缀身份 → 命中 → 引用保护\n释放与淘汰分离'
            if ch==7 and key=='scheduler':body='容量失败 → 抢占 → 等待\n保留 token 历史，重新建立 KV'
            box(ax,x,y,360,180,title,body,c if key in active else GRAY)
            if key in active:
                ax.add_patch(Rectangle((x-5,y-5),370,190,fill=False,lw=2.6,edgecolor=c,zorder=1))
                txt(ax,x+330,y-22,'展开',11,c)
        arrow(ax,[(420,230),(520,230)],'请求')
        arrow(ax,[(880,225),(980,225)],'计划')
        arrow(ax,[(980,305),(880,305)],'结果')
        arrow(ax,[(630,345),(630,550)],'调度／回填',(568,445))
        arrow(ax,[(775,550),(775,345)],'计划／输出',(833,445))
        arrow(ax,[(880,605),(980,605)],'申请')
        arrow(ax,[(980,695),(880,695)],'块／失败')
        arrow(ax,[(520,285),(470,285),(470,640),(420,640)],'核心输出',(470,455))
        txt(ax,240,775,'↓ 用户取得结果',14,GREEN)
        zoom={1:'本章放大：责任 → 类型与关系 → 调用路径 → 可观察状态',2:'本章放大：配置约束 → 对象装配 → 设备与模型 → KV 资源 → 就绪',3:'本章放大：渲染 → 身份分配 → 输出侧登记 → 核心 Request → 等待队列',
              4:'本章放大：SchedulerOutput → Future／模型结果 → 请求进度回填 → 前端交付',
              5:'本章放大：读取进度差 → 裁剪预算 → 申请 KV → 区分新旧请求 → SchedulerOutput',
              6:'本章放大：增长需求 → 跨组容量检查 → 请求块表 → 新编号 → 释放引用',
              7:'本章放大：哈希身份 → 命中并持有 → 释放／淘汰；容量不足 → 抢占／重算',
              8:'本章放大：执行器契约 → 本地／消息分发 → Worker → Runner → 输出归属',
              9:'本章放大：请求 ID → 持久行号 → 本轮 batch 顺序 → 拼接 token → InputBatch',
              10:'本章放大：模型配置 → 实现类 → 权重映射 → 模型层 → 隐藏状态',
              11:'本章放大：Q/K/V + 元数据 → 层上下文 → KV 写入／读取 → Attention 输出',
              12:'本章放大：logits → 采样 → 可读结果 → 请求回填与停止 → 用户文本',
              13:'本章放大：在途队列 → 占位兑现 → 旧结果消账 → 缓冲区生命周期',
              14:'本章放大：编译模型 → 捕获候选 → 匹配执行描述 → 稳定输入 → 重放或回退',
              15:'本章放大：协议适配 → 异步接纳 → 共享接收与分发 → 独立消费 → 取消闭环',
              16:'本章放大：rank 与组 → 参数 / 层段 → 控制命令 → 张量通信 → 返回',
              17:'本章放大：负载估计 → 选核心 → 归属登记 → 结果 / 取消 → 跨核心协调',
              18:'本章放大：token 路由 → 专家归属 → 分发 → 本地计算 → 加权合并',
              19:'本章放大：发现平台 → 配置装配 → 注册 / 继承 / 补丁 → 查证实际方法',
              20:'本章放大：候选 → 目标验证 → 连续接受前缀 → 状态修正 → 后续候选',
              21:'本章放大：规则编译 → 请求语法状态 → 合法集合 → 采样 → 正式推进',
              22:'本章放大：媒体与占位 → encoder调度 → 特征缓存 → 切片融合 → 语言前向',
              28:'本章收拢：状态所有者 → 交接产物 → 三轮闭环 → 设计取舍 → 新源码入口',
              27:'本章放大：身份与事件 → 时钟域 → 指标产物 → 错误传播 → 责任定位',
              26:'本章放大：外部命中 → 本地分配 → 数据交接 → 完成 / 失败 → 安全释放',
              25:'本章放大：状态规格 → 物理分组 → 释放规则 → 共同命中 → 执行状态恢复',
              24:'本章放大：量化配置 → 参数创建 → 装载后转换 → 内核选择 → 前向',
              23:'本章放大：adapter身份 → 加载 / 激活 → token映射 → 槽位 → 增量计算'}[ch]
        box(ax,60,840,1280,90,'',zoom,GRAY)
        txt(ax,700,990,'启动（第 2 章）准备这些对象；服务、并行与扩展章节继续在相同主干上展开。',14,GRAY)
        save(fig,f'fig-{ch}-0-map')


def class_figures():
    legend='UML：实线箭头＝可导航关联；虚线箭头＝使用依赖；空心三角＝泛化，指向父类。数字＝目标数量。'
    fig,ax=canvas('图 3.2  前端的类关系：持有协作者，分离请求数据',
        '类框从上到下：类名、属性、操作；只选本章所需成员。{abstract} 表示抽象类。',1190,footer=legend)
    klass(ax,520,135,360,'LLMEngine','input_processor · output_processor\nengine_core: EngineCoreClient','add_request() · step()',BLUE)
    klass(ax,60,450,360,'InputProcessor','renderer','process_inputs()\nassign_request_id()',BLUE)
    klass(ax,520,450,360,'OutputProcessor','request_states: dict','add_request()\nprocess_outputs()',GREEN)
    klass(ax,980,450,360,'EngineCoreClient','—','add_request() · get_output()',BLUE,True)
    klass(ax,60,820,360,'EngineCoreRequest','request_id · prompt_token_ids\nsampling_params','—',BLUE)
    klass(ax,520,820,360,'RequestState','external_req_id · detokenizer\noutput_kind','from_new_request()',GREEN)
    klass(ax,980,820,360,'InprocClient','engine_core: EngineCore','add_request() · get_output()',BLUE)
    relation(ax,[(560,345),(560,390),(240,390),(240,450)],'input_processor  1',(340,390))
    relation(ax,[(700,345),(700,450)],'output_processor  1',(805,430))
    relation(ax,[(840,345),(840,390),(1160,390),(1160,450)],'engine_core  1',(1050,390))
    relation(ax,[(240,660),(240,820)],'«create»',(300,740),'dependency')
    relation(ax,[(700,660),(700,820)],'request_states  0..*',(812,740))
    relation(ax,[(1160,820),(1160,660)],'继承',(1220,740),'generalization')
    txt(ax,700,1100,'engine.engine_core 是客户端；client.engine_core 才是本例的核心对象。同名属性不能混读。',14,GRAY)
    save(fig,'fig-3-2-objects')

    fig,ax=canvas('图 4.1  核心协调依赖什么契约',
        'EngineCore 持有协作者；继承发生在实现类与抽象基类之间，不发生在协调者与协作者之间。',1110,footer=legend)
    klass(ax,460,125,480,'EngineCore','scheduler: SchedulerInterface\nmodel_executor: Executor','step()',ORANGE)
    klass(ax,80,445,540,'SchedulerInterface','—','schedule() → SchedulerOutput\nupdate_from_output() → 核心输出',ORANGE,True)
    klass(ax,780,445,540,'Executor','vllm_config','execute_model() · sample_tokens()',PURPLE,True)
    klass(ax,80,785,540,'Scheduler','requests · running · waiting','schedule() · update_from_output()',ORANGE)
    klass(ax,780,785,540,'UniProcExecutor','driver_worker','execute_model() · sample_tokens()',PURPLE)
    relation(ax,[(550,335),(550,385),(350,385),(350,445)],'scheduler  1',(350,380))
    relation(ax,[(850,335),(850,385),(1050,385),(1050,445)],'model_executor  1',(1080,380))
    relation(ax,[(350,785),(350,655)],'继承',(410,715),'generalization')
    relation(ax,[(1050,785),(1050,655)],'继承',(1110,715),'generalization')
    save(fig,'fig-4-1-contracts')

    fig,ax=canvas('图 5.1  调度器：长期状态与当步计划分开',
        '关联描述对象持有关系；创建依赖描述本轮产物。没有把 Request 画成执行计划的组成部分。',1240,footer=legend)
    klass(ax,480,130,440,'Scheduler','requests: dict · running: list\nwaiting: RequestQueue','schedule() · update_from_output()',ORANGE)
    klass(ax,60,470,360,'Request','status · num_computed_tokens\nnum_in_flight_tokens','append_output_token_ids()',ORANGE)
    klass(ax,520,470,360,'SchedulerOutput','num_scheduled_tokens\nfinished_req_ids','—',PURPLE)
    klass(ax,980,470,360,'KVCacheManager','coordinator · block_pool','allocate_slots() · free()',GREEN)
    klass(ax,100,870,520,'NewRequestData','req_id · prompt_token_ids\nblock_ids','from_request()',BLUE)
    klass(ax,780,870,520,'CachedRequestData','req_ids · new_block_ids\nnum_computed_tokens','—',ORANGE)
    relation(ax,[(520,340),(520,400),(240,400),(240,470)],'requests  0..*',(270,395))
    relation(ax,[(700,340),(700,470)],'«create»',(760,400),'dependency')
    relation(ax,[(880,340),(880,400),(1160,400),(1160,470)],'kv_cache_manager  1',(1120,395))
    relation(ax,[(600,680),(600,770),(360,770),(360,870)],'scheduled_new_reqs  0..*',(370,765))
    relation(ax,[(800,680),(800,770),(1040,770),(1040,870)],'scheduled_cached_reqs  1',(1080,765))
    txt(ax,700,1140,'一个 CachedRequestData 可承载零条或多条请求的更新；对象数量 1 不等于请求数量 1。',14,GRAY)
    save(fig,'fig-5-1-state-plan')


def main():
    maps();class_figures()


if __name__=='__main__':main()
