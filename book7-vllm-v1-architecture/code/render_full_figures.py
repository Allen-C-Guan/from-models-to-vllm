"""第 3–12 章中文架构教学图。source ./env.sh 后运行；不启动设备推理。"""
from render_figures import (canvas, box, group, arrow, text, note, save,
                            BLUE, ORANGE, GREEN, PURPLE, MUTED, FILL, OUT)
from matplotlib.patches import Rectangle
import unicodedata


def page(ch, num, title, subtitle, active=(), height=850):
    if isinstance(active, int):
        active = (active,)
    fig, ax = canvas(f"图 {ch}.{num}  {title}", subtitle, height)
    for i, (label, color) in enumerate(zip(
            ["输入适配", "核心／调度", "KV 管理", "执行／设备", "结果交付"],
            [BLUE, ORANGE, GREEN, PURPLE, BLUE])):
        c = color if i in active else MUTED
        box(ax, 40 + i * 228, 112, 208, 42, label, color=c)
    note(ax, height - 24, "顶部为全书职责索引；高亮表示本图关注范围。示意图，非性能测量。")
    return fig, ax


def node(ax, x, y, title, body="", color=BLUE, w=270, h=100):
    box(ax, x, y, w, h, "", color=color)
    def fit(value, maximum):
        count = max(sum(1 if unicodedata.east_asian_width(c) in "WF" else .56 for c in line) for line in value.splitlines())
        return min(maximum, (w - 24) * .72 / max(count, 1))
    if body:
        text(ax, x+w/2, y+h*.28, title, fit(title,18), color, weight="bold")
        text(ax, x+w/2, y+h*.70, body, fit(body,14))
    else:
        text(ax, x+w/2, y+h/2, title, fit(title,18), color, weight="bold")


def line(ax, points, label="", at=None, color=MUTED, dashed=False):
    arrow(ax, points, label or None, at, color, dashed, 14)


def flow(ch, num, slug, title, subtitle, cards, active=(), foot=""):
    """三列蛇形活动图；每张图由独立的语义卡和箭头标签定义。"""
    fig, ax = page(ch, num, title, subtitle, active)
    loc = [(50, 235), (465, 235), (880, 235), (880, 530), (465, 530), (50, 530)]
    w = 260
    edge_labels = {
        4: ["配置", "执行资源", "报告需求", "容量结果", "资源就位"],
        7: ["拼接", "输入", "隐藏表示", "选定行", "logits"],
        11: ["候选", "验证计划", "验证结果", "接受数量", "已确认 token"],
        12: ["提问", "定位", "查证", "检验", "归纳"],
    }[ch]
    for i, (head, body, col) in enumerate(cards):
        x, y = loc[i]; node(ax, x, y, head, body, col, w=w, h=120)
        if i:
            px, py = loc[i - 1]
            if py == y:
                a, b = ((px + w, py + 60), (x, y + 60)) if x > px else ((px, py + 60), (x + w, y + 60))
                line(ax, [a, b], edge_labels[i-1], color=col)
            else:
                line(ax, [(px + w/2, py+120), (x+w/2, y)], edge_labels[i-1], color=col)
    if foot: note(ax, 745, foot)
    save(fig, slug)


def ch3():
    fig, ax = page(3, 1, "职责怎样落到进程和设备", "单设备 · 核心多进程 · UniProcExecutor；CPU 控制代码与 NPU 算子分开", (0,1,3,4))
    group(ax, 40, 210, 330, 340, "用户／前端进程", BLUE)
    node(ax, 70, 275, "输入与输出", "Renderer · OutputProcessor", BLUE)
    node(ax, 70, 410, "核心客户端", "提交请求 · 接收结果", BLUE)
    group(ax, 540, 210, 620, 340, "EngineCore 进程", ORANGE)
    node(ax, 565, 275, "核心协调", "Scheduler · KV 管理", ORANGE)
    node(ax, 865, 275, "执行控制", "Executor\nWorker · Runner", PURPLE, w=270)
    text(ax, 850, 470, "输入线程／队列 · 主循环 · 输出线程／队列", 16)
    line(ax, [(370,420),(540,420)], "请求消息", (455,390), BLUE)
    line(ax, [(540,510),(370,510)], "内部结果", (455,530), BLUE)
    line(ax, [(835,325),(865,325)], color=PURPLE)
    node(ax, 710, 650, "NPU 设备", "权重 · KV 张量 · 算子", PURPLE, w=390)
    line(ax, [(1000,375),(1135,375),(1135,590),(1000,590),(1000,650)], "提交", (1070,590), PURPLE)
    line(ax, [(760,650),(760,580),(590,580),(590,550)], "结果／完成", (670,580), PURPLE)
    save(fig, "fig-3-1-deployment")

    fig, ax = page(3, 2, "消息与计算在核心进程中交接", "纵向是逻辑先后；C 到达不等于立即进入当前设备批次", (0,1,3,4), 950)
    xs=[125,435,745,1055]
    for x,t,c in zip(xs,["通信线程","输入／输出队列","核心主循环","执行／设备"],[BLUE,BLUE,ORANGE,PURPLE]):
        node(ax,x-110,205,t,color=c,w=220,h=62)
        ax.plot([x,x],[270,860],ls="--",color="#BCC7D4",lw=1)
    line(ax,[(745,310),(1055,310)],"B 的当步计划",(900,285),PURPLE)
    node(ax,950,350,"B 的计算",color=PURPLE,w=210,h=110)
    line(ax,[(125,385),(435,385)],"收到 C，放入队列",(280,360),BLUE)
    line(ax,[(1055,510),(745,510)],"当步结果",(900,490),PURPLE)
    line(ax,[(745,570),(435,570)],"内部输出入队",(590,545),ORANGE)
    line(ax,[(435,640),(125,640)],"发送给前端",(280,615),BLUE)
    line(ax,[(435,710),(745,710)],"核心处理 C",(590,685),ORANGE)
    line(ax,[(745,800),(1055,800)],"下一计划：是否包含 C？",(900,775),PURPLE)
    save(fig,"fig-3-2-lanes")

    fig, ax = page(3,3,"请求的简化状态与终止","圆角框是状态；箭头标原因；抢占可恢复，取消不恢复",(0,1,2,3,4))
    node(ax,65,260,"等待","请求已存在\n尚未取得本步安排",BLUE)
    node(ax,465,260,"运行","可持续参与调度\n不等于时刻占用设备",ORANGE)
    node(ax,865,260,"正常结束","达到长度／停止条件",GREEN)
    node(ax,465,590,"已抢占","释放引用 · 保留历史\n等待恢复",GREEN)
    node(ax,865,590,"取消／错误终止","不再恢复生成\n协调清理在途状态",MUTED)
    line(ax,[(335,310),(465,310)],"资源允许",color=ORANGE)
    line(ax,[(735,310),(865,310)],"完成",color=GREEN)
    line(ax,[(600,360),(600,590)],"容量压力",(600,475),GREEN)
    line(ax,[(465,640),(200,640),(200,360)],"重新排队",(310,640),BLUE)
    line(ax,[(735,350),(800,350),(800,640),(865,640)],"取消／失败",(800,480),MUTED)
    note(ax,745,"等待状态也可能被取消；实际枚举还区分语法、远端 KV 等等待原因。")
    save(fig,"fig-3-3-states")


def ch4():
    fig, ax = page(4,1,"启动把配置变成全系统约束","配置的一致性跨越前端、调度、存储和执行",(0,1,2,3))
    node(ax,50,230,"模型与任务","结构 · 数据类型 · 长度",BLUE)
    node(ax,465,230,"执行与平台","设备 · 并行 · 后端",PURPLE)
    node(ax,880,230,"资源与策略","容量预算 · 调度 · 编译",ORANGE)
    node(ax,360,440,"统一配置与兼容检查","约束协调 · 必要时反馈有效值",ORANGE,w=480)
    for x in [185,600,1015]:
        line(ax,[(x,330),(x,390),(600,390),(600,440)],color=ORANGE)
    for x,t,c in [(50,"输入限制",BLUE),(465,"调度／KV 容量",GREEN),(880,"实际执行形态",PURPLE)]:
        node(ax,x,650,t,color=c,w=270,h=70)
        line(ax,[(600,540),(600,590),(x+135,590),(x+135,650)],color=c)
    save(fig,"fig-4-1-config")
    flow(4,2,"fig-4-2-startup","从模型准备到引擎就绪","执行侧报告真实资源；核心形成缓存配置，再让各方按同一配置工作",[
        ("配置与兼容","决定模型／平台／执行方式",BLUE),
        ("建立执行资源","设备初始化 · 组织 Worker",PURPLE),
        ("模型准备","识别实现 · 构建 · 加载",PURPLE),
        ("规格与可用容量","执行侧 → 核心",GREEN),
        ("缓存配置与分配","核心决策 → 执行侧落实",GREEN),
        ("预热与就绪","必要准备 · 启动握手",ORANGE)],(0,1,2,3),"逻辑阶段示意；平台内部可穿插编译、检查和额外初始化。")
    fig,ax=page(4,3,"引擎资源与请求状态的寿命","横向只表示先后；请求完成不等于销毁模型与设备存储池",(1,2,3))
    for y,t,start,end,col in [(260,"权重／执行资源",280,1130,PURPLE),(360,"KV 存储池",280,1130,GREEN),(480,"A 的块引用",360,650,GREEN),(570,"B 的块引用",440,950,GREEN),(660,"C 的块引用",750,1080,GREEN)]:
        text(ax,40,y+28,t,17,ha="left")
        ax.add_patch(Rectangle((start,y),end-start,56,color=FILL[col],ec=col,lw=1.8))
    for x in [430,540,650,760,870,980]:
        ax.add_patch(Rectangle((x,755),55,24,color=PURPLE,alpha=.6))
    text(ax,40,767,"当步输入",17,ha="left")
    line(ax,[(280,210),(1130,210)],"启动 → 接纳请求 → 持续执行 → 引擎退出",(700,190))
    save(fig,"fig-4-3-lifetimes")


def ch5():
    fig,ax=page(5,1,"调度器连接请求进度、KV 与执行","调度决定必须同时满足工作预算和状态存储约束",(1,2))
    node(ax,40,330,"输入侧","新请求 · 取消",BLUE)
    node(ax,465,330,"调度器","观察进度 → 形成计划",ORANGE)
    node(ax,890,330,"执行层","张量准备 · 计算 · 采样",PURPLE)
    node(ax,465,610,"KV 管理","可用块 · 命中 · 分配",GREEN)
    node(ax,465,205,"本步约束","token／请求数／其他依赖",ORANGE,h=75)
    line(ax,[(310,380),(465,380)],"请求事件",color=BLUE)
    line(ax,[(735,365),(890,365)],"调度计划",color=ORANGE)
    line(ax,[(890,410),(735,410)],"执行结果",(812,440),PURPLE)
    line(ax,[(600,280),(600,330)],color=ORANGE)
    line(ax,[(540,430),(540,610)],"申请",(500,530),GREEN)
    line(ax,[(660,610),(660,430)],"可行性",(710,530),GREEN)
    save(fig,"fig-5-1-focus")
    fig,ax=page(5,2,"一步计划的形成与资源反馈","先确定可推进工作，再确保它拥有合法存储位置",(1,2),950)
    for y,t,b,c in [(205,"候选请求","当前进度与待处理工作",BLUE),(355,"约束本步工作量","剩余预算 · 长度 · 其他依赖",ORANGE),(505,"申请 KV 位置","已有状态 + 本步新增需求",GREEN),(705,"加入当步计划","记录 token 数与块信息",ORANGE)]:
        node(ax,120,y,t,b,c,w=450,h=100)
    for y in [305,455]:line(ax,[(345,y),(345,y+50)],color=ORANGE)
    line(ax,[(345,605),(345,705)],"分配成功",color=GREEN)
    node(ax,785,505,"暂时不可行","新请求等待\n或选择可抢占请求",MUTED,w=350,h=120)
    line(ax,[(570,555),(785,555)],"分配失败",color=GREEN)
    line(ax,[(960,625),(960,670),(680,670),(680,585),(570,585)],"释放后重试分配",(850,700),GREEN)
    note(ax,870,"失败支线不表示直接加入计划；必须重新满足资源条件。")
    save(fig,"fig-5-2-decision")


def ch6():
    fig,ax=page(6,1,"KV 的管理信息与设备数据","块号表达管理与访问的约定，不是 CPU 内保存全部 KV 数值",(1,2,3))
    node(ax,40,285,"调度器","是否能继续推进？",ORANGE)
    node(ax,465,285,"KV 管理器","请求 → 块\n引用／缓存／可用队列",GREEN)
    node(ax,890,285,"执行侧映射","本步块表与写入位置",PURPLE)
    group(ax,280,590,760,150,"设备存储",PURPLE)
    node(ax,330,650,"每层 KV 张量","模型算子真正读写的内容",GREEN,w=660,h=65)
    line(ax,[(310,335),(465,335)],"分配／释放",color=GREEN)
    line(ax,[(735,335),(890,335)],"计划携带块信息",color=PURPLE)
    line(ax,[(600,385),(600,590)],"映射约束",(600,495),GREEN,True)
    line(ax,[(1025,385),(1025,525),(925,525),(925,590)],"算子访问",(1025,490),PURPLE)
    save(fig,"fig-6-1-ownership")
    fig,ax=page(6,2,"逻辑位置经过块表找到物理槽位","教学块长 2；逻辑顺序连续，物理块可不连续",(2,3))
    node(ax,110,245,"A：逻辑块 0","位置 0、1",BLUE,w=300)
    node(ax,670,245,"A：逻辑块 1","位置 2、3（3 尚未写入）",BLUE,w=400)
    for i,(bid,content,col) in enumerate([(2,"位置 2 │ 空槽",GREEN),(4,"其他内容",MUTED),(7,"位置 0 │ 位置 1",GREEN),(9,"其他内容",MUTED)]):
        node(ax,40+i*290,595,f"物理块 {bid}",content,col,w=250,h=110)
    line(ax,[(260,345),(260,460),(745,460),(745,595)],"块表第 0 项 = 7",(500,435),GREEN,True)
    line(ax,[(870,345),(870,515),(165,515),(165,595)],"块表第 1 项 = 2",(500,540),GREEN,True)
    note(ax,755,"例：位置 2 → 逻辑块 1／偏移 0 → 物理块 2／偏移 0。")
    save(fig,"fig-6-2-blocks")
    fig,ax=page(6,3,"块的引用与缓存内容分别变化","空闲不一定空白；引用为零才可能被重新分配",(2,3))
    node(ax,60,260,"活跃引用","请求仍依赖该块\n不能随意覆盖",GREEN,w=310,h=120)
    node(ax,815,260,"空闲但保有缓存","引用归零\n内容仍可能命中",GREEN,w=310,h=120)
    node(ax,435,600,"被用于新内容","旧缓存身份失效\n建立新的使用关系",ORANGE,w=330,h=120)
    line(ax,[(370,290),(815,290)],"最后一个请求释放引用",(590,265),GREEN)
    line(ax,[(815,350),(370,350)],"新请求命中并重新引用",(590,380),GREEN)
    line(ax,[(970,380),(970,660),(765,660)],"分配／覆盖",(970,510),ORANGE)
    line(ax,[(435,660),(215,660),(215,380)],"开始新的活跃寿命",(215,525),GREEN)
    save(fig,"fig-6-3-lifecycle")


def ch7_roles():
    """一个模型副本、单机三卡 PP 的对象与设备拓扑；可独立重绘。"""
    fig,ax=page(7,1,"一个执行器组织三个设备工作角色",
                "单机 · 一个模型副本 · 六层模型分三段 · 不混用其他并行方式",(1,3),1100)
    group(ax,330,185,540,200,"核心进程（CPU）：EngineCore",ORANGE)
    text(ax,600,245,"取得调度计划 → 调用执行器 → 交回结果",15)
    node(ax,410,280,"Executor × 1","组织整组三个 Worker 的调用",BLUE,w=380,h=80)
    for i,x in enumerate([40,440,840]):
        center=x+160
        group(ax,x,455,320,285,f"工作进程 {i}（CPU）",PURPLE)
        node(ax,x+30,505,f"Worker {i}","本地设备环境 · 阶段交接",PURPLE,w=260,h=80)
        node(ax,x+30,635,f"Runner {i}","持有模型分片 · 组织本地计算",PURPLE,w=260,h=80)
        arrow(ax,[(center,585),(center,635)],color=PURPLE)
        text(ax,center,610,"持有并调用",12,PURPLE,bg=True)
        points=[(600,360),(600,400),(x+280,400),(x+280,505)]
        arrow(ax,points,color=BLUE)
        arrow(ax,[(center,715),(center,845)],color=PURPLE)
        text(ax,center,790,"组织并提交设备计算",13,PURPLE,bg=True)
        detail=["嵌入与第 1～2 层","第 3～4 层","第 5～6 层与输出处理"][i]
        node(ax,x+10,845,f"物理 NPU {i}",detail+"\n本阶段权重与 KV",PURPLE,w=300,h=110)
    for x in [330,730]:
        arrow(ax,[(x,545),(x+140,545)],color=GREEN,dashed=True)
        text(ax,x+70,590,"中间状态\n(4,H)",12,GREEN,bg=True)
    text(ax,600,425,"下发同一轮执行调用；返回路径省略",14,BLUE,bg=True)
    text(ax,600,990,"绿色虚线：Worker 组织的阶段间张量传输，不要求经 CPU 中转",14,GREEN)
    text(ax,600,1030,"蓝线：执行调用　紫线：本地调用／提交计算　H：隐藏维度",14,MUTED)
    save(fig,"fig-7-1-roles")
    # matplotlib 的 SVG 路径默认带行尾空格，规范化后便于审阅 diff。
    svg=OUT/"fig-7-1-roles.svg"
    svg.write_text("\n".join(row.rstrip() for row in svg.read_text().splitlines())+"\n")


def ch7():
    ch7_roles()
    flow(7,2,"fig-7-2-tensors","混合批次从请求计划变成张量","T=4 个输入 token，R=2 条请求；本例两条请求都产生一个输出",[
        ("请求切片","A：1 token\nC：3 token",BLUE),
        ("连续输入","ids (4,)\npositions (4,)",PURPLE),
        ("模型计算","(4,) → (4,H)\n层输入／输出 (4,H)",PURPLE),
        ("选取生成位置","(4,H) → (2,H)",PURPLE),
        ("词表投影","(2,H) × (H,V)\n→ logits (2,V)",PURPLE),
        ("采样并关联请求","(2,V) → tokens (2,)\n归属 A、C",BLUE)],(1,3,4),"查询分段 [0,1,4]，上下文长度 [4,3]；不画张量并行与图形态填充。")
    fig,ax=page(7,3,"从稳定请求状态收集当步输入","MRV2 教学视图：持久槽位与本轮排列是两种索引",(3))
    group(ax,70,225,450,490,"持久请求状态",GREEN)
    for y,t,b in [(300,"槽位 2：C","C 的状态与块映射"),(460,"槽位 5：A","A 的状态与块映射")]:node(ax,120,y,t,b,GREEN,w=350)
    group(ax,730,225,400,490,"当步连续输入",PURPLE)
    node(ax,770,300,"第 0 段：A","本轮处理 1 token",PURPLE,w=320)
    node(ax,770,530,"第 1 段：C","本轮处理 3 token",PURPLE,w=320)
    line(ax,[(470,510),(635,510),(635,350),(770,350)],"收集 A",(635,430),PURPLE)
    line(ax,[(470,350),(580,350),(580,580),(770,580)],"收集 C",(660,605),PURPLE)
    save(fig,"fig-7-3-state")


def ch8():
    fig,ax=page(8,1,"模型语义与执行后端的边界","后端实现共同语义，同时声明自己支持的输入与执行形态",(3))
    node(ax,55,260,"Model Runner","准备输入与元数据",PURPLE,w=310)
    node(ax,475,260,"模型层","层组织与计算语义",PURPLE,w=310)
    node(ax,475,490,"Attention 契约","张量 · 元数据 · 能力",GREEN,w=310)
    node(ax,55,680,"后端实现 A","算子／布局／形态",PURPLE,w=310,h=85)
    node(ax,875,680,"后端实现 B","算子／布局／形态",PURPLE,w=270,h=85)
    line(ax,[(365,310),(475,310)],"张量",color=PURPLE)
    line(ax,[(630,360),(630,490)],"调用",color=PURPLE)
    line(ax,[(210,360),(210,540),(475,540)],"计算说明",(335,515),GREEN)
    line(ax,[(630,590),(630,630),(210,630),(210,680)],"按能力选择",(400,630),PURPLE)
    line(ax,[(630,630),(1010,630),(1010,680)],color=PURPLE)
    save(fig,"fig-8-1-boundary")
    fig,ax=page(8,2,"张量与元数据共同定义 Attention 工作","当前数值并不自带请求边界、历史位置与缓存写入信息",(2,3),900)
    node(ax,45,245,"数值张量","Q (4,4,8)\nK、V 各 (4,2,8)",PURPLE,w=420,h=140)
    node(ax,735,245,"元数据","分段 [0,1,4] · 长度 [4,3]\n块表 · 当前写入槽位",GREEN,w=420,h=140)
    node(ax,390,495,"Attention 后端","读历史 · 写新 KV · 因果计算",PURPLE,w=420,h=110)
    line(ax,[(255,385),(255,440),(500,440),(500,495)],color=PURPLE)
    line(ax,[(945,385),(945,440),(700,440),(700,495)],color=GREEN)
    node(ax,65,720,"输出表示","(4,4,8) → 合并头 (4,32)",PURPLE,w=450,h=85)
    node(ax,735,720,"设备 KV 状态","按层布局和块映射更新",GREEN,w=420,h=85)
    line(ax,[(500,605),(500,665),(290,665),(290,720)],color=PURPLE)
    line(ax,[(700,605),(700,665),(945,665),(945,720)],color=GREEN)
    save(fig,"fig-8-2-metadata")
    fig,ax=page(8,3,"Ascend 在平台边界接入","独立插件提供平台实现；版本与功能组合仍需匹配",(1,2,3))
    node(ax,420,220,"平台插件入口","发现 NPUPlatform",BLUE,w=360)
    node(ax,420,420,"平台配置与能力","选择实现 · 检查约束",ORANGE,w=360)
    line(ax,[(600,320),(600,420)],color=ORANGE)
    for i,(t,b) in enumerate([("NPUWorker","设备／模型／执行"),("Attention 后端","元数据／KV／算子"),("设备通信","并行协作"),("图与编译封装","ACLGraph 等")]):
        x=25+i*300;node(ax,x,660,t,b,PURPLE,w=250)
        line(ax,[(600,520),(600,595),(x+125,595),(x+125,660)],color=PURPLE)
    save(fig,"fig-8-3-platform")


def ch9():
    fig,ax=page(9,1,"CPU 与设备的串行组织和重叠机会","只重叠依赖允许的工作；图宽不是耗时，不表示未知 token 可提前使用",(1,3,4),900)
    def bar(x,y,w,t,c):
        box(ax,x,y,w,66,t,color=c)
    text(ax,45,235,"串行示意",18,ha="left")
    for y,t in [(280,"CPU"),(380,"设备"),(570,"CPU"),(670,"设备")]:text(ax,45,y+33,t,17,ha="left")
    bar(180,280,210,"准备 N",ORANGE);bar(410,380,260,"计算 N",PURPLE)
    bar(690,280,210,"处理／准备",ORANGE);bar(920,380,220,"计算 N+1",PURPLE)
    line(ax,[(390,313),(410,413)],color=MUTED);line(ax,[(670,413),(690,313)],color=MUTED)
    line(ax,[(900,313),(920,413)],color=MUTED)
    text(ax,45,520,"重叠机会",18,ha="left")
    bar(180,570,210,"准备 N",ORANGE);bar(410,670,300,"计算 N",PURPLE)
    bar(435,570,250,"独立部分准备",ORANGE);bar(725,570,190,"依赖就绪",GREEN)
    bar(930,670,210,"计算 N+1",PURPLE)
    line(ax,[(390,603),(410,703)],color=MUTED)
    line(ax,[(710,703),(725,603)],"结果依赖",(790,760),GREEN)
    line(ax,[(915,603),(930,703)],color=PURPLE)
    save(fig,"fig-9-1-timeline")
    fig,ax=page(9,2,"编译与设备图重放的不同职责","编译优化计算实现；图重放减少重复提交；能力不满足时保留回退",(3))
    group(ax,45,215,470,450,"编译路径",PURPLE)
    group(ax,685,215,470,450,"设备图路径",GREEN)
    node(ax,85,285,"模型计算","可分析的计算区域",PURPLE,w=390)
    node(ax,85,515,"编译后的实现","算子组合与计算优化",PURPLE,w=390)
    node(ax,725,285,"选择执行形态","后端能力／形状／缓冲区",GREEN,w=390)
    node(ax,725,515,"捕获与重放","复用可支持的执行序列",GREEN,w=390)
    line(ax,[(280,385),(280,515)],"编译",color=PURPLE)
    line(ax,[(920,385),(920,515)],"准备后复用",color=GREEN)
    line(ax,[(475,565),(725,565)],"可以配合",color=PURPLE)
    text(ax,600,745,"也可选择分段或即时执行；不是两个开关必须同时开或同时关。",17)
    save(fig,"fig-9-2-compile")
    fig,ax=page(9,3,"动态批次选择执行形态","匹配的不只是 token 总数，还包括后端和批次能力",(1,3))
    node(ax,40,270,"本步计划","3 个有效 token（构造例）",ORANGE,w=330)
    node(ax,470,270,"模式分派","检查形态与能力",PURPLE,w=300)
    line(ax,[(370,320),(470,320)],color=ORANGE)
    for y,t,b,c in [(200,"匹配图形态","例如容量 4：3 有效 + 1 填充",GREEN),(420,"分段执行","可捕获片段 + 灵活部分",PURPLE),(640,"即时执行","不满足复用约束时使用",MUTED)]:
        node(ax,860,y,t,b,c,w=310,h=105)
        line(ax,[(770,320),(805,320),(805,y+52),(860,y+52)],color=c)
    note(ax,735,"填充位置必须被正确屏蔽，不能污染用户输出或 KV 状态。",x=40)
    save(fig,"fig-9-3-dispatch")


def ch10():
    # 独立维护第十章的五张差异教学图，避免全册重绘时恢复旧拓扑。
    from render_ch10 import main as render_ch10
    render_ch10()


def ch11():
    fig,ax=page(11,1,"扩展功能围绕主架构接入","扩展点表示影响范围，不代表功能只属于一个组件",(0,1,2,3,4),900)
    for x,t,b,c in [(30,"输入与交付","表达需求／返回结果",BLUE),(465,"调度与 KV","推进／约束／状态",ORANGE),(900,"执行","张量／前向／采样",PURPLE)]:node(ax,x,450,t,b,c,w=270)
    line(ax,[(300,480),(465,480)],"请求",color=BLUE)
    line(ax,[(465,530),(300,530)],"结果",(382,565),BLUE)
    line(ax,[(735,480),(900,480)],"计划",color=PURPLE)
    line(ax,[(900,530),(735,530)],"结果",(817,565),PURPLE)
    for x,y,t,b,c in [(40,220,"多模态／LoRA","输入特征与模型身份",BLUE),(465,220,"混合状态／PD","规格、远端就绪与寿命",GREEN),(890,220,"投机解码","候选、验证与校正",PURPLE),(465,710,"结构化输出","语法状态、掩码与接受",ORANGE)]:
        node(ax,x,y,t,b,c,w=280,h=100)
        if y<450:line(ax,[(x+140,y+100),(x+140,450)],color=c,dashed=True)
        else:line(ax,[(x+140,y),(x+140,550)],color=c,dashed=True)
    save(fig,"fig-11-1-extensions")
    from render_ch11 import main as draw_ch11_details
    draw_ch11_details()


def ch12():
    fig,ax=page(12,1,"一次生成的整机闭环","启动建立约束，平台与多设备组织承接执行；请求循环贯穿全系统",(0,1,2,3,4),900)
    node(ax,40,270,"输入适配","用户需求 → 核心请求",BLUE,w=280)
    node(ax,460,270,"核心／调度","进度 + 预算 → 计划",ORANGE,w=280)
    node(ax,880,270,"执行层","计划 → 张量 → 结果",PURPLE,w=280)
    node(ax,460,590,"KV 管理","分配 · 复用 · 回收",GREEN,w=280)
    node(ax,40,590,"结果交付","token → 文本与完成",BLUE,w=280)
    node(ax,880,590,"设备／后端","权重 · KV · 算子",PURPLE,w=280)
    line(ax,[(320,320),(460,320)],"请求",color=BLUE)
    line(ax,[(740,300),(880,300)],"计划",color=ORANGE)
    line(ax,[(880,345),(740,345)],"结果",(810,380),PURPLE)
    line(ax,[(540,370),(540,590)],"申请状态",(530,475),GREEN)
    line(ax,[(660,590),(660,370)],"容量反馈",(700,475),GREEN)
    line(ax,[(460,350),(390,350),(390,640),(320,640)],"内部结果",(390,500),BLUE)
    line(ax,[(970,370),(970,590)],"执行",(930,480),PURPLE)
    line(ax,[(1080,590),(1080,370)],"结果",(1125,480),PURPLE)
    line(ax,[(180,590),(180,470),(490,470),(490,370)],"停止／取消反馈",(230,470),BLUE)
    text(ax,600,795,"外围条件：启动与配置 · 平台适配 · 编译／图执行 · 多设备协作",17)
    save(fig,"fig-12-1-map")
    flow(12,2,"fig-12-2-investigation","从症状进入职责边界，再寻找证据","每次阅读都带着可验证的问题；症状不是唯一故障原因",[
        ("观察现象","例如首 token 等待长",BLUE),
        ("拆开时间与状态","输入？排队？prefill？",ORANGE),
        ("选责任边界","找相应生产者与消费者",PURPLE),
        ("核查证据","配置／事件／输入输出",GREEN),
        ("情景复盘","加入／结束／容量／取消",ORANGE),
        ("回答后停止","记录版本与适用边界",BLUE)],(0,1,2,3,4),"不以阅读行数衡量理解；证据不足时保留多个解释。")


def ch7_panorama():
    """第 2～7 章阶段地图；独立调用只重绘本图，不改其他插图。"""
    fig, ax = canvas("图 7.4  请求、计划、张量与结果：第 2～7 章全景地图",
                     "主图：单设备 · 核心独立进程 · UniProcExecutor · 普通文本生成；编号对应正文表 7.4。按因果读图，不代表耗时或默认部署。",
                     height=2960, width=2200)

    def card(x, y, w, h, title, body, color=BLUE, size=16):
        box(ax, x, y, w, h, "", color=color)
        heading = text(ax, x+20, y+26, title, 20, color, ha="left", weight="bold")
        detail = text(ax, x+20, y+55, body, size, ha="left")
        detail.set_verticalalignment("top")
        detail.set_linespacing(1.2)
        # 按真实字形边界收紧字号，防止多行文字越过框底或遮住箭头。
        renderer = fig.canvas.get_renderer()
        for artist, max_height in [(heading, 36), (detail, h-65)]:
            bounds = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
            ratio = min((w-40)/abs(bounds.width), max_height/abs(bounds.height), 1)
            artist.set_fontsize(artist.get_fontsize()*ratio)

    def route(points, label, at, color=BLUE, dashed=False):
        arrow(ax, points, label, at, color, dashed, size=16)

    # 启动带：逻辑准备阶段，避免把资源检查画成每请求执行。
    group(ax, 35, 110, 2130, 180, "运行之前｜第 4 章：资源先就绪，随后才能接纳请求", MUTED)
    starts = [
        ("统一配置", "模型／并行／长度\n每步预算／KV 约束", BLUE),
        ("执行侧准备", "设备与通信 · 模型构建\n加载器 → 权重映射", PURPLE),
        ("容量往返", "执行侧 → 规格、可用容量\n核心 → KVCacheConfig", GREEN),
        ("两侧一致后就绪", "核心块管理 ↔ 设备 KV 池\n所需预热 · 长期保留资源", ORANGE),
    ]
    for i, (title, body, col) in enumerate(starts):
        x=55+i*535
        card(x, 163, 505, 108, title, body, col, 15)
        if i: route([(x-30,216),(x,216)], "", None, MUTED)

    text(ax, 40, 330, "运行时拓扑与往返｜第 2、3、5、6、7 章", 23, weight="bold", ha="left")
    text(ax, 2160, 330, "大框＝进程或设备　小框＝对象／职责　箭头＝交接", 16, MUTED, ha="right")
    group(ax, 35, 365, 600, 820, "用户／前端进程：输入适配与结果交付", BLUE)
    group(ax, 735, 365, 1430, 820, "EngineCore 进程：下列核心与执行对象均在 CPU 侧", ORANGE)
    group(ax, 755, 420, 650, 740, "EngineCore：协调接收、调度、执行、更新与输出", ORANGE)
    group(ax, 1490, 420, 650, 740, "单卡执行对象：本例没有另起 Worker 进程", PURPLE)

    card(60, 425, 550, 95, "用户调用  LLM.generate()", "文本／token 输入 + SamplingParams", BLUE)
    card(60, 572, 550, 150, "LLMEngine：输入协调", "① Renderer：渲染／分词\nInputProcessor：校验、建立内部身份\nOutputProcessor 另行登记 RequestState", BLUE)
    route([(335,520),(335,572)], "① 输入", (405,545), BLUE)
    card(60, 785, 550, 110, "EngineCoreClient（本例 SyncMPClient）", "② ADD 请求向核心发送\n⑪ 接收核心的内部结果", BLUE)
    route([(335,722),(335,785)], "EngineCoreRequest", (335,754), BLUE)
    card(60, 960, 550, 180, "OutputProcessor → 用户结果", "⑪ 根据 RequestState 累积、反分词\n停止字符串检查 · 恢复外部请求身份\n⑫ RequestOutput：文本／IDs／结束原因\n离线收齐返回；在线按协议交付", BLUE)
    route([(335,895),(335,960)], "⑪ 内部更新", (335,927), BLUE)
    route([(610,1120),(650,1120),(650,470),(610,470)], "⑫", (650,940), BLUE)

    card(775, 475, 610, 110, "接收路径 → 输入队列 → 主循环", "② EngineCoreRequest → 核心 Request\n保存身份、prompt、输出历史、进度与状态", ORANGE)
    route([(610,815),(685,815),(685,530),(775,530)], "② ADD", (683,505), BLUE)
    card(775, 635, 610, 120, "Scheduler：选择本步工作", "③ add_request / schedule：等待、运行、预算\n⑤ 形成 SchedulerOutput\n⑩ update_from_output：更新、结束、下一轮", ORANGE)
    route([(1080,585),(1080,635)], "③ 登记请求", (1080,610), ORANGE)
    card(775, 820, 610, 120, "KVCacheManager → 块管理／BlockPool", "④ 查前缀、分配块、维护引用与可用容量\n以 Request 的身份和进度解释新增需求\n管理块号与归属；KV 数值在设备上", GREEN)
    route([(960,755),(960,820)], "④ 申请", (910,786), GREEN)
    route([(1190,820),(1190,755)], "块／失败", (1260,786), GREEN)
    card(775, 1005, 610, 135, "核心输出路径：回前端，也准备下一轮", "⑨ 接收 ModelRunnerOutput\n⑩ 原计划 + 结果 → Scheduler 更新请求\n⑪ EngineCoreOutputs → 输出队列／发送", ORANGE)
    route([(775,1080),(685,1080),(685,870),(610,870)], "⑪", (685,1045), BLUE)
    route([(1385,1030),(1420,1030),(1420,730),(1385,730)], "⑩", (1420,980), ORANGE, True)

    card(1510, 475, 610, 110, "Executor：整组 Worker 的调用入口", "⑤ 收到计划；本例为 UniProcExecutor\n组织执行／采样调用，取得约定结果", PURPLE)
    route([(1385,685),(1445,685),(1445,530),(1510,530)], "⑤ 计划", (1445,610), PURPLE)
    card(1510, 635, 610, 120, "Worker：一个设备执行角色", "⑥ 初始化设备与通信，协调本地资源\n把计划交给 Runner；PP 时组织阶段收发\n持有一个 Runner，本例使用 NPU 0", PURPLE)
    route([(1815,585),(1815,635)], "⑥ 调用", (1815,609), PURPLE)
    card(1510, 820, 610, 170, "Runner：本角色的模型执行组织", "⑥ 更新请求记录、块表及当步排列\n⑦ input_ids / positions + Attention 元数据\n⑧ 调模型、选生成位置、投影与采样\n⑨ 以请求 ID 包装执行结果", PURPLE)
    route([(1815,755),(1815,820)], "计划 + 本地状态", (1815,786), PURPLE)
    card(1510, 1040, 610, 100, "返回：Runner → Worker → Executor", "⑨ ModelRunnerOutput → EngineCore\n执行与采样可分入口；结果可经异步对象取得", PURPLE, 15)
    route([(1510,1090),(1385,1090)], "⑨", (1445,1060), PURPLE)
    route([(1510,940),(1470,940),(1470,1090),(1510,1090)], "", None, PURPLE)
    # CPU 到设备的双向边界：向下提交和向上结果分别沿外侧布线。
    route([(2120,915),(2180,915),(2180,1265),(2120,1265)], "⑦／⑧ 提交", (2070,1210), PURPLE)
    route([(1510,1375),(1450,1375),(1450,1120),(1510,1120)], "⑧ 完成", (1448,1210), PURPLE)
    group(ax, 35, 1230, 1360, 260, "同一条请求在四处留下状态｜身份关联，不是跨进程共享一个对象", MUTED)
    for i, (title, body, col) in enumerate([
        ("前端输出状态", "RequestState\n累计 token／文本\n反分词与交付进度", BLUE),
        ("核心计算状态", "Request + 块引用\n等待／运行／结束\n计算进度与输出历史", ORANGE),
        ("Runner 执行状态", "请求记录 + 块表\n首次登记后跨轮保留\n再收集为当步张量", PURPLE),
        ("设备历史状态", "各层 KV 张量\n按块映射访问\n随请求增长与复用", GREEN),
    ]):
        card(55+i*335, 1290, 315, 175, title, body, col, 15)
    group(ax, 1490, 1230, 650, 260, "物理 NPU 0：权重、KV、临时张量与设备算子", PURPLE)
    text(ax, 1510, 1300, "⑧ 编号 (4,) → 隐藏表示 (4,H)\n各层读取历史 KV，写入本步 KV\n生成位置 (2,H) → logits (2,V) → IDs (2,)\n结果留在执行链路；并非逐层搬回 CPU", 18, ha="left").set_verticalalignment("top")

    text(ax, 40, 1530, "沿接口读载荷｜编号与主图、表 7.4 一致；以下为字段节选和本章构造值", 23, ha="left", weight="bold")
    exchanges = [
        ("①～② 输入 → 核心请求", "prompt=[11,27,42]；max_tokens=2\nEngineCoreRequest：ID、prompt_token_ids、参数\nADD 携请求；ABORT 携终止 ID 列表", BLUE),
        ("③～④ 请求 → 工作量与容量判断", "A：已知 4，已处理 3，尚有 1 个位置\nallocate_slots(request_a, num_new_tokens=1)\n返回块对象／块号；无需新增 ≠ 分配失败", GREEN),
        ("⑤～⑥ 计划 → 执行侧状态", "num_scheduled_tokens={A:1, C:3}；总量 4\nA：已有记录的增量；C：完整首次资料\n同时带块变化、结束 ID；不传整份 KV", PURPLE),
        ("⑦～⑧ 状态 + 计划 → 张量", "IDs=[101,31,32,33]；位置=[3,0,1,2]，均 (4,)\n查询边界=[0,1,4] (3,)；上下文=[4,3] (2,)\n块表／槽位决定去哪读写，请求历史仍隔离", PURPLE),
        ("⑨～⑩ 执行结果 → 生命周期更新", "ModelRunnerOutput：req_ids=[A,C]\nsampled_token_ids=[[102],[301]]，附行映射\n原计划 + 结果 → A 达上限结束，C 继续", ORANGE),
        ("⑪～⑫ 内部更新 → 用户可见结果", "EngineCoreOutput：A 新增 [102]，结束原因 LENGTH\n前端接续已有 [101] → [101,102]，恢复文本\nRequestOutput 含文本、IDs、finished 等", BLUE),
    ]
    for i,(title,body,col) in enumerate(exchanges):
        card(40+(i%2)*1090, 1580+(i//2)*195, 1040, 170, title, body, col, 18)

    text(ax, 40, 2190, "换一种部署再定位职责｜纯 PP 三卡示例：1 个核心 → 1 个执行器 → 3 个 Worker → 3 个 Runner → 3 张卡", 22, ha="left", weight="bold")
    card(40, 2235, 460, 240, "核心进程", "EngineCore\n    └─ MultiprocExecutor\n整组调用、资源查询与结果收集\n核心仍持有调度与 KV 管理职责", ORANGE, 18)
    for i in range(3):
        x=620+i*530
        group(ax,x,2235,490,155,f"工作进程 {i} · 流水线阶段 {i}",PURPLE)
        text(ax,x+20,2290,f"Worker {i}：本角色设备与阶段收发\n    └─ Runner {i}：组织本段模型执行",18,ha="left").set_verticalalignment("top")
        box(ax,x+20,2400,450,55,f"NPU {i}：第 {i*2+1}～{i*2+2} 层",color=PURPLE)
        route([(x+245,2375),(x+245,2400)],"",None,PURPLE)
        # 控制总线与中间张量线路分开，避免误读为 Executor 传递隐藏表示。
        route([(500,2265),(565,2265),(565,2220),(x+245,2220),(x+245,2235)],"",None,ORANGE)
        if i<2:
            route([(x+490,2305),(x+530,2305)],"",None,GREEN,True)
    text(ax, 1110, 2510, "橙线：Executor 组织计划与调用；绿色虚线：Worker 间传 IntermediateTensors（隐藏表示 (4,H)，还可带残差等）", 18, MUTED)
    text(ax, 1110, 2545, "本轮前向依次经过三段；输出阶段参与最终生成，按执行协议回收结果。层数为构造值，非三份独立完整模型。", 18, MUTED)

    group(ax,35,2590,2130,285,"生命周期反馈｜这些变化还会改变下一轮计划，不是收到一个 token 就结束整个闭环",MUTED)
    events=[
        ("正常结束", "核心检查上限／EOS 等\n释放 KV 块引用\nfinished_req_ids → Runner 清理\n前端完成交付", ORANGE),
        ("取消／文本停止", "前端 → 核心：终止 ID\n核心停止后续安排\n设备已提交工作可能仍完成\n晚到结果不得重新激活请求", BLUE),
        ("容量不足／抢占", "分配失败 → 调整本轮计划\n普通路径释放引用、重置进度\n历史 token 保留，重入等待\n恢复时更新映射并可能重算", GREEN),
        ("前缀复用", "按计算前缀身份查块\n命中后落实使用引用\n执行侧按块表读取历史\n缓存的是 KV，不是整段回答", GREEN),
    ]
    for i,(title,body,col) in enumerate(events):
        card(55+i*535,2645,505,205,title,body,col,16)
    text(ax,40,2910,"证据：vLLM v0.30.0 接口与本章已核验的执行职责；教学示意，非运行抓取。H＝隐藏维，V＝词表维。在线 HTTP／AsyncLLM 接入左侧，核心职责不变。",16,MUTED,ha="left")
    save(fig, "fig-7-4-panorama")
    svg = OUT / "fig-7-4-panorama.svg"
    svg.write_text("\n".join(s.rstrip() for s in svg.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    for render in [ch3,ch4,ch5,ch6,ch7,ch7_panorama,ch8,ch9,ch10,ch11,ch12]:
        render()
