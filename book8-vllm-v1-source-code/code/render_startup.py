"""第 1–2 章：对象实例、装配关系与启动依赖。"""
from matplotlib.patches import Rectangle
from render_figures import canvas,box,txt,arrow,save,BLUE,ORANGE,GREEN,PURPLE,GRAY
from render_architecture import maps,klass,relation


def main():
    maps((1,2))
    fig,ax=canvas('图 1.1  同一进程中的实例引用','UML 对象图：下划线头部＝实例别名：类型；连线＝实例链接，文字＝属性角色。',940,
        footer='依据 v0.30.0 InprocClient 路径；教学别名不是源码变量；省略其他协作者。')
    rows=[(90,170,520,'engine: LLMEngine','前端对象',BLUE),(790,170,520,'client: InprocClient','客户端对象',BLUE),
          (790,550,520,'core: EngineCore','核心对象',ORANGE),(90,550,520,'executor: UniProcExecutor','执行器对象',PURPLE)]
    for x,y,w,name,body,c in rows:
        box(ax,x,y,w,160,name,body,c)
        ax.plot([x+50,x+w-50],[y+65,y+65],c=c,lw=1)
    relation(ax,[(610,250),(790,250)],'engine_core',(700,220))
    relation(ax,[(1050,330),(1050,550)],'engine_core',(1150,435))
    relation(ax,[(790,630),(610,630)],'model_executor',(700,600))
    txt(ax,700,815,'同一个属性名必须结合所有者阅读：engine.engine_core ≠ client.engine_core。',16,GRAY)
    save(fig,'fig-1-1-instances')

    fig,ax=canvas('图 2.1  配置驱动的装配','UML：实线箭头＝可导航关联；虚线 «create»＝创建依赖。省略参数与辅助对象。',1220,
        footer='依据 v0.30.0；展示 InprocClient 分支。Executor 框表示接口，实际实例由所选 executor_class 创建。')
    klass(ax,60,145,360,'EngineArgs','用户入口参数','create_engine_config()',BLUE)
    klass(ax,520,145,360,'VllmConfig','模型／缓存／并行／调度配置','—',GRAY)
    klass(ax,980,145,360,'LLMEngine','vllm_config · engine_core','from_engine_args()',BLUE)
    klass(ax,980,520,360,'InprocClient','engine_core','__init__()',BLUE)
    klass(ax,520,520,360,'EngineCore','model_executor · scheduler','__init__()',ORANGE)
    klass(ax,60,520,360,'Executor','vllm_config','初始化／执行契约',PURPLE,True)
    klass(ax,520,900,360,'SchedulerInterface','—','schedule()',ORANGE,True)
    relation(ax,[(420,250),(520,250)],'«create»',(470,215),'dependency')
    relation(ax,[(980,250),(880,250)],'vllm_config',(930,215))
    relation(ax,[(1160,355),(1160,520)],'engine_core',(1250,435))
    relation(ax,[(980,630),(880,630)],'«create»',(930,590),'dependency')
    relation(ax,[(980,695),(880,695)],'engine_core',(930,745))
    relation(ax,[(520,625),(420,625)],'model_executor',(470,585))
    relation(ax,[(520,695),(420,695)],'«create»',(470,740),'dependency')
    relation(ax,[(700,730),(700,900)],'scheduler',(775,820))
    txt(ax,230,950,'EngineCore 先创建执行器，\n取得 KV 配置后\n再实例化所选调度类。',15,GRAY)
    txt(ax,1130,950,'VllmConfig 被传给多方；\n为可读性省略其余\n配置关联线。',15,GRAY)
    save(fig,'fig-2-1-assembly')

    fig,ax=canvas('图 2.2  启动必须兑现哪些依赖','活动／依赖视图：箭头标明本例先后条件；框宽、间距均不代表耗时。',1050)
    box(ax,60,150,350,150,'配置解析','模型与平台约束\n选出执行器类',BLUE)
    box(ax,525,150,350,150,'执行侧准备','Worker → init_device\n加载模型与权重',PURPLE)
    box(ax,990,150,350,150,'询问需求与容量','模型 KV spec\n本例：手动 64 MiB',GREEN)
    arrow(ax,[(410,225),(525,225)],'构造')
    arrow(ax,[(875,225),(990,225)],'模型已存在')
    box(ax,990,475,350,150,'求解缓存配置','执行侧：布局与存储\n调度侧：块容量',GREEN)
    box(ax,525,475,350,150,'兑现设备资源','initialize_from_config\ncompile_or_warm_up_model',PURPLE)
    box(ax,60,475,350,150,'核心完成装配','创建 Scheduler\n选择 step_fn',ORANGE)
    arrow(ax,[(1165,300),(1165,475)],'需求 + 预算',(1240,385))
    arrow(ax,[(990,550),(875,550)],'配置')
    arrow(ax,[(525,550),(410,550)],'可兑现容量')
    box(ax,60,795,1280,100,'外层入口完成准备','LLM 还要完成 Renderer 预热等工作，之后才提交 A、B',BLUE)
    arrow(ax,[(235,625),(235,795)],'核心返回',(300,710))
    save(fig,'fig-2-2-startup')


if __name__=='__main__':main()
