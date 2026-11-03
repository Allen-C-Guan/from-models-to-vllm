"""参数与异构状态章节的结构图。"""
from render_figures import canvas,box,txt,arrow,save,BLUE,ORANGE,GREEN,PURPLE,GRAY
from render_architecture import maps,klass,relation

def quant():
    maps((24,))
    fig,ax=canvas('图 24.1  配置选择方法，方法装配内核','UML：实线关联；虚线创建依赖；空心三角指向父类。省略其他量化实现。',1530)
    klass(ax,60,150,540,'QuantizationConfig','格式与层级策略','get_quant_method(layer,prefix)',BLUE,True)
    klass(ax,60,480,540,'Fp8Config','激活策略 / 忽略层 / block shape','get_quant_method()',BLUE)
    relation(ax,[(330,480),(330,360)],'继承',(410,420),'generalization')
    klass(ax,800,150,540,'QuantizeMethodBase','参数生命周期契约','create / postprocess / apply',PURPLE,True)
    klass(ax,800,480,540,'LinearMethodBase','线性层方法契约','create_weights / apply',PURPLE,True)
    relation(ax,[(1070,480),(1070,360)],'继承',(1150,420),'generalization')
    klass(ax,800,810,540,'Fp8LinearMethod','quant_config · fp8_linear','create / postprocess / apply',PURPLE)
    relation(ax,[(1070,810),(1070,690)],'继承',(1150,750),'generalization')
    relation(ax,[(600,585),(700,585),(700,875),(800,875)],'«create»',(700,735),'dependency')
    klass(ax,60,810,540,'LinearBase','quant_method · weight / scales','具体并行层组织forward',PURPLE)
    relation(ax,[(600,970),(800,970)],'持有 1方法',(700,1020))
    box(ax,800,1210,540,160,'选中的 FP8 线性内核','层配置 + 参数布局\nprocess_weights / apply_weights',GREEN)
    relation(ax,[(1070,1020),(1070,1210)],'fp8_linear 1',(1180,1110))
    txt(ax,700,1440,'参数属于层；方法协调表示与使用；内核负责满足硬件和布局约束的计算。',15,GRAY)
    save(fig,'fig-24-1-classes')
    fig,ax=canvas('图 24.2  参数的生命周期与计算契约','离线FP8普通per-tensor分支示意；多scale、转置与重打包随实现不同。',1160)
    rows=[('创建参数壳','weight [N,K] + scale；记录分片 / loader；选择兼容内核',BLUE),('读取 checkpoint','按权重名称、TP分片与scale规则填入数据',BLUE),('装载后转换','统一表示 → [K,N] / 运行scale → 内核进一步处理',ORANGE),('运行期消费','X [T,K] + 当前参数与scale → Y [T,N]',GREEN)]
    for i,(t,b,c) in enumerate(rows):
        y=160+220*i;box(ax,80,y,1240,155,t,b,c)
        if i:arrow(ax,[(700,y-65),(700,y)],'')
    txt(ax,700,1065,'转换可以替换Parameter；已预加工权重必须避免重复转换。',15,GRAY)
    save(fig,'fig-24-2-lifecycle')



def hybrid():
    maps((25,))
    fig,ax=canvas('图 25.1  状态描述与管理行为分开扩展','UML：泛化指向父类；实线关联；左侧描述布局，右侧维护请求块归属。',1510)
    klass(ax,60,140,540,'KVCacheSpec','block_size / page字节 / 布局','单层状态契约',BLUE,True)
    klass(ax,800,140,540,'SingleTypeKVCacheManager','spec · block_pool · req_to_blocks','allocate / free / cache hit',GREEN,True)
    klass(ax,60,490,540,'AttentionSpec','KV头 / 维度 / dtype','Full / SlidingWindow等派生',BLUE)
    klass(ax,60,840,540,'MambaSpec','状态shapes / dtypes / mode','page字节与状态保留量',BLUE)
    relation(ax,[(330,490),(330,350)],'继承',(405,420),'generalization')
    relation(ax,[(60,945),(30,945),(30,245),(60,245)],'继承',(30,700),'generalization')
    klass(ax,800,490,540,'SlidingWindowManager','窗口 / 额外保留量','过期区间 / 窗口命中',GREEN)
    klass(ax,800,840,540,'MambaManager','检查点 / align状态块位置','离散命中 / 状态块归属',GREEN)
    relation(ax,[(1070,490),(1070,350)],'继承',(1160,420),'generalization')
    relation(ax,[(1340,945),(1380,945),(1380,245),(1340,245)],'继承',(1380,710),'generalization')
    relation(ax,[(800,945),(600,945)],'关联规格',(700,910))
    box(ax,60,1210,1280,170,'HybridKVCacheCoordinator → 各组管理器 → BlockPool','规格决定布局，组规则决定有效性，协调者求共同边界；执行端另持有实际张量。',ORANGE)
    save(fig,'fig-25-1-classes')
    fig,ax=canvas('图 25.2  块表位置与状态恢复边界','构造例：窗口W4、块B4；已处理位置0至6，下一位置7。非模型运行日志。',1140)
    for x,t,b,c in [(60,'逻辑列0：位置0–3','过期 → null',GRAY),(520,'逻辑列1：位置4–7','保留历史4/5/6 + 当前7',GREEN),(980,'逻辑列2：位置8–11','仍是原逻辑列2',GREEN)]:box(ax,x,180,360,180,t,b,c)
    txt(ax,700,430,'释放实体块；不压紧列号。额外预留块与窗口历史是两回事。',15,GRAY)
    box(ax,80,600,1240,160,'递推检查点：可用边界 {4,12}','S4 与 S12 是不同历史的状态；有 S12 不代表能截出 S8。',PURPLE)
    box(ax,80,850,1240,160,'共同恢复边界','Full {0,4,8,12} ∩ 组2 {0,4,12} ∩ 组3 {0,4,8} → 最大共同边界4',ORANGE)
    arrow(ax,[(700,760),(700,850)],'各组共同接受')
    save(fig,'fig-25-2-boundary')

if __name__=='__main__':
    import sys
    hybrid() if len(sys.argv)>1 and sys.argv[1]=='25' else quant()
