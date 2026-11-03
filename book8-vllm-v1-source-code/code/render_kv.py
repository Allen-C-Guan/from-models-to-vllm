"""第 6–7 章元数据关系、块映射与恢复图。"""
from render_figures import canvas,box,txt,arrow,save,lanes,BLUE,ORANGE,GREEN,PURPLE,GRAY
from render_architecture import maps,klass,relation


def main():
    maps((6,7))
    fig,ax=canvas('图 6.1  谁管理请求块表，谁管理物理编号','UML：实线＝关联；虚线＝使用依赖；空心三角＝泛化。数量标在关联角色旁。',1220,
        footer='依据 v0.30.0 普通 KV group 路径；特殊独立池与混合策略留第 25 章。框不代表进程。')
    klass(ax,50,145,360,'Scheduler','kv_cache_manager','schedule()',ORANGE)
    klass(ax,520,145,360,'KVCacheManager','coordinator · block_pool','allocate_slots() · free()',GREEN)
    klass(ax,990,145,360,'KVCacheBlocks','blocks: tuple[Sequence, ...]','get_block_ids()',GREEN)
    klass(ax,50,525,440,'BlockPool','blocks · free_block_queue\ncached_block_hash_to_block','get_new_blocks() · free_blocks()',GREEN)
    klass(ax,590,525,600,'KVCacheCoordinator','single_type_managers · block_pool','allocate_new_blocks() · free()',GREEN,True)
    klass(ax,590,890,600,'SingleTypeKVCacheManager','req_to_blocks: 请求 → 块序列','allocate_new_blocks() · free()',GREEN,True)
    klass(ax,50,890,440,'FullAttentionManager','全注意力保留策略','find_longest_cache_hit()',GREEN)
    relation(ax,[(410,245),(520,245)],'1',(470,210))
    relation(ax,[(700,355),(700,525)],'coordinator  1',(810,440))
    relation(ax,[(880,245),(990,245)],'返回接口',(935,210),'dependency')
    relation(ax,[(590,625),(490,625)],'block_pool  1',(540,580))
    relation(ax,[(890,735),(890,890)],'single_type_managers  1..*',(1070,815))
    relation(ax,[(490,990),(590,990)],'继承',(540,950),'generalization')
    relation(ax,[(650,890),(650,810),(270,810),(270,735)],'block_pool  1',(425,805))
    save(fig,'fig-6-1-classes')

    fig,ax=canvas('图 6.2  位置增长，不一定需要新块','单组、全注意力、块大小 128；编号来自独立 CPU 元数据实验，不展示设备张量。',920)
    box(ax,70,150,520,140,'请求 A：逻辑 token 位置','原来 0–4 → 下一步增加位置 5',ORANGE)
    box(ax,810,150,520,140,'请求 B：逻辑 token 位置','0–5',ORANGE)
    box(ax,70,420,520,130,'A 的组内块表','逻辑块 0 → 物理编号 1',GREEN)
    box(ax,810,420,520,130,'B 的组内块表','逻辑块 0 → 物理编号 2',GREEN)
    arrow(ax,[(330,290),(330,420)],'位置 // 128 → 表下标')
    arrow(ax,[(1070,290),(1070,420)],'位置 // 128 → 表下标')
    for x,n,label,c in [(70,0,'null，占位',GRAY),(410,1,'A 引用',GREEN),(750,2,'B 引用',GREEN),(1090,3,'空闲',GRAY)]:
        box(ax,x,700,240,110,f'块 {n}',label,c)
    arrow(ax,[(330,550),(330,620),(530,620),(530,700)],'编号寻址',(530,610))
    arrow(ax,[(1070,550),(1070,620),(870,620),(870,700)],'编号寻址',(870,610))
    save(fig,'fig-6-2-mapping')

    fig,ax=canvas('图 7.1  请求引用与缓存身份是两条状态轴','普通完整块的状态图；条件写在箭头上。null block 与部分命中复制不在本图范围。',1060)
    box(ax,70,180,520,150,'可再分配，无缓存身份','ref_cnt = 0；在空闲队列',GRAY)
    box(ax,810,180,520,150,'使用中，尚无缓存身份','ref_cnt > 0；不在空闲队列',GREEN)
    box(ax,810,620,520,150,'使用中，有缓存身份','ref_cnt > 0；可查找，不能任意覆盖',GREEN)
    box(ax,70,620,520,150,'可回收，有缓存身份','ref_cnt = 0；可命中，也可淘汰',ORANGE)
    arrow(ax,[(590,225),(810,225)],'get_new_blocks')
    arrow(ax,[(810,300),(590,300)],'最后引用释放')
    arrow(ax,[(1070,330),(1070,620)],'完整前缀身份登记',(1160,470))
    arrow(ax,[(810,665),(590,665)],'最后引用释放')
    arrow(ax,[(590,735),(810,735)],'命中后 touch')
    arrow(ax,[(330,620),(330,330)],'再分配前淘汰旧身份',(430,470))
    txt(ax,700,920,'身份登记只表示元数据进展；设备数据是否可读，还要满足执行顺序与同步约束。',15,GRAY)
    save(fig,'fig-7-1-states')

    fig,ax=canvas('图 7.2  同步路径中的抢占与恢复','源码推导的时序；不是本书 A/B 运行中已观测的抢占。虚线＝返回或通知。',1040)
    lanes(ax,[('Scheduler',ORANGE),('Request / 队列',ORANGE),('KV 管理',GREEN),('执行侧',PURPLE)],[170,510,850,1190],900)
    arrow(ax,[(170,265),(850,265)],'allocate_slots')
    arrow(ax,[(850,330),(170,330)],'None：容量不足',dashed=True)
    arrow(ax,[(170,405),(510,405)],'选中并移出 running')
    arrow(ax,[(170,475),(850,475)],'释放被抢占请求的块引用')
    arrow(ax,[(170,550),(510,550)],'PREEMPTED；computed=0；入 waiting')
    arrow(ax,[(170,625),(1190,625)],'计划携带抢占 ID，撤下旧状态',dashed=True)
    arrow(ax,[(170,730),(850,730)],'后续重新接纳：命中查询与分配')
    arrow(ax,[(170,825),(1190,825)],'携带重建进度与块表的新计划')
    save(fig,'fig-7-2-recovery')


if __name__=='__main__':main()
