"""第19章平台绑定与继承。"""
from render_figures import canvas,box,txt,arrow,save,BLUE,ORANGE,GREEN,PURPLE,GRAY
from render_architecture import maps,klass,relation


def main():
    maps((19,))
    fig,ax=canvas('图 19.1  平台契约与真实继承边界','UML：空心三角指向父类。每行独立，不把Worker与Runner当作同一继承树。',1440)
    pairs=[('Platform','NPUPlatform','平台能力 / 实现选择',BLUE),('WorkerBase','NPUWorker','设备资源 / Runner组织',PURPLE),('GPUModelRunner','NPUModelRunner','本章仅MRV2路径',PURPLE),('AttentionBackend','AscendAttentionBackend','元数据 / 实现工厂',GREEN)]
    for i,(base,sub,note,c) in enumerate(pairs):
        y=150+i*285
        klass(ax,60,y,520,base,'上游契约或实现','',GRAY)
        klass(ax,820,y,520,sub,note,'',c)
        relation(ax,[(820,y+105),(580,y+105)],'继承',(700,y+70),'generalization')
    txt(ax,700,1360,'NPUWorker持有Runner；它没有继承GPUWorker。实现选择与对象包含分开读。',15,GRAY)
    save(fig,'fig-19-1-classes')
    fig,ax=canvas('图 19.2  绑定时点决定调试入口','启动装配与运行查证的活动图；每个框是阶段，不是一个进程。',1140)
    box(ax,60,160,360,180,'发现平台','entry point → 限定类名\n解析 current_platform',BLUE)
    box(ax,520,160,360,180,'形成有效配置','worker_cls / Attention选择\n设备约束与资源参数',ORANGE)
    box(ax,980,160,360,180,'创建实际对象','Worker → Runner → 模型\n注册表选择具体层类',PURPLE)
    arrow(ax,[(420,250),(520,250)],'平台能力')
    arrow(ax,[(880,250),(980,250)],'实现选择')
    box(ax,60,490,560,180,'注册 / 补丁','注册：创建对象时查替代类\n补丁：导入时改变模块名称绑定',GRAY)
    box(ax,780,490,560,180,'运行核对','type(obj) → MRO → 方法来源\n再检查有效配置与分支',GREEN)
    arrow(ax,[(620,580),(780,580)],'影响具体实现')
    arrow(ax,[(1160,340),(1160,490)],'实际实例')
    box(ax,60,840,1280,170,'基础案例的检查结果','BalanceScheduler + balance_enabled=False → super().schedule\nNPU MRV2 execute_model → 平台前后处理 + 上游主执行逻辑',GREEN)
    arrow(ax,[(1060,670),(1060,840)],'与记录对齐')
    txt(ax,700,1065,'固定源码说明可执行分支；真实记录说明本次选择了什么。',15,GRAY)
    save(fig,'fig-19-2-binding')


if __name__=='__main__':main()
