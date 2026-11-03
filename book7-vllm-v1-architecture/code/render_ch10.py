"""ch10 单卡对照、TP、PP、链路、DP 教学图；仅 CPU 制图。
从工作区 source ./env.sh 后运行本文件。图源随书，图注给出版本和边界。
"""
from render_figures import canvas, box, group, arrow, text, save, BLUE, ORANGE, GREEN, PURPLE, MUTED


def card(ax, x, y, w, h, title, body='', color=PURPLE):
    box(ax, x, y, w, h, '', color=color)
    artists = [text(ax, x+w/2, y+(h*.28 if body else h/2), title, 17, color, weight='bold')]
    if body:
        artists.append(text(ax, x+w/2, y+h*.68, body, 13))
    ax.figure.canvas.draw()
    for a in artists:
        bounds = a.get_window_extent(ax.figure.canvas.get_renderer()).transformed(ax.transData.inverted())
        factor = min(1, (w-20)/abs(bounds.width), (h*(.45 if body else .8))/abs(bounds.height))
        a.set_fontsize(a.get_fontsize()*factor)


def edge(ax, points, label=None, at=None, color=PURPLE, dashed=False):
    arrow(ax, points, label, at, color, dashed, 13)


def overview():
    fig, ax = canvas('图 10.1  单卡 → TP／PP：对象数量相似，模型分工不同',
        '橙：请求计划；紫：本地调用／提交；绿：张量协作；蓝：请求与结果。设备框不属于 CPU 进程。', 1420, 1800)
    titles=['单卡：UniProcExecutor', 'TP=2，PP=1：层内分片', 'TP=1，PP=2：按层分段']
    for i, title in enumerate(titles):
        x=30+i*590
        card(ax,x,115,560,70,title,color=BLUE)
        card(ax,x+100,215,360,65,'前端／核心客户端','接收请求；交付结果',BLUE)
        group(ax,x,325,560,590 if i==0 else 280,'EngineCore 进程',ORANGE)
        card(ax,x+30,380,500,65,'EngineCore','持有 Scheduler 与 Executor',ORANGE)
        card(ax,x+30,460,500,65,'Scheduler ↔ KVCacheManager','请求进度、预算、逻辑块分配',ORANGE)
        edge(ax,[(x+230,280),(x+230,380)],'请求',(x+195,307),BLUE)
        edge(ax,[(x+330,380),(x+330,280)],'结果',(x+365,307),BLUE)
        if i==0:
            card(ax,x+30,540,500,55,'UniProcExecutor',color=ORANGE)
            card(ax,x+30,610,500,55,'Worker → Runner',color=PURPLE)
            edge(ax,[(x+280,525),(x+280,540)],color=ORANGE)
            edge(ax,[(x+280,595),(x+280,610)])
            card(ax,x+30,750,500,110,'模型控制代码：第 1—6 层','完整模型；由 Runner 调用',PURPLE)
            # 属于同一核心进程的模型控制代码单独标注，不画成新进程。
            text(ax,x+280,715,'下框仍是本进程中的模型控制代码',13,MUTED)
            edge(ax,[(x+280,665),(x+280,750)])
            card(ax,x+30,1030,500,120,'NPU 0','完整权重／KV；执行全部层算子',PURPLE)
            edge(ax,[(x+280,860),(x+280,1030)],'提交算子／设备执行',(x+280,950))
        else:
            # Executor 位于核心进程；下方为独立 Worker 进程。
            card(ax,x+30,530,500,55,'MultiprocExecutor',color=ORANGE)
            for j in range(2):
                xx=x+15+j*280
                group(ax,xx,650,250,280,f'工作进程 {j} · CPU',MUTED)
                card(ax,xx+15,700,220,50,f'Worker {j}')
                card(ax,xx+15,775,220,50,f'Runner {j}')
                body=f'第 1—6 层的分片 {j}' if i==1 else ('嵌入＋第 1—3 层' if j==0 else '第 4—6 层＋输出头')
                card(ax,xx+15,850,220,60,'本地模型',body)
                edge(ax,[(xx+125,750),(xx+125,775)])
                edge(ax,[(xx+125,825),(xx+125,850)])
                edge(ax,[(x+280,585),(x+280,620),(xx+125,620),(xx+125,700)],color=ORANGE)
                card(ax,xx+15,1030,220,120,f'NPU {j}', '本地权重／KV\n执行本地模型算子')
                edge(ax,[(xx+125,910),(xx+125,1030)],'提交',(xx+125,980))
            text(ax,x+280,635,'同一 SchedulerOutput → 两个 Worker',13,ORANGE)
            edge(ax,[(x+250,1090),(x+310,1090)],color=GREEN)
            if i==1:
                edge(ax,[(x+310,1120),(x+250,1120)],color=GREEN)
            text(ax,x+280,1190,'层内：部分结果交换／归约' if i==1 else '阶段间：中间张量 → 后段',15,GREEN)
        text(ax,x+280,1270,['不跨卡；模型完整位于一张设备','切分点：模型并行层／权重加载\n通信点：模型层调用 TP 组','切分点：模型构造／本地层范围\n通信点：Worker 调用 PP 组'][i],16,MUTED)
    text(ax,40,1380,'每列均是一份模型副本、一个 Scheduler；模型算子在卡上执行，Python 对象在 CPU 侧。六层为教学构造。',14,MUTED,ha='left')
    save(fig,'fig-10-1-topology')


def tp():
    fig,ax=canvas('图 10.2  TP 层内一次 All-Reduce：两边都执行到同一通信位置',
        '向下是本进程调用；底部绿色是设备数据交换。Worker 的调用未结束，但张量无须返回 Worker 转发。',1350,1500)
    for i,x in enumerate([75,875]):
        group(ax,x,135,550,840,f'工作进程 {i} · CPU 控制代码',MUTED)
        rows=[('Worker → Runner','调用本 rank 的模型'),('模型中的行并行线性层',f'本地乘法得到 P{i}，形状 (2,4)'),('TP All-Reduce 接口','tensor_model_parallel_all_reduce'),('本进程的 TP 组对象','相同参与组 {rank 0, rank 1}'),('设备通信器 → 分布式后端','Ascend 普通路径：PyTorch → HCCL')]
        for j,(a,b) in enumerate(rows):
            y=200+j*150
            card(ax,x+35,y,480,100,a,b,GREEN if j>=2 else PURPLE)
            if j: edge(ax,[(x+275,y-50),(x+275,y)],color=GREEN if j>=2 else PURPLE)
        card(ax,x+35,1040,480,100,f'NPU {i}：设备内存与通信资源',f'本地贡献 P{i} (2,4)；通信后取得 Y (2,4)',PURPLE)
        edge(ax,[(x+275,900),(x+275,1040)],'提交通信操作',(x+275,995),GREEN)
    edge(ax,[(590,1080),(910,1080)],'链路交换＋归约',(750,1040),GREEN)
    edge(ax,[(910,1120),(590,1120)],color=GREEN)
    ax.plot([40,1460],[1010,1010],ls='--',color=MUTED,lw=1)
    text(ax,750,1210,'Y = P0 + P1；两卡各自取得完整结果后，继续执行本地模型。',18,GREEN)
    text(ax,750,1280,'Executor 不归并这些矩阵；一次执行中可发生多次层内通信。省略融合与自定义算子包装。',14,MUTED)
    save(fig,'fig-10-2-communication')


def pp():
    fig,ax=canvas('图 10.3  PP：相同计划下，两个 Worker 接起不同模型阶段',
        'TP=1、PP=2；普通同步路径。绿色为张量交接，蓝色为结果／下一轮 token 的逻辑回程。',1440,1600)
    card(ax,450,120,700,75,'EngineCore · Scheduler · KVCacheManager','本轮 A:1、B:1；接收采样结果并生成下一轮计划',ORANGE)
    card(ax,550,245,500,65,'MultiprocExecutor','向两阶段发命令；从末阶段取普通结果',ORANGE)
    edge(ax,[(800,195),(800,245)],color=ORANGE)
    for i,x in enumerate([70,930]):
        group(ax,x,405,600,600,f'工作进程 {i} · PP 阶段 {i}',MUTED)
        card(ax,x+35,470,530,110,f'Worker {i}', '先调用 Runner；返回中间张量后发往后段' if i==0 else '先接收中间张量；交给 Runner 继续计算')
        card(ax,x+35,640,530,85,f'Runner {i}','组织本阶段输入与模型调用' if i==0 else '组织后段计算；采样形成 token (2,)')
        card(ax,x+35,785,530,120,'本地模型控制代码', '嵌入＋第 1—3 层\n返回 hidden_states、residual，各 (2,4)' if i==0 else '第 4—6 层＋归一化／输出头\n隐藏 (2,4) → logits (2,V)')
        edge(ax,[(800,310),(800,355),(x+300,355),(x+300,470)],color=ORANGE)
        edge(ax,[(x+300,580),(x+300,640)])
        edge(ax,[(x+300,725),(x+300,785)])
        card(ax,x+35,1070,530,110,f'NPU {i}', '前段权重与 KV；前段计算' if i==0 else '后段权重与 KV；后段计算')
        edge(ax,[(x+300,905),(x+300,1070)],'提交本地算子',(x+300,1030))
    edge(ax,[(635,520),(965,520)],'PP 组：发送／接收',(800,485),GREEN)
    text(ax,800,570,'两张设备张量\n各 (2,4)',13,GREEN)
    edge(ax,[(635,1120),(965,1120)],'实际链路',(800,1090),GREEN)
    # 返回使用外侧通道，避免穿过本地模型卡片。
    edge(ax,[(1495,520),(1560,520),(1560,275),(1050,275)],'末阶段执行结果',(1300,250),BLUE)
    edge(ax,[(900,245),(900,195)],color=BLUE)
    edge(ax,[(450,155),(350,155),(350,275),(550,275)],'下一轮计划',(350,225),BLUE,dashed=True)
    text(ax,800,1280,'下一轮反馈：末阶段生成的 token → 核心更新 → 同步 PP 的 new_token_ids → 首阶段',16,BLUE)
    text(ax,800,1360,'绿色横线表示交接关系；设备数据走下方链路，不在两侧 Python 对象之间复制整个张量。',14,MUTED)
    save(fig,'fig-10-3-pipeline')


def links():
    fig,ax=canvas('图 10.4  CPU 发起通信，与张量实际走的线路',
        '各行是可选路径，不是同一机器同时具备的连线；物理链路须由拓扑与运行记录确认。',1250,1700)
    card(ax,80,120,1540,80,'模型并行层（TP）／Worker 阶段交接（PP） → 通信组 → NCCL 或 HCCL','CPU 侧可提交地址、长度、通信任务；不等于 CPU 核心逐元素搬运张量',ORANGE)
    rows=[('同机 GPU：NVLink','GPU 1 显存','NVLink／NVSwitch','GPU 2 显存'),('同机 GPU：PCIe P2P','GPU 1 显存','PCIe 交换／允许 P2P 的 I/O 路径','GPU 2 显存'),('GPU：主机内存中转','GPU 1 显存','PCIe → 主机共享缓冲区 → PCIe','GPU 2 显存'),('跨机 GPU：GPUDirect RDMA','GPU 1 显存','网卡 → IB／RoCE 网络 → 网卡','GPU 2 显存'),('NPU：设备间直接通路','NPU 1 HBM','设备搬运 → HCCS／PCIe → 对端','NPU 2 HBM'),('NPU：设备侧 RoCE 通路','NPU 1 HBM','设备网络接口 → RoCE → 对端接口','NPU 2 HBM')]
    for j,(label,a,b,c) in enumerate(rows):
        y=290+j*140
        text(ax,70,y-22,label,15,MUTED,ha='left')
        for x,w,t in [(70,340,a),(550,600,b),(1290,340,c)]:card(ax,x,y,w,70,t,color=GREEN if x==550 else PURPLE)
        edge(ax,[(410,y+35),(550,y+35)],color=GREEN)
        edge(ax,[(1150,y+35),(1290,y+35)],color=GREEN)
    text(ax,850,1185,'CPU 执行 API ≠ 数据经过 CPU 的 I/O 硬件 ≠ 张量在主机内存中转。上图不表示归约算法的全部轮次。',15,ORANGE)
    save(fig,'fig-10-4-links')


def dp():
    fig,ax=canvas('图 10.5  DP=2 × TP=2：路由分请求，副本内部再分模型计算',
        '普通稠密模型。大框表示副本归属；Core 进程和 Worker 进程分别标出。两个副本不交换逐层计算结果。',1450,1800)
    card(ax,490,120,820,100,'前端／DP 核心客户端：请求路由与结果关联','A → 副本 0；B → 副本 1；保存请求与目标核心的对应关系',BLUE)
    for i,x in enumerate([40,950]):
        group(ax,x,300,810,1030,f'模型副本 {i} · DP rank {i}（资源归属框）',BLUE)
        group(ax,x+30,370,750,285,f'EngineCore {i} 进程',ORANGE)
        card(ax,x+60,420,690,65,f'Core {i} → Scheduler {i} ↔ KVCacheManager {i}', '本副本请求进度／预算／KV 元数据',ORANGE)
        card(ax,x+60,540,690,70,f'Executor {i}', 'A 的计划：例如 A:1' if i==0 else 'B 的计划：例如 B:3',ORANGE)
        edge(ax,[(x+405,485),(x+405,540)],color=ORANGE)
        edge(ax,[(800 if i==0 else 1000,220),(800 if i==0 else 1000,265),(x+355,265),(x+355,420)],color=BLUE)
        edge(ax,[(x+455,420),(x+455,285),(850 if i==0 else 1050,285),(850 if i==0 else 1050,220)],color=BLUE,dashed=True)
        for j in range(2):
            xx=x+30+j*390
            group(ax,xx,735,360,300,f'Worker 进程 · TP rank {j}',MUTED)
            card(ax,xx+20,790,320,55,f'Worker {2*i+j} → Runner {2*i+j}')
            card(ax,xx+20,910,320,85,'本 rank 模型分片',f'处理副本 {i} 的请求；层内分工')
            edge(ax,[(xx+180,845),(xx+180,910)])
            edge(ax,[(x+405,610),(x+405,695),(xx+180,695),(xx+180,790)],color=ORANGE)
            card(ax,xx+20,1130,320,100,f'NPU {2*i+j}','本地权重分片／本地 KV')
            edge(ax,[(xx+180,995),(xx+180,1130)],'提交',(xx+180,1080))
        edge(ax,[(x+370,1170),(x+440,1170)],color=GREEN)
        edge(ax,[(x+440,1200),(x+370,1200)],color=GREEN)
        text(ax,x+405,1285,'仅本副本的 TP 组：层内协作',15,GREEN)
    text(ax,900,1390,'总计：2 Core、2 Scheduler、2 KV 管理器、2 Executor、4 Worker、4 Runner、4 NPU。',16,MUTED)
    save(fig,'fig-10-5-dp')


def main():
    # 书仓代码快照可在原位执行；工作区版本沿用 render_figures 的草稿目录。
    from pathlib import Path
    import render_figures
    if Path(__file__).resolve().parent.name == 'code':
        render_figures.OUT = Path(__file__).resolve().parent.parent / 'figures'
    for draw in (overview, tp, pp, links, dp):
        draw()


if __name__ == '__main__':
    main()
