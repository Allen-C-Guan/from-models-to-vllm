"""11–14/18–20理论地图与机制图；数据来自对应正文的构造例。"""
from render_theory import MAPS,maps
from render_paging import canvas,box,arrow,txt,save,BLUE,GREEN,ORANGE,GRAY,PURPLE,PALE
from matplotlib.patches import Rectangle
MAPS.clear();MAPS.update({11:(3,['稳定softmax','三状态分块合并','IO与并行划分']),12:(3,['格点与截断','尺度、粒度与格式','存储、计算、累加']),13:(3,['输入相关性与补偿','通道缩放','反量化与执行布局']),14:(3,['架构、表示、保留集合','K/V非对称分组','质量、容量与维护成本']),18:(5,['提交、完成与依赖','重叠与资源条件','编译、捕获与重放']),19:(5,['计算与状态归属','通信与物理拓扑','保留原并行专题']),20:(5,['容量、速度与质量','工作负载与测量','进入架构与源码'])})
def diagrams():
 f,a=canvas('图 11.2  将两块的分子、分母换到同一指数尺度','分数[0, ln2]与[ln6]；V=[10,20]与[40]。',680)
 box(a,60,175,340,145,'块A：基准 ln2','l=3/2，u=25',BLUE);box(a,60,405,340,145,'块B：基准 ln6','l=1，u=40',BLUE)
 box(a,560,275,350,175,'统一基准 ln6','l=(3/2)/3+1=3/2\nu=25/3+40=145/3',ORANGE);box(a,1070,285,270,150,'最终输出','u/l=290/9',GREEN)
 arrow(a,(400,245),(560,310),'乘 exp(ln2−ln6)=1/3',(490,210));arrow(a,(400,470),(560,415),'乘1',(485,500));arrow(a,(910,360),(1070,360));save(f,'fig-11-02-merge')
 f,a=canvas('图 11.3  小分数块消费后丢弃，输出状态持续更新','只展示一个query块；不同query块可能重新读取KV。',750)
 box(a,50,210,290,135,'HBM：K/V历史','逐块读取',BLUE);a.add_patch(Rectangle((425,140),890,420,fc='none',ec=ORANGE,ls='--',lw=2));txt(a,850,175,'片上工作集',18,ORANGE)
 box(a,470,245,330,125,'当前Q块 × 当前K块','临时分数 Bq × Bk',BLUE);box(a,945,245,320,125,'更新每行 m / l / u','临时分数随后丢弃',ORANGE)
 arrow(a,(340,280),(470,280),'读块');arrow(a,(800,305),(945,305),'贡献');arrow(a,(1090,370),(1090,470),'下一块继续归约',(1100,415));txt(a,870,510,'保留Q块、必要K/V块与每行摘要，不保存完整S²分数矩阵',14)
 box(a,940,610,330,75,'最终O写回HBM',color=GREEN);arrow(a,(1100,550),(1100,610));save(f,'fig-11-03-io')
 f,a=canvas('图 12.2  均匀格点：范围内舍入，范围外截断','s=1、z=1、无符号码0/1/2/3 → 实数−1/0/1/2。',650)
 arrow(a,(100,330),(1310,330),'实数',(1280,370))
 for value in [-1,0,1,2]:
  x=350+value*240;a.plot([x,x],[310,350],color=BLUE,lw=2);txt(a,x,390,str(value),18)
 for value,target,label in [(.6,1,'0.6 → 1：舍入'),(2.7,2,'2.7 → 2：截断')]:
  x=350+value*240;tx=350+target*240;txt(a,x,190,label,18,ORANGE);arrow(a,(x,225),(tx,310),color=ORANGE)
 txt(a,700,500,'未截断时最近舍入误差≤s/2；截断误差可以超过s/2。',18);save(f,'fig-12-02-grid')
 f,a=canvas('图 12.3  同一激活矩阵：行是token，列是通道','边框表示哪些元素共用尺度；分组必须指明轴。',710)
 for z,title in enumerate(['整个张量一份尺度','每个token一份尺度','每个通道一份尺度']):
  x=75+z*450;txt(a,x+160,180,title,18)
  for r in range(3):
   for c in range(4):
    a.add_patch(Rectangle((x+c*80,250+r*80),80,80,fc=PALE[BLUE],ec='white'));txt(a,x+c*80+40,290+r*80,f'x{r}{c}',14)
  if z==0:a.add_patch(Rectangle((x,250),320,240,fc='none',ec=ORANGE,lw=3))
  elif z==1:
   for r in range(3):a.add_patch(Rectangle((x,250+r*80),320,80,fc='none',ec=ORANGE,lw=3))
  else:
   for c in range(4):a.add_patch(Rectangle((x+c*80,250),80,240,fc='none',ec=ORANGE,lw=3))
 txt(a,700,600,'尺度更细可以隔离离群值，也增加元数据与恢复尺度的成本。',18);save(f,'fig-12-03-granularity')
 f,a=canvas('图 13.2  GPTQ思想的二维构造例','G=[[1,0.5],[0.5,1]]；整数格点；按第1、第2权重依次固定。',760)
 labels=[('原权重','[0.6, 0.6]'),('固定第1个并补偿','[1, 0.4]'),('再量化第2个','[1, 0]')]
 for i,(t,b) in enumerate(labels):
  x=50+i*450;box(a,x,200,350,150,t,b,BLUE if i==0 else ORANGE)
  if i<2:arrow(a,(x+350,275),(x+450,275))
 box(a,500,430,350,135,'逐步方案总误差','[0.4, −0.6]；损失0.28',GREEN);box(a,950,430,350,135,'独立RTN总误差','[0.4, 0.4]；损失0.48',GRAY)
 txt(a,700,650,'中间0.4是连续补偿值；最终仍须量化。例子不证明全离散全局最优。',17);save(f,'fig-13-02-compensate')
 f,a=canvas('图 14.2  非对称量化先明确数据轴','单头K/V均为(S,d)；演示4个token、2个通道。',720)
 for z,title in enumerate(['K：同一通道跨token分组','V：同一token跨通道分组']):
  x=200+z*680;txt(a,x+140,180,title,18)
  for r in range(4):
   for c in range(2):
    a.add_patch(Rectangle((x+c*140,240+r*70),140,70,fc=PALE[BLUE],ec='white'));txt(a,x+c*140+70,275+r*70,str([[1,100],[2,110],[3,120],[4,130]][r][c]) if z==0 else f'v{r}{c}',17)
  if z==0:
   for c in range(2):
    for g in range(2):a.add_patch(Rectangle((x+c*140,240+g*140),140,140,fc='none',ec=ORANGE,lw=3))
  else:
   for r in range(4):a.add_patch(Rectangle((x,240+r*70),280,70,fc='none',ec=GREEN,lw=3))
 txt(a,700,610,'K示意每2个token成组；V示意本行2个通道成组。不是矩阵转置。',17);save(f,'fig-14-02-axes')
 f,a=canvas('图 14.3  选择性保留使用不同的预测信号','删掉历史位置会改变未来Attention；没有分布不变保证。',720)
 rows=[(200,'H2O','生成中累积注意力','历史重要位置＋近期窗口'),(420,'SnapKV','提示词观察窗投票、池化','选定提示词位置＋观察窗')]
 for y,t,b,end in rows:
  box(a,50,y,250,130,t,color=BLUE);box(a,450,y,390,130,b,color=ORANGE);box(a,990,y,350,130,end,color=GREEN);arrow(a,(300,y+65),(450,y+65));arrow(a,(840,y+65),(990,y+65))
 txt(a,700,630,'已保留位置还需保持原位置身份；压缩后数组下标不是原序列位置。',17);save(f,'fig-14-03-selection')
 f,a=canvas('图 18.2  跨流依赖：消费者等待生产者完成','CPU可以继续提交；事件边负责设备侧先后关系。',720)
 txt(a,100,250,'流S1',18);txt(a,100,470,'流S2',18)
 box(a,220,200,360,100,'复制输入X',color=BLUE);box(a,680,200,220,100,'记录事件E',color=ORANGE);arrow(a,(580,250),(680,250))
 box(a,680,420,220,100,'等待事件E',color=ORANGE);box(a,1030,420,300,100,'读取X并计算WX',color=GREEN);arrow(a,(790,300),(790,420),'完成依赖',(900,360));arrow(a,(900,470),(1030,470));txt(a,700,610,'缓冲X在最后一次读取完成前不可被下一批覆盖。',18);save(f,'fig-18-02-events')
 f,a=canvas('图 18.3  三批独立工作：24ms串行 → 18ms理想流水线','每批准备2ms、计算5ms、返回处理1ms；具备独立资源与缓冲。',720)
 jobs=[('准备',[(0,2,'1'),(2,4,'2'),(7,9,'3')],BLUE),('设备',[(2,7,'1'),(7,12,'2'),(12,17,'3')],ORANGE),('返回',[(7,8,'1'),(12,13,'2'),(17,18,'3')],GREEN)]
 for j,(label,times,col) in enumerate(jobs):
  y=190+j*130;txt(a,100,y+35,label,18)
  for s,e,l in times:box(a,200+s*60,y,(e-s)*60,70,l,color=col)
 arrow(a,(200,610),(1300,610))
 for t in [0,2,7,12,17,18]:txt(a,200+t*60,650,str(t),13)
 save(f,'fig-18-03-pipeline')
 f,a=canvas('图 18.4  编译与图执行回答不同问题','上：生成执行程序；下：组织重复提交。并非所有动态行为均可捕获。',720)
 for x,t,b,col in [(50,'计算描述','张量运算与依赖',BLUE),(520,'编译/优化','融合、布局、特化',ORANGE),(990,'设备kernel','具体执行程序',GREEN)]:box(a,x,180,350,135,t,b,col)
 arrow(a,(400,247),(520,247));arrow(a,(870,247),(990,247))
 for x,t,b,col in [(50,'任务依赖','复制、kernel、事件',BLUE),(520,'构造/捕获并准备','可重复执行的图',ORANGE),(990,'反复重放','符合契约的新输入',GREEN)]:box(a,x,430,350,135,t,b,col)
 arrow(a,(400,497),(520,497));arrow(a,(870,497),(990,497));arrow(a,(1165,315),(695,430),'作为图中操作',(950,370));save(f,'fig-18-04-graph')
 f,a=canvas('图 20.2  局部优化会改变其他系统条件','箭头表示影响关系，不表示单调收益；所有判断受质量与SLO约束。',780)
 box(a,60,200,340,140,'有效KV占用','长度、共享、表示',BLUE);box(a,530,200,340,140,'可容纳的请求','容量与工作区约束',ORANGE);box(a,1000,200,340,140,'批量与计算形状','复用、执行与通信',GREEN)
 arrow(a,(400,270),(530,270));arrow(a,(870,270),(1000,270));box(a,530,480,340,140,'等待与用户延迟','调度、公平、准入',ORANGE);arrow(a,(1160,340),(870,520),'改变每轮成本',(1100,430));arrow(a,(530,550),(230,340),'反过来限制并发',(270,490));txt(a,700,700,'容量更多不自动意味着延迟更小；单项加速比不能直接相乘。',18);save(f,'fig-20-02-system')
 f,a=canvas('图 20.3  闭合与开放负载的区别在于谁决定下次发送','上路等待响应；下路按外部计划继续到达。',720)
 box(a,55,200,300,110,'闭合：发请求1',color=BLUE);box(a,520,200,300,110,'收到响应1',color=GREEN);box(a,990,200,340,110,'随后发请求2',color=BLUE);arrow(a,(355,255),(520,255),'等待');arrow(a,(820,255),(990,255),'响应触发')
 box(a,55,450,300,110,'开放：独立时钟',color=ORANGE)
 for x,t in [(520,'t=0发请求1'),(925,'t=1发请求2')]:box(a,x,450,350,110,t,color=BLUE)
 arrow(a,(355,500),(520,500));arrow(a,(870,500),(925,500));txt(a,700,640,'下路请求2不等待请求1完成；必须记录实际到达是否跟上计划。',17);save(f,'fig-20-03-load')
if __name__=='__main__':maps();diagrams()
