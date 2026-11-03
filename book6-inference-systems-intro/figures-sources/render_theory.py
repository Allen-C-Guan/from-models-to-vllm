"""全册新增理论图，与获批分页样稿共用坐标与图形语汇。"""
from render_paging import canvas,box,arrow,route,txt,save,cells,BLUE,ORANGE,GREEN,PURPLE,GRAY,PALE
from matplotlib.patches import Rectangle
GROUPS=[('工作负载与成本','01 服务指标\n02 存储层次 · 03 成本估算'),('状态与存储','04 KV状态 · 05 地址分页\n06 分页KV · 07 前缀复用'),('批处理与调度','08 批处理 · 09 排队\n10 分块预填充与抢占'),('计算与表示','11 Attention IO\n12 低精度 · 13 权重 · 14 KV压缩'),('采样与投机','15 概率语义\n16 投机正确性 · 17 收益与候选'),('执行与系统判断','18 主机设备协同\n19 并行拓扑 · 20 综合判断')]
MAPS={15:(4,['条件分布与采样','温度、惩罚与截断','序列概率与搜索']),16:(4,['提议与目标的条件','重合质量与残差修正','首拒截断与状态提交']),17:(4,['平均产出与轮成本','候选来源与树验证','负载、回滚与收益边界'])}
def maps():
 for ch,(focus,labels) in MAPS.items():
  f,a=canvas(f'图 {ch}.{0 if ch==19 else 1}  全书理论地图：本章展开{GROUPS[focus][0]}','同一位置、同一坐标；灰色为全书背景，橙色为本章范围。',850)
  for i,(t,b) in enumerate(GROUPS):
   x=60+(i%3)*445;y=145+(i//3)*190;box(a,x,y,390,140,t,b,ORANGE if i==focus else GRAY)
   if i%3<2:arrow(a,(x+390,y+70),(x+445,y+70))
  txt(a,700,525,f'本章展开：第{ch}章',17,ORANGE);arrow(a,(700,552),(700,582),color=ORANGE)
  for i,l in enumerate(labels):
   x=60+i*445;box(a,x,590,390,95,l,color=ORANGE)
   if i<2:arrow(a,(x+390,637),(x+445,637))
  txt(a,700,750,'箭头表示理解关系；不是请求调用顺序，也不是软件组件或进程拓扑。',15,GRAY);save(f,f'fig-{ch:02d}-01-map')
def sampling():
 f,a=canvas('图 15.2  区间长度就是选择概率','四词构造例：p = [0.4, 0.3, 0.2, 0.1]；均匀随机数 u = 0.65。',620)
 x=100
 for label,p,c in zip('abcd',[.4,.3,.2,.1],[BLUE,GREEN,ORANGE,PURPLE]):
  w=1200*p;box(a,x,210,w,100,label,f'{p}',c);x+=w
 for v in [0,.4,.7,.9,1]:txt(a,100+1200*v,350,str(v),14)
 arrow(a,(880,450),(880,315),'u = 0.65 → b',(880,480),ORANGE)
 save(f,'fig-15-02-sampling')
 f,a=canvas('图 15.3  交换操作顺序，可能改变候选集合','固定原分布 [0.4,0.3,0.2,0.1]；T = 0.5，top-p阈值 = 0.5。',650)
 box(a,60,160,350,120,'同一原分布','a的概率0.4',BLUE)
 box(a,510,160,360,120,'先温度','[16,9,4,1] / 30',ORANGE)
 box(a,980,160,350,120,'再top-p','只留a：概率1',GREEN)
 arrow(a,(410,220),(510,220));arrow(a,(870,220),(980,220))
 box(a,60,370,350,120,'同一原分布','a的概率0.4',BLUE)
 box(a,510,370,360,120,'先top-p','保留a、b：[4,3] / 7',ORANGE)
 box(a,980,370,350,120,'再温度','a、b：[16,9] / 25',GREEN)
 arrow(a,(410,430),(510,430));arrow(a,(870,430),(980,430));save(f,'fig-15-03-order')
def speculation():
 f,a=canvas('图 16.2  验证分布属于各自的条件前缀','第二个候选拒绝后，后面的分布和暂存路径不能继续使用。',720)
 for i,(head,body) in enumerate([('已确认 h','提供p1'),('候选 x1','输入x1后提供p2'),('候选 x2','输入x2后提供p3'),('候选 x3','输入x3后提供p4')]):
  x=55+i*340;box(a,x,180,280,120,head,body,BLUE if i==0 else ORANGE)
  if i<3:arrow(a,(x+280,240),(x+340,240))
 for i,(head,body) in enumerate([('p1 验证x1','接受'),('p2 验证x2','拒绝 → 残差采样r'),('p3、p4','依赖x2，失效')]):box(a,60+i*445,400,390,120,head,body,GREEN if i==0 else (ORANGE if i==1 else GRAY))
 txt(a,700,610,'确认输出：h、x1、r；可保留KV：h、x1。r尚需经过真实前向。',18);save(f,'fig-16-02-alignment')
 f,a=canvas('图 16.3  接受与修正共同填满目标概率','p = [0.5,0.3,0.2]；q = [0.2,0.5,0.3]；总接受概率0.7。',760)
 for i,(v,p,q,m) in enumerate(zip('abc',[.5,.3,.2],[.2,.5,.3],[.2,.3,.2])):
  x=220+i*420;base=540;scale=600
  a.add_patch(Rectangle((x,base-m*scale),160,m*scale,fc=PALE[GREEN],ec=GREEN,lw=2))
  if p>m:a.add_patch(Rectangle((x,base-p*scale),160,(p-m)*scale,fc=PALE[ORANGE],ec=ORANGE,lw=2))
  txt(a,x+80,base+40,v,22);txt(a,x+80,150,f'p={p}；q={q}',18)
  txt(a,x+80,base-m*scale/2,f'接受 {m}',16,GREEN)
  if p>m:txt(a,x+80,base-(p+m)*scale/2,'修正 0.3',16,ORANGE)
 txt(a,700,650,'绿色：min(p,q)；橙色：(p−q)的正部。不是“拒绝后再从完整p重采”。',17);save(f,'fig-16-03-mass')
 f,a=canvas('图 17.2  后面的产出必须先到得了','条件接受率0.8、0.5、0.9；尾概率相乘，平均产出按尾概率求和。',700)
 for i,(prob,label) in enumerate(zip([.8,.4,.36],['至少1个','至少2个','至少3个'])):
  x=180+i*420;a.add_patch(Rectangle((x,520-prob*400),220,prob*400,fc=PALE[GREEN],ec=GREEN,lw=2));txt(a,x+110,490-prob*400,str(prob),20);txt(a,x+110,555,label,18)
 txt(a,700,620,'E[L] = 0.8 + 0.4 + 0.36 = 1.56；未提前停止时 E[N] = 2.56。',18);save(f,'fig-17-02-expectation')
 f,a=canvas('图 17.3  树注意力只沿自己的祖先路径读取','分支同深度使用同一位置号；数组先后顺序不能替代祖先关系。',800)
 box(a,60,290,220,100,'共享前缀 h',color=BLUE)
 for x,y,t in [(400,170,'a'),(400,470,'b'),(740,170,'c'),(740,470,'d')]:box(a,x,y,190,100,t,color=GREEN)
 arrow(a,(280,315),(400,220));arrow(a,(280,365),(400,520));arrow(a,(590,220),(740,220));arrow(a,(590,520),(740,520))
 box(a,1010,170,330,150,'c允许看','h、a、c',ORANGE);box(a,1010,470,330,150,'d允许看','h、b、d',ORANGE)
 txt(a,700,700,'a与b互不可见；c与d互不可见。树计算正确，仍需正确的候选验收算法。',17);save(f,'fig-17-03-tree')
def main():maps();sampling();speculation()
if __name__=='__main__':main()
