"""01–04/07–10的地图和机制图；纯CPU构造例。"""
from render_theory import MAPS,maps
from render_paging import canvas,box,arrow,txt,save,BLUE,GREEN,ORANGE,GRAY,PURPLE,PALE,FONT
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
MAPS.clear();MAPS.update({1:(0,['请求的依赖链','时间轴与指标','全书问题与阅读路线']),2:(0,['CPU与设备','容量、流量、带宽','复用与执行依赖']),3:(0,['形状 → FLOPs与字节','Roofline约束','关键路径与估算边界']),4:(1,['因果不变量与增量计算','KV形状与架构差异','状态寿命与完成进度']),7:(1,['相同计算的身份','哈希链与基数树','命中、引用与驱逐']),8:(2,['权重与批量复用','迭代级成员变化','打包布局与请求边界']),9:(2,['到达与服务需求','人数时间面积','选择、公平与准入']),10:(2,['chunk与因果范围','工作预算与干扰','抢占、换出和重算'])})
def core():
 f,a=canvas('图 1.2  请求时间轴：首输出与后续间隔','构造例，客户端观测；测量起点与计数口径固定。',650)
 arrow(a,(100,300),(1320,300),'时间 / ms',(1250,340))
 for t,l in [(0,'请求起点'),(12,'第1 token'),(17,'第2 token'),(23,'第3 token'),(30,'第4 token')]:
  x=100+t*38;a.plot([x,x],[280,315],color=BLUE);txt(a,x,240,l,14);txt(a,x,355,str(t),16)
 box(a,100,420,360,100,'TTFT = 12ms',color=ORANGE);box(a,515,420,360,100,'ITL = 5 / 6 / 7ms',color=GREEN);box(a,930,420,360,100,'TPOT = 6ms',color=BLUE);save(f,'fig-01-02-timeline')
 f,a=canvas('图 2.2  不同存储边界，不同传输成本','概念层次：不指定某一NPU的缓存级数或实测带宽。',650)
 labels=[('主机内存','输入、权重加载'),('设备 HBM','权重、KV、工作区'),('片上存储','缓存或局部缓冲'),('运算单元','矩阵与向量运算')]
 for i,(t,b) in enumerate(labels):
  x=50+i*345;box(a,x,200,270,145,t,b,BLUE if i<2 else GREEN)
  if i<3:arrow(a,(x+270,270),(x+345,270),'搬运',(x+310,390))
 txt(a,700,480,'权重已经在HBM，不代表运算单元不再读取它；每条边必须分别计量。',18);save(f,'fig-02-02-hierarchy')
 # 标准科学坐标：构造设备，不使用真实硬件数值。
 f,a=plt.subplots(figsize=(12,7));xs=[.1,.2,.5,1,2,5,10,20,50,100,200,500,1000];a.loglog(xs,[min(100,x) for x in xs],color=BLUE,lw=3)
 a.axvline(100,color=GRAY,ls='--');a.scatter([10,100],[10,100],color=ORANGE,s=70)
 a.set_xlabel('算术强度 I（FLOP / byte）',fontproperties=FONT,fontsize=14);a.set_ylabel('可达性能上界（TFLOP / s）',fontproperties=FONT,fontsize=14)
 a.set_title('图 3.2  构造设备：100 TFLOP/s、1 TB/s；脊点100 FLOP/byte',fontproperties=FONT,fontsize=16,pad=20)
 a.grid(True,which='both',alpha=.2);a.text(.15,45,'实际性能可低于上界\n曲线不是硬件实测',fontproperties=FONT,fontsize=14,color=GRAY)
 f.tight_layout();save(f,'fig-03-02-roofline')
 f,a=canvas('图 4.2  采出token，比计算它的KV早一个步骤','提示词a、b、c；每行是一个已完成前向后的状态。',730)
 for i,(head,tok,kv) in enumerate([('prefill结束','a b c x','a b c'),('计算x后','a b c x y','a b c x'),('计算y后','a b c x y z','a b c x y')]):
  y=170+i*150;box(a,60,y,300,100,head,color=BLUE);box(a,470,y,380,100,'已确认token',tok,ORANGE);box(a,960,y,380,100,'已计算KV',kv,GREEN)
 save(f,'fig-04-02-progress')
 f,a=canvas('图 7.2  当前块相同，父身份不同，缓存键仍不同','按块的哈希链：键依赖此前全部前缀及必要计算条件。',700)
 for y,first,parent in [(180,'a b c d','H0'),(430,'u v w x','J0')]:
  box(a,60,y,350,130,'第一块',first,BLUE);box(a,520,y,350,130,'父身份',parent,ORANGE);box(a,980,y,350,130,'第二块：e f g h',f'Hash({parent}, efgh, 条件)',GREEN)
  arrow(a,(410,y+65),(520,y+65));arrow(a,(870,y+65),(980,y+65))
 save(f,'fig-07-02-hash')
 f,a=canvas('图 7.3  基数树在公共前缀结束处拆边','先abcdef，再abcdxy，再abcdxz；物理KV可以分页存放。',720)
 box(a,50,260,300,140,'输入序列','abcdef\nabcdxy · abcdxz',BLUE);box(a,470,280,230,100,'abcd',color=ORANGE)
 box(a,840,150,180,90,'ef',color=GREEN);box(a,840,450,180,90,'x',color=GREEN);box(a,1170,370,150,90,'y',color=GREEN);box(a,1170,550,150,90,'z',color=GREEN)
 arrow(a,(350,330),(470,330));arrow(a,(700,300),(840,195));arrow(a,(700,360),(840,495));arrow(a,(1020,475),(1170,415));arrow(a,(1020,515),(1170,595));save(f,'fig-07-03-radix')
 f,a=canvas('图 8.2  每轮重新组批，结束位置可让给新请求','仅比较4轮的成员；固定轮宽不代表不同批耗时相同。',780)
 for i in range(4):txt(a,430+i*240,150,f'轮{i+1}',17)
 rows=[('固定批 A',['A','A','A','A']),('固定批 B',['B','B','空','空']),('连续批 A',['A','A','A','A']),('连续批 B/C',['B','B','C','C'])]
 for j,(label,v) in enumerate(rows):
  y=210+j*115;txt(a,155,y+35,label,16)
  for i,s in enumerate(v):box(a,330+i*240,y,200,75,s,color=GRAY if s=='空' else (GREEN if s=='C' else BLUE))
 txt(a,700,710,'C在轮3开始前已就绪；若尚未prefill，还需安排其输入计算。',17);save(f,'fig-08-02-batches')
 f,a=canvas('图 9.2  同一面积，既可按请求求和，也可按时间求和','A停留4秒、B停留2秒、C停留3秒；完整观测窗口0–6秒。',740)
 for j,(label,start,end) in enumerate([('A',0,4),('B',1,3),('C',3,6)]):
  y=190+j*110;txt(a,80,y+30,label,20);box(a,180+start*180,y,(end-start)*180,65,f'{end-start} 请求·秒',color=BLUE)
 arrow(a,(180,550),(1290,550))
 for t in range(7):txt(a,180+t*180,590,str(t),15)
 txt(a,700,650,'面积9；平均人数9/6=1.5；到达率3/6=0.5；平均停留9/3=3秒。',17);save(f,'fig-09-02-little')
 f,a=canvas('图 10.2  切分长prefill，改变连续等待区间','独立串行教学时间线：prefill总12ms；每次decode 2ms。',680)
 txt(a,115,220,'整体prefill',16);box(a,270,170,630,100,'C prefill 12ms',color=ORANGE);box(a,900,170,105,100,'D\n2ms',color=GREEN)
 txt(a,115,430,'分三段',16)
 x=270
 for i in range(3):
  box(a,x,380,210,100,f'C块{i+1}\n4ms',color=ORANGE);x+=210;box(a,x,380,105,100,'D\n2ms',color=GREEN);x+=105
 txt(a,700,570,'下排总18ms；decode有更多机会，C首输出更晚。真实混合批成本需另测。',17);save(f,'fig-10-02-chunks')
def main():maps();core()
if __name__=='__main__':main()
