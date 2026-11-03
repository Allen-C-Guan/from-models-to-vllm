"""CPU纸面例与边界验证；不运行模型，不代替数学证明或教学审读。"""
from pathlib import Path
from fractions import Fraction as F
import math,itertools,json,hashlib
OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(parents=True, exist_ok=True)
CHECKS=[]
def check(name,cond,detail=None):
    assert cond,name
    CHECKS.append({'name':name,'detail':detail})
def close(a,b):return abs(a-b)<1e-9
def softmax(z):
    m=max(z);v=[math.exp(x-m) for x in z];return [x/sum(v) for x in v]
def nucleus(p,rho):
    order=sorted(range(len(p)),key=lambda i:-p[i]);keep=[];total=0
    for i in order:
        keep.append(i);total+=p[i]
        if total>=rho:break
    return [p[i]/total if i in keep else 0 for i in range(len(p))]
def sampling():
    p=[.4,.3,.2,.1];z=[math.log(x) for x in p]
    cold=softmax([x/.5 for x in z]);check('15 温度平方并归一',all(close(a,b/30) for a,b in zip(cold,[16,9,4,1])))
    first=nucleus(cold,.5);trunc=nucleus(p,.5);other=[x*x/sum(y*y for y in trunc) for x in trunc]
    check('15 温度和top-p不交换',first==[1,0,0,0] and close(other[0],.64),[first,other])
    check('15 正负重复惩罚',[2/2,-2*2]==[1,-4])
    check('15 beam两步反例',.3*.9>.4*.4)
    p=list(map(F,['.5','.3','.2']));q=list(map(F,['.2','.5','.3']))
    m=[min(x,y) for x,y in zip(p,q)];beta=sum(m);res=[max(x-y,0)/(1-beta) for x,y in zip(p,q)]
    check('16 重合质量与残差',m==[F(1,5),F(3,10),F(1,5)] and res==[1,0,0])
    wrong=[x+(1-beta)*y for x,y in zip(m,p)]
    check('16 直接目标重采有偏',wrong==list(map(F,['.35','.39','.26'])),list(map(float,wrong)))
    # 枚举四分之一网格的全部三词分布；含零支撑、完全相同和不相交。
    dist=[[F(a,4),F(b,4),F(4-a-b,4)] for a in range(5) for b in range(5-a)]
    for p,q in itertools.product(dist,repeat=2):
        mass=[min(a,b) for a,b in zip(p,q)];beta=sum(mass)
        out=mass if beta==1 else [m+max(a-b,0) for m,a,b in zip(mass,p,q)]
        assert out==p
    check('16 全枚举225对分布的质量守恒',True)
    check('16 首拒状态保留', ['h','x','r']!=['h','x','y'] and len(['h','x'])==2)
    a=[.8,.5,.9];reach=[];prod=1
    for x in a:prod*=x;reach.append(prod)
    n=1+sum(reach);check('17 条件尾和式',close(n,2.56),{'tail':reach,'output':n})
    check('17 成本翻转',close(16/n,6.25) and 29/n>10)
    check('17 边际扩展反而变慢',18/(n+.18)>16/n)
    for alpha in [0,.2,.8,1]:
        direct=sum(alpha**i for i in range(5));formula=5 if alpha==1 else (1-alpha**5)/(1-alpha)
        assert close(direct,formula)
    check('17 几何式含零一边界',True)
    # token节点a,b,c,d；祖先分别[],[],[a],[b]，加自身与共享前缀。
    visible=[{'h','a'},{'h','b'},{'h','a','c'},{'h','b','d'}]
    check('17 树mask隔离兄弟', 'b' not in visible[2] and 'a' not in visible[3])
    s=1;states=[]
    for x in [2,3,4]:s=2*s+x;states.append(s)
    check('17 递归回滚',[4,11,26]==states and 2*states[0]+5==13)
def core():
    times=[12,17,23,30]
    check('01 TTFT与TPOT',times[0]==12 and (times[-1]-times[0])/(len(times)-1)==6)
    check('02 小传输固定成本',close(5e-6+1000/100e9,5.01e-6))
    check('02 搬运计算流水',2+3*3==11 and 3*(2+3)==15)
    check('03 矩阵FLOPs与流量',[(2*m*4*4,2*(m*4+16+m*4)) for m in [1,8]]==[(32,48),(256,160)])
    check('03 Roofline下界',max(1e12/100e12,20e9/1e12)==.02)
    check('03 Amdahl',close(1/(.7+.3/3),1.25))
    check('04 头共享容量',[2*2*6*h*2*2 for h in [4,2,1]]==[384,192,96])
    check('07 复用后缀配对',2*4+2*3//2==11 and 6*7//2-11==10)
    check('08 请求与token计数',len([2,1,3])==3 and sum([2,1,3])==6)
    intervals=[(0,4),(1,3),(3,6)]
    area=sum(b-a for a,b in intervals);by_time=sum(sum(a<=t<b for a,b in intervals) for t in range(6))
    check('09 人数时间面积',area==by_time==9 and area/6==.5*(area/3))
    check('09 排序改变平均完成时间',sum([8,9,10])/3==9 and close(sum([1,2,10])/3,13/3))
    pairs=[]
    for start in [0,2,4]:
        for q in range(start,start+2):pairs.extend((q,k) for k in range(q+1))
    check('10 chunk与完整因果范围',pairs==[(q,k) for q in range(6) for k in range(q+1)])
    check('10 swap双向成本',close(2*64e6/8e9,.016))
def remaining():
    def summarize(z,v):
        if not z:return None
        m=max(z);w=[math.exp(x-m) for x in z]
        return m,sum(w),sum(a*b for a,b in zip(w,v))
    def merge(a,b):
        if a is None:return b
        if b is None:return a
        m=max(a[0],b[0]);sa=math.exp(a[0]-m);sb=math.exp(b[0]-m)
        return m,sa*a[1]+sb*b[1],sa*a[2]+sb*b[2]
    z=[0,math.log(2),math.log(6)];v=[10,20,40]
    r=merge(summarize(z[:2],v[:2]),summarize(z[2:],v[2:]))
    check('11 分块三状态',close(r[1],1.5) and close(r[2],145/3) and close(r[2]/r[1],290/9))
    for mask in itertools.product([False,True],repeat=3):
        zs=[x for x,m in zip(z,mask) if m];vs=[x for x,m in zip(v,mask) if m]
        full=summarize(zs,vs);part=None
        for x,y,m in zip(z,v,mask):part=merge(part,summarize([x],[y]) if m else None)
        assert full is None and part is None or all(close(x,y) for x,y in zip(full,part))
    check('11 掩码与空块全部8种组合',True)
    check('11 大分数稳定',all(math.isfinite(x) for x in softmax([1000,1001])))
    q=lambda x:min(3,max(0,round(x)+1))-1
    check('12 舍入与截断',q(.6)==1 and q(2.7)==2)
    check('12 离群值使小值归零',[round(x) for x in [-.2,.1,.3,7]]==[0,0,0,7])
    check('12 元数据位宽',4+16/32==4.5 and 4+16/128==4.125)
    loss=lambda x,y:x*x+x*y+y*y
    check('13 补偿与最终离散误差',close(loss(.4,-.2),.12) and close(loss(.4,-.6),.28) and close(loss(.4,.4),.48))
    # 连续补偿在附近扰动下不增加损失。
    check('13 约束极小点',all(loss(.4,-.2+e)>=loss(.4,-.2)-1e-12 for e in [-1,-.1,0,.1,1]))
    check('13 缩放等价',10*.1+1*1==(10/10)*(.1*10)+1*1)
    # 数值差分检查softmax对分数扰动的一阶导数。
    z=[.1,.7,-.2];p=softmax(z);h=1e-6;delta=[0,h,0];p2=softmax([a+b for a,b in zip(z,delta)])
    predicted=[pi*((h if i==1 else 0)-p[1]*h) for i,pi in enumerate(p)]
    check('14 softmax扰动',max(abs(b-a-d) for a,b,d in zip(p,p2,predicted))<1e-12)
    residue=[];quantized=[];history=[]
    for i in range(1,6):
        residue.append(i)
        if len(residue)==4:quantized.extend(residue);residue=[]
        history.append([quantized.copy(),residue.copy()])
    check('14 K残差流式刷新',history[2]==[[],[1,2,3]] and history[3]==[[1,2,3,4],[]] and history[4]==[[1,2,3,4],[5]])
    check('14 混合位宽容量',10*2+40*.5==40 and 200*.5*.25==25)
    # 依赖驱动三阶段流水线，独立准备/计算/返回资源。
    prep=device=ret=0
    for i in range(3):
        prep+=2;device=max(device,prep)+5;ret=max(ret,device)+1
    check('18 流水线递推',ret==18 and 3*(2+5+1)==24)
    check('18 准备成本摊销',12/.2==60 and close(12/(.2-.15),240))
    logical=[{7,2},{5},{7,0}];physical=set().union(*logical)
    check('20 共享实际物理容量',len(physical)==4 and sum(map(len,logical))==5 and len(physical)*64==256)
    check('20 合格吞吐',1000/10==100 and 800/10==80)
def main():
    sampling();core();remaining()
    data={'kind':'CPU constructed examples, not hardware measurements','checks':CHECKS,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'theory-checks.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(len(CHECKS),'checks passed')
if __name__=='__main__':main()
