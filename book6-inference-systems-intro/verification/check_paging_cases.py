"""第一组独立构造例核验。纯 Python CPU，无 vLLM/NPU、无性能结论。

运行：python code/Book6-推理系统原理/check_paging_cases.py
输出为纸面例的可复算证据，不是生产分配器或并发测试。
"""
from pathlib import Path
from collections import Counter
import copy
import hashlib
import json
import math

OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(parents=True, exist_ok=True)


class Pool:
    """串行教学模型：固定池、前向完成后更新有效长度，显式引用。"""
    def __init__(self):
        self.free = [7,2,5,0,9,1,3,4,6,8,10,11]
        self.rows = {}
        self.refs = Counter()
        self.values = {}
        self.inflight = set()
        self.retiring = set()

    def alloc(self):
        p = self.free.pop(0)
        self.refs[p] = 1
        self.values[p] = [None]*4
        return p

    def append(self, name, items):
        table, size = self.rows.get(name, ([],0))
        # 预检同时计入新增尾块和共享尾块的私有副本。
        growth = (size+len(items)+3)//4 - (size+3)//4
        cow = bool(items and size%4 and self.refs[table[-1]]>1)
        if growth+int(cow)>len(self.free):
            return False
        table = list(table)
        if cow:
            old = table[-1]; new = self.alloc()
            self.values[new][:size%4] = self.values[old][:size%4]
            self.refs[old] -= 1
            table[-1] = new
        for value in items:
            if size%4==0:table.append(self.alloc())
            self.values[table[size//4]][size%4] = value
            size += 1
        self.rows[name]=(table,size)
        self.invariant()
        return True

    def fork(self, src, dst):
        table,size=self.rows[src]
        self.rows[dst]=(list(table),size)
        self.refs.update(table)
        self.invariant()

    def logical(self, name):
        table,size=self.rows[name]
        return [self.values[table[i//4]][i%4] for i in range(size)]

    def release(self, name):
        table,_=self.rows.pop(name)
        for p in table:
            self.refs[p]-=1
            if self.refs[p]==0:
                if p in self.inflight:self.retiring.add(p)
                else:self.free.append(p)
        self.invariant()

    def complete(self, p):
        self.inflight.remove(p)
        if p in self.retiring:
            self.retiring.remove(p);self.free.append(p)
        self.invariant()

    def snapshot(self):
        return {'tables':{n:t for n,(t,s) in self.rows.items()},'lengths':{n:s for n,(t,s) in self.rows.items()},'refs':{str(p):n for p,n in sorted(self.refs.items()) if n},'free_count':len(self.free),'retiring':sorted(self.retiring)}

    def invariant(self):
        actual=Counter(p for table,size in self.rows.values() for p in table)
        assert actual == +self.refs
        assert len(set(self.free))==len(self.free)
        assert not set(self.free)&(set(actual)|self.retiring)
        assert set(self.free)|set(actual)|self.retiring == set(range(12))
        for table,size in self.rows.values():
            assert len(table)==(size+3)//4


def base():
    pool=Pool()
    assert pool.append('A',list('abcdef')) and pool.append('B',list('uvw'))
    return pool


def main():
    checks=[];data={'kind':'constructed_examples_cpu_only','not_evidence_for':['NPU性能','vLLM实际布局','并发正确性']}
    def check(name, condition):
        assert condition,name
        checks.append(name)
    pages=[5,1,6,3]
    translate=lambda v:16*pages[v//16]+v%16
    addresses=[translate(v) for v in range(64)]
    check('全部64个地址保持页内偏移且一一对应',len(set(addresses))==64 and all(addresses[v]%16==v%16 for v in range(64)))
    check('37翻译为101',translate(37)==101)
    check('跨页30..33',addresses[30:34]==[30,31,96,97])
    check('48字节空闲不满足连续24',sum([16]*3)==48 and max([16]*3)<24)
    data['os_addresses']={'table':pages,'v37':translate(37),'cross_page':addresses[30:34]}
    page_cost=[]
    for p in [4,8,16]:
        counts=[(n+p-1)//p for n in [5,9,17]]
        page_cost.append({'size':p,'counts':counts,'allocated':sum(counts)*p,'waste':sum(counts)*p-31})
        waste=[((n+p-1)//p)*p-n for n in range(1,p+1)]
        check(f'页大小{p}的范围和均匀期望',min(waste)==0 and max(waste)==p-1 and sum(waste)/p==(p-1)/2)
    check('页大小手算表',[x['allocated'] for x in page_cost]==[40,48,64])
    data['page_cost']=page_cost
    # COW的独立字节数组演示，验证修改副本不影响原页。
    a=[10,20,30,40];c=list(a);c[1]=99
    check('OS私有写隔离',a[1]==20 and c[1]==99)
    p=base();growth=[p.snapshot()]
    for token in ['g','h','i']:
        check(f'追加{token}',p.append('A',[token]))
        growth.append(p.snapshot())
    check('增长块表及空闲量',[s['tables']['A'] for s in growth]==[[7,2],[7,2],[7,2],[7,2,0]] and [s['free_count'] for s in growth]==[9,9,9,8])
    check('逻辑数据顺序',p.logical('A')==list('abcdefghi'))
    p.release('A');check('释放整条A后只留B',p.snapshot()['free_count']==11 and p.rows['B'][0]==[5])
    data['growth']=growth
    p=base();p.fork('A','C');branch=[p.snapshot()]
    assert p.append('A',['x']);branch.append(p.snapshot())
    check('COW后C原前缀未变',p.logical('C')==list('abcdef'))
    assert p.append('C',['y']);branch.append(p.snapshot())
    check('A/C分歧隔离',p.logical('A')==list('abcdefx') and p.logical('C')==list('abcdefy'))
    p.release('A');branch.append(p.snapshot());p.release('C');branch.append(p.snapshot())
    check('共享分支各行空闲量',[s['free_count'] for s in branch]==[9,8,8,9,11])
    data['shared_branch']=branch
    # 只余一个块时，先需要COW再需要新尾块，总需求两个；不能部分修改。
    p=base();p.fork('A','C');assert p.append('D',['z']*32)
    before=copy.deepcopy(p.__dict__)
    check('缺块时拒绝整轮',not p.append('A',['x','x','x']))
    check('失败无部分修改',p.__dict__==before)
    # 共享页尚被C持有；A私有尾页在途，释放需等待。
    p=base();p.fork('A','C');p.append('A',['x']);p.inflight.add(0);p.release('A')
    check('在途块不提前复用',0 in p.retiring and 0 not in p.free and len(p.free)==8)
    p.complete(0);check('完成后才能回收',0 in p.free and len(p.free)==9)
    # 单头Attention直接与按块求和是两条计算路径。
    keys=[0,math.log(2),math.log(6)];values=[10,20,40]
    weights=[math.exp(x) for x in keys]
    direct=sum(w*v for w,v in zip(weights,values))/sum(weights)
    storage={4:([keys[0],keys[1]],[10,20]),1:([keys[2],math.log(100)],[40,999])}
    nums=[];dens=[]
    for j,physical in enumerate([4,1]):
        k,v=storage[physical];valid=min(2,3-j*2)
        w=[math.exp(x) for x in k[:valid]]
        nums.append(sum(x*y for x,y in zip(w,v[:valid])));dens.append(sum(w))
    paged=sum(nums)/sum(dens);wrong=sum(n/d for n,d in zip(nums,dens))/2
    unmasked=(sum(nums)+100*999)/(sum(dens)+100)
    check('分块全局归一化等于直接结果',math.isclose(paged,direct) and math.isclose(paged,290/9))
    check('局部等权平均反例',math.isclose(wrong,85/3) and not math.isclose(wrong,direct))
    check('尾槽残留必须mask',not math.isclose(unmasked,direct))
    data['attention']={'weights':weights,'block_numerators':nums,'block_denominators':dens,'direct':direct,'paged':paged,'wrong_equal_mean':wrong,'wrong_unmasked':unmasked}
    kv_cost=[]
    for b in [2,4,8]:
        counts=[(n+b-1)//b for n in [6,3]];slots=sum(counts)*b
        kv_cost.append({'b':b,'counts':counts,'slots':slots,'empty_slots':slots-9,'bytes':slots*16,'entries':sum(counts)})
    check('KV大小比较',[x['bytes'] for x in kv_cost]==[160,192,256])
    check('容量预算',12*4*16==768 and 2*24*16==768)
    check('滑窗不能过早释放部分有效块',set(range(4,8))&set(range(5,10))=={5,6,7})
    data['kv_cost']=kv_cost;data['checks']=checks
    data['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'cases-verified.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(f'{len(checks)} checks passed; CPU constructed examples only')


if __name__=='__main__':main()
