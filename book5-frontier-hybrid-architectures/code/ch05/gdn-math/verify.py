"""复现样稿数学检查；输出为CPU数值验证，不是设备性能实测。"""
from pathlib import Path
import argparse, ast, hashlib, importlib.util, json, math, re
import torch
import torch.nn.functional as F
import sympy as sp
from gdn_math import recurrent, chunkwise, affine, block, coefficients, toy

BOOK = Path(__file__).resolve().parents[3]
ROOT = next((p for p in Path(__file__).resolve().parents if (p/"env.sh").is_file()), Path.cwd())
OUT = None
CHECKS = []

def equal(name, x, y, atol=1e-10, rtol=1e-10):
    error = float((x-y).abs().max().detach())
    assert torch.allclose(x, y, atol=atol, rtol=rtol), (name, error)
    CHECKS.append(dict(name=name, max_abs_error=error, atol=atol, rtol=rtol))

def serial(x):
    if isinstance(x, torch.Tensor): return x.detach().tolist()
    if isinstance(x, dict): return {k:serial(v) for k,v in x.items()}
    if isinstance(x, list): return [serial(v) for v in x]
    return x

def data(n, seed=42):
    gen=torch.Generator().manual_seed(seed)
    def rand(*shape): return torch.randn(*shape, generator=gen, dtype=torch.float64)
    q,k,v=rand(2,2,n,3), F.normalize(rand(2,2,n,3),dim=-1), rand(2,2,n,2)
    a,b=rand(2,2,n).sigmoid(),rand(2,2,n).sigmoid()
    return q,k,v,a,b,rand(2,2,3,2)

def main():
    global OUT
    parser=argparse.ArgumentParser(description="GDN CPU数学验证；拒绝覆盖已有结果")
    parser.add_argument("--out-name",default="run1")
    args=parser.parse_args()
    if not args.out_name or Path(args.out_name).name!=args.out_name or args.out_name in (".",".."):
        raise ValueError("--out-name 必须为单个目录名")
    OUT=ROOT/"log/gdn-math"/args.out_name
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    # 多头多样本、非方形状态、尾块；边界不只用同一组随机数。
    for n in (1,7,8,9):
        for variant in ('regular','zero_state','no_decay','no_write','zero_decay'):
            args=list(data(n,42+n))
            if variant=='zero_state': args[-1]=torch.zeros_like(args[-1])
            if variant=='no_decay': args[3]=torch.ones_like(args[3])
            if variant=='no_write': args[4]=torch.zeros_like(args[4])
            if variant=='zero_decay': args[3][...,n//2]=0
            ro,rs,_=recurrent(*args); ao,ass=affine(*args)
            equal(f'{n}/{variant}/affine-out',ro,ao);equal(f'{n}/{variant}/affine-state',rs,ass)
            for c in (1,3,8):
                co,cs,_=chunkwise(*args,chunk=c)
                equal(f'{n}/{variant}/C{c}/output',ro,co);equal(f'{n}/{variant}/C{c}/state',rs,cs)
    args=data(7)
    # 与三种记忆情景直接对应的解析期望，不让两条实现互相作唯一裁判。
    t=lambda a:torch.tensor(a,dtype=torch.float64)
    for name,p,expected in (('same-key',[1,0],[3,0]),('orthogonal',[0,1],[0,7]),('overlap',[.6,.8],[1.8,5.6])):
        case=(t([p]),t([[1,0]]),t([[3,0]]),t([1]),t([1]),t([[5,0],[0,7]]))
        for path in (recurrent,chunkwise):
            equal(f'geometry/{name}/{path.__name__}',path(*case)[0][0],t(expected))
    repeat=(t([[1,0],[1,0]]),t([[1,0],[1,0]]),t([[5,0],[3,0]]),t([1,1]),t([1,1]),t([[0,0],[0,0]]))
    equal('same-key-rewrite',chunkwise(*repeat,chunk=1)[0],t([[5,0],[3,0]]))
    nw=list(args);nw[4]=torch.zeros_like(nw[4]);no,ns,_=chunkwise(*nw)
    gamma=nw[3].cumprod(-1)
    equal('no-write-analytic-state',ns,gamma[...,-1,None,None]*nw[-1])
    equal('no-write-analytic-output',no,gamma[..., :,None]*(nw[0]@nw[-1]))
    # 改动位置4以后的q/k/v/门；0..3输出严格不变。
    changed=[x.clone() for x in args]
    for i in range(5):
        if i<3: changed[i][...,4:,:]+=0.2
        else: changed[i][...,4:]*=.3
    equal('causality',chunkwise(*args)[0][...,:4,:],chunkwise(*changed)[0][...,:4,:],0,0)
    # 任意分割位置，不限在块边界。
    first=[x[...,:4,:] if i<3 else x[...,:4] if i<5 else x for i,x in enumerate(args)]
    second=[x[...,4:,:] if i<3 else x[...,4:] if i<5 else x for i,x in enumerate(args)]
    o1,s1,_=chunkwise(*first,chunk=3);second[-1]=s1
    o2,s2,_=chunkwise(*second,chunk=2)
    equal('resume-output',torch.cat([o1,o2],-2),recurrent(*args)[0]);equal('resume-state',s2,recurrent(*args)[1])
    # 输出及末状态都进loss，覆盖六组变量，避免漏测state/门路径。
    grad_errors={}
    for path in (recurrent,chunkwise,affine):
        aa=[x.clone().requires_grad_(True) for x in args]
        o,s=path(*aa)[:2]
        grads=torch.autograd.grad(o.square().sum()+s.square().sum(),aa)
        if path is recurrent: baseline=grads
        else:
            for name,g,r in zip(('q','k','v','alpha','beta','S_in'),grads,baseline):
                equal(f'gradient/{path.__name__}/{name}',g,r)
    small=[x[0,0,:2].clone().requires_grad_(True) for x in args[:5]]+[args[-1][0,0].clone().requires_grad_(True)]
    assert torch.autograd.gradcheck(lambda *a:chunkwise(*a,chunk=2)[:2],tuple(small),eps=1e-6,atol=1e-5,rtol=1e-4)
    CHECKS.append({'name':'finite-difference-gradcheck','passed':True})
    # 两种门系数表达一致，验证因果及对角。
    a=args[3];d,g,e=coefficients(a);c=a.log().cumsum(-1)
    equal('log-vs-product',d,torch.tril((c[..., :,None]-c[...,None,:]).exp()))
    equal('D-diagonal',d.diagonal(dim1=-2,dim2=-1),torch.ones_like(a),0,0)
    # WY/门控WY：另走稠密乘积。
    q,k,v,a,b,s=args
    _,_,info=block(q,k,v,a,b,s)
    equal('L-diagonal-zero',info['L'].diagonal(dim1=-2,dim2=-1),torch.zeros_like(a),0,0)
    equal('A-diagonal-one',info['A'].diagonal(dim1=-2,dim2=-1),torch.ones_like(a),0,0)
    equal('P-future-zero',torch.triu(info['P'],diagonal=1),torch.zeros_like(info['P']),0,0)
    eye=torch.eye(3,dtype=k.dtype).expand(2,2,3,3)
    product=eye.clone()
    for i in range(k.shape[-2]):
        ki=k[...,i,:]; product=a[...,i,None,None]*(eye-b[...,i,None,None]*ki[..., :,None]*ki[...,None,:])@product
    representation=info['gamma'][...,-1,None,None]*eye-k.transpose(-1,-2)@(info['eta'][..., :,None]*info['W'])
    equal('gated-WY',product,representation)
    # 核注意力显式因果加权 vs 双累加器；使用正特征，保留分母。
    q,k,v,a,b,s=data(7);q,k=F.elu(q)+1,F.elu(k)+1
    weights=torch.tril(q@k.transpose(-1,-2));dense=weights@v/weights.sum(-1,keepdim=True)
    state=torch.zeros_like(s);z=torch.zeros_like(k[...,0,:]);outs=[]
    for i in range(7):
        state=state+k[...,i,:,None]*v[...,i,None,:];z=z+k[...,i,:]
        outs.append((q[...,i,:,None]*state).sum(-2)/(q[...,i,:]*z).sum(-1,keepdim=True))
    equal('normalized-linear-causal',dense,torch.stack(outs,-2))
    # 主例使用有理数独立计算，不从torch输出生成期望值。
    tq,tk,tv,ta,tb,ts=toy();ro,rs,rt=recurrent(*toy());co,cs,ct=block(*toy())
    S=sp.Matrix([[2,0],[0,0]]); exact=[]
    rational=lambda x:sp.Rational(str(x))
    for i in range(3):
        ki=sp.Matrix([rational(x) for x in tk[i].tolist()]);vi=sp.Matrix([rational(x) for x in tv[i].tolist()])
        faded=rational(ta[i].item())*S;r=vi-faded.T*ki;S=faded+ki*r.T
        exact.append({'S':str(S),'r':str(r),'o':str(S.T*ki)})
        equal(f'exact-state-{i}',rt[i]['state'],torch.tensor(S.tolist(),dtype=torch.float64))
    equal('toy-output',co,ro);equal('toy-state',cs,rs)
    # 直接从正文抽取两个代码片段执行，避免“配套程序正确、书里代码失真”。
    chapter=BOOK/'ch05-线性注意力与GDN.md'
    snippets=re.findall(r'```python\n(.*?)```',chapter.read_text(),re.S)
    assert len(snippets)==2
    q,k,v,a,b,s=toy()
    scope=dict(q=q,k=k,v=v,alpha=a,beta=b,S=s.clone(),N=3,o=torch.zeros_like(v))
    exec(compile(snippets[0],str(chapter)+'#snippet1','exec'),scope)
    equal('prose-code-recurrent-output',scope['o'],ro);equal('prose-code-recurrent-state',scope['S'],rs)
    d,g,e=coefficients(a)
    scope=dict(Q=q,K=k,V=v,beta=b,S_in=s,C=3,D=d,gamma=g,eta=e,tril=torch.tril,
               eye=lambda n:torch.eye(n,dtype=torch.float64),solve_triangular=torch.linalg.solve_triangular)
    exec(compile(snippets[1],str(chapter)+'#snippet2','exec'),scope)
    equal('prose-code-chunk-output',scope['O'],co);equal('prose-code-chunk-state',scope['S_out'],cs)
    # 抽取已固定的三个参考函数，保留函数原文；不导入整个模型也不加载权重。
    source=Path(importlib.util.find_spec("transformers").origin).parent/"models/qwen3_next/modeling_qwen3_next.py"
    tree=ast.parse(source.read_text())
    names={'l2norm','torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule'}
    body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    assert len(body)==3
    scope={'torch':torch,'F':F};exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec'),scope)
    a=data(7);q,k,v,al,be,s=[x.float() for x in a]
    # 参考接口(B,N,H,d)，q内部有1/sqrt(d_k)；本书核心q已经是有效查询。
    kwargs=dict(query=q.transpose(1,2),key=k.transpose(1,2),value=v.transpose(1,2),g=al.log().transpose(1,2),beta=be.transpose(1,2),initial_state=s,output_final_state=True,use_qk_l2norm_in_kernel=False)
    for name in ('torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule'):
        kw=dict(kwargs)
        if 'chunk' in name: kw['chunk_size']=3
        oo,ss=scope[name](**kw)
        our=chunkwise(q/math.sqrt(3),k,v,al,be,s,chunk=3)
        equal('HF/'+name+'/output',our[0],oo.transpose(1,2),2e-5,2e-5)
        equal('HF/'+name+'/state',our[1],ss,2e-5,2e-5)
    (OUT/'case.json').write_text(json.dumps(serial({'inputs':dict(zip(('Q','K','V','alpha','beta','S_in'),toy())),'steps':rt,'block':ct,'exact_sympy':exact}),ensure_ascii=False,indent=2)+'\n')
    result={'kind':'CPU mathematical verification; no model training or device benchmark','torch':torch.__version__,'transformers':__import__('transformers').__version__,'dtype':'float64 (HF parity float32)','checks':CHECKS,'prose_snippet_hashes':[hashlib.sha256(s.encode()).hexdigest() for s in snippets],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'script_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('gdn_math.py'))}}
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'checks_passed':len(CHECKS),'max_float64_error':max(x.get('max_abs_error',0) for x in CHECKS if not x['name'].startswith('HF/')),'toy':serial(ct)},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
