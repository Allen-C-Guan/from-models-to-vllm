"""CPU数学练习：梯度、状态等价与账本；不测硬件性能，不替代长训练。"""
from pathlib import Path
import json,math,sys,importlib.util
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'log/math-examples/results.json';torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(20261103);rows=[]
def eq(name,a,b,tol=1e-10):
 a=torch.as_tensor(a);b=torch.as_tensor(b);err=(a-b).abs().max().item();assert err<=tol,(name,err);rows.append({'name':name,'max_error':err})
z=torch.tensor([2.,.5,1.2,-.3],requires_grad=True);idx=z.topk(2).indices;g=z[idx].softmax(0);e=torch.tensor([2.,-1.]);y=(g*e).sum();grad=torch.autograd.grad(y,z)[0];expected=torch.zeros(4);expected[idx]=g*(e-y);eq('selected-softmax-gradient',grad,expected)
z=torch.tensor([math.log(3),0.],requires_grad=True);p=z.softmax(0);eq('switch-top1-output',2*p[0],1.5);eq('switch-top1-gradient',torch.autograd.grad(2*p[0],z)[0],[.375,-.375])
z=torch.tensor([math.log(3),0.],requires_grad=True);y=2*z[:1].softmax(0).sum();eq('renormalized-top1-zero-gradient',torch.autograd.grad(y,z)[0],[0.,0.])
a=torch.tensor([3.,1.]);eq('cv-squared',a.var(unbiased=False)/a.mean()**2,.25)
p=torch.tensor([[.51,.49]]*3+[[.01,.99]]);f=torch.tensor([.75,.25]);eq('fp-surrogate-not-uniform-minimum',2*(f*p.mean(0)).sum(),.885)
z=torch.randn(5,4,requires_grad=True);f=torch.tensor([.4,.3,.2,.1]);p=z.softmax(-1);loss=.01*4*(f*p.mean(0)).sum();ex=.01*4/5*p*(f-(p*f).sum(-1,keepdim=True));eq('balance-gradient',torch.autograd.grad(loss,z)[0],ex)
z=torch.randn(4,requires_grad=True);lse=z.logsumexp(0);eq('z-loss-gradient',torch.autograd.grad(lse**2,z)[0],2*lse*z.softmax(0));eq('lse-shift',torch.logsumexp(z+3,0),lse+3)
scores=torch.tensor([.6,.5,.2,.1]);bias=torch.tensor([-.5,0.,.4,0.]);idx=(scores+bias).topk(2).indices;eq('noaux-choice',idx,[2,1]);eq('noaux-original-scores',scores[idx]/scores[idx].sum(),[2/7,5/7])
for B,H,N,R,D,V in [(1,2,1,2,2,3),(2,3,5,4,3,2),(1,4,7,3,2,4)]:
 c=torch.randn(B,N,R,requires_grad=True);q=torch.randn(B,H,N,D,requires_grad=True);uk=torch.randn(H,D,R,requires_grad=True);uv=torch.randn(H,V,R,requires_grad=True);qr=torch.randn(B,H,N,2,requires_grad=True);kr=torch.randn(B,N,2,requires_grad=True);wo=torch.randn(H,V,6,requires_grad=True)
 mask=torch.ones(N,N,dtype=torch.bool).triu(1);k=torch.einsum('bnr,hdr->bhnd',c,uk);v=torch.einsum('bnr,hvr->bhnv',c,uv);s=(q@k.transpose(-1,-2)+qr@kr[:,None].transpose(-1,-2))/math.sqrt(D+2);alpha=s.masked_fill(mask,-torch.inf).softmax(-1);o=torch.einsum('bhnv,hvo->bno',alpha@v,wo)
 qhat=torch.einsum('bhnd,hdr->bhnr',q,uk);s2=(qhat@c[:,None].transpose(-1,-2)+qr@kr[:,None].transpose(-1,-2))/math.sqrt(D+2);a2=s2.masked_fill(mask,-torch.inf).softmax(-1);mixed=a2@c[:,None];v2=torch.einsum('bhnr,hvr->bhnv',mixed,uv);o2=torch.einsum('bhnv,hvo->bno',v2,wo)
 eq(f'mla-scores-{N}',s,s2);eq(f'mla-output-{N}',o,o2);params=[c,q,uk,uv,qr,kr,wo];g1=torch.autograd.grad(o.square().sum(),params,retain_graph=True);g2=torch.autograd.grad(o2.square().sum(),params)
 for j,(x,y) in enumerate(zip(g1,g2)):eq(f'mla-gradient-{N}-{j}',x,y,2e-9)
# Joint rank, rotation and hand example.
eq('joint-rank-counterexample',torch.linalg.matrix_rank(torch.eye(2)),2)
U=torch.tensor([[1.],[0.]]);Rot=torch.tensor([[0.,-1.],[1.,0.]]);eq('rotation-preserves-rank',torch.linalg.matrix_rank(Rot@U),torch.linalg.matrix_rank(U));eq('rotation-leaves-subspace',Rot@U,[[0.],[1.]])
C=torch.tensor([[1.,2.],[3.,-1.]]);uk=torch.diag(torch.tensor([1.,2.]));q=torch.tensor([2.,1.]);eq('mla-hand-scores',C@uk.T@q,[6.,4.]);eq('mla-hand-value',(torch.tensor([.25,.75])@C)@torch.tensor([2.,-1.]),5.25)
eq('gptoss-window-128',2*8*64*(12*128+12*128),24576*128)
eq('gptoss-20b-total',2*201088*2880+24*(2*2880*64*(64+8)+(64+16)*64+2880+64+2880*32+32+32*(3*2880**2+3*2880)+5760)+2880,20914757184)
AG=3145728;AM=2686976;d=1024;w=(AG-AM-512+3*d*2816-8*d)/(3*d*9);eq('equal-total-width',w,329+1/6);eq('rounding-gap',3*d*9*(w-329)*12,55296,1e-6)
eq('dense-active-excluding-input',207119360-32000*1024,174351360)
eq('fp8-normal-range',448/2**-6,28672);eq('fp8-subnormal',2**-3*2**-6,2**-9);eq('mxfp4-bits',4+8/32,4.25)
q=torch.tensor([3.,4.]);k=torch.tensor([4.,3.]);eq('qk-cosine',(q/q.norm())@(k/k.norm()),.96);eq('rms-versus-l2',(q/q.square().mean().sqrt())@(k/k.square().mean().sqrt()),1.92)
x=torch.diag(torch.tensor([4.,1.]));x=x/x.norm()
for _ in range(12):x=.5*x@(3*torch.eye(2)-x.T@x)
eq('newton-schulz-prototype',x,torch.eye(2),1e-9)
qc=torch.randn(3);kc=torch.randn(3);qr=torch.randn(2);kr=torch.randn(2);g=.5;eq('mla-qk-clip',((qc*g**.5)@(kc*g**.5)+(qr*g)@kr),g*(qc@kc+qr@kr))
eq('mtp-independent-joint-total',sum([.25]*4),1.)
t=torch.tensor([[1,2,3,4,5]]);eq('mtp-input',t[:,:-2],[[1,2,3]]);eq('mtp-next',t[:,1:-1],[[2,3,4]]);eq('mtp-target',t[:,2:],[[3,4,5]])
P=torch.tensor([[1.,0.]]);U=torch.tensor([[1.],[1.]]);x=torch.tensor([2.,3.]);v=P@x;y=U@(.25*v**2+.75*v.relu());eq('latentmoe-hand',y,[2.5,2.5]);eq('latentmoe-null-direction',P@x,P@torch.tensor([2.,9.]))
eq('latent-budget-experts',512*1024,128*4096);eq('latent-communication-ratio',22*1024/(6*4096),22/24)
OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps({'device':'CPU','dtype':'float64','checks':rows},ensure_ascii=False,indent=2)+'\n');print('PASS',len(rows))
