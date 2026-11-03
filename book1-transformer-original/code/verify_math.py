"""CPU float64：独立解析式、自动微分、有限差分、等价重排与信息不变性。"""
from pathlib import Path
import hashlib,json,math,re,contextlib,io
import torch
from model import Attention,Translator,positions
from paths import DRAFT, RESEARCH as OUT
torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(19)
checks=[]
def close(name,a,b,atol=1e-10,rtol=1e-9):
 a=torch.as_tensor(a);b=torch.as_tensor(b)
 err=(a-b).abs().max().item();assert torch.allclose(a,b,atol=atol,rtol=rtol),(name,err)
 checks.append(dict(name=name,max_abs_error=err))
def finite_jac(fn,x,eps=1e-6):
 cols=[]
 for i in range(x.numel()):
  e=torch.zeros_like(x);e[i]=eps;cols.append((fn(x+e)-fn(x-e))/(2*eps))
 return torch.stack(cols,-1)
# 循环状态雅可比连乘与实际复合函数。
w=torch.randn(3,3)*.2;inputs=torch.randn(4,3);h0=torch.randn(3,requires_grad=True)
def recurrence(h):
 for e in inputs:h=torch.tanh(w@h+e)
 return h
h=h0;product=torch.eye(3)
for e in inputs:
 h=torch.tanh(w@h+e);product=torch.diag(1-h*h)@w@product
close('2.RNN复合雅可比',product,torch.autograd.functional.jacobian(recurrence,h0))
close('2.RNN有限差分',product,finite_jac(recurrence,h0),atol=1e-8)
f=torch.tensor([.8,.4,.9]);writes=torch.tensor([.3,-.2,.7]);c=torch.tensor(2.)
for i in range(3):c=f[i]*c+writes[i]
expanded=f.prod()*2+sum(f[i+1:].prod()*writes[i] for i in range(3))
close('2.门控展开',c,expanded)
# softmax雅可比、读取梯度、错误分类梯度。
s=torch.randn(4,requires_grad=True);v=torch.randn(4,3);p=s.softmax(0)
j=torch.diag(p)-p[:,None]*p[None,:]
close('6.softmax解析雅可比',j,torch.autograd.functional.jacobian(lambda z:z.softmax(0),s))
close('6.softmax有限差分',j,finite_jac(lambda z:z.softmax(0),s),atol=1e-8)
close('6.平移零方向',j@torch.ones(4),torch.zeros(4))
g=torch.randn(3);o=p@v;analytic=p*((v-o)@g)
close('3/6.读出分数梯度',analytic,torch.autograd.grad(o@g,s)[0])
logits=torch.tensor([math.log(999),0.],requires_grad=True)
l=torch.nn.functional.cross_entropy(logits[None,:],torch.tensor([1]))
close('6.自信错误CE梯度',torch.autograd.grad(l,logits)[0],torch.tensor([.999,-.999]))
# 方差检查报告观察，不用紧容差伪装随机统计恒等式。
variance=[]
for d in [8,64,256]:
 q=torch.randn(10000,d);k=torch.randn_like(q);a=(q*k).sum(-1)
 variance.append(dict(d=d,raw_variance=a.var().item(),scaled_variance=(a/math.sqrt(d)).var().item()))
# 多头拆装对比每头独立公式、官方MHA。
a=Attention(8,2).eval();x=torch.randn(2,3,8);mem=torch.randn(2,4,8);allowed=torch.ones(2,1,3,4,dtype=torch.bool)
y=a(x,mem,allowed);heads=[]
for head in range(2):
 sl=slice(head*4,(head+1)*4)
 q=a.q(x)[...,sl];k=a.k(mem)[...,sl];vv=a.v(mem)[...,sl]
 heads.append((q@k.transpose(-1,-2)/2).softmax(-1)@vv)
close('7.独立头与打包输出',y,a.out(torch.cat(heads,-1)))
ref=torch.nn.MultiheadAttention(8,2,batch_first=True,dropout=0).eval()
with torch.no_grad():
 ref.in_proj_weight.copy_(torch.cat([a.q.weight,a.k.weight,a.v.weight]));ref.in_proj_bias.copy_(torch.cat([a.q.bias,a.k.bias,a.v.bias]))
 ref.out_proj.load_state_dict(a.out.state_dict())
close('7.官方MHA对拍',y,ref(x,mem,mem,need_weights=False)[0])
# 正弦位移、等变与位置破坏。
pe=positions(19,8,torch.float64,'cpu');delta=3;omega=10000**(-torch.arange(0,8,2)/8)
for i in range(4):
 angle=delta*omega[i];rot=torch.tensor([[angle.cos(),angle.sin()],[-angle.sin(),angle.cos()]])
 close(f'9.位置平移频率{i}',pe[3:19,2*i:2*i+2],pe[:16,2*i:2*i+2]@rot.T)
close('9.位置内积',pe[2]@pe[7],torch.cos((2-7)*omega).sum())
perm=torch.tensor([2,0,1]);all3=torch.ones(1,1,3,3,dtype=torch.bool)
close('9.无位置自注意等变',a(x[:,perm],x[:,perm],all3),a(x,x,all3)[:,perm])
# FFN算例、逐位置等价。
w1=torch.tensor([[1.,-1.,0.],[0.,0.,1.]]);w2=torch.tensor([[1.,0.],[-1.,0.],[0.,1.]])
close('10.FFN算例',(torch.tensor([[2.,-1.],[-2.,1.]])@w1).relu()@w2,torch.tensor([[2.,0.],[-2.,1.]]))
ff=torch.nn.Sequential(torch.nn.Linear(8,16),torch.nn.ReLU(),torch.nn.Linear(16,8));xx=torch.randn(2,3,8)
close('10.逐位置与批量',ff(xx),torch.stack([ff(row) for row in xx.reshape(-1,8)]).reshape(2,3,8))
# LayerNorm包含epsilon的解析导数。
for eps in [1e-5,.1]:
 z=torch.randn(5,requires_grad=True);gamma=torch.randn(5);beta=torch.randn(5)
 fn=lambda t:torch.nn.functional.layer_norm(t,(5,),gamma,beta,eps)
 centered=z-z.mean();scale=(centered.square().mean()+eps).sqrt();C=torch.eye(5)-torch.ones(5,5)/5
 norm_j=C/scale-centered[:,None]*centered[None,:]/(5*scale**3)
 analytic=gamma[:,None]*norm_j
 close(f'11.LN自动微分eps{eps}',analytic,torch.autograd.functional.jacobian(fn,z))
 close(f'11.LN有限差分eps{eps}',analytic,finite_jac(fn,z),atol=1e-8)
 close(f'11.LN平移eps{eps}',fn(z+3),fn(z))
 close(f'11.LN径向eps{eps}',norm_j@centered,eps*centered/scale**3)
# 多层整机因果、源填充与梯度。
net=Translator(d=8,h=2,layers=2,ff=16,drop=0).eval();src=torch.tensor([[3,7,5]]);tgt=torch.tensor([[1,3,7,5]])
a0=net(src,tgt);changed=tgt.clone();changed[:,2:]=torch.tensor([4,6])
close('8.两层整机未来不影响过去',a0[:,:2],net(src,changed)[:,:2])
close('8.追加源padding不影响有效输出',a0,net(torch.cat([src,torch.zeros(1,2,dtype=torch.long)],1),tgt))
# 跨位置输入梯度在过去输出上为零，用解码嵌入hook保留梯度。
held=[]
def hook(module,args,out):out.retain_grad();held.append(out)
handle=net.target.register_forward_hook(hook);net(src,tgt)[:,0].sum().backward();handle.remove()
close('8.未来输入嵌入梯度为零',held[0].grad[:,1:],torch.zeros_like(held[0].grad[:,1:]))
# 平滑、优化器偏差修正、学习率交点。
logits=torch.randn(4,requires_grad=True);q=torch.tensor([.025,.925,.025,.025]);loss=-(q*logits.log_softmax(0)).sum()
close('12.平滑梯度',torch.autograd.grad(loss,logits)[0],logits.softmax(0)-q)
for b in [.9,.98]:
 value=0.
 for t in range(1,8):
  value=b*value+(1-b)*2
  close(f'13.恒定梯度偏差修正{b}-{t}',value/(1-b**t),2.)
lr=lambda t:512**-.5*min(t**-.5,t*4000**-1.5)
close('13.调度峰值',lr(4000),1/math.sqrt(512*4000));close('13.四倍步骤半峰值',lr(16000),lr(4000)/2)
close('14.概率树归一',.3+.3+.396+.004,1.)
# 直接执行正文短代码，保留实际输出。
outputs={}
for chapter in sorted(DRAFT.glob('ch*.md')):
 for i,snippet in enumerate(re.findall(r'```python\n(.*?)```',chapter.read_text(),re.S)):
  stream=io.StringIO()
  with contextlib.redirect_stdout(stream):exec(compile(snippet,str(chapter),'exec'),{})
  outputs[f'{chapter.name}:{i}']=dict(sha256=hashlib.sha256(snippet.encode()).hexdigest(),stdout=stream.getvalue())
report=dict(device='CPU',dtype='float64',torch=torch.__version__,checks=checks,count=len(checks),variance_observations=variance,prose_snippets=outputs,
 code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('model.py')]})
(OUT/'math-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('PASS',len(checks),'max_abs_error',max(c['max_abs_error'] for c in checks));print(json.dumps(variance))
