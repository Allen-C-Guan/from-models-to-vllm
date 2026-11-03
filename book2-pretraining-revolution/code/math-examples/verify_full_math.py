"""Book2 全册新增推导的 CPU float64 检查；不运行训练，不推断性能。"""
from pathlib import Path
import json, math, importlib.util
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'log/book2-polish-verification'
torch.set_default_dtype(torch.float64);torch.manual_seed(7)
results={}
def check(name,actual,expected,atol=1e-10):
 a=torch.as_tensor(actual,dtype=torch.float64);b=torch.as_tensor(expected,dtype=torch.float64)
 assert torch.allclose(a,b,atol=atol,rtol=1e-9),(name,a,b)
 results[name]={'actual':a.tolist(),'expected':b.tolist(),'atol':atol}
# 负采样：梯度来自二元交叉熵，不是全词表 softmax 近似梯度。
v=torch.tensor([1.,0.],requires_grad=True);u=torch.tensor([0.,1.]);loss=torch.nn.functional.softplus(-(v@u));g=torch.autograd.grad(loss,v)[0]
check('SGNS梯度',g,(torch.sigmoid(v@u)-1)*u)
check('SGNS一步更新',v-.1*g,[1,.05])
check('GloVe概率比',math.log(.8/.1),math.log(.8)-math.log(.1))
weights=torch.softmax(torch.tensor([0.,1.,2.]),0);states=torch.randn(3,1024);mix=(weights[:,None]*states).sum(0)
assert mix.shape==(1024,);check('ELMo混合权重和',weights.sum(),1)
T,cut=100,10
p=lambda t:t/cut if t<cut else (T-t)/(T-cut)
check('STLR端点',[p(0),p(cut),p(T)],[0,1,0])
# Unigram：路径求和不等于最优路径。
paths=[.1*.1*.1,.2*.2,.3*.1]
check('Unigram字符串边缘概率',sum(paths),.071);check('Unigram最优路径',max(paths),.04)
# BERT 只对选中位置计损失；未选位置可经注意力收到梯度。
z=torch.randn(2,5,requires_grad=True);target=torch.tensor([1,3]);ce=torch.nn.functional.cross_entropy(z,target)
g=torch.autograd.grad(ce,z)[0];e=torch.softmax(z,1);e[torch.arange(2),target]-=1
check('BERT-CE梯度',g,e/2)
check('两位置目标概率损失',-(math.log(.2)+math.log(.5))/2,1.1512925464970227)
x=torch.randn(3,4,requires_grad=True);att=torch.softmax(x@x.T,1);h=att@x;h[1].square().sum().backward();assert x.grad[0].norm()>0
results['未计损失位置经注意力收到梯度']={'norm':x.grad[0].norm().item()}
# T5：指定例子 8 个原始 token，遮 3 个，2 段，含末尾 sentinel 与 EOS。
check('T5输入输出长度',[8-3+2,3+2+1+1],[7,7])
# RoBERTa：动态掩码与实际日程累计位置。
check('动态掩码覆盖',[1-.85**r for r in [1,4,10]],[.15,.47799375,.8031255956592774])
bert=256*(900000*128+100000*512);rob=8192*512*500000
check('BERT实际token位置',bert,42598400000);check('RoBERTa位置倍率',rob/bert,49.23076923076923)
# ICL、EM和Brier：连续指标与有限样本观测的区别。
check('四位全对',.8**4,.4096);check('20题全零概率', (1-.3**4)**20,.85,atol=2e-4)
check('Brier二类',[2*(1-.49)**2,2*(1-.51)**2],[.5202,.4802])
# loss scaling 与 clipping 的操作次序。
g=torch.tensor([3.,4.]);S=8.;c=1.
correct=g*min(1,c/g.norm());wrong=(S*g)*min(1,c/(S*g).norm())/S
check('先缩放后裁剪会缩小阈值',wrong,correct/S)
w=torch.tensor(1.,dtype=torch.float16);master=torch.tensor(1.,dtype=torch.float32)
for _ in range(20):w=(w-torch.tensor(3e-5,dtype=torch.float16));master-=3e-5
check('低精度逐步更新丢失',float(w),1)
assert float(master.half())<1;results['fp32主权重保留小更新']={'master':float(master),'cast':float(master.half())}
ks=torch.arange(1,65,dtype=torch.float64);check('64层检查点简化模型最优段长',ks[torch.argmin(ks+64/ks)],8)
# GPT-2 参数账、共享梯度、logsumexp 二阶项。
check('GPT2独立参数数',50257*768+1024*768+12*(12*768**2+13*768)+2*768,124439808)
E=torch.randn(7,4,requires_grad=True);h=torch.randn(4);y=torch.tensor([2]);ell=torch.nn.functional.cross_entropy((E@h)[None,:],y);g=torch.autograd.grad(ell,E)[0]
prob=torch.softmax(E@h,0);check('未作为目标的输出行梯度',g[6],prob[6]*h);assert g[6].norm()>0
z=torch.tensor([-.2,.1,.3,-.1])*.01;y=2
actual=torch.logsumexp(z,0)-z[y];approx=math.log(4)+z.mean()-z[y]+.5*(z.square().mean()-z.mean().square())
check('logsumexp二阶近似',actual,approx,atol=1e-8)
for temp in [.5,1,2]:
 prob=torch.softmax(torch.tensor([2.,0.])/temp,0);check(f'温度概率比-{temp}',prob[0]/prob[1],math.exp(2/temp))
# 完整轻量 GPT 模型：实际项目源码；检查因果性与权重绑定，不下载。
spec=importlib.util.spec_from_file_location('book2_model',ROOT/'code/ch10/model.py');mod=importlib.util.module_from_spec(spec)
import sys;sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
model=mod.GPT2(mod.GPT2Config(vocab_size=17,n_positions=8,n_layer=2,n_head=2,n_embd=8,dropout=0)).double().eval()
ids=torch.tensor([[1,2,3,4]]);other=ids.clone();other[:,3]=9
with torch.no_grad():a=model(ids)[0];b=model(other)[0]
check('项目模型未来不影响过去',a[:,:3],b[:,:3]);assert model.head.weight is model.wte.weight
results['实际项目模型']={'shape':list(a.shape),'tied':True}
# 15点预算表：直接解析正文，防止验证另造一份数字。
import re
s=next((ROOT).glob('ch11-*.md')).read_text()
rows=[]
for line in s.splitlines():
 if re.match(r'\| c[123][a-z] \|',line):
  cols=[v.strip().replace(',','').replace('**','') for v in line.strip('|').split('|')]
  n,d,steps=int(cols[2]),int(cols[3]),int(cols[4]);c=[6e14,2e15,6e15][int(cols[0][1])-1]
  assert d==steps*8192;assert abs(6*n*d/c-1)<.0004
  rows.append({'point':cols[0],'N':n,'D':d,'loss':float(cols[5]),'C':c})
assert len(rows)==15;results['预算表15点']={'max_relative_compute_error':max(abs(6*r['N']*r['D']/r['C']-1) for r in rows)}
for group in [1,2,3]:
 rr=[r for r in rows if r['point'].startswith('c'+str(group))];xx=np.log([r['N'] for r in rr]);yy=np.array([r['loss'] for r in rr]);coef=np.polyfit(xx,yy,2)
 results[f'历史表格二次拟合C{group}']={'vertex':float(np.exp(-coef[1]/(2*coef[0]))),'interpretation':'插值或外推候选，不是已证最优'}
OUT.mkdir(parents=True,exist_ok=True);(OUT/'math-full.json').write_text(json.dumps({'device':'CPU','dtype':'float64（主权重示例另注明）','results':results,'grid':rows},ensure_ascii=False,indent=2)+'\n')
print('PASS',len(results),'组机制/算术检查；历史训练未重跑。')
if __name__=='__main__':pass
