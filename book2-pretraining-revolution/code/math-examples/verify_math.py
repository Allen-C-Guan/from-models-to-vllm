"""两篇润色样稿的CPU双精度核验。历史表格重算不等于训练复现。"""
from pathlib import Path
import argparse, json, math, re
import numpy as np
import torch
from scipy.optimize import least_squares, minimize_scalar, brentq
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'log/book2-polish-verification'
OUT.mkdir(parents=True,exist_ok=True)
N = np.array([991616,3114432,7108352,14796160,19519872],dtype=float)
LOSSES = np.array([5.101,4.862,4.710,4.571,4.501])

def close(a,b,atol=1e-10):
    np.testing.assert_allclose(np.asarray(a),np.asarray(b),atol=atol,rtol=atol)

def chapter4():
    torch.set_default_dtype(torch.float64)
    x=torch.tensor([1.,2.,3.]); one=torch.ones(3); radial=x-x.mean(); tangent=torch.tensor([1.,-2.,1.])
    cases=[]
    for eps in [0.,1e-5,.1]:
        def ln(x): return (x-x.mean())/torch.sqrt(((x-x.mean())**2).mean()+eps)
        v=x-x.mean(); s=torch.sqrt((v*v).mean()+eps)
        analytic=(torch.eye(3)-torch.ones(3,3)/3-torch.outer(v,v)/(3*s*s))/s
        actual=torch.autograd.functional.jacobian(ln,x)
        close(analytic,actual); close(actual@one,torch.zeros(3))
        close(actual@radial,eps*radial/s**3);close(actual@tangent,tangent/s)
        pre=torch.autograd.functional.jacobian(lambda z:z+.1*ln(z),x)
        post=torch.autograd.functional.jacobian(lambda z:ln(1.1*z),x)
        close(pre,torch.eye(3)+.1*actual)
        close(post,1.1*torch.autograd.functional.jacobian(ln,1.1*x))
        cases.append({'epsilon':eps,'J':actual.tolist(),'pre_mean':(pre@one).tolist(),'post_mean':(post@one).tolist()})
    # 对任意非均匀gamma再检验，方向解释不默认为全一gamma。
    gamma=torch.tensor([.7,1.3,2.])
    actual=torch.autograd.functional.jacobian(lambda z:gamma*(z-z.mean())/torch.sqrt(((z-z.mean())**2).mean()+.1),x)
    close(actual,torch.diag(gamma)@analytic)
    # 同一标量loss中，绑权梯度应等于独立输入/输出表梯度之和。
    E=torch.tensor([[1.,0.],[0.,1.],[1.,1.]],requires_grad=True)
    ids=torch.tensor([0,1,0]); labels=torch.tensor([1,2,2]); W=torch.tensor([[.2,.4],[.1,.8]])
    def calc(Ein,Eout):
        hidden=Ein[ids]@W
        logits=hidden@Eout.T
        return torch.nn.functional.cross_entropy(logits,labels)
    tied=torch.autograd.grad(calc(E,E),E)[0]
    ein=E.detach().clone().requires_grad_();eout=E.detach().clone().requires_grad_()
    gi,go=torch.autograd.grad(calc(ein,eout),(ein,eout));close(tied,gi+go)
    logits=torch.tensor([2.,1.])@E.detach().T
    close(logits,[2.,1.,3.])
    # 位置查表的反向是按索引散加；重复位置的梯度相加。
    P=torch.zeros(4,2,requires_grad=True); pos=torch.tensor([0,2,2]);w=torch.tensor([[1.,2.],[3.,4.],[5.,6.]])
    pg=torch.autograd.grad((P[pos]*w).sum(),P)[0];close(pg,[[1,2],[0,0],[8,10],[0,0]])
    d=768;V=50257;T=1024;L=12
    count={'wte':V*d,'wpe':T*d,'block':12*d*d+13*d,'ln_f':2*d}
    count['total']=count['wte']+count['wpe']+L*count['block']+count['ln_f']
    count['untied']=count['total']+V*d
    assert count['total']==124439808 and count['untied']==163037184
    # 独立通过meta模块形状计数，避免只重复同一个总公式。
    with torch.device('meta'):
        modules=torch.nn.ModuleList([torch.nn.Embedding(V,d),torch.nn.Embedding(T,d),torch.nn.LayerNorm(d)])
        for _ in range(L):
            modules.append(torch.nn.ModuleList([torch.nn.Linear(d,3*d),torch.nn.Linear(d,d),torch.nn.Linear(d,4*d),torch.nn.Linear(4*d,d),torch.nn.LayerNorm(d),torch.nn.LayerNorm(d)]))
    assert sum(p.numel() for p in modules.parameters())==count['total']
    cross={str(d):4*d*d+6*d for d in [384,768]};assert cross['768']==2363904
    # 因果attention的前缀不受未来输入修改影响。
    torch.manual_seed(41);q=torch.randn(5,3); k=torch.randn(5,3);v=torch.randn(5,2)
    mask=torch.triu(torch.ones(5,5,dtype=torch.bool),diagonal=1)
    att=lambda k,v:torch.softmax((q@k.T/math.sqrt(3)).masked_fill(mask,-torch.inf),-1)@v
    original=att(k,v);k2=k.clone();v2=v.clone();k2[3:]+=100;v2[3:]-=100
    close(original[:3],att(k2,v2)[:3])
    return {'kind':'CPU float64数学/构造核验，无模型训练','LN':cases,'tied_gradient_max_error':float((tied-gi-go).abs().max()),'toy_logits':logits.tolist(),'toy_probability':torch.softmax(logits,0).tolist(),'position_gradient':pg.tolist(),'parameters':count,'cross_sub_layer':cross,'causal_prefix_unchanged':True}

def fit_points(E=0):
    slope,intercept=np.polyfit(np.log(N),np.log(LOSSES-E),1)
    return {'E':E,'A':float(np.exp(intercept)),'alpha':float(-slope)}

def chapter8():
    # 临界批次经验双曲线及两个极限。
    bcrit=2.1e8/4.6**(1/.21);smin=100.;tmin=bcrit*smin
    batch=[]
    for b in [bcrit*.001,8192,bcrit,bcrit*1000]:
        s=smin*(1+bcrit/b);t=tmin*(1+b/bcrit)
        close((s/smin-1)*(t/tmin-1),1,1e-9);close(b*s,t,1e-9)
        batch.append({'B':b,'S_over_Smin':s/smin,'T_over_Tmin':t/tmin})
    # Kaplan隐式关系：归一化常数A=K=1，先逐点解L，再数值优化N。
    a,s,b=.077,.76,.21
    eta=1/(1/a+1/s+1/b);p=eta/a
    def implicit_l(logn,compute):
        n=np.exp(logn)
        fn=lambda l:l-n**(-a)-n**s*compute**(-s)*l**(-s/b)
        return brentq(fn,n**(-a),max(10,n**(-a)*100))
    opt=[]
    for c in [1e2,1e3]:
        res=minimize_scalar(lambda z:implicit_l(z,c),bounds=(-10,30),method='bounded',options={'xatol':1e-10})
        n=np.exp(res.x);l=res.fun
        close(a*n**(-a),s*n**s*c**(-s)*l**(-s/b),1e-7)
        opt.append({'C':c,'N':n,'L':l})
    close(opt[1]['N']/opt[0]['N'],10**p,1e-5)
    # Chinchilla解析最优点与数值搜索及梯度三方核验。
    chin=[]
    for A,B,alpha,beta,K in [(1,1,.5,.5,100),(406.4,410.7,.34,.28,1e20/6)]:
        G=(alpha*A/(beta*B))**(1/(alpha+beta));n=G*K**(beta/(alpha+beta));d=K/n
        def f(z):return A*np.exp(-alpha*z)+B*K**(-beta)*np.exp(beta*z)
        r=minimize_scalar(f,bounds=(np.log(n)-10,np.log(n)+10),method='bounded')
        close(np.exp(r.x)/n,1,1e-6);close(alpha*A*n**(-alpha),beta*B*d**(-beta))
        z=torch.tensor(np.log(n),dtype=torch.float64,requires_grad=True)
        val=A*torch.exp(-alpha*z)+B*K**(-beta)*torch.exp(beta*z)
        grad=torch.autograd.grad(val,z,create_graph=True)[0];hess=torch.autograd.grad(grad,z)[0]
        assert abs(float(grad.detach()))<1e-10 and float(hess)>0
        chin.append({'N':n,'D':d,'loss_above_E':f(np.log(n)),'log_hessian':float(hess),'N_exponent':beta/(alpha+beta)})
    close(chin[0]['N'],10);close(chin[0]['loss_above_E'],2/np.sqrt(10))
    for (layers,d),actual in zip([(5,128),(7,192),(9,256),(12,320),(11,384)],N):assert layers*(12*d*d+13*d)+2*d==actual
    T=4882*8192;costs=6*N*T
    fit=fit_points();pred=fit['A']*N**(-fit['alpha']);r2=1-((LOSSES-pred)**2).sum()/((LOSSES-LOSSES.mean())**2).sum()
    close(fit['alpha'],-np.cov(np.log(N),np.log(LOSSES),ddof=0)[0,1]/np.var(np.log(N)))
    # 在同一原loss残差目标上比较：参数用log幅度提高优化数值尺度。
    fixed=least_squares(lambda p:np.exp(p[0])*N**(-p[1])-LOSSES,[np.log(9),.04],bounds=([-20,.001],[20,5]),xtol=1e-13,ftol=1e-13,gtol=1e-13)
    free=least_squares(lambda p:p[0]+np.exp(p[1])*N**(-p[2])-LOSSES,[.1,np.log(9),.04],bounds=([0,-20,.001],[4.4,20,5]),max_nfev=10000,xtol=1e-13,ftol=1e-13,gtol=1e-13)
    assert fixed.success and free.success
    # 正相关反例：均值方差含协方差；独立公式会低估。
    cov=np.full((10,10),.1);np.fill_diagonal(cov,1);assert cov.sum()/100>1/10
    return {'kind':'CPU数学与历史圆整终点重算；无长训练/历史bootstrap复现','critical_batch':bcrit,'batch_cases':batch,'kaplan_rounded_constants':{'eta':eta,'N_exponent':p,'numerical_optima':opt},'chinchilla':chin,'processed_tokens':T,'costs_FLOPs':costs.tolist(),'family_PF_days':float(costs.sum()/8.64e19),'iid_se_only':2.5/math.sqrt(401408),'rounded_point_fit':{**fit,'prediction':pred.tolist(),'residual':(LOSSES-pred).tolist(),'r2':r2},'same_raw_loss_objective':{'fixed_E0':{'A':float(np.exp(fixed.x[0])),'alpha':float(fixed.x[1]),'sse':float((fixed.fun**2).sum())},'free_E':{'E':float(free.x[0]),'A':float(np.exp(free.x[1])),'alpha':float(free.x[2]),'sse':float((free.fun**2).sum())}},'fixed_floor_teaching_fits':[fit_points(e) for e in [0,2,3]],'local_slopes':(np.diff(np.log(LOSSES))/np.diff(np.log(N))).tolist(),'kaplan_threshold_plug_only':5e3*(20e6)**.74}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--chapter',choices=['4','8','all'],default='all');args=parser.parse_args()
    report={'device':'CPU','dtype':'float64','historical_training_rerun':False}
    for number,fn in [('4',chapter4),('8',chapter8)]:
        if args.chapter in ['all',number]:
            result=fn();report[number]=result
            (OUT/f'math-ch{number}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
