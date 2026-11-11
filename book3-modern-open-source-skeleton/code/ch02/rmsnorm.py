# Book3第2章：LLaMA式无偏置RMSNorm、性质探针与同条件对拍。
# 在已有环境中调用；只运行CPU局部检查，不训练、不下载权重。
# 默认结果写当前工作目录log；公式不变性与有限精度近似分开报告。
import argparse
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn as nn

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'00-feasibility'))
from llama_slots import RMSNorm as SlotRMSNorm

SEED=20261002


class RMSNorm(nn.Module):
    """输入(B,n,C)，统计量(B,n,1)，输出同形状；增益(C,)，无偏置。

    本复刻路径总在fp32算均方（float64输入也会降为fp32），先将归一化结果
    转回输入dtype再乘增益。最终dtype还受增益dtype及类型提升影响。
    eps=0时仅非零输入有定义；默认正eps在零点附近也改变函数及导数。
    """
    def __init__(self,hidden_size,eps=1e-6):
        super().__init__()
        if not isinstance(hidden_size,int) or hidden_size<=0:raise ValueError('hidden_size须为正整数')
        if not math.isfinite(eps) or eps<0:raise ValueError('eps须为有限非负数')
        self.weight=nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon=eps

    def forward(self,x):
        if not x.is_floating_point() or x.ndim==0 or x.shape[-1]!=self.weight.numel():
            raise ValueError('输入须为末维匹配的浮点张量')
        dtype=x.dtype
        x32=x.float()                                    # (...,C) -> (...,C)，fp32统计
        ms=x32.pow(2).mean(-1,keepdim=True)              # (...,C) -> (...,1)
        x32=x32*torch.rsqrt(ms+self.variance_epsilon)    # 广播缩放，仍为(...,C)
        return self.weight*x32.to(dtype)               # (C,)广播；最终dtype由两乘数决定


@torch.no_grad()
def probe_properties():
    """同一构造输入验证代数边界，不把有epsilon的近似叫精确尺度不变。"""
    torch.manual_seed(SEED)
    x=torch.randn(2,5,8);ln=nn.LayerNorm(8,eps=1e-6);rms=RMSNorm(8,eps=1e-6)
    d_ln=float((ln(x+3)-ln(x)).abs().max());d_rms=float((rms(x+3)-rms(x)).abs().max())
    assert d_ln<2e-6 and d_rms>.1
    scaling={name:float((module(2.5*x)-module(x)).abs().max()) for name,module in [('LN',ln),('RMS',rms)]}
    x0=x-x.mean(-1,keepdim=True);equal_error=float((ln(x0)-rms(x0)).abs().max());assert equal_error<1e-6
    ms=x.square().mean(-1);mu=x.mean(-1);var=x.var(-1,unbiased=False)
    moment_error=float((ms-var-mu.square()).abs().max());assert moment_error<1e-6
    a=torch.tensor([1.,2.,3.,6.]);mean=float(a.mean());variance=float(a.var(unbiased=False));rms2=float(a.square().mean())
    assert (mean,variance,rms2)==(3.,3.5,12.5)
    # 纯RMS正齐次而非线性；负缩放翻方向，正eps破坏精确尺度不变。
    pure=RMSNorm(8,eps=0);assert torch.allclose(pure(2.5*x),pure(x),atol=1e-6)
    assert torch.allclose(pure(-x),-pure(x),atol=1e-6)
    tiny=torch.full((1,8),1e-5);eps_norm=RMSNorm(8,eps=1e-5)
    eps_counter=float((eps_norm(10*tiny)-eps_norm(tiny)).abs().max());assert eps_counter>.02
    return dict(translation={'ln_max_error':d_ln,'rms_changed_by':d_rms},positive_scaling_with_eps=scaling,
                zero_mean_max_error=equal_error,moment_identity_error=moment_error,
                four_dim_example={'mean':mean,'variance':variance,'rms_squared':rms2,
                    'LN':torch.nn.functional.layer_norm(a,(4,),eps=1e-6).tolist(),
                    'RMS':(a/torch.sqrt(a.square().mean()+1e-6)).tolist()},
                positive_eps_scale_counterexample=eps_counter,
                interpretation='代数性质以正文条件为准；误差是此输入的浮点读数')


@torch.no_grad()
def probe_parity():
    """同权重/同输入/同eps的有限对拍；assert容差，另外如实记录torch.equal。"""
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    torch.manual_seed(SEED);x=torch.randn(4,64,512);results=[]
    for eps in (1e-6,1e-5):
        ours=RMSNorm(512,eps);slot=SlotRMSNorm(512,eps);hf=LlamaRMSNorm(512,eps)
        weight=torch.randn(512)
        with torch.no_grad():
            for m in (ours,slot,hf):m.weight.copy_(weight)
            yo,ys,yh=ours(x),slot(x),hf(x)
        assert torch.allclose(yo,ys,atol=1e-6,rtol=1e-6) and torch.allclose(yo,yh,atol=1e-6,rtol=1e-6)
        results.append(dict(eps=eps,slot_max=float((yo-ys).abs().max()),hf_max=float((yo-yh).abs().max()),
                            slot_bitwise=torch.equal(yo,ys),hf_bitwise=torch.equal(yo,yh)))
    x16=torch.randn(64,256,dtype=torch.bfloat16)
    m16=x16.square().mean(-1,keepdim=True);m32=x16.float().square().mean(-1,keepdim=True)
    rel=float(((m16.float()-m32).abs()/m32).max())
    promoted=RMSNorm(256)(x16)
    return dict(parity=results,bf16_mean_square_relative_error=rel,input_dtype=str(x16.dtype),
                gain_dtype='torch.float32',output_dtype=str(promoted.dtype),
                interpretation='均方相对误差不等于归一化输出相对误差；rsqrt敏感度须区分绝对与相对量')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path.cwd()/'log/book3-ch02/rmsnorm-check.json');a=ap.parse_args()
    if a.out.exists():raise FileExistsError('不覆盖已有核验：'+str(a.out))
    import transformers
    report=dict(torch=str(torch.__version__),transformers=transformers.__version__,device='cpu',
                properties=probe_properties(),parity=probe_parity(),training_rerun=False)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
