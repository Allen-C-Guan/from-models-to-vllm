"""Book5第8章：位置公式与边界的CPU小例；不训练、不测长文质量。
从本册目录运行 python code/ch08/position_checks.py --out-name run1。
"""
from pathlib import Path
import argparse,json,math
import torch

def frequencies(head_dim,rho,theta):
    """偶数头维D；返回旋转子空间的r个频率，CPU float64。"""
    if head_dim<2 or head_dim%2 or not 0<=rho<=1 or theta<=0:
        raise ValueError("偶数头维、0<=rho<=1、正theta")
    width=int(head_dim*rho)
    if width%2:raise ValueError("本练习要求实际旋转宽度为偶数")
    if width==0:return torch.empty(0,dtype=torch.float64)
    return theta**(-torch.arange(0,width,2,dtype=torch.float64)/width)

def rotate(x,position,inv_freq):
    """x:(...,D)，相邻两维一对；HF split-half布局可通过固定维度排列互换。"""
    width=2*inv_freq.numel()
    if width>x.shape[-1]:raise ValueError("旋转子空间超过头维")
    if width==0:return x.clone()
    pairs=x[...,:width].reshape(*x.shape[:-1],-1,2)
    angle=torch.as_tensor(position,dtype=x.dtype)*inv_freq
    c,s=angle.cos(),angle.sin()
    a,b=pairs.unbind(-1)
    result=torch.stack((a*c-b*s,a*s+b*c),-1).flatten(-2)
    return torch.cat((result,x[...,width:]),-1)

def checks():
    torch.set_num_threads(1);torch.manual_seed(81);result={}
    dtype=torch.float64
    q=torch.tensor([1,0,3,-2,5,1,0,-4],dtype=dtype)
    freq=torch.tensor([math.pi/6],dtype=dtype)  # 构造频率；真实theta公式首频率是1。
    out=rotate(q,2,freq)
    expected=torch.tensor([.5,math.sqrt(3)/2,3,-2,5,1,0,-4],dtype=dtype)
    assert torch.allclose(out,expected,atol=1e-12,rtol=0)
    result["toy_rotation"]=out.tolist()
    for rho in (0,.25,1):
        fs=frequencies(8,rho,1e4)
        a,b=torch.randn(8,dtype=dtype),torch.randn(8,dtype=dtype)
        w=2*len(fs)
        left=rotate(a,3,fs)@rotate(b,11,fs)
        right=a[:w]@rotate(b[:w],8,fs)+a[w:]@b[w:]
        assert torch.allclose(left,right,atol=1e-12,rtol=0)
        result[f"relative_identity_rho_{rho}"]=float((left-right).abs())
    good=frequencies(256,.25,1e7)
    truncated=1e7**(-torch.arange(0,64,2,dtype=dtype)/256)
    result["frequency_denominators"]={"partial_64_slowest":float(good[-1]),"full256_truncated_slowest":float(truncated[-1])}
    # 因果掩码+可区分BOS产生位置线索；全部相同值是相反例。
    n=6;score=torch.zeros(n,n,dtype=dtype).masked_fill(torch.ones(n,n,dtype=torch.bool).triu(1),-float("inf"))
    weights=score.softmax(-1)
    bos=torch.zeros(n,dtype=dtype);bos[0]=1
    signal=weights@bos;constant=weights@torch.ones(n,dtype=dtype)
    assert torch.allclose(signal,1/torch.arange(1,n+1,dtype=dtype),atol=1e-14,rtol=0)
    assert torch.allclose(constant,torch.ones(n,dtype=dtype),atol=1e-14,rtol=0)
    result["NoPE_BOS_signal"]=signal.tolist();result["identical_values_counterexample"]=constant.tolist()
    positions=torch.tensor([0,8190,8191,16383],dtype=dtype)
    gain=1+.1*torch.log(torch.floor((positions+1)/8192)+1)
    result["Llama4_NoRoPE_query_gain"]=dict(zip(map(str,positions.int().tolist()),gain.tolist()))
    result["state_crossings"]={}
    for name,matrix,conv,slope in [
        ("toy",196608,27648,320),("Qwen3-Next",18874368,884736,12288),("K3",108527616,7630848,13824)]:
        result["state_crossings"][name]=dict(matrix_only=matrix/slope,with_min_conv=(matrix+conv)/slope,
            first_integer_exceeding_matrix=matrix//slope+1,first_integer_exceeding_with_conv=(matrix+conv)//slope+1)
    result["Book3_8K_deltas"]={k:v-4.254 for k,v in dict(none=5.397,pi=4.689,ntk=4.531,yarn_lite=4.491).items()}
    result["Book3_none_16K_delta"]=5.694-4.254
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--out-name",default="run1");args=ap.parse_args()
    if not args.out_name or Path(args.out_name).name!=args.out_name or args.out_name in (".",".."):
        raise ValueError("输出名必须为单个文件名")
    root=next((p for p in Path(__file__).resolve().parents if (p/"env.sh").is_file()),Path.cwd())
    out=root/"log/book5-ch08"/f"positions_{args.out_name}.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    result=dict(passed=True,torch=torch.__version__,kind="CPU数学与边界检查，不是质量或性能评测",checks=checks())
    with out.open("x") as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
