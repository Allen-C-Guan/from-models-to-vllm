"""CPU执行固定源码FP8标量缩放函数；不是设备FP8矩阵乘。"""
import ast,hashlib,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]

def main():
    path=ROOT/'repos/vllm/vllm/model_executor/layers/quantization/utils/fp8_utils.py'
    fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='input_to_float8')
    ns={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    cases=[]
    for name,x in [('mixed',torch.tensor([[-2.,-1.,-.3,0.],[.1,.5,1.,2.]])),('zero',torch.zeros(2,4))]:
        q,s=ns['input_to_float8'](x,torch.float8_e4m3fn);restored=q.float()*s
        assert q.shape==x.shape and s.ndim==0 and torch.isfinite(restored).all()
        err=(x-restored).abs().max().item();assert err<.04
        cases.append({'name':name,'input':x.tolist(),'dtype':str(q.dtype),'encoded_values':q.float().tolist(),'encoded_bytes':q.view(torch.uint8).tolist(),'dequant_scale':s.item(),'restored':restored.tolist(),'max_abs_error':err})
    out=ROOT/'drafts/Book8-vLLM源码剖析/evidence/fp8-encoding';out.mkdir(exist_ok=True)
    data={'kind':'CPU fixed-source AST function with explicit dtype; no GEMM/model/NPU execution','probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()},'cases':cases}
    (out/'trace.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(cases,indent=2))

if __name__=='__main__':main()
