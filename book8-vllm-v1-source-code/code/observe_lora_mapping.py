"""MRV2真实LoraState与固定槽位转换函数；CPU构造状态，无adapter权重。"""
import ast,hashlib,inspect,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from vllm.lora.request import LoRARequest
from vllm.v1.worker.gpu.lora_utils import LoraState
ROOT=Path(__file__).resolve().parents[2]

def main():
    state=LoraState(4)
    state.add_request('A',3,LoRARequest('city-a',11,'/synthetic/not-loaded/a'))
    state.add_request('B',2,None)
    state.add_request('C',1,LoRARequest('city-c',22,'/synthetic/not-loaded/c'))
    prompt,tokens,active=state.make_lora_inputs(['B','A','C'],np.array([2,3,1]),np.array([1,2,1]))
    assert tuple(map(int,tokens))==(0,11,11,22)
    path=ROOT/'repos/vllm/vllm/lora/punica_wrapper/utils.py';tree=ast.parse(path.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='convert_mapping')
    ns={'torch':torch,'async_tensor_h2d':lambda values,dtype,device:torch.tensor(values,dtype=dtype,device=device)}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    mapping=SimpleNamespace(index_mapping=tokens,prompt_mapping=prompt)
    layouts=[]
    for slots in [[22,11,None],[11,22,None]]:
        base,sampler,_,_,_=ns['convert_mapping'](mapping,slots,3,16,torch.device('cpu'))
        layouts.append({'slot_to_adapter':slots,'token_slots':base.tolist(),'request_slots':sampler.tolist()})
    assert layouts[0]['token_slots']==[-1,1,1,0] and layouts[1]['token_slots']==[-1,0,0,1]
    files={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()}
    for cls,rel in [(LoraState,'vllm/v1/worker/gpu/lora_utils.py'),(LoRARequest,'vllm/lora/request.py')]:
        p=ROOT/'repos/vllm'/rel;assert Path(inspect.getsourcefile(cls)).read_bytes()==p.read_bytes();files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    out=ROOT/'drafts/Book8-vLLM源码剖析/evidence/lora-mapping';out.mkdir(exist_ok=True)
    data={'kind':'CPU actual LoraState and fixed-source convert_mapping AST; transfer helper replaced by CPU torch.tensor; no weights or kernels','probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':files,'batch':{'req_ids':['B','A','C'],'persistent_rows':[2,3,1],'num_scheduled_tokens':[1,2,1],'request_adapter_ids':list(map(int,prompt)),'token_adapter_ids':list(map(int,tokens)),'active_adapter_ids':sorted(a.lora_int_id for a in active)},'layouts':layouts}
    (out/'trace.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');print('CPU LoRA logical IDs, token expansion, two slot layouts passed')

if __name__=='__main__':main()
