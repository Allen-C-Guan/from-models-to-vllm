"""从固定源码AST执行专家布局纯函数；不运行MoE模型或设备内核。"""
import ast,hashlib,json
from pathlib import Path
from typing import Literal
import torch
ROOT=Path(__file__).resolve().parents[2]


def main():
    path=ROOT/'repos/vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py'
    tree=ast.parse(path.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='determine_expert_map')
    ns={'torch':torch,'ExpertPlacementStrategy':Literal['linear','round_robin']}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    traces={}
    for strategy in ['linear','round_robin']:
        traces[strategy]=[]
        for rank in range(2):
            n,m,mask=ns['determine_expert_map'](2,rank,4,strategy)
            traces[strategy].append({'rank':rank,'local_num_experts':n,'expert_map':m.tolist()})
    assert traces['linear'][0]['expert_map']==[0,1,-1,-1]
    assert traces['linear'][1]['expert_map']==[-1,-1,0,1]
    assert traces['round_robin'][0]['expert_map']==[0,-1,1,-1]
    result={'kind':'CPU execution of fixed-source function via AST; layouts synthetic; no full placement selection or model run',
        'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'layouts':traces}
    out=ROOT/'drafts/Book8-vLLM源码剖析/evidence/expert-map';out.mkdir(exist_ok=True)
    (out/'trace.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('CPU fixed-source expert maps: linear and round_robin checked; no kernel execution')


if __name__=='__main__':main()
