"""真实缓存规格与滑窗元数据，CPU构造状态，无设备缓存。"""
import hashlib,inspect,json
from pathlib import Path
from types import SimpleNamespace
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec,SlidingWindowSpec,MambaSpec
from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
ROOT=Path(__file__).resolve().parents[2]

def main():
    full=FullAttentionSpec(block_size=4,num_kv_heads=1,head_size=4,dtype=torch.float32)
    small=SlidingWindowSpec(block_size=4,num_kv_heads=1,head_size=2,dtype=torch.float32,sliding_window=4)
    mamba=MambaSpec(block_size=4,shapes=((2,3),(2,2)),dtypes=(torch.float32,torch.float32))
    specs={'full':full,'window':small,'mamba':mamba}
    def describe(d):return {k:{'type':type(v).__name__,'block_size':v.block_size,'page_bytes':v.page_size_bytes,'content_bytes':v.real_page_size_bytes} for k,v in d.items()}
    pages={'before':describe(specs),'after':describe(unify_kv_cache_spec_page_size(specs))}
    pool=BlockPool(num_gpu_blocks=8,enable_caching=False,hash_block_size=4)
    mgr=SlidingWindowManager(small,block_pool=pool,enable_caching=False,kv_cache_group_id=0,scheduler_block_size=4)
    mgr.req_to_blocks['C-window']=pool.get_new_blocks(3)
    def snap():return {'block_ids':[b.block_id for b in mgr.req_to_blocks['C-window']],'is_null':[b.is_null for b in mgr.req_to_blocks['C-window']],'free_blocks':pool.get_num_free_blocks()}
    before=snap();mgr.remove_skipped_blocks('C-window',7);after=snap()
    assert after['is_null']==[True,False,False] and after['free_blocks']==before['free_blocks']+1
    modes=[]
    for mode in ['none','align','all']:
        spec=MambaSpec(block_size=4,shapes=((2,3),(2,2)),dtypes=(torch.float32,torch.float32),mamba_cache_mode=mode)
        conf=SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode=mode),model_config=SimpleNamespace(max_model_len=16))
        modes.append({'mode':mode,'max_state_bytes':spec.max_memory_usage_bytes(conf),'block_table_columns':spec.max_num_blocks_per_req(conf,16)})
    files={}
    for obj in [FullAttentionSpec,unify_kv_cache_spec_page_size,BlockPool,SlidingWindowManager]:
        p=Path(inspect.getsourcefile(obj));rel=Path('vllm')/str(p).rsplit('/vllm/',1)[1];src=ROOT/'repos/vllm'/rel
        assert p.read_bytes()==src.read_bytes();files[str(src.relative_to(ROOT))]=hashlib.sha256(src.read_bytes()).hexdigest()
    out=ROOT/'drafts/Book8-vLLM源码剖析/evidence/hybrid-state';out.mkdir(exist_ok=True)
    data={'kind':'CPU actual specs and manager; manually seeded blocks; no device state/model','probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':files,'pages':pages,'window':{'window_size':4,'block_size':4,'computed':7,'skipped_tokens':mgr.get_num_skipped_tokens(7),'before':before,'after':after},'mamba_modes':modes}
    (out/'trace.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))

if __name__=='__main__':main()
