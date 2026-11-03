"""执行上游 KV 元数据对象；没有模型前向、设备 KV 或 NPU 性能结论。"""
from pathlib import Path
import hashlib
import importlib.metadata
import inspect
import json
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher,init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec,KVCacheConfig,KVCacheGroupSpec
from vllm.v1.request import Request

ROOT=Path(__file__).resolve().parents[2]
A=[785,6722,315,9625,374]
B=[785,7772,17951,389,9237,374]


def manager(size,caching):
    config=KVCacheConfig(num_blocks=4,kv_cache_tensors=[],kv_cache_groups=[KVCacheGroupSpec(['teaching_layer'],FullAttentionSpec(block_size=size,num_kv_heads=1,head_size=1,dtype=torch.float32))])
    return KVCacheManager(config,max_model_len=128,scheduler_block_size=size,hash_block_size=size,enable_caching=caching)


def request(name,tokens,size):
    return Request(request_id=name,prompt_token_ids=tokens,sampling_params=SamplingParams(max_tokens=3),pooling_params=None,block_hasher=get_request_block_hasher(size,sha256))


def snapshot(m):
    pool=m.block_pool
    return {'num_free_blocks':pool.get_num_free_blocks(),
            'free_order':[b.block_id for b in pool.free_block_queue.get_all_free_blocks()],
            'blocks':[{'block_id':b.block_id,'ref_cnt':b.ref_cnt,'is_null':b.is_null,'has_hash':b.block_hash is not None} for b in pool.blocks],
            'req_to_blocks':[{rid:[b.block_id for b in blocks] for rid,blocks in g.req_to_blocks.items()} for g in m.coordinator.single_type_managers]}


def main():
    init_none_hash(sha256)
    import vllm.v1.core.block_pool as poolmod
    import vllm.v1.core.single_type_kv_cache_manager as singlemod
    import vllm.v1.core.kv_cache_coordinator as coordmod
    checked={}
    for obj in [KVCacheManager,poolmod,singlemod,coordmod]:
        f=Path(inspect.getfile(obj));relative=f.relative_to(Path(importlib.metadata.distribution('vllm').locate_file('')))
        source=ROOT/'repos/vllm'/relative
        assert f.read_bytes()==source.read_bytes(),relative
        checked[str(relative)]=hashlib.sha256(f.read_bytes()).hexdigest()
    data={'kind':'CPU metadata-only execution; no model forward or KV tensors','source_sha256':checked,'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    m=manager(128,False);a=request('A',A,128);b=request('B',B,128)
    rows=[{'stop':'initial',**snapshot(m)}]
    for r in [a,b]:
        allocated=m.allocate_slots(r,len(r.prompt_token_ids));assert allocated is not None
        rows.append({'stop':'allocate_'+r.request_id,'returned_block_ids':allocated.get_block_ids(),**snapshot(m)})
    m.free(a);rows.append({'stop':'free_A',**snapshot(m)})
    c=request('C',[785,6722,315,6323,374],128)
    allocated=m.allocate_slots(c,5);rows.append({'stop':'allocate_C','returned_block_ids':allocated.get_block_ids(),**snapshot(m)})
    data['allocation']={row['stop']:row for row in rows}
    m=manager(4,True);a=request('A-prefix',A,4)
    assert m.allocate_slots(a,5) is not None
    rows=[{'stop':'A_registered_5_slots',**snapshot(m)}]
    # 驱动只推进元数据；没有执行 attention，也没有生成 KV 内容。
    a.num_computed_tokens=5
    m.free(a);rows.append({'stop':'A_released',**snapshot(m)})
    again=request('A-again',A,4)
    hits,n,boundary=m.get_computed_blocks(again)
    rows.append({'stop':'lookup_A_again','hit_block_ids':hits.get_block_ids(),'num_hit_tokens':n,'shared_prefix_boundary':boundary,**snapshot(m)})
    assert n==4
    allocated=m.allocate_slots(again,1,num_new_computed_tokens=n,new_computed_blocks=hits)
    assert allocated is not None
    rows.append({'stop':'attach_hit_and_allocate_tail','returned_block_ids':allocated.get_block_ids(),**snapshot(m)})
    assert m.block_pool.blocks[1].ref_cnt==1
    data['prefix']={row['stop']:row for row in rows}
    out=ROOT/'drafts/Book8-vLLM源码剖析/evidence/kv-metadata';out.mkdir(parents=True,exist_ok=True)
    (out/'trace.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(out/'trace.json'),'allocation_stops':len(data['allocation']),'prefix_stops':len(rows),'source_files_verified':len(checked)},ensure_ascii=False))


if __name__=='__main__':main()
