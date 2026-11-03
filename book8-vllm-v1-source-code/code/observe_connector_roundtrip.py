"""ExampleConnector真实CPU文件/槽拷贝；构造对象，非引擎PD部署。"""
import hashlib,inspect,json
from pathlib import Path
from types import SimpleNamespace
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import ExampleConnector,ExampleConnectorMetadata,ReqMeta
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import OffloadingWorkerMetadata
ROOT=Path(__file__).resolve().parents[2]

def main():
    # 仅给真实方法所需状态；不假装走过引擎初始化或网络握手。
    c=object.__new__(ExampleConnector);c._block_size=4;c._storage_path=str(ROOT/'log/book8-source-rewrite/connector-cpu-storage')
    tokens=[785,6722,315,9625,374]
    store=ExampleConnectorMetadata();store.add_request(tokens,[1,2],4,True,[])
    load=ExampleConnectorMetadata();load.add_request(tokens,[3,4],4,False,[])
    src=torch.arange(6*1*4*2,dtype=torch.float32).reshape(6,1,4,2)
    dst=torch.full_like(src,-1)
    c.bind_connector_metadata(store);c.save_kv_layer('teaching_layer',src,SimpleNamespace())
    c.clear_connector_metadata();assert not c.has_connector_metadata()
    c.bind_connector_metadata(load)
    ctx=SimpleNamespace(attn_metadata={'teaching_layer':SimpleNamespace()},no_compile_layers={'teaching_layer':SimpleNamespace(kv_cache=dst)})
    c.start_load_kv(ctx);c.clear_connector_metadata()
    assert torch.equal(src[1],dst[3]) and (dst[:3]==-1).all() and (dst[4:]==-1).all()
    a=OffloadingWorkerMetadata();b=OffloadingWorkerMetadata();a.mark_completed(7);b.mark_completed(7);merged=a.aggregate(b)
    def meta(m):return {'token_ids':m.token_ids.tolist(),'slot_mapping':m.slot_mapping.tolist(),'is_store':m.is_store,'mm_hashes':m.mm_hashes}
    files={}
    for obj in [ExampleConnector,OffloadingWorkerMetadata]:
        p=Path(inspect.getsourcefile(obj));rel=Path('vllm')/str(p).rsplit('/vllm/',1)[1];s=ROOT/'repos/vllm'/rel;assert p.read_bytes()==s.read_bytes();files[str(s.relative_to(ROOT))]=hashlib.sha256(s.read_bytes()).hexdigest()
    out=ROOT/'drafts/Book8-vLLM源码剖析/evidence/connector-cpu';out.mkdir(exist_ok=True)
    data={'kind':'CPU real ExampleConnector methods on manually initialized object and synthetic tensors; no scheduler/network/NPU','probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':files,'roundtrip':{'shape':list(src.shape),'store_meta':meta(store.requests[0]),'load_meta':meta(load.requests[0]),'source_block_1':src[1].tolist(),'destination_block_3':dst[3].tolist(),'other_blocks_untouched':True},'completion_counts':{'worker0':a.completed_jobs,'worker1':b.completed_jobs,'aggregate':merged.completed_jobs}}
    (out/'trace.json').write_text(json.dumps(data,indent=2)+'\n');print('CPU ExampleConnector tensor roundtrip and 2-worker completion aggregation passed')

if __name__=='__main__':main()
