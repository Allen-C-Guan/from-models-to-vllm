"""CPU驱动真实DP路由方法；构造负载，不启动任何引擎/网络。"""
from pathlib import Path
from types import SimpleNamespace,MethodType
from collections import Counter
import asyncio,hashlib,inspect,json
from vllm.v1.engine.core_client import DPLBAsyncMPClient
ROOT=Path(__file__).resolve().parents[2]


def client(counts):
    c=SimpleNamespace(core_engines=[b'E0',b'E1'],lb_engines=counts,client_count=1,
        eng_start_index=0,engine_inflight=Counter(),reqs_in_flight={},resources=SimpleNamespace(engine_dead=False),sent=[])
    async def send(self,kind,ids,engine):self.sent.append({'kind':kind.name,'ids':ids,'engine':engine.decode()})
    c._send_input=MethodType(send,c)
    c._abort_requests=MethodType(DPLBAsyncMPClient._abort_requests,c)
    return c


def snap(c):
    return {'counts':c.lb_engines,'inflight':{k.decode():v for k,v in c.engine_inflight.items()},
            'owners':{k:v.decode() for k,v in c.reqs_in_flight.items()},'scan_start':c.eng_start_index}


def choose(c,rid,rank=None):
    request=SimpleNamespace(request_id=rid,data_parallel_rank=rank,pooling_params=None)
    chosen=DPLBAsyncMPClient.get_core_engine_for_request(c,request)
    return json.loads(json.dumps({'request':rid,'chosen':chosen.decode(),'after':snap(c)}))


def main():
    source=Path(inspect.getfile(DPLBAsyncMPClient));fixed=ROOT/'repos/vllm/vllm/v1/engine/core_client.py'
    assert source.read_bytes()==fixed.read_bytes()
    c=client([[0,0,0.0],[0,0,0.0]]);burst=[choose(c,'A'),choose(c,'B')]
    c.lb_engines=[[0,0,0.0],[0,0,0.0]]
    burst.append(choose(c,'C'))
    assert [v['chosen'] for v in burst]==['E0','E1','E0']
    asyncio.run(DPLBAsyncMPClient.abort_requests_async(c,['A','B','C']))
    assert c.sent==[{'kind':'ABORT','ids':['A','C'],'engine':'E0'},{'kind':'ABORT','ids':['B'],'engine':'E1'}]
    pressure=client([[1,0,.95],[1,0,.1]])
    low=choose(pressure,'C-pressure');assert low['chosen']=='E1'
    forced=choose(pressure,'C-forced',0);assert forced['chosen']=='E0'
    result={'kind':'CPU real methods, synthetic state; send replaced by recorder; no engine execution',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'burst':burst,'cancel_messages':c.sent,'pressure':low,'explicit_rank':forced}
    out=ROOT/'drafts/Book8-vLLM源码剖析/evidence/dp-contract';out.mkdir(exist_ok=True)
    (out/'trace.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('DP CPU contract: stale snapshots, KV pressure, explicit rank and grouped abort passed')


if __name__=='__main__':main()
