"""用构造时钟驱动真实指标对象，并验证请求收集器错误唤醒。"""
import asyncio,hashlib,inspect,json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from vllm.v1.metrics.stats import IterationStats,RequestStateStats,LoRARequestStates
from vllm.v1.engine import EngineCoreOutput,EngineCoreEvent,EngineCoreEventType,FinishReason
from vllm.v1.engine.output_processor import OutputProcessor,RequestOutputCollector
from vllm.sampling_params import RequestOutputKind
ROOT=Path(__file__).resolve().parents[2]

async def errors():
    queues={r:RequestOutputCollector(RequestOutputKind.DELTA,r) for r in ['A','B']}
    state=SimpleNamespace(request_states={r:SimpleNamespace(queue=q) for r,q in queues.items()})
    tasks={r:asyncio.create_task(q.get()) for r,q in queues.items()}
    await asyncio.sleep(0)
    OutputProcessor.propagate_error(state,RuntimeError('synthetic core failure'))
    result={}
    for r,t in tasks.items():
        try:await t
        except RuntimeError as e:result[r]={'type':type(e).__name__,'message':str(e),'ready_after_get':queues[r].ready.is_set()}
    assert len(result)==2
    return result

def main():
    req=RequestStateStats(arrival_time=100.0);lora=LoRARequestStates();rows=[]
    for i,(token,wall,core) in enumerate([(279,100.5,10.4),(16462,100.65,10.55),(21575,100.8,10.7)]):
        stats=IterationStats();stats.iteration_timestamp=wall
        events=[EngineCoreEvent(EngineCoreEventType.QUEUED,10.),EngineCoreEvent(EngineCoreEventType.SCHEDULED,10.1)] if i==0 else ([EngineCoreEvent(EngineCoreEventType.PREEMPTED,10.5),EngineCoreEvent(EngineCoreEventType.SCHEDULED,10.53)] if i==1 else [])
        output=EngineCoreOutput(request_id='B-metrics',new_token_ids=[token],events=events)
        stats.update_from_output(output,core,i==0,req,lora,None)
        rows.append({'frontend_time':wall,'core_time':core,'ttft_samples':stats.time_to_first_tokens_iter,'itl_samples':stats.inter_token_latencies_iter,'scheduled_ts':req.scheduled_ts,'preemptions':req.num_preemptions})
    stats.update_from_finished_request(FinishReason.LENGTH,'B-metrics',6,3,req)
    finished=asdict(stats.finished_requests[0]);assert abs(finished['mean_time_per_output_token']-.15)<1e-8 and req.scheduled_ts==10.1
    files={}
    for obj in [IterationStats,OutputProcessor,EngineCoreEvent]:
        p=Path(inspect.getsourcefile(obj));rel=Path('vllm')/str(p).rsplit('/vllm/',1)[1];src=ROOT/'repos/vllm'/rel;assert p.read_bytes()==src.read_bytes();files[str(src.relative_to(ROOT))]=hashlib.sha256(src.read_bytes()).hexdigest()
    out=ROOT/'drafts/Book8-vLLM源码剖析/evidence/metrics-contract';out.mkdir(exist_ok=True)
    data={'kind':'CPU actual statistics and collector methods with synthetic timestamps/events/error; not latency benchmark','probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':files,'updates':rows,'finished':finished,'error_delivery':asyncio.run(errors())}
    (out/'trace.json').write_text(json.dumps(data,indent=2)+'\n');print('CPU clock-domain formulas, preemption and collector exception delivery passed')

if __name__=='__main__':main()
