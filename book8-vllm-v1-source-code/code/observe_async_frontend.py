"""AsyncLLM 真实 NPU 观测：并发 A/B 的 DELTA 输出、关闭 C 生成器后取消。
不启动 HTTP；核心仍显式使用同步调度，以分离前端异步与核心异步。
"""
from pathlib import Path
import asyncio,hashlib,importlib.metadata,json,os,time
from observe_requests import MODEL,ROOT,snapshot,typename,location


async def run():
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.sampling_params import SamplingParams,RequestOutputKind
    out=ROOT/'log/book8-source-rewrite/async-frontend'
    out.mkdir(exist_ok=False)
    settings=dict(model=str(MODEL),tokenizer=str(MODEL),dtype='bfloat16',max_model_len=128,
        max_num_seqs=4,max_num_batched_tokens=32,enforce_eager=True,async_scheduling=False,
        enable_prefix_caching=False,enable_chunked_prefill=True,gpu_memory_utilization=.15,
        kv_cache_memory_bytes=64*1024*1024,seed=0,disable_log_stats=True)
    engine=AsyncLLM.from_engine_args(AsyncEngineArgs(**settings))
    rows=[]
    def emit(event,**kw):
        row={'seq':len(rows)+1,'event':event,'pid':os.getpid(),'monotonic_ns':time.monotonic_ns(),**kw}
        rows.append(row)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    original_abort=engine.abort
    async def abort(request_id,internal=False):
        emit('abort.call',request_id=request_id,internal=internal,unfinished_before=engine.get_num_unfinished_requests())
        await original_abort(request_id,internal)
        emit('abort.return',unfinished_after=engine.get_num_unfinished_requests())
    engine.abort=abort
    manifest={'packages':{n:importlib.metadata.version(n) for n in ['vllm','vllm-ascend','torch','torch-npu']},
        'requested_settings':settings,'effective':{'client':typename(engine.engine_core),'async_scheduling':engine.vllm_config.scheduler_config.async_scheduling,'use_v2_model_runner':engine.vllm_config.use_v2_model_runner},
        'model_revision':MODEL.name,'visible_devices':os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'methods':{'generate':location(engine.generate),'abort':location(original_abort)},
        'boundary':'AsyncLLM frontend observed; no HTTP server or core-process internal probes'}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    def generate(rid,prompt,length):
        rendered=engine.renderer.render_cmpl([{'prompt':prompt}])[0]
        return engine.generate(rendered,SamplingParams(temperature=0,max_tokens=length,ignore_eos=True,seed=0,output_kind=RequestOutputKind.DELTA),rid)
    collected={}
    async def consume(rid,prompt,length):
        values=[];text='';tokens=[]
        async for item in generate(rid,prompt,length):
            value=snapshot(item);emit('user.delta',external_id=rid,output=value);values.append(value)
            text+=item.outputs[0].text;tokens.extend(item.outputs[0].token_ids)
        collected[rid]={'text':text,'token_ids':tokens,'chunks':len(values),'last_finished':values[-1]['finished']}
    try:
        await asyncio.gather(consume('A-online','The capital of France is',2),consume('B-online','The largest ocean on Earth is',3))
        assert collected['A-online']['token_ids']==[12095,13]
        assert collected['B-online']['token_ids']==[279,16462,21575]
        gen=generate('C-cancel','The capital of Japan is',32)
        first=await anext(gen);assert not first.finished
        emit('user.cancel_after_first',output=snapshot(first))
        await gen.aclose()
        # 等待核心处理完已发出的取消与在途工作；不以任意sleep猜测完成。
        await engine.pause_generation(mode='wait',clear_cache=False)
        emit('core.wait_idle_returned',frontend_unfinished=engine.get_num_unfinished_requests())
        assert engine.get_num_unfinished_requests()==0
        result={'assembled':collected,'cancel_first':snapshot(first),'aborts':[r for r in rows if r['event'].startswith('abort.')],
            'idle':rows[-1]}
        (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(result,ensure_ascii=False))
    finally:engine.shutdown()


def main():
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='1'
    os.environ['VLLM_USE_V2_MODEL_RUNNER']='1'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
    asyncio.run(run())


if __name__=='__main__':main()
