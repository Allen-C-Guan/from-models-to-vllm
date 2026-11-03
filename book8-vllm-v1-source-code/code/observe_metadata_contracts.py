"""CPU 契约驱动：调用真实回填方法与固定源码的纯匹配函数；不启动推理。"""
from pathlib import Path
import ast, dataclasses, hashlib, inspect, json
from types import SimpleNamespace
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.config.compilation import CUDAGraphMode

ROOT=Path(__file__).resolve().parents[2]


def main():
    installed=Path(inspect.getfile(AsyncScheduler))
    fixed=ROOT/'repos/vllm/vllm/v1/core/sched/async_scheduler.py'
    assert installed.read_bytes()==fixed.read_bytes()
    commits={str(fixed.relative_to(ROOT)):hashlib.sha256(fixed.read_bytes()).hexdigest()}
    calls=[]
    # 只装配该方法需要的依赖；记录 cache_blocks 调用，不分配 KV。
    scheduler=object.__new__(AsyncScheduler)
    scheduler.max_model_len=128
    scheduler.kv_cache_manager=SimpleNamespace(cache_blocks=lambda request,n:calls.append({'id':request.request_id,'num_tokens':n}))
    def request(name):
        return Request(name,[785,6722,315,9625,374],SamplingParams(max_tokens=2,ignore_eos=True),None)
    def snap(r):
        return {'id':r.request_id,'status':r.status.name,'num_computed_tokens':r.num_computed_tokens,'num_output_placeholders':r.num_output_placeholders,'output_token_ids':list(r.output_token_ids),'cache_calls':list(calls)}
    a=request('A-contract');a.status=RequestStatus.RUNNING
    # 明确构造初态：已有两轮排定，CPU 尚未接收其输出。
    a.num_computed_tokens=6;a.num_output_placeholders=2
    result={'kind':'CPU method contract driver; manually constructed initial state; no engine or device execution','async':{'initial':snap(a)}}
    for token,key in [(12095,'first_result'),(13,'second_result')]:
        output,stopped=scheduler._update_request_with_output(a,[token])
        result['async'][key]={'returned':output,'stopped':stopped,**snap(a)}
    assert a.num_output_placeholders==0 and a.status==RequestStatus.FINISHED_LENGTH_CAPPED
    stale=request('A-stale');stale.status=RequestStatus.PREEMPTED;stale.num_output_placeholders=0
    scheduler._update_request_with_output(stale,[12095],is_stale=True)
    result['async']['stale_delivery']=snap(stale)
    assert stale.num_output_placeholders==0 and len(calls)==2

    # 只执行纯 dataclass 与布尔匹配函数的原始 AST，不导入 GPU 图捕获模块。
    file=ROOT/'repos/vllm/vllm/v1/worker/gpu/cudagraph_utils.py'
    tree=ast.parse(file.read_text());nodes=[n for n in tree.body if getattr(n,'name',None) in {'BatchExecutionDescriptor','_is_compatible'}]
    ns={'dataclass':dataclasses.dataclass,'CUDAGraphMode':CUDAGraphMode,'__name__':__name__}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(file),'exec'),ns)
    desc=ns['BatchExecutionDescriptor'](CUDAGraphMode.FULL,4,4,uniform_token_count=1,max_query_len=1)
    cases={
        'two_decode':dict(num_reqs=2,num_tokens=2,uniform_token_count=1,num_active_loras=0,max_query_len=1,num_ubatches=1),
        'one_prefill_two_tokens':dict(num_reqs=1,num_tokens=2,uniform_token_count=None,num_active_loras=0,max_query_len=2,num_ubatches=1),
        'too_many_tokens':dict(num_reqs=5,num_tokens=5,uniform_token_count=1,num_active_loras=0,max_query_len=1,num_ubatches=1),
        'lora_mismatch':dict(num_reqs=2,num_tokens=2,uniform_token_count=1,num_active_loras=1,max_query_len=1,num_ubatches=1)}
    result['graph']={'kind':'fixed-source pure compatibility function; no graph capture or replay','descriptor':{**dataclasses.asdict(desc),'cg_mode':desc.cg_mode.name},'cases':{k:{'input':v,'compatible':ns['_is_compatible'](desc,**v)} for k,v in cases.items()}}
    assert [x['compatible'] for x in result['graph']['cases'].values()]==[True,False,False,False]
    commits[str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
    result['source_sha256']=commits
    result['probe_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out=ROOT/'drafts/Book8-vLLM源码剖析/evidence/metadata-contracts';out.mkdir(parents=True,exist_ok=True)
    (out/'trace.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('CPU: async placeholder/stale-output contract and 4 graph compatibility cases passed; no device execution')


if __name__=='__main__':main()
