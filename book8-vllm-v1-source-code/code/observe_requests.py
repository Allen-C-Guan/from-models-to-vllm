"""第 3–5 章真实请求观测；只包装本次创建的对象，不修改安装包或上游仓。

运行前 npu-smi info，选择空闲卡。在根目录：
export ASCEND_RT_VISIBLE_DEVICES=0
source ./env.sh
python code/Book8-vLLM源码剖析/observe_requests.py --case baseline
默认输出 log/book8-source-rewrite/<case>-<deployment>-<traced|plain>/。
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import dataclasses
import enum
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'log/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca'


def typename(obj):
    return f'{type(obj).__module__}.{type(obj).__qualname__}'


def location(method):
    fn = inspect.unwrap(method)
    try:
        file = inspect.getsourcefile(fn)
        return {'symbol': f'{fn.__module__}.{fn.__qualname__}',
                'file': file, 'line': inspect.getsourcelines(fn)[1],
                'sha256': hashlib.sha256(Path(file).read_bytes()).hexdigest()}
    except (TypeError, OSError):
        return {'symbol': str(fn)}


def snapshot(obj, depth=0):
    """JSON 字段投影，保留容器内容；不遍历任意对象的内部属性图。"""
    if isinstance(obj, enum.Enum):
        return obj.name
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if depth > 8:
        return {'type': typename(obj), 'omitted': 'depth'}
    if isinstance(obj, dict):
        return {str(k): snapshot(v, depth+1) for k, v in obj.items()}
    if isinstance(obj, (Sequence, set)):
        values = sorted(obj) if isinstance(obj, set) else obj
        return [snapshot(v, depth+1) for v in values]
    if dataclasses.is_dataclass(obj):
        return {'type': typename(obj), **{f.name: snapshot(getattr(obj, f.name), depth+1)
                for f in dataclasses.fields(obj)}}
    if hasattr(obj, '__struct_fields__'):
        return {'type': typename(obj), **{n: snapshot(getattr(obj, n), depth+1)
                for n in obj.__struct_fields__}}
    if hasattr(obj, 'shape') and hasattr(obj, 'dtype'):
        result = {'type': typename(obj), 'shape': list(obj.shape), 'dtype': str(obj.dtype)}
        if hasattr(obj, 'device'):
            result['device'] = str(obj.device)
        # 仅在显式边界复制小张量；会引入同步，不用于性能测量。
        size = obj.numel() if hasattr(obj, 'numel') else obj.size
        if size <= 64:
            result['values'] = obj.tolist()
        return result
    if hasattr(obj, 'outputs') and hasattr(obj, 'request_id'):
        return pick(obj, ['request_id', 'prompt', 'prompt_token_ids', 'outputs', 'finished'])
    if type(obj).__name__ == 'CompletionOutput':
        return pick(obj, ['index', 'text', 'token_ids', 'finish_reason', 'stop_reason'])
    return {'type': typename(obj)}


def pick(obj, fields):
    return {'type': typename(obj), **{f: snapshot(getattr(obj, f)) for f in fields if hasattr(obj, f)}}


def request_view(req):
    return pick(req, ['request_id', 'status', 'prompt_token_ids', 'output_token_ids',
        'num_prompt_tokens', 'num_tokens', 'num_tokens_with_spec', 'num_computed_tokens',
        'num_in_flight_tokens', 'num_output_placeholders', 'is_prefill_chunk', 'max_tokens'])


def scheduler_view(scheduler):
    return {'current_step': scheduler.current_step,
            'waiting': [r.request_id for r in scheduler.waiting],
            'running': [r.request_id for r in scheduler.running],
            'requests': {k: request_view(v) for k, v in scheduler.requests.items()}}


def output_states(processor):
    return {k: pick(v, ['request_id', 'external_req_id', 'prompt', 'prompt_token_ids',
            'prompt_len', 'max_tokens_param', 'output_kind', 'is_prefilling', 'sent_tokens_offset'])
            for k, v in processor.request_states.items()}


class Recorder:
    def __init__(self, out):
        self.path = out / 'events.jsonl'
        self.n = 0
        self.step = 0
        self.methods = {}

    def emit(self, event, **values):
        self.n += 1
        row = {'seq': self.n, 'pid': os.getpid(), 'monotonic_ns': time.monotonic_ns(),
               'front_step': self.step, 'event': event, **snapshot(values)}
        with self.path.open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    def wrap(self, obj, method, before=None, after=None):
        original = getattr(obj, method)
        name = f'{typename(obj)}.{method}'
        self.methods[name] = location(original)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if before:
                before(args, kwargs)
            result = original(*args, **kwargs)
            if after:
                after(args, kwargs, result)
            return result
        setattr(obj, method, wrapped)


def instrument(engine, rec):
    ip, op, client = engine.input_processor, engine.output_processor, engine.engine_core
    rec.wrap(ip, 'process_inputs', after=lambda a,k,r: rec.emit('input.processed', request=r))
    rec.wrap(ip, 'assign_request_id', after=lambda a,k,r: rec.emit('input.identified', request=a[0]))
    rec.wrap(op, 'add_request', after=lambda a,k,r: rec.emit('output.registered', states=output_states(op)))
    rec.wrap(op, 'process_outputs',
        before=lambda a,k: rec.emit('output.received', outputs=a[0]),
        after=lambda a,k,r: rec.emit('output.processed', outputs=r, states=output_states(op)))
    rec.wrap(client, 'add_request', before=lambda a,k: rec.emit('client.add', request=a[0]))
    rec.wrap(client, 'get_output', after=lambda a,k,r: rec.emit('client.output', outputs=r))
    core = getattr(client, 'engine_core', None)
    if core is None:
        return
    sched = core.scheduler
    rec.wrap(core, 'preprocess_add_request', after=lambda a,k,r: rec.emit('core.preprocessed', request=request_view(r[0])))
    rec.wrap(core, 'add_request', after=lambda a,k,r: rec.emit('core.registered', scheduler=scheduler_view(sched)))
    rec.wrap(core, 'step_fn',
        before=lambda a,k: rec.emit('core.step.begin', scheduler=scheduler_view(sched)),
        after=lambda a,k,r: rec.emit('core.step.end', outputs=r, scheduler=scheduler_view(sched)))
    rec.wrap(sched, 'schedule',
        before=lambda a,k: rec.emit('schedule.before', scheduler=scheduler_view(sched)),
        after=lambda a,k,r: rec.emit('schedule.after', plan=r, scheduler=scheduler_view(sched)))
    rec.wrap(sched, 'update_from_output',
        before=lambda a,k: rec.emit('update.before', model_output=a[1], scheduler=scheduler_view(sched)),
        after=lambda a,k,r: rec.emit('update.after', outputs=r, scheduler=scheduler_view(sched)))
    manager = sched.kv_cache_manager
    rec.wrap(manager, 'allocate_slots',
        after=lambda a,k,r: rec.emit('kv.allocated', request_id=a[0].request_id,
            requested_tokens=a[1] if len(a)>1 else k.get('num_new_tokens'),
            granted=r is not None, block_ids=r.get_block_ids() if r is not None else None))
    rec.wrap(manager, 'free', before=lambda a,k: rec.emit('kv.free', request_id=a[0].request_id, request=request_view(a[0])))
    executor = core.model_executor
    rec.wrap(executor, 'execute_model',
        before=lambda a,k: rec.emit('execute.call', non_block=k.get('non_block'), plan=a[0]),
        after=lambda a,k,r: rec.emit('execute.return', returned_type=typename(r)))
    rec.wrap(executor, 'sample_tokens',
        before=lambda a,k: rec.emit('sample.call', grammar=a[0] if a else k),
        after=lambda a,k,r: rec.emit('sample.return', output=r))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['baseline', 'chunked'], default='baseline')
    parser.add_argument('--deployment', choices=['inproc', 'mp'], default='inproc')
    parser.add_argument('--plain', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    out = args.out or ROOT / 'log/book8-source-rewrite' / f'{args.case}-{args.deployment}-{"plain" if args.plain else "traced"}'
    out.mkdir(parents=True, exist_ok=False)
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0' if args.deployment == 'inproc' else '1'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '1'
    from vllm import LLM, SamplingParams

    settings = dict(model=str(MODEL), tokenizer=str(MODEL), dtype='bfloat16',
        max_model_len=128, max_num_seqs=4, max_num_batched_tokens=32 if args.case=='baseline' else 4,
        enforce_eager=True, async_scheduling=False, enable_prefix_caching=False,
        enable_chunked_prefill=True, gpu_memory_utilization=0.15,
        kv_cache_memory_bytes=64*1024*1024, seed=0, disable_log_stats=True)
    start = time.monotonic()
    llm = LLM(**settings)
    engine = llm.llm_engine
    rec = Recorder(out)
    if not args.plain:
        instrument(engine, rec)
    core = getattr(engine.engine_core, 'engine_core', None)
    cfg = engine.vllm_config
    manifest = {'case':args.case, 'deployment':args.deployment, 'instrumented':not args.plain,
        'packages':{n:importlib.metadata.version(n) for n in ['vllm','vllm-ascend','torch','torch-npu']},
        'requested_settings':settings,
        'effective':{'async_scheduling':cfg.scheduler_config.async_scheduling,
            'max_num_batched_tokens':cfg.scheduler_config.max_num_batched_tokens,
            'max_num_seqs':cfg.scheduler_config.max_num_seqs,
            'use_v2_model_runner':cfg.use_v2_model_runner,
            'block_size':cfg.cache_config.block_size,
            'num_gpu_blocks':cfg.cache_config.num_gpu_blocks,
            'client':typename(engine.engine_core)},
        'model_revision':MODEL.name, 'methods':rec.methods,
        'pid':os.getpid(), 'visible_devices':os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if core:
        worker = core.model_executor.driver_worker.worker
        manifest['effective'].update(scheduler=typename(core.scheduler), executor=typename(core.model_executor),
            worker=typename(worker), runner=typename(worker.model_runner),
            balance_enabled=getattr(core.scheduler,'_balance_enabled',None),
            scheduler_mro=[f'{c.__module__}.{c.__qualname__}' for c in type(core.scheduler).__mro__],
            core_step=location(core.step_fn), runner_execute=location(worker.model_runner.execute_model))
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    requests = [('A','The capital of France is',2),('B','The largest ocean on Earth is',3)]
    final = {}

    def submit(rid, text, length):
        rendered = engine.renderer.render_cmpl([{'prompt':text}])[0]
        params = SamplingParams(temperature=0, max_tokens=length, ignore_eos=True, seed=0)
        rec.emit('user.submit', external_id=rid, text=text, rendered=rendered, params=params)
        internal = engine.add_request(rid, rendered, params)
        rec.emit('user.accepted', external_id=rid, internal_id=internal)

    try:
        for req in requests:
            submit(*req)
        while engine.has_unfinished_requests():
            rec.step += 1
            if rec.step > 50:
                raise RuntimeError('Too many steps for the short teaching workload')
            outputs = engine.step()
            rec.emit('user.step', outputs=outputs)
            for output in outputs:
                if output.finished:
                    final[output.request_id] = snapshot(output)
            if args.case == 'chunked' and rec.step == 2:
                submit('C','The capital of Japan is',2)
        rec.emit('user.complete', final=final,
            output_states=output_states(engine.output_processor),
            scheduler=scheduler_view(core.scheduler) if core else None,
            pending_finished_req_ids=core.scheduler.finished_req_ids if core else None)
        (out/'result.json').write_text(json.dumps({'final':final,'steps':rec.step,
            'wall_seconds_including_startup':time.monotonic()-start},ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'out':str(out),'steps':rec.step,'outputs':final},ensure_ascii=False))
    finally:
        engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
