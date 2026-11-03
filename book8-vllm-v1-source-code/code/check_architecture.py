"""用固定源码 AST 核对类图关键边和调用树路径；不导入运行引擎。"""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research/Book8-源码重写/2026-11-02/revision-01'
FILES={
 'front':'vllm/v1/engine/llm_engine.py','client':'vllm/v1/engine/core_client.py',
 'input':'vllm/v1/engine/input_processor.py','output':'vllm/v1/engine/output_processor.py',
 'core':'vllm/v1/engine/core.py','scheduler':'vllm/v1/core/sched/scheduler.py',
 'contract':'vllm/v1/core/sched/interface.py','executor':'vllm/v1/executor/abstract.py',
 'uni':'vllm/v1/executor/uniproc_executor.py','plan':'vllm/v1/core/sched/output.py',
}


def main():
    trees={k:ast.parse((ROOT/'repos/vllm'/p).read_text()) for k,p in FILES.items()}
    verified=[]
    def cls(file,name):return next(n for n in trees[file].body if isinstance(n,ast.ClassDef) and n.name==name)
    def method(file,name,op):return next(n for n in cls(file,name).body if isinstance(n,ast.FunctionDef) and n.name==op)
    def record(kind,file,node,relation):
        verified.append({'kind':kind,'file':FILES[file],'line':node.lineno,'relation':relation})
    for file,child,parent in [('client','InprocClient','EngineCoreClient'),
        ('scheduler','Scheduler','SchedulerInterface'),('uni','UniProcExecutor','Executor')]:
        node=cls(file,child);assert parent in [ast.unparse(b) for b in node.bases]
        record('generalization',file,node,f'{child} -> {parent}')
    for file,owner,attr in [('front','LLMEngine','input_processor'),('front','LLMEngine','output_processor'),
        ('front','LLMEngine','engine_core'),('output','OutputProcessor','request_states'),
        ('client','InprocClient','engine_core'),('core','EngineCore','scheduler'),
        ('core','EngineCore','model_executor'),('scheduler','Scheduler','requests'),
        ('scheduler','Scheduler','kv_cache_manager')]:
        assignment=next(n for n in ast.walk(cls(file,owner)) if isinstance(n,(ast.Assign,ast.AnnAssign)) and
            any(ast.unparse(t)=='self.'+attr for t in (n.targets if isinstance(n,ast.Assign) else [n.target])))
        record('stored_reference',file,assignment,f'{owner}.{attr}')
    for attr,expected in [('scheduled_new_reqs','list[NewRequestData]'),('scheduled_cached_reqs','CachedRequestData')]:
        node=next(n for n in cls('plan','SchedulerOutput').body if isinstance(n,ast.AnnAssign) and ast.unparse(n.target)==attr)
        assert ast.unparse(node.annotation)==expected
        record('typed_plan_field','plan',node,f'{attr}: {expected}')
    calls=[
        ('input','InputProcessor','process_inputs','EngineCoreRequest'),
        ('front','LLMEngine','add_request','self.input_processor.process_inputs'),
        ('front','LLMEngine','add_request','self.input_processor.assign_request_id'),
        ('front','LLMEngine','add_request','self.output_processor.add_request'),
        ('front','LLMEngine','add_request','self.engine_core.add_request'),
        ('output','OutputProcessor','add_request','RequestState.from_new_request'),
        ('client','InprocClient','add_request','self.engine_core.preprocess_add_request'),
        ('client','InprocClient','add_request','self.engine_core.add_request'),
        ('core','EngineCore','preprocess_add_request','Request.from_engine_core_request'),
        ('core','EngineCore','add_request','self.scheduler.add_request'),
        ('scheduler','Scheduler','add_request','self._enqueue_waiting_request'),
        ('front','LLMEngine','step','self.engine_core.get_output'),
        ('client','InprocClient','get_output','self.engine_core.step_fn'),
        ('client','InprocClient','get_output','self.engine_core.post_step'),
        ('core','EngineCore','step','self.scheduler.schedule'),
        ('core','EngineCore','step','self.model_executor.execute_model'),
        ('core','EngineCore','step','future.result'),
        ('core','EngineCore','step','self.model_executor.sample_tokens'),
        ('core','EngineCore','step','self.scheduler.update_from_output'),
        ('front','LLMEngine','step','self.output_processor.process_outputs'),
        ('front','LLMEngine','step','self.engine_core.abort_requests'),
        ('scheduler','Scheduler','schedule','self.kv_cache_manager.allocate_slots'),
        ('scheduler','Scheduler','schedule','SchedulerOutput'),
    ]
    for file,owner,op,callee in calls:
        node=next(n for n in ast.walk(method(file,owner,op)) if isinstance(n,ast.Call) and ast.unparse(n.func)==callee)
        record('call_site',file,node,f'{owner}.{op} -> {callee}')
    draft=ROOT/'drafts/Book8-vLLM源码剖析'
    for ch in [3,4,5]:
        doc=next(draft.glob(f'ch0{ch}-*.md'));s=doc.read_text()
        assert s.index(f'fig-{ch}-0-map.png')<s.index('## ')
        assert re.search(r'```text\n.*?(?:LLMEngine|EngineCore).*?```',s,re.S)
    before=json.loads((OUT/'before.json').read_text())
    preserved=[name for name in before if '/evidence/' in name or name.endswith('/observe_requests.py')]
    assert preserved
    for name in preserved:assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==before[name],name
    result={'verified_relations_and_calls':verified,'unchanged_observation_files':len(preserved),
        'chapter_maps_before_first_section':3,'call_path_blocks':3,
        'scope':'AST 核对源码中的边；不证明全部分支执行或运行时堆栈。'}
    (OUT/'architecture-validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(f'{len(verified)} source relationships/call sites; {len(preserved)} observation files preserved')


if __name__=='__main__':main()
