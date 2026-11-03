"""保存 AsyncLLM 原始记录，验证增量拼接，并执行真实输出收集器的CPU合并。"""
from pathlib import Path
import hashlib,json,shutil
from vllm.outputs import RequestOutput,CompletionOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.output_processor import RequestOutputCollector
ROOT=Path(__file__).resolve().parents[2]


def main():
    src=ROOT/'log/book8-source-rewrite/async-frontend';out=ROOT/'drafts/Book8-vLLM源码剖析/evidence/async-frontend';out.mkdir(exist_ok=True)
    for name in ['events.jsonl','manifest.json','result.json']:
        dest=out/name
        if dest.exists():assert dest.read_bytes()==(src/name).read_bytes()
        else:shutil.copyfile(src/name,dest)
    console=src.parent/'async-frontend-console.log';dest=out.parent/console.name
    if dest.exists():assert dest.read_bytes()==console.read_bytes()
    else:shutil.copyfile(console,dest)
    manifest=json.loads((out/'manifest.json').read_text());assert manifest['probe_sha256']==hashlib.sha256((ROOT/'code/Book8-vLLM源码剖析/observe_async_frontend.py').read_bytes()).hexdigest()
    rows=[json.loads(s) for s in (out/'events.jsonl').read_text().splitlines()]
    result=json.loads((out/'result.json').read_text());deltas={};objects=[]
    for row in rows:
        if row['event']!='user.delta':continue
        rid=row['external_id'];o=row['output'];c=o['outputs'][0]
        deltas.setdefault(rid,[]).append({'seq':row['seq'],'text':c['text'],'token_ids':c['token_ids'],'finished':o['finished'],'finish_reason':c['finish_reason']})
        if rid=='A-online':
            args={k:v for k,v in o.items() if k not in ['type','outputs']}
            objects.append(RequestOutput(**args,prompt_logprobs=None,outputs=[CompletionOutput(**{k:v for k,v in c.items() if k!='type'})]))
    for rid,items in deltas.items():
        assert ''.join(i['text'] for i in items)==result['assembled'][rid]['text']
        assert sum([i['token_ids'] for i in items],[])==result['assembled'][rid]['token_ids']
    collector=RequestOutputCollector(RequestOutputKind.DELTA,'A-online')
    for o in objects:collector.put(o)
    merged=collector.get_nowait();assert merged is not None
    assert merged.outputs[0].text==' Paris.' and list(merged.outputs[0].token_ids)==[12095,13] and merged.finished
    projection={'events_sha256':hashlib.sha256((out/'events.jsonl').read_bytes()).hexdigest(),'deltas':deltas,
        'collector_cpu':{'kind':'replay recorded deltas through actual CPU RequestOutputCollector, without consumer reads between puts',
            'put_count':len(objects),'get_count':1,'text':merged.outputs[0].text,'token_ids':list(merged.outputs[0].token_ids),'finished':merged.finished}}
    (out/'projection.json').write_text(json.dumps(projection,ensure_ascii=False,indent=2)+'\n')
    print('Async frontend: A/B delta assembly and CPU collector coalescing verified; cancellation and wait-idle retained')


if __name__=='__main__':main()
