"""从固定源码和真实 JSONL 生成正文证据块；保留人工撰写的其他段落。"""
from pathlib import Path
import hashlib
import json
import re
import shutil
from source_notes import NOTES, annotate

ROOT=Path(__file__).resolve().parents[2]
DRAFT=ROOT/'drafts/Book8-vLLM源码剖析'
LOG=ROOT/'log/book8-source-rewrite'
RESEARCH=ROOT/'research/Book8-源码重写/2026-11-02'
COMMIT='ced6857afa0ea7b2e3f0846a62e1394e90f15607'
ASCEND='c09d3e20b9c45c80b34ae3781440d90904a44023'
SOURCES={
 'core_request':('vllm','vllm/v1/engine/__init__.py',107,122),
 'assign_id':('vllm','vllm/v1/engine/input_processor.py',260,279),
 'register':('vllm','vllm/v1/engine/llm_engine.py',271,285),
 'core_step':('vllm','vllm/v1/engine/core.py',589,619),
 'advance':('vllm','vllm/v1/core/sched/scheduler.py',1520,1534),
 'append_stop':('vllm','vllm/v1/core/sched/scheduler.py',2347,2364),
 'frontend_step':('vllm','vllm/v1/engine/llm_engine.py',310,326),
 'inproc_output':('vllm','vllm/v1/engine/core_client.py',361,364),
 'demand':('vllm','vllm/v1/core/sched/scheduler.py',656,665),
 'allocate':('vllm','vllm/v1/core/sched/scheduler.py',1159,1180),
 'plan_definition':('vllm','vllm/v1/core/sched/output.py',224,240),
 'balance_delegate':('vllm-ascend','vllm_ascend/patch/platform/patch_balance_schedule.py',163,165),
}


def records(run):
    return [json.loads(s) for s in (LOG/run/'events.jsonl').read_text().splitlines()]


def select(rows,event,step=None):
    return next(r for r in rows if r['event']==event and (step is None or r['front_step']==step))


def fields(obj,*keys):
    return {k:obj[k] for k in keys}


def pretty(obj):
    s=json.dumps(obj,ensure_ascii=False,indent=2)
    # 单纯标量数组收成一行；仍是可解析、与投影严格相等的 JSON。
    s=re.sub(r'\[\n([^[\]{}]+?)\n\s*\]',lambda m:'['+' '.join(m[1].split())+']',s)
    return s


def main():
    blocks={};index={}
    for name,(repo,file,start,end) in SOURCES.items():
        path=ROOT/'repos'/repo/file
        lines=path.read_text().splitlines()
        source='\n'.join(lines[start-1:end])
        commit=COMMIT if repo=='vllm' else ASCEND
        link=f'https://github.com/vllm-project/{repo}/blob/{commit}/{file}#L{start}-L{end}'
        key='source:'+name
        teaching=annotate(name,lines[start-1:end],start)
        blocks[key]=f'源码教学注释版，{repo} v0.30.0：[ `{file}`，原文件 L{start}–L{end}]({link})。仅插入 `# 【解读】` 注释，其余内容保留原文；行号指向原文件。\n\n```python\n{teaching}\n```'
        index[key]={'repo':repo,'path':file,'start':start,'end':end,'commit':commit,
            'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'snippet_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'teaching_notes':NOTES[name]}

    base_name='baseline-inproc-traced';chunk_name='chunked-inproc-traced'
    base,chunk=records(base_name),records(chunk_name)
    def data(name,run,events,value):
        key='data:'+name
        seqs=[r['seq'] for r in events]
        links=f'[原始事件 {", ".join(map(str,seqs))}](evidence/{run}/events.jsonl)'
        blocks[key]=f'实测对象字段节选，{links}；JSON 是阅读投影，保留原始字段值。\n\n```json\n{pretty(value)}\n```'
        index[key]={'run':run,'seqs':seqs,'projection':value}

    r=select(base,'user.submit');data('rendered_a',base_name,[r],fields(r['rendered'],'type','prompt_token_ids','prompt'))
    r=select(base,'input.identified');aid=r['request']['request_id']
    req=fields(r['request'],'request_id','external_req_id','prompt_token_ids')
    req['sampling_params']=fields(r['request']['sampling_params'],'temperature','max_tokens','ignore_eos','output_kind')
    data('identified_a',base_name,[r],req)
    r=select(base,'output.registered');data('output_state_a',base_name,[r],r['states'][aid])
    r=select(base,'core.preprocessed');data('core_request_a',base_name,[r],fields(r['request'],
        'type','request_id','status','prompt_token_ids','output_token_ids','num_computed_tokens','num_in_flight_tokens','max_tokens'))
    r=[r for r in base if r['event']=='core.registered'][-1]
    s=r['scheduler'];data('admitted',base_name,[r],{'waiting':s['waiting'],'running':s['running'],
        'requests':{k:fields(v,'status','num_computed_tokens','num_in_flight_tokens') for k,v in s['requests'].items()}})
    first=select(base,'schedule.after',1)
    data('first_plan',base_name,[first],fields(first['plan'],'type','num_scheduled_tokens','total_num_scheduled_tokens','finished_req_ids'))
    moments=[select(base,e,1) for e in ['schedule.before','schedule.after','update.after']]
    data('progress',base_name,moments,[{'event':r['event'],**fields(r['scheduler']['requests'][aid],
        'num_tokens','num_computed_tokens','num_in_flight_tokens','output_token_ids')} for r in moments])
    events=[r for r in base if r['front_step']==1 and r['event'] in ['execute.return','sample.call','sample.return']]
    data('execution_events',base_name,events,[{'event':r['event'],**(
        {'returned_type':r['returned_type']} if 'returned_type' in r else
        {'grammar':r['grammar']} if 'grammar' in r else {'returned_type':r['output']['type']})} for r in events])
    r=select(base,'update.before',1);data('model_output',base_name,[r],fields(r['model_output'],
        'type','req_ids','req_id_to_index','sampled_token_ids'))
    r=select(base,'user.step',2);a=next(o for o in r['outputs'] if o['request_id']=='A')
    v=fields(a,'request_id','finished');v['outputs']=[fields(a['outputs'][0],'text','token_ids','finish_reason')]
    data('final_a',base_name,[r],v)
    r=select(base,'kv.allocated');data('allocation',base_name,[r],fields(r,'request_id','requested_tokens','granted','block_ids'))
    a=next(v for v in first['plan']['scheduled_new_reqs'] if v['req_id']==aid)
    data('new_a',base_name,[first],fields(a,'type','req_id','prompt_token_ids','block_ids','num_computed_tokens','prefill_token_ids'))
    r=select(base,'schedule.after',2);data('cached_second',base_name,[r],r['plan']['scheduled_cached_reqs'])
    rs=[select(chunk,e,1) for e in ['schedule.after','update.before','update.after']]
    rid=next(iter(rs[0]['plan']['num_scheduled_tokens']))
    data('chunk_first',chunk_name,rs,{
        'num_scheduled_tokens':rs[0]['plan']['num_scheduled_tokens'],
        'sampled_token_ids':rs[1]['model_output']['sampled_token_ids'],
        'request_after_update':fields(rs[2]['scheduler']['requests'][rid],
            'status','num_computed_tokens','num_in_flight_tokens','output_token_ids')})

    for name,run,rows in [('baseline',base_name,base),('chunked',chunk_name,chunk)]:
        ids={r['internal_id']:r['external_id'] for r in rows if r['event']=='user.accepted'}
        plans=[r for r in rows if r['event']=='schedule.after']
        def pretty_ids(d):return '；'.join(f'{ids[k]}：{v}' for k,v in d.items()) or '无'
        table=['| 轮次 | 实际调度的输入 token 数 | 本轮生成 token | 回填后仍在等待 | 回填后仍在 running |',
               '|---|---|---|---|---|']
        for p in plans:
            step=p['front_step'];output=select(rows,'update.before',step)['model_output']
            state=select(rows,'update.after',step)['scheduler']
            generated=dict(zip(output['req_ids'],output['sampled_token_ids']))
            table.append(f'| {step} | {pretty_ids(p["plan"]["num_scheduled_tokens"])} | {pretty_ids(generated)} | '
                +('、'.join(ids[k] for k in state['waiting']) or '无')+' | '
                +('、'.join(ids[k] for k in state['running']) or '无')+' |')
        key='table:'+name
        blocks[key]='\n'.join(table)+f'\n\n表据[原始运行记录](evidence/{run}/events.jsonl)生成；A/B/C 为内部 ID 的外部名称映射。空列表 `[]` 表示该请求本轮没有新输出 token。'
        index[key]={'run':run,'alias_map':ids,'plan_seqs':[r['seq'] for r in plans]}

    used=[]
    for doc in sorted(DRAFT.glob('ch*.md')):
        text=doc.read_text()
        def initial(match):
            key=match[1];used.append(key)
            return f'<!-- evidence:{key}:start -->\n{blocks[key]}\n<!-- evidence:{key}:end -->'
        def update(match):
            key=match[1];used.append(key)
            return f'<!-- evidence:{key}:start -->\n{blocks[key]}\n<!-- evidence:{key}:end -->'
        text=re.sub(r'<!-- evidence:([^\s]+):start -->.*?<!-- evidence:\1:end -->',update,text,flags=re.S)
        text=re.sub(r'\{\{((?:source|data|table):[^}]+)\}\}',initial,text)
        doc.write_text(text)
    assert set(used)==set(blocks),(set(used)^set(blocks))
    (RESEARCH/'evidence-index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
    for run in [base_name,chunk_name,'baseline-mp-traced','baseline-inproc-plain']:
        dest=DRAFT/'evidence'/run;dest.mkdir(parents=True,exist_ok=True)
        for name in ['events.jsonl','manifest.json','result.json']:
            shutil.copyfile(LOG/run/name,dest/name)
    for name in ['baseline-inproc-console.log','chunked-inproc-console.log','baseline-mp-console.log','baseline-plain-console.log']:
        shutil.copyfile(LOG/name,DRAFT/'evidence'/name)
    print(f'{len(SOURCES)} source excerpts; {len(blocks)-len(SOURCES)} observation blocks; 4 raw runs copied')


if __name__=='__main__':main()
