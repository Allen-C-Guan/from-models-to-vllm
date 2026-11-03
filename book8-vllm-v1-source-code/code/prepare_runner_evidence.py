"""保留 Runner 观测原件，生成可追溯投影并验证输入／块／输出一致性。"""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2]
LOG=ROOT/'log/book8-source-rewrite/runner-inproc-traced'
OUT=ROOT/'drafts/Book8-vLLM源码剖析/evidence/runner-inproc-traced'


def main():
    OUT.mkdir(exist_ok=True)
    for name in ['events.jsonl','manifest.json','result.json','runner-provenance.json']:
        target=OUT/name
        if target.exists():assert target.read_bytes()==(LOG/name).read_bytes()
        else:shutil.copyfile(LOG/name,target)
    raw=OUT/'events.jsonl';rows=[json.loads(s) for s in raw.read_text().splitlines()]
    provenance=json.loads((OUT/'runner-provenance.json').read_text())
    assert provenance['probe_sha256']==hashlib.sha256((ROOT/'code/Book8-vLLM源码剖析/observe_runner.py').read_bytes()).hexdigest()
    base=ROOT/'drafts/Book8-vLLM源码剖析/evidence/baseline-inproc-plain/result.json'
    assert json.loads((OUT/'result.json').read_text())['final']==json.loads(base.read_text())['final']
    result={'events_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'steps':{}}
    def get(event,step):return next(r for r in rows if r['event']==event and r['front_step']==step)
    for step in [1,2,3]:
        records={name:get(event,step) for name,event in [('registered','runner.registered'),('inputs','runner.inputs'),('attn','runner.attn'),('hidden','runner.hidden'),('logits','model.logits'),('execute','execute.call'),('returned','execute.return'),('sample','sample.return')]}
        b=records['inputs']['batch'];q=b['query_start_loc']['values'];counts=b['num_scheduled_tokens']['values']
        assert [y-x for x,y in zip(q,q[1:])]==counts and q[-1]==b['num_tokens']
        mapping=records['registered']['req_id_to_index']
        assert b['idx_mapping']['values']==[mapping[rid] for rid in b['req_ids']]
        blocks=records['attn']['block_tables'][0]['values'];positions=b['positions']['values']
        slots=records['attn']['slot_mappings']['values'][0]
        for idx in range(b['num_reqs']):
            assert slots[q[idx]:q[idx+1]]==[blocks[idx][p//128]*128+p%128 for p in positions[q[idx]:q[idx+1]]]
        result['steps'][str(step)]=records
    (OUT/'projection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Runner: 3 rounds, request-index mapping, ragged offsets, slot mapping and plain-run outputs verified')


if __name__=='__main__':main()
