"""从同一次既有NPU记录抽取闭环；不启动模型。"""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    base=ROOT/'drafts/Book8-vLLM源码剖析/evidence/baseline-inproc-traced'
    ep=base/'events.jsonl';rp=base/'result.json';events=[json.loads(l) for l in ep.read_text().splitlines()];result=json.loads(rp.read_text());rows=[]
    for step in [1,2,3]:
        plan=next(e for e in events if e['event']=='schedule.after' and e['front_step']==step)
        sample=next(e for e in events if e['event']=='sample.return' and e['front_step']==step)
        assert plan['pid']==sample['pid']
        rows.append({'front_step':step,'plan_event_seq':plan['seq'],'sample_event_seq':sample['seq'],'pid':plan['pid'],'num_scheduled_tokens':plan['plan']['num_scheduled_tokens'],'finished_req_ids_in_plan':plan['plan']['finished_req_ids'],'sampled':dict(zip(sample['output']['req_ids'],sample['output']['sampled_token_ids']))})
    finals={r:{'token_ids':v['outputs'][0]['token_ids'],'text':v['outputs'][0]['text'],'finish_reason':v['outputs'][0]['finish_reason']} for r,v in result['final'].items()}
    data={'kind':'projection of one existing NPU baseline run; no re-execution','probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ep,rp]},'rounds':rows,'final':finals}
    out=ROOT/'drafts/Book8-vLLM源码剖析/evidence/closing';out.mkdir(exist_ok=True);(out/'trace.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');print('3 plan/sample pairs from same original run verified')

if __name__=='__main__':main()
