"""保存双NPU观测及失败历史；核对探针、分片与最终token。"""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2]


def keep(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():assert dst.read_bytes()==src.read_bytes()
    else:shutil.copyfile(src,dst)


def main():
    raw=ROOT/'log/book8-source-rewrite';dest=ROOT/'drafts/Book8-vLLM源码剖析/evidence';code=Path(__file__).parent
    for run in ['tp2-final','pp2']:
        for name in ['manifest.json','result.json']:keep(raw/run/name,dest/run/name)
        keep(raw/(run+'-console.log'),dest/(run+'-console.log'))
        m=json.loads((dest/run/'manifest.json').read_text());r=json.loads((dest/run/'result.json').read_text())
        assert m['probe_sha256']==hashlib.sha256((code/'observe_parallel.py').read_bytes()).hexdigest()
        assert m['model_probe_sha256']==hashlib.sha256((code/'parallel_probe.py').read_bytes()).hexdigest()
        assert [o['outputs'][0]['token_ids'] for o in r['outputs']]==[[12095,13],[279,16462,21575]]
        if run=='tp2-final':assert all(v['qkv_weight']==[2048,1024] and v['layer_range']==[0,28] for v in r['ranks'])
        else:assert [v['layer_range'] for v in r['ranks']]==[[0,14],[14,28]]
    for run in ['tp2','tp2-observed']:keep(raw/(run+'-console.log'),dest/'parallel-failed'/(run+'-console.log'))
    print('TP2 and PP2: probes, rank metadata and A/B outputs verified; failed attempts and shutdown warnings retained')


if __name__=='__main__':main()
