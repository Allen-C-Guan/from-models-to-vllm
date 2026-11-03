"""执行固定源码n-gram纯函数体；无模型、无NPU、非MRV2 proposer。"""
import ast, hashlib, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=ROOT/'repos/vllm/vllm/v1/spec_decode/ngram_proposer.py'
    tree=ast.parse(p.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_find_longest_matched_ngram_and_propose_tokens')
    fn.decorator_list=[]
    ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns)
    rows=[]
    for history,limit in [([10,20,30,40,10,20],32),([10,20,30],32),([10,20,30,40,10,20],6)]:
        result=ns[fn.name](np.array(history,dtype=np.int32),2,2,limit,2)
        rows.append({'history':history,'max_model_len':limit,'min_ngram':2,'max_ngram':2,'k':2,'draft':result.tolist()})
    assert [r['draft'] for r in rows]==[[30,40],[],[]]
    out=ROOT/'drafts/Book8-vLLM源码剖析/evidence/ngram-contract';out.mkdir(exist_ok=True)
    (out/'trace.json').write_text(json.dumps({'kind':'CPU fixed-source function body via AST, decorator removed; synthetic IDs; MRV1 branch, no model execution','source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':rows},ensure_ascii=False,indent=2)+'\n')
    print('ngram: match, no match and max length verified on CPU')

if __name__=='__main__':main()
