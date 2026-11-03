"""实际XGrammar/vLLM包装的CPU状态实验；小词表，不运行生成模型。"""
import hashlib, inspect, json
from concurrent.futures import Future
from pathlib import Path
from importlib.metadata import version
import xgrammar as xgr
from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.sampling_params import StructuredOutputsParams
ROOT=Path(__file__).resolve().parents[2]

def main():
    vocab=['P','a','r','i','s','T','o','k','y','<eos>']
    info=xgr.TokenizerInfo(vocab,vocab_type=xgr.VocabType.RAW,vocab_size=len(vocab),stop_token_ids=[9])
    ctx=xgr.GrammarCompiler(info).compile_grammar('root ::= "Paris" | "Tokyo"')
    gram=XgrammarGrammar(vocab_size=len(vocab),matcher=xgr.GrammarMatcher(ctx,max_rollback_tokens=8),ctx=ctx)
    def snapshot():
        if gram.is_terminated():
            return {'mask_int32':None,'allowed_ids':None,'processed':gram.num_processed_tokens,'terminated':True}
        mask=xgr.allocate_token_bitmask(1,len(vocab));gram.fill_bitmask(mask,0)
        packed=mask.tolist()[0]
        return {'mask_int32':packed,'allowed_ids':[i for i in range(len(vocab)) if (packed[i//32]>>(i%32))&1], 'processed':gram.num_processed_tokens,'terminated':gram.is_terminated()}
    before=snapshot();valid=gram.validate_tokens([0,1,2,3,4]);after_validate=snapshot()
    assert before==after_validate and valid==[0,1,2,3,4]
    assert gram.accept_tokens('A-grammar',[0]);after_p=snapshot()
    assert before['allowed_ids']==[0,5] and after_p['allowed_ids']==[1]
    assert gram.accept_tokens('A-grammar',[1,2,3,4]);after_word=snapshot()
    assert after_word['allowed_ids']==[9]
    assert gram.accept_tokens('A-grammar',[9]);after_eos=snapshot()
    req=StructuredOutputRequest(StructuredOutputsParams(grammar='root ::= "Paris"'))
    f=Future();req.grammar=f;pending={'ready':req.is_grammar_ready,'grammar_is_none':req.grammar is None};f.set_result(gram)
    ready={'ready':req.is_grammar_ready,'same_object':req.grammar is gram}
    bad=Future();req.grammar=bad;bad.set_exception(ValueError('constructed compile failure'))
    failed={'ready':req.is_grammar_ready,'error_type':type(req.grammar).__name__}
    files={}
    for cls,rel in [(XgrammarGrammar,'vllm/v1/structured_output/backend_xgrammar.py'),(StructuredOutputRequest,'vllm/v1/structured_output/request.py')]:
        installed=Path(inspect.getsourcefile(cls));fixed=ROOT/'repos/vllm'/rel
        assert installed.read_bytes()==fixed.read_bytes();files[rel]=hashlib.sha256(fixed.read_bytes()).hexdigest()
    out=ROOT/'drafts/Book8-vLLM源码剖析/evidence/grammar-cpu';out.mkdir(exist_ok=True)
    data={'kind':'CPU actual XGrammar and vLLM grammar wrapper; synthetic vocabulary; no model/GPU mask kernel', 'xgrammar_version':version('xgrammar'),'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':files,'vocab':vocab,'states':{'before':before,'validated_prefix':valid,'after_validate':after_validate,'after_P':after_p,'after_Paris':after_word,'after_eos':after_eos},'future':{'pending':pending,'ready':ready,'failed':failed}}
    (out/'trace.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');print(json.dumps(data['states'],indent=2))

if __name__=='__main__':main()
