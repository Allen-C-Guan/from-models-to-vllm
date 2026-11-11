# prep_sp32k.py —— Book3第10章：sp32000与OWT重切；不覆盖缺身份的旧数据。
# 运行：source env.sh && python <本文件路径> --out-dir log/book3-ch10-new
# 默认沿用历史分块估算、文档后加EOS、尾部按token切eval；不保证文档互斥或内容去重。
# 分词器训练分片应由上游单独采集；本程序记录两个输入哈希，不能单凭文件名证明无重叠。
import argparse,hashlib,json,multiprocessing,os,shutil,tempfile,time
from pathlib import Path
import numpy as np
import sentencepiece as spm
V=32000
OUT_DIR=Path.cwd()/'log/book3-ch10'
TOK_TXT=Path.cwd()/'log/book2-scaling/owt_tokenizer_corpus.txt'
DOCS_JSONL=Path.cwd()/'log/book2-scaling/owt_docs.jsonl'
TRAIN_TARGET=42_000_000;EVAL_TOKENS=1_200_000;SAMPLE_DOCS=2000
_SP=None


def file_identity(path):
    h=hashlib.sha256();path=Path(path)
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return dict(bytes=path.stat().st_size,sha256=h.hexdigest())


def save_json(path,value):
    payload=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    temp=Path(str(path)+'.tmp');temp.write_text(payload);os.replace(temp,path)


def _init_worker(model_path,docs_path):
    global _SP,DOCS_JSONL
    DOCS_JSONL=docs_path;_SP=spm.SentencePieceProcessor(model_file=model_path)


def _encode_range(job):
    """按原文行区间编码，文档后加EOS；worker显式接收原文路径，不依赖fork继承。"""
    start,end,path=job;buf=[];tokens=unknown=docs=0
    with open(path,'xb') as output,open(DOCS_JSONL,encoding='utf-8') as source:
        for i,line in enumerate(source):
            if i<start:continue
            if i>=end:break
            if not line.strip():continue
            text=json.loads(line)['text'];ids=_SP.encode(text,out_type=int,add_eos=True)
            if ids and (min(ids)<0 or max(ids)>65535):raise ValueError('token ID不能以uint16保存')
            buf.extend(ids);tokens+=len(ids);unknown+=sum(v==_SP.unk_id() for v in ids);docs+=1
            if len(buf)>=4_000_000:output.write(np.asarray(buf,dtype=np.uint16).tobytes());buf=[]
        if buf:output.write(np.asarray(buf,dtype=np.uint16).tobytes())
    return str(path),tokens,unknown,docs


def ensure_tokenizer(V):
    if not isinstance(V,int) or not 1<=V<=65536:raise ValueError('uint16词表大小须在1—65536')
    folder=Path(OUT_DIR);folder.mkdir(parents=True,exist_ok=True);prefix=folder/f'sp{V}_owt'
    model=Path(str(prefix)+'.model');vocab=Path(str(prefix)+'.vocab');metadata=Path(str(prefix)+'.identity.json')
    identity=dict(V=V,corpus=file_identity(TOK_TXT),sentencepiece=spm.__version__,model_type='bpe',byte_fallback=True,character_coverage=1.)
    if model.exists() or vocab.exists() or metadata.exists():
        if not (model.exists() and vocab.exists() and metadata.exists()):raise FileExistsError('分词器产物不完整或缺身份，使用新out-dir')
        previous=json.loads(metadata.read_text())
        if previous['identity']!=identity or previous['model']!=file_identity(model):raise ValueError('已有分词器身份不匹配')
        return str(model)
    with tempfile.TemporaryDirectory(prefix=f'tokenizer-v{V}-',dir=folder) as tmp:
        temporary=Path(tmp)/'sp'
        spm.SentencePieceTrainer.Train(input=str(TOK_TXT),model_prefix=str(temporary),vocab_size=V,model_type='bpe',character_coverage=1.,byte_fallback=True)
        sp=spm.SentencePieceProcessor(model_file=str(temporary)+'.model')
        if sp.get_piece_size()!=V or sp.eos_id()<0:raise ValueError('词表大小或EOS不符')
        shutil.copyfile(str(temporary)+'.model',model);shutil.copyfile(str(temporary)+'.vocab',vocab)
    save_json(metadata,dict(identity=identity,model=file_identity(model)))
    return str(model)


def estimate_docs_needed(sp_path,target_tokens):
    sp=spm.SentencePieceProcessor(model_file=sp_path);tokens=docs=0
    with open(DOCS_JSONL,encoding='utf-8') as f:
        for i,line in enumerate(f):
            if i>=200:break
            if not line.strip():continue
            tokens+=len(sp.encode(json.loads(line)['text'],out_type=int,add_eos=True));docs+=1
    if not docs or not tokens:raise ValueError('前200行无可编码文本，不能估计流容量')
    return int(target_tokens/(tokens/docs)*1.2)+1000


def slice_stream(V,sp_path,workers):
    """保持历史“估计行块直至足额、尾部留eval”方案；EOF明确失败，不声称各臂原文对齐。"""
    if workers<=0 or TRAIN_TARGET<=0 or EVAL_TOKENS<=0:raise ValueError('workers和池目标须为正')
    if not 1<=V<=65536:raise ValueError('uint16词表超界')
    tokenizer=spm.SentencePieceProcessor(model_file=sp_path)
    if tokenizer.get_piece_size()!=V or tokenizer.eos_id()<0:raise ValueError('tokenizer词表大小/EOS与切流配置不符')
    folder=Path(OUT_DIR);folder.mkdir(parents=True,exist_ok=True)
    train=folder/'tokens32k.bin';ev=folder/'eval32k.bin';meta_path=folder/'slice_meta_32k.json'
    identity=dict(V=V,tokenizer=file_identity(sp_path),source=file_identity(DOCS_JSONL),train_target=TRAIN_TARGET,eval_tokens=EVAL_TOKENS,
                  workers=workers,slice_rule='ordered_estimated_line_chunks_tail_eval_v2',runner=file_identity(__file__))
    if any(p.exists() for p in (train,ev,meta_path)):
        if not all(p.exists() for p in (train,ev,meta_path)):raise FileExistsError('已有流不完整或缺身份，使用新out-dir')
        meta=json.loads(meta_path.read_text())
        if meta.get('identity')!=identity or meta.get('train_file')!=file_identity(train) or meta.get('eval_file')!=file_identity(ev):raise ValueError('切流缓存来源或内容不匹配')
        return meta
    target=TRAIN_TARGET+EVAL_TOKENS;start=0;end=estimate_docs_needed(sp_path,target);total=unknown=docs=0;parts=[];begin=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f'encode-v{V}-',dir=folder) as tmp:
        tmp=Path(tmp)
        while total<target:
            stride=max(1,(end-start+workers-1)//workers)
            jobs=[(i,min(i+stride,end),tmp/f'part-{i}.bin') for i in range(start,end,stride)]
            if workers==1:
                _init_worker(sp_path,str(DOCS_JSONL));rows=list(map(_encode_range,jobs))
            else:
                with multiprocessing.get_context('spawn').Pool(min(workers,len(jobs)),initializer=_init_worker,initargs=(sp_path,str(DOCS_JSONL))) as pool:rows=pool.map(_encode_range,jobs)
            gained=sum(v[1] for v in rows)
            if gained==0:raise ValueError(f'原文已耗尽，仅有{total}个token，未达到声明目标{target}')
            parts.extend(rows);total+=gained;unknown+=sum(v[2] for v in rows);docs+=sum(v[3] for v in rows)
            if total<target:start,end=end,end+estimate_docs_needed(sp_path,target-total)
        combined=tmp/'combined.bin'
        with combined.open('xb') as dest:
            for path,ntok,_,_ in parts:
                if Path(path).stat().st_size!=2*ntok:raise ValueError('part长度与记录不符')
                with open(path,'rb') as src:shutil.copyfileobj(src,dest,length=1024*1024)
        data=np.memmap(combined,dtype=np.uint16,mode='r')
        if len(data)!=total or int(data.max())>=V:raise ValueError('拼流计数或ID范围错误')
        n_train=total-EVAL_TOKENS
        sp=spm.SentencePieceProcessor(model_file=sp_path)
        eos=sp.eos_id()
        if eos<0 or int((data==eos).sum())!=docs:
            raise ValueError('EOS计数与文档数不符')
        first=[];seen=0
        with open(DOCS_JSONL,encoding='utf-8') as source:
            for line in source:
                if not line.strip():continue
                first.extend(sp.encode(json.loads(line)['text'],out_type=int,add_eos=True));seen+=1
                if seen==3:break
        if data[:len(first)].tolist()!=first:
            raise ValueError('流首部与前三篇有效文档的重新编码不符')
        # 分块写临时文件，全部验证后才发布这次自建产物；不整流额外复制进内存。
        for name,lo,hi in [('train',0,n_train),('eval',n_train,total)]:
            with (tmp/name).open('xb') as f:
                for k in range(lo,hi,1_000_000):f.write(data[k:min(k+1_000_000,hi)].tobytes())
        del data
        os.replace(tmp/'train',train);os.replace(tmp/'eval',ev)
    meta=dict(identity=identity,V=V,tokenizer=str(sp_path),train_bin=str(train),eval_bin=str(ev),train_tokens=n_train,eval_tokens=EVAL_TOKENS,
              train_file=file_identity(train),eval_file=file_identity(ev),docs_encoded=docs,unk_tokens=unknown,unk_rate=unknown/total,
              checks=['词表大小/EOS有效','ID范围','训练池足额','评估池精确长度','磁盘字节与token计数','EOS数等于文档数','首三篇有效文档回读'],
              wallclock_sec=time.perf_counter()-begin,protocol='原文行顺序编码并加EOS，尾部按token切分；不是文档边界切分，也不保证跨词表同评估文本')
    save_json(meta_path,meta);return meta


def measure_compression(sp_path):
    """首SAMPLE_DOCS行中的实际有效文档；raw UTF-8字节/含EOS token，不是eval实际B/T。"""
    sp=spm.SentencePieceProcessor(model_file=sp_path);tokens=byte_count=words=docs=0
    with open(DOCS_JSONL,encoding='utf-8') as f:
        for i,line in enumerate(f):
            if i>=SAMPLE_DOCS:break
            if not line.strip():continue
            text=json.loads(line)['text'];tokens+=len(sp.encode(text,out_type=int,add_eos=True))
            byte_count+=len(text.encode('utf-8'));words+=len(text.split());docs+=1
    if tokens<=0 or byte_count<=0:raise ValueError('压缩样本没有有效token或原始字节')
    return dict(sample_docs=docs,sample_line_limit=SAMPLE_DOCS,sample_bytes=byte_count,sample_tokens=tokens,
                bytes_per_token=byte_count/tokens,tokens_per_word=tokens/words if words else None,
                source=file_identity(DOCS_JSONL),tokenizer=file_identity(sp_path),add_eos=True,
                note='样本独立于评估池；原始字节与分词归一化后的内容可能不同，bpb换算仅为估计')


def main():
    global OUT_DIR,TOK_TXT,DOCS_JSONL,TRAIN_TARGET,EVAL_TOKENS
    ap=argparse.ArgumentParser(description='第10章独立分词器训练和带身份切流；缺原文即停止')
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--out-dir',type=Path,default=OUT_DIR)
    ap.add_argument('--tokenizer-corpus',type=Path,default=TOK_TXT)
    ap.add_argument('--docs',type=Path,default=DOCS_JSONL)
    ap.add_argument('--train-target',type=int,default=TRAIN_TARGET)
    ap.add_argument('--eval-tokens',type=int,default=EVAL_TOKENS)
    a=ap.parse_args()
    OUT_DIR=a.out_dir;TOK_TXT=a.tokenizer_corpus;DOCS_JSONL=a.docs
    TRAIN_TARGET=a.train_target;EVAL_TOKENS=a.eval_tokens
    if a.workers<1 or min(TRAIN_TARGET,EVAL_TOKENS)<1:raise ValueError('worker/目标池大小须正')
    if TOK_TXT.resolve()==DOCS_JSONL.resolve():raise ValueError('分词器语料不能直接使用同一LM主语料文件')
    begin=time.perf_counter()
    model=ensure_tokenizer(V)
    sliced=slice_stream(V,model,a.workers)
    compression=measure_compression(model)
    out=Path(OUT_DIR)/'prep_sp32k_report.json'
    content=dict(V=V,tokenizer=model,tokenizer_identity=file_identity(model),slice=sliced,compression=compression)
    if out.exists():
        old=json.loads(out.read_text())
        if any(old.get(k)!=v for k,v in content.items()):raise ValueError('已有总报告与已核数据不一致')
        print('已核对并复用：',out);return
    save_json(out,dict(content,elapsed_seconds=time.perf_counter()-begin))
    print('数据已保存：',out)


if __name__=='__main__':main()
