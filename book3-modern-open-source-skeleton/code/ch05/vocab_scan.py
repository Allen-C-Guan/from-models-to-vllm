# Book3第5章：词表四臂默认预算保留，输入/输出默认工作目录log；--help不训练。
# 原bpb协议使用独立压缩样本，只作换算估计；同token数不等同算力或同原文。
# 数据准备核来源/模型hash，EOF明确失败；断点核完整身份，非有限梯度更新前拒绝，成功权重保留。
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import tempfile
import shutil
import multiprocessing
import sentencepiece as spm

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
for candidate in (HERE.parents[2]/'book2-pretraining-revolution/code/ch10',HERE.parents[1]/'Book2-预训练革命/ch10'):
    if candidate.is_dir():
        sys.path.insert(0,str(candidate));break
else:raise ImportError('找不到配套Book2第10章模型，请保留相邻书目录布局')
from model import GPT2,GPT2Config

SEED=20261002
OUT_DIR=str(Path.cwd()/'log/book3-ch05')
BOOK2_DIR=str(Path.cwd()/'log/book2-ch08')
TOK_TXT=str(Path.cwd()/'log/book2-scaling/owt_tokenizer_corpus.txt')
DOCS_JSONL=str(Path.cwd()/'log/book2-scaling/owt_docs.jsonl')
BATCH=16;BLOCK=512;TPB=BATCH*BLOCK;EVAL_BATCHES=49
PEAK_LR=1e-3;MIN_LR_RATIO=.1;WARMUP=100;WDECL=.1;TAIL=300
N_LAYER,N_EMBD,N_HEAD,N_POSITIONS=6,512,8,1024
ARMS={'v8192':8192,'v16384':16384,'v32768':32768,'v49152':49152}
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
    folder=Path(OUT_DIR);folder.mkdir(parents=True,exist_ok=True)
    train=folder/f'stream_v{V}_train.bin';ev=folder/f'stream_v{V}_eval.bin';meta_path=folder/f'slice_meta_v{V}.json'
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
        # 分块写临时文件，全部验证后才发布这次自建产物；不整流额外复制进内存。
        for name,lo,hi in [('train',0,n_train),('eval',n_train,total)]:
            with (tmp/name).open('xb') as f:
                for k in range(lo,hi,1_000_000):f.write(data[k:min(k+1_000_000,hi)].tobytes())
        del data
        os.replace(tmp/'train',train);os.replace(tmp/'eval',ev)
    meta=dict(identity=identity,V=V,tokenizer=str(sp_path),train_bin=str(train),eval_bin=str(ev),train_tokens=n_train,eval_tokens=EVAL_TOKENS,
              train_file=file_identity(train),eval_file=file_identity(ev),docs_encoded=docs,unk_tokens=unknown,unk_rate=unknown/total,
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


def lr_at(step,total):
    if WARMUP>0 and step<=WARMUP:return PEAK_LR*step/WARMUP
    progress=(step-WARMUP)/max(1,total-WARMUP)
    return PEAK_LR*(MIN_LR_RATIO+(1-MIN_LR_RATIO)*.5*(1+math.cos(math.pi*progress)))


class TrainStream:
    """第k步的输入/目标均为(BATCH,BLOCK)，目标右移一位；位置一次通过不证明文本去重。"""
    def __init__(self,path):
        self.path=str(path)
        if Path(path).stat().st_size%2:raise ValueError('uint16流字节数须为偶数')
        self.arr=np.memmap(path,dtype=np.uint16,mode='r');self.max_step=(len(self.arr)-1)//TPB
    def batch(self,step,device):
        if not 1<=step<=self.max_step:raise ValueError('训练步数越过流容量')
        start=(step-1)*TPB;span=np.asarray(self.arr[start:start+TPB+1],dtype=np.int64)
        return (torch.from_numpy(span[:-1].reshape(BATCH,BLOCK)).to(device),
                torch.from_numpy(span[1:].reshape(BATCH,BLOCK)).to(device))


def build_eval_batches(eval_bin,device):
    """全部合法起点中固定抽49批×16窗；不同池、种子或抽样版本不能称同窗。"""
    ev=np.memmap(eval_bin,dtype=np.uint16,mode='r')
    if len(ev)<BLOCK+1:raise ValueError('评估池不足一个输入/目标窗口')
    gen=torch.Generator().manual_seed(SEED)
    starts=torch.randint(0,len(ev)-BLOCK,(EVAL_BATCHES*BATCH,),generator=gen).tolist()
    x=torch.tensor(np.stack([ev[o:o+BLOCK] for o in starts]).astype(np.int64)).reshape(EVAL_BATCHES,BATCH,BLOCK)
    y=torch.tensor(np.stack([ev[o+1:o+BLOCK+1] for o in starts]).astype(np.int64)).reshape(EVAL_BATCHES,BATCH,BLOCK)
    return x.to(device),y.to(device)


@torch.no_grad()
def eval_loss(model,ex,ey,return_batches=False):
    """等长窗口均值及批间SE；相关/重叠窗不保证独立，单批不能估SE。退出恢复原模式。"""
    if ex.shape!=ey.shape or len(ex)==0:raise ValueError('评估输入/目标形状不符或无批')
    before=model.training;model.eval()
    try:losses=[float(model(ex[i],ey[i])[1]) for i in range(len(ex))]
    finally:model.train(before)
    if not all(math.isfinite(v) for v in losses):raise FloatingPointError('非有限评估损失')
    mean=sum(losses)/len(losses)
    se=float(np.std(losses,ddof=1)/math.sqrt(len(losses))) if len(losses)>1 else None
    return (mean,se,losses) if return_batches else (mean,se)


def build_arm(V):
    """固定种子、只改词表几何；不同V会改变RNG消耗，不承诺主体初值逐位相同。"""
    torch.manual_seed(SEED)
    model=GPT2(GPT2Config(vocab_size=V,n_positions=N_POSITIONS,n_layer=N_LAYER,n_embd=N_EMBD,n_head=N_HEAD))
    expected=V*N_EMBD+N_POSITIONS*N_EMBD+N_LAYER*(12*N_EMBD**2+13*N_EMBD)+2*N_EMBD
    assert model.n_params()==expected
    return model


def common_identity(steps,eval_every,device):
    return dict(schema=2,seed=SEED,steps=steps,eval_every=eval_every,device=str(device),torch=str(torch.__version__),sentencepiece=spm.__version__,
        model=[N_LAYER,N_EMBD,N_HEAD,N_POSITIONS],batch=BATCH,block=BLOCK,eval_batches=EVAL_BATCHES,
        learning_rate=[PEAK_LR,MIN_LR_RATIO,WARMUP],weight_decay=WDECL,sampler='all_legal_starts_v2',
        runner=file_identity(__file__),factory=file_identity(sys.modules[GPT2.__module__].__file__),docs=file_identity(DOCS_JSONL),
        tokenizer_corpus=file_identity(TOK_TXT) if Path(TOK_TXT).is_file() else None)


def run_identity(arm,V,steps,eval_every,device,stream,report):
    meta=report['meta'];ev=meta['slices'][arm]['eval_bin']
    return dict(**common_identity(steps,eval_every,device),arm=arm,V=V,train=file_identity(stream.path),eval=file_identity(ev),
                tokenizer=file_identity(meta['tokenizers'][arm]),compression=meta['compression'][arm])


def usable(record):
    return (record.get('completed') is True and not record.get('diverged') and
            record.get('steps_done')==record.get('steps_planned') and record.get('steps_done',0)>0
            and isinstance(record.get('eval_final'),(int,float)) and math.isfinite(record['eval_final']))


def run_arm(arm,V,steps,eval_every,device,stream,ex,ey,report,out_json):
    """有效身份下从eval边界恢复；CPU小模型续跑可测试，MPS不承诺逐位确定性。"""
    device=torch.device(device)
    if not 0<=WARMUP<steps or eval_every<=0 or steps>stream.max_step:raise ValueError('步数、warmup或评估间隔无效')
    identity=run_identity(arm,V,steps,eval_every,device,stream,report)
    name=report['meta']['out_name'];folder=Path(OUT_DIR);folder.mkdir(parents=True,exist_ok=True)
    ckpt=folder/f'ckpt_{arm}_{name}.pt';curve=folder/f'curve_{arm}_{name}.csv'
    if curve.exists() and not ckpt.exists():raise FileExistsError('同名曲线无可核断点，请改out-name')
    model=build_arm(V).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=PEAK_LR,betas=(.9,.95),weight_decay=WDECL)
    train_losses=[];lrs=[];times=[];evals=[];start=0;train_time=eval_time=0.;memory=[]
    if ckpt.exists():
        saved=torch.load(ckpt,map_location=device,weights_only=True)
        if saved.get('run_identity')!=identity:raise ValueError('断点配置/数据/实现身份不同，禁止混接')
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['opt'])
        torch.set_rng_state(saved['cpu_rng'].cpu())
        if device.type=='mps' and saved.get('mps_rng') is not None:torch.mps.set_rng_state(saved['mps_rng'].cpu())
        start=saved['step'];train_losses=saved['train_losses'];lrs=saved['lrs'];times=saved['times'];evals=saved['evals']
        train_time=saved['elapsed_train'];eval_time=saved['elapsed_eval'];memory=saved.get('memory_samples',[])
        if not 0<=start<=steps or len(train_losses)!=start or len(lrs)!=start or len(times)!=start:raise ValueError('断点步数与曲线不符')
    model.train();begin=time.perf_counter();failure=None
    for step in range(start+1,steps+1):
        lr=lr_at(step,steps)
        for group in opt.param_groups:group['lr']=lr
        x,y=stream.batch(step,device);t0=time.perf_counter()
        with torch.autocast('mps',dtype=torch.bfloat16) if device.type=='mps' else torch.autocast('cpu',enabled=False):
            _,loss=model(x,y)
        value=float(loss.detach())
        if not math.isfinite(value) or value>100:
            failure={'step':step,'reason':'非有限或超过本实验阈值的loss，未更新'};break
        opt.zero_grad(set_to_none=True);loss.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0))
        if not math.isfinite(norm):failure={'step':step,'reason':'非有限梯度，未更新'};break
        opt.step()
        if device.type=='mps':torch.mps.synchronize()
        elapsed=time.perf_counter()-t0;times.append(elapsed);train_time+=elapsed;train_losses.append(value);lrs.append(lr)
        if device.type=='mps':memory.append(torch.mps.current_allocated_memory()/2**20)
        if step==1 or step%100==0:print(f'[{arm}] {step}/{steps} loss={value:.4f} 累计训练秒/步={train_time/len(times):.4f}',flush=True)
        if step%eval_every==0 or step==steps:
            t0=time.perf_counter();el,se,batches=eval_loss(model,ex,ey,True)
            if device.type=='mps':torch.mps.synchronize()
            eval_time+=time.perf_counter()-t0
            evals.append(dict(step=step,eval_loss=round(el,5),eval_se=round(se,5) if se is not None else None,
                              batch_losses=batches,n_tokens=ey.numel(),se_definition='批均值的样本std/sqrt(批数)，不保证独立'))
            saved=dict(model=model.state_dict(),opt=opt.state_dict(),step=step,train_losses=train_losses,lrs=lrs,times=times,evals=evals,
                elapsed_train=train_time,elapsed_eval=eval_time,run_identity=identity,cpu_rng=torch.get_rng_state(),
                mps_rng=torch.mps.get_rng_state() if device.type=='mps' else None,memory_samples=memory)
            torch.save(saved,str(ckpt)+'.tmp');os.replace(str(ckpt)+'.tmp',ckpt)
    count=len(train_losses);steady=times[20:] if len(times)>20 else times;tail=train_losses[-TAIL:]
    completed=failure is None and count==steps and bool(evals) and evals[-1]['step']==steps
    record=dict(arm=arm,V=V,embedding_params=V*N_EMBD,embedding_share=V*N_EMBD/model.n_params(),run_identity=identity,params_total=model.n_params(),params_non_embedding=model.n_params(non_embedding=True),
        config=dict(d=N_EMBD,L=N_LAYER,h=N_HEAD,V=V,B=BATCH,T=BLOCK),steps_done=count,steps_planned=steps,
        resumed_from_step=start or None,completed=completed,diverged=failure is not None,failure=failure,
        tail300_train_loss=round(sum(tail)/len(tail),5) if tail else None,tail300_def=f'最后min({TAIL},已完成步数)步的均值',
        last_step_loss=train_losses[-1] if train_losses else None,min_train_loss=min(train_losses) if train_losses else None,
        evals=evals,eval_final=evals[-1]['eval_loss'] if completed else None,eval_final_se=evals[-1]['eval_se'] if completed else None,
        sec_per_step_steady=sum(steady)/len(steady) if steady else None,sec_step_std=float(np.std(steady)) if steady else None,
        sec_per_step_all=train_time/count if count else None,wallclock_sec=time.perf_counter()-begin,
        wallclock_scope='本次调用片段；累计训练与eval分别记录',elapsed_train=train_time,eval_overhead_sec=eval_time,
        max_current_mib_after_step=max(memory) if memory else None,memory_scope='更新后离散current读数，非训练峰值',
        precision='MPS bf16 autocast训练/fp32 eval' if device.type=='mps' else 'CPU fp32训练/eval',checkpoint=str(ckpt))
    with curve.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['step','train_loss','lr']);writer.writerows(zip(range(1,count+1),train_losses,lrs))
    report['arms'][arm]=record;save_json(out_json,report)
    print(f'[{arm} {"完成" if completed else "未完成"}] {count}/{steps}，eval={record["eval_final"]}',flush=True)
    del model,opt
    if device.type=='mps':torch.mps.empty_cache()
    return record


def add_tradeoff_summary(report):
    rows=[]
    for arm,record in report['arms'].items():
        if not usable(record):continue
        comp=report['meta']['compression'][arm];beta=comp['bytes_per_token']
        if not math.isfinite(beta) or beta<=0:raise ValueError('压缩系数无效')
        rows.append(dict(arm=arm,V=record['V'],embedding_share=record['embedding_share'],bytes_per_token=beta,
            tokens_per_word=comp['tokens_per_word'],eval_nats_per_token=record['eval_final'],
            eval_bpb=record['eval_final']/(math.log(2)*beta),sec_per_step=record['sec_per_step_steady'],
            bpb_scope='独立压缩样本系数的换算估计，不是同一评分文本的总NLL/字节'))
    report['tradeoff_table']=rows
    report['tradeoff_note']='同token预算不等同算力或同原文；bpb为估计，跨臂验证流与可见字节上下文不同；不作纯词表因果结论'


def main():
    global OUT_DIR,BOOK2_DIR,TOK_TXT,DOCS_JSONL,WARMUP,SEED
    ap=argparse.ArgumentParser(description='词表四臂扫描；明确数据身份，bpb仅样本压缩系数换算估计')
    ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--out-name',default='fast');ap.add_argument('--cells',default=','.join(ARMS))
    ap.add_argument('--eval-every',type=int,default=500);ap.add_argument('--warmup',type=int,default=WARMUP);ap.add_argument('--seed',type=int,default=SEED)
    ap.add_argument('--workers',type=int,default=8);ap.add_argument('--out-dir',type=Path,default=Path(OUT_DIR))
    ap.add_argument('--book2-dir',type=Path,default=Path(BOOK2_DIR));ap.add_argument('--tokenizer-corpus',type=Path,default=Path(TOK_TXT));ap.add_argument('--docs',type=Path,default=Path(DOCS_JSONL))
    ap.add_argument('--device',choices=['cpu','mps'],default='mps' if torch.backends.mps.is_available() else 'cpu');a=ap.parse_args()
    OUT_DIR=str(a.out_dir);BOOK2_DIR=str(a.book2_dir);TOK_TXT=str(a.tokenizer_corpus);DOCS_JSONL=str(a.docs);WARMUP=a.warmup;SEED=a.seed
    if not 0<=WARMUP<a.steps or a.eval_every<=0 or a.workers<=0:raise ValueError('日程、评估间隔与workers无效')
    if Path(a.out_name).name!=a.out_name or a.out_name in ('','.','..'):raise ValueError('out-name须单个名称')
    arms=a.cells.split(',')
    if not arms or len(set(arms))!=len(arms) or any(x not in ARMS for x in arms):raise ValueError('无效或重复实验臂')
    common=common_identity(a.steps,a.eval_every,a.device);path=a.out_dir/f'vocab_scan_{a.out_name}.json'
    if path.exists():
        report=json.loads(path.read_text())
        if report.get('meta',{}).get('run_identity')!=common:raise ValueError('报告配置/来源/实现身份不同，请使用新输出名或目录')
    else:report=dict(meta=dict(out_name=a.out_name,run_identity=common,slices={},compression={},tokenizers={}),arms={})
    a.out_dir.mkdir(parents=True,exist_ok=True)
    for arm in arms:
        V=ARMS[arm]
        if arm=='v8192':
            tokenizer=str(a.book2_dir/'sp8k_owt.model');tb=a.book2_dir/'tokens.bin';eb=a.book2_dir/'eval.bin'
            if any(p.stat().st_size%2 for p in (tb,eb)):raise ValueError('旧uint16流字节长度不正确')
            prepared=dict(train_bin=str(tb),eval_bin=str(eb),train_tokens=tb.stat().st_size//2,eval_tokens=eb.stat().st_size//2,
                          train_file=file_identity(tb),eval_file=file_identity(eb),protocol='读取现有Book2流，不沿用固定token计数常量')
        else:
            tokenizer=ensure_tokenizer(V);prepared=slice_stream(V,tokenizer,a.workers)
        sp=spm.SentencePieceProcessor(model_file=tokenizer)
        if sp.get_piece_size()!=V:raise ValueError('词表实物大小与模型不符')
        compression=measure_compression(tokenizer)
        for key,value in [('slices',prepared),('compression',compression),('tokenizers',tokenizer)]:
            previous=report['meta'][key].get(arm)
            if previous is not None and previous!=value:raise ValueError('同名实验臂的准备产物已变化')
            report['meta'][key][arm]=value
        stream=TrainStream(prepared['train_bin'])
        if a.steps>stream.max_step:raise ValueError('训练预算超出流容量')
        identity=run_identity(arm,V,a.steps,a.eval_every,a.device,stream,report)
        done=report['arms'].get(arm,{})
        if done and done.get('run_identity')!=identity:raise ValueError('实验臂身份变化')
        if usable(done):continue
        ex,ey=build_eval_batches(prepared['eval_bin'],a.device)
        result=run_arm(arm,V,a.steps,a.eval_every,a.device,stream,ex,ey,report,path)
        if not usable(result):raise RuntimeError('实验臂失败，保留有效断点，不汇总为成功结果')
    add_tradeoff_summary(report);save_json(path,report);print(path)


if __name__=='__main__':main()
