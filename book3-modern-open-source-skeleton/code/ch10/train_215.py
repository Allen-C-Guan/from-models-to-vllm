# train_215.py —— Book3第10章整机冒烟、整合短训与全量入口。
# 运行：source env.sh && python <本文件路径> --task smoke --out-name new1
# 历史默认：B8/T1024、AdamW(.9,.95)、wd.1、clip1、峰值1e-3、warmup100。
# smoke20步全部处于warmup；integrated1000步才包含后续余弦下降。全量默认按4.3B目标向上取整。
# 只在显式执行时训练。数据/配置/日程/代码身份一致才续跑；完成权重保留，不自动删除或覆盖旧实验。
import argparse,csv,hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from llama215 import LLAMA_215M,Llama215,LLaMAConfig,hand_account

SEED=20261002
BATCH,BLOCK,EVAL_BATCHES=8,1024,49
PEAK_LR,MIN_LR_RATIO,WARMUP,WDECL=1e-3,.1,100,.1
OUT_DIR=Path.cwd()/'log/book3-ch10'
TOKENS_BIN=OUT_DIR/'tokens32k.bin';EVAL_BIN=OUT_DIR/'eval32k.bin'
TASKS={'smoke':dict(steps=20,eval_every=10),'integrated':dict(steps=1000,eval_every=100),
       'full':dict(steps=math.ceil(4_300_000_000/(BATCH*BLOCK)),eval_every=1000)}


def identity(path):
    h=hashlib.sha256();p=Path(path)
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return dict(bytes=p.stat().st_size,sha256=h.hexdigest())


def save_json(path,value):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)


def config_from_dict(data):
    data=dict(data);hd=data.pop('head_dim',None)
    if data.get('num_key_value_heads') == 0:raise ValueError('num_key_value_heads不能为0')
    cfg=LLaMAConfig(**data)
    if hd is not None and hd!=cfg.head_dim:raise ValueError('独立head_dim不受支持')
    return cfg


def token_array(path,vocab):
    p=Path(path)
    if p.stat().st_size==0 or p.stat().st_size%2:raise ValueError('token流须为非空uint16裸流')
    arr=np.memmap(p,dtype=np.uint16,mode='r')
    if int(arr.max())>=vocab:raise ValueError('token ID越过模型词表，须核对tokenizer身份')
    return arr


class TrainStream:
    """第k步输入从(k-1)*B*T起取B*T项；目标右移1，读取跨度B*T+1。"""
    def __init__(self,path,batch=BATCH,block=BLOCK,vocab=32000):
        if batch<=0 or block<=0:raise ValueError('batch/block须为正数')
        self.path=Path(path);self.batch_size=batch;self.block=block;self.tpb=batch*block
        self.arr=token_array(path,vocab);self.max_step=(len(self.arr)-1)//self.tpb
    def batch(self,step,device):
        if not 1<=step<=self.max_step:raise ValueError('训练步超出流容量')
        offset=(step-1)*self.tpb
        span=np.asarray(self.arr[offset:offset+self.tpb+1],dtype=np.int64)
        return (torch.from_numpy(span[:-1].reshape(self.batch_size,self.block)).to(device),
                torch.from_numpy(span[1:].reshape(self.batch_size,self.block)).to(device))


def corpus_stats(arr,n_tokens,vocab=32000):
    """指定前缀的经验一元熵；不是上下文模型损失下界，也不自动成为验证交叉熵。"""
    if not 0<n_tokens<=len(arr):raise ValueError('统计前缀超出数据')
    counts=np.bincount(np.asarray(arr[:n_tokens]),minlength=vocab)
    prob=counts[counts>0]/n_tokens
    return dict(tokens=n_tokens,active_types=int((counts>0).sum()),unigram_entropy_nat=float(-(prob*np.log(prob)).sum()))


def build_eval_batches(path,device,batch=BATCH,block=BLOCK,n_batches=EVAL_BATCHES,vocab=32000):
    ev=token_array(path,vocab)
    if len(ev)<block+1 or n_batches<1:raise ValueError('评估数据或批数不足')
    gen=torch.Generator().manual_seed(SEED)
    # 包括末个合法起点；旧抽样漏掉该点，新旧分数须分别记录。
    starts=torch.randint(0,len(ev)-block,(n_batches*batch,),generator=gen).tolist()
    x=torch.tensor(np.stack([ev[o:o+block] for o in starts]).astype(np.int64)).reshape(n_batches,batch,block)
    y=torch.tensor(np.stack([ev[o+1:o+block+1] for o in starts]).astype(np.int64)).reshape(n_batches,batch,block)
    return x.to(device),y.to(device)


@torch.no_grad()
def eval_loss(model,ex,ey):
    before=model.training;model.eval()
    try:
        losses=[float(model(ex[i],ey[i])[1]) for i in range(len(ex))]
        if not losses or not all(math.isfinite(v) for v in losses):raise FloatingPointError('评估损失非有限')
        return sum(losses)/len(losses)
    finally:model.train(before)


def lr_at(step,total,warmup=WARMUP):
    if step<=warmup:return PEAK_LR*step/warmup
    p=(step-warmup)/max(1,total-warmup)
    return PEAK_LR*(MIN_LR_RATIO+(1-MIN_LR_RATIO)*.5*(1+math.cos(math.pi*p)))


def sync(device):
    if device.type=='mps':torch.mps.synchronize()
    elif device.type=='cuda':torch.cuda.synchronize(device)


def rng_state(device):
    return dict(cpu=torch.get_rng_state(),mps=torch.mps.get_rng_state() if device.type=='mps' else None,
                cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)


def restore_rng(state,device):
    torch.set_rng_state(state['cpu'].cpu())
    if device.type=='mps':torch.mps.set_rng_state(state['mps'].cpu())
    if device.type=='cuda':torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def write_curve(path,rows):
    temp=Path(str(path)+'.tmp')
    with temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['step','train_loss','eval_loss','lr','sec_step'])
        writer.writeheader();writer.writerows(rows)
    os.replace(temp,path)


def run(task,out_name,steps=None,device_pref='cpu',*,data_dir=None,out_dir=None,config=None,
        batch=BATCH,block=BLOCK,eval_batches=EVAL_BATCHES,eval_every=None,warmup=WARMUP,stop_after=None):
    """stop_after只供短测试/主动分段，在eval边界返回；总日程steps不变。"""
    if task not in TASKS:raise ValueError('未知task')
    if min(batch,block)<=0:raise ValueError('batch/block须正')
    steps=(math.ceil(4_300_000_000/(batch*block)) if task=='full' else TASKS[task]['steps']) if steps is None else steps
    every=TASKS[task]['eval_every'] if eval_every is None else eval_every
    if min(steps,batch,block,eval_batches,every)<=0 or warmup<0:raise ValueError('步数/尺寸/间隔需正，warmup非负')
    if stop_after is not None and (not 0<stop_after<=steps or (stop_after%every and stop_after!=steps)):
        raise ValueError('stop_after须在总日程的eval边界')
    if Path(out_name).name!=out_name or out_name in ('','.','..'):raise ValueError('out-name须为单名')
    device=torch.device(device_pref)
    if device.type not in ('cpu','mps','cuda'):raise ValueError('本脚本只支持cpu/mps/cuda')
    if device.type=='mps' and not torch.backends.mps.is_available():raise ValueError('MPS不可用，请显式选择cpu')
    if device.type=='cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError('需要可用且支持BF16的CUDA设备')
    c=dict(LLAMA_215M if config is None else config);cfg=config_from_dict(c)
    detail,hand=hand_account(cfg)
    if config is None and hand!=207119360:raise AssertionError('默认207M参数账异常')
    data=Path(data_dir or OUT_DIR);folder=Path(out_dir or OUT_DIR)
    train_path=data/'tokens32k.bin';eval_path=data/'eval32k.bin'
    stream=TrainStream(train_path,batch,block,cfg.vocab_size)
    if steps>stream.max_step:raise ValueError('训练池不足，禁止用重复遍历冒充位置一次通过')
    ex,ey=build_eval_batches(eval_path,device,batch,block,eval_batches,cfg.vocab_size)
    bookcode=Path(__file__).resolve().parent.parent
    sources=['ch10/train_215.py','ch10/llama215.py','ch02/rmsnorm.py','ch03/swiglu.py',
             'ch04/rope.py','ch06/gqa.py','00-feasibility/llama_slots.py']
    run_id=dict(schema=2,task=task,steps=steps,config=vars(cfg),batch=batch,block=block,seed=SEED,
                eval_batches=eval_batches,eval_every=every,warmup=warmup,lr=[PEAK_LR,MIN_LR_RATIO],
                wd=WDECL,betas=[.9,.95],clip=1.,torch=str(torch.__version__),device=str(device),
                train=identity(train_path),eval=identity(eval_path),sampler='all_legal_starts_v2',
                tokenizer=identity(data/'sp32000_owt.model') if (data/'sp32000_owt.model').exists() else None,
                code={f:identity(bookcode/f) for f in sources})
    stem=f'{task}_{out_name}';ckpt=folder/f'ckpt_{stem}.pt';report_path=folder/f'train215_{stem}.json';curve_path=folder/f'curve_{stem}.csv'
    if not ckpt.exists() and (report_path.exists() or curve_path.exists()):
        raise FileExistsError('同名产物缺可核断点，请使用新out-name')
    folder.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED);model=Llama215(cfg).to(device)
    if model.n_params()!=hand:raise AssertionError('手算与整机参数不符')
    opt=torch.optim.AdamW(model.parameters(),lr=PEAK_LR,betas=(.9,.95),weight_decay=WDECL)
    rows=[];health=None;start=0;train_time=0.;eval_time=0.
    if ckpt.exists():
        saved=torch.load(ckpt,map_location=device,weights_only=True)
        if saved.get('run_identity')!=run_id:raise ValueError('断点配置/日程/数据/实现身份不同；禁止混接')
        start=saved['step'];rows=saved['curve'];health=saved['health']
        if not 0<=start<=steps or [r['step'] for r in rows]!=list(range(1,start+1)):raise ValueError('断点曲线与步号不符')
        model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['opt'])
        restore_rng(saved['rng'],device);train_time=saved['train_seconds'];eval_time=saved['eval_seconds']
        write_curve(curve_path,rows)  # 从有效断点重建对应曲线，也适用于CSV缺失。
    began=time.perf_counter();failure=None;model.train();limit=stop_after or steps
    for step in range(start+1,limit+1):
        lr=lr_at(step,steps,warmup)
        for group in opt.param_groups:group['lr']=lr
        x,y=stream.batch(step,device);sync(device);tick=time.perf_counter()
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type!='cpu'):
            logits,loss=model(x,y)
        value=float(loss.detach())
        if not math.isfinite(value):failure=dict(step=step,reason='非有限loss，更新前停止');break
        if step==1:
            with torch.no_grad():
                variance=float(logits.float().var(unbiased=False))
                target_mean=float(logits.gather(-1,y.unsqueeze(-1)).float().mean())
            prediction=math.log(cfg.vocab_size)+cfg.hidden_size*cfg.initializer_range**2/2
            health=dict(init_loss=value,logit_var=variance,target_logit_mean=target_mean,
                        variance_prediction=prediction,within_diagnostic_band=abs(value-prediction)<.05)
        opt.zero_grad(set_to_none=True);loss.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        if not math.isfinite(norm):failure=dict(step=step,reason='非有限梯度，更新前停止');break
        opt.step();sync(device);seconds=time.perf_counter()-tick;train_time+=seconds
        rows.append(dict(step=step,train_loss=value,lr=lr,sec_step=seconds))
        if step%every==0 or step==steps:
            tick=time.perf_counter()
            try:ev=eval_loss(model,ex,ey)
            except FloatingPointError:
                failure=dict(step=step,reason='非有限eval，保留此前有效断点');break
            sync(device);eval_time+=time.perf_counter()-tick;rows[-1]['eval_loss']=ev
            saved=dict(model=model.state_dict(),opt=opt.state_dict(),step=step,config=vars(cfg),
                       run_identity=run_id,curve=rows,health=health,rng=rng_state(device),
                       train_seconds=train_time,eval_seconds=eval_time)
            torch.save(saved,str(ckpt)+'.tmp');os.replace(str(ckpt)+'.tmp',ckpt)
            write_curve(curve_path,rows)
            print(f'{task} {step}/{steps}: train={value:.4f}, eval={ev:.4f}',flush=True)
    completed=failure is None and len(rows)==steps and 'eval_loss' in rows[-1]
    done=len(rows);final_eval=rows[-1].get('eval_loss') if completed else None
    observations=[r['eval_loss'] for r in rows if 'eval_loss' in r]
    checks=dict(loss_finite=all(math.isfinite(r['train_loss']) for r in rows),
                initial_scale=health['within_diagnostic_band'] if health else None,
                train_last_below_first=(rows[-1]['train_loss']<rows[0]['train_loss']) if len(rows)>1 else None,
                eval_last_below_first=(observations[-1]<observations[0]) if len(observations)>1 else None)
    record=dict(task=task,out_name=out_name,run_identity=run_id,params=hand,config=vars(cfg),account=detail,checks=checks,
                completed=completed,failure=failure,steps_done=done,steps_planned=steps,targets=done*stream.tpb,
                input_stream_tokens=done*stream.tpb+1 if done else 0,health=health,
                first_train_loss=rows[0]['train_loss'] if rows else None,last_train_loss=rows[-1]['train_loss'] if rows else None,
                final_eval=final_eval,resumed_from_step=start,checkpoint=str(ckpt),
                train_seconds=train_time,eval_seconds=eval_time,invocation_seconds=time.perf_counter()-began,
                precision='CPU fp32 train/eval' if device.type=='cpu' else f'{device.type} BF16 autocast train / fp32 eval',
                corpus_stats=corpus_stats(stream.arr,done*stream.tpb+1,cfg.vocab_size) if done else None,
                sampling_note='新抽样含末合法起点；旧MPS分数不作为当前运行预期',
                continuation_note='累计计时只含已保存轨迹；中断后丢弃重算的工作须另记进程记录')
    save_json(report_path,record)
    if failure:raise FloatingPointError(str(failure))
    return record


if __name__=='__main__':
    ap=argparse.ArgumentParser(description='207M整机训练；默认设置保留历史配方，显式新实验名与路径')
    ap.add_argument('--task',choices=TASKS,default='smoke');ap.add_argument('--out-name',required=True)
    ap.add_argument('--steps',type=int);ap.add_argument('--device',default='mps' if torch.backends.mps.is_available() else 'cpu')
    ap.add_argument('--data-dir',type=Path,default=OUT_DIR);ap.add_argument('--out-dir',type=Path,default=OUT_DIR)
    ap.add_argument('--config',type=Path,help='可选小模型配置JSON；报告明确记录实际配置')
    ap.add_argument('--batch',type=int,default=BATCH);ap.add_argument('--block',type=int,default=BLOCK)
    ap.add_argument('--eval-batches',type=int,default=EVAL_BATCHES);ap.add_argument('--eval-every',type=int)
    ap.add_argument('--warmup',type=int,default=WARMUP);a=ap.parse_args()
    c=json.loads(a.config.read_text()) if a.config else None
    run(a.task,a.out_name,a.steps,a.device,data_dir=a.data_dir,out_dir=a.out_dir,config=c,
        batch=a.batch,block=a.block,eval_batches=a.eval_batches,eval_every=a.eval_every,warmup=a.warmup)
