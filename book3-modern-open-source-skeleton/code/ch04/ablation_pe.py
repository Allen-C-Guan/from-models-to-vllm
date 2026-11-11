# Book3第4章：位置表与RoPE独立模块实验，保持历史默认训练几何、日程与长窗评价构造。
# 读取当前工作目录log；同名须数据/配置/日程/代码身份一致；完成权重保留，失败臂不比较。
# --help不训练。训练与长窗读数必须分开；同窗随机采样、长窗取池首部，不能单独分离长度因果。
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import types
import torch.nn.functional as F

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'00-feasibility'))
for candidate in (HERE.parents[2]/'book2-pretraining-revolution/code/ch10',HERE.parents[1]/'Book2-预训练革命/ch10'):
    if candidate.is_dir():
        sys.path.insert(0,str(candidate));break
else:raise ImportError('找不到配套Book2第10章模型，请保留相邻书目录布局')
from model import GPT2,GPT2Config
from llama_slots import apply_rope,build_rope_cache

SEED=20261002
OUT_DIR=str(Path.cwd()/'log/book3-ch04')
TOKENS_BIN=str(Path.cwd()/'log/book2-ch08/tokens.bin')
EVAL_BIN=str(Path.cwd()/'log/book2-ch08/eval.bin')
VOCAB=8192;BATCH=16;BLOCK=512;TPB=BATCH*BLOCK;EVAL_BATCHES=49
PEAK_LR=1e-3;MIN_LR_RATIO=.1;WARMUP=100;WDECL=.1;TAIL=300
N_LAYER,N_EMBD,N_HEAD,N_POSITIONS=6,512,8,1024
ROPE_THETA=10000.;EXTRAP_N=2048;EXTRAP_WINDOWS=192;EXTRAP_BATCH=4
EXTRAP_BUCKETS=[0,512,1024,1536,2048]
_MASK_CACHE={}


def causal_mask(n,device):
    """保持因果规则，按当前长度缓存(1,1,n,n)布尔掩码；长度越大平方成本越高。"""
    key=(n,str(device))
    if key not in _MASK_CACHE:_MASK_CACHE[key]=torch.ones(n,n,device=device,dtype=torch.bool).tril().view(1,1,n,n)
    return _MASK_CACHE[key]


def file_identity(path):
    h=hashlib.sha256();path=Path(path)
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return dict(bytes=path.stat().st_size,sha256=h.hexdigest())


def save_json(path,value):
    payload=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    temp=Path(str(path)+'.tmp');temp.write_text(payload);os.replace(temp,path)


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


def build_eval_batches(device):
    """全部合法起点中固定抽49批×16窗；不同池、种子或抽样版本不能称同窗。"""
    ev=np.memmap(EVAL_BIN,dtype=np.uint16,mode='r')
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


def _rope_attn_forward(self, x):
    """rope 臂的注意力前向（换装件）：与底座 CausalSelfAttention 同一零件（c_attn/c_proj 权重不变），
    仅在 Q/K 投影后施加 RoPE 旋转。输入 (B,n,C) -> 输出 (B,n,C)。v 不旋（式 11 只约束 QK 内积）。

    cos/sin 由 llama_slots.build_rope_cache 按 n 现算：(1,n,hd)——无学习式表长限制，实际仍受内存和精度约束。
    """
    B, n, C = x.shape                                              # (B,n,C)
    hd = C // self.n_head                                          # 64
    q, k, v = self.c_attn(x).split(C, dim=2)                       # (B,n,3C) -> 各 (B,n,C)
    q = q.view(B, n, self.n_head, hd).transpose(1, 2)              # (B,h,n,hd)
    k = k.view(B, n, self.n_head, hd).transpose(1, 2)              # (B,h,n,hd)
    v = v.view(B, n, self.n_head, hd).transpose(1, 2)              # (B,h,n,hd)
    cos, sin = build_rope_cache(n, hd, ROPE_THETA, x.device)       # (1,n,hd) x2（llama_slots 正身）
    q, k = apply_rope(q, k, cos, sin)                              # 只旋 q/k -> (B,h,n,hd)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)                # (B,h,n,hd)·(B,h,hd,n) -> (B,h,n,n)
    att = att.masked_fill(causal_mask(n, x.device)[:, :, :n, :n] == 0, float("-inf"))
    att = self.attn_drop(F.softmax(att, dim=-1))                   # 行和=1（pdrop=0）
    y = att @ v                                                    # (B,h,n,n)·(B,h,n,hd) -> (B,h,n,hd)
    y = y.transpose(1, 2).contiguous().view(B, n, C)               # (B,n,h*hd)=(B,n,C)
    return self.resid_drop(self.c_proj(y))                         # (B,n,C)


class GPT2RoPE(GPT2):
    """rope 臂整机：底座构造后删 wpe、六处注意力换 RoPE 前向（改装第三例的实验落地）。

    构造 RNG 与 wpe 臂完全一致（super().__init__ 先建全套含 wpe，再删），因此 wte/blocks/ln_f 初始
    权重逐位相同——改动范围为位置编码件与动态掩码存储（wpe 表 524,288 参数 vs RoPE 零参数，如实标注）。
    """

    def __init__(self, cfg: GPT2Config):
        super().__init__(cfg)
        self.wpe = None                                            # 删 wpe：参数与查表一并消失
        for blk in self.blocks:                                    # 6 处注意力换 RoPE 前向（权重不动）
            blk.attn.forward = types.MethodType(_rope_attn_forward, blk.attn)

    def forward(self, idx, targets=None):
        B, n = idx.shape                                           # (B,n)，无学习式位置表上限
        if n<=0:raise ValueError('序列不能为空')
        if targets is not None and targets.shape!=idx.shape:raise ValueError('targets须与idx同形')
        x = self.drop(self.wte(idx))                               # (B,n,C)（只加词嵌入，位置进 attention 旋）
        for blk in self.blocks:                                    # 6 × (B,n,C)
            x = blk(x)
        x = self.ln_f(x)                                           # (B,n,C)
        logits = self.head(x)                                      # (B,n,C)·(C,V) -> (B,n,V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                  targets.reshape(-1))
        return logits, loss

    def n_params(self, non_embedding=False):
        """参数账：rope 臂无 wpe，non_embedding 只扣 wte（tied head 同存储不计）。"""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel()
        return n


def build_arm(arm):
    """构造两臂：同种子先建底座（RNG 消耗序列一致），rope 臂再换装位置编码件。"""
    torch.manual_seed(SEED)
    cfg = GPT2Config(vocab_size=VOCAB, n_positions=N_POSITIONS,
                     n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD)
    if arm == "wpe":
        return GPT2(cfg)                                           # 底座原样（学习式位置查表）
    if arm == "rope":
        return GPT2RoPE(cfg)                                       # 删 wpe + Q/K 旋转
    raise ValueError(arm)


def run_identity(steps,eval_every,device,stream):
    return dict(schema=2,seed=SEED,steps=steps,eval_every=eval_every,device=str(device),torch=str(torch.__version__),
        model=[VOCAB,N_LAYER,N_EMBD,N_HEAD,N_POSITIONS],batch=BATCH,block=BLOCK,eval_batches=EVAL_BATCHES,
        rope_theta=ROPE_THETA,extrap=[EXTRAP_N,EXTRAP_WINDOWS,EXTRAP_BATCH,EXTRAP_BUCKETS],
        learning_rate=[PEAK_LR,MIN_LR_RATIO,WARMUP],weight_decay=WDECL,
        train=file_identity(stream.path),eval=file_identity(EVAL_BIN),sampler='all_legal_starts_v2',
        runner=file_identity(__file__),factory=file_identity(sys.modules[GPT2.__module__].__file__),
        position=file_identity(sys.modules[apply_rope.__module__].__file__))


def validate_extrap_pool():
    if EXTRAP_N<=0 or EXTRAP_WINDOWS<=0 or EXTRAP_BATCH<=0:raise ValueError('外推窗口/批数须为正')
    if EXTRAP_BUCKETS[0]!=0 or EXTRAP_BUCKETS[-1]!=EXTRAP_N or any(a>=b for a,b in zip(EXTRAP_BUCKETS,EXTRAP_BUCKETS[1:])):
        raise ValueError('分桶须严格递增且覆盖完整外推窗')
    if Path(EVAL_BIN).stat().st_size//2<EXTRAP_WINDOWS*EXTRAP_N+1:raise ValueError('外推池不足所声明的输入/目标位置数')


@torch.no_grad()
def eval_extrap(model,device):
    """连续输入窗不重叠；目标右移一位。不同于随机同窗评估，不能单独分离长度因果效应。"""
    validate_extrap_pool();before=model.training;model.eval()
    ev=np.memmap(EVAL_BIN,dtype=np.uint16,mode='r');loss_sum=0.;count=0
    bsum=[0.]*(len(EXTRAP_BUCKETS)-1);bcnt=[0]*len(bsum)
    try:
        for start in range(0,EXTRAP_WINDOWS,EXTRAP_BATCH):
            nb=min(EXTRAP_BATCH,EXTRAP_WINDOWS-start)
            xs=[ev[i*EXTRAP_N:(i+1)*EXTRAP_N] for i in range(start,start+nb)]
            ys=[ev[i*EXTRAP_N+1:(i+1)*EXTRAP_N+1] for i in range(start,start+nb)]
            x=torch.tensor(np.stack(xs).astype(np.int64),device=device);y=torch.tensor(np.stack(ys).astype(np.int64),device=device)
            logits,_=model(x,y)                  # (nb,n,V)，以逐目标CE汇总
            losses=F.cross_entropy(logits.reshape(-1,logits.shape[-1]).float(),y.reshape(-1),reduction='none').view_as(y)
            if not torch.isfinite(losses).all():raise FloatingPointError('外推出现非有限损失')
            loss_sum+=float(losses.sum());count+=y.numel()
            for j,(lo,hi) in enumerate(zip(EXTRAP_BUCKETS,EXTRAP_BUCKETS[1:])):
                bsum[j]+=float(losses[:,lo:hi].sum());bcnt[j]+=nb*(hi-lo)
    finally:model.train(before)
    avg=loss_sum/count
    if avg>math.log(sys.float_info.max):raise FloatingPointError('ppl超出浮点范围')
    buckets={f'pos[{a}:{b}]':bsum[j]/bcnt[j] for j,(a,b) in enumerate(zip(EXTRAP_BUCKETS,EXTRAP_BUCKETS[1:]))}
    assert count==EXTRAP_WINDOWS*EXTRAP_N and sum(bcnt)==count
    return avg,math.exp(avg),buckets,count


def probe_wpe_wall(model,device):
    """只把预期位置范围错误记为边界证据；其他异常继续抛出。"""
    if EXTRAP_N<=model.block_size:raise ValueError('边界探针长度须大于位置表')
    rec=dict(n_positions=model.block_size,probe_len=EXTRAP_N,lookup_device='cpu')
    x=torch.zeros(1,EXTRAP_N,dtype=torch.long,device=device)
    try:
        with torch.no_grad():model(x)
    except (ValueError,AssertionError) as err:
        message=str(err)
        if '序列长' not in message or str(EXTRAP_N) not in message:raise
        rec['forward_error']=dict(type=type(err).__name__,message=message)
    else:raise AssertionError('未触发预期模型位置边界')
    try:F.embedding(torch.arange(EXTRAP_N),model.wpe.weight.detach().cpu())
    except IndexError as err:rec['wpe_lookup_error']=dict(type='IndexError',message=str(err))
    else:raise AssertionError('裸位置表没有按预期拒绝越界')
    rec['verified']=True;rec['note']='当前模型前向错误与CPU裸查表错误；异常名称可随底座实现变化，不外推MPS行为'
    return rec


def usable(record):
    return (record.get('completed') is True and not record.get('diverged') and
            record.get('steps_done')==record.get('steps_planned') and record.get('steps_done',0)>0
            and isinstance(record.get('eval_final'),(int,float)) and math.isfinite(record['eval_final']))


def run_arm(arm,steps,eval_every,device,stream,ex,ey,report,out_json):
    """有效身份下从eval边界恢复；CPU小模型续跑可测试，MPS不承诺逐位确定性。"""
    device=torch.device(device)
    if not 0<=WARMUP<steps or eval_every<=0 or steps>stream.max_step:raise ValueError('步数、warmup或评估间隔无效')
    identity={**run_identity(steps,eval_every,device,stream),'arm':arm}
    name=report['meta']['out_name'];folder=Path(OUT_DIR);folder.mkdir(parents=True,exist_ok=True)
    ckpt=folder/f'ckpt_{arm}_{name}.pt';curve=folder/f'curve_{arm}_{name}.csv'
    if curve.exists() and not ckpt.exists():raise FileExistsError('同名曲线无可核断点，请改out-name')
    model=build_arm(arm).to(device)
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
    record=dict(arm=arm,run_identity=identity,params_total=model.n_params(),params_non_embedding=model.n_params(non_embedding=True),
        config=dict(d=N_EMBD,L=N_LAYER,h=N_HEAD,V=VOCAB,B=BATCH,T=BLOCK,n_positions=N_POSITIONS,rope_theta=ROPE_THETA),
        pos_params=N_POSITIONS*N_EMBD if arm=='wpe' else 0,steps_done=count,steps_planned=steps,
        resumed_from_step=start or None,completed=completed,diverged=failure is not None,failure=failure,
        tail300_train_loss=round(sum(tail)/len(tail),5) if tail else None,tail300_def=f'最后min({TAIL},已完成步数)步的均值',
        last_step_loss=train_losses[-1] if train_losses else None,min_train_loss=min(train_losses) if train_losses else None,
        evals=evals,eval_final=evals[-1]['eval_loss'] if completed else None,eval_final_se=evals[-1]['eval_se'] if completed else None,
        sec_per_step_steady=sum(steady)/len(steady) if steady else None,sec_step_std=float(np.std(steady)) if steady else None,
        sec_per_step_all=train_time/count if count else None,wallclock_sec=time.perf_counter()-begin,
        wallclock_scope='本次训练/同窗评估片段；不含另列长窗评估与边界探针，累计训练与eval分别记录',elapsed_train=train_time,eval_overhead_sec=eval_time,
        max_current_mib_after_step=max(memory) if memory else None,memory_scope='更新后离散current读数，非训练峰值',
        precision='MPS bf16 autocast训练/fp32 eval' if device.type=='mps' else 'CPU fp32训练/eval',checkpoint=str(ckpt))
    if completed:
        try:
            if arm=='rope':
                t0=time.perf_counter();xl,xp,buckets,ntok=eval_extrap(model,device)
                record['extrap_2048']=dict(loss=xl,ppl=xp,buckets=buckets,n_tokens=ntok,windows=EXTRAP_WINDOWS,
                    context_length=EXTRAP_N,wall_sec=time.perf_counter()-t0,
                    note='字段名保留历史2048口径；实际长度看context_length；与同窗评估不是同一窗集')
            else:record['wall_2048']=probe_wpe_wall(model,device)
        except Exception as err:
            record['completed']=False;record['post_eval_error']=f'{type(err).__name__}: {err}'
            report['arms'][arm]=record;save_json(out_json,report);raise
    with curve.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['step','train_loss','lr']);writer.writerows(zip(range(1,count+1),train_losses,lrs))
    report['arms'][arm]=record;save_json(out_json,report)
    print(f'[{arm} {"完成" if completed else "未完成"}] {count}/{steps}，eval={record["eval_final"]}',flush=True)
    del model,opt
    if device.type=='mps':torch.mps.empty_cache()
    return record


def add_pair_summary(report):
    a,b=report['arms'].get('wpe'),report['arms'].get('rope')
    if not a or not b or not usable(a) or not usable(b):
        report['pair_summary']={'status':'双臂及各自终局评估完整后才比较'};return
    ia={k:v for k,v in a['run_identity'].items() if k!='arm'};ib={k:v for k,v in b['run_identity'].items() if k!='arm'}
    if ia!=ib:raise ValueError('双臂身份不同')
    diffs=np.asarray(a['evals'][-1]['batch_losses'])-np.asarray(b['evals'][-1]['batch_losses'])
    report['pair_summary']=dict(eval_final_diff_wpe_minus_rope=float(diffs.mean()),
        paired_batch_se=float(diffs.std(ddof=1)/math.sqrt(len(diffs))) if len(diffs)>1 else None,
        params_diff_rope_minus_wpe=b['params_total']-a['params_total'],
        note='不同参数预算的位置组件比较；配对SE不含训练种子方差；长窗与同窗采用不同样本构造')


def main():
    global OUT_DIR,TOKENS_BIN,EVAL_BIN,WARMUP,SEED
    ap=argparse.ArgumentParser(description='学习式位置表/RoPE对照及长窗评估；同名严格续跑')
    ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--out-name');ap.add_argument('--cells',default='wpe,rope')
    ap.add_argument('--eval-every',type=int,default=500);ap.add_argument('--warmup',type=int,default=WARMUP);ap.add_argument('--seed',type=int,default=SEED)
    ap.add_argument('--out-dir',type=Path,default=Path(OUT_DIR));ap.add_argument('--data-dir',type=Path,default=Path(TOKENS_BIN).parent)
    ap.add_argument('--device',choices=['cpu','mps'],default='mps' if torch.backends.mps.is_available() else 'cpu');args=ap.parse_args()
    WARMUP=args.warmup;SEED=args.seed;OUT_DIR=str(args.out_dir);TOKENS_BIN=str(args.data_dir/'tokens.bin');EVAL_BIN=str(args.data_dir/'eval.bin')
    if not 0<=WARMUP<args.steps or args.eval_every<=0:raise ValueError('需要0<=warmup<steps且eval-every正')
    name=args.out_name or ('fast' if args.steps<=1500 else 'full')
    if Path(name).name!=name or name in ('','.','..'):raise ValueError('out-name须为单个名称')
    arms=args.cells.split(',')
    if not arms or len(set(arms))!=len(arms) or any(x not in ('wpe','rope') for x in arms):raise ValueError('无效或重复实验臂')
    if 'rope' in arms:validate_extrap_pool()
    if 'wpe' in arms and EXTRAP_N<=N_POSITIONS:raise ValueError('wpe探针须越过位置表范围')
    stream=TrainStream(TOKENS_BIN)
    if args.steps>stream.max_step:raise ValueError('步数超过位置一次通过容量')
    identity=run_identity(args.steps,args.eval_every,args.device,stream);path=args.out_dir/f'ablation_pe_{name}.json'
    if path.exists():
        report=json.loads(path.read_text())
        if report.get('meta',{}).get('run_identity')!=identity:raise ValueError('旧报告身份不同或缺失，使用新out-name')
    else:
        report=dict(meta=dict(out_name=name,run_identity=identity,protocol='相同配置/数据身份下的同位置批；非内容去重证明',
            eval_protocol=f'{EVAL_BATCHES}批固定窗；新版包含末合法起点，不要求逐值复现旧窗口',
            precision='MPS bf16/fp32' if args.device=='mps' else 'CPU fp32',
            historical_note='包含末合法评估起点；当前错误类型和运行环境可变，不把旧分数继承为本次结果'),arms={})
    args.out_dir.mkdir(parents=True,exist_ok=True);ex,ey=build_eval_batches(args.device)
    for arm in arms:
        if usable(report['arms'].get(arm,{})):continue
        result=run_arm(arm,args.steps,args.eval_every,args.device,stream,ex,ey,report,path)
        if not usable(result):raise RuntimeError('实验臂未完成，保留有效断点，不进行双臂结论')
    add_pair_summary(report);save_json(path,report);print(path)


if __name__=='__main__':main()
