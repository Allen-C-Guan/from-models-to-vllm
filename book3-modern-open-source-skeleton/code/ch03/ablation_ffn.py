# Book3第3章：GELU-4C与SwiGLU-8/3C前馈对照，默认几何与配方保留历史设置。
# 工作目录log读取语料并另存输出；--help不训练；仅显式执行训练入口时消耗预算。
# 核对配置/日程/数据/代码身份后续跑，成功权重保留；未完成或非有限臂不参与比较。
# main只对齐主矩阵，GELU偏置另列；sanity512矩阵盈余12.5%。历史MPS读数不冒称当前复测。
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'00-feasibility'))
for candidate in (HERE.parents[2]/'book2-pretraining-revolution/code/ch10',HERE.parents[1]/'Book2-预训练革命/ch10'):
    if candidate.is_dir():
        sys.path.insert(0,str(candidate));break
else:raise ImportError('找不到配套Book2第10章模型，请保留相邻书目录布局')
from model import GPT2,GPT2Config
from llama_slots import SwiGLU,LLaMAConfig

SEED=20261002
OUT_DIR=str(Path.cwd()/'log/book3-ch03')
TOKENS_BIN=str(Path.cwd()/'log/book2-ch08/tokens.bin')
EVAL_BIN=str(Path.cwd()/'log/book2-ch08/eval.bin')
VOCAB=8192;BATCH=16;BLOCK=512;TPB=BATCH*BLOCK;EVAL_BATCHES=49
PEAK_LR=1e-3;MIN_LR_RATIO=.1;WARMUP=100;WDECL=.1;TAIL=300
N_POSITIONS=1024
CELLS={
    'main':dict(d=768,L=6,h=12,d_ff_gelu=3072,d_ff_swiglu=2048,note='主矩阵参数相等，GELU偏置另计'),
    'sanity512':dict(d=512,L=6,h=8,d_ff_gelu=2048,d_ff_swiglu=1536,note='SwiGLU主矩阵参数多12.5%，不是同参比较'),
}
ARMS={'main':('gelu4c','swiglu83'),'sanity512':('gelu4c_s512','swiglu83_s512')}


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


def build_arm(arm,cell):
    """先同种子构造GPT2；共享件相同，SwiGLU只改前馈内部，保留LayerNorm。

    输入/输出(B,n,C)，门和值各(B,n,d_ff)。0.02为标准差；down沿残差出口约定缩放，
    不是已证明最优的门控方差配方。此处实际使用Book2第10章模型。
    """
    c=CELLS[cell]
    if arm not in ARMS[cell]:raise ValueError('实验臂与格不匹配')
    torch.manual_seed(SEED)
    model=GPT2(GPT2Config(vocab_size=VOCAB,n_positions=N_POSITIONS,n_layer=c['L'],n_embd=c['d'],n_head=c['h']))
    if arm.startswith('swiglu83'):
        for block in model.blocks:
            block.mlp=SwiGLU(LLaMAConfig(hidden_size=c['d'],intermediate_size=c['d_ff_swiglu'],num_attention_heads=c['h']))
            for projection in (block.mlp.gate_proj,block.mlp.up_proj):
                torch.nn.init.normal_(projection.weight,mean=0.,std=.02)
            torch.nn.init.normal_(block.mlp.down_proj.weight,mean=0.,std=.02/math.sqrt(2*c['L']))
    return model


def parameter_ledger(cell):
    c=CELLS[cell];gelu=2*c['d']*c['d_ff_gelu'];swiglu=3*c['d']*c['d_ff_swiglu']
    if cell=='main':assert gelu==swiglu==8*c['d']**2, '主矩阵预算未对齐'
    if cell=='sanity512':assert math.isclose(swiglu/gelu-1,.125), 'sanity盈余应为12.5%'
    return dict(gelu_matrix_per_layer=gelu,swiglu_matrix_per_layer=swiglu,
                swiglu_surplus_pct=100*(swiglu/gelu-1),gelu_bias_total=c['L']*(c['d_ff_gelu']+c['d']))


def run_identity(steps,eval_every,device,stream):
    return dict(schema=2,seed=SEED,steps=steps,eval_every=eval_every,device=str(device),torch=str(torch.__version__),
        model=dict(vocab=VOCAB,positions=N_POSITIONS,cells=CELLS),batch=BATCH,block=BLOCK,eval_batches=EVAL_BATCHES,
        learning_rate=[PEAK_LR,MIN_LR_RATIO,WARMUP],weight_decay=WDECL,
        train=file_identity(stream.path),eval=file_identity(EVAL_BIN),sampler='all_legal_starts_v2',
        runner=file_identity(__file__),factory=file_identity(sys.modules[GPT2.__module__].__file__),
        component=file_identity(sys.modules[SwiGLU.__module__].__file__))


def usable(record):
    return (record.get('completed') is True and not record.get('diverged') and
            record.get('steps_done')==record.get('steps_planned') and record.get('steps_done',0)>0
            and isinstance(record.get('eval_final'),(int,float)) and math.isfinite(record['eval_final']))


def run_arm(arm,cell,steps,eval_every,device,stream,ex,ey,report,out_json):
    """有效身份下从eval边界恢复；CPU小模型续跑可测试，MPS不承诺逐位确定性。"""
    device=torch.device(device)
    if not 0<=WARMUP<steps or eval_every<=0 or steps>stream.max_step:raise ValueError('步数、warmup或评估间隔无效')
    identity={**run_identity(steps,eval_every,device,stream),'cell':cell,'arm':arm}
    name=report['meta']['out_name'];folder=Path(OUT_DIR);folder.mkdir(parents=True,exist_ok=True)
    ckpt=folder/f'ckpt_{arm}_{name}.pt';curve=folder/f'curve_{arm}_{name}.csv'
    if curve.exists() and not ckpt.exists():raise FileExistsError('同名曲线无可核断点，请改out-name')
    c=CELLS[cell];ledger=parameter_ledger(cell)
    model=build_arm(arm,cell).to(device)
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
    record=dict(arm=arm,cell=cell,run_identity=identity,ffn_param_alignment=ledger,params_total=model.n_params(),params_non_embedding=model.n_params(non_embedding=True),
        config=dict(**c,V=VOCAB,B=BATCH,T=BLOCK),steps_done=count,steps_planned=steps,
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


def add_pair_summary(report):
    """同格双臂完整、有效且身份一致后，按相同评估批计算配对差值。"""
    report['pair_summaries']={}
    for cell,(ga,sa) in ARMS.items():
        a,b=report['arms'].get(ga),report['arms'].get(sa)
        if not a or not b or not usable(a) or not usable(b):continue
        ia={k:v for k,v in a['run_identity'].items() if k!='arm'}
        ib={k:v for k,v in b['run_identity'].items() if k!='arm'}
        if ia!=ib:raise ValueError('双臂运行身份不一致')
        diffs=np.asarray(a['evals'][-1]['batch_losses'])-np.asarray(b['evals'][-1]['batch_losses'])
        se=float(diffs.std(ddof=1)/math.sqrt(len(diffs))) if len(diffs)>1 else None
        report['pair_summaries'][cell]=dict(eval_final_diff_gelu4c_minus_swiglu83=float(diffs.mean()),
            paired_batch_se=se,within_historical_band=bool(abs(diffs.mean())<=.02),
            params_total_diff=b['params_total']-a['params_total'],
            tail300_diff_gelu4c_minus_swiglu83=a['tail300_train_loss']-b['tail300_train_loss'],
            sec_per_step_diff=a['sec_per_step_steady']-b['sec_per_step_steady'],
            note='正损失差表示GELU更高；0.02是描述带，配对批SE不包含训练种子方差；sanity512不是等参实验')


def main():
    global OUT_DIR,TOKENS_BIN,EVAL_BIN,WARMUP,SEED
    ap=argparse.ArgumentParser(description='GELU/SwiGLU前馈实验；同名严格续跑，成功权重保留')
    ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--out-name');ap.add_argument('--cells',default='main')
    ap.add_argument('--eval-every',type=int,default=500);ap.add_argument('--warmup',type=int,default=WARMUP);ap.add_argument('--seed',type=int,default=SEED)
    ap.add_argument('--out-dir',type=Path,default=Path(OUT_DIR));ap.add_argument('--data-dir',type=Path,default=Path(TOKENS_BIN).parent)
    ap.add_argument('--device',choices=['cpu','mps'],default='mps' if torch.backends.mps.is_available() else 'cpu');args=ap.parse_args()
    WARMUP=args.warmup;SEED=args.seed;OUT_DIR=str(args.out_dir);TOKENS_BIN=str(args.data_dir/'tokens.bin');EVAL_BIN=str(args.data_dir/'eval.bin')
    if not 0<=WARMUP<args.steps or args.eval_every<=0:raise ValueError('需要0<=warmup<steps且eval-every正')
    name=args.out_name or ('fast' if args.steps<=1500 else 'full')
    if Path(name).name!=name or name in ('','.','..'):raise ValueError('out-name须为单个名称')
    cells=args.cells.split(',')
    if not cells or len(set(cells))!=len(cells) or any(x not in CELLS for x in cells):raise ValueError('无效或重复实验格')
    stream=TrainStream(TOKENS_BIN)
    if args.steps>stream.max_step:raise ValueError('步数超过位置一次通过容量')
    identity=run_identity(args.steps,args.eval_every,args.device,stream);path=args.out_dir/f'ablation_ffn_{name}.json'
    if path.exists():
        report=json.loads(path.read_text())
        if report.get('meta',{}).get('run_identity')!=identity:raise ValueError('旧报告身份不同或缺失，使用新out-name')
    else:
        report=dict(meta=dict(out_name=name,run_identity=identity,protocol='相同配置/数据身份下的同位置批；非内容去重证明',
            eval_protocol=f'{EVAL_BATCHES}批固定窗；新版包含末合法起点，不要求逐值复现旧窗口',
            precision='MPS bf16/fp32' if args.device=='mps' else 'CPU fp32',
            historical_note='包含末合法评估起点，实际版本与历史不同；不将历史分数当本次结果'),arms={})
    args.out_dir.mkdir(parents=True,exist_ok=True);ex,ey=build_eval_batches(args.device)
    for cell in cells:
        for arm in ARMS[cell]:
            if usable(report['arms'].get(arm,{})):continue
            result=run_arm(arm,cell,args.steps,args.eval_every,args.device,stream,ex,ey,report,path)
            if not usable(result):raise RuntimeError('实验臂未完成，保留有效断点，不进行双臂结论')
    add_pair_summary(report);save_json(path,report);print(path)


if __name__=='__main__':main()
