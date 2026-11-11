# extrapolate.py —— Book3 ch7 外推核心实验：2k 窗 mini 训练 + 四干预免重训 eval（none/PI/NTK/YaRN-lite）
# 用途：(1) 训一个 2k 窗 RoPE mini（llama_slots.LLaMA：d=256/L=4/h=4/h_kv=4/V=8192，7.4M，
#           B=8/T=2048，五件套配方，tokens.bin 顺序分块一次通过）；
#       (2) 训练后**免重训**只改 RoPE 参数做四干预 eval（s=4，2k→8k 名义窗）：
#           none      原样 RoPE（theta=10000，位置原尺度）——直接外推对照
#           pi        位置缩放（PI 式 4：位置 m→m/s，等价 inv_freq/=s；HF rope_type=linear 同式）
#           ntk       base 放大（YaRN 式 15 转述 bloc97：b'=b·s^(d/(d-2))，d=head_dim=64——
#                     高频外推、低频插值的「改 base 不改尺度」）
#           yarn_lite ntk 的 base + 注意力温度（YaRN v2式22/v3式15：√(1/t)=0.1·ln s+1；HF get_mscale
#                     口径 = mscale 乘 cos/sin，q/k 各乘一次 → logits×mscale²=1/t）
#           （YaRN 本体还有 NTK-by-parts 的波长分段斜坡——本实验为「冻结权重的自定义组合」取其
#            base+温度两件的 lite 版，分段斜坡归正文讲解；命名如实带 _lite 后缀）
#       (3) eval 三窗长 2048（同窗）/8192（4× 目标窗）/16384（二次越界）× 位置分桶长文 ppl，
#           全部 fp32；同一 heldout 段、同 token 预算（各窗长均 393,216 token）。
# 运行方式：python extrapolate.py [--steps 2000] [--out-name fast] [--eval-every 500]
#           两档位：fast=2000 步（~12 min）/ full=5000 步（~31 min）；训练断点续跑。
# 产物（log/book3-ch07/，不入库）：extrapolate_{out}.json + curve_train_{out}.csv
#           + mini2k_{out}.pt（终态 ckpt：model+config，供 passkey.py 与后续复用）。
import argparse
import csv
import json
import inspect
import hashlib
from pathlib import Path
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "00-feasibility")))
from llama_slots import LLaMA, LLaMAConfig          # noqa: E402  本册插槽组件库（整机正身）

# ---------------- 固定口径 ----------------
SEED = 20261002
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = str(Path.cwd()/"log/book3-ch07")
TOKENS_BIN = str(Path.cwd()/"log/book2-ch08/tokens.bin")
EVAL_BIN = str(Path.cwd()/"log/book2-ch08/eval.bin")     # 1.2M heldout（沿用Book2拆分；独立文件本身不证明文本去重）

MINI_CFG = dict(vocab_size=8192, hidden_size=256, intermediate_size=704,
                num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
                max_position_embeddings=2048, rope_theta=10000.0)       # 探针 G 同 config（7,407,872 参数）
BATCH, BLOCK = 8, 2048                # 16,384 tok/步（探针 G 实测 ~0.38 s/step）
TPB = BATCH * BLOCK
PEAK_LR, MIN_LR_RATIO, WARMUP, WDECL, TAIL = 1e-3, 0.1, 100, 0.1, 300

SCALE = 4                              # s：2k→8k 名义扩窗（四干预共用）
HEAD_DIM = 64                          # d=256/h=4
ROPE_BASE = 10000.0
NTK_BASE = ROPE_BASE * SCALE ** (HEAD_DIM / (HEAD_DIM - 2))   # YaRN 式 15：b·s^(d/(d-2))
YARN_MSCALE = 0.1 * math.log(SCALE) + 1                        # YaRN v2式22/v3式15：√(1/t)=0.1·ln s+1（HF get_mscale）

# 下列NTK式15/温度式22为旧稿所据YaRN v2编号；v3温度移为式15。
# 四干预定义（免重训，只改 RoPE 参数；eval 时逐干预重建 cos/sin）
INTERVENTIONS = {
    "none":      {"pos_scale": 1.0, "base": ROPE_BASE, "mscale": 1.0,
                  "def": "原样 RoPE（直接外推对照）"},
    "pi":        {"pos_scale": 1.0 / SCALE, "base": ROPE_BASE, "mscale": 1.0,
                  "def": f"PI 位置缩放 s={SCALE}：位置 m→m/s（等价 inv_freq/=s；论文式 4/HF linear）"},
    "ntk":       {"pos_scale": 1.0, "base": NTK_BASE, "mscale": 1.0,
                  "def": f"NTK base 放大：b'=b·s^(d/(d-2))={NTK_BASE:.1f}（YaRN 式 15 转述 bloc97）"},
    "yarn_lite": {"pos_scale": 1.0, "base": NTK_BASE, "mscale": YARN_MSCALE,
                  "def": f"YaRN-lite：NTK base + 温度 mscale=0.1·ln s+1={YARN_MSCALE:.4f} 乘 cos/sin"
                         f"（式 22 √(1/t)；HF get_mscale 口径；不含 by-parts 波长分段）"},
}

# eval 窗长协议：各窗长固定批大小与窗数（token 预算一律 393,216——目标索引一致，重置边界不同）
EVAL_PLAN = {2048: {"windows": 192, "batch": 8, "buckets": [0, 512, 1024, 2048]},
             8192: {"windows": 48, "batch": 4, "buckets": [0, 512, 1024, 2048, 4096, 8192]},
             16384: {"windows": 24, "batch": 1, "buckets": [0, 1024, 2048, 4096, 8192, 16384]}}


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


def rope_cache_variant(n, head_dim, base, device, pos_scale=1.0, mscale=1.0):
    """四干预共用的 RoPE cos/sin 查表（llama_slots.build_rope_cache 的参数化推广）。

    inv_freq_i = 1/base^(2i/head_dim)；t = pos_scale·arange(n)（PI 的位置缩放就在这一步）；
    emb = cat(freqs, freqs)（rotate_half 约定需整维表）；mscale 乘 cos/sin（YaRN 温度的
    合入位置表的实现（实际开销仍需测量）——q/k 各乘一次 → logits ×mscale²）。返回 (1,n,hd) fp32 各一。
    """
    if not isinstance(n,int) or n<0 or not isinstance(head_dim,int) or head_dim<=0 or head_dim%2:
        raise ValueError("非负长度、正偶数头维")
    if not all(math.isfinite(v) and v>0 for v in (base,pos_scale,mscale)):raise ValueError("频率与缩放须有限正数")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(n, dtype=torch.float32, device=device) * pos_scale      # (n,)
    freqs = torch.outer(t, inv_freq)                                          # (n,hd/2)
    emb = torch.cat((freqs, freqs), dim=-1)                                   # (n,hd)
    return (emb.cos() * mscale)[None], (emb.sin() * mscale)[None]             # (1,n,hd) x2


def make_interventions(head_dim=HEAD_DIM,base=ROPE_BASE,scale=SCALE):
    if head_dim<=2 or head_dim%2 or not math.isfinite(base) or base<=0 or not math.isfinite(scale) or scale<1:
        raise ValueError('NTK需大于2的偶数旋转维度，base正、scale>=1')
    ntk=base*scale**(head_dim/(head_dim-2));mscale=1+.1*math.log(scale)
    return {'none':dict(pos_scale=1.,base=base,mscale=1.),'pi':dict(pos_scale=1/scale,base=base,mscale=1.),
            'ntk':dict(pos_scale=1.,base=ntk,mscale=1.),'yarn_lite':dict(pos_scale=1.,base=ntk,mscale=mscale)}


def rope_cache_for(intervention,n,device,head_dim=HEAD_DIM,base=ROPE_BASE,scale=SCALE):
    spec=make_interventions(head_dim,base,scale)[intervention]
    return rope_cache_variant(n,head_dim,spec['base'],device,pos_scale=spec['pos_scale'],mscale=spec['mscale'])


def forward_with_rope(model, idx, cos, sin, targets=None):
    """显式 cos/sin 注入的前向（绕过 backbone 内置 rope 构建——干预只发生在 eval 侧，权重不动）。

    idx (B,n) -> logits (B,n,V)；逐层 x = layer(x, cos, sin)（llama_slots.LLaMADecoderLayer 原生接口）。
    """
    x = model.model.embed_tokens(idx)                                    # (B,n,C)
    for layer in model.model.layers:                                     # 4 × (B,n,C)
        x = layer(x, cos, sin)
    x = model.model.norm(x)                                              # (B,n,C)
    logits = model.lm_head(x)                                            # (B,n,C)·(C,V) -> (B,n,V)
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))
    return logits, loss


@torch.no_grad()
def eval_windows(model, device, n, windows, batch, buckets, intervention=None):
    """heldout 首部不重叠窗一次通过：token 加权 loss -> ppl + 位置分桶 loss（全 fp32）。

    intervention=None 用模型原生前向（训练中途 eval）；否则注入该干预的 cos/sin（免重训）。
    """
    if min(n,windows,batch)<=0 or buckets[0]!=0 or buckets[-1]!=n or any(a>=b for a,b in zip(buckets,buckets[1:])):
        raise ValueError('窗口/批量须正，分桶须严格覆盖0..n')
    if Path(EVAL_BIN).stat().st_size%2:raise ValueError('uint16流字节数须偶数')
    was_training=model.training
    try:
        model.eval()
        ev = np.memmap(EVAL_BIN, dtype=np.uint16, mode="r")
        if len(ev)<n*windows+1:raise ValueError("评估流不足完整输入/目标范围")
        losses, ntok = 0.0, 0
        bsum = [0.0] * (len(buckets) - 1)
        bcnt = [0] * (len(buckets) - 1)
        for b0 in range(0, windows, batch):
            nb = min(batch, windows - b0)
            xs = [ev[i * n:(i + 1) * n] for i in range(b0, b0 + nb)]
            ys = [ev[i * n + 1:(i + 1) * n + 1] for i in range(b0, b0 + nb)]
            x = torch.from_numpy(np.stack(xs).astype(np.int64)).to(device)      # (nb,n)
            y = torch.from_numpy(np.stack(ys).astype(np.int64)).to(device)      # (nb,n)
            if intervention is None:
                logits, loss = model(x, y)
            else:
                cos, sin = rope_cache_for(intervention,n,device,model.cfg.head_dim,model.cfg.rope_theta,SCALE)
                logits, loss = forward_with_rope(model, x, cos, sin, y)
            k = x.numel()
            if not torch.isfinite(loss):raise FloatingPointError("评估损失非有限")
            losses += loss.item() * k
            ntok += k
            pos_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                       y.reshape(-1), reduction="none").view(x.shape).mean(dim=0)  # (n,)
            for bi in range(len(bsum)):
                lo, hi = buckets[bi], min(buckets[bi + 1], n)
                if hi <= lo:
                    continue
                bsum[bi] += pos_loss[lo:hi].sum().item() * nb
                bcnt[bi] += (hi - lo) * nb
    finally:model.train(was_training)
    avg = losses / ntok
    out = {f"pos[{buckets[i]}:{buckets[i+1]}]": round(bsum[i] / max(1, bcnt[i]), 4) for i in range(len(bsum))}
    return avg, math.exp(avg), out, ntok




def config_dict(model):
    return dict(vars(model.cfg))


def build_model(config=None,device='cpu'):
    values=dict(MINI_CFG if config is None else config)
    if "head_dim" in values and values["head_dim"]!=values["hidden_size"]//values["num_attention_heads"]:
        raise ValueError("本教学模型仅支持head_dim=hidden_size/num_attention_heads")
    allowed=set(inspect.signature(LLaMAConfig).parameters)
    values={k:v for k,v in values.items() if k in allowed}
    torch.manual_seed(SEED);model=LLaMA(LLaMAConfig(**values)).to(device)
    c=model.cfg;d=c.hidden_size;L=c.num_hidden_layers
    hand=2*c.vocab_size*d+L*(2*d*c.num_attention_heads*c.head_dim+2*d*c.num_key_value_heads*c.head_dim+3*d*c.intermediate_size+2*d)+d
    assert model.n_params()==hand
    if config is None and MINI_CFG['hidden_size']==256 and MINI_CFG['num_hidden_layers']==4 and MINI_CFG['vocab_size']==8192:
        assert hand==7407872
    return model


def run_identity(steps,eval_every,device,stream,config,init_from=None):
    return dict(schema=2,steps=steps,eval_every=eval_every,seed=SEED,device=str(device),torch=str(torch.__version__),
        config=config,batch=BATCH,block=BLOCK,warmup=WARMUP,lr=[PEAK_LR,MIN_LR_RATIO],weight_decay=WDECL,
        train=file_identity(stream.path),eval=file_identity(EVAL_BIN),eval_plan={str(n):p for n,p in EVAL_PLAN.items()},
        init_from=file_identity(init_from) if init_from else None,runner=file_identity(__file__),
        factory=file_identity(sys.modules[LLaMA.__module__].__file__),scale=SCALE)


def save_checkpoint(path,payload):
    torch.save(payload,str(path)+'.tmp');os.replace(str(path)+'.tmp',path)


def train_model(steps,eval_every,device,stream,name,init_from=None):
    """默认mini配方；续跑要匹配身份与日程，热起只载权重并明确重启日程。"""
    if not 0<=WARMUP<steps or eval_every<=0 or steps>stream.max_step:raise ValueError('无效步数、warmup或容量')
    device=torch.device(device);folder=Path(OUT_DIR);folder.mkdir(parents=True,exist_ok=True)
    model=build_model(device=device);identity=run_identity(steps,eval_every,device,stream,config_dict(model),init_from)
    ckpt=folder/f'ckpt_train_{name}.pt';curve=folder/f'curve_train_{name}.csv';final=folder/f'mini2k_{name}.pt'
    report_path=folder/f'extrapolate_{name}.json'
    if report_path.exists():
        previous=json.loads(report_path.read_text())
        if previous.get('meta',{}).get('run_identity')!=identity:raise ValueError('旧报告身份不同，请改out-name')
    if (curve.exists() or final.exists()) and not ckpt.exists():raise FileExistsError('已有输出却没有可核断点，请改out-name')
    opt=torch.optim.AdamW(model.parameters(),lr=PEAK_LR,betas=(.9,.95),weight_decay=WDECL)
    losses=[];lrs=[];times=[];evals=[];start=0;elapsed_train=elapsed_eval=0.
    if ckpt.exists():
        ck=torch.load(ckpt,map_location=device,weights_only=True)
        if ck.get('run_identity')!=identity:raise ValueError('断点配置/数据/日程/代码不同')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['opt']);start=ck['step']
        losses=ck['train_losses'];lrs=ck['lrs'];times=ck['times'];evals=ck['evals'];elapsed_train=ck['elapsed_train'];elapsed_eval=ck['elapsed_eval']
        if start>steps or len(losses)!=start or len(lrs)!=start or len(times)!=start:raise ValueError('断点曲线与步数不一致')
        torch.set_rng_state(ck['cpu_rng'].cpu())
        if device.type=='mps' and ck.get('mps_rng') is not None:torch.mps.set_rng_state(ck['mps_rng'].cpu())
    elif init_from:
        ck=torch.load(init_from,map_location=device,weights_only=True)
        other=build_model(ck['config'],device)
        if config_dict(other)!=config_dict(model):raise ValueError('热起checkpoint配置不匹配')
        model.load_state_dict(ck['model']);del other
    failure=None;started=time.perf_counter();model.train()
    for step in range(start+1,steps+1):
        lr=lr_at(step,steps)
        for group in opt.param_groups:group['lr']=lr
        x,y=stream.batch(step,device);begin=time.perf_counter()
        with torch.autocast('mps',dtype=torch.bfloat16) if device.type=='mps' else torch.autocast('cpu',enabled=False):
            _,loss=model(x,y)
        value=float(loss.detach())
        if not math.isfinite(value) or value>100:failure=dict(step=step,reason='非有限或超过阈值loss，未更新');break
        opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        if not math.isfinite(norm):failure=dict(step=step,reason='非有限梯度，未更新');break
        opt.step()
        if device.type=='mps':torch.mps.synchronize()
        times.append(time.perf_counter()-begin);elapsed_train+=times[-1];losses.append(value);lrs.append(lr)
        if step==1 or step%100==0:print(f'[train] {step}/{steps} loss={value:.4f}',flush=True)
        if step%eval_every==0 or step==steps:
            n=BLOCK;p=EVAL_PLAN[n];begin=time.perf_counter()
            el,ep,_,ntok=eval_windows(model,device,n,p['windows'],p['batch'],p['buckets'])
            if device.type=='mps':torch.mps.synchronize()
            elapsed_eval+=time.perf_counter()-begin
            evals.append(dict(step=step,eval_loss_2k=round(el,5),ppl_2k=round(ep,3),window=n,n_tokens=ntok))
            save_checkpoint(ckpt,dict(model=model.state_dict(),opt=opt.state_dict(),step=step,train_losses=losses,lrs=lrs,times=times,evals=evals,
                elapsed_train=elapsed_train,elapsed_eval=elapsed_eval,run_identity=identity,config=config_dict(model),cpu_rng=torch.get_rng_state(),
                mps_rng=torch.mps.get_rng_state() if device.type=='mps' else None))
    count=len(losses);completed=failure is None and count==steps and bool(evals) and evals[-1]['step']==steps
    tail=losses[-TAIL:];steady=times[20:] if len(times)>20 else times
    with curve.open('w',newline='') as out:
        writer=csv.writer(out);writer.writerow(['step','train_loss','lr']);writer.writerows(zip(range(1,count+1),losses,lrs))
    rec=dict(steps_done=count,steps_planned=steps,completed=completed,failure=failure,resumed_from_step=start or None,tokens_seen=count*TPB,
        tail300_train_loss=sum(tail)/len(tail) if tail else None,first_loss=losses[0] if losses else None,last_loss=losses[-1] if losses else None,
        sec_per_step_steady=sum(steady)/len(steady) if steady else None,train_wall_sec=time.perf_counter()-started,
        elapsed_train=elapsed_train,eval_overhead_sec=elapsed_eval,mid_evals=evals,wall_scope='train_wall_sec仅本次调用；累计项另列')
    report=dict(meta=dict(out_name=name,run_identity=identity,model=config_dict(model),scale=SCALE,
        precision='MPS bf16训练/fp32评估' if device.type=='mps' else 'CPU fp32',init_from=str(init_from) if init_from else None,
        eval_protocol='相同流目标索引1..n*windows；跨长度重置边界不同。独立文件不自动证明文本无重复'),train=rec)
    save_json(report_path,report)
    if not completed:raise RuntimeError('训练未完成，保留有效断点，不评四干预或保存成功终态')
    save_checkpoint(final,dict(model=model.state_dict(),config=config_dict(model),steps=count,completed=True,seed=SEED,out_name=name,run_identity=identity))
    return model,report,final


def evaluate_interventions(model,device):
    """同一权重逐干预，目标索引相同；返回完整表，不在失败后补空/零结果。"""
    table={};before={k:v.detach().clone() for k,v in model.state_dict().items()}
    for iv in INTERVENTIONS:
        table[iv]={}
        for n,plan in EVAL_PLAN.items():
            begin=time.perf_counter();loss,ppl,buckets,count=eval_windows(model,device,n,plan['windows'],plan['batch'],plan['buckets'],iv)
            if torch.device(device).type=='mps':torch.mps.synchronize()
            table[iv][str(n)]=dict(loss=round(loss,5),ppl=round(ppl,3),buckets=buckets,n_tokens=count,wall_sec=time.perf_counter()-begin)
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items()),'评估改变了权重'
    return table


def main():
    global OUT_DIR,TOKENS_BIN,EVAL_BIN,WARMUP,SEED
    ap=argparse.ArgumentParser(description='2k mini训练/四干预评估；默认配置保留历史配方')
    ap.add_argument('--steps',type=int,default=2000);ap.add_argument('--out-name');ap.add_argument('--eval-every',type=int,default=500)
    ap.add_argument('--warmup',type=int,default=WARMUP);ap.add_argument('--seed',type=int,default=SEED)
    ap.add_argument('--init-from',type=Path,help='同配置权重热起；日程重启，不是接续原总步数')
    ap.add_argument('--eval-only',type=Path,help='已有checkpoint仅评估；不训练')
    ap.add_argument('--historical-checkpoint',action='store_true',help='显式读取缺运行身份的可信历史checkpoint')
    ap.add_argument('--out-dir',type=Path,default=Path(OUT_DIR));ap.add_argument('--data-dir',type=Path,default=Path(TOKENS_BIN).parent)
    ap.add_argument('--device',choices=['cpu','mps'],default='mps' if torch.backends.mps.is_available() else 'cpu');args=ap.parse_args()
    OUT_DIR=str(args.out_dir);TOKENS_BIN=str(args.data_dir/'tokens.bin');EVAL_BIN=str(args.data_dir/'eval.bin');WARMUP=args.warmup;SEED=args.seed
    name=args.out_name or ('eval' if args.eval_only else 'fast' if args.steps<=3000 else 'full')
    if Path(name).name!=name or name in ('','.','..'):raise ValueError('无效输出名称')
    if args.eval_only and args.init_from:raise ValueError('eval-only与init-from不能并用')
    output=args.out_dir/f'extrapolate_{name}.json';device=torch.device(args.device)
    if args.eval_only:
        if output.exists():raise FileExistsError('仅评估输出已存在，请更换out-name')
        ck=torch.load(args.eval_only,map_location=device,weights_only=True)
        if ck.get('completed') is False:raise ValueError('失败checkpoint不可评估')
        if ck.get('completed') is not True and not args.historical_checkpoint:raise ValueError('旧checkpoint需显式historical-checkpoint')
        model=build_model(ck['config'],device);model.load_state_dict(ck['model'])
        report=dict(meta=dict(out_name=name,eval_only=True,checkpoint=file_identity(args.eval_only),eval=file_identity(EVAL_BIN),
            model=config_dict(model),scale=SCALE,torch=str(torch.__version__),historical_checkpoint=args.historical_checkpoint),train=dict(steps_done=ck.get('steps')))
    else:
        stream=TrainStream(TOKENS_BIN)
        model,report,_=train_model(args.steps,args.eval_every,device,stream,name,args.init_from)
    report['meta']['interventions']=make_interventions(model.cfg.head_dim,model.cfg.rope_theta,SCALE)
    report['intervention_eval']=evaluate_interventions(model,device);report['completed']=True
    args.out_dir.mkdir(parents=True,exist_ok=True);save_json(output,report);print(output)


if __name__=='__main__':main()
