# passkey.py —— Book3 ch7 passkey 检索协议（ppl 之外的第二口径；KV cache 增量解码的正身实现）
# 用途：对 extrapolate.py 训好的 2k 窗 mini ckpt 做「大海捞针」检索评测——真实 filler（eval.bin heldout
#       段）中段藏一个 5 位 passkey（needle），末尾提问（query），贪心解码 12 token，解码文本包含
#       passkey 即命中。双口径：in-window（1.5k < 训练窗 2k）与 out-window（4k / 8k，2×/4× 越窗）；
#       四干预（none/PI/NTK/YaRN-lite）× 三窗长逐一跑——「ppl 高 ≠ 检索失效」的实验落地
#      （papers/05 §1.3：YaRN v3附录B.5 "perplexity may not be a great indicator…"）。
#       解码用 KV cache 增量式（ch6 增量生成算法的第一次实战：prefill 一次 + 每步 O(t) 单 query），
#       启动前跑「缓存解码 vs 全量重算解码」的自检（同 token 序列 assert，防缓存实现错）。
# 协议细节：needle = " The passkey is {key}. Remember it. "，query = " The passkey is"；
#       同一 (窗长, 深度, trial) 的上下文跨四干预完全复用（配对比较，压方差）；命中判定 = 精确
#       包含 5 位数字串。
# 运行方式：python passkey.py [--ckpt log/book3-ch07/mini2k_full.pt] [--out-name full]
#           [--trials 8] [--depths 0.25,0.5,0.75] [--lengths 1536,4096,8192]
#           fast 档：--trials 4 --depths 0.5（协议跑通，数字仅协议验证）；full 档：全套。
# 产物（log/book3-ch07/，不入库）：passkey_{out}.json（准确率矩阵 + 示例 + 自检记录）。
import argparse
import json
import hashlib
from pathlib import Path
from functools import wraps
import math
import os
import sys
import time

import numpy as np
import sentencepiece as spm
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "00-feasibility")))
from llama_slots import LLaMA, LLaMAConfig, apply_rope, repeat_kv     # noqa: E402
from extrapolate import INTERVENTIONS, forward_with_rope, rope_cache_for, build_model, make_interventions, file_identity, save_json   # noqa: E402  四干预正身（同目录）

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = str(Path.cwd()/"log/book3-ch07")
EVAL_BIN = str(Path.cwd()/"log/book2-ch08/eval.bin")     # filler来源（沿用heldout文件；是否文本去重另核）
SP_MODEL = str(Path.cwd()/"log/book2-ch08/sp8k_owt.model")
SEED = 20261002
DECODE_N = 12       # 贪心解码 token 数
QUERY = " The passkey is"


def layer_step(layer, x, pos_lo, caches, cos, sin):
    """单层前进一步（prefill 或解码），带 KV cache。x (1,t,C)，pos_lo = 本步起始绝对位置。

    k/v 追加进 caches[layer_idx]（(1,h_kv,t_cache,hd)）；注意力 t_query x t_cache——解码步
    t=1 时全历史可见（因果性），prefill 步用 tril 掩码。返回 (1,t,C)。
    """
    if x.ndim!=3 or x.shape[1]<1 or pos_lo<0:raise ValueError("非空三维输入与非负位置")
    previous=caches[layer._idx]
    cached=0 if previous is None else previous[0].shape[2]
    if cached!=pos_lo:raise ValueError("缓存有效长度与绝对位置不一致")
    if cos.shape!=sin.shape or cos.shape[1]<pos_lo+x.shape[1]:raise ValueError("位置表不足")
    attn = layer.self_attn
    B, t, C = x.shape
    h, h_kv, hd = attn.n_head, attn.n_kv_head, attn.head_dim
    h_in = layer.input_layernorm(x)                                        # (1,t,C)
    q = attn.q_proj(h_in).view(B, t, h, hd).transpose(1, 2)                # (1,h,t,hd)
    k = attn.k_proj(h_in).view(B, t, h_kv, hd).transpose(1, 2)             # (1,h_kv,t,hd)
    v = attn.v_proj(h_in).view(B, t, h_kv, hd).transpose(1, 2)             # (1,h_kv,t,hd)
    cos_t = cos[:, pos_lo:pos_lo + t]                                      # (1,t,hd) 本步位置行
    sin_t = sin[:, pos_lo:pos_lo + t]
    q, k = apply_rope(q, k, cos_t, sin_t)                                  # 只旋本步 q/k
    if caches[layer._idx] is not None:
        k = torch.cat([caches[layer._idx][0], k], dim=2)                   # (1,h_kv,t_cache+t,hd)
        v = torch.cat([caches[layer._idx][1], v], dim=2)
    caches[layer._idx] = (k, v)
    k_rep, v_rep = repeat_kv(k, attn.n_rep), repeat_kv(v, attn.n_rep)      # -> (1,h,t_cache+t,hd)
    att = (q @ k_rep.transpose(-2, -1)) * attn.scaling                     # (1,h,t,t_cache+t)
    tc = k_rep.shape[2]
    if t > 1:                                                              # prefill：因果掩码（历史列全可见）
        hist = tc - t                                                      # 已缓存列（右侧新块才需 tril）
        mask = torch.ones(t, tc, device=x.device)
        mask[:, hist:] = torch.tril(torch.ones(t, t, device=x.device))
        att = att.masked_fill(mask.view(1, 1, t, tc) == 0, float("-inf"))
    att = torch.softmax(att.float(), dim=-1).to(q.dtype)                   # fp32 softmax（对齐 HF）
    y = (att @ v_rep).transpose(1, 2).contiguous().view(B, t, C)           # (1,t,C)
    x = x + attn.o_proj(y)                                                 # 残差一
    x = x + layer.mlp(layer.post_attention_layernorm(x))                   # 残差二
    return x


def evaluation_mode(fn):
    @wraps(fn)
    def wrapped(model,*args,**kwargs):
        before=model.training;model.eval()
        try:return fn(model,*args,**kwargs)
        finally:model.train(before)
    return wrapped


def validate_decode(model,ctx_ids,n_new):
    if not isinstance(n_new,int) or isinstance(n_new,bool) or n_new<0:raise ValueError('n_new须非负整数')
    if not len(ctx_ids) or any(not 0<=int(v)<model.cfg.vocab_size for v in ctx_ids):raise ValueError('上下文非空且ID不越界')
    for i,layer in enumerate(model.model.layers):layer._idx=i


@torch.no_grad()
@evaluation_mode
def greedy_decode(model, ctx_ids, n_new, intervention, device):
    """KV cache 贪心解码（正身）：prefill 一次保留 caches，之后每步喂 1 个 token。

    ctx_ids (n,) -> 新增 n_new 个 token id。干预通过 rope_cache_for 注入（免重训，权重不动）。
    位置口径：prefill 覆盖绝对位置 0..n-1；第 j 个新 token（j≥1）落在绝对位置 n+j-1——
    RoPE 的 cos/sin 行号必须用绝对位置（此处曾出 bug：从 0 起数，自检拦截）。
    """
    validate_decode(model,ctx_ids,n_new)
    if n_new==0:return []
    n = len(ctx_ids)
    cos, sin = rope_cache_for(intervention,n+n_new,device,model.cfg.head_dim,model.cfg.rope_theta)             # (1,n+n_new,hd)
    caches = [None] * len(model.model.layers)
    x = torch.tensor([ctx_ids], dtype=torch.long, device=device)           # (1,n)
    h = model.model.embed_tokens(x)                                        # (1,n,C)
    for layer in model.model.layers:                                       # prefill（逐层带 cache，位置 0..n-1）
        h = layer_step(layer, h, 0, caches, cos, sin)
    logits = model.lm_head(model.model.norm(h[:, -1:, :]))                 # (1,1,V) 取末位
    out = [int(logits[0, -1].argmax())]
    pos = n                                                                # 下一个被喂入 token 的绝对位置
    for _ in range(n_new - 1):
        tok = torch.tensor([[out[-1]]], dtype=torch.long, device=device)   # (1,1)
        h = model.model.embed_tokens(tok)                                  # (1,1,C)
        for layer in model.model.layers:                                   # 解码步：单 query O(t)
            h = layer_step(layer, h, pos, caches, cos, sin)
        pos += 1
        logits = model.lm_head(model.model.norm(h))                        # (1,1,V)
        out.append(int(logits[0, -1].argmax()))
    return out


@torch.no_grad()
@evaluation_mode
def greedy_decode_full_recompute(model, ctx_ids, n_new, intervention, device):
    """全量重算贪心（探针 G 口径）——仅自检用：每步对整段 ctx 重新前向。"""
    validate_decode(model,ctx_ids,n_new)
    ctx = list(ctx_ids)
    for _ in range(n_new):
        x = torch.tensor([ctx], dtype=torch.long, device=device)           # (1,t)
        cos, sin = rope_cache_for(intervention,len(ctx),device,model.cfg.head_dim,model.cfg.rope_theta)
        logits, _ = forward_with_rope(model, x, cos, sin)
        ctx.append(int(logits[0, -1].argmax()))
    return ctx[len(ctx_ids):]


def self_check(model, device):
    """协议守门：缓存解码 vs 全量重算解码，同一小上下文上 token 序列必须逐位一致。"""
    rng = np.random.default_rng(SEED)
    ev = np.memmap(EVAL_BIN, dtype=np.uint16, mode="r")
    if len(ev)<2:raise ValueError("自检filler太短")
    size=min(320,len(ev))
    start=int(rng.integers(0,len(ev)-size+1))
    ctx=np.asarray(ev[start:start+size],dtype=int).tolist()
    rec = {}
    for iv in INTERVENTIONS:
        a = greedy_decode(model, ctx, 8, iv, device)
        b = greedy_decode_full_recompute(model, ctx, 8, iv, device)
        rec[iv] = {"cached": a, "full": b, "identical": a == b}
        assert a == b, f"缓存解码自检失败（{iv}）：{a} != {b}"
    rec["note"] = f"{size} token上下文×8步×四干预，缓存与全量重算逐token一致；这是样例校验，不是所有长度的证明"
    return rec


def build_contexts(length, depth, trials, sp):
    """构造 (length, depth) 的 trials 个上下文（同种子确定性；跨干预复用同一批上下文）。

    结构：filler_a + needle + filler_b + query；filler 取 eval.bin 随机段（真实文本），
    needle 深度 = filler_a 占比 depth。返回 [{ctx, key, ctx_len}]。
    """
    if not isinstance(length,int) or length<1 or not math.isfinite(depth) or not 0<=depth<=1 or trials<1:raise ValueError("长度/深度/试验数无效")
    ev = np.memmap(EVAL_BIN, dtype=np.uint16, mode="r")
    needle_tpl = " The passkey is {key}. Remember it. "
    query_ids = sp.EncodeAsIds(QUERY)
    out = []
    for trial in range(trials):
        rng = np.random.default_rng(SEED + trial * 1000 + length + int(depth * 100))
        key = int(rng.integers(10000, 100000))
        needle_ids = sp.EncodeAsIds(needle_tpl.format(key=key))
        nfill = length - len(needle_ids) - len(query_ids)
        if nfill<0 or len(ev)<nfill:raise ValueError("请求长度装不下针/问题或filler不足")
        start=int(rng.integers(0,len(ev)-nfill+1))
        fill_a = ev[start:start + int(nfill * depth)]
        fill_b = ev[start + int(nfill * depth):start + nfill]
        ctx = np.concatenate([fill_a, np.array(needle_ids, dtype=np.uint16), fill_b,
                              np.array(query_ids, dtype=np.uint16)]).astype(np.int64)
        out.append({"ctx": ctx, "key": key, "ctx_len": int(len(ctx))})
    return out




def main():
    global OUT_DIR,EVAL_BIN,SP_MODEL,DECODE_N
    ap=argparse.ArgumentParser(description='冻结checkpoint的配对passkey测试；历史包含匹配口径')
    ap.add_argument('--ckpt',type=Path,default=Path(OUT_DIR)/'mini2k_full.pt');ap.add_argument('--out-name')
    ap.add_argument('--trials',type=int,default=8);ap.add_argument('--depths',default='0.5');ap.add_argument('--lengths',default='1536,4096,8192')
    ap.add_argument('--decode-tokens',type=int,default=DECODE_N);ap.add_argument('--historical-checkpoint',action='store_true')
    ap.add_argument('--out-dir',type=Path,default=Path(OUT_DIR));ap.add_argument('--data-dir',type=Path,default=Path(EVAL_BIN).parent)
    ap.add_argument('--tokenizer',type=Path);ap.add_argument('--device',choices=['cpu','mps'],default='mps' if torch.backends.mps.is_available() else 'cpu');args=ap.parse_args()
    lengths=[int(x) for x in args.lengths.split(',')];depths=[float(x) for x in args.depths.split(',')]
    if args.trials<1 or args.decode_tokens<1 or not lengths or min(lengths)<=0 or len(set(lengths))!=len(lengths):raise ValueError('长度/试验/解码数须正且长度不重复')
    if not depths or len(set(depths))!=len(depths) or any(not math.isfinite(d) or not 0<=d<=1 for d in depths):raise ValueError('深度须在0..1且不重复')
    name=args.out_name or ('fast' if args.trials<=4 else 'full')
    if Path(name).name!=name or name in ('','.','..'):raise ValueError('无效输出名称')
    OUT_DIR=str(args.out_dir);EVAL_BIN=str(args.data_dir/'eval.bin');SP_MODEL=str(args.tokenizer or args.data_dir/'sp8k_owt.model');DECODE_N=args.decode_tokens
    output=args.out_dir/f'passkey_{name}.json'
    if output.exists():raise FileExistsError('已有报告，请改out-name')
    device=torch.device(args.device);ck=torch.load(args.ckpt,map_location=device,weights_only=True)
    if ck.get('completed') is False:raise ValueError('失败checkpoint不能冒作成功模型')
    if ck.get('completed') is not True and not args.historical_checkpoint:raise ValueError('旧checkpoint需显式historical-checkpoint')
    model=build_model(ck['config'],device);model.load_state_dict(ck['model']);model.eval()
    sp=spm.SentencePieceProcessor(model_file=SP_MODEL)
    if sp.vocab_size()!=model.cfg.vocab_size:raise ValueError('分词器词表尺寸与模型不一致')
    # 尺寸相同仍不证明词表ID语义一致；记录实际模型hash供读者核对原训练资产。
    eval_stream=np.memmap(EVAL_BIN,dtype=np.uint16,mode='r')
    if len(eval_stream)==0 or int(eval_stream.max())>=sp.vocab_size():raise ValueError('filler含越界词元')
    check=self_check(model,device)
    results={iv:{} for iv in INTERVENTIONS};case_identity={};started=time.perf_counter()
    for length in lengths:
        for depth in depths:
            cases=build_contexts(length,depth,args.trials,sp)
            if any(len(sp.EncodeAsIds(str(c['key'])))>DECODE_N for c in cases):raise ValueError('解码预算不足以表示完整数字答案')
            cell=f'len{length}@d{depth}';case_identity[cell]=hashlib.sha256(b''.join(c['ctx'].tobytes()+str(c['key']).encode() for c in cases)).hexdigest()
            for iv in INTERVENTIONS:
                hits=0;examples=[]
                for case in cases:
                    ids=greedy_decode(model,case['ctx'].tolist(),DECODE_N,iv,device);text=sp.DecodeIds(ids)
                    ok=str(case['key']) in text;hits+=int(ok)
                    if len(examples)<2:examples.append(dict(key=case['key'],ctx_len=case['ctx_len'],decoded=text,hit=ok))
                results[iv][cell]=dict(accuracy=hits/args.trials,hits=hits,trials=args.trials,mean_ctx_len=length,examples=examples)
    summary={};counts={}
    for iv in INTERVENTIONS:
        summary[iv]={};counts[iv]={}
        for length in lengths:
            cells=[v for k,v in results[iv].items() if k.startswith(f'len{length}@')];total=sum(v['trials'] for v in cells);hits=sum(v['hits'] for v in cells)
            summary[iv][str(length)]=hits/total;counts[iv][str(length)]=dict(hits=hits,trials=total)
    report=dict(meta=dict(out_name=name,seed=SEED,device=str(device),torch=str(torch.__version__),ckpt=str(args.ckpt),ckpt_steps=ck.get('steps'),
        checkpoint_identity=file_identity(args.ckpt),tokenizer_identity=file_identity(SP_MODEL),filler_identity=file_identity(EVAL_BIN),config=dict(vars(model.cfg)),
        runner_identity=file_identity(__file__),decode_tokens=DECODE_N,trials=args.trials,depths=depths,lengths=lengths,
        interventions=make_interventions(model.cfg.head_dim,model.cfg.rope_theta),
        protocol='固定真实filler中插入五位数字，末尾提问；命中=目标数字串包含于续写，不是整句精确相等；默认12token且不因EOS提前终止',
        paired_design='同一长度/深度/trial跨四干预共享完整上下文；不同干预不是独立试验',
        sampler='v2：数字10000..99999含端点；所有合法filler起点',
        compatibility_note='词表大小相同仍需核对原训练分词器身份；此报告记录实际hash，不追认未知历史词表'),
        self_check=check,results=results,by_length_summary=summary,by_length_counts=counts,context_hashes=case_identity,
        completed=True,wallclock_sec=time.perf_counter()-started)
    args.out_dir.mkdir(parents=True,exist_ok=True);save_json(output,report);print(output)


if __name__=='__main__':main()
