# llama215.py —— Book3 ch10 大项目整合件：LLaMA 式 207M 整机（四插槽 import 自第 2/3/4/6 章）
# 用途：本册单轨代码主线的收官形态——「改装史写在 import 语句里」：
#       归一化插槽   import 自 ch02/rmsnorm.py::RMSNorm(hidden_size, eps)
#       前馈插槽     import 自 ch03/swiglu.py::SwiGLUMLP(cfg)
#       位置插槽     import 自 ch04/rope.py::build_rope_cache(seq_len, head_dim, theta, device)
#                     与 apply_rope(q, k, cos, sin)（ch06/gqa.py 内部同样依赖本件）
#       注意力插槽   import 自 ch06/gqa.py::GroupedQueryAttention(cfg).forward(x, cos, sin)
#       四件均承 00-feasibility/llama_slots.py 正身的类名/签名/HF 键位（插槽 import 契约），
#       本文件只做「组装」：DecoderLayer / Backbone / 整机 + untied lm_head + 参数逐项账 assert。
# config 定版 cand1：d=1024 / L=12 / h=16 / h_kv=8 / d_ff=2816 / V=32000 untied /
#       RoPE theta=10000（无 wpe）/ rms_norm_eps=1e-5 / 无 bias —— 总参数 207,119,360。
# 所属章节：Book3 第 10 章（10.2 整机实现 / 10.3 冒烟与对拍）；DESIGN-215M.md 的架构正身。
# 运行方式（三档，全 CPU 可跑除注明外）：
#   自测（秒级）        ：source env.sh && python llama215.py
#                        —— 参数逐项账 assert 207,119,360 + 随机 token 初始 loss 自检（ln V + sigma^2/2）
#   HF 结构对拍（分钟级）：python llama215.py --parity
#                        —— 同 config 随机权重 strict 直搬 HF LlamaForCausalLM，CPU fp32 对拍 logits
#   TinyLlama 实权重对拍：python llama215.py --tinyllama --snapshot /本地完整快照目录
#                        —— 载入 TinyLlama_v1.1（Apache-2.0，HF 缓存实物）真权重到本类，对拍 + 贪心续写
# 产物：对拍 JSON 落 log/book3-ch10/{llama215_parity, llama215_tinyllama}_{out}.json（不入库）。
# 依赖链：ch02/ch03/ch04/ch06 四章教学重写件（各自与 llama_slots 正身逐位一致——对拍已在前章验收）
#       ＋ 00-feasibility/llama_slots.py（LLaMAConfig 鸭子类型正身）——五个目录见文件内 sys.path 段；
#       transformers仅在对拍段import；实际版本写入报告，本轮5.14.1。
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
BOOK3 = os.path.abspath(os.path.join(HERE, ".."))
for sub in ("ch02", "ch03", "ch04", "ch06", "00-feasibility"):      # 插槽 import 契约的物理落点
    sys.path.insert(0, os.path.join(BOOK3, sub))

from rmsnorm import RMSNorm                                # noqa: E402  归一化插槽（第 2 章）
from swiglu import SwiGLUMLP                               # noqa: E402  前馈插槽（第 3 章）
from rope import apply_rope, build_rope_cache              # noqa: E402  位置插槽（第 4 章）
from gqa import GroupedQueryAttention                       # noqa: E402  注意力插槽（第 6 章）
from llama_slots import LLaMAConfig                        # noqa: E402  config 鸭子类型正身（组件库）

REPO_ROOT = os.getcwd()  # 从工作区根目录启动；数据与产物路径可显式传入。
OUT_DIR = os.path.join(REPO_ROOT, "log", "book3-ch10")
SEED = 20261002                                            # 全书统一种子

# ---------------- 215M 定版 config（cand1；逐字段依据见 DESIGN-215M.md §3） ----------------
LLAMA_215M = dict(vocab_size=32000, hidden_size=1024, intermediate_size=2816,
                  num_hidden_layers=12, num_attention_heads=16, num_key_value_heads=8,
                  max_position_embeddings=8192, rope_theta=10000.0, rms_norm_eps=1e-5)


def validate_config(cfg):
    for key in ('vocab_size','hidden_size','intermediate_size','num_hidden_layers',
                'num_attention_heads','num_key_value_heads','head_dim','max_position_embeddings'):
        if type(getattr(cfg,key)) is not int or getattr(cfg,key)<=0:
            raise ValueError(f'{key}必须为正整数')
    if cfg.hidden_size!=cfg.num_attention_heads*cfg.head_dim or cfg.head_dim%2 or cfg.num_attention_heads%cfg.num_key_value_heads:
        raise ValueError('整机需d=h*head_dim、偶数head_dim、查询头数为KV头数的整数倍')
    for key in ('rope_theta','rms_norm_eps'):
        if not math.isfinite(getattr(cfg,key)) or getattr(cfg,key)<=0:
            raise ValueError(f'{key}必须为有限正数')
    if not math.isfinite(cfg.initializer_range) or cfg.initializer_range<0:
        raise ValueError('initializer_range必须为有限非负数')


class Llama215DecoderLayer(nn.Module):
    """pre-RMSNorm 残差块——拓扑仍是 Book2 三插槽 Block，两个插槽里装的是本册新件。

        x = x + attn(input_layernorm(x))      RMSNorm(ch2) -> GQA(ch6, 内旋 RoPE(ch4))
        x = x + mlp(post_attention_layernorm(x))   RMSNorm(ch2) -> SwiGLU(ch3)
    输入/输出 (B,n,C)——层内四件套换装，残差河恒宽不变（图 0.1 的插座①②③④全在这 12 行里）。
    """

    def __init__(self, cfg: LLaMAConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)          # 插槽①（第 2 章）
        self.self_attn = GroupedQueryAttention(cfg)                                # 插槽④（第 6 章）
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)  # 插槽①（第 2 章）
        self.mlp = SwiGLUMLP(cfg)                                                  # 插槽②（第 3 章）

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)   # (B,n,C) 残差河
        x = x + self.mlp(self.post_attention_layernorm(x))          # (B,n,C)
        return x


class Llama215Backbone(nn.Module):
    """embed_tokens + L 层 DecoderLayer + 终态 RMSNorm。命名对齐 HF LlamaModel（model.*）。

    入口无位置查表（wpe 已在第 4 章拆除）：位置信息由每层注意力内的 apply_rope 现做，
    cos/sin按当前窗长现算，无学习式位置表硬上限；实际长度仍受内存、数值与训练范围限制。
    """

    def __init__(self, cfg: LLaMAConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)            # (V,C)
        self.layers = nn.ModuleList([Llama215DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)                       # 出口前最后一枚

    def forward(self, idx):
        B, n = idx.shape                                                     # (B,n)
        cfg = self.cfg
        x = self.embed_tokens(idx)                                           # (B,n) -> (B,n,C)
        cos, sin = build_rope_cache(n, cfg.head_dim, cfg.rope_theta, idx.device)  # (1,n,hd) x2（第 4 章）
        for layer in self.layers:                                            # L x (B,n,C)
            x = layer(x, cos, sin)
        return self.norm(x)                                                  # (B,n,C)


class Llama215(nn.Module):
    """LLaMA 式整机（207M 定版）：wte + L x [pre-RMSNorm -> GQA | pre-RMSNorm -> SwiGLU]
    + 终态 RMSNorm + untied lm_head。前向接口与 Book2 GPT2 一致：idx (B,n) -> logits (B,n,V)；
    targets须已右移一位；返回fp32交叉熵在全部有效目标上的均值标量。

    state_dict 键与 HF LlamaForCausalLM 完全同名（model.embed_tokens.weight /
    model.layers.i.self_attn.q_proj.weight / ... / model.norm.weight / lm_head.weight），
    strict搬运检查键与形状，随后比较指定输入的数值；只支持本组件可表达的同型配置。
    cfg 缺省 = 215M 定版；init_std 缺省 = config initializer_range 0.02（Book2 式 (10.3) 的纪律）。
    """

    def __init__(self, cfg: LLaMAConfig = None, init_std: float | None = None):
        super().__init__()
        cfg = cfg or LLaMAConfig(**LLAMA_215M)
        validate_config(cfg)
        self.cfg = cfg
        self.model = Llama215Backbone(cfg)                                    # 命名对齐 HF（model.*）
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)  # untied（第 5 章附加一刀）
        std = cfg.initializer_range if init_std is None else init_std
        if not math.isfinite(std) or std < 0:raise ValueError("init_std必须为有限非负数")
        self.apply(lambda m: self._init(m, std))

    @staticmethod
    def _init(module, std):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def n_params(self, non_embedding=False):
        """参数量。non_embedding=True 时扣 wte 与 lm_head（untied 各扣一份）。"""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.model.embed_tokens.weight.numel() + self.lm_head.weight.numel()
        return n

    def forward(self, idx, targets=None):
        if idx.ndim != 2 or min(idx.shape) <= 0:raise ValueError("idx须为非空(B,n)")
        if targets is not None and targets.shape != idx.shape:raise ValueError("targets须与idx同形且预先右移")
        B, n = idx.shape                                               # (B,n)
        x = self.model(idx)                                            # (B,n,C)（backbone 含 RoPE 现算）
        logits = self.lm_head(x)                                       # (B,n,C)·(C,V) -> (B,n,V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),   # (B*n,V) fp32
                                  targets.reshape(-1))                             # (B*n,)
        return logits, loss


def hand_account(cfg: LLaMAConfig):
    """参数逐项手算（无 bias、untied、无 wpe）。返回 (明细 dict, 总数)——assert 的对账另一侧。"""
    validate_config(cfg)
    d, L, V, f = cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size, cfg.intermediate_size
    attn = d * d * 2 + d * cfg.head_dim * cfg.num_key_value_heads * 2      # q/o 全宽 + k/v 按 h_kv 缩（GQA）
    mlp = 3 * d * f                                                        # gate/up/down 三投影（SwiGLU）
    per_layer = attn + mlp + 2 * d                                         # + 两枚 RMSNorm（各只有 weight）
    total = per_layer * L + V * d * 2 + d                                  # untied：wte 与 lm_head 各一份 + 终态 norm
    detail = {"attn_per_layer": attn, "mlp_per_layer": mlp, "two_norms": 2 * d,
              "per_layer": per_layer, "layers": per_layer * L,
              "wte": V * d, "lm_head_untied": V * d, "final_norm": d, "total": total}
    return detail, total


# ---------------- 自测：参数逐项账 assert + 初始 loss 的数学自检 ----------------
def self_test():
    torch.manual_seed(SEED)
    cfg = LLaMAConfig(**LLAMA_215M)
    m = Llama215(cfg)
    detail, hand = hand_account(cfg)
    total = m.n_params()
    print("[config] cand1:", {k: LLAMA_215M[k] for k in LLAMA_215M})
    for k, v in detail.items():
        print(f"  {k:>18} = {v:,}")
    print(f"[参数账] 手算 = {hand:,}   实测 = {total:,}   non-emb = {m.n_params(True):,}")
    assert total == hand == 207_119_360, f"参数逐项对账失败：{total} != {hand} != 207,119,360"

    # 独立输出头：二阶近似E[loss]≈lnV+Var(logits)/2，并非精确等式。
    # 末层RMSNorm增益初值1，典型||h||²≈d，输出头每行方差s²，故logit方差≈ds²。
    # 目标logit样本均值可非零；不能称为信息泄漏，也不能将一次测量当严格期望。
    g = torch.Generator().manual_seed(SEED)
    x = torch.randint(0, cfg.vocab_size, (2, 256), generator=g)         # (2,256)
    xp, yp = x[:, :-1], x[:, 1:]                                        # next-token 目标（训练口径）
    with torch.no_grad():
        logits, loss = m(xp, yp)                                        # (2,255,V)
        z_t = logits.gather(-1, yp.unsqueeze(-1)).squeeze(-1)           # (2,255) 目标位 logit
        var = logits.var().item()
    ln_v = math.log(cfg.vocab_size)
    sig2_pred = cfg.hidden_size * cfg.initializer_range ** 2
    print(f"[初始 loss] 实测 {loss.item():.4f} | ln V = {ln_v:.4f} | 预言 ln V + d*s^2/2 = "
          f"{ln_v + sig2_pred / 2:.4f}（sigma^2 = {sig2_pred:.4f}，实测 logit 方差 {var:.4f}）"
          f" | E[z_target] = {z_t.mean().item():+.4f}（目标logit样本均值，期望近似为0）")
    assert abs(loss.item() - (ln_v + sig2_pred / 2)) < 0.05, "初始 loss 偏离 ln V + d*s^2/2 的预言"
    assert abs(z_t.mean().item()) < 0.05, "目标logit样本均值偏离本自测容差；检查尺度与目标构造"
    assert loss.item() - ln_v < 1.0, "初始 loss 距 ln V 超过 1 nat——初始化或前向有病"
    print("llama215自测通过（参数207,119,360；初始loss位于指定近似容差内）")


# ---------------- 可复查的参数搬运与指定输入数值检查 ----------------
def output_path(kind, out_name, out_dir=None):
    from pathlib import Path
    if Path(out_name).name != out_name or out_name in ('', '.', '..'):
        raise ValueError('out-name须为单个文件名')
    path=Path(out_dir or OUT_DIR)/f'llama215_{kind}_{out_name}.json'
    if path.exists():raise FileExistsError('已有同名产物，请更换out-name')
    path.parent.mkdir(parents=True,exist_ok=True)
    return path


def save_report(path,report):
    with path.open('x',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    if report['verdict']!='PASS':raise AssertionError(f'数值检查失败，详见{path}')
    return report


def parity_hf(out_name='run1',out_dir=None,config=None,batch=2,length=128):
    """CPU fp32指定输入对拍；config可缩小以检查接口，不将小档结果写成207M实测。"""
    from transformers import LlamaConfig,LlamaForCausalLM
    path=output_path('parity',out_name,out_dir)
    if batch<1 or length<1:raise ValueError('batch/length须正')
    c=dict(LLAMA_215M if config is None else config)
    torch.manual_seed(SEED)
    ours=Llama215(LLaMAConfig(**c)).eval()
    hf_cfg=LlamaConfig(**c,tie_word_embeddings=False,attention_bias=False,mlp_bias=False,attention_dropout=0.)
    hf_cfg._attn_implementation='eager'
    hf=LlamaForCausalLM(hf_cfg).eval()
    hf.load_state_dict(ours.state_dict(),strict=True,assign=True)
    g=torch.Generator().manual_seed(SEED)
    tokens=torch.randint(0,ours.cfg.vocab_size,(batch,length+1),generator=g)
    x,y=tokens[:,:-1],tokens[:,1:]  # 两边都用同一批next-token目标，而非x,x。
    started=time.perf_counter()
    with torch.no_grad():
        lo,loss=ours(x,y);lh=hf(x).logits
    delta=float((lo-lh).abs().max())
    argmax=float((lo.argmax(-1)==lh.argmax(-1)).float().mean())
    k=min(5,ours.cfg.vocab_size)
    same_set=(lo.topk(k,-1).indices.sort(-1).values==lh.topk(k,-1).indices.sort(-1).values).all(-1)
    topk=float(same_set.float().mean())
    hf_loss=float(F.cross_entropy(lh.reshape(-1,ours.cfg.vocab_size).float(),y.reshape(-1)))
    torch.manual_seed(SEED+1)
    other=Llama215(LLaMAConfig(**c)).eval()
    with torch.no_grad():sanity=float((other(x)[0]-lh).abs().max())
    loss_gap=abs(float(loss)-hf_loss)
    passed=all(math.isfinite(v) for v in (delta,argmax,topk,sanity,loss_gap)) and delta<1e-4 and argmax==1 and topk==1 and loss_gap<1e-4 and sanity>.1
    report=dict(seed=SEED,config=vars(ours.cfg),torch_version=str(torch.__version__),
                transformers_version=__import__('transformers').__version__,precision='CPU fp32 / HF eager',
                input_shape=list(x.shape),target_protocol='next-token shift by one',
                parity=dict(ours_params=ours.n_params(),max_abs_diff_logit=delta,argmax_agreement=argmax,
                            top5_set_agreement=topk,top_k=k,loss_abs_diff=loss_gap,
                            sanity_unshared_max_abs_diff=sanity,wall_sec=time.perf_counter()-started),
                verdict='PASS' if passed else 'FAIL')
    print(report['parity'])
    return save_report(path,report)


def compatible_config(hc):
    """仅接受本组件线可表达的无偏置、untied、默认RoPE Llama；未知变体不能静默近似。"""
    rope=getattr(hc,'rope_parameters',None) or getattr(hc,'rope_scaling',None) or {}
    if hc.model_type!='llama' or hc.tie_word_embeddings or getattr(hc,'attention_bias',False) or getattr(hc,'mlp_bias',False):
        raise ValueError('仅支持Llama无偏置untied配置')
    if hc.hidden_act!='silu' or rope.get('rope_type',rope.get('type','default'))!='default':
        raise ValueError('不支持此激活或RoPE缩放；不能只读theta便忽略其他字段')
    if getattr(hc,'head_dim',hc.hidden_size//hc.num_attention_heads)!=hc.hidden_size//hc.num_attention_heads:
        raise ValueError('本整机不支持独立head_dim')
    theta=rope.get('rope_theta',getattr(hc,'rope_theta',10000.))
    return LLaMAConfig(vocab_size=hc.vocab_size,hidden_size=hc.hidden_size,intermediate_size=hc.intermediate_size,
                      num_hidden_layers=hc.num_hidden_layers,num_attention_heads=hc.num_attention_heads,
                      num_key_value_heads=hc.num_key_value_heads,max_position_embeddings=hc.max_position_embeddings,
                      rope_theta=theta,rms_norm_eps=hc.rms_norm_eps,initializer_range=hc.initializer_range)


def parity_tinyllama(out_name='run1',snapshot=None,out_dir=None):
    """只读取显式本地快照；模型与tokenizer来自同一目录。逐步比较两个模型的贪心选择。"""
    from pathlib import Path
    from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
    if snapshot is None or not Path(snapshot).is_dir():raise ValueError('请用--snapshot提供已有本地完整快照，本程序不自动下载权重')
    path=output_path('tinyllama',out_name,out_dir);snapshot=Path(snapshot).resolve()
    hc=AutoConfig.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    cfg=compatible_config(hc)
    hf=AutoModelForCausalLM.from_pretrained(snapshot,dtype=torch.float32,local_files_only=True,
                                          trust_remote_code=False,attn_implementation='eager').eval()
    ours=Llama215(cfg).eval();ours.load_state_dict(hf.state_dict(),strict=True)
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    ids_in_vocab=list(tok.get_vocab().values())
    if not ids_in_vocab or min(ids_in_vocab)<0 or max(ids_in_vocab)>=cfg.vocab_size:
        raise ValueError('tokenizer ID超出模型词表')
    prompt='The capital of France is';ids=tok(prompt,return_tensors='pt').input_ids
    gen=ids.clone();deltas=[];argmax=[];greedy_match=[]
    # 两模型在相同前缀比较；若选择不同，即失败，不再把单边生成说成逐token一致。
    with torch.no_grad():
        for step in range(12):
            lo=ours(gen)[0];lh=hf(gen,use_cache=False).logits
            deltas.append(float((lo-lh).abs().max()))
            argmax.append(float((lo.argmax(-1)==lh.argmax(-1)).float().mean()))
            a=lo[:,-1].argmax(-1,keepdim=True);b=lh[:,-1].argmax(-1,keepdim=True)
            same=torch.equal(a,b);greedy_match.append(same)
            if not same:break
            gen=torch.cat([gen,a],dim=1)
    passed=len(greedy_match)==12 and all(greedy_match) and all(math.isfinite(d) and d<1e-4 for d in deltas) and all(a==1 for a in argmax)
    import hashlib
    fingerprints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir()
                  if p.is_file() and p.name in ('config.json','tokenizer.json','tokenizer.model','tokenizer_config.json','special_tokens_map.json')}
    report=dict(snapshot=str(snapshot),small_file_hashes=fingerprints,params=ours.n_params(),
                tensors=len(ours.state_dict()),config=vars(cfg),transformers_version=__import__('transformers').__version__,
                parity=dict(max_abs_diff_by_step=deltas,argmax_agreement_by_step=argmax,greedy_matches=greedy_match,
                            prompt=prompt,greedy_continuation_ours=tok.decode(gen[0,ids.shape[1]:])),
                note='固定提示、最多12次选择；不评价生成质量，不从短输出推断训练阶段',
                verdict='PASS' if passed else 'FAIL')
    print(report['parity'])
    return save_report(path,report)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description='207M整机自测、随机权重与显式本地快照对拍')
    ap.add_argument('--parity',action='store_true')
    ap.add_argument('--tinyllama',action='store_true')
    ap.add_argument('--snapshot',help='已有本地模型和tokenizer同一快照目录')
    ap.add_argument('--out-name',default='run1')
    ap.add_argument('--out-dir',default=OUT_DIR)
    args=ap.parse_args()
    if args.parity:parity_hf(args.out_name,args.out_dir)
    if args.tinyllama:parity_tinyllama(args.out_name,args.snapshot,args.out_dir)
    if not (args.parity or args.tinyllama):self_test()
