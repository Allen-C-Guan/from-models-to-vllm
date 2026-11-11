# Book3第5章：配置参数账与五步核对。历史锚、当前计算和当前缓存实读分列；不自动联网。
# 在已激活环境中运行python code/ch05/config_infer.py --out-name 新名称；输出当前目录log。
# 本公式限无偏置、独立层、普通SwiGLU/RMS结构；不靠参数总量推作者动机或完整架构。
import argparse
import hashlib
import json
import os
from pathlib import Path
HF_HUB=os.environ.get('HF_HUB_CACHE',str(Path.cwd()/'log/huggingface/hub'))
OUT_DIR=str(Path.cwd()/'log/book3-ch05')
SNAPSHOTS={}

CONFIGS = {
    "gpt2_124M": dict(model_type="gpt2", architectures=["GPT2LMHeadModel"], vocab_size=50257,
                      n_embd=768, n_layer=12, n_head=12, n_positions=1024,
                      tie_word_embeddings=True, _src="本地缓存 openai-community/gpt2（Book2 遗产）",
                      _repo="models--openai-community--gpt2"),
    "llama1_7B": dict(model_type="llama", architectures=["LlamaForCausalLM"], vocab_size=32000,
                      hidden_size=4096, intermediate_size=11008, num_hidden_layers=32,
                      num_attention_heads=32, tie_word_embeddings=False, _transformers_version="(4.28 时代)",
                      _src="镜像 huggyllama/llama-7b + 论文 Table 2 交叉锚定", _repo="models--huggyllama--llama-7b"),
    "llama1_13B": dict(model_type="llama", architectures=["LlamaForCausalLM"], vocab_size=32000,
                       hidden_size=5120, intermediate_size=13824, num_hidden_layers=40,
                       num_attention_heads=40, tie_word_embeddings=False,
                       _src="论文 Table 2（13.0B 档）——缓存无实物，内嵌 canonical"),
    "llama1_33B": dict(model_type="llama", architectures=["LlamaForCausalLM"], vocab_size=32000,
                       hidden_size=6656, intermediate_size=17920, num_hidden_layers=60,
                       num_attention_heads=52, tie_word_embeddings=False,
                       _src="论文 Table 2（32.5B 档）——缓存无实物，内嵌 canonical"),
    "llama1_65B": dict(model_type="llama", architectures=["LlamaForCausalLM"], vocab_size=32000,
                       hidden_size=8192, intermediate_size=22016, num_hidden_layers=80,
                       num_attention_heads=64, tie_word_embeddings=False,
                       _src="镜像 huggyllama/llama-65b + 论文 Table 2 交叉锚定", _repo="models--huggyllama--llama-65b"),
    "llama2_7B": dict(model_type="llama", architectures=["LlamaForCausalLM"], vocab_size=32000,
                      hidden_size=4096, intermediate_size=11008, num_hidden_layers=32,
                      num_attention_heads=32, tie_word_embeddings=False,
                      _src="镜像 NousResearch/Llama-2-7b-hf（snapshot 8efe6c9b）",
                      _repo="models--NousResearch--Llama-2-7b-hf"),
    "llama2_70B": dict(model_type="llama", architectures=["LlamaForCausalLM"], vocab_size=32000,
                       hidden_size=8192, intermediate_size=28672, num_hidden_layers=80,
                       num_attention_heads=64, num_key_value_heads=8, tie_word_embeddings=False,
                       _src="镜像 NousResearch/Llama-2-70b-hf + 论文 Table 1（GQA=34B/70B 双档）"),
    "llama3_8B": dict(model_type="llama", architectures=["LlamaForCausalLM"], vocab_size=128256,
                      hidden_size=4096, intermediate_size=14336, num_hidden_layers=32,
                      num_attention_heads=32, num_key_value_heads=8, rope_theta=500000.0,
                      max_position_embeddings=8192, tie_word_embeddings=False,
                      _src="镜像 NousResearch/Meta-Llama-3-8B + 论文 Table 3",
                      _repo="models--NousResearch--Meta-Llama-3-8B"),
    "llama3_405B": dict(model_type="llama", architectures=["LlamaForCausalLM"], vocab_size=128256,
                        hidden_size=16384, intermediate_size=53248, num_hidden_layers=126,
                        num_attention_heads=128, num_key_value_heads=8, rope_theta=500000.0,
                        tie_word_embeddings=False,
                        _src="论文 Table 3（405B 档）——缓存无实物，内嵌 canonical"),
    "llama32_1B": dict(model_type="llama", architectures=["LlamaForCausalLM"], vocab_size=128256,
                       hidden_size=2048, intermediate_size=8192, num_hidden_layers=16,
                       num_attention_heads=32, num_key_value_heads=8, tie_word_embeddings=True,
                       _src="镜像 unsloth + NousResearch 双镜像一致（2024-09，冻结点外——仅 tie 级 config 事实）",
                       _repo="models--NousResearch--Llama-3.2-1B"),
    # 本册 207M 定版（plan/Book3 卷级大纲 cand1；llama_slots.LLAMA_215M 同款）
    "book3_207M": dict(model_type="llama(手写)", architectures=["(本册单轨整机)"], vocab_size=32000,
                       hidden_size=1024, intermediate_size=2816, num_hidden_layers=12,
                       num_attention_heads=16, num_key_value_heads=8, rope_theta=10000.0,
                       tie_word_embeddings=False, _src="本册定版 config（untied 出厂口径）"),
}

# 历史算术/实物对照锚点（证据等级见各条；调研 papers/03 §3.2 逐条落盘）
ANCHORS = {
    "gpt2_124M": 124_439_808,      # 本地 checkpoint 计数（Book2 ch4 已核，124M 档）
    "llama1_7B": 6_738_415_616,     # 论文 6.7B / HF 卡 6.74B 的精确底数（index total_size 含缓冲，见下）
    "llama1_13B": 13_015_864_320,   # 论文 13.0B
    "llama1_33B": 32_528_943_616,   # 论文 32.5B
    "llama1_65B": 65_285_660_672,   # 所列配置算术值；论文65.2B为另一精度的规格记法，不当精确独立锚
    "llama2_7B": 6_738_415_616,
    "llama2_70B": 68_976_648_192,   # 名义70B，不是将此精确数按十进制个位四舍五入
    "llama3_8B": 8_030_261_248,     # 四方逐位：手算=index/2=远程 header 和=meta 建模
    "llama3_405B": 405_853_388_800, # 所列405B配置的公式计数，不能从论文型号名直接得到此整数
    "llama32_1B": 1_235_814_400,    # 名义 1B（tied：词表账只计一份）
    "book3_207M": 207_119_360,      # 本册定版（llama_slots 自测同数）
}

# 调研期远程读数（HTTP Range 读 safetensors header，不下载权重；papers/03 §3.2）
REMOTE_ANCHORS = {
    "llama2_7B_index_total_size": 13_476_839_424,   # bytes（2字节权重 + 32 个 F32 inv_freq 缓冲）
    "llama3_8B_index_total_size": 16_060_522_496,   # bytes（291 张量，纯 bf16 参数）
    "llama3_8B_remote_header_sum": 8_030_261_248,   # 远程逐张量 numel 求和
}


def positive_ints(*values):
    if any(not isinstance(v,int) or isinstance(v,bool) or v<=0 for v in values):raise ValueError('尺寸须为正整数')


def llama_params(V,d,f,L,H,h_kv,tied,head_dim=None):
    """无偏置、每层两个RMS增益、独立层、普通SwiGLU的参数账；显式头维可不满足H*d_k=d。"""
    positive_ints(V,d,f,L,H,h_kv)
    if not isinstance(tied,bool):raise ValueError('tied须为布尔值')
    if H%h_kv:raise ValueError('本GQA结构要求查询头数可被KV头数整除')
    if head_dim is None:
        if d%H:raise ValueError('未给head_dim时要求d可被H整除')
        head_dim=d//H
    positive_ints(head_dim)
    vocab=V*d*(1 if tied else 2)
    attn=2*d*H*head_dim+2*d*h_kv*head_dim   # Q/O各d*(H*d_k)，K/V各d*(h_kv*d_k)
    mlp=3*d*f;layer=attn+mlp+2*d;total=vocab+L*layer+d
    return dict(V=V,d=d,f=f,L=L,H=H,h_kv=h_kv,tied=tied,d_k=head_dim,vocab_acct=vocab,
                attn_per_layer=attn,mlp_per_layer=mlp,per_layer=layer,total=total,vocab_share=vocab/total,
                scope='无偏置普通LLaMA式；额外投影、共享层或混合层需另算')


def gpt2_params(V,C,L,H,n_ctx):
    """绑定GPT2：QKV/输出/MLP均含偏置，两个LN/层及末端LN，学习式位置表。"""
    positive_ints(V,C,L,H,n_ctx)
    if C%H:raise ValueError('GPT2宽度须能被头数整除')
    wte=V*C;wpe=n_ctx*C;block=12*C*C+13*C;total=wte+wpe+L*block+2*C
    return dict(wte=wte,wpe=wpe,block=block,total=total,wte_share_tied=wte/total,
                untie_extra=wte,untie_extra_pct=wte/total,saving_fraction_of_untied=wte/(total+wte))


def file_info(path):
    p=Path(path);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def cached_file(name,fname):
    """只读本地缓存；多修订不任选首项，要求用--snapshot明确指定。"""
    repo=CONFIGS[name].get('_repo')
    if not repo:return None
    base=Path(HF_HUB)/repo/'snapshots';revision=SNAPSHOTS.get(name)
    if revision:
        if Path(revision).name!=revision or revision in ('.','..'):raise ValueError('无效snapshot修订')
        p=base/revision/fname;return str(p) if p.is_file() else None
    revisions=sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []
    if len(revisions)>1:raise ValueError(f'{name}缓存有多个修订，请用--snapshot {name}=修订号')
    p=revisions[0]/fname if revisions else None
    return str(p) if p is not None and p.is_file() else None


def cached_config(name):
    p=cached_file(name,'config.json');return json.loads(Path(p).read_text()) if p else None


def step1_family(report):
    rows={}
    for name in ('gpt2_124M','llama1_7B','llama2_7B','llama3_8B','llama32_1B'):
        embedded=CONFIGS[name];path=cached_file(name,'config.json');live=cached_config(name)
        source=live if live is not None else embedded
        # 本程序只演练所选历史几何，不把不同模型缓存默默当原模型验证。
        if live is not None:
            if live.get('model_type')!=embedded['model_type']:raise ValueError(f'{name}缓存模型家族不同或未声明')
            if live.get('architectures',embedded['architectures'])!=embedded['architectures']:raise ValueError(f'{name}缓存任务类与本例不同')
            default_tie=name.startswith('gpt2')
            if live.get('tie_word_embeddings',default_tie)!=embedded['tie_word_embeddings']:raise ValueError(f'{name}有效绑定设置与本例不同')
            if not name.startswith('gpt2'):
                if live.get('attention_bias',False) or live.get('mlp_bias',False):raise ValueError('本例参数式不含投影bias')
                if live.get('hidden_act','silu')!='silu':raise ValueError('本例限定SwiGLU，其他激活需另核实现')
                expected_kv=embedded.get('num_key_value_heads',embedded['num_attention_heads'])
                actual_kv=live.get('num_key_value_heads') or live.get('num_attention_heads')
                if actual_kv!=expected_kv:raise ValueError('缓存有效KV头数与本例不同')
                expected_hd=embedded.get('head_dim',embedded['hidden_size']//embedded['num_attention_heads'])
                actual_hd=live.get('head_dim') or live.get('hidden_size',0)//live.get('num_attention_heads',1)
                if actual_hd!=expected_hd:raise ValueError('缓存有效头维与本例不同')
            keys=('vocab_size','n_embd','n_layer','n_head','n_positions','tie_word_embeddings') if name.startswith('gpt2') else ('vocab_size','hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim','tie_word_embeddings')
            for key in keys:
                if key in live and key in embedded and live[key]!=embedded[key]:raise ValueError(f'{name}缓存字段{key}与本例内置几何不同')
        rows[name]=dict(model_type=source.get('model_type'),architectures=source.get('architectures'),
            tie_word_embeddings=source.get('tie_word_embeddings','未显式提供；须查实现默认'),
            transformers_version=source.get('transformers_version','未提供，不推断年代'),
            observed_absent_fields=[k for k in ('rope_theta','num_key_value_heads') if k not in source],
            source_kind='当前缓存实读' if live is not None else '内置历史配置摘录，未获取配置',
            file=file_info(path) if path else None,
            note='字段缺失须查对应实现默认；model_type/architectures不代替实际代码与权重核对')
        print(name,rows[name]['source_kind'])
    report['step1_family']=rows


def step2_blueprint(report):
    fields=[
      ('vocab_size','词嵌入/读出表行数起点，核实际权重形状'),
      ('hidden_size / head_dim','残差宽与显式头维分别读；仅未显式指定时按所据实现计算d/H'),
      ('intermediate_size','所选SwiGLU的门/值中间宽；三矩阵连接来自实现'),
      ('num_hidden_layers','所选独立Block数量；共享或混合层另核'),
      ('num_attention_heads / num_key_value_heads','查询头与KV头数量；缺项含义由实现默认决定'),
      ('rope_theta / rope_parameters','频率基底/规则；与KV头数无直接推导关系'),
      ('tie_word_embeddings','运行时参数共享；序列化可能保留别名或去重'),
      ('attention_bias / mlp_bias','标准注意力四投影、SwiGLU三投影的偏置；本公式假定关闭'),
      ('rms_norm_eps','加到均方后开方的稳定项'),
      ('max_position_embeddings','实现配置长度字段；不单独证明训练最大长度/推理上限'),
      ('bos/eos/pad_token_id','与分词器词条、数据整理及mask约定分别核对'),
    ]
    report['step2_field_map']=[dict(field=k,how_to_read=v) for k,v in fields]


def step3_accounts(report):
    accounts={}
    for name,c in CONFIGS.items():
        if name=='gpt2_124M':acc=gpt2_params(c['vocab_size'],c['n_embd'],c['n_layer'],c['n_head'],c['n_positions'])
        else:acc=llama_params(c['vocab_size'],c['hidden_size'],c['intermediate_size'],c['num_hidden_layers'],c['num_attention_heads'],c.get('num_key_value_heads',c['num_attention_heads']),c['tie_word_embeddings'],c.get('head_dim'))
        assert acc['total']==ANCHORS[name],(name,acc['total'],ANCHORS[name])
        accounts[name]=acc
    c3=CONFIGS['llama3_8B'];four={'hand':dict(value=accounts['llama3_8B']['total'],source='本次内置几何算术'),
        'remote_header_sum':dict(value=REMOTE_ANCHORS['llama3_8B_remote_header_sum'],source='历史远程头部记录，本次未发网络请求')}
    index_path=cached_file('llama3_8B','model.safetensors.index.json')
    if index_path:
        index=json.loads(Path(index_path).read_text());size=index['metadata']['total_size']
        four['index_total_size_div2']=dict(value=size//2,source='当前缓存index；仅在全2字节独立参数且无缓冲前提下解释',file=file_info(index_path))
        assert size%2==0 and size//2==accounts['llama3_8B']['total'],'缓存index不符合本例两字节前提，须逐张量追查'
    else:four['index_total_size_div2']=dict(status='未执行：无缓存index')
    try:
        import torch,transformers
        from transformers import LlamaConfig,LlamaForCausalLM
        with torch.device('meta'):
            model=LlamaForCausalLM(LlamaConfig(**{k:v for k,v in c3.items() if not k.startswith('_')}))
        count=sum(p.numel() for p in model.parameters());assert count==accounts['llama3_8B']['total']
        four['meta_modeling']=dict(value=count,source='当前库meta建模',torch=str(torch.__version__),transformers=transformers.__version__)
    except ImportError as exc:four['meta_modeling']=dict(status='未执行：缺可导入的模型库',reason=str(exc))
    report['step3_llama3_8b_four_way']=four
    # 此处是历史混合dtype案例的算术复核；未声称读到了本次远程张量。
    l2=accounts['llama2_7B']['total'];inv_elements=32*64;inv_bytes=inv_elements*4;size=REMOTE_ANCHORS['llama2_7B_index_total_size']
    assert 2*l2+inv_bytes==size
    report['step3_llama2_7b_invfreq']=dict(arch_params=l2,index_total_size=size,inv_freq_elements=inv_elements,inv_freq_bytes=inv_bytes,
        div2_phantom_extra=size//2-l2,source='历史远程读数的字节算术，2字节权重+F32缓冲；不是本次远程审计')
    tie=[]
    for name in ('gpt2_124M','llama1_7B','llama3_8B','llama32_1B'):
        c=CONFIGS[name];acc=accounts[name];d=c.get('hidden_size',c.get('n_embd'));V=c['vocab_size'];bound=c['tie_word_embeddings']
        total=acc['total'];Nt=total if bound else total-V*d;Nu=Nt+V*d;vocab=V*d*(1 if bound else 2)
        tie.append(dict(name=name,tie=bound,V=V,d=d,vocab_acct=vocab,total=total,share=vocab/total,
                        untie_extra=V*d,tied_total=Nt,untied_total=Nu,extra_over_tied=V*d/Nt,extra_share_in_untied=V*d/Nu))
    report['step3_tie_switch']=tie
    toy=[]
    for label,V in [('sp8k',8192),('LLaMA 32k',32000),('GPT2 50257',50257),('Llama3 128256',128256)]:
        a=llama_params(V,768,2048,12,12,12,True);u=llama_params(V,768,2048,12,12,12,False)
        toy.append(dict(label=label,V=V,embed=a['vocab_acct'],tied_total=a['total'],embed_share=a['vocab_share'],untied_share=u['vocab_share']))
    report['step3_skeleton_vocab_table']=toy;report['step3_accounts']=accounts
    report['step3_all_asserts_passed']=True;report['step3_scope']='内置几何算术和本机可得检查通过，不表示缺失证据已执行'
    print('内置参数算术：',len(accounts),'档通过；当前/历史证据分别列于报告')


def step4_periphery(report):
    result={};path=cached_file('llama2_7B','tokenizer.model')
    if path:
        import sentencepiece as spm
        sp=spm.SentencePieceProcessor(model_file=path);n=sp.get_piece_size()
        byte_ids=[i for i in range(n) if sp.is_byte(i)];special_ids=[i for i in range(n) if sp.is_control(i) or sp.is_unknown(i)]
        result['llama2_sp']=dict(file=file_info(path),piece_size=n,byte_ids=byte_ids,special_ids=special_ids,
            ordinary_pieces=n-len(byte_ids)-len(special_ids),pad_id=sp.pad_id(),pad_lookup=sp.piece_to_id('<pad>'),
            pad_lookup_piece=sp.id_to_piece(sp.piece_to_id('<pad>')),note='普通piece含基础字符，不等合并规则数；未命中字符串可返回unk')
    else:result['llama2_sp']=dict(status='未执行：本地无tokenizer.model')
    path=cached_file('llama3_8B','tokenizer.json')
    if path:
        import tokenizers
        tok=tokenizers.Tokenizer.from_file(path);added=tok.get_added_tokens_decoder()
        result['llama3_tokenizers']=dict(file=file_info(path),vocab_size=tok.get_vocab_size(),added_count=len(added),
          non_reserved={str(i):v.content for i,v in added.items() if 'reserved' not in v.content},
          reserved_count=sum('reserved' in v.content for v in added.values()),note='当前所选快照，不推广其他Llama3版本特殊词定义')
    else:result['llama3_tokenizers']=dict(status='未执行：本地无tokenizer.json')
    report['step4_periphery']=result


def step5_evidence(report):
    report['step5_evidence_ladder']=[
        dict(kind='配置/源码/权重形状',scope='互相核对结构；字段名和参数总量不足以唯一确定连接'),
        dict(kind='论文',scope='核作者披露的方法/动机/训练及实验条件；未披露留未知'),
        dict(kind='公开镜像与缓存',scope='写来源、revision及hash，不冒称与官方权重逐字节同一'),
        dict(kind='历史header记录',scope='本程序仅复算内置值，不发网络请求；当前重取需另核HTTP与分片身份'),
        dict(kind='机制推断',scope='与配置事实分开，不能仅凭参数占比还原作者唯一动机'),
    ]


def main():
    global HF_HUB,OUT_DIR,SNAPSHOTS
    ap=argparse.ArgumentParser(description='配置算术与可得本地证据；不联网，不把历史常量当当前审计')
    ap.add_argument('--out-name',default='run1');ap.add_argument('--out-dir',type=Path,default=Path(OUT_DIR))
    ap.add_argument('--cache-dir',type=Path,default=Path(HF_HUB));ap.add_argument('--snapshot',action='append',default=[],help='模型内置名称=明确缓存修订，可重复')
    a=ap.parse_args()
    if Path(a.out_name).name!=a.out_name or a.out_name in ('','.','..'):raise ValueError('out-name须单个名称')
    path=a.out_dir/f'config_infer_{a.out_name}.json'
    if path.exists():raise FileExistsError('不覆盖已有报告：'+str(path))
    HF_HUB=str(a.cache_dir);OUT_DIR=str(a.out_dir);SNAPSHOTS={}
    for pair in a.snapshot:
        name,revision=pair.split('=',1)
        if name not in CONFIGS or name in SNAPSHOTS:raise ValueError('未知或重复模型snapshot')
        SNAPSHOTS[name]=revision
    report=dict(meta=dict(scope='当前本地计算/缓存读取与历史锚分列，network_requests=0',cache_dir=HF_HUB))
    step1_family(report);step2_blueprint(report);step3_accounts(report);step4_periphery(report);step5_evidence(report)
    payload=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n';a.out_dir.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:f.write(payload)
    print(path)


if __name__=='__main__':main()
