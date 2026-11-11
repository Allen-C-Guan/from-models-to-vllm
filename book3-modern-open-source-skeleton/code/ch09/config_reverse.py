# config_reverse.py —— Book3第9章：从固定配置快照核对结构与两本参数账。
# 运行：source env.sh && python <本文件路径> --model scout --out-name run1
# 只取config/index小文件；不下载权重，不执行远程代码。产物默认写工作区log。
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

SNAPSHOTS = {
    'scout': [('unsloth/Llama-4-Scout-17B-16E-Instruct', 'afd8e498c87bda51c7ea8ec68ea2f7c066e6340b'),
              ('RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic', 'fc1b4f4cea92cfc13570f504a209eb4d4ca81534')],
    'maverick': [('unsloth/Llama-4-Maverick-17B-128E-Instruct', '86c5ecb6f2fbddf604c60120125fb202e09f2556')],
}
CORE_FIELDS = ('hidden_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads',
               'head_dim', 'num_local_experts', 'num_experts_per_tok', 'intermediate_size',
               'intermediate_size_mlp', 'max_position_embeddings', 'vocab_size', 'rms_norm_eps',
               'tie_word_embeddings', 'moe_layers', 'no_rope_layers', 'use_qk_norm',
               'attention_bias', 'attention_chunk_size', 'floor_scale', 'attn_scale',
               'attn_temperature_tuning', 'rope_parameters')


def load_json(path, **source):
    data = Path(path).read_bytes()
    return json.loads(data), dict(source, path=str(Path(path).resolve()),
                                  sha256=hashlib.sha256(data).hexdigest())


def fetch(model, cache_dir, local_only=False):
    """按不可变revision读取所有列出的镜像；缺项即失败，不冒称完成交叉检查。"""
    from huggingface_hub import hf_hub_download
    result = []
    for repo, revision in SNAPSHOTS[model]:
        path = hf_hub_download(repo, 'config.json', revision=revision,
                               cache_dir=str(cache_dir), local_files_only=local_only)
        result.append(load_json(path, repo=repo, revision=revision, file='config.json'))
    return result


def effective(cfg):
    """读取当前安装库的生效配置；版本随报告保存，不假定等于序列化版本。"""
    from transformers.models.llama4 import Llama4Config
    if cfg.get('model_type') != 'llama4' or not isinstance(cfg.get('text_config'), dict):
        raise ValueError('本例只支持带text_config的Llama4整机配置')
    return Llama4Config.from_dict(copy.deepcopy(cfg)).text_config


def param_account(t):
    """式9.1/9.2：仅计本章无偏置、每层两个RMS增益、共享专家宽等于f的结构。"""
    d, L, h, hkv, dk = t.hidden_size, t.num_hidden_layers, t.num_attention_heads, t.num_key_value_heads, t.head_dim
    V, E, k, f, fm = t.vocab_size, t.num_local_experts, t.num_experts_per_tok, t.intermediate_size, t.intermediate_size_mlp
    if any(type(x) is not int or x <= 0 for x in (d, L, h, hkv, dk, V, E, k, f, fm)):
        raise ValueError('尺寸必须为正整数')
    if h % hkv or k > E or t.attention_bias:
        raise ValueError('查询头数须为KV头数的整数倍、k不超过E，且本公式不支持attention bias')
    layers = t.moe_layers
    if len(set(layers)) != len(layers) or any(type(i) is not int or not 0 <= i < L for i in layers):
        raise ValueError('专家层索引重复或越界')
    if len(t.no_rope_layers) != L or any(i not in (0, 1) for i in t.no_rope_layers):
        raise ValueError('旋转层布局与层数不符')
    attn = 2*d*h*dk + 2*d*hkv*dk  # Q/O可能不是d×d，保持一般头宽的整数账。
    moe = d*E + 3*d*f*(E+1)
    dense = 3*d*fm
    n = len(layers)
    ends = V*d*(2-int(t.tie_word_embeddings)) + d
    act_moe = attn + 2*d + d*E + 3*d*f*(1+k)
    act_dense = attn + 2*d + dense
    total = ends + L*(attn+2*d) + n*moe + (L-n)*dense
    active = ends + n*act_moe + (L-n)*act_dense
    return dict(attn_per_layer=attn, moe_ffn_per_layer=moe, dense_ffn_per_layer=dense,
                n_moe_layers=n, text_total=total, moe_layer_active=act_moe,
                dense_layer_active=act_dense, active_total=active, total_over_active=total/active)


def infer_rows(t):
    """字段到结构；旋转布尔语义已对照v4.52.4使用点。"""
    n = sum(t.no_rope_layers)
    rope = t.rope_parameters
    return [
        ('hidden/layers', [t.hidden_size,t.num_hidden_layers], '残差流宽/层数'),
        ('heads/KV/head_dim', [t.num_attention_heads,t.num_key_value_heads,t.head_dim], '每KV组对应h/hkv个查询头'),
        ('experts/top_k', [t.num_local_experts,t.num_experts_per_tok], '路由份数/每token选中份数，另有共享一份'),
        ('moe_layers', t.moe_layers, '索引从0开始；空表与缺省值分别处理'),
        ('FFN widths', [t.intermediate_size,t.intermediate_size_mlp], '专家/稠密层宽'),
        ('no_rope_layers', [n,len(t.no_rope_layers)-n], '旋转/不旋转；值1对应RoPE'),
        ('attention_chunk_size', t.attention_chunk_size, '旋转层按绝对位置分块，块内因果；NoPE层全局因果'),
        ('rope_parameters', rope, '保留全部阈值；Scout两阈值相等，不冒称有平滑过渡区'),
        ('context', t.max_position_embeddings, '声明窗口，不等于任务能力验证'),
        ('vocab/tied', [t.vocab_size,t.tie_word_embeddings], '词表大小/出入口是否共享'),
        ('use_qk_norm', t.use_qk_norm, '若启用，仅旋转层Q/K作无参数均方根缩放'),
        ('temperature/floor/scale', [t.attn_temperature_tuning,t.floor_scale,t.attn_scale], 'NoPE层按查询位置缩放Q，非YaRN常数缩放'),
        ('norm_eps/attention_bias', [t.rms_norm_eps,t.attention_bias], 'RMS epsilon/注意力偏置'),
    ]


def meta_count(cfg):
    """只构造meta模型，无参数存储；仍使用Python对象和张量元数据内存。"""
    import torch
    from transformers.models.llama4 import Llama4Config, Llama4ForConditionalGeneration
    with torch.device('meta'):
        model = Llama4ForConditionalGeneration(Llama4Config.from_dict(copy.deepcopy(cfg)))
    total = sum(p.numel() for p in model.parameters())
    vision = sum(p.numel() for name,p in model.named_parameters()
                 if 'vision_model.' in name or 'multi_modal_projector.' in name)
    if vision <= 0:
        raise ValueError('当前库视觉模块命名与计数约定不符，需重新核对')
    return dict(text_total=total-vision, vision_total=vision, grand_total=total)


def index_check(index, grand_total):
    """索引总字节不是张量元素数。只检查两字节假设的相容性，不推断dtype。"""
    total = index['metadata']['total_size']
    if type(total) is not int or total <= 0 or not isinstance(index['weight_map'], dict):
        raise ValueError('索引格式错误')
    compatible = total == 2*grand_total
    if not compatible:
        raise ValueError(f'索引字节{total}与参数数目{grand_total}的两字节假设不符，须检查dtype/缓冲/量化')
    return dict(total_size_bytes=total, compatible_with_two_bytes_per_parameter=True,
                dtype_verified=False, tensor_headers_read=False)


def draw_fig9_1(t, acct, meta, path):
    """Scout示意：完整主路径与视觉入口；不从粗粒度规模反推结构。"""
    if (t.hidden_size,t.num_hidden_layers,t.num_local_experts) != (5120,48,16):
        raise ValueError('图9.1仅适用于本章Scout配置')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8})
    fig,ax=plt.subplots(figsize=(10,6),dpi=300)
    ax.set(xlim=(0,12),ylim=(0,10));ax.axis('off')
    def box(x,y,w,h,label,color='#2a78d6'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04',facecolor='white',edgecolor=color))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=7.5)
    def arrow(x,y,X,Y):
        ax.annotate('',xy=(X,Y),xytext=(x,y),arrowprops=dict(arrowstyle='->',color='#52514e'))
    ax.text(6,9.7,'Llama 4 Scout: config + reference implementation',ha='center',fontsize=12)
    box(.2,8.3,3.1,.8,'Text IDs (B,n)\nEmbedding -> (B,n,5120)')
    box(.2,5.9,3.1,1.7,'Image tile (T,3,336,336)\n34-layer vision tower\n(T,577,1408), then remove CLS\nshuffle: (T,144,5632)', '#1baf7a')
    box(.2,4.2,3.1,1,'Adapter -> (T,144,4096)\nProject -> (T,144,5120)', '#1baf7a')
    arrow(1.75,5.9,1.75,5.2)
    box(4.3,8.3,7.3,.8,'Merge image features at image-token slots -> x (B,n,5120)')
    arrow(3.3,8.7,4.3,8.7);arrow(3.3,4.7,4.05,4.7);arrow(4.05,4.7,4.05,8.5);arrow(4.05,8.5,4.3,8.5)
    box(4.7,4.05,6.9,3.5,'48 blocks: z = x + Attention(RMSNorm(x))\ny = z + MoE(RMSNorm(z))\nQ (B,40,n,128); K/V (B,8,n,128) -> (B,n,5120)\nMoE: one routed expert + one shared; f=8192\n\nAttention pattern within these 48 blocks:\n1-3 of each 4: RoPE + QK RMS + chunked causal (8192)\n4th: NoPE + full causal + position-dependent Q scale')
    arrow(7.9,8.3,7.9,7.55)
    box(4.7,2.5,6.9,.8,'Final RMSNorm -> untied LM head\n(B,n,5120) -> logits (B,n,202048)')
    arrow(7.9,4.05,7.9,3.3)
    ax.text(6,1.7,f"Text {acct['text_total']:,} + vision/projector {meta['vision_total']:,}",ha='center',fontsize=8)
    ax.text(6,1.25,f"Total {meta['grand_total']:,}; active text accounting {acct['active_total']:,}",ha='center',fontsize=8)
    ax.text(6,.65,'Official model-card claims: 109B total / 17B active / ~40T training / 10M context',ha='center',fontsize=8)
    ax.text(6,.22,'Schematic; not a throughput or maximum-context test. Each image tile contributes 144 image features.',ha='center',fontsize=7,color='#52514e')
    fig.tight_layout();fig.savefig(path);plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=SNAPSHOTS,default='scout')
    ap.add_argument('--out-name',default='run1')
    ap.add_argument('--out-dir',type=Path,default=Path('log/book3-ch09'))
    ap.add_argument('--cache-dir',type=Path,default=Path('log/book3-ch09/hf-cache'))
    ap.add_argument('--local-files-only',action='store_true')
    ap.add_argument('--config',type=Path,help='显式本地底本；不自动联网补其它镜像')
    ap.add_argument('--compare-config',type=Path)
    ap.add_argument('--index',type=Path,help='与底本同revision的本地索引')
    ap.add_argument('--no-figure',action='store_true')
    a=ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',a.out_name) or a.out_name in ('.','..'):
        ap.error('out-name只允许简单文件名')
    if not a.config and (a.compare_config or a.index):
        ap.error('本地compare-config/index须配合本地config，防止混用revision')
    out=a.out_dir/f'config_reverse_{a.model}_{a.out_name}.json'
    figure=a.out_dir/f'fig-9-1-{a.out_name}.png'
    if out.exists() or (not a.no_figure and figure.exists()):
        raise FileExistsError('产物已存在，请选择新out-name')
    a.out_dir.mkdir(parents=True,exist_ok=True)
    if a.config:
        mirrors=[load_json(a.config,mode='explicit-local')]
        if a.compare_config:mirrors.append(load_json(a.compare_config,mode='explicit-local-comparison'))
        idx=load_json(a.index,mode='explicit-local-same-revision-required') if a.index else None
    else:
        mirrors=fetch(a.model,a.cache_dir,a.local_files_only)
        from huggingface_hub import hf_hub_download
        repo,revision=SNAPSHOTS[a.model][0]
        p=hf_hub_download(repo,'model.safetensors.index.json',revision=revision,
                          cache_dir=str(a.cache_dir),local_files_only=a.local_files_only)
        idx=load_json(p,repo=repo,revision=revision,file='model.safetensors.index.json')
    cfg=mirrors[0][0];t=effective(cfg)
    expected_experts = 16 if a.model == 'scout' else 128
    if t.num_local_experts != expected_experts:
        raise ValueError('config专家数与--model标签不符')
    acct=param_account(t);meta=meta_count(cfg)
    if acct['text_total']!=meta['text_total']:raise AssertionError('手算与当前库meta模型不一致')
    diffs=[]
    for other,_ in mirrors[1:]:
        ot=effective(other)
        diffs.extend(k for k in CORE_FIELDS if getattr(t,k)!=getattr(ot,k))
    if diffs:raise AssertionError('镜像结构字段不一致：'+str(diffs))
    import transformers
    report=dict(model=a.model,transformers_runtime=transformers.__version__,
                raw_configs=[c for c,_ in mirrors],sources=[s for _,s in mirrors],
                compared_config_count=len(mirrors),equal_fields=list(CORE_FIELDS) if len(mirrors)>1 else [],
                structure=infer_rows(t),account=acct,meta=meta,
                index=index_check(idx[0],meta['grand_total']) if idx else {'status':'not supplied'},
                index_source=idx[1] if idx else None,
                peripheral={'token_ids':{k:cfg.get(k,t.to_dict().get(k)) for k in
                            ['bos_token_id','pad_token_id','boi_token_index','eoi_token_index','image_token_index']},
                            'tokenizer_contents_verified':False,'generation_config_verified':False},
                limits=['参数一致不等于前向等价或能力验证','索引未逐张量核dtype','本例仅模型结构，不评测训练和推理速度'])
    if not a.no_figure:
        if a.model!='scout':raise ValueError('Maverick请使用--no-figure；图9.1仅Scout')
        draw_fig9_1(t,acct,meta,figure)
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('手算与meta模型一致；已记录配置数：',len(mirrors))
    print('索引：',report['index']);print('产物：',out)


if __name__=='__main__':main()
