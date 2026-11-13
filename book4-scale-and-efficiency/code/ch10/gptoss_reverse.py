# gptoss_reverse.py —— Book4第10章：配置、逻辑参数、存储载荷与KV账
# --no-net严格只读同目录configs/快照；在线档另外核HF API及20b同revision分片头。
# 历史整数锚不是官方逐位发布值；网络失败和未执行项会写入结果。
# --real-weights仅对本地唯一完整快照取token0首选一个专家，不是完整top4或整机前向。
# --figure默认none，启用时图另存log/book4-ch10/figures_<out-name>/；已有结果拒绝覆盖。
# 在工作区根目录source ./env.sh后运行本脚本；所有产物与缓存留log。
import argparse
import hashlib
from pathlib import Path
import json
import math
import os
import re
import struct
import time
import urllib.request

SEED = 20261002
REPO = os.getcwd()  # 工作区根目录；教学快照另按__file__定位
OUT_DIR = os.path.join(REPO, "log", "book4-ch10")
FIG_DIR = os.path.join(REPO, "log", "book4-ch10", "figures")
HF_HOME = os.environ.get("HF_HOME", os.path.join(REPO, "log", "huggingface"))

# 以下为本书历史公式/审计整数锚；模型卡Table1只给舍入概数。
HISTORICAL = {
    "openai/gpt-oss-20b": dict(total=20_914_757_184, mlp=19_119_145_728, attn=637_203_456,
                               ent=1_158_266_880, active_one=3_608_307_264,
                               active_two=4_187_440_704, ckpt_bytes=13_761_264_768),
    "openai/gpt-oss-120b": dict(total=116_829_156_672, mlp=114_714_874_368, attn=955_805_184,
                                ent=1_158_266_880, active_one=5_132_849_472,
                                active_two=5_711_982_912, ckpt_gib=60.8),
}
# MXFP4 存储审计锚点（20b 分片头逐张量，2026-10-04；块=物理字节，fp4 逻辑参数=块×2）
MXFP4_AUDIT_20B = dict(gu_blocks=6_370_099_200, gu_scales=398_131_200,
                       dn_blocks=3_185_049_600, dn_scales=199_065_600, bf16=1_804_459_584)
# 速览表 Book3 已核资产（指回锚：ch5 式 5.2 四方对拍 / ch8 表 8.2 / ch9 式 9.1-9.2 三方逐位）
BOOK3_ANCHORS = dict(llama3_8b=8_030_261_248, scout_text=107_769_861_120, scout_active=17_172_894_720,
                     maverick_text=400_711_848_960, maverick_active=17_184_691_200)


BUNDLED = Path(__file__).with_name("configs")
OFFLINE = False


def load_config(repo_id, fname=None):
    """离线只读随书快照；在线优先HF，失败回落并明确来源，不下载权重。"""
    fname = fname or "config.json"
    error = None
    if not OFFLINE:
        try:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(repo_id, fname))
            raw = path.read_bytes()
            return json.loads(raw), {"repo": repo_id, "mode": "HF online/cache",
                "path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    path = BUNDLED / (repo_id.replace("/", "_") + ".config.json")
    if not path.is_file():
        raise FileNotFoundError(f"无随书配置：{repo_id}；在线错误={error}")
    raw = path.read_bytes()
    return json.loads(raw), {"repo": repo_id, "mode": "bundled snapshot", "online_error": error,
                           "sha256": hashlib.sha256(raw).hexdigest()}


# ---------------- STEP1/2 认家族 + 结构推断表 ----------------
def infer_rows(c):
    """表 10.1 的机器可读版：字段 → 20b/120b 值 → 部件/读法（等级注记写进正文表）。"""
    d, L, E = c["hidden_size"], c["num_hidden_layers"], c["num_local_experts"]
    hd = c["num_attention_heads"] * c["head_dim"]
    lt = c["layer_types"]
    return [
        ("model_type / architectures / transformers_version",
         f'gpt_oss / GptOssForCausalLM / {c["transformers_version"]}',
         "认家族三件套；配置中记录的实现版本线索，不证明精确写入日期"),
        ("hidden_size", str(d), "残差流宽d；两档保持宽度，增加专家数与层数，不据此推断训练配方"),
        ("num_hidden_layers", f'{L}（论文 §2.2 同值）', "Block 数 L"),
        ("num_local_experts / num_experts_per_tok", f'{E} / {c["num_experts_per_tok"]}',
         f'专家份数 E / 每 token 点燃 k（论文 "top-4"；html 渲染损伤为 "top-44"，以 config 为准）'),
        ("intermediate_size", str(c["intermediate_size"]), "专家宽f=d；逻辑三矩阵为方阵，融合gate/up不是；此模型无稠密FFN层"),
        ("num_attention_heads / num_key_value_heads / head_dim",
         f'{c["num_attention_heads"]} / {c["num_key_value_heads"]} / {c["head_dim"]}',
         f"GQA 8 组；h·d_k = {hd} > d = {d}：q/o 是 {d}↔{hd} 的矩形投影（按实际头维计数）"),
        ("attention_bias", str(c["attention_bias"]), "四个注意力投影全带偏置（分片头证实 o_proj.bias 也在）；属于所核模型之间的结构差异"),
        ("layer_types", f'{len(lt)} 项，sliding/full 逐层交替（首层 sliding）', "1:1 交错注意力（非传闻的 3:1）"),
        ("sliding_window", str(c["sliding_window"]), "滑窗带宽 128 token（只作 config 事实，机制→Book5 第 2 章）"),
        ("rope_theta / rope_scaling",
         f'{c["rope_theta"]:g} / yarn factor {c["rope_scaling"]["factor"]:g}（4096→131072）',
         "RoPE 底数 1.5e5 + YaRN 32×（缩放常数住实现里：0.1·ln32+1≈1.3466）"),
        ("max_position_embeddings", f'{c["max_position_embeddings"]:,}', "声明上下文 128k"),
        ("swiglu_limit", str(c["swiglu_limit"]), "SwiGLU 激活限幅常数（config 字段与实现硬编码冗余并置）"),
        ("router_aux_loss_coef / output_router_logits",
         f'{c["router_aux_loss_coef"]} / {c["output_router_logits"]}',
         "默认不输出router logits；HF5.14.1启用该输出并提供labels时会加aux，原训练使用情况未知"),
        ("quantization_config", 'mxfp4；不转换=self_attn/router/embed/lm_head', "专家投影用MXFP4；历史检查点其余张量含偏置/norm为BF16，运行dtype另看实现"),
        ("vocab_size / tie_word_embeddings", f'{c["vocab_size"]:,} / {c["tie_word_embeddings"]}',
         "o200k_harmony 词表，untied 出入口两份"),
        ("rms_norm_eps", str(c["rms_norm_eps"]), "pre-RMSNorm（fp32 上抛的数值路径差异，不在结构）"),
        ("（config 无字段）self_attn.sinks", f"每头 1 可学标量（{L}×{c['num_attention_heads']} 枚 BF16）",
         "权重审计+源码双跳才读得到（分片头：sinks [64]/层）"),
        ("（config 无字段）experts 偏置", "gate_up/down 两级都带偏置", "分片头证实（HF 键位：experts.gate_up_proj_bias 等）"),
        ("（config 无字段）QK-Norm", "无", "所核双实现（OpenAI参考代码与transformers5.14.1）注意力通路均无 q/k 归一化——负结论"),
    ]


# ---------------- STEP3 三方逐位参数账 ----------------
def gptoss_account(c):
    """式 10.1/10.2 手算：总账与活账两本（gpt-oss 特化=注意力偏置+sink、专家偏置）。

    返回逐项整数账（单位=参数元素个数）。
    """
    d, L = c["hidden_size"], c["num_hidden_layers"]
    h, h_kv, d_k = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    E, k, f, V = c["num_local_experts"], c["num_experts_per_tok"], c["intermediate_size"], c["vocab_size"]
    hd, kvd = h * d_k, h_kv * d_k
    attn_w = 2 * d * hd + 2 * d * kvd          # q/o 全宽(矩形 2880↔4096) + k/v 按 8 KV 组缩
    attn_b = hd + 2 * kvd + d                  # q/k/v/o 四偏置
    sinks = h                                   # 每头 1 枚可学标量
    router = d * E + E                          # router 权重 + 常规偏置（同路受梯度）
    exp_w, exp_b = E * 3 * d * f, E * (2 * f + d)   # 专家三投影权重 + gate_up/down 两级偏置
    norms = 2 * d
    per_layer = attn_w + attn_b + sinks + router + exp_w + exp_b + norms
    total = 2 * V * d + L * per_layer + d       # untied 出入口 + L 层 + 收尾 norm
    # 活账·本册口径=含 lm_head、不含输入 embedding lookup（数值上=官方「单份」口径）
    act_one = V * d + L * (attn_w + attn_b + sinks + router + k * (3 * d * f + 2 * f + d) + norms) + d
    return dict(d=d, L=L, E=E, k=k, f=f, V=V, attn_w=attn_w, attn_b=attn_b, sinks=sinks,
                router=router, exp_w=exp_w, exp_b=exp_b, norms=norms, per_layer=per_layer,
                total=total, active_one=act_one, active_two=act_one + V * d,
                mlp_family=L * (exp_w + exp_b + router), attn_family=L * (attn_w + attn_b + sinks),
                ent_family=2 * V * d,
                active_layer=per_layer - exp_w - exp_b + k * (3 * d * f + 2 * f + d))


def hf_api_safetensors(repo_id):
    """HF API 对拍方：total 与 dtype 分布（gpt-oss 的 U8 计数=fp4 逻辑参数，块×2、不含 scale）。"""
    r = json.load(urllib.request.urlopen(
        f"https://huggingface.co/api/models/{repo_id}?expand[]=safetensors", timeout=30))
    st = r.get("safetensors", {})
    return dict(total=st.get("total"), parameters=st.get("parameters"))


def read_range(url, start, end):
    """服务器必须返回指定206范围；在读取正文前拒绝整片200，最多读取请求字节+1。"""
    length = end - start + 1
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=30) as handle:
        if handle.status != 206:
            raise ValueError("服务器未遵守Range，拒绝读取整片权重")
        content_range = handle.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes {start}-{end}/"):
            raise ValueError(f"Content-Range不符：{content_range}")
        data = handle.read(length + 1)
        if len(data) != length:raise ValueError("Range长度不符")
        return data


def shard_header_audit(repo_id):
    """读取同一revision的索引/头部，核对键和数据字节区间；不读取张量载荷。"""
    info = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo_id}",timeout=30))
    revision = info["sha"]
    base = f"https://huggingface.co/{repo_id}/resolve/{revision}/"
    idx = json.load(urllib.request.urlopen(base+"model.safetensors.index.json",timeout=30))
    shards = sorted(set(idx["weight_map"].values()))
    fam, dtype_ct, grand, payload, headers = {}, {}, 0, 0, {}
    widths={"U8":1,"I8":1,"BF16":2,"F16":2,"F32":4,"F64":8,"I32":4,"I64":8}
    for shard in shards:
        if Path(shard).name!=shard:raise ValueError("非法分片名")
        url=base+shard
        n=struct.unpack("<Q",read_range(url,0,7))[0]
        if not 2<=n<=64*1024*1024:raise ValueError("头部长度异常")
        hdr=json.loads(read_range(url,8,7+n))
        declared={k for k,v in idx["weight_map"].items() if v==shard}
        if set(hdr)-{"__metadata__"} != declared:raise ValueError("索引与分片键不一致")
        intervals=[]
        for key,meta in hdr.items():
            if key=="__metadata__":continue
            numel=math.prod(meta["shape"]);a,b=meta["data_offsets"]
            if a<0 or b-a!=numel*widths[meta["dtype"]]:raise ValueError("dtype/形状/字节区间不符")
            intervals.append((a,b));payload+=b-a;grand+=numel
            family=re.sub(r"\d+","N",key);fam[family]=fam.get(family,0)+numel
            dtype_ct[meta["dtype"]]=dtype_ct.get(meta["dtype"],0)+numel
        intervals.sort()
        if any(a<prev_b for (_,prev_b),(a,_) in zip(intervals,intervals[1:])):raise ValueError("载荷区间重叠")
        headers[shard]=n
    return dict(revision=revision,shards=shards,grand=grand,families=fam,dtypes=dtype_ct,
                payload_bytes=payload,header_bytes=headers)


def mxfp4_storage(u8_blocks, u8_scales, logical_fp4):
    """4.25 bits/param 自证：物理字节（块+scale）×8 bits ÷ fp4 逻辑参数数。"""
    bits = (u8_blocks + u8_scales) * 8 / logical_fp4
    return bits


def kv_account(c):
    """KV 元素账（Book3 ch6 公式直系；单位=元素数）：2·L·h_kv·d_k，另报 bf16 字节。"""
    L, h_kv, d_k = c["num_hidden_layers"], c["num_key_value_heads"], c["head_dim"]
    elems = 2 * L * h_kv * d_k
    return dict(elements=elems, kib_bf16=elems * 2 / 1024)


# ---------------- STEP4 跨厂商活账口径 ----------------
def _mla_attn(c):
    """修正版 MLA 每层注意力账（含 kv_b_proj 上投影与两枚 RMSNorm——P8 销项正身口径）。"""
    d, H = c["hidden_size"], c["num_attention_heads"]
    ql, kvl = c.get("q_lora_rank"), c["kv_lora_rank"]
    qn, qr, vh = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
    q_path = (ql * d + ql + ql * H * (qn + qr)) if ql else d * H * (qn + qr)
    kv_path = d * (kvl + qr) + kvl + H * (qn + vh) * kvl
    return q_path + kv_path + d * H * vh


def ds_moe_account(c):
    """DSV2-Lite/V3 总账+活账（本册口径=含 lm_head 不含输入 lookup；双份另报）。

    口径注：noaux偏置（主模型58层×256，MTP另有256；FP32）与 FP8 scale_inv 同记辅助张量侧、
    不入主账——与 notes/08 §2.2 的 F32 分解归档及登记值 671,026,404,352/15,706,484,224 逐位一致。
    """
    d, L, V = c["hidden_size"], c["num_hidden_layers"], c["vocab_size"]
    E, sh, k = c["n_routed_experts"], c["n_shared_experts"], c["num_experts_per_tok"]
    fm, fd, nd = c["moe_intermediate_size"], c["intermediate_size"], c["first_k_dense_replace"]
    attn = _mla_attn(c)
    moe_layer = attn + E * 3 * d * fm + sh * 3 * d * fm + d * E + 2 * d
    dense_layer = attn + 3 * d * fd + 2 * d
    total = 2 * V * d + nd * dense_layer + (L - nd) * moe_layer + d
    moe_active = attn + k * 3 * d * fm + sh * 3 * d * fm + d * E + 2 * d
    act_one = V * d + nd * dense_layer + (L - nd) * moe_active + d
    return dict(total=total, active_one=act_one, active_two=act_one + V * d, attn_per_layer=attn,
                kv_token=L * (c["kv_lora_rank"] + c["qk_rope_head_dim"]))


def qwen_moe_account(c):
    """Qwen3MoE完整参数账，计入每层2d残差norm及2d_k的共享QK-Norm。"""
    d, L, V = c["hidden_size"], c["num_hidden_layers"], c["vocab_size"]
    H, KV, d_k = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    E, k, f = c["num_experts"], c["num_experts_per_tok"], c["moe_intermediate_size"]
    attn = 2 * d * (H * d_k) + 2 * d * (KV * d_k)
    total = 2 * V * d + L * (attn + E * 3 * d * f + d * E + 2*d + 2*d_k) + d
    act_one = V * d + L * (attn + k * 3 * d * f + d * E + 2*d + 2*d_k) + d
    return dict(total=total, active_one=act_one, active_two=act_one + V * d)


def gemma_text_account(c):
    """Gemma2/3 文本栈（三明治四 norm；Gemma3 计 q_norm/k_norm——P5 双镜像已权重级印证）。"""
    t = c.get("text_config", c)
    d, L = t["hidden_size"], t["num_hidden_layers"]
    H, KV, d_k = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    f, V = t["intermediate_size"], t["vocab_size"]
    tied = t.get("tie_word_embeddings", True)
    attn = 2 * d * (H * d_k) + 2 * d * (KV * d_k)
    qkn = 2*d_k if t.get("model_type") == "gemma3_text" else 0  # 各头复用同一d_k增益，不乘头数
    total = (V * d if tied else 2 * V * d) + L * (attn + qkn + 3 * d * f + 4 * d) + d
    return dict(total=total, tied=tied, attn_rect=f"h·d_k={H * d_k} vs d={d}")


def llama3_account(d, L, H, KV, d_k, f, V):
    """Llama3 系（GQA：q/o 全宽、k/v 按 KV 组缩；无偏置、双 norm、untied）。"""
    attn = 2 * d * (H * d_k) + 2 * d * (KV * d_k)
    return 2 * V * d + L * (attn + 3 * d * f + 2 * d) + d


# ---------------- 条件性真权重：MXFP4 解包与单专家前向 ----------------
FP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
           -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def my_mxfp4_dequant(blocks, scales):
    """有限E8M0块解包：(E,R,G,16)U8 + (E,R,G)U8 -> (E,R,32G)FP32。
    低/高半字节依次为偶/奇位置；255是NaN编码，本有限数据教学入口显式拒绝。
    """
    import torch
    if blocks.dtype!=torch.uint8 or scales.dtype!=torch.uint8 or blocks.ndim!=4 or blocks.shape[-1]!=16:
        raise ValueError("需要(E,R,G,16)uint8 blocks及uint8 scales")
    if blocks.shape[:-1]!=scales.shape or blocks.device!=scales.device:raise ValueError("blocks/scales形状或设备不同")
    if (scales==255).any():raise ValueError("本有限数据入口不处理E8M0 NaN编码255")
    lut=torch.tensor(FP4_LUT,dtype=torch.float32,device=blocks.device)
    lo=lut[(blocks&15).long()];hi=lut[(blocks>>4).long()]
    vals=torch.stack([lo,hi],-1).reshape(*blocks.shape[:-1],32)
    vals=torch.ldexp(vals,scales.to(torch.int32).unsqueeze(-1)-127)
    return vals.reshape(*blocks.shape[:-2],-1)


def find_local_snapshot(repo_id):
    """仅接受索引中所有分片齐备的唯一snapshot，不凭存在某个分片断言完整。"""
    root=Path(HF_HOME)/"hub"/("models--"+repo_id.replace("/","--"))/"snapshots"
    valid=[]
    for snap in root.glob("*"):
        index=snap/"model.safetensors.index.json"
        if not index.is_file():continue
        mapping=json.loads(index.read_text()).get("weight_map",{})
        if mapping and all(Path(s).name==s and (snap/s).is_file() for s in set(mapping.values())):valid.append(snap)
    if len(valid)>1:raise ValueError("存在多个完整snapshot，请明确整理目标revision")
    return str(valid[0]) if valid else None


def real_weights_demo(verbose=True):
    """只读取token0所选一个专家，核两投影解包；输出是一个门控贡献，不是完整MoE。"""
    import torch
    snap=find_local_snapshot("openai/gpt-oss-20b")
    if snap is None:
        return {"executed":False,"reason":"本地未找到完整唯一gpt-oss-20b snapshot"}
    from safetensors import safe_open
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors
    mapping=json.loads((Path(snap)/"model.safetensors.index.json").read_text())["weight_map"]
    prefix="model.layers.0.mlp."
    def tensor(name,expert=None):
        key=prefix+name
        with safe_open(Path(snap)/mapping[key],framework="pt") as handle:
            return handle.get_tensor(key) if expert is None else handle.get_slice(key)[expert:expert+1]
    rw,rb=tensor("router.weight").float(),tensor("router.bias").float()
    value=torch.randn(1,rw.shape[1],generator=torch.Generator().manual_seed(SEED))
    top=(value@rw.T+rb).topk(4,-1);gates=top.values.softmax(-1);expert=int(top.indices[0,0])
    t0=time.perf_counter();weights={};diffs={}
    for name in ("gate_up_proj","down_proj"):
        blocks=tensor("experts."+name+"_blocks",expert);scales=tensor("experts."+name+"_scales",expert)
        ours=my_mxfp4_dequant(blocks,scales).transpose(1,2).contiguous()
        ref=convert_moe_packed_tensors(blocks,scales,dtype=torch.bfloat16).float()
        delta=(ours.to(torch.bfloat16).float()-ref).abs().max().item()
        if not math.isfinite(delta) or delta!=0:raise AssertionError(f"{name}解包不一致：{delta}")
        weights[name]=ours[0];diffs[name]=delta
    gu=value@weights["gate_up_proj"]+tensor("experts.gate_up_proj_bias",expert)[0].float()
    gate,up=gu[...,::2].clamp(max=7),gu[...,1::2].clamp(-7,7)
    hidden=gate*torch.sigmoid(1.702*gate)*(up+1)
    result=(hidden@weights["down_proj"]+tensor("experts.down_proj_bias",expert)[0].float())*gates[0,0]
    if not torch.isfinite(result).all():raise AssertionError("单专家贡献非有限")
    report={"executed":True,"snapshot":snap,"expert_index":expert,"input_shape":list(value.shape),
        "dequant_bf16_max_diff":diffs,"output_shape":list(result.shape),"output_absmax":result.abs().max().item(),
        "scope":"token0的首选专家门控贡献，不是完整top4合并输出","wall_s":time.perf_counter()-t0}
    if verbose:print(report)
    return report


# ---------------- 图 10.1 / 图 10.2 ----------------
def _figure_base():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    def box(ax,x,y,w,h,text,color="#2a78d6",size=9,dashed=False):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.006",facecolor="white",
                     edgecolor=color,lw=1.5,linestyle="--" if dashed else "-"))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=size,color="#202020",linespacing=1.4)
    def arrow(ax,x,y,xx,yy):
        ax.annotate("",xy=(xx,yy),xytext=(x,y),arrowprops={"arrowstyle":"->","color":"#52514e","lw":1.5})
    return plt,box,arrow


def draw_fig10_1(acct20,path):
    """完整Block内含注意力与MoE两条残差；来源说明不串入前向。"""
    if Path(path).exists():raise FileExistsError(path)
    plt,box,arrow=_figure_base()
    fig,ax=plt.subplots(figsize=(10,7));ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    ax.text(.5,.98,"gpt-oss-20b: forward structure and evidence",ha="center",fontsize=14,fontweight="bold")
    box(ax,.06,.83,.56,.085,"Embedding: (B,n) -> (B,n,2880)")
    box(ax,.06,.61,.56,.175,"RMSNorm -> Attention -> add residual\n"
        "Q (B,64,n,64); K/V (B,8,n,64)\n"
        "Output (B,n,2880); four projections with bias\n"
        "1 learned sink / query head; no QK-Norm\n"
        "Sliding-128 / full alternate, first layer sliding",size=8.6)
    box(ax,.06,.415,.56,.15,"RMSNorm -> MoE -> add residual\n"
        "32 experts, top-4; width 2880; no shared expert\n"
        "Router (Bn,2880) -> (Bn,32), with bias\n"
        "Selected-logit softmax -> output (B,n,2880)",color="#1baf7a",size=8.8)
    box(ax,.06,.265,.56,.10,"Final RMSNorm -> vocabulary head\n(B,n,2880) -> logits (B,n,201088)")
    for y,yy in [(.83,.785),(.61,.565),(.415,.365)]:arrow(ax,.34,y,.34,yy)
    ax.plot([.025,.025],[.415,.785],color="#52514e")
    ax.text(.015,.6,"repeat both sublayers 24 times",rotation=90,va="center",ha="right",fontsize=8)
    box(ax,.69,.76,.28,.15,"Model-card values\n20.91B total / 3.61B active\nActive includes output head,\nexcludes input embedding","#eb6834",8.3,True)
    box(ax,.69,.54,.28,.16,"Published expert matrices:\nMXFP4 blocks + E8M0 scales\n32 values share 1 scale\n4.25 bits / matrix value","#eb6834",8.4,True)
    box(ax,.69,.32,.28,.16,"Other published tensors: BF16\nRuntime dtype is a separate choice\nYaRN factor 32, theta 150000\nDeclared context 131072","#eb6834",8.0,True)
    ax.text(.06,.18,f"Exact logical count: {acct20['total']:,}; active convention: {acct20['active_one']:,}",
            fontsize=10,color="#202020")
    ax.text(.06,.12,"Sources: official config + reference code; 20b shard headers; model card v1.\n"
            "Metadata agreement does not verify full-model numerical inference.",fontsize=9,va="top",linespacing=1.6)
    fig.tight_layout();fig.savefig(path,dpi=300,facecolor="white");plt.close(fig)


def draw_fig10_2(path):
    """区分gpt-oss的logit偏置和V3的选择偏置，箭头按前向顺序。"""
    if Path(path).exists():raise FileExistsError(path)
    plt,box,arrow=_figure_base()
    fig,axes=plt.subplots(1,2,figsize=(10,7))
    data=[
      ("gpt-oss",32,4,"#2a78d6",[
       "Logits = X W^T + ordinary bias\n(M,d) -> (M,E); E=32 / 128",
       "Select top-4 logits\nindices and logits: (M,4)",
       "Weights = softmax(selected logits)\n(M,4); each row sums to 1",
       "Weighted sum of 4 expert outputs\n(M,4,d) -> (M,d); no shared branch"],
       "Original training balance recipe: not fully disclosed\nHF 5.14.1 has an opt-in auxiliary-loss path"),
      ("DeepSeek-V3",256,8,"#1baf7a",[
       "Raw affinities s = sigmoid(X W^T)\n(M,d) -> (M,256); no ordinary logit bias",
       "Selection scores = s + correction bias\n8 groups -> 4 groups -> top-8 experts",
       "Gather RAW s at selected indices\nnormalize selected s, then multiply by 2.5",
       "Weighted sum of 8 routed experts\nplus 1 shared expert; output (M,d)"],
       "Correction bias affects selection only\nTraining also retains small-weight SeqBal")]
    for ax,(name,E,k,color,lines,note) in zip(axes,data):
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
        ax.text(.5,.97,name,ha="center",fontsize=15,fontweight="bold")
        ys=[.76,.56,.36,.16]
        for y,line in zip(ys,lines):box(ax,.06,y,.88,.15,line,color,9)
        for upper,lower in zip(ys,ys[1:]):arrow(ax,.5,upper,.5,lower+.15)
        ax.text(.5,.09,note,ha="center",va="top",fontsize=8.2,linespacing=1.4)
    fig.text(.5,.015,"M = B x n tokens. Bias placement changes selection and gradient semantics.",ha="center",fontsize=9)
    fig.tight_layout();fig.savefig(path,dpi=300,facecolor="white");plt.close(fig)


# ---------------- 主流程 ----------------
def main():
    global OFFLINE,FIG_DIR
    ap=argparse.ArgumentParser(description="gpt-oss配置、参数、存储与KV复算；证据来源逐项记录")
    ap.add_argument("--out-name",default="run1")
    ap.add_argument("--no-net",action="store_true",help="只读随书配置，跳过网络审计")
    ap.add_argument("--real-weights",action="store_true",help="只用本地完整快照演示一个专家")
    ap.add_argument("--figure",choices=["all","structure","routing","none"],default="none")
    args=ap.parse_args();OFFLINE=args.no_net
    if Path(args.out_name).name!=args.out_name or args.out_name in {".",".."}:ap.error("out-name须为单个后缀")
    target=Path(OUT_DIR)/f"gptoss_reverse_{args.out_name}.json"
    if target.exists():raise FileExistsError(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    FIG_DIR=str(target.parent/f"figures_{args.out_name}")
    out={"offline":OFFLINE,"sources":{},"network_audit":{},"limitations":[]}
    def config(repo):
        obj,source=load_config(repo);out["sources"][repo]=source
        return obj

    # 1/2：认家族与字段比较。差异只针对这两份配置，不推出相同权重或训练配方。
    cfgs={r:config(r) for r in HISTORICAL}
    c20,c120=cfgs["openai/gpt-oss-20b"],cfgs["openai/gpt-oss-120b"]
    out["config_diff"]=sorted(k for k in set(c20)|set(c120) if c20.get(k)!=c120.get(k))
    out["structure_rows"]=infer_rows(c20)
    print("STEP1/2 配置差异：",out["config_diff"])

    # 3：公式与历史整数锚核对，官方Table1的舍入值单列在正文。
    accts={r:gptoss_account(c) for r,c in cfgs.items()}
    for repo,a in accts.items():
        old=HISTORICAL[repo]
        for key,field in [("total","total"),("mlp_family","mlp"),("attn_family","attn"),("ent_family","ent"),
                          ("active_one","active_one"),("active_two","active_two")]:
            if a[key]!=old[field]:raise AssertionError(f"{repo}:{key}与历史整数锚不同")
        if a["active_layer"]*a["L"]+a["V"]*a["d"]+a["d"]!=a["active_one"]:
            raise AssertionError("单层活账与整机活账不一致")
        print(f"STEP3 {repo}: total={a['total']:,}, active={a['active_one']:,}")
    out["account"]=accts
    audit20=None
    if not OFFLINE:
        for repo in cfgs:
            try:
                result=hf_api_safetensors(repo)
            except (OSError,ValueError) as exc:
                out["network_audit"][repo]={"performed":False,"error":str(exc)}
            else:
                out["network_audit"][repo]={"performed":True,**result,
                    "matches_formula":result["total"]==accts[repo]["total"]}
        try:
            audit20=shard_header_audit("openai/gpt-oss-20b")
        except (OSError,ValueError) as exc:
            out["network_audit"]["headers20b"]={"performed":False,"error":str(exc)}
        else:
            out["network_audit"]["headers20b"]={"performed":True,**audit20}
    else:
        out["limitations"].append("网络API与分片头未读取；历史审计数只用于离线算术")

    # 4：跨厂商定义；数值接近不等于原作者采用同一口径。
    qw=qwen_moe_account(config("Qwen/Qwen3-30B-A3B"))
    ds={name:ds_moe_account(config(repo)) for name,repo in [
        ("v2lite","deepseek-ai/DeepSeek-V2-Lite"),("v3","deepseek-ai/DeepSeek-V3")]}
    out["cross_vendor"]={"qwen3_30b":qw,**ds,
        "interpretation":"gpt-oss表注明确计输出头不计输入表；其他品牌公开概数不能唯一确定细项口径"}
    out["limitations"].append("V3官方36.7B含双表与本书未吸收投影账未精确对平；V2-Lite/Qwen概数不证明唯一单/双表定义")
    print("STEP4 跨厂商账已分列；未用接近概数判定定义。")

    # 5：物理载荷与逻辑参数分开。在线成功时所有项取在线同一审计，不混历史BF16数。
    if audit20:
        fam=audit20["families"]
        prefix="model.layers.N.mlp.experts."
        data={key:fam[prefix+name] for key,name in [
            ("gu_blocks","gate_up_proj_blocks"),("gu_scales","gate_up_proj_scales"),
            ("dn_blocks","down_proj_blocks"),("dn_scales","down_proj_scales")]}
        data["bf16"]=audit20["dtypes"].get("BF16",0)
        source="本次同revision分片头"
    else:
        data=dict(MXFP4_AUDIT_20B);source="历史审计汇总，非本次读取"
    blocks=data["gu_blocks"]+data["dn_blocks"];scales=data["gu_scales"]+data["dn_scales"]
    logical=2*blocks;bits=mxfp4_storage(blocks,scales,logical)
    payload=blocks+scales+2*data["bf16"]
    if data["bf16"]+logical!=accts["openai/gpt-oss-20b"]["total"]:raise AssertionError("逻辑参数与公式不同")
    if bits!=4.25:raise AssertionError("专家位宽与32值一scale不符")
    if audit20 and payload!=audit20["payload_bytes"]:raise AssertionError("物理字节账不一致")
    out["mxfp4"]={**data,"source":source,"logical_fp4":logical,"bits":bits,
                   "payload_bytes":payload,"gib":payload/2**30,"includes_file_headers":False}
    out["kv"]={r:kv_account(c) for r,c in cfgs.items()}
    print(f"STEP5 expert bits={bits}; payload={payload:,} bytes; full-prefix KV={out['kv']}")

    # 6：速览保留全部模型，QK-Norm计入精确整数；镜像一致只限所列字段。
    q32=config("Qwen/Qwen3-32B")
    q32total=llama3_account(q32["hidden_size"],q32["num_hidden_layers"],q32["num_attention_heads"],
        q32["num_key_value_heads"],q32["head_dim"],q32["intermediate_size"],q32["vocab_size"])
    q32total+=q32["num_hidden_layers"]*2*q32["head_dim"]
    g2=config("unsloth/gemma-2-9b");g3a=config("unsloth/gemma-3-27b-it")
    g3b=config("RedHatAI/gemma-3-27b-it-FP8-dynamic")
    keys=["hidden_size","num_hidden_layers","num_attention_heads","num_key_value_heads","head_dim",
          "intermediate_size","vocab_size","sliding_window_pattern","sliding_window","rope_local_base_freq","rope_theta"]
    same=all(g3a["text_config"].get(k)==g3b["text_config"].get(k) for k in keys)
    out["speedview"]={"llama3_8b":llama3_account(4096,32,32,8,128,14336,128256),
        "llama3_70b":llama3_account(8192,80,64,8,128,28672,128256),
        "llama4_book3_anchors":BOOK3_ANCHORS,"qwen3_32b":q32total,
        "qwen3_30b":qw,"gemma2_9b":gemma_text_account(g2),
        "gemma3_27b":gemma_text_account(g3a),"gemma3_mirror_fields":keys,"gemma3_mirror_agreement":same,
        **ds,"gptoss":accts}
    print("STEP6 精确账：Qwen3-32B",q32total,"Gemma3",out["speedview"]["gemma3_27b"]["total"])
    if args.real_weights:out["real_weights"]=real_weights_demo()
    if args.figure!="none":
        Path(FIG_DIR).mkdir(parents=True,exist_ok=False)
        if args.figure in {"all","structure"}:
            draw_fig10_1(accts["openai/gpt-oss-20b"],str(Path(FIG_DIR)/"fig-10-1-gptoss-inferred-structure.png"))
        if args.figure in {"all","routing"}:
            draw_fig10_2(str(Path(FIG_DIR)/"fig-10-2-routing-two-schools.png"))
    with target.open("x",encoding="utf-8") as handle:json.dump(out,handle,ensure_ascii=False,indent=2)
    print("产物：",target)


if __name__ == "__main__":
    main()
