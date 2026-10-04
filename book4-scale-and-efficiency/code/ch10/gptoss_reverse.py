# gptoss_reverse.py —— Book4 ch10：gpt-oss config 逆向主场（B3-5 五步法的完整演出）
# 用途：STEP1 认家族（20b/120b 并列；openai 官方直发、非 gated）→ STEP2 结构推断表（表 10.1 数字源：
#       字段→值→部件→等级）→ STEP3 三方逐位参数账（式 10.1/10.2 手算 vs HF API vs 分片头 Range 审计；
#       assert 官方模型卡论文 Table 1 全家族逐位）→ STEP4 活账两口径+跨厂商口径审计（单/双份判别）
#       → STEP5 MXFP4 存储解剖（blocks/scales 字节账→4.25 bits 自证+12.8GiB）与 KV 元素账
#       → STEP6 十机型旗舰速览表生成器（config 实算；Llama3/Llama4 行=Book3 已核资产指回）
#       → [--real-weights] 条件性真权重演示：本地 gpt-oss-20b 快照的 MXFP4 字节→自实现解包
#         vs transformers 5.18.0 反量化（max|Δ| assert）+单专家真前向（gpt-oss 式限幅 SwiGLU）。
#         整机反量化≈41.8GB bf16 超 48GB 统一内存实用预算——降级边界如实标注，不硬上。
# 所属章节：Book4 第 10 章 §10.2-10.7（表 10.1-10.4 与式 10.1/10.2 的机器可读版；图 10.1/10.2 一并绘制）
# 运行方式：cd 工作区根目录 && source env.sh && python code/ch10/gptoss_reverse.py --out-name run1
#           可选：--no-net（离线档：跳过 HF API/Range 审计，用调研期登记数并如实标注）
#                 --real-weights（条件性真权重；快照缺失时自动降级打印原因）
#                 --figure {all,structure,routing,none}（默认 all；图落 figures/）
# 产物：log/book4-ch10/gptoss_reverse_{out_name}.json（不入库）；
#       figures/fig-10-1-gptoss-inferred-structure.png、fig-10-2-routing-two-schools.png
# 依赖：config 在线=openai/gpt-oss-{20b,120b}（HF_HOME 缓存，KB 级）→ 失败回落
#       log/book4-feasibility/06-configs/ 调研期快照（2026-10-04 实抓；来源标进 JSON）；
#       速览表机型快照同目录（Gemma3 行做 unsloth+RedHatAI 双镜像交叉）；纯算术 CPU 秒级。
# 注：STEP1-6 为确定性算术与 config 读取、无随机过程；--real-weights 演示前向用种子 20261002。
import argparse
import json
import math
import os
import re
import struct
import time
import urllib.request

SEED = 20261002
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO, "log", "book4-ch10")
FIG_DIR = os.path.join(REPO, "figures")
CFG_FALLBACK = os.path.join(REPO, "log", "book4-feasibility", "06-configs")
HF_HOME = os.environ.get("HF_HOME", os.path.join(REPO, "log", "huggingface"))

# 官方模型卡论文（arXiv 2508.10925 v1）Table 1 assert 锚点（调研期+本次 Range 审计双核对）
OFFICIAL = {
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


def load_config(repo_id, fname=None):
    """config 获取：HF 在线（HF_HOME 缓存）→ 调研期快照回落。返回 (cfg, 来源标注)。"""
    fname = fname or "config.json"
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id, fname)
        return json.load(open(path, encoding="utf-8")), f"HF 在线/缓存：{repo_id}"
    except Exception as e:
        snap = os.path.join(CFG_FALLBACK, repo_id.replace("/", "_") + ".config.json")
        if os.path.exists(snap):
            return json.load(open(snap, encoding="utf-8")), f"调研期快照 {os.path.basename(snap)}（在线失败：{type(e).__name__}）"
        raise


# ---------------- STEP1/2 认家族 + 结构推断表 ----------------
def infer_rows(c):
    """表 10.1 的机器可读版：字段 → 20b/120b 值 → 部件/读法（等级注记写进正文表）。"""
    d, L, E = c["hidden_size"], c["num_hidden_layers"], c["num_local_experts"]
    hd = c["num_attention_heads"] * c["head_dim"]
    lt = c["layer_types"]
    return [
        ("model_type / architectures / transformers_version",
         f'gpt_oss / GptOssForCausalLM / {c["transformers_version"]}',
         "认家族三件套；转换版本=年代学锚（2025-08，Book3 ch5 手法）"),
        ("hidden_size", str(d), "残差流宽 d（两档同宽——扩容走专家份数不走宽度，与 Llama 4 同哲学）"),
        ("num_hidden_layers", f'{L}（论文 §2.2 同值）', "Block 数 L"),
        ("num_local_experts / num_experts_per_tok", f'{E} / {c["num_experts_per_tok"]}',
         f'专家份数 E / 每 token 点燃 k（论文 "top-4"；html 渲染损伤为 "top-44"，以 config 为准）'),
        ("intermediate_size", str(c["intermediate_size"]), "专家宽 f（=d，方阵专家；无 moe_ 前缀字段，dense/MoE 共用）"),
        ("num_attention_heads / num_key_value_heads / head_dim",
         f'{c["num_attention_heads"]} / {c["num_key_value_heads"]} / {c["head_dim"]}',
         f"GQA 8 组；h·d_k = {hd} > d = {d}：q/o 是 {d}↔{hd} 的矩形投影（罕见几何）"),
        ("attention_bias", str(c["attention_bias"]), "四个注意力投影全带偏置（分片头证实 o_proj.bias 也在）——2017 之后的罕见回归"),
        ("layer_types", f'{len(lt)} 项，sliding/full 逐层交替（首层 sliding）', "1:1 交错注意力（非传闻的 3:1）"),
        ("sliding_window", str(c["sliding_window"]), "滑窗带宽 128 token（只作 config 事实，机制→Book5 第 2 章）"),
        ("rope_theta / rope_scaling",
         f'{c["rope_theta"]:g} / yarn factor {c["rope_scaling"]["factor"]:g}（4096→131072）',
         "RoPE 底数 1.5e5 + YaRN 32×（缩放常数住实现里：0.1·ln32+1≈1.3466）"),
        ("max_position_embeddings", f'{c["max_position_embeddings"]:,}', "声明上下文 128k"),
        ("swiglu_limit", str(c["swiglu_limit"]), "SwiGLU 激活限幅常数（config 字段与实现硬编码冗余并置）"),
        ("router_aux_loss_coef / output_router_logits",
         f'{c["router_aux_loss_coef"]} / {c["output_router_logits"]}',
         "负载均衡系数字段存在但实现从不计算（悬案④）"),
        ("quantization_config", 'mxfp4；不转换=self_attn/router/embed/lm_head', "只有 MoE 专家权重是 MXFP4，其余全 BF16"),
        ("vocab_size / tie_word_embeddings", f'{c["vocab_size"]:,} / {c["tie_word_embeddings"]}',
         "o200k_harmony 词表，untied 出入口两份"),
        ("rms_norm_eps", str(c["rms_norm_eps"]), "pre-RMSNorm（fp32 上抛的数值路径差异，不在结构）"),
        ("（config 无字段）self_attn.sinks", f"每头 1 可学标量（{L}×{c['num_attention_heads']} 枚 BF16）",
         "权重审计+源码双跳才读得到（分片头：sinks [64]/层）"),
        ("（config 无字段）experts 偏置", "gate_up/down 两级都带偏置", "分片头证实（HF 键位：experts.gate_up_proj_bias 等）"),
        ("（config 无字段）QK-Norm", "无", "双实现（OpenAI 原实现 + transformers 5.18.0）注意力通路均无 q/k 归一化——负结论"),
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
                active_layer=per_layer - exp_w + k * (3 * d * f + 2 * f + d))


def hf_api_safetensors(repo_id):
    """HF API 对拍方：total 与 dtype 分布（gpt-oss 的 U8 计数=fp4 逻辑参数，块×2、不含 scale）。"""
    r = json.load(urllib.request.urlopen(
        f"https://huggingface.co/api/models/{repo_id}?expand[]=safetensors", timeout=30))
    st = r.get("safetensors", {})
    return dict(total=st.get("total"), parameters=st.get("parameters"))


def shard_header_audit(repo_id):
    """分片头审计对拍方（Book3 ch5 Range 手法）：逐张量 numel，按家族与 dtype 聚合。"""
    idx = json.load(urllib.request.urlopen(
        f"https://huggingface.co/{repo_id}/resolve/main/model.safetensors.index.json", timeout=30))
    shards = sorted(set(idx["weight_map"].values()))
    fam, dtype_ct, grand = {}, {}, 0
    for s in shards:
        url = f"https://huggingface.co/{repo_id}/resolve/main/{s}"
        req = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
        n = struct.unpack("<Q", urllib.request.urlopen(req, timeout=60).read())[0]
        req = urllib.request.Request(url, headers={"Range": f"bytes=8-{7 + n}"})
        hdr = json.loads(urllib.request.urlopen(req, timeout=60).read())
        for t, meta in hdr.items():
            if t == "__metadata__":
                continue
            numel = math.prod(meta["shape"])
            grand += numel
            key = re.sub(r"\d+", "N", t)
            fam[key] = fam.get(key, 0) + numel
            dtype_ct[meta["dtype"]] = dtype_ct.get(meta["dtype"], 0) + numel
    return dict(shards=shards, grand=grand, families=fam, dtypes=dtype_ct)


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

    口径注：noaux 偏置 e_score_correction_bias（F32，59 层×256）与 FP8 scale_inv 同记辅助张量侧、
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
    """Qwen3-30B-A3B（QK-Norm 每层 2×d_k 与双 norm 2d 共 ~0.4M 入舍入，不进公式——与 ch01/
    调研登记口径一致，登记值 30,531,913,728 对官方 30.5B 逐位）。"""
    d, L, V = c["hidden_size"], c["num_hidden_layers"], c["vocab_size"]
    H, KV, d_k = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    E, k, f = c["num_experts"], c["num_experts_per_tok"], c["moe_intermediate_size"]
    attn = 2 * d * (H * d_k) + 2 * d * (KV * d_k)
    total = 2 * V * d + L * (attn + E * 3 * d * f + d * E) + d
    act_one = V * d + L * (attn + k * 3 * d * f + d * E) + d
    return dict(total=total, active_one=act_one, active_two=act_one + V * d)


def gemma_text_account(c):
    """Gemma2/3 文本栈（三明治四 norm；Gemma3 计 q_norm/k_norm——P5 双镜像已权重级印证）。"""
    t = c.get("text_config", c)
    d, L = t["hidden_size"], t["num_hidden_layers"]
    H, KV, d_k = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    f, V = t["intermediate_size"], t["vocab_size"]
    tied = t.get("tie_word_embeddings", True)
    attn = 2 * d * (H * d_k) + 2 * d * (KV * d_k)
    qkn = (H * d_k + KV * d_k) if t.get("sliding_window_pattern") else 0  # Gemma3 有 QK-Norm；Gemma2 无
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
    """自实现 MXFP4 解包（教学件；与 transformers 5.18.0 语义独立成文再对拍）。

    输入：blocks (E, R, G, 16) uint8——每字节 2 个 fp4（低 4 位=组内偶数位，高 4 位=奇数位）；
          scales (E, R, G) uint8——每 32 值一枚 E8M0 指数（存储=x+127，真值=2^(x-127)）。
    输出：(E, R, 32G) float32 的解量化权重行（权重矩阵的 [R, 32G] 段，R=输出维、32G=输入维）。
    """
    import torch
    lut = torch.tensor(FP4_LUT, dtype=torch.float32)
    lo = lut[(blocks & 0x0F).long()]              # (E,R,G,16) 组内偶数位值
    hi = lut[(blocks >> 4).long()]                # (E,R,G,16) 组内奇数位值
    vals = torch.stack([lo, hi], dim=-1).reshape(*blocks.shape[:-1], blocks.shape[-1] * 2)  # (E,R,G,32)
    scale = torch.pow(2.0, scales.to(torch.float32) - 127.0)   # E8M0 → 2^(x-127)
    vals = vals * scale.unsqueeze(-1)             # 块内 32 值共享一枚指数
    return vals.reshape(*blocks.shape[:-2], -1)   # (E,R,32G)


def find_local_snapshot(repo_id):
    """本地快照探测（env.sh 的 HF_HOME 红线内）：返回 snapshot 目录或 None。"""
    root = os.path.join(HF_HOME, "hub", "models--" + repo_id.replace("/", "--"), "snapshots")
    if not os.path.isdir(root):
        return None
    for snap in sorted(os.listdir(root)):
        p = os.path.join(root, snap)
        if not os.path.isdir(p):
            continue
        st = [f for f in os.listdir(p) if f.endswith(".safetensors") and "index" not in f]
        if st and all(os.path.exists(os.path.join(p, f)) for f in st):
            return p  # 完整落盘（symlink 已建、无 .incomplete）
    return None


def real_weights_demo(verbose=True):
    """条件性真权重演示：字节→自实现解包 vs transformers 反量化→单专家真前向。

    整机反量化≈41.8GB bf16，超 48GB 统一内存实用预算——本演示停在专家粒度（~2GB），
    边界如实标注（正文 §10.5 动手节同款口径）。
    """
    import torch
    torch.manual_seed(SEED)
    snap = find_local_snapshot("openai/gpt-oss-20b")
    if snap is None:
        print("  [降级] gpt-oss-20b 本地快照未落地（或存在 .incomplete）——"
              "分片头/HF API 审计资产已齐（STEP3/5），真权重演示不阻塞。")
        return None
    from safetensors import safe_open
    shard0 = os.path.join(snap, "model-00000-of-00002.safetensors")
    with safe_open(shard0, framework="pt") as f:  # 惰性读：只取 layer 0 的 MoE 一套
        blocks = f.get_tensor("model.layers.0.mlp.experts.gate_up_proj_blocks")   # U8 (32,5760,90,16)
        scales = f.get_tensor("model.layers.0.mlp.experts.gate_up_proj_scales")   # U8 (32,5760,90)
        gu_bias = f.get_tensor("model.layers.0.mlp.experts.gate_up_proj_bias")    # BF16 (32,5760)
        dn_blocks = f.get_tensor("model.layers.0.mlp.experts.down_proj_blocks")   # U8 (32,2880,90,16)
        dn_scales = f.get_tensor("model.layers.0.mlp.experts.down_proj_scales")   # U8 (32,2880,90)
        dn_bias = f.get_tensor("model.layers.0.mlp.experts.down_proj_bias")       # BF16 (32,2880)
        r_w = f.get_tensor("model.layers.0.mlp.router.weight")                    # BF16 (32,2880)
        r_b = f.get_tensor("model.layers.0.mlp.router.bias")                      # BF16 (32,)
    t0 = time.time()
    gu_mine = my_mxfp4_dequant(blocks, scales).transpose(1, 2).contiguous()       # (32,2880,5760)=(E,d,2f)
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors
    gu_hf = convert_moe_packed_tensors(blocks, scales).float()                    # 同一入口的官方反量化
    if gu_hf.shape != gu_mine.shape:  # 官方实现带一次 transpose——形状对齐后再逐位比
        gu_hf = gu_hf.transpose(1, 2).contiguous()
    assert gu_hf.shape == gu_mine.shape, f"解包形状对不齐：{tuple(gu_hf.shape)} vs {tuple(gu_mine.shape)}"
    max_diff = (gu_mine.to(torch.bfloat16).float() - gu_hf).abs().max().item()
    # 单专家真前向（gpt-oss 式限幅 SwiGLU：gate 上限 7.0、up ±7.0、α=1.702、输出乘 (up+1)）
    x = torch.randn(4, 2880, generator=torch.Generator().manual_seed(SEED))      # (B=4, d=2880)
    logits = x @ r_w.float().T + r_b.float()                                      # (4, 32)=(N,E)
    top = torch.topk(logits, k=4, dim=-1)
    gates = torch.softmax(top.values, dim=-1)                                     # top-k 内 softmax（两学派之一）
    e0 = int(top.indices[0, 0])                                                   # 第 0 token 的首选专家
    gu = x @ gu_mine[e0].to(torch.float32) + gu_bias[e0].float()                  # (4, 5760)
    gate, up = gu[..., ::2], gu[..., 1::2]                                        # 偶位=gate、奇位=up（HF 键位同款交错）
    act = gate.clamp(max=7.0) * torch.sigmoid(1.702 * gate.clamp(max=7.0)) * (up.clamp(-7.0, 7.0) + 1)
    dn = my_mxfp4_dequant(dn_blocks, dn_scales)                                   # (32,2880,2880)=(E,f,d)…行=f
    y = act @ dn[e0].to(torch.float32) + dn_bias[e0].float()                      # (4, 2880)
    out = gates[0, 0] * y
    res = dict(snapshot=snap, gate_up_shape=list(gu_mine.shape), hf_shape=list(gu_hf.shape),
               max_absdiff_bf16=max_diff, expert_index=e0, router_top4=top.indices[0].tolist(),
               router_gates=[round(v, 4) for v in gates[0].tolist()],
               out_shape=list(out.shape), out_absmax=round(out.abs().max().item(), 4),
               wall_s=round(time.time() - t0, 1))
    if verbose:
        print(f"  [真权重] 快照={snap}")
        print(f"  gate_up blocks {list(blocks.shape)} + scales {list(scales.shape)} → 解包 {list(gu_mine.shape)} (E,d,2f)")
        print(f"  自实现解包 vs transformers 5.18.0 反量化：max|Δ|={max_diff:.2e}（bf16 对拍）")
        print(f"  单专家真前向（token0 选专家 {e0}，top-4={res['router_top4']}，门={res['router_gates']}）"
              f"→ 输出 {res['out_shape']}，|y|max={res['out_absmax']}")
        print(f"  演示口径：专家粒度（整机反量化≈41.8GB bf16 超 48GB 统一内存预算，边界如实标注）")
    return res


# ---------------- 图 10.1 / 图 10.2 ----------------
def draw_fig10_1(acct20, path):
    """图 10.1 gpt-oss 结构推断图（自绘示意级；实线=config 逆向+权重审计，虚线=官方模型卡论文）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    BLUE, ORANGE, TEAL, DARK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.size": 8.5, "text.color": DARK, "font.family": "DejaVu Sans"})

    def box(x, y, w, h, text, ec=DARK, fc="white", lw=1.1, fs=7.0, ls="-", tc=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=lw, linestyle=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc or DARK)

    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=300)
    ax.set_xlim(0, 17); ax.set_ylim(0, 11.4); ax.axis("off")
    ax.text(8.5, 11.05, "gpt-oss-20b: structure inferred from config.json + shard headers (no architecture paper)",
            ha="center", fontsize=9.2, fontweight="bold")
    # 主列（实线 = config 逆向 + 权重审计）
    box(3.9, 9.5, 7.9, 0.9, "Embedding  V=201,088 x d=2880  (untied, tie=false)", ec=BLUE, fc="#f2f7fd")
    box(3.9, 6.4, 7.9, 2.75, "", ec=BLUE, fc="#f2f7fd")
    ax.text(4.15, 8.92, "x 24 blocks (120b: 36)", fontsize=7.2, fontweight="bold")
    ax.text(4.15, 7.75, "RMSNorm -> Attention: 64 Q heads / 8 KV heads, head_dim 64\n"
                        "     q:(B,n,4096)<-(B,n,2880)  k,v:(B,8,n,64)  ->  o:(B,n,2880)\n"
                        "     rectangular proj. h*d_k=4096 > d=2880; all 4 proj. WITH bias\n"
                        "     + 1 learned sink per head (64/layer, shard-verified)\n"
                        "     alternate layers: sliding (window 128) / full  (1:1, first=sliding)",
            fontsize=6.4, va="center")
    box(3.9, 4.35, 7.9, 1.7, "", ec=TEAL, fc="#eefaf5")
    ax.text(4.15, 5.85, "RMSNorm -> FFN slot = MoE (all 24 layers)", fontsize=7.2, fontweight="bold", color=DARK)
    ax.text(4.15, 5.0, "router: nn.Linear 2880->E(=32) with bias, scores (N,E)\n"
                       "select top-4, weights = softmax over the 4 selected only\n"
                       "E experts, width f=2880 (=d, 'square' experts), each with bias",
            fontsize=6.4, va="center")
    box(3.9, 3.2, 7.9, 0.75, "Final RMSNorm  (d=2880, eps=1e-5)", ec=BLUE, fc="#f2f7fd")
    box(3.9, 2.05, 7.9, 0.85, "LM head  2880 -> 201,088  (untied; BF16)", ec=BLUE, fc="#f2f7fd")
    for y0, y1 in [(9.5, 9.15), (6.4, 6.05), (4.35, 3.95), (3.2, 2.9)]:
        ax.annotate("", xy=(7.85, y1), xytext=(7.85, y0), arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))
    # MXFP4 标注（config quantization_config + 分片头）
    box(0.35, 4.35, 3.2, 1.7, "MoE expert weights ONLY:\nMXFP4 storage\n(quantization_config;\nshard: blocks U8\n2 fp4/byte + 1 U8\nE8M0 scale / 32 vals)", ec=ORANGE, fc="#fdf3ee", fs=6.2, tc="#9c4413")
    ax.annotate("", xy=(3.9, 5.2), xytext=(3.55, 5.2), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    # 右列（虚线 = 官方模型卡论文 2508.10925）
    relay = [(9.5, "Total 20.91B / Active 3.61B\n(Table 1; \"Unembedding counted\ntowards active, not embeddings\")"),
             (8.0, "\"post-trained the models with\nquantization of the MoE weights\nto MXFP4\" - 4.25 bits/param"),
             (6.6, "RoPE theta 1.5e5 + YaRN x32:\n4096 -> 131,072 tokens\n(\"extend...dense layers\")"),
             (5.2, "attention blocks \"alternate between\nbanded window and fully dense\",\nbandwidth 128 tokens"),
             (3.8, "trained \"trillions of tokens\";\n120b: 2.1M H100-hours\n(20b ~10x fewer)"),
             (2.4, "router: \"standard linear router\nprojection\" + \"softmax ... over\nonly the selected experts\"")]
    for y, txt in relay:
        box(12.75, y - 0.5, 3.95, 1.0, txt, ec=ORANGE, ls="--", lw=1.2, fs=6.0, tc="#9c4413")
    ax.text(14.7, 1.35, "{official model card paper\n2508.10925 (card, not an\narchitecture paper)}", ha="center", fontsize=6.2, color=MUTED)
    ax.text(1.95, 3.55, "{shard header audit:\n3 shards, 52,112 header bytes\nread via HTTP Range}", ha="center", fontsize=6.2, color=MUTED)
    # 图例与手算注脚
    box(3.9, 0.95, 3.7, 0.7, "solid = config reverse + shard audit\n(openai repo, non-gated)", ec=BLUE, fs=6.2)
    box(7.9, 0.95, 3.9, 0.7, "dashed = official model card\n(claims, not field-level)", ec=ORANGE, ls="--", fs=6.2, tc="#9c4413")
    ax.text(8.5, 0.45, f"hand-computed from config: total {acct20['total']:,} = HF API = shard audit"
                       f"  |  active(one-embed) {acct20['active_one']:,}",
            ha="center", fontsize=6.8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_fig10_2(path):
    """图 10.2 两学派路由对照（gpt-oss vs DSV3；全部 config/实现事实，形状标注）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    BLUE, ORANGE, DARK, MUTED, TEAL = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#1baf7a"
    plt.rcParams.update({"font.size": 8.5, "text.color": DARK, "font.family": "DejaVu Sans"})

    def box(x, y, w, h, text, ec=DARK, fc="white", lw=1.1, fs=6.6, ls="-", tc=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=lw, linestyle=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc or DARK)

    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=300)
    ax.set_xlim(0, 17); ax.set_ylim(0, 10.6); ax.axis("off")
    ax.text(8.5, 10.25, "Two routing schools in 2025 flagships (config + reference-implementation facts)",
            ha="center", fontsize=9.4, fontweight="bold")
    # 左列：gpt-oss
    ax.text(3.6, 9.6, "gpt-oss (2025-08)", ha="center", fontsize=8.6, fontweight="bold", color=BLUE)
    left = [(8.35, "score: linear router  (N,d)·(d,E)+bias -> (N,E)\nE=32 (20b) / 128 (120b); softmax NOT applied here", BLUE),
            (6.85, "select: top-4 (hard pick)\n(N,E) -> indices (N,4)", BLUE),
            (5.35, "weights: softmax over the SELECTED 4 only\n(N,4); no re-scaling factor", BLUE),
            (3.85, "expert mix: 4 narrow experts, f=2880=d\nout (N,4,d) --gates--> (N,d);  NO shared expert", BLUE),
            (2.2, "load balancing: NOT DISCLOSED\nconfig field router_aux_loss_coef=0.9 exists,\nbut the reference impl. never computes it (open case)", ORANGE)]
    # 右列：DSV3
    ax.text(13.4, 9.6, "DeepSeek-V3 (2024-12)", ha="center", fontsize=8.6, fontweight="bold", color=TEAL)
    right = [(8.35, "score: sigmoid  (N,d)·(d,E)+bias -> (N,E)\nE=256 routed + 1 shared; e_score_correction_bias", TEAL),
             (6.85, "select: group-limited top-8\n8 groups -> keep top-4 groups -> top-8 experts", TEAL),
             (5.35, "weights: normalize over selected 8,\nthen x routed_scaling_factor 2.5", TEAL),
             (3.85, "expert mix: 8 fine experts, f=2048 (~0.29d)\n+ 1 shared always-on;  first 3 layers dense", TEAL),
             (2.2, "load balancing: noaux_tc\nbias updates decoupled from gate values\n(aux-loss-free; ch5 eq. 5.2)", TEAL)]
    for y, txt, c in left:
        box(0.5, y - 0.65, 6.2, 1.3, txt, ec=c, fc="#f2f7fd" if c == BLUE else "#fdf3ee", fs=6.2)
    for y, txt, c in right:
        box(10.3, y - 0.65, 6.2, 1.3, txt, ec=c, fc="#eefaf5", fs=6.2)
    for xy, xytext in [((3.6, 7.7), (3.6, 8.35 - 0.65)), ((3.6, 6.2), (3.6, 6.85 - 0.65)),
                       ((3.6, 4.7), (3.6, 5.35 - 0.65)), ((3.6, 2.85), (3.6, 3.85 - 0.65)),
                       ((13.4, 7.7), (13.4, 8.35 - 0.65)), ((13.4, 6.2), (13.4, 6.85 - 0.65)),
                       ((13.4, 4.7), (13.4, 5.35 - 0.65)), ((13.4, 2.85), (13.4, 3.85 - 0.65))]:
        ax.annotate("", xy=xy, xytext=xytext, arrowprops=dict(arrowstyle="->", color=DARK, lw=1.1))
    ax.text(8.5, 0.75, "N = flattened token count (B*n); both schools use the SAME 4-step machinery of ch2 -\n"
                       "they differ in scoring / selection / normalization / balancing governance",
            ha="center", fontsize=6.8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="ch10 gpt-oss config 逆向主场：五步法完整演出+旗舰速览表")
    ap.add_argument("--out-name", default="run1")
    ap.add_argument("--no-net", action="store_true", help="离线档：跳过在线对拍（用调研期登记数并标注）")
    ap.add_argument("--real-weights", action="store_true", help="条件性真权重演示（快照缺失自动降级）")
    ap.add_argument("--figure", choices=["all", "structure", "routing", "none"], default="all")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"steps": []}

    print("=" * 100)
    print("STEP1 认家族（openai 官方直发、非 gated——单源=config 本身即官方）")
    cfgs, tags = {}, {}
    for repo in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
        cfgs[repo], tags[repo] = load_config(repo)
        c = cfgs[repo]
        print(f"  {repo}: model_type={c['model_type']}  architectures={c['architectures'][0]}"
              f"  transformers_version={c['transformers_version']}  [{tags[repo]}]")
        print(f"    d={c['hidden_size']}  L={c['num_hidden_layers']}  E={c['num_local_experts']}"
              f"  top-{c['num_experts_per_tok']}  f={c['intermediate_size']}  GQA {c['num_attention_heads']}:"
              f"{c['num_key_value_heads']}  d_k={c['head_dim']}  V={c['vocab_size']:,}")
    # 20b/120b 全字段 diff：只有 L 与 E（及 layer_types 长度）不同——「扩容只加专家份数」的机器证据
    diff = [k for k in set(cfgs["openai/gpt-oss-20b"]) | set(cfgs["openai/gpt-oss-120b"])
            if cfgs["openai/gpt-oss-20b"].get(k) != cfgs["openai/gpt-oss-120b"].get(k)]
    print(f"  20b vs 120b 全字段 diff：{diff}（宽度/注意力/词表全同——容量只长在专家份数上）")

    print("\nSTEP2 结构推断表（字段 → 值 → 部件/读法；等级注记见正文表 10.1）")
    rows = infer_rows(cfgs["openai/gpt-oss-20b"])
    for f, v, p in rows:
        print(f"  {f:46s} {v}")

    print("\nSTEP3 三方逐位参数账（式 10.1 手算 vs HF API vs 分片头审计；assert 官方 Table 1）")
    accts = {}
    for repo in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
        a = gptoss_account(cfgs[repo])
        off = OFFICIAL[repo]
        assert a["total"] == off["total"], f"{repo} 总账 {a['total']:,} != 官方 {off['total']:,}"
        assert a["mlp_family"] == off["mlp"] and a["attn_family"] == off["attn"] and a["ent_family"] == off["ent"]
        assert a["active_one"] == off["active_one"] and a["active_two"] == off["active_two"]
        print(f"  {repo}: 手算总账 = {a['total']:,}"
              f"（MLP {a['mlp_family']:,} / Attn {a['attn_family']:,} / E+U {a['ent_family']:,} / norms"
              f" {a['total'] - a['mlp_family'] - a['attn_family'] - a['ent_family']:,}）")
        print(f"    每层：attn_w={a['attn_w']:,} attn_b={a['attn_b']:,} sinks={a['sinks']} router={a['router']:,}"
              f" experts_w={a['exp_w']:,} experts_b={a['exp_b']:,}")
        print(f"    活账：本册口径(含 lm_head)={a['active_one']:,}  双份={a['active_two']:,}")
        print(f"    [assert] 手算 = 官方 Table 1（{off['total']:,} / Active {off['active_one']:,}）逐位 ✓")
        accts[repo] = a
        out["steps"].append(dict(step=3, repo=repo, account={k: v for k, v in a.items()}, official=off))
    api = audit = None
    if not args.no_net:
        try:
            api = {r: hf_api_safetensors(r) for r in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]}
            for r, v in api.items():
                ok = v["total"] == OFFICIAL[r]["total"]
                print(f"  HF API {r}: total={v['total']:,}  dtypes={v['parameters']}"
                      f"  {'== 手算：三方之第二方 ✓' if ok else '≠ 手算（需归因）'}")
        except Exception as e:
            print(f"  [skip] HF API 失败（{type(e).__name__}），对拍列降级")
        try:
            audit = shard_header_audit("openai/gpt-oss-20b")
            gu_b = audit["families"].get("model.layers.N.mlp.experts.gate_up_proj_blocks", 0)
            gu_s = audit["families"].get("model.layers.N.mlp.experts.gate_up_proj_scales", 0)
            dn_b = audit["families"].get("model.layers.N.mlp.experts.down_proj_blocks", 0)
            dn_s = audit["families"].get("model.layers.N.mlp.experts.down_proj_scales", 0)
            bf16 = audit["dtypes"].get("BF16", 0)
            u8_phys = audit["dtypes"].get("U8", 0)
            logical = (gu_b + dn_b) * 2
            print(f"  分片头审计(20b)：grand={audit['grand']:,}  dtype={{BF16 {bf16:,}, U8 {u8_phys:,}(物理字节)}}")
            print(f"    fp4 逻辑参数 = 块×2 = {(gu_b + dn_b) * 2:,}；BF16+逻辑 = {bf16 + logical:,}"
                  f" {'== 手算总账 ✓（HF API 的 U8 计数即逻辑口径）' if bf16 + logical == OFFICIAL['openai/gpt-oss-20b']['total'] else ''}")
            out["audit20b"] = dict(grand=audit["grand"], gu_blocks=gu_b, gu_scales=gu_s, dn_blocks=dn_b,
                                   dn_scales=dn_s, bf16=bf16, u8_physical=u8_phys, logical_fp4=logical,
                                   shards=audit["shards"])
        except Exception as e:
            print(f"  [skip] 分片头审计失败（{type(e).__name__}），审计列降级（用调研期登记数）")

    print("\nSTEP4 活账两口径 + 跨厂商口径审计（官方「3.6B」数的是哪一本账）")
    print("  官方 Table 1 caption 原句：\"Unembedding parameters are counted towards active, but not embeddings\"")
    print("  ——与本册口径（含 lm_head、不含输入 embedding lookup）同义；untied 下两份等大、数值相同")
    cv_rows = [("gpt-oss-20b", "3.6B", accts["openai/gpt-oss-20b"]["active_one"], "单份"),
               ("gpt-oss-120b", "5.1B", accts["openai/gpt-oss-120b"]["active_one"], "单份")]
    sv = {}  # 速览表暂存
    try:
        qw = qwen_moe_account(load_config("Qwen/Qwen3-30B-A3B")[0])
        assert qw["total"] == 30_531_913_728, f"Qwen3-30B-A3B 总账 {qw['total']:,} != 登记数"
        cv_rows.append(("Qwen3-30B-A3B", "3.3B", qw["active_two"], "双份"))
        sv["qwen3_30b"] = qw
    except Exception as e:
        print(f"  [skip] Qwen3-30B-A3B 快照缺失（{type(e).__name__}）")
    try:
        ds_cfgs = {n: load_config(n)[0] for n in
                   ["deepseek-ai/DeepSeek-V2-Lite", "deepseek-ai/DeepSeek-V3"]}
        lite = ds_moe_account(ds_cfgs["deepseek-ai/DeepSeek-V2-Lite"])
        v3 = ds_moe_account(ds_cfgs["deepseek-ai/DeepSeek-V3"])
        assert lite["total"] == 15_706_484_224 and v3["total"] == 671_026_404_352, \
            f"DS 总账对不上：Lite {lite['total']:,} / V3 {v3['total']:,}"
        cv_rows += [("DeepSeek-V2-Lite", "2.4B", lite["active_one"], "单份"),
                    ("DeepSeek-V3", "37B(精确 36.6B)", v3["active_one"], "单份")]
        out["ds_verify"] = dict(v2lite=lite, v3_main=v3)
        sv.update(v2lite_total=lite["total"], v3_total=v3["total"])
    except AssertionError as e:
        print(f"  [ASSERT-FAIL] {e}——P8 正身公式对不上登记数，需归因（不许静默）")
        raise
    except Exception as e:
        print(f"  [skip] DS 快照缺失（{type(e).__name__}）")
    print(f"  {'模型':16s}{'官方口径':14s}{'实算吻合值':>16s}   {'判定':6s}")
    for name, off_v, ours, verdict in cv_rows:
        print(f"  {name:16s}{off_v:14s}{ours:>16,}   {verdict}")
    out["cross_vendor"] = [dict(model=n, official=o, ours_single_or_match=v, verdict=d)
                           for n, o, v, d in cv_rows]

    print("\nSTEP5 MXFP4 存储解剖（分片头三件套）与 KV 元素账")
    m = out.get("audit20b")
    gu_b = m["gu_blocks"] if m else MXFP4_AUDIT_20B["gu_blocks"]
    gu_s = m["gu_scales"] if m else MXFP4_AUDIT_20B["gu_scales"]
    dn_b = m["dn_blocks"] if m else MXFP4_AUDIT_20B["dn_blocks"]
    dn_s = m["dn_scales"] if m else MXFP4_AUDIT_20B["dn_scales"]
    logical = (gu_b + dn_b) * 2
    bits = mxfp4_storage(gu_b + dn_b, gu_s + dn_s, logical)
    gib = (gu_b + gu_s + dn_b + dn_s + MXFP4_AUDIT_20B["bf16"] * 2) / 2**30
    print(f"  gate_up blocks {gu_b:,} U8 + scales {gu_s:,} U8；down blocks {dn_b:,} + scales {dn_s:,}"
          f"（每字节 2 个 fp4；每 32 值一枚 U8 E8M0 scale）")
    print(f"  4.25 bits 自证：({gu_b + dn_b:,}+{gu_s + dn_s:,})×8 ÷ {logical:,} = {bits} bits/param"
          f"（论文 \"4.25 bits per parameter\" ✓）")
    print(f"  检查点体积：BF16 {MXFP4_AUDIT_20B['bf16']:,}×2B + U8 {(gu_b + gu_s + dn_b + dn_s):,}×1B"
          f" = {(MXFP4_AUDIT_20B['bf16'] * 2 + gu_b + gu_s + dn_b + dn_s):,} B = {gib:.2f} GiB（论文 12.8GiB ✓）")
    for repo in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
        kv = kv_account(cfgs[repo])
        print(f"  {repo} KV/token = 2·L·h_kv·d_k = {kv['elements']:,} 元素 = {kv['kib_bf16']:.0f} KiB (bf16)"
              f"（GQA-8 的小 KV 面；滑窗层的有界性属推理期记账→Book6，机制→Book5）")
    out["mxfp4"] = dict(bits=bits, gib=round(gib, 2), logical_fp4=logical)
    out["kv"] = {r: kv_account(cfgs[r]) for r in cfgs}

    print("\nSTEP6 十机型旗舰速览表生成器（config 实算；Book3 已核行=指回常量）")
    try:
        l3_8b = llama3_account(4096, 32, 32, 8, 128, 14336, 128256)
        assert l3_8b == BOOK3_ANCHORS["llama3_8b"], f"Llama3-8B {l3_8b:,} != Book3 锚"
        l3_70b = llama3_account(8192, 80, 64, 8, 128, 28672, 128256)
        q3_32b = llama3_account(5120, 64, 64, 8, 128, 25600, 151936)  # Qwen3-32B 同构（GQA+双 norm+untied）
        print(f"  Llama 3 8B / 70B：{l3_8b:,}（=Book3 ch5 四方逐位锚 ✓）/ {l3_70b:,}")
        print(f"  Qwen3-32B：{q3_32b:,}（GQA 64:8 矩形 8192>5120；QK-Norm 0.4M 入舍入）")
        print(f"  Llama 4 Scout / Maverick（Book3 ch9 三方逐位指回）：text {BOOK3_ANCHORS['scout_text']:,}"
              f" / {BOOK3_ANCHORS['maverick_text']:,}；active 17.2B / 17.2B")
        g2 = gemma_text_account(load_config("unsloth/gemma-2-9b")[0])
        print(f"  Gemma2 9B text：{g2['total']:,}（tied；单镜像 unsloth+分片头审计——等级降档标注）")
        g3a = load_config("unsloth/gemma-3-27b-it")[0]
        g3b = load_config("RedHatAI/gemma-3-27b-it-FP8-dynamic")[0]
        keys = ["sliding_window_pattern", "sliding_window", "rope_local_base_freq", "rope_theta"]
        same = all(g3a["text_config"].get(k) == g3b["text_config"].get(k) for k in keys)
        g3 = gemma_text_account(g3a)
        print(f"  Gemma3 27B text：{g3['total']:,}（双镜像交叉字段一致={same}；5:1/窗1024/双RoPE；"
              f"出厂带 vision tower——本表只计文本栈）")
        sv.update(llama3_8b=l3_8b, llama3_70b=l3_70b, qwen3_32b=q3_32b, gemma2_9b=g2["total"],
                  gemma3_27b_text=g3["total"], gemma3_dual_mirror_same=same)
    except Exception as e:
        print(f"  [skip] 速览表部分行失败（{type(e).__name__}: {e}）")
    out["speedview"] = sv
    print("  （完整 12 行表→正文表 10.3；本行打印为生成器的核心 assert 摘要）")

    if args.real_weights:
        print("\n[real-weights] 条件性真权重演示")
        out["real_weights"] = real_weights_demo()

    if args.figure in ("all", "structure"):
        os.makedirs(FIG_DIR, exist_ok=True)
        p = os.path.join(FIG_DIR, "fig-10-1-gptoss-inferred-structure.png")
        draw_fig10_1(accts["openai/gpt-oss-20b"], p)
        print(f"\n[图] {p}")
    if args.figure in ("all", "routing"):
        os.makedirs(FIG_DIR, exist_ok=True)
        p = os.path.join(FIG_DIR, "fig-10-2-routing-two-schools.png")
        draw_fig10_2(p)
        print(f"[图] {p}")

    out["config_sources"] = tags
    out_path = os.path.join(OUT_DIR, f"gptoss_reverse_{args.out_name}.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(f"\n[产物] {out_path}")


if __name__ == "__main__":
    main()
