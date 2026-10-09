# qwen3next_mini.py —— Book5 ch5 正式件：MiniQwen3Next 缩玩具整机 + HF strict 直搬对拍
# 用途：ch5 5.5/5.7「Qwen3-Next 整机 config 逆向 + 键位对齐」的本机实验正身——
#   ① 自研缩玩具整机 MiniQwen3Next：键位严格对齐 transformers 5.18.0 Qwen3NextForCausalLM 的
#      state_dict（model.embed_tokens / model.layers.N.{linear_attn|self_attn, mlp.*, input_layernorm,
#      post_attention_layernorm} / model.norm / lm_head）——load_state_dict(HF 整机, strict) 直搬；
#   ② 三级对拍：op 级（gdn.py 双 kernel vs HF，快检）→ 层级（GDN 层 strict，含 h_v=2·h_k 的
#      GQA 比 + K1 交错头排布）→ 整机级（mini 随机权重 strict 直搬，max|Δlogits|）；
#   ③ 键位六坑 K1-K6 注释——「strict 直搬第一现场」素材（预研期逐个排掉，正文各一行）。
# 【键位六坑（HF 5.18.0 qwen3_next 口径；坑=不照做就 strict 失败或对拍不过）】
#   K1 交错头排布：HF in_proj_qkvz 输出按 k 头交错 [q,k,v,z] 切块（fix_query_key_value_ordering），
#      不是连续 [Q,K,V,Z]——strict 直搬第一坑（本文件 HFCompatGDN 与 gdn.py 教学骨架的关键差异）；
#   K2 两个 RMSNorm 不同构：Qwen3NextRMSNorm 是 (1+w) 缩放，Qwen3NextRMSNormGated 是 w 缩放 +
#      silu(gate) 门——同文件两个类、两种语义；
#   K3 partial RoPE：rotary_dim = cos 最后一维（head_dim·partial_rotary_factor），q/k 只旋转前
#      rotary_dim 维、后段直通；
#   K4 门控注意力：q_proj 宽 2·h·d（q + 逐维 sigmoid 门），输出 attn_out·sigmoid(gate) 后才 o_proj；
#   K5 conv1d：padding=k−1 卷积后取前 n 位（因果短卷积），权重按头 squeeze(1)；
#   K6 dt_bias/A_log：g = −exp(A_log)·softplus(a + dt_bias)，fp32 计算防溢出。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch05/qwen3next_mini.py" [--out-name run1]
#   全程 CPU fp32（数值对拍口径）；预计 2-3 分钟。
#   --params-80b：只出 80B 真机参数账（meta device 零分配，秒级；正文表 5.3 数据源）——
#     三口径（总参/非嵌入/激活）+ 逐层公式 + checkpoint MTP 补齐逐位闭合 + 分类型状态账与 n*。
# 产物：log/book5-ch05/qwen3next_mini_{out}.json 与 qwen3next_params_{out}.json（不入库）
import argparse
import importlib.util
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book5-ch05")
SEED = 20261002

_GDN = None


def _gdn():
    """惰性加载同目录 gdn.py（双 kernel），进程内只加载一次。"""
    global _GDN
    if _GDN is None:
        _GDN = load_sibling("gdn")
    return _GDN


def load_sibling(name):
    p = os.path.join(HERE, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save_json(name, obj, out_name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}_{out_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[产物] {path}")
    return path


# ---------------- 自研 HF 兼容件（键位严格对齐 transformers 5.18.0 qwen3_next） ----------------
class RMSNorm1pW(nn.Module):
    """Qwen3NextRMSNorm 同构：x·rsqrt(mean x²+eps)·(1+w)——K2 的 (1+w) 形。"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(var + self.eps)).to(x.dtype) * (1.0 + self.weight.float())


class RMSNormGated(nn.Module):
    """Qwen3NextRMSNormGated 同构：norm·w·silu(gate)（w 缩放——注意与上面 (1+w) 的不同构，K2）。"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, gate):
        var = x.float().pow(2).mean(-1, keepdim=True)
        h = (x.float() * torch.rsqrt(var + self.eps)) * self.weight.float()
        return (h * F.silu(gate.float())).to(x.dtype)


class HFCompatGDN(nn.Module):
    """自研 GDN 层（HF 键位 strict 直搬版）——键位/语义对齐 Qwen3NextGatedDeltaNet。

    交错头排布（K1）：in_proj_qkvz 输出 view 为 (B,S,h_k, 2d_k + 2·d_v·h_v/h_k) 后逐 k 头切
    [q, k, v, z]；in_proj_ba 同理切 [b, a]。h_v = 2·h_k 时 v/z/b/a 按 2 倍宽切（GQA 比）。
    """

    def __init__(self, hidden, h_k, h_v, d_k, d_v, conv_kernel=4, chunk=64):
        super().__init__()
        self.h_k, self.h_v, self.d_k, self.d_v = h_k, h_v, d_k, d_v
        self.chunk = chunk
        self.key_dim, self.value_dim = h_k * d_k, h_v * d_v
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.in_proj_qkvz = nn.Linear(hidden, 2 * self.key_dim + 2 * self.value_dim, bias=False)
        self.in_proj_ba = nn.Linear(hidden, 2 * h_v, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, conv_kernel, groups=self.conv_dim,
                                padding=conv_kernel - 1, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(h_v))
        self.A_log = nn.Parameter(torch.log(torch.empty(h_v).uniform_(0.01, 16.0)))
        self.norm = RMSNormGated(d_v, eps=1e-6)
        self.out_proj = nn.Linear(self.value_dim, hidden, bias=False)

    def forward(self, x):
        gdn = _gdn()
        B, S, H = x.shape
        ratio = self.h_v // self.h_k
        qkvz = self.in_proj_qkvz(x).view(B, S, self.h_k,
                                         2 * self.d_k + 2 * self.d_v * ratio)       # (B,S,h_k,·) 交错块（K1）
        q, k, v, z = qkvz.split([self.d_k, self.d_k, self.d_v * ratio, self.d_v * ratio], dim=-1)
        ba = self.in_proj_ba(x).view(B, S, self.h_k, 2 * ratio)
        b, a = ba.split([ratio, ratio], dim=-1)
        v = v.reshape(B, S, self.h_v, self.d_v)                                   # (B,S,h_v,d_v)
        z = z.reshape(B, S, self.h_v, self.d_v)
        b = b.reshape(B, S, self.h_v)                                             # (B,S,h_v)
        a = a.reshape(B, S, self.h_v)
        qkv = torch.cat([q.reshape(B, S, -1), k.reshape(B, S, -1), v.reshape(B, S, -1)],
                        dim=-1).transpose(1, 2)                                   # (B,conv_dim,S)
        qkv = F.silu(self.conv1d(qkv)[..., :S])                                   # K5 因果短卷积
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=1)
        q = q.transpose(1, 2).view(B, S, self.h_k, self.d_k)
        k = k.transpose(1, 2).view(B, S, self.h_k, self.d_k)
        v = v.transpose(1, 2).view(B, S, self.h_v, self.d_v)
        if ratio > 1:
            q = q.repeat_interleave(ratio, dim=2)                                 # k 侧 GQA 展开（HF 同款）
            k = k.repeat_interleave(ratio, dim=2)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)       # K6：fp32 防溢出
        core, _ = gdn.gdn_chunkwise(q, k, v, g, beta, chunk=self.chunk)           # (B,S,h_v,d_v)
        core = self.norm(core, z)                                                 # 带权 RMSNormGated（K2）
        return self.out_proj(core.reshape(B, S, self.value_dim))                   # (B,S,H)


class MiniQwen3NextAttention(nn.Module):
    """Qwen3NextAttention 同构：门控注意力（K4）+ QK-Norm（head_dim 维）+ partial RoPE（K3）。

    键位：q_proj（宽 2·h·d：q+gate）/k_proj/v_proj/o_proj/q_norm/k_norm。
    """

    def __init__(self, hidden, n_head, n_kv, head_dim, rope_theta=10000.0, partial=0.5, eps=1e-6):
        super().__init__()
        self.h, self.h_kv, self.d = n_head, n_kv, head_dim
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(hidden, n_head * head_dim * 2, bias=False)        # K4：宽 2·h·d
        self.k_proj = nn.Linear(hidden, n_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, n_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(n_head * head_dim, hidden, bias=False)
        self.q_norm = RMSNorm1pW(head_dim, eps)
        self.k_norm = RMSNorm1pW(head_dim, eps)
        self.rope_theta, self.partial = rope_theta, partial
        self.rotary_dim = int(head_dim * partial)                                 # K3：rotary_dim=cos 最后一维

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        B, S, H = x.shape
        qg = self.q_proj(x).view(B, S, -1, 2 * self.d)                            # (B,S,h,2d)
        q, gate = qg.chunk(2, dim=-1)                                             # (B,S,h,d) x2——K4
        gate = gate.reshape(B, S, -1)
        q = self.q_norm(q)
        k = self.k_norm(self.k_proj(x).view(B, S, self.h_kv, self.d))
        v = self.v_proj(x).view(B, S, self.h_kv, self.d)
        q = q.transpose(1, 2)                                                     # (B,h,S,d)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # partial RoPE（K3）：只旋转前 rotary_dim 维，后段直通；cat(freqs,freqs)+rotate_half
        rd = self.rotary_dim
        inv = 1.0 / (self.rope_theta ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
        ang = torch.outer(torch.arange(S, dtype=torch.float32), inv)              # (S, rd/2)
        freqs = torch.cat((ang, ang), dim=-1)                                     # (S, rd)
        cos, sin = freqs.cos()[None, None], freqs.sin()[None, None]
        rot_q = q[..., :rd] * cos + self._rotate_half(q[..., :rd]) * sin
        rot_k = k[..., :rd] * cos + self._rotate_half(k[..., :rd]) * sin
        q = torch.cat([rot_q, q[..., rd:]], dim=-1)
        k = torch.cat([rot_k, k[..., rd:]], dim=-1)
        rep = self.h // self.h_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        att = (q @ k.transpose(-2, -1)) * self.scale                              # (B,h,S,S)
        att = att.masked_fill(~torch.ones(S, S, dtype=torch.bool).tril(), float("-inf"))
        p = F.softmax(att.float(), dim=-1).to(x.dtype)
        o = (p @ v).transpose(1, 2).contiguous().view(B, S, self.h * self.d)
        o = o * torch.sigmoid(gate)                                               # K4 逐维门
        return self.o_proj(o)


class MiniExperts(nn.Module):
    """Qwen3NextExperts 同构：融合 3D 专家（E,2w,H)/(E,H,w)。"""

    def __init__(self, E, hidden, inter):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(E, 2 * inter, hidden))
        self.down_proj = nn.Parameter(torch.empty(E, hidden, inter))

    def forward(self, h, idx, w):
        out = torch.zeros_like(h)
        for e in range(self.gate_up_proj.shape[0]):
            tok, slot = torch.nonzero(idx == e, as_tuple=True)
            if tok.numel() == 0:
                continue
            gate, up = F.linear(h[tok], self.gate_up_proj[e]).chunk(2, dim=-1)
            y = F.linear(F.silu(gate) * up, self.down_proj[e])
            out.index_add_(0, tok, y * w[tok, slot].unsqueeze(1))
        return out


class MiniMoE(nn.Module):
    """Qwen3NextSparseMoeBlock 同构：router(norm_topk_prob) + 融合专家 + sigmoid 门共享专家。

    键位：mlp.gate.weight（路由门）/ mlp.experts.{gate_up,down}_proj / mlp.shared_expert.* /
    mlp.shared_expert_gate.weight——本类 gate 用 Module 参数包装成同名键。
    """

    def __init__(self, hidden, E, top_k, inter, shared_inter):
        super().__init__()
        self.gate = nn.Module()
        self.gate.weight = nn.Parameter(torch.zeros(E, hidden))                   # HF 键位 mlp.gate.weight
        self.experts = MiniExperts(E, hidden, inter)
        self.shared_expert = nn.Module()
        self.shared_expert.gate_proj = nn.Linear(hidden, shared_inter, bias=False)
        self.shared_expert.up_proj = nn.Linear(hidden, shared_inter, bias=False)
        self.shared_expert.down_proj = nn.Linear(shared_inter, hidden, bias=False)
        self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)
        self.top_k, self.E = top_k, E

    def _shared(self, x):
        s = self.shared_expert
        return s.down_proj(F.silu(s.gate_proj(x)) * s.up_proj(x))

    def forward(self, x):
        shape = x.shape
        h = x.reshape(-1, shape[-1])
        logits = F.linear(h, self.gate.weight)                                    # (N,E)
        probs = logits.float().softmax(-1)
        w, idx = torch.topk(probs, self.top_k, dim=-1)                            # (N,k)
        w = w / w.sum(-1, keepdim=True)                                           # norm_topk_prob=True
        out = self.experts(h, idx, w.to(h.dtype)) + torch.sigmoid(self.shared_expert_gate(h)) * self._shared(h)
        return out.view(*shape)


class MiniDenseMLP(nn.Module):
    """Qwen3NextMLP 同构（dense 层用；整机 mini 全 MoE 步长=1 时不出现）。"""

    def __init__(self, hidden, inter):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MiniQwen3Next(nn.Module):
    """整机同构（键位对齐 Qwen3NextForCausalLM 的 state_dict）——strict 直搬对拍正身。

    mini config（正文 5.5 的缩微口径）：V=256/d=128/4 层 [linear×3, full]/GDN 插槽 2k4v/头维 32/
    MoE 8 选 2 + 共享——结构谱系与真机 80B-A3B 同构（3:1 + 门控注意力 + partial RoPE 0.5）。
    """

    def __init__(self, cfg):
        super().__init__()
        H, E, topk = cfg["hidden"], cfg["E"], cfg["top_k"]
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(cfg["vocab"], H)
        self.model.layers = nn.ModuleList()
        for i, kind in enumerate(cfg["layer_types"]):
            layer = nn.Module()
            layer.input_layernorm = RMSNorm1pW(H, cfg["eps"])
            layer.post_attention_layernorm = RMSNorm1pW(H, cfg["eps"])
            if kind == "linear_attention":
                layer.linear_attn = HFCompatGDN(H, cfg["l_h_k"], cfg["l_h_v"], cfg["l_d"], cfg["l_d"],
                                                cfg["conv_k"], cfg["chunk"])
            else:
                layer.self_attn = MiniQwen3NextAttention(H, cfg["n_head"], cfg["n_kv"], cfg["head_dim"],
                                                         cfg["rope_theta"], cfg["partial"], cfg["eps"])
            layer.mlp = MiniMoE(H, E, topk, cfg["moe_inter"], cfg["shared_inter"])
            self.model.layers.append(layer)
        self.model.norm = RMSNorm1pW(H, cfg["eps"])
        self.lm_head = nn.Linear(H, cfg["vocab"], bias=False)

    def forward(self, idx):
        x = self.model.embed_tokens(idx)                                          # (B,S,H)
        for layer in self.model.layers:
            r = x
            h = layer.input_layernorm(x)
            mixer = layer.linear_attn(h) if hasattr(layer, "linear_attn") else layer.self_attn(h)
            x = r + mixer
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.lm_head(self.model.norm(x))                                   # (B,S,V)


# ---------------- 80B 真机参数账（meta device 零分配；config 逆向正档） ----------------
# Qwen/Qwen3-Next-80B-A3B-Thinking config.json 逐字段（2026-10-05 亲核下载，落
# log/book5-feasibility/qwen3next_config.json；此处内嵌字段以保证脚本离线可复现）
QWEN3NEXT_80B_FIELDS = dict(
    vocab_size=151936, hidden_size=2048, num_hidden_layers=48,
    num_attention_heads=16, num_key_value_heads=2, head_dim=256,
    intermediate_size=5120,
    linear_num_key_heads=16, linear_num_value_heads=32,
    linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
    num_experts=512, num_experts_per_tok=10,
    moe_intermediate_size=512, shared_expert_intermediate_size=512,
    decoder_sparse_step=1, mlp_only_layers=[],
    max_position_embeddings=262144, tie_word_embeddings=False,
    rope_theta=10_000_000.0, partial_rotary_factor=0.25,
)
# checkpoint 索引正身（model.safetensors.index.json metadata.total_size，BF16 字节 → 元素 /2）
CKPT_TOTAL_ELEMENTS = 162_649_725_440 // 2  # = 81,324,862,720（48 层 HF 模型之外还带 1 个 mtp.* 模块）


def params_account_80b():
    """Qwen3-Next 80B 参数账三口径 + 逐项公式 + MTP 补齐对账 checkpoint 索引。

    meta device 只建模块树不分配数据（Book4 ch10/ch5 ch04 同款手艺）；三层 assert：
    ① 公式逐项之和 == meta 实测总参；② +MTP 模块公式 == checkpoint 索引 total_size/2（逐位闭合）；
    ③ README 三口径（80B 总参 / 79B non-embedding / 3B activated）舍入对平。
    返回 dict（正文表 5.3 的数据源）。
    """
    from accelerate import init_empty_weights
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextForCausalLM

    d, V = 2048, 151936
    h, h_kv, hd = 16, 2, 256            # 全局层 GQA 8:1（QK-Norm 512 维）
    h_k, h_v, d_k, d_v, conv = 16, 32, 128, 128, 4   # GDN：V:K 头数比 2:1
    E, top_k, w_i, w_s = 512, 10, 512, 512

    # 逐项公式（元素数；Book4 ch1 单双份判定法：嵌入 untied 出入双份、激活只报计算账）
    full_attn = 2 * h * hd * d + h_kv * hd * d * 2 + h * hd * d + 2 * hd          # q(含 gate)+k+v+o+QK-Norm
    gdn = d * (2 * h_k * d_k + 2 * h_v * d_v) + d * 2 * h_v + (2 * h_k * d_k + h_v * d_v) * conv \
        + h_v * 2 + d_v + h_v * d_v * d                                          # qkvz+ba+conv+A_log/dt+norm+out
    moe = d * E + E * 3 * d * w_i + 3 * d * w_s + d                              # router+专家群+共享+sigmoid 门
    moe_active = d * E + top_k * 3 * d * w_i + 3 * d * w_s + d                   # 每 token 实际动用（10 专家）
    embed = V * d * 2                                                             # untied 出入双份
    norms = 48 * 2 * d + d                                                        # 双层 norm + final norm
    mtp = 2 * d * d + full_attn + 5 * d + moe                                     # fc + 注意力层 + 5 组 norm + MoE

    cfg = Qwen3NextConfig(**QWEN3NEXT_80B_FIELDS)
    counts = {t: cfg.layer_types.count(t) for t in sorted(set(cfg.layer_types))}
    assert counts == {"full_attention": 12, "linear_attention": 36}, counts       # 3:1（interval=4 展开）
    with init_empty_weights():
        model = Qwen3NextForCausalLM(cfg)
    meta_total = sum(p.numel() for p in model.parameters())                        # meta 张量 numel 可读、无数据

    formula_total = counts["full_attention"] * full_attn + counts["linear_attention"] * gdn \
        + 48 * moe + embed + norms
    assert formula_total == meta_total, (formula_total, meta_total)                # ① 公式=meta 逐位
    assert formula_total + mtp == CKPT_TOTAL_ELEMENTS, (formula_total + mtp, CKPT_TOTAL_ELEMENTS)  # ② MTP 补齐=索引

    active_non_embed = counts["full_attention"] * full_attn + counts["linear_attention"] * gdn \
        + 48 * moe_active
    ledger = {  # 分类型状态账（ch1 式 (1.3) 的 Qwen3-Next 行；单位=元素）
        "gdn_state_fixed": counts["linear_attention"] * h_v * d_k * d_v,           # 524,288/层，与 n 无关
        "gdn_conv_state_fixed": counts["linear_attention"] * (conv - 1) * (2 * h_k * d_k + h_v * d_v),
        "full_kv_per_token": 12 * 2 * h_kv * hd,                                   # 1,024 元素/token，无界
        "crossover_n_star": counts["linear_attention"] * h_v * d_k * d_v // (12 * 2 * h_kv * hd),
    }
    return {"config_fields": QWEN3NEXT_80B_FIELDS, "layer_counts": counts,
            "per_layer": {"full_attn": full_attn, "gdn": gdn, "moe": moe, "moe_active": moe_active},
            "embed_pair": embed, "norms": norms, "mtp_module": mtp,
            "total_formula": formula_total, "total_meta": meta_total,
            "non_embedding": formula_total - embed,
            "active_non_embedding": active_non_embed,
            "ckpt_index_total_elements": CKPT_TOTAL_ELEMENTS,
            "ledger": ledger,
            "readme_check": {"total_80B": round(formula_total / 1e9, 2),
                             "nonemb_79B": round((formula_total - embed) / 1e9, 2),
                             "activated_3B": round(active_non_embed / 1e9, 2)}}


# ---------------- 主流程：三级对拍 ----------------
def main():
    ap = argparse.ArgumentParser(description="Book5 ch5 MiniQwen3Next 整机 strict 直搬对拍")
    ap.add_argument("--out-name", default="run1")
    ap.add_argument("--params-80b", action="store_true",
                    help="80B 真机参数账（meta device 零分配，秒级）——不跑对拍，只出账")
    args = ap.parse_args()
    if args.params_80b:
        acc = params_account_80b()
        print(f"[参数账] 3:1 层计数 = {acc['layer_counts']}")
        print(f"[参数账] 每层：全局注意力 {acc['per_layer']['full_attn']:,} | GDN {acc['per_layer']['gdn']:,} "
              f"| MoE {acc['per_layer']['moe']:,}（激活 {acc['per_layer']['moe_active']:,}）")
        print(f"[参数账] 总参 {acc['total_meta']:,}（公式=meta 逐位 ✓）"
              f" + MTP {acc['mtp_module']:,} = checkpoint 索引 {acc['ckpt_index_total_elements']:,}（逐位闭合 ✓）")
        print(f"[参数账] 非嵌入 {acc['non_embedding']:,} ≈ {acc['readme_check']['nonemb_79B']}B"
              f" | 激活非嵌入 {acc['active_non_embedding']:,} ≈ {acc['readme_check']['activated_3B']}B（A3B）")
        print(f"[参数账] README 对账：{acc['readme_check']}（80B/79B/3B 三口径舍入对平）")
        print(f"[状态账] GDN 固定 {acc['ledger']['gdn_state_fixed']:,}+卷积 {acc['ledger']['gdn_conv_state_fixed']:,} "
              f"| 全局层 {acc['ledger']['full_kv_per_token']:,}/token 无界 | n* = {acc['ledger']['crossover_n_star']}")
        save_json("qwen3next_params", acc, args.out_name)
        return
    gdn = load_sibling("gdn")
    torch.manual_seed(SEED)
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextGatedDeltaNet, Qwen3NextForCausalLM, torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule)
    results = {"seed": SEED, "hf_version": __import__("transformers").__version__}

    # ① op 级复检（快——gdn.py 全量对拍另有 main，此处口径同种子）
    B, S, Hh, Dk, Dv = 2, 256, 2, 32, 32
    torch.manual_seed(SEED)
    q = torch.randn(B, S, Hh, Dk); k = torch.randn(B, S, Hh, Dk); v = torch.randn(B, S, Hh, Dv)
    beta = torch.randn(B, S, Hh).sigmoid()
    g = -F.softplus(torch.randn(B, S, Hh) * 1.5)
    o_hr, _ = torch_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    o_mr, _ = gdn.gdn_recurrent(q, k, v, g, beta)
    o_hc, _ = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    o_mc, _ = gdn.gdn_chunkwise(q, k, v, g, beta)
    results["op_level"] = {"recurrent_max_diff": float((o_hr - o_mr).abs().max()),
                           "chunk_max_diff": float((o_hc - o_mc).abs().max())}
    print(f"[op 级] recurrent max|Δ|={results['op_level']['recurrent_max_diff']:.2e} | "
          f"chunk max|Δ|={results['op_level']['chunk_max_diff']:.2e}")

    # ② 层级 strict 直搬（GDN 层，含 h_v=2·h_k 的 GQA 比 + 交错排布 K1）
    cfg_l = Qwen3NextConfig(hidden_size=128, num_hidden_layers=1,
                            linear_num_key_heads=2, linear_num_value_heads=4,
                            linear_key_head_dim=32, linear_value_head_dim=32,
                            linear_conv_kernel_dim=4,
                            layer_types=["linear_attention"])
    torch.manual_seed(SEED)
    hf_layer = Qwen3NextGatedDeltaNet(cfg_l, layer_idx=0)
    my_layer = HFCompatGDN(128, 2, 4, 32, 32, conv_kernel=4, chunk=32)
    res_load = my_layer.load_state_dict(hf_layer.state_dict(), strict=True)       # strict：键位逐个咬合
    x = torch.randn(2, 64, 128)
    y_hf = hf_layer(hidden_states=x, cache_params=None, attention_mask=None)
    y_my = my_layer(x)
    results["layer_level_gdn_max_diff"] = float((y_hf - y_my).abs().max())
    print(f"[层级 GDN strict] load strict ✓ | max|Δ|={results['layer_level_gdn_max_diff']:.2e}")

    # ③ 整机 strict 直搬（MiniQwen3Next vs Qwen3NextForCausalLM）
    mini_cfg = dict(vocab=256, hidden=128, n_head=4, n_kv=2, head_dim=32, eps=1e-6,
                    rope_theta=10000.0, partial=0.5, l_h_k=2, l_h_v=4, l_d=32, conv_k=4, chunk=32,
                    E=8, top_k=2, moe_inter=64, shared_inter=64,
                    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"])
    hf_cfg = Qwen3NextConfig(
        vocab_size=256, hidden_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, intermediate_size=112,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, decoder_sparse_step=1, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=64, shared_expert_intermediate_size=64, norm_topk_prob=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        layer_types=mini_cfg["layer_types"], tie_word_embeddings=False)
    hf_cfg._attn_implementation = "eager"
    torch.manual_seed(SEED)
    hf_model = Qwen3NextForCausalLM(hf_cfg)
    my_model = MiniQwen3Next(mini_cfg)
    sd = hf_model.state_dict()
    load_res = my_model.load_state_dict(sd, strict=True)                          # 整机 strict：一键咬合
    idx_in = torch.randint(0, 256, (2, 48))
    torch.manual_seed(SEED)
    logits_hf = hf_model(input_ids=idx_in).logits
    logits_my = my_model(idx_in)
    results["model_level"] = {
        "strict_load_missing": list(load_res.missing_keys),
        "strict_load_unexpected": list(load_res.unexpected_keys),
        "max_abs_diff_logits": float((logits_hf - logits_my).abs().max()),
    }
    print(f"[整机 strict] missing={results['model_level']['strict_load_missing']} "
          f"unexpected={results['model_level']['strict_load_unexpected']} | "
          f"max|Δlogits|={results['model_level']['max_abs_diff_logits']:.2e}")
    assert results["model_level"]["max_abs_diff_logits"] < 1e-4, "整机 strict 直搬对拍失败"
    save_json("qwen3next_mini", results, args.out_name)
    print("ch05 qwen3next_mini.py 完成。")


if __name__ == "__main__":
    main()
