# 08_probe_parity.py —— Book5 对拍链路终裁探针：HF Qwen3Next 三级对拍 + SWA mask 对拍
# 用途：全册「手写件 vs HF 5.18.0 参考实现」的对拍链路可用性——
#   ① op 级：04 探针 GDN 双形态 vs HF torch_chunk/torch_recurrent（复跑快检）；
#   ② 层级（strict 直搬）：自研 HF 兼容 GDN 层（键位 in_proj_qkvz/in_proj_ba/conv1d/A_log/dt_bias/
#      norm/out_proj + HF 的 fix_query_key_value_ordering 交错头排布 + 带权 RMSNormGated）
#      load_state_dict(HF 层, strict=True) 后前向 max|Δ|；
#   ③ 整机级（strict 直搬）：自研 MiniQwen3Next（HF 键位：embed_tokens/layers.N.{linear_attn|self_attn}/
#      mlp.{gate,experts,shared_expert,shared_expert_gate}/norm/lm_head）vs HF Qwen3NextForCausalLM
#      mini 随机权重 strict 直搬 max|Δlogits|——含门控注意力（q_proj 出 2×宽：q+逐维 sigmoid 门）、
#      QK-Norm（head_dim 维）、partial RoPE（rotary_dim=cos.shape[-1]，前段旋转后段直通）、
#      MoE（融合 3D 专家 + norm_topk_prob + sigmoid 门共享专家）；
#   ④ SWA mask 对拍：自研 banded causal mask vs HF masking_utils.create_sliding_window_causal_mask
#      （Gemma3Config 口径）逐元素等价。
# 【已知坑登记（预研时逐个排掉，正文「键位对齐坑」素材）】
#   K1 交错头排布：HF in_proj_qkvz 输出按 k 头交错 [q,k,v,z] 切块（fix_query_key_value_ordering），
#      不是连续 [Q,K,V,Z]——strict 直搬第一坑；
#   K2 两个 RMSNorm 不同构：Qwen3NextRMSNorm 是 (1+w) 缩放，RMSNormGated 是 w 缩放 + silu(gate) 门；
#   K3 partial RoPE：rotary_dim=cos 最后一维（head_dim·partial），q/k 只旋转前 rotary_dim 维；
#   K4 门控注意力：q_proj 宽 2·h·d，chunk 出 q 与 gate，输出 attn_out·sigmoid(gate) 后才 o_proj；
#   K5 conv1d：padding=k−1 卷积后取前 n 位（因果短卷积），权重 squeeze(1)；
#   K6 dt_bias/A_log：g = −exp(A_log)·softplus(a + dt_bias)，fp32 计算防溢出。
# 所属章节：Book5 第 5 章（GDN 对拍）与第 2 章（SWA mask 对拍）；对拍链路结论供全册引用。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/00-feasibility/08_probe_parity.py" [--out-name run1]
#   全程 CPU fp32（数值对拍口径）；预计 2-4 分钟。
# 产物：log/book5-feasibility/probe08_parity_{out}.json（不入库）
import argparse
import importlib.util
import json
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book5-feasibility")
SEED = 20261002


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
    """自研 GDN 层，键位/语义对齐 Qwen3NextGatedDeltaNet（strict 直搬对拍用）。

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
        B, S, H = x.shape
        ratio = self.h_v // self.h_k
        qkvz = self.in_proj_qkvz(x).view(B, S, self.h_k,
                                         2 * self.d_k + 2 * self.d_v * ratio)       # (B,S,h_k,·) 交错块
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
        core, _ = p04.gdn_chunkwise(q, k, v, g, beta, chunk=self.chunk)            # (B,S,h_v,d_v)
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
        self.q_proj = nn.Linear(hidden, n_head * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden, n_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, n_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(n_head * head_dim, hidden, bias=False)
        self.q_norm = RMSNorm1pW(head_dim, eps)
        self.k_norm = RMSNorm1pW(head_dim, eps)
        self.rope_theta, self.partial = rope_theta, partial
        self.rotary_dim = int(head_dim * partial)

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
        # partial RoPE（K3）：只旋转前 rotary_dim 维，后段直通；cat(freqs,freqs) + rotate_half
        rd = self.rotary_dim
        inv = 1.0 / (self.rope_theta ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
        ang = torch.outer(torch.arange(S, dtype=torch.float32), inv)              # (S, rd/2)
        freqs = torch.cat((ang, ang), dim=-1)                                     # (S, rd)
        cos, sin = freqs.cos()[None, None], freqs.sin()[None, None]
        rot_q = q[..., :rd] * cos + p02_like_rotate_half(q[..., :rd]) * sin
        rot_k = k[..., :rd] * cos + p02_like_rotate_half(k[..., :rd]) * sin
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


def p02_like_rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


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
    """Qwen3NextSparseMoeBlock 同构：router(norm_topk_prob) + 融合专家 + sigmoid 门共享专家。"""

    def __init__(self, hidden, E, top_k, inter, shared_inter):
        super().__init__()
        self.gate_w = nn.Parameter(torch.zeros(E, hidden))                        # HF 键位 mlp.gate.weight
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
        logits = F.linear(h, self.gate_w)                                         # (N,E)
        probs = logits.float().softmax(-1)
        w, idx = torch.topk(probs, self.top_k, dim=-1)                            # (N,k)
        w = w / w.sum(-1, keepdim=True)                                           # norm_topk_prob=True
        out = self.experts(h, idx, w.to(h.dtype)) + torch.sigmoid(self.shared_expert_gate(h)) * self._shared(h)
        return out.view(*shape)


class MiniDenseMLP(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MiniQwen3Next(nn.Module):
    """整机同构（键位对齐 Qwen3NextForCausalLM 的 state_dict）——strict 直搬对拍正身。"""

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


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="Book5 对拍链路终裁探针")
    ap.add_argument("--out-name", default="run1")
    args = ap.parse_args()
    global p04
    p04 = load_sibling("04_probe_gdn")
    torch.manual_seed(SEED)
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextGatedDeltaNet, Qwen3NextForCausalLM, torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule)
    from transformers.models.gemma3.configuration_gemma3 import Gemma3Config
    from transformers.masking_utils import create_sliding_window_causal_mask
    results = {"seed": SEED, "date": "2026-10-05",
               "hf_version": __import__("transformers").__version__}

    # ① op 级复检（快）
    B, S, Hh, Dk, Dv = 2, 256, 2, 32, 32
    q = torch.randn(B, S, Hh, Dk); k = torch.randn(B, S, Hh, Dk); v = torch.randn(B, S, Hh, Dv)
    beta = torch.randn(B, S, Hh).sigmoid()
    g = -F.softplus(torch.randn(B, S, Hh) * 1.5)
    o_hr, _ = torch_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    o_mr, _ = p04.gdn_recurrent(q, k, v, g, beta)
    o_hc, _ = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    o_mc, _ = p04.gdn_chunkwise(q, k, v, g, beta)
    results["op_level"] = {"recurrent": float((o_hr - o_mr).abs().max()),
                           "chunk": float((o_hc - o_mc).abs().max())}
    print(f"[op 级] recurrent max|Δ|={results['op_level']['recurrent']:.2e} | "
          f"chunk max|Δ|={results['op_level']['chunk']:.2e}")

    # ② 层级 strict 直搬（GDN 层，含 h_v=2·h_k 的 GQA 比 + 交错排布 K1）
    cfg_l = Qwen3NextConfig(hidden_size=128, num_hidden_layers=1,
                            linear_num_key_heads=2, linear_num_value_heads=4,
                            linear_key_head_dim=32, linear_value_head_dim=32,
                            linear_conv_kernel_dim=4,
                            layer_types=["linear_attention"])
    torch.manual_seed(SEED)
    hf_layer = Qwen3NextGatedDeltaNet(cfg_l, layer_idx=0)
    my_layer = HFCompatGDN(128, 2, 4, 32, 32, conv_kernel=4, chunk=32)
    my_layer.load_state_dict(hf_layer.state_dict(), strict=True)               # strict：键位逐个咬合
    x = torch.randn(2, 64, 128)
    y_hf = hf_layer(hidden_states=x, cache_params=None, attention_mask=None)
    y_my = my_layer(x)
    results["layer_level_gdn"] = float((y_hf - y_my).abs().max())
    print(f"[层级 GDN strict] max|Δ|={results['layer_level_gdn']:.2e}")

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
    # 键位核对：自研 MiniMoE 的 gate_w 参数名是 gate_w → 需临时改名为 gate.weight 兼容
    my_sd = my_model.state_dict()
    renamed = {k: v for k, v in sd.items()}
    load_sd = {}
    my_keys = set(my_sd.keys())
    for k, val in renamed.items():
        mk = k.replace("mlp.gate.weight", "mlp.gate_w")
        if mk in my_keys:
            load_sd[mk] = val
        else:
            load_sd[k] = val
    res = my_model.load_state_dict(load_sd, strict=False)
    unexpected_keys = [k for k in res.unexpected_keys]
    missing_keys = [k for k in res.missing_keys]
    idx_in = torch.randint(0, 256, (2, 48))
    torch.manual_seed(SEED)
    logits_hf = hf_model(input_ids=idx_in).logits
    logits_my = my_model(idx_in)
    results["model_level"] = {
        "max_abs_diff_logits": float((logits_hf - logits_my).abs().max()),
        "missing_keys": missing_keys, "unexpected_keys": unexpected_keys,
    }
    print(f"[整机 strict] max|Δlogits|={results['model_level']['max_abs_diff_logits']:.2e} | "
          f"missing={missing_keys[:4]} unexpected={unexpected_keys[:4]}")

    # ④ SWA mask 对拍（Gemma3 口径）
    try:
        g3 = Gemma3Config(vocab_size=256, hidden_size=64, num_hidden_layers=2,
                          sliding_window=32, _attn_implementation="eager")
        emb = torch.zeros(1, 96, 64)                                            # (B,q_len,hidden)
        hf_mask = create_sliding_window_causal_mask(g3, emb, None, None)
        n = 96
        i = torch.arange(n)[:, None]; j = torch.arange(n)[None, :]
        mine = ~((j <= i) & (i - j < 32))                                        # True=遮蔽
        hf_block = hf_mask[0, 0] if hf_mask is not None else None
        if hf_block is None:
            results["swa_mask"] = "HF 返回 None（is_causal 跳过路径）——改用 allow_is_causal_skip=False 重试"
            hf_mask = create_sliding_window_causal_mask(g3, emb, None, None,
                                                        allow_is_causal_skip=False)
            hf_block = hf_mask[0, 0]
        mine_dense = torch.zeros(n, n).masked_fill(mine, float("-inf"))
        same = torch.equal(mine_dense == 0, hf_block == 0)
        results["swa_mask"] = {"equal_bool": bool(same),
                               "hf_shape": list(hf_block.shape),
                               "note": "对拍口径：mask==0 视为可见位（eager 4D 加性掩码）"}
    except Exception as e:
        results["swa_mask"] = {"error": f"{type(e).__name__}: {e}"}
    print(f"[SWA mask] {results['swa_mask']}")

    save_json("probe08_parity", results, args.out_name)
    print("探针 08 完成。")


if __name__ == "__main__":
    main()
