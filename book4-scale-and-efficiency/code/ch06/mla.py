# mla.py —— Book4 ch6 教学件：MultiHeadLatentAttention（MLA 双路径：显式上投影 / 权重吸收）
# 用途：正文 6.3-6.8 节的代码正身——把 MLA 的三步机构（KV 联合低秩压缩 / 解耦 RoPE / 权重吸收）
#       写成一个可直接运行、可对拍的注意力插槽。对拍五层（全部 CPU fp32、种子 20261002）：
#       ①教学件 vs 正身 moe_mla_slots.MLAAttention（同权重直搬，q_lora 两变体，期望逐位一致 0.00e+00）；
#       ②本件双路径互拍（显式 vs 吸收：整机输出 max|Δ| + 注意力权重 max|Δ|，多窗长复验）；
#       ③吸收路径 vs 正身 mla_absorbed_forward（同权重，期望逐位一致）；
#       ④HF 整机对拍：SlotLLaMA 注意力插槽换成本件 vs DeepseekV2ForCausalLM 小 config 随机权重
#         strict 直搬（探针 G 同口径复跑；含 q_lora_rank=null 的 V2-Lite 式变体）；
#       ⑤V2-Lite 真权重条件性核验：config 已核则报字段；safetensors 分片完整则 safe_open 惰性读
#         逐层 kv_b_proj（W^UK/W^UV 折叠）形状账本；未落地则如实记录降级（结构对拍已足够支撑正文）。
# 所属章节：Book4 第 6 章（MLA：把 KV 装进小箱子）。
# 运行方式：cd 工作区根目录 && source env.sh && python code/ch06/mla.py --out-name run1
#   （CPU fp32，对拍秒级、HF 臂分钟级；产物 log/book4-ch06/mla_parity_run1.json，不入库）
# 契约（plan 卷级大纲「插槽 import 契约 v1.1」）：类名 MultiHeadLatentAttention；签名
#   forward(x, cos, sin)（对齐 Book3 GroupedQueryAttention 插槽——llama409 整机可统一调用）；
#   path='explicit'|'absorbed' 双路径；支持 q_lora_rank=None（V2-Lite 式单 q_proj 分支）；
#   键位对齐 HF DeepseekV2Attention（q_a_proj/q_a_layernorm/q_b_proj|q_proj/kv_a_proj_with_mqa/
#   kv_a_layernorm/kv_b_proj/o_proj）。注意：解耦 RoPE 通道（d_h^R 维）自带自己的频率阶梯
#   （inv_freq 按 d_h^R 归一），插槽级的全宽 cos/sin 表无法切片复用——签名保留以对齐整机调用，
#   旋转角在模块内自建（与正身同款行为）。
import argparse
import glob
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "00-feasibility")))
from moe_mla_slots import (SEED, MLAAttention, SlotConfig, SlotLLaMA,       # noqa: E402
                           bootstrap_book3, mla_absorbed_forward)

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book4-ch06")


def save_json(name, obj, out_name="run1"):
    """产物落本章 log 目录（防覆写后缀）。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}_{out_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[产物] {path}")
    return path
LITE_SNAP = os.path.join(REPO_ROOT, "log", "huggingface", "hub",
                         "models--deepseek-ai--DeepSeek-V2-Lite", "snapshots")


_UNSET = object()


def _field(src, *names, default=_UNSET):
    """按别名族读 config 字段（num_attention_heads|n_head 等——HF attribute_map 同款别名）。
    default 显式传 None 时返回 None（q_lora_rank=null 的 V2-Lite 分支靠它）。"""
    for nm in names:
        v = getattr(src, nm, None) if not isinstance(src, dict) else src.get(nm)
        if v is not None:
            return v
    if default is not _UNSET:
        return default
    raise AttributeError(f"cfg 缺少字段（试过 {list(names)}）")


class MultiHeadLatentAttention(nn.Module):
    """多头潜在注意力（MLA）教学件——DSV2 §2.1 公式的代码形态，双路径前向。

    记号承 DSV2 v5 论文正字（与 config 字段对照见正文表 6.x）：
      d_c   = kv_lora_rank          KV 联合压缩维（箱子容量，DSV2=512）
      d_c'  = q_lora_rank           查询压缩维（DSV2=1536；V2-Lite=null 无查询压缩）
      d_h^R = qk_rope_head_dim      解耦 RoPE 维（共享位置键的宽度，=64）
      d_h   = qk_nope_head_dim      每头内容维（nope 段，=128）；d_v = v_head_dim 每头输出维
    键位 ↔ 论文记号：kv_a_proj_with_mqa = [W^DKV; W^KR]（行拼接，输出 split 出 c_KV 与 k_R）；
      kv_b_proj = [W^UK; W^UV]（按头行块拼接，输出 split 出 k_nope 与 v）；
      q_a_proj/q_b_proj = W^DQ/W^UQ（q_lora_rank=None 时退单 q_proj=W^Q）；o_proj = W^O。
    forward(x, cos, sin, path): x (B,n,d) -> (B,n,d)。path='explicit' 显式上投影
      （训练 / HF eager 同构）；'absorbed' 权重吸收（推理形态：W^UK 吸进 q 侧、
      W^UV 吸到输出侧，注意力直接在 d_c 维潜空间算——等价性=矩阵结合律，正文 6.6 推导）。
    """

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        src = cfg if cfg is not None else kwargs
        slots = bootstrap_book3()
        d = _field(src, "hidden_size")
        self.h = _field(src, "num_attention_heads", "n_head")
        self.d_c = _field(src, "kv_lora_rank")
        self.d_nope = _field(src, "qk_nope_head_dim")
        self.d_rope = _field(src, "qk_rope_head_dim")
        self.d_v = _field(src, "v_head_dim")
        self.q_lora_rank = _field(src, "q_lora_rank", default=None)
        self.rope_theta = _field(src, "rope_theta", default=10000.0)
        self.rope_style = _field(src, "rope_style", default="complex")   # complex=HF DSV2 约定
        self.qk_head_dim = self.d_nope + self.d_rope
        self.scaling = self.qk_head_dim ** -0.5        # √(d_h+d_h^R)：192^-0.5 档（非 128 非 512）
        self._mask_cache = {}
        # 潜向量 RMSNorm 的 eps 固定 1e-6——对齐 HF：DeepseekV2RMSNorm 不随 config.rms_norm_eps
        #（探针 G 踩坑：层内 LN 用 config eps、潜向量 LN 用类默认）
        if self.q_lora_rank is None:
            # V2-Lite 式分支：无查询压缩，q 整体投影 (d → h·(d_n+d_r))
            self.q_proj = nn.Linear(d, self.h * self.qk_head_dim, bias=False)
            self.q_a_proj, self.q_a_layernorm, self.q_b_proj = None, None, None
        else:
            self.q_proj = None
            self.q_a_proj = nn.Linear(d, self.q_lora_rank, bias=False)          # W^DQ (d → d_c')
            self.q_a_layernorm = slots.RMSNorm(self.q_lora_rank, 1e-6)          # LN_q（住查询潜向量）
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.h * self.qk_head_dim, bias=False)  # W^UQ+W^QR
        self.kv_a_proj_with_mqa = nn.Linear(d, self.d_c + self.d_rope, bias=False)  # [W^DKV;W^KR] (d→d_c+d_r)
        self.kv_a_layernorm = slots.RMSNorm(self.d_c, 1e-6)                     # LN_c（只住 c_KV 段）
        self.kv_b_proj = nn.Linear(self.d_c, self.h * (self.d_nope + self.d_v), bias=False)  # [W^UK;W^UV]
        self.o_proj = nn.Linear(self.h * self.d_v, d, bias=False)               # W^O
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # ---------------- 三件内部工具：旋转角 / 因果掩码 / 潜向量 ----------------
    def _rope_angles(self, n, device):
        """解耦 RoPE 通道自己的频率阶梯：inv_freq_i = θ^(-2i/d_h^R)，i=0..d_h^R/2-1。
        返回 (cos, sin) 各 (1,1,n,d_h^R/2)——复数约定的成对旋转角。"""
        d = self.d_rope
        inv = 1.0 / (self.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32, device=device) / d))
        ang = torch.outer(torch.arange(n, dtype=torch.float32, device=device), inv)   # (n, d/2)
        return ang.cos()[None, None], ang.sin()[None, None]

    def _rope_apply(self, q_pe, k_r):
        """只旋转位置截（q_pe (B,h,n,d_r)、k_r (B,n,d_r)；v 与 c_KV 不转）。
        complex=相邻维成对（HF DSV2 的 view_as_complex 同构）；half=两半配对（Book3 约定）。"""
        if self.rope_style == "complex":
            n, d = q_pe.shape[2], self.d_rope
            cos, sin = self._rope_angles(n, q_pe.device)
            k_in = k_r.unsqueeze(1)                                        # (B,1,n,d) 单 KV 头（全头共享）
            q0, q1 = q_pe[..., 0::2].float(), q_pe[..., 1::2].float()      # (B,h,n,d/2) 相邻成对
            k0, k1 = k_in[..., 0::2].float(), k_in[..., 1::2].float()      # (B,1,n,d/2)
            q_out = torch.stack((q0 * cos - q1 * sin, q0 * sin + q1 * cos), dim=-1).flatten(-2)
            k_out = torch.stack((k0 * cos - k1 * sin, k0 * sin + k1 * cos), dim=-1).flatten(-2)
            return q_out.to(q_pe.dtype), k_out.to(k_r.dtype)
        slots = bootstrap_book3()
        cos, sin = slots.build_rope_cache(q_pe.shape[2], self.d_rope, self.rope_theta, q_pe.device)
        return slots.apply_rope(q_pe, k_r.unsqueeze(1), cos, sin)

    def _causal_mask(self, n, device):
        if n not in self._mask_cache or self._mask_cache[n].device != device:
            self._mask_cache[n] = torch.tril(torch.ones(n, n, device=device)).view(1, 1, n, n)
        return self._mask_cache[n]

    def latents(self, x):
        """产出「可缓存」的两份潜向量——KV cache 只存它们（写入点在压缩态）：
        c_KV (B,n,d_c，已过 kv_a_layernorm) 与 k_R (B,n,d_r，未旋转——旋转后定型写缓存亦可)。"""
        c_raw = self.kv_a_proj_with_mqa(x)                                 # (B,n,d_c+d_r) [W^DKV;W^KR] 一次算
        c_kv, k_r = c_raw.split([self.d_c, self.d_rope], dim=-1)           # (B,n,d_c)+(B,n,d_r)
        return self.kv_a_layernorm(c_kv), k_r                              # LN 只住 c 段（解耦的一部分）

    def _queries(self, x):
        """查询全件：内容截 q_nope + 位置截 q_pe（未旋转）。返回 (B,h,n,d_n),(B,h,n,d_r)。"""
        B, n, _ = x.shape
        if self.q_lora_rank is not None:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))        # (B,n,h·(d_n+d_r)) 低秩两步
        else:
            q = self.q_proj(x)                                             # (B,n,h·(d_n+d_r)) 整体一步
        q = q.view(B, n, self.h, self.qk_head_dim).transpose(1, 2)         # (B,h,n,d_n+d_r)
        return q.split([self.d_nope, self.d_rope], dim=-1)

    def _w_uk_uv(self, e):
        """kv_b_proj 第 e 头的行块：W^UK_e (d_n,d_c) 与 W^UV_e (d_v,d_c)——吸收路径的切片正身。"""
        W = self.kv_b_proj.weight                                          # (h·(d_n+d_v), d_c)
        base = e * (self.d_nope + self.d_v)
        return W[base: base + self.d_nope], W[base + self.d_nope: base + self.d_nope + self.d_v]

    # ---------------- 前向：双路径 ----------------
    def forward(self, x, cos=None, sin=None, path="explicit", return_attn_probs=False):
        """x (B,n,d) -> (B,n,d)。cos/sin 为插槽契约参数（整机统一调用），解耦 RoPE 通道
        自建角度表（见类 docstring）；path 见类 docstring。return_attn_probs=True 额外返回
        注意力权重 (B,h,n,n)（双路径互拍用）。"""
        B, n, _ = x.shape
        q_nope, q_pe = self._queries(x)                                    # (B,h,n,d_n) / (B,h,n,d_r)
        c_kv, k_r = self.latents(x)                                        # (B,n,d_c) / (B,n,d_r) ←缓存写入点
        q_pe, k_pe = self._rope_apply(q_pe, k_r)                           # 只旋转位置截（内容截不转）

        if path == "explicit":
            # 形态一（训练 / HF eager 同构）：K/V 各自物化——上投影后拼头，标准注意力
            kv = self.kv_b_proj(c_kv).view(B, n, self.h, self.d_nope + self.d_v).transpose(1, 2)
            k_nope, v = kv.split([self.d_nope, self.d_v], dim=-1)          # (B,h,n,d_n) / (B,h,n,d_v)
            k = torch.cat([k_nope, k_pe.expand(B, self.h, n, self.d_rope)], dim=-1)  # (B,h,n,192)
            q_full = torch.cat([q_nope, q_pe], dim=-1)                     # (B,h,n,192)
            att = (q_full @ k.transpose(-2, -1)) * self.scaling            # (B,h,n,n) 192 维逻辑内积
        elif path == "absorbed":
            # 形态二（推理）：W^UK 吸进查询、W^UV 吸到输出——注意力在 d_c 维潜空间算
            scores = q_pe @ k_pe.transpose(-2, -1)                         # (B,h,n,n) 位置车道（k_pe (B,1,n,d_r) 广播共享）
            for e in range(self.h):
                w_uk, _ = self._w_uk_uv(e)
                q_hat = q_nope[:, e] @ w_uk                                # (B,n,d_n)·(d_n,d_c)→(B,n,d_c) 箱子语查询
                scores[:, e] += q_hat @ c_kv.transpose(-1, -2)             # (B,n,d_c)·(d_c,n)→(B,n,n) 内容车道
            att = scores * self.scaling                                    # (B,h,n,n)（scale 仍按 192 算——数学等价）
        else:
            raise ValueError(path)

        att = att.masked_fill(self._causal_mask(n, x.device)[:, :, :n, :n] == 0, float("-inf"))
        att = F.softmax(att.float(), dim=-1).to(x.dtype)                   # fp32 softmax（对齐 HF）

        if path == "explicit":
            y = att @ v                                                    # (B,h,n,d_v)
        else:
            u = None
            for e in range(self.h):
                _, w_uv = self._w_uk_uv(e)
                c_mix = att[:, e] @ c_kv                                   # (B,n,n)·(n,d_c)→(B,n,d_c) 先搅匀（箱内加权）
                y_e = c_mix @ w_uv.T                                       # (B,n,d_c)·(d_c,d_v)→(B,n,d_v) 再分杯
                u = y_e.unsqueeze(2) if u is None else torch.cat([u, y_e.unsqueeze(2)], dim=2)
            y = u.reshape(B, n, self.h * self.d_v)                         # (B,n,h·d_v)
        if path == "explicit":
            y = y.transpose(1, 2).contiguous().view(B, n, self.h * self.d_v)
        y = self.o_proj(y)                                                 # (B,n,d)
        return (y, att) if return_attn_probs else y

    # ---------------- 参数账（手算正身，正文表 6.x 与 kv_probe 复用） ----------------
    def param_ledger(self):
        """每层 MLA 参数逐项手算（含两枚潜向量 RMSNorm 的 weight）。返回 dict——与 numel() 对账。"""
        d = self.kv_a_proj_with_mqa.in_features
        q_side = ((d * self.q_lora_rank + self.q_lora_rank
                   + self.q_lora_rank * self.h * self.qk_head_dim) if self.q_lora_rank is not None
                  else d * self.h * self.qk_head_dim)
        kv_down = d * (self.d_c + self.d_rope) + self.d_c                  # [W^DKV;W^KR] + LN_c
        kv_up = self.d_c * self.h * (self.d_nope + self.d_v)               # [W^UK;W^UV]
        o = self.h * self.d_v * d
        return {"q_side": q_side, "kv_down": kv_down, "kv_up": kv_up, "o": o,
                "total": q_side + kv_down + kv_up + o}


# ---------------- 对拍 ①：教学件 vs 正身（同权重，期望逐位一致） ----------------
def check_vs_canonical(geom, q_lora, verbose=True):
    """同 seed 同几何：正身 MLAAttention 与教学件各自初始化后直搬权重，同输入前向对拍。"""
    torch.manual_seed(SEED)
    ref = MLAAttention(geom["d"], geom["h"], geom["d_c"], geom["d_n"], geom["d_r"], geom["d_v"],
                       q_lora_rank=q_lora, rope_theta=10000.0, rope_style="complex")
    torch.manual_seed(SEED)
    teach = MultiHeadLatentAttention(hidden_size=geom["d"], num_attention_heads=geom["h"],
                                     kv_lora_rank=geom["d_c"], qk_nope_head_dim=geom["d_n"],
                                     qk_rope_head_dim=geom["d_r"], v_head_dim=geom["d_v"],
                                     q_lora_rank=q_lora, rope_style="complex")
    teach.load_state_dict(ref.state_dict(), strict=True)                   # strict 直搬（异常即抛）
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(2, 48, geom["d"], generator=g)                          # (2,48,d)
    with torch.no_grad():
        yr = ref(x)                                                         # 正身显式路径
        yt = teach(x)                                                       # 教学件显式路径
        ya = teach(x, path="absorbed")                                      # 教学件吸收路径
        yabs = mla_absorbed_forward(ref, x)                                 # 正身吸收路径
    d_expl = (yt - yr).abs().max().item()
    d_abs = (ya - yabs).abs().max().item()
    if verbose:
        tag = f"q_lora={q_lora if q_lora else 'null'}"
        print(f"[① vs 正身 d={geom['d']} h={geom['h']} d_c={geom['d_c']} {tag}] "
              f"显式 max|Δ| = {d_expl:.2e} | 吸收路径 vs 正身吸收 max|Δ| = {d_abs:.2e}")
    assert d_expl == 0.0 and d_abs == 0.0, "教学件与正身不逐位一致"
    return d_expl, d_abs


# ---------------- 对拍 ②：双路径互拍（本件自身，多窗长复验 + 注意力权重级） ----------------
def check_dual_path(geom, q_lora, seq_lens=(16, 256, 1024)):
    torch.manual_seed(SEED)
    teach = MultiHeadLatentAttention(hidden_size=geom["d"], num_attention_heads=geom["h"],
                                     kv_lora_rank=geom["d_c"], qk_nope_head_dim=geom["d_n"],
                                     qk_rope_head_dim=geom["d_r"], v_head_dim=geom["d_v"],
                                     q_lora_rank=q_lora, rope_style="complex").float()
    g = torch.Generator().manual_seed(SEED)
    out = {"seq_lens": {}, "q_lora": q_lora}
    with torch.no_grad():
        for n in seq_lens:
            x = torch.randn(2, n, geom["d"], generator=g)                   # (2,n,d)
            y_e, p_e = teach(x, return_attn_probs=True)                     # 显式：输出+注意力权重
            y_a, p_a = teach(x, path="absorbed", return_attn_probs=True)    # 吸收：同上
            out["seq_lens"][n] = {
                "logits_max_abs": float(f"{(y_e - y_a).abs().max().item():.3e}"),
                "attn_probs_max_abs": float(f"{(p_e - p_a).abs().max().item():.3e}"),
                "argmax_agree": (y_e.argmax(-1) == y_a.argmax(-1)).float().mean().item(),
            }
        n0 = seq_lens[0]
        x = torch.randn(2, n0, geom["d"], generator=g)
        y_e, y_a = teach(x), teach(x, path="absorbed")
        rel = ((y_e - y_a).norm() / y_e.norm()).item()
    row = out["seq_lens"][n0]
    print(f"[② 双路径 d={geom['d']} h={geom['h']} d_c={geom['d_c']} q_lora={q_lora or 'null'}] "
          f"n={n0}: 输出 max|Δ| = {row['logits_max_abs']:.2e}（相对 {rel:.1e}）| "
          f"注意力权重 max|Δ| = {row['attn_probs_max_abs']:.2e} | 多窗长 "
          + " ".join(f"n={n}:{v['logits_max_abs']:.1e}" for n, v in out["seq_lens"].items()))
    assert all(v["logits_max_abs"] < 1e-4 and v["argmax_agree"] == 1.0 for v in out["seq_lens"].values())
    out["rel_err"] = float(f"{rel:.3e}")
    return out


# ---------------- 对拍 ④：HF 整机（SlotLLaMA 注意力插槽换教学件 vs DeepseekV2ForCausalLM） ----------------
def check_hf_model(q_lora):
    """探针 G 同口径复跑：小 config 随机权重 strict 直搬。注意力插槽=本教学件（其余=正身件），
    state_dict 键名与 MLAAttention 一致，直搬不受影响——「插槽可换」本身的现场验证。"""
    from transformers import DeepseekV2Config, DeepseekV2ForCausalLM
    torch.manual_seed(SEED)
    mla = dict(kv_lora_rank=64, q_lora_rank=q_lora, qk_nope_head_dim=32,
               qk_rope_head_dim=16, v_head_dim=32, rope_style="complex")
    ours = SlotLLaMA(SlotConfig(vocab_size=512, hidden_size=256, num_hidden_layers=2,
                                num_attention_heads=4, num_key_value_heads=1, intermediate_size=704,
                                moe=dict(n_expert=4, top_k=2, expert_dim=128, n_shared=1,
                                         variant="hf_deepseek", routed_scaling_factor=1.0),
                                moe_first_k_dense=1, mla=mla, rope_theta=10000.0, rms_norm_eps=1e-5))
    for layer in ours.model.layers:                                        # 注意力插槽换成本件（键名不变）
        layer.self_attn = MultiHeadLatentAttention(
            {**mla, "hidden_size": 256, "num_attention_heads": 4, "rope_theta": 10000.0})
    cfg = DeepseekV2Config(vocab_size=512, hidden_size=256, intermediate_size=704,
                           moe_intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                           num_key_value_heads=1, first_k_dense_replace=1, n_routed_experts=4,
                           n_shared_experts=1, num_experts_per_tok=2, kv_lora_rank=64,
                           q_lora_rank=q_lora, qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
                           topk_method="greedy", routed_scaling_factor=1.0, norm_topk_prob=False,
                           rope_theta=10000.0, rms_norm_eps=1e-5, tie_word_embeddings=False,
                           max_position_embeddings=512)
    hf = DeepseekV2ForCausalLM(cfg).eval()
    hf.load_state_dict(ours.state_dict(), strict=True, assign=True)        # strict 直搬
    g = torch.Generator().manual_seed(SEED)
    x = torch.randint(0, 512, (2, 128), generator=g)
    with torch.no_grad():
        lo, _ = ours(x, x)
        lh = hf(x).logits
    delta = (lo - lh).abs().max().item()
    agree = (lo.argmax(-1) == lh.argmax(-1)).float().mean().item()
    print(f"[④ vs HF 5.18.0 q_lora={q_lora if q_lora else 'null'}] strict 直搬通过 "
          f"{sum(p.numel() for p in ours.parameters()):,} 参 | max|Δlogits| = {delta:.3e} | "
          f"argmax 一致率 = {agree:.4f}")
    assert delta < 1e-4 and agree == 1.0
    return {"max_abs_diff_logit": delta, "argmax_agreement": agree,
            "params": sum(p.numel() for p in ours.parameters())}


# ---------------- 对拍 ⑤：V2-Lite 真权重条件性（config 已核；权重完整则 safe_open 逐层形状账本） ----------------
def check_v2lite():
    """条件性真权重核验（大纲前置任务 13）：
    a) config.json（本地缓存，一手）字段核对——Lite 的 MLA 几何与 q_lora_rank=null 事实；
    b) safetensors 分片若已完整落地：safe_open 惰性读（零数值加载）逐层验证
       kv_b_proj（W^UK/W^UV 折叠在一个矩阵）等五投影形状账本；
    c) 未落地/不完整：如实记录降级——结构对拍（①④）已足够支撑正文。"""
    res = {"config_verified": False, "weights_complete": False, "shapes": {}, "degraded": False}
    # snapshots/<revision>/ 目录下才是 config.json 与分片——glob 必须再进一层（写浅一层会永远找不到文件）
    snaps = sorted(glob.glob(os.path.join(LITE_SNAP, "*", "*")))
    cfg_path = next((p for p in snaps if p.endswith("config.json")), None)
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            c = json.load(f)
        res["config"] = {k: c.get(k) for k in
                         ("hidden_size", "num_hidden_layers", "num_attention_heads", "kv_lora_rank",
                          "q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
                          "n_routed_experts", "n_shared_experts", "num_experts_per_tok",
                          "first_k_dense_replace")}
        ok = (c["hidden_size"] == 2048 and c["num_hidden_layers"] == 27 and c["num_attention_heads"] == 16
              and c["kv_lora_rank"] == 512 and c["q_lora_rank"] is None
              and c["qk_nope_head_dim"] == 128 and c["qk_rope_head_dim"] == 64 and c["v_head_dim"] == 128
              and c["n_routed_experts"] == 64 and c["num_experts_per_tok"] == 6
              and c["n_shared_experts"] == 2 and c["first_k_dense_replace"] == 1)
        res["config_verified"] = ok
        print(f"[⑤ V2-Lite config] 本地缓存核验{'通过' if ok else '失败'}：n_h=16、q_lora_rank=null、"
              f"d_c=512、d_h^R=64、L=27（64 选 6+2 共享、首层稠密）")
    st_files = [p for p in snaps if p.endswith(".safetensors") and os.path.exists(os.path.realpath(p))]
    total = sum(os.path.getsize(p) for p in st_files)
    res["safetensors_files"], res["safetensors_bytes"] = len(st_files), total
    if not st_files or total < 30 * 2**30:
        res["degraded"] = True
        res["note"] = (f"权重分片未完整落地（{len(st_files)} 个文件 / {total/2**30:.1f} GiB，"
                       f"预期约 31.4 GiB）——按预登记降级：config 已核 + 结构对拍（①④）足够支撑正文；"
                       "权重落地后重跑本臂可补逐层形状账本")
        print(f"[⑤ V2-Lite 权重] 降级记录：{res['note']}")
        return res
    # 完整落地：safe_open 惰性读逐层形状账本（峰值单张量数百 MB 以内——只读 kv_b_proj）
    from safetensors import safe_open
    h, d_c, d_n, d_r, d_v, d, L = 16, 512, 128, 64, 128, 2048, 27
    expect = {"kv_a_proj_with_mqa": (d_c + d_r, d), "kv_b_proj": (h * (d_n + d_v), d_c),
              "o_proj": (d, h * d_v), "q_proj": (h * (d_n + d_r), d)}
    n_checked = 0
    for path in st_files:
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if ".self_attn.kv_" in key or ".self_attn.o_proj" in key or ".self_attn.q_proj." in key:
                    name = key.split(".")[-2]
                    if name not in expect:
                        continue
                    shape = tuple(f.get_slice(key).get_shape())
                    assert shape == expect[name], (key, shape, expect[name])
                    res["shapes"][f"L{key.split('.')[2]}.{name}"] = shape
                    n_checked += 1
    res["weights_complete"] = True
    res["n_tensors_checked"] = n_checked
    print(f"[⑤ V2-Lite 权重] {len(st_files)} 分片完整：逐层投影形状账本核验 {n_checked} 项全过"
          f"（kv_b_proj {expect['kv_b_proj']} = W^UK/W^UV 折叠、kv_a_proj_with_mqa {expect['kv_a_proj_with_mqa']}）")
    return res


def main(out_name):
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.perf_counter()
    torch.manual_seed(SEED)
    print(f"[环境] CPU fp32 | 种子 {SEED}")
    geom_reg = dict(d=64, h=4, d_c=32, d_n=16, d_r=8, d_v=16)       # parity 注册表小档
    geom_probe = dict(d=512, h=8, d_c=256, d_n=64, d_r=64, d_v=64)  # 探针 C 档

    # ① vs 正身（两变体：q_lora=64 与 q_lora=null——V2-Lite 式分支的逐位验收）
    vs_canon = {}
    for q_lora in (64, None):
        d_expl, d_abs = check_vs_canonical(geom_reg, q_lora)
        vs_canon[str(q_lora)] = {"explicit": d_expl, "absorbed": d_abs}

    # ② 双路径互拍（probe C 几何 + q_lora=null 变体；注意力权重级 + 多窗长）
    dual = {"q128": check_dual_path(geom_probe, 128), "qnull": check_dual_path(geom_probe, None)}

    # ③（并入 ①）吸收路径 vs 正身吸收——同权重逐位（check_vs_canonical 已含）

    # ④ HF 整机对拍（两变体；q_lora=null 即 V2-Lite 式整机）
    hf = {"q_lora=64": check_hf_model(64), "q_lora=null": check_hf_model(None)}

    # ⑤ V2-Lite 真权重条件性
    lite = check_v2lite()

    # 参数账 assert（两分支手算=实测）
    torch.manual_seed(SEED)
    for q_lora, tag in ((64, "q64"), (None, "qnull")):
        m = MultiHeadLatentAttention(hidden_size=64, num_attention_heads=4, kv_lora_rank=32,
                                     qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
                                     q_lora_rank=q_lora)
        led = m.param_ledger()
        n_real = sum(p.numel() for p in m.parameters())
        assert led["total"] == n_real, (tag, led["total"], n_real)
        print(f"[参数账] {tag}: 手算 {led['total']:,} = 实测 {n_real:,}（q_side={led['q_side']:,} "
              f"kv_down={led['kv_down']:,} kv_up={led['kv_up']:,} o={led['o']:,}）")

    report = {"seed": SEED, "device": "cpu fp32",
              "geom_registry": geom_reg, "geom_probe": geom_probe,
              "vs_canonical_bitwise": vs_canon, "dual_path": dual, "hf_model": hf,
              "v2lite_real_weights": lite,
              "wall_sec": round(time.perf_counter() - t0, 1), "verdict": "PASS"}
    save_json("mla_parity", report, out_name)
    print(f"[判定] PASS | 墙钟 {report['wall_sec']} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch6 教学件 MLA：五层对拍 + V2-Lite 条件性核验")
    ap.add_argument("--out-name", default="run1", help="产物 JSON 后缀（防覆写）")
    args = ap.parse_args()
    main(args.out_name)
