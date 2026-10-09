# hybrid409.py —— Book5 ch9 正式件：三刀整机（两刀机 9 层注意力插槽换 GDN → 3:1 混合，等总参 ≈409M）
# 用途：单轨主线演化线本册段收官——「三刀改装史写在 import 语句里」，供货方即三代改装史：
#       归一化插槽（不动件）   import 自 Book3 ch02/rmsnorm.py::RMSNorm(hidden_size, eps)（经由 llama409）
#       首层稠密件（机构保留） import 自 Book3 ch03/swiglu.py::SwiGLUMLP(cfg)（经由 llama409；定版不启用）
#       位置插槽（不动件）     import 自 Book3 ch04/rope.py::build_rope_cache（经由 llama409）
#       前馈插槽（第一刀）     import 自 Book4 ch05/deepseek_moe.py::DeepSeekMoE（经由 llama409）
#       注意力插槽（第二刀）   import 自 Book4 ch06/mla.py::MultiHeadLatentAttention（经由 llama409）
#                               —— 仅 3 层保留（config 0 起 3/7/11；人读第 4/8/12 层，末层全局承 K3 惯例）
#       注意力插槽（第三刀）   import 自本册 ch05/gdn.py::GatedDeltaNet(d, h_k, h_v, d_k, d_v)
#                               —— 其余 9 层换线性注意力（h_k=h_v=16 / d_k=128 / d_v=64，层参 7,393,312）
# 底座=Book4 cand2 两刀机（409,115,648；d=1024/L=12/MLA_LITE/E8 top2 s1 w=938）；
#   等总参补偿：w_expert 938→810（ΔMoE=12×3,538,944=42,467,328 抵 Δattn=9×4,705,824=42,352,416）；
#   净差 −114,912（−0.028%）→ 三刀总参 409,000,736（与 cand2 等总参的 ≈409M 口径；精确值以本件 assert 为正身）。
#   —— 三形态等参对照链：Book3 稠密 207M（207,119,360）→ Book4 两刀 409M（409,115,648）→ 本册三刀 409M（409,000,736）。
# 【层号双口径（全文一套）】层模式 [gdn,gdn,gdn,mla]×3 为 config 0 起索引——MLA 在 3/7/11；
#   行文/例题固定人读 1 起：第 4/8/12 层 MLA（首次出现处加换算句）。
# 【分类型状态账】MLA 3 层：每 token (d_c+d_h^R)=320 元素 × 3 层 = 960 元素/token，随 n 线性（无界）；
#   GDN 9 层：每层固定 h_v·d_k·d_v + (k_conv−1)·conv_dim = 131,072+15,360 = 146,432，× 9 = 1,317,888（与 n 无关）。
# 对拍口径声明（大纲钉死）：无整机正身可 strict 直搬——GDN 件层级已与 HF 双 kernel 对拍（ch05 gdn.py
#   1e-8 级）+ 线性层文献互证（Qwen3-Next/Kimi Linear 同族机构）；MLA/MoE 件承 Book4 对拍资产。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch09/hybrid409.py" [--out-name run1]
#   自测 CPU fp32 秒~分钟级（409M 参数构建 + 双口径 assert + 分类型账打印 + 初始 loss 自检）。
# 产物：log/book5-ch09/hybrid409_selftest_{out}.json（不入库）
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
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book5-ch09")
SEED = 20261002                                            # 全书统一种子

# ---------------- 三候选 bootstrap（llama409 `_B3_CANDS` 已验证模式；书仓定稿后同款可命中） ----------------
_B4_CH09_CANDS = [
    os.path.join(REPO_ROOT, "code", "Book4-规模与效率", "ch09"),                          # 工作区布局
    os.path.join(REPO_ROOT, "from-models-to-vllm", "book4-scale-and-efficiency", "code", "ch09"),
    os.path.join(REPO_ROOT, "book4-scale-and-efficiency", "code", "ch09"),                # 书仓裸布局
]
_B5_CH05_CANDS = [
    os.path.join(REPO_ROOT, "code", "Book5-前沿混合架构", "ch05"),
    os.path.join(REPO_ROOT, "from-models-to-vllm", "book5-frontier-hybrid-architectures", "code", "ch05"),
    os.path.join(REPO_ROOT, "book5-frontier-hybrid-architectures", "code", "ch05"),
]


def bootstrap():
    """挂两刀机正身（llama409——import 时自挂 Book3 四件+Book4 ch05/ch06）与本册 gdn 件的物理落点。"""
    for cands, need in ((_B4_CH09_CANDS, "llama409.py"), (_B5_CH05_CANDS, "gdn.py")):
        for p in cands:
            if os.path.exists(os.path.join(p, need)):
                if p not in sys.path:
                    sys.path.insert(0, p)
                break
        else:
            raise FileNotFoundError(f"未找到兄弟目录（需要 {need}）：{cands}")
    import llama409                                       # noqa: F401  Book4 ch9 两刀机（其 bootstrap 挂 Book3/Book4）
    import gdn                                            # noqa: F401  本册 ch5 GDN 件
    return llama409, gdn


L409, _gdn_mod = bootstrap()
# —— 三刀改装史写在 import 语句里（供货方=改装史；与两刀机同款纪律：import 复用不复制）——
from llama409 import (RMSNorm, SwiGLUMLP, rope_mod,        # noqa: E402  Book3 不动件（归一化/稠密/位置——经由两刀机）
                      MultiHeadLatentAttention, DeepSeekMoE,           # noqa: E402  Book4 两刀（MLA 插槽/MoE 前馈）
                      TwoKnifeConfig, MLA_LITE, CANDS)      # noqa: E402  Book4 config 正身与双臂定版
from gdn import GatedDeltaNet                              # noqa: E402  本册第三刀：线性注意力插槽

# ---------------- 三刀 config 定版（大纲 v1.1 §9.5；数字正身=本件 assert） ----------------
GDN_GEO = dict(h_k=16, h_v=16, d_k=128, d_v=64)            # GDN 插槽几何（conv_kernel=4 默认）
GDN_LAYER_PARAMS = 7_393_312                               # 手算正身（下方 hand_account 复算 assert）
MLA_ATTN_PARAMS = 2_687_488                                # MLA_LITE @ d=1024（含 q_a/kv_a 两枚潜向量 RMSNorm）
HYBRID_W_EXPERT = 810                                      # 等总参补偿宽（938→810）
EXPECT_TOTAL = 409_000_736                                 # 三刀总参正身（与 cand2 差 −114,912 = −0.028%）
EXPECT_ACT_TOTAL = 197_073_696                             # 激活参正身（含 lm_head/不含输入 embedding/含路由门）
EXPECT_ACT_NONEMB = 164_305_696                            # 激活 non-embedding（= 激活参 − lm_head）
CAND2_TOTAL = CANDS["cand2"]["expect_total"]               # 409,115,648（两刀机，对照锚）
LAYER_PATTERN = ["gdn", "gdn", "gdn", "mla"] * 3           # config 0 起：MLA@3/7/11（人读第 4/8/12 层）


class ThreeKnifeConfig(TwoKnifeConfig):
    """三刀整机 config：两刀 config 加两字段——layer_types（0 起层模式）与 gdn（GDN 插槽几何）。

    字段名 layer_types 对齐 2026 机型 config 惯例（Qwen3.5/gpt-oss 同名）；"mla"/"gdn" 为本件教学取值。
    first_k_dense_replace 定版 0（12/12 MoE——承 cand2 定版；首层稠密机构不在三刀机档位）。
    """

    def __init__(self, vocab_size=32000, hidden_size=1024, num_hidden_layers=12,
                 num_attention_heads=16, num_key_value_heads=8, head_dim=64,
                 intermediate_size=2816, rope_theta=10000.0, rms_norm_eps=1e-5,
                 initializer_range=0.02, n_expert=8, top_k=2, n_shared=1,
                 w_expert=HYBRID_W_EXPERT, mla=None, layer_types=None, gdn=None,
                 first_k_dense_replace=0, scoring="softmax", routed_scaling_factor=1.0,
                 max_position_embeddings=8192):
        super().__init__(vocab_size=vocab_size, hidden_size=hidden_size,
                         num_hidden_layers=num_hidden_layers,
                         num_attention_heads=num_attention_heads,
                         num_key_value_heads=num_key_value_heads, head_dim=head_dim,
                         intermediate_size=intermediate_size, rope_theta=rope_theta,
                         rms_norm_eps=rms_norm_eps, initializer_range=initializer_range,
                         n_expert=n_expert, top_k=top_k, n_shared=n_shared,
                         w_expert=w_expert, mla=(mla or MLA_LITE),
                         first_k_dense_replace=first_k_dense_replace, scoring=scoring,
                         routed_scaling_factor=routed_scaling_factor,
                         max_position_embeddings=max_position_embeddings)
        assert first_k_dense_replace == 0, "三刀机定版 12/12 MoE（承 cand2）"
        self.gdn = dict(gdn or GDN_GEO)
        self.layer_types = list(layer_types or LAYER_PATTERN)
        assert len(self.layer_types) == self.num_hidden_layers
        assert set(self.layer_types) <= {"mla", "gdn"}


def make_hybrid_cfg(**over):
    """三刀定版 config（over 覆盖任意字段——mini 冒烟臂用）。"""
    return ThreeKnifeConfig(**over)


class ThreeKnifeDecoderLayer(nn.Module):
    """残差河恒宽的三刀层——拓扑仍是 Book2/3 三插槽 Block；注意力插槽按 layer_types 逐层供货：

        x = x + attn(input_layernorm(x), cos, sin)     RMSNorm(Book3 ch2) -> MLA(Book4 ch6) | GDN(本册 ch5)
        x = x + mlp(post_attention_layernorm(x))       RMSNorm(Book3 ch2) -> MoE(Book4 ch5, w=810)
    输入/输出 (B,n,C)。cos/sin 为插槽契约参数：MLA 解耦 RoPE 通道自建阶梯、GDN 结构级 NoPE——
    两者都不消费此表（与两刀机同款契约，层循环照发保持统一调用）。
    """

    def __init__(self, cfg: ThreeKnifeConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)       # 插槽①（Book3 第 2 章）
        if cfg.layer_types[layer_idx] == "mla":                                 # 插槽④第二刀（Book4 第 6 章）
            self.self_attn = MultiHeadLatentAttention(
                dict(hidden_size=cfg.hidden_size, num_attention_heads=cfg.num_attention_heads,
                     rope_theta=cfg.rope_theta, rope_style="complex", **cfg.mla))
        else:                                                                   # 插槽④第三刀（本册第 5 章）
            g = cfg.gdn
            self.self_attn = GatedDeltaNet(cfg.hidden_size, g["h_k"], g["h_v"],
                                           g["d_k"], g["d_v"])
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = DeepSeekMoE(hidden_size=cfg.hidden_size, n_expert=cfg.n_expert,   # 第一刀（Book4 第 5 章）
                               top_k=cfg.top_k, expert_dim=cfg.w_expert,
                               n_shared=cfg.n_shared, scoring=cfg.scoring,
                               n_group=1, topk_group=1, norm_topk_prob=False,
                               routed_scaling_factor=cfg.routed_scaling_factor)
        self.layer_idx = layer_idx

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)   # (B,n,C) 残差河
        x = x + self.mlp(self.post_attention_layernorm(x))          # (B,n,C)
        return x


class ThreeKnifeBackbone(nn.Module):
    """embed_tokens + L 层 ThreeKnifeDecoderLayer + 终态 RMSNorm（命名对齐 HF DeepseekV2Model 惯例）。"""

    def __init__(self, cfg: ThreeKnifeConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)       # (V,C)
        self.layers = nn.ModuleList([ThreeKnifeDecoderLayer(cfg, i)
                                     for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)                  # 出口前最后一枚

    def forward(self, idx):
        B, n = idx.shape                                                       # (B,n)
        cfg = self.cfg
        x = self.embed_tokens(idx)                                             # (B,n) -> (B,n,C)
        cos, sin = rope_mod.build_rope_cache(n, cfg.head_dim, cfg.rope_theta,   # (1,n,64) x2（Book3 第 4 章；
                                            idx.device)                         #  MLA 自建阶梯/GDN NoPE，契约用）
        for layer in self.layers:
            x = layer(x, cos, sin)                                             # L x (B,n,C)
        return self.norm(x)                                                    # (B,n,C)


class ThreeKnifeLM(nn.Module):
    """三刀整机：wte + L x [pre-RMSNorm -> MLA|GDN | pre-RMSNorm -> MoE] + 终态 RMSNorm + untied lm_head。

    前向接口与 Book2/3/4 全线一致：idx (B,n) -> logits (B,n,V)；给 targets 返回逐 token 交叉熵（fp32）。
    键位沿用两刀机族（self_attn.* / mlp.* / 两枚 layernorm）；GDN 层的 self_attn 键位=本册 ch5 教学件
    （in_proj_qkvz/in_proj_ba/conv1d/A_log/dt_bias/out_proj——HF Qwen3Next 同族命名）。
    """

    def __init__(self, cfg: ThreeKnifeConfig = None, init_std: float | None = None):
        super().__init__()
        cfg = cfg or make_hybrid_cfg()
        self.cfg = cfg
        self.model = ThreeKnifeBackbone(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)   # untied（Book3 第 5 章口径）
        std = init_std or cfg.initializer_range
        self.apply(lambda m: self._init(m, std))

    @staticmethod
    def _init(module, std):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        # conv1d/A_log/dt_bias 保留 GatedDeltaNet 构造器初始化（ch5 正身件行为，不改）

    def n_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.model.embed_tokens.weight.numel() + self.lm_head.weight.numel()
        return n

    def forward(self, idx, targets=None):
        x = self.model(idx)                                                    # (B,n,C)
        logits = self.lm_head(x)                                               # (B,n,C)·(C,V) -> (B,n,V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),  # (B*n,V) fp32
                                  targets.reshape(-1))
        return logits, loss


# ---------------- 参数逐项手算账（assert 的对账另一侧；公式来源：Book4 DESIGN-409M §3 + 本册 ch5） ----------------
def gdn_layer_params(g=None, d=1024, conv_kernel=4):
    """GDN 插槽层参数手算：in_proj_qkvz + in_proj_ba + conv1d + A_log/dt_bias + out_proj。"""
    g = g or GDN_GEO
    conv_dim = 2 * g["h_k"] * g["d_k"] + g["h_v"] * g["d_v"]                    # 2·16·128+16·64 = 5,120
    in_qkvz = d * (2 * g["h_k"] * g["d_k"] + 2 * g["h_v"] * g["d_v"])          # 1024×6,144 = 6,291,456
    in_ba = d * 2 * g["h_v"]                                                   # 1024×32 = 32,768
    conv = conv_dim * conv_kernel                                              # 5,120×4 = 20,480
    gates = 2 * g["h_v"]                                                       # A_log+dt_bias = 32
    out_p = g["h_v"] * g["d_v"] * d                                            # 1,024×1,024 = 1,048,576
    return in_qkvz + in_ba + conv + gates + out_p


def mla_layer_params(cfg: ThreeKnifeConfig):
    """MLA 插槽层参数手算（含 q_a/kv_a 两枚潜向量 RMSNorm）——Book4 hand_account 同式。"""
    d, h, m = cfg.hidden_size, cfg.num_attention_heads, cfg.mla
    r_q = m.get("q_lora_rank")
    q_side = (d * r_q + r_q * h * (m["qk_nope_head_dim"] + m["qk_rope_head_dim"]))
    kv_down = d * (m["kv_lora_rank"] + m["qk_rope_head_dim"])                   # [W^DKV;W^KR] 融合一件
    kv_up = m["kv_lora_rank"] * h * (m["qk_nope_head_dim"] + m["v_head_dim"])   # [W^UK;W^UV] 按头行块拼接
    o_side = h * m["v_head_dim"] * d
    return q_side + kv_down + kv_up + o_side + (m["kv_lora_rank"] + (r_q or 0))


def moe_layer_params(cfg: ThreeKnifeConfig):
    """MoE 层参数手算：路由门 E·d + 路由专家 E·3dw + 共享 n_shared·3dw。"""
    d = cfg.hidden_size
    return (cfg.n_expert * d + cfg.n_expert * 3 * d * cfg.w_expert
            + cfg.n_shared * 3 * d * cfg.w_expert)


def moe_activated_params(cfg: ThreeKnifeConfig):
    """MoE 激活参数手算（每 token 实际动用）：路由门 E·d + (top_k 路由+共享) 专家各 3dw。"""
    d = cfg.hidden_size
    return cfg.n_expert * d + (cfg.top_k + cfg.n_shared) * 3 * d * cfg.w_expert


def activated_account(cfg: ThreeKnifeConfig):
    """激活参逐项手算（本册口径=含 lm_head、不含输入 embedding、含路由门；MoE 按 top_k+shared 计）。

    GDN/MLA 层各 = 注意力插槽 + 激活 MoE + 2 枚 RMSNorm；出口 = untied lm_head + 终态 norm。
    正身：GDN 层 14,868,512、MLA 层 10,162,688、总 197,073,696、non-emb 164,305,696
    （DESIGN-3TO1 §3 激活参行——2026-10-09 审计勘正 16,868,512/215,073,696 旧数后立此 assert）。
    """
    d, L, V = cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size
    gdn_attn, mla_attn = gdn_layer_params(cfg.gdn, d), mla_layer_params(cfg)
    moe_act = moe_activated_params(cfg)
    n_mla = sum(1 for t in cfg.layer_types if t == "mla")
    n_gdn = L - n_mla
    gdn_layer = gdn_attn + moe_act + 2 * d
    mla_layer = mla_attn + moe_act + 2 * d
    lm_head = V * d
    total = n_gdn * gdn_layer + n_mla * mla_layer + lm_head + d
    return {"gdn_layer_activated": gdn_layer, "mla_layer_activated": mla_layer,
            "moe_activated_per_layer": moe_act, "lm_head": lm_head,
            "total_activated": total, "non_embedding": total - lm_head}


def hand_account(cfg: ThreeKnifeConfig):
    """三刀总参逐项手算（无 bias、untied、RMSNorm 只 weight）。返回明细 dict。"""
    d, L, V = cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size
    gdn_attn, mla_attn = gdn_layer_params(cfg.gdn, d), mla_layer_params(cfg)
    moe = moe_layer_params(cfg)
    n_mla = sum(1 for t in cfg.layer_types if t == "mla")
    n_gdn = L - n_mla
    layers = n_mla * (mla_attn + 2 * d + moe) + n_gdn * (gdn_attn + 2 * d + moe)
    emb = V * d * 2 + d                                                       # untied 两份 + 终态 norm
    return {"gdn_attn_per_layer": gdn_attn, "mla_attn_per_layer": mla_attn,
            "mlp_moe_per_layer": moe, "n_mla_layers": n_mla, "n_gdn_layers": n_gdn,
            "layers_total": layers, "embedding_and_head": emb, "total": layers + emb}


# ---------------- 分类型状态账（本册新账——三刀机的 KV/状态读数） ----------------
def typed_state_account(cfg: ThreeKnifeConfig, conv_kernel=4, ns=(1024, 4096, 16384, 65536, 262144)):
    """分类型 KV/状态账（元素数口径，论文 Table 1 同款）：

    MLA 层（3 层）：每 token (d_c + d_h^R) = 320 元素——潜向量 c_KV 一份 + 解耦共享键 k_R 一份；
                   n token 共 3×320×n（无界，随 n 线性）。
    GDN 层（9 层）：每层固定 h_v·d_k·d_v（递归状态矩阵 S）+ (k_conv−1)·conv_dim（因果短卷积滞留）
                   = 131,072 + 15,360 = 146,432——与 n 无关；9 层合计 1,317,888。
    交叉点 n*：全 MLA 等价账 = 9 层 GDN 固定账 ÷ 每 token MLA 账 = 1,317,888/960 ≈ 1,373——
               n 超过此值后「9 层 GDN 的全部状态」仍恒定，而 3 层 MLA 的账继续线性上涨。
    """
    g = cfg.gdn
    mla_idx = [i for i, t in enumerate(cfg.layer_types) if t == "mla"]
    gdn_idx = [i for i, t in enumerate(cfg.layer_types) if t == "gdn"]
    per_tok_mla = len(mla_idx) * (cfg.mla["kv_lora_rank"] + cfg.mla["qk_rope_head_dim"])
    conv_dim = 2 * g["h_k"] * g["d_k"] + g["h_v"] * g["d_v"]
    gdn_fixed_per_layer = g["h_v"] * g["d_k"] * g["d_v"] + (conv_kernel - 1) * conv_dim
    gdn_fixed = len(gdn_idx) * gdn_fixed_per_layer
    crossover = (gdn_fixed / per_tok_mla) if per_tok_mla else None
    totals = {n: {"mla_unbounded_elems": per_tok_mla * n, "gdn_fixed_elems": gdn_fixed,
                  "total_elems": per_tok_mla * n + gdn_fixed} for n in ns}
    return {"mla_layer_indices_config0": mla_idx, "gdn_layer_indices_config0": gdn_idx,
            "mla_per_token_elems": per_tok_mla, "gdn_fixed_per_layer_elems": gdn_fixed_per_layer,
            "gdn_fixed_total_elems": gdn_fixed, "crossover_n": round(crossover, 1) if crossover else None,
            "totals_by_n": totals}


# ---------------- 自测：双口径 assert + 分类型账打印 + 初始 loss 数学自检 ----------------
def self_test(out_name):
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.perf_counter()
    torch.manual_seed(SEED)
    report = {"seed": SEED, "date": "2026-10-08", "arms": {}}

    cfg = make_hybrid_cfg()
    model = ThreeKnifeLM(cfg)
    hand = hand_account(cfg)
    total = model.n_params()

    # ① 双口径 assert：模块枚举 == 手算逐项 == 定版常数（精确值以代码为正身——逐位报数）
    assert total == hand["total"] == EXPECT_TOTAL, \
        f"三刀参数账失败：枚举 {total} vs 手算 {hand['total']} vs 定版 {EXPECT_TOTAL}"
    assert hand["gdn_attn_per_layer"] == GDN_LAYER_PARAMS
    assert hand["mla_attn_per_layer"] == MLA_ATTN_PARAMS
    # ② 层模式逐层核（config 0 起 3/7/11 为 MLA——人读第 4/8/12 层）
    got_types = ["mla" if isinstance(l.self_attn, MultiHeadLatentAttention) else "gdn"
                 for l in model.model.layers]
    assert got_types == cfg.layer_types == LAYER_PATTERN, got_types
    gdn_layer_n = sum(p.numel() for p in model.model.layers[0].self_attn.parameters())
    mla_layer_n = sum(p.numel() for p in model.model.layers[3].self_attn.parameters())
    assert gdn_layer_n == GDN_LAYER_PARAMS and mla_layer_n == MLA_ATTN_PARAMS
    gap = total - CAND2_TOTAL
    print(f"[三刀机] 总参 {total:,}（枚举=手算=定版，逐位一致）| GDN 层 {gdn_layer_n:,} × 9 + "
          f"MLA 层 {mla_layer_n:,} × 3 | vs cand2 两刀 {CAND2_TOTAL:,}：差 {gap:+,}"
          f"（{gap / CAND2_TOTAL * 100:+.3f}%——等总参补偿 w 938→810 的净残差）")
    report["arms"]["hybrid409"] = {
        "total": total, "gdn_attn_per_layer": gdn_layer_n, "mla_attn_per_layer": mla_layer_n,
        "vs_cand2_gap": gap, "vs_cand2_pct": round(gap / CAND2_TOTAL * 100, 4),
        "layer_types_config0": got_types, "mla_human_reading": "第 4/8/12 层（1 起）= 3/7/11（0 起）",
        "account": hand}

    # ①′ 激活参 assert（手算=定版；本册口径=含 lm_head、不含输入 embedding、含路由门——MoE 激活按
    #     top_k 路由+共享逐 token 计，属公式量；与总参账的勾稽：总参−输入嵌入−激活参=闲置路由专家参数）
    act = activated_account(cfg)
    assert act["total_activated"] == EXPECT_ACT_TOTAL, \
        f"激活参账失败：手算 {act['total_activated']} vs 定版 {EXPECT_ACT_TOTAL}"
    assert act["non_embedding"] == EXPECT_ACT_NONEMB
    assert act["gdn_layer_activated"] == 7_393_312 + 7_464_960 + 8_192 + 2_048   # 14,868,512
    assert act["mla_layer_activated"] == 2_687_488 + 7_464_960 + 8_192 + 2_048   # 10,162,688
    idle_experts = ((cfg.n_expert - cfg.top_k) * 3 * cfg.hidden_size * cfg.w_expert
                    * cfg.num_hidden_layers)                                 # 未选中路由专家（12 层合计）
    assert total - model.model.embed_tokens.weight.numel() - act["total_activated"] == idle_experts
    print(f"[激活参] {act['total_activated']:,}（手算=定版；含 lm_head 32,768,000、不含输入 embedding）"
          f"| GDN 层 {act['gdn_layer_activated']:,} × 9 + MLA 层 {act['mla_layer_activated']:,} × 3"
          f" | non-emb {act['non_embedding']:,}（勾稽：总参−嵌入−激活=闲置路由专家 {idle_experts:,} ✓）")
    report["arms"]["hybrid409"]["activated_account"] = act
    del model

    # ③ 分类型状态账打印（本册新账——三刀机首算）
    st = typed_state_account(cfg)
    print(f"[分类型状态账] MLA {len(st['mla_layer_indices_config0'])} 层：{st['mla_per_token_elems']} 元素/token"
          f"（无界，随 n 线性）| GDN {len(st['gdn_layer_indices_config0'])} 层：固定 "
          f"{st['gdn_fixed_total_elems']:,}（每层 {st['gdn_fixed_per_layer_elems']:,} = 递归 131,072 + "
          f"卷积滞留 15,360，与 n 无关）| 交叉 n* = {st['crossover_n']}")
    for n, row in st["totals_by_n"].items():
        print(f"  n={n:>6}: MLA 无界账 {row['mla_unbounded_elems']:>10,} + GDN 固定账 "
              f"{row['gdn_fixed_elems']:>10,} = {row['total_elems']:>11,} 元素"
              f"（fp16 字节 {row['total_elems'] * 2 / 1024 / 1024:,.1f} MiB）")
    report["arms"]["hybrid409"]["typed_state_account"] = st

    # ④ 初始 loss 自检（随机 token，CPU fp32）：预言 ln V + d·σ²/2（untied 无自泄漏；承 Book3/4 判据）
    torch.manual_seed(SEED)
    model = ThreeKnifeLM(cfg).eval()
    g = torch.Generator().manual_seed(SEED)
    x = torch.randint(0, cfg.vocab_size, (2, 256), generator=g)                # (2,256)
    xp, yp = x[:, :-1], x[:, 1:]                                               # next-token 目标
    with torch.no_grad():
        logits, loss = model(xp, yp)                                           # (2,255,V) / 标量
        z_t = logits.gather(-1, yp.unsqueeze(-1)).squeeze(-1)                  # (2,255) 目标位 logit
    ln_v, sig2 = math.log(cfg.vocab_size), cfg.hidden_size * cfg.initializer_range ** 2
    print(f"[初始 loss] 实测 {loss.item():.4f} | 预言 ln V + d·σ²/2 = {ln_v + sig2 / 2:.4f}"
          f"（ln V = {ln_v:.4f}，σ² = {sig2:.4f}，实测 logit 方差 {logits.var().item():.4f}，"
          f"E[z_t] = {z_t.mean().item():+.4f}）")
    assert abs(loss.item() - (ln_v + sig2 / 2)) < 0.06, "初始 loss 偏离 ln V + d·σ²/2"
    assert math.isfinite(loss.item())
    report["init_loss"] = {"measured": loss.item(), "predicted": ln_v + sig2 / 2,
                           "logit_var": logits.var().item(), "zt_mean": z_t.mean().item()}
    del model

    # ⑤ mini 冒烟臂（层循环/账本函数的独立复核：改几何后手算必须仍与枚举逐位一致）
    mini = make_hybrid_cfg(vocab_size=512, hidden_size=256, num_hidden_layers=4,
                           num_attention_heads=4, num_key_value_heads=1, head_dim=64,
                           intermediate_size=704, n_expert=4, top_k=2, n_shared=1, w_expert=128,
                           mla=dict(kv_lora_rank=64, q_lora_rank=64, qk_nope_head_dim=32,
                                    qk_rope_head_dim=16, v_head_dim=32),
                           layer_types=["gdn", "mla", "gdn", "mla"],
                           gdn=dict(h_k=2, h_v=2, d_k=32, d_v=32))
    m = ThreeKnifeLM(mini)
    mh = hand_account(mini)
    assert m.n_params() == mh["total"], (m.n_params(), mh["total"])
    ms = typed_state_account(mini)
    assert ms["mla_per_token_elems"] == 2 * (64 + 16)                          # 2 层 MLA × (d_c 64 + d_h^R 16)
    print(f"[mini 复核] d=256/L=4/[gdn,mla,gdn,mla]：总参 {m.n_params():,}（枚举=手算逐位）| "
          f"MLA/token {ms['mla_per_token_elems']} + GDN 固定 {ms['gdn_fixed_total_elems']:,}")
    report["mini_check"] = {"params": m.n_params(), "layer_types": mini.layer_types}
    del m

    report["parity_note"] = ("无整机正身可 strict 直搬（大纲口径）：GDN 件层级对拍承 ch05 gdn.py"
                             "（vs HF torch 双 kernel 1e-8 级）+ 线性层机构文献互证（Qwen3-Next/Kimi Linear）；"
                             "MLA/MoE 件承 Book4 对拍资产（DeepseekV2 strict 直搬）。")
    report["wall_sec"] = round(time.perf_counter() - t0, 1)
    report["verdict"] = "PASS"
    path = os.path.join(OUT_DIR, f"hybrid409_selftest_{out_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[判定] PASS | 墙钟 {report['wall_sec']} s | 产物 {path}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch9 三刀整机 hybrid409：双口径 assert + 分类型状态账 + 初始 loss 自检")
    ap.add_argument("--out-name", default="run1", help="产物 JSON 后缀（防覆写）")
    args = ap.parse_args()
    self_test(args.out_name)
