# llama409.py —— Book4 ch9 大项目整合件：两刀整机（Book3 207M 骨架 FFN→MoE + GQA→MLA）
# 用途：单轨主线本册段收官——「两刀改装史写在 import 语句里」，四个插槽的供货方即改装史：
#       归一化插槽（不动件） import 自 Book3 ch02/rmsnorm.py::RMSNorm(hidden_size, eps)
#       稠密前馈件（首层稠密机构） import 自 Book3 ch03/swiglu.py::SwiGLUMLP(cfg)
#       位置插槽（不动件）   import 自 Book3 ch04/rope.py::build_rope_cache(seq_len, head_dim, theta, device)
#       注意力插槽（第二刀） import 自本册 ch06/mla.py::MultiHeadLatentAttention(cfg).forward(x, cos, sin)
#                            ——GQA 四投影退役进虚影，换 DSV2 键位五投影（q_a/q_a_ln/q_b/kv_a/kv_a_ln/kv_b/o）
#       前馈插槽（第一刀）   import 自本册 ch05/deepseek_moe.py::DeepSeekMoE(cfg)
#                            ——dense SwiGLU 退役，E=8/top-2/共享 1、softmax+greedy+rsf 1.0（DSV2-Lite 式）
#       config 双臂：cand2 主臂（等激活·轻 MLA，总参 409,115,648）与 cand3 等参对照臂（207,064,064，
#       与 Book3 207M 差 55,296=0.03%）；cand1（重 MLA 512 档）只留参数账教学点（+3,735,552/层，见 ch6）。
#       首层稠密（first_k_dense_replace，DSV2 同款机构）由层循环处理；cand2/cand3 定版取 0（12/12 层 MoE，
#       与探针 F 锁定参数账逐位一致——置 1 的反事实账与出入登记见 DESIGN-409M.md §3）。
# 所属章节：Book4 第 9 章（9.4 两刀机 / 9.5 整合验收）；DESIGN-409M.md 的架构正身。
# 运行方式（三档；自测与对拍 CPU 可跑）：
#   自测（秒级）           ：source env.sh && python llama409.py            —— 双臂参数账 assert + 初始 loss 自检
#   HF 结构对拍（分钟级）   ：python llama409.py --parity                   —— mini 臂（探针 G 路径复跑）+ cand2 全机
#   V2-Lite 真权重条件性    ：python llama409.py --v2lite                   —— 落地则 safe_open 选择性张量，否则降级声明
# 产物：log/book4-ch09/{llama409_selftest, llama409_parity, llama409_v2lite}_{out}.json（不入库）
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
BOOK4 = os.path.abspath(os.path.join(HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(BOOK4, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book4-ch09")
SEED = 20261002                                            # 全书统一种子

# ---------------- 跨册双候选 bootstrap：Book3/Book4 兄弟目录的物理落点，存在即插入 ----------------
_B3_CANDS = [
    os.path.join(REPO_ROOT, "code", "Book3-现代开源骨架"),                   # 工作区布局（REPO_ROOT=model_analysis）
    os.path.join(REPO_ROOT, "from-models-to-vllm", "book3-modern-open-source-skeleton", "code"),
    os.path.join(REPO_ROOT, "book3-modern-open-source-skeleton", "code"),    # 书仓布局（REPO_ROOT=书仓根）
]


def bootstrap():
    """插入五件 import 的物理落点（Book3 双候选 + 本册兄弟章目录），返回 (rmsnorm 模块, swiglu, rope)。

    Book3 兄弟章（ch02/ch03/ch04）按「工作区优先」双候选探测——书仓布局（定稿快照）亦可运行；
    本册 ch05/ch06 教学件各自在其文件头内自行挂好 00-feasibility/ch02 依赖，这里只补本目录挂点。
    """
    for root in _B3_CANDS:
        if os.path.isdir(os.path.join(root, "ch02")):
            for sub in ("ch02", "ch03", "ch04", "00-feasibility"):
                p = os.path.join(root, sub)
                if os.path.isdir(p) and p not in sys.path:
                    sys.path.insert(0, p)
            break
    else:
        raise FileNotFoundError(f"未找到 Book3 兄弟目录：{_B3_CANDS}")
    for sub in ("ch02", "ch05", "ch06", "00-feasibility"):            # 本册兄弟章（对拍与组件复用）
        p = os.path.join(BOOK4, sub)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    from rmsnorm import RMSNorm                                        # noqa: F401  Book3 第 2 章（不动件）
    import swiglu                                                      # noqa: F401  Book3 第 3 章（首层稠密件）
    import rope                                                        # noqa: F401  Book3 第 4 章（位置插槽）
    return RMSNorm, swiglu.SwiGLUMLP, rope


RMSNorm, SwiGLUMLP, rope_mod = bootstrap()
from deepseek_moe import DeepSeekMoE              # noqa: E402  本册第 5 章：DSV2 键位 MoE（第一刀）
from mla import MultiHeadLatentAttention           # noqa: E402  本册第 6 章：MLA 双路径（第二刀）

# ---------------- 两刀 config 定版（探针 F 锁定；逐字段依据见 DESIGN-409M.md §3） ----------------
# 共用底座=Book3 207M 骨架：d=1024 / L=12 / h=16 / V=32000 untied / theta=10000 / eps=1e-5 / 无 bias / init 0.02
# MoE 刀：E=8 路由专家 / top-2 / 共享 1 / softmax+greedy / rsf=1.0（DSV2-Lite 式路由）
#   等激活宽 w = floor(d_ff/(k+n_shared)) = floor(2816/3) = 938（每层激活 FFN 宽 3*938=2814 ≈ dense 2816）
#   等总参宽 w=329（扫描解：总参最逼近 207,119,360 的取值）
# MLA 刀两档几何：轻档 d_c=256 / d_c'=256 / d_h=64 / d_h^R=64 / d_v=64（cand2/cand3 共用）；
#   重档 d_c=512 / d_c'=512 / d_h=128 / d_h^R=64 / d_v=128（cand1 教学点，DSV2 旗舰比例缩小）
MLA_LITE = dict(kv_lora_rank=256, q_lora_rank=256, qk_nope_head_dim=64,
                qk_rope_head_dim=64, v_head_dim=64)
MLA_HEAVY = dict(kv_lora_rank=512, q_lora_rank=512, qk_nope_head_dim=128,
                 qk_rope_head_dim=64, v_head_dim=128)

CANDS = {
    "cand2": dict(w_expert=938, mla=MLA_LITE,  expect_total=409_115_648, note="等激活·轻 MLA（主臂，409M 两刀机）"),
    "cand3": dict(w_expert=329, mla=MLA_LITE,  expect_total=207_064_064, note="等总参·轻 MLA（对照臂，207M 等参两刀机）"),
    "cand1": dict(w_expert=938, mla=MLA_HEAVY, expect_total=459_453_440, note="等激活·重 MLA（参数账教学点，不设训练档）"),
}
BASE_TOTAL_207M = 207_119_360       # Book3 定版（对照锚）


class TwoKnifeConfig:
    """两刀整机 config。字段与 HF DeepseekV2Config 同名族（d=L 对齐 num_*；MoE/MLA 字段见 CANDS）。

    first_k_dense_replace：前 k 层保持稠密 SwiGLU（DSV2 同款机构，由层循环处理）；
    cand2/cand3 定版=0（12/12 层 MoE，探针 F 锁定账）；对拍 mini 臂取 1 演练混排。
    """

    def __init__(self, vocab_size=32000, hidden_size=1024, num_hidden_layers=12,
                 num_attention_heads=16, num_key_value_heads=8, head_dim=64,
                 intermediate_size=2816, rope_theta=10000.0, rms_norm_eps=1e-5,
                 initializer_range=0.02, n_expert=8, top_k=2, n_shared=1,
                 w_expert=938, mla=None, first_k_dense_replace=0,
                 scoring="softmax", routed_scaling_factor=1.0,
                 max_position_embeddings=8192):
        self.vocab_size, self.hidden_size = vocab_size, hidden_size
        self.num_hidden_layers, self.num_attention_heads = num_hidden_layers, num_attention_heads
        self.num_key_value_heads, self.head_dim = num_key_value_heads, head_dim
        self.intermediate_size, self.rope_theta = intermediate_size, rope_theta
        self.rms_norm_eps, self.initializer_range = rms_norm_eps, initializer_range
        self.n_expert, self.top_k, self.n_shared = n_expert, top_k, n_shared
        self.w_expert, self.mla = w_expert, (mla or MLA_LITE)
        self.first_k_dense_replace = first_k_dense_replace
        self.scoring, self.routed_scaling_factor = scoring, routed_scaling_factor
        self.max_position_embeddings = max_position_embeddings


def make_cfg(arm="cand2", **over):
    """按 CANDS 定版构造 config（over 可覆盖任意字段——对拍 mini 臂/首层稠密变体用）。"""
    spec = CANDS[arm]
    kw = dict(w_expert=spec["w_expert"], mla=dict(spec["mla"]))
    kw.update(over)
    return TwoKnifeConfig(**kw)


class TwoKnifeDecoderLayer(nn.Module):
    """残差河恒宽的两刀层——拓扑仍是 Book2/3 三插槽 Block，两个插槽里换的是本册新件：

        x = x + attn(input_layernorm(x), cos, sin)          RMSNorm(Book3 ch2) -> MLA(本册 ch6)
        x = x + mlp(post_attention_layernorm(x))            RMSNorm(Book3 ch2) -> MoE(本册 ch5) | 稠密(首层)
    输入/输出 (B,n,C)。键名与 HF DeepseekV2 DecoderLayer 严格同名（self_attn/mlp/两枚 layernorm）。
    cos/sin 为插槽契约参数：MLA 的解耦 RoPE 通道（d_h^R=64 维）自带频率阶梯（ch6 契约——
    全宽表无法切片复用），本参数在此被忽略；层循环仍照发，保持与 Book3 GQA 插槽统一调用。
    """

    def __init__(self, cfg: TwoKnifeConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)       # 插槽①（Book3 第 2 章）
        self.self_attn = MultiHeadLatentAttention(                              # 插槽④（本册第 6 章）
            dict(hidden_size=cfg.hidden_size, num_attention_heads=cfg.num_attention_heads,
                 rope_theta=cfg.rope_theta, rope_style="complex", **cfg.mla))
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if layer_idx < cfg.first_k_dense_replace:                               # 首层稠密由层循环处理
            dense_cfg = type("Dense_cfg", (), {"hidden_size": cfg.hidden_size,
                                               "intermediate_size": cfg.intermediate_size})()
            self.mlp = SwiGLUMLP(dense_cfg)                                     # (B,n,C)->(B,n,C) 稠密件
        else:
            self.mlp = DeepSeekMoE(hidden_size=cfg.hidden_size, n_expert=cfg.n_expert,
                                   top_k=cfg.top_k, expert_dim=cfg.w_expert,
                                   n_shared=cfg.n_shared, scoring=cfg.scoring,
                                   n_group=1, topk_group=1, norm_topk_prob=False,
                                   routed_scaling_factor=cfg.routed_scaling_factor)  # DSV2 键位
        self.layer_idx = layer_idx

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)   # (B,n,C) 残差河
        x = x + self.mlp(self.post_attention_layernorm(x))          # (B,n,C) MoE/稠密件同接口
        return x


class TwoKnifeBackbone(nn.Module):
    """embed_tokens + L 层 TwoKnifeDecoderLayer + 终态 RMSNorm（命名对齐 HF DeepseekV2Model）。"""

    def __init__(self, cfg: TwoKnifeConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)       # (V,C)
        self.layers = nn.ModuleList([TwoKnifeDecoderLayer(cfg, i)
                                     for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)                  # 出口前最后一枚

    def forward(self, idx):
        B, n = idx.shape                                                       # (B,n)
        cfg = self.cfg
        x = self.embed_tokens(idx)                                             # (B,n) -> (B,n,C)
        cos, sin = rope_mod.build_rope_cache(n, cfg.head_dim, cfg.rope_theta,   # (1,n,64) x2（Book3 第 4 章；
                                            idx.device)                         #  MLA 自建阶梯，此表仅供契约）
        for layer in self.layers:
            x = layer(x, cos, sin)                                             # L x (B,n,C)
        return self.norm(x)                                                    # (B,n,C)


class TwoKnifeLM(nn.Module):
    """两刀整机：wte + L x [pre-RMSNorm -> MLA | pre-RMSNorm -> MoE] + 终态 RMSNorm + untied lm_head。

    前向接口与 Book2/3 全线一致：idx (B,n) -> logits (B,n,V)；给 targets 返回逐 token 交叉熵（fp32）。
    state_dict 键与 HF DeepseekV2ForCausalLM 完全同名（model.embed_tokens.weight /
    model.layers.i.self_attn.q_a_proj.weight / model.layers.i.mlp.gate.weight /
    model.layers.i.mlp.experts.gate_up_proj / model.layers.i.mlp.shared_experts.* / lm_head.weight），
    strict 直搬即结构对拍（--parity）；首层稠密层的 mlp 键退回 gate/up/down_proj（DeepseekV2MLP 同名）。
    """

    def __init__(self, cfg: TwoKnifeConfig = None, init_std: float | None = None):
        super().__init__()
        cfg = cfg or make_cfg("cand2")
        self.cfg = cfg
        self.model = TwoKnifeBackbone(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)   # untied（Book3 第 5 章口径）
        std = init_std or cfg.initializer_range
        self.apply(lambda m: self._init(m, std))

    @staticmethod
    def _init(module, std):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

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


# ---------------- 参数逐项手算账（assert 的对账另一侧；公式见 DESIGN-409M.md §3） ----------------
def hand_account(cfg: TwoKnifeConfig):
    """总参逐项手算（无 bias、untied、RMSNorm 只 weight）。返回明细 dict。

    每层：MLA 投影（q 低秩两步 + kv 下投影 + kv 上投影 + o）+ 两枚层内 RMSNorm + 两枚潜向量 RMSNorm
    （d_c 与 d_c' 各一枚）+ FFN（MoE=路由门 E·d + 路由专家 E·3dw + 共享 n_shared·3dw；稠密层=3·d·f）。
    """
    d, L, V, f = cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size, cfg.intermediate_size
    h, m = cfg.num_attention_heads, cfg.mla
    r_q = m.get("q_lora_rank")
    q_side = (d * r_q + r_q * h * (m["qk_nope_head_dim"] + m["qk_rope_head_dim"])
              if r_q is not None else d * h * (m["qk_nope_head_dim"] + m["qk_rope_head_dim"]))
    kv_down = d * (m["kv_lora_rank"] + m["qk_rope_head_dim"])                   # [W^DKV;W^KR] 融合一件
    kv_up = m["kv_lora_rank"] * h * (m["qk_nope_head_dim"] + m["v_head_dim"])   # [W^UK;W^UV] 按头行块拼接
    o_side = h * m["v_head_dim"] * d
    attn_proj = q_side + kv_down + kv_up + o_side
    latent_norms = (m["kv_lora_rank"] + (r_q or 0))                             # 潜向量 RMSNorm 两枚
    gate = cfg.n_expert * d                                                     # 路由门 W_g ∈ R^{E×d}
    experts = cfg.n_expert * 3 * d * cfg.w_expert                               # 路由专家（融合 3D 等价账）
    shared = cfg.n_shared * 3 * d * cfg.w_expert                                # 共享专家（加宽 SwiGLU）
    moe_layer = gate + experts + shared
    dense_layer = 3 * d * f
    n_moe = sum(1 for i in range(L) if i >= cfg.first_k_dense_replace)
    n_dense = L - n_moe
    layers = n_moe * (attn_proj + latent_norms + 2 * d + moe_layer) \
        + n_dense * (attn_proj + latent_norms + 2 * d + dense_layer)
    emb = V * d * 2 + d                                                         # untied 两份 + 终态 norm
    return {"attn_proj_per_layer": attn_proj, "q_side": q_side, "kv_down": kv_down,
            "kv_up": kv_up, "o": o_side, "latent_norms": latent_norms,
            "mlp_moe_per_layer": moe_layer, "mlp_dense_per_layer": dense_layer,
            "gate": gate, "experts": experts, "shared": shared,
            "n_moe_layers": n_moe, "n_dense_layers": n_dense,
            "layers_total": layers, "embedding_and_head": emb, "total": layers + emb}


def activated_account(cfg: TwoKnifeConfig):
    """单 token 激活参数口径族（ch1 口径纪律的代码版）：

    返回 dict：non_emb（不含两份嵌入）、with_lm_head（本册正文口径=含 lm_head、不含输入 embedding
    lookup；两级均含路由门）、dual（双份嵌入+路由门——正文修正版账本）、dual_probe（探针 F JSON 的
    历史口径：双份嵌入、不含路由门，对照用；与 dual 的差额=n_moe_layers×gate）。
    MoE 层激活 FFN = 路由门 E·d + top_k 份路由专家 + n_shared 份共享；稠密层全激活。
    """
    d, L, V = cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size
    acc = hand_account(cfg)
    attn_and_norms = acc["attn_proj_per_layer"] + acc["latent_norms"] + 2 * d
    body = 0
    for i in range(L):
        if i >= cfg.first_k_dense_replace:
            body += attn_and_norms + acc["gate"] \
                + 3 * d * (cfg.top_k + cfg.n_shared) * cfg.w_expert
        else:
            body += attn_and_norms + acc["mlp_dense_per_layer"]
    body += d                                                                   # 终态 norm
    probe_dual = body - acc["n_moe_layers"] * acc["gate"] + V * d * 2           # 探针历史口径：不含路由门、双份嵌入
    dual_gates = body + V * d * 2                                               # 修正版账本：双份嵌入+路由门
    return {"non_emb": body, "with_lm_head": body + V * d,
            "dual": dual_gates, "dual_probe": probe_dual}


def kv_per_token_elements(cfg: TwoKnifeConfig):
    """KV cache 每 token 元素账（单位=元素数，论文 Table 1 口径）：

    两刀 MLA = L·(d_c + d_h^R)——潜向量 c_KV 一份 + 解耦共享键 k_R 一份（K/V 上投影移到读出端不进缓存）；
    退役 GQA 基线 = 2·L·h_kv·d_k（Book3 式 (6.2) 直系，h_kv=8/d_k=64）。
    """
    mla_elems = cfg.num_hidden_layers * (cfg.mla["kv_lora_rank"] + cfg.mla["qk_rope_head_dim"])
    gqa_elems = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim
    return {"mla": mla_elems, "gqa_retired": gqa_elems, "saving": round(1 - mla_elems / gqa_elems, 4)}


# ---------------- 自测：双臂参数账 assert + 初始 loss 的数学自检 ----------------
def self_test(out_name):
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    report = {"seed": SEED, "arms": {}}
    print(f"[bootstrap] Book3 兄弟目录自：{next(p for p in _B3_CANDS if os.path.isdir(os.path.join(p, 'ch02')))}")
    for arm in ("cand2", "cand3", "cand1"):
        cfg = make_cfg(arm)
        model = TwoKnifeLM(cfg)
        hand, total = hand_account(cfg), model.n_params()
        exp = CANDS[arm]["expect_total"]
        assert total == hand["total"] == exp, f"{arm}: {total} != {hand['total']} != {exp}"
        act = activated_account(cfg)
        row = {"note": CANDS[arm]["note"], "total": total,
               "active_non_emb": act["non_emb"], "active_with_lm_head": act["with_lm_head"],
               "active_dual": act["dual"], "active_dual_probe": act["dual_probe"],
               "kv": kv_per_token_elements(cfg)}
        print(f"[{arm}] {row['note']}：总参 {total:,}（手算=实测=探针 F 锁定数，逐位）| "
              f"激活 non-emb {act['non_emb']:,} / 含 lm_head {act['with_lm_head']:,} / "
              f"双份(含门) {act['dual']:,} / 双份(探针口径,不含门) {act['dual_probe']:,} | "
              f"KV/token {row['kv']['mla']:,} 元素 vs 退役 GQA "
              f"{row['kv']['gqa_retired']:,}（−{row['kv']['saving'] * 100:.2f}%）")
        report["arms"][arm] = row
        del model

    # cand3 与 Book3 207M 的等参对账（同预算两刀 vs 稠密的对照臂资格）
    gap = CANDS["cand3"]["expect_total"] - BASE_TOTAL_207M
    print(f"[等参对账] cand3 {CANDS['cand3']['expect_total']:,} vs Book3 207M {BASE_TOTAL_207M:,}："
          f"差 {gap:,} = {abs(gap) / BASE_TOTAL_207M * 100:.3f}%")
    report["cand3_vs_207m_gap"] = gap

    # 初始 loss 自检（cand2，随机 token，CPU fp32）：预言 ln V + d*s^2/2 = 10.578（untied 无自泄漏）
    cfg = make_cfg("cand2")
    model = TwoKnifeLM(cfg).eval()
    g = torch.Generator().manual_seed(SEED)
    x = torch.randint(0, cfg.vocab_size, (2, 256), generator=g)                # (2,256)
    xp, yp = x[:, :-1], x[:, 1:]                                               # next-token 目标
    with torch.no_grad():
        logits, loss = model(xp, yp)                                           # (2,255,V) / 标量
        z_t = logits.gather(-1, yp.unsqueeze(-1)).squeeze(-1)                  # (2,255) 目标位 logit
    ln_v, sig2 = math.log(cfg.vocab_size), cfg.hidden_size * cfg.initializer_range ** 2
    print(f"[初始 loss] 实测 {loss.item():.4f} | 预言 ln V + d*s^2/2 = {ln_v + sig2 / 2:.4f}"
          f"（ln V = {ln_v:.4f}，sigma^2 = {sig2:.4f}，实测 logit 方差 {logits.var().item():.4f}，"
          f"E[z_t] = {z_t.mean().item():+.4f}）")
    assert abs(loss.item() - (ln_v + sig2 / 2)) < 0.06, "初始 loss 偏离 ln V + d*s^2/2"
    assert math.isfinite(loss.item())
    report["init_loss"] = {"measured": loss.item(), "predicted": ln_v + sig2 / 2}

    # 首层稠密机构演练（mini 几何 first_k_dense=1：层循环切换稠密/MoE，账本跟着分册）
    mini = TwoKnifeConfig(vocab_size=512, hidden_size=256, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=1, head_dim=64,
                          intermediate_size=704, n_expert=4, top_k=2, n_shared=1,
                          w_expert=128, mla=dict(kv_lora_rank=64, q_lora_rank=64,
                                                 qk_nope_head_dim=32, qk_rope_head_dim=16,
                                                 v_head_dim=32), first_k_dense_replace=1)
    m = TwoKnifeLM(mini)
    hand = hand_account(mini)
    assert m.n_params() == hand["total"] == 1_493_504, (m.n_params(), hand["total"])
    assert hand["n_dense_layers"] == 1 and hand["n_moe_layers"] == 1
    assert isinstance(m.model.layers[0].mlp, SwiGLUMLP) and not isinstance(m.model.layers[1].mlp, SwiGLUMLP)
    print(f"[首层稠密机构] mini（L=2/first_k_dense=1）：层 0 稠密 {hand['mlp_dense_per_layer']:,} + "
          f"层 1 MoE {hand['mlp_moe_per_layer']:,}，总参 {m.n_params():,} 逐位（层循环分册记账）")
    report["first_k_dense_mini"] = {"params": m.n_params(), "n_dense": 1, "n_moe": 1}

    report["verdict"] = "PASS"
    with open(os.path.join(OUT_DIR, f"llama409_selftest_{out_name}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[判定] PASS | 产物 {OUT_DIR}/llama409_selftest_{out_name}.json")
    return report


# ---------------- HF 结构对拍：DeepseekV2ForCausalLM 同 config 随机权重 strict 直搬 ----------------
def _hf_kwargs(cfg: TwoKnifeConfig):
    """TwoKnifeConfig -> DeepseekV2Config kwargs（字段同名族一一对应）。"""
    return dict(vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
                intermediate_size=cfg.intermediate_size,
                moe_intermediate_size=cfg.w_expert, num_hidden_layers=cfg.num_hidden_layers,
                num_attention_heads=cfg.num_attention_heads,
                num_key_value_heads=cfg.num_key_value_heads,
                first_k_dense_replace=cfg.first_k_dense_replace,
                n_routed_experts=cfg.n_expert, n_shared_experts=cfg.n_shared,
                num_experts_per_tok=cfg.top_k,
                kv_lora_rank=cfg.mla["kv_lora_rank"], q_lora_rank=cfg.mla["q_lora_rank"],
                qk_nope_head_dim=cfg.mla["qk_nope_head_dim"],
                qk_rope_head_dim=cfg.mla["qk_rope_head_dim"],
                v_head_dim=cfg.mla["v_head_dim"], topk_method="greedy",
                scoring_func="softmax", routed_scaling_factor=cfg.routed_scaling_factor,
                norm_topk_prob=False, rope_theta=cfg.rope_theta,
                rms_norm_eps=cfg.rms_norm_eps, tie_word_embeddings=False,
                max_position_embeddings=cfg.max_position_embeddings)


def parity_one(tag, cfg: TwoKnifeConfig, x_len=128):
    """一臂对拍：本类 vs HF DeepseekV2ForCausalLM——strict 直搬 + CPU fp32 logits 对拍。

    实现注记（5.18.0 坑，本机实测）：HF 的融合专家默认按 config._experts_implementation 分发到
    grouped_mm 内核，后者要求权重步长 16 字节对齐（fp32 下即各维为 4 的倍数）——官方旗舰档
    （1536/2048/1408）天然满足，教学档奇宽（cand2 的 938、cand3 的 329）会在前向崩
    「strides should be multiple of 16 bytes」（strict 搬运本身已通过——键位/形状证书先拿到）。
    对拍切回 "eager"（逐专家循环，与本册教学件同构、语义等价）。
    """
    from transformers import DeepseekV2Config, DeepseekV2ForCausalLM
    torch.manual_seed(SEED)
    ours = TwoKnifeLM(cfg).eval()
    hf = DeepseekV2ForCausalLM(DeepseekV2Config(**_hf_kwargs(cfg))).eval()
    hf.config._experts_implementation = "eager"          # grouped_mm 的 16 字节对齐约束——见 docstring
    hf.load_state_dict(ours.state_dict(), strict=True, assign=True)            # 键不一致直接抛错——本身就是验收
    g = torch.Generator().manual_seed(SEED)
    x = torch.randint(0, cfg.vocab_size, (2, x_len), generator=g)              # (2,128)
    with torch.no_grad():
        lo, _ = ours(x, x)                                                     # (2,128,V)
        lh = hf(x).logits                                                      # (2,128,V)
    delta = (lo - lh).abs().max().item()
    agree = (lo.argmax(-1) == lh.argmax(-1)).float().mean().item()
    top5 = (lo.topk(5, -1).indices.sort(-1).values
            == lh.topk(5, -1).indices.sort(-1).values).float().mean().item()
    torch.manual_seed(SEED + 1)                                                # sanity：未搬运对照应显著不同
    other = TwoKnifeLM(cfg).eval()
    with torch.no_grad():
        sanity = (other(x)[0] - lh).abs().max().item()
    ok = delta < 1e-4 and agree == 1.0 and sanity > 0.1
    print(f"[{tag}] strict 直搬通过 {ours.n_params():,} 参 | max|Δlogits| = {delta:.3e} | "
          f"argmax 一致率 = {agree:.4f} | top-5 一致率 = {top5:.4f} | sanity(未搬运) = {sanity:.3f}"
          f" | {'PASS' if ok else 'FAIL'}")
    return {"params": ours.n_params(), "max_abs_diff_logit": delta, "argmax_agreement": agree,
            "top5_agreement": top5, "sanity_unshared": sanity, "verdict": "PASS" if ok else "FAIL"}


def parity_hf(out_name):
    t0 = time.perf_counter()
    print(f"[对拍] transformers {__import__('transformers').__version__} | CPU fp32 | 种子 {SEED}")
    # 臂 A mini（探针 G 路径复跑：d=256/L=2/first_k_dense=1/E4/top2/共享1/MLA 64 档）
    mini_cfg = TwoKnifeConfig(vocab_size=512, hidden_size=256, num_hidden_layers=2,
                              num_attention_heads=4, num_key_value_heads=1, head_dim=64,
                              intermediate_size=704, max_position_embeddings=512,
                              n_expert=4, top_k=2, n_shared=1, w_expert=128,
                              mla=dict(kv_lora_rank=64, q_lora_rank=64, qk_nope_head_dim=32,
                                       qk_rope_head_dim=16, v_head_dim=32),
                              first_k_dense_replace=1)
    arm_a = parity_one("A mini（探针 G 路径）", mini_cfg)
    # 臂 B cand2 全机（409M 两刀机 vs HF 同 config——本机最大的 strict 直搬）
    arm_b = parity_one("B cand2 全机 409M", make_cfg("cand2"))
    report = {"seed": SEED, "transformers": __import__("transformers").__version__,
              "device": "cpu fp32", "mini": arm_a, "cand2_full": arm_b,
              "wall_sec": round(time.perf_counter() - t0, 1),
              "verdict": "PASS" if arm_a["verdict"] == "PASS" and arm_b["verdict"] == "PASS" else "FAIL"}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"llama409_parity_{out_name}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[判定] {report['verdict']} | 墙钟 {report['wall_sec']} s | 产物 "
          f"{OUT_DIR}/llama409_parity_{out_name}.json")
    return report


# ---------------- V2-Lite 真权重条件性核验（落地则 safe_open 选择性张量；否则降级声明） ----------------
LITE_SNAP = os.path.join(REPO_ROOT, "log", "huggingface", "hub",
                         "models--deepseek-ai--DeepSeek-V2-Lite", "snapshots")


def check_v2lite(out_name):
    """两刀机的「大表哥」实物核验（条件性，失败降级不阻塞）：

    DeepSeek-V2-Lite 与 cand2 同族（MLA+细粒度 MoE+共享专家+softmax greedy），且 q_lora_rank=null
    是 cand2 未覆盖的分支。权重完整落地（≈31.4 GiB）则 safe_open 惰性读逐层形状账本（零数值加载，
    峰值单张量数百 MB）：kv_a/kv_b/q_proj/o_proj 与 shared_experts 宽度（n_shared·w 的活证据）。
    """
    import glob
    res = {"checked": False, "degraded": True}
    snaps = sorted(glob.glob(os.path.join(LITE_SNAP, "*", "*")))
    st = [p for p in snaps if p.endswith(".safetensors") and os.path.exists(os.path.realpath(p))]
    total = sum(os.path.getsize(p) for p in st)
    res["safetensors_files"], res["safetensors_bytes"] = len(st), total
    if not st or total < 30 * 2 ** 30:
        res["note"] = (f"权重未完整落地（{len(st)} 分片 / {total / 2 ** 30:.1f} GiB，预期 ≈31.4 GiB）——"
                       "按预登记降级：随机权重结构对拍（--parity 两臂）已足够支撑正文；"
                       "权重落地后重跑 --v2lite 补逐层形状账本")
        print(f"[V2-Lite] 降级记录：{res['note']}")
    else:
        from safetensors import safe_open
        h, d_c, d_n, d_r, d_v, d = 16, 512, 128, 64, 128, 2048
        expect = {"kv_a_proj_with_mqa": (d_c + d_r, d), "kv_b_proj": (h * (d_n + d_v), d_c),
                  "o_proj": (d, h * d_v), "q_proj": (h * (d_n + d_r), d),         # Lite 无 q 压缩
                  "shared_experts.gate_proj": (2 * 1408, d)}                       # 共享 2 × w 1408 的活证据
        shapes, n = {}, 0
        for path in st:
            with safe_open(path, framework="pt") as f:
                for key in f.keys():
                    for name, want in expect.items():
                        if key.endswith(f".{name}.weight") and ".0." not in key:
                            got = tuple(f.get_slice(key).get_shape())
                            assert got == want, (key, got, want)
                            shapes[name] = got
                            n += 1
        res.update({"checked": True, "degraded": False, "shapes": shapes, "n_tensors": n})
        print(f"[V2-Lite] {len(st)} 分片完整：逐层形状账本 {n} 项全过——kv_b_proj {shapes['kv_b_proj']} "
              f"（W^UK/W^UV 折叠）、shared_experts.gate_proj {shapes['shared_experts.gate_proj']} "
              f"（共享 2×1408 的加宽实物）、q_proj {shapes['q_proj']}（Lite 不压 q）")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"llama409_v2lite_{out_name}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch9 两刀整机 llama409：自测 / HF 结构对拍 / V2-Lite 条件性核验")
    ap.add_argument("--parity", action="store_true", help="HF 结构对拍（CPU fp32，分钟级）")
    ap.add_argument("--v2lite", action="store_true", help="V2-Lite 真权重条件性核验（落地则 safe_open 选择性张量）")
    ap.add_argument("--out-name", default="run1", help="产物 JSON 后缀（防覆写）")
    args = ap.parse_args()
    if args.parity:
        parity_hf(args.out_name)
    if args.v2lite:
        check_v2lite(args.out_name)
    if not (args.parity or args.v2lite):
        self_test(args.out_name)
