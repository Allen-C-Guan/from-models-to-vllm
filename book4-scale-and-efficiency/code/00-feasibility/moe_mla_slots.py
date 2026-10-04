# moe_mla_slots.py —— Book4 插槽组件库：第一刀 MoE / 第二刀 MLA + 可插拔 SlotLLaMA 底座
# 用途：为 00-feasibility 各探针（b/d 模块训练、c MLA 对拍、f 两刀整机、g HF 对拍）提供统一组件：
#       ① bootstrap_book3()：跨册双候选引入 Book3 的 llama_slots（工作区/书仓两布局自适应——Book3 终校先例）；
#       ② SparseMoE（教学版 MoE：top-k 路由 + 专家 ModuleList + 可选共享专家）与 HF 键位对齐版
#          HFMixtralMoE / HFDeepseekMoE（5.18.0 起专家权重为融合 3D 张量 experts.gate_up_proj/down_proj，
#          不再是 4.x 的 experts.N.* ModuleList——strict 直搬必须按新键位）；
#       ③ MLAAttention（手写 MLA：低秩 KV 压缩 + 解耦 RoPE + q 侧低秩分解；键位对齐 HF
#          DeepseekV2Attention：q_a_proj/q_a_layernorm/q_b_proj/kv_a_proj_with_mqa/kv_a_layernorm/
#          kv_b_proj/o_proj——论文记号 W_DKV/W_UK/W_UV/W_KR 与 HF 键的映射表见类 docstring）；
#       ④ SlotConfig/SlotLLaMA：残差河恒宽的插槽式整机——注意力插槽（GQA|MLA）与 FFN 插槽
#          （dense|MoE，支持 DSV2 式 first_k_dense_replace 前几层保持 dense）任意组合，
#          这正是本册两刀（FFN→MoE、GQA→MLA）的代码形态；
#       ⑤ hand_account/activated_account/kv_per_token：总参/激活参/KV 三口径手算账（与实测 assert 对账）。
# 所属章节：Book4 ch2-6（MoE/MLA 机制正身的 mini 版）与 ch9（两刀整合底座）；00-feasibility 全体探针共用。
# 运行方式：作为模块被探针 import；自测（秒级，CPU）：source env.sh && python moe_mla_slots.py
#   —— 自测内容：三口径参数手算账 vs 实测逐位 assert + MLA 显式/吸收两路等价快检 + MoE 路由 shape 检查。
# 数值口径：softmax/方差一律 fp32（对齐 HF）；MLA 前向为「显式上投影」路径（与 HF DeepseekV2 eager 同构），
#   「权重吸收」路径在 03_probe_c_mla.py 中以本库权重另行推导对拍。
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # 三级上溯=工作区根/书仓根（书仓候选 book3-modern-open-source-skeleton 的定位锚，组装归一化按跨册 bootstrap 行豁免本行）
OUT_DIR = os.path.join(REPO_ROOT, "log", "book4-feasibility")
SEED = 20261002                                            # 全书统一种子

# 跨册语料（Book3 遗产，见 plan/Book4-开工指南 §1）：uint16 token 流 + sp 模型
TOKENS_SP8K = os.path.join(REPO_ROOT, "log", "book2-ch08", "tokens.bin")       # V=8192, 306M tok
EVAL_SP8K = os.path.join(REPO_ROOT, "log", "book2-ch08", "eval.bin")
TOKENS_SP32K = os.path.join(REPO_ROOT, "log", "book3-ch10", "tokens32k.bin")   # V=32000, 56.3M tok
EVAL_SP32K = os.path.join(REPO_ROOT, "log", "book3-ch10", "eval32k.bin")


# ---------------- 跨册双候选 bootstrap：llama_slots 的两种物理落点，存在即插入 ----------------
_SLOTS_CANDIDATES = [
    os.path.abspath(os.path.join(REPO_ROOT, "code", "Book3-现代开源骨架", "00-feasibility")),
    os.path.abspath(os.path.join(REPO_ROOT, "from-models-to-vllm", "book3-modern-open-source-skeleton",
                                 "code", "00-feasibility")),
    os.path.abspath(os.path.join(REPO_ROOT, "book3-modern-open-source-skeleton", "code", "00-feasibility")),
]
_slots_cache = {}


def bootstrap_book3():
    """双候选引入 Book3 llama_slots：工作区 code/ 与书仓 from-models-to-vllm/ 两布局自适应。

    候选按「工作区优先」逐个探测，命中第一个即插入 sys.path（Book3 终校「存在即插入」先例），
    随后 import llama_slots 并缓存。返回 llama_slots 模块（组件正身：
    LLaMAConfig/RMSNorm/SwiGLU/build_rope_cache/apply_rope/GroupedQueryAttention）。
    """
    if "slots" in _slots_cache:
        return _slots_cache["slots"]
    found = [p for p in _SLOTS_CANDIDATES if os.path.exists(os.path.join(p, "llama_slots.py"))]
    if not found:
        raise FileNotFoundError(f"两布局均未找到 llama_slots.py：{_SLOTS_CANDIDATES}")
    sys.path.insert(0, found[0])                 # 只插入第一个命中（工作区优先）——后插会把先插的挤到后面
    slots = importlib.import_module("llama_slots")
    _slots_cache.update(slots=slots, paths=found)
    return slots


def slots_paths():
    bootstrap_book3()
    return list(_slots_cache["paths"])


# ---------------- 训练数据流：uint16 token 流的确定性顺序批（Book3 train_215 同口径） ----------------
class TokenStream:
    """np.memmap 顺序批加载器。batch(step) 返回 (x,y)，形状 (B,T)/(B,T) 的 int64——next-token 目标。

    第 step 批取自偏移 step*B*T 的连续段（无重叠、无洗牌——可行性探针的确定性口径）。
    """

    def __init__(self, path, batch, block, holdout_first=0):
        self.arr = np.memmap(path, dtype=np.uint16, mode="r")
        self.B, self.T = batch, block
        self.holdout = holdout_first                  # 前若干 token 留作观察（探针不用于 eval，默认 0）
        self.max_step = (len(self.arr) - 1 - self.holdout) // (batch * block) - 1

    def batch(self, step, device):
        base = self.holdout + step * self.B * self.T
        span = np.asarray(self.arr[base: base + self.B * self.T + 1]).astype(np.int64)
        x = torch.from_numpy(span[:-1].reshape(self.B, self.T)).to(device)   # (B,T)
        y = torch.from_numpy(span[1:].reshape(self.B, self.T)).to(device)    # (B,T)
        return x, y


# ---------------- 第一刀：MoE 插件 ----------------
class TopkGate(nn.Module):
    """top-k 路由器（无 bias 线性 + softmax + top-k）。两种模式，前向语义分别对齐 HF 5.18.0：

    - mode="mixtral"：logits 用原 dtype 线性 → softmax(fp32) → top-k → 权重归一化（和为 1）
      （Mixtral 论文口径；MixtralTopKRouter.forward 同构）。
    - mode="deepseek"：logits 强制 fp32 线性 → softmax(fp32) → top-k → 权重 × routed_scaling_factor，
      不做归一化（DeepseekV2TopkRouter 的 greedy 路径同构——5.18.0 无 norm_topk_prob 分支，见探针 g 记录）。
    返回 (router_logits, topk_weights, topk_indices)，形状 (N,E)/(N,k)/(N,k)，N=B*n 为展平 token 数。
    """

    def __init__(self, hidden_size, n_expert, top_k, mode="mixtral", routed_scaling_factor=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_expert, hidden_size))       # (E,d) —— HF 键名 gate.weight
        self.n_expert, self.top_k, self.mode = n_expert, top_k, mode
        self.routed_scaling_factor = routed_scaling_factor
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x):
        h = x.reshape(-1, x.shape[-1])                                        # (B,n,d)->(N,d)
        if self.mode == "deepseek":
            logits = F.linear(h.float(), self.weight.float())                 # (N,E) fp32（HF 同构）
            scores = logits.softmax(dim=-1)
            w, idx = torch.topk(scores, self.top_k, dim=-1, sorted=False)     # (N,k)
            w = w * self.routed_scaling_factor                                # 不归一化（记录于探针 g）
        else:
            logits = F.linear(h, self.weight)                                 # (N,E)
            probs = logits.float().softmax(dim=-1)
            w, idx = torch.topk(probs, self.top_k, dim=-1)                    # (N,k)
            w = w / w.sum(dim=-1, keepdim=True)                               # 归一化（Mixtral 口径）
        return logits, w, idx


class ExpertSwiGLU(nn.Module):
    """单个专家 = 窄 SwiGLU（gate/up/down 三投影，键名对齐 HF DeepseekV2MLP/LlamaMLP）。

    共享专家与路由专家同构——「共享专家」= 每个 token 无条件激活的专家（DeepSeekMoE 口径）。
    DSV2 式共享专家宽度 = n_shared * w_e，用本类一个加宽实例表达（参数账逐位相同）。
    """

    def __init__(self, hidden_size, expert_dim):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, expert_dim, bias=False)      # (B',d)->(B',w)
        self.up_proj = nn.Linear(hidden_size, expert_dim, bias=False)
        self.down_proj = nn.Linear(expert_dim, hidden_size, bias=False)
        self.expert_dim = expert_dim

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))   # (B',d)


class SparseMoE(nn.Module):
    """教学版稀疏 MoE：路由专家 E 个（ModuleList）+ 可选共享专家 + top-k 路由。

    前向：y = Σ_{e∈top-k} g_e(x)·Expert_e(x) + Shared(x)——每 token 只激活 k(+n_shared) 个专家。
    forward 额外返回负载统计 counts (E,)（本批各专家被选中的次数——负载不均的现场素材）。
    variant="teach"：专家存 ModuleList（教学直观）；HF 对拍请用 HFMixtralMoE/HFDeepseekMoE。
    """

    def __init__(self, hidden_size, n_expert, top_k, expert_dim, n_shared=0,
                 mode="mixtral", routed_scaling_factor=1.0, variant="teach"):
        super().__init__()
        assert variant == "teach"
        self.gate = TopkGate(hidden_size, n_expert, top_k, mode, routed_scaling_factor)
        self.experts = nn.ModuleList([ExpertSwiGLU(hidden_size, expert_dim) for _ in range(n_expert)])
        self.n_shared = n_shared
        self.shared_experts = (ExpertSwiGLU(hidden_size, n_shared * expert_dim)
                               if n_shared > 0 else None)
        self.n_expert, self.top_k, self.expert_dim = n_expert, top_k, expert_dim

    def forward(self, x):
        shape = x.shape                                                       # (B,n,d)
        h = x.reshape(-1, shape[-1])                                          # (N,d)
        _, w, idx = self.gate(x)                                              # (N,k),(N,k)
        out = torch.zeros_like(h)                                             # (N,d)
        counts = torch.zeros(self.n_expert, dtype=torch.long)
        for e in range(self.n_expert):
            tok, slot = torch.nonzero(idx == e, as_tuple=True)                # 选中专家 e 的 (token, 槽位)
            if tok.numel() == 0:
                continue
            y = self.experts[e](h[tok])                                       # (n_e,d)
            out.index_add_(0, tok, (y * w[tok, slot].to(y.dtype).unsqueeze(1)).to(out.dtype))
            counts[e] = tok.numel()
        out = out.view(*shape)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)                                # 共享专家：全 token 无条件激活
        return out, counts

    def expert_selection_counts(self, x):
        """只算路由直方图不算专家输出（轻量口径，供长训练逐段采样负载分布）。"""
        _, _, idx = self.gate(x)
        return torch.bincount(idx.reshape(-1), minlength=self.n_expert)       # (E,)


class _FusedExperts(nn.Module):
    """HF 5.18.0 键位的融合专家库：experts.gate_up_proj (E,2w,d) 与 experts.down_proj (E,d,w)。

    4.x 的 experts.N.gate_proj ModuleList 已被 5.x 融合 3D 张量取代——strict 直搬的键位正身。
    前向按被命中的专家逐个 F.linear（与 HF MixtralExperts/DeepseekV2Experts 同构，含
    「one_hot(E+1) 多出的填充槽」之外的数值语义；权重×topk 权重后 index_add 汇聚）。
    """

    def __init__(self, hidden_size, n_expert, expert_dim):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(n_expert, 2 * expert_dim, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(n_expert, hidden_size, expert_dim))
        for p in (self.gate_up_proj, self.down_proj):
            nn.init.normal_(p, mean=0.0, std=0.02)
        self.n_expert, self.expert_dim = n_expert, expert_dim

    def forward(self, h, topk_weights, topk_idx):
        out = torch.zeros_like(h)                                             # (N,d)
        for e in range(self.n_expert):
            tok, slot = torch.nonzero(topk_idx == e, as_tuple=True)
            if tok.numel() == 0:
                continue
            gate, up = F.linear(h[tok], self.gate_up_proj[e]).chunk(2, dim=-1)  # (n_e,w) x2
            y = F.linear(F.silu(gate) * up, self.down_proj[e])                # (n_e,d)
            out.index_add_(0, tok, (y * topk_weights[tok, slot].to(y.dtype).unsqueeze(1)).to(out.dtype))
        return out


class HFMixtralMoE(nn.Module):
    """Mixtral 式 MoE（HF 键位 strict 直搬版）：mlp.gate.weight + mlp.experts.{gate_up,down}_proj。

    路由 = softmax+top-k+归一化（MixtralTopKRouter 语义）；无共享专家；无 jitter（eval/教学口径）。
    """

    def __init__(self, hidden_size, n_expert, top_k, expert_dim):
        super().__init__()
        self.gate = TopkGate(hidden_size, n_expert, top_k, mode="mixtral")
        self.experts = _FusedExperts(hidden_size, n_expert, expert_dim)

    def forward(self, x):
        shape = x.shape
        h = x.reshape(-1, shape[-1])
        _, w, idx = self.gate(x)
        out = self.experts(h, w, idx)
        return out.view(*shape)


class HFDeepseekMoE(nn.Module):
    """DeepSeekMoE 式 MoE（HF 键位 strict 直搬版）：路由专家 + 共享专家，DSV2 路由语义。

    键位：mlp.gate.weight / mlp.experts.{gate_up,down}_proj / mlp.shared_experts.{gate,up,down}_proj.weight
    （shared_experts 宽度 = n_shared * w_e，与 HF DeepseekV2Moe.shared_experts 同构）。
    路由 = fp32 线性+softmax+top-k（greedy）+ routed_scaling_factor，不归一化。
    """

    def __init__(self, hidden_size, n_expert, top_k, expert_dim, n_shared=1, routed_scaling_factor=1.0):
        super().__init__()
        self.gate = TopkGate(hidden_size, n_expert, top_k, mode="deepseek",
                             routed_scaling_factor=routed_scaling_factor)
        self.experts = _FusedExperts(hidden_size, n_expert, expert_dim)
        self.shared_experts = ExpertSwiGLU(hidden_size, n_shared * expert_dim)

    def forward(self, x):
        shape = x.shape
        h = x.reshape(-1, shape[-1])
        _, w, idx = self.gate(x)
        out = self.experts(h, w, idx).view(*shape)
        return out + self.shared_experts(x)


# ---------------- 第二刀：MLA 插件 ----------------
class MLAAttention(nn.Module):
    """手写 MLA（多头潜在注意力）——显式上投影路径，键位对齐 HF DeepseekV2Attention。

    论文记号 → HF 键映射（正文对照表用）：
      W_DKV（KV 下投影，产出压缩潜向量 c_KV 与解耦 RoPE 键 k_R） ↔ kv_a_proj_with_mqa (d→d_c+d_r)
      LN_c（潜向量 RMSNorm）                                       ↔ kv_a_layernorm (d_c)
      W_UK/W_UV（K/V 上投影，按头切片融合存一个矩阵）              ↔ kv_b_proj (d_c→h*(d_n+d_v))
      W_DQ/W_UQ（q 侧低秩分解，可选）                              ↔ q_a_proj/q_a_layernorm/q_b_proj
      W_KR 严格说并入 kv_a_proj_with_mqa 的后 d_r 列；W_QR ↔ q_b_proj 的 rope 列
      W_O                                                          ↔ o_proj (h*d_v→d)
    几何：h 头 × (d_nope 压缩键维 + d_rope 解耦 RoPE 维)；v 独立维 d_v（可与 d_nope 不同——DSV3 教学点）。
    KV cache 每 token 每层只存 (d_c + d_r) 个元素（c_KV 与 k_R 两份），
    K/V 头在推理时由上投影现算——「KV 折叠进低秩潜空间」的代码形态。
    注意：RoPE 只作用于 q_pe/k_pe 的 d_r 维（llama_slots.apply_rope 的 rotate_half 约定；
    HF DSV2 用复数约定，两者对 q·k 点积数学等价——q/k 同约定则旋转正交一致，见探针 g 记录）。
    """

    def __init__(self, hidden_size, n_head, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
                 v_head_dim, q_lora_rank=None, rope_theta=10000.0, rms_norm_eps=1e-6,
                 rope_style="half"):
        super().__init__()
        slots = bootstrap_book3()
        d = hidden_size
        self.h, self.d_c = n_head, kv_lora_rank
        self.d_nope, self.d_rope, self.d_v = qk_nope_head_dim, qk_rope_head_dim, v_head_dim
        self.q_lora_rank = q_lora_rank
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.scaling = self.qk_head_dim ** -0.5               # HF：qk_head_dim**-0.5（yarn 默认 mscale=1）
        self.rope_theta = rope_theta
        self.rope_style = rope_style                          # "half"=rotate_half（llama 系）|
                                                              # "complex"=复数约定（HF DSV2 view_as_complex）
        self._build_rope = slots.build_rope_cache
        self._rope = slots.apply_rope
        self._mask_cache = {}
        # 潜向量 LN 的 eps 固定 1e-6——对齐 HF：DeepseekV2RMSNorm(config.kv_lora_rank) 不传 config
        # rms_norm_eps，走类默认 1e-6（探针 G 踩坑记录：层内 LN 用 config eps、潜向量 LN 用默认）
        if q_lora_rank is None:
            self.q_proj = nn.Linear(d, n_head * self.qk_head_dim, bias=False)
            self.q_a_proj, self.q_a_layernorm, self.q_b_proj = None, None, None
        else:
            self.q_proj = None
            self.q_a_proj = nn.Linear(d, q_lora_rank, bias=False)            # W_DQ (d→r_q)
            self.q_a_layernorm = slots.RMSNorm(q_lora_rank, 1e-6)            # LN_q（HF 默认 eps）
            self.q_b_proj = nn.Linear(q_lora_rank, n_head * self.qk_head_dim, bias=False)  # W_UQ (r_q→h*(d_n+d_r))
        self.kv_a_proj_with_mqa = nn.Linear(d, kv_lora_rank + qk_rope_head_dim, bias=False)  # W_DKV+W_KR 融合
        self.kv_a_layernorm = slots.RMSNorm(kv_lora_rank, 1e-6)
        self.kv_b_proj = nn.Linear(kv_lora_rank, n_head * (qk_nope_head_dim + v_head_dim), bias=False)
        self.o_proj = nn.Linear(n_head * v_head_dim, d, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope_apply(self, q_pe, k_r):
        """对解耦 rope 维施加旋转。rope_style="half"：llama_slots.apply_rope（rotate_half，
        维度配对 (i, i+d/2)）；"complex"：复数约定（相邻维配对 (2i, 2i+1)，对齐 HF DSV2 的
        view_as_complex——两种约定各自内部自洽，但共享权重时 q·k 点积不同，跨实现对拍必须同约定）。"""
        if self.rope_style == "half":
            cos, sin = self._build_rope(q_pe.shape[2], self.d_rope, self.rope_theta, q_pe.device)
            return self._rope(q_pe, k_r.unsqueeze(1), cos, sin)
        n, d, device = q_pe.shape[2], self.d_rope, q_pe.device
        k_in = k_r.unsqueeze(1)                                              # (B,1,n,d) 单 KV 头（潜空间共享）
        inv = 1.0 / (self.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32, device=device) / d))
        ang = torch.outer(torch.arange(n, dtype=torch.float32, device=device), inv)   # (n,d/2)
        cos, sin = ang.cos()[None, None], ang.sin()[None, None]                       # (1,1,n,d/2)
        q0, q1 = q_pe[..., 0::2].float(), q_pe[..., 1::2].float()                     # (B,h,n,d/2)
        k0, k1 = k_in[..., 0::2].float(), k_in[..., 1::2].float()                     # (B,1,n,d/2)
        q_out = torch.stack((q0 * cos - q1 * sin, q0 * sin + q1 * cos), dim=-1).flatten(-2)
        k_out = torch.stack((k0 * cos - k1 * sin, k0 * sin + k1 * cos), dim=-1).flatten(-2)
        return q_out.to(q_pe.dtype), k_out.to(k_r.dtype)

    def _causal_mask(self, n, device):
        if n not in self._mask_cache or self._mask_cache[n].device != device:
            self._mask_cache[n] = torch.tril(torch.ones(n, n, device=device)).view(1, 1, n, n)
        return self._mask_cache[n]

    def latents(self, x):
        """产出「可缓存」的两份潜向量：c_KV (B,n,d_c，已 LN) 与 k_R (B,n,d_r，未旋转）。
        KV cache 存的就是这两份——权重吸收/账本实验的口径正身。"""
        c_raw = self.kv_a_proj_with_mqa(x)                                   # (B,n,d_c+d_r)
        c_kv, k_r = c_raw.split([self.d_c, self.d_rope], dim=-1)
        return self.kv_a_layernorm(c_kv), k_r

    def forward(self, x, cos=None, sin=None):
        B, n, d = x.shape
        if self.q_lora_rank is not None:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))          # (B,n,h*(d_n+d_r))
        else:
            q = self.q_proj(x)
        q = q.view(B, n, self.h, self.qk_head_dim).transpose(1, 2)           # (B,h,n,d_n+d_r)
        q_nope, q_pe = q.split([self.d_nope, self.d_rope], dim=-1)           # (B,h,n,·) x2

        c_kv, k_r = self.latents(x)                                          # (B,n,d_c),(B,n,d_r)
        q_pe, k_pe = self._rope_apply(q_pe, k_r)                             # 只旋转 rope 维；v 不转

        kv = self.kv_b_proj(c_kv).view(B, n, self.h, self.d_nope + self.d_v).transpose(1, 2)
        k_nope, v = kv.split([self.d_nope, self.d_v], dim=-1)                # (B,h,n,·)
        k = torch.cat([k_nope, k_pe.expand(B, self.h, n, self.d_rope)], dim=-1)  # (B,h,n,d_n+d_r)
        q_full = torch.cat([q_nope, q_pe], dim=-1)                           # (B,h,n,d_n+d_r)

        att = (q_full @ k.transpose(-2, -1)) * self.scaling                  # (B,h,n,n)
        att = att.masked_fill(self._causal_mask(n, x.device)[:, :, :n, :n] == 0, float("-inf"))
        att = F.softmax(att.float(), dim=-1).to(x.dtype)                     # fp32 softmax
        y = att @ v                                                          # (B,h,n,d_v)
        y = y.transpose(1, 2).contiguous().view(B, n, self.h * self.d_v)     # (B,n,h*d_v)
        return self.o_proj(y)                                                # (B,n,d)


def mla_absorbed_forward(attn: MLAAttention, x):
    """MLA 的「权重吸收」前向：把 K/V 上投影吸收进 q 侧与输出侧，注意力直接在潜空间算。

    与显式路径的恒等式（正文推导正身）：
      score = q_nopeᵀ(W_UK c_KV) + q_peᵀk_pe = (q_nopeᵀ W_UK) c_KV + q_peᵀk_pe
      out_head = Σ_j a_j · W_UV c_KV_j = W_UV (Σ_j a_j c_KV_j)
    即 W_UK 吸收进 q（q̂ = W_UKᵀ q_nope，(d_nope→d_c)），W_UV 吸收进输出（û = A·c_KV 后再 W_UVᵀ）。
    推理时 cache 只存 c_KV/k_R、每步现算 q̂——矩阵吸收的收益在「免 per-head 上投影」。
    键位切片：kv_b_proj.weight 第 e 头的行 [e*(d_n+d_v), e*(d_n+d_v)+d_n) = W_UK_e，其后 d_v 行 = W_UV_e。
    """
    B, n, d = x.shape
    h, d_c, d_n, d_r, d_v = attn.h, attn.d_c, attn.d_nope, attn.d_rope, attn.d_v
    if attn.q_lora_rank is not None:
        q = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(x)))
    else:
        q = attn.q_proj(x)
    q = q.view(B, n, h, attn.qk_head_dim).transpose(1, 2)                    # (B,h,n,d_n+d_r)
    q_nope, q_pe = q.split([d_n, d_r], dim=-1)
    c_kv, k_r = attn.latents(x)                                             # (B,n,d_c),(B,n,d_r)
    q_pe, k_pe = attn._rope_apply(q_pe, k_r)                                # 与显式路径同一 rope（同约定）

    scores = q_pe @ k_pe.transpose(-2, -1)                                  # (B,h,n,n) rope 项（广播到头）
    u = None
    W = attn.kv_b_proj.weight                                               # (h*(d_n+d_v), d_c)
    for e in range(h):
        W_UK = W[e * (d_n + d_v): e * (d_n + d_v) + d_n]                    # (d_n,d_c)
        W_UV = W[e * (d_n + d_v) + d_n: (e + 1) * (d_n + d_v)]              # (d_v,d_c)
        q_hat = q_nope[:, e] @ W_UK                                         # (B,n,d_c) ← 吸收 W_UK（q^T·W_UK）
        s = q_hat @ c_kv.transpose(-1, -2)                                  # (B,n,n) 潜空间内积
        scores[:, e] += s
    att = scores * attn.scaling
    att = att.masked_fill(attn._causal_mask(n, x.device)[:, :, :n, :n] == 0, float("-inf"))
    att = F.softmax(att, dim=-1)                                            # fp32（CPU 探针口径）
    for e in range(h):
        W_UV = W[e * (d_n + d_v) + d_n: (e + 1) * (d_n + d_v)]              # (d_v,d_c)
        u_e = att[:, e] @ c_kv                                              # (B,n,d_c) ← cache 直连注意力
        y_e = u_e @ W_UV.T                                                  # (B,n,d_v) ← 吸收 W_UV
        u = y_e.unsqueeze(2) if u is None else torch.cat([u, y_e.unsqueeze(2)], dim=2)
    y = u.reshape(B, n, h * d_v)                                            # (B,n,h*d_v)
    return attn.o_proj(y)                                                   # (B,n,d)


# ---------------- 插槽式整机：两刀的代码形态 ----------------
class SlotConfig:
    """SlotLLaMA 的配置。注意力插槽 mla=None → 沿用 Book3 GQA；dict → MLA（第二刀）。
    FFN 插槽 moe=None → dense SwiGLU；dict → MoE（第一刀），字段：
      n_expert/top_k/expert_dim/n_shared/mode/routed_scaling_factor/variant
      variant ∈ {"teach"(ModuleList 教学版) | "hf_mixtral" | "hf_deepseek"}；
    moe_first_k_dense：前 k 层保持 dense（DSV2 first_k_dense_replace 同构，默认 0 全 MoE）。
    """

    def __init__(self, vocab_size=8192, hidden_size=512, num_hidden_layers=6,
                 num_attention_heads=8, num_key_value_heads=None, head_dim=None,
                 rope_theta=10000.0, rms_norm_eps=1e-5, initializer_range=0.02,
                 intermediate_size=1408, moe=None, moe_first_k_dense=0, mla=None):
        self.vocab_size, self.hidden_size = vocab_size, hidden_size
        self.num_hidden_layers, self.num_attention_heads = num_hidden_layers, num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.rope_theta, self.rms_norm_eps = rope_theta, rms_norm_eps
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.moe, self.moe_first_k_dense, self.mla = moe, moe_first_k_dense, mla
        assert num_attention_heads % self.num_key_value_heads == 0


class SlotDecoderLayer(nn.Module):
    """残差河恒宽的插槽层——拓扑仍是 Book2/3 的三插槽 Block：

        x = x + attn(input_layernorm(x))      插槽① RMSNorm → 插槽④ GQA|MLA
        x = x + mlp(post_attention_layernorm(x))   插槽① RMSNorm → 插槽② SwiGLU|MoE
    键名与 HF Llama/Mixtral/DeepseekV2 的 DecoderLayer 一致（input_layernorm/self_attn/
    post_attention_layernorm/mlp）——strict 直搬的层内正身。
    """

    def __init__(self, cfg: SlotConfig, layer_idx: int):
        super().__init__()
        slots = bootstrap_book3()
        self.input_layernorm = slots.RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if cfg.mla is not None:
            self.self_attn = MLAAttention(cfg.hidden_size, cfg.num_attention_heads,
                                          cfg.mla["kv_lora_rank"], cfg.mla["qk_nope_head_dim"],
                                          cfg.mla["qk_rope_head_dim"], cfg.mla["v_head_dim"],
                                          q_lora_rank=cfg.mla.get("q_lora_rank"),
                                          rope_theta=cfg.rope_theta, rms_norm_eps=cfg.rms_norm_eps,
                                          rope_style=cfg.mla.get("rope_style", "half"))
        else:
            like = type("GQA_cfg", (), {})()                                  # 鸭子类型：GroupedQueryAttention 只读这四件
            like.hidden_size, like.head_dim = cfg.hidden_size, cfg.head_dim
            like.num_attention_heads = cfg.num_attention_heads
            like.num_key_value_heads = cfg.num_key_value_heads
            self.self_attn = slots.GroupedQueryAttention(like)
        self.post_attention_layernorm = slots.RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        dense = slots.SwiGLU(type("S_cfg", (), {"hidden_size": cfg.hidden_size,
                                                "intermediate_size": cfg.intermediate_size})())
        if cfg.moe is None or layer_idx < cfg.moe_first_k_dense:
            self.mlp = dense                                                  # dense 槽（含 first_k_dense 层）
        else:
            m = cfg.moe
            common = (cfg.hidden_size, m["n_expert"], m["top_k"], m["expert_dim"])
            if m.get("variant", "teach") == "teach":
                self.mlp = SparseMoE(*common, n_shared=m.get("n_shared", 0),
                                     mode=m.get("mode", "mixtral"),
                                     routed_scaling_factor=m.get("routed_scaling_factor", 1.0))
            elif m["variant"] == "hf_mixtral":
                self.mlp = HFMixtralMoE(*common)
            elif m["variant"] == "hf_deepseek":
                self.mlp = HFDeepseekMoE(*common, n_shared=m.get("n_shared", 1),
                                         routed_scaling_factor=m.get("routed_scaling_factor", 1.0))
            else:
                raise ValueError(m["variant"])

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)             # (B,n,C) 残差河
        mlp_out = self.mlp(self.post_attention_layernorm(x))
        if isinstance(mlp_out, tuple):                                        # SparseMoE 返回 (out, counts)
            mlp_out = mlp_out[0]
        x = x + mlp_out                                                       # (B,n,C)
        return x


class SlotLLaMA(nn.Module):
    """插槽式 LLaMA 整机：Book3 207M 骨架的「两刀改装台」。

    前向接口与 Book2/3 一致：idx (B,n) -> logits (B,n,V)；给 targets 返回逐 token 交叉熵（fp32）。
    state_dict 键与 HF Llama 同构（model.embed_tokens/layers.i.../norm/lm_head）——GQA+teach 版
    可直接对拍 LlamaForCausalLM；HF 变体插槽可对拍 Mixtral/DeepseekV2ForCausalLM（探针 g）。
    注意：MLA 插槽自带 rope（作用于 d_r 维），忽略 backbone 按 GQA head_dim 现算的 cos/sin。
    """

    def __init__(self, cfg: SlotConfig, init_std=None):
        super().__init__()
        slots = bootstrap_book3()
        self.cfg = cfg
        self.model = _Backbone(cfg, slots)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)  # untied
        std = init_std or cfg.initializer_range
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=std)

    def n_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.model.embed_tokens.weight.numel() + self.lm_head.weight.numel()
        return n

    def forward(self, idx, targets=None):
        x = self.model(idx)                                                   # (B,n,C)
        logits = self.lm_head(x)                                              # (B,n,V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   targets.reshape(-1))
        return logits, loss


class _Backbone(nn.Module):
    def __init__(self, cfg, slots):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([SlotDecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = slots.RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, idx):
        cfg = self.cfg
        B, n = idx.shape
        x = self.embed_tokens(idx)                                            # (B,n,C)
        cos, sin = bootstrap_book3().build_rope_cache(n, cfg.head_dim, cfg.rope_theta, idx.device)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)                                                   # (B,n,C)


# ---------------- 三口径手算账：总参 / 激活参 / KV cache ----------------
def hand_account(cfg: SlotConfig):
    """参数逐项手算（无 bias、untied、RMSNorm 只 weight）。返回明细 dict——与 n_params() assert 对账。

    注意力每层：GQA = q/o 全宽 + k/v 按 h_kv 缩（Book3 公式）；MLA = q 侧低秩 + KV 下投影
    + 上投影 + o（见 MLAAttention 键映射）。FFN 每层：dense = 3·d·d_ff；MoE = 路由门 E·d
    + 路由专家 E·3·d·w + 共享专家 n_shared·3·d·w。
    """
    d, L, V = cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size
    if cfg.mla is not None:
        m = cfg.mla
        r_q = m.get("q_lora_rank")
        q_side = (d * r_q + r_q * cfg.num_attention_heads * (m["qk_nope_head_dim"] + m["qk_rope_head_dim"])
                  if r_q is not None else d * cfg.num_attention_heads * (m["qk_nope_head_dim"] + m["qk_rope_head_dim"]))
        kv_side = (d * (m["kv_lora_rank"] + m["qk_rope_head_dim"])
                   + m["kv_lora_rank"] * cfg.num_attention_heads * (m["qk_nope_head_dim"] + m["v_head_dim"]))
        o_side = cfg.num_attention_heads * m["v_head_dim"] * d
        attn_per_layer = q_side + kv_side + o_side
        attn_detail = {"q_side": q_side, "kv_down": d * (m["kv_lora_rank"] + m["qk_rope_head_dim"]) + m["kv_lora_rank"],
                       "kv_up": m["kv_lora_rank"] * cfg.num_attention_heads * (m["qk_nope_head_dim"] + m["v_head_dim"]),
                       "o": o_side}
    else:
        attn_per_layer = d * d * 2 + d * cfg.head_dim * cfg.num_key_value_heads * 2
        attn_detail = {"q_o": d * d * 2, "kv": d * cfg.head_dim * cfg.num_key_value_heads * 2}

    mlp_dense = 3 * d * cfg.intermediate_size
    mlp_moe = None
    if cfg.moe is not None:
        w, E = cfg.moe["expert_dim"], cfg.moe["n_expert"]
        n_shared = cfg.moe.get("n_shared", 0) if cfg.moe.get("variant", "teach") != "hf_mixtral" else 0
        mlp_moe = {"gate": E * d, "experts": E * 3 * d * w, "shared": n_shared * 3 * d * w}

    per_layer, mlp_total = [], 0
    for i in range(L):
        norms = 2 * d + (cfg.mla["kv_lora_rank"] if cfg.mla else 0) \
                + (cfg.mla.get("q_lora_rank") or 0 if cfg.mla else 0)
        mlp = mlp_dense if (cfg.moe is None or i < cfg.moe_first_k_dense) else sum(mlp_moe.values())
        mlp_total += mlp
        per_layer.append(attn_per_layer + mlp + norms)
    emb = V * d * 2 + d                                                      # untied 两份 + 终态 norm
    total = sum(per_layer) + emb
    return {"attn_per_layer": attn_per_layer, "attn_detail": attn_detail,
            "mlp_dense": mlp_dense, "mlp_moe": mlp_moe, "mlp_total": mlp_total,
            "layers": sum(per_layer), "embedding_and_head": emb, "total": total}


def activated_account(cfg: SlotConfig):
    """单 token 激活参数手算：注意力全激活；MoE 层 FFN 只激活 top_k 个路由专家 + 全部共享专家；
    dense 层全激活。返回 (含嵌入与 lm_head 的口径, non-embedding 口径)——DSV3「37B/671B」双口径的 mini 版。"""
    d, L, V = cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size
    act = hand_account(cfg)
    attn = act["attn_per_layer"]
    mlp_active_total = 0
    for i in range(L):
        if cfg.moe is None or i < cfg.moe_first_k_dense:
            mlp_active_total += act["mlp_dense"]
        else:
            m = cfg.moe
            n_shared = m.get("n_shared", 0) if m.get("variant", "teach") != "hf_mixtral" else 0
            mlp_active_total += 3 * d * (m["top_k"] * m["expert_dim"] + n_shared * m["expert_dim"])
    norms_per_layer = 2 * d + (cfg.mla["kv_lora_rank"] if cfg.mla else 0) \
        + (cfg.mla.get("q_lora_rank") or 0 if cfg.mla else 0)
    body = L * (attn + norms_per_layer) + mlp_active_total + d
    return body + V * d * 2, body                                            # (含 emb+head, non-emb)


def kv_per_token_elements(cfg: SlotConfig, dtype_bytes=2):
    """KV cache 每 token 字节账（batch=1）。GQA = 2(K,V)·L·h_kv·d_k（Book3 ch6 公式续用）；
    MLA = L·(d_c+d_r)——潜向量 c_KV 一份 + 解耦 rope 键 k_R 一份（K/V 上投影移到读出端，不进 cache）。"""
    L = cfg.num_hidden_layers
    if cfg.mla is not None:
        elems = L * (cfg.mla["kv_lora_rank"] + cfg.mla["qk_rope_head_dim"])
    else:
        elems = 2 * L * cfg.num_key_value_heads * cfg.head_dim
    return elems, elems * dtype_bytes


# ---------------- 小工具 ----------------
def lr_at(step, total, peak=1e-3, warmup=20):
    """模块档学习率：常数 peak + 线性 warmup（Book2 五件套的探针简化版——可行性只看可训性）。"""
    return peak * min(1.0, step / warmup)


def eval_loss(model, device, batches=8, batch=8, block=512, eval_path=None):
    """heldout eval loss（eval.bin 独立文件；MPS bf16 autocast / CPU fp32）。"""
    import numpy as np
    arr = np.memmap(eval_path or EVAL_SP8K, dtype=np.uint16, mode="r")
    offs = np.linspace(0, len(arr) - batch * block - 1, batches).astype(int)
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for o in offs:
            span = np.asarray(arr[o: o + batch * block + 1]).astype(np.int64)
            x = torch.from_numpy(span[:-1].reshape(batch, block)).to(device)
            y = torch.from_numpy(span[1:].reshape(batch, block)).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "mps" \
                    else torch.autocast("cpu", enabled=False):
                _, loss = model(x, y)
            losses.append(loss.item())
    if was_training:
        model.train()
    return float(np.mean(losses))


def train_loop(model, steps, batch, block, device, tokens_path, peak_lr=1e-3, warmup=20,
               log_every=None):
    """共用短训环：AdamW(0.9,0.95,wd0.1)+clip1.0+warmup（Book3 train_215 同配方），
    MPS bf16 autocast（fp32 权重——同口径），逐步 sync 计时。返回 (secs, losses, stream)。"""
    stream = TokenStream(tokens_path, batch, block)
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1)
    model.train()
    secs, losses = [], []
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, peak_lr, warmup)
        x, y = stream.batch(step, device)
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "mps" \
                else torch.autocast("cpu", enabled=False):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device.type == "mps":
            torch.mps.synchronize()
        secs.append(time.perf_counter() - t0)
        losses.append(loss.item())
        if log_every and step % log_every == 0:
            print(f"    step {step}/{steps} loss {loss.item():.4f}", flush=True)
    return secs, losses, stream


def pick_device(pref="mps"):
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mps_peak_mb(reset=False):
    """MPS 统一内存口径：torch.mps.current_allocated_memory 峰值（无则返回 None）。"""
    try:
        if reset:
            torch.mps.reset_peak_memory_stats()
            return None
        return torch.mps.current_allocated_memory() / 1024 / 1024
    except Exception:
        return None


def save_json(name, obj, out_name="run1"):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}_{out_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[产物] {path}")
    return path


# ---------------- 自测（秒级，CPU） ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Book4 插槽组件库自测")
    ap.add_argument("--out-name", default="selftest", help="产物 JSON 后缀（防覆写）")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    print(f"[bootstrap] llama_slots 引入自：{slots_paths()}")

    # ① 模块档 dense（26.1M）与 MoE（59.2M）参数账 assert
    dense_cfg = SlotConfig(vocab_size=8192, hidden_size=512, num_hidden_layers=6,
                           num_attention_heads=8, num_key_value_heads=4, intermediate_size=1408)
    moe_cfg = SlotConfig(vocab_size=8192, hidden_size=512, num_hidden_layers=6,
                         num_attention_heads=8, num_key_value_heads=4, intermediate_size=1408,
                         moe=dict(n_expert=8, top_k=2, expert_dim=704, n_shared=0, variant="teach"))
    for tag, c in (("dense", dense_cfg), ("moe", moe_cfg)):
        m = SlotLLaMA(c)
        hand = hand_account(c)
        assert m.n_params() == hand["total"], f"{tag}: {m.n_params()} != {hand['total']}"
        act, act_ne = activated_account(c)
        print(f"[{tag}] 总参 {m.n_params():,}（手算逐位一致）| 激活参 {act:,}（non-emb {act_ne:,}）"
              f"| KV/token {kv_per_token_elements(c)}")

    # ② MoE 前向 + 负载统计 shape 检查
    x = torch.randint(0, 8192, (2, 64))
    moe = SlotLLaMA(moe_cfg)
    logits, loss = moe(x, x)
    assert logits.shape == (2, 64, 8192) and torch.isfinite(loss)
    counts = moe.model.layers[0].mlp.expert_selection_counts(moe.model.embed_tokens(x))
    assert counts.sum().item() == 2 * 64 * 2                                # 每 token 恰选 top-2
    print(f"[moe 路由] counts(第 0 层, 随机初权重) = {counts.tolist()}（Σ = 每 token k=2 ✓）")

    # ③ MLA 显式 vs 吸收两路等价快检（小几何，fp32 CPU）
    mla_cfg = SlotConfig(vocab_size=512, hidden_size=256, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=1, intermediate_size=704,
                         mla=dict(kv_lora_rank=64, q_lora_rank=64, qk_nope_head_dim=32,
                                  qk_rope_head_dim=16, v_head_dim=32))
    mm = SlotLLaMA(mla_cfg)
    hand = hand_account(mla_cfg)
    assert mm.n_params() == hand["total"], f"{mm.n_params()} != {hand['total']}"
    attn = mm.model.layers[0].self_attn
    h = torch.randn(2, 96, 256)
    y1 = attn(h)
    y2 = mla_absorbed_forward(attn, h)
    dmax = (y1 - y2).abs().max().item()
    assert dmax < 1e-4, f"MLA 两路不等价：max|Δ| = {dmax}"
    print(f"[MLA 等价] 显式 vs 吸收 max|Δ| = {dmax:.2e}（fp32 CPU）| "
          f"KV/token elems: GQA口径 {2 * 2 * 1 * 64} vs MLA {2 * (64 + 16)}")
    save_json("slots_selftest", {"seed": SEED, "dense_params": SlotLLaMA(dense_cfg).n_params(),
                                 "moe_params": SlotLLaMA(moe_cfg).n_params(),
                                 "mla_parity_max_abs": dmax}, args.out_name)
    print("moe_mla_slots 自测全部通过")
