# gdn.py —— Book5 ch5 正式件：Gated DeltaNet 双形态（recurrent / chunkwise）+ 对拍 + scaling
# 用途：ch5（路线三·线性注意力，数学最重章）的本机实验正身——
#   ① op 级双形态算子：gdn_recurrent（逐 token 递归——推理形态，机制正身）与 gdn_chunkwise
#      （块内 UT 变换并行 + 块间线性扫描——训练形态）；两者是同一递推的两种写法（对拍互证）；
#   ② GatedDeltaNet(cfg) 层模块：教学骨架（in_proj_qkvz/in_proj_ba/conv1d/A_log/dt_bias/out_proj，
#      连续 [q,k,v,z] 切分——教学口径；HF 键位交错版见本目录 qwen3next_mini.py 的 K1 坑），
#      forward(path="chunk"|"recurrent") 一键切形态——「训练要并行、推理要省」同一层的两条路；
#      本层同时是 ch6 K3 玩具的 GDN 替身插槽（对 KDA 做「门粒度」单点替换的基线臂）；
#   ③ 对拍函数内置（对拍驱动开发）：自研 recurrent vs HF torch_recurrent_gated_delta_rule /
#      自研 chunk vs HF torch_chunk_gated_delta_rule / 双形态互拍（含 initial_state 与
#      output_final_state 的状态级对拍）——三方两两 max|Δ| 全绿才算数；
#   ④ scaling 函数：MPS 上 chunkwise vs 满注意力 n=512..8192 的 wall-clock，log-log 斜率
#      ——显式启用的重新测量入口；旧记录入口轴错误，历史曲线不得继续支持复杂度结论。
# 【公式正身（与 HF transformers 5.18.0 qwen3_next L378-574 逐行互证，探针期已核）】
#   R1 门方向：g ≤ 0（log 空间衰减；模块侧 g = −exp(A_log)·softplus(a + dt_bias)，恒 ≤ 0）；
#      递归中状态先乘 exp(g_t) 衰减、再读预测/写入——「先忘后写」。
#   R2 归一化位置：q/k 各自 l2norm（eps=1e-6，FLA 口径 x/sqrt(Σx²+eps)）先于 q×Dk^{-1/2}。
#   R3 β 施加位置：β 乘在 k 与 v（k_beta=βk、v_beta=βv）——delta = β(v−Sᵀk) 的矩阵形态。
#   R4 UT 系统：块内 (I+M)（M=tril(βKKᵀ·D̂,−1)）经三角求解（solve_triangular unitriangular=True）
#      ——不显式求逆；旧记录报告特定MPS样例约40倍差异，不作通用速度保证（首版误写 (I−M)^{-1} 被对拍抓出）。
#   R5 chunk 衰减三处：q 侧乘 exp(c)；k 侧乘 exp(c_last−c)（跨块读老状态补齐到块尾）；状态更新乘 exp(c_last)。
# 复杂度双口径（纪律⑨）：训练/并行形态 O(n·C)（块间线性扫描、块内二次）；推理/递归形态每步
#   O(Dk·Dv) 状态更新与 n 无关——「线性」的两句话各说各的，不混。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch05/gdn.py" [--out-name run1]
#   默认仅CPU fp32对拍；--scaling才做计时，实际设备与版本写入结果。
# 产物：log/book5-ch05/gdn_{out}.json（不入库）
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "env.sh").is_file()), Path.cwd())
OUT_DIR = os.path.join(REPO_ROOT, "log", "book5-ch05")
SEED = 20261002


def save_json(name, obj, out_name):
    if not out_name or Path(out_name).name != out_name or out_name in (".", ".."):
        raise ValueError("--out-name 必须为单个文件名")
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}_{out_name}.json")
    with open(path, "x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[产物] {path}")
    return path


def l2norm(x, dim=-1, eps=1e-6):
    """FLA 口径 l2norm：x / sqrt(Σx² + eps)（对齐 HF qwen3_next 的 fla 对齐注释）。"""
    return x / torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)


# ---------------- 形态一：recurrent（逐 token 递归——推理形态，机制正身） ----------------
def _check_inputs(q, k, v, g, beta, initial_state):
    """教学接口要求(B,S,H,D)，门(B,S,H)，末状态(B,H,Dk,Dv)。"""
    if k.ndim != 4 or q.shape != k.shape or v.ndim != 4 or v.shape[:3] != k.shape[:3]:
        raise ValueError("q/k/v 必须为同批、同序列、同头的 (B,S,H,D)")
    B, n, H, dk = k.shape
    if min(B, n, H, dk, v.shape[-1]) <= 0 or g.shape != k.shape[:3] or beta.shape != k.shape[:3]:
        raise ValueError("非空输入；g/beta 形状必须为 (B,S,H)")
    if initial_state is not None and initial_state.shape != (B,H,dk,v.shape[-1]):
        raise ValueError("initial_state 形状必须为 (B,H,Dk,Dv)")
    if not bool(torch.isfinite(g).all()) or bool((g > 0).any()):
        raise ValueError("log 衰减 g 必须有限且非正；精确 alpha=0 用 gdn_math 的有限乘积版")


def gdn_recurrent(q, k, v, g, beta, initial_state=None, output_final_state=False):
    """gated delta rule 递归形态。输入布局 (B,S,H,D)（q/k 同 H；v 同 H）——与 HF 参考一致便于对拍。

    每步（先忘后写，R1）：
      S ← S·exp(g_t)                          # (B,H,Dk,Dv) 状态衰减
      delta_t = β_t·(v_t − Sᵀk_t)             # 老状态预测的残差 = 增量
      S ← S + k_t ⊗ delta_t                   # 外积写入（秩一更新）
      o_t = Sᵀ q_t                            # 读出
    q/k 先 l2norm，q 再乘 Dk^{-1/2}（R2）。
    """
    _check_inputs(q, k, v, g, beta, initial_state)
    out_dtype = q.dtype
    B, S_len, H, Dk = k.shape
    Dv = v.shape[-1]
    q = l2norm(q.float()).transpose(1, 2)                                       # (B,H,S,Dk)
    k = l2norm(k.float()).transpose(1, 2)                                       # (B,H,S,Dk)
    v = v.float().transpose(1, 2)                                               # (B,H,S,Dv)
    beta = beta.float().transpose(1, 2)                                         # (B,H,S)
    g = g.float().transpose(1, 2)                                               # (B,H,S)
    q = q * (Dk ** -0.5)
    S = (torch.zeros(B, H, Dk, Dv, device=q.device, dtype=torch.float32) if initial_state is None
         else initial_state.float().to(q.device))
    out = torch.empty(B, H, q.shape[2], Dv, device=q.device, dtype=torch.float32)
    for t in range(q.shape[2]):
        q_t, k_t, v_t = q[:, :, t], k[:, :, t], v[:, :, t]                      # (B,H,Dk)/(B,H,Dk)/(B,H,Dv)
        S = S * g[:, :, t].exp()[..., None, None]                               # (B,H,Dk,Dv) 先忘
        kv_mem = torch.einsum("bhkv,bhk->bhv", S, k_t)                          # (B,H,Dv) Sᵀk
        delta = (v_t - kv_mem) * beta[:, :, t][..., None]                       # (B,H,Dv) 增量
        S = S + torch.einsum("bhk,bhv->bhkv", k_t, delta)                       # 秩一写入
        out[:, :, t] = torch.einsum("bhkv,bhk->bhv", S, q_t)                    # (B,H,Dv) 读出
    out = out.transpose(1, 2).contiguous()                                      # (B,S,H,Dv)
    return out.to(out_dtype), (S if output_final_state else None)


# ---------------- 形态二：chunkwise（块并行——训练形态，WY/UT 表示） ----------------
def gdn_chunkwise(q, k, v, g, beta, chunk=64, initial_state=None, output_final_state=False):
    """gated delta rule 块并行形态（与 HF 参考数学等价；本版使用三角求解，R4/R5）。

    输入布局 (B,S,H,D)；输出 (B,S,H,Dv)。S 非 chunk 倍数时 pad（pad 位 β=0、g=0 → 不写入不衰减）。
    块内：u=(I+M)^{-1}vβ、k̃=(I+M)^{-1}(βk·e^c)（UT 三角求解）；块间：状态线性扫描。
    """
    _check_inputs(q, k, v, g, beta, initial_state)
    out_dtype = q.dtype
    if not isinstance(chunk, int) or chunk <= 0:
        raise ValueError("chunk 必须是正整数")
    B, S_len, H, Dk = k.shape
    Dv = v.shape[-1]
    q = l2norm(q.float()).transpose(1, 2) * (Dk ** -0.5)                        # (B,H,S,Dk) l2norm + scale（R2）
    k = l2norm(k.float()).transpose(1, 2)                                       # (B,H,S,Dk)
    v = v.float().transpose(1, 2)                                               # (B,H,S,Dv)
    beta = beta.float().transpose(1, 2)                                         # (B,H,S)
    g = g.float().transpose(1, 2)                                               # (B,H,S)

    pad = (chunk - S_len % chunk) % chunk
    if pad:
        q, k, v = F.pad(q, (0, 0, 0, pad)), F.pad(k, (0, 0, 0, pad)), F.pad(v, (0, 0, 0, pad))
        beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))                      # pad 位 β=0、g=0 → 不写入不衰减
    NC = (S_len + pad) // chunk

    # 块化视图（索引约定：w=块号，i/j=块内位置，d=Dk，e=Dv）
    qb = q.view(B, H, NC, chunk, Dk)                                            # (B,H,w,i,d)
    kb = k.view(B, H, NC, chunk, Dk)                                            # (B,H,w,j,d)
    vb = v.view(B, H, NC, chunk, Dv)                                            # (B,H,w,j,e)
    k_beta = (k * beta[..., None]).view(B, H, NC, chunk, Dk)                    # (B,H,w,i,d) βk（R3）
    v_beta = (v * beta[..., None]).view(B, H, NC, chunk, Dv)                    # (B,H,w,i,e) βv

    c = g.view(B, H, NC, chunk).cumsum(dim=-1)                                  # (B,H,w,i) 块内累积 log 衰减
    if not bool(torch.isfinite(c).all()):
        raise ValueError("块内累计 log 衰减溢出；请减小块长或检查门参数")
    D_pair = (c.unsqueeze(-1) - c.unsqueeze(-2)).masked_fill(                   # (B,H,w,i,j) D̂=exp(c_i−c_j)
        torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).triu(1), float("-inf")).exp()

    # UT 系统（R4）：M = tril(βKKᵀ·D̂, -1)；u = (I+M)^{-1}·vβ、k̃ = (I+M)^{-1}·(βk·e^c)
    #   经三角求解直接得 u 与 k̃（不显式求逆；HF 5.14.1 参考则用行递推构造逆的作用；
    #   首版误写 (I−M)^{-1} 被对拍抓出；旧记录报告特定MPS样例约40倍差异，不作通用速度保证——如实记录）
    M = (k_beta @ kb.transpose(-1, -2)) * D_pair                                # (B,H,w,i,j) βk·kᵀ·D̂
    M = M - torch.diag_embed(M.diagonal(dim1=-2, dim2=-1))                      # 去对角 → 严格下三角
    eye = torch.eye(chunk, device=q.device, dtype=torch.float32).expand(B, H, NC, chunk, chunk).contiguous()

    u = torch.linalg.solve_triangular(eye + M, v_beta, upper=False, unitriangular=True)      # (B,H,w,i,e)
    k_tilde = torch.linalg.solve_triangular(eye + M, k_beta * c.exp().unsqueeze(-1),         # (B,H,w,i,d)
                                            upper=False, unitriangular=True)
    intra = (qb @ kb.transpose(-1, -2)) * D_pair                                # (B,H,w,i,j) 块内注意 (qkᵀ)·D̂
    q_dec = qb * c.exp().unsqueeze(-1)                                          # (B,H,w,i,d) q·e^{c}（R5）
    k_cross = kb * (c[..., -1:] - c).exp().unsqueeze(-1)                        # (B,H,w,j,d) k·e^{c_last−c}
    chunk_decay = c[..., -1].exp()                                              # (B,H,w) e^{c_last}

    S = (torch.zeros(B, H, Dk, Dv, device=q.device, dtype=torch.float32) if initial_state is None
         else initial_state.float().to(q.device))
    out = torch.empty(B, H, NC, chunk, Dv, device=q.device, dtype=torch.float32)
    for w in range(NC):                                                         # 块间线性扫描——「O(n) 在这里」
        v_new = u[:, :, w] - k_tilde[:, :, w] @ S                               # (B,H,i,e) 减老状态可预测部分
        out[:, :, w] = (q_dec[:, :, w] @ S                                      # (B,H,i,e) 跨块读老状态
                        + intra[:, :, w] @ v_new)                               # 块内注意
        S = S * chunk_decay[:, :, w, None, None] \
            + k_cross[:, :, w].transpose(-1, -2) @ v_new                        # (B,H,d,e) 状态更新（先忘后写 R1）
    out = out.reshape(B, H, NC * chunk, Dv)[:, :, :S_len].transpose(1, 2).contiguous()
    return out.to(out_dtype), (S if output_final_state else None)


# ---------------- 层模块：教学骨架（连续 [q,k,v,z] 切分；ch6 替身插槽同款） ----------------
class GatedDeltaNet(nn.Module):
    """GDN 注意力层（教学骨架）——ch6 K3 玩具的 GDN 替身插槽。

    结构（连续 [q,k,v,z] 切分的教学口径；HF 键位交错排布的 strict 直搬版见 qwen3next_mini.py）：
      in_proj_qkvz: d → 2·h_k·d_k + 2·h_v·d_v（q/k/v/z 融合投影，连续四段）
      in_proj_ba:   d → 2·h_v（β 与门 a 输入）
      conv1d:       因果短卷积 (k_conv=4, groups=conv_dim, silu；pad=k−1 后取前 n 位)
      A_log/dt_bias: 门 g = −exp(A_log)·softplus(a + dt_bias) ≤ 0（R1；fp32 计算降低低精度误差，仍需检查有限值）
      核心算子:     gdn_chunkwise / gdn_recurrent（path 切换）
      out 前处理:   RMSNormGated（z 为门，silu 门控；教学骨架无 weight——参数账与 toyA 正身
                    对齐；HF 带权版见 qwen3next_mini.py）
      out_proj:     h_v·d_v → d
    forward(x, cos=None, sin=None, path="chunk") 兼容 SlotDecoderLayer 的注意力插槽签名（忽略 cos/sin——
    线性层不做RoPE；因果递推与短卷积仍含顺序）。
    """

    def __init__(self, d, h_k, h_v, d_k, d_v, conv_kernel=4, chunk=64):
        super().__init__()
        if min(d, h_k, h_v, d_k, d_v, conv_kernel, chunk) <= 0 or h_v % h_k:
            raise ValueError("维度/块长须为正，值头数必须是键头数的整数倍")
        self.h_k, self.h_v, self.d_k, self.d_v = h_k, h_v, d_k, d_v
        self.chunk = chunk
        self.conv_dim = 2 * h_k * d_k + h_v * d_v
        self.in_proj_qkvz = nn.Linear(d, 2 * h_k * d_k + 2 * h_v * d_v, bias=False)
        self.in_proj_ba = nn.Linear(d, 2 * h_v, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, conv_kernel, groups=self.conv_dim,
                                padding=conv_kernel - 1, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(h_v).uniform_(0.01, 8.0)))
        self.dt_bias = nn.Parameter(torch.randn(h_v) * 0.1)
        self.out_proj = nn.Linear(h_v * d_v, d, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, cos=None, sin=None, path="chunk"):
        B, n, d = x.shape
        qkvz = self.in_proj_qkvz(x)                                              # (B,n,2h_kd_k+2h_vd_v)
        q, k, v, z = qkvz.split([self.h_k * self.d_k, self.h_k * self.d_k,
                                 self.h_v * self.d_v, self.h_v * self.d_v], dim=-1)
        ba = self.in_proj_ba(x)                                                  # (B,n,2h_v)
        b_raw, a = ba.split([self.h_v, self.h_v], dim=-1)
        beta = b_raw.sigmoid()                                                   # (B,n,h_v) 写入门
        # 因果短卷积（pad=k−1 后取前 n 位——HF causal_conv1d_fn 同口径）+ silu
        qkv = torch.cat([q, k, v], dim=-1).transpose(1, 2)                       # (B,conv_dim,n)
        qkv = F.silu(self.conv1d(qkv)[..., :n])                                  # (B,conv_dim,n)
        q, k, v = qkv.split([self.h_k * self.d_k, self.h_k * self.d_k, self.h_v * self.d_v],
                            dim=1)
        q = q.transpose(1, 2).view(B, n, self.h_k, self.d_k)                     # (B,n,h_k,d_k)
        k = k.transpose(1, 2).view(B, n, self.h_k, self.d_k)
        v = v.transpose(1, 2).view(B, n, self.h_v, self.d_v)
        if self.h_v != self.h_k:                                                 # GQA 比：k 侧 repeat（HF 同款）
            rep = self.h_v // self.h_k
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)     # (B,n,h_v) ≤ 0
        if path == "chunk":
            core, _ = gdn_chunkwise(q, k, v, g, beta, chunk=self.chunk)          # (B,n,h_v,d_v)
        elif path == "recurrent":
            core, _ = gdn_recurrent(q, k, v, g, beta)                            # (B,n,h_v,d_v)
        else:
            raise ValueError(path)
        # 输出门控 RMSNorm（z 为门；无权重版——教学骨架与 toyA 参数账正身：
        # HF Qwen3NextRMSNormGated 带 weight，strict 直搬版（带权）在 qwen3next_mini.py）
        core = core.reshape(-1, self.d_v)
        z2 = z.reshape(-1, self.d_v)
        var = core.float().pow(2).mean(-1, keepdim=True)
        core = (core.float() * torch.rsqrt(var + 1e-6) * F.silu(z2.float())).to(core.dtype)
        core = core.view(B, n, self.h_v * self.d_v)                              # (B,n,h_v·d_v)
        return self.out_proj(core)                                               # (B,n,d)


# ---------------- 对拍（内置；三方两两） ----------------
def parity_hf(B=2, S=384, H=2, Dk=64, Dv=64, seed=SEED, chunk=64):
    """自研双形态 vs HF 参考实现（transformers 5.18.0 qwen3_next torch 双 kernel）。

    返回 {对拍名: max|Δ|}；含 initial_state 传递与 output_final_state 的状态级对拍。
    """
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule)
    torch.manual_seed(seed)
    q = torch.randn(B, S, H, Dk)
    k = torch.randn(B, S, H, Dk)
    v = torch.randn(B, S, H, Dv)
    beta = torch.randn(B, S, H).sigmoid()
    g = -F.softplus(torch.randn(B, S, H) * 1.5)                                  # ≤ 0：log 空间衰减（R1）
    S0 = torch.randn(B, H, Dk, Dv) * 0.05                                        # 非零初始状态（对拍 state 传递）

    out_r, S_r = gdn_recurrent(q, k, v, g, beta, initial_state=S0, output_final_state=True)
    out_c, S_c = gdn_chunkwise(q, k, v, g, beta, chunk=chunk, initial_state=S0,
                               output_final_state=True)
    out_hr, S_hr = torch_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=S0,
                                                    output_final_state=True, use_qk_l2norm_in_kernel=True)
    out_hc, S_hc = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=S0,
                                                output_final_state=True, use_qk_l2norm_in_kernel=True)
    parity = {
        "mine_recurrent_vs_mine_chunk_out": float((out_r - out_c).abs().max()),
        "mine_recurrent_vs_mine_chunk_state": float((S_r - S_c).abs().max()),
        "mine_recurrent_vs_hf_out": float((out_r - out_hr).abs().max()),
        "mine_recurrent_vs_hf_state": float((S_r - S_hr).abs().max()),
        "mine_chunk_vs_hf_out": float((out_c - out_hc).abs().max()),
        "mine_chunk_vs_hf_state": float((S_c - S_hc).abs().max()),
    }
    return parity


# ---------------- scaling（MPS 计时口径） ----------------
def full_attention_ms(q, k, v):
    """满注意力（计时对照）：q/k/v (B,h,n,d) → (B,h,n,d)——因果 softmax。"""
    n = q.shape[2]
    att = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    att = att.masked_fill(~torch.ones(n, n, dtype=torch.bool, device=q.device).tril(), float("-inf"))
    return F.softmax(att.float(), dim=-1).to(q.dtype) @ v


def scaling_mps(ns=(512, 1024, 2048, 4096, 8192), B=2, H=4, D=64, chunk=64, seed=SEED):
    """MPS wall-clock：chunkwise vs 满注意力，log-log 斜率（各 10 次取中位）。"""
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    scaling = {}
    for n in ns:
        torch.manual_seed(seed + n)
        qm = torch.randn(B, H, n, D, device=device)
        km = torch.randn(B, H, n, D, device=device)
        vm = torch.randn(B, H, n, D, device=device)
        # 满注意力用(B,H,n,D)，GDN接口用(B,n,H,D)，不可把H误当序列轴。
        qg, kg, vg = (x.transpose(1, 2).contiguous() for x in (qm, km, vm))
        gm = -F.softplus(torch.randn(B, n, H, device=device) * 1.5)
        bm = torch.randn(B, n, H, device=device).sigmoid()
        with torch.no_grad():
            for _ in range(3):                                                   # 预热
                gdn_chunkwise(qg, kg, vg, gm, bm, chunk=chunk)
                full_attention_ms(qm, km, vm)
            if device.type == "mps":
                torch.mps.synchronize()
            t = {}
            for tag, fn in (("chunk_gdn", lambda: gdn_chunkwise(qg, kg, vg, gm, bm, chunk=chunk)),
                            ("full_attn", lambda: full_attention_ms(qm, km, vm))):
                ts = []
                for _ in range(10):
                    t0 = time.perf_counter()
                    fn()
                    if device.type == "mps":
                        torch.mps.synchronize()
                    ts.append(time.perf_counter() - t0)
                t[tag] = float(np.median(ts))
        scaling[n] = t
        print(f"  n={n}: chunk {t['chunk_gdn']*1000:.1f} ms | full {t['full_attn']*1000:.1f} ms")
    nsa = np.array(sorted(scaling), dtype=float)
    slope_chunk = float(np.polyfit(np.log(nsa), np.log([scaling[int(n)]["chunk_gdn"] for n in nsa]), 1)[0])
    slope_full = float(np.polyfit(np.log(nsa), np.log([scaling[int(n)]["full_attn"] for n in nsa]), 1)[0])
    return {"device": str(device), "B": B, "H": H, "D": D, "chunk": chunk,
            "times": {str(k): v for k, v in scaling.items()},
            "slopes": {"chunk_gdn": slope_chunk, "full_attn": slope_full}}


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="Book5 ch5 GDN 正式件：双形态对拍 + scaling")
    ap.add_argument("--out-name", default="run1")
    ap.add_argument("--scaling", action="store_true", help="显式启用长度扫描；默认只做数值验证")
    ap.add_argument("--skip-scaling", action="store_true", help="兼容旧命令：禁用长度扫描")
    args = ap.parse_args()
    torch.set_num_threads(2)

    results = {"seed": SEED, "hf_reference": "transformers " + __import__("transformers").__version__ + " qwen3_next torch 双 kernel",
               "parity_cpu_fp32": parity_hf()}
    for kk, vv in results["parity_cpu_fp32"].items():
        print(f"[对拍] {kk}: max|Δ| = {vv:.2e}")
    assert results["parity_cpu_fp32"]["mine_recurrent_vs_hf_out"] < 1e-4, "自研 recurrent 与 HF 不一致"
    assert results["parity_cpu_fp32"]["mine_chunk_vs_hf_out"] < 1e-4, "自研 chunkwise 与 HF 不一致"
    assert results["parity_cpu_fp32"]["mine_recurrent_vs_mine_chunk_out"] < 1e-4, "双形态互拍不一致"

    assert all(v < 1e-4 for v in results["parity_cpu_fp32"].values()), "输出或末状态对拍失败"

    # 层模块双形态自洽快检（教学骨架 path 切换的两条路给同一答案）
    torch.manual_seed(SEED)
    layer = GatedDeltaNet(128, 2, 2, 32, 32, chunk=32)
    x = torch.randn(2, 96, 128)
    with torch.no_grad():
        y_c = layer(x, path="chunk")
        y_r = layer(x, path="recurrent")
    results["layer_dual_path_max_diff"] = float((y_c - y_r).abs().max().detach())
    print(f"[层模块] path=chunk vs recurrent max|Δ| = {results['layer_dual_path_max_diff']:.2e}")
    assert results["layer_dual_path_max_diff"] < 1e-4, "层模块双形态不一致"

    if args.scaling and not args.skip_scaling:
        print("[scaling] 自动选择可用MPS或CPU：chunkwise vs 满注意力，n=512..8192（B=2/H=4/D=64/chunk=64）")
        results["scaling"] = scaling_mps()
        sc = results["scaling"]["slopes"]
        print(f"[scaling] log-log 斜率：chunk {sc['chunk_gdn']:.2f}（近线性≈1）/ "
              f"full {sc['full_attn']:.2f}（二次≈2）")
        results["scaling_note"] = ("「线性 O(n)」口径：训练并行形态（chunk 间线性扫描 + 块内二次）；"
                                   "推理递归形态每步 O(Dk·Dv) 与 n 无关——两形态分列（纪律⑨）。"
                                   "本版已修复GDN序列/头轴；不得沿用旧入口曲线作为复杂度证据。")
    save_json("gdn", results, args.out_name)
    print("ch05 gdn.py 完成。")


if __name__ == "__main__":
    main()
