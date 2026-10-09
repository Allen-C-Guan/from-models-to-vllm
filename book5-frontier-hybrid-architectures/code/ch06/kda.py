# kda.py —— Book5 ch6 正式件：KimiDeltaAttention（KDA——K3 注意力侧正身件）
# 用途：ch6 6.2/6.7「KDA 正身 + 玩具替身」的组件正身——把 ch5 的 gated delta rule 的
#   per-token 标量遗忘门换成 K3 的 per-channel 通道遗忘门（K3 报告 §2.1.1 Eq.1-6）：
#   ① 递归形态 kda_recurrent（Eq.1 三步读法：遗忘→对齐检索→写差值——机制正身）；
#   ② chunkwise 形态 kda_chunkwise（Eq.3-4：通道累积衰减 γ 的 UT 变换 + 块内并行 + 块间递推）；
#   ③ KimiDeltaAttention(cfg) 层模块：K3 化三件——per-channel 门（Eq.5 下界衰减
#      g = g_min·Sigmoid(e^{A^h}·z)，g_min=−5）、低秩遗忘门投影（z = W↑W↓x + b，rank=d_k）、
#      满秩输出门（Eq.6：y = W_o[Sigmoid(W_g x) ⊙ RMSNorm(õ)]——玩具 512×512=d 同构缩微，
#      真机 7168×7168，两口径勿混）；
#   ④ 对拍内置：自研双形态 vs transformers 5.18.0 kimi_linear 的 torch 参考 kernel
#      （chunk_kimi_delta_attention / recurrent_kimi_delta_attention——「KDA essentially the same
#      as GDN but decay is per-channel」官方差分句的实现正身）；双形态互拍；
#   ⑤ gate_mode 开关："per_channel"（K3 正身）| "per_token"（GDN 粒度替身臂——z 沿通道取均值
#      后广播，每头每步一个标量；**单变量 ablation**：与 per_channel 臂同参数同结构，只有
#      门的读出粒度不同——门粒度实验的判据预注册见 notes/07）。
# 【KDA vs GDN 的公式差异（全部凭 K3 报告 §2.1.1 三句 + HF docstring，papers/04 §3.5）】
#   D1 遗忘门粒度：GDN 每头每步一个标量 g_t（S ← S·e^{g_t} 整体打折）；KDA 每头每步 d_k 维
#      通道向量 g_t ∈ (g_min,0)^{d_k}（S ← Diag(e^{g_t})·S 逐键通道打折——128 个抽屉各自折扣率）。
#   D2 衰减参数化：Kimi Linear −e^{A}Softplus(z) 无下界；K3 g_min·Sigmoid(e^{A}z) 下界 −5
#      （α > e^{−5}≈6.7e−3 → 16-token 块累积 log 衰减 ∈ (−80,0) → 倒数重标 < e^{80} 在 BF16
#      动态域内——机制改动直接改变 kernel 形态：对角块并入 Tensor Core 稠密乘，K3 报告 Fig.3b）。
#   D3 输出门：GDN 低秩 z 门（in_proj 出）；KDA 满秩 Sigmoid(W_g x)（d×d——K3 改动二）。
#   D4 共同基座：delta rule 递归（先忘后写 + 写残差）、短卷积+Swish、q/k L2Norm——同族不变。
# 【实现口径（正文 6.8 素材）】
#   - 本件 chunkwise 用「逐对衰减」D[i,j,c]=exp(c_i[c]−c_j[c]) 直接计算（i≥j 时 ∈(0,1]，
#     永不溢出——数学上的安全形态）；K3 的 g_min=−5 是为「因式分解形态」（k̃=k⊙e^{−c} 重标，
#     e^{−c} 可爆）买回 BF16 动态域——同一数学的两种工程立场，正文对照教学点。
#   - chunk 默认 16：与 K3 报告的 16-token 数字链对齐（若走因式分解形态，16 是 g_min=−5 下
#     不溢出的块长；本件逐对形态任意 chunk 都安全，16 取报告口径）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch06/kda.py" [--out-name run1]
#   对拍 CPU fp32；预计 <1 分钟（秒级）。
# 产物：log/book5-ch06/kda_{out}.json（不入库）
import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book5-ch06")
SEED = 20261002


def save_json(name, obj, out_name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}_{out_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[产物] {path}")
    return path


def l2norm(x, dim=-1, eps=1e-6):
    """FLA 口径 l2norm：x / sqrt(Σx² + eps)（kimi_linear 参考实现同款）。"""
    return x / torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)


# ---------------- 形态一：recurrent（逐 token 递归——推理形态，Eq.1 三步读法） ----------------
def kda_recurrent(q, k, v, g, beta, initial_state=None, output_final_state=False):
    """KDA 递归形态。g 布局 (B,S,H,Dk)——per-channel log 衰减（与 GDN 的 (B,S,H) 标量门唯一差异）。

    每步（先忘后写——通道版）：
      S ← Diag(e^{g_t})·S                        # (B,H,Dk,Dv) 逐键通道衰减（遗忘）
      kv_mem = Sᵀk_t                              # 对齐检索：旧状态沿当前键读出
      delta = (v_t − kv_mem)·β_t                  # 写差值：新值减旧记忆
      S ← S + k_t ⊗ delta                         # 秩一写入
      o_t = Sᵀ q_t                                # 读出（读的是写入后的状态——块形态对角线保留的来源）
    q/k 先 l2norm，q 再乘 Dk^{-1/2}。
    """
    B, S_len, H, Dk = k.shape
    Dv = v.shape[-1]
    q = l2norm(q.float()).transpose(1, 2) * (Dk ** -0.5)                          # (B,H,S,Dk)
    k = l2norm(k.float()).transpose(1, 2)                                        # (B,H,S,Dk)
    v = v.float().transpose(1, 2)                                                # (B,H,S,Dv)
    beta = beta.float().transpose(1, 2)                                          # (B,H,S)
    g = g.float().transpose(1, 2)                                                # (B,H,S,Dk)
    S = (torch.zeros(B, H, Dk, Dv, device=q.device) if initial_state is None
         else initial_state.float().to(q.device))
    out = torch.empty(B, H, q.shape[2], Dv, device=q.device)
    for t in range(q.shape[2]):
        q_t, k_t, v_t = q[:, :, t], k[:, :, t], v[:, :, t]                       # (B,H,Dk)/(B,H,Dk)/(B,H,Dv)
        S = S * g[:, :, t].exp()[..., None]                                      # (B,H,Dk,Dv)⊙(B,H,Dk,1) 逐通道遗忘
        kv_mem = torch.einsum("bhkv,bhk->bhv", S, k_t)                           # (B,H,Dv) 对齐检索
        delta = (v_t - kv_mem) * beta[:, :, t][..., None]                        # (B,H,Dv) 写差值
        S = S + torch.einsum("bhk,bhv->bhkv", k_t, delta)                        # 秩一写入
        out[:, :, t] = torch.einsum("bhkv,bhk->bhv", S, q_t)                     # (B,H,Dv) 读出
    out = out.transpose(1, 2).contiguous()                                       # (B,S,H,Dv)
    return out, (S if output_final_state else None)


# ---------------- 形态二：chunkwise（块并行——训练形态，Eq.3-4） ----------------
def kda_chunkwise(q, k, v, g, beta, chunk=16, initial_state=None, output_final_state=False):
    """KDA 块并行形态：通道累积衰减 c=Σg（Eq.3 的 log 空间版）+ UT 三角求解 + 块间递推。

    输入布局 (B,S,H,D)：q/k 为 Dk、v 为 Dv、g 为 Dk（per-channel）、beta (B,S,H)。
    块内（逐对衰减形态——见文件头「实现口径」）：
      D[i,j,c] = e^{c_i[c]−c_j[c]}（j≤i；j>i 位掩 0）
      M[i,j]   = β_i·Σ_c k_i[c]k_j[c]D[i,j,c]            # UT 系统矩阵（严格下三角）
      u        = (I+M)^{-1}(βv)、k̃ = (I+M)^{-1}(βk⊙e^c)  # 三角求解
      intra[i,j] = Σ_c q_i[c]k_j[c]D[i,j,c]（j≤i，含对角——读的是写入后状态）
    块间：v_new = u − k̃S；out = (q⊙e^c)S + intra·v_new；S ← Diag(e^{c_last})S + (k⊙e^{c_last−c})ᵀv_new。
    """
    B, S_len, H, Dk = k.shape
    Dv = v.shape[-1]
    q = l2norm(q.float()).transpose(1, 2) * (Dk ** -0.5)                         # (B,H,S,Dk)
    k = l2norm(k.float()).transpose(1, 2)                                        # (B,H,S,Dk)
    v = v.float().transpose(1, 2)                                                # (B,H,S,Dv)
    beta = beta.float().transpose(1, 2)                                          # (B,H,S)
    g = g.float().transpose(1, 2)                                                # (B,H,S,Dk)

    pad = (chunk - S_len % chunk) % chunk
    if pad:
        q, k, v = F.pad(q, (0, 0, 0, pad)), F.pad(k, (0, 0, 0, pad)), F.pad(v, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))                                             # pad 位 β=0 → 不写入
        g = F.pad(g, (0, 0, 0, pad))                                             # pad 位 g=0 → 不衰减
    NC = (S_len + pad) // chunk

    c = g.view(B, H, NC, chunk, Dk).cumsum(dim=-2)                               # (B,H,w,i,c) 通道累积 log 衰减
    beta_c = beta.view(B, H, NC, chunk)                                          # (B,H,w,i)
    qb = q.view(B, H, NC, chunk, Dk)                                             # (B,H,w,i,d)
    kb = k.view(B, H, NC, chunk, Dk)                                             # (B,H,w,j,d)
    vb = v.view(B, H, NC, chunk, Dv)                                             # (B,H,w,j,e)
    k_beta = (k * beta[..., None]).view(B, H, NC, chunk, Dk)                     # (B,H,w,i,d) βk
    v_beta = (v * beta[..., None]).view(B, H, NC, chunk, Dv)                     # (B,H,w,i,e) βv

    # 逐对衰减 D（j≤i 位 (0,1]，永不溢出）；严格上三角位（j>i）先填 −inf 再 exp → 0
    #   （坑：先填 0 再 exp 会把掩蔽位变成 e⁰=1——被对拍抓出的实现错误，正文「对拍驱动开发」素材）
    iu = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).triu(1)     # (i,j) j>i 为 True
    D = (c.unsqueeze(-2) - c.unsqueeze(-3)).masked_fill(iu[..., None], float("-inf")).exp()  # (B,H,w,i,j,c)

    # UT 系统：M = tril(β_i·Σ_c k_i k_j D, −1)——对角元 β_iΣ_c k_i[c]²·1 必须再去掉（单位对角补偿）
    M = torch.einsum("bhwid,bhwjd,bhwijd->bhwij", k_beta, kb, D)                 # (B,H,w,i,j)
    M = M - torch.diag_embed(M.diagonal(dim1=-2, dim2=-1))                       # 严格下三角
    eye = torch.eye(chunk, device=q.device).expand(B, H, NC, chunk, chunk).contiguous()

    u = torch.linalg.solve_triangular(eye + M, v_beta, upper=False, unitriangular=True)   # (B,H,w,i,e)
    k_tilde = torch.linalg.solve_triangular(eye + M, k_beta * c.exp(),                     # (B,H,w,i,d)
                                            upper=False, unitriangular=True)
    intra = torch.einsum("bhwid,bhwjd,bhwijd->bhwij", qb, kb, D)                 # (B,H,w,i,j) 含对角
    q_dec = qb * c.exp()                                                         # (B,H,w,i,d) q⊙e^{c}
    k_cross = kb * (c[..., -1:, :] - c).exp()                                    # (B,H,w,j,d) k⊙e^{c_last−c}
    c_last = c[..., -1, :].exp()                                                 # (B,H,w,c) e^{c_last}（先 exp 再乘——漏 exp 会被对拍抓出）

    S = (torch.zeros(B, H, Dk, Dv, device=q.device) if initial_state is None
         else initial_state.float().to(q.device))
    out = torch.empty(B, H, NC, chunk, Dv, device=q.device)
    for w in range(NC):                                                          # 块间线性扫描
        v_new = u[:, :, w] - k_tilde[:, :, w] @ S                                # (B,H,i,e) 真差值 δ
        out[:, :, w] = (q_dec[:, :, w] @ S                                       # (B,H,i,e) 跨块读老状态
                        + intra[:, :, w] @ v_new)                                # 块内注意（含对角）
        S = S * c_last[:, :, w][..., None] \
            + k_cross[:, :, w].transpose(-1, -2) @ v_new                         # (B,H,d,e) Diag(e^{c_last})·S+写
    out = out.reshape(B, H, NC * chunk, Dv)[:, :, :S_len].transpose(1, 2).contiguous()
    return out, (S if output_final_state else None)


# ---------------- 层模块：K3 化 KDA 正身件 ----------------
class KimiDeltaAttention(nn.Module):
    """KDA 注意力层（K3 口径：per-channel 下界遗忘门 + 满秩输出门）。

    结构（参数账与 K3 报告 Table 1 侧算同构——papers/04 §2 config 逆向）：
      q/k/v_proj:    d → h·d_k 各一（K3 账「q/k/v 投影 3×d×(h·d_k)」）
      conv1d:        因果短卷积 (k_conv=4, groups=3h·d_k, silu)——作用于拼接的 qkv
      f_a/f_b_proj:  低秩遗忘门投影 z = W↑(W↓x)（rank=d_k——KimiLinearForgetGate 口径）
      dt_bias:       (h·d_k,) 逐通道偏置（Eq.2 的 b^h_α）
      A_log:         (h,) 每头 log-scale（Eq.5 的 e^{A^h}；初始化 0——A_log=0）
      b_proj:        d → h（β 写强度，sigmoid）
      o_norm:        head-wise RMSNorm（weight (d_v,) 头间共享——KimiLinearRMSNormGated 去门版）
      g_proj(W_g):   d → d 满秩输出门（Eq.6 Sigmoid(W_g x)；玩具 d=h·d_v 时维度自洽——
                     真机 7168×7168 与 h·d_v=12288 的广播口径报告未明，notes/07 登记）
      o_proj:        h·d_v → d
    gate_mode="per_token"：z 沿 d_k 取均值后广播（GDN 粒度——每头每步一个标量）。
    forward(x, cos, sin, path) 兼容 SlotDecoderLayer 插槽签名（线性层 NoPE，忽略 cos/sin）。
    """

    def __init__(self, d, h, d_k, d_v, conv_kernel=4, chunk=16, gate_mode="per_channel",
                 g_min=-5.0, rank=None):
        super().__init__()
        assert gate_mode in ("per_channel", "per_token")
        assert h * d_v == d, ("满秩输出门维度自洽条件：h·d_v == d（玩具口径 d=512=h·d_v——"
                              "真机 7168 vs 12288 的广播口径报告未明，见 notes/07 登记项）")
        self.h, self.d_k, self.d_v = h, d_k, d_v
        self.chunk, self.gate_mode, self.g_min = chunk, gate_mode, g_min
        self.qkv_dim = h * d_k
        self.conv_dim = 3 * self.qkv_dim
        rank = rank or d_k                                                        # 低秩秩数 = head_dim（KimiLinear 口径）
        self.q_proj = nn.Linear(d, self.qkv_dim, bias=False)
        self.k_proj = nn.Linear(d, self.qkv_dim, bias=False)
        self.v_proj = nn.Linear(d, self.qkv_dim, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, conv_kernel,
                                groups=self.conv_dim, padding=conv_kernel - 1, bias=False)
        self.f_a_proj = nn.Linear(d, rank, bias=False)                            # 遗忘门低秩 ↓
        self.f_b_proj = nn.Linear(rank, self.qkv_dim, bias=False)                 # 遗忘门低秩 ↑
        self.dt_bias = nn.Parameter(torch.zeros(self.qkv_dim))
        self.A_log = nn.Parameter(torch.zeros(h))                                 # e^{A^h} 初始 1（K3 口径）
        self.b_proj = nn.Linear(d, h, bias=False)
        self.o_norm_w = nn.Parameter(torch.ones(d_v))                             # head-wise RMSNorm 权重
        self.g_proj = nn.Linear(d, d, bias=False)                                 # 满秩输出门 W_g
        self.o_proj = nn.Linear(h * d_v, d, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forget_gate(self, x):
        """g = g_min·Sigmoid(e^{A^h}·z) ∈ (g_min, 0)——per-channel（Eq.5）。返回 (B,S,h,d_k)。"""
        B, n, _ = x.shape
        z = self.f_b_proj(self.f_a_proj(x)) + self.dt_bias                        # (B,n,h·d_k)
        z = z.float().view(B, n, self.h, self.d_k)
        if self.gate_mode == "per_token":                                         # GDN 粒度臂：通道均值读出
            z = z.mean(dim=-1, keepdim=True).expand(-1, -1, -1, self.d_k)
        a = self.A_log.float().exp().view(1, 1, self.h, 1)                        # e^{A^h} 每头一个
        return self.g_min * torch.sigmoid(a * z)                                  # (B,n,h,d_k) ∈ (−5,0)

    def forward(self, x, cos=None, sin=None, path="chunk"):
        B, n, d = x.shape
        qkv = torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)  # (B,n,3h·d_k)
        qkv = qkv.transpose(1, 2)                                                 # (B,3h·d_k,n)
        qkv = F.silu(self.conv1d(qkv)[..., :n])                                   # 因果短卷积 + silu
        q, k, v = qkv.split([self.qkv_dim] * 3, dim=1)
        q = q.transpose(1, 2).view(B, n, self.h, self.d_k)                        # (B,n,h,d_k)
        k = k.transpose(1, 2).view(B, n, self.h, self.d_k)
        v = v.transpose(1, 2).view(B, n, self.h, self.d_v)                        # (B,n,h,d_v)
        beta = torch.sigmoid(self.b_proj(x))                                      # (B,n,h)
        g = self.forget_gate(x)                                                   # (B,n,h,d_k)
        if path == "chunk":
            core, _ = kda_chunkwise(q, k, v, g, beta, chunk=self.chunk)           # (B,n,h,d_v)
        elif path == "recurrent":
            core, _ = kda_recurrent(q, k, v, g, beta)
        else:
            raise ValueError(path)
        # head-wise RMSNorm（weight (d_v,) 头间共享）→ 满秩输出门 Sigmoid(W_g x)⊙·（Eq.6）
        var = core.float().pow(2).mean(-1, keepdim=True)
        core = (core.float() * torch.rsqrt(var + 1e-6)) * self.o_norm_w.float().view(1, 1, 1, self.d_v)
        gate = torch.sigmoid(self.g_proj(x))                                      # (B,n,d) 满秩门
        core = core * gate.view(B, n, self.h, self.d_v)                           # 玩具 d=h·d_v 自洽
        return self.o_proj(core.reshape(B, n, self.h * self.d_v))                 # (B,n,d)


# ---------------- 对拍（内置；vs HF kimi_linear torch 双 kernel + 双形态互拍） ----------------
def parity_hf(B=2, S=256, H=2, Dk=32, Dv=32, seed=SEED, chunk=16):
    """自研 KDA 双形态 vs HF kimi_linear 参考实现（K3 式 g 直接喂给 HF kernel——kernel 只吃
    log 衰减 g，参数化在模块侧，对拍正合「同一递归、不同门」的教学点）。"""
    from transformers.models.kimi_linear.modeling_kimi_linear import (
        chunk_kimi_delta_attention, recurrent_kimi_delta_attention)
    torch.manual_seed(seed)
    q = torch.randn(B, S, H, Dk)
    k = torch.randn(B, S, H, Dk)
    v = torch.randn(B, S, H, Dv)
    beta = torch.randn(B, S, H).sigmoid()
    g = -5.0 * torch.sigmoid(torch.randn(B, S, H, Dk) * 1.5)                      # K3 式 (−5,0) per-channel
    S0 = torch.randn(B, H, Dk, Dv) * 0.05

    out_r, S_r = kda_recurrent(q, k, v, g, beta, initial_state=S0, output_final_state=True)
    out_c, S_c = kda_chunkwise(q, k, v, g, beta, chunk=chunk, initial_state=S0,
                               output_final_state=True)
    out_hr, S_hr = recurrent_kimi_delta_attention(q, k, v, g, beta, initial_state=S0,
                                                  output_final_state=True,
                                                  use_qk_l2norm_in_kernel=True)
    out_hc, S_hc = chunk_kimi_delta_attention(q, k, v, g, beta, chunk_size=chunk,
                                              initial_state=S0, output_final_state=True,
                                              use_qk_l2norm_in_kernel=True)
    parity = {
        "mine_recurrent_vs_mine_chunk_out": float((out_r - out_c).abs().max()),
        "mine_recurrent_vs_mine_chunk_state": float((S_r - S_c).abs().max()),
        "mine_recurrent_vs_hf_out": float((out_r - out_hr).abs().max()),
        "mine_recurrent_vs_hf_state": float((S_r - S_hr).abs().max()),
        "mine_chunk_vs_hf_out": float((out_c - out_hc).abs().max()),
        "mine_chunk_vs_hf_state": float((S_c - S_hc).abs().max()),
    }
    return parity


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="Book5 ch6 KDA 正式件：双形态 + kimi_linear 对拍")
    ap.add_argument("--out-name", default="run1")
    args = ap.parse_args()

    results = {"seed": SEED,
               "hf_reference": "transformers 5.18.0 kimi_linear torch 双 kernel",
               "parity_cpu_fp32": parity_hf()}
    for kk, vv in results["parity_cpu_fp32"].items():
        print(f"[对拍] {kk}: max|Δ| = {vv:.2e}")
    for key in ("mine_recurrent_vs_hf_out", "mine_chunk_vs_hf_out",
                "mine_recurrent_vs_mine_chunk_out"):
        assert results["parity_cpu_fp32"][key] < 1e-4, f"{key} 对拍失败"

    # 层模块双形态自洽 + 两 gate_mode 同构检查（单变量臂：参数完全相同；d=h·d_v=64 自洽口径）
    torch.manual_seed(SEED)
    layer = KimiDeltaAttention(64, 2, 32, 32, chunk=16)
    x = torch.randn(2, 96, 64)
    with torch.no_grad():
        y_c = layer(x, path="chunk")
        y_r = layer(x, path="recurrent")
    results["layer_dual_path_max_diff"] = float((y_c - y_r).abs().max())
    torch.manual_seed(SEED)
    layer_pt = KimiDeltaAttention(64, 2, 32, 32, chunk=16, gate_mode="per_token")
    layer_pt.load_state_dict(layer.state_dict())                                  # 同权重直搬——单变量
    with torch.no_grad():
        y_pt = layer_pt(x, path="chunk")
    results["gate_mode_param_identical"] = (sum(p.numel() for p in layer.parameters())
                                            == sum(p.numel() for p in layer_pt.parameters()))
    results["per_token_vs_per_channel_max_diff"] = float((y_pt - y_c).abs().max())  # 应显著非零（门粒度不同）
    results["kda_layer_params_128d_2h_32dim"] = sum(p.numel() for p in layer.parameters())
    print(f"[层模块] chunk vs recurrent max|Δ| = {results['layer_dual_path_max_diff']:.2e} | "
          f"per_token vs per_channel max|Δ| = {results['per_token_vs_per_channel_max_diff']:.2e}（非零=粒度生效） | "
          f"参数同构 = {results['gate_mode_param_identical']}")
    assert results["layer_dual_path_max_diff"] < 1e-4, "层模块双形态不一致"
    assert results["gate_mode_param_identical"], "两 gate_mode 参数应完全相同（单变量臂）"
    assert results["per_token_vs_per_channel_max_diff"] > 1e-4, "门粒度未生效"

    save_json("kda", results, args.out_name)
    print("ch06 kda.py 完成。")


if __name__ == "__main__":
    main()
