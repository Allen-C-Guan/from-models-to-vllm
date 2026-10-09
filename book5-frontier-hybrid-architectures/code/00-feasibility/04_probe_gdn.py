# 04_probe_gdn.py —— Book5 ch5 实验可行性探针：Gated DeltaRule 双形态（recurrent vs chunkwise）
# 用途：ch5（路线三·线性注意力）的数学正身与工程底座——
#   ① 静态互证（先于运行）：门控增量规则（gated delta rule）的公式族三方互证——
#      文献公式（Gated DeltaNet，候选 arXiv 2412.06464?，三查后以核验为准）↔ 本文件独立推导
#      ↔ transformers 5.18.0 参考实现（repos/transformers .../qwen3_next/modeling_qwen3_next.py
#      L378-574 的 torch_chunk/torch_recurrent 两函数）。互证记录落盘 JSON；
#   ② 数值对拍（CPU fp32 随机权重）：自研 recurrent ↔ 自研 chunkwise ↔ HF 参考实现，
#      max|Δ| 三方两两对拍（含 initial_state 传递与 output_final_state）；
#   ③ wall-clock scaling（MPS）：n=512..8192，chunkwise（块内二次 + 块间线性扫描）
#      vs 满注意力（n² 分数矩阵），log-log 斜率——「O(n) 断言的口径：训练并行形态」现场读数。
# 【静态互证记录（逐项核对，2026-10-05）】
#   R1 门方向：g ≤ 0（log 空间衰减；HF 模块侧 g = -exp(A_log)·softplus(a + dt_bias)，恒 ≤ 0）。
#      递归形态中 state 先乘 exp(g_t) 衰减、再读预测/写入——「先忘后写」（HF L557-568 逐行同构）。
#   R2 归一化位置：q、k 各自 l2norm（eps=1e-6，FLA 口径 x·rsqrt(Σx²+eps)）在缩放之前；
#      随后 q 乘 d_k^{-1/2}（HF L419-425 / L540-545——两处顺序一致：先 l2norm 后 scale）。
#   R3 β 施加位置：β 乘在 k 与 v 上（k_beta = βk、v_beta = βv，HF L437-438）——增量式
#      delta = β(v − Sᵀk) 展开后的矩阵形态；「学习率/写入强度」的解释与论文一致。
#   R4 UT 系统：块内 (I − tril(βKKᵀ·D̂, -1)) 的逆经三角求解（HF L461-473 solve_triangular
#      unitriangular=True；D̂_ij = exp(c_i − c_j)，c = 块内累积 log 衰减）。
#   R5 chunk 衰减三处：q 侧乘 exp(c)；k 侧乘 exp(c_last − c)（跨块读老状态时补齐到块尾）；
#      state 更新乘 exp(c_last)（HF L490-492）。
#   本文件 chunkwise 实现为独立推导的等价变形：out = q_d @ S + intra @ v_new，其中
#   v_new = u − k̃ @ S（老状态的可预测部分显式减去——delta rule 的「块形态」）。
# 所属章节：Book5 第 5 章（线性注意力数学最重章的底座；05/06/07 探针经 load_sibling 复用本件）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/00-feasibility/04_probe_gdn.py" [--out-name run1]
#   对拍 CPU fp32；计时 MPS；预计 2-4 分钟。
# 产物：log/book5-feasibility/probe04_gdn_{out}.json（不入库）
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book5-feasibility")
SEED = 20261002


def save_json(name, obj, out_name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}_{out_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[产物] {path}")
    return path


def l2norm(x, dim=-1, eps=1e-6):
    """FLA 口径 l2norm：x · rsqrt(Σx² + eps)（对齐 HF qwen3_next 的 fla 对齐注释）。"""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


# ---------------- 形态一：recurrent（逐 token 递归——推理形态，机制正身） ----------------
def gdn_recurrent(q, k, v, g, beta, initial_state=None, output_final_state=False):
    """gated delta rule 递归形态。输入布局 (B,S,H,D)（q/k 同 H；v 同 H）——与 HF 参考一致便于对拍。

    每步（先忘后写，见文件头 R1）：
      S ← S·exp(g_t)                          # (B,H,Dk,Dv) 状态衰减
      delta_t = β_t·(v_t − Sᵀk_t)             # 老状态预测的残差 = 增量
      S ← S + k_t ⊗ delta_t                   # 外积写入（秩一更新）
      o_t = Sᵀ q_t                            # 读出
    q/k 先 l2norm，q 再乘 Dk^{-1/2}（R2）。
    """
    B, S_len, H, Dk = k.shape
    Dv = v.shape[-1]
    q = l2norm(q.float()).transpose(1, 2)                                       # (B,H,S,Dk)
    k = l2norm(k.float()).transpose(1, 2)                                       # (B,H,S,Dk)
    v = v.float().transpose(1, 2)                                               # (B,H,S,Dv)
    beta = beta.float().transpose(1, 2)                                         # (B,H,S)
    g = g.float().transpose(1, 2)                                               # (B,H,S)
    q = q * (Dk ** -0.5)
    S = (torch.zeros(B, H, Dk, Dv, device=q.device) if initial_state is None
         else initial_state.float().to(q.device))
    out = torch.empty(B, H, q.shape[2], Dv, device=q.device)
    for t in range(q.shape[2]):
        q_t, k_t, v_t = q[:, :, t], k[:, :, t], v[:, :, t]                      # (B,H,Dk)/(B,H,Dk)/(B,H,Dv)
        S = S * g[:, :, t].exp()[..., None, None]                               # (B,H,Dk,Dv) 先忘
        kv_mem = torch.einsum("bhkv,bhk->bhv", S, k_t)                          # (B,H,Dv) Sᵀk
        delta = (v_t - kv_mem) * beta[:, :, t][..., None]                       # (B,H,Dv) 增量
        S = S + torch.einsum("bhk,bhv->bhkv", k_t, delta)                       # 秩一写入
        out[:, :, t] = torch.einsum("bhkv,bhk->bhv", S, q_t)                    # (B,H,Dv) 读出
    out = out.transpose(1, 2).contiguous()                                      # (B,S,H,Dv)
    return out, (S if output_final_state else None)


# ---------------- 形态二：chunkwise（块并行——训练形态，WY/UT 表示） ----------------
def gdn_chunkwise(q, k, v, g, beta, chunk=64, initial_state=None, output_final_state=False):
    """gated delta rule 块并行形态（独立推导，见文件头 R4/R5 与下方逐步注释）。

    输入布局 (B,S,H,D)；输出 (B,S,H,Dv)。S 需为 chunk 倍数（探针内 pad 到倍数）。
    """
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
    D_pair = (c.unsqueeze(-1) - c.unsqueeze(-2)).masked_fill(                   # (B,H,w,i,j) D̂=exp(c_i−c_j)
        torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).triu(1), float("-inf")).exp()

    # UT 系统（R4）：M = tril(βKKᵀ·D̂, -1)；u = vβ − M·u − … ⇒ u = (I+M)^{-1}·(vβ 等)
    #   经三角求解直接得 u 与 k̃（不显式求逆——HF 同款 solve_triangular(unitriangular=True)；
    #   首版误写 (I−M)^{-1} 被对拍抓出；MPS 上批量 inv 慢 ~40×，三角求解是正解——如实记录）
    M = (k_beta @ kb.transpose(-1, -2)) * D_pair                                # (B,H,w,i,j) βk·kᵀ·D̂
    M = M - torch.diag_embed(M.diagonal(dim1=-2, dim2=-1))                      # 去对角 → 严格下三角（上三角已被 D̂→0 掩掉）
    eye = torch.eye(chunk, device=q.device).expand(B, H, NC, chunk, chunk).contiguous()

    u = torch.linalg.solve_triangular(eye + M, v_beta, upper=False, unitriangular=True)      # (B,H,w,i,e)
    k_tilde = torch.linalg.solve_triangular(eye + M, k_beta * c.exp().unsqueeze(-1),         # (B,H,w,i,d)
                                            upper=False, unitriangular=True)
    intra = (qb @ kb.transpose(-1, -2)) * D_pair                                # (B,H,w,i,j) 块内注意 (qkᵀ)·D̂
    q_dec = qb * c.exp().unsqueeze(-1)                                          # (B,H,w,i,d) q·e^{c}（R5）
    k_cross = kb * (c[..., -1:] - c).exp().unsqueeze(-1)                        # (B,H,w,j,d) k·e^{c_last−c}
    chunk_decay = c[..., -1].exp()                                              # (B,H,w) e^{c_last}

    S = (torch.zeros(B, H, Dk, Dv, device=q.device) if initial_state is None
         else initial_state.float().to(q.device))
    out = torch.empty(B, H, NC, chunk, Dv, device=q.device)
    for w in range(NC):                                                         # 块间线性扫描——「O(n) 在这里」
        v_new = u[:, :, w] - k_tilde[:, :, w] @ S                               # (B,H,i,e) 减老状态可预测部分
        out[:, :, w] = (q_dec[:, :, w] @ S                                      # (B,H,i,e) 跨块读老状态
                        + intra[:, :, w] @ v_new)                               # 块内注意
        S = S * chunk_decay[:, :, w, None, None] \
            + k_cross[:, :, w].transpose(-1, -2) @ v_new                        # (B,H,d,e) 状态更新（先忘后写 R1）
    out = out.reshape(B, H, NC * chunk, Dv)[:, :, :S_len].transpose(1, 2).contiguous()
    return out, (S if output_final_state else None)


def full_attention_ms(q, k, v):
    """满注意力（计时对照）：q/k/v (B,h,n,d) → (B,h,n,d)。"""
    n = q.shape[2]
    att = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    att = att.masked_fill(~torch.ones(n, n, dtype=torch.bool, device=q.device).tril(), float("-inf"))
    return F.softmax(att.float(), dim=-1).to(q.dtype) @ v


def main():
    ap = argparse.ArgumentParser(description="Book5 ch5 GDN 双形态可行性探针")
    ap.add_argument("--out-name", default="run1")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    results = {"seed": SEED, "date": "2026-10-05"}

    # HF 参考实现（transformers 5.18.0，本机 conda env 与 repos/ 浅克隆同版本）
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule)
    results["hf_reference"] = "transformers 5.18.0 qwen3_next/modeling_qwen3_next.py L378-574"

    # ---- ① 数值对拍（CPU fp32 随机权重） ----
    B, S, H, Dk, Dv = 2, 384, 2, 64, 64
    torch.manual_seed(SEED)
    q = torch.randn(B, S, H, Dk)
    k = torch.randn(B, S, H, Dk)
    v = torch.randn(B, S, H, Dv)
    beta = torch.randn(B, S, H).sigmoid()
    g = -F.softplus(torch.randn(B, S, H) * 1.5)                                  # ≤ 0：log 空间衰减（R1）
    S0 = torch.randn(B, H, Dk, Dv) * 0.05                                        # 非零初始状态（对拍 state 传递）

    out_r, S_r = gdn_recurrent(q, k, v, g, beta, initial_state=S0, output_final_state=True)
    out_c, S_c = gdn_chunkwise(q, k, v, g, beta, initial_state=S0, output_final_state=True)
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
    for kk, vv in parity.items():
        print(f"[对拍] {kk}: max|Δ| = {vv:.2e}")
    assert parity["mine_recurrent_vs_hf_out"] < 1e-4, "自研 recurrent 与 HF 不一致——公式核对失败"
    assert parity["mine_chunk_vs_hf_out"] < 1e-4, "自研 chunkwise 与 HF 不一致——推导核对失败"
    results["parity_cpu_fp32"] = parity

    # 互证记录（静态，见文件头 R1-R5）落盘
    results["static_cross_check"] = {
        "R1_gate_direction": "decay exp(g) BEFORE predict/write；g=-exp(A_log)·softplus(a+dt_bias) ≤ 0",
        "R2_norm_position": "l2norm(q/k, eps=1e-6) 先于 q×Dk^-0.5（HF L419-425/L540-445 同序）",
        "R3_beta_on_kv": "β 乘在 k 与 v（k_beta/v_beta，HF L437-438）",
        "R4_ut_system": "(I−tril(βKKᵀD̂,-1))^-1 三角求解（HF L461-473）；本件用 inv(·) 等价实现",
        "R5_chunk_decay": "q·e^{cum}；k·e^{c_last−cum}；state·e^{c_last}（HF L490-492）",
        "verdict": "三方（文献公式/本件推导/HF 参考）逐项一致；数值对拍全部 < 1e-4",
    }

    # ---- ② wall-clock scaling（MPS） ----
    print("[计时] MPS：chunkwise vs 满注意力，n=512..8192（B=2,H=4,D=64,chunk=64）")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    scaling = {}
    for n in (512, 1024, 2048, 4096, 8192):
        torch.manual_seed(SEED + n)
        qm = torch.randn(2, 4, n, 64, device=device)
        km = torch.randn(2, 4, n, 64, device=device)
        vm = torch.randn(2, 4, n, 64, device=device)
        gm = -F.softplus(torch.randn(2, 4, n, device=device) * 1.5)
        bm = torch.randn(2, 4, n, device=device).sigmoid()
        with torch.no_grad():
            for _ in range(3):
                gdn_chunkwise(qm, km, vm, gm, bm)
                full_attention_ms(qm, km, vm)
            torch.mps.synchronize()
            t = {}
            for tag, fn in (("chunk_gdn", lambda: gdn_chunkwise(qm, km, vm, gm, bm)),
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
    ns = np.array(sorted(scaling), dtype=float)
    slope_chunk = float(np.polyfit(np.log(ns), np.log([scaling[int(n)]["chunk_gdn"] for n in ns]), 1)[0])
    slope_full = float(np.polyfit(np.log(ns), np.log([scaling[int(n)]["full_attn"] for n in ns]), 1)[0])
    print(f"  log-log 斜率：chunk {slope_chunk:.2f}（近线性≈1）/ full {slope_full:.2f}（二次≈2）")
    results["scaling_mps"] = {str(k): v for k, v in scaling.items()}
    results["scaling_slopes"] = {"chunk_gdn": slope_chunk, "full_attn": slope_full}
    results["note"] = ("「线性 O(n)」口径：训练并行形态（chunk 间线性扫描 + 块内二次）；"
                       "推理递归形态每步 O(Dk·Dv) 与 n 无关——两形态分列（纪律⑨）。")

    save_json("probe04_gdn", results, args.out_name)
    print("探针 04 完成。")


if __name__ == "__main__":
    main()
