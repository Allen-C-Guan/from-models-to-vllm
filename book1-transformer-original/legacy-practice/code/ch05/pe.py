# -*- coding: utf-8 -*-
# 用途：第 5 章「正弦位置编码」全部实验——论文式 PE 实现、波长/性质数值验证、
#       相对位移性质（旋转矩阵 M(k)）与点积恒等式、置换等变最小实验、
#       t2t/HF 第三方对拍（已先核维度排布口径）、出图 fig-5-1 / fig-5-2。
# 所属：《从模型到 vLLM》Book1 第 5 章（原论文 v7 §3.5 / Table 3 (E)）
# 运行：source env.sh && python code/ch05/pe.py
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap

FIG_DIR = Path(__file__).resolve().parents[3] / "drafts" / "Book1-Transformer原典" / "figures"
# _写作规范.md 第 4 节配色：主色蓝 #2a78d6；发散型（有极性）蓝-灰-红，中点 #f0efec
BLUE, AX_GREY, TITLE, NOTE, GRID = "#2a78d6", "#898781", "#0b0b0b", "#52514e", "#e1e0d9"
CMAP_DIV = LinearSegmentedColormap.from_list("div", ["#0d366b", "#f0efec", "#c14a39"])


def sinusoidal_pe(length, d_model, dtype=torch.float64):
    """论文式正弦位置编码（v7 §3.5）：偶维 sin、奇维 cos（interleave 排布）。
    频率口径：第 i 对维度的角频率为 10000^{-2i/d_model}（指数分母是 d_model）。
    形状：输入标量 length(=L) 与 d_model(=d)；返回 pe (L, d)——第 pos 行即位置 pos 的编码向量。"""
    pos = torch.arange(length, dtype=dtype).unsqueeze(1)            # (L,1)
    pair = torch.arange(d_model // 2, dtype=dtype)                  # (d/2,)，i = 0..d/2-1
    inv_tau = 10000.0 ** (-2.0 * pair / d_model)                    # (d/2,)，1/τ_i
    pe = torch.zeros(length, d_model, dtype=dtype)                  # (L,d)
    pe[:, 0::2] = torch.sin(pos * inv_tau)                          # (L,1)×(d/2,) 广播 -> (L,d/2)，写入偶维列
    pe[:, 1::2] = torch.cos(pos * inv_tau)                          # 同上，写入奇维列
    return pe                                                       # (L,d)


def t2t_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4, start_index=0):
    """tensor2tensor get_timing_signal_1d 的逐算子 torch 移植，用于第三方对拍。
    来源：tensor2tensor/layers/common_attention.py，Apache License 2.0，
    Copyright The Tensor2Tensor Authors（master 分支，2026-10 取回）。
    与论文式的两处口径差异（先核口径再对拍，见正文考据框）：
    1) 排布：sin/cos 沿通道维 concat（前半 sin、后半 cos），论文是 interleave；
       t2t 源码自注 "this slightly differs from the published paper"（PR #177）。
    2) 频率：等比数列端点精确落在 max_timescale（分母 num_timescales-1），
       论文指数为 2i/d_model（端点只到 10000^{(d-2)/d}）。
    形状：输入标量 length/channels；返回 signal (L, channels)——前半 sin、后半 cos（concat 排布）。"""
    position = (torch.arange(length, dtype=torch.float32) + start_index)   # (L,)
    num_timescales = channels // 2                                        # 频率根数 = channels/2
    log_inc = math.log(max_timescale / min_timescale) / max(num_timescales - 1, 1)
    inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales, dtype=torch.float32) * -log_inc)  # (num_timescales,)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)     # (L,1)×(1,num_timescales) -> (L,num_timescales)
    signal = torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)     # (L,channels)：前半 sin、后半 cos
    if channels % 2:
        signal = torch.cat([signal, torch.zeros(length, 1)], dim=1)       # 奇数通道补零列，保持 (L,channels)
    return signal  # (L, channels)，对应 t2t 返回的 [1, length, channels] 去掉 batch 维


def rotation_m(k, d_model, dtype=torch.float64):
    """相对位移性质的线性变换 M(k)：使 PE_{pos+k} = M(k)·PE_pos（本书补写推导，见 5.3）。
    分块对角 2x2 旋转，第 i 块角度 δ_i = k/τ_i = k·10000^{-2i/d_model}；与 pos 无关。
    形状：输入标量 k 与 d_model；返回 m (d, d) 块对角——作用在 (d,) 的 PE 向量上。"""
    delta = k * (10000.0 ** (-2.0 * torch.arange(d_model // 2, dtype=dtype) / d_model))  # (d/2,)，各块转角 δ_i
    m = torch.zeros(d_model, d_model, dtype=dtype)              # (d,d)
    m[0::2, 0::2] = torch.diag(torch.cos(delta))               # 偶-偶块 (d/2,d/2) 对角：cos δ_i
    m[0::2, 1::2] = torch.diag(torch.sin(delta))               # 偶-奇块 (d/2,d/2) 对角：sin δ_i
    m[1::2, 0::2] = torch.diag(-torch.sin(delta))              # 奇-偶块 (d/2,d/2) 对角：−sin δ_i
    m[1::2, 1::2] = torch.diag(torch.cos(delta))               # 奇-奇块 (d/2,d/2) 对角：cos δ_i
    return m                                                    # (d,d)


def attn_forward(x, w, causal=False):
    """单层单头缩放点积注意力（手写 matmul+softmax，见第 2 章式 (2.1)）。
    不含 PE、dropout、偏置；用于置换等变最小实验。
    形状：输入 x (n,d)、w=(wq,wk,wv,wo) 各 (d,d)；返回 (n,d)。"""
    wq, wk, wv, wo = w
    q, k, v = x @ wq, x @ wk, x @ wv            # (n,d)@(d,d) -> (n,d)，投影逐位置进行
    score = q @ k.T / math.sqrt(x.shape[1])     # (n,d)@(d,n) -> (n,n) 打分矩阵
    if causal:  # 因果 mask：只许看过去（第 4 章 4.2）
        n = x.shape[0]
        score = score.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))  # (n,n) 上三角置 -inf
    return torch.softmax(score, dim=-1) @ v @ wo  # (n,n)@(n,d)@(d,d) -> (n,d)


def main():
    torch.manual_seed(5)
    d, length = 512, 1024
    pe64 = sinusoidal_pe(length, d)                 # float64：性质验证
    pe32 = sinusoidal_pe(length, d, torch.float32)  # float32：工程精度与出图

    print(f"== 1) 波长表（d_model={d}，λ_i = 2π·10000^(2i/d)）==")
    for i in (0, 64, 128, 192, 255):
        tau = 10000.0 ** (2 * i / d)
        print(f"  i={i:3d}  τ_i={tau:11.3f}  λ_i={2*math.pi*tau:.2f}")

    print("== 2) 解析断言：PE[0] 交替 0/1；首对维度恰为 sin(pos)/cos(pos)；范数恒 √(d/2) ==")
    pos = torch.arange(length, dtype=torch.float64)
    assert torch.allclose(pe64[0, 0::2], torch.zeros(d // 2, dtype=torch.float64))
    assert torch.allclose(pe64[0, 1::2], torch.ones(d // 2, dtype=torch.float64))
    assert torch.allclose(pe64[:, 0], pos.sin()) and torch.allclose(pe64[:, 1], pos.cos())
    dev_norm = (pe64.norm(dim=1) - math.sqrt(d / 2)).abs().max()
    print(f"  PE[0] 前 6 维 = {pe64[0, :6].tolist()}")
    print(f"  max | ‖PE_pos‖ − √(d/2)=16 | = {dev_norm:.3e}")

    print("== 3) 相对位移性质 PE_{pos+k} = M(k)·PE_pos（命题属原论文假设句，推导为本书补写）==")
    for k in (1, 7, 137):
        mk = rotation_m(k, d)
        mk32 = rotation_m(k, d, torch.float32)
        err64 = max((pe64[pos + k] - mk @ pe64[pos]).abs().max() for pos in range(0, 400))
        err32 = max((pe32[pos + k] - mk32 @ pe32[pos]).abs().max() for pos in range(0, 400))
        print(f"  k={k:3d}: max|PE_{{pos+k}} − M(k)PE_pos|  float64={err64:.3e}  float32={err32:.3e}")
    print(f"  旋转可加性 max|M(3)M(5) − M(8)| = {(rotation_m(3, d) @ rotation_m(5, d) - rotation_m(8, d)).abs().max():.3e}")
    print(f"  M(k) 正交性 max|M(7)ᵀM(7) − I| = {(rotation_m(7, d).T @ rotation_m(7, d) - torch.eye(d)).abs().max():.3e}")

    print("== 4) 点积恒等式 PE_pos·PE_{pos+k} = Σ_i cos(k/τ_i)（恰只依赖相对偏移 k，本书补写）==")
    inv_tau = 10000.0 ** (-2.0 * torch.arange(d // 2, dtype=torch.float64) / d)  # (d/2,)，1/τ_i
    g = pe64 @ pe64.T                                    # (1024,1024) 内积矩阵，沿反对角线恒定（Toeplitz）
    dev_ident, dev_collapse = 0.0, 0.0
    for pos in range(0, 300):
        for k in range(0, 60):
            theory = torch.cos(k * inv_tau).sum()
            dev_ident = max(dev_ident, abs(g[pos, pos + k] - theory))
            if pos > 0:
                dev_collapse = max(dev_collapse, abs(g[pos, pos + k] - g[0, k]))
    print(f"  max|点积 − Σcos(k/τ_i)| = {dev_ident:.3e}；max|S(pos,k) − S(0,k)| = {dev_collapse:.3e}")

    print("== 5) 置换等变最小实验（单头注意力，随机固定权重，d=64, L=12）==")
    gen = torch.Generator().manual_seed(5)
    w = [torch.randn(64, 64, generator=gen) / 8 for _ in range(4)]   # wq,wk,wv,wo 各 (64,64)
    x = torch.randn(12, 64, generator=gen)                           # (12,64)：12 个位置 × 64 维
    pi = torch.randperm(12, generator=gen)                           # 固定随机置换 π
    def perm_gap(y_a, y_b):  # 等变即 f(X)[π] == f(X[π])：逐位比较 Y_a[π(i)] 与 Y_b[i]
        return (y_a[pi] - y_b).abs().max().item()
    d_pe = sinusoidal_pe(12, 64, torch.float32)                      # (12,64)，与 x 同形方可相加
    y1, y2 = attn_forward(x, w), attn_forward(x[pi], w)              # 各 (12,64)
    y3, y4 = attn_forward(x + d_pe, w), attn_forward(x[pi] + d_pe, w)  # 加 PE：(12,64)+(12,64) 逐元素相加（批量情形即正文 5.2 的 (B,n,512)+(n,512) 广播，此处 B=1 省去批维）
    y5, y6 = attn_forward(x, w, causal=True), attn_forward(x[pi], w, causal=True)
    print(f"  无 PE 双向:  max|Y[π] − Y_打乱| = {perm_gap(y1, y2):.3e}   → 等变成立")
    print(f"  有 PE 双向:  同一检查           = {perm_gap(y3, y4):.3f}   → 等变被打破（顺序被感知）")
    print(f"  无 PE 因果:  同一检查           = {perm_gap(y5, y6):.3f}   → 连等变都不成立")

    print("== 6) 第三方对拍（先核口径：t2t/HF 用 concat + 端点 10000，论文用 interleave + 指数 2i/d）==")
    t2t = t2t_timing_signal_1d(length, d)
    paper_concat = torch.cat([pe32[:, 0::2], pe32[:, 1::2]], dim=1)  # 论文式 (1024,512) 改 concat 排布以对齐：两半各 (1024,256)
    gap = (t2t - paper_concat).abs()
    i_worst, j_worst = divmod(gap.argmax().item(), d)
    print(f"  排布对齐后 max|t2t − 论文式| = {gap.max():.4f}（出现在 pos={i_worst}，第 {j_worst if j_worst < 256 else j_worst - 256} 对维度；"
          f"原因=频率口径差 255 档 vs 256 档分母，端点 10000 vs {10000.0 ** (510 / 512):.1f}）")
    try:
        from transformers.models.fsmt.modeling_fsmt import SinusoidalPositionalEmbedding
        hf = SinusoidalPositionalEmbedding.get_embedding(length, d, None)
        ok = torch.allclose(t2t, hf, rtol=1e-5, atol=1e-8)
        print(f"  本文 t2t 移植 vs HF(fairseq 系) get_embedding: allclose={ok}, max|diff|={(t2t - hf).abs().max():.3e}")
    except Exception as e:  # 环境无 transformers 时如实降级
        print(f"  [跳过 HF 对拍：{type(e).__name__}: {e}]")

    # ---- 图 5.1：PE 值热图（前 100 个位置 × 512 维）----
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.imshow(pe32[:100].numpy(), cmap=CMAP_DIV, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xlabel("dimension (even: sin / odd: cos)", color=AX_GREY)
    ax.set_ylabel("position pos", color=AX_GREY)
    ax.set_title("Sinusoidal positional encoding values (d_model = 512)", color=TITLE)
    ax.tick_params(colors=AX_GREY)
    for s in ax.spines.values():
        s.set_color(GRID)
    cb = fig.colorbar(ax.images[0], ax=ax, pad=0.01)
    cb.ax.tick_params(colors=AX_GREY)
    cb.outline.set_edgecolor(GRID)
    cb.set_label("value", color=AX_GREY)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig-5-1-sinusoidal-pe-values.png", facecolor="white")
    plt.close(fig)

    # ---- 图 5.2：相对位置点积相似度（左：Toeplitz 热图；右：随偏移 k 的曲线）----
    cos_sim = (pe64[:100] @ pe64[:100].T / (d / 2)).numpy()          # (100,100)：前 100 位置的内积/归一
    k_axis = torch.arange(300, dtype=torch.float64)
    s_k = torch.stack([torch.cos(kk * inv_tau).sum() for kk in k_axis]) / (d / 2)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    a1.imshow(cos_sim, cmap=CMAP_DIV, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    a1.set_xlabel("position j", color=AX_GREY)
    a1.set_ylabel("position i", color=AX_GREY)
    a1.set_title("PE_i · PE_j / (d/2): Toeplitz", color=TITLE)
    a1.tick_params(colors=AX_GREY)
    for s in a1.spines.values():
        s.set_color(GRID)
    a2.plot(k_axis.numpy(), s_k.numpy(), color=BLUE, lw=2)
    a2.axhline(0, color=GRID, lw=1)
    a2.grid(color=GRID, lw=0.5)
    a2.set_axisbelow(True)
    a2.set_xlabel("relative offset k", color=AX_GREY)
    a2.set_ylabel("PE_pos · PE_(pos+k) / (d/2)", color=AX_GREY)
    a2.set_title("Similarity depends only on k", color=TITLE)
    a2.tick_params(colors=AX_GREY)
    for s in a2.spines.values():
        s.set_color(AX_GREY)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig-5-2-relative-position-similarity.png", facecolor="white")
    plt.close(fig)
    print(f"== 图已保存至 {FIG_DIR} ==")


if __name__ == "__main__":
    main()
