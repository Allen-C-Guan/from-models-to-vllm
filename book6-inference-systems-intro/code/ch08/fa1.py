# fa1.py —— Book6 ch8 代码件：FlashAttention-1 的 online softmax 从零实现与对拍
# 用途：①「铺表」基线（naive：全量打分矩阵 materialize——O(n²) 中间量）与 online softmax 分块
#   实现（running max/sum 两状态、逐 tile 修正——O(n) 中间量）双实现；
#   ②CPU fp32 数值对拍 torch.sdpa（allclose 打印）——两实现 vs sdpa 三方一致；
#   ③NPU bf16 wall-clock 对照（naive vs 分块 vs sdpa：中间量的访存账变成时延差——n 扫描）。
# 数值口径：softmax 全程 fp32（对拍约定）；分块 tile 大小 --tile 可调。
# 所属章节：Book6 ch8 §8.4；设计书=plan/Book6-推理系统导论.md ch8
# 运行：cd <workspace> && source env.sh && ASCEND_RT_VISIBLE_DEVICES=<卡> \
#      python code/Book6-推理系统导论/ch08/fa1.py [--n 1024 --tile 128 --bench-npu --out-name base]
# 产物：log/book6-ch08/fa1_<out-name>.json（对拍 allclose + NPU 时延表）
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F


def naive_attention(q, k, v):
    """铺表基线：打分矩阵整个 materialize。q (B,h,n,hd) -> (B,h,n,hd)。"""
    n = q.shape[2]
    att = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)          # (B,h,n,n) <- 中间量 1
    mask = torch.tril(torch.ones(n, n, device=q.device, dtype=torch.bool))
    att = att.masked_fill(~mask, float("-inf"))
    att = F.softmax(att.float(), dim=-1).to(q.dtype)                  # (B,h,n,n) fp32 中转
    return att @ v                                                    # 中间量峰值：2×(B,h,n,n)


def flash_attn1(q, k, v, tile=128):
    """FA1 式分块：逐查询块算注意力，m/l 两状态在线修正——打分矩阵每块即弃。

    形状流转（tile=T）：q 块 (B,h,T,hd)；k/v 全长 (B,h,n,hd)；块打分 (B,h,T,n) 即弃；
    状态 m (B,h,T,1)、l (B,h,T,1)——中间量峰值 O(T·n) 而非 O(n²)。
    数值：softmax 以 m/l 缩放修正（FA1 式两遍 online——教学版按块处理键维即全宽）。
    """
    B, h, n, hd = q.shape
    scale = hd ** -0.5
    T = min(tile, n)
    out = torch.zeros_like(q)
    m = torch.full((B, h, n, 1), float("-inf"), device=q.device, dtype=torch.float32)   # 各查询行 running max
    l = torch.zeros(B, h, n, 1, device=q.device, dtype=torch.float32)                   # running sum
    acc = torch.zeros(B, h, n, hd, device=q.device, dtype=torch.float32)                # 加权累积
    for i0 in range(0, n, T):
        q_blk = q[:, :, i0:i0 + T]                                     # (B,h,T,hd)
        s = (q_blk.float() @ k.float().transpose(-2, -1)) * scale      # (B,h,T,n) 块中间量
        s = s.masked_fill(~_causal_rows(T, n, i0, q.device), float("-inf"))
        m_blk = s.max(dim=-1, keepdim=True).values                     # (B,h,T,1)
        p = torch.exp(s - m_blk)                                       # 数值安全：减块内 max
        l_blk = p.sum(dim=-1, keepdim=True)
        # 与已累积部分合并（online 修正）：老 acc 以老 m 归一、并入新块
        m_new = torch.maximum(m[:, :, i0:i0 + T], m_blk)
        alpha = torch.exp(m[:, :, i0:i0 + T] - m_new)                  # 老状态的缩放修正
        beta = torch.exp(m_blk - m_new)
        acc[:, :, i0:i0 + T] = acc[:, :, i0:i0 + T] * alpha + (p * beta) @ v.float()
        l[:, :, i0:i0 + T] = l[:, :, i0:i0 + T] * alpha + l_blk * beta
        m[:, :, i0:i0 + T] = m_new
    out = (acc / l).to(q.dtype)
    return out


def _causal_rows(T, n, i0, device):
    """查询块 [i0,i0+T) 对全部 n 个键的因果可见性 (T,n)。"""
    mask = torch.zeros(T, n, device=device, dtype=torch.bool)
    for i in range(T):
        mask[i, :i0 + i + 1] = True
    return mask


def make_qkv(B, h, n, hd, device, dtype, seed=20261002):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, h, n, hd, generator=g).to(device=device, dtype=dtype)
    k = torch.randn(B, h, n, hd, generator=g).to(device=device, dtype=dtype)
    v = torch.randn(B, h, n, hd, generator=g).to(device=device, dtype=dtype)
    return q, k, v


def is_causal_sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="FA1 online softmax 从零实现与对拍")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--bench-npu", action="store_true")
    ap.add_argument("--out-name", default="base")
    a = ap.parse_args()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "log", "book6-ch08")
    os.makedirs(out_dir, exist_ok=True)

    # ① 数值对拍（CPU fp32）：naive / flash1 / sdpa 三方
    q, k, v = make_qkv(2, 8, a.n, 64, "cpu", torch.float32)
    r_naive, r_fa1, r_sdpa = naive_attention(q, k, v), flash_attn1(q, k, v, a.tile), is_causal_sdpa(q, k, v)
    d1 = (r_fa1 - r_naive).abs().max().item()
    d2 = (r_fa1 - r_sdpa).abs().max().item()
    print(f"[对拍 CPU fp32] flash1 vs naive max|Δ|={d1:.2e} | flash1 vs sdpa max|Δ|={d2:.2e}")
    assert d1 < 1e-4 and d2 < 1e-4

    res = dict(n=a.n, tile=a.tile, parity_cpu_fp32=dict(fa1_vs_naive=d1, fa1_vs_sdpa=d2),
               npu_bench=[])

    # ② NPU bf16 时延对照（naive vs flash1 vs sdpa：中间量访存账的时延化身）
    if a.bench_npu and torch.npu.is_available():
        dev = "npu"
        for n in (512, 1024, 2048, 4096):
            q, k, v = make_qkv(1, 16, n, 64, dev, torch.bfloat16)
            row = dict(n=n)
            fns = {"naive": lambda: naive_attention(q, k, v),
                   "flash1": lambda: flash_attn1(q, k, v, a.tile),
                   "sdpa": lambda: is_causal_sdpa(q, k, v)}
            for _ in range(1):                      # warmup（首跑编译/缓存不入账）
                fns["sdpa"](); fns["naive"](); fns["flash1"]()
            for name, fn2 in fns.items():
                torch.npu.synchronize()
                t0 = time.perf_counter()
                for _ in range(3):
                    fn2()
                torch.npu.synchronize()
                row[name + "_ms"] = round((time.perf_counter() - t0) / 3 * 1e3, 3)
            res["npu_bench"].append(row)
            print(f"[NPU n={n}] naive {row['naive_ms']} ms | flash1 {row['flash1_ms']} ms | sdpa {row['sdpa_ms']} ms")
            del q, k, v
            torch.npu.empty_cache()

    p = os.path.join(out_dir, f"fa1_{a.out_name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[产物] {p}")


if __name__ == "__main__":
    main()
