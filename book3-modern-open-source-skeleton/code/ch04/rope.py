# rope.py —— Book3 ch4 教学重写件：旋转位置编码 RoPE（两半式正身 + 复数相邻配对对照 + 性质验证三件）
# 用途：ch4「实现考据三连」与「机器验证」的换装件正身。教学重写 = 不复制 llama_slots 的 RoPE 代码，
#       而是按式 (4.6)/(4.10) 从零写一遍，再过两道对拍验收：
#       ① 与 00-feasibility/llama_slots.py 的 build_rope_cache/apply_rope（组件库正身）CPU fp32 逐位一致；
#       ② 与 HF transformers 5.18.0 的 LlamaRotaryEmbedding + apply_rotary_pos_emb（rotate_half 两半式约定）
#         同 config 同 position_ids 对拍。
#       自测第三件：性质验证三件（平移不变 / 保范数 / 共轭置换等价）——式 (4.3)/(4.8) 的机器验证。
# 所属章节：Book3 ch4（4.7 实现三型 / 4.9 机器验证）；ch06/gqa.py 与 ch10 整机 import 本件作位置插槽
#       （插槽 import 契约：ch04/rope.py::build_rope_cache(seq_len, head_dim, theta, device) 与
#         apply_rope(q, k, cos, sin)——签名/行为承 llama_slots 正身，教学重写件须逐位一致）。
# 运行方式：cd code/ch04 && python rope.py   （全 CPU fp32 秒级，无命令行参数）
# 依赖：sys.path 两级 bootstrap（00-feasibility 的 llama_slots）；transformers 仅在对拍段 import。
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "00-feasibility")))
from llama_slots import apply_rope as slots_apply_rope          # noqa: E402  组件库正身（对拍对象①）
from llama_slots import build_rope_cache as slots_build_cache   # noqa: E402


# ---------------- 正身：两半式（rotate_half）实现 —— 对拍锚定 HF，键位契约承 llama_slots ----------------

def build_rope_cache(seq_len, head_dim, theta, device):
    """RoPE 的 cos/sin 表。返回 (cos, sin)，各 (1, n, head_dim)，fp32。

    inv_freq_i = 1 / theta^(2i/head_dim)，i = 0..head_dim/2-1（0 起口径，HF 源码同款；
    论文 §3.2.2 写 1 起的 10000^{-2(i-1)/d}，两式是同一个集合——记号考据见正文 4.4 节脚注）。
    freqs[t, i] = t * inv_freq_i（位置 t 在第 i 对维度上转过的角度）；
    emb = cat(freqs, freqs)——两半式约定需要整维宽的表（前半与后半用同一组角度）。
    n 可以任意大：没有查表上限，这正是「墙变软墙」的代码实体（ch7 外推不改代码的原因）。
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))  # (hd/2,)
    t = torch.arange(seq_len, dtype=torch.float32, device=device)                # (n,)
    freqs = torch.outer(t, inv_freq)                                             # (n, hd/2)
    emb = torch.cat((freqs, freqs), dim=-1)                                      # (n, hd) 前后两半同角度
    return emb.cos()[None], emb.sin()[None]                                      # (1,n,hd) 各一


def rotate_half(x):
    """(…, hd) -> (…, hd)：后一半取负放前、前一半放后。

    2D 旋转矩阵 (cos -sin; sin cos) 作用在第 (j, hd/2+j) 维对上时，实数写法拆成
    「x*cos + rotate_half(x)*sin」——复数乘法 (x+iy)(cos+i·sin) 的实数化（式 (4.10)）。
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """对 q/k 施加旋转（v 不旋——式 (4.3) 只约束 QK 内积，见正文 4.6 节三层论证）。

    q (B,h,n,hd)、k (B,h_kv,n,hd)；cos/sin (1,n,hd) 沿批与头维广播。
    返回 (q', k')，形状不变——RoPE 不引入新形状、不引入新参数，位置只作为旋转角进入。
    """
    cos, sin = cos.to(q.dtype), sin.to(q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)      # (B,h,n,hd)
    k_embed = (k * cos) + (rotate_half(k) * sin)      # (B,h_kv,n,hd)
    return q_embed, k_embed


# ---------------- 对照实现：复数相邻配对（interleaved）—— Meta 原版 llama 的约定 ----------------

def rope_interleaved_complex(x, pos, head_dim, theta):
    """复数 view_as_complex 版：第 i 对维度取 (x[2i], x[2i+1]) 视作复数，乘 e^{i·m·θ_i}。

    x (…, hd) -> 输出 (…, hd)。与两半式数学等价、只差一个固定置换（本文件 main 的
    性质三验证这件事）。Meta 官方 llama/model.py 的参考实现即此约定（相邻配对）。
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=x.dtype) / head_dim))  # (hd/2,)
    angles = pos * inv_freq                                     # (hd/2,) 位置 m 在各对上转过的角度
    fcis = torch.polar(torch.ones_like(angles), angles)         # e^{i·m·θ_i}（欧拉公式）
    xc = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())   # (…, hd/2) 复数
    out = xc * fcis                                             # 复数乘 = 转角（形状不变）
    return torch.view_as_real(out).reshape(*x.shape[:-1], head_dim)             # (…, hd)


def half_split_perm(head_dim):
    """τ：把两半式配对 (j, hd/2+j) 挪到相邻位 (2j, 2j+1) 的维度重排（共轭置换）。

    y = x[..., perm] 后，两半式的每一对搭档恰好变成相邻配对的第 j 对——
    于是「两半式(x) == 相邻式(x[perm])[argsort(perm)]」（输出侧用 τ^{-1}=argsort，排坑记录见正文 4.7）。
    权重侧同理：W_q/W_k 的输出行做同一置换，两套约定即可互换（Meta→HF 转换脚本的 permute）。
    """
    j = torch.arange(head_dim // 2)
    perm = torch.empty(head_dim, dtype=torch.long)
    perm[2 * j] = j                     # 前半第 j 维 -> 偶数位 2j
    perm[2 * j + 1] = head_dim // 2 + j  # 后半第 j 维 -> 奇数位 2j+1
    return perm


def _rope_half_at(x, pos, head_dim, theta):
    """两半式旋转的定点版（性质验证用）：只把位置 pos 的那一行 cos/sin 应到 x 上。

    dtype 跟随 x（fp64 验证恒等式、fp32 复现工程口径——大角度下两者的差是角度舍入、
    不是恒等性失效，见 main 性质一的对照报告）。
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=x.dtype) / head_dim))  # (hd/2,)
    angles = pos * inv_freq                                          # (hd/2,)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)            # (hd,)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)            # (hd,)
    return (x * cos) + (rotate_half(x) * sin)                        # (…, hd)


# ---------------- 自测：对拍两道 + 性质验证三件 ----------------

def main():
    torch.manual_seed(20261002)          # 全书统一种子
    device = torch.device("cpu")         # 数值对拍一律 CPU fp32 报数（写作规范 §8）
    hd, theta, n = 64, 10000.0, 128      # 模块档头维 / LLaMA1-2 基底 / 序列长
    B, h, h_kv = 2, 8, 4

    cos, sin = build_rope_cache(n, hd, theta, device)                # (1,n,hd) x2
    q = torch.randn(B, h, n, hd)                                      # (2,8,128,64)
    k = torch.randn(B, h_kv, n, hd)                                   # (2,4,128,64)
    qr, kr = apply_rope(q, k, cos, sin)                               # 各形状不变
    print(f"== 形状检查 ==\n  q {tuple(q.shape)} -> {tuple(qr.shape)}；k {tuple(k.shape)} -> {tuple(kr.shape)}"
          f"（RoPE 不引入新形状；cos/sin {tuple(cos.shape)} 沿 B 与头维广播）")

    # ---- 对拍①：vs llama_slots 正身（CPU fp32 逐位一致——插槽 import 契约的验收） ----
    cos0, sin0 = slots_build_cache(n, hd, theta, device)
    qr0, kr0 = slots_apply_rope(q, k, cos0, sin0)
    d_cache = max((cos - cos0).abs().max().item(), (sin - sin0).abs().max().item())
    d_apply = max((qr - qr0).abs().max().item(), (kr - kr0).abs().max().item())
    print(f"== 对拍① vs llama_slots（组件库正身，strict 契约）==")
    print(f"  cos/sin 表 max|Δ| = {d_cache:.3e}；apply 后 max|Δ| = {d_apply:.3e}（逐位一致判据 =0）")
    assert d_cache == 0.0 and d_apply == 0.0, "教学重写件与 llama_slots 正身出现数值分叉——契约破坏"

    # ---- 对拍②：vs HF transformers 5.18.0（rotate_half 两半式约定） ----
    from transformers import LlamaConfig as HFLlamaConfig
    from transformers.models.llama.modeling_llama import (LlamaRotaryEmbedding,
                                                          apply_rotary_pos_emb)
    cfg = HFLlamaConfig(hidden_size=hd * h, num_attention_heads=h, num_key_value_heads=h_kv,
                        head_dim=hd, rope_theta=theta, max_position_embeddings=n)
    rot = LlamaRotaryEmbedding(config=cfg)
    pos_ids = torch.arange(n).expand(B, n)                            # (B,n) 位置号
    cos_hf, sin_hf = rot(q.float(), pos_ids)                          # (B,n,hd) x2（内部强制 fp32）
    qr_hf, kr_hf = apply_rotary_pos_emb(q, k, cos_hf, sin_hf)         # unsqueeze(1) 广播到头维
    d_hf = max((qr - qr_hf).abs().max().item(), (kr - kr_hf).abs().max().item())
    print(f"== 对拍② vs HF LlamaRotaryEmbedding（transformers 5.18.0，CPU fp32）==")
    print(f"  q/k 旋转后 max|Δ| = {d_hf:.3e}（inv_freq 外积 vs 矩阵乘两种写法、数值同源）")
    assert d_hf < 1e-6

    # ---- 性质一：平移不变（式 (4.3) 的机器验证）—— 内积只依赖 m-n ----
    # 口径说明：恒等式的机器验证用 fp64（身份本身是精确的）；fp32 下大角度的舍入会盖过
    # 恒等差——pos≈5100、θ_0=1 时 fp32 角度绝对误差已达 ~5e-4 rad（HF 强制 fp32 算 cos/sin
    # 的数值纪律与此同源，正文 4.9 节一句注记）。
    print(f"== 性质一：平移不变（d={hd}，随机 200 对 q/k，fp64）==")
    torch.manual_seed(20261002)
    worst = 0.0
    for _ in range(200):
        qv, kv = torch.randn(hd, dtype=torch.float64), torch.randn(hd, dtype=torch.float64)
        m, nn, t = 100, 37, 5000                                  # (m,n) 与整体平移 t 后比较
        ip1 = torch.dot(_rope_half_at(qv, m, hd, theta), _rope_half_at(kv, nn, hd, theta))
        ip2 = torch.dot(_rope_half_at(qv, m + t, hd, theta), _rope_half_at(kv, nn + t, hd, theta))
        worst = max(worst, abs(ip1 - ip2).item())
    print(f"  ⟨R_m q, R_n k⟩ vs ⟨R_(m+t) q, R_(n+t) k⟩ 的 max|Δ| = {worst:.2e}"
          f"（fp64——绝对位置在共轭相乘中消掉，式 (4.3) 闭合）")
    assert worst < 1e-9
    # fp32 对照（如实报数，不进判据）：
    torch.manual_seed(20261002)
    worst32 = 0.0
    for _ in range(200):
        qv, kv = torch.randn(hd), torch.randn(hd)
        ip1 = torch.dot(_rope_half_at(qv, 100, hd, theta), _rope_half_at(kv, 37, hd, theta))
        ip2 = torch.dot(_rope_half_at(qv, 5100, hd, theta), _rope_half_at(kv, 5037, hd, theta))
        worst32 = max(worst32, abs(ip1 - ip2).item())
    print(f"  fp32 对照（同试验，pos 5100）：max|Δ| = {worst32:.2e}——大角度舍入成为主导，"
          f"这是角度精度问题、不是恒等性失效")

    # ---- 性质二：保范数（正交性——位置只改方向不改长度） ----
    torch.manual_seed(20261002)
    x = torch.randn(2, 4, 9, hd)                                   # (B,h,n,hd)
    n_before = x.norm(dim=-1)
    n_after = apply_rope(x, x, cos[:, :9], sin[:, :9])[0].norm(dim=-1)
    rel = ((n_after - n_before).abs() / n_before).max().item()
    print(f"== 性质二：保范数 ==\n  旋转前后 ‖x‖ 相对偏差 max = {rel:.2e}（fp32 舍入级——R 正交，"
          f"论文所谓 encoding 稳定性；恒等本身精确，见性质一的 fp64 口径）")
    assert rel < 1e-5

    # ---- 性质三：共轭置换等价（两半式 == 相邻配对 ∘ τ） ----
    print(f"== 性质三：共轭置换（两半式 vs 复数相邻配对，d=8/64/128 × pos=13/1234/9999）==")
    torch.manual_seed(20261002)
    worst3, rows = 0.0, []
    for d in (8, 64, 128):
        for pos in (13, 1234, 9999):
            x = torch.randn(2, 3, 5, d)                            # (B,h,n,d)
            half = _rope_half_at(x, pos, d, theta)                 # 两半式输出
            perm = half_split_perm(d)                              # τ：半式搭档 -> 相邻
            inter = rope_interleaved_complex(x[..., perm], pos, d, theta)   # 先置换再相邻配对
            back = inter[..., torch.argsort(perm)]                 # 输出侧 τ^{-1} = argsort（排坑点）
            diff = (half - back).abs().max().item()
            worst3 = max(worst3, diff)
            rows.append((d, pos, diff))
    for d, pos, diff in rows:
        print(f"  d={d:3d} pos={pos:5d}  max|Δ| = {diff:.2e}")
    print(f"  三组九格全过，worst = {worst3:.2e}（fp32 噪声级——两种实现只差一个固定置换）")
    assert worst3 < 1e-5

    # ---- 波长谱表（表 4.1 的数据源）：d=128 档逐维波长 ----
    print(f"== 波长谱（d=128，theta={theta:.0f}）：λ_i = 2π/θ_i = 转一圈所需位置步数 ==")
    for i in (0, 1, 31, 32, 63):
        th = theta ** (-2 * i / 128)
        print(f"  i={i:2d}  θ_i = {th:.4e}  λ_i = {2 * torch.pi / th:.1f} 步/圈"
              f"{'（最快：每步转 1 弧度）' if i == 0 else ''}"
              f"{'（最慢：训练窗内从未转完一圈）' if i == 63 else ''}")

    print("rope.py 自测全部通过（对拍①逐位一致 / 对拍② HF 同构 / 性质三件 / 波长谱）")


if __name__ == "__main__":
    main()
