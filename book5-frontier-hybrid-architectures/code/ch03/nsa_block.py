# nsa_block.py —— Book5 ch3 正式实验件：NSA 式块稀疏注意力（blockwise top-k 选择 + gather）
# 用途（大纲 ch3「路线二·可学习稀疏：NSA→DSA」正身实验；探针 03 升格——升格判据：凡入正文
#   数字=本脚本重跑+JSON 落 log/book5-ch03/；探针 03 JSON 只作设计记录）：
#   ① 机制件：块级 top-k 选择注意力——「块压缩分数（查询块 × 块均值键）→ 因果资格过滤 →
#      top-k 块选择 → gather 全精度块 → 块内带因果位注意力」的 blockwise gather-scatter 实现
#      （NSA 选择分支的教学骨架；逐运算形状注释 + 形状流转打印）；
#   ② 正确性（CPU fp32）：k_sel=全部块 时与满注意力逐位对拍（max|Δ|，期望 ~1e-6 级=仅求和序差）；
#      k_sel<全部块 时近似 gap（rel-L2 阶梯，k_sel=4/8/12/16 四点——16=选全块=阶梯地板）——
#      「丢块=有损压缩」的现场读数；
#   ③ 质量对照（MPS，fast 150 步正式档）：MiniLM T=1024 full vs 块稀疏(k_sel=8/16)——可训性同档可比；
#   ④ 解码账本 Table 4 复现（P7）：NSA 超参 l=32/d=16/l'=64/n=16/w=512，逐档断言
#      load(t)=t//d + n*l' + w ≡ 论文 Table 4 的 NSA 行（8192→2048 / 16384→2560 /
#      32768→3584 / 65536→5632 四列全中）；同时记录 Eq.7 字面 (t-l)//d 与表格的 1-2 项差
#      （考据框素材：数值表按 t/d 计）。
# 口径声明：本实现是 NSA「选择分支」的教学骨架（压缩分支用块均值键近似、滑窗分支不单独实现），
#   与 NSA 论文三分支完整形的差异在 ch3 正文展开；计时口径承探针 03 结论（教学实现比满注意力慢，
#   只看机制与 gap 不看绝对速度），正式件不再单列计时节。
# 几何教训（探针 03 run1 教训，防退化）：n=512/b=64 只有 8 块，k_sel≥8 钳到全块、gap 恒 0——
#   对拍几何固定 n=1024/b=64（16 块），k_sel 阶梯 {4,8,12} 均有效。
# 类名契约：BlockSparseAttention(cfg)（大纲 ch3 插槽契约；cfg=NSAConfig 数据类）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch03/nsa_block.py" [--out-name fast150]
#   正确性 CPU fp32；质量对照 MPS bf16；预计 4-6 分钟。
# 产物：log/book5-ch03/nsa_block_{out-name}.json（不入库）
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book5-ch03")
SEED = 20261002

# 跨册三候选 bootstrap（llama409 `_B3_CANDS` 同款已验证模式）+ 本册 ch02 兄弟章（needle_probe 复用）
_B4_FEAS_CANDS = [
    os.path.join(REPO_ROOT, "code", "Book4-规模与效率", "00-feasibility"),
    os.path.join(REPO_ROOT, "from-models-to-vllm", "book4-scale-and-efficiency", "code", "00-feasibility"),
    os.path.join(REPO_ROOT, "book4-scale-and-efficiency", "code", "00-feasibility"),
]
sys.path.insert(0, os.path.join(HERE, "..", "ch02"))        # needle_probe（MiniLM/FullAttn/SpBatch/训练环）
import needle_probe  # noqa: E402  ch02 兄弟章：MiniLM/FullAttn/SpBatch/train_mini/apply_rope（main 守卫内安全）


def bootstrap():
    hits = [p for p in _B4_FEAS_CANDS if os.path.exists(os.path.join(p, "moe_mla_slots.py"))]
    if not hits:
        raise FileNotFoundError(_B4_FEAS_CANDS)
    sys.path.insert(0, hits[0])
    import moe_mla_slots as m4
    import needle_probe as p2
    return m4, p2, hits


# ---------------- 块稀疏注意力核心（op 级，含形状流转） ----------------
def block_sparse_attention(q, k, v, block, n_selected, return_shapes=False):
    """块级 top-k 选择注意力（NSA 选择分支教学骨架）。

    输入：q (B,h,n,d_k) / k,v (B,h,n,d_k)（已展开到 h 头、已过 rope）；
    选择分 = 查询块 × 块均值键 的点积和；因果资格：查询块 i 只可选键块 j ≤ i；
    选中块 gather 全精度 K/V，块内做带因果位的注意力。返回 (B,h,n,d_k)。"""
    B, h, n, d_k = q.shape
    nb = n // block
    assert n == nb * block, f"n={n} 需为 block={block} 的倍数"
    scale = d_k ** -0.5

    qb = q.view(B, h, nb, block, d_k)                                            # (B,h,nb,b,d_k) 查询块
    kb = k.view(B, h, nb, block, d_k)                                            # (B,h,nb,b,d_k) 键块
    vb = v.view(B, h, nb, block, d_k)                                            # (B,h,nb,b,d_k) 值块
    k_mean = kb.mean(dim=3)                                                      # (B,h,nb,d_k) 块压缩键（NSA 压缩分支近似）

    # 块选择分：查询块内 token 与压缩键的点积求和（相对序不变，不另除 b；einsum 字母不重用：
    # z=B,h=头,q=查询块,k=键块,w=块内位,d=d_k——探针踩坑账①）
    blk_scores = torch.einsum("zhqwd,zhkd->zhqk", qb, k_mean) * scale            # (B,h,nb,nb) 选择矩阵
    elig = torch.ones(nb, nb, dtype=torch.bool, device=q.device).tril()          # (nb,nb) 因果资格：j ≤ i
    blk_scores = blk_scores.masked_fill(~elig, float("-inf"))                   # 未来块不可选

    k_sel = min(n_selected, nb)
    _, sel_idx = torch.topk(blk_scores, k_sel, dim=-1)                           # (B,h,nb,k_sel) 选中块号

    # gather 键/值块：torch.gather 不能增维——按选中槽位逐个 gather（k_sel ≤ nb，小循环清晰）
    def gather_blocks(xb):
        parts = [xb.gather(2, sel_idx[..., i:i + 1].unsqueeze(-1)
                           .expand(B, h, nb, block, xb.shape[-1]))
                 for i in range(k_sel)]                                          # 各 (B,h,nb,block,d_k)
        return torch.stack(parts, dim=3).reshape(B, h, nb, k_sel * block, xb.shape[-1])
    kg = gather_blocks(kb)                                                       # (B,h,nb,k_sel·b,d_k) gather 键
    vg = gather_blocks(vb)                                                       # (B,h,nb,k_sel·b,d_k) gather 值

    att = torch.einsum("zhqwd,zhqsd->zhqws", qb, kg) * scale                     # (B,h,nb,b,k_sel·b) 块内分数
    # 因果位掩码：键全局位 = 块号·b + 块内位；查询全局位同理；allowed = kp ≤ qp（广播）
    kp = (sel_idx.unsqueeze(-1) * block + torch.arange(block, device=q.device)).view(B, h, nb, 1, k_sel * block)
    qp = (torch.arange(nb, device=q.device).view(nb, 1) * block + torch.arange(block, device=q.device)).view(1, 1, nb, block, 1)
    allowed = kp <= qp                                                           # (B,h,nb,b,k_sel·b)
    att = att.masked_fill(~allowed, float("-inf"))
    p = F.softmax(att.float(), dim=-1).to(q.dtype)                               # fp32 softmax → 原 dtype
    out = torch.einsum("zhqws,zhqsd->zhqwd", p, vg)                              # (B,h,nb,b,d_k)
    out = out.reshape(B, h, n, d_k)                                              # (B,h,n,d_k)
    if return_shapes:
        shapes = {"blk_scores": tuple(blk_scores.shape), "sel_idx": tuple(sel_idx.shape),
                  "gathered_k": tuple(kg.shape), "intra_att": tuple(att.shape), "out": tuple(out.shape)}
        return out, shapes
    return out


def full_attention_ref(q, k, v):
    """满因果注意力（对拍参照）：q,k,v (B,h,n,d_k) → (B,h,n,d_k)。"""
    n = q.shape[2]
    att = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)                      # (B,h,n,n)
    mask = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()
    att = att.masked_fill(~mask, float("-inf"))
    p = F.softmax(att.float(), dim=-1).to(q.dtype)
    return p @ v                                                                # (B,h,n,d_k)


# ---------------- 注意力插槽（MiniLM 兼容：forward(x, cos, sin) → (B,n,d)） ----------------
@dataclass
class NSAConfig:
    """块稀疏插槽 config（大纲 ch3 插槽契约：BlockSparseAttention(cfg) 单参构造）。"""
    d: int = 256
    n_head: int = 8
    n_kv_head: int = 4
    d_k: int = 64
    block: int = 64            # 块宽 b
    n_selected: int = 8        # 每查询块选 k_sel 个键块
    use_rope: bool = True


class BlockSparseAttention(nn.Module):
    """块稀疏注意力插槽（大纲类名契约）：GQA 四投影 + rope + 块级 top-k 选择。

    核心调 block_sparse_attention（op 级教学件）；形状流转：
    (B,n,d) →q/k/v 投影→ (B,h,n,d_k) →块化+选择+gather→ (B,h,nb,k_sel·b,d_k) →块内注意→ (B,n,d)。"""

    def __init__(self, cfg: NSAConfig):
        super().__init__()
        self.h, self.h_kv, self.d_k = cfg.n_head, cfg.n_kv_head, cfg.d_k
        self.block, self.n_selected, self.use_rope = cfg.block, cfg.n_selected, cfg.use_rope
        self.q_proj = nn.Linear(cfg.d, cfg.n_head * cfg.d_k, bias=False)         # (B,n,d)→(B,n,h·d_k)
        self.k_proj = nn.Linear(cfg.d, cfg.n_kv_head * cfg.d_k, bias=False)
        self.v_proj = nn.Linear(cfg.d, cfg.n_kv_head * cfg.d_k, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * cfg.d_k, cfg.d, bias=False)

    def forward(self, x, cos, sin):
        B, n, d = x.shape
        q = self.q_proj(x).view(B, n, self.h, self.d_k).transpose(1, 2)          # (B,h,n,d_k)
        k = self.k_proj(x).view(B, n, self.h_kv, self.d_k).transpose(1, 2)       # (B,h_kv,n,d_k)
        v = self.v_proj(x).view(B, n, self.h_kv, self.d_k).transpose(1, 2)
        if self.use_rope:
            q, k = needle_probe.apply_rope(q, cos, sin), needle_probe.apply_rope(k, cos, sin)
        rep = self.h // self.h_kv
        k = k.repeat_interleave(rep, dim=1)                                      # (B,h,n,d_k) GQA 展开
        v = v.repeat_interleave(rep, dim=1)
        y = block_sparse_attention(q, k, v, self.block, self.n_selected)         # (B,h,n,d_k)
        y = y.transpose(1, 2).contiguous().view(B, n, self.h * self.d_k)         # (B,n,h·d_k)
        return self.o_proj(y)                                                    # (B,n,d)


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="Book5 ch3 NSA 式块稀疏正式件（对拍+gap+质量）")
    ap.add_argument("--out-name", default="fast150")
    ap.add_argument("--steps", type=int, default=150, help="质量对照训练步数（fast=150 正式档）")
    args = ap.parse_args()
    m4, needle_probe, hits = bootstrap()
    device = m4.pick_device("mps")
    print(f"[bootstrap] moe_mla_slots 自：{hits[0]} | needle_probe 复用（ch02 兄弟章）| device {device}",
          flush=True)
    torch.manual_seed(SEED)
    results = {"seed": SEED, "date": "2026-10-05", "steps": args.steps}

    # ① 正确性（CPU fp32）：k_sel=nb 全块逐位对拍 + k_sel<nb 近似 gap 阶梯 + 形状流转
    B, h, n, d_k, block = 2, 4, 1024, 64, 64
    nb = n // block
    q = torch.randn(B, h, n, d_k)
    k = torch.randn(B, h, n, d_k)
    v = torch.randn(B, h, n, d_k)
    ref = full_attention_ref(q, k, v)                                            # (B,h,n,d_k)
    out_full, shapes = block_sparse_attention(q, k, v, block, nb, return_shapes=True)
    print(f"[形状流转] {shapes}", flush=True)
    parity = {"n_blocks": nb,
              "sel_all_max_abs": float((out_full - ref).abs().max())}            # 期望 ~1e-6（仅求和序差）
    gaps = {}
    for ks in (4, 8, 12, 16):
        out_s = block_sparse_attention(q, k, v, block, ks)
        gaps[ks] = float((out_s - ref).norm() / ref.norm())                      # rel-L2：丢块的近似代价
    parity["approx_gap_rel_l2_by_ksel"] = gaps
    print(f"[对拍] k_sel=全部块({nb}) max|Δ|={parity['sel_all_max_abs']:.2e} | 近似 gap rel-L2: "
          f"{ {kk: round(vv, 4) for kk, vv in gaps.items()} }", flush=True)
    results["cpu_parity"] = parity
    results["shape_flow"] = {kk: list(vv) for kk, vv in shapes.items()}

    # ④ 解码账本 Table 4 复现（P7）：NSA 超参 l=32/d=16/l'=64/n=16/w=512（论文 §4.1 逐字）
    l_cmp, stride, l_sel, n_sel, win = 32, 16, 64, 16, 512
    table4_paper = {8192: 2048, 16384: 2560, 32768: 3584, 65536: 5632}          # Table 4 NSA 行
    table4 = {}
    for t_len, expect in table4_paper.items():
        load = t_len // stride + n_sel * l_sel + win                             # 账本口径 t/d + n·l' + w
        eq7_literal = (t_len - l_cmp) // stride                                  # Eq.7 字面 (t-l)/d（差 1-2 项）
        table4[t_len] = {"load_formula": load, "table4_nsa": expect,
                         "match": load == expect, "eq7_literal_compress": eq7_literal,
                         "full_attention_load": t_len}
        assert load == expect, f"Table 4 复现失败: t={t_len} 算得 {load} ≠ 论文 {expect}"
    results["table4_account"] = {"hyper": {"l": l_cmp, "d": stride, "l_prime": l_sel,
                                           "n": n_sel, "w": win}, "rows": table4}
    print(f"[账本] t/d+n·l'+w 四档复现 Table 4：{ {k: v['load_formula'] for k, v in table4.items()} }"
          f"（全中；Eq.7 字面 (t-l)/d={' 与表格差 1-2 项，见 JSON eq7_literal_compress'}）", flush=True)

    # ② 质量对照（MPS，fast 150 步正式档）：MiniLM T=1024，full vs 块稀疏(k_sel=8/16)
    T = 1024
    print(f"[质量] MiniLM T={T}：full vs 块稀疏(b=64, k_sel=8/16) 各 {args.steps} 步", flush=True)
    quality = {}
    for tag, attns in (
        ("full", [needle_probe.FullAttn(256, 8, 4, 64) for _ in range(4)]),
        ("bs8", [BlockSparseAttention(NSAConfig(n_selected=8)) for _ in range(4)]),
        ("bs16", [BlockSparseAttention(NSAConfig(n_selected=16)) for _ in range(4)]),
    ):
        torch.manual_seed(SEED)
        m = needle_probe.MiniLM(d=256, n_layer=4, d_ff=704, attn_modules=attns).to(device)
        bf = needle_probe.SpBatch(m4, 4, T)
        res = needle_probe.train_mini(m, args.steps, bf, device, tag=f"q-{tag}", log_every=50)
        ev = m4.eval_loss(m, device, batch=4, block=T)
        quality[tag] = {"train": res, "eval_loss": ev}
        print(f"  [{tag}] train {res['loss_first']:.3f}→{res['loss_last_20_mean']:.3f} | eval {ev:.4f} "
              f"| {res['sec_per_step_p50']:.3f} s/step", flush=True)
        del m
    results["quality_mps"] = quality
    results["note"] = ("教学实现口径（gather/scatter 未融合）：计时只作相对比值叙事（承探针 03 结论："
                       "教学实现慢于满注意力，绝对速度不作正文结论）；正文价值在机制与 gap 阶梯。")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"nsa_block_{args.out_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[产物] {path}", flush=True)
    print("nsa_block 完成。", flush=True)


if __name__ == "__main__":
    main()
