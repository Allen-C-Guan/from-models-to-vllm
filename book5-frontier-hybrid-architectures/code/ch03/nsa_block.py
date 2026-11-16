# nsa_block.py —— Book5 第3章：因果块选择教学件，并非完整 NSA 三分支。
# 运行：source env.sh && python code/ch03/nsa_block.py --out-name verify
# 默认仅做 CPU 算子检查与 Table4 整数复算；--steps 150 才执行语料质量实验。
# 历史前视版本的gap、训练和计时只保留在正文历史记录，不作为当前版本验收阈值。
# 当前选择：块首query给严格过去块打分，固定保留当前块；每头独选，未实现论文GQA组共享选择。
# 不含可学压缩MLP、摘要输出、门及滑窗支路；没有完整NSA的压缩分支梯度路径。
# 产物 log/book5-ch03/，拒覆盖；训练使用第2章同日程checkpoint，--resume恢复已有各臂进度。
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = str(next((p for p in Path(__file__).resolve().parents if (p / "env.sh").is_file()), Path.cwd()))
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
    m4, hits = needle_probe.bootstrap_book4()
    return m4, needle_probe, hits


# ---------------- 块稀疏注意力核心（op 级，含形状流转） ----------------
def block_sparse_attention(q, k, v, block, n_selected, return_shapes=False):
    """块级 top-k 选择注意力（NSA 选择分支教学骨架）。

    输入：q/k (B,h,n,d_k)、v (B,h,n,d_v)（已展开到 h 头、已过 rope）；
    选择分只用查询块首query与严格过去块的均值键；当前块固定保留；
    选中块 gather 全精度 K/V，块内做带因果位的注意力。返回 (B,h,n,d_v)。"""
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("q/k应为同形(B,h,n,d_k)，v为(B,h,n,d_v)")
    if block < 1 or n_selected < 1 or q.shape[2] < 1:
        raise ValueError("block、n_selected 与序列长度必须为正")
    B, h, n, d_k = q.shape
    d_v = v.shape[-1]
    nb = n // block
    assert n == nb * block, f"n={n} 需为 block={block} 的倍数"
    scale = d_k ** -0.5

    qb = q.reshape(B, h, nb, block, d_k)                                            # (B,h,nb,b,d_k) 查询块
    kb = k.reshape(B, h, nb, block, d_k)                                            # (B,h,nb,b,d_k) 键块
    vb = v.reshape(B, h, nb, block, d_v)                                            # (B,h,nb,b,d_v) 值块
    k_mean = kb.mean(dim=3)                                                      # (B,h,nb,d_k) 块压缩键（NSA 压缩分支近似）

    # 当前块固定保留，选择只由块首query和严格过去的完整键块决定。
    # 因而改变块首之后的query/key不会改变较早位置的参与集合。
    blk_scores = torch.einsum("zhqd,zhkd->zhqk", qb[:, :, :, 0], k_mean) * scale
    k_sel = max(1, min(n_selected, nb))
    choices = []
    for i in range(nb):
        cur = torch.full((B, h, 1), i, dtype=torch.long, device=q.device)
        count = min(k_sel - 1, i)
        past = blk_scores[:, :, i, :i].argsort(dim=-1, descending=True, stable=True)[..., :count] if count else cur[..., :0]
        # 早期不足预算时以未来块占位；最终逐位置mask完全排除这些占位。
        pad = torch.arange(i + 1, i + 1 + k_sel - 1 - count, device=q.device)
        pad = pad.view(1, 1, -1).expand(B, h, -1)
        choices.append(torch.cat([cur, past, pad], dim=-1))
    sel_idx = torch.stack(choices, dim=2)  # (B,h,nb,k_sel)，每行索引不重复

    # gather 键/值块：torch.gather 不能增维——按选中槽位逐个 gather（k_sel ≤ nb，小循环清晰）
    def gather_blocks(xb):
        parts = [xb.gather(2, sel_idx[..., i:i + 1].unsqueeze(-1)
                           .expand(B, h, nb, block, xb.shape[-1]))
                 for i in range(k_sel)]                                          # 各 (B,h,nb,block,d_k)
        return torch.stack(parts, dim=3).reshape(B, h, nb, k_sel * block, xb.shape[-1])
    kg = gather_blocks(kb)                                                       # (B,h,nb,k_sel·b,d_k) gather 键
    vg = gather_blocks(vb)                                                       # (B,h,nb,k_sel·b,d_v) gather 值

    att = torch.einsum("zhqwd,zhqsd->zhqws", qb, kg) * scale                     # (B,h,nb,b,k_sel·b) 块内分数
    # 因果位掩码：键全局位 = 块号·b + 块内位；查询全局位同理；allowed = kp ≤ qp（广播）
    kp = (sel_idx.unsqueeze(-1) * block + torch.arange(block, device=q.device)).view(B, h, nb, 1, k_sel * block)
    qp = (torch.arange(nb, device=q.device).view(nb, 1) * block + torch.arange(block, device=q.device)).view(1, 1, nb, block, 1)
    allowed = kp <= qp                                                           # (B,h,nb,b,k_sel·b)
    att = att.masked_fill(~allowed, float("-inf"))
    p = F.softmax(att, dim=-1, dtype=torch.float32 if att.dtype in (torch.float16, torch.bfloat16) else att.dtype).to(v.dtype)                               # 低精度用FP32归一，再转回V的dtype
    out = torch.einsum("zhqws,zhqsd->zhqwd", p, vg)                              # (B,h,nb,b,d_v)
    out = out.reshape(B, h, n, d_v)                                              # (B,h,n,d_v)
    if return_shapes:
        shapes = {"blk_scores": tuple(blk_scores.shape), "sel_idx": tuple(sel_idx.shape),
                  "gathered_k": tuple(kg.shape), "intra_att": tuple(att.shape), "out": tuple(out.shape)}
        return out, shapes
    return out


def full_attention_ref(q, k, v):
    """满因果注意力（对拍参照）：q/k (B,h,n,d_k)、v (B,h,n,d_v) → (B,h,n,d_v)。"""
    n = q.shape[2]
    att = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)                      # (B,h,n,n)
    mask = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()
    att = att.masked_fill(~mask, float("-inf"))
    p = F.softmax(att, dim=-1, dtype=torch.float32 if att.dtype in (torch.float16, torch.bfloat16) else att.dtype).to(v.dtype)
    return p @ v                                                                # (B,h,n,d_v)


# ---------------- 注意力插槽（MiniLM 兼容：forward(x, cos, sin) → (B,n,d)） ----------------
@dataclass
class NSAConfig:
    """块稀疏插槽 config（大纲 ch3 插槽契约：BlockSparseAttention(cfg) 单参构造）。"""
    d: int = 256
    n_head: int = 8
    n_kv_head: int = 4
    d_k: int = 64
    block: int = 64            # 块宽 b
    n_selected: int = 8        # 每查询块最多k_sel个键块（包含固定当前块）
    use_rope: bool = True


    def __post_init__(self):
        if min(self.d,self.n_head,self.n_kv_head,self.d_k,self.block,self.n_selected)<1:
            raise ValueError("维度及选择预算必须为正")
        if self.n_head % self.n_kv_head or (self.use_rope and self.d_k % 2):
            raise ValueError("GQA分组或RoPE维度无效")


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
    ap.add_argument("--steps", type=int, default=0, help="默认0只检算子；指定正数才运行语料训练")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.steps<0:
        raise ValueError("steps不得为负")
    os.makedirs(OUT_DIR,exist_ok=True)
    path = os.path.join(OUT_DIR,f"nsa_block_{args.out_name}.json")
    if os.path.exists(path):
        raise FileExistsError(path)
    m4, needle_probe, hits = bootstrap()
    device = m4.pick_device("mps")
    print(f"[bootstrap] moe_mla_slots 自：{hits[0]} | needle_probe 复用（ch02 兄弟章）| device {device}",
          flush=True)
    torch.manual_seed(SEED)
    results = {"seed": SEED, "date": datetime.now(timezone.utc).isoformat(), "steps": args.steps,
               "torch": torch.__version__, "device": str(device), "implementation": "causal_block_first_query_v2"}

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
    assert parity["sel_all_max_abs"] < 2e-6
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
        eq7_literal = max(0, (t_len - l_cmp) // stride + 1)                                  # Eq.7闭区间枚举含起点0
        table4[t_len] = {"load_formula": load, "table4_nsa": expect,
                         "match": load == expect, "eq7_literal_compress": eq7_literal,
                         "full_attention_load": t_len}
        assert load == expect, f"Table 4 复现失败: t={t_len} 算得 {load} ≠ 论文 {expect}"
    results["table4_account"] = {"hyper": {"l": l_cmp, "d": stride, "l_prime": l_sel,
                                           "n": n_sel, "w": win}, "rows": table4}
    print(f"[账本] t/d+n·l'+w 四档复现 Table 4：{ {k: v['load_formula'] for k, v in table4.items()} }"
          f"（全中；Eq.7完整块数比t/d少1项，见 JSON eq7_literal_compress）", flush=True)

    quality = {}
    if args.steps > 0:
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
            ckpath = os.path.join(OUT_DIR, f"nsa_block_{args.out_name}_{tag}.pt")
            res = needle_probe.train_mini(m, args.steps, bf, device, tag=f"q-{tag}", log_every=50,
                                          checkpoint_path=ckpath,
                                          resume=args.resume and os.path.exists(ckpath))
            ev = m4.eval_loss(m, device, batch=4, block=T)
            quality[tag] = {"train": res, "eval_loss": ev}
            print(f"  [{tag}] train {res['loss_first']:.3f}→{res['loss_last_20_mean']:.3f} | eval {ev:.4f} "
                  f"| {res['sec_per_step_p50']:.3f} s/step", flush=True)
            del m
    results["quality"] = quality
    results["note"] = ("教学实现口径（gather/scatter 未融合）：计时只作相对比值叙事（承探针 03 结论："
                       "教学实现慢于满注意力，绝对速度不作正文结论）；正文价值在机制与 gap 阶梯。")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"nsa_block_{args.out_name}.json")
    with open(path, "x", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[产物] {path}", flush=True)
    print("nsa_block 完成。", flush=True)


if __name__ == "__main__":
    main()
