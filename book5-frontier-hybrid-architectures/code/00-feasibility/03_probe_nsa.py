# 第二轮修订：核心改为复用ch03因果函数；旧前视版本的历史计时不由当前版本重现。
# 03_probe_nsa.py —— Book5 ch3 实验可行性探针：NSA 式块稀疏注意力（blockwise top-k 选择 + gather）
# 用途：ch3（路线二·可学习稀疏：NSA→DSA）的机制件在本机是否成立——
#   ① 机制件：块级 top-k 选择注意力——「块压缩分数 → 因果资格过滤 → top-k 块选择 →
#      gather 全精度块 → 块内注意力」的 blockwise gather-scatter 实现（NSA 选择分支的教学骨架；
#      逐运算形状注释 + 形状流转打印：选择矩阵/块聚合/gather 后张量的形状）；
#   ② 正确性（CPU fp32）：k_sel=全部块 时与满注意力逐位对拍（max|Δ|）；k_sel<全部 时量化
#      近似 gap（rel-L2）——「丢块 = 有损压缩」的现场读数；
#   ③ 质量（MPS）：MiniLM 小档 full vs 块稀疏 各 150 步（sp-8k, T=1024）——可训性对照；
#   ④ 计时（MPS）：n=2048 前向 wall-clock：full vs k_sel=8/16（块 64）。
# 口径声明：本实现是 NSA「选择分支」的教学骨架（压缩分支用块均值键近似、滑窗分支不单独实现），
#   与 NSA 论文（候选 arXiv 2502.11089?，三查后以核验为准）的三分支完整形差异在正文展开。
# 所属章节：Book5 第 3 章（可学习稀疏正身实验的底座）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/00-feasibility/03_probe_nsa.py" [--out-name run1]
#   正确性 CPU fp32；训练/计时 MPS bf16；预计 3-5 分钟。
# 产物：log/book5-feasibility/probe03_nsa_{out}.json（不入库）
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = str(next((p for p in Path(__file__).resolve().parents if (p / "env.sh").is_file()), Path.cwd()))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book5-feasibility")
SEED = 20261002

_B4_FEAS_CANDS = [
    os.path.join(REPO_ROOT, "code", "Book4-规模与效率", "00-feasibility"),
    os.path.join(REPO_ROOT, "from-models-to-vllm", "book4-scale-and-efficiency", "code", "00-feasibility"),
    os.path.join(REPO_ROOT, "book4-scale-and-efficiency", "code", "00-feasibility"),
]


def bootstrap_book4():
    hits = [p for p in _B4_FEAS_CANDS if os.path.exists(os.path.join(p, "moe_mla_slots.py"))]
    if not hits:
        raise FileNotFoundError(_B4_FEAS_CANDS)
    sys.path.insert(0, hits[0])
    import moe_mla_slots as m4
    return m4, hits


def load_sibling(name):
    """同目录探针模块装载（文件路径定位，避免数字开头的模块名不可 import 问题）。"""
    p = os.path.join(HERE, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save_json(name, obj, out_name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}_{out_name}.json")
    with open(path, "x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[产物] {path}")
    return path


# ---------------- 块稀疏注意力核心（op 级，含形状流转） ----------------
def block_sparse_attention(q, k, v, block, n_selected, return_shapes=False):
    """复用正式因果核心；旧前视版本仅在冻结底本中留存。"""
    sys.path.insert(0,str(Path(HERE).parent/"ch03"))
    from nsa_block import block_sparse_attention as causal
    return causal(q,k,v,block,n_selected,return_shapes=return_shapes)


def full_attention_ref(q, k, v):
    """满因果注意力（对拍参照）：q,k,v (B,h,n,d_k) → (B,h,n,d_k)。"""
    n = q.shape[2]
    att = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)                      # (B,h,n,n)
    mask = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()
    att = att.masked_fill(~mask, float("-inf"))
    p = F.softmax(att.float(), dim=-1).to(q.dtype)
    return p @ v                                                                # (B,h,n,d_k)


# ---------------- 注意力插槽（MiniLM 兼容：forward(x, cos, sin)） ----------------
class BlockSparseAttn(nn.Module):
    """块稀疏注意力插槽：GQA 四投影 + rope + 块级 top-k 选择（核心调 block_sparse_attention）。"""

    def __init__(self, d, n_head, n_kv_head, d_k, block, n_selected):
        super().__init__()
        self.h, self.h_kv, self.d_k = n_head, n_kv_head, d_k
        self.block, self.n_selected = block, n_selected
        self.q_proj = nn.Linear(d, n_head * d_k, bias=False)
        self.k_proj = nn.Linear(d, n_kv_head * d_k, bias=False)
        self.v_proj = nn.Linear(d, n_kv_head * d_k, bias=False)
        self.o_proj = nn.Linear(n_head * d_k, d, bias=False)

    def forward(self, x, cos, sin):
        B, n, d = x.shape
        q = self.q_proj(x).view(B, n, self.h, self.d_k).transpose(1, 2)          # (B,h,n,d_k)
        k = self.k_proj(x).view(B, n, self.h_kv, self.d_k).transpose(1, 2)       # (B,h_kv,n,d_k)
        v = self.v_proj(x).view(B, n, self.h_kv, self.d_k).transpose(1, 2)       # (B,h_kv,n,d_k)
        q, k = p02.apply_rope(q, cos, sin), p02.apply_rope(k, cos, sin)
        rep = self.h // self.h_kv
        k = k.repeat_interleave(rep, dim=1)                                      # (B,h,n,d_k)
        v = v.repeat_interleave(rep, dim=1)                                      # (B,h,n,d_k)
        y = block_sparse_attention(q, k, v, self.block, self.n_selected)         # (B,h,n,d_k)
        y = y.transpose(1, 2).contiguous().view(B, n, self.h * self.d_k)         # (B,n,h·d_k)
        return self.o_proj(y)                                                    # (B,n,d)


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="Book5 ch3 NSA 式块稀疏可行性探针")
    ap.add_argument("--out-name", default="run1")
    args = ap.parse_args()
    output = os.path.join(OUT_DIR,f"probe03_nsa_{args.out_name}.json")
    if os.path.exists(output):
        raise FileExistsError(output)
    global p02
    m4, hits = bootstrap_book4()
    p02 = load_sibling("02_probe_swa_sink")                                      # 复用 FullAttn/MiniLM/训练环
    device = m4.pick_device("mps")
    print(f"[bootstrap] moe_mla_slots 自：{hits[0]} | 02 探针复用（FullAttn/MiniLM）| device {device}")
    torch.manual_seed(SEED)
    results = {"seed": SEED, "date": datetime.now(timezone.utc).isoformat(), "device": str(device), "implementation": "causal_core_from_ch03"}

    # ① 正确性（CPU fp32）：k_sel=nb 逐位对拍 + k_sel<nb 近似 gap + 形状流转
    #   （run1 教训：n=512/block=64 只有 8 块——k_sel=8/16 被钳到全块，gap 阶梯退化；改 n=1024 → 16 块）
    B, h, n, d_k, block = 2, 4, 1024, 64, 64
    nb = n // block
    q = torch.randn(B, h, n, d_k)
    k = torch.randn(B, h, n, d_k)
    v = torch.randn(B, h, n, d_k)
    ref = full_attention_ref(q, k, v)                                            # (B,h,n,d_k)
    _, shapes = block_sparse_attention(q, k, v, block, nb, return_shapes=True)
    print(f"[形状流转] {shapes}")
    parity = {"n_blocks": nb}
    out_full = block_sparse_attention(q, k, v, block, nb)
    parity["sel_all_max_abs"] = float((out_full - ref).abs().max())              # 期望 ~1e-6（仅求和序差）
    gaps = {}
    for ks in (4, 8, 12):
        out_s = block_sparse_attention(q, k, v, block, ks)
        gaps[ks] = float((out_s - ref).norm() / ref.norm())                      # rel-L2：丢块的近似代价
    parity["approx_gap_rel_l2_by_ksel"] = gaps
    print(f"[对拍] k_sel=全部块({nb}) max|Δ|={parity['sel_all_max_abs']:.2e} | "
          f"近似 gap rel-L2: { {k: round(v, 4) for k, v in gaps.items()} }")
    results["cpu_parity"] = parity
    results["shape_flow"] = {k: list(v) for k, v in shapes.items()}

    # ② 质量（MPS）：MiniLM full vs 块稀疏 各 150 步（sp-8k, T=1024）
    print("[质量] MiniLM T=1024：full vs 块稀疏(b=64, k_sel=8/16) 各 150 步")
    T = 1024
    cfg_common = dict(d=256, L=4, h=8, h_kv=4, d_k=64, d_ff=704)
    quality = {}
    for tag, attns in (
        ("full", [p02.FullAttn(256, 8, 4, 64) for _ in range(4)]),
        ("bs8", [BlockSparseAttn(256, 8, 4, 64, 64, 8) for _ in range(4)]),
        ("bs16", [BlockSparseAttn(256, 8, 4, 64, 64, 16) for _ in range(4)]),
    ):
        torch.manual_seed(SEED)
        m = p02.MiniLM(d=256, n_layer=4, d_ff=704, attn_modules=attns).to(device)
        bf = p02.SpBatch(m4, 4, T)
        res = p02.train_mini(m, 150, bf, device, tag=f"q-{tag}")
        ev = m4.eval_loss(m, device, batch=4, block=T)
        quality[tag] = {"train": res, "eval_loss": ev}
        print(f"  [{tag}] train {res['loss_first']:.3f}→{res['loss_last_20_mean']:.3f} | eval {ev:.4f} "
              f"| {res['sec_per_step_p50']:.3f} s/step")
        del m
    results["quality_mps"] = quality

    # ③ 计时（MPS）：n=2048 前向 wall-clock
    print("[计时] n=2048 前向（B=2, h=8, d_k=64，20 次取中位）")
    B2, h2, n2, d2 = 2, 8, 2048, 64
    q2 = torch.randn(B2, h2, n2, d2, device=device)
    k2 = torch.randn(B2, h2, n2, d2, device=device)
    v2 = torch.randn(B2, h2, n2, d2, device=device)
    timing = {}
    with torch.no_grad():
        for tag, fn in (
            ("full", lambda: full_attention_ref(q2, k2, v2)),
            ("bs_k8", lambda: block_sparse_attention(q2, k2, v2, 64, 8)),
            ("bs_k16", lambda: block_sparse_attention(q2, k2, v2, 64, 16)),
        ):
            for _ in range(3):
                fn()
            if device.type == "mps":
                torch.mps.synchronize()
            ts = []
            for _ in range(20):
                t0 = time.perf_counter()
                fn()
                if device.type == "mps":
                    torch.mps.synchronize()
                ts.append(time.perf_counter() - t0)
            timing[tag] = float(np.median(ts))
            print(f"  [{tag}] {timing[tag]*1000:.1f} ms")
    results["timing_n2048_ms"] = timing
    results["note"] = ("计时为教学实现口径（gather/scatter 未融合），相对比值供参考；"
                       "绝对速度与融合 kernel 不可比（纪律：不教 FlashAttention 机制）。")

    save_json("probe03_nsa", results, args.out_name)
    print("探针 03 完成。")


if __name__ == "__main__":
    main()
