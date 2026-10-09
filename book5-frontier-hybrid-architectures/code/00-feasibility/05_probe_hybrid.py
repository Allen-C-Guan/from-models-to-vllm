# 05_probe_hybrid.py —— Book5 ch9 前置可行性探针：SlotLLaMA 改 3:1 混合（GDN 线性层 + 全局层）+ 分类型 KV/状态账
# 用途：①「注意力插槽换血」的整机冒烟——Book4 SlotLLaMA（GQA 全局底座）逐层替换注意力插槽为
#         GDN 线性层（Qwen3-Next 式 3:1：L=8 中第 3/7 层保留全局 GQA，其余 6 层换 GDN——
#         对应 full_attention_interval=4 的层型模式），小档前向+反向 10 步（MPS sec/step vs 全局底座）；
#       ② 分类型 KV/状态账本首算（本册新账，Book3/Book4 的 KV 账本扩展）：
#         全局层：KV/token = 2·h_kv·d_k 元素，n token 共 2·h_kv·d_k·n（无界，随 n 线性）；
#         滑窗层：状态上限 = 2·h_kv·d_k·w（有界，与 n 无关）；
#         线性层（GDN）：固定状态 = h_v·d_k·d_v（递归状态矩阵）+ (k_conv−1)·conv_dim（短卷积状态）
#         ——与 n 无关。三型并存的账表 + 与全全局底座的交叉点。
# 所属章节：Book5 第 9 章（2026 混合配方谱系的整机底座）与第 5 章（GDN 插槽落地）共用。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/00-feasibility/05_probe_hybrid.py" [--out-name run1]
#   MPS bf16 autocast；预计 2-3 分钟。
# 产物：log/book5-feasibility/probe05_hybrid_{out}.json（不入库）
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
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
    p = os.path.join(HERE, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save_json(name, obj, out_name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}_{out_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[产物] {path}")
    return path


class GDNLayer(nn.Module):
    """GDN 注意力插槽（Qwen3-Next GatedDeltaNet 教学骨架；无 rope——线性层位置自由）。

    结构（键位对齐 HF Qwen3NextGatedDeltaNet，ch08 对拍用同族）：
      in_proj_qkvz: d → 2·h_k·d_k + 2·h_v·d_v（q/k/v/z 融合投影）
      in_proj_ba:   d → 2·h_v（β 与门 a 输入）
      conv1d:       因果短卷积 (k_conv=4, groups=conv_dim, silu)
      A_log/dt_bias: 门 g = −exp(A_log)·softplus(a + dt_bias) ≤ 0（04 探针 R1）
      核心算子: gdn_chunkwise（04 探针——已与 HF 参考对拍到 1e-7 级）
      out_proj: h_v·d_v → d
    forward(x, cos, sin) 兼容 SlotDecoderLayer 的注意力插槽签名（忽略 cos/sin）。
    """

    def __init__(self, d, h_k, h_v, d_k, d_v, conv_kernel=4, chunk=64):
        super().__init__()
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

    def forward(self, x, cos=None, sin=None):
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
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)      # (B,n,h_v) ≤ 0
        core, _ = p04.gdn_chunkwise(q, k, v, g, beta, chunk=self.chunk)           # (B,n,h_v,d_v)
        # 输出门控 RMSNorm（z 为门；HF RMSNormGated 同构）
        core = core.reshape(-1, self.d_v)
        z2 = z.reshape(-1, self.d_v)
        var = core.float().pow(2).mean(-1, keepdim=True)
        core = core * torch.rsqrt(var + 1e-6) * F.silu(z2.float())
        core = core.view(B, n, self.h_v * self.d_v)                              # (B,n,h_v·d_v)
        return self.out_proj(core)                                               # (B,n,d)


def make_hybrid(m4, cfg, global_at="gqa", pattern=None):
    """SlotLLaMA 构建后逐层换注意力插槽。pattern: 长度 L 的列表，"linear"→GDNLayer，"global"→保留。"""
    model = m4.SlotLLaMA(cfg)
    for i, kind in enumerate(pattern):
        if kind == "linear":
            model.model.layers[i].self_attn = GDNLayer(
                cfg.hidden_size, h_k=8, h_v=8, d_k=64, d_v=64).to(
                next(model.parameters()).dtype)
    return model


def main():
    ap = argparse.ArgumentParser(description="Book5 3:1 混合整机冒烟 + 分类型账本")
    ap.add_argument("--out-name", default="run1")
    args = ap.parse_args()
    global p04
    m4, hits = bootstrap_book4()
    p04 = load_sibling("04_probe_gdn")
    device = m4.pick_device("mps")
    print(f"[bootstrap] moe_mla_slots 自：{hits[0]} | 04 探针 gdn_chunkwise 复用 | device {device}")
    torch.manual_seed(SEED)

    L = 8
    cfg = m4.SlotConfig(vocab_size=8192, hidden_size=512, num_hidden_layers=L,
                        num_attention_heads=8, num_key_value_heads=4, intermediate_size=1408)
    pattern = ["linear" if (i + 1) % 4 else "global" for i in range(L)]           # 3:1——第 3/7 层全局
    print(f"[层型模式] {pattern}（full_attention_interval=4 同构，Qwen3-Next 式）")

    # ① 基线（全全局 GQA）10 步
    torch.manual_seed(SEED)
    base = m4.SlotLLaMA(cfg).to(device)
    secs_b, losses_b, _ = m4.train_loop(base, steps=10, batch=8, block=512, device=device,
                                        tokens_path=m4.TOKENS_SP8K, peak_lr=1e-3, warmup=2)
    print(f"[全全局底座] {base.n_params():,} 参 | {np.median(secs_b):.3f} s/step | loss {losses_b[0]:.3f}→{losses_b[-1]:.3f}")

    # ② 3:1 混合 10 步 + 梯度流检查
    torch.manual_seed(SEED)
    hyb = make_hybrid(m4, cfg, pattern=pattern).to(device)
    assert hyb.model.layers[0].self_attn.__class__.__name__ == "GDNLayer"
    secs_h, losses_h, _ = m4.train_loop(hyb, steps=10, batch=8, block=512, device=device,
                                        tokens_path=m4.TOKENS_SP8K, peak_lr=1e-3, warmup=2)
    g = hyb.model.layers[0].self_attn.in_proj_qkvz.weight.grad
    grad_ok = g is not None and torch.isfinite(g).all().item()
    print(f"[3:1 混合] {hyb.n_params():,} 参 | {np.median(secs_h):.3f} s/step | "
          f"loss {losses_h[0]:.3f}→{losses_h[-1]:.3f} | GDN 梯度有限={grad_ok}")

    # ③ 分类型 KV/状态账本（新账首算：全局无界 / 滑窗有界 / 线性固定）
    h_kv, d_k = cfg.num_key_value_heads, cfg.head_dim
    h_v, d_v, conv_k, conv_dim = 8, 64, 4, 2 * 8 * 64 + 8 * 64
    n_gdn, n_glo = pattern.count("linear"), pattern.count("global")
    ledger = {
        "per_layer_formulas": {
            "global_gqa_kv_per_token_elems": 2 * h_kv * d_k,                     # 512——随 n 线性（无界）
            "swa_bounded_elems(if 窗 w)": f"2·{h_kv}·{d_k}·w（有界，与 n 无关）",
            "linear_gdn_fixed_state_elems": h_v * d_k * d_v + (conv_k - 1) * conv_dim,  # 固定
        },
        "linear_gdn_fixed_state_elems": h_v * d_k * d_v + (conv_k - 1) * conv_dim,
        "totals_by_n": {},
    }
    print("[分类型账本] 每 token/每层元素数：全局=512（无界）| 线性层固定状态="
          f"{ledger['linear_gdn_fixed_state_elems']:,}（递归 {h_v*d_k*d_v:,} + 卷积 {(conv_k-1)*conv_dim:,}）")
    for n in (512, 2048, 8192, 32768, 131072):
        all_global = L * 2 * h_kv * d_k * n
        hybrid = n_glo * 2 * h_kv * d_k * n + n_gdn * ledger["linear_gdn_fixed_state_elems"]
        ledger["totals_by_n"][n] = {"all_global_elems": all_global, "hybrid_3to1_elems": hybrid,
                                    "ratio": round(hybrid / all_global, 4)}
        print(f"  n={n:>6}: 全全局 {all_global:>12,} | 3:1 混合 {hybrid:>10,} | 比率 {hybrid/all_global:.3f}")
    crossover = n_gdn * ledger["linear_gdn_fixed_state_elems"] / (n_glo * 2 * h_kv * d_k) \
        if n_glo else None
    # 交叉点：n_glo·512·n > n_gdn·fixed ⇔ n > n_gdn·fixed/(n_glo·512)——本配置恒小（混合在一切 n 下省）
    print(f"  [口径] 混合架构状态账恒小于全全局（交叉 n = {crossover:.1f}，n>此值即混合占优）")

    save_json("probe05_hybrid", {
        "seed": SEED, "date": "2026-10-05", "L": L, "pattern": pattern,
        "baseline": {"params": base.n_params(), "sec_per_step_p50": float(np.median(secs_b)),
                     "loss_last": losses_b[-1]},
        "hybrid": {"params": hyb.n_params(), "sec_per_step_p50": float(np.median(secs_h)),
                   "loss_last": losses_h[-1], "gdn_grad_finite": bool(grad_ok)},
        "ledger": ledger,
    }, args.out_name)
    print("探针 05 完成。")


if __name__ == "__main__":
    main()
