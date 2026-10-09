# 06_probe_k3toy.py —— Book5 ch6/ch7 大项目前置探针：K3 玩具替身 config 手算账 + 冒烟 + MXFP4 钩子
# 用途：K3 玩具替身（20-50M，ch6/ch7 正菜）的定档依据——
#   ① 2-3 个候选 config 的参数逐项手算账（总参/激活参双口径）+ assert（与实测逐位对账）；
#   ② 各候选 MPS 前向+反向 10 步 sec/step（可训性冒烟）；
#   ③ MXFP4 模拟量化钩子的训练循环嵌入方式验证（复用 Book4 ch07 fp8_sim.mxfp4_rt；
#      STE 直通估计器——前向用块量化权重、反传走恒等；量 MoE 专家权重，量/不量的开销差）；
#   ④ 给正式替身定档建议（精确 config 字段值）。
# 【替身口径声明】KDA（Kimi Delta Attention）正身以 K3 官方报告逐节精读为准（另路调研）；
#   本探针以 GDN（gated delta rule，Qwen3-Next 同族、HF 5.18.0 参考实现已对拍）作 delta-rule 族
#   可行性替身——transformers kimi_linear 文档自证「KDA essentially the same as GDN but decay is
#   per-channel instead of per-token」：门粒度差异（per-token vs per-channel）是 K3 报告精读后
#   唯一预期改动点，组件几何（qkv 投影/短卷积/递归状态/输出门控 norm）不变。
#   其余件：Gated MLA = Book4 MLAAttention 复用（Gated 变体的门在 K3 精读后加）；
#   超稀疏 MoE = 64 选 4 + 共享 1（K3 896 选 16 的 1/14 比例缩样）；MXFP4 = fp8_sim 块模拟。
# 所属章节：Book5 第 6/7 章（K3 玩具替身大项目的定档底座）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/00-feasibility/06_probe_k3toy.py" [--out-name run1]
#   MPS bf16 autocast；预计 3-5 分钟。
# 产物：log/book5-feasibility/probe06_k3toy_{out}.json（不入库）
import argparse
import importlib.util
import json
import os
import sys

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
_B4_CH07_CANDS = [
    os.path.join(REPO_ROOT, "code", "Book4-规模与效率", "ch07"),
    os.path.join(REPO_ROOT, "from-models-to-vllm", "book4-scale-and-efficiency", "code", "ch07"),
]


def bootstrap_book4():
    hits = [p for p in _B4_FEAS_CANDS if os.path.exists(os.path.join(p, "moe_mla_slots.py"))]
    if not hits:
        raise FileNotFoundError(_B4_FEAS_CANDS)
    sys.path.insert(0, hits[0])
    import moe_mla_slots as m4
    return m4, hits


def bootstrap_fp8sim():
    hits = [p for p in _B4_CH07_CANDS if os.path.exists(os.path.join(p, "fp8_sim.py"))]
    if not hits:
        raise FileNotFoundError(_B4_CH07_CANDS)
    sys.path.insert(0, hits[0])
    import fp8_sim
    return fp8_sim, hits


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


# ---------------- K3 玩具替身：整机装配（KDA 替身=GDN 插槽 + MLA 插槽 + 超稀疏 MoE 插槽） ----------------
MLA_GEOM = dict(kv_lora_rank=128, qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64,
                q_lora_rank=None)                                               # q 侧直投（轻档）


def k3_hand_account(V, d, L, pattern, E, top_k, w_e, n_shared, h_k, h_v, d_k, d_v, conv_k=4):
    """参数逐项手算（整数精确）。pattern: "kda"|"mla" 逐层。返回明细 dict。"""
    conv_dim = 2 * h_k * d_k + h_v * d_v
    kda_attn = (d * (2 * h_k * d_k + 2 * h_v * d_v)                              # in_proj_qkvz
                + d * 2 * h_v                                                    # in_proj_ba
                + conv_dim * conv_k                                              # conv1d
                + h_v + h_v                                                      # A_log + dt_bias
                + h_v * d_v * d)                                                 # out_proj
    m = MLA_GEOM
    h = 8
    mla_attn = (d * h * (m["qk_nope_head_dim"] + m["qk_rope_head_dim"])          # q_proj
                + d * (m["kv_lora_rank"] + m["qk_rope_head_dim"]) + m["kv_lora_rank"]   # kv_a + LN
                + m["kv_lora_rank"] * h * (m["qk_nope_head_dim"] + m["v_head_dim"])    # kv_b
                + h * m["v_head_dim"] * d)                                       # o
    moe = E * d + E * 3 * d * w_e + n_shared * 3 * d * w_e                       # 门 + 路由专家 + 共享
    per_layer, detail = [], {"kda_attn": kda_attn, "mla_attn": mla_attn, "moe": moe}
    for kind in pattern:
        norms = 2 * d                                                            # mla 的潜向量 LN 已计入 mla_attn
        per_layer.append((kda_attn if kind == "kda" else mla_attn) + moe + norms)
    emb = 2 * V * d + d                                                          # untied 两份 + 终态 norm
    total = sum(per_layer) + emb
    act_body = sum((kda_attn if k == "kda" else mla_attn) + 3 * d * (top_k * w_e + n_shared * w_e)
                   + 2 * d for k in pattern) + d
    return {"total": total, "activated_with_emb": act_body + 2 * V * d,
            "activated_non_emb": act_body, "detail": detail, "emb": emb}


def build_k3toy(m4, p05, V, d, L, pattern, E, top_k, w_e, n_shared):
    """SlotLLaMA 底座装配：MoE 全层（Book4 第一刀）；注意力插槽逐层换 KDA 替身（GDN）/ MLA（Book4 第二刀）。"""
    cfg = m4.SlotConfig(vocab_size=V, hidden_size=d, num_hidden_layers=L, num_attention_heads=8,
                        num_key_value_heads=4, intermediate_size=1408,
                        moe=dict(n_expert=E, top_k=top_k, expert_dim=w_e, n_shared=n_shared,
                                 mode="mixtral", variant="teach"))
    model = m4.SlotLLaMA(cfg)
    for i, kind in enumerate(pattern):
        if kind == "kda":
            model.model.layers[i].self_attn = p05.GDNLayer(d, h_k=8, h_v=8, d_k=64, d_v=64)
        else:
            model.model.layers[i].self_attn = m4.MLAAttention(
                d, 8, MLA_GEOM["kv_lora_rank"], MLA_GEOM["qk_nope_head_dim"],
                MLA_GEOM["qk_rope_head_dim"], MLA_GEOM["v_head_dim"],
                q_lora_rank=MLA_GEOM["q_lora_rank"], rope_theta=10000.0)
    return model, cfg


# ---------------- MXFP4 STE 钩子（复用 Book4 ch07 fp8_sim） ----------------
class MXFP4Linear(nn.Module):
    """MXFP4 模拟量化线性层（STE 直通）：前向用块量化权重（OCP 口径 32 元素块 + E8M0 共享指数），
    反传对原权重恒等。weight 与被替换的 nn.Linear 共享同一 Parameter（优化器无缝）。"""

    def __init__(self, lin, block=32):
        super().__init__()
        self.weight = lin.weight                                                 # 共享 Parameter（不复制）
        self.block = block

    def forward(self, x):
        w_q, _ = fp8.mxfp4_rt(self.weight.data, self.block)                      # (out,in) 量化-反量化
        w = self.weight + (w_q - self.weight).detach()                           # STE：前向 w_q、反传恒等
        return F.linear(x, w)


def apply_mxfp4_to_experts(model, m4):
    """把所有 SparseMoE 专家的 12 个 Linear 换成 MXFP4Linear（路由门/共享专家/注意力不量——K3 侧口径）。"""
    n_swapped = 0
    for layer in model.model.layers:
        mlp = layer.mlp
        if isinstance(mlp, m4.SparseMoE):
            for e in mlp.experts:
                e.gate_proj = MXFP4Linear(e.gate_proj)
                e.up_proj = MXFP4Linear(e.up_proj)
                e.down_proj = MXFP4Linear(e.down_proj)
                n_swapped += 3
    return n_swapped


# ---------------- 候选与主流程 ----------------
CANDIDATES = {
    "toyA_L8_E64t4s1_w32": dict(L=8, pattern=["kda"] * 8, E=64, top_k=4, w_e=32, n_shared=1),
    "toyB_L8_E32t2s1_w48": dict(L=8, pattern=["kda"] * 8, E=32, top_k=2, w_e=48, n_shared=1),
    "toyC_L6_E64t4s1_w64": dict(L=6, pattern=["kda"] * 6, E=64, top_k=4, w_e=64, n_shared=1),
}


def inject_global(pattern, interval=4):
    """按 full_attention_interval 把全局层（MLA）注入 pattern（第 interval-1, 2·interval-1, ... 层）。"""
    return ["mla" if (i + 1) % interval == 0 else k for i, k in enumerate(pattern)]


def main():
    ap = argparse.ArgumentParser(description="Book5 K3 玩具替身定档探针")
    ap.add_argument("--out-name", default="run1")
    args = ap.parse_args()
    global fp8
    m4, hits = bootstrap_book4()
    fp8, ch07_hits = bootstrap_fp8sim()
    p05 = load_sibling("05_probe_hybrid")
    p05.p04 = load_sibling("04_probe_gdn")                                       # GDNLayer 依赖 04 的 gdn_chunkwise
    device = m4.pick_device("mps")
    print(f"[bootstrap] moe_mla_slots 自 {hits[0]} | fp8_sim 自 {ch07_hits[0]} | device {device}")
    torch.manual_seed(SEED)

    V, d = 8192, 512
    h_k = h_v = 8
    results = {"seed": SEED, "date": "2026-10-05", "candidates": {}}

    # ① 手算账 + assert + 10 步冒烟
    for name, c in CANDIDATES.items():
        pattern = inject_global(list(c["pattern"]))                              # 3:1 注入 MLA 全局层
        hand = k3_hand_account(V, d, c["L"], pattern, c["E"], c["top_k"], c["w_e"],
                               c["n_shared"], h_k, h_v, 64, 64)
        torch.manual_seed(SEED)
        model, cfg = build_k3toy(m4, p05, V, d, c["L"], pattern, c["E"], c["top_k"],
                                 c["w_e"], c["n_shared"])
        actual = model.n_params()
        assert actual == hand["total"], f"{name}: 实测 {actual} != 手算 {hand['total']}"
        secs, losses, _ = m4.train_loop(model.to(device), steps=10, batch=8, block=512,
                                        device=device, tokens_path=m4.TOKENS_SP8K,
                                        peak_lr=1e-3, warmup=2)
        row = {"pattern": pattern, **{k: v for k, v in c.items() if k != "pattern"},
               "total_params": actual, "activated_non_emb": hand["activated_non_emb"],
               "hand_detail": hand["detail"],
               "sec_per_step_p50": float(np.median(secs)), "loss_last": losses[-1]}
        results["candidates"][name] = row
        print(f"[{name}] 总参 {actual:,}（手算逐位 ✓）| 激活 non-emb {hand['activated_non_emb']:,} "
              f"| {np.median(secs):.3f} s/step | loss→{losses[-1]:.3f}")
        del model

    # ② MXFP4 钩子：推荐候选（toyA）量/不量各 10 步
    reco = "toyA_L8_E64t4s1_w32"
    c = CANDIDATES[reco]
    pattern = inject_global(list(c["pattern"]))
    torch.manual_seed(SEED)
    m_plain, _ = build_k3toy(m4, p05, V, d, c["L"], pattern, c["E"], c["top_k"], c["w_e"], c["n_shared"])
    secs_plain, losses_plain, _ = m4.train_loop(m_plain.to(device), steps=10, batch=8, block=512,
                                                device=device, tokens_path=m4.TOKENS_SP8K,
                                                peak_lr=1e-3, warmup=2)
    del m_plain
    torch.manual_seed(SEED)
    m_q, _ = build_k3toy(m4, p05, V, d, c["L"], pattern, c["E"], c["top_k"], c["w_e"], c["n_shared"])
    n_swapped = apply_mxfp4_to_experts(m_q, m4)
    assert m_q.n_params() == results["candidates"][reco]["total_params"], "量化钩子不得改参数账"
    secs_q, losses_q, _ = m4.train_loop(m_q.to(device), steps=10, batch=8, block=512,
                                        device=device, tokens_path=m4.TOKENS_SP8K,
                                        peak_lr=1e-3, warmup=2)
    results["mxfp4_hook"] = {"n_linear_swapped": n_swapped,
                             "sec_per_step_plain": float(np.median(secs_plain)),
                             "sec_per_step_quant": float(np.median(secs_q)),
                             "overhead_ratio": float(np.median(secs_q) / np.median(secs_plain)),
                             "loss_last_plain": losses_plain[-1], "loss_last_quant": losses_q[-1],
                             "embedding": "STE 直通：w_q=mxfp4_rt(w)；w+(w_q−w).detach()；专家 12 Linear/层全量"}
    print(f"[MXFP4 钩子] 换 {n_swapped} 个 Linear | {np.median(secs_plain):.3f}→{np.median(secs_q):.3f} s/step "
          f"（×{np.median(secs_q)/np.median(secs_plain):.2f}）| loss→{losses_q[-1]:.3f}")

    # ③ 定档建议
    reco_fields = {"vocab_size": V, "hidden_size": d, "num_hidden_layers": c["L"], "layer_pattern": pattern,
                   "kda_slot(GDN 替身)": {"h_k": 8, "h_v": 8, "d_k": 64, "d_v": 64, "conv_kernel": 4, "chunk": 64},
                   "mla_slot": {"num_heads": 8, **MLA_GEOM},
                   "moe": {"n_expert": c["E"], "top_k": c["top_k"], "expert_dim": c["w_e"], "n_shared": 1},
                   "mxfp4": "专家权重 STE 模拟量化（fp8_sim.mxfp4_rt，块 32）",
                   "total_params": results["candidates"][reco]["total_params"],
                   "activated_non_emb": results["candidates"][reco]["activated_non_emb"]}
    results["recommendation"] = reco_fields
    print(f"[定档建议] 正式替身 = {reco}：{reco_fields['total_params']:,} 参 "
          f"（激活 {reco_fields['activated_non_emb']:,}），训练数百步档位见 JSON。")

    save_json("probe06_k3toy", results, args.out_name)
    print("探针 06 完成。")


if __name__ == "__main__":
    main()
