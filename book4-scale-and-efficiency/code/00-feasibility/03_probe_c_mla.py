# 03_probe_c_mla.py —— Book4 探针 C：mini MLA 前向探针（显式 vs 权重吸收 + KV 字节账实测）
# 用途：MLA（第二刀）在 CPU fp32 上的两个实现与一本账——ch6（本册数学最重章）的实验底座：
#       ①手写 MLA 注意力（d=512、d_c=256、d_c'=64 档；W_DKV/W_UK/W_UV/W_KR/q 侧低秩分解，
#         键位对齐 HF DeepseekV2），显式上投影版（K/V 各自上投影再注意力）与吸收版
#         （W_UK 吸收进 q、W_UV 吸收进输出，注意力直接在 d_c 维潜空间算）等价性对拍 max|Δ|；
#       ②KV 字节账实测：构造 GQA 的 KV cache 张量对 (1,L,n,h_kv,d_k)×2 vs MLA 的
#         (1,L,n,d_c)+(1,L,n,d_c')（bf16 元素大小计量），与手算公式对拍
#         （GQA = 2·L·h_kv·d_k；MLA = L·(d_c+d_c')——Book3 ch6 账本的续算口径）；
#       ③多几何对照表（probe 档 / 207M 两刀档 / DSV2-Lite 官方几何）——
#         「KV −93.3%」这类旗舰口径在什么几何下出现（正文引用前回论文原文核，本探针只自算）。
# 所属章节：Book4 ch6（MLA：KV 低秩压缩、解耦 RoPE、权重吸收）；00-feasibility 探针 C。
# 运行方式：source env.sh && python 03_probe_c_mla.py [--out-name run1]（秒级，CPU fp32）
# 产物：log/book4-feasibility/probeC_mla_{out}.json（不入库）
import argparse

import torch

from moe_mla_slots import (SEED, MLAAttention, SlotConfig, activated_account, bootstrap_book3,
                           hand_account, kv_per_token_elements, mla_absorbed_forward, save_json)


def kv_account_row(name, L, gqa_elems_fn, mla_elems_fn, n=1024, dtype_bytes=2, batch=1):
    """一行账：GQA 口径 vs MLA 口径的每 token KV 元素/字节（n=1024 全窗总量）。elems 公式：
    GQA = 2(K,V)·L·h_kv·d_k；MLA = L·(d_c+d_c')——c_KV 一份（K/V 信息都在潜向量里，读出端现算
    上投影）+ 解耦 RoPE 键 k_R 一份。"""
    g, m = gqa_elems_fn(), mla_elems_fn()
    return {"geometry": name, "L": L, "seq": n,
            "gqa_kv_elems_per_token": g, "mla_kv_elems_per_token": m,
            "gqa_kv_bytes_full": g * n * dtype_bytes * batch,
            "mla_kv_bytes_full": m * n * dtype_bytes * batch,
            "mla_vs_gqa_ratio": round(m / g, 4), "saving_pct": round((1 - m / g) * 100, 1)}


def main(out_name):
    torch.manual_seed(SEED)
    bootstrap_book3()
    device = torch.device("cpu")

    # ① 显式 vs 吸收：probe 档几何 d=512 / h=8 / d_c=256 / d_c'=64 / d_nope=64 / d_v=64 / r_q=128
    attn = MLAAttention(hidden_size=512, n_head=8, kv_lora_rank=256, qk_nope_head_dim=64,
                        qk_rope_head_dim=64, v_head_dim=64, q_lora_rank=128).to(device).float()
    n_params = sum(p.numel() for p in attn.parameters())
    x = torch.randn(2, 128, 512)                      # (B,n,d)
    y_exp = attn(x)                                   # 显式上投影版 (B,n,d)
    y_abs = mla_absorbed_forward(attn, x)             # 吸收版 (B,n,d)
    dmax = (y_exp - y_abs).abs().max().item()
    rel = ((y_exp - y_abs).norm() / y_exp.norm()).item()
    argmax_agree = (y_exp.argmax(-1) == y_abs.argmax(-1)).float().mean().item()
    print(f"[MLA 对拍] 显式 vs 吸收：max|Δ| = {dmax:.2e} | 相对误差 {rel:.2e} | argmax 一致率 {argmax_agree:.4f}")
    print(f"[MLA 几何] d=512/h=8/d_c=256/d_c'=64/d_nope=64/d_v=64/r_q=128 | 单层注意力参数 {n_params:,} "
          f"| 缓存每 token 每层 {256 + 64} elems")

    # 多输入尺寸复验（短窗/长窗）
    sizes = {}
    for n in (16, 256, 1024):
        xx = torch.randn(2, n, 512)
        d = (attn(xx) - mla_absorbed_forward(attn, xx)).abs().max().item()
        sizes[n] = float(f"{d:.2e}")
        assert d < 1e-4, f"n={n} 等价性失败：{d}"

    # ② KV 字节账实测：构造真实 cache 张量（bf16），tensors vs 公式逐位对拍
    L, n, h_kv, d_k, d_c, d_r = 12, 1024, 8, 64, 256, 64
    gqa_cache_k = torch.zeros(1, L, n, h_kv, d_k, dtype=torch.bfloat16)   # (1,L,n,h_kv,d_k)
    gqa_cache_v = torch.zeros_like(gqa_cache_k)
    mla_cache_c = torch.zeros(1, L, n, d_c, dtype=torch.bfloat16)         # c_KV 潜向量
    mla_cache_r = torch.zeros(1, L, n, d_r, dtype=torch.bfloat16)         # k_R 解耦 rope 键
    gqa_bytes_meas = (gqa_cache_k.element_size() * gqa_cache_k.numel()
                      + gqa_cache_v.element_size() * gqa_cache_v.numel())
    mla_bytes_meas = (mla_cache_c.element_size() * mla_cache_c.numel()
                      + mla_cache_r.element_size() * mla_cache_r.numel())
    gqa_formula = 2 * L * n * h_kv * d_k * 2                            # 2(K,V)·L·n·h_kv·d_k·2字节
    mla_formula = L * n * (d_c + d_r) * 2                               # L·n·(d_c+d_c')·2字节
    assert gqa_bytes_meas == gqa_formula, (gqa_bytes_meas, gqa_formula)
    assert mla_bytes_meas == mla_formula, (mla_bytes_meas, mla_formula)
    print(f"[KV 实测] GQA@207M几何(L=12,h_kv=8,d_k=64,n=1024,bf16) = {gqa_bytes_meas/1024/1024:.2f} MiB"
          f" | MLA(d_c=256,d_c'=64) = {mla_bytes_meas/1024/1024:.2f} MiB "
          f"| 节省 {(1 - mla_bytes_meas/gqa_bytes_meas)*100:.1f}%（张量构造=公式逐位一致）")

    # ③ 多几何对照表（公式口径；旗舰行只自算，引用数字待回论文核）
    rows = [
        kv_account_row("probe 档（h=8, GQA h_kv=4/d_k=64 vs MLA d_c=256/d_c'=64）", 6,
                       lambda: 2 * 6 * 4 * 64, lambda: 6 * (256 + 64)),
        kv_account_row("207M 两刀档（h=16, GQA h_kv=8/d_k=64 vs MLA d_c=256/d_c'=64）", 12,
                       lambda: 2 * 12 * 8 * 64, lambda: 12 * (256 + 64)),
        kv_account_row("207M 两刀档 vs MHA 反事实（h_kv=16）", 12,
                       lambda: 2 * 12 * 16 * 64, lambda: 12 * (256 + 64)),
        kv_account_row("DSV2-Lite 官方几何（MHA h=16/d_h=128 vs MLA d_c=512/d_c'=64；{官方 config 核实}"
                       "：L=27, q_lora_rank=null, 64 专家 top-6 + 2 共享）", 27,
                       lambda: 2 * 27 * 16 * 128, lambda: 27 * (512 + 64)),
        kv_account_row("DSV2 236B 量级（MHA h=128/d_h=128 vs MLA d_c=512/d_c'=64；量级示意，L=60 待核）", 60,
                       lambda: 2 * 60 * 128 * 128, lambda: 60 * (512 + 64)),
    ]
    for r in rows:
        print(f"[账表] {r['geometry']}: GQA/MHA {r['gqa_kv_elems_per_token']:,} vs MLA "
              f"{r['mla_kv_elems_per_token']:,} elems/token | 节省 {r['saving_pct']}%")

    # ④ SlotLLaMA(MLA) 整机口径（参数账 assert 已在组件自测；此处报激活账与 KV 账的联合口径）
    cfg = SlotConfig(vocab_size=8192, hidden_size=512, num_hidden_layers=6, num_attention_heads=8,
                     num_key_value_heads=4, intermediate_size=1408,
                     mla=dict(kv_lora_rank=256, q_lora_rank=128, qk_nope_head_dim=64,
                              qk_rope_head_dim=64, v_head_dim=64))
    act, act_ne = activated_account(cfg)
    print(f"[SlotLLaMA+MLA 档] 激活参 {act:,}（non-emb {act_ne:,}）| KV/token {kv_per_token_elements(cfg)}")

    report = {"seed": SEED, "device": "cpu fp32",
              "mla_geoms": {"d": 512, "h": 8, "d_c": 256, "d_c_prime": 64, "d_nope": 64,
                            "d_v": 64, "q_lora_rank": 128, "attn_params": n_params},
              "parity_explicit_vs_absorbed": {"max_abs_diff": dmax, "rel_err": rel,
                                              "argmax_agreement": argmax_agree,
                                              "multi_seq_len_max_abs": sizes},
              "kv_bytes_measured": {"gqa_L12_hkv8_dk64_n1024_bf16": gqa_bytes_meas,
                                    "mla_dc256_dr64_n1024_bf16": mla_bytes_meas,
                                    "formula_match": True,
                                    "saving_pct": round((1 - mla_bytes_meas / gqa_bytes_meas) * 100, 2)},
              "kv_account_table": rows,
              "slot_mla_cfg": {"activated": act, "activated_nonemb": act_ne}}
    verdict = "PASS" if dmax < 1e-4 and argmax_agree == 1.0 else "FAIL"
    report["verdict"] = verdict
    print(f"[判定] {verdict}")
    save_json("probeC_mla", report, out_name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="探针 C：MLA 显式 vs 吸收对拍 + KV 字节账")
    ap.add_argument("--out-name", default="run1", help="产物 JSON 后缀（防覆写）")
    args = ap.parse_args()
    main(args.out_name)
