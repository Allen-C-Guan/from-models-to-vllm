# arithmetic.py —— Book6 ch2 代码件：推理算术体系手算函数族与三机型账本表
# 用途：①每 token FLOPs 的逐层推导（「2N 不是魔法是加法」——打印逐组件乘加账）；
#   ②权重/KV 字节账（元素数×dtype 字节换算式）；③decode 带宽下限与「没有引擎的价格」；
#   ④三机型算例矩阵（GPT-2 124M / llama215 207M / 两刀机 409M 激活 168.9M——B4-7 算例）；
#   ⑤MFU 两口径（microbench 饱和度 vs 真实负载）。全部 CPU 秒级、纯手算口径（不跑模型）。
# 所属章节：Book6 ch2 全章（§2.2-2.6 的账本正身）；设计书=plan/Book6-推理系统导论.md ch2
# 运行：cd <workspace> && source env.sh && python code/Book6-推理系统导论/ch02/arithmetic.py [--out-name base]
# 产物：log/book6-ch02/arithmetic_<out-name>.json
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(ROOT, "log", "book6-ch02")

BW_910B3 = 1190.0        # GB/s，本机实测拷贝稳态（notes/04 P2；官方无公开数——正文双口径）
PEAK_910B3 = 269.3       # TFLOPS bf16 matmul 实测（官方换算口径 313 并列）


def dense_config(name, V, d, L, h, h_kv, d_ff, tie=False):
    return dict(name=name, vocab=V, d_model=d, n_layers=L, n_heads=h,
                n_kv_heads=h_kv, d_ff=d_ff, tie=tie)


# 三机型（GPT-2 124M/llama215 207M 承各自 config；两刀机激活侧=DESIGN-409M §3 工程参数）
MACHINES = {
    "gpt2_124m": dict(dense_config("GPT-2 124M", 50257, 768, 12, 12, 12, 3072, tie=True), arch="gpt2"),
    "llama215_207m": dense_config("llama215 207M", 32000, 1024, 12, 16, 8, 2816, tie=False),
    "twoknife_409m": dense_config("两刀机 409M（激活侧）", 32000, 1024, 12, 16, 8, 2816, tie=False),
}
# 两刀机激活侧修正（MLA 注意力 + MoE FFN，激活参数按 DESIGN-409M §3：168,877,056 含 lm_head）
TWOKNIFE_ACT_PARAMS = 168_877_056
# llama215 参数账（llama215.hand_account 同款——assert 逐位）
LLAMA215_TOTAL = 207_119_360
LLAMA215_LAYER = (3_145_728 + 8_650_752 + 2_048)   # attn + mlp + 2×RMSNorm
GPT2_TOTAL = 124_356_864                            # GPT-2 124M 主项口径（官方全量 124,439,808 含 bias 82,944）
# 两刀机 KV：MLA latent 3,840 元素/token（DESIGN-409M §10；对退役 GQA 12,288 = −68.75%）
TWOKNIFE_KV_PER_TOK = 3840


def params_dense(cfg):
    """稠密机总参逐项账（GPT-2 形状：MHA+两矩阵 FFN+学习式 PE+wpe；LLaMA 形状：GQA+SwiGLU+RoPE 无 PE 表）。

    主项口径=矩阵权重（GPT-2 官方全量 124,439,808 另含 bias 82,944——正文注明，不进主项）。
    返回 (total_main, breakdown_dict)。
    """
    V, d, L, dff = cfg["vocab"], cfg["d_model"], cfg["n_layers"], cfg["d_ff"]
    h, hkv, dk = cfg["n_heads"], cfg["n_kv_heads"], cfg["d_model"] // cfg["n_heads"]
    if cfg.get("arch", "llama") == "gpt2":            # GPT-2：MHA（kv 头=全头）+ Conv1D 两矩阵 FFN
        attn = 4 * d * d                               # c_attn(d→3d) + c_proj(d→d)
        mlp = 2 * d * dff                              # c_fc + c_proj（两矩阵，无门控）
        pe = 1024 * d                                  # wpe 学习式位置表（n_ctx=1024）
        norms = 2 * (2 * d)                            # ln_1/ln_2 主项
        per_layer = attn + mlp + norms
        emb = V * d
        total = L * per_layer + emb + pe + 2 * d       # +ln_f 主项
        return total, {"attn_per_layer": attn, "mlp_per_layer": mlp, "norms": norms,
                       "wpe": pe, "per_layer": per_layer, "layers": L * per_layer,
                       "embedding": emb, "final_norm": 2 * d, "lm_head_untied": 0 if cfg["tie"] else emb,
                       "bias_note": "GPT-2 官方全量 124,439,808 = 主项 124,356,864 + bias 82,944（Conv1D/LN 偏置，正文注明）"}
    attn = h * d * dk + 2 * hkv * d * dk + h * dk * d  # Wq(d→h·dk) Wk,Wv(d→hkv·dk) Wo(h·dk→d)——GQA：kv 投影窄
    mlp = 3 * d * dff                                  # SwiGLU 三矩阵（gate/up/down）
    norms = 2 * d
    per_layer = attn + mlp + norms
    emb = V * d
    total = L * per_layer + emb + d + (0 if cfg["tie"] else emb)
    return total, {"attn_per_layer": attn, "mlp_per_layer": mlp, "norms": norms,
                   "per_layer": per_layer, "layers": L * per_layer, "embedding": emb,
                   "final_norm": d, "lm_head_untied": 0 if cfg["tie"] else emb}


def flops_per_token(cfg, n=1):
    """每 token 前向 FLOPs（matmul 乘加账 ≈ 2×激活参数；注意力分数项 O(n·d) 另列）。"""
    total, br = params_dense(cfg)
    return {"matmul_2N": 2 * total, "attn_scores_2LHn": 2 * cfg["n_layers"] * cfg["n_heads"] * n * (cfg["d_model"] // cfg["n_heads"])}


def kv_bytes_per_token(cfg, dtype_bytes=2):
    """KV 账（GQA 口径：2·L·h_kv·d_k 元素/token；dtype 字节换算）。"""
    dk = cfg["d_model"] // cfg["n_heads"]
    elems = 2 * cfg["n_layers"] * cfg["n_kv_heads"] * dk
    return {"elements": elems, "bytes": elems * dtype_bytes}


def machine_rows():
    rows = []
    for key in ("gpt2_124m", "llama215_207m"):
        cfg = MACHINES[key]
        total, br = params_dense(cfg)
        kv = kv_bytes_per_token(cfg)
        rows.append({
            "machine": cfg["name"], "total_params": total,
            "act_params_per_token": total,          # 稠密：激活=总参
            "flops_per_token": 2 * total,
            "weights_bf16_MB": round(total * 2 / 2**20, 1),
            "kv_elem_per_token": kv["elements"], "kv_B_per_token_bf16": kv["bytes"],
            "decode_floor_ms@1190": round(total * 2 / (BW_910B3 * 1e9) * 1e3, 3),
        })
    # 两刀机：激活侧按 DESIGN-409M（MoE top2——激活 FFN 只算被选专家；MLA KV 3,840）
    rows.append({
        "machine": MACHINES["twoknife_409m"]["name"],
        "total_params": 409_115_648,
        "act_params_per_token": TWOKNIFE_ACT_PARAMS,      # 168.9M（含 lm_head，DESIGN-409M §3）
        "flops_per_token": 2 * TWOKNIFE_ACT_PARAMS,
        "weights_bf16_MB": round(409_115_648 * 2 / 2**20, 1),
        "kv_elem_per_token": TWOKNIFE_KV_PER_TOK,
        "kv_B_per_token_bf16": TWOKNIFE_KV_PER_TOK * 2,
        "decode_floor_ms@1190": round(TWOKNIFE_ACT_PARAMS * 2 / (BW_910B3 * 1e9) * 1e3, 3),
        "note": "激活≠总参（MoE top2）；KV=MLA latent 3,840（对退役 GQA 12,288 −68.75%）",
    })
    return rows


def no_engine_price():
    """「没有引擎的价格」：ch1 实测稳态 vs 带宽下限（notes/01 §五 定稿口径）。"""
    floor_ms = LLAMA215_TOTAL * 2 / (BW_910B3 * 1e9) * 1e3
    measured_ms = 25.15        # bf16 稳态（serve_baseline_base.json n0=512 档）
    return {"bandwidth_floor_ms": round(floor_ms, 3), "measured_ms": measured_ms,
            "ratio": round(measured_ms / floor_ms, 1),
            "gap_decomposition": "launch/算子调度开销（主导——25ms 与 dtype/n0 无关）+ 无批处理权重不摊薄 + fp32 路径"}


def mfu(flops_per_tok, tokens_per_s, peak=PEAK_910B3 * 1e12):
    return flops_per_tok * tokens_per_s / peak


def main() -> None:
    ap = argparse.ArgumentParser(description="ch2 手算账本：三机型矩阵+基线价格账")
    ap.add_argument("--out-name", default="base")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    # assert 正身（与 llama215.hand_account 对齐的量）
    _, l215 = params_dense(MACHINES["llama215_207m"])
    assert l215["attn_per_layer"] == 3_145_728 and l215["mlp_per_layer"] == 8_650_752, l215
    total215, _ = params_dense(MACHINES["llama215_207m"])
    assert total215 == LLAMA215_TOTAL, total215
    g2, _ = params_dense(MACHINES["gpt2_124m"])
    assert g2 == GPT2_TOTAL, g2

    out = {
        "per_layer_breakdown_llama215": l215,
        "flops_per_token_llama215": flops_per_token(MACHINES["llama215_207m"], n=1024),
        "machine_matrix": machine_rows(),
        "no_engine_price": no_engine_price(),
        "mfu_example": {
            "desc": "ch1 基线 bf16 n0=512：39.77 tok/s × 2×207,119,360 FLOP/tok",
            "achieved_TFLOPS": round(2 * LLAMA215_TOTAL * 39.77 / 1e12, 2),
            "vs_microbench_peak_269.3_pct": round(2 * LLAMA215_TOTAL * 39.77 / 1e12 / 269.3 * 100, 3),
            "bw_util_pct": round(LLAMA215_TOTAL * 2 / (25.15e-3) / (BW_910B3 * 1e9) * 100, 2),
        },
        "constants": {"BW_910B3_GBps_measured": BW_910B3, "PEAK_910B3_TFLOPS_measured": PEAK_910B3,
                      "official_PEAK_313_TFLOPS_note": "厂商口径换算（2.504P÷8），正文双口径并列"},
    }
    p = os.path.join(OUT_DIR, f"arithmetic_{a.out_name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("[llama215 逐层账] ", {k: f"{v:,}" for k, v in l215.items()})
    for r in out["machine_matrix"]:
        print(f"[{r['machine']:>22}] 总参 {r['total_params']:,} | 激活/token {r['act_params_per_token']:,} "
              f"| FLOP/token {r['flops_per_token']:,} | bf16 权重 {r['weights_bf16_MB']:.0f}MB "
              f"| KV {r['kv_elem_per_token']:,} 元素/token | decode 下限 {r['decode_floor_ms@1190']} ms")
    ne = out["no_engine_price"]
    print(f"[没有引擎的价格] 实测 {ne['measured_ms']} ms vs 带宽下限 {ne['bandwidth_floor_ms']} ms ≈ {ne['ratio']}×")
    m = out["mfu_example"]
    print(f"[利用率] 算力 {m['achieved_TFLOPS']} TFLOPS=峰值 {m['vs_microbench_peak_269.3_pct']}% | 带宽 {m['bw_util_pct']}%（单请求裸跑连零头都用不上）")
    print(f"[产物] {p}")


if __name__ == "__main__":
    main()
