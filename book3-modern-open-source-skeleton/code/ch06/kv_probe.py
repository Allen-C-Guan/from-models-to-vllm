# kv_probe.py —— Book3 ch6 KV cache 显存账本 probe 正身（章交付物；收编 00-feasibility/06 探针）
# 用途：(1) 手算公式 bytes = 2(K+V) * L * n * h_kv * d_k * dtype_bytes——逐档列 MHA/GQA(4组)/MQA
#           在 124M(GPT-2: L=12,h=12,d_k=64) / 207M(本册候选: L=12,h=16,d_k=64) / 7B(LLaMA2: L=32,h=32,d_k=128)
#           三档模型、n=1024/4096/8192、B=1、fp16 的 KV 字节表；
#       (2) 两套命名账（几何标签齐全，防混用）：207M 候选（L=12,h=16,h_kv=8,d_k=64）@2048=48MiB
#           （MHA 反事实 96 / MQA 6）；216M 等参数三兄弟（L=16,h=16,d_k=64，V=8k）@2048=128/32/8MiB
#           （每 token 64/16/4 KiB）——三兄弟参数账逐位 assert（attn+FFN 每层 12,451,840 三臂逐位相同，
#           仿 LLaMA2 A.2.1 的 FFN 补齐协议；登记数 216,006,656 为不含 RMSNorm 口径，含 13 处 norm
#           精确值 216,040,448——两口径都报，防回流）；
#       (3) 实测对拍：真实分配 (1,L,n,h_kv,d_k) fp16 张量对（逐层布局），MPS current_allocated_memory
#           差值 vs 手算逐 case 对拍（含 207M@2048 与三兄弟 @2048 三个命名 case）。
#       （证据等级约定：本表数字一律自算自测，不引二手汇总——ch6「显存账本第一次手算」的章级口径。）
# 所属章节：Book3 ch6（KV 账本 + F4/F16/N7 回收的实验正身）。
# 运行方式：python kv_probe.py [--out-name kv_probe]     （秒级，MPS 或 CPU）
#   产物：log/book3-ch06/kv_probe_{out}.json（不入库）
import argparse
import json
import os
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book3-ch06")
SEED = 20261002

# 三档模型的注意力几何（来源：GPT-2 官方 config / 本册 207M 候选（plan 定版 cand1）/ LLaMA2-7B 官方 config）
MODELS = {
    "124M(GPT-2)":  {"L": 12, "h": 12, "d_k": 64},
    "207M(本册)":    {"L": 12, "h": 16, "d_k": 64},
    "7B(LLaMA2)":   {"L": 32, "h": 32, "d_k": 128},
}
SCHEMES = {"MHA": None, "GQA(4组)": 4, "MQA": 1}      # None -> h_kv = h（MHA 即全头独立）
SEQS = [1024, 4096, 8192]
DTYPE_BYTES = 2                                       # fp16/bf16

# 216M 等参数三兄弟（papers/04 §7.3 / plan v1.1：L=16/V=8192 几何，只算账不训练）
BROTHERS = [
    {"name": "MHA",   "h_kv": 16, "d_ff": 2688},
    {"name": "GQA-4", "h_kv": 4,  "d_ff": 3200},
    {"name": "MQA",   "h_kv": 1,  "d_ff": 3328},
]
BROTHERS_GEOM = {"d": 1024, "L": 16, "h": 16, "d_k": 64, "V": 8192, "untied": True}


def kv_bytes(L, n, h_kv, d_k, bytes_per=DTYPE_BYTES, batch=1):
    """KV cache 字节手算：K 与 V 各一份，形状 (B,L,n,h_kv,d_k)。"""
    return 2 * L * n * h_kv * d_k * bytes_per * batch


def brothers_param_account(b):
    """三兄弟单臂参数手算（papers/04 §7.3 口径：attn+FFN，不含 RMSNorm）。

    每层 attn = 2d²（q/o 全宽）+ 2·d·h_kv·d_k（k/v 按 KV 组缩）；每层 FFN = 3·d·d_ff；
    总 = L×每层 + 2·V·d（untied 两份嵌入）。FFN 宽度逐臂调档使每层 attn+FFN 逐位相等（参数补齐）。
    """
    d, L, h, d_k, V = (BROTHERS_GEOM[k] for k in ("d", "L", "h", "d_k", "V"))
    attn = 2 * d * d + 2 * d * b["h_kv"] * d_k
    ffn = 3 * d * b["d_ff"]
    per_layer = attn + ffn
    total = L * per_layer + 2 * V * d
    return {"attn_per_layer": attn, "ffn_per_layer": ffn, "per_layer_attn_ffn": per_layer,
            "total_no_norms": total, "total_with_norms": total + (2 * L * d + d)}


def measure_case(L, n, h_kv, d_k, device):
    """逐 case 实测：逐层构造 (1,h_kv,n,d_k) 的 K/V 张量对（真实 cache 布局），MPS 分配差 vs 手算。"""
    hand = kv_bytes(L, n, h_kv, d_k)
    if device.type != "mps":
        return {"hand_bytes": hand, "mps_alloc_delta_bytes": None, "match": None,
                "note": "CPU 口径不测分配（逐字节账 numel 直接核对）"}
    torch.mps.empty_cache()
    torch.mps.synchronize()
    before = torch.mps.current_allocated_memory()
    cache = [torch.randn(1, h_kv, n, d_k, dtype=torch.float16, device=device) for _ in range(2 * L)]
    torch.mps.synchronize()
    delta = torch.mps.current_allocated_memory() - before
    del cache
    torch.mps.empty_cache()
    return {"hand_bytes": hand, "mps_alloc_delta_bytes": delta,
            "match": bool(delta == hand),
            "dev_pct": round(abs(delta - hand) / hand * 100, 3) if hand else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="kv_probe")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.perf_counter()

    # ---- (1) 手算表：三档模型 × 三方案 × 三窗长 ----
    hand_table = []
    for mname, m in MODELS.items():
        for sname, h_kv in SCHEMES.items():
            h_kv_eff = h_kv if h_kv is not None else m["h"]
            row = {"model": mname, "geometry": f"L={m['L']},h={m['h']},d_k={m['d_k']}",
                   "scheme": sname, "h_kv": h_kv_eff,
                   "kv_per_token_bytes": kv_bytes(m["L"], 1, h_kv_eff, m["d_k"])}
            for n in SEQS:
                row[f"n={n}_bytes"] = kv_bytes(m["L"], n, h_kv_eff, m["d_k"])
                row[f"n={n}_mib"] = round(row[f"n={n}_bytes"] / 2**20, 2)
            hand_table.append(row)
            print(f"[手算] {mname:10s} {sname:8s} h_kv={h_kv_eff:2d}  "
                  + "  ".join(f"n={n}:{row[f'n={n}_mib']:8.1f}MiB" for n in SEQS), flush=True)

    # ---- (2a) 命名账一：207M 候选 @2048（含 MHA 反事实 / MQA）----
    g = MODELS["207M(本册)"]
    named_207m = {
        "geometry": f"207M cand1：L={g['L']}, h=16, h_kv=8, d_k=64（V=32k untied）",
        "gqa2_at_2048_mib": round(kv_bytes(g["L"], 2048, 8, g["d_k"]) / 2**20, 2),
        "mha_counterfactual_at_2048_mib": round(kv_bytes(g["L"], 2048, 16, g["d_k"]) / 2**20, 2),
        "mqa_counterfactual_at_2048_mib": round(kv_bytes(g["L"], 2048, 1, g["d_k"]) / 2**20, 2),
        "kv_per_token_bytes": kv_bytes(g["L"], 1, 8, g["d_k"]),
    }
    assert named_207m["gqa2_at_2048_mib"] == 48.0, "207M@2048=48MiB 命名账失守"
    print(f"[命名账] 207M(L=12,h_kv=8)@2048 = {named_207m['gqa2_at_2048_mib']}MiB"
          f"（MHA 反事实 {named_207m['mha_counterfactual_at_2048_mib']} / "
          f"MQA {named_207m['mqa_counterfactual_at_2048_mib']}）", flush=True)

    # ---- (2b) 命名账二：216M 等参数三兄弟（参数账 assert + KV 账）----
    brothers = []
    for b in BROTHERS:
        acc = brothers_param_account(b)
        assert acc["per_layer_attn_ffn"] == 12_451_840, f"三兄弟每层参数补齐失守：{acc}"
        assert acc["total_no_norms"] == 216_006_656, f"三兄弟总参数失守：{acc}"
        rec = {"arm": b["name"], "h_kv": b["h_kv"], "d_ff": b["d_ff"], **acc,
               "kv_per_token_kib": round(kv_bytes(BROTHERS_GEOM["L"], 1, b["h_kv"],
                                                  BROTHERS_GEOM["d_k"]) / 2**10, 2),
               "kv_at_2048_mib": round(kv_bytes(BROTHERS_GEOM["L"], 2048, b["h_kv"],
                                                BROTHERS_GEOM["d_k"]) / 2**20, 2)}
        brothers.append(rec)
        print(f"[三兄弟] {b['name']:5s} h_kv={b['h_kv']:2d} d_ff={b['d_ff']}  "
              f"每层 attn+FFN={acc['per_layer_attn_ffn']:,}（逐位同）  "
              f"总={acc['total_no_norms']:,}（不含 norm；含 norm {acc['total_with_norms']:,}）  "
              f"KV@2048={rec['kv_at_2048_mib']}MiB", flush=True)
    assert [b["kv_at_2048_mib"] for b in brothers] == [128.0, 32.0, 8.0], "三兄弟 @2048 = 128/32/8MiB 命名账失守"
    named_brothers = {
        "geometry": f"216M 等参数三兄弟：d=1024/L={BROTHERS_GEOM['L']}/h=16/d_k=64/V=8192 untied",
        "arms": brothers,
        "param_convention_note": ("登记数 216,006,656 = 不含 13 处 RMSNorm（33,792 参数）的 papers/04 §7.3 "
                                  "口径；含 norm 精确值 216,040,448——引用时注明口径，防回流"),
        "kv_note": "每 token 64/16/4 KiB（任务书原文「64/32/8KiB」勘正——@2048 才是 128/32/8 MiB）",
    }

    # ---- (3) 实测对拍：207M 几何 n 扫描 + 三兄弟 @2048 三臂 ----
    measured = []
    for n in [1024, 2048, 4096, 8192]:
        rec = {"case": f"207M 几何 (1,L=12,h_kv=8,d_k=64) n={n}", "n": n,
               **measure_case(12, n, 8, 64, device)}
        measured.append(rec)
        print(f"[实测] 207M 几何 n={n:5d}  手算 {rec['hand_bytes']/2**20:8.1f} MiB  "
              f"MPS 差 {(rec['mps_alloc_delta_bytes'] or 0)/2**20:8.1f} MiB  "
              f"match={rec['match']}", flush=True)
    for b in BROTHERS:
        rec = {"case": f"三兄弟 {b['name']} (1,L=16,h_kv={b['h_kv']},d_k=64) n=2048", "n": 2048,
               **measure_case(16, 2048, b["h_kv"], 64, device)}
        measured.append(rec)
        print(f"[实测] 三兄弟 {b['name']:5s} @2048  手算 {rec['hand_bytes']/2**20:8.1f} MiB  "
              f"MPS 差 {(rec['mps_alloc_delta_bytes'] or 0)/2**20:8.1f} MiB  "
              f"match={rec['match']}", flush=True)

    n_match = sum(1 for r in measured if r["match"] is True)
    n_meas = sum(1 for r in measured if r["match"] is not None)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{args.out_name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"date": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "device": str(device),
                   "dtype_bytes": DTYPE_BYTES, "formula": "2*L*n*h_kv*d_k*bytes（B=1）",
                   "hand_table": hand_table,
                   "named_207m": named_207m, "named_brothers_216m": named_brothers,
                   "measured": measured,
                   "measured_summary": {"cases": n_meas, "exact_match": n_match,
                                        "note": "match = MPS 分配差与手算逐字节相等"},
                   "wallclock_sec": round(time.perf_counter() - t0, 1)},
                  f, ensure_ascii=False, indent=2)
    print(f"[完成] 实测 {n_meas} case 中 {n_match} 个逐字节吻合 -> {out}", flush=True)


if __name__ == "__main__":
    main()
