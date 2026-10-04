# 06_probe_f_kv.py —— 探针 F：KV cache 显存账本实测对拍（ch6 正菜的底数）
# 用途：(1) 手算公式 bytes = 2(K+V) * L * n * h_kv * d_k * dtype_bytes——逐档列 MHA/GQA(4组)/MQA
#           在 124M(GPT-2: L=12,h=12,d_k=64) / 215M(本册: L=12,h=16,d_k=64) / 7B(LLaMA2: L=32,h=32,d_k=128)
#           三档模型、n=1024/4096/8192、B=1、fp16 的 KV 字节表；
#       (2) 小档实测：真实分配 (1,L,n,h_kv,d_k) fp16 张量对，MPS current_allocated_memory 差值 vs 手算对拍。
#       （证据等级约定：本表数字一律自算自测，不引二手汇总——ch6「显存账本第一次手算」的口径预演。）
# 所属章节：Book3 00-feasibility（ch6 KV cache 显存账本 + F4/F16/N7 回收的实验底数）。
# 运行方式：cd code/00-feasibility && python 06_probe_f_kv.py [--out-name NAME]
#   产物：log/book3-feasibility/{out-name}.json（不 commit）
import argparse
import json
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book3-feasibility")
SEED = 20261002

# 三档模型的注意力几何（来源：GPT-2 官方 config / 本册 215M 候选 / LLaMA2-7B 官方 config，ch6 正文引用处核）
MODELS = {
    "124M(GPT-2)":   {"L": 12, "h": 12, "d_k": 64},
    "215M(本册)":     {"L": 12, "h": 16, "d_k": 64},
    "7B(LLaMA2)":    {"L": 32, "h": 32, "d_k": 128},
}
SCHEMES = {"MHA": None, "GQA(4组)": 4, "MQA": 1}      # None -> h_kv = h（MHA 即全头独立）
SEQS = [1024, 4096, 8192]
DTYPE_BYTES = 2                                       # fp16/bf16


def kv_bytes(L, n, h_kv, d_k, bytes_per=DTYPE_BYTES):
    """KV cache 字节手算：K 与 V 各一份，形状 (B,L,n,h_kv,d_k)。"""
    return 2 * L * n * h_kv * d_k * bytes_per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="probe_f_kv")
    args = ap.parse_args()
    torch.manual_seed(SEED)

    # ---- (1) 手算表 ----
    table = []
    for mname, m in MODELS.items():
        for sname, h_kv in SCHEMES.items():
            h_kv_eff = h_kv if h_kv is not None else m["h"]
            row = {"model": mname, "scheme": sname, "h": m["h"], "h_kv": h_kv_eff, "L": m["L"], "d_k": m["d_k"]}
            for n in SEQS:
                row[f"n={n}"] = kv_bytes(m["L"], n, h_kv_eff, m["d_k"])
            table.append(row)
            print(f"[手算] {mname:10s} {sname:8s} h_kv={h_kv_eff:2d}  "
                  + "  ".join(f"n={n}:{row[f'n={n}']/2**20:7.1f}MiB" for n in SEQS))

    # ---- (2) 小档实测对拍（MPS；h_kv=8/d_k=64/L=12 = 215M 主 config 几何） ----
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    measured = []
    L, h_kv, d_k = 12, 8, 64
    for n in SEQS:
        hand = kv_bytes(L, n, h_kv, d_k)
        if device.type == "mps":
            torch.mps.empty_cache()
            torch.mps.synchronize()
            before = torch.mps.current_allocated_memory()
            cache = [torch.randn(1, L, n, h_kv, d_k, dtype=torch.float16, device=device) for _ in range(2)]
            torch.mps.synchronize()
            after = torch.mps.current_allocated_memory()
            delta = after - before
            numel = cache[0].numel() * 2 * 2                     # K+V 两张，各 numel 元素 ×2 字节
            del cache
            torch.mps.empty_cache()
        else:                                                    # CPU 口径：element 逐字节账
            delta = None
            numel = 2 * L * n * h_kv * d_k * 2
        measured.append({"n": n, "hand_bytes": hand, "mps_alloc_delta_bytes": delta, "numel_bytes": numel,
                         "match": (abs(delta - hand) / hand < 0.05) if delta else None})
        print(f"[实测] n={n:5d}  手算 {hand/2**20:8.1f} MiB  MPS 分配差 "
              f"{(delta/2**20 if delta else 0):8.1f} MiB  偏差 "
              f"{(abs(delta-hand)/hand*100 if delta else float('nan')):.2f}%")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{args.out_name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "dtype_bytes": DTYPE_BYTES, "formula": "2*L*n*h_kv*d_k*bytes",
                   "hand_table": table, "measured": measured, "device": str(device)},
                  f, ensure_ascii=False, indent=2)
    print(f"产物已写：{out}")


if __name__ == "__main__":
    main()
