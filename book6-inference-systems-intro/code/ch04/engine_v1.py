# engine_v1.py —— Book6 大项目 v1：naive 引擎（静态批 + 无 KV 全量重算——明知故犯的反面教材）
# 用途：ch4 §4.4 的正身载体。三宗「罪」全部保留：①静态批（一批算到全批完成为止，先完成的
#   请求空转陪跑——decode 期浪费 token 位的账由此可测）；②无 KV cache（每步全量重算整段序列，
#   与 ch1 基线同口径——O(n²) 的账保留）；③逐批串行（批间无重叠）。v2 逐条偿还。
#   workload：等长 prompt（tokens32k 同一起点切 128 token）+ 变长输出（16-64）——prefill 无 pad
#   干扰（教学焦点在 decode 空转），输出长短不齐制造静态批的经典浪费。
# 所属章节：Book6 ch4 §4.4；设计书=plan/Book6-推理系统导论.md ch4 节列表（v1 契约）
# 运行：cd <workspace> && source env.sh && npu-smi info 挑卡后
#      ASCEND_RT_VISIBLE_DEVICES=<卡> python code/Book6-推理系统导论/ch04/engine_v1.py \
#        [--batch 8 --smoke --out-name s1]
# 产物：log/book6-ch04/engine_v1_<out-name>.json（吞吐/TPOT/浪费位/显存峰值/时延分解）
import argparse
import importlib.util
import json
import os
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(ROOT, "log", "book6-ch04")
CKPT = os.path.join(ROOT, "log", "book3-ch10", "ckpt_integrated_npu1.pt")
TOK = os.path.join(ROOT, "log", "book3-ch10", "tokens32k.bin")

_L215_CANDS = [
    os.path.join(ROOT, "code", "Book3-现代开源骨架", "ch10", "llama215.py"),
    os.path.join(ROOT, "from-models-to-vllm", "book3-modern-open-source-skeleton", "code", "ch10", "llama215.py"),
    os.path.join(ROOT, "from-models-to-vllm", "code", "ch10", "llama215.py"),
]
SEED = 20261002


def load_llama215():
    for c in _L215_CANDS:
        if os.path.isfile(c):
            spec = importlib.util.spec_from_file_location("llama215", c)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("llama215.py 三候选均未命中")


def make_workload(n_req, p_len, max_span, smoke=False):
    """等长 prompt + 变长输出。max_new 在 [max_span//4, max_span] 均匀取（构造 workload，教学口径）。"""
    import numpy as np
    g = torch.Generator().manual_seed(SEED)
    stream = np.fromfile(TOK, dtype=np.uint16, count=p_len + 4096)
    prompts = torch.tensor([stream[i: i + p_len].tolist() for i in range(0, n_req * 8, 8)][:n_req])
    hi = 16 if smoke else max_span
    lo = max(4, hi // 4)
    max_new = torch.randint(lo, hi + 1, (n_req,), generator=g)
    return prompts, max_new


def main() -> None:
    ap = argparse.ArgumentParser(description="v1 naive 引擎：静态批+无 KV 全量重算（反面教材）")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--p-len", type=int, default=128)
    ap.add_argument("--max-span", type=int, default=64)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-name", default=None)
    a = ap.parse_args()
    name = a.out_name or ("smoke" if a.smoke else "base")
    os.makedirs(OUT_DIR, exist_ok=True)
    if a.smoke:
        a.batch, a.p_len, a.max_span = 4, 64, 16

    l215 = load_llama215()
    device = "npu" if torch.npu.is_available() else "cpu"
    torch.manual_seed(SEED)
    model = l215.Llama215()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(device=device, dtype=torch.bfloat16)

    prompts, max_new = make_workload(a.batch, a.p_len, a.max_span, a.smoke)
    prompts = prompts.to(device)                                  # (B, n0) 等长
    B = prompts.shape[0]
    gen = prompts.clone()                                         # (B, n) 全量重算口径：序列只增不减
    done = torch.zeros(B, dtype=torch.bool, device=device)

    if device == "npu":
        torch.npu.reset_peak_memory_stats()
    torch.npu.synchronize() if device == "npu" else None
    t0 = time.perf_counter()

    n_steps = 0
    while not bool(done.all()):
        with torch.no_grad():
            logits = model(gen)[0][:, -1]                         # (B, V)——全量重算取末位
        probs = torch.softmax(logits.float(), dim=-1)
        nxt = torch.multinomial(probs, 1)                         # (B, 1) 采样
        gen = torch.cat([gen, nxt], dim=1)                        # (B, n+1)
        n_steps += 1
        done = done | (n_steps >= max_new.to(device))              # 第 n_steps 步后：已生成 n_steps 个 token

    torch.npu.synchronize() if device == "npu" else None
    wall = time.perf_counter() - t0
    peak = (torch.npu.max_memory_allocated() / 2**20 if device == "npu" else 0)

    useful = int(max_new.sum().item())                            # 有效 token 总数
    wasted = B * n_steps - useful                                 # 空转陪跑的 token 位
    out = {
        "engine": "v1_naive", "device": device, "dtype": "bfloat16",
        "workload": {"batch": B, "p_len": a.p_len,
                     "max_new_min": int(max_new.min()), "max_new_max": int(max_new.max())},
        "decode_steps": n_steps, "wall_s": round(wall, 3),
        "throughput_tok_s": round(useful / wall, 2),
        "per_req_tpot_ms": round(wall / n_steps * 1e3 / 1, 3),    # 静态批：全批同速=批速
        "useful_tokens": useful, "wasted_slots": wasted,
        "waste_ratio_pct": round(wasted / (B * n_steps) * 100, 1),
        "peak_mem_MiB": round(peak, 1),
        "kv": "none（全量重算——v2 的改造对象）",
        "sins": ["static_batch（先完成者空转）", "no_kv_cache（O(n²) 重算）", "batch 内串行陪跑"],
    }
    p = os.path.join(OUT_DIR, f"engine_v1_{name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[v1] {B} 条请求 × p_len {a.p_len} | decode {n_steps} 步 | wall {wall:.2f}s")
    print(f"     吞吐 {out['throughput_tok_s']} tok/s | 浪费位 {wasted}/{B*n_steps}（{out['waste_ratio_pct']}%）"
          f" | 峰值 {out['peak_mem_MiB']} MiB")
    print(f"[产物] {p}")


if __name__ == "__main__":
    main()
