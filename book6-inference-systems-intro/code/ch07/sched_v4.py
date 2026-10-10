# sched_v4.py —— Book6 ch7 代码件：mini 引擎调度整合（chunked prefill + decode 搭车，干扰复现 A/B）
# 用途：ch7 §7.5 正身。同一 workload（4 条 decode 在途 + 第 8 步一条长 prefill 闯入）跑两档调度：
#   ①none（不切）：长 prefill 一次性整段前向，期间 decode 集体停摆——generation stall 的教学复现；
#   ②chunked（切块）：长 prefill 按 --chunk 逐块推进，每迭代「一块前向 + 一步 decode」——decode 搭车
#     的教学形态（工业版把两者拼进同一次前向；本件同迭代背靠背，迭代有界性等价）。
#   核心读数：decode 迭代间隔序列（gap）——none 档出一个尖峰（=整段 prefill 时长）、chunked 档平坦。
# 预注册（正档默认：long=2048、chunk=256、4 条短请求×128 prompt×40 步、第 8 步到达）：
#   ①尖峰比（none 的 max_gap / chunked 的 max_gap）预期 >4×（2048 行整段前向 vs 256 行块+一步 decode）；
#   ②长请求 TTFT：chunked 略高于 none（同量 prefill 摊进 8 个迭代、每迭代背一步 decode；教学版另有
#     逐块 python 常数税——对应 SARATHI Fig 14「切块不是免费」的诚实披露）；
#   ③两档总墙钟同量级（同量计算、分布不同）。
# 机制复用：chunk 化 prefill = engine_v4.prefill_incremental（hit_len=块偏移——与 ch6 命中免算同机制：
#   跨块 attention 读全历史（梯形 mask）、KV 逐块增量写入）。
# 所属章节：Book6 ch7 §7.5；设计书=plan/Book6-推理系统导论.md ch7
# 运行：cd <workspace> && source env.sh && ASCEND_RT_VISIBLE_DEVICES=<卡> \
#      python code/Book6-推理系统导论/ch07/sched_v4.py [--smoke --out-name s1]
# 产物：log/book6-ch07/sched_v4_<out-name>.json（两档 gap 序列/尖峰/长请求 TTFT）
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(ROOT, "log", "book6-ch07")
CKPT = os.path.join(ROOT, "log", "book3-ch10", "ckpt_integrated_npu1.pt")
TOK = os.path.join(ROOT, "log", "book3-ch10", "tokens32k.bin")
SEED = 20261002


def load_mod(relpath):
    for c in [os.path.join(ROOT, relpath),
              os.path.join(ROOT, "from-models-to-vllm", "book6-inference-systems-intro", "code", relpath),
              os.path.join(ROOT, "from-models-to-vllm", "code", relpath)]:
        if os.path.isfile(c):
            spec = importlib.util.spec_from_file_location(os.path.basename(relpath), c)
            m = importlib.util.module_from_spec(spec)
            sys.path.insert(0, os.path.dirname(c))
            spec.loader.exec_module(m)
            return m
    raise FileNotFoundError(relpath)


def run_mode(mode, a, st, pool, ps, ev3, ev4, prompts, long_ids):
    """一档调度时间线。返回 (gaps_ms, kinds, long_ttft_ms, n_iters)。"""
    torch.manual_seed(SEED)
    tables, last, done = [], [], [0] * len(prompts)
    with torch.no_grad():
        for p in prompts:                                 # 初始 4 条短请求 prefill
            t = ev3.BlockTable()
            last.append(ps.prefill(p.unsqueeze(0).to(st.device), t))
            tables.append(t)
    n_short = len(tables)
    long_table, long_logits, long_offset = None, None, 0
    long_t0, long_ttft, long_decoded = None, None, False
    gaps, kinds = [], []
    t_prev = time.perf_counter()
    it = 0
    with torch.no_grad():
        while True:
            shorts_left = [i for i in range(n_short) if done[i] < a.gen_len]
            if not shorts_left and long_decoded:
                break
            if it > a.arrival + a.gen_len + 64:           # 兜底（理论不可达）
                break
            kind = "decode"
            if it == a.arrival:                            # 长请求到达：建表
                long_t0 = time.perf_counter()
                long_table = ev3.BlockTable()
                if mode == "none":                         # 整段前向（一次 2048 行）
                    long_table.ensure_capacity(pool, a.long_p)
                    long_logits = ev4.prefill_incremental(
                        st, pool, long_table, long_ids, 0, a.block_size)
                    kind = "full-prefill"
            if mode == "chunked" and long_table is not None and long_offset < a.long_p:
                end = min(long_offset + a.chunk, a.long_p)  # 本迭代一块
                long_table.ensure_capacity(pool, end)
                long_logits = ev4.prefill_incremental(
                    st, pool, long_table, long_ids[:, :end], long_offset, a.block_size)
                long_offset = end
                kind = "chunk"
            # decode 一步：短请求（+ 首 logits 已就绪的长请求——从 prefill 完成的下一拍起搭车）
            idxs = list(shorts_left)
            if long_table is not None and long_table.n >= a.long_p and long_logits is not None:
                idxs.append(-1)
            if not idxs:
                it += 1
                continue
            rows = [last[i] if i >= 0 else long_logits for i in idxs]
            nxt = torch.multinomial(torch.softmax(torch.stack(rows).float(), -1), 1)
            logits = ps.step_batch(nxt.to(st.device),
                                   [tables[i] if i >= 0 else long_table for i in idxs])
            t_end = time.perf_counter()
            gaps.append(round((t_end - t_prev) * 1e3, 1))
            kinds.append(kind)
            t_prev = t_end
            for k, i in enumerate(idxs):
                if i >= 0:
                    last[i] = logits[k]
                    done[i] += 1
                else:
                    long_logits = logits[k]
                    if not long_decoded:
                        long_decoded = True
                        long_ttft = (t_end - long_t0) * 1e3   # 到达 -> 第一个生成 token
            it += 1
    return gaps, kinds, long_ttft, it


def main() -> None:
    ap = argparse.ArgumentParser(description="v4 调度整合：chunked prefill 干扰复现 A/B")
    ap.add_argument("--long-p", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--n-short", type=int, default=4)
    ap.add_argument("--p-len", type=int, default=128)
    ap.add_argument("--gen-len", type=int, default=40)
    ap.add_argument("--arrival", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-name", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.long_p, a.chunk, a.p_len, a.gen_len, a.arrival = 512, 128, 64, 16, 4
    name = a.out_name or ("smoke" if a.smoke else "base")
    os.makedirs(OUT_DIR, exist_ok=True)

    ev2 = load_mod(os.path.join("code", "Book6-推理系统导论", "ch04", "engine_v2.py"))
    ev3 = load_mod(os.path.join("code", "Book6-推理系统导论", "ch05", "engine_v3.py"))
    ev4 = load_mod(os.path.join("code", "Book6-推理系统导论", "ch06", "engine_v4.py"))
    l215 = ev2.load_llama215()
    device = "npu" if torch.npu.is_available() else "cpu"
    torch.manual_seed(SEED)
    model = l215.Llama215()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(device=device, dtype=torch.bfloat16)
    st = ev2.Stepper(model)

    stream = np.fromfile(TOK, dtype=np.uint16, count=16384)
    prompts = [torch.tensor(stream[i * 128: i * 128 + a.p_len].tolist()) for i in range(a.n_short)]
    long_ids = torch.tensor([stream[8192: 8192 + a.long_p].tolist()]).to(device)

    out = {"engine": "sched_v4", "device": device, "dtype": "bfloat16",
           "workload": dict(n_short=a.n_short, p_len=a.p_len, gen=a.gen_len, long_p=a.long_p,
                            chunk=a.chunk, arrival=a.arrival, block_size=a.block_size),
           "seed": SEED, "modes": {}}
    for mode in ("none", "chunked"):
        pool = ev3.BlockFreeList(model.cfg, 4096, a.block_size, device, torch.bfloat16)
        ps = ev3.PagedStepper(st, pool)
        gaps, kinds, ttft, n_it = run_mode(mode, a, st, pool, ps, ev3, ev4, prompts, long_ids)
        mx = max(gaps) if gaps else 0
        med = sorted(gaps)[len(gaps) // 2] if gaps else 0
        out["modes"][mode] = dict(gaps_ms=gaps, kinds=kinds, max_gap_ms=mx,
                                  median_gap_ms=med, long_ttft_ms=round(ttft, 1) if ttft else None,
                                  n_iters=n_it)
        print(f"[{mode:7s}] decode 间隔尖峰 {mx} ms | 中位 {med} ms | 长请求 TTFT "
              f"{round(ttft, 1) if ttft else '-'} ms | 迭代数 {n_it}")
    n_max = out["modes"]["none"]["max_gap_ms"]
    c_max = out["modes"]["chunked"]["max_gap_ms"]
    print(f"[结论] 尖峰比 none/chunked = {n_max}/{c_max} = {n_max / c_max:.1f}×"
          f"（chunked 的长请求 TTFT 见上——预算旋钮的两面）")
    p = os.path.join(OUT_DIR, f"sched_v4_{name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[产物] {p}")


if __name__ == "__main__":
    main()
