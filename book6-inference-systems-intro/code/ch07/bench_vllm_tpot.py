# bench_vllm_tpot.py —— Book6 ch7 第三层对拍：vllm 运行栈（0.27.1+vllm-ascend）单请求 TPOT 水位
# 用途：ch7 §7.5 的「真实引擎水位线」——运行栈跑 Qwen3-0.6B 的离线单请求生成，报每 token 均摊时延，
#   与 mini 引擎（v2 手写步进 59 ms 量级）做趋势级对照：绝对数不可比（模型不同、实现成熟度不同），
#   只标「工业水位」供量级定位（黑盒对拍纪律——vLLM 只作黑盒，源码剖析在 Book7）。
# 口径注：①两轮生成（首轮暖机——引擎初始化/kernel 编译/图捕获），报第二轮；
#   ②「每 token 均摊」= 生成墙钟 ÷ max_tokens（含 TTFT 摊入，非严格 TPOT——离线 API 拿不到
#   首 token 时刻，如实标注）；③贪心解码（temperature=0）消除采样随机性。
# vLLM 脚本纪律：落盘运行（本文件）+ __main__ 守卫（VLLM_WORKER_MULTIPROC_METHOD=spawn 重导入安全）；
#   模型走本地 HF 缓存（log/huggingface——缓存红线，不落 workspace 外）。
# 所属章节：Book6 ch7 §7.5；设计书=plan/Book6-推理系统导论.md ch7
# 运行：cd <workspace> && source env.sh && ASCEND_RT_VISIBLE_DEVICES=<卡> \
#      python code/Book6-推理系统导论/ch07/bench_vllm_tpot.py [--out-name base]
# 产物：log/book6-ch07/vllm_tpot_<out-name>.json
import argparse
import glob
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(ROOT, "log", "book6-ch07")


def find_qwen():
    cands = glob.glob(os.path.join(ROOT, "log", "huggingface", "hub", "models--Qwen--Qwen3-0.6B",
                                   "snapshots", "*"))
    for c in cands:
        if os.path.isfile(os.path.join(c, "config.json")):
            return c
    raise FileNotFoundError("Qwen3-0.6B 本地缓存未命中（log/huggingface）")


def main() -> None:
    ap = argparse.ArgumentParser(description="vllm 运行栈单请求 TPOT 水位（黑盒对拍）")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out-name", default="base")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    from vllm import LLM, SamplingParams
    model = find_qwen()
    t_init0 = time.perf_counter()
    llm = LLM(model=model, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.6)          # 默认执行栈（含图模式默认值——真实引擎行为）
    t_init = time.perf_counter() - t_init0
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
    prompt = ("The following is a long intro paragraph for a systems book about LLM inference. "
              * 12)                                     # ~200 token 量级 prompt

    for rnd, tag in (("warmup", 0), ("measured", 1)):
        t0 = time.perf_counter()
        outs = llm.generate([prompt], sp)
        wall = time.perf_counter() - t0
        ntok = len(outs[0].outputs[0].token_ids)
        if tag == 1:
            out = {"engine": "vllm-runtime", "vllm_version": "0.27.1+vllm-ascend 0.27.1rc1",
                   "device": "npu", "model": "Qwen3-0.6B", "dtype": "bfloat16",
                   "init_s": round(t_init, 1), "max_tokens": ntok,
                   "gen_wall_s": round(wall, 3),
                   "per_token_avg_ms": round(wall / ntok * 1e3, 1),
                   "note": "每 token 均摊=生成墙钟/max_tokens（TTFT 摊入；贪心；第二轮暖机后）"}
            p = os.path.join(OUT_DIR, f"vllm_tpot_{a.out_name}.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=1)
            print(f"[vllm:{rnd}] {ntok} tok | 墙钟 {wall:.3f} s | 每 token 均摊 "
                  f"{out['per_token_avg_ms']} ms（引擎初始化 {t_init:.0f} s 不计入）")
            print(f"[产物] {p}")
        else:
            print(f"[vllm:{rnd}] 暖机轮 {ntok} tok {wall:.3f} s（不计入）")


if __name__ == "__main__":
    main()
