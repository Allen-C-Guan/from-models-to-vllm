# exp_ch05.py —— Book7 ch05 代码件：调度器两实验（preemption 场景构造 + token 预算两档对照）
# 用途：ch05《调度器》实测——
#   实验① preemption 场景：低 KV 预算（gpu_memory_utilization=0.30）+ 长请求并发，构造 KV 先于预算耗尽、
#     触发 RECOMPUTE 抢占；判据预注册（notes/14 §8）：周期统计行出现「Preemptions: K」段（仅 K>0 出现——
#     出现即判据达成）、KV usage 行逼近 100%、至少一条请求完成；预期行为：抢占步不收新请求。
#   实验② max-num-batched-tokens 两档（512 vs 2048）：同 workload（一条 2k 长预填+3 条短请求），
#     判据预注册：512 档长请求被切 4 段、TTFT 变长、短请求首 token 提前（Book6 ch07 sched_v4 双档的引擎版）。
# 所属章节：Book7 ch05（§5.4/§5.5 实验）；设计书=notes/14 §8。
# 运行：cd <workspace> && source env.sh && npu-smi info 选卡 && ASCEND_RT_VISIBLE_DEVICES=<卡> \
#        python code/Book7-vLLM源码剖析上/ch05/exp_ch05.py [--skip-preempt] [--skip-budget]
# 产物：log/book7-ch05/{preempt_run.log(自采), preempt.json, budget_<arm>.json, exp_ch05_summary.json}
# 关键坑（notes/14）：offline LLM 默认强制 disable_log_stats=True——必须显式传 False 才有周期统计行；
#   VLLM_LOG_STATS_INTERVAL=2 缩短周期。TTFT 从 RequestOutput.metrics（first_token_time-arrival_time）读。
import argparse
import json
import os
import time

os.environ.setdefault("VLLM_LOG_STATS_INTERVAL", "2")

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MODEL = os.path.join(WS, "log", "book7-00", "hf207m")
OUT = os.path.join(WS, "log", "book7-ch05")


def long_prompt(seed: int, n: int) -> list[int]:
    import random
    r = random.Random(seed)
    return [r.randrange(3, 32000) for _ in range(n)]


def preempt_run() -> dict:
    from vllm import LLM, SamplingParams
    # 容量账（预注册，v2——吸取三跑教训）：固定开销≈0.65 GiB；u=0.015 → KV≈12.7k token（实测 12,672）。
    # v1 配方（prompt 1800+输出 248）实测：KV 压到 98% 但零抢占——prompt 总量超容量时调度器用
    # waiting 侧「装不下就等」消化（8 进 6 等 2），抢占不发生。v2 配方改为「先全准入、后长大越界」：
    # prompt 1024×8=8.2k < 12.7k 全部进 running；每条 max_tokens=1000 → 长大后 8×2024=16.2k > 12.7k
    # → running 者生长撞墙 → 必然触发抢占（RECOMPUTE）。
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.015,
              enforce_eager=True, disable_log_stats=False)
    sp = SamplingParams(temperature=0.0, max_tokens=1000)
    # 逐条异种子：同种子=8 条相同 prompt → APC 全命中、KV 需求塌缩（首跑意外实证，见正文注记）
    prompts = [long_prompt(20261011 + i, 1024) for i in range(8)]
    t0 = time.perf_counter()
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], sp)
    wall = round(time.perf_counter() - t0, 1)
    finished = sum(1 for o in outs if o.finished)
    # 0.27.1 的请求级指标字段：first_token_latency 现成（时钟能不混减——arrival_time 墙钟/ts 单调钟）
    ttfts = [round(o.metrics.first_token_latency, 3) if o.metrics else None for o in outs]
    return dict(exp="preempt", wall_s=wall, n_requests=len(prompts), finished=finished,
                ttft_s=ttfts,
                note="请求级抢占计数在 0.27.1 的 outputs 无字段——引擎级证据=日志『Preemptions:』段")


def budget_arm(mnbt: int) -> dict:
    from vllm import LLM, SamplingParams
    # 注：hf207m 的 config 上限 max_position_embeddings=2048——长预填取 2040+输出 8（合计 2048 内）
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.4,
              enforce_eager=True, disable_log_stats=False, max_num_batched_tokens=mnbt)
    sp = SamplingParams(temperature=0.0, max_tokens=8)
    items = ([{"prompt_token_ids": long_prompt(20261100, 2040)}]          # 一条长预填
             + [{"prompt_token_ids": long_prompt(20261101 + i, 64)} for i in range(3)])  # 三条短
    t0 = time.perf_counter()
    outs = llm.generate(items, sp)
    wall = round(time.perf_counter() - t0, 1)
    ttfts = [round(o.metrics.first_token_latency, 3) if o.metrics else None for o in outs]
    return dict(exp="budget", mnbt=mnbt, wall_s=wall, ttft_long=ttfts[0], ttft_shorts=ttfts[1:])


def main() -> None:
    ap = argparse.ArgumentParser(description="ch05 调度器两实验")
    ap.add_argument("--skip-preempt", action="store_true")
    ap.add_argument("--skip-budget", action="store_true")
    ap.add_argument("--budget-arm", type=int, choices=[512, 2048],
                    help="单跑一个 budget 臂（一臂一进程纪律——NPU 同进程双 LLM 必死）")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.budget_arm:
        r = budget_arm(a.budget_arm)
        print(f"[budget {a.budget_arm}]", json.dumps(r, ensure_ascii=False))
        with open(os.path.join(OUT, f"budget_{a.budget_arm}.json"), "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=1)
        return
    summary = {}
    if not a.skip_preempt:
        summary["preempt"] = preempt_run()
        print("[preempt]", json.dumps(summary["preempt"], ensure_ascii=False))
    if not a.skip_budget:
        for mnbt in (512, 2048):
            summary[f"budget_{mnbt}"] = budget_arm(mnbt)
            print(f"[budget {mnbt}]", json.dumps(summary[f"budget_{mnbt}"], ensure_ascii=False))
    with open(os.path.join(OUT, "exp_ch05_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"[产物] {os.path.join(OUT, 'exp_ch05_summary.json')}")


if __name__ == "__main__":
    main()
