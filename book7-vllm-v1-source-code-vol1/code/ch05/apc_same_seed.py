# apc_same_seed.py —— Book7 ch05 代码件：同种子 APC 防抢占意外档（批一三镜头 B3 销项重跑）
# 用途：固定种子生成 8 条完全相同的 prompt——APC 高命中、KV 需求塌缩、零抢占——
#       「前缀缓存防抢占」的意外实证（ch05 表 5.2 意外档读数的底档）。
# 所属章节：Book7 ch05（§5.8 表 5.2 意外档）；运行：source env.sh && ASCEND_RT_VISIBLE_DEVICES=<卡> \
#        python code/Book7-vLLM源码剖析上/ch05/apc_same_seed.py
# 产物：log/book7-ch05/apc_same_seed.json（命中率读数在 serve 侧周期行——本件日志 grep）
import json, os, random, time
os.environ.setdefault("VLLM_LOG_STATS_INTERVAL", "2")
from vllm import LLM, SamplingParams

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "log", "book7-00", "hf207m")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "log", "book7-ch05")

if __name__ == "__main__":
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.015,
              enforce_eager=True, disable_log_stats=False)
    sp = SamplingParams(temperature=0.0, max_tokens=1000)
    r = random.Random(20261011)                  # 固定种子 → 8 条完全相同
    p = [r.randrange(3, 32000) for _ in range(1024)]
    outs = llm.generate([{"prompt_token_ids": p} for _ in range(8)], sp)
    res = dict(exp="apc_same_seed", finished=sum(1 for o in outs if o.finished),
               ttft_s=[round(o.metrics.first_token_latency, 3) if o.metrics else None for o in outs],
               note="同种子=8 条同 prompt → APC 高命中防抢占的意外实证（批一 B3 销项重跑；命中率见日志周期行）")
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "apc_same_seed.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))
