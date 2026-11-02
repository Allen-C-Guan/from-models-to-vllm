# bench_207m.py —— Book7 ch10 代码件：大项目对照读数（vllm 载 207M 的单请求 TPOT）
# 用途：表 10.4 的 vllm 侧读数——同 207M 模型（与 Book6 mini 引擎 v4 同一权重）在 vllm 运行栈上的
#   单请求 decode TPOT，Book6 ch13 的 59 ms（mini v2 档、趋势级）与 ch01 的 8.9 ms（Qwen3-0.6B）之间定位。
#   判据预注册：TPOT 显著低于 mini 档（工程红利）；与 8.9 ms 不可比（跨模型——四要素声明）。
# 所属章节：Book7 ch10（§10.7 表 10.4）；设计书=notes/18。
# 运行：cd <workspace> && source env.sh && npu-smi info 选卡 && ASCEND_RT_VISIBLE_DEVICES=<卡> \
#        python code/Book7-vLLM源码剖析上/ch10/bench_207m.py [--out-name base]
# 产物：log/book7-ch10/bench_207m_<out-name>.json
import argparse, json, os, time

WS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
MODEL = os.path.join(WS, "log", "book7-00", "hf207m")
OUT = os.path.join(WS, "log", "book7-ch10")

if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="base")
    a = ap.parse_args()
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=0.4,
              enforce_eager=True, disable_log_stats=False)
    # 单请求贪心 64-token decode：TPOT = (总时-prefill)/63 的近似——用逐 token 时间戳更准
    sp = SamplingParams(temperature=0.0, max_tokens=64)
    prompt = [{"prompt_token_ids": [9707, 11, 1499, 290]}]  # 固定 4-token prompt（prefill 极小）
    outs = llm.generate(prompt, sp)
    o = outs[0]
    m = o.metrics
    # first_token_ts=首 token（含 prefill）；last_token_ts=末 token
    # TPOT=(last-first)/(n-1)=首 token 后均摊——与 Book6 生成墙钟口径差一个 TTFT（§10.7 口径注）
    n = len(o.outputs[0].token_ids)
    tpot_ms = round((m.last_token_ts - m.first_token_ts) / (n - 1) * 1000, 3) if m else None
    ttft_ms = round(m.first_token_latency * 1000, 3) if m else None
    res = dict(model="hf207m (207,119,360 params, bf16, NPU)", n_tokens=n,
               ttft_ms=ttft_ms, tpot_ms=tpot_ms,
               runtime="vllm 0.27.1 + vllm-ascend 0.27.1rc1（Atlas 910B3, eager）",
               refs="Book6 mini v2: 59 ms (trend-level); Qwen3-0.6B: 8.9 ms (cross-model, 4-factor caveat)")
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"bench_207m_{a.out_name}.json")
    json.dump(res, open(p, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print("[产物]", p)
