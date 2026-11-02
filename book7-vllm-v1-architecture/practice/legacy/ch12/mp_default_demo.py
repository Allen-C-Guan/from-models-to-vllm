# mp_default_demo.py —— Book7 ch12 方法①实证：默认多进程形态下「日志怎么流回主进程」
# 目标证据（本机运行栈 0.27.1 + vllm-ascend 0.27.1rc1，Atlas 910B3）：
#   E1 EngineCore 以 spawn 子进程存在（进程名/父子关系）
#   E2 子进程日志经继承的 stdout/stderr 流回同一终端，且带 decorate_logs 前缀
#      “(EngineCore pid=XXXX)”（源码锚 utils/system_utils.py::decorate_logs /
#      v1/engine/core.py::run_engine_core 中 decorate_logs() 调用）
#   E3 主进程周期统计行 “Engine 000: Avg prompt throughput ... Prefix cache hit rate”
#      （源码锚 v1/metrics/loggers.py::LoggingStatLogger.log，间隔由
#      VLLM_LOG_STATS_INTERVAL 控制，默认 10s，本跑设 2s 便于截获）
#   E4 VLLM_LOGGING_LEVEL=DEBUG 开闸后能看到的额外引擎侧日志
# 运行：cd <workspace> && source env.sh && npu-smi info 选卡后
#   ASCEND_RT_VISIBLE_DEVICES=<卡> VLLM_LOG_STATS_INTERVAL=2. \
#   VLLM_LOGGING_LEVEL=INFO python code/.../mp_default_demo.py
# 产物：stdout；由外层 tee 到 log/book7-r6/mp_default_demo.log 供 grep 证据
import os
import time


def main() -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(model="Qwen/Qwen3-0.6B", dtype="bfloat16", max_model_len=1024,
              gpu_memory_utilization=0.4, enforce_eager=True,
              disable_log_stats=False)  # LLM 默认关统计（llm.py:225-226），显式开
    t0 = time.perf_counter()
    outs = llm.generate(["Count from one to ten:"] * 4,
                        SamplingParams(temperature=0.0, max_tokens=384,
                                       ignore_eos=True))
    dur = time.perf_counter() - t0
    print(f"[mp_default] generated {len(outs)} completions in {dur:.1f}s "
          f"(log window >2s 以触发统计行)")


if __name__ == "__main__":
    main()
