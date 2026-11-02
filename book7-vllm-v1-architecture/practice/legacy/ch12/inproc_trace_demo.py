# inproc_trace_demo.py —— Book7 ch12 方法②：单进程模式跟跑（VLLM_ENABLE_V1_MULTIPROCESSING=0）
# 证据链（源码锚，均已在运行栈 0.27.1 核过）：
#   - vllm/v1/engine/llm_engine.py:174  from_engine_args 中 VLLM_ENABLE_V1_MULTIPROCESSING=1
#     时强制 enable_multiprocessing=True —— 即该环境变量是 0.27.1 单/多进程的总开关
#   - vllm/v1/engine/core_client.py:90  EngineCoreClient.make_client 按
#     multiprocess_mode 分派：False→InprocClient（同进程直调）/True→SyncMPClient
#   - docs/usage/troubleshooting.md:63  官方调试指引：置 0 后 scheduler 与主进程同进程，
#     可用原生 pdb 断点
# 本脚本演示同进程「只读 instrumentation」三钩子（不改 site-packages 任何一行）：
#   钩1 make_client 分派结果（InprocClient 正身）
#   钩2 Scheduler.schedule 逐调度步打印 SchedulerStats（running/waiting/KV 用量/prefix 命中）
#   钩3 LLMEngine.step 步计数（主循环节奏）
# 运行：cd <workspace> && source env.sh && npu-smi info 选卡后
#   ASCEND_RT_VISIBLE_DEVICES=<卡> python code/Book7-vLLM源码剖析上/ch12/inproc_trace_demo.py
# 产物：stdout（跟跑证据行）+ log/book7-r6/inproc_trace_demo.json
import json
import os

# 官方推荐做法：在 import vllm 之前置 0（晚于 import 则 EngineArgs 缓存可能已读旧值）
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

MODELS = "Qwen/Qwen3-0.6B"  # 本地 HF 缓存已有；如需离线可换绝对路径
OUT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                 "log", "book7-r6", "inproc_trace_demo.json"))


def main() -> None:
    from vllm import LLM, SamplingParams
    from vllm.v1.engine import core_client, llm_engine
    from vllm.v1.core.sched import scheduler as sched_mod

    # ── 钩1：make_client 分派正身 ────────────────────────────────────────────
    orig_make = core_client.EngineCoreClient.make_client  # staticmethod，类上直取即函数

    def traced_make(*args, **kw):
        cls = orig_make(*args, **kw)  # 调用点 llm_engine.py:105 全以 kwargs 传参
        print(f"[hook1] make_client -> {type(cls).__name__} "
              f"(multiprocess_mode={kw.get('multiprocess_mode')}, "
              f"asyncio_mode={kw.get('asyncio_mode')})")
        return cls

    core_client.EngineCoreClient.make_client = staticmethod(traced_make)

    llm = LLM(model=MODELS, dtype="bfloat16", max_model_len=1024,
              gpu_memory_utilization=0.4, enforce_eager=True,
              disable_log_stats=False)  # LLM 默认关统计（llm.py:225-226），显式开

    # ── 钩2：Scheduler.schedule 逐调度步观测（同进程才能这样直接包） ─────────
    sched = llm.llm_engine.engine_core.engine_core.scheduler
    print(f"[hook2] scheduler instance = {type(sched).__module__}.{type(sched).__name__}")
    n_steps = {"n": 0}
    orig_schedule = type(sched).schedule

    def traced_schedule(self, *a, **kw):
        out = orig_schedule(self, *a, **kw)
        n_steps["n"] += 1
        # EngineCore.step 内部同款读法（sched/scheduler.py:2495 make_stats）；
        # log_stats=False 时返回 None，退回 scheduler 内部直读
        st = self.make_stats()
        pc = st.prefix_cache_stats if st is not None else None
        if n_steps["n"] <= 5 or n_steps["n"] % 20 == 0:
            if st is not None:
                print(f"[hook2] step#{n_steps['n']:>3} running={st.num_running_reqs} "
                      f"waiting={st.num_waiting_reqs} kv={st.kv_cache_usage:.2f} "
                      f"prefix(hits/queries={pc.hits}/{pc.queries} tokens) "
                      f"scheduled={len(out.scheduled_new_reqs)}new+"
                      f"{len(out.scheduled_cached_reqs.req_ids)}cached")
            else:
                print(f"[hook2] step#{n_steps['n']:>3} "
                      f"requests={len(self.requests)} "
                      f"scheduled={len(out.scheduled_new_reqs)}new")
        return out

    type(sched).schedule = traced_schedule

    # ── 钩3：LLMEngine.step 主循环计数 ──────────────────────────────────────
    n_engine = {"n": 0}
    orig_step = llm_engine.LLMEngine.step

    def traced_step(self, *a, **kw):
        n_engine["n"] += 1
        return orig_step(self, *a, **kw)

    llm_engine.LLMEngine.step = traced_step

    outs = llm.generate(["The capital of France is",
                         "Machine learning is a field of"],
                        SamplingParams(temperature=0.0, max_tokens=24))

    # 进程正身：单进程应无 EngineCore 子进程
    import psutil
    children = psutil.Process().children(recursive=True)
    body = [o.outputs[0].text for o in outs]
    print(f"[proc] child processes = {[(c.name(), c.pid) for c in children]}")
    print(f"[result] steps: engine={n_engine['n']} scheduler={n_steps['n']}")
    print(f"[result] completion[0] = {body[0]!r}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"child_procs": [(c.name(), c.pid) for c in children],
                   "engine_steps": n_engine["n"], "scheduler_steps": n_steps["n"],
                   "completions": body,
                   "mode": "VLLM_ENABLE_V1_MULTIPROCESSING=0"}, f, ensure_ascii=False, indent=1)
    print(f"[产物] {OUT}")


if __name__ == "__main__":
    main()
