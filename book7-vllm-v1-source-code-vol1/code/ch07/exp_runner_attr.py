# exp_runner_attr.py —— Book7 ch07 代码件：NPU runner 归属两臂观察（判据 J1-J5 预注册）
# 用途：ch07 实验①——验证 NPU 双 runner 分派的实际归属与 patch env-only 判据：
#   臂 A（默认）：MRV1（patch_use_v2_model_runner 未设 env→恒 False）——J1 日志 [model_runner_v1.py: 前缀行；
#     J2 全文无「Model Runner V2」上游回退警告（判据链被 patch 旁路的负证据）；J3 config 回显含 AscendCompiler。
#   臂 B（翻转）：VLLM_USE_V2_MODEL_RUNNER=1——J4 日志出现「npu model runner v2 is in developing」（worker.py#L492 原话）；
#     J5 加载行前缀变 [v2/model_runner.py:。风险预注册：臂 B 若启动失败（V2 未成熟），失败栈本身入档为
#     「opt-in 的成本」读数，J4 在失败前日志中仍可核——负结果如实。
# 所属章节：Book7 ch07（§7.3/§7.5 实验）；设计书=notes/15 §8。
# 运行：cd <workspace> && source env.sh && npu-smi info 选卡 && ASCEND_RT_VISIBLE_DEVICES=<卡> \
#        python code/Book7-vLLM源码剖析上/ch07/exp_runner_attr.py [--arm A|B] [--out-name base]
# 产物：log/book7-ch07/runner_attr_<arm>_<out-name>.json
# 纪律：一臂一进程（NPU 同进程双 LLM 必死——notes/06 §9）；__main__ 守卫。
import argparse
import json
import os
import re
import time


def main() -> None:
    ap = argparse.ArgumentParser(description="ch07 runner 归属两臂")
    ap.add_argument("--arm", choices=["A", "B"], required=True)
    ap.add_argument("--out-name", default="base")
    a = ap.parse_args()
    if a.arm == "B":
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"      # 必须在 import vllm 前

    from vllm import LLM, SamplingParams
    WS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    MODEL = os.path.join(WS, "log", "book7-00", "hf207m")
    OUT = os.path.join(WS, "log", "book7-ch07")

    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=0.4,
              enforce_eager=True, disable_log_stats=False)
    t0 = time.perf_counter()
    outs = llm.generate(["One, two, three,"], SamplingParams(temperature=0.0, max_tokens=8))
    gen_s = round(time.perf_counter() - t0, 2)
    text = outs[0].outputs[0].text

    # 对象级断言：默认多进程（SyncMPClient）下 EngineCore 在子进程——对象树不可达。
    # 判据改用日志前缀（文件名即答案——notes/15 §1.G.4 的最廉价观测法）：
    # 本进程把 vllm 日志重定向到文件后 grep。
    mc = llm.llm_engine.vllm_config
    log_lines = open(os.environ.get("BOOK7_RUNNER_LOG", "/dev/null"), encoding="utf-8", errors="ignore").read() \
        if os.environ.get("BOOK7_RUNNER_LOG") else ""
    import re as _re
    m_v1 = _re.findall(r"\[(model_runner_v1\.py):(\d+)\]", log_lines)
    m_v2 = _re.findall(r"\[(v2/model_runner\.py):(\d+)\]", log_lines)
    runner_cls = "(from log prefix) " + (m_v1[0][0] if m_v1 else (m_v2[0][0] if m_v2 else "not-found"))
    res = dict(arm=a.arm, env_v2=os.environ.get("VLLM_USE_V2_MODEL_RUNNER"),
               use_v2_model_runner=bool(mc.use_v2_model_runner),
               runner_class=runner_cls,
               J1_v1_prefix=bool(m_v1), J5_v2_prefix=bool(m_v2),
               v1_lines=len(m_v1), v2_lines=len(m_v2),
               gen_s=gen_s, sample=text,
               runtime="vllm 0.27.1 + vllm-ascend 0.27.1rc1（Atlas 910B3）",
               note="判据=日志前缀 grep（BOOK7_RUNNER_LOG 指向重定向的运行日志；文件名即答案）")
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"runner_attr_{a.arm}_{a.out_name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"[产物] {p}")


if __name__ == "__main__":
    main()
