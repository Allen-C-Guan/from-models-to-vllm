# exp1_compile_timing.py —— ch3 实验 1：grammar 编译时间（inproc 钩子法，两法中的主法）
# 所属章节：Book8 ch3；设计书=notes/12 §四 exp1（判据预注册 J1-J3 含结果分支）
# 判据：J1 miss 工况 compile_ms>0 记中位数（<5ms→如实报「小编译账」，叙事以 miss/hit 比值为主）；
#       J2 hit 工况 compile_ms≈0（非 0→查缓存键登记）；TTFT 差分留 exp2（serve 口径）。
# 运行：cd <workspace> && source env.sh && ASCEND_RT_VISIBLE_DEVICES=<空闲卡> \
#        VLLM_ENABLE_V1_MULTIPROCESSING=0 python code/Book8-vLLM源码剖析下/ch03/exp1_compile_timing.py [--smoke]
# 产物：log/book8-ch03/exp1_compile.json
import argparse
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(WS, "log/book8-ch03")
import sys
sys.path.insert(0, HERE)
from schemas_ch03 import VARIANTS, PROMPTS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="只跑 2 个变体（分文件）")
    a = ap.parse_args()
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from vllm import LLM, SamplingParams

    # inproc 钩子：包住 StructuredOutputManager._create_grammar 记 per-request compile_ms
    # （签名按运行栈 0.27.1 单参；v0.30.0 为双参 (request, structured_output_request)——差异 #15 系）
    from vllm.v1.structured_output import StructuredOutputManager
    timing = {"last_ms": None}
    orig = StructuredOutputManager._create_grammar

    def timed(self, request):
        t0 = time.perf_counter()
        r = orig(self, request)
        timing["last_ms"] = (time.perf_counter() - t0) * 1000.0
        return r

    StructuredOutputManager._create_grammar = timed

    llm = LLM(model="Qwen/Qwen3-0.6B", dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.4, enforce_eager=True, seed=0)
    names = list(VARIANTS) if not a.smoke else ["obj_basic", "enum_basic"]
    rows = []
    from vllm.sampling_params import StructuredOutputsParams
    for name in names:
        schema = VARIANTS[name]
        prompt = PROMPTS[name]
        for rep in ("miss", "hit"):
            timing["last_ms"] = None
            sp = SamplingParams(temperature=0.0, max_tokens=48,
                                structured_outputs=StructuredOutputsParams(json=schema))
            out = llm.generate([prompt], sp)
            ms = timing["last_ms"]
            rows.append(dict(schema_id=name, variant=name, rep=rep,
                             compile_ms=None if ms is None else round(ms, 3),
                             text_head=out[0].outputs[0].text[:60]))
            print(f"[{name}/{rep}] compile_ms={ms if ms is None else round(ms, 3)}", flush=True)
    StructuredOutputManager._create_grammar = orig

    os.makedirs(OUT, exist_ok=True)
    fname = "exp1_compile_smoke.json" if a.smoke else "exp1_compile.json"
    res = dict(meta=dict(date="2026-10-12", model="Qwen/Qwen3-0.6B", runtime="vllm 0.27.1+vllm-ascend（Atlas 910B3）",
                         method="inproc hook on StructuredOutputManager._create_grammar"),
               rows=rows)
    with open(os.path.join(OUT, fname), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("[产物]", os.path.join(OUT, fname))


if __name__ == "__main__":
    main()
