# exp3_correctness.py —— ch3 实验 3：结构化输出正确性（jsonschema 双断言+贪心两轮一致）
# 所属章节：Book8 ch3；设计书=notes/12 §四 exp3（J1 32/32 双过；J2 两轮 hash 一致）
# 运行：source env.sh && ASCEND_RT_VISIBLE_DEVICES=<卡> VLLM_ENABLE_V1_MULTIPROCESSING=0 \
#        python code/Book8-vLLM源码剖析下/ch03/exp3_correctness.py
# 产物：log/book8-ch03/exp3_correct.json
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(WS, "log/book8-ch03")
sys.path.insert(0, HERE)
from schemas_ch03 import VARIANTS, PROMPTS, CORRECTNESS_SET


def main() -> None:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    import jsonschema

    llm = LLM(model="Qwen/Qwen3-0.6B", dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.4, enforce_eager=True, seed=0)
    rows = []
    for name in CORRECTNESS_SET:
        schema = VARIANTS[name]
        sp = SamplingParams(temperature=0.0, max_tokens=128,
                            structured_outputs=StructuredOutputsParams(json=schema))
        outs = []
        for _round in (1, 2):
            o = llm.generate([PROMPTS[name]] * 8, sp)   # 8 请求/变体
            outs.append([x.outputs[0].text for x in o])
        for i in range(8):
            text = outs[0][i]
            parse_ok = False
            schema_ok = False
            try:
                obj = json.loads(text)          # 严判：截断的 JSON 必须算 parse 失败
                parse_ok = True
                jsonschema.validate(instance=obj, schema=schema)
                schema_ok = True
            except json.JSONDecodeError:
                parse_ok = False
            except Exception:
                pass
            h1 = hashlib.md5(outs[0][i].encode()).hexdigest()[:10]
            h2 = hashlib.md5(outs[1][i].encode()).hexdigest()[:10]
            rows.append(dict(schema_id=name, req=i, parse_ok=parse_ok, schema_ok=schema_ok,
                             round1_hash=h1, round2_hash=h2, deterministic=(h1 == h2)))
    n_ok = sum(r["schema_ok"] for r in rows)
    n_det = sum(r["deterministic"] for r in rows)
    res = dict(meta=dict(date="2026-10-12", model="Qwen/Qwen3-0.6B", n=len(rows),
                         validator="jsonschema 4.26.0", runtime="vllm 0.27.1+vllm-ascend（Atlas 910B3）"),
               summary=dict(schema_ok=f"{n_ok}/{len(rows)}", deterministic=f"{n_det}/{len(rows)}"),
               rows=rows)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "exp3_correct.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res["summary"], ensure_ascii=False))
    print("[产物]", os.path.join(OUT, "exp3_correct.json"))


if __name__ == "__main__":
    main()
