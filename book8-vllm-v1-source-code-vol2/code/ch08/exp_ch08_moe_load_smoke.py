# 用途：Book8 ch08 实验——MoE w4a16 检查点装载冒烟（ch6 欠账「ch8 开工首跑」的兑现；
#       权重 blob 已于 2026-10-13 上午补全落位——本脚本按 J3 预注册分支跑：装得上=读张量
#       计数/装载日志行/一次生成；装不上=负结果原文落档）
# 所属章节：code/Book8-vLLM源码剖析下/ch08/（notes/18 §五；正文 8.7）
# 运行方式：source env.sh && ASCEND_RT_VISIBLE_DEVICES=<卡> python exp_ch08_moe_load_smoke.py

import json
import os
import time
import traceback

MODEL = "nm-testing/Qwen1.5-MoE-A2.7B-Chat-quantized.w4a16"


def main() -> None:
    rec = {
        "exp": "book8-ch08 MoE w4a16 checkpoint load smoke (ch6 promise: first run at ch8)",
        "model": MODEL,
        "hardware": "Atlas 910B3, runtime vllm 0.27.1 + vllm-ascend 0.27.1rc1",
        "note": "arch qwen2_moe NOT in ascend supported_models green list (notes/16 R4) - J3 negative branch preregistered",
    }
    try:
        from vllm import LLM, SamplingParams

        t0 = time.time()
        llm = LLM(model=MODEL, max_model_len=4096, gpu_memory_utilization=0.85,
                  enable_prefix_caching=False)
        t1 = time.time()
        rec["load_s"] = round(t1 - t0, 1)
        outs = llm.generate(["用一句话说明什么是混合专家模型。"],
                            SamplingParams(temperature=0.0, max_tokens=32), use_tqdm=False)
        t2 = time.time()
        rec["gen_s"] = round(t2 - t1, 3)
        rec["text"] = outs[0].outputs[0].text[:120]
        rec["load_ok"] = True
    except Exception as e:  # noqa: BLE001
        rec["load_ok"] = False
        rec["error_type"] = type(e).__name__
        rec["error_text"] = str(e)[:600]
        rec["traceback_tail"] = traceback.format_exc().strip().split("\n")[-10:]

    out = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "..", "..", "log", "book8-ch08",
                                       "exp_ch08_moe_load_smoke.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("[moe-smoke] load_ok =", rec.get("load_ok"),
          "| load_s =", rec.get("load_s"), "| text/error =",
          (rec.get("text") or rec.get("error_text", ""))[:120])
    print("[moe-smoke] wrote", out)


if __name__ == "__main__":
    main()
