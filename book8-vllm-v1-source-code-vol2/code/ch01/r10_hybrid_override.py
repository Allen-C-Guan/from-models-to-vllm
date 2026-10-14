# r10_hybrid_override.py —— 【Book8 批前预跑①】hf207m 零下载 hybrid 变身（R10 待核项 1，2026-10-12）
# 目的：验证 --hf-overrides 注入 layer_types（full_attention/sliding_attention 交替）+sliding_window
#       能否把满注意力 hf207m 变成双组混合模型并激活 Hybrid 协调器（运行栈 0.27.1+vllm-ascend）。
#       过则 Book8 ch1 实验面升「仿真+本机引擎实测」两段式（大纲 v0.9 §1.6）。
# 判据：J1 引擎带 overrides 正常起；J2 hybrid 证据（日志出现滑窗组/混合组/两组块规格——grep 底档）；
#       J3 生成非空且贪心两轮一致。
# 运行：cd <workspace> && source env.sh && ASCEND_RT_VISIBLE_DEVICES=2 \
#        python log/book8-00/r10_hybrid_override.py
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, "..", ".."))
MODEL = os.path.join(WS, "log/book7-00/hf207m")

LAYER_TYPES = ["full_attention", "sliding_attention"] * 6   # 12 层交替
SLIDING_WINDOW = 256


def main() -> None:
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=MODEL, dtype="bfloat16", max_model_len=1024,
        gpu_memory_utilization=0.4, enforce_eager=True,
        hf_overrides={"layer_types": LAYER_TYPES, "sliding_window": SLIDING_WINDOW},
    )
    prompts = [
        "The capital of France is Paris. The capital of Germany is",
        "one two three four five six seven",
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=24)
    t1 = [o.outputs[0].text for o in llm.generate(prompts, sp)]
    t2 = [o.outputs[0].text for o in llm.generate(prompts, sp)]
    ok = all(t.strip() for t in t1) and t1 == t2
    res = dict(probe="r10_hybrid_override", j1_engine_up=True, j2_hybrid_evidence="见 log grep（sliding/group/block）",
               j3_determinism=bool(ok),
               cfg=f"layer_types=FA/SW 交替×12, sliding_window={SLIDING_WINDOW}",
               outputs=[{"prompt": p, "completion": t} for p, t in zip(prompts, t1)],
               runtime="vllm 0.27.1 + vllm-ascend 0.27.1rc1（Atlas 910B3）", date="2026-10-12")
    with open(os.path.join(HERE, "r10_hybrid_override.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "outputs"}, ensure_ascii=False))
    print("[产物]", os.path.join(HERE, "r10_hybrid_override.json"))


if __name__ == "__main__":
    main()
