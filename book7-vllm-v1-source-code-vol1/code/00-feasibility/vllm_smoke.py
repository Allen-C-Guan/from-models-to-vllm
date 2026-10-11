# vllm_smoke.py —— Book7 00-feasibility 代码件：HF 格式 207M 目录经 vllm 运行栈（NPU）加载冒烟
# 用途：大项目可行性预跑的正身——vllm 0.27.1+vllm-ascend 0.27.1rc1 以内置 LlamaForCausalLM 注册路径
#       加载 log/book7-00/hf207m/（make_hf207m.py 产物，llama215 对拍 1.05e-05），offline LLM API 生成，
#       验证「Book3 终态模型注册进 vllm-ascend」全链可用；记录引擎侧事实（后端选择/attention/kv 块）
#       与质量 sanity（贪心续写非乱码、固定 prompt 两轮一致）。
# 所属章节：Book7 ch10（大项目可行性）；设计书=plan/Book7-vLLM源码剖析上.md
# 运行：cd <workspace> && source env.sh && npu-smi info 选卡 && ASCEND_RT_VISIBLE_DEVICES=<卡> \
#        python code/Book7-vLLM源码剖析上/00-feasibility/vllm_smoke.py [--eager] [--out-name base]
# 产物：log/book7-00/vllm_smoke_<out-name>.json（引擎事实+生成样例+计时）
# 依据：vLLM 脚本纪律——落盘+__main__ 守卫（spawn 重导入）；权重 fp32→dtype bfloat16 由引擎铸。
import argparse
import json
import os
import time

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                          "log", "book7-00", "hf207m"))
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                       "log", "book7-00"))

PROMPTS = [
    "The capital of France is",
    "Machine learning is a field of",
    "One, two, three, four,",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="vllm NPU 冒烟：207M HF 目录加载+生成")
    ap.add_argument("--eager", action="store_true", help="enforce_eager=True（跳过图模式——冒烟最小风险档）")
    ap.add_argument("--out-name", default="base")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    t0 = time.perf_counter()
    llm = LLM(
        model=MODELS_DIR,
        dtype="bfloat16",
        max_model_len=1024,
        gpu_memory_utilization=0.4,          # 207M 小模型，低占用档
        enforce_eager=a.eager,
        disable_log_stats=False,
    )
    t_load = time.perf_counter() - t0

    sp = SamplingParams(temperature=0.0, max_tokens=32)   # 贪心——两轮一致即确定性 sanity
    t1 = time.perf_counter()
    outs = llm.generate(PROMPTS, sp)
    t_gen = time.perf_counter() - t1
    texts_r1 = [o.outputs[0].text for o in outs]
    outs2 = llm.generate(PROMPTS, sp)                     # 第二轮：确定性检查
    texts_r2 = [o.outputs[0].text for o in outs2]

    # 引擎侧事实（供 ch10/ch12 正文与欠账核对；字段缺失不炸——版本差异容错）
    facts = {}
    try:
        eng = llm.llm_engine
        cfg = getattr(eng, "engine_config", None) or getattr(eng, "vllm_config", None)
        if cfg is not None:
            cache = getattr(cfg, "cache_config", None)
            if cache is not None:
                facts["block_size"] = getattr(cache, "block_size", None)
                facts["gpu_mem_util"] = getattr(cache, "gpu_memory_utilization", None)
            mc = getattr(cfg, "model_config", None)
            if mc is not None:
                facts["dtype"] = str(getattr(mc, "dtype", None))
                facts["hf_arch"] = getattr(getattr(mc, "hf_config", None), "architectures", None)
    except Exception as e:  # noqa: BLE001 —— 事实采集失败不影响冒烟判定
        facts["collect_error"] = str(e)

    res = dict(
        model_dir=MODELS_DIR, enforce_eager=a.eager,
        load_s=round(t_load, 1), gen_s=round(t_gen, 2),
        determinism_ok=(texts_r1 == texts_r2),
        outputs=[{"prompt": p, "completion": t} for p, t in zip(PROMPTS, texts_r1)],
        engine_facts=facts,
        runtime="vllm 0.27.1 + vllm-ascend 0.27.1rc1（Atlas 910B3）",
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, f"vllm_smoke_{a.out_name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"[产物] {p}")


if __name__ == "__main__":
    main()
