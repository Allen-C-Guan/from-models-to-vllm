# 用途：Book8 ch08 实验——fp8 检查点在 910B3 上的反量化服务观测（双臂各一进程）
#       臂 A：Qwen/Qwen3-1.7B-FP8（fp8 检查点）；臂 B：Qwen/Qwen3-1.7B（bf16 对照臂）
#       判据预注册（沿 Book7 ch10 §10.5 + notes/18 §五）：若 fp8_block.py docstring 为真，
#       两臂生成速率同量级；实测第一轮发现 fp8 臂装载期即撞算子错（aclnnInplaceCopy
#       561103 @ resolve_block_scales）——装载失败本身按负结果分支落档，不吞不改。
# 所属章节：code/Book8-vLLM源码剖析下/ch08/
# 运行方式：source env.sh 后，每臂各起一进程（一进程一 LLM 纪律）：
#   ASCEND_RT_VISIBLE_DEVICES=6 python exp_ch08_fp8_dequant.py --arm fp8 --out-name exp_ch08_fp8_arm.json
# 口径：贪心；TPOT=两点法（单请求 max_tokens=1 与 128 的墙钟差 /127——offline 路径
#       RequestOutput.metrics 恒 None，改两点法+聚合吞吐；两臂同协议，趋势级可比）。

import argparse
import json
import os
import time
import traceback

PROMPTS = [
    "请用三段话介绍一下注意力机制的历史，从 2014 年的 Bahdanau 注意力讲到 2017 年的 Transformer。",
    "Explain why decode-phase LLM inference is memory-bandwidth bound, in about 150 words.",
    "把下面这句话翻译成英文并解释其中的双关：工程师最喜欢的浪漫是「我对你的爱是 O(1)」。",
    "写一个 Python 函数计算两个大整数矩阵乘法的朴素实现，并注释每一步的形状变化。",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["fp8", "bf16"], required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()

    model = {"fp8": "Qwen/Qwen3-1.7B-FP8", "bf16": "Qwen/Qwen3-1.7B"}[args.arm]
    out_name = args.out_name or f"exp_ch08_{args.arm}_arm.json"
    out_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "..", "..", "..", "log", "book8-ch08"))

    rec = {
        "exp": "book8-ch08 fp8 dequant service observation (two-point TPOT protocol)",
        "arm": args.arm,
        "model": model,
        "hardware": "Atlas 910B3, runtime vllm 0.27.1 + vllm-ascend 0.27.1rc1",
        "config": {"max_model_len": args.max_model_len,
                   "gpu_memory_utilization": args.gpu_mem_util,
                   "temperature": 0.0, "max_tokens": args.max_tokens,
                   "num_prompts": len(PROMPTS), "prefix_caching": False},
    }

    try:
        from vllm import LLM, SamplingParams

        t_start = time.time()
        llm = LLM(
            model=model,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,
            enable_prefix_caching=False,
        )
        t_loaded = time.time()
        rec["load_s"] = round(t_loaded - t_start, 1)
        rec["load_failed"] = False

        # ①两点法 TPOT：同一 prompt，1 token 与 128 token 两次生成（前缀缓存已关）
        g = SamplingParams(temperature=0.0, max_tokens=1)
        t0 = time.time(); llm.generate([PROMPTS[0]], g, use_tqdm=False); t1 = time.time()
        g128 = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
        t0b = time.time(); out1 = llm.generate([PROMPTS[0]], g128, use_tqdm=False); t2 = time.time()
        n1 = len(out1[0].outputs[0].token_ids)
        rec["two_point"] = {
            "t_gen1_s": round(t1 - t0, 3),
            "t_gen128_s": round(t2 - t0b, 3),
            "n_tokens": n1,
            "tpot_ms": round((t2 - t0b - (t1 - t0)) * 1000.0 / max(1, n1 - 1), 2),
            "ttft_proxy_ms": round((t1 - t0) * 1000.0, 2),
        }

        # ②四请求聚合吞吐
        t3 = time.time()
        outs = llm.generate(PROMPTS, g128, use_tqdm=False)
        t4 = time.time()
        total_tok = sum(len(o.outputs[0].token_ids) for o in outs)
        rec["batch4"] = {
            "wall_s": round(t4 - t3, 3),
            "total_tokens": total_tok,
            "output_toks_per_s": round(total_tok / (t4 - t3), 1),
        }

        hbm = {}
        try:
            import torch
            import torch_npu  # noqa: F401
            free, total = torch.npu.mem_get_info()
            hbm = {"free_GB": round(free / 1e9, 2), "total_GB": round(total / 1e9, 2)}
        except Exception as e:  # noqa: BLE001
            hbm = {"note": f"mem_get_info unavailable: {e}"}
        rec["hbm_after_run"] = hbm
        rec["text_head"] = outs[0].outputs[0].text[:60]
    except Exception as e:  # noqa: BLE001
        rec["load_failed"] = True
        rec["error_type"] = type(e).__name__
        rec["error_text"] = str(e)[:500]
        rec["traceback_tail"] = traceback.format_exc().strip().split("\n")[-12:]

    rec["preregistered_expectation"] = ("two arms generation rate same order of magnitude "
                                        "(fp8 served as bf16 on 910B3 per fp8_block.py docstring)")
    out = os.path.join(out_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("[exp] wrote", out)
    if rec.get("load_failed"):
        print("[exp]", args.arm, "LOAD FAILED:", rec["error_text"][:200])
    else:
        print("[exp]", args.arm, "tpot_ms =", rec["two_point"]["tpot_ms"],
              "toks/s =", rec["batch4"]["output_toks_per_s"],
              "load_s =", rec["load_s"])


if __name__ == "__main__":
    main()
