# 用途：Book8 ch08 实验——进门账探针：给平台层喂一个白名单外的量化方法名，
#       捕获报错原文（报错文案应自带设备名与白名单六法——verify_quantization 机制的正向验证）。
#       CPU 即可跑（引擎 config 构建期即抛，不触碰 NPU）。
# 所属章节：code/Book8-vLLM源码剖析下/ch08/（notes/18 §五 E-probe）
# 运行方式：source env.sh && python exp_ch08_whitelist_probe.py

import json
import os


def main() -> None:
    from vllm.engine.arg_utils import EngineArgs

    err_text = None
    err_type = None
    try:
        # 本地在位模型 + 白名单外方法名 gptq（社区最流行、NPU 白名单不在列的那家）
        EngineArgs(model="Qwen/Qwen3-0.6B", quantization="gptq").create_engine_config()
        err_text = "NO ERROR RAISED (unexpected)"
    except Exception as e:  # noqa: BLE001
        err_type = type(e).__name__
        err_text = str(e)

    rec = {
        "exp": "book8-ch08 checkpoint whitelist door probe",
        "probe": "EngineArgs(model=Qwen3-0.6B, quantization='gptq').create_engine_config()",
        "error_type": err_type,
        "error_text": err_text,
        "runtime": "vllm 0.27.1 + vllm-ascend 0.27.1rc1",
    }
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                           "log", "book8-ch08")
    out = os.path.abspath(os.path.join(out_dir, "exp_ch08_whitelist_probe.json"))
    os.makedirs(out_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("[probe]", err_type, ":", err_text[:400])
    print("[probe] wrote", out)


if __name__ == "__main__":
    main()
