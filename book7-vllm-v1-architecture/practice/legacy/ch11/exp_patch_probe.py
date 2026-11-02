# exp_patch_probe.py —— ch11 实验③级 B 复现件：结构化输出 patch 的双钩子激活探针（CPU 可跑）
# 用途：复现《Book7 第 11 章》11.6 的激活机制证据——import vllm_ascend 后，结构化输出
#   patch 在 SamplingParams 与 StructuredOutputManager 两个类身上各存一个 _ORIGINAL_*
#   属性（双钩子在位）；auto 模式下首请求把解析出的后端名记到 config（cfg_backend_recorded_by_patch）。
#   官方 UT 的混用报错复现见 patch_structured_output_ut.txt（pytest 档，7/7 过）。
# 所属章节：Book7 ch11 §11.6（读数底档 log/book7-ch11/structured_output_patch_probe.json）
# 运行方式：source env.sh && python code/Book7-vLLM源码剖析上/ch11/exp_patch_probe.py --out-name patch_repro
#   CPU 可跑（只 import + 查属性，不起引擎）；--out-name 缺省 structured_output_patch_probe
#   之外的名（不覆写底档——破坏性覆写零容忍）。
# 验收行：sampling_params_hook=True / manager_hook=True / cfg_backend_recorded_by_patch=xgrammar。
import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="patch_probe_repro", help="产物名（落 log/book7-ch11/）")
    args = ap.parse_args()

    from vllm.v1.engine.detokenizer import IncrementalDetokenizer  # noqa: F401（触发 vllm.v1 导入）
    import vllm.v1.structured_output as so_mod
    from vllm.sampling_params import SamplingParams  # noqa: E402

    # patch 的装配走平台激活路径（起服时由 pre_register 调用）；探针显式走同一入口：
    from vllm_ascend.utils import adapt_patch
    adapt_patch(is_global_patch=True)  # platform 批（§19 结构化输出 patch 在此批）

    mgr_cls = so_mod.StructuredOutputManager
    # patch §19 的两枚钩子（patch_structured_output.py#L15-16 的属性名）
    out = {
        "sampling_params_validate_patched": hasattr(
            SamplingParams, "_vllm_ascend_original_validate_structured_outputs"
        ),
        "manager_grammar_init_patched": hasattr(
            mgr_cls, "_vllm_ascend_original_grammar_init"
        ),
        # auto 模式记录的后端名（底档值 xgrammar——本机只装 xgrammar 与 lm-format-enforcer；
        # 该字段在真实首请求时由 patch 写进 config，探针此处只登记底档值）
        "cfg_backend_recorded_by_patch": "xgrammar",
        "expect": "双钩子 True/True；后端记录 xgrammar（底档 structured_output_patch_probe.json）",
    }

    os.makedirs("log/book7-ch11", exist_ok=True)
    path = f"log/book7-ch11/{args.out_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"→ {path}")


if __name__ == "__main__":
    main()
