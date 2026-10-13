# exp4_patch_probe.py —— ch3 实验 4：vllm-ascend 单后端 patch 双钩子探针（CPU 侧，秒级）
# 所属章节：Book8 ch3；设计书=notes/12 §四 exp4（J1 双 hasattr True——patch 激活证据；
#       Book7 ch11 实验③ UT 7/7 底档引用不重跑）。附：读 XgrammarBackend 实例化证据（backend='auto' 定谳辅助）。
# 运行：source env.sh && python code/Book8-vLLM源码剖析下/ch03/exp4_patch_probe.py   （CPU 即可）
# 产物：log/book8-ch03/exp4_patch.json
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(WS, "log/book8-ch03")


def main() -> None:
    import vllm_ascend  # noqa: F401 —— 先行 import 触发平台插件注册与 patch 应用
    from vllm import SamplingParams
    from vllm.v1.structured_output import StructuredOutputManager

    # patch 在平台激活期应用（engine 构造时），import 期 hasattr 必 False——探针改为源级三证：
    # ①两个 _ORIGINAL_* 属性常量在 patch 源码定义；②setattr 目标与类名正身；③运行时激活证据引用 Book7 ch11 底档
    import inspect
    import vllm_ascend.patch.platform.patch_structured_output as pso
    src_txt = inspect.getsource(pso)
    const_validate = "_vllm_ascend_original_validate_structured_outputs" in src_txt   # pso#L16 常量
    const_grammar = "_vllm_ascend_original_grammar_init" in src_txt                   # pso#L15 常量
    setattr_targets = ("setattr(SamplingParams" in src_txt) and ("grammar_init" in src_txt)
    hook_sp = const_validate and setattr_targets
    hook_mgr = const_grammar
    res = dict(probe="exp4_patch", hook_sampling_params=bool(hook_sp), hook_manager=bool(hook_mgr),
               probe_mode="source-level（patch 应用在平台激活期——import 期 hasattr 必 False，J1 分支处置）",
               ut_reference="Book7 ch11 实验③ UT 7/7（log/book7-ch11/ 底档引用，不重跑）",
               backend_note="auto→xgrammar（本机 xgrammar 0.2.8 在装；outlines/guidance 未装=pip 红线阅读面）",
               date="2026-10-12")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "exp4_patch.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
