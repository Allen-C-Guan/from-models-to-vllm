# vllm_smoke_reg.py —— Book7 00-feasibility 代码件：大项目 A 路（外部注册）NPU 冒烟
# 用途：验证「自写类名 Llama215ForCausalLM 经 model_class_overrides 注册进运行栈并加载同一份 207M 权重」
#       全链可用——大项目 C+A 双档的 A 路冒烟（notes/07 §5.1 设计、§5.2 判据 A1-A4）：
#       A1 解析断言（resolved 类=自写类且非 Transformers* 泛型——防 model_impl=auto 静默回退）；
#       A2 权重严格对账（AutoWeightsLoader track_weights_loading 默认开，漏 key 必炸）；
#       A3 determinism（贪心两轮逐字一致）；A4 跨路线对拍（与 C 路 vllm_smoke_eager1 同 prompt 同参数
#       生成逐字一致——「内置类与自写类同一份权重、同一输出」=注册机制正确性的端到端证明）。
# 所属章节：Book7 ch10（大项目 A 路）；设计书=plan/Book7-vLLM源码剖析上.md §3/§10.7
# 运行：cd <workspace> && source env.sh && npu-smi info 选卡 && ASCEND_RT_VISIBLE_DEVICES=<卡> \
#        PYTHONPATH=code/Book7-vLLM源码剖析上/00-feasibility \
#        python code/Book7-vLLM源码剖析上/00-feasibility/vllm_smoke_reg.py [--eager] [--out-name regA1]
# 产物：log/book7-00/vllm_smoke_reg_<out-name>.json（含 cross_route_parity 判据读数）
# 注意：hf207m_reg/ 是 hf207m/ 的副本、仅改 config.json 的 architectures 字段（C 路资产不动）。
import argparse
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC_DIR = os.path.join(WS, "log", "book7-00", "hf207m")
REG_DIR = os.path.join(WS, "log", "book7-00", "hf207m_reg")      # A 路模型目录（architectures 改名）
C_JSON = os.path.join(WS, "log", "book7-00", "vllm_smoke_eager1.json")  # C 路读数（A4 对拍基准）
OUT_DIR = os.path.join(WS, "log", "book7-00")

PROMPTS = [
    "The capital of France is",
    "Machine learning is a field of",
    "One, two, three, four,",
]
ARCH = "Llama215ForCausalLM"


def ensure_reg_dir() -> str:
    """A 路模型目录：复制 C 路 HF 目录、仅改 architectures 字段。"""
    cfg_p = os.path.join(REG_DIR, "config.json")
    if not os.path.exists(cfg_p):
        os.makedirs(REG_DIR, exist_ok=True)
        for f in os.listdir(SRC_DIR):
            if f != "model.safetensors":                      # 大文件用硬链接（同盘）
                shutil.copyfile(os.path.join(SRC_DIR, f), os.path.join(REG_DIR, f))
        os.link(os.path.join(SRC_DIR, "model.safetensors"), os.path.join(REG_DIR, "model.safetensors"))
        with open(os.path.join(SRC_DIR, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["architectures"] = [ARCH]
        with open(cfg_p, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=1)
    return REG_DIR


def main() -> None:
    ap = argparse.ArgumentParser(description="大项目 A 路注册冒烟（NPU）")
    ap.add_argument("--eager", action="store_true", help="enforce_eager=True（冒烟最小风险档）")
    ap.add_argument("--out-name", default="regA1")
    a = ap.parse_args()
    reg_dir = ensure_reg_dir()

    # A1 预检（CPU 侧，不起引擎）：注册-解析链路先证（notes/07 §1.1.3 探针同款）
    from vllm import ModelRegistry
    import llama215_vllm                                            # PYTHONPATH 注入后可达
    ModelRegistry.register_model(ARCH, llama215_vllm.Llama215ForCausalLM)
    assert ARCH in ModelRegistry.get_supported_archs(), "A1 预检失败：注册后架构名不在支持集"

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=reg_dir,
        dtype="bfloat16",
        max_model_len=1024,
        gpu_memory_utilization=0.4,
        enforce_eager=a.eager,
        model_class_overrides={ARCH: "llama215_vllm:Llama215ForCausalLM"},  # worker 侧经此重注册
    )

    # A1 终检：引擎实际解析到的类必须是自写类（防 auto 静默回退 transformers backend）
    mc = llm.llm_engine.vllm_config.model_config
    resolved = mc.registry.resolve_model_cls([ARCH], mc)[0]
    resolved_name = f"{resolved.__module__}.{resolved.__qualname__}"
    a1_ok = resolved_name.startswith("llama215_vllm.") and not resolved_name.startswith("transformers")

    sp = SamplingParams(temperature=0.0, max_tokens=32)
    outs = llm.generate(PROMPTS, sp)
    texts_r1 = [o.outputs[0].text for o in outs]
    outs2 = llm.generate(PROMPTS, sp)
    texts_r2 = [o.outputs[0].text for o in outs2]
    a3_ok = texts_r1 == texts_r2

    # A4 跨路线对拍：与 C 路（内置 LlamaForCausalLM 直载）同 prompt 同参数逐字比
    a4 = None
    if os.path.exists(C_JSON):
        with open(C_JSON, encoding="utf-8") as f:
            c = json.load(f)
        c_texts = [o["completion"] for o in c["outputs"]]
        identical = c_texts == texts_r1
        first_div = next((i for i, (x, y) in enumerate(zip(c_texts, texts_r1)) if x != y), None)
        a4 = {"vs": "vllm_smoke_eager1", "identical": bool(identical),
              "first_divergence_prompt": first_div}

    res = dict(
        route="A", arch_registered=ARCH, resolved_class=resolved_name, a1_resolve_ok=bool(a1_ok),
        enforce_eager=a.eager, determinism_ok=bool(a3_ok), cross_route_parity=a4,
        outputs=[{"prompt": p, "completion": t} for p, t in zip(PROMPTS, texts_r1)],
        runtime="vllm 0.27.1 + vllm-ascend 0.27.1rc1（Atlas 910B3）",
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, f"vllm_smoke_reg_{a.out_name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"[产物] {p}")


if __name__ == "__main__":
    main()
