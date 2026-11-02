# exp_backend_probe.py —— Book7 ch09 代码件：后端选择探针（不起引擎、秒级）
# 用途：ch09 实验①——直接调 selector.get_attn_backend 探测 NPU 的后端选择：
#   判据 A：常规 GQA（head_size=64, bf16）→ AscendAttentionBackend（attention_v1.py#L79）；
#   变臂 B：use_mla=True → AscendMLABackend（mla_v1.py）；
#   变臂 C（负例）：use_sparse=True 且 use_mla=False → 预注册 KeyError/报错形态（如实抄录）。
# 所属章节：Book7 ch09（§9.3/§9.6 实验）；设计书=notes/17 §8。
# 运行：cd <workspace> && source env.sh && python code/Book7-vLLM源码剖析上/ch09/exp_backend_probe.py（纯 CPU 可跑，无需选卡）
# 产物：log/book7-ch09/backend_probe.json
import json, os

WS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
OUT = os.path.join(WS, "log", "book7-ch09")


def probe(tag, **kw):
    try:
        # 探针路线 B：绕过 selector 懒导入链（device_op 的环形导入在纯 CPU 进程触发），
        # 直接调平台的查表函数——selector 在引擎里的最终委托点（selector.py#L208-212）
        from vllm_ascend.platform import NPUPlatform
        from types import SimpleNamespace
        import vllm_ascend.ascend_config as ac
        from vllm.config import VllmConfig, set_current_vllm_config
        cfg = VllmConfig()
        with set_current_vllm_config(cfg):
            ac.init_ascend_config(cfg)          # 全局 _ASCEND_CONFIG 就位（守卫放行）
        cfg_ns = SimpleNamespace(use_mla=kw.get("use_mla", False),
                                 use_sparse=kw.get("use_sparse", False),
                                 use_compress=False, use_pcp=False, use_dcp=False)
        cls = NPUPlatform.get_attn_backend_cls(selected_backend=None,
                                               attn_selector_config=cfg_ns)
        backend = cls if isinstance(cls, str) else f"{cls.__module__}.{cls.__qualname__}"
        return dict(tag=tag, kwargs={k: str(v) for k, v in kw.items()},
                    backend=backend)
    except Exception as e:  # noqa: BLE001 —— 负例的报错形态本身是判据
        return dict(tag=tag, kwargs={k: str(v) for k, v in kw.items()},
                    error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    import torch
    res = {}
    res["A_gqa"] = probe("A_gqa", head_size=64, dtype=torch.bfloat16, kv_cache_dtype=None)
    res["B_mla"] = probe("B_mla", head_size=64, dtype=torch.bfloat16, kv_cache_dtype=None,
                         use_mla=True)
    res["C_sparse_neg"] = probe("C_sparse_neg", head_size=64, dtype=torch.bfloat16,
                                kv_cache_dtype=None, use_sparse=True)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "backend_probe.json")
    json.dump(res, open(p, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print("[产物]", p)
