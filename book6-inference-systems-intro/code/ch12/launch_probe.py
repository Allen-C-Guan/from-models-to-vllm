# launch_probe.py —— Book6 ch12 代码件：launch 开销 microbench 与图执行实证锚
# 用途：①kernel 发射开销实测：小算子串（如 207M 一层的逐算子序列）逐个发射 vs 等价大算子
#   一次发射的每步时延对照——launch bound 的直接证据（ch1 推断的实证收口）；
#   ②ACLGraph 实证锚（收窄口径：默认值读数+开关对拍，机制不展开——规范边界条款①）：
#   读 vllm-ascend 图模式默认配置 + npu 图执行接口在位性（黑盒）。
# 所属章节：Book6 ch12 §12.2/12.4；设计书=plan/Book6-推理系统导论.md ch12
# 运行：cd <workspace> && source env.sh && ASCEND_RT_VISIBLE_DEVICES=<卡> \
#      python code/Book6-推理系统导论/ch12/launch_probe.py [--out-name base]
# 产物：log/book6-ch12/launch_probe_<out-name>.json
import argparse
import json
import os
import time

import torch


def bench(fn, n=50, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / n * 1e3     # ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="base")
    a = ap.parse_args()
    assert torch.npu.is_available(), "本件为 NPU 性能件（选卡纪律见 env.sh）"
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "log", "book6-ch12")
    os.makedirs(out_dir, exist_ok=True)
    dev = "npu"

    # ① 算子拆分 vs 融合：模拟一层 RMSNorm 的「逐算子串」与「一次 torch.rms_norm」对照
    x = torch.randn(8, 1024, device=dev, dtype=torch.bfloat16)
    w = torch.randn(1024, device=dev, dtype=torch.bfloat16)

    def many_small():
        v = x * x                                    # 平方
        v = v.mean(-1, keepdim=True)                 # 均值
        v = x / (v.float().sqrt().to(x.dtype) + 1e-6)
        return v * w

    def one_fused():
        return torch.nn.functional.rms_norm(x, (1024,), w, 1e-6)

    row_small = bench(many_small)
    row_fused = bench(one_fused)
    # ② 纯发射开销：最小算子（1 元素加）× N 次串行——发射成本的裸读数
    t1 = torch.ones(1, device=dev)
    t2 = torch.ones(1, device=dev)
    row_launch = bench(lambda: t1.add_(t2), n=2000, warmup=100)

    # ②' 图执行对照（torch.npu.NPUGraph capture/replay）：同一条 8 语句链（profiler 实证每语句
    #     拆 2 个设备算子=Muls+Adds 各 8，共 16 kernel）eager 逐个发射 vs 录一次后 replay
    #     ——「N 次发射并成 1 次」的本机读数（每算子账按 16 计：21.4/1.9 µs）
    row_graph = {}
    try:
        gs = torch.npu.Stream()
        gp = torch.npu.graphs.graph_pool_handle()

        def chain(x_):
            y = x_
            for _ in range(8):                    # 8 个小算子的直线链（无分支、静态形状）
                y = y * 1.0001 + 0.001
            return y

        gx = torch.randn(8, 1024, device=dev, dtype=torch.bfloat16)
        # 预热（ allocator 稳定后再录——图执行的红线：地址静态）
        for _ in range(3):
            chain(gx)
        torch.npu.synchronize()
        with torch.npu.stream(gs):
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, pool=gp):
                chain(gx)
        row_graph["eager_ms"] = round(bench(lambda: chain(gx), n=500, warmup=50), 4)
        torch.npu.synchronize()
        row_graph["replay_ms"] = round(bench(lambda: graph.replay(), n=500, warmup=50), 4)
        row_graph["ratio"] = round(row_graph["eager_ms"] / max(1e-9, row_graph["replay_ms"]), 1)
        row_graph["note"] = "同一条 8 算子链：eager 逐个发射 vs NPUGraph 录一次整链 replay（capture/replay 的本机正身读数）"
    except Exception as e:
        row_graph["probe"] = f"skip: {type(e).__name__}: {str(e)[:120]}"

    # ③ ACLGraph 实证锚（黑盒）：接口在位性与默认配置读数（机制不展开——红线）
    anchor = {}
    try:
        anchor["NPUGraph_api"] = hasattr(torch.npu, "NPUGraph") or "NpuGraph" in dir(torch.npu.graphs) if hasattr(torch, "npu") else False
    except Exception:
        anchor["NPUGraph_api"] = "probe_failed"
    try:
        from vllm_ascend.platform import NPUPlatform  # noqa: F401
        import inspect
        src = inspect.getsource(NPUPlatform)
        anchor["vllm_ascend_cudagraph_default"] = ("FULL_AND_PIECEWISE" in src)
        anchor["breakable_optin_env"] = "VLLM_USE_BREAKABLE_CUDAGRAPH" in src
    except Exception as e:
        anchor["vllm_ascend_probe"] = f"skip: {type(e).__name__}"

    res = dict(
        split_vs_fused=dict(rmsnorm_split_ms=round(row_small, 4),
                            rmsnorm_fused_ms=round(row_fused, 4),
                            ratio=round(row_small / max(1e-9, row_fused), 1),
                            note="同层数学两种发射法：拆成 4 个小算子串行发射 vs 一个融合算子——拆分税即发射税的层内缩影"),
        bare_launch_us=round(row_launch * 1e3, 2),
        graph_vs_eager=row_graph,
        aclgraph_anchor=anchor,
        note="裸发射≈每 kernel 的固定成本下限；融合与图执行是把 N 次发射并成 1 次的两条路（前者编译期、后者运行期）",
    )
    p = os.path.join(out_dir, f"launch_probe_{a.out_name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"[产物] {p}")


if __name__ == "__main__":
    main()
