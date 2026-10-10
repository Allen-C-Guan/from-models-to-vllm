# int8_probe.py —— Book6 ch11 代码件：W8A8 原生算子 wall-clock 探针（npu_quant_matmul 对 bf16 torch.mm）
# 用途：910B3 的 INT8 W8A8 通道吞吐实测——npu_quant_matmul(int8×int8) 对同形状 torch.mm(bf16×bf16)
#   的对照；方法学与 notes/04 P2 对齐（warmup 3 + 计时 10 次取 median、synchronize+perf_counter），
#   bf16 基线须落在 P2 五点曲线同档（4096²→250.6 TFLOPS 量级）方为有效档——否则该档判无效。
# 所属章节：Book6 ch11 §11.6（实物 11.4 数据源）；设计书=plan/Book6-推理系统导论.md ch11
# 运行：cd <workspace> && source env.sh && npu-smi info 选卡 && ASCEND_RT_VISIBLE_DEVICES=<卡> \
#      python code/Book6-推理系统导论/ch11/int8_probe.py [--n 4096] [--out-name base]
# 产物：log/book6-ch11/int8_probe_<out-name>.json
import argparse
import json
import os
import time

import torch


def bench_median(fn, n=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]           # median（P2 同款口径）


def main() -> None:
    ap = argparse.ArgumentParser(description="W8A8 npu_quant_matmul 探针")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--out-name", default="base")
    a = ap.parse_args()
    assert torch.npu.is_available(), "本件为 NPU 性能件（先 npu-smi 选卡）"
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "log", "book6-ch11")
    os.makedirs(out_dir, exist_ok=True)
    dev = "npu"
    m = k = n = a.n

    # bf16 基线（P2 同款：randn 直接上卡、median 计时）
    xb = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
    wb = torch.randn(k, n, dtype=torch.bfloat16, device=dev)
    t_bf16 = bench_median(lambda: torch.mm(xb, wb))
    bf16_tops = 2 * m * k * n / t_bf16 / 1e12

    # int8 W8A8（npu_quant_matmul：x1/x2 int8 + scale 逐张量 fp32）
    g = torch.Generator().manual_seed(20261010)
    x8 = torch.randint(-127, 127, (m, k), generator=g, dtype=torch.int8).to(dev)
    w8 = torch.randint(-127, 127, (k, n), generator=g, dtype=torch.int8).to(dev)
    sc = torch.tensor([0.01], dtype=torch.float32, device=dev)
    y = torch.ops.npu.npu_quant_matmul(x8, w8, sc)      # 输出 int8 (m,n)——反量化在算子外（教学口径）
    t_int8 = bench_median(lambda: torch.ops.npu.npu_quant_matmul(x8, w8, sc))
    int8_tops = 2 * m * k * n / t_int8 / 1e12

    res = dict(
        shape=f"{m}x{k}x{n}",
        bf16=dict(ms=round(t_bf16 * 1e3, 3), tflops=round(bf16_tops, 1),
                  p2_ref="4096^2 档参考 250.6 TFLOPS（notes/04 P2 五点）——本档有效判据：同量级"),
        int8=dict(ms=round(t_int8 * 1e3, 3), tops=round(int8_tops, 1)),
        ratio=round(int8_tops / bf16_tops, 2),
        note="median of 10（warmup 3，逐次 synchronize）；int8 输出为 int8（反量化在算子外——与引擎 fused 反量化口径差一拍的固定开销）",
    )
    p = os.path.join(out_dir, f"int8_probe_{a.out_name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"[产物] {p}")


if __name__ == "__main__":
    main()
