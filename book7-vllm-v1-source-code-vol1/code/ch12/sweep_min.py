# sweep_min.py —— Book7 ch12 参数扫描最小骨架：vllm bench latency + --output-json 逐点落盘
# 双轴扫描（调度器两级预算——ch05 §5.3 结构账的实验延伸）：
#   axis=mnbt：max-num-batched-tokens（token 预算）三档——小预算下批需分步装完、大预算单步装完；
#   axis=mns ：max-num-seqs（序列预算）三档——批内并发上限对单批延迟的作用。
# 运行：cd <workspace> && source env.sh && npu-smi info 选卡后
#   ASCEND_RT_VISIBLE_DEVICES=<卡> python code/Book7-vLLM源码剖析上/ch12/sweep_min.py [mnbt|mns]
# 产物：log/book7-ch12/lat_<轴><值>.json 逐点 + stdout 汇总表
# 依据：vllm bench latency 参数面（vllm/benchmarks/latency.py#L30-L77，EngineArgs 尾部追加）；
#       纪律——逐点 --output-json 落盘可续跑可复核（Book6 对照组对齐底档教训延续）。
import json
import os
import subprocess
import sys

GRIDS = {
    "mnbt": [2048, 4096, 8192],   # token 预算扫描点（fast 档；full 档扩到 16384/32768）
    "mns": [8, 16, 32],           # 序列预算扫描点（batch-size=16 夹在中间：8 收窄/16 基线/32 放宽）
}
OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                 "log", "book7-ch12"))


def main() -> None:
    axis = sys.argv[1] if len(sys.argv) > 1 else "mnbt"
    assert axis in GRIDS, f"axis must be one of {list(GRIDS)}"
    flag = "--max-num-batched-tokens" if axis == "mnbt" else "--max-num-seqs"
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for v in GRIDS[axis]:
        out = os.path.join(OUT_DIR, f"lat_{axis}{v}.json")
        subprocess.run(
            ["vllm", "bench", "latency",
             "--model", "Qwen/Qwen3-0.6B",
             "--max-model-len", "1024", "--gpu-memory-utilization", "0.4",
             "--enforce-eager",
             "--input-len", "512", "--output-len", "64", "--batch-size", "16",
             "--num-iters-warmup", "2", "--num-iters", "3",
             flag, str(v),
             "--output-json", out],
            check=True)
        r = json.load(open(out, encoding="utf-8"))
        rows.append({axis: v, "avg_s": round(r["avg_latency"], 4),
                     "p50_s": round(r["percentiles"]["50"], 4),
                     "p99_s": round(r["percentiles"]["99"], 4)})
        print(f"[sweep] {axis}={v}: {rows[-1]}", flush=True)
    print(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"[产物] {OUT_DIR}/lat_{axis}*.json")


if __name__ == "__main__":
    main()
