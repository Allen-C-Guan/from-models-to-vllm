"""第13章 HCCL collective 正确性与通信微基准；不是模型性能。

运行前选空闲卡：export ASCEND_RT_VISIBLE_DEVICES=0,1; source ./env.sh
torchrun --nnodes=1 --master-addr=127.0.0.1 --master-port=29813 --nproc-per-node=2 code/Book6-推理系统导论/ch13/collectives.py --mode fast --out-name hccl2
"""
import argparse
import json
import os
import time
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
import torch_npu  # registers HCCL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "fast", "full"], default="fast")
    ap.add_argument("--out-name", required=True)
    a = ap.parse_args()
    path = Path("log/book6-ch13") / ("collectives_" + a.out_name + ".json")
    if path.exists():
        raise FileExistsError(path)
    rank = int(os.environ["RANK"])
    n = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    rows = []
    try:
        # Verify values before benchmarking zero-filled buffers.
        x = torch.full((n * 4,), float(rank + 1), device=device)
        dist.all_reduce(x)
        assert torch.all(x == n * (n + 1) / 2).item()
        src = torch.full((4,), float(rank), device=device)
        out = torch.empty(n * 4, device=device)
        dist.all_gather_into_tensor(out, src)
        assert out.cpu().tolist() == [float(r) for r in range(n) for _ in range(4)]
        src = torch.arange(n, device=device, dtype=torch.float32).repeat_interleave(4) + rank * 10
        dist.all_to_all_single(out, src)
        assert out.cpu().tolist() == [float(r * 10 + rank) for r in range(n) for _ in range(4)]
        src = torch.full((n * 4,), float(rank + 1), device=device)
        small = torch.empty(4, device=device)
        dist.reduce_scatter_tensor(small, src)
        assert torch.all(small == n * (n + 1) / 2).item()
        sizes = {"smoke": [1024], "fast": [1024, 65536, 1048576],
                 "full": [256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304]}[a.mode]
        repeats = {"smoke": 5, "fast": 20, "full": 50}[a.mode]
        for count in sizes:
            count = count // n * n
            x = torch.zeros(count, device=device)
            y = torch.empty_like(x)
            small = torch.zeros(count // n, device=device)
            ops = {"all_reduce": lambda: dist.all_reduce(x),
                   "all_gather": lambda: dist.all_gather_into_tensor(y, small),
                   "reduce_scatter": lambda: dist.reduce_scatter_tensor(small, x),
                   "all_to_all": lambda: dist.all_to_all_single(y, x)}
            for name, op in ops.items():
                for _ in range(3):
                    op()
                torch.npu.synchronize()
                dist.barrier()
                start = time.perf_counter()
                for _ in range(repeats):
                    op()
                torch.npu.synchronize()
                elapsed = torch.tensor([(time.perf_counter() - start) / repeats], device=device)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                rows.append({"op": name, "logical_full_bytes": count * 4,
                             "max_rank_mean_us": elapsed.item() * 1e6})
        if rank == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"world_size": n, "dtype": "float32", "mode": a.mode,
                "correctness": "pass", "timing": "synchronized wall clock; maximum rank mean; Python launch included",
                "rows": rows}, indent=2))
            print(path)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
