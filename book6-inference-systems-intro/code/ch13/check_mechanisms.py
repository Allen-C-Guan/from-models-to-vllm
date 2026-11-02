"""第13章：TP、EP、SP与分片attention的独立数值验证（CPU FP32）。

运行：source ./env.sh; python code/Book6-推理系统导论/ch13/check_mechanisms.py --out-name fast
使用小尺寸同构计算验证恒等式，不模拟设备性能，不声称运行了完整模型。
"""
import argparse
import json
from pathlib import Path
import torch


def verify():
    torch.manual_seed(20261030)
    torch.set_num_threads(1)
    results = {}

    def check(name, actual, expected):
        error = float((actual - expected).abs().max())
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        results[name] = {"max_abs_error": error, "shape": list(actual.shape)}

    x = torch.randn(4, 8)  # (tokens, hidden)
    up, gate, down = torch.randn(8, 12), torch.randn(8, 12), torch.randn(12, 8)
    ref = (torch.nn.functional.silu(x @ gate) * (x @ up)) @ down
    partials = [
        (torch.nn.functional.silu(x @ g) * (x @ u)) @ d
        for g, u, d in zip(gate.chunk(2, 1), up.chunk(2, 1), down.chunk(2, 0))
    ]  # two (4,8) partial sums, not two halves of the answer
    check("tp_swiglu", sum(partials), ref)
    # ReduceScatter along tokens; token-local RMSNorm; AllGather restores layout.
    def rms(t):
        return t * torch.rsqrt(t.square().mean(-1, keepdim=True) + 1e-6)
    shards = sum(partials).chunk(2, 0)
    check("sp_token_local_rms", torch.cat([rms(t) for t in shards]), rms(ref))

    experts = torch.randn(4, 8, 8)
    routes = torch.tensor([[0, 2], [1, 2], [2, 3], [0, 3]])
    weights = torch.tensor([[.7, .3], [.2, .8], [.6, .4], [.5, .5]])
    reference = torch.stack([
        sum(weights[t, j] * (x[t] @ experts[routes[t, j]]) for j in range(2))
        for t in range(4)
    ])
    combined = torch.zeros_like(x)
    dispatch = []
    for expert in range(4):
        token, choice = torch.where(routes == expert)
        local = x[token] @ experts[expert]  # (assignments,8) -> (assignments,8)
        combined.index_add_(0, token, local * weights[token, choice, None])
        dispatch.append({"expert": expert, "owner_rank": expert // 2,
                         "token_ids": token.tolist()})
    check("ep_dispatch_combine", combined, reference)
    results["ep_dispatch"] = dispatch

    q, k, v = torch.randn(2, 3, 1, 8), torch.randn(2, 3, 7, 8), torch.randn(2, 3, 7, 5)
    scores = q @ k.transpose(-1, -2) / 8**.5  # (2,3,1,7)
    full = scores.softmax(-1) @ v  # (2,3,1,5)
    # Unequal shards make accidentally averaging local softmax outputs fail.
    local_o, local_lse = [], []
    for s, values in zip(scores.split([2, 5], -1), v.split([2, 5], -2)):
        local_o.append(s.softmax(-1) @ values)
        local_lse.append(s.logsumexp(-1, keepdim=True))
    lse = torch.stack(local_lse, 0)
    merged = (torch.stack(local_o, 0) * lse.softmax(0)).sum(0)
    check("dcp_lse_merge", merged, full)
    wrong = torch.stack(local_o).mean(0)
    results["wrong_unweighted_mean_error"] = float((wrong - full).abs().max())
    assert results["wrong_unweighted_mean_error"] > .01
    # Empty shard: numerator=0, denominator=0. Never evaluate -inf - -inf.
    lse = torch.cat([lse, torch.full_like(lse[:1], -torch.inf)], 0)
    local_o.append(torch.zeros_like(local_o[0]))
    finite = torch.isfinite(lse)
    mass = torch.where(finite, torch.exp(lse - lse.max(0).values), 0.)
    check("merge_with_empty_shard", (torch.stack(local_o) * mass).sum(0) / mass.sum(0), full)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", required=True)
    a = ap.parse_args()
    p = Path("log/book6-ch13") / ("mechanisms_" + a.out_name + ".json")
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(p)
    data = verify()
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(data, indent=2))
