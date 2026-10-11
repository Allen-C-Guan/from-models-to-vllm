# exp_aclgraph.py —— Book7 ch08 代码件：ACLGraph（默认档）vs eager 对拍
# 用途：B6-6 实测兑现——默认 FULL_AND_PIECEWISE+图捕获对 --enforce-eager 的同 workload 对拍。
#   判据预注册：①图档稳态 decode TPOT 低于 eager 档（NPUGraph 11.4× 是 kernel 链级、引擎级预期一个数量级内）；
#   ②图档启动时间高于 eager 档（捕获成本——与 ch07 实验③的 84 s 默认档读数可互参）。
#   口径纪律：同模型同卡同 workload；TPOT 用 vllm bench latency 的逐 token 均值（同工具同参数）。
# 所属章节：Book7 ch08（§8.6 实验）；设计书=notes/16。
# 运行：cd <workspace> && source env.sh && npu-smi info 选卡 && ASCEND_RT_VISIBLE_DEVICES=<卡> \
#        python code/Book7-vLLM源码剖析上/ch08/exp_aclgraph.py
# 产物：log/book7-ch08/aclgraph_eager_<arm>.json + summary
# 纪律：一臂一进程；Qwen3-0.6B fast 档。
import argparse, json, os, re, subprocess, time

WS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
HUB = os.path.join(WS, "log", "huggingface", "hub", "models--Qwen--Qwen3-0.6B", "snapshots")
MODEL = os.path.join(HUB, next(s for s in os.listdir(HUB)) if os.listdir(HUB) else "Qwen/Qwen3-0.6B")
OUT = os.path.join(WS, "log", "book7-ch08")


def bench_arm(tag, extra):
    raw = os.path.join(OUT, f"bench_{tag}")
    os.makedirs(raw, exist_ok=True)
    cmd = ["vllm", "bench", "latency", "--model", MODEL, "--max-model-len", "1024",
           "--gpu-memory-utilization", "0.4", "--input-len", "128", "--output-len", "64",
           "--batch-size", "4", "--num-iters-warmup", "2", "--num-iters", "5",
           "--output-json", os.path.join(raw, "r.json"), *extra]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=raw, timeout=900)
    wall = round(time.perf_counter() - t0, 1)
    j = {}
    p = os.path.join(raw, "r.json")
    if os.path.exists(p):
        j = json.load(open(p))
    return dict(tag=tag, extra=" ".join(extra) or "(default=graph)", wall_s=wall, exit=r.returncode,
                avg_latency=j.get("avg_latency"), p50=j.get("latencies", [None])[0] if j.get("latencies") else None,
                note="avg_latency 为整批 64-token 生成均值 (s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="graph,eager")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res = {}
    if "graph" in a.arms:
        res["graph"] = bench_arm("graph", [])
        print("[graph]", json.dumps(res["graph"], ensure_ascii=False))
    if "eager" in a.arms:
        res["eager"] = bench_arm("eager", ["--enforce-eager"])
        print("[eager]", json.dumps(res["eager"], ensure_ascii=False))
    if "graph" in res and "eager" in res:
        g, e = res["graph"].get("avg_latency"), res["eager"].get("avg_latency")
        if g and e:
            res["verdict"] = dict(graph_s=g, eager_s=e, ratio=round(e / g, 2),
                                  graph_faster=bool(g < e))
    json.dump(res, open(os.path.join(OUT, "aclgraph_summary.json"), "w"), ensure_ascii=False, indent=1)
    print("[产物]", os.path.join(OUT, "aclgraph_summary.json"))
