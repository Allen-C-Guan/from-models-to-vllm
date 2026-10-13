# exp_ch01_hit.py —— ch1 实验 B：hybrid 变身命中率实测（Book7 ch06 例 6.4 workload 续账）
# 所属章节：Book8 ch1（§1.6）；设计书=notes/13 §四 实验 B（判据三分支预注册：主 32.0%/分支②56.0%/分支③偏离查粒度）
# workload：4 前缀×4 请求、prefix 1000 tok（不整除 128）、suffix 200 tok/请求、输出 32、请求率 2/s；
#           prompt 以 token id 数组下发（精确 token 数控制）。
# 两臂各一进程：python .../exp_ch01_hit.py --arm hybrid --sliding-window 512 [--out-name hit_hybrid_w512]
#              python .../exp_ch01_hit.py --arm baseline    [--out-name hit_fa_base]
# 产物：log/book8-ch01/exp_ch01_hit_<out-name>.json
import argparse
import json
import os
import random
import subprocess
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(WS, "log/book8-ch01")
MODEL = os.path.join(WS, "log/book7-00/hf207m")
PORT = 18041
SEED = 20261012
N_PREFIX, REQ_PER_PREFIX, PREFIX_LEN, SUFFIX_LEN, OUT_LEN, RATE = 4, 4, 1000, 200, 32, 2.0


def build_workload():
    rng = random.Random(SEED)
    prefixes = [[rng.randint(3, 31999) for _ in range(PREFIX_LEN)] for _ in range(N_PREFIX)]
    reqs = []
    for p in prefixes:
        for j in range(REQ_PER_PREFIX):
            suffix = [rng.randint(3, 31999) for _ in range(SUFFIX_LEN)]
            reqs.append(p + suffix)
    return reqs


def serve_cmd(arm, sliding_window):
    base = ["vllm", "serve", MODEL, "--dtype", "bfloat16", "--max-model-len", "2048",
            "--gpu-memory-utilization", "0.4", "--enforce-eager", "--seed", str(SEED),
            "--host", "127.0.0.1", "--port", str(PORT)]
    if arm == "hybrid":
        overrides = json.dumps({"layer_types": ["full_attention", "sliding_attention"] * 6,
                                "sliding_window": sliding_window})
        base += ["--hf-overrides", overrides]
    return base


def fire(reqs):
    t0 = time.time()
    for i, ids in enumerate(reqs):
        body = json.dumps({"model": MODEL, "prompt": ids, "max_tokens": OUT_LEN,
                           "temperature": 0, "ignore_eos": True}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=180).read()
        wait = (i + 1) / RATE - (time.time() - t0)
        if wait > 0:
            time.sleep(wait)


def metrics():
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=30) as r:
        txt = r.read().decode()
    vals = {}
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].split("{")[0].startswith("vllm:prefix_cache"):
            vals[parts[0].split("{")[0].replace("vllm:", "").replace("_total", "")] = float(parts[1])
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["hybrid", "baseline"], required=True)
    ap.add_argument("--sliding-window", type=int, default=512)
    ap.add_argument("--out-name", default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out_name = a.out_name or f"hit_{'hybrid_w' + str(a.sliding_window) if a.arm == 'hybrid' else 'fa_base'}"

    log = open(os.path.join(OUT, f"{out_name}_serve.log"), "w")
    proc = subprocess.Popen(serve_cmd(a.arm, a.sliding_window), stdout=log, stderr=subprocess.STDOUT)
    with open(os.path.join(OUT, "serve.pid"), "w") as f:
        f.write(str(proc.pid))
    code = "000"
    for _ in range(75):
        time.sleep(2)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as r:
                code = str(r.status)
            if code == "200":
                break
        except Exception:
            pass
    if code != "200":
        print("serve 未就绪"); proc.terminate()
        return
    m0 = metrics()
    reqs = build_workload()
    fire(reqs)
    time.sleep(3)
    m1 = metrics()
    hits = m1.get("prefix_cache_hits", 0) - m0.get("prefix_cache_hits", 0)
    queries = m1.get("prefix_cache_queries", 0) - m0.get("prefix_cache_queries", 0)
    rate_pct = round(hits / queries * 100, 1) if queries else None
    proc.terminate()
    for _ in range(20):
        time.sleep(2)
        if proc.poll() is not None:
            break
    branch = ("main" if rate_pct and abs(rate_pct - 32.0) <= 3 else
              "b2" if rate_pct and abs(rate_pct - 56.0) <= 3 else "b3")
    res = dict(probe="exp_ch01_hit", arm=a.arm,
               hf_overrides=(serve_cmd(a.arm, a.sliding_window)[-1] if a.arm == "hybrid" else None),
               max_model_len=2048, apc=True, enforce_eager=True,
               workload=dict(prefixes=N_PREFIX, reqs_per_prefix=REQ_PER_PREFIX,
                             prefix_len=PREFIX_LEN, suffix_len=SUFFIX_LEN,
                             output_len=OUT_LEN, rate=RATE),
               hits=hits, queries=queries, hit_rate_pct=rate_pct,
               ceiling_main=32.0, ceiling_branch2=56.0, branch_taken=branch,
               runtime="vllm 0.27.1 + vllm-ascend 0.27.1rc1（Atlas 910B3）", date="2026-10-12")
    with open(os.path.join(OUT, f"exp_ch01_hit_{out_name}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "hf_overrides"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
