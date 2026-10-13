# exp2_overhead.py —— ch3 实验 2：稳态约束开销（serve 口径流式客户端：guided vs plain 两臂差分）
# 所属章节：Book8 ch3；设计书=notes/12 §四 exp2（J1 guided ITL>plain 个位数百分比；J2 缓存热后 TTFT 差<exp1 首批）
# 设计：单 serve 实例同接两臂（response_format 是请求级）——先 guided 预热缓存，再 plain 批、guided 批
#       各 32 请求×64 token，流式记 TTFT/ITL（每 token 口径——防输出长度污染吞吐）。
# 运行：先起 serve（12 号模板改：Qwen3-0.6B/--port 18023），然后
#        python code/Book8-vLLM源码剖析下/ch03/exp2_overhead.py --host 127.0.0.1 --port 18023
# 产物：log/book8-ch03/exp2_overhead.json
import argparse
import json
import os
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(WS, "log/book8-ch03")
sys_path = HERE
import sys
sys.path.insert(0, sys_path)
from schemas_ch03 import VARIANTS, PROMPTS, CORRECTNESS_SET


def stream_one(host, port, prompt, schema, max_tokens=64):
    """一次流式请求：返回 ttft_ms 与 itl_ms 列表（每 token 口径）。schema=None → plain 臂。"""
    body = {"model": "Qwen/Qwen3-0.6B",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    if schema is not None:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "exp2", "schema": schema}}
    req = urllib.request.Request(f"http://{host}:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    itls = []
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            delta = d.get("choices", [{}])[0].get("delta", {}).get("content")
            if delta is None:
                continue
            now = (time.perf_counter() - t0) * 1000
            if ttft is None:
                ttft = now
            else:
                itls.append(now)
    if itls:
        # itls 存的是各 token 到达时刻（ms 自请求起）——差分得每 token 间隔
        gaps = [itls[0] - ttft] + [itls[i] - itls[i - 1] for i in range(1, len(itls))]
    else:
        gaps = []
    return ttft, gaps


def run_arm(host, port, guided, n=32):
    ttfts, gaps_all = [], []
    for i in range(n):
        name = CORRECTNESS_SET[i % len(CORRECTNESS_SET)]
        schema = VARIANTS[name] if guided else None
        ttft, gaps = stream_one(host, port, PROMPTS[name], schema)
        ttfts.append(ttft)
        gaps_all.extend(gaps)
    gaps_sorted = sorted(gaps_all)
    mid = len(gaps_sorted) // 2
    return dict(n=n, ttft_mean_ms=round(sum(ttfts) / len(ttfts), 1),
                ttft_p50_ms=round(gaps_sorted and sorted(ttfts)[len(ttfts) // 2] or 0, 1),
                itl_mean_ms=round(sum(gaps_all) / max(1, len(gaps_all)), 2),
                itl_p50_ms=round(gaps_sorted[mid], 2) if gaps_sorted else None,
                n_itl_samples=len(gaps_all))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18023)
    a = ap.parse_args()
    print("[warmup] guided 预热（缓存 miss→hit）……")
    run_arm(a.host, a.port, guided=True, n=4)
    print("[arm plain] ……")
    plain = run_arm(a.host, a.port, guided=False)
    print("[arm guided] ……")
    guided = run_arm(a.host, a.port, guided=True)
    res = dict(meta=dict(date="2026-10-12", model="Qwen/Qwen3-0.6B", port=a.port,
                         runtime="vllm 0.27.1+vllm-ascend（Atlas 910B3）"),
               plain=plain, guided=guided,
               itl_ratio=round(guided["itl_mean_ms"] / plain["itl_mean_ms"], 3) if plain["itl_mean_ms"] else None,
               ttft_diff_ms=round(guided["ttft_mean_ms"] - plain["ttft_mean_ms"], 1))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "exp2_overhead.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))
    print("[产物]", os.path.join(OUT, "exp2_overhead.json"))


if __name__ == "__main__":
    main()
