"""第13章：OpenAI completion 流式端到端压测，支持多个独立副本。

运行：source ./env.sh; python .../bench_serving.py --urls http://127.0.0.1:8130 --mode fast --out-name tp1
客户端 TTFT 包含请求排队、HTTP；TPOT=(最后token到达-首token到达)/(输出token数-1)。
使用服务端 return_token_ids，避免把文本chunk数量当成token数量。
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
import requests


def request(url, tokens, output):
    session = requests.Session()
    session.trust_env = False
    start = time.perf_counter()
    stamps, ids = [], []
    body = {"model": "deepseek-v2-lite", "prompt": tokens, "max_tokens": output,
            "temperature": 0, "ignore_eos": True, "stream": True,
            "return_token_ids": True, "seed": 20261030}
    with session.post(url + "/v1/completions", json=body, stream=True, timeout=(10, 600)) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            data = json.loads(line[6:])
            if "error" in data:
                raise RuntimeError(data["error"])
            for choice in data.get("choices", []):
                new = choice.get("token_ids") or []
                if new:
                    ids.extend(new)
                    stamps.extend([time.perf_counter()] * len(new))
    session.close()
    if len(ids) != output:
        raise RuntimeError(f"expected {output} token ids, got {len(ids)}")
    return {"input_tokens": len(tokens), "output_tokens": len(ids),
            "ttft_ms": (stamps[0] - start) * 1000,
            "tpot_ms": (stamps[-1] - stamps[0]) * 1000 / max(1, len(ids) - 1),
            "e2e_ms": (stamps[-1] - start) * 1000, "token_ids": ids,
            "token_arrival_ms": [(t - start) * 1000 for t in stamps]}


def percentile(values, fraction):
    values = sorted(values)
    pos = (len(values) - 1) * fraction
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", nargs="+", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--mode", choices=["smoke", "fast", "full"], default="fast")
    ap.add_argument("--out-name", required=True)
    a = ap.parse_args()
    path = Path("log/book6-ch13") / ("serving_" + a.out_name + ".json")
    if path.exists():
        raise FileExistsError(path)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model_path)
    base = tokenizer.encode("Explain why distributed inference communicates partial results between devices. ",
                            add_special_tokens=False)
    output = {"smoke": 8, "fast": 32, "full": 128}[a.mode]
    # Fixed closed-loop concurrency, not an open-loop arrival process or SLO capacity test.
    scenarios = [(128, 1, 2)] if a.mode == "smoke" else [
        (128, 1, 5 if a.mode == "fast" else 30),
        (4096 if a.mode == "fast" else 16384, 1, 5 if a.mode == "fast" else 30),
        (128, 8, 16 if a.mode == "fast" else 128)]
    result = {"mode": a.mode, "urls": a.urls, "workload": "closed-loop fixed concurrency; synthetic repeated token prefix; prefix cache disabled",
              "rows": []}
    # Fixed teacher-forced prefix probabilities are a numerical comparison, not a language-quality test.
    probe_tokens = (base * 8)[:64]
    probe_session = requests.Session()
    probe_session.trust_env = False
    probe = probe_session.post(a.urls[0] + "/v1/completions", json={
        "model": "deepseek-v2-lite", "prompt": probe_tokens, "max_tokens": 8,
        "temperature": 0, "ignore_eos": True, "return_token_ids": True,
        "logprobs": 5, "echo": True, "seed": 20261030}, timeout=600)
    probe.raise_for_status()
    result["correctness_probe"] = {"prompt_ids": probe_tokens, "response": probe.json()}
    if a.mode != "smoke":
        # Check long-prefill numerics separately: a short-prefix match cannot
        # certify the chunked-prefill / pipeline path used by the benchmark.
        long_tokens = (base * (4096 // len(base) + 1))[:4096]
        long_probe = probe_session.post(a.urls[0] + "/v1/completions", json={
            "model": "deepseek-v2-lite", "prompt": long_tokens, "max_tokens": 8,
            "temperature": 0, "ignore_eos": True, "return_token_ids": True,
            "logprobs": 0, "echo": True, "seed": 20261030}, timeout=600)
        long_probe.raise_for_status()
        result["long_correctness_probe"] = {"prompt_ids": long_tokens, "response": long_probe.json()}
    probe_session.close()
    for length, concurrency, count in scenarios:
        tokens = (base * (length // len(base) + 1))[:length]
        # Warm each replica at this length. Warmup excluded from results.
        for url in a.urls:
            request(url, tokens, min(8, output))
        begin = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            records = list(pool.map(lambda i: request(a.urls[i % len(a.urls)], tokens, output), range(count)))
        wall = time.perf_counter() - begin
        row = {"input_tokens": length, "concurrency": concurrency, "requests": count,
               "wall_s": wall, "output_tokens_per_s": sum(r["output_tokens"] for r in records) / wall,
               "records": records}
        for metric in ["ttft_ms", "tpot_ms", "e2e_ms"]:
            row[metric] = {"mean": statistics.mean(r[metric] for r in records),
                           "p50": percentile([r[metric] for r in records], .5),
                           "p95": percentile([r[metric] for r in records], .95)}
        result["rows"].append(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in row.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
