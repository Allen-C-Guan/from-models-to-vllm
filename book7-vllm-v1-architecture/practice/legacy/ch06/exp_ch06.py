# exp_ch06.py —— Book7 ch06 代码件：KV 管理三实验（形状探针 / APC 开关对拍 / 块 16 vs 128 对拍）
# 用途：ch06《KV cache 管理全景》的实测底档——
#   实验③ KV 形状探针：离线 LLM 起 207M（hf207m），读 cache_config 的 block_size/num_gpu_blocks，
#     按 NPU 形状正身 (2, num_blocks, block_size, h_kv, head_size) 联算 KV 显存账（与起服日志互证）；
#   实验① APC 开/关对拍：vllm serve + bench serve（prefix_repetition 数据集：4 前缀×4 请求、
#     prefix 1000（故意不整除 128）+suffix 200），读 /metrics 的 prefix_cache_hits/queries 与 bench TTFT；
#   实验② 块 16 vs 128 对拍：同 workload 两档 --block-size，命中率差=块对齐损耗
#     （理论天花板：16 档每请求命中 ⌊1000/16⌋×16=992、128 档 896；全队列天花板
#      16 档 3×4×992/(16×1200)=62.0%、128 档 3×4×896/(16×1200)=56.0%——预注册）。
# 所属章节：Book7 ch06（§6.6 实验）；设计书=plan/Book7-vLLM源码剖析上.md §ch06 + notes/10 §8
# 运行：cd <workspace> && source env.sh && npu-smi info 选卡 && ASCEND_RT_VISIBLE_DEVICES=<卡> \
#        python code/Book7-vLLM源码剖析上/ch06/exp_ch06.py [--skip-probe] [--skip-serve]
# 产物：log/book7-ch06/{kv_probe.json, exp_<arm>.json, bench_raw/}；判据预注册见各函数 docstring。
# 纪律：vllm 脚本落盘+__main__ 守卫（spawn）；起服进程 terminate 收尾；--out-name 防覆写。
import argparse
import json
import os
import re
import subprocess
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MODEL = os.path.join(WS, "log", "book7-00", "hf207m")
OUT = os.path.join(WS, "log", "book7-ch06")
PORT = 8021

# workload 定版（预注册）：4 前缀 × 每前缀 4 请求 = 16 请求；prefix 1000（不整除 128）+ suffix 200 + 输出 32
WL = ["--dataset-name", "prefix_repetition",
      "--prefix-repetition-prefix-len", "1000",
      "--prefix-repetition-suffix-len", "200",
      "--prefix-repetition-num-prefixes", "4",
      "--num-prompts", "16",
      "--prefix-repetition-output-len", "32",
      "--request-rate", "2.0"]
CEIL = {"block16": round(3 * 4 * 992 / (16 * 1200), 4), "block128": round(3 * 4 * 896 / (16 * 1200), 4)}


def kv_probe() -> dict:
    """实验③：离线 LLM 读 KV 事实。判据：block_size==128（NPU 默认）；
    tokens 容量=num_gpu_blocks×block_size；KV 字节账=容量×2(K/V)×h_kv(8)×head(64)×2B×L(12)。"""
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.4,
              enforce_eager=True, disable_log_stats=False)
    cc = llm.llm_engine.vllm_config.cache_config
    bs, nb = cc.block_size, getattr(cc, "num_gpu_blocks", None)
    h_kv, head, L = 8, 64, 12                                   # 207M 实参（Book6 ch05 同账）
    tok_cap = nb * bs
    kv_bytes = tok_cap * 2 * h_kv * head * 2 * L                # K/V 两份 × bf16
    res = dict(block_size=bs, num_gpu_blocks=nb, tokens_capacity=tok_cap,
               kv_shape_per_layer=f"(2, {nb}, {bs}, {h_kv}, {head})",
               kv_total_gib=round(kv_bytes / 2**30, 2),
               per_token_bytes_all_layers=2 * h_kv * head * 2 * L,
               runtime="vllm 0.27.1 + vllm-ascend 0.27.1rc1（Atlas 910B3）",
               note="形状正身=vllm_ascend/attention/attention_v1.py#L109-115 get_kv_cache_shape")
    llm.generate(["One, two, three,"], SamplingParams(temperature=0.0, max_tokens=4))
    return res


def parse_metrics(text: str) -> dict:
    """从 /metrics 文本抽 prefix cache 计数（exposition 名带/不带 _total 两态兼容）。"""
    out = {}
    for line in text.splitlines():
        m = re.match(r'^vllm:prefix_cache_(queries|hits)(_total)?\s+([0-9.e+]+)', line)
        if m:
            out[f"{m.group(1)}"] = float(m.group(3))
    return out


def run_arm(arm: str, extra_serve: list) -> dict:
    """起服→bench→metrics→收尾。判据预注册：APC-off 臂命中率≈0；block128 臂≈CEIL['block128']±5pp；
    block16 臂≈CEIL['block16']±5pp 且 >block128 臂（对齐损耗方向）。"""
    log_p = os.path.join(OUT, f"serve_{arm}.log")
    serve_cmd = ["vllm", "serve", MODEL, "--port", str(PORT), "--max-model-len", "2048",
                 "--gpu-memory-utilization", "0.4", "--enforce-eager", *extra_serve]
    with open(log_p, "w") as lf:
        proc = subprocess.Popen(serve_cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=WS)
    try:
        t0 = time.time()
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                if proc.poll() is not None:
                    return dict(arm=arm, error=f"serve died (see {log_p})")
                if time.time() - t0 > 420:
                    proc.terminate()
                    return dict(arm=arm, error="health timeout 420s")
                time.sleep(3)
        load_s = round(time.time() - t0, 1)
        time.sleep(2)
        m0 = parse_metrics(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=10).read().decode())
        bench_raw = os.path.join(OUT, "bench_raw")
        os.makedirs(bench_raw, exist_ok=True)
        bench_cmd = ["vllm", "bench", "serve", "--port", str(PORT), *WL, "--save-result"]
        t1 = time.time()
        bp = subprocess.run(bench_cmd, capture_output=True, text=True, cwd=bench_raw, timeout=600)
        bench_s = round(time.time() - t1, 1)
        with open(os.path.join(bench_raw, f"bench_{arm}.out"), "w") as f:
            f.write(bp.stdout[-8000:] + "\n===STDERR===\n" + bp.stderr[-2000:])
        m1 = parse_metrics(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=10).read().decode())
        hits = m1.get("hits", 0.0) - m0.get("hits", 0.0)
        queries = m1.get("queries", 0.0) - m0.get("queries", 0.0)
        ttft = None
        for ln in bp.stdout.splitlines():
            if "Mean TTFT (ms)" in ln:
                ttft = ln.strip()
        # bench 落盘的 result json（vllm bench serve --save-result 在 cwd 写 benchmark_serve_*.json）
        res = dict(arm=arm, serve_flags=" ".join(extra_serve) or "(default)", load_s=load_s, bench_s=bench_s,
                   hits=hits, queries=queries,
                   hit_rate=round(hits / queries, 4) if queries else None,
                   ceiling_note=CEIL, ttft_line=ttft, bench_exit=bp.returncode)
        return res
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()


def main() -> None:
    ap = argparse.ArgumentParser(description="ch06 KV 管理三实验")
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--skip-serve", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    if not a.skip_probe:
        summary["kv_probe"] = kv_probe()
        with open(os.path.join(OUT, "kv_probe.json"), "w", encoding="utf-8") as f:
            json.dump(summary["kv_probe"], f, ensure_ascii=False, indent=1)
        print("[probe]", json.dumps(summary["kv_probe"], ensure_ascii=False))
    if not a.skip_serve:
        arms = [
            ("apc_off_b128", ["--no-enable-prefix-caching"]),   # 实验① 负臂
            ("apc_on_b128", []),                                  # 实验① 正臂 = 实验② 的 128 档
            ("apc_on_b16", ["--block-size", "16"]),              # 实验② 的 16 档
        ]
        for arm, extra in arms:
            r = run_arm(arm, extra)
            summary[arm] = r
            print(f"[{arm}]", json.dumps(r, ensure_ascii=False))
    with open(os.path.join(OUT, "exp_ch06_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"[产物] {os.path.join(OUT, 'exp_ch06_summary.json')}")


if __name__ == "__main__":
    main()
