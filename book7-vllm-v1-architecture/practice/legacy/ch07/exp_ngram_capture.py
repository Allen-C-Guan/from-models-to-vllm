# exp_ngram_capture.py —— Book7 ch07 代码件：实验② ngram 冒烟 + 实验③ capture sizes 观测
# 用途：实验②——ngram 投机开关冒烟（K1 两臂均 MRV1：NPU 恒 MRV1×ngram MRV1-only 叠加验证；
#   K2 吞吐/延迟方向双分支如实）。实验③——capture sizes 三观察（S1 默认档尺寸表回显；S2 ngram k=3
#   后上限按 max_num_seqs×4 与 512 的 min 移动；D1 910B3 不应出现剪枝告警——能力位负判据）。
# 所属章节：Book7 ch07（§7.7/实验③）；设计书=notes/15 §8。
# 运行：cd <workspace> && source env.sh && npu-smi info 选卡 && ASCEND_RT_VISIBLE_DEVICES=<卡> \
#        python code/Book7-vLLM源码剖析上/ch07/exp_ngram_capture.py
# 产物：log/book7-ch07/ngram_capture_summary.json
import json, os, re, subprocess, time, urllib.request

WS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
MODEL = os.path.join(WS, "log", "huggingface", "hub", "models--Qwen--Qwen3-0.6B", "snapshots")
SNAP = os.path.join(next(iter(os.listdir(MODEL))) if os.listdir(MODEL) else "", "") if os.listdir(MODEL) else ""
MODEL_PATH = os.path.join(MODEL, SNAP) if SNAP else "Qwen/Qwen3-0.6B"
OUT = os.path.join(WS, "log", "book7-ch07")
PORT = 8024


def run_arm(tag, extra):
    log_p = os.path.join(OUT, f"serve_{tag}.log")
    with open(log_p, "w") as lf:
        proc = subprocess.Popen(["vllm", "serve", MODEL_PATH, "--port", str(PORT), "--max-model-len", "1024",
                                 "--gpu-memory-utilization", "0.4", *extra],
                                stdout=lf, stderr=subprocess.STDOUT, cwd=WS)
    try:
        t0 = time.perf_counter()
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                if proc.poll() is not None:
                    return dict(tag=tag, error=f"serve died, see {log_p}")
                if time.perf_counter() - t0 > 480:
                    return dict(tag=tag, error="health timeout")
                time.sleep(3)
        load_s = round(time.perf_counter() - t0, 1)
        log = open(log_p, encoding="utf-8", errors="ignore").read()
        cap = re.search(r"'cudagraph_capture_sizes': \[([^\]]*)\]", log)
        return dict(tag=tag, load_s=load_s,
                    capture_sizes=cap.group(1).strip() if cap else None,
                    K1_runner_v1=bool(re.findall(r"\[model_runner_v1\.py:", log)),
                    runner_v2_warn="npu model runner v2" in log,
                    D1_prune_warn="Adjusted ACL graph batch sizes" in log,
                    capturing_progress="Capturing CUDA graphs" in log or "Capturing aclgraph" in log,
                    spec_line=[ln.strip()[:90] for ln in log.splitlines() if "ngram" in ln.lower()][:2])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    summary["S1_default"] = run_arm("default", [])
    print("[S1]", json.dumps(summary["S1_default"], ensure_ascii=False)[:300])
    summary["S2_ngram"] = run_arm("ngram", ["--speculative-config",
                                            '{"method":"ngram","num_speculative_tokens":3}'])
    print("[S2]", json.dumps(summary["S2_ngram"], ensure_ascii=False)[:300])
    json.dump(summary, open(os.path.join(OUT, "ngram_capture_summary.json"), "w"),
              ensure_ascii=False, indent=1)
    print("[产物]", os.path.join(OUT, "ngram_capture_summary.json"))
