# 00_download_assets.py —— Book2 可行性探针的资产预下载（gpt2 / bert-base / bert-tiny / GLUE 数据）
# 下载一律走 env.sh 重定向后的 HF_HOME（log/huggingface），产物不入库；本脚本耗时与体量会记录到
# log/book2-feasibility/download_report.json，供 ch10/ch11/ch3 的预算表引用。
# 用法：python 00_download_assets.py（可重复运行，命中缓存则跳过）

import json
import os
import time
import traceback

OUT_DIR = os.path.join("log", "book2-feasibility")


def _cache_size_mb() -> float:
    """统计 HF 缓存目录当前体量（MB），用于估算新增下载量。"""
    hf_home = os.environ.get("HF_HOME", "")
    total = 0
    for root, _dirs, files in os.walk(hf_home):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1024 / 1024


def timed(fn, *args, **kwargs):
    """包装：记录函数 wall-clock（秒）与是否成功，异常不中断后续下载。"""
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        return {"ok": True, "sec": round(time.perf_counter() - t0, 1), "result": result}
    except Exception:
        return {"ok": False, "sec": round(time.perf_counter() - t0, 1),
                "error": traceback.format_exc()[-2000:]}


def download_gpt2():
    from huggingface_hub import snapshot_download
    # gpt2 仓库共 26 个文件 ≈1.6GB（含 TF/Flax/Rust 权重的三套重复格式）——只取 PT/分词器所需
    path = snapshot_download(
        "openai-community/gpt2",
        allow_patterns=["config.json", "model.safetensors", "vocab.json", "merges.txt",
                        "tokenizer.json", "tokenizer_config.json"])
    return {"path": path, "files": sorted(os.listdir(path))}


def download_bert_base():
    from huggingface_hub import snapshot_download
    path = snapshot_download("google-bert/bert-base-uncased",
                             allow_patterns=["config.json", "vocab.txt", "tokenizer*",
                                             "model.safetensors"])
    return {"path": path, "files": sorted(os.listdir(path))}


def download_bert_tiny():
    from huggingface_hub import snapshot_download
    path = snapshot_download("google/bert_uncased_L-2_H-128_A-2",
                             allow_patterns=["config.json", "vocab.txt", "tokenizer*",
                                             "model.safetensors"])
    return {"path": path, "files": sorted(os.listdir(path))}


def download_glue():
    from datasets import load_dataset
    out = {}
    for subset in ["sst2", "mrpc"]:
        ds = load_dataset("nyu-mll/glue", subset)
        builder_sizes = {k: len(v) for k, v in ds.items()}
        lic = getattr(ds["train"], "info", None) and getattr(ds["train"].info, "license", None)
        out[subset] = {"splits": builder_sizes, "license": str(lic)}
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    size_before = _cache_size_mb()
    report = {"cache_mb_before": round(size_before, 1), "hf_home": os.environ.get("HF_HOME", "")}

    # 未鉴权请求可能被限速甚至挂起：每项最多重试 3 次（挂起由外层 timeout 兜底杀掉）
    jobs = {"gpt2": download_gpt2, "bert_base": download_bert_base,
            "bert_tiny": download_bert_tiny, "glue": download_glue}
    for name, fn in jobs.items():
        for attempt in range(1, 4):
            r = timed(fn)
            r["attempt"] = attempt
            report[name] = r
            if r["ok"]:
                break
            print(f"[retry] {name} attempt {attempt} failed", flush=True)

    report["cache_mb_after"] = round(_cache_size_mb(), 1)
    report["cache_mb_delta"] = round(report["cache_mb_after"] - report["cache_mb_before"], 1)

    out_path = os.path.join(OUT_DIR, "download_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "glue"}, ensure_ascii=False, indent=2))
    print("glue:", json.dumps(report["glue"], ensure_ascii=False)[:500])
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
