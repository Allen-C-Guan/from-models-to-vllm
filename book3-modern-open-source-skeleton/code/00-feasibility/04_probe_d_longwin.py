# 04_probe_d_longwin.py —— 探针 D：长上下文成本曲线 + 长文语料盘点 + Gutenberg 下载管道冒烟
# 用途：(1) 同一小档 RoPE 模型（d=256/L=4/h=4）在 seq=1024/2048/4096/8192 四档实测 sec/step 与
#           MPS 内存峰值——ch7 长上下文实验的 O(n^2) 成本地板；
#       (2) 流式扫描 log/book2-scaling/owt_docs.jsonl 的文档字符长度分布（直方图落 JSON），
#           按 Book2 实测口径 bytes_per_token=3.82 换算，估算「≥8k token 长文档可切出多少 8k 训练窗」；
#       (3) Project Gutenberg 一本公版书（Alice, #11）下载管道冒烟（走 env.sh 缓存落 log/）。
# 所属章节：Book3 00-feasibility（ch7 长上下文工程线实验底数与语料预算）。
# 运行方式：cd code/00-feasibility && python 04_probe_d_longwin.py [--out-name NAME]
#   产物：log/book3-feasibility/{out-name}.json + gutenberg/（下载件；均不 commit）
import argparse
import json
import os
import sys
import time
import urllib.request

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from llama_slots import LLaMA, LLaMAConfig   # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book3-feasibility")
OWT_JSONL = os.path.join(REPO_ROOT, "log", "book2-scaling", "owt_docs.jsonl")
SEED = 20261002
BYTES_PER_TOKEN = 3.82          # Book2 ch08 data_meta.json 实测口径（同一语料 sp-8k 分词），chars->tokens 换算系数


def longwin_timing(device):
    """seq 四档 sec/step 实测（B=1，fwd+bwd+step；丢前 2 步预热）。"""
    torch.manual_seed(SEED)
    cfg = LLaMAConfig(vocab_size=8192, hidden_size=256, intermediate_size=704,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=8192)
    model = LLaMA(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
    rows = []
    for n in [1024, 2048, 4096, 8192]:
        g = torch.Generator().manual_seed(SEED)
        x = torch.randint(0, 8192, (1, n), generator=g).to(device)
        y = torch.randint(0, 8192, (1, n), generator=g).to(device)
        base = torch.mps.current_allocated_memory() / 2**20 if device.type == "mps" else 0.0
        peak, times = base, []
        for _ in range(6):
            t0 = time.perf_counter()
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if device.type == "mps":
                torch.mps.synchronize()
            times.append(time.perf_counter() - t0)
            if device.type == "mps":
                peak = max(peak, torch.mps.current_allocated_memory() / 2**20)
        steady = sum(times[2:]) / len(times[2:])
        rows.append({"seq": n, "sec_step": round(steady, 4), "mem_peak_delta_mb": round(peak - base, 1)})
        print(f"[seq={n:5d}] sec/step = {steady:.4f}  峰值增量 = {peak - base:.0f} MB", flush=True)
        del x, y
    # O(n^2) 检查：注意力分附近的倍率（sec 比值 vs n 比值）
    s = {r["seq"]: r["sec_step"] for r in rows}
    if 1024 in s and 8192 in s:
        rows.append({"check_n2": "8192/1024 sec 倍率 = %.2f（n^2 纯注意力理论 = 64；实测含线性项<64）" % (s[8192] / s[1024])})
        print(rows[-1]["check_n2"])
    return rows


def corpus_survey():
    """流式扫 jsonl：文档字符长度分布 + ≥8k-token 窗估算。"""
    t0 = time.perf_counter()
    import numpy as np
    lengths = []
    with open(OWT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                doc = json.loads(line)
                lengths.append(len(doc.get("text", "")))
            except json.JSONDecodeError:
                continue
    lengths = np.array(lengths, dtype=np.int64)
    tok = lengths / BYTES_PER_TOKEN                        # 换算为 token 数（口径见文件头注释）
    bins_chars = [0, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, np.inf]
    hist, _ = np.histogram(lengths, bins=bins_chars)
    # 每个 ≥8k token 文档可切出的不重叠 8k 窗数（下取整）
    windows_8k = int(np.maximum(np.floor(tok / 8192), 0).sum())
    survey = {
        "docs_total": int(len(lengths)),
        "chars_total": int(lengths.sum()),
        "tokens_est_total": int(tok.sum()),
        "bytes_per_token_basis": "Book2 ch08 data_meta.json 实测 3.82（同语料 sp-8k）",
        "char_len_quantiles": {q: int(np.percentile(lengths, q)) for q in [50, 75, 90, 95, 99]},
        "char_len_max": int(lengths.max()),
        "hist_chars": {f"{int(bins_chars[i])}-{int(bins_chars[i+1]) if np.isfinite(bins_chars[i+1]) else 'inf'}": int(hist[i])
                       for i in range(len(hist))},
        "docs_ge_8k_tokens": int((tok >= 8192).sum()),
        "docs_ge_8k_tokens_frac": round(float((tok >= 8192).mean()), 4),
        "trainable_8k_windows_total": windows_8k,
        "docs_ge_30k_tokens": int((tok >= 30720).sum()),
        "scan_sec": round(time.perf_counter() - t0, 1),
    }
    print(f"[语料] {survey['docs_total']} docs / {survey['chars_total']/1e9:.2f} G chars ≈ "
          f"{survey['tokens_est_total']/1e6:.0f} M tok；≥8k tok 文档 {survey['docs_ge_8k_tokens']} 篇"
          f"（{survey['docs_ge_8k_tokens_frac']*100:.1f}%），可切 8k 窗共 {windows_8k:,} 个（扫描 {survey['scan_sec']}s）")
    return survey


def gutenberg_smoke():
    """Project Gutenberg 下载冒烟：Alice in Wonderland（#11，公版）。走 log/ 缓存目录。"""
    url = "https://www.gutenberg.org/files/11/11-0.txt"
    dest_dir = os.path.join(OUT_DIR, "gutenberg")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "alice_11.txt")
    rec = {"url": url, "dest": dest}
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
        rec.update({"status": "cached", "bytes": os.path.getsize(dest)})
    else:
        try:
            t0 = time.perf_counter()
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (book3-feasibility probe)"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                f.write(r.read())
            rec.update({"status": "ok", "bytes": os.path.getsize(dest),
                        "sec": round(time.perf_counter() - t0, 1)})
        except Exception as e:                              # 网络失败：如实记录，不算探针失败
            rec.update({"status": "failed", "error": str(e)[:200]})
    print(f"[gutenberg] {rec['status']}  {rec.get('bytes', 0)} bytes  -> {dest}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="probe_d_longwin")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    result = {"seed": SEED, "device": str(device),
              "model": "LLaMA d=256/L=4/h=4/h_kv=4 (RoPE), B=1, fwd+bwd+AdamW",
              "longwin_timing": longwin_timing(device),
              "corpus_survey": corpus_survey(),
              "gutenberg": gutenberg_smoke()}
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{args.out_name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"产物已写：{out}")


if __name__ == "__main__":
    main()
