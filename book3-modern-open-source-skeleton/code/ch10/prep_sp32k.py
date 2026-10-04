# prep_sp32k.py —— Book3 ch10 大项目前置（P-03）：sp-32k 分词器 + OWT 语料重切
# 用途：为 ch10 整合件（llama215，V=32000 LLaMA 口径）备料——
#   ① 在独立分片 log/book2-scaling/owt_tokenizer_corpus.txt（6000 篇/29MB，与 LM 语料互斥）
#     上训练 sp-BPE V=32000 分词器（bpe/byte_fallback/coverage=1.0——Book2 ch8 sp-8k 同配方，
#     只变 vocab_size；不复用 ch05 的 sp32768：V 不同，ch10 config 定版 V=32000，
#     初始 loss 判据锚 ln 32000=10.37）；
#   ② 把 LM 主语料 log/book2-scaling/owt_docs.jsonl 流首部逐篇重切为 token 流
#     （doc 间插 EOS，尾 1.2M tok 留独立 eval 池）——切片协议与 ch05 vocab_scan 四臂同口径：
#     同文本起点（流首部）、同池深目标、doc 粒度推进，便于跨章对照。
# 池深：train 池目标 42M tok（= 5000 步 × 8192 tok/步 的余量；ch10 整合短训只用前 8.2M），
#   eval 池 1.2M tok（对齐 Book2 ch08 / ch05 各臂体量）。
# 校验（参数带校验，销项要求）：token 数（train ≥ 目标 / eval == 1.2M）、id 范围（< 32000，
#   uint16 安全）、文档边界（EOS 计数 == 编码篇数 + 首 3 篇回读逐 id 比对）、磁盘字节数 == 2×token 数。
# 运行：source env.sh && python prep_sp32k.py [--workers 8]；预算 <30 分钟（CPU 为主）。
# 产物（log/book3-ch10/，不入库）：sp32000_owt.model/.vocab + tokens32k.bin + eval32k.bin
#   + slice_meta_32k.json（ch10 写作 agent 的数据读法见该 json 字段）。
import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import sentencepiece as spm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# ---------------- 固定口径 ----------------
V = 32000                                   # LLaMA 词表口径（ch10 config 定版）
TOK_TXT = os.path.join(REPO_ROOT, "log", "book2-scaling", "owt_tokenizer_corpus.txt")  # 分词器独立分片
DOCS_JSONL = os.path.join(REPO_ROOT, "log", "book2-scaling", "owt_docs.jsonl")         # LM 主语料
OUT_DIR = os.path.join(REPO_ROOT, "log", "book3-ch10")
TRAIN_TARGET = 42_000_000                   # train 池目标（5000 步 × 8192 tok 余量，ch05 同口径）
EVAL_TOKENS = 1_200_000                     # 独立 eval 池（尾切）
SAMPLE_DOCS = 2000                          # 压缩率固定样本（首 2000 篇，与 ch05 同文本可比）

_SP = None      # worker 内持有分词器


def _init_worker(model_path):
    global _SP
    _SP = spm.SentencePieceProcessor()
    _SP.Load(model_path)


def _encode_range(job):
    """编码 jsonl 行区间 [start,end) -> uint16 part 文件（doc 间插 EOS，Book2 ch8 同口径）。"""
    start, end, part_path = job
    sp = _SP
    buf, n_tok, n_unk, n_doc = [], 0, 0, 0
    with open(DOCS_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= end:
                break
            line = line.strip()
            if not line:
                continue
            ids = sp.encode(json.loads(line)["text"], out_type=int, add_eos=True)  # EOS 结尾
            buf.extend(ids)
            n_tok += len(ids)
            n_unk += sum(1 for t in ids if t == 0)
            n_doc += 1
            if len(buf) > 4_000_000:                       # 分批落盘控内存
                with open(part_path, "ab") as g:
                    g.write(np.asarray(buf, dtype=np.uint16).tobytes())
                buf = []
    if buf:
        with open(part_path, "ab") as g:
            g.write(np.asarray(buf, dtype=np.uint16).tobytes())
    return part_path, n_tok, n_unk, n_doc


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def train_tokenizer():
    """分词器训练（存在即复用）：sp-BPE / byte_fallback / coverage=1.0——sp-8k 同配方只变 V。"""
    prefix = os.path.join(OUT_DIR, "sp32000_owt")
    model_path = prefix + ".model"
    if os.path.exists(model_path):
        print(f"[分词器] 复用已存在：{model_path}", flush=True)
    else:
        print(f"[分词器] 训练 sp-{V}（独立分片 {TOK_TXT}，bpe/byte_fallback/coverage=1.0）…", flush=True)
        t0 = time.perf_counter()
        spm.SentencePieceTrainer.Train(
            input=TOK_TXT, model_prefix=prefix, vocab_size=V,
            model_type="bpe", character_coverage=1.0, byte_fallback=True,
        )
        print(f"[分词器] sp-{V} 完成（{time.perf_counter()-t0:.1f}s）", flush=True)
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    assert sp.GetPieceSize() == V, f"词表 {sp.GetPieceSize()} != {V}"
    eos_id = sp.piece_to_id("</s>")
    meta = {"path": model_path, "piece_size": sp.GetPieceSize(),
            "unk_id": sp.unk_id(), "bos_id": sp.bos_id(), "eos_id": eos_id,
            "sha256": sha256_of(model_path), "bytes": os.path.getsize(model_path),
            "train_input": TOK_TXT,
            "recipe": "sentencepiece bpe / byte_fallback / character_coverage=1.0（Book2 ch8 sp-8k 同配方，只变 vocab_size）",
            "why_not_reuse_ch05_sp32768": "ch05 词表扫描臂 V=32768 与 ch10 config 定版 V=32000 不同值；"
                                          "ch10 初始 loss 判据锚 ln 32000=10.37，须 32000 词表",
            }
    return model_path, sp, meta


def estimate_docs_needed(sp_path, target_tokens):
    """用首 200 篇估计 tokens/doc，算覆盖 target（+20% 余量）所需行数（ch05 同法）。"""
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_path)
    n_tok, n_doc = 0, 0
    with open(DOCS_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 200:
                break
            line = line.strip()
            if not line:
                continue
            n_tok += len(sp.encode(json.loads(line)["text"], out_type=int, add_eos=True))
            n_doc += 1
    per_doc = n_tok / max(1, n_doc)
    return int(target_tokens / per_doc * 1.2) + 1000


def slice_stream(sp_path, workers):
    """自切 LM 流：编码流首部直至 ≥ TRAIN_TARGET+EVAL_TOKENS，尾 EVAL_TOKENS 留 eval（ch05 同协议）。"""
    train_bin = os.path.join(OUT_DIR, "tokens32k.bin")
    eval_bin = os.path.join(OUT_DIR, "eval32k.bin")
    meta_path = os.path.join(OUT_DIR, "slice_meta_32k.json")
    if os.path.exists(train_bin) and os.path.exists(eval_bin) and os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

    target = TRAIN_TARGET + EVAL_TOKENS
    parts_dir = os.path.join(OUT_DIR, "tmp_parts_32k")
    if os.path.exists(parts_dir):                       # 清理旧 part（重跑口径：全量重算，编码确定性）
        for p in os.listdir(parts_dir):
            os.remove(os.path.join(parts_dir, p))
    os.makedirs(parts_dir, exist_ok=True)
    t0 = time.perf_counter()
    need_from = 0
    need_lines = estimate_docs_needed(sp_path, target)
    chunks, total_tok, total_unk, total_doc = [], 0, 0, 0
    while True:   # 分块推进直至达标（估不准时自动续块）
        jobs = []
        nchunk = max(1, (need_lines - need_from + workers - 1) // workers)
        for k in range(need_from, need_lines, nchunk):
            jobs.append((k, min(k + nchunk, need_lines), os.path.join(parts_dir, f"part_{k:08d}.bin")))
        with Pool(len(jobs), initializer=_init_worker, initargs=(sp_path,)) as pool:
            results = pool.map(_encode_range, jobs)
        chunks.extend(results)
        total_tok += sum(r[1] for r in results)
        total_unk += sum(r[2] for r in results)
        total_doc += sum(r[3] for r in results)
        print(f"  [切流] 行 [0:{need_lines}) 完成，累计 {total_tok:,} tok（{total_doc:,} 篇）", flush=True)
        if total_tok >= target:
            break
        need_from, need_lines = need_lines, need_lines + estimate_docs_needed(sp_path, target - total_tok)
    # 按行序拼接（order-preserving，Book2 ch8 同法）
    stream = np.empty(total_tok, dtype=np.uint16)
    pos = 0
    for part_path, n_tok, _, _ in chunks:
        with open(part_path, "rb") as f:
            arr = np.frombuffer(f.read(), dtype=np.uint16)
        stream[pos:pos + n_tok] = arr
        pos += n_tok
    assert pos == total_tok
    eval_arr = stream[-EVAL_TOKENS:].copy()
    train_arr = stream[:-EVAL_TOKENS].copy()
    train_arr.tofile(train_bin)
    eval_arr.tofile(eval_bin)
    for part_path, _, _, _ in chunks:
        os.remove(part_path)
    os.rmdir(parts_dir)
    wallclock = round(time.perf_counter() - t0, 1)
    # ---- 切片参数校验（「参数带校验」销项要求）----
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_path)
    eos_id = sp.piece_to_id("</s>")
    assert int(stream.max()) < V, f"校验失败：id 越界（max={stream.max()} ≥ V={V}）"
    assert int(stream.min()) >= 0
    assert len(train_arr) >= TRAIN_TARGET, f"校验失败：训练池 {len(train_arr):,} < {TRAIN_TARGET:,}"
    assert len(eval_arr) == EVAL_TOKENS
    assert os.path.getsize(train_bin) == 2 * len(train_arr), "校验失败：train_bin 字节数 != 2×token 数"
    assert os.path.getsize(eval_bin) == 2 * len(eval_arr), "校验失败：eval_bin 字节数 != 2×token 数"
    n_eos = int((stream == eos_id).sum())                # 文档边界：EOS 计数 == 编码篇数
    assert n_eos == total_doc, f"校验失败：EOS 数 {n_eos} != 篇数 {total_doc}"
    head = []                                             # 文档边界：首 3 篇逐 id 回读比对
    with open(DOCS_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            head.extend(sp.encode(json.loads(line)["text"], out_type=int, add_eos=True))
    assert list(stream[:len(head)]) == head, "校验失败：流首部与首 3 篇重编码不一致"
    meta = {"V": V, "dtype": "uint16", "tokenizer": sp_path,
            "docs_encoded": total_doc, "unk_tokens": total_unk,
            "unk_rate": total_unk / total_tok,
            "train_tokens": int(len(train_arr)), "eval_tokens": int(len(eval_arr)),
            "train_bin": train_bin, "eval_bin": eval_bin,
            "train_bytes": os.path.getsize(train_bin), "eval_bytes": os.path.getsize(eval_bin),
            "eos_id": eos_id, "eos_count": n_eos,
            "id_min": int(stream.min()), "id_max": int(stream.max()),
            "protocol": ("owt_docs.jsonl 流首部逐篇编码（EOS 分隔），尾 1.2M tok 留独立 eval 池；"
                         "池深目标 42M+1.2M（ch05 vocab_scan 同协议同文本起点，便于跨章对照）；"
                         "ch10 整合短训 1000 步 × 8192 tok 只用前 8.2M"),
            "checks": ["piece_size == 32000", "0 <= id < 32000 (uint16 安全)",
                       "train_tokens >= 42,000,000", "eval_tokens == 1,200,000",
                       "磁盘字节数 == 2 x token 数", "EOS 计数 == 编码篇数（文档边界）",
                       "流首部与首 3 篇重编码逐 id 一致（文档边界）"],
            "wallclock_sec": wallclock}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  [切流] train {len(train_arr):,} tok / eval {len(eval_arr):,} tok / unk {total_unk} / "
          f"EOS {n_eos} == 篇数 {total_doc}（{wallclock}s）", flush=True)
    return meta


def measure_compression(sp_path):
    """固定样本压缩率（首 SAMPLE_DOCS 篇，与 ch05 四臂同文本可比）：bytes/token 与 tokens/word。"""
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_path)
    n_tok, n_bytes, n_words = 0, 0, 0
    with open(DOCS_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= SAMPLE_DOCS:
                break
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)["text"]
            n_tok += len(sp.encode(text, out_type=int, add_eos=True))
            n_bytes += len(text.encode("utf-8"))
            n_words += len(text.split())
    return {"sample_docs": SAMPLE_DOCS, "sample_bytes": n_bytes, "sample_tokens": n_tok,
            "bytes_per_token": round(n_bytes / n_tok, 4), "tokens_per_word": round(n_tok / max(1, n_words), 4)}


def main():
    ap = argparse.ArgumentParser(description="ch10 前置 P-03：sp-32k 分词器训练 + owt_docs.jsonl 重切")
    ap.add_argument("--workers", type=int, default=8, help="语料编码并行 worker 数")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    t_all = time.perf_counter()

    tok_meta_path, _, tok_meta = train_tokenizer()
    slice_meta = slice_stream(tok_meta_path, args.workers)
    compression = measure_compression(tok_meta_path)
    print(f"[压缩率] sp-{V}: {compression['bytes_per_token']} B/tok, "
          f"{compression['tokens_per_word']} tok/word（首 {SAMPLE_DOCS} 篇，与 ch05 同文本）", flush=True)

    report = {"V": V, "tokenizer": tok_meta, "slice": slice_meta, "compression": compression,
              "total_wallclock_sec": round(time.perf_counter() - t_all, 1),
              "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_json = os.path.join(OUT_DIR, "prep_sp32k_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[全部完成] {report['total_wallclock_sec']}s -> {out_json}", flush=True)
    print(f"  分词器: {tok_meta_path}（sha256 {tok_meta['sha256'][:16]}…）", flush=True)
    print(f"  train : {slice_meta['train_bin']}（{slice_meta['train_tokens']:,} tok / "
          f"{slice_meta['train_bytes']:,} B）", flush=True)
    print(f"  eval  : {slice_meta['eval_bin']}（{slice_meta['eval_tokens']:,} tok）", flush=True)


if __name__ == "__main__":
    main()
