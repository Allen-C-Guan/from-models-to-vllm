# data_prep.py —— Book2 ch8 Kaplan 单曲线族：数据制备（分词器训练 + LM 语料编码 + eval 池留出）
# 所属章节：Book2 第 8 章 §8.3（ch08-ScalingLaws.md）
# 设计书：notes/01-实验可行性.md §4（正式族口径）
#   * 分词器 sp-8k 只在独立分片 owt_tokenizer_corpus.txt（6000 篇 / 29MB）上训练——与 LM 语料分离
#     （冒烟档「分词器与 LM 语料未分离」教训的正式档修正，见 01-实验可行性.md §4 冒烟教训）。
#   * LM 主语料 owt_docs.jsonl（232,920 篇 / 1.15GB）逐篇编码为 uint16 token 流，文档间插 EOS(id=2)
#     后扁平拼接（GPT-2 式 end-of-text 拼接流；训练时 256-token 窗口可跨文档，与 GPT-2 预训练同口径）。
#   * 末尾 EVAL_TOKENS=1,200,000 token 留作 eval 池（>=120 万，设计书要求），与训练段（取自流首部）互斥。
#   * byte_fallback=true：256 个字节 piece 兜底，任何字符串可编码（unk=0 构造性保证，GPT-2 字节级
#     BPE 的 sentencepiece 等价物）；实测 unk 计数仍逐篇统计上报。
# 运行方式：python data_prep.py [--workers 10]
# 产物：log/book2-ch08/{sp8k_owt.model, sp8k_owt.vocab, tokens.bin, eval.bin, data_meta.json}

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import sentencepiece as spm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "log", "book2-scaling")
OUT_DIR = os.path.join(REPO_ROOT, "log", "book2-ch08")
TOK_TXT = os.path.join(DATA_DIR, "owt_tokenizer_corpus.txt")   # 分词器分片（与 LM 语料互斥的分片）
DOCS_JSONL = os.path.join(DATA_DIR, "owt_docs.jsonl")          # LM 主语料
MODEL_PREFIX = os.path.join(OUT_DIR, "sp8k_owt")
EVAL_TOKENS = 1_200_000
SEED = 20261002  # 全书统一种子（本脚本无随机源：SP 训练确定性、编码确定性、切分按固定长度）

_SP = None  # worker 内持有，避免每篇重载


def _init_worker(model_path):
    global _SP
    _SP = spm.SentencePieceProcessor()
    _SP.Load(model_path)


def _encode_range(job):
    """编码 jsonl 的行区间 [start,end)（每行 {"text": ...}）-> uint16 part 文件。
    逐篇 add_eos=True（文档间插 EOS id=2）；unk(id=0) 逐篇计数。返回 (part_path, n_tok, n_unk, n_doc)。"""
    start, end, part_idx, part_path = job
    sp = _SP
    buf = []
    n_tok = n_unk = n_doc = 0
    with open(DOCS_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= end:
                break
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)["text"]
            ids = sp.encode(text, out_type=int, add_eos=True)  # EOS(id=2) 结尾
            unk = sum(1 for t in ids if t == 0)
            buf.extend(ids)
            n_tok += len(ids)
            n_unk += unk
            n_doc += 1
            if len(buf) > 4_000_000:  # 分批落盘，控内存
                with open(part_path, "ab") as g:
                    g.write(np.asarray(buf, dtype=np.uint16).tobytes())
                buf = []
    if buf:
        with open(part_path, "ab") as g:
            g.write(np.asarray(buf, dtype=np.uint16).tobytes())
    return part_path, n_tok, n_unk, n_doc


def count_lines(path):
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.perf_counter()
    meta = {"seed_declared": SEED, "date": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ---- 1) 分词器：只在独立分片上训 sp-8k（byte_fallback，coverage=1.0）----
    model_path = MODEL_PREFIX + ".model"
    if os.path.exists(model_path):
        print(f"[分词器] 已存在，复用 {model_path}", flush=True)
    else:
        print(f"[分词器] 在分词器分片上训练 sp-8k（vocab=8192, bpe, byte_fallback=true）…", flush=True)
        spm.SentencePieceTrainer.Train(
            input=TOK_TXT, model_prefix=MODEL_PREFIX, vocab_size=8192,
            model_type="bpe", character_coverage=1.0, byte_fallback=True,
            # 注意：sentencepiece 0.2.2 无 seed_sentencepiece_seed 字段（01-实验可行性.md §8-⑥）；
        )
        print(f"[分词器] 完成 {time.perf_counter()-t0:.1f}s", flush=True)
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    vocab = sp.GetPieceSize()
    assert vocab == 8192, f"词表 {vocab} != 8192"
    eos_id, bos_id, unk_id = sp.eos_id(), sp.bos_id(), sp.unk_id()
    meta["tokenizer"] = {
        "type": "sentencepiece BPE vocab=8192 (sp-8k)", "byte_fallback": True,
        "trained_on": "owt_tokenizer_corpus.txt（6000 篇独立分片，与 LM 语料互斥）",
        "shard_bytes": os.path.getsize(TOK_TXT), "vocab_size": vocab,
        "eos_id": eos_id, "bos_id": bos_id, "unk_id": unk_id,
        "doc_separator": f"每篇末尾 append EOS(id={eos_id})；无 BOS",
    }

    # ---- 2) LM 语料全量编码（multiprocessing，按行区间切分，part 文件按序拼接）----
    n_lines = count_lines(DOCS_JSONL)
    print(f"[语料] {DOCS_JSONL}：{n_lines} 行，编码 workers={args.workers}", flush=True)
    parts_dir = os.path.join(OUT_DIR, "tmp_parts")
    os.makedirs(parts_dir, exist_ok=True)
    n_chunk = (n_lines + args.workers - 1) // args.workers
    jobs = []
    for w in range(args.workers):
        start, end = w * n_chunk, min((w + 1) * n_chunk, n_lines)
        if start >= end:
            break
        jobs.append((start, end, w, os.path.join(parts_dir, f"part_{w:02d}.bin")))
    for _, _, _, p in jobs:  # 清理旧 part（断点重跑口径：全量重算，编码确定性）
        if os.path.exists(p):
            os.remove(p)
    t1 = time.perf_counter()
    results = []
    with Pool(len(jobs), initializer=_init_worker, initargs=(model_path,)) as pool:
        for part_path, n_tok, n_unk, n_doc in pool.imap_unordered(_encode_range, jobs):
            results.append((part_path, n_tok, n_unk, n_doc))
            print(f"  [part 完成] {os.path.basename(part_path)}: {n_doc} 篇 / {n_tok:,} tok / unk {n_unk}", flush=True)
    order = {os.path.basename(p): i for i, (_, _, _, p) in enumerate(jobs)}
    results.sort(key=lambda r: order[os.path.basename(r[0])])
    total_tok = sum(r[1] for r in results)
    total_unk = sum(r[2] for r in results)
    total_doc = sum(r[3] for r in results)
    print(f"[编码] {total_doc} 篇 -> {total_tok:,} tokens（unk {total_unk}），耗时 {time.perf_counter()-t1:.1f}s", flush=True)

    # ---- 3) 拼接为单一 uint16 流 + 末尾切 eval 池 ----
    stream = np.empty(total_tok, dtype=np.uint16)
    pos = 0
    for part_path, n_tok, _, _ in results:
        with open(part_path, "rb") as f:
            arr = np.frombuffer(f.read(), dtype=np.uint16)
        assert len(arr) == n_tok
        stream[pos:pos + n_tok] = arr
        pos += n_tok
    eval_bin = stream[-EVAL_TOKENS:].copy()
    train_bin = stream[:-EVAL_TOKENS].copy()
    train_bin.tofile(os.path.join(OUT_DIR, "tokens.bin"))
    eval_bin.tofile(os.path.join(OUT_DIR, "eval.bin"))
    for p in os.listdir(parts_dir):  # part 临时件清理
        os.remove(os.path.join(parts_dir, p))
    os.rmdir(parts_dir)

    # ---- 4) 统计与元数据 ----
    nbytes = os.path.getsize(DOCS_JSONL)
    # tok/word 换算系数实测（01-实验可行性.md §5 要求在目标语料上实测）：前 2000 篇样本
    sample_ids = sample_words = 0
    with open(DOCS_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 2000:
                break
            text = json.loads(line)["text"]
            sample_ids += len(sp.encode(text, out_type=int, add_eos=True))
            sample_words += len(text.split())
    uniq, counts = np.unique(stream[:10_000_000], return_counts=True)  # 首 10M token 频率谱（够稳定）
    p = counts / counts.sum()
    ent = float(-(p * np.log(p)).sum())
    meta.update({
        "corpus": {"path": DOCS_JSONL, "bytes": nbytes, "docs": total_doc,
                   "note": "OpenWebText 子采样（CC0 包装层）；分片来源与 subsample.log 同源"},
        "encoding": {"dtype": "uint16", "total_tokens": int(total_tok),
                     "train_pool_tokens": int(len(train_bin)), "eval_pool_tokens": int(len(eval_bin)),
                     "eval_heldout_from": "stream tail（与训练段互斥：训练只读流首部 ≤40M token）",
                     "unk_tokens": int(total_unk), "unk_rate": total_unk / total_tok,
                     "bytes_per_token": nbytes / total_tok,
                     "tokens_per_word_sample2000": sample_ids / max(1, sample_words),
                     "token_entropy_nats_first10M": ent,
                     "vocab_usage_first10M": int((counts > 0).sum()), "vocab_size": vocab,
                     "id_max": int(stream.max()), },
        "wallclock_sec": round(time.perf_counter() - t0, 1),
    })
    out_json = os.path.join(OUT_DIR, "data_meta.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta["encoding"], ensure_ascii=False, indent=2), flush=True)
    print(f"[完成] tokens.bin {len(train_bin):,} tok / eval.bin {len(eval_bin):,} tok -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
