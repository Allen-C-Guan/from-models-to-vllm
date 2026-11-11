# data_prep.py —— Book2 ch8 Kaplan 单曲线族：数据制备（分词器训练 + LM 语料编码 + eval 池留出）
# 所属章节：Book2 第8章8.5节（drafts/Book2-预训练革命/ch08-ScalingLaws.md）
# 设计书：research/Book2-预训练革命/notes/01-实验可行性.md §4（正式族口径）
#   * 分词器 sp-8k 只在独立分片 owt_tokenizer_corpus.txt（6000 篇 / 29MB）上训练——与 LM 语料分离
#     这是历史输入准备约定；文件名不同本身不能证明文档内容或近重复已经隔离。
#   * LM 主语料 owt_docs.jsonl（232,920 篇 / 1.15GB）逐篇编码为 uint16 token 流，文档间插 EOS(id=2)
#     后扁平拼接（GPT-2 式 end-of-text 拼接流；训练时 256-token 窗口可跨文档，与 GPT-2 预训练同口径）。
#   * 末尾 EVAL_TOKENS=1,200,000 token 留作 eval 池（>=120 万，设计书要求），与训练段（取自流首部）互斥。
#   * byte_fallback=true：保留256个字节piece，编码后逐篇统计UNK；仍经过SentencePiece自身规范化，
#     不等于GPT-2的预切分、字节映射与合并规则。
# 运行方式：python data_prep.py [--workers 10]
# 产物：log/book2-ch08/{sp8k_owt.model, sp8k_owt.vocab, tokens.bin, eval.bin, data_meta.json}

import argparse
import json
import os
from pathlib import Path
import hashlib
import time
from multiprocessing import Pool

import numpy as np
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2

REPO_ROOT = os.getcwd()  # 所有缓存相对当前运行目录；可用命令行指定数据与产物目录。
DATA_DIR = os.path.join(REPO_ROOT, "log", "book2-scaling")
OUT_DIR = os.path.join(REPO_ROOT, "log", "book2-ch08")
TOK_TXT = os.path.join(DATA_DIR, "owt_tokenizer_corpus.txt")   # 分词器分片（与 LM 语料互斥的分片）
DOCS_JSONL = os.path.join(DATA_DIR, "owt_docs.jsonl")          # LM 主语料
MODEL_PREFIX = os.path.join(OUT_DIR, "sp8k_owt")
EVAL_TOKENS = 1_200_000
SEED = 20261002  # 历史实验标签；此常量并未作为SentencePiece随机种子传入，不能保证跨版本逐位一致。

_SP = None  # worker 内持有，避免每篇重载


def _init_worker(model_path, docs_path):
    global _SP, DOCS_JSONL
    DOCS_JSONL = docs_path
    _SP = spm.SentencePieceProcessor()
    _SP.Load(model_path)


def _encode_range(job):
    """编码 jsonl 的行区间 [start,end)（每行 {"text": ...}）-> uint16 part 文件。
    逐篇 add_eos=True（文档间插 EOS id=2）；unk(id=0) 逐篇计数。返回 (part_path, n_tok, n_unk, n_doc, raw_text_bytes)。"""
    start, end, part_idx, part_path = job
    sp = _SP
    buf = []
    n_tok = n_unk = n_doc = raw_text_bytes = 0
    with open(part_path, "xb"):
        pass  # 每个worker只创建自己的新文件，已有分片绝不覆盖。
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
            if not isinstance(text, str):
                raise ValueError(f"第{i+1}行text不是字符串")
            raw_text_bytes += len(text.encode("utf-8"))
            ids = sp.encode(text, out_type=int, add_eos=True)  # EOS(id=2)结尾
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
    return part_path, n_tok, n_unk, n_doc, raw_text_bytes


def count_lines(path):
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def main():
    global DATA_DIR, OUT_DIR, TOK_TXT, DOCS_JSONL, MODEL_PREFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    if args.workers <= 0:
        ap.error("--workers必须为正数")
    DATA_DIR, OUT_DIR = os.path.abspath(args.data_dir), os.path.abspath(args.out_dir)
    TOK_TXT = os.path.join(DATA_DIR, "owt_tokenizer_corpus.txt")
    DOCS_JSONL = os.path.join(DATA_DIR, "owt_docs.jsonl")
    MODEL_PREFIX = os.path.join(OUT_DIR, "sp8k_owt")
    protected = ["tokens.bin", "eval.bin", "data_meta.json", "tmp_parts"]
    if any((Path(OUT_DIR) / name).exists() for name in protected):
        raise FileExistsError("输出目录已有语料或临时分片；请指定新的--out-dir，先核对已有产物")
    if not Path(DOCS_JSONL).is_file() or not Path(TOK_TXT).is_file():
        raise FileNotFoundError("需要owt_docs.jsonl与owt_tokenizer_corpus.txt两份输入")
    if count_lines(DOCS_JSONL) == 0:
        raise ValueError("LM语料为空")
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.perf_counter()
    meta = {"seed_declared": SEED, "seed_used_by_sentencepiece": False,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "document_overlap_verified": False}

    # ---- 1) 分词器：只在独立分片上训 sp-8k（byte_fallback，coverage=1.0）----
    model_path = MODEL_PREFIX + ".model"
    if not os.path.exists(model_path) and Path(MODEL_PREFIX + ".vocab").exists():
        raise FileExistsError("仅有旧vocab而无model，请换新--out-dir，避免覆盖")
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
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(Path(model_path).read_bytes())
    if (proto.trainer_spec.model_type != sentencepiece_model_pb2.TrainerSpec.BPE
            or not proto.trainer_spec.byte_fallback or (eos_id, bos_id, unk_id) != (2, 1, 0)):
        raise ValueError("分词器需为BPE并开启byte_fallback，且EOS/BOS/UNK编号为2/1/0")
    model_sha = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    meta["tokenizer"] = {
        "type": "sentencepiece BPE vocab=8192 (sp-8k)", "byte_fallback": True,
        "training_input_declared": TOK_TXT,
        "training_input_note": "历史约定为6000篇独立分片；复用模型时不能由本次文件名证明训练来源",
        "model_sha256": model_sha,
        "normalizer_name": proto.normalizer_spec.name,
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
    t1 = time.perf_counter()
    results = []
    with Pool(len(jobs), initializer=_init_worker, initargs=(model_path, DOCS_JSONL)) as pool:
        for part_path, n_tok, n_unk, n_doc, raw_text_bytes in pool.imap_unordered(_encode_range, jobs):
            results.append((part_path, n_tok, n_unk, n_doc, raw_text_bytes))
            print(f"  [part 完成] {os.path.basename(part_path)}: {n_doc} 篇 / {n_tok:,} tok / unk {n_unk}", flush=True)
    order = {os.path.basename(p): i for i, (_, _, _, p) in enumerate(jobs)}
    results.sort(key=lambda r: order[os.path.basename(r[0])])
    total_tok = sum(r[1] for r in results)
    total_unk = sum(r[2] for r in results)
    total_doc = sum(r[3] for r in results)
    raw_text_bytes = sum(r[4] for r in results)
    if total_tok <= EVAL_TOKENS:
        raise ValueError("编码语料不足以同时保留评估池与非空训练池；临时文件保留供检查")
    print(f"[编码] {total_doc} 篇 -> {total_tok:,} tokens（unk {total_unk}），耗时 {time.perf_counter()-t1:.1f}s", flush=True)

    # ---- 3) 拼接为单一 uint16 流 + 末尾切 eval 池 ----
    stream = np.empty(total_tok, dtype=np.uint16)
    pos = 0
    for part_path, n_tok, _, _, _ in results:
        with open(part_path, "rb") as f:
            arr = np.frombuffer(f.read(), dtype=np.uint16)
        assert len(arr) == n_tok
        stream[pos:pos + n_tok] = arr
        pos += n_tok
    eval_bin = stream[-EVAL_TOKENS:].copy()
    train_bin = stream[:-EVAL_TOKENS].copy()
    train_bin.tofile(os.path.join(OUT_DIR, "tokens.bin"))
    eval_bin.tofile(os.path.join(OUT_DIR, "eval.bin"))
    for part_path, *_ in results:  # 只清理本次已拼接的分片
        os.remove(part_path)
    os.rmdir(parts_dir)

    # ---- 4) 统计与元数据 ----
    nbytes = os.path.getsize(DOCS_JSONL)
    # tok/word 换算系数实测（01-实验可行性.md §5 要求在目标语料上实测）：前 2000 篇样本
    sample_ids = sample_words = 0
    with open(DOCS_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 2000:
                break
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            sample_ids += len(sp.encode(text, out_type=int, add_eos=True))
            sample_words += len(text.split())
    uniq, counts = np.unique(stream[:10_000_000], return_counts=True)  # 首 10M token 频率谱（够稳定）
    p = counts / counts.sum()
    ent = float(-(p * np.log(p)).sum())
    meta.update({
        "corpus": {"path": DOCS_JSONL, "bytes": nbytes, "raw_text_utf8_bytes": raw_text_bytes, "docs": total_doc,
                   "note": "预期输入为OpenWebText子采样；实际来源与分片隔离需由输入准备记录核对"},
        "encoding": {"dtype": "uint16", "total_tokens": int(total_tok),
                     "train_pool_tokens": int(len(train_bin)), "eval_pool_tokens": int(len(eval_bin)),
                     "eval_heldout_from": "stream tail；位置区间与train_pool互斥，不保证文档/近重复隔离",
                     "unk_tokens": int(total_unk), "unk_rate": total_unk / total_tok,
                     "bytes_per_token": nbytes / total_tok,
                     "bytes_per_token_definition": "JSONL存储文件字节/含文档EOS的token；不是净文本评估分母",
                     "raw_text_bytes_per_token": raw_text_bytes / total_tok,
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
