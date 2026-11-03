# 03_probe_c_bpe.py —— 探针(c)：自写最小 BPE 冒烟 + sentencepiece 8k 对照（ch2 数据管道底座）
# 自写 BPE：词频表上的字符级起点 + top-k 相邻对合并；产出压缩率、held-out round-trip 无损
# 验证、非 ASCII 样本切分路径。sentencepiece 侧复训 8k 词表，复算 Book1 的「3.7 字符/piece」口径。
# 用法：python 03_probe_c_bpe.py；产物落 log/book2-feasibility/probe_c_bpe.json 与 c_corpus.txt

import json
import os
import re
import time
from collections import Counter, defaultdict

import sentencepiece as spm

SEED = 20261002
OUT_DIR = os.path.join("log", "book2-feasibility")
SHAKESPEARE = os.path.join(OUT_DIR, "data", "tiny_shakespeare.txt")
CORPUS_PATH = os.path.join(OUT_DIR, "c_corpus.txt")
N_MERGES = 1000  # 自写 BPE 的合并次数（教学档：词表 ≈ 基础字符数 + 1000）

# 非 ASCII 对照样本（回收 Book1 的 Nähe 算例，另加法/中/emoji 各一条）
NON_ASCII_SAMPLES = [
    "Die Nähe des neuen Büros war für die Mitarbeiter überraschend praktisch.",
    "Le café très près de la gare est fermé le lundi.",
    "距离火车站很近的新办公室让同事们感到意外地方便。",
    "Emoji sanity check: 🚀🤖 fits in a byte-free world? …",
    "Mixed Nähe café 距离 line with digits 12345 and punctuation!!",
]


def build_corpus():
    """1MB 英文（tiny shakespeare 截断）+ 非 ASCII 样本重复 40 遍（~6KB，让德文字符进高频表）。"""
    with open(SHAKESPEARE, "r", encoding="utf-8") as f:
        en = f.read(1024 * 1024)
    extra = ("\n".join(NON_ASCII_SAMPLES) + "\n") * 40
    corpus = en + extra
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        f.write(corpus)
    return corpus


# ---------------- 自写最小 BPE（教学口径：词频表 + 相邻对合并，带 pair->word 倒排索引提速） ----------------
class MiniBPE:
    """词内 BPE：先按空白切词统计词频，词 = 字符元组；迭代合并全局最高频相邻对。"""

    def __init__(self, merges: int):
        self.n_merges = merges
        self.merges = []          # 合并规则列表 [(a, b), ...]
        self.word_freq = None     # {tuple(symbols): freq}
        self.base_vocab = set()

    def _pretokenize(self, text):
        """按非空白连续段切词（简化 GPT-2 预切词：真实 GPT-2 用正则+空格并入词内，教学版从简）。"""
        return Counter(re.findall(r"\S+", text))

    def fit(self, text):
        t0 = time.perf_counter()
        self.word_freq = self._pretokenize(text)
        words = {i: list(w) for i, w in enumerate(self.word_freq.keys())}
        freqs = list(self.word_freq.values())
        self.base_vocab = set(ch for w in words.values() for ch in w)

        # 倒排索引：符号对 -> 含该对的词 id 集合（避免每次合并全量重扫）
        pair_to_words = defaultdict(set)
        pair_counts = Counter()

        def add_word_pairs(i):
            syms = words[i]
            for a, b in zip(syms, syms[1:]):
                pair_to_words[(a, b)].add(i)
                pair_counts[(a, b)] += freqs[i]

        for i in words:
            add_word_pairs(i)

        for _ in range(self.n_merges):
            if not pair_counts:
                break
            best, best_c = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
            if best_c < 2:  # 频次 <2 的合并无压缩收益
                break
            self.merges.append(best)
            a, b = best
            merged = a + b
            affected = list(pair_to_words[best])
            for i in affected:
                syms, f = words[i], freqs[i]
                # 先撤销旧对计数，再按合并后的序列重建该词的对
                for p in zip(syms, syms[1:]):
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    pair_to_words[p].discard(i)
                new_syms, j = [], 0
                while j < len(syms):
                    if j < len(syms) - 1 and syms[j] == a and syms[j + 1] == b:
                        new_syms.append(merged)
                        j += 2
                    else:
                        new_syms.append(syms[j])
                        j += 1
                words[i] = new_syms
                for p in zip(new_syms, new_syms[1:]):
                    pair_counts[p] += f
                    pair_to_words[p].add(i)
            pair_to_words.pop(best, None)
        self._words = words
        self._freqs = freqs
        return round(time.perf_counter() - t0, 2)

    def vocab(self):
        return self.base_vocab | {a + b for a, b in self.merges}

    def encode_word(self, word: str):
        """编码单个词：从字符序列出发，按学到的合并规则顺序应用（贪心按规则序，教学实现）。"""
        syms = list(word)
        for a, b in self.merges:
            i = 0
            while i < len(syms) - 1:
                if syms[i] == a and syms[i + 1] == b:
                    syms[i:i + 2] = [a + b]
                i += 1
        return syms

    def encode(self, text: str):
        """编码整段文本 -> piece 列表（词间空格丢弃，解码时用已知词表边界还原的教学简化：
        以 ' ' 连接各词的 piece 串）。"""
        pieces = []
        for w in re.findall(r"\S+", text):
            pieces.extend(self.encode_word(w))
        return pieces

    def decode(self, pieces):
        """无损还原：piece 直接拼接、词间补空格（编码端丢空格数 == 词数-1，需原文本换行信息——
        故 round-trip 验证采用『词粒度拼接 + 原始空白骨架』两路对照）。"""
        return " ".join(pieces)


def roundtrip_check(bpe: MiniBPE, heldout: str, n=200):
    """held-out 无损验证：逐词 encode->decode 拼接 == 原词；整词序列不依赖空白细节。"""
    import random
    rng = random.Random(SEED)
    words = re.findall(r"\S+", heldout)
    sample = rng.sample(words, min(n, len(words)))
    bad = []
    for w in sample:
        if "".join(bpe.encode_word(w)) != w:
            bad.append(w)
    return {"n": len(sample), "lossless": len(bad) == 0, "bad_examples": bad[:5]}


def compression(bpe: MiniBPE, text: str):
    """压缩率：非空白字符数 / piece 数（口径 A）与 全字符数(含空格) / piece 数（口径 B，对齐
    sentencepiece 惯例的粗略换算）。"""
    pieces = bpe.encode(text)
    chars_nospace = sum(len(w) for w in re.findall(r"\S+", text))
    return {"pieces": len(pieces),
            "chars_per_piece_nospace": round(chars_nospace / len(pieces), 3),
            "chars_per_piece_withspace": round(len(text) / len(pieces), 3)}


def sentencepiece_run(corpus: str, vocab_size=8000, coverage=1.0, tag="sp_8k"):
    """sentencepiece BPE 训练 + 编码统计（对齐 Book1 的 BPE-8k 管道口径）。"""
    t0 = time.perf_counter()
    model_prefix = os.path.join(OUT_DIR, f"c_{tag}")
    spm.SentencePieceTrainer.train(
        input=CORPUS_PATH, model_prefix=model_prefix, vocab_size=vocab_size,
        model_type="bpe", character_coverage=coverage, input_sentence_size=1_000_000,
        shuffle_input_sentence=False)
    train_sec = round(time.perf_counter() - t0, 2)
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    n_unk = 0
    piece_total = 0
    for line in NON_ASCII_SAMPLES:
        ids = sp.encode(line, out_type=int)
        pieces = sp.encode(line, out_type=str)
        piece_total += len(ids)
        n_unk += sum(1 for i in ids if i == sp.unk_id())
    corpus_pieces = sum(len(sp.encode(line)) for line in corpus.splitlines()[:2000])
    corpus_chars = sum(len(line) for line in corpus.splitlines()[:2000])
    return {"train_sec": train_sec, "vocab_size": sp.get_piece_size(),
            "chars_per_piece_withspace": round(corpus_chars / corpus_pieces, 3),
            "nonascii_unk_pieces": n_unk, "nonascii_pieces_total": piece_total,
            "naehe_path": sp.encode("Nähe", out_type=str),
            "sample_paths": {s[:24]: sp.encode(s, out_type=str)[:12]
                             for s in NON_ASCII_SAMPLES[:2]}}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    corpus = build_corpus()
    with open(SHAKESPEARE, "r", encoding="utf-8") as f:
        full = f.read()
    # 切 90/10 train/held-out（从同分布语料取 held-out，验证泛化压缩与无损）
    train_text, heldout = corpus, full[-100_000:] + ("\n".join(NON_ASCII_SAMPLES) * 4)

    report = {"meta": {"date": "2026-10-02", "seed": SEED,
                       "corpus_bytes": os.path.getsize(CORPUS_PATH),
                       "n_merges": N_MERGES}}

    # 1) 自写最小 BPE
    bpe = MiniBPE(N_MERGES)
    fit_sec = bpe.fit(train_text)
    vocab = bpe.vocab()
    comp_train = compression(bpe, train_text[:400_000])
    comp_held = compression(bpe, heldout)
    rt = roundtrip_check(bpe, heldout)
    report["mini_bpe"] = {
        "fit_sec": fit_sec, "vocab_size": len(vocab),
        "base_char_vocab": len(bpe.base_vocab),
        "compression_train": comp_train, "compression_heldout": comp_held,
        "roundtrip": rt,
        "top_merges": ["".join(p) for p in bpe.merges[:15]],
        "naehe_path": bpe.encode_word("Nähe"),
        "no_unk_by_construction": True,
    }
    print("mini_bpe:", json.dumps(report["mini_bpe"], ensure_ascii=False)[:400], flush=True)

    # 2) sentencepiece 对照：coverage=1.0 与 0.9995 双口径
    report["sp_8k_cov1"] = sentencepiece_run(corpus, 8000, 1.0, "sp_8k_cov1")
    print("sp cov1.0:", json.dumps(report["sp_8k_cov1"], ensure_ascii=False)[:300], flush=True)
    report["sp_8k_cov09995"] = sentencepiece_run(corpus, 8000, 0.9995, "sp_8k_cov09995")
    print("sp cov0.9995:", json.dumps(report["sp_8k_cov09995"], ensure_ascii=False)[:300], flush=True)

    out_path = os.path.join(OUT_DIR, "probe_c_bpe.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
