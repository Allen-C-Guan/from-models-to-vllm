# -*- coding: utf-8 -*-
# 用途：Book2 第 2 章「BPE 从零讲透」主脚本：Sennrich 手算例复跑、自写 mini-BPE（pair 倒排索引）、
#       Book1 F7 欠条四数字复算钉死、两语料对照（Multi30K vs Gutenberg 德文）、非 ASCII unk 闭环、
#       GPT-2 词表 50257 分解、tiktoken≡HF B 级对拍；全部数字落 log/book2-ch02/bpe_report.json
# 所属章节：《预训练革命》第 2 章（ch02-分词器）
# 运行方式：source env.sh && python "code/ch02/bpe.py"（CPU 约 46 s；Faust 语料缺失时联网下载一次）
import json
import os
import re
import time
from collections import Counter, defaultdict

import sentencepiece as spm
import tiktoken

SEED = 20261003
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "log", "book2-ch02")
M30K = os.path.join(ROOT, "log", "Book1-ch09", "multi30k_corpus.txt")   # Book1 同源语料（CC BY-NC-SA，产物落 log/）
BOOK1_MODEL = os.path.join(ROOT, "log", "Book1-ch09", "bpe_8k.model")   # Book1 自训 shared BPE-8k
SHAKESPEARE = os.path.join(ROOT, "log", "book2-feasibility", "tiny_shakespeare.txt")
FAUST = os.path.join(OUT, "gutenberg_de.txt")                           # Goethe《浮士德》(PG#2229，公版)
OPENAI_VOCAB = os.path.join(ROOT, "log", "Book2-ch02-research", "openai-gpt2")
CJK_PROBES = ["北京欢迎你，世界！", "距离火车站很近的新办公室让同事们感到意外地方便。", "人工智能正在改变世界。"]

# ---------------- 1) Sennrich Algorithm 1 逐字复刻（词典插入序决胜 = 论文代码语义） ----------------
def sennrich_toy(word_freq: dict, n_merges: int) -> list:
    """论文 Algorithm 1 最小复刻。输入 {符号串: 词频}（词内空格分符号），输出前 n_merges 条合并 [(a,b),...]。
    形状语义：'l o w </w>'(4 符号) --合并--> 'lo w </w>'(3 符号)；max() 平票取插入序首个（Python=论文代码语义）。"""
    vocab, merges = dict(word_freq), []
    for _ in range(n_merges):
        pairs = defaultdict(int)
        for word, freq in vocab.items():                      # 词典上按词频加权数对（type 级，不扫语料）
            syms = word.split()
            for a, b in zip(syms, syms[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        pat = re.compile(r"(?<!\S)" + re.escape(" ".join(best)) + r"(?!\S)")
        vocab = {pat.sub("".join(best), w): f for w, f in vocab.items()}
        merges.append(best)
    return merges

# ---------------- 2) 自写 mini-BPE：pair 倒排索引版（fit/encode/decode/压缩率三口径） ----------------
class MiniBPE:
    """词内 BPE：非空白预分词 -> 词频表 -> 字符起步 -> 迭代合并全局最高频相邻对。
    形状语义：fit(text:L_c 字符) 用 {词: 频}；encode(text) -> piece 列表（L_p 个）；encode_word(w) 无损可逆。"""

    def __init__(self, n_merges: int):
        self.n_merges = n_merges
        self.merges = []            # 合并规则表 [(a, b), ...]，行序 = rank = 「顺序即模型」
        self.base_vocab = set()

    def fit(self, text: str) -> float:
        """训练，返回耗时秒。倒排索引 pair -> 含该对的词 id 集：每次合并只重算受影响词（核心提速）。"""
        t0 = time.perf_counter()
        wf = Counter(re.findall(r"\S+", text))
        words = {i: list(w) for i, w in enumerate(wf)}
        freqs = list(wf.values())
        self.base_vocab = {ch for w in words.values() for ch in w}
        pair_counts, pair_to_words = Counter(), defaultdict(set)
        for i, syms in words.items():
            for p in zip(syms, syms[1:]):
                pair_counts[p] += freqs[i]
                pair_to_words[p].add(i)
        for _ in range(self.n_merges):
            if not pair_counts:
                break
            best = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            if pair_counts[best] < 2:                          # 频次 <2 的合并无压缩收益
                break
            self.merges.append(best)
            a, b = best
            for i in list(pair_to_words[best]):                # 撤旧对 -> 合并 -> 建新对
                syms, f = words[i], freqs[i]
                for p in zip(syms, syms[1:]):
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    pair_to_words[p].discard(i)
                new, j = [], 0
                while j < len(syms):
                    if j < len(syms) - 1 and syms[j] == a and syms[j + 1] == b:
                        new.append(a + b)
                        j += 2
                    else:
                        new.append(syms[j])
                        j += 1
                words[i] = new
                for p in zip(new, new[1:]):
                    pair_counts[p] += f
                    pair_to_words[p].add(i)
            pair_to_words.pop(best, None)
        return round(time.perf_counter() - t0, 2)

    def vocab_size(self) -> int:
        return len(self.base_vocab) + len(self.merges)         # 词表 = 基础字符集 + 合并数（唯一超参数）

    def encode_word(self, word: str) -> list:
        """测试时切词：字符起步，按学到的合并表依序合并（贪心按规则序）——任何词必可切 -> 构造上无 unk。"""
        syms = list(word)
        for a, b in self.merges:
            i = 0
            while i < len(syms) - 1:
                if syms[i] == a and syms[i + 1] == b:
                    syms[i:i + 2] = [a + b]
                i += 1
        return syms

    def encode(self, text: str) -> list:
        out = []
        for w in re.findall(r"\S+", text):
            out.extend(self.encode_word(w))
        return out

    def compression(self, text: str) -> dict:
        """压缩率三口径：A=非空白字符/piece（Book1 口径）；B=全字符含空格/piece；C=UTF-8 字节/piece。"""
        pieces = self.encode(text)
        n = len(pieces)
        chars_ns = sum(len(w) for w in re.findall(r"\S+", text))
        return {"pieces": n, "A_nospace": round(chars_ns / n, 3),
                "B_withspace": round(len(text) / n, 3),
                "C_bytes": round(len(text.encode("utf-8")) / n, 3)}

def build_c_corpus() -> tuple:
    """教学语料（探针 c 同源）：1MB shakespeare + 非 ASCII 样本 ×40（让德文字符进高频表）；held-out 另取。"""
    with open(SHAKESPEARE, encoding="utf-8") as f:
        full = f.read()
    samples = ["Die Nähe des neuen Büros war für die Mitarbeiter überraschend praktisch.",
               "Le café très près de la gare est fermé le lundi.",
               "距离火车站很近的新办公室让同事们感到意外地方便。",
               "Emoji sanity check: 🚀🤖 fits in a byte-free world? …",
               "Mixed Nähe café 距离 line with digits 12345 and punctuation!!"]
    return full[:1024 * 1024] + ("\n".join(samples) + "\n") * 40, full[-100_000:] + ("\n".join(samples) + "\n") * 4

# ---------------- 3) 各实验 ----------------
def exp_toy() -> dict:
    """手算例：论文代码词典与图注词典各跑一遍（两者不一致 -> 正文考据框）。"""
    code_dict = {"l o w </w>": 5, "l o w e r </w>": 2, "n e w e s t </w>": 6, "w i d e s t </w>": 3}
    cap_dict = {"l o w </w>": 5, "l o w e s t </w>": 2, "n e w e r </w>": 6, "w i d e r </w>": 3}
    m_code, m_cap = sennrich_toy(code_dict, 10), sennrich_toy(cap_dict, 10)
    m = MiniBPE(10**9)
    m.merges = m_cap                                              # 复用 encode_word 验证 OOV 切分
    return {"merges_code_dict": [" ".join(p) for p in m_code],
            "merges_caption_dict": [" ".join(p) for p in m_cap],
            "oov_lower_path": m.encode_word("lower")}

def exp_mini_bpe(corpus, heldout) -> dict:
    """自写 mini-BPE 主实验：fit 1000 合并 + round-trip 无损 + 压缩率 + 词表扫描（供图 2.2）。"""
    bpe = MiniBPE(1000)
    fit_sec = bpe.fit(corpus)
    words_h = re.findall(r"\S+", heldout)
    rt_ok = all("".join(bpe.encode_word(w)) == w for w in words_h[:: max(1, len(words_h) // 200)][:200])
    sweep = []
    for k in [0, 200, 500, 1000, 2000, 4000]:
        m = MiniBPE(k)
        t = m.fit(corpus)
        rec = {"n_merges": k, "vocab": m.vocab_size(), "fit_sec": t,
               "train": m.compression(corpus[:400_000])["A_nospace"],
               "heldout": m.compression(heldout)["A_nospace"]}
        sweep.append(rec)
        print(f"  sweep merges={k:4d} vocab={rec['vocab']:5d} train={rec['train']:.3f} held={rec['heldout']:.3f} ({t}s)", flush=True)
    return {"n_merges": 1000, "fit_sec": fit_sec, "vocab_size": bpe.vocab_size(),
            "base_chars": len(bpe.base_vocab), "roundtrip_200words": rt_ok,
            "comp_train": bpe.compression(corpus[:400_000]), "comp_heldout": bpe.compression(heldout),
            "top_merges": [a + b for a, b in bpe.merges[:15]],
            "naehe": bpe.encode_word("Nähe"), "sweep": sweep}

def sp_stats(sp, lines: list) -> dict:
    """sentencepiece 语料统计：piece 数 / unk piece 数（piece 级口径）/ 压缩率三口径 / 每词片数。"""
    n_pieces = n_unk = n_words = n_chars = n_spaces = n_bytes = 0
    for line in lines:
        ids = sp.encode(line)
        n_pieces += len(ids)
        n_unk += sum(1 for i in ids if i == sp.unk_id())
        n_words += len(line.split())
        n_chars += len(line) - line.count(" ")
        n_spaces += line.count(" ")
        n_bytes += len(line.encode("utf-8"))
    return {"lines": len(lines), "pieces": n_pieces, "unk_pieces": n_unk,
            "unk_rate_piece": round(100 * n_unk / n_pieces, 2), "words": n_words,
            "A_nospace": round(n_chars / n_pieces, 3), "B_withspace": round((n_chars + n_spaces) / n_pieces, 3),
            "C_bytes": round(n_bytes / n_pieces, 3), "pieces_per_word": round(n_pieces / n_words, 3),
            "chars_per_word": round(n_chars / n_words, 2)}

def exp_book1_recompute() -> dict:
    """第二幕核心：Book1 bpe_8k.model 在同源 Multi30K 上复算四欠条数字 + 同规格重训计时与确定性校验。"""
    lines = open(M30K, encoding="utf-8").read().splitlines()
    sp = spm.SentencePieceProcessor(model_file=BOOK1_MODEL)
    stats = sp_stats(sp, lines)
    t0 = time.perf_counter()
    spm.SentencePieceTrainer.train(
        input=M30K, model_prefix=os.path.join(OUT, "bpe_8k_reverify"), vocab_size=8000, model_type="bpe",
        character_coverage=0.995, pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        shuffle_input_sentence=False, hard_vocab_limit=False)
    retrain_sec = round(time.perf_counter() - t0, 2)
    sp2 = spm.SentencePieceProcessor(model_file=os.path.join(OUT, "bpe_8k_reverify.model"))
    identical = [sp.id_to_piece(i) for i in range(sp.get_piece_size())] == \
                [sp2.id_to_piece(i) for i in range(sp2.get_piece_size())]
    return {"stats": stats, "retrain_sec": retrain_sec, "retrain_identical_vocab": identical,
            "paths": {"Nähe": sp.encode("Nähe", out_type=str), "Büsche": sp.encode("Büsche", out_type=str)},
            "vocab_size": sp.get_piece_size()}

def exp_two_corpora() -> dict:
    """两语料对照（同 coverage 0.995/同 8k/同 bpe）：Multi30K vs 浮士德——切分路径是语料频率的函数。"""
    if not os.path.exists(FAUST):                                # PG#2229 Goethe《浮士德·第一部》（公版）
        os.system(f'curl -sL --max-time 90 -o "{FAUST}.raw" '
                  '"https://www.gutenberg.org/cache/epub/2229/pg2229.txt"')
        raw = open(FAUST + ".raw", encoding="utf-8").read()
        m = re.search(r"\*\*\* START OF.*?\*\*\*(.*?)\*\*\* END OF", raw, re.S)
        open(FAUST, "w", encoding="utf-8").write(m.group(1) if m else raw)
    spm.SentencePieceTrainer.train(
        input=FAUST, model_prefix=os.path.join(OUT, "sp_faust_8k"), vocab_size=8000, model_type="bpe",
        character_coverage=0.995, pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        shuffle_input_sentence=False, hard_vocab_limit=False)
    sp = spm.SentencePieceProcessor(model_file=os.path.join(OUT, "sp_faust_8k.model"))
    return {"faust_bytes": os.path.getsize(FAUST),
            "Nähe": sp.encode("Nähe", out_type=str), "Büsche": sp.encode("Büsche", out_type=str)}

def exp_nonascii() -> dict:
    """非 ASCII 闭环：ASCII-only 20k 行训两套 sp（字符级 vs byte_fallback）+ tiktoken，三测试集对照。"""
    lines = open(M30K, encoding="utf-8").read().splitlines()
    ascii_lines = [l for l in lines if l.isascii()]
    train, ascii_test = ascii_lines[:20000], ascii_lines[20000:20200]
    de_test = [l for l in lines if not l.isascii()][:200]
    open(os.path.join(OUT, "train_ascii20k.txt"), "w", encoding="utf-8").write("\n".join(train))
    common = dict(input=os.path.join(OUT, "train_ascii20k.txt"), vocab_size=4000, model_type="bpe",
                  character_coverage=1.0, pad_id=0, unk_id=1, bos_id=2, eos_id=3,
                  shuffle_input_sentence=False, hard_vocab_limit=False)
    spm.SentencePieceTrainer.train(model_prefix=os.path.join(OUT, "sp_char"), byte_fallback=False, **common)
    spm.SentencePieceTrainer.train(model_prefix=os.path.join(OUT, "sp_byte"), byte_fallback=True, **common)
    sp_char = spm.SentencePieceProcessor(model_file=os.path.join(OUT, "sp_char.model"))
    sp_byte = spm.SentencePieceProcessor(model_file=os.path.join(OUT, "sp_byte.model"))
    tkt = tiktoken.get_encoding("gpt2")

    def tk_stats(texts):
        toks = sum(len(tkt.encode(t)) for t in texts)
        return {"unk_rate_piece": 0.0, "B_withspace": round(sum(len(t) for t in texts) / toks, 2)}

    def sp_stats_test(sp, texts):
        pieces = sum(len(sp.encode(t)) for t in texts)
        unk = sum(sum(1 for i in sp.encode(t) if i == sp.unk_id()) for t in texts)
        return {"unk_rate_piece": round(100 * unk / pieces, 2),
                "B_withspace": round(sum(len(t) for t in texts) / pieces, 2)}

    def pieces_shown(sp, text):
        """unk piece 显示为 <unk>：out_type=str 会渲染成未知段原文，易误读成『收录』。"""
        return [p if i != sp.unk_id() else "<unk>"
                for i, p in zip(sp.encode(text), sp.encode(text, out_type=str))]

    tests = {"ascii200": ascii_test, "german200": de_test, "cjk_probe": CJK_PROBES}
    table = {name: {"sp_char": sp_stats_test(sp_char, ts), "sp_byte": sp_stats_test(sp_byte, ts),
                    "tiktoken": tk_stats(ts)} for name, ts in tests.items()}
    return {"table": table,
            "footnote_paths": {"北京欢迎你，世界！": pieces_shown(sp_char, "北京欢迎你，世界！"),
                               "Nähe": pieces_shown(sp_char, "Nähe")}}

def exp_gpt2_vocab() -> dict:
    """GPT-2 官方词表产物程序化验证：256+50000+1 分解、首批合并、regex quirk、CJK 碎片化、一词三切。"""
    vocab = json.load(open(os.path.join(OPENAI_VOCAB, "encoder.json"), encoding="utf-8"))
    merges = [l for l in open(os.path.join(OPENAI_VOCAB, "vocab.bpe"), encoding="utf-8")
              .read().split("\n")[1:] if l.strip()]
    tkt = tiktoken.get_encoding("gpt2")
    import glob
    t5p = glob.glob(os.path.join(ROOT, "log", "huggingface", "hub", "models--t5-base", "snapshots", "*", "spiece.model"))
    t5 = spm.SentencePieceProcessor(model_file=t5p[0]) if t5p else None
    cjk = [{"text": s, "chars": len(s), "tokens": len(tkt.encode(s)),
            "tokens_per_char": round(len(tkt.encode(s)) / len(s), 2)}
           for s in ["北京", "北京清华大学"] + CJK_PROBES]
    return {"encoder_json_entries": len(vocab), "merge_lines": len(merges),
            "single_char_tokens": sum(1 for t in vocab if len(t) == 1), "first_merges": merges[:3],
            "the_id": tkt.encode("the"), "space_the_id": tkt.encode(" the"),
            "dont": [tkt.decode([i]) for i in tkt.encode("don't")],
            "DONT": [tkt.decode([i]) for i in tkt.encode("DON'T")], "cjk": cjk,
            "triple_cut": {"book1_8k": ["▁", "N", "ähe"],
                           "t5_base_unigram": t5.encode("Nähe", out_type=str) if t5 else None,
                           "gpt2_byte": [tkt.decode([i]) for i in tkt.encode("Nähe")]}}

def exp_tiktoken_vs_hf() -> dict:
    """B 级对拍：tiktoken gpt2 ≡ HF transformers GPT2Tokenizer——受控样例 + Multi30K 前 2000 行（含 id 全等）。"""
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tkt = tiktoken.get_encoding("gpt2")
    edge = ["don't stop", "DON'T STOP", "Nähe café 🚀", "北京 welcome", "a  double  spaces",
            "user@mail.com 2023-01-01", "trailing space ", "  leading", "3.14159", "Hello world", " Hello world"]
    lines = open(M30K, encoding="utf-8").read().splitlines()[:2000]
    n_id_mismatch = rt_fail = 0
    for s in edge + lines:
        a, b = tkt.encode(s), hf.encode(s)
        if a != b:
            n_id_mismatch += 1
        if tkt.decode(a) != hf.decode(b) or tkt.decode(a) != s:
            rt_fail += 1
    return {"edge_samples": len(edge), "corpus_lines": len(lines),
            "id_mismatch": n_id_mismatch, "roundtrip_fail": rt_fail}

def main():
    os.makedirs(OUT, exist_ok=True)
    report = {"meta": {"seed": SEED, "date": "2026-10", "machine": "CPU (M5 Pro)",
                       "corpus_note": "Multi30K CC BY-NC-SA，产物落 log/"}}
    print("[1/7] Sennrich 手算例 …", flush=True); report["toy"] = exp_toy()
    print("[2/7] 自写 mini-BPE …", flush=True)
    corpus, heldout = build_c_corpus()
    report["mini_bpe"] = exp_mini_bpe(corpus, heldout)
    print("[3/7] Book1 四欠条复算 …", flush=True); report["book1_recompute"] = exp_book1_recompute()
    print("[4/7] 两语料对照 …", flush=True); report["two_corpora"] = exp_two_corpora()
    print("[5/7] 非 ASCII 闭环 …", flush=True); report["nonascii"] = exp_nonascii()
    print("[6/7] GPT-2 词表分解 …", flush=True); report["gpt2_vocab"] = exp_gpt2_vocab()
    print("[7/7] tiktoken≡HF 对拍 …", flush=True); report["tiktoken_vs_hf"] = exp_tiktoken_vs_hf()
    out_path = os.path.join(OUT, "bpe_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("saved ->", out_path)

if __name__ == "__main__":
    main()
