# -*- coding: utf-8 -*-
# 用途：Book2 第 2 章「BPE 从零讲透」主脚本：Sennrich 手算例复跑、自写 mini-BPE（pair 倒排索引）、
#       Book1 F7 欠条四数字复算钉死、两语料对照（Multi30K vs Gutenberg 德文）、非 ASCII unk 闭环、
#       GPT-2 词表 50257 分解、tiktoken≡HF B 级对拍；全部数字落 log/book2-ch02/bpe_report.json
# 所属章节：《预训练革命》第 2 章（ch02-分词器）
# 运行方式：python "code/ch02/bpe.py" --toy-only；去掉参数运行完整实验（需先准备原料）。
# BOOK2_LOG_ROOT/BOOK1_LOG_ROOT可指定外层缓存根；新结果不保证复现历史时间或旧有重叠切片曲线。
import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import time
from collections import Counter, defaultdict

SEED = 20261003
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_ROOT = os.path.abspath(os.environ.get("BOOK2_LOG_ROOT", os.path.join(ROOT, "log")))
OUT = os.path.join(LOG_ROOT, "book2-ch02")
BOOK1_LOG_ROOT = os.path.abspath(os.environ.get("BOOK1_LOG_ROOT", LOG_ROOT))
M30K = os.path.join(BOOK1_LOG_ROOT, "Book1-ch09", "multi30k_corpus.txt")
BOOK1_MODEL = os.path.join(BOOK1_LOG_ROOT, "Book1-ch09", "bpe_8k.model")
SHAKESPEARE = os.path.join(LOG_ROOT, "book2-feasibility", "tiny_shakespeare.txt")
FAUST = os.path.join(OUT, "gutenberg_de.txt")                           # Goethe《浮士德》(PG#2229，公版)
OPENAI_VOCAB = os.path.join(LOG_ROOT, "Book2-ch02-research", "openai-gpt2")
CJK_PROBES = ["北京欢迎你，世界！", "距离火车站很近的新办公室让同事们感到意外地方便。", "人工智能正在改变世界。"]

# ---------------- 1) Sennrich Algorithm 1 思路复刻（本书Python 3的字典插入序决胜） ----------------
def sennrich_toy(word_freq: dict, n_merges: int) -> list:
    """论文 Algorithm 1 最小复刻。输入 {符号串: 词频}（词内空格分符号），输出前 n_merges 条合并 [(a,b),...]。
    形状语义：'l o w </w>'(4 符号) --合并--> 'lo w </w>'(3 符号)；max() 平票取当前Python字典遍历序中的首个；这是本实现的确定规则。"""
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

# ---------------- 2) 自写 mini-BPE：pair倒排索引版（fit/encode/单词拼接还原/压缩率三口径） ----------------
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
        self.merges = []  # fit建立新模型；重复调用不能残留旧语料规则。
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
            if pair_counts[best] < 2:                          # 教学停止规则：不为只出现一次的对新增规则；并非它完全没有压缩收益。
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

    def vocabulary(self) -> set:
        """保留基础字符与所有合并产物；不同规则可能生成同一字符串。"""
        return self.base_vocab | {a + b for a, b in self.merges}

    def vocab_size(self) -> int:
        return len(self.vocabulary())

    def encode_word(self, word: str) -> list:
        """测试时切词：字符起步，按学到的合并表依序合并（贪心按规则序）；这里只返回字符串，可能含训练未见字符，不等于固定ID词表已覆盖。"""
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
        """压缩率三口径：A=去全部正则空白后的码点/piece（与sp_stats仅去普通空格不同）；B=全字符含空格/piece；C=UTF-8 字节/piece。"""
        pieces = self.encode(text)
        n = len(pieces)
        chars_ns = sum(len(w) for w in re.findall(r"\S+", text))
        if n == 0:
            raise ValueError("没有可编码的非空白内容，压缩率无定义")
        return {"pieces": n, "A_nospace": round(chars_ns / n, 3),
                "B_withspace": round(len(text) / n, 3),
                "C_bytes": round(len(text.encode("utf-8")) / n, 3)}

def build_c_corpus() -> tuple:
    """训练前部至多1Mi个码点，加非ASCII样句×40；末100K码点独立留出，不复制样句。"""
    with open(SHAKESPEARE, encoding="utf-8") as f:
        full = f.read()
    samples = ["Die Nähe des neuen Büros war für die Mitarbeiter überraschend praktisch.",
               "Le café très près de la gare est fermé le lundi.",
               "距离火车站很近的新办公室让同事们感到意外地方便。",
               "Emoji sanity check: 🚀🤖 fits in a byte-free world? …",
               "Mixed Nähe café 距离 line with digits 12345 and punctuation!!"]
    if len(full) <= 100_000:
        raise ValueError("教学语料须多于100000个码点，才能另留末尾切片")
    split = min(1024 * 1024, len(full) - 100_000)
    # 训练区间与保留区间不重叠；扩展样句只加在训练侧。
    return full[:split] + ("\n".join(samples) + "\n") * 40, full[-100_000:]

# ---------------- 3) 各实验 ----------------
def exp_toy() -> dict:
    """手算例：论文代码词典与图注词典各跑一遍（两者不一致 -> 正文考据框）。"""
    code_dict = {"l o w </w>": 5, "l o w e r </w>": 2, "n e w e s t </w>": 6, "w i d e s t </w>": 3}
    cap_dict = {"l o w </w>": 5, "l o w e s t </w>": 2, "n e w e r </w>": 6, "w i d e r </w>": 3}
    m_code, m_cap = sennrich_toy(code_dict, 10), sennrich_toy(cap_dict, 10)
    symbols = list("lower") + ["</w>"]
    for a, b in m_cap:
        new, i = [], 0
        while i < len(symbols):
            if i + 1 < len(symbols) and symbols[i:i + 2] == [a, b]:
                new.append(a + b)
                i += 2
            else:
                new.append(symbols[i])
                i += 1
        symbols = new
    return {"merges_code_dict": [" ".join(p) for p in m_code],
            "merges_caption_dict": [" ".join(p) for p in m_cap],
            "oov_lower_path": symbols}

def exp_mini_bpe(corpus, heldout) -> dict:
    """自写 mini-BPE 主实验：fit 1000 合并 + round-trip 无损 + 压缩率 + 词表扫描（供图 2.2）。"""
    bpe = MiniBPE(1000)
    fit_sec = bpe.fit(corpus)
    words_h = re.findall(r"\S+", heldout)
    tested_words = words_h[::max(1, len(words_h) // 200)][:200]
    rt_ok = all("".join(bpe.encode_word(w)) == w for w in tested_words)
    uncovered = sorted({piece for w in tested_words for piece in bpe.encode_word(w)
                        if piece not in bpe.vocabulary()})
    sweep = []
    for k in [0, 200, 500, 1000, 2000, 4000]:
        m = MiniBPE(k)
        t = m.fit(corpus)
        rec = {"n_merges_requested": k, "n_merges": len(m.merges), "vocab": m.vocab_size(), "fit_sec": t,
               "train": m.compression(corpus[:400_000])["A_nospace"],
               "heldout": m.compression(heldout)["A_nospace"]}
        sweep.append(rec)
        print(f"  sweep merges={k:4d} vocab={rec['vocab']:5d} train={rec['train']:.3f} held={rec['heldout']:.3f} ({t}s)", flush=True)
    return {"n_merges_requested": 1000, "n_merges": len(bpe.merges), "fit_sec": fit_sec, "vocab_size": bpe.vocab_size(),
            "roundtrip_words_tested": len(tested_words), "uncovered_pieces_in_tested_words": uncovered,
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
    if n_pieces == 0 or n_words == 0:
        raise ValueError("统计分母为零：需要有有效piece及空白分词单元的文本")
    return {"lines": len(lines), "pieces": n_pieces, "unk_pieces": n_unk,
            "A_definition": "原文仅移除U+0020普通空格；不等于移除全部Unicode空白",
            "unk_rate_piece": round(100 * n_unk / n_pieces, 2), "words": n_words,
            "A_nospace": round(n_chars / n_pieces, 3), "B_withspace": round((n_chars + n_spaces) / n_pieces, 3),
            "C_bytes": round(n_bytes / n_pieces, 3), "pieces_per_word": round(n_pieces / n_words, 3),
            "chars_per_word": round(n_chars / n_words, 2)}

def exp_book1_recompute() -> dict:
    """第二幕核心：Book1 bpe_8k.model 在同源 Multi30K 上复算四欠条数字 + 同规格重训计时与确定性校验。"""
    import sentencepiece as spm
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
            "equality_scope": "只比较按ID排列的piece字符串，不证明整个proto或全部行为相同",
            "original_model_sha256": hashlib.sha256(open(BOOK1_MODEL, "rb").read()).hexdigest(),
            "retrained_model_sha256": hashlib.sha256(open(os.path.join(OUT, "bpe_8k_reverify.model"), "rb").read()).hexdigest(),
            "paths": {"Nähe": sp.encode("Nähe", out_type=str), "Büsche": sp.encode("Büsche", out_type=str)},
            "vocab_size": sp.get_piece_size()}

def exp_two_corpora() -> dict:
    """两语料对照（同 coverage 0.995/同 8k/同 bpe）：Multi30K vs 浮士德——切分路径是语料频率的函数。"""
    import sentencepiece as spm
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
    import sentencepiece as spm
    import tiktoken
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
            "selection_note": "german200是旧键名，实际按非ASCII筛选，未作语言识别；两套4000词表的字节预留预算不同",
            "footnote_paths": {"北京欢迎你，世界！": pieces_shown(sp_char, "北京欢迎你，世界！"),
                               "Nähe": pieces_shown(sp_char, "Nähe")}}

def exp_gpt2_vocab() -> dict:
    """GPT-2 官方词表产物程序化验证：256+50000+1 分解、首批合并、regex quirk、CJK 碎片化、一词三切。"""
    import sentencepiece as spm
    import tiktoken
    vocab = json.load(open(os.path.join(OPENAI_VOCAB, "encoder.json"), encoding="utf-8"))
    merges = [l for l in open(os.path.join(OPENAI_VOCAB, "vocab.bpe"), encoding="utf-8")
              .read().split("\n")[1:] if l.strip()]
    tkt = tiktoken.get_encoding("gpt2")
    import glob
    hf_hub = os.environ.get("HF_HUB_CACHE", os.path.join(LOG_ROOT, "huggingface", "hub"))
    t5p = sorted(glob.glob(os.path.join(hf_hub, "models--google-t5--t5-base", "snapshots", "*", "spiece.model"))
                 + glob.glob(os.path.join(hf_hub, "models--t5-base", "snapshots", "*", "spiece.model")))
    t5 = spm.SentencePieceProcessor(model_file=t5p[0]) if t5p else None
    original_sp = spm.SentencePieceProcessor(model_file=BOOK1_MODEL) if os.path.exists(BOOK1_MODEL) else None
    cjk = [{"text": s, "chars": len(s), "tokens": len(tkt.encode(s)),
            "tokens_per_char": round(len(tkt.encode(s)) / len(s), 2)}
           for s in ["北京", "北京清华大学"] + CJK_PROBES]
    return {"encoder_json_entries": len(vocab), "merge_lines": len(merges),
            "single_char_tokens": sum(1 for t in vocab if len(t) == 1), "first_merges": merges[:3],
            "the_id": tkt.encode("the"), "space_the_id": tkt.encode(" the"),
            "dont": [tkt.decode([i]) for i in tkt.encode("don't")],
            "DONT": [tkt.decode([i]) for i in tkt.encode("DON'T")], "cjk": cjk,
            "triple_cut": {"book1_8k": original_sp.encode("Nähe", out_type=str) if original_sp else None,
                           "t5_base_unigram": t5.encode("Nähe", out_type=str) if t5 else None,
                           "gpt2_byte": [tkt.decode([i]) for i in tkt.encode("Nähe")]}}

def exp_tiktoken_vs_hf() -> dict:
    """B 级对拍：tiktoken gpt2 ≡ HF transformers GPT2Tokenizer——受控样例 + Multi30K 前 2000 行（含 id 全等）。"""
    import tiktoken
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
            "hf_tokenizer_class": type(hf).__name__, "hf_is_fast": getattr(hf, "is_fast", None),
            "tiktoken_version": getattr(tiktoken, "__version__", None),
            "id_mismatch": n_id_mismatch, "roundtrip_fail": rt_fail}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy-only", action="store_true", help="仅运行手算词典，不下载或训练历史语料")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    report = {"meta": {"seed_label": SEED, "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       "machine": platform.platform(), "python": platform.python_version(),
                       "protocol": "toy-only" if args.toy_only else "disjoint-text-slices",
                       "corpus_note": "运行结果归本次环境；历史数字不作为逐值验收线"}}
    if args.toy_only:
        report["toy"] = exp_toy()
        path = os.path.join(OUT, "bpe_toy_report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    required = [M30K, BOOK1_MODEL, SHAKESPEARE,
                os.path.join(OPENAI_VOCAB, "encoder.json"), os.path.join(OPENAI_VOCAB, "vocab.bpe")]
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("完整实验缺少原料；可先用--toy-only：" + "; ".join(missing))
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
