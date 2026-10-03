# data.py —— Book2 ch11 数据管道正式化（11.1 节：OWT 子采样→分词器分离→两种块化→换算盒→WikiText-2 对照）
# 用途：把散在 00-feasibility/08_owt_subsample.py 与 ch08/data_prep.py、ch11/train_124m.py --prep-only
#   的三段管道收拢为一个可复算的入口，并产出 ch11 特有的三件教学件：
#   ①两种块化切法对比（GPT-2 拼接流切块 vs 文档边界对齐切块——零浪费换跨文档上下文）；
#   ②token↔word 换算盒（同一文档集上 sp-8k 与 gpt2 两套 n_t/n_w 实测）；
#   ③WikiText-2 word-level 对照（word 级 unigram/bigram 基线——换算盒的着陆坐标）。
# 所属章节：Book2 第 11 章 11.1 节。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch11/data.py"
#   （前置产物缺失时按提示先跑 08_owt_subsample.py / ch08/data_prep.py / ch11/train_124m.py --prep-only）
# 产物：log/book2-ch11/data_pipeline.json + figures/fig-11-1-data-pipeline.png
import json
import os
import time
from collections import Counter

import numpy as np

SEED = 20261002
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCALING_DIR = os.path.join(REPO_ROOT, "log", "book2-scaling")
CH08_DIR = os.path.join(REPO_ROOT, "log", "book2-ch08")
OUT_DIR = os.path.join(REPO_ROOT, "log", "book2-ch11")
FIG_DIR = os.path.join(REPO_ROOT, "figures")

DOCS_JSONL = os.path.join(SCALING_DIR, "owt_docs.jsonl")            # LM 主语料（1.15GB / 232,920 篇）
TOK_TXT = os.path.join(SCALING_DIR, "owt_tokenizer_corpus.txt")     # 分词器分片（独立，29MB / 6000 篇）
SP8K_MODEL = os.path.join(CH08_DIR, "sp8k_owt.model")
TOKENS50K = os.path.join(OUT_DIR, "tokens50k.bin")                  # 50257 词表流（train_124m.py --prep-only 产物）
BLOCK = 256                                                          # 训练窗长 s（全书统一）
WT2_MAX_WORDS = 2_000_000                                            # WikiText-2 统计上限（语料本身 ~2M word 级）


def need(path, hint):
    if not os.path.exists(path):
        raise SystemExit(f"[前置缺失] {path}\n  先跑：{hint}")


# ---------------- ①语料与分片分离的核账 ----------------
def step_corpus():
    """OWT 子采样件核账 + 分词器分片分离声明（写作纪律：分离写进运行元数据）。"""
    need(DOCS_JSONL, "python code/00-feasibility/08_owt_subsample.py --target-mb 1150")
    n_docs, n_bytes = 0, 0
    with open(DOCS_JSONL, "rb") as f:
        for _ in f:
            n_docs += 1
    n_bytes = os.path.getsize(DOCS_JSONL)
    return {"lm_corpus": {"path": DOCS_JSONL, "docs": n_docs, "mb": round(n_bytes / 1024 / 1024, 1),
                          "license": "OpenWebText 打包层 CC0（HF 卡片 cc0-1.0）；原文各自版权。"
                                     "WebText 原版=Reddit karma>=3 外链（GPT-2 论文 §2.1）；OpenWebText 的 karma 管道"
                                     "说法出自其 GitHub 项目自述（第三方复刻），HF 卡片未载明 karma 细节"},
            "tokenizer_shard": {"path": TOK_TXT, "mb": round(os.path.getsize(TOK_TXT) / 1024 / 1024, 1),
                                "separation": "分词器只在独立分片（前 6000 篇）训练，与 LM 语料（自第 6001 篇起）互斥——"
                                              "写进运行元数据的纪律（ch08 冒烟档教训的正式修正）"}}


# ---------------- ②两种块化切法对比（教学件） ----------------
def step_chunking(n_docs_demo=2000, block=BLOCK):
    """同一批文档、两种切成 (B,256) 训练窗的方法对比。

    法 A（GPT-2 拼接流，全书与官方同口径）：文档间插 EOT 后扁平拼接成一条流，按 256 定长切窗——
      窗可以跨文档边界（EOT 就是官方教的「文档到此为止」信号），零填充零丢弃。
    法 B（文档边界对齐）：每篇独立切窗、末窗不足 256 丢弃——窗内纯单文档，代价是短文档尾料浪费。
    返回两种方法的窗口数/利用率/跨文档窗占比/示例窗。
    """
    need(SP8K_MODEL, "python \"code/ch08/data_prep.py\"")
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(SP8K_MODEL)
    eos = sp.eos_id()
    lens, first_doc_ids = [], None
    with open(DOCS_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_docs_demo:
                break
            ids = sp.encode(json.loads(line)["text"], out_type=int, add_eos=True)  # +EOS(id=2) 文档分隔
            lens.append(len(ids))
            if first_doc_ids is None:
                first_doc_ids = ids
    total_tok = sum(lens)
    # 法 A：拼接流定长切窗
    n_win_a = total_tok // block
    cross, pos = 0, 0
    for L in lens:
        end = pos + L
        if pos // block != (end - 1) // block and end <= n_win_a * block:
            cross += 1                      # 该文档的结束边界落进某个窗内 => 该窗跨文档
        pos = end
    # 法 B：逐篇切窗丢尾
    n_win_b = sum(L // block for L in lens)
    used_b = n_win_b * block
    # 示例：拼接流里第一个文档交界（doc1 末尾 + EOS + doc2 开头）——EOT 教边界的实物
    with open(DOCS_JSONL, "r", encoding="utf-8") as f:
        f.readline()
        d2 = sp.encode(json.loads(f.readline())["text"], out_type=int)[:6]  # 下一篇开头 6 token
    junction = first_doc_ids[-8:] + d2                                     # 末 8 token（含 EOS）+ 下一篇开头
    return {
        "demo_docs": n_docs_demo, "block": block, "total_tokens": total_tok,
        "A_concat_stream": {"windows": n_win_a, "utilization": 1.0,
                            "cross_doc_windows": cross, "cross_frac": round(cross / n_win_a, 3),
                            "note": "GPT-2 式拼接流：文档间插 EOS/EOT 后按 256 定长切窗；窗跨文档边界是特性不是 bug——"
                                    "官方预训练同口径，模型顺带学会 EOT 前后该断则断"},
        "B_per_doc_aligned": {"windows": n_win_b, "utilization": round(used_b / total_tok, 4),
                              "discarded_tokens": total_tok - used_b,
                              "note": "文档边界对齐：窗内纯单文档、无 EOT 学习压力；代价=每篇尾料 <256 丢弃"},
        "junction_decoded": sp.decode(junction),
        "junction_ids": junction,
        "junction_note": "第一个文档交界处：...doc1 末尾 [EOS=2] doc2 开头...——拼接流里模型「看见」的边界就长这样",
    }


# ---------------- ③token↔word 换算盒（同一文档集、两套词表） ----------------
def step_conversion(n_docs=56000):
    """换算盒实测：同一批文档（前 56,000 篇完整文档）上 gpt2(50257) 与 sp-8k 的 n_t/n_w 与 B/token。
    注意口径：data50k_meta 的 total_tokens=60,000,000 是截断后的训练流，其逐篇累计统计覆盖的
    是 56,000 篇完整文档（含截断尾段，gpt2 全量 ~63.3M）——本函数在前 56,000 行上直接重数三种单位，
    保证三个比值同一分母。"""
    need(TOKENS50K, "python \"code/ch11/train_124m.py\" --prep-only")
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    n_words, n_bytes, n_gpt = 0, 0, 0
    with open(DOCS_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_docs:
                break
            text = json.loads(line)["text"]
            n_words += len(text.split())
            n_bytes += len(text.encode("utf-8"))
            n_gpt += len(enc.encode(text, disallowed_special=())) + 1     # +EOT（与训练流拼接同口径）
    with open(os.path.join(OUT_DIR, "data50k_meta.json"), encoding="utf-8") as f:
        meta50k = json.load(f)
    sp_tok = meta50k["encoding"]["sp8k_tokens_same_docs"]             # 73,598,712（train_124m 同一批文档）
    return {
        "docs": n_docs, "bytes": n_bytes, "words": n_words,
        "gpt2_50257": {"tokens": n_gpt, "tokens_per_word": round(n_gpt / n_words, 4),
                       "bytes_per_token": round(n_bytes / n_gpt, 3)},
        "sp8k_8192": {"tokens": sp_tok, "tokens_per_word": round(sp_tok / n_words, 4),
                      "bytes_per_token": round(n_bytes / sp_tok, 3)},
        "stream_note": "训练流取 gpt2 拼接流前 60,000,000 token（约前 53,000 篇）；本表比值按 56,000 篇完整文档计（同一分母）",
        "formula": "ln PPL_word = (n_t/n_w) * ln PPL_token（token 模型的损失换算到社区 word 级坐标系）",
        "ch08_ref": "ch08 sp-8k 全语料（232,920 篇）B/token=3.82、2000 篇样本 n_t/n_w=1.5887",
    }


# ---------------- ④WikiText-2 word-level 对照基线 ----------------
def step_wikitext2():
    """WikiText-2（word-level 传统基准语料）上的三件数：
    (a) word 级 unigram / bigram PPL 基线（教学级实现：train 频次 + <unk> 回退 + 计数插值平滑，
        与社区 word 级协议同型：valid OOV 记 <unk>）；
    (b) 同一 valid 切片上 sp-8k 与 gpt2 两种分词器的 n_t/n_w——换算盒的系数在本语料上的取值；
    (c) 演示：若某 token 模型在 WT2 valid 上达 L nats/token，其 word PPL 读数=(b) 的系数换算。"""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")     # 许可：CC BY-SA（卡片文本 4.0 与
    #   metadata 3.0+GFDL 双标不一致——保守口径：落 log/ 不分发、引用时注明二者）

    def to_words(split):
        out = []
        for t in split["text"]:
            out.extend(t.split())                                     # 行内按空白切（保留前导空格切出的词）
        return out

    tr, va = to_words(ds["train"]), to_words(ds["validation"])
    uni = Counter(tr)
    bi = Counter(zip(tr, tr[1:]))
    n_tr, n_va = len(tr), len(va)
    unk_hits = sum(1 for w in va if w not in uni)
    p_uni = {w: c / n_tr for w, c in uni.items()}
    p_unk = 1.0 / n_tr                                                # <unk> 概率（训练集单例折算，教学级）

    def get_p1(w):
        return p_uni.get(w, p_unk)                                    # OOV -> <unk>

    ll_uni = sum(-np.log(get_p1(w)) for w in va)
    gamma = 5.0                                                       # 计数插值：P(w2|w1)=(c12+γ·P1(w2))/(c1+γ)
    ll_bi = 0.0
    for w1, w2 in zip(va, va[1:]):
        p1 = get_p1(w2)
        c1 = uni.get(w1, 0)
        p2 = (bi.get((w1, w2), 0) + gamma * p1) / (c1 + gamma) if c1 else p1
        ll_bi += -np.log(p2)
    # 两种分词器在本语料 valid 上的 n_t/n_w
    import sentencepiece as spm
    import tiktoken
    sp = spm.SentencePieceProcessor()
    sp.Load(SP8K_MODEL)
    enc = tiktoken.get_encoding("gpt2")
    va_text = "".join(t for t in ds["validation"]["text"])            # 原始文本流（不重组，避免引入口径差）
    n_sp = len(sp.encode(va_text, out_type=int))
    n_gpt = len(enc.encode(va_text))
    return {"valid_words": n_va, "train_words": n_tr, "valid_oov_rate": round(unk_hits / n_va, 4),
            "word_ppl_unigram": round(float(np.exp(ll_uni / n_va)), 1),
            "word_ppl_bigram_interp": round(float(np.exp(ll_bi / (n_va - 1))), 1),
            "note_baseline": "教学级 word 级基线（<unk> 回退 + γ=5 计数插值，非 SOTA）——给换算盒一个 word 级着陆坐标；"
                             "社区 word 级 WT2 强基线在数十量级（LSTM 传统，第三方文献口径，本书不展开）",
            "nt_nw_wt2": {"sp8k": round(n_sp / n_va, 4), "gpt2": round(n_gpt / n_va, 4)},
            "demo": f"124M fast 档若在 WT2 上取 L nats/token，word PPL = exp({n_gpt / n_va:.4f}*L)（gpt2 在 WT2 的系数）"}


# ---------------- 图 11.1：管道流 ----------------
def make_figure(rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})
    BLUE, ORANGE, TEAL, YELLOW, GRID, TICK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e1e0d9", "#898781"
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.set_xlim(0, 16); ax.set_ylim(0, 8); ax.axis("off")

    def box(x, y, w, h, title, body, color=BLUE):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="white", edgecolor=color, lw=1.4))
        ax.text(x + w / 2, y + h - 0.52, title, ha="center", va="center", fontsize=8,
                color="#0b0b0b", weight="bold")
        ax.text(x + w / 2, y + (h - 0.52) / 2, body, ha="center", va="center", fontsize=6.6, color="#52514e")

    def arrow(x1, y1, x2, y2, color=BLUE):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2))

    # 上层：语料与分词器（分离纪律）
    box(0.3, 5.4, 4.4, 2.2, "OWT stream subsample",
        "232,920 docs / 1.15 GB\n(CC0 packaging)\nshard 0-6k  |  shard 6k+")
    box(5.6, 5.4, 4.6, 2.2, "tokenizer (separate!)", "sp-8k trained on shard 0-6k ONLY\n(29 MB)  ->  sp8k_owt.model", ORANGE)
    box(11.1, 5.4, 4.6, 2.2, "gpt2 50257 (given)",
        "tiktoken gpt2, not trained here\n(ch2 parity: tiktoken == HF)", TEAL)
    arrow(4.0, 5.4, 6.6, 3.6, ORANGE)
    arrow(13.4, 5.4, 13.9, 3.6, TEAL)
    # 下层：两条编码流
    box(3.1, 1.4, 5.0, 2.2, "sp-8k stream (ch08/ch11-iso)",
        "tokens.bin (306M,) uint16\nB/token=3.78  tok/word=1.600\n(same 56k docs basis)", ORANGE)
    box(10.4, 1.4, 5.0, 2.2, "gpt2 stream (124M fast)",
        "tokens50k.bin (60M,) uint16\nB/token=4.398  tok/word=1.376", TEAL)
    arrow(7.9, 4.15, 7.9, 3.6, ORANGE)
    arrow(12.4, 4.15, 12.4, 3.6, TEAL)
    # 块化注记（虚线框）
    box(0.4, 0.2, 2.4, 2.4, "chunking A/B",
        "A concat-stream: (32,256)\nzero waste, cross-doc\nB per-doc aligned:\npad-free, tail loss", YELLOW)
    ax.text(8.1, 0.8, "eval pool = stream tail (mutually exclusive, 401,408 tok/eval >= 400k protocol)",
            fontsize=6.6, color="#52514e")
    ax.set_title("ch11 data pipeline: one doc stream, two tokenizers, separate shards",
                 fontsize=9, color="#0b0b0b")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-11-1-data-pipeline.png")
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"[figure] -> {out}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.perf_counter()
    rep = {"meta": {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED,
                    "note": "数据管道正式化：本脚本只做核账与教学件，重活由 08_owt_subsample / ch08/data_prep / "
                            "ch11/train_124m --prep-only 完成（断点产物已存在则复用）"}}
    print("===== ① 语料与分片分离核账 =====", flush=True)
    rep["corpus"] = step_corpus()
    print(json.dumps(rep["corpus"]["lm_corpus"], ensure_ascii=False, indent=1))
    print("===== ② 两种块化切法对比 =====", flush=True)
    rep["chunking"] = step_chunking()
    print(json.dumps({k: v for k, v in rep["chunking"].items()
                      if k not in ("junction_ids", "junction_decoded")}, ensure_ascii=False, indent=1))
    print("  junction:", rep["chunking"]["junction_ids"], "->", repr(rep["chunking"]["junction_decoded"]))
    print("===== ③ token<->word 换算盒（同一文档集两套词表） =====", flush=True)
    rep["conversion"] = step_conversion()
    print(json.dumps(rep["conversion"], ensure_ascii=False, indent=1))
    print("===== ④ WikiText-2 word-level 对照 =====", flush=True)
    rep["wikitext2"] = step_wikitext2()
    print(json.dumps(rep["wikitext2"], ensure_ascii=False, indent=1))
    rep["meta"]["wallclock_sec"] = round(time.perf_counter() - t0, 1)
    out_json = os.path.join(OUT_DIR, "data_pipeline.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"[完成] -> {out_json}")
    make_figure(rep)


if __name__ == "__main__":
    main()
