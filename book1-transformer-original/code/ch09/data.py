# -*- coding: utf-8 -*-
# 用途：Book1 第 9 章——Multi30K 数据管道：加载、shared BPE-8k 自训、编码、批处理与动态填充
# 所属章节：《Transformer 原典》第 9 章（从零实现与小型翻译实验）
# 运行：source env.sh && python "code/ch09/data.py"（冒烟自测，MPS/CPU 均可，约 1 分钟）
# 合规：Multi30K 为 CC BY-NC-SA——语料、BPE 模型等一切产物落 log/Book1-ch09/，不入库。
#
# 工具链借用框（算法留给 Book2）：sentencepiece 只用三个 API——
#   train(input=语料文件) → .model 文件；sp.encode(句子) → piece 序列；sp.id_to_piece / sp.decode。
import os
import random

import torch
from datasets import load_dataset
import sentencepiece as spm

LOG_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "log", "Book1-ch09"))

# ---- 全局口径（ch9 参考配置；与 feasibility notes/01 实测对齐） ----
VOCAB_SIZE = 8000        # shared BPE-8k：en+de 合并训练（探针实测训练 0.2 s，unk 率 ~1.9%）
BATCH = 64               # 探针实测该档 MPS 43.5 ms/step
MAX_TRAIN_PIECES = 16    # 训练句长上限 L（BPE piece 数，双侧同限）——外推实验以它的 1.5~2 倍为靶
EXTRAP_BAND = (24, 32)   # = [1.5L, 2L]：外推评测集的 piece 长度带（见 eval.py）
DECODE_MAX_PIECES = 64   # 解码硬上限（正弦 PE 可即时生成任意长度，这里取数据天然上界之上）

# 特殊 token：sentencepiece 训练时显式指定 pad/unk/bos/eos 的 id，全书统一按此引用
PAD, UNK, SOS, EOS = 0, 1, 2, 3


def ensure_bpe(log_dir: str = LOG_DIR, vocab_size: int = VOCAB_SIZE) -> spm.SentencePieceProcessor:
    """自训（或复用已训）shared BPE。en+de 交替写入同一语料 → 单一词表（原论文 shared vocab 的缩小版）。
    语料与模型文件都落 log/，重跑时若模型已存在则直接加载（确定性：固定随机种子 + 不打乱输入行）。"""
    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(log_dir, "bpe_8k.model")
    if not os.path.exists(model_path):
        tr = load_dataset("bentrevett/multi30k")["train"]
        corpus_path = os.path.join(log_dir, "multi30k_corpus.txt")
        with open(corpus_path, "w", encoding="utf-8") as f:
            for ex in tr:
                f.write(ex["en"].replace("\n", " ").strip() + "\n")
                f.write(ex["de"].replace("\n", " ").strip() + "\n")
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=os.path.join(log_dir, "bpe_8k"),
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=0.995,     # de 变音符：覆盖率 0.995 避免词表被单字符占满
            pad_id=PAD, unk_id=UNK, bos_id=SOS, eos_id=EOS,
            shuffle_input_sentence=False,  # 固定语料顺序 → 词表可复现
            hard_vocab_limit=False,
        )
    sp = spm.SentencePieceProcessor(model_file=model_path)
    assert (sp.pad_id(), sp.unk_id(), sp.bos_id(), sp.eos_id()) == (PAD, UNK, SOS, EOS)
    return sp


def load_raw() -> dict:
    """Multi30K（bentrevett 镜像，jsonl 原生无加载脚本）：train 29000 / validation 1014 / test 1000 句对。"""
    ds = load_dataset("bentrevett/multi30k")
    return {split: [(ex["en"], ex["de"]) for ex in ds[split]] for split in ds}


def encode_sent(sp: spm.SentencePieceProcessor, sent: str, cap: int = DECODE_MAX_PIECES) -> list:
    """str → [SOS] + pieces + [EOS] 的 id 列表（总长 ≤ cap+1；内容截到 cap-2 个 piece——
    训练侧 cap=MAX_TRAIN_PIECES 起过滤作用，正常不触发）。"""
    ids = sp.encode(sent, out_type=int)[: cap - 2]
    return [SOS] + ids + [EOS]


def encode_pairs(sp, pairs, cap: int = DECODE_MAX_PIECES):
    return [(encode_sent(sp, en, cap), encode_sent(sp, de, cap)) for en, de in pairs]


def filter_pairs(sp, pairs, max_pieces: int = MAX_TRAIN_PIECES):
    """训练长度上限 L：只保留双侧内容 piece 数 ≤ L 的句对（外推实验的「训练 max_len」即此 L）。
    返回 (过滤后句对, 保留率)。"""
    kept = [(en, de) for en, de in pairs
            if len(sp.encode(en, out_type=int)) <= max_pieces
            and len(sp.encode(de, out_type=int)) <= max_pieces]
    return kept, len(kept) / len(pairs)


def batches(id_pairs, batch_size: int = BATCH, shuffle: bool = True, seed: int = 2017, epoch: int = 0):
    """动态填充批生成器：每批 yield (src, tgt)，均为 (B, m) 的 long 张量——
    m = 批内两侧最长的 id 序列长（含 SOS/EOS；训练侧 m ≤ L+2 = 18），批与批不同。
    训练循环随后从 tgt (B,m) 错位切出 tgt[:, :-1] 与 tgt[:, 1:] 各 (B, m-1)（教师强制，见 train.py）。
    探针踩坑实录（ch9 9.2 节写作素材）：
      ① 变长句拼张量必须先在 list 层补 pad 再 torch.tensor（先 tensor 化再 pad 会触发设备拷贝报错）；
      ② tgt[:, 1:] 等切片非连续张量，损失内部展平须用 .reshape 而非 .view。"""
    idx = list(range(len(id_pairs)))
    if shuffle:
        random.Random(seed + epoch).shuffle(idx)  # 逐 epoch 换洗牌种子，可复现
    for s in range(0, len(idx), batch_size):
        chunk = [id_pairs[i] for i in idx[s:s + batch_size]]
        m = max(max(len(a) for a, _ in chunk), max(len(b) for _, b in chunk))  # 批内最长句（含 sos/eos）

        def pad(rows):
            return torch.tensor([r + [PAD] * (m - len(r)) for r in rows], dtype=torch.long)  # list 层补齐 -> (B, m)
        src, tgt = zip(*chunk)
        yield pad(list(src)), pad(list(tgt))


def ids_to_text(sp, ids) -> str:
    """piece id 序列（list[int]，可含 PAD/SOS 位与批填充）→ 可读文本：截到首个 EOS、剔除特殊位，▁ 还原为空格。"""
    out = []
    for i in ids:
        if i == EOS:
            break
        if i in (PAD, SOS):
            continue
        out.append(sp.id_to_piece(int(i)))
    return "".join(p.replace("▁", " ") for p in out).strip()


def piece_len_hist(sp, pairs, cuts=(8, 12, 16, 20, 24, 32)):
    """双侧内容 piece 长度分布（供决定/核验 MAX_TRAIN_PIECES 用）。"""
    ls = [max(len(sp.encode(en, out_type=int)), len(sp.encode(de, out_type=int))) for en, de in pairs]
    rows = []
    lo = 0
    for c in list(cuts) + [10 ** 9]:
        rows.append((lo, c, sum(1 for l in ls if lo < l <= c)))
        lo = c
    return rows, max(ls)


def main():  # 冒烟自测：加载→自训 BPE→统计→过滤→出一批
    import time
    t0 = time.perf_counter()
    sp = ensure_bpe()
    raw = load_raw()
    print(f"[数据] Multi30K 句对数 train/val/test = {len(raw['train'])}/{len(raw['validation'])}/{len(raw['test'])}"
          f"（BPE 词表 {sp.get_piece_size()}，加载+分词 {time.perf_counter() - t0:.1f} s）")

    demo = raw["train"][0]
    print(f"[分词] {demo[0]!r} -> {sp.encode(demo[0], out_type=str)}")
    print(f"[分词] {demo[1]!r} -> {sp.encode(demo[1], out_type=str)}")

    rows, mx = piece_len_hist(sp, raw["train"])
    print(f"[长度] train 双侧 piece 长度分布（max={mx}）：")
    for lo, hi, n in rows:
        print(f"    ({lo:>2},{hi if hi < 10**9 else 'inf':>3}]  {n:>6}  {n / len(raw['train']):6.1%}")

    kept, keep_rate = filter_pairs(sp, raw["train"])
    print(f"[过滤] L={MAX_TRAIN_PIECES}：保留 {len(kept)}/{len(raw['train'])}（{keep_rate:.1%}）")

    id_pairs = encode_pairs(sp, kept[:128])
    it = batches(id_pairs, shuffle=True, seed=2017)
    src, tgt = next(it)
    print(f"[批次] src {tuple(src.shape)} tgt {tuple(tgt.shape)}  pad 占比 "
          f"{(src == PAD).float().mean():.1%}/{(tgt == PAD).float().mean():.1%}（动态填充生效）")
    back = ids_to_text(sp, tgt[0].tolist())
    print(f"[回读] {back!r}")


if __name__ == "__main__":
    main()
