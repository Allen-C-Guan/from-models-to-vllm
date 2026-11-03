# -*- coding: utf-8 -*-
# 用途：Book1 第 9 章——评测：sacreBLEU(13a) + 手写朴素 BLEU 对拍自证 + 按参考句长分桶 + 外推实测 + 三方对照 + 组批对照
# 所属章节：《Transformer 原典》第 9 章 9.6；BLEU 公式出处 Papineni 2002（速览见 ch1）
# 运行：
#   source env.sh && python "code/ch09/eval.py" --ckpt transformer          # 单模型全 test 评测
#   source env.sh && python "code/ch09/eval.py" --compare                   # 三方对照表
#   source env.sh && python "code/ch09/eval.py" --ckpt transformer --extrap # 外推实测
#   source env.sh && python "code/ch09/eval.py" --ckpt gru_noattn --sort-by-len  # 组批对照（9.6 组批税实录复现入口）
import argparse
import json
import math
import os
import random
import time
from collections import Counter

import torch
import sacrebleu

from data import (LOG_DIR, PAD, ensure_bpe, load_raw, encode_pairs, encode_sent, batches,
                  ids_to_text, MAX_TRAIN_PIECES, EXTRAP_BAND)
from decode import decode_corpus, beam_search, BEAM, ALPHA
from train import label_smoothing_loss, nll_loss

BUCKETS = ((0, 7, "≤7"), (8, 13, "8-13"), (14, 10 ** 9, "≥14"))  # 按参考句词数（ch1 图 1.2 对读口径）


# ---------------- 手写朴素 BLEU（口径自证用） ----------------
def naive_corpus_bleu(hyps: list, refs: list, max_n: int = 4) -> float:
    """corpus 级朴素 BLEU（式 (9.2) 的手写版）：clipped n-gram 精确率求和后取 4-gram 几何平均 × BP，
    无平滑（任一 n-gram 命中为 0 → 0 分，与 sacreBLEU 默认口径一致）。token = 空白切分。
    hyps/refs: str 列表（逐句对齐）→ 0—100 标量。"""
    match = [0] * max_n
    total = [0] * max_n
    hyp_len = ref_len = 0
    for h, r in zip(hyps, refs):
        ht, rt = h.split(), r.split()
        hyp_len += len(ht)
        ref_len += len(rt)
        for n in range(1, max_n + 1):
            hn = Counter(tuple(ht[i:i + n]) for i in range(len(ht) - n + 1))
            rn = Counter(tuple(rt[i:i + n]) for i in range(len(rt) - n + 1))
            match[n - 1] += sum(min(c, rn[g]) for g, c in hn.items())
            total[n - 1] += max(sum(hn.values()), 0)
    if min(total) == 0 or min(match) == 0:
        return 0.0
    log_p = sum(math.log(match[n] / total[n]) for n in range(max_n)) / max_n
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / hyp_len)  # 简单 BP（Papineni 式）
    return 100.0 * bp * math.exp(log_p)


# ---------------- 模型装载（Transformer 与 GRU 双 baseline 同一工厂） ----------------
def build_model(arch: str, config: dict, device):
    if arch == "transformer":
        from model import Transformer
        return Transformer(config["vocab"], n_layer=config["n_layer"], d_model=config["d_model"],
                           heads=config["heads"], d_ff=config["d_ff"], p_drop=config["p_drop"]).to(device)
    from gru_baseline import GRUBahdanau, GRUNoAttn
    cls = {"gru_noattn": GRUNoAttn, "gru_bahdanau": GRUBahdanau}[arch]
    return cls(config["vocab"], hidden=config["hidden"],
               **({"attn_dim": config["attn_dim"]} if "attn_dim" in config else {})).to(device)


def load_ckpt(tag: str, device):
    # torch.load(map_location=mps) 会产生 placeholder storage（MPS 已知坑），先载 CPU 再搬运
    ckpt = torch.load(os.path.join(LOG_DIR, f"{tag}.pt"), map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], ckpt["config"], device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


# ---------------- 主评测：全 test beam 解码 + 三口径 BLEU + 句长分桶 ----------------
@torch.no_grad()
def evaluate(tag: str, device, also_greedy: bool = False, limit: int = None, log_dir: str = LOG_DIR,
             sort_by_len: bool = False):
    """主评测（表 9.2）：test 全集 beam 解码 → 三口径 BLEU + 句长分桶。
    解码侧批形状：混批 src (B,m)（m=批内两侧最长 id 长，解码只消费源侧，但批宽由两侧最长句决定——
    批内最长可达 50+ piece，组批税的温床）；sort_by_len=True 时按源句长排序、只按源侧定宽
    （m=批内最长源 id 长，填充最小化，9.6 干净口径），结果落 eval_<tag>.json。"""
    sp = ensure_bpe(log_dir)
    model, ckpt = load_ckpt(tag, device)
    pairs = load_raw()["test"]
    if limit:
        pairs = pairs[:limit]
    refs_raw = [de for _, de in pairs]

    hyp_ids, dec_sec = decode_corpus(model, sp, pairs, device, k=BEAM, alpha=ALPHA)
    hyps_raw = [ids_to_text(sp, ids) for ids in hyp_ids]

    bleu_13a = sacrebleu.corpus_bleu(hyps_raw, [refs_raw], tokenize="13a").score
    bleu_none = sacrebleu.corpus_bleu(hyps_raw, [refs_raw], tokenize="none").score
    bleu_naive = naive_corpus_bleu(hyps_raw, refs_raw)

    res = {"tag": tag, "arch": ckpt["arch"], "n_test": len(pairs), "steps": ckpt["steps"],
           "beam": BEAM, "alpha": ALPHA, "batching": "mixed",
           "decode_seconds": round(dec_sec, 1),
           "bleu_13a": round(bleu_13a, 2), "bleu_none": round(bleu_none, 2),
           "bleu_naive": round(bleu_naive, 2),
           "naive_minus_none": round(bleu_naive - bleu_none, 6)}

    # 按参考句长分桶（词数）：与 ch1 图 1.2「容量墙」预言对读
    res["buckets"] = {}
    for lo, hi, name in BUCKETS:
        idx = [i for i, r in enumerate(refs_raw) if lo <= len(r.split()) <= hi]
        res["buckets"][name] = {
            "n": len(idx),
            "bleu_13a": round(sacrebleu.corpus_bleu([hyps_raw[i] for i in idx],
                                                    [[refs_raw[i] for i in idx]], tokenize="13a").score, 2)
            if idx else None,
            "avg_ref_words": round(sum(len(refs_raw[i].split()) for i in idx) / len(idx), 2) if idx else None,
            "avg_hyp_words": round(sum(len(hyps_raw[i].split()) for i in idx) / len(idx), 2) if idx else None,
        }

    if sort_by_len:  # 组批对照（9.6「组批税」实录的复现入口）：按源句 piece 长排序组批再解码同一 test
        order = sorted(range(len(pairs)), key=lambda i: len(sp.encode(pairs[i][0], out_type=int)))
        srcs = [encode_sent(sp, pairs[i][0]) for i in order]
        s_ids, t0 = [], time.perf_counter()
        for s in range(0, len(srcs), 64):
            chunk = srcs[s:s + 64]
            m = max(len(r) for r in chunk)  # 解码只消费源侧：填充到批内最长源句即止（填充最小化）
            src = torch.tensor([r + [PAD] * (m - len(r)) for r in chunk], dtype=torch.long, device=device)  # (B,m)
            s_ids.extend(beam_search(model, src, k=BEAM, alpha=ALPHA).tolist())
            if device.type == "mps":
                torch.mps.synchronize()
        s_sec = time.perf_counter() - t0
        s_by_orig = [None] * len(pairs)  # 排序后译文放回原始下标，与混批评数逐句对齐
        for pos, i in enumerate(order):
            s_by_orig[i] = ids_to_text(sp, s_ids[pos])
        s_buckets = {}
        for lo, hi, name in BUCKETS:
            idx = [i for i, r in enumerate(refs_raw) if lo <= len(r.split()) <= hi]
            if idx:
                s_buckets[name] = round(sacrebleu.corpus_bleu([s_by_orig[i] for i in idx],
                                                              [[refs_raw[i] for i in idx]],
                                                              tokenize="13a").score, 2)
        res["sorted_by_len"] = {
            "batching": "按源句 piece 长升序排序组批（论文 §5.1「近似句长分桶」口径）· batch 64 · "
                        "源侧填充到批内最长源句（解码不消费目标侧，填充最小化）；beam/α 与主口径同",
            "bleu_13a": round(sacrebleu.corpus_bleu(s_by_orig, [refs_raw], tokenize="13a").score, 2),
            "decode_seconds": round(s_sec, 1),
            "avg_hyp_words": round(sum(len(h.split()) for h in s_by_orig) / len(s_by_orig), 2),
            "diff_hyps_vs_mixed": sum(1 for a, b in zip(hyps_raw, s_by_orig) if a != b),
            "word_buckets": s_buckets,
        }

    if also_greedy:  # greedy 对照（beam 增益的活体数字）
        g_ids, g_sec = decode_corpus(model, sp, pairs, device, k=1)
        g_raw = [ids_to_text(sp, ids) for ids in g_ids]
        res["bleu_13a_greedy"] = round(sacrebleu.corpus_bleu(g_raw, [refs_raw], tokenize="13a").score, 2)
        res["greedy_seconds"] = round(g_sec, 1)

    res["samples"] = [{"src": pairs[i][0], "ref": refs_raw[i], "hyp": hyps_raw[i]} for i in range(3)]
    with open(os.path.join(log_dir, f"eval_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return res


# ---------------- 外推实测：训练长度 L 的 1.5~2 倍上评 loss / BLEU（F6 兑现，正弦 PE 固定） ----------------
@torch.no_grad()
def _loss_rows(model, id_pairs_by_name, device):
    """逐集计算（label-smoothed 损失, per-piece NLL/PPL）——外推 loss 与域内 loss 的对照行。
    每批同训练口径：src/tgt (B,m) → logits (B,m-1,V) → 每集标量。"""
    stats = {}
    for name, idp in id_pairs_by_name.items():
        ls_sum = nl_sum = tok = 0
        for src, tgt in batches(idp, 64, shuffle=False):
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt[:, :-1])
            n = int((tgt[:, 1:] != PAD).sum())
            ls_sum += label_smoothing_loss(logits, tgt[:, 1:]).item() * n
            nl_sum += nll_loss(logits, tgt[:, 1:]).item() * n
            tok += n
        stats[name] = {"ls_loss": round(ls_sum / tok, 4), "nll_per_piece": round(nl_sum / tok, 4),
                       "ppl_piece": round(math.exp(nl_sum / tok), 3), "pairs": len(idp)}
    return stats


@torch.no_grad()
def evaluate_extrap(tag: str, device, log_dir: str = LOG_DIR):
    """外推实测（9.6）：训练 L=16 的模型不重训，在 band=[24,32] 上评 loss/BLEU。
    loss 侧同训练口径（src/tgt (B,m) → logits (B,m-1,V)）；解码侧外推源句 (B,m) 的 m 可达 32+——
    正弦 PE 按需生成、形状不限，learned PE 查表 18 位截位。结果落 extrap_<tag>.json。"""
    sp = ensure_bpe(log_dir)
    model, ckpt = load_ckpt(tag, device)
    raw = load_raw()
    test_pairs = raw["test"]
    lo, hi = EXTRAP_BAND  # [1.5L, 2L]，L=MAX_TRAIN_PIECES=16

    # 外推集构造：test 不相交两两拼接（拼接句内容 piece 长落带内）+ 落带内的天然长句（分开统计）
    order = list(range(len(test_pairs)))
    random.Random(2017).shuffle(order)
    concat_pairs = []
    for i in range(0, len(order) - 1, 2):
        a, b = test_pairs[order[i]], test_pairs[order[i + 1]]
        src, ref = a[0] + " " + b[0], a[1] + " " + b[1]
        if lo <= len(sp.encode(src, out_type=int)) <= hi:
            concat_pairs.append((src, ref))
    natural_pairs = [p for p in test_pairs if lo <= len(sp.encode(p[0], out_type=int)) <= hi]

    stats = _loss_rows(model, {
        "val_natural": encode_pairs(sp, raw["validation"]),        # 域内（验证集，≤自然长度）
        "test_natural": encode_pairs(sp, test_pairs),              # 域内（测试集，含少量天然长句）
        "extrap_concat[24,32]": encode_pairs(sp, concat_pairs),    # 外推主体：拼接对
        "extrap_natural[24,32]": encode_pairs(sp, natural_pairs),  # 外推天然长句
    }, device)

    bleu_rows = {}
    for name, pairs in (("extrap_all[24,32]", concat_pairs + natural_pairs),
                        ("extrap_concat[24,32]", concat_pairs),
                        ("extrap_natural[24,32]", natural_pairs)):
        hyp_ids, dec_sec = decode_corpus(model, sp, pairs, device, k=BEAM, alpha=ALPHA)
        hyps_raw = [ids_to_text(sp, ids) for ids in hyp_ids]
        refs_raw = [de for _, de in pairs]
        bleu_rows[name] = {"n": len(pairs),
                           "bleu_13a": round(sacrebleu.corpus_bleu(hyps_raw, [refs_raw], tokenize="13a").score, 2),
                           "seconds": round(dec_sec, 1)}
        if name == "extrap_all[24,32]":
            samples = [{"src": pairs[i][0][:90], "ref": pairs[i][1][:90], "hyp": hyps_raw[i][:90]} for i in range(3)]

    res = {"tag": tag, "band": [lo, hi], "L_train": MAX_TRAIN_PIECES,
           "n_concat_pairs": len(concat_pairs), "n_natural_in_band": len(natural_pairs),
           "loss_stats": stats, "bleu_extrap": bleu_rows, "extrap_samples": samples}
    with open(os.path.join(log_dir, f"extrap_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return res


def print_eval(res):
    print(f"[{res['tag']}] test {res['n_test']} 句 · beam={res['beam']} α={res['alpha']} · "
          f"解码 {res['decode_seconds']}s · 训练步数 {res['steps']}")
    print(f"  BLEU(13a)={res['bleu_13a']}  BLEU(none)={res['bleu_none']}  手写朴素={res['bleu_naive']}"
          f"  （朴素-none 差 {res['naive_minus_none']} → 口径自证）")
    if "sorted_by_len" in res:
        s = res["sorted_by_len"]
        print(f"  [组批对照] 混批 {res['bleu_13a']} → 按长排序 {s['bleu_13a']}"
              f"（{s['diff_hyps_vs_mixed']}/{res['n_test']} 句译文变化，平均译长 {s['avg_hyp_words']} 词，"
              f"词桶 {s['word_buckets']}）")
    if "bleu_13a_greedy" in res:
        print(f"  贪心 BLEU(13a)={res['bleu_13a_greedy']}（{res['greedy_seconds']}s）→ "
              f"beam 增益 {res['bleu_13a'] - res['bleu_13a_greedy']:+.2f}")
    for name in ("≤7", "8-13", "≥14"):
        b = res["buckets"][name]
        print(f"  句长桶 {name:>5} 词：n={b['n']:>4}  BLEU={b['bleu_13a']:>6}  "
              f"平均译长/参考长={b['avg_hyp_words']}/{b['avg_ref_words']}")


def main():
    ap = argparse.ArgumentParser(description="ch9 评测：BLEU 三口径 + 句长分桶 + 外推 + 三方对照")
    ap.add_argument("--ckpt", default="transformer", help="log/Book1-ch09/ 下的 tag（如 gru_noattn）")
    ap.add_argument("--compare", action="store_true", help="三方对照表（transformer / gru_noattn / gru_bahdanau）")
    ap.add_argument("--extrap", action="store_true", help="外推实测（1.5~2 倍训练长度）")
    ap.add_argument("--only-extrap", action="store_true", help="跳过主评测，只跑外推（主评测已有 JSON 时用）")
    ap.add_argument("--greedy", action="store_true", help="附带贪心解码对照")
    ap.add_argument("--sort-by-len", action="store_true",
                    help="组批对照：主口径（数据集顺序混批）之外，另按源句 piece 长排序组批解码同一 test，"
                         "结果入 JSON 的 sorted_by_len 块（9.6「组批税」实录的复现入口）")
    ap.add_argument("--limit", type=int, default=None, help="只评前 N 句（冒烟用）")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    args = ap.parse_args()
    device = torch.device(args.device)

    tags = ["transformer", "gru_noattn", "gru_bahdanau"] if args.compare else [args.ckpt]
    for tag in tags:
        if not args.only_extrap:
            res = evaluate(tag, device, also_greedy=args.greedy or (tag == "transformer" and not args.compare),
                           limit=args.limit, sort_by_len=args.sort_by_len)
            print_eval(res)
        if args.extrap or args.only_extrap:
            ex = evaluate_extrap(tag, device)
            print(f"  [外推] 带 {ex['band']}（L={ex['L_train']}）：拼接 {ex['n_concat_pairs']} 对 + "
                  f"天然长句 {ex['n_natural_in_band']} 句")
            for k2, v in ex["loss_stats"].items():
                print(f"    loss[{k2:<22}] ls={v['ls_loss']:.3f}  nll/piece={v['nll_per_piece']:.3f}  "
                      f"PPL={v['ppl_piece']:8.2f}  (n={v['pairs']})")
            for k2, v in ex["bleu_extrap"].items():
                print(f"    BLEU[{k2:<22}] n={v['n']:>3}  BLEU(13a)={v['bleu_13a']:>6}  ({v['seconds']}s)")


if __name__ == "__main__":
    main()
