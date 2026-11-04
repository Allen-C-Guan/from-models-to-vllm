# -*- coding: utf-8 -*-
# 用途：Book1 第 10 章——消融网格编排器：13 配置（base×3 种子 + 10 行单种子消融）复用 ch09 公平协议长训，
#       增量落盘 log/Book1-ch10/grid_results.json + 断点续跑 + 方向判据汇总（plan 10.2/10.3）。
# 所属章节：《Transformer 原典》第 10 章（效果实证：趋势级复现与引用对照）
# 运行：
#   source env.sh && python "code/ch10/ablation.py" --smoke            # 冒烟：base 2 epochs 全链（结果落 grid_smoke.json，不触碰 grid_results.json）
#   source env.sh && nohup python "code/ch10/ablation.py" \
#       > log/Book1-ch10/grid.log 2>&1 &                                                     # 全网格（MPS 约 1 h）
#   source env.sh && python "code/ch10/ablation.py" --extrap-learnedpe  # learned PE 外推补测（约 1 min）
# 依据：plan/Book1-Transformer原典.md 10.2/10.3；research/.../notes/01-实验可行性.md 第 (7) 节判定标准。
# 协议（ch09 公平协议原样复用，gru_baseline.py 同款五条）：同 shared BPE-8k 词表（直接加载 ch09 已训
#   bpe_8k.model）· 同数据（L=16 双侧过滤 17555 对、batch 64、30 epochs=8250 步=同 token 预算）·
#   同优化调度（Noam warmup 4000/factor 1.0 + Adam β2=0.98 eps=1e-9；所有行 d_model=256 → lr 逐 step 相同）·
#   checkpoint=末 epoch（Table 3 口径：无 averaging、不挑 best-val）· beam=4 α=0.6 全 test 1000 句 ·
#   sacreBLEU tokenize=13a · 句长桶 ≤7/8-13/≥14（按参考句词数，与 ch9 eval 同一口径常量）。
# 诚实条款（notes/01 (7)）：绝对 BLEU/PPL 与 Table 3 不可比（Multi30K≠WMT14、算力 ~1/100），只宣称方向；
#   PPL 为 per-wordpiece 口径；消融行单种子 + base 3 种子方差代理，局限随 JSON 落盘。
#
# ── 适配说明（ch09 一行不改，变体全部以 wrapper 在本文件解决）─────────────────────────
#   ① 无 PE / learned PE：子类覆盖 Transformer._add_pe（NoPETransformer / LearnedPETransformer）；
#   ② 去 √d_k / d_k 缩减：MultiHeadAttention 子类（NoScaleMHA / ReducedDKMHA），构建后逐层原位替换
#      self_attn / cross_attn 子模块（nn.Module.__setattr__ 对已注册子模块名即原位换装，FFN/子层不动）；
#   ③ ε_ls 与种子可变：ch09 train.run_epoch/validate 把 eps=0.1 与洗牌种子 2017 硬编码在函数体内，
#      无法传参 → 本文件按 ch09 同款复刻并参数化（run_epoch_ls / validate_ls，逐行对齐原实现）；
#   ④ 种子作用域 = 参数初始化（torch.manual_seed）+ 逐 epoch 洗牌（batches(seed=seed, epoch=epoch)）
#      + dropout 掩码（全局 RNG）；机制与 ch09 的 2017 完全一致，仅换成配置种子。
import argparse
import datetime as dt
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import sacrebleu

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "ch09")))  # 复用 ch09 管线模块
from data import (ensure_bpe, load_raw, encode_pairs, filter_pairs, batches, ids_to_text,
                  PAD, BATCH, MAX_TRAIN_PIECES, EXTRAP_BAND, LOG_DIR as CH9_LOG_DIR)
from model import Transformer, MultiHeadAttention, sinusoidal_pe, n_params
from model import N_LAYER, D_MODEL, HEADS, D_FF, P_DROP
from train import NoamLR, WARMUP, label_smoothing_loss, nll_loss
from decode import decode_corpus, BEAM, ALPHA
from eval import BUCKETS, _loss_rows  # 句长桶与外推 loss 行口径与 ch9 同源

LOG_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "log", "Book1-ch10"))
# 落盘目标按模式选定（一处改动）：--smoke 冒烟档隔离落 grid_smoke.json，绝不触碰正式网格的 grid_results.json
RESULTS_JSON = os.path.join(LOG_DIR, "grid_smoke.json" if "--smoke" in sys.argv else "grid_results.json")


# ---------------- 模型变体（ch09 子类，见头部「适配说明」） ----------------
class NoScaleMHA(MultiHeadAttention):
    """去 √d_k 缩放行（必做三行之三；论文无对应行，ch2 猜想的训练级验证）：
    scores = QK^T，不再除以 √d_k。除 scores 一行外，forward 与 ch09 MultiHeadAttention 逐行一致。
    形状差异：无——只改打分的数值尺度，不动任何投影：Q/K/V 仍 (B,n,256)→逐头 (B,h,n,d_k)，
    打分矩阵仍 (B,h,n,m)，输出仍 (B,n,256)（本行 d_k=32，与 base 逐张量同形）。"""

    def forward(self, q_in, k_in, v_in, mask, return_weights=False):
        b = q_in.size(0)
        split = lambda x: x.reshape(b, -1, self.h, self.d_k).transpose(1, 2)  # (B,n,256) -> (B,h,n,d_k)
        q, k, v = split(self.wq(q_in)), split(self.wk(k_in)), split(self.wv(v_in))  # 各 (B,h,n,d_k)
        scores = q @ k.transpose(-2, -1)  # (B,h,n,d_k)×(B,h,d_k,m) -> (B,h,n,m)；← 与 ch09 的唯一差别：去掉 / math.sqrt(d_k)
        scores = scores.masked_fill(mask, float("-inf"))  # mask (B,1,1,S)/(B,1,T,T) 广播到 (B,h,n,m)
        probs = F.softmax(scores, dim=-1)  # (B,h,n,m)，行和为 1
        out = probs @ v  # (B,h,n,m)×(B,h,m,d_k) -> (B,h,n,d_k)
        out = out.transpose(1, 2).reshape(b, -1, self.h * self.d_k)  # 拼回 (B,n,h·d_k)
        out = self.wo(out)  # (B,n,h·d_k) -> (B,n,256)
        return (out, probs) if return_weights else out


class ReducedDKMHA(MultiHeadAttention):
    """d_k 缩减行——Table 3 (B) 语义在 mini 版上的映射（本书口径）：
    保持 h=8，把每头 Q/K/V 投影从 d_k=d_v=32（=256/8）降到 16——即 wq/wk/wv 投影到 h·16=128 维、
    输出投影 wo 从 128 收回 256；注意力在更低维空间打分，参数量随之下降（论文 65M→60M/58M 同方向）。
    与论文 (B) 的差异如实注记：论文 (B) 固定 h=8、只把 d_k 从 64 缩到 32/16（d_v 不动，参数 60M/58M
    由 65M 扣 W^Q/W^K 而来，ch03 表 3.1 已核对）；本书 base 为 h=8/d_k=d_v=32，取「每头 dk=dv 同步
    减半到 16」——d_v 一并缩半是本书口径与论文 (B) 的幅度差异（mini 模型投影收缩后拼回维度需对齐），
    语义同为「缩小每头打分维度」。
    形状差异（对照 base d_k=d_v=32）：wq/wk/wv (B,n,256)→(B,n,h·d_k=128)，逐头拆分后 (B,h,n,32)→(B,h,n,16)；
    打分矩阵仍 (B,h,n,m)——缩的是每头打分/取值维度，注意力矩阵形状不变；拼回 (B,n,128) 再由 wo 收回
    (B,n,256)，子层对外 I/O 与 base 同形（forward 完全继承 ch09，split/√dk 缩放/拼回均按 self.d_k 自适应）。"""

    def __init__(self, d_model: int, h: int, p_drop: float, d_k: int):
        nn.Module.__init__(self)  # 跳过父类构造（其投影固定 d_model×d_model），自建收缩投影
        self.h, self.d_k = h, d_k
        self.wq, self.wk, self.wv = (nn.Linear(d_model, h * d_k) for _ in range(3))  # (B,n,256) -> (B,n,h·d_k)
        self.wo = nn.Linear(h * d_k, d_model)  # (B,n,h·d_k) -> (B,n,256)


class NoPETransformer(Transformer):
    """无 PE 行（必做三行之二）：PE 置零，只保留 嵌入×√d_model——模型退化为词袋（置换等变，ch2 回收）。
    教学性消融，Table 3 无对应行（显式标注）。
    形状差异：无——嵌入输出仍 (B,n,256)，只是加法里少了「位置项」这个加数，逐张量与 base 同形。"""

    def _add_pe(self, ids):
        return self.emb_drop(self.embed(ids) * math.sqrt(self.d_model))  # (B,n,256)：PE 项置零，形状与 base 一致


class LearnedPETransformer(Transformer):
    """learned PE 行（Table 3 (E) 语义）：nn.Embedding 位置嵌入替代正弦。
    尺寸=训练 max_len：L=16 双侧内容过滤 + SOS/EOS → 训练序列长上限 18。
    推理遇超长序列（test 未按 L 过滤、解码上限 64）时位置下标截到最后一位——learned PE 无法外推，
    正是与正弦 PE 的对照点（ch9 外推实测的反面）。初始化取正弦值（Annotated Transformer 同款做法），
    训练中可学习。
    形状差异：PE 项来源从「按需生成的 (n,256) 正弦矩阵」换成「(18,256) 查表」——论文 (max_len,512)
    位置查表的本书缩尺寸版；查表 I/O：pos 下标 (n,) → (n,256)，广播加到嵌入 (B,n,256) 上，
    加法与输出形状和正弦版完全同形，故与 base 可在同一条管线互换。"""

    def __init__(self, vocab: int, pe_len: int, **kw):
        super().__init__(vocab, **kw)
        self.pe = nn.Embedding(pe_len, self.d_model)  # 位置查表 (18,256)：查 (n,) 得 (n,256)
        with torch.no_grad():
            self.pe.weight.copy_(sinusoidal_pe(pe_len, self.d_model))

    def _add_pe(self, ids):
        pos = torch.arange(ids.size(1), device=ids.device)  # (n,)
        pos = torch.clamp(pos, max=self.pe.num_embeddings - 1)  # 超长输入截位（无法外推的诚实处理）
        return self.emb_drop(self.embed(ids) * math.sqrt(self.d_model) + self.pe(pos))  # (B,n,256)+(n,256) 广播相加


def _swap_attn(model, mha_factory):
    """逐层原位替换注意力子模块（self_attn / cross_attn）；FFN 与残差子层保持 ch09 原装。
    换装后子层对外形状不变（(B,n,256) → (B,n,256)）——变体差异全部封闭在注意力内部。"""
    for layer in [*model.enc_layers, *model.dec_layers]:
        layer.self_attn = mha_factory()
    for layer in model.dec_layers:
        layer.cross_attn = mha_factory()


def build_model(cfg: dict, vocab: int) -> nn.Module:
    """按网格行实例化模型变体；无 override 键 = ch09 原版 Transformer。
    外形契约与 ch09 一致：forward(src (B,S), tgt_in (B,T)) → logits (B,T,V)；各行差异只在内部形状（见各类 docstring）。"""
    kw = dict(n_layer=cfg.get("n_layer", N_LAYER), d_model=D_MODEL, heads=cfg.get("heads", HEADS),
              d_ff=D_FF, p_drop=cfg.get("p_drop", P_DROP))
    pe = cfg.get("pe", "sin")
    if pe == "learned":
        model = LearnedPETransformer(vocab, MAX_TRAIN_PIECES + 2, **kw)
    elif pe == "none":
        model = NoPETransformer(vocab, **kw)
    else:
        model = Transformer(vocab, **kw)
    if cfg.get("d_k"):
        _swap_attn(model, lambda: ReducedDKMHA(D_MODEL, kw["heads"], kw["p_drop"], cfg["d_k"]))
    if cfg.get("scale") is False:
        _swap_attn(model, lambda: NoScaleMHA(D_MODEL, kw["heads"], kw["p_drop"]))
    return model


# ---------------- 消融网格（13 运行：base×3 种子 + 10 行单种子消融，seed 0） ----------------
GRID = [
    # base ×3 种子：估计 σ_seed（方向判据的分母）+ Table 3 base 行锚点
    dict(run_id="base_s0", row="base", seed=0, table3="base: PPL 4.92 / BLEU 25.8 / 65M"),
    dict(run_id="base_s1", row="base", seed=1, table3="base: PPL 4.92 / BLEU 25.8 / 65M"),
    dict(run_id="base_s2", row="base", seed=2, table3="base: PPL 4.92 / BLEU 25.8 / 65M"),
    # 必做三行（plan 10.3）
    dict(run_id="h1_s0", row="h=1", seed=0, heads=1, table3="(A) h=1 → 24.9", note="必做三行之一"),
    dict(run_id="nope_s0", row="no_pe", seed=0, pe="none", table3="无对应行（教学性消融，显式标注）",
         note="必做三行之二"),
    dict(run_id="noscale_s0", row="no_scale", seed=0, scale=False,
         table3="无对应行（论文未单列；ch2 √dk 猜想的训练级验证）", note="必做三行之三"),
    # 论文锚点行（Table 3 对照）
    dict(run_id="h4_s0", row="h=4", seed=0, heads=4, table3="(A) h=4 → 25.5"),
    dict(run_id="h16_s0", row="h=16", seed=0, heads=16, table3="(A) h=16 → 25.8"),
    dict(run_id="dk16_s0", row="d_k=16", seed=0, d_k=16,
         table3="(B) d_k=16（d_v 不变）→ 25.1；本书变体 dk=dv=16 映射见 ReducedDKMHA"),
    dict(run_id="n2_s0", row="N=2", seed=0, n_layer=2, table3="(C) N=2 → 23.7"),
    dict(run_id="drop0_s0", row="Pdrop=0", seed=0, p_drop=0.0, table3="(D) Pdrop=0 → 24.6"),
    dict(run_id="ls0_s0", row="eps_ls=0", seed=0, eps_ls=0.0,
         table3="(D) εls=0 → PPL 4.67 / BLEU 25.3（双指标反向）"),
    dict(run_id="learnedpe_s0", row="learned_pe", seed=0, pe="learned", table3="(E) learned PE → 25.7 ≈ base"),
]


# ---------------- 训练 / 评测（ch09 复刻参数化，见头部「适配说明」③） ----------------
def run_epoch_ls(model, id_pairs, opt_sched, device, epoch, seed, eps, clip=None):
    """ch09 train.run_epoch 的参数化复刻（eps 与洗牌种子可变）；其余逐行一致（Noam step、pad 位不计损失）。"""
    model.train()
    total_loss, total_tok, n_batch = 0.0, 0, 0
    for src, tgt in batches(id_pairs, BATCH, shuffle=True, seed=seed, epoch=epoch):
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src, tgt[:, :-1])
        loss = label_smoothing_loss(logits, tgt[:, 1:], eps=eps)
        opt_sched.opt.zero_grad()
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt_sched.step()
        n_valid = int((tgt[:, 1:] != PAD).sum())
        total_loss += loss.item() * n_valid
        total_tok += n_valid
        n_batch += 1
    return total_loss / total_tok, n_batch


@torch.no_grad()
def validate_ls(model, id_pairs, device, eps):
    """ch09 train.validate 的参数化复刻（eps 可变）。返回（LS 损失, 每 piece NLL）：
    NLL/PPL 与 eps 无关、跨行可比（Table 3 的 PPL 即 per-wordpiece 口径）；LS 损失各行用各自 eps，只作训练曲线。"""
    model.eval()
    ls, nl, tok = 0.0, 0.0, 0
    for src, tgt in batches(id_pairs, BATCH, shuffle=False):
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src, tgt[:, :-1])
        ls += label_smoothing_loss(logits, tgt[:, 1:], eps=eps).item() * int((tgt[:, 1:] != PAD).sum())
        nl += nll_loss(logits, tgt[:, 1:]).item() * int((tgt[:, 1:] != PAD).sum())
        tok += int((tgt[:, 1:] != PAD).sum())
    return ls / tok, nl / tok


def evaluate_bleu(model, sp, test_pairs, device):
    """全 test beam=4/α=0.6 解码 + sacreBLEU(13a) + 句长桶——与 ch9 eval.evaluate 同口径同代码路径。"""
    hyp_ids, dec_sec = decode_corpus(model, sp, test_pairs, device, k=BEAM, alpha=ALPHA)
    hyps = [ids_to_text(sp, ids) for ids in hyp_ids]
    refs = [de for _, de in test_pairs]
    res = {"n_test": len(test_pairs), "beam": BEAM, "alpha": ALPHA, "decode_seconds": round(dec_sec, 1),
           "bleu_13a": round(sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a").score, 2), "buckets": {}}
    for lo, hi, name in BUCKETS:
        idx = [i for i, r in enumerate(refs) if lo <= len(r.split()) <= hi]
        res["buckets"][name] = {
            "n": len(idx),
            "bleu_13a": round(sacrebleu.corpus_bleu([hyps[i] for i in idx],
                                                    [[refs[i] for i in idx]], tokenize="13a").score, 2) if idx else None,
            "avg_ref_words": round(sum(len(refs[i].split()) for i in idx) / len(idx), 2) if idx else None,
            "avg_hyp_words": round(sum(len(hyps[i].split()) for i in idx) / len(idx), 2) if idx else None,
        }
    res["samples"] = [{"src": test_pairs[i][0], "ref": refs[i], "hyp": hyps[i]} for i in range(2)]
    return res


def run_config(cfg: dict, epochs: int, device, sp, data: dict) -> dict:
    """单配置全链：训练（逐 epoch 验证）→ 存末 epoch checkpoint（Table 3 口径）→ 全 test 评测。"""
    run_id = cfg["run_id"]
    started = dt.datetime.now().isoformat(timespec="seconds")
    print(f"[{run_id}] 开始：row={cfg['row']} seed={cfg['seed']} · Table 3 锚点：{cfg['table3']}", flush=True)
    t0 = time.perf_counter()
    torch.manual_seed(cfg["seed"])  # 参数初始化 + dropout 掩码（洗牌种子另由 batches(seed=...) 控制）
    model = build_model(cfg, sp.get_piece_size()).to(device)
    n_par, n_emb = n_params(model), model.embed.weight.numel()
    print(f"[{run_id}] 参数 {n_par:,}（非嵌入 {n_par - n_emb:,}）· epochs={epochs}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-7, betas=(0.9, 0.98), eps=1e-9)
    sched = NoamLR(opt, D_MODEL, WARMUP)
    eps_ls = cfg.get("eps_ls", 0.1)
    history = []
    t_train = time.perf_counter()
    for epoch in range(epochs):
        te = time.perf_counter()
        tr_loss, _ = run_epoch_ls(model, data["train_ids"], sched, device, epoch, cfg["seed"], eps_ls)
        val_ls, val_nll = validate_ls(model, data["val_ids"], device, eps_ls)
        if device.type == "mps":
            torch.mps.synchronize()
        sec = time.perf_counter() - te
        history.append({"epoch": epoch + 1, "train_loss": round(tr_loss, 4), "val_ls_loss": round(val_ls, 4),
                        "val_nll": round(val_nll, 4), "val_ppl_piece": round(math.exp(val_nll), 3),
                        "lr": round(sched.rate(), 8), "seconds": round(sec, 1)})
        print(f"[{run_id}] epoch {epoch + 1:>2}/{epochs}  train {tr_loss:.3f}  val_ls {val_ls:.3f}  "
              f"PPL(piece) {math.exp(val_nll):7.2f}  lr {sched.rate():.2e}  {sec:5.1f}s", flush=True)
    train_seconds = time.perf_counter() - t_train
    torch.save({"run_id": run_id, "cfg": cfg, "model": model.state_dict(), "steps": sched.step_num,
                "arch_note": "ch10 消融变体；重建方式见 ablation.build_model"},
               os.path.join(LOG_DIR, f"grid_{run_id}.pt"))
    ev = evaluate_bleu(model, sp, data["test_pairs"], device)
    wall = time.perf_counter() - t0
    rec = {"run_id": run_id, "row": cfg["row"], "seed": cfg["seed"], "table3_ref": cfg["table3"],
           "note": cfg.get("note"), "cfg": cfg, "params": n_par, "nonembed_params": n_par - n_emb,
           "epochs": epochs, "steps": sched.step_num, "eps_ls": eps_ls, "vocab": sp.get_piece_size(),
           "history": history, "final_val_ls_loss": history[-1]["val_ls_loss"],
           "final_val_ppl_piece": history[-1]["val_ppl_piece"],
           "train_seconds": round(train_seconds, 1), "wall_clock_seconds": round(wall, 1),
           "started": started, "finished": dt.datetime.now().isoformat(timespec="seconds")}
    rec.update(ev)
    if device.type == "mps":
        torch.mps.empty_cache()
    return rec


# ---------------- 落盘 / 续跑 ----------------
def load_results() -> dict:
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {"runs": []}


def save_results(res: dict):
    """每配置完成即原子落盘（tmp + os.replace）——中途断电/杀进程不损坏 JSON，重启按 run_id 续跑。"""
    tmp = RESULTS_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    os.replace(tmp, RESULTS_JSON)


def variant_selftest(sp, device):
    """网格启动前自检：13 个变体逐个构建 + 前向 + 反传一步 + 解码接口（约 10 s，长跑前的形状护栏）。
    前向契约在此断言：src (1,S) + tgt_in (1,T) → logits (1,T,V)；含 20 长序列，learned PE（pe_len=18）截位路径一并覆盖。"""
    vocab = sp.get_piece_size()
    src = torch.tensor([[2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 3, 0, 0]], device=device)
    tgt_in = torch.tensor([[2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 3, 0]], device=device)
    tgt_out = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 3, 0, 0]], device=device)
    for cfg in GRID:  # 含 20 长度序列：learned PE（尺寸 18）的截位路径一并覆盖
        model = build_model(cfg, vocab).to(device)
        model.eval()  # 关 dropout：两次前向（整段 vs decode_step）才逐位一致（ch09 model.main 同款检查）
        logits = model(src, tgt_in)
        assert logits.shape == (1, tgt_in.size(1), vocab), cfg["run_id"]
        loss = label_smoothing_loss(logits, tgt_out)
        loss.backward()
        last = model.decode_step(model.encode(src), tgt_in)
        assert torch.allclose(last, logits[:, -1], atol=1e-5), cfg["run_id"]
        print(f"[自检] {cfg['run_id']:<14} ok（loss {loss.item():.2f}，参数 {n_params(model):,}）", flush=True)
    print("[自检] 13 个变体构建+前向+反传+解码接口 全部通过", flush=True)


# ---------------- 方向判据汇总（plan 10.2 ①-⑦ + 信息行） ----------------
def summarize(res: dict):
    runs = res["runs"]
    bases = [r for r in runs if r["row"] == "base"]
    if not bases:
        return
    mean = sum(b["bleu_13a"] for b in bases) / len(bases)
    sigma = (sum((b["bleu_13a"] - mean) ** 2 for b in bases) / max(len(bases) - 1, 1)) ** 0.5  # 样本标准差
    ppl_mean = sum(b["final_val_ppl_piece"] for b in bases) / len(bases)
    thr = max(2 * sigma, 0.5)
    by = {r["run_id"]: r for r in runs}
    print(f"\n[汇总] base {len(bases)} 种子：BLEU 均值 {mean:.2f}，σ_seed {sigma:.2f}；"
          f"PPL(piece) 均值 {ppl_mean:.2f}；阈值 |ΔBLEU| ≥ max(2σ, 0.5) = {thr:.2f}", flush=True)
    print("[汇总] 各行 vs base（单种子局限：Δ 含种子噪声，判定只看方向与阈值）", flush=True)
    rows = []
    for cfg in GRID:
        if cfg["row"] == "base":
            continue
        r = by.get(cfg["run_id"])
        if r is None:
            continue
        d_bleu, d_ppl = r["bleu_13a"] - mean, r["final_val_ppl_piece"] - ppl_mean
        hit = abs(d_bleu) >= thr
        rows.append({"run_id": cfg["run_id"], "row": cfg["row"], "table3_ref": cfg["table3"],
                     "bleu_13a": r["bleu_13a"], "delta_bleu": round(d_bleu, 2),
                     "ppl_piece": r["final_val_ppl_piece"], "delta_ppl": round(d_ppl, 2),
                     "threshold": round(thr, 3), "beyond_seed_noise": hit})
        print(f"  {cfg['run_id']:<14} BLEU {r['bleu_13a']:>6}（Δ{d_bleu:+6.2f}）  PPL {r['final_val_ppl_piece']:>8.2f}"
              f"（Δ{d_ppl:+7.2f}）  超噪声 {'是' if hit else '否'}  ← {cfg['table3']}", flush=True)

    def d(rid):
        r = by.get(rid)
        return None if r is None else r["bleu_13a"] - mean

    def dp(rid):
        r = by.get(rid)
        return None if r is None else r["final_val_ppl_piece"] - ppl_mean

    def fmt(rid):
        return "未跑" if d(rid) is None else f"ΔBLEU={d(rid):+.2f}" + (
            "" if dp(rid) is None else f"，ΔPPL={dp(rid):+.2f}")

    checks = [
        ("① h=1 显著差", d("h1_s0") is not None and d("h1_s0") <= -thr, fmt("h1_s0")),
        ("② 缩 d_k 变差（(B) 语义单点）", d("dk16_s0") is not None and d("dk16_s0") < 0, fmt("dk16_s0")),
        ("③ N=2 变差（单边证据）", d("n2_s0") is not None and d("n2_s0") < 0, fmt("n2_s0")),
        ("④ Pdrop=0 变差 ≥1 BLEU", d("drop0_s0") is not None and -d("drop0_s0") >= 1.0, fmt("drop0_s0")),
        ("⑤ εls=0 双指标反向（PPL 变好且 BLEU 变差）",
         d("ls0_s0") is not None and d("ls0_s0") < 0 and dp("ls0_s0") is not None and dp("ls0_s0") < 0, fmt("ls0_s0")),
        ("⑥ learned PE ≈ 正弦（|Δ| < σ_seed）",
         d("learnedpe_s0") is not None and abs(d("learnedpe_s0")) < sigma, fmt("learnedpe_s0")),
        ("⑦ 无 PE 崩溃（教学性消融，无 Table 3 行）",
         d("nope_s0") is not None and d("nope_s0") <= -5 * thr, fmt("nope_s0")),
        ("⑧ 去 √d_k（论文无行，不判定，如实呈现）", None, fmt("noscale_s0")),
    ]
    print("[汇总] plan 10.2 方向判据命中情况（如实）：", flush=True)
    for name, hit, detail in checks:
        verdict = "信息行" if hit is None else ("命中" if hit else "未命中")
        print(f"  {name:<28} {verdict}  （{detail}）", flush=True)
    res["summary"] = {"base_n_seeds": len(bases), "base_mean_bleu": round(mean, 2),
                      "base_sigma_bleu": round(sigma, 3), "base_mean_ppl_piece": round(ppl_mean, 2),
                      "threshold": round(thr, 3),
                      "threshold_rule": "|ΔBLEU| ≥ max(2σ_seed, 0.5)；σ=样本标准差（3 种子，ddof=1）",
                      "single_seed_caveat": "消融行为单种子，Δ 含种子噪声；结论只宣称方向（绝对值与 Table 3 不可比）",
                      "rows": rows, "direction_checks": [{"criterion": n, "hit": h, "detail": s} for n, h, s in checks]}


# ---------------- learned PE 外推补测（评 grid_learnedpe_s0，ch9 eval.evaluate_extrap 同协议） ----------------
@torch.no_grad()
def evaluate_extrap_learnedpe(device):
    """learned PE 行（判据⑥）的外推补测：沿 ch09 eval.py 的外推同协议——band [24,32]（=1.5L~2L，L=16），
    测试集不相交两两拼接 286 对 + 天然长句 43 句（同种子 2017），逐集 LS 损失 / per-piece PPL + beam BLEU，
    落 log/Book1-ch10/learnedpe_extrap.json（与 extrap_transformer.json 并排即「正弦 vs 学习式」收口）。
    截位行为如实记录：learned PE 是 pe_len=18 的 nn.Embedding（L=16 内容 + SOS/EOS），
    序列位置 ≥18 一律 clamp 到末位（第 17 号）向量复用——查表结构上无法外推的诚实处理（LearnedPETransformer._add_pe）。
    形状口径：外推序列长 n∈[24,32] > pe_len=18——位置下标 (n,) 先 clamp 再查表得 (n,256)，加法与输出
    (B,n,256) 形状不变；打分/前馈/logits 各张量形状与域内一致，变的只是查表来源（18 号以内向量被反复复用）。"""
    ckpt = torch.load(os.path.join(LOG_DIR, "grid_learnedpe_s0.pt"), map_location="cpu", weights_only=False)
    sp = ensure_bpe(CH9_LOG_DIR)
    model = build_model(ckpt["cfg"], sp.get_piece_size()).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    pe_len = model.pe.num_embeddings

    raw = load_raw()
    test_pairs = raw["test"]
    lo, hi = EXTRAP_BAND
    order = list(range(len(test_pairs)))  # 外推集构造：与 ch09 eval.evaluate_extrap 完全同种子同口径
    random.Random(2017).shuffle(order)
    concat_pairs = []
    for i in range(0, len(order) - 1, 2):
        a, b = test_pairs[order[i]], test_pairs[order[i + 1]]
        src, ref = a[0] + " " + b[0], a[1] + " " + b[1]
        if lo <= len(sp.encode(src, out_type=int)) <= hi:
            concat_pairs.append((src, ref))
    natural_pairs = [p for p in test_pairs if lo <= len(sp.encode(p[0], out_type=int)) <= hi]
    assert (len(concat_pairs), len(natural_pairs)) == (286, 43), "外推集与 ch09 eval.py 口径不一致"

    stats = _loss_rows(model, {
        "val_natural": encode_pairs(sp, raw["validation"]),
        "test_natural": encode_pairs(sp, test_pairs),
        "extrap_concat[24,32]": encode_pairs(sp, concat_pairs),
        "extrap_natural[24,32]": encode_pairs(sp, natural_pairs),
    }, device)

    bleu_rows, samples = {}, None
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

    # 截位统计：外推集（拼接+天然）里被 clamp 的位置占比——「一张表用到尽头」的量化
    seq_lens = [len(sp.encode(p[0], out_type=int)) + 2 for p in concat_pairs + natural_pairs]
    clamped = sum(max(0, l - pe_len) for l in seq_lens)
    res = {"tag": "learnedpe_s0", "arch_note": "ch10 消融网格 learned PE 行（LearnedPETransformer：18×256 位置查表，"
                                               "初始化取正弦值后可学习；域内 test BLEU 22.19 见 grid_results.json）",
           "protocol": "ch09 eval.evaluate_extrap 同协议：band=[24,32]（1.5L~2L，L=16），拼接 286 对 + 天然长句 43 句"
                       "（种子 2017）；beam=4 α=0.6，sacreBLEU 13a；PPL 为 per-piece 口径",
           "band": [lo, hi], "L_train": MAX_TRAIN_PIECES, "pe_len": pe_len,
           "n_concat_pairs": len(concat_pairs), "n_natural_in_band": len(natural_pairs),
           "truncation": {"rule": f"位置下标 clamp 到 pe_len-1={pe_len - 1}：序列第 {pe_len} 个及以后的位置"
                                  f"一律复用末位（第 {pe_len - 1} 号）向量",
                          "extrap_clamped_position_share": round(clamped / sum(seq_lens), 4),
                          "note": "learned PE 查表结构上无法外推（pos ≥ pe_len 无向量可用），截位复用末位是诚实处理；"
                                  "正弦 PE 无此限制（闭合公式按需生成任意长度，对照数字见 sinusoid_ch9_ref）"},
           "loss_stats": stats, "bleu_extrap": bleu_rows, "extrap_samples": samples,
           "sinusoid_ch9_ref": {"source": "log/Book1-ch09/extrap_transformer.json（同协议正弦侧，ch9 9.6 引用）",
                                "bleu_concat": 9.03, "bleu_natural": 6.62,
                                "ppl_concat": 96.98, "ppl_natural": 154.57}}
    out = os.path.join(LOG_DIR, "learnedpe_extrap.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[learned PE 外推] 带 {res['band']}（L={res['L_train']}，pe_len={pe_len}，截位复用末位向量）："
          f"拼接 {len(concat_pairs)} 对 + 天然 {len(natural_pairs)} 句，"
          f"被截位位置占外推集全部位置的 {res['truncation']['extrap_clamped_position_share']:.1%}")
    for k2, v in stats.items():
        print(f"    loss[{k2:<22}] ls={v['ls_loss']:.3f}  nll/piece={v['nll_per_piece']:.3f}  "
              f"PPL={v['ppl_piece']:8.2f}  (n={v['pairs']})")
    for k2, v in bleu_rows.items():
        print(f"    BLEU[{k2:<22}] n={v['n']:>3}  BLEU(13a)={v['bleu_13a']:>6}  ({v['seconds']}s)")
    print(f"[learned PE 外推] → {out}（正弦侧同协议对照：拼接 9.03 / 天然 6.62）")


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="ch10 消融网格：13 配置复用 ch09 公平协议；增量落盘 + 断点续跑")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--epochs", type=int, default=30, help="每配置 epochs（协议=30；冒烟自动改 2）")
    ap.add_argument("--smoke", action="store_true", help="冒烟档：仅 base seed0、2 epochs，跑通训练+评测+落盘全链")
    ap.add_argument("--extrap-learnedpe", action="store_true",
                    help="只跑 learned PE 外推补测（评已训 grid_learnedpe_s0，ch9 外推同协议，"
                         "落 learnedpe_extrap.json；不动网格。learned PE 为 18×256 查表，"
                         "band 内位置下标 clamp 到末位——形状口径见 evaluate_extrap_learnedpe() docstring）")
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(LOG_DIR, exist_ok=True)

    if args.extrap_learnedpe:
        evaluate_extrap_learnedpe(device)
        return

    t0 = time.perf_counter()
    sp = ensure_bpe(CH9_LOG_DIR)  # 「同词表」协议：直接复用 ch09 已训 BPE，不另训
    raw = load_raw()
    kept, keep_rate = filter_pairs(sp, raw["train"])
    data = {"train_ids": encode_pairs(sp, kept), "val_ids": encode_pairs(sp, raw["validation"]),
            "test_pairs": raw["test"], "train_pairs": len(kept), "keep_rate": round(keep_rate, 4)}
    print(f"[数据] 复用 ch09 BPE（词表 {sp.get_piece_size()}）· train {len(kept)}（{keep_rate:.1%}）· "
          f"val {len(data['val_ids'])} · test {len(raw['test'])} · 准备 {time.perf_counter() - t0:.1f}s", flush=True)

    if args.smoke:
        grid, epochs = [dict(GRID[0], run_id="smoke_base_s0")], 2
    else:
        grid, epochs = GRID, args.epochs

    variant_selftest(sp, device)

    res = load_results()
    keep_ids = {c["run_id"] for c in grid}
    res["runs"] = [r for r in res["runs"] if r["run_id"] in keep_ids]  # 剔除不在当前网格的记录（如冒烟残留）
    res.pop("summary", None)
    if args.smoke:
        res["smoke"] = True
        res.pop("protocol", None)
    else:
        res.pop("smoke", None)
        res["protocol"] = {
            "pipeline": "ch09 公平协议原样复用（data/model/train/decode/eval）；变体 wrapper 见 ablation.py 头注",
            "bpe": f"shared BPE-8k 复用 log/Book1-ch09/bpe_8k.model（词表 {sp.get_piece_size()}）",
            "train_pairs": len(kept), "keep_rate": round(keep_rate, 4), "max_train_pieces": MAX_TRAIN_PIECES,
            "batch": BATCH, "epochs_per_config": epochs,
            "optim": f"Noam(warmup={WARMUP}, factor=1.0, d_model={D_MODEL}) + Adam(β2=0.98, eps=1e-9)，ls=0.1",
            "checkpoint": "末 epoch（Table 3 口径：无 averaging、不挑 best-val）",
            "eval": f"beam={BEAM} α={ALPHA} 全 test {len(raw['test'])} 句 · sacreBLEU 13a · 句长桶 ≤7/8-13/≥14（参考句词数）",
            "metric_caveat": "绝对值与 Table 3 不可比（Multi30K≠WMT14，算力 ~1/100）；PPL=per-wordpiece",
        }
    save_results(res)

    done = {r["run_id"] for r in res["runs"] if "bleu_13a" in r}  # 仅成功记录算完成（失败/中断行重启重跑）
    t_grid, n_done = time.perf_counter(), 0
    for i, cfg in enumerate(grid, 1):
        if cfg["run_id"] in done:
            print(f"[续跑] 跳过已完成：{cfg['run_id']}", flush=True)
            continue
        print(f"\n===== [{i}/{len(grid)}] {cfg['run_id']} =====", flush=True)
        try:
            rec = run_config(cfg, epochs, device, sp, data)
        except Exception as e:  # 长跑护栏：单行崩溃不拖垮整网格；错误入 JSON，重启自动重试该行
            import traceback
            err = f"{type(e).__name__}: {e}"
            print(f"[{cfg['run_id']}] 异常（已记录，继续下一行）：{err}\n{traceback.format_exc()}", flush=True)
            res["runs"] = [r for r in res["runs"] if r["run_id"] != cfg["run_id"]]
            res["runs"].append({"run_id": cfg["run_id"], "row": cfg["row"], "error": err,
                                "finished": dt.datetime.now().isoformat(timespec="seconds")})
            save_results(res)
            continue
        res["runs"] = [r for r in res["runs"] if r["run_id"] != cfg["run_id"]]
        res["runs"].append(rec)
        save_results(res)  # 每配置完成即落盘
        n_done += 1
        el = time.perf_counter() - t_grid
        left = len(grid) - i
        print(f"[{cfg['run_id']}] 完成：BLEU(13a) {rec['bleu_13a']} · 末 epoch PPL(piece) {rec['final_val_ppl_piece']} · "
              f"wall {rec['wall_clock_seconds']}s · 网格已用 {el / 60:.1f} min"
              + (f" · 剩余 ~{left * el / n_done / 60:.0f} min" if left and n_done else ""), flush=True)

    if not args.smoke:
        summarize(res)
    res["grid_seconds_total"] = round(time.perf_counter() - t_grid, 1)
    save_results(res)
    print(f"\n[结束] {len(res['runs'])}/{len(grid)} 配置在档 → {RESULTS_JSON}"
          f"{'（冒烟档：验证后请删除本 JSON 与 grid_smoke_*.pt）' if args.smoke else ''}", flush=True)


if __name__ == "__main__":
    main()
