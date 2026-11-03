# -*- coding: utf-8 -*-
# 用途：Book1 第 9 章——推理解码：贪心 + beam search（beam=4、GNMT 式长度惩罚 α=0.6）
# 所属章节：《Transformer 原典》第 9 章 9.6；beam 组件讲解见 ch8 8.5
# 运行（库：被 eval.py / gru_baseline.py 导入；CLI 冒烟）：
#   source env.sh && python "code/ch09/decode.py" --ckpt transformer --n 5
# 模型统一接口（Transformer 与 GRU 双 baseline 都实现）：encode(src)→state、decode_step(state, ys)→(B,V) logits。
#   decode_step 整段重算（无增量缓存）——「每生成一个 token 重算整段解码器」，2017 口径（ch7 欠账 F4 的活体）。
import argparse
import os
import time

import torch
import torch.nn.functional as F

from data import (LOG_DIR, PAD, SOS, EOS, ensure_bpe, load_raw, encode_sent, encode_pairs,
                  batches, ids_to_text, DECODE_MAX_PIECES)

BEAM, ALPHA = 4, 0.6  # 原论文 §5.2：beam=4、α=0.6（长度惩罚 GNMT eq(14) 直系血统）


def length_penalty(length: int, alpha: float = ALPHA) -> float:
    """GNMT eq(14)：lp(Y) = (5+|Y|)^α / 6^α。|Y| = 已生成 token 数（含 EOS）。标量 → 标量。"""
    return ((5.0 + length) ** alpha) / 6.0 ** alpha


@torch.no_grad()
def greedy_decode(model, src, max_len: int = DECODE_MAX_PIECES):
    """贪心解码：每步取 argmax；已出 EOS 的行冻结为 EOS。
    src (B,S) → 返回 (B, ≤max_len) id 序列（SOS 起头）。model 须已 eval()。
    束状态即 state（B 行）与活路 ys (B,1)→(B,t)。"""
    state = model.encode(src)
    ys = torch.full((src.size(0), 1), SOS, dtype=torch.long, device=src.device)  # (B,1) 起步
    done = torch.zeros(src.size(0), 1, dtype=torch.bool, device=src.device)
    for _ in range(max_len - 1):
        logits = model.decode_step(state, ys)   # (B,t) -> (B,V)
        nxt = logits.argmax(-1, keepdim=True)   # (B,1)
        nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
        ys = torch.cat([ys, nxt], 1)            # (B,t+1)
        done = done | (nxt == EOS)
        if done.all():
            break
    return ys


@torch.no_grad()
def beam_search(model, src, k: int = BEAM, alpha: float = ALPHA, max_len: int = DECODE_MAX_PIECES):
    """批量 beam search。打分：score(Y) = logP(Y) / lp(|Y|)（§5.2 / GNMT eq(14)）。
    束状态形状：state 沿 batch 复制 k 份（memory (B,S,d)→(B·k,S,d)）；活路 ys (B·k, t)；
    累积 logP scores (B,k)；每步候选 cand (B,k,V) 展平成 (B,k·V) 取 top-2k；返回 (B, ≤max_len)（批内补 PAD）。
    实现要点：存活 beam 长度同步增长 → 步内 lp 相同，按原始 logP 剪枝与按惩罚分剪枝等价；
    EOS 候选立即定稿入 finished（记各自长度的惩罚分），每句凑满 k 个定稿或达 max_len 停。"""
    model.eval()
    b, device = src.size(0), src.device
    state = _expand_state(model.encode(src), k)  # state 的 batch 维扩成 b*k 份

    ys = torch.full((b * k, 1), SOS, dtype=torch.long, device=device)  # (B·k, 1) 活路序列
    scores = torch.full((b, k), -1.0e18, device=device)                # (B,k) 累积 logP
    scores[:, 0] = 0.0                        # 起步只有 beam 0 存活
    alive = torch.ones(b, k, dtype=torch.bool, device=device)
    alive[:, 1:] = False
    finished = [[] for _ in range(b)]         # 每句定稿候选 (惩罚分, tokens)

    for t in range(max_len - 1):
        logits = model.decode_step(state, ys)                        # (B·k, t) -> (B·k, V)
        v = logits.size(-1)
        logp = F.log_softmax(logits.float(), dim=-1).view(b, k, v)    # (B·k,V) -> (B,k,V)
        cand = scores.unsqueeze(-1) + logp                            # (B,k,V) 累积 logP
        cand[~alive] = -1.0e18                                       # 死行不参选
        n_sel = min(2 * k, k * v)                                    # 取 top-2k：EOS 定稿 + k 个活行都有席位
        top_scores, top_idx = cand.view(b, k * v).topk(n_sel, dim=-1)  # (B,k·V) -> (B,2k)
        beam_idx, tokens = top_idx // v, top_idx % v                 # (B,2k)：来源束号 / 新 token
        src_alive = alive.gather(1, beam_idx)                        # (B,2k) 候选来源行是否存活
        sel_alive = src_alive & (tokens != EOS)                      # 可延续（非 EOS）候选
        eos_hit = src_alive & (tokens == EOS)                        # 定稿候选

        eos_i, eos_j = eos_hit.nonzero(as_tuple=True)
        for i, j in zip(eos_i.tolist(), eos_j.tolist()):             # 定稿（数量少，Python 循环开销可忽略）
            n_tok = t + 1                                            # 本迭代是第 t+1 个生成 token（含此 EOS）
            finished[i].append((top_scores[i, j].item() / length_penalty(n_tok, alpha),
                                ys[beam_idx[i, j] + i * k, 1:].tolist() + [int(tokens[i, j])]))

        # 活行继承：按得分序取前 k 个非 EOS 候选 → 新 slot（向量化的「第 j 个 True」定位）
        cs = torch.cumsum(sel_alive.int(), dim=1)                    # (B,2k) 第 j 个可延续候选的累计位
        slots = torch.arange(k, device=device)
        hit = (cs.unsqueeze(1) == (slots + 1).view(1, k, 1)) & sel_alive.unsqueeze(1)  # (B,k,2k)
        col = torch.argmax(hit.int(), dim=2)                         # (B,k)：slot j 的来源列
        has = (slots < sel_alive.sum(1, keepdim=True))               # (B,k) 该 slot 是否有候选
        src_row = (beam_idx.gather(1, col) + torch.arange(b, device=device)[:, None] * k).view(-1)
        src_row = torch.where(has.view(-1), src_row, torch.arange(b * k, device=device))  # 空位指回自身
        ys = torch.cat([ys.index_select(0, src_row), tokens.gather(1, col).view(-1, 1)], 1)  # (B·k, t+1)
        scores = torch.where(has, top_scores.gather(1, col), torch.full_like(top_scores[:, :k], -1.0e18))
        alive = has
        if all(len(f) >= k for f in finished):
            break

    out = []
    for i in range(b):
        if len(finished[i]) < k:                                      # max_len 截断：活行定稿
            for j in range(k):
                if alive[i, j]:
                    finished[i].append((scores[i, j].item() / length_penalty(ys.size(1) - 1, alpha),
                                        ys[i * k + j, 1:].tolist()))
        best = max(finished[i], key=lambda x: x[0])[1] if finished[i] else [EOS]
        out.append([SOS] + best)

    m = max(len(x) for x in out)
    return torch.tensor([x + [PAD] * (m - len(x)) for x in out], dtype=torch.long, device=device)  # (B, m)


def _expand_state(state, k):
    """encode() 返回的 state 沿 batch 维复制 k 份：张量 (B,…) → (B·k,…)（Transformer 的
    (mem, mask) 元组与 GRU 的张量通吃；bool 掩码同样按束复制）。"""
    if isinstance(state, tuple):
        return tuple(_expand_state(s, k) for s in state)
    if torch.is_tensor(state):
        return state.repeat_interleave(k, dim=0)
    return state


def decode_corpus(model, sp, raw_pairs, device, k: int = BEAM, alpha: float = ALPHA, batch_size: int = 64,
                  max_pieces: int = DECODE_MAX_PIECES):
    """整集解码（beam；k=1 走贪心）：每批 src (B,m)（m=批内两侧最长 id 长，解码只消费源侧）
    → 返回 (假设 id 列表, wall-clock 秒)。"""
    id_pairs = encode_pairs(sp, raw_pairs, cap=max_pieces)
    hyps, t0 = [], time.perf_counter()
    for src, _ in batches(id_pairs, batch_size, shuffle=False):
        src = src.to(device)
        out = beam_search(model, src, k=k, alpha=alpha) if k > 1 else greedy_decode(model, src)
        hyps.extend(out.tolist())
        if device.type == "mps":
            torch.mps.synchronize()
    return hyps, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description="ch9 解码冒烟：加载 checkpoint，前 N 句贪心+beam 对照")
    ap.add_argument("--ckpt", default="transformer", help="log/Book1-ch09/ 下的 tag")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    args = ap.parse_args()
    device = torch.device(args.device)

    from model import Transformer
    ckpt = torch.load(os.path.join(LOG_DIR, f"{args.ckpt}.pt"), map_location="cpu", weights_only=False)
    model = Transformer(ckpt["config"]["vocab"],
                        **{k2: v for k2, v in ckpt["config"].items() if k2 != "vocab"}).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    sp = ensure_bpe()
    print(f"[checkpoint] tag={ckpt['tag']} steps={ckpt['steps']} config={ckpt['config']}")
    for en, de in load_raw()["test"][: args.n]:
        ids = torch.tensor([encode_sent(sp, en)], device=device)
        g = ids_to_text(sp, greedy_decode(model, ids)[0].tolist())
        bm = ids_to_text(sp, beam_search(model, ids)[0].tolist())
        print(f"  src   : {en}\n  ref   : {de}\n  greedy: {g}\n  beam4 : {bm}\n")


if __name__ == "__main__":
    main()
