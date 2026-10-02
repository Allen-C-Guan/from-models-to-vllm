# -*- coding: utf-8 -*-
# 用途：Book1 第 9 章——GRU 双 baseline：无 attention 版（ch1 固定向量瓶颈的活体）与 Bahdanau 版（缓解的活体）
# 所属章节：《Transformer 原典》第 9 章 9.5；结构出处 Cho et al. 1406.1078 / Bahdanau et al. 1409.0473
# 运行：source env.sh && python "code/ch09/gru_baseline.py" --variant both --device mps --epochs 30
#
# 公平协议（与 Transformer 对照）：
#   ① 同一 shared BPE-8k 词表、同一嵌入维度 256、三处权重共享（编码/解码输入嵌入 + 输出投影经 256 维适配后共享）；
#   ② 同一训练数据（L=16 过滤后 17555 对）、同 batch=64、同 epochs → 同 token 预算；
#   ③ 同优化器（Adam β2=0.98 + Noam warmup 4000，d 取 256）同 label smoothing 0.1；
#   ④ 非嵌入参数对齐到与 Transformer（7.37M）同量级：noattn H=896 → ~8.8M；Bahdanau 双向 2×576 → ~8.1M；
#   ⑤ 同推理协议（beam=4、α=0.6，与 decode.py 同一代码路径）。
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import LOG_DIR, PAD, ensure_bpe, load_raw, encode_pairs, filter_pairs, MAX_TRAIN_PIECES, BATCH
from train import NoamLR, run_epoch, validate, WARMUP

D_EMB = 256  # 与 Transformer 的 d_model 对齐（共享嵌入的维度）


class GRUNoAttn(nn.Module):
    """Cho 1406.1078 的 RNN Encoder–Decoder（无 attention）：整句压进 c = 编码器最后隐状态，
    c 注入每个解码步——ch1「固定向量装不下整句」的活体展示。
    形状：encode src (B,S) → state (B,1,H)；decode tgt_in (B,T) → logits (B,T,V)。"""

    def __init__(self, vocab: int, hidden: int = 896, p_drop: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab, D_EMB, padding_idx=PAD)
        self.encoder = nn.GRU(D_EMB, hidden, batch_first=True)          # 单向：h_T 即句向量
        self.decoder = nn.GRU(D_EMB + hidden, hidden, batch_first=True)  # 输入 [emb(y_{t-1}); c]
        self.out_proj = nn.Linear(hidden, D_EMB)                        # 适配到嵌入维，输出投影共享嵌入
        self.emb_drop = nn.Dropout(p_drop)

    def encode(self, src):
        """src (B,S) → state (B,1,H)。注意末态吃完全部位置（含尾部 PAD）——无掩码可贴，组批税的根源。"""
        _, h_n = self.encoder(self.emb_drop(self.embed(src)))           # h_n: (1, B, H)
        return h_n.transpose(0, 1).contiguous()                         # → (B, 1, H)：batch 在前，beam 可沿 dim0 复制

    def decode(self, tgt_in, state):
        """tgt_in (B,T) + state (B,1,H) → logits (B,T,V)。"""
        h0 = state.transpose(0, 1).contiguous()                         # → (1, B, H)
        c = state[:, -1]                                                # (B, H)：整句压缩成的固定向量
        emb = self.emb_drop(self.embed(tgt_in))                         # (B, T, 256)
        dec_in = torch.cat([emb, c.unsqueeze(1).expand(-1, tgt_in.size(1), -1)], dim=-1)  # (B,T,256+H)
        out, _ = self.decoder(dec_in, h0)                               # s_0 = h_T，且 c 逐步注入；(B,T,H)
        return self.out_proj(out) @ self.embed.weight.t()               # (B,T,H)->(B,T,256)@(256,V)->(B,T,V)

    def forward(self, src, tgt_in):
        """src (B,S) + tgt_in (B,T) → logits (B,T,V)。"""
        return self.decode(tgt_in, self.encode(src))

    def decode_step(self, state, ys):
        """ys (B,t) → (B,V)：整段重算取末位（与 Transformer 同接口）。"""
        return self.decode(ys, state)[:, -1]


class GRUBahdanau(nn.Module):
    """Bahdanau 1409.0473：双向 GRU 编码 + 加性注意力。e_ij = v^T tanh(W s_{i-1} + U h_j)、
    c_i = Σ_j α_ij h_j，解码步输入 [emb(y_{t-1}); c_i]——「缓解但不痊愈」的活体展示。
    形状：encode src (B,S) → state=(enc_out (B,S,2H), h_final (B,2H), enc_mask (B,S))；
    decode tgt_in (B,T) → logits (B,T,V)；α 打分表 (B,S)——无头维，全模型单头。"""

    def __init__(self, vocab: int, hidden: int = 576, attn_dim: int = 576, p_drop: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab, D_EMB, padding_idx=PAD)
        self.encoder = nn.GRU(D_EMB, hidden, batch_first=True, bidirectional=True)  # 编码输出 2H
        self.w_dec = nn.Linear(hidden, attn_dim)          # 对 s_{i-1}
        self.w_enc = nn.Linear(2 * hidden, attn_dim)      # 对 h_j
        self.v = nn.Linear(attn_dim, 1, bias=False)       # 打分向量（加性＝Luong 术语的 general 前身之争见 ch2）
        self.cell = nn.GRUCell(D_EMB + 2 * hidden, hidden)
        self.s0_proj = nn.Linear(2 * hidden, hidden)      # s_0 = tanh(W h_T)（Bahdanau 原式）
        self.out_proj = nn.Linear(hidden, D_EMB)
        self.emb_drop = nn.Dropout(p_drop)

    def encode(self, src):
        """src (B,S) → (enc_out (B,S,2H), h_final (B,2H), enc_mask (B,S))。"""
        emb = self.emb_drop(self.embed(src))              # (B,S,256)
        out, h_n = self.encoder(emb)                       # out: (B,S,2H)；h_n: (2,B,H)
        h_final = torch.cat([h_n[0], h_n[1]], dim=-1)      # 前向末 + 后向末 → (B, 2H)
        return (out, h_final, src != PAD)

    def _attend(self, s_prev, enc_out, enc_mask):
        """s_prev (B,H) → (c_i (B,2H), α (B,S))：加性打分 → softmax → 加权求和（式 (4.4)/(1.3)）。"""
        e = self.v(torch.tanh(self.w_dec(s_prev).unsqueeze(1) + self.w_enc(enc_out))).squeeze(-1)  # (B,S)
        e = e.masked_fill(~enc_mask, float("-inf"))
        a = F.softmax(e, dim=-1)                           # (B,S)，PAD 位经 -inf 归零
        return (a.unsqueeze(-1) * enc_out).sum(1), a       # c_i: (B,2H)，α: (B,S)

    def decode(self, tgt_in, state):
        """tgt_in (B,T) + state → logits (B,T,V)：逐步 attend 再转移（串行，无并行红利）。"""
        enc_out, h_final, enc_mask = state
        s = torch.tanh(self.s0_proj(h_final))              # (B,H)
        outs = []
        for t in range(tgt_in.size(1)):                    # RNN 解码本质串行：逐步 attend 再转移
            c, _ = self._attend(s, enc_out, enc_mask)      # c: (B,2H)
            inp = torch.cat([self.emb_drop(self.embed(tgt_in[:, t])), c], dim=-1)  # (B,256+2H)
            s = self.cell(inp, s)                          # (B,H)
            outs.append(self.out_proj(s))                  # (B,256)
        return torch.stack(outs, dim=1) @ self.embed.weight.t()  # (B,T,256)@(256,V) -> (B,T,V)

    def forward(self, src, tgt_in):
        """src (B,S) + tgt_in (B,T) → logits (B,T,V)。"""
        return self.decode(tgt_in, self.encode(src))

    def decode_step(self, state, ys):
        """ys (B,t) → (B,V)：整段重算取末位（与 Transformer 同接口）。"""
        return self.decode(ys, state)[:, -1]

    def attention_weights(self, src, tgt):
        """导出对齐矩阵 α（图 9.2 素材，Bahdanau Fig.3 的真模型重演）。src (1,S)、tgt (1,T) → (T,S)。"""
        enc_out, h_final, enc_mask = self.encode(src)
        s = torch.tanh(self.s0_proj(h_final))
        alphas = []
        for t in range(tgt.size(1)):
            c, a = self._attend(s, enc_out, enc_mask)      # a: (1,S)
            s = self.cell(torch.cat([self.emb_drop(self.embed(tgt[:, t])), c], dim=-1), s)
            alphas.append(a)
        return torch.stack(alphas, dim=1)[0].detach().cpu()  # (T,S)


ARCHS = {"noattn": (GRUNoAttn, dict(hidden=896)), "bahdanau": (GRUBahdanau, dict(hidden=576, attn_dim=576))}


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def train_gru(variant: str, epochs: int, device_str: str, log_dir: str = LOG_DIR):
    assert variant in ARCHS
    device = torch.device(device_str)
    torch.manual_seed(2017)
    sp = ensure_bpe(log_dir)
    raw = load_raw()
    kept, keep_rate = filter_pairs(sp, raw["train"])
    train_ids = encode_pairs(sp, kept)
    val_ids = encode_pairs(sp, raw["validation"])

    cls, kw = ARCHS[variant]
    model = cls(sp.get_piece_size(), **kw).to(device)
    print(f"[模型] gru_{variant}：参数 {n_params(model):,}"
          f"（非嵌入 {n_params(model) - model.embed.weight.numel():,}，配置 {kw}）")
    opt = torch.optim.Adam(model.parameters(), lr=1e-7, betas=(0.9, 0.98), eps=1e-9)
    sched = NoamLR(opt, D_EMB, WARMUP)  # 公平协议：与 Transformer 同调度形状（d 取共享嵌入维 256）

    tag = f"gru_{variant}"
    log = {"tag": tag, "device": device_str, "epochs": epochs, "vocab": sp.get_piece_size(),
           "params": n_params(model), "config": f"gru_{variant},{kw},ls=0.1", "train_pairs": len(train_ids),
           "keep_rate": round(keep_rate, 4), "max_train_pieces": MAX_TRAIN_PIECES, "batch": BATCH,
           "warmup": WARMUP, "history": []}
    t_all = time.perf_counter()
    for epoch in range(epochs):
        t0 = time.perf_counter()
        tr_loss, _ = run_epoch(model, train_ids, sched, device, epoch)
        val_ls, val_nll = validate(model, val_ids, device)
        if device.type == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
        log["history"].append({"epoch": epoch + 1, "train_loss": round(tr_loss, 4), "val_loss": round(val_ls, 4),
                               "val_nll": round(val_nll, 4), "val_ppl_piece": round(math.exp(val_nll), 3),
                               "lr": round(sched.rate(), 8), "seconds": round(dt, 1)})
        print(f"[{tag} epoch {epoch + 1:>2}/{epochs}] train {tr_loss:.3f}  val {val_ls:.3f}  "
              f"PPL(piece) {math.exp(val_nll):7.3f}  {dt:5.1f}s", flush=True)

    torch.save({"tag": tag, "model": model.state_dict(), "arch": tag,
                "config": dict(vocab=sp.get_piece_size(), **kw), "steps": sched.step_num, "log": log},
               os.path.join(log_dir, f"{tag}.pt"))
    log["total_seconds"] = round(time.perf_counter() - t_all, 1)
    log["final_val_loss"], log["final_val_nll"] = round(val_ls, 4), round(val_nll, 4)
    with open(os.path.join(log_dir, f"{tag}_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"[完成] gru_{variant} checkpoint → {os.path.join(log_dir, tag + '.pt')}；总时长 {log['total_seconds']}s")


def main():
    ap = argparse.ArgumentParser(description="ch9 GRU 双 baseline 训练（公平协议见文件头）")
    ap.add_argument("--variant", default="both", choices=["noattn", "bahdanau", "both"])
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    for v in (["noattn", "bahdanau"] if args.variant == "both" else [args.variant]):
        train_gru(v, args.epochs, args.device)


if __name__ == "__main__":
    main()
