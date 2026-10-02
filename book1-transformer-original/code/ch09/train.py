# -*- coding: utf-8 -*-
# 用途：Book1 第 9 章——Transformer 训练循环：Noam 调度 + label smoothing + Adam(β2=0.98)，checkpoint 落 log/
# 所属章节：《Transformer 原典》第 9 章（从零实现与小型翻译实验）；配方依据见 ch8
# 运行：source env.sh && python "code/ch09/train.py" --device mps --epochs 30
#       （fast 冒烟档：--epochs 2；CPU 约 6 倍时长）
# 配方（原论文 §5 / ch8）：Noam 公式 lrate = d_model^{-0.5}·min(step^{-0.5}, step·warmup^{-1.5})，
#   warmup=4000、factor=1.0（论文口径）；label smoothing ε=0.1（§5.4）；Adam β2=0.98、eps=1e-9。
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from data import (LOG_DIR, PAD, ensure_bpe, load_raw, encode_pairs, filter_pairs, batches,
                  MAX_TRAIN_PIECES, BATCH)
from model import Transformer, n_params, N_LAYER, D_MODEL, HEADS, D_FF, P_DROP

WARMUP = 4000


class NoamLR:
    """原论文式 (3) 的 Noam 调度（ch8 8.1）。factor=1、warmup=4000 为论文 base 口径。
    之所以先升后降：post-LN 下初始大学习率会放大残差和的尺度抖动——warmup 是「止痛药」（欠账 F5）。"""

    def __init__(self, optimizer, d_model: int, warmup: int = WARMUP, factor: float = 1.0):
        self.opt, self.d, self.warmup, self.factor, self.step_num = optimizer, d_model, warmup, factor, 0

    def rate(self, step=None) -> float:
        step = step or self.step_num
        return self.factor * self.d ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))

    def step(self):
        self.step_num += 1
        for g in self.opt.param_groups:
            g["lr"] = self.rate()
        self.opt.step()


def label_smoothing_loss(logits, tgt_out, eps: float = 0.1, pad: int = PAD):
    """§5.4：ε=0.1 的 label smoothing 交叉熵，pad 位不计。
    logits (B,T,V) + tgt_out (B,T) → 标量。展平须用 .reshape（tgt[:, 1:] 切片非连续，.view 报错）。"""
    logits = logits.reshape(-1, logits.size(-1)).float()  # (B,T,V) -> (B·T,V)；探针踩坑：.reshape 不用 .view
    tgt_out = tgt_out.reshape(-1)                          # (B,T) -> (B·T,)
    keep = tgt_out != pad                                  # (B·T,) 填充位不计
    lsm = F.log_softmax(logits, dim=-1)
    nll = -lsm.gather(1, tgt_out.unsqueeze(1)).squeeze(1)  # (B·T,)
    smooth = -lsm.mean(dim=1)                              # (B·T,)
    loss = (1 - eps) * nll + eps * smooth                  # (B·T,) 每位损失
    return loss[keep].sum() / keep.sum()                   # pad 掩蔽求和 / 有效位数 -> 标量


def nll_loss(logits, tgt_out, pad: int = PAD):
    """无平滑 NLL（per-piece 口径）——评测期报告 PPL 用（Table 3 的 PPL 亦 per-wordpiece）。
    logits (B,T,V) + tgt_out (B,T) → 标量（有效 piece 平均）。"""
    logits = logits.reshape(-1, logits.size(-1)).float()   # (B,T,V) -> (B·T,V)
    tgt_out = tgt_out.reshape(-1)                          # (B,T) -> (B·T,)
    keep = tgt_out != pad
    nll = F.cross_entropy(logits[keep], tgt_out[keep], reduction="sum")
    return nll / keep.sum()


def run_epoch(model, id_pairs, opt_sched, device, epoch, clip=None):
    """一个训练 epoch：每步 src/tgt (B,m) → logits (B,m-1,V) → 标量损失 → 反传 → NoamLR.step()。"""
    model.train()
    total_loss, total_tok, n_batch = 0.0, 0, 0
    for src, tgt in batches(id_pairs, BATCH, shuffle=True, epoch=epoch):
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src, tgt[:, :-1])      # tgt (B,m) -> 输入侧 (B,m-1)；logits (B,m-1,V)
        loss = label_smoothing_loss(logits, tgt[:, 1:])  # 标签侧 (B,m-1) -> 标量
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
def validate(model, id_pairs, device):
    """验证：与 run_epoch 同形状口径（教师强制整句前向），返回（平滑损失，每 piece NLL）两标量。"""
    model.eval()
    ls, nl, tok = 0.0, 0.0, 0
    for src, tgt in batches(id_pairs, BATCH, shuffle=False):
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src, tgt[:, :-1])                # (B,m-1,V)
        ls += label_smoothing_loss(logits, tgt[:, 1:]).item() * int((tgt[:, 1:] != PAD).sum())
        nl += nll_loss(logits, tgt[:, 1:]).item() * int((tgt[:, 1:] != PAD).sum())
        tok += int((tgt[:, 1:] != PAD).sum())
    return ls / tok, nl / tok  # （平滑损失，每 piece NLL）


def train_transformer(epochs: int, device_str: str, tag: str = "transformer", log_dir: str = LOG_DIR):
    device = torch.device(device_str)
    torch.manual_seed(2017)
    sp = ensure_bpe(log_dir)
    raw = load_raw()
    kept, keep_rate = filter_pairs(sp, raw["train"])  # 训练长度上限 L（外推实验基准）
    train_ids = encode_pairs(sp, kept)
    val_ids = encode_pairs(sp, raw["validation"])
    print(f"[数据] L={MAX_TRAIN_PIECES} 过滤后 train {len(train_ids)}（{keep_rate:.1%}），val {len(val_ids)}，"
          f"词表 {sp.get_piece_size()}")

    model = Transformer(sp.get_piece_size()).to(device)
    print(f"[模型] N={N_LAYER} d_model={D_MODEL} h={HEADS} d_ff={D_FF} p_drop={P_DROP}，"
          f"参数 {n_params(model):,}（非嵌入 {n_params(model) - model.embed.weight.numel():,}）")
    opt = torch.optim.Adam(model.parameters(), lr=1e-7, betas=(0.9, 0.98), eps=1e-9)  # §5.3：β2=0.98
    sched = NoamLR(opt, D_MODEL, WARMUP)

    log = {"tag": tag, "device": device_str, "epochs": epochs, "vocab": sp.get_piece_size(),
           "params": n_params(model), "config": f"N={N_LAYER},d={D_MODEL},h={HEADS},dff={D_FF},ls=0.1,drop={P_DROP}",
           "train_pairs": len(train_ids), "keep_rate": round(keep_rate, 4),
           "max_train_pieces": MAX_TRAIN_PIECES, "batch": BATCH, "warmup": WARMUP, "history": []}
    t_all = time.perf_counter()
    for epoch in range(epochs):
        t0 = time.perf_counter()
        tr_loss, n_batch = run_epoch(model, train_ids, sched, device, epoch)
        val_ls, val_nll = validate(model, val_ids, device)
        if device.type == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
        lr_now = sched.rate()
        log["history"].append({"epoch": epoch + 1, "train_loss": round(tr_loss, 4),
                               "val_loss": round(val_ls, 4), "val_nll": round(val_nll, 4),
                               "val_ppl_piece": round(math.exp(val_nll), 3),
                               "lr": round(lr_now, 8), "seconds": round(dt, 1)})
        print(f"[epoch {epoch + 1:>2}/{epochs}] train {tr_loss:.3f}  val {val_ls:.3f}  "
              f"PPL(piece) {math.exp(val_nll):7.3f}  lr {lr_now:.2e}  {dt:5.1f}s", flush=True)

    ckpt = {"tag": tag, "model": model.state_dict(), "arch": "transformer",
            "config": dict(n_layer=N_LAYER, d_model=D_MODEL, heads=HEADS, d_ff=D_FF, p_drop=P_DROP,
                           vocab=sp.get_piece_size()),
            "steps": sched.step_num, "log": log}
    ckpt_path = os.path.join(log_dir, f"{tag}.pt")
    torch.save(ckpt, ckpt_path)
    log["total_seconds"] = round(time.perf_counter() - t_all, 1)
    log["final_val_loss"], log["final_val_nll"] = round(val_ls, 4), round(val_nll, 4)
    log["final_val_ppl_piece"] = round(math.exp(val_nll), 3)
    with open(os.path.join(log_dir, f"{tag}_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"[完成] checkpoint → {ckpt_path}；总时长 {log['total_seconds']}s")
    return ckpt_path


def main():
    ap = argparse.ArgumentParser(description="ch9 Transformer 参考训练（Noam + label smoothing + Adam β2=0.98）")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tag", default="transformer", help="checkpoint 文件名（log/Book1-ch09/<tag>.pt）")
    args = ap.parse_args()
    train_transformer(args.epochs, args.device, args.tag)


if __name__ == "__main__":
    main()
