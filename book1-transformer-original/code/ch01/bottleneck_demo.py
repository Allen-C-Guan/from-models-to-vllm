# -*- coding: utf-8 -*-
"""Book1 第 1 章微实验：固定上下文向量的容量墙（seq2seq 瓶颈演示）。

用途：
  在合成「随机符号序列的变换映射」任务上，训练无注意力的小型 GRU encoder-decoder——
  源序列信息只经一个固定维度的上下文向量 c（编码器末步隐状态）转运——再按序列长度
  分桶评测，产出「质量-长度」衰减曲线（图 1.4）：本书自己的固定向量容量墙证据，
  对读 Sutskever Fig.3 / Bahdanau Fig.2 双证（正文 1.2 与 1.6 节）。
  评测双协议：
    A 内容召回（主指标）——长度由参考侧给定（与 BLEU 对参考译文计分同理），解码器
      逐位在内容符号内取 argmax，统计逐位准确率与「整句内容全对率」；
    B 整句精确率（严格指标）——自由贪心解码，须自行产出全部符号并停在对的位置。
所属章节：drafts/Book1-Transformer原典/ch01-前史动机.md
运行方式：source env.sh && python code/ch01/bottleneck_demo.py
          （CPU 约 2 分钟；数据落 log/Book1-ch01/，图落 drafts/.../figures/）
"""
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(2017)  # 固定种子：训练与采样全程可复现

V = 10                          # 符号表大小：源符号 1..V
TRAIN_MAX, EVAL_MAX = 12, 30    # 训练长度上限 / 评测长度上限（评测超出训练范围）
BOS, EOS, PAD = 0, V + 1, V + 2
VOCAB = V + 3
STEPS, BATCH, LR, LR_DROP = 12000, 64, 1e-3, 3e-4
N_EVAL = 200                    # 每个长度的评测样本数（独立固定种子）
# 任务：固定变换映射（源符号 s 的「译文」为 PERM[s-1]，保持词序）。解码器每步只知道
# c 与已输出符号，必须从固定向量 c 中按位置读出整条源序列——「装不下」即露馅。
PERM = torch.randperm(V, generator=torch.Generator().manual_seed(2017)) + 1


def make_batch(lengths: torch.Tensor):
    """按长度采样源序列，返回 (x, y)。x = [符号序列, <EOS>, PAD...]（Sutskever 图 1 式
    源端句末标记）；y = [PERM 逐位映射, EOS, PAD...]。
    I/O 形状：lengths (B,) → x (B, n+1)、y (B, n+1)，均为词 id 整数张量（n=批内最大长度）。"""
    b, n = len(lengths), int(lengths.max())
    pos = torch.arange(n)
    x = torch.randint(1, V + 1, (b, n))
    x = torch.where(pos < lengths[:, None], x, torch.full_like(x, PAD))
    x = torch.cat([x, torch.full((b, 1), PAD)], 1)
    x[torch.arange(b), lengths] = EOS
    y = torch.full((b, n + 1), PAD)
    y[:, :n] = torch.where(pos < lengths[:, None], PERM[(x[:, :n] - 1).clamp(0, V - 1)], y[:, :n])
    y[torch.arange(b), lengths] = EOS
    return x, y


class EncDec(nn.Module):
    """无注意力 GRU encoder-decoder：整句压进编码器末步隐状态 c（「8000 个实数」的缩小版）；
    解码器以 c 为初始隐状态逐词展开（Sutskever §2 的 v 即此设计）。
    I/O 形状：x (B,Lx) 词 id、y_in (B,Ly) 词 id → logits (B,Ly,VOCAB)。"""

    def __init__(self, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, 32, padding_idx=PAD)
        self.enc = nn.GRU(32, hidden, batch_first=True)
        self.dec = nn.GRU(32, hidden, batch_first=True)
        self.out = nn.Linear(hidden, VOCAB)

    def forward(self, x, y_in):
        _, h = self.enc(self.emb(x))               # (B,Lx)→(B,Lx,32)，末步 h [1,B,hidden] = 固定上下文向量 c
        logits, _ = self.dec(self.emb(y_in), h)    # (B,Ly,32)＋c 作初始状态 → (B,Ly,hidden)：逐词展开
        return self.out(logits)                    # (B,Ly,hidden)→(B,Ly,VOCAB)

    @torch.no_grad()
    def greedy(self, x, max_len: int, free: bool):
        """逐步贪心解码。free=True 为协议 B（全词表，含 EOS）；free=False 为协议 A
        （屏蔽 EOS/PAD/BOS，长度由调用方给定——只考内容召回）。
        I/O 形状：x (B,Lx) 词 id → 输出 (B,max_len) 词 id（自回归：逐步取 argmax 回喂）。"""
        _, h = self.enc(self.emb(x))               # (B,Lx)→(B,Lx,32)→h [1,B,hidden] = c
        tok = torch.full((len(x), 1), BOS)         # (B,1)：首步回喂起始符 <BOS>
        outs = []
        for _ in range(max_len):
            o, h = self.dec(self.emb(tok), h)       # 一步 GRU：tok (B,1)→(B,1,32)，同时产出本步输出与新状态
            logits = self.out(o[:, -1])             # (B,hidden)→(B,VOCAB)
            if not free:
                logits[:, [BOS, EOS, PAD]] = float("-inf")
            tok = logits.argmax(-1, keepdim=True)   # (B,VOCAB)→(B,1)：抽中的词回喂下一步
            outs.append(tok)
        return torch.cat(outs, 1)


def train(hidden: int):
    """训练一个给定 c 维度的模型，返回 (模型, 训练秒数, 末段损失)。"""
    model = EncDec(hidden)
    opt = torch.optim.Adam(model.parameters(), LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    t0, loss = time.time(), None
    for step in range(STEPS):
        if step == int(STEPS * 2 / 3):              # 后 1/3 降学习率，收敛更稳
            for group in opt.param_groups:
                group["lr"] = LR_DROP
        x, y = make_batch(torch.randint(1, TRAIN_MAX + 1, (BATCH,)))
        y_in = torch.cat([torch.full((len(x), 1), BOS), y[:, :-1]], 1)  # y (B,n+1) → y_in (B,n+1)：右移一位（teacher forcing）
        loss = loss_fn(model(x, y_in).reshape(-1, VOCAB), y.reshape(-1))  # logits (B,L,VOCAB)→(B*L,VOCAB) 对 (B*L,)
        opt.zero_grad(), loss.backward(), opt.step()
    return model, time.time() - t0, float(loss.detach())


def evaluate(model: EncDec):
    """按长度 1..EVAL_MAX 双协议评测，返回逐长度指标字典。"""
    g = torch.Generator().manual_seed(1409)  # 评测采样独立固定种子
    res = {}
    for length in range(1, EVAL_MAX + 1):
        x = torch.randint(1, V + 1, (N_EVAL, length), generator=g)
        src = torch.cat([x, torch.full((N_EVAL, 1), EOS)], 1)   # (N_EVAL,length)→(N_EVAL,length+1)：补句末 EOS
        tgt = PERM[x - 1]                                       # 标准答案 (N_EVAL,length)
        pa = model.greedy(src, length, free=False)          # 协议 A：内容召回，出 (N_EVAL,length)
        pb = model.greedy(src, EVAL_MAX + 4, free=True)     # 协议 B：自由解码，出 (N_EVAL,EVAL_MAX+4)
        ok_a = pa == tgt
        ok_b = pb[:, :length] == tgt
        res[length] = {
            "token": ok_a.float().mean().item(),            # A：逐位准确率
            "seq": ok_a.all(1).float().mean().item(),       # A：整句内容全对率
            "exact": (ok_b.all(1) & (pb[:, length] == EOS)).float().mean().item(),  # B
        }
    return res


def main() -> None:
    results = {}
    for hidden in (32, 128):  # 两个容量档：c 维度越小，墙出现越早
        model, sec, loss = train(hidden)
        results[f"hidden={hidden}"] = {"train_seconds": round(sec, 1), "final_loss": round(loss, 4),
                                       **{k: {m: round(v, 3) for m, v in d.items()}
                                          for k, d in evaluate(model).items()}}
        print(f"[hidden={hidden}] 训练 {sec:.0f}s × {STEPS} 步，末段损失 {loss:.4f}")

    # 分桶表（正文 1.6 节分桶数字的数据源；桶宽 5，桶内对各长度取平均）
    heads = [("seq", "句内容全对(A)"), ("token", "逐位准确(A)"), ("exact", "整句精确(B)")]
    print("句长桶   " + "".join(f"{n}(h={h})".rjust(14) for h in (32, 128) for _, n in heads))
    for lo in range(1, EVAL_MAX, 5):
        hi = min(lo + 4, EVAL_MAX)
        cells = "".join(f"{sum(results[f'hidden={h}'][l][m] for l in range(lo, hi + 1)) / (hi - lo + 1):.3f}".rjust(14)
                        for h in (32, 128) for m, _ in heads)
        print(f"{lo:>2}-{hi:<4}{cells}")

    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "log", "Book1-ch01"))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "bottleneck_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    # 图 1.4：质量-长度衰减曲线（印刷向规格见 _写作规范.md 第 4 节）
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#898781",
        "axes.labelcolor": "#52514e", "xtick.color": "#898781", "ytick.color": "#898781",
        "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.5, "axes.axisbelow": True,
        "font.family": "DejaVu Sans",
    })
    fig, ax = plt.subplots(figsize=(6.4, 4))
    lengths = list(range(1, EVAL_MAX + 1))
    for hidden, color in ((32, "#eb6834"), (128, "#2a78d6")):
        ax.plot(lengths, [results[f"hidden={hidden}"][l]["seq"] for l in lengths], "-o",
                color=color, linewidth=2, markersize=4, label=f"c dim = {hidden}")
    ax.axvline(TRAIN_MAX, color="#52514e", linestyle="--", linewidth=1)
    ax.annotate("train length cap = 12", xy=(TRAIN_MAX, 0.84), xytext=(14.5, 0.84),
                fontsize=8, color="#52514e")
    ax.set_xlabel("sequence length", fontsize=9)
    ax.set_ylabel("all-content-correct rate (protocol A)", fontsize=9)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig_path = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "drafts", "Book1-Transformer原典",
        "figures", "fig-1-4-capacity-wall-curve.png"))
    fig.savefig(fig_path, dpi=300)
    print(f"图已保存：{fig_path}\n数据已保存：{os.path.join(out, 'bottleneck_results.json')}")


if __name__ == "__main__":
    main()
