# 用途：Book1 第 4 章「三种注意力角色与双塔协作」配套实验——
#   ① 三种注意力角色（encoder 双向自注意 / decoder 因果自注意 / cross-attention）的
#      形状与掩码演练（手写 matmul+softmax，单头演示，不用 SDPA）；
#   ② 官方 nn.MultiheadAttention 的三种调用姿势（实现对照，非原论文内容）；
#   ③ 合成词典逐词映射任务上训练小型 GRU+Bahdanau 注意力模型（CPU 数分钟），
#      导出真实训练所得的注意力对齐热图（图 4.3）与关键数字。
# 所属章节：code/ch04/roles_demo.py
# 运行：source env.sh && python "code/ch04/roles_demo.py"
#      （固定种子 42，CPU 可跑；热图同时落 drafts/figures 与 log/ch04/）

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "drafts" / "Book1-Transformer原典" / "figures"
LOG_DIR = ROOT / "log" / "ch04"
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


# ---------- Part 1：手写缩放点积注意力（第 2 章式 (2.1)，单头演示） ----------
def attention(Q, K, V, causal=False):
    """缩放点积注意力（第 2 章式 (2.1)，单头演示）。

    输入：Q (..., n, d_k)、K (..., m, d_k)、V (..., m, d_v)；前面若有 B/h 维则一路保留。
    返回：(输出 (..., n, d_v), 注意力权重 A (..., n, m))。
    causal=True 时在 softmax 输入处把 j>i 的连接置 -inf
    （v7 §3.2.3 原文："masking out (setting to -inf) all values in the input of the
    softmax which correspond to illegal connections"）。"""
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (...,n,dk)×(...,dk,m) -> (...,n,m)
    if causal:
        n = Q.size(-2)
        nosee = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)  # (n,m) 禁区表，j>i 为真
        scores = scores.masked_fill(nosee, float("-inf"))  # 掩码沿 B/h 维广播，未来格置 -inf
    A = torch.softmax(scores, dim=-1)                    # 逐行 softmax，行和 1
    return A @ V, A                                      # (...,n,m)×(...,m,dv) -> (...,n,dv)


def demo_roles(n=8, d=16):
    """三种角色的形状/掩码演练：同一套 Q/K/V 机制，三处不同的喂法。

    演示为二维单头（无 B/h 维）：x_enc/y_dec 均为 (n, d)；三种权重 A_enc/A_dec 各 (n, n)、
    A_cross (n_tgt, n_tgt=src)（此处 n_tgt=n_src=n）。整批多头口径即在最前补 B、h 两维。"""
    x_enc = torch.randn(n, d)   # 编码器某层的输出（记忆 memory）：(n_src, d)
    y_dec = torch.randn(n, d)   # 解码器上一层输出：(n_tgt, d)
    # 角色一：encoder 自注意——Q=K=V 同源，无掩码，全句互见（双向）
    _, A_enc = attention(x_enc, x_enc, x_enc)          # A: (n_src, n_src)，全非零
    # 角色二：decoder 因果自注意——Q=K=V 同源，下三角可见（只许看过去）
    _, A_dec = attention(y_dec, y_dec, y_dec, causal=True)   # A: (n_tgt, n_tgt)，严格下三角
    # 角色三：cross——Q 来自解码器，K=V 来自编码器输出，无因果掩码
    _, A_cross = attention(y_dec, x_enc, x_enc)        # A: (n_tgt, n_src)，行=目标步、列=源位
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    print(f"[角色演练] n={n}, d={d}")
    print(f"  encoder 自注意权重形状 {tuple(A_enc.shape)}，min={A_enc.min():.3e}（全句互见，无零元）")
    print(f"  decoder 因果权重形状   {tuple(A_dec.shape)}，上三角（未来位）和 = {A_dec[upper].sum():.3e}")
    print(f"  cross 权重形状        {tuple(A_cross.shape)}（行=目标步 i，列=源位 j）")
    for name, A in [("enc", A_enc), ("dec", A_dec), ("cross", A_cross)]:
        assert torch.allclose(A.sum(-1), torch.ones(n), atol=1e-5), f"{name} 行和 != 1"
    assert A_dec[upper].sum() == 0.0, "因果掩码未屏蔽未来位置"
    print("  断言通过：三种权重行和均为 1；因果矩阵严格下三角")


def demo_api_postures(n=8, d=16):
    """实现对照框：官方 nn.MultiheadAttention 承载三种角色的三种调用姿势。
    （实现对照，非原论文内容；bool 掩码 True=屏蔽，此处用 float -inf 形式避免歧义。）

    输入 x/y 均为 (B=1, n, d)（batch_first=True）；返回权重形状 (N=1, L, S)，取 [0] 得 (n, n)。
    输出张量（第一个返回值）为 (1, n, d)。"""
    mha = nn.MultiheadAttention(d, num_heads=1, batch_first=True, dropout=0.0).eval()
    x = torch.randn(1, n, d)                            # (B, n_src, d)：编码器侧
    y = torch.randn(1, n, d)                            # (B, n_tgt, d)：解码器侧
    causal = torch.triu(torch.full((n, n), float("-inf")), diagonal=1)
    with torch.no_grad():
        _, a_enc = mha(x, x, x, need_weights=True)                      # 姿势一：encoder 自注意
        _, a_dec = mha(y, y, y, attn_mask=causal, need_weights=True)    # 姿势二：因果自注意
        _, a_cross = mha(y, x, x, need_weights=True)                    # 姿势三：cross（Q=y, K=V=x）
    a_enc, a_dec, a_cross = a_enc[0], a_dec[0], a_cross[0]              # (N,L,S)→(L,S)
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    print("[API 姿势] nn.MultiheadAttention 三种喂法（实现对照，非原论文内容）")
    print(f"  self(x,x,x) 权重 {tuple(a_enc.shape)}；self(y,y,y,mask) 权重 {tuple(a_dec.shape)}，"
          f"上三角和 {a_dec[upper].sum():.3e}；cross(y,x,x) 权重 {tuple(a_cross.shape)}")
    assert a_dec[upper].sum() < 1e-8


# ---------- Part 2：合成词典任务 + GRU+Bahdanau 注意力短训 ----------
SRC = ["apple", "bird", "cat", "dog", "sun", "moon", "tree", "lake", "star", "fish", "king", "wind"]
TGT = ["Apfel", "Vogel", "Katze", "Hund", "Sonne", "Mond", "Baum", "See", "Stern", "Fisch", "Koenig", "Wind"]
DICT = dict(zip(SRC, TGT))
S2I = {w: i for i, w in enumerate(SRC)}
T2I = {w: i + 2 for i, w in enumerate(TGT)}          # 0=<sos> 1=<eos>，正文词从 2 起
T_WORDS = ["<sos>", "<eos>"] + TGT
MIN_L, MAX_L = 6, 9


def make_pair(rng):
    """源句 = 从 12 词中无重复抽 L 个；目标 = 词典直译，但**源句首词后置到目标句末**
    （模拟德语动词后置式语序差异 → 对齐非单调，呼应 Bahdanau Fig.3(a)）。"""
    L = rng.randint(MIN_L, MAX_L)
    src = rng.sample(SRC, L)
    tgt = [DICT[w] for w in src[1:]] + [DICT[src[0]]]  # 首词搬到最后
    gold = list(range(1, L)) + [0]                     # 对齐真值：目标内容步 i(0 起)→源位 i+1，末步→源位 0
    return src, tgt, gold


def make_batch(rng, B):
    """构造一个变长批。返回：src (B, L)（-1=填充）、tgt_in (B, L+1)（<sos> 打头）、
    tgt_out (B, L+1)（内容词 + <eos>）、valid (B, L+1)（有效监督步）、pairs 原始句对。"""
    pairs = [make_pair(rng) for _ in range(B)]
    L = max(len(s) for s, _, _ in pairs)
    src = torch.full((B, L), -1, dtype=torch.long)     # -1 = 填充
    tgt_in = torch.zeros(B, L + 1, dtype=torch.long)   # <sos> + L 个内容步
    tgt_out = torch.ones(B, L + 1, dtype=torch.long)   # 内容词 + <eos>（eos=1）
    valid = torch.zeros(B, L + 1, dtype=torch.bool)    # 有效监督步（含该句的 <eos> 步）
    for b, (s, t, _) in enumerate(pairs):
        src[b, : len(s)] = torch.tensor([S2I[w] for w in s])
        ids = [T2I[w] for w in t]
        tgt_in[b, 1 : len(ids) + 1] = torch.tensor(ids)
        tgt_out[b, : len(ids) + 1] = torch.tensor(ids + [1])
        valid[b, : len(ids) + 1] = True
    return src, tgt_in, tgt_out, valid, pairs


class Encoder(nn.Module):
    """双向 GRU 编码器：Bahdanau 1409.0473 §3.2 的拼接标注 h_j = [h→_j; h←_j]。
    变长批用 pack_padded_sequence，避免填充位污染双向读取。

    输入 src (B, L)（-1 为填充）→ 输出标注 h (B, L, 2*hid)（双向拼接，填充位为 0）。"""

    def __init__(self, vocab, emb=32, hid=48):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.gru = nn.GRU(emb, hid, batch_first=True, bidirectional=True)

    def forward(self, src):                            # src: (B, L)，-1 为填充
        lens = (src >= 0).sum(1).cpu()
        e = self.emb(src.clamp(min=0))                 # (B, L) -> (B, L, emb)
        packed = nn.utils.rnn.pack_padded_sequence(e, lens, batch_first=True, enforce_sorted=False)
        h, _ = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=src.size(1))
        return h                                      # (B, L, 2*hid)，填充位为 0


class BahdanauAttention(nn.Module):
    """加性打分 e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j)（1409.0473 附录 A.1.2）。

    输入 s (B, hid)、h (B, L, hid)、src_mask (B, L) → 输出 α_i (B, L)（行和 1）。"""

    def __init__(self, hid):
        super().__init__()
        self.W = nn.Linear(hid, hid, bias=False)       # 作用于解码状态 s
        self.U = nn.Linear(hid, hid, bias=False)       # 作用于编码标注 h
        self.v = nn.Linear(hid, 1, bias=False)

    def forward(self, s, h, src_mask):                 # s:(B,hid) h:(B,L,hid)
        e = self.v(torch.tanh(self.W(s).unsqueeze(1) + self.U(h))).squeeze(-1)
        # 形状：s (B,hid)->(B,1,hid) 与 h (B,L,hid) 广播相加 -> tanh -> v 压维 -> e (B,L)
        e = e.masked_fill(~src_mask, float("-inf"))    # 填充位打分置 -inf，softmax 后为 0
        return torch.softmax(e, dim=-1)                # α_i: (B, L)


class Decoder(nn.Module):
    """GRU 解码器，每步以 [y_{i-1} 嵌入; c_i] 为输入（教师强制）。

    step() 输入 y_prev (B,)、s (B, hid) → 返回 (新 s (B, hid), logits (B, V), α (B, L))。"""

    def __init__(self, vocab, emb=32, hid=96):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.cell = nn.GRUCell(emb + hid, hid)
        self.attn = BahdanauAttention(hid)
        self.out = nn.Linear(hid, vocab)

    def step(self, y_prev, s, h, src_mask):
        a = self.attn(s, h, src_mask)                  # (B, L)
        c = (a.unsqueeze(-1) * h).sum(1)               # (B,L,1)×(B,L,hid) 加权求和 -> (B, hid)
        s = self.cell(torch.cat([self.emb(y_prev), c], -1), s)  # [y 嵌入;c] (B,emb+hid) -> (B, hid)
        return s, self.out(s), a                       # logits: (B, V)


def masked_ce(logits, tgt_out, valid):
    """只在有效步（含各句自己的 <eos> 步）上计损失，排除批内填充监督。
    输入 logits (B, T, V)、tgt_out/valid (B, T) → 返回标量损失。"""
    ce = nn.functional.cross_entropy(logits.flatten(0, 1), tgt_out.flatten(), reduction="none")
    return (ce * valid.flatten()).sum() / valid.sum()


def train(steps=3000, B=64, lr=1e-3):
    """教师强制短训（Adam）。每批 src (B, L)、tgt_in/tgt_out (B, L+1)；
    返回 (enc, dec, 参数量, 末步 loss)。"""
    rng = random.Random(SEED)
    enc, dec = Encoder(len(SRC)), Decoder(len(T_WORDS))
    params = list(enc.parameters()) + list(dec.parameters())
    n_par = sum(p.numel() for p in params)
    opt = torch.optim.Adam(params, lr=lr)
    for step in range(1, steps + 1):
        src, tgt_in, tgt_out, valid, _ = make_batch(rng, B)
        h = enc(src)                                   # (B, L, 2*hid)
        s = h.new_zeros(B, dec.cell.hidden_size)       # s_0: (B, hid)
        logits = []
        for t in range(tgt_in.size(1)):
            s, lg, _ = dec.step(tgt_in[:, t], s, h, src >= 0)
            logits.append(lg)                          # 每 t 一行 (B, V)
        loss = masked_ce(torch.stack(logits, 1), tgt_out, valid)   # 堆成 (B, T, V)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"  step {step:5d}  loss {loss.item():.4f}")
    return enc, dec, n_par, loss.item()


def evaluate(enc, dec, n=1024):
    """held-out 句级精确匹配率 + 对齐 argmax 命中率 + 真值源位上的平均权重（对齐锐度）。"""
    rng = random.Random(SEED + 1)
    exact = hit = total = 0
    gold_w = 0.0
    with torch.no_grad():
        for _ in range(n):
            src, tgt_in, tgt_out, _, pairs = make_batch(rng, 1)
            h = enc(src)                               # (1, L, 2*hid)
            s = h.new_zeros(1, dec.cell.hidden_size)   # s_0: (1, hid)
            pred_ids, amax, rows = [], [], []
            for t in range(tgt_in.size(1)):
                s, lg, a = dec.step(tgt_in[:, t], s, h, src >= 0)   # a: (1, L)
                pred_ids.append(int(lg.argmax(-1)))
                amax.append(int(a.argmax(-1)))
                rows.append(a[0])                      # 每步的对齐行 (L,)
            if pred_ids == tgt_out[0].tolist():
                exact += 1
            gold = pairs[0][2]                        # 仅 L 个内容步有对齐真值
            for i, g in enumerate(gold):
                total += 1
                hit += int(amax[i] == g)
                gold_w += float(rows[i][g])
    return exact / n, hit / total, gold_w / total


def export_heatmap(enc, dec, fname="fig-4-3-alignment-heatmap.png"):
    """取两条固定测试句（L=6 与 L=9），导出对齐热图（真实训练产物）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    rng = random.Random(SEED + 7)
    picks = []
    for L in (6, 9):
        while True:
            p = make_pair(rng)
            if len(p[0]) == L:
                picks.append(p)
                break
    cmap = LinearSegmentedColormap.from_list("bookblue", ["#cde2fb", "#0d366b"])
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    last_im = None
    for ax, (src_w, tgt_w, _) in zip(axes, picks):
        src = torch.tensor([[S2I[w] for w in src_w]])
        tgt_in = torch.tensor([[0] + [T2I[w] for w in tgt_w]])
        with torch.no_grad():
            h = enc(src)
            s = h.new_zeros(1, dec.cell.hidden_size)
            rows = []
            for t in range(tgt_in.size(1) - 1):        # 只画 L 个内容步
                s, _, a = dec.step(tgt_in[:, t], s, h, src >= 0)
                rows.append(a[0])
        A = torch.stack(rows).numpy()                  # (L, L)：行=目标步，列=源位
        last_im = ax.imshow(A, cmap=cmap, vmin=0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(src_w)), src_w, rotation=45, fontsize=6, color="#898781")
        ax.set_yticks(range(len(tgt_w)), tgt_w, fontsize=6, color="#898781")
        ax.set_xlabel("source positions (encoder input)", fontsize=7, color="#898781")
        ax.set_ylabel("target steps (decoder)", fontsize=7, color="#898781")
        for sp in ax.spines.values():
            sp.set_color("#e1e0d9")
    fig.colorbar(last_im, ax=axes, shrink=0.85, pad=0.03).ax.tick_params(labelsize=6, colors="#898781")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / fname)
    fig.savefig(LOG_DIR / fname)
    print(f"[热图] 已导出 {FIG_DIR / fname}（另存 log/ch04/）")
    return picks


def main():
    print("== Part 1a 三种注意力角色：形状/掩码演练 ==")
    demo_roles()
    print("== Part 1b 官方 API 三种调用姿势（实现对照）==")
    demo_api_postures()
    print("== Part 2 合成词典任务：GRU+Bahdanau 短训 ==")
    enc, dec, n_par, last_loss = train()
    acc, align, gold_w = evaluate(enc, dec)
    print(f"  参数量 {n_par:,}；最后一步 loss {last_loss:.4f}")
    print(f"  held-out 1024 句：句级精确匹配率 {acc:.1%}；对齐 argmax 命中率 {align:.1%}；"
          f"对齐位平均权重 {gold_w:.3f}")
    picks = export_heatmap(enc, dec)
    for src_w, tgt_w, gold in picks:
        print(f"  样例 src={src_w}")
        print(f"        tgt={tgt_w}")
        print(f"        对齐真值(目标步→源位)={gold}")


if __name__ == "__main__":
    main()
