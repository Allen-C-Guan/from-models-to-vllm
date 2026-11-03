# -*- coding: utf-8 -*-
# 用途：Book1 第 9 章——手写 mini 原版 Transformer（零魔改，仅缩尺寸）：整机定义
# 所属章节：《Transformer 原典》第 9 章（从零实现与小型翻译实验）；组件细节见 ch2-6，整机 mask 全景见 ch7
# 运行：source env.sh && python "code/ch09/model.py"（冒烟自测：形状/参数量/前向/解码接口）
#
# 零魔改清单（对照原论文 v7，仅缩小尺寸）：
#   post-LN（残差后 LayerNorm，§3.1 图 1 的 Add & Norm 位置）· 正弦 PE（§3.5）
#   · encoder 双向 self-attn / decoder masked self-attn + cross-attn（§3.2.3）
#   · FFN = ReLU 两层（§3.3）· dropout 布点两处（§5.4：子层输出 + embedding 与 PE 之和）
#   · 三处权重共享：encoder 输入嵌入 = decoder 输入嵌入 = decoder 输出 pre-softmax 线性（§3.4，
#     shared BPE 单词表使该共享成立——原论文「similar to [30]」的做法）
#   · 注意力手写 matmul + softmax（不用 F.scaled_dot_product_attention / nn.MultiheadAttention）
# 缩尺寸：N=4, d_model=256, h=8, d_ff=1024（base 65M → 9.42M，numel 实数；词表 37k→8k）
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import PAD, SOS, ensure_bpe

# ch9 参考配置（feasibility notes/01 推荐：N=4/d=256/h=8/dff=1024）
N_LAYER, D_MODEL, HEADS, D_FF, P_DROP = 4, 256, 8, 1024, 0.1


def sinusoidal_pe(length: int, d_model: int) -> torch.Tensor:
    """正弦位置编码（§3.5 式；与 ch5/ch07 同一实现）：PE(pos,2i)=sin(pos/10000^{2i/d})，奇数维 cos。
    输入标量长度，返回 (length, d_model)——按需生成任意长度，外推实验（1.5~2 倍训练长度）不需要任何额外参数。"""
    pe = torch.zeros(length, d_model)
    pos = torch.arange(length).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


def attention(q, k, v, mask, return_weights=False):
    """缩放点积注意力（§3.2.1）：softmax(QK^T/√d_k)V。
    q: (B,H,Lq,Dk)、k: (B,H,Lk,Dk)、v: (B,H,Lk,Dv)；mask: bool，True=屏蔽，形如 (B,1,1,Lk) 可广播。
    返回 out (B,H,Lq,Dv)，return_weights 时另附 probs (B,H,Lq,Lk)。"""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))  # (B,H,Lq,Dk)@(B,H,Dk,Lk) -> (B,H,Lq,Lk)
    scores = scores.masked_fill(mask, float("-inf"))
    probs = F.softmax(scores, dim=-1)                          # (B,H,Lq,Lk)，每行和为 1
    out = probs @ v                                            # (B,H,Lq,Lk)@(B,H,Lk,Dv) -> (B,H,Lq,Dv)
    return (out, probs) if return_weights else out


class MultiHeadAttention(nn.Module):
    """多头注意力（§3.2.2）：投影→按头 reshape→单头注意力→拼回→输出投影 W^O。
    forward: q_in (B,Lq,d)、k_in/v_in (B,Lk,d) → out (B,Lq,d)，return_weights 时另附 probs (B,h,Lq,Lk)。"""

    def __init__(self, d_model: int, h: int, p_drop: float):
        super().__init__()
        self.h, self.d_k = h, d_model // h
        self.wq, self.wk, self.wv = (nn.Linear(d_model, d_model) for _ in range(3))
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, q_in, k_in, v_in, mask, return_weights=False):
        b = q_in.size(0)
        split = lambda x: x.reshape(b, -1, self.h, self.d_k).transpose(1, 2)  # .reshape：切片来源的张量不保证连续
        q, k, v = split(self.wq(q_in)), split(self.wk(k_in)), split(self.wv(v_in))  # (B,L,d) -> (B,h,L,dk)
        out, probs = attention(q, k, v, mask, return_weights=True)                 # out: (B,h,Lq,dk)
        out = out.transpose(1, 2).reshape(b, -1, self.h * self.d_k)                # (B,h,Lq,dk) -> (B,Lq,d)
        out = self.wo(out)
        return (out, probs) if return_weights else out


class PostLNSublayer(nn.Module):
    """残差 + post-LN 包装（§3.1/§5.4）：dropout 施加于子层输出、求和、归一化之前。x: (B,L,d) → (B,L,d)。"""

    def __init__(self, d_model: int, p_drop: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer()))


class EncoderLayer(nn.Module):
    """编码器一层：x (B,S,d) + pad_mask (B,1,1,S) → (B,S,d)；自注意 scores (B,h,S,S)，FFN 鼓包 (B,S,d_ff)。"""

    def __init__(self, d_model: int, h: int, d_ff: int, p_drop: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, p_drop)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.sub1, self.sub2 = PostLNSublayer(d_model, p_drop), PostLNSublayer(d_model, p_drop)

    def forward(self, x, pad_mask):
        x = self.sub1(x, lambda: self.self_attn(x, x, x, pad_mask))
        return self.sub2(x, lambda: self.ffn(x))


class DecoderLayer(nn.Module):
    """解码器一层：x (B,T,d) + mem (B,S,d)、self_mask (B,1,T,T)、cross_mask (B,1,1,S) → (B,T,d)。
    自注意 scores (B,h,T,T)（因果），cross scores (B,h,T,S)（查询来自解码器、键值来自 memory）。"""

    def __init__(self, d_model: int, h: int, d_ff: int, p_drop: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, p_drop)
        self.cross_attn = MultiHeadAttention(d_model, h, p_drop)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.sub1 = PostLNSublayer(d_model, p_drop)
        self.sub2 = PostLNSublayer(d_model, p_drop)
        self.sub3 = PostLNSublayer(d_model, p_drop)

    def forward(self, x, mem, self_mask, cross_mask):
        x = self.sub1(x, lambda: self.self_attn(x, x, x, self_mask))        # masked self-attn（因果）
        x = self.sub2(x, lambda: self.cross_attn(x, mem, mem, cross_mask))  # cross-attn（双塔交汇）
        return self.sub3(x, lambda: self.ffn(x))


class Transformer(nn.Module):
    """mini 原版 Transformer（翻译整机）。解码统一接口 encode()/decode_step() 供贪心/beam 复用（decode.py）。
    forward: src (B,S) + tgt_in (B,T) → logits (B,T,V)；全章形状流转表见书表 9.1。"""

    def __init__(self, vocab: int, n_layer=N_LAYER, d_model=D_MODEL, heads=HEADS, d_ff=D_FF, p_drop=P_DROP):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.emb_drop = nn.Dropout(p_drop)  # §5.4：作用于 embedding 与 PE 之和
        self.enc_layers = nn.ModuleList(EncoderLayer(d_model, heads, d_ff, p_drop) for _ in range(n_layer))
        self.dec_layers = nn.ModuleList(DecoderLayer(d_model, heads, d_ff, p_drop) for _ in range(n_layer))

    def _add_pe(self, ids):
        """ids (B,L) → (B,L,d)：查表 ×√d_model、+正弦 PE、dropout（§3.4/§5.4）。"""
        x = self.embed(ids) * math.sqrt(self.d_model) + sinusoidal_pe(ids.size(1), self.d_model).to(ids.device)
        return self.emb_drop(x)

    def encode(self, src):
        """src: (B, S) → memory: (B, S, d) 与 cross 用的源侧 padding mask (B,1,1,S)。"""
        pad_mask = (src == PAD)[:, None, None, :]  # (B,S) -> (B,1,1,S)
        x = self._add_pe(src)                       # (B,S) -> (B,S,d)
        for layer in self.enc_layers:
            x = layer(x, pad_mask)                  # 每层 (B,S,d) -> (B,S,d)
        return (x, pad_mask)

    def decode(self, tgt_in, state):
        """tgt_in (B,T) + state=(mem (B,S,d), cross_mask (B,1,1,S)) → logits (B,T,V)。"""
        mem, cross_mask = state
        length = tgt_in.size(1)
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=tgt_in.device), 1)
        self_mask = causal[None, None] | (tgt_in == PAD)[:, None, None, :]  # (1,1,T,T) | (B,1,1,T) -> (B,1,T,T)
        x = self._add_pe(tgt_in)                    # (B,T) -> (B,T,d)
        for layer in self.dec_layers:
            x = layer(x, mem, self_mask, cross_mask)  # 每层 (B,T,d) -> (B,T,d)
        logits = x @ self.embed.weight.t()  # §3.4 权重共享：(B,T,d)@(d,V) -> (B,T,V)
        return logits

    def forward(self, src, tgt_in):
        """src (B,S) + tgt_in (B,T) → logits (B,T,V)（教师强制：tgt_in 为右移一位的目标前缀）。"""
        return self.decode(tgt_in, self.encode(src))

    def decode_step(self, state, ys):
        """贪心/beam 的统一接口：给定整段前缀 ys (B,t)，返回最后一步的 logits (B, V)。
        无缓存整段重算——「每生成一个 token 要重算什么」的活体展示（K/V 缓存账是后世欠账，详见 Book3 第 6 章）。"""
        return self.decode(ys, state)[:, -1]

    def cross_attention_weights(self, src, tgt, layer: int = -1):
        """导出指定解码层的 cross-attn 对齐矩阵（图 9.2 素材）：layer 支持负索引（-1=末层，默认）与
        非负索引（0=首层）——非负统一换算成负索引再匹配；8 头平均，取 batch 0。
        src (1,S)、tgt (1,T) → 返回 (T, S)。"""
        if layer >= 0:  # 统一负索引：非负 layer 若直接套 i == len + layer 的偏移判断会永不命中（索引 bug；负索引路径行为不变）
            layer -= len(self.dec_layers)
        mem, cross_mask = self.encode(src)
        length = tgt.size(1)
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=tgt.device), 1)
        self_mask = causal[None, None] | (tgt == PAD)[:, None, None, :]
        x = self._add_pe(tgt)
        for i, layer_ in enumerate(self.dec_layers):
            if i == len(self.dec_layers) + layer:
                _, w = layer_.cross_attn(x, mem, mem, cross_mask, return_weights=True)
                align = w.mean(1)[0].detach().cpu()  # (h,T,S) 头平均取 batch 0 -> (T,S)
            x = layer_(x, mem, self_mask, cross_mask)
        return align


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main():  # 冒烟自测
    torch.manual_seed(2017)
    vocab = ensure_bpe().get_piece_size()
    model = Transformer(vocab)
    n_emb = model.embed.weight.numel()
    print(f"[参数] 总数 {n_params(model):,}（非嵌入 {n_params(model) - n_emb:,}，共享嵌入 {n_emb:,}）"
          f"≈ {n_params(model) / 1e6:.2f}M")
    src = torch.tensor([[SOS, 5, 6, 7, 8, PAD, PAD]])
    tgt_in = torch.tensor([[SOS, 9, 10, 11]])
    model.eval()
    with torch.no_grad():
        logits = model(src, tgt_in)
        print(f"[前向] src {tuple(src.shape)} + tgt_in {tuple(tgt_in.shape)} → logits {tuple(logits.shape)}")
        state = model.encode(src)
        last = model.decode_step(state, tgt_in)
        print(f"[解码接口] decode_step → {tuple(last.shape)}；与整段前向最后一步一致："
              f"{torch.allclose(last, logits[:, -1], atol=1e-5)}")
        align = model.cross_attention_weights(src, tgt_in)
        print(f"[对齐] cross-attn 权重（末层头平均）{tuple(align.shape)}，行和≈1："
              f"{align.sum(-1).mean():.4f}")
        model.train()
        _ = model(src, tgt_in)  # 训练态（dropout 开）也能前向


if __name__ == "__main__":
    main()
