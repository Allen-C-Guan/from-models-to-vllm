# -*- coding: utf-8 -*-
# 用途：Book1 第 7 章「整机组装」——复用第 2-6 章手写组件拼出完整 Transformer 整机
#       （论文 §3.1/§3.4 口径：Embedding×√d_model→6+6 层→共享 pre-softmax 线性 + softmax），
#       与官方 nn.Transformer(batch_first=True) 权重搬运对拍（含栈顶 LN 差异的分项演示），
#       并以 numel 实数复核 base 的 65M 参数账本。
# 所属章节：《Transformer 原典》第 7 章（整机组装与 mask 全景）
# 跨章复用（读者复现方式）：本文件把 code/ 注入 sys.path 后按包导入
#       ch03/ch05/ch06（各章目录已配 __init__.py；该目录名含连字符与中文、不能作包名，
#       故以 __file__ 相对定位注入路径，效果与平级包一致，任意 cwd 下可运行）。
# 运行：source env.sh && python "code/ch07/transformer.py"（CPU 分钟级）
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_PKG_ROOT = Path(__file__).resolve().parents[1]                 # code/
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from ch03.mha import MyMultiHeadAttention                       # 第 3 章：多头注意力（bool 掩码，True=屏蔽）
from ch05.pe import sinusoidal_pe                               # 第 5 章：正弦位置编码（interleave 排布）
from ch06.blocks import PositionwiseFFN, PostLNSublayer         # 第 6 章：FFN 与 残差+post-LN 包装


# ---------- 1. 整机三层结构：编码器层 / 解码器层 / 整机（§3.1 + §3.4） ----------

class EncoderLayer(nn.Module):
    """§3.1 编码器层：自注意力 + FFN 两个子层，各套残差 + post-LN（第 6 章式 (6.4)）。"""

    def __init__(self, d_model=512, h=8, d_ff=2048, p_drop=0.1):
        super().__init__()
        self.self_attn = MyMultiHeadAttention(d_model, h)       # 第 3 章组件
        self.ffn = PositionwiseFFN(d_model, d_ff)               # 第 6 章组件
        self.attn_block = PostLNSublayer(d_model, p_drop)       # LN(x + Dropout(Sublayer(x)))
        self.ffn_block = PostLNSublayer(d_model, p_drop)

    def forward(self, x, src_pad):
        """编码器层前向：x (B,S,512)、src_pad (B,1,1,S) bool（True=屏蔽）→ (B,S,512)。"""
        # 编码器自注意力：双向可见，只需挡掉 padding（(B,1,1,S) 广播到 (B,h,S,S)）
        x = self.attn_block(x, lambda t: self.self_attn(t, t, t, mask=src_pad)[0])  # (B,S,512) -> (B,S,512)
        return self.ffn_block(x, self.ffn)                                          # (B,S,512) -> (B,S,512)（FFN 车间内 (B,S,2048)）


class DecoderLayer(nn.Module):
    """§3.1 解码器层：掩码自注意力 + cross 注意力 + FFN 三个子层，各套残差 + post-LN。"""

    def __init__(self, d_model=512, h=8, d_ff=2048, p_drop=0.1):
        super().__init__()
        self.self_attn = MyMultiHeadAttention(d_model, h)       # 解码器带掩码自注意力
        self.cross_attn = MyMultiHeadAttention(d_model, h)      # cross：Q 来自解码器，K=V=memory
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.sub1 = PostLNSublayer(d_model, p_drop)
        self.sub2 = PostLNSublayer(d_model, p_drop)
        self.sub3 = PostLNSublayer(d_model, p_drop)

    def forward(self, x, memory, self_mask, cross_mask):
        """解码器层前向：x (B,T,512)、memory (B,S,512)、self_mask (B,1,T,T) 与 cross_mask (B,1,1,S) 均 bool → (B,T,512)。"""
        x = self.sub1(x, lambda t: self.self_attn(t, t, t, mask=self_mask)[0])       # 因果∪填充；scores (B,h,T,T)
        x = self.sub2(x, lambda t: self.cross_attn(t, memory, memory, mask=cross_mask)[0])  # scores (B,h,T,S)
        return self.sub3(x, self.ffn)                                                # (B,T,512)（FFN 车间内 (B,T,2048)）


class TransformerMT(nn.Module):
    """整机组装（§3.4）：查表 ×√d_model + 正弦 PE → N+N 层 → 共享 pre-softmax 线性 + softmax。

    I/O 形状：forward(src (B,S) int, tgt (B,T) int, src_pad (B,S) / tgt_pad (B,T) bool)
    → logits (B,T,V)；B=batch、S=源句长、T=目标句长、V=词表 37000。
    stack_top_ln=False 默认忠于论文 Fig 1（两塔出口无栈顶 LN，canonical 口径 C-G13）；
    True 仅为与官方 nn.Transformer 对拍而设（torch 有栈顶 LN，属实现族谱差异）。
    """

    def __init__(self, vocab=37000, n_layer=6, d_model=512, h=8, d_ff=2048,
                 p_drop=0.1, stack_top_ln=False, max_pe=5120):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab, d_model)               # 三处共享的同一个矩阵（§3.4）
        self.emb_dropout = nn.Dropout(p_drop)                   # §5.4：embedding 与 PE 之和
        self.enc_layers = nn.ModuleList(EncoderLayer(d_model, h, d_ff, p_drop) for _ in range(n_layer))
        self.dec_layers = nn.ModuleList(DecoderLayer(d_model, h, d_ff, p_drop) for _ in range(n_layer))
        self.top_norm_enc = nn.LayerNorm(d_model) if stack_top_ln else None
        self.top_norm_dec = nn.LayerNorm(d_model) if stack_top_ln else None
        pe = sinusoidal_pe(max_pe, d_model, dtype=torch.float32)
        self.register_buffer("pe_table", pe, persistent=False)  # 预生成查表：任意句长 ≤ max_pe

    def embed_tokens(self, ids):
        """入口（§3.4/§3.5/§5.4）：查表 ×√d_model，加正弦 PE，再过 dropout。ids (B,L) int → (B,L,512)。"""
        x = self.embed(ids) * math.sqrt(self.d_model)           # (B,L) -> (B,L,512)：§3.4 的 ×√d_model
        x = x + self.pe_table[: ids.size(1)]                    # (B,L,512) + (L,512) 逐位置广播，形状不变
        return self.emb_dropout(x)                              # (B,L,512)

    def run_encoder(self, x, src_pad):
        """层栈主体（对拍入口：输入已是连续张量）。x (B,S,512)、src_pad (B,S) bool（True=padding）→ memory (B,S,512)。"""
        mask = src_pad[:, None, None, :]                        # (B,S) -> (B,1,1,S)：在 scores (B,h,S,S) 上广播
        for layer in self.enc_layers:                           # 每层 (B,S,512) -> (B,S,512)，六层形状不变
            x = layer(x, mask)
        return self.top_norm_enc(x) if self.top_norm_enc is not None else x

    def run_decoder(self, x, memory, tgt_pad, src_pad):
        """x (B,T,512)、memory (B,S,512)、tgt_pad (B,T)/src_pad (B,S) bool → (B,T,512)。"""
        T = x.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)  # (T,T)：上三角=屏蔽
        self_mask = causal | tgt_pad[:, None, None, :]          # (T,T)|(B,1,1,T) 广播 -> (B,1,T,T)：因果∪填充（实验 3 验证）
        cross_mask = src_pad[:, None, None, :]                  # (B,S) -> (B,1,1,S)：贴在 scores (B,h,T,S) 上
        for layer in self.dec_layers:                           # 每层 (B,T,512) -> (B,T,512)，六层形状不变
            x = layer(x, memory, self_mask, cross_mask)
        return self.top_norm_dec(x) if self.top_norm_dec is not None else x

    def forward(self, src, tgt, src_pad=None, tgt_pad=None):
        """src/tgt: (B,S)/(B,T) 词元 id；*_pad 不传则视为无填充。返回 logits (B,T,V)。"""
        B = src.size(0)
        src_pad = torch.zeros(B, src.size(1), dtype=torch.bool, device=src.device) if src_pad is None else src_pad
        tgt_pad = torch.zeros(B, tgt.size(1), dtype=torch.bool, device=src.device) if tgt_pad is None else tgt_pad
        memory = self.run_encoder(self.embed_tokens(src), src_pad)                    # (B,S) -> memory (B,S,512)
        dec_out = self.run_decoder(self.embed_tokens(tgt), memory, tgt_pad, src_pad)  # (B,T) -> (B,T,512)
        logits = dec_out @ self.embed.weight.t()                # (B,T,512)@(512,V) -> (B,T,V)；§3.4：pre-softmax 线性 = 转置的嵌入矩阵
        return logits

    def decode_step(self, src, src_pad, ys):
        """自回归解码一步（无任何缓存，整段前缀重算）——第 7.4 节「重算什么」的活体。
        ys (B,t) int → P (B,V)：只取最后一步的下一词元分布。"""
        return F.softmax(self.forward(src, ys, src_pad), dim=-1)[:, -1]


# ---------- 2. 对拍一：官方默认超参 = 论文 base 逐项断言（表 7.4 的机器复核） ----------

def exp_default_hyperparams():
    t = nn.Transformer()                                        # 全默认构造
    got = dict(N_enc=t.encoder.num_layers, N_dec=t.decoder.num_layers,
               d_model=t.d_model, h=t.encoder.layers[0].self_attn.num_heads,
               d_ff=t.encoder.layers[0].linear1.out_features,
               dropout=t.encoder.layers[0].dropout.p,
               act="relu", norm_first=t.encoder.layers[0].norm_first,
               top_LN=t.encoder.norm is not None)
    base = dict(N_enc=6, N_dec=6, d_model=512, h=8, d_ff=2048, dropout=0.1,
                act="relu", norm_first=False, top_LN=False)    # top_LN：论文 Fig 1 两塔出口无栈顶 LN（C-G13）
    print("== 1) nn.Transformer() 默认超参 vs 论文 Table 3 base ==")
    for k in base:
        flag = "一致" if got[k] == base[k] else "差异"
        print(f"  {k:<10s} 官方默认 {str(got[k]):<6s} 论文 base {str(base[k]):<6s} {flag}")


# ---------- 3. 对拍二：权重搬运整机对拍（含栈顶 LN 差异的分项演示） ----------

def _copy_mha(mine, offi):
    """四组权重搬进官方 MHA：in_proj_weight (3d,d) 沿 dim0 按 Q/K/V 打包（第 3 章口径），W^O 走 out_proj。"""
    d = mine.w_q.weight.shape[0]
    offi.in_proj_weight[:d].copy_(mine.w_q.weight)              # in_proj 沿 dim0 按 Q/K/V 打包（第 3 章）
    offi.in_proj_weight[d:2 * d].copy_(mine.w_k.weight)
    offi.in_proj_weight[2 * d:].copy_(mine.w_v.weight)
    offi.in_proj_bias[:d].copy_(mine.w_q.bias)
    offi.in_proj_bias[d:2 * d].copy_(mine.w_k.bias)
    offi.in_proj_bias[2 * d:].copy_(mine.w_v.bias)
    offi.out_proj.weight.copy_(mine.w_o.weight)                 # W^O
    offi.out_proj.bias.copy_(mine.w_o.bias)


def _transfer(mine, offi):
    """把本书整机的全部权重搬进 nn.Transformer（嵌入/输出层官方没有，对拍只比层栈）。"""
    for m_l, o_l in zip(mine.enc_layers, offi.encoder.layers):
        _copy_mha(m_l.self_attn, o_l.self_attn)
        o_l.linear1.weight.copy_(m_l.ffn.w1.weight); o_l.linear1.bias.copy_(m_l.ffn.w1.bias)
        o_l.linear2.weight.copy_(m_l.ffn.w2.weight); o_l.linear2.bias.copy_(m_l.ffn.w2.bias)
        o_l.norm1.weight.copy_(m_l.attn_block.norm.weight); o_l.norm1.bias.copy_(m_l.attn_block.norm.bias)
        o_l.norm2.weight.copy_(m_l.ffn_block.norm.weight); o_l.norm2.bias.copy_(m_l.ffn_block.norm.bias)
    for m_l, o_l in zip(mine.dec_layers, offi.decoder.layers):
        _copy_mha(m_l.self_attn, o_l.self_attn)
        _copy_mha(m_l.cross_attn, o_l.multihead_attn)
        o_l.linear1.weight.copy_(m_l.ffn.w1.weight); o_l.linear1.bias.copy_(m_l.ffn.w1.bias)
        o_l.linear2.weight.copy_(m_l.ffn.w2.weight); o_l.linear2.bias.copy_(m_l.ffn.w2.bias)
        o_l.norm1.weight.copy_(m_l.sub1.norm.weight); o_l.norm1.bias.copy_(m_l.sub1.norm.bias)
        o_l.norm2.weight.copy_(m_l.sub2.norm.weight); o_l.norm2.bias.copy_(m_l.sub2.norm.bias)
        o_l.norm3.weight.copy_(m_l.sub3.norm.weight); o_l.norm3.bias.copy_(m_l.sub3.norm.bias)
    if mine.top_norm_enc is not None:                           # 栈顶 LN 对照档才有得搬
        offi.encoder.norm.weight.copy_(mine.top_norm_enc.weight)
        offi.encoder.norm.bias.copy_(mine.top_norm_enc.bias)
        offi.decoder.norm.weight.copy_(mine.top_norm_dec.weight)
        offi.decoder.norm.bias.copy_(mine.top_norm_dec.bias)


def exp_parity(d_model=512, h=8, n_layer=6, d_ff=2048, B=2, S=10, T=9):
    """整机权重搬运对拍：同权重、同输入、同掩码 → allclose；再做栈顶 LN 分项对照。"""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    torch.manual_seed(7)
    print("== 2) 整机对拍：本书手写（复用 ch03/ch05/ch06 组件）vs nn.Transformer(batch_first=True) ==")
    print("  口径：eval（两侧 dropout 全关，官方 FFN 内 ReLU 后 dropout 在 eval 下同为恒等）")
    print("        + torch.backends.mha fast path 关闭 + SDPA 锁 MATH 后端；掩码分开传官方、bool 合并传本书")
    mine = TransformerMT(n_layer=n_layer, d_model=d_model, h=h, d_ff=d_ff,
                         stack_top_ln=True).eval()              # 对照档：补栈顶 LN 对齐官方
    offi = nn.Transformer(d_model=d_model, nhead=h, num_encoder_layers=n_layer,
                          num_decoder_layers=n_layer, dim_feedforward=d_ff,
                          dropout=0.1, batch_first=True).eval()
    with torch.no_grad():
        _transfer(mine, offi)
    h_src, h_tgt = torch.randn(B, S, d_model), torch.randn(B, T, d_model)  # 对拍只比层栈：从嵌入之后喂 (B,S,512)/(B,T,512)
    src_pad = torch.zeros(B, S, dtype=torch.bool); src_pad[0, 7:] = True   # 右填充（无全屏蔽行），(B,S)
    tgt_pad = torch.zeros(B, T, dtype=torch.bool); tgt_pad[1, 6:] = True   # (B,T)
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)    # (T,T)
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)              # 关原生 fast path，走可复现慢路径
    try:
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            mem_mine = mine.run_encoder(h_src, src_pad)
            out_mine = mine.run_decoder(h_tgt, mem_mine, tgt_pad, src_pad)
            out_offi = offi(h_src, h_tgt, src_key_padding_mask=src_pad,
                            tgt_mask=causal, tgt_key_padding_mask=tgt_pad,
                            memory_key_padding_mask=src_pad)
            # 分项一：论文口径整机（无栈顶 LN）vs 官方（有，γ=1/β=0 初始）——C-G13 的数值活体
            mine.top_norm_enc, mine.top_norm_dec = None, None
            mem_paper = mine.run_encoder(h_src, src_pad)
            out_paper = mine.run_decoder(h_tgt, mem_paper, tgt_pad, src_pad)
            # 分项二：官方栈顶 LN 的 γ/β 偏离初始值（模拟训练后）——差异即刻显形
            gen = torch.Generator().manual_seed(7)
            for norm in (offi.encoder.norm, offi.decoder.norm):
                norm.weight.copy_(torch.rand(d_model, generator=gen) + 0.5)      # γ ∈ [0.5, 1.5)
                norm.bias.copy_(torch.rand(d_model, generator=gen) - 0.5)        # β ∈ [-0.5, 0.5)
            out_offi2 = offi(h_src, h_tgt, src_key_padding_mask=src_pad,
                             tgt_mask=causal, tgt_key_padding_mask=tgt_pad,
                             memory_key_padding_mask=src_pad)
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    diff = (out_mine - out_offi).abs().max().item()
    print(f"  对照档（本书+栈顶 LN vs 官方）          ：max|Δ| = {diff:.2e}"
          f"  allclose(atol=1e-5) = {torch.allclose(out_mine, out_offi, rtol=0, atol=1e-5)}")
    diff_ln = (out_paper - out_offi).abs().max().item()
    print(f"  论文档（本书无栈顶 LN vs 官方，γ=1/β=0）：max|Δ| = {diff_ln:.2e}——post-LN 塔顶输出已逐位置"
          f"零均值/单位方差，LN(γ=1,β=0) 近似恒等（差异被 ε 与方差口径压在 1e-6 量级）")
    diff_ln2 = (out_paper - out_offi2).abs().max().item()
    print(f"  论文档 vs 官方（γ/β 训练后偏离初始）   ：max|Δ| = {diff_ln2:.2f}——"
          f"一旦仿射参数离开 1/0，栈顶 LN 就是一处实打实的结构差异（多 2×512 个参数与一次归一化）")


# ---------- 4. 参数量账本：65M 逐项数（numel 实数，非公式推算） ----------

def exp_ledger(vocab=37000):
    print("== 3) base 参数量账本（整机实例化后逐部件 numel 实数；词表 37000 = §5.1 共享 BPE 口径）==")
    torch.manual_seed(7)
    m = TransformerMT(vocab=vocab)                              # 论文口径整机（无栈顶 LN）
    n = lambda mod: sum(p.numel() for p in mod.parameters())
    attn_p = n(m.enc_layers[0].self_attn)
    ffn_p = n(m.enc_layers[0].ffn)
    ln_p = 2 * m.d_model
    enc_l = n(m.enc_layers[0])
    dec_l = n(m.dec_layers[0])
    emb_p = m.embed.weight.numel()
    total = n(m)                                                # 共享权重按张量只计一次
    untied = total + 2 * emb_p                                  # 反事实：三处不共享
    print(f"  注意力子层 {attn_p:>10,}   FFN 子层 {ffn_p:>10,}   LayerNorm {ln_p:>6,}")
    print(f"  编码器层小计 {enc_l:>8,} ×6 = {6 * enc_l:>10,}")
    print(f"  解码器层小计 {dec_l:>8,} ×6 = {6 * dec_l:>10,}")
    print(f"  词嵌入（三处共享一个矩阵） {emb_p:>10,}")
    print(f"  合计 {total:,} ≈ {total / 1e6:.1f}M（Table 3 记 65M，差 {(65e6 - total) / 1e6:.1f}M ≈ 3%，构成预估口径）")
    print(f"  反事实：三处不共享 = {untied:,} ≈ {untied / 1e6:.1f}M——权重共享省下 2×{emb_p/1e6:.1f}M = {2*emb_p/1e6:.1f}M（省 37.5%，超过三分之一）")
    print(f"  共享自证：合计恰等于 6×编码器层 + 6×解码器层 + 单份嵌入 = "
          f"{total == 6 * enc_l + 6 * dec_l + emb_p}（同一张量三处引用，numel 只数到一份）")
    # 「模型不止参数」：自回归解码的重算账（每个新 token 都把整段前缀重过一遍解码器）
    T = 9
    recompute = T * (T + 1) // 2
    print(f"  重算账（T={T} 逐 token 解码）：累计前向位置数 = 1+2+…+{T} = {recompute}，"
          f"是一次整段前向（{T}）的 {recompute / T:.0f} 倍——序列越长浪费越大（回收地址见章末欠账）")


if __name__ == "__main__":
    exp_default_hyperparams()   # 1) 官方默认超参逐项断言
    exp_parity()                # 2) 整机对拍 + 栈顶 LN 分项
    exp_ledger()                # 3) 65M 账本 numel 实数 + 重算账
