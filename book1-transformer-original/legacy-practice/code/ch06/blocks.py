# 用途：Book1 第 6 章——逐位置 FFN、残差连接与 post-LN 子层块的实现、官方对拍与形状表生成
# 所属章节：《Transformer 原典》第 6 章（FFN、残差与 post-LN：一条 token 的数据流）
# 运行：source env.sh && python "code/ch06/blocks.py"（CPU 秒级）
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(2017)  # 固定种子（2017＝原论文年份，全部结果可复现）

D_MODEL, D_FF, H, N_LAYER, VOCAB = 512, 2048, 8, 6, 37000  # Table 3 base 超参（词表取 §5.1 的 37k）


# ---------- 1) 逐位置前馈网络：式 (6.1) FFN(x) = max(0, xW1+b1)W2+b2 ----------
class PositionwiseFFN(nn.Module):
    """式 (6.1)：(b,s,d_model) -> (b,s,d_ff) -> (b,s,d_model)，逐位置独立。"""
    def __init__(self, d_model=D_MODEL, d_ff=D_FF):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)   # 升维 W1：512 → 2048（4:1 扩展率）
        self.w2 = nn.Linear(d_ff, d_model)   # 降维 W2：2048 → 512
    def forward(self, x):                    # x: (b, s, d_model)；Linear 只作用于最后一维＝逐位置独立
        return self.w2(F.relu(self.w1(x)))   # (b,s,512) -> (b,s,2048) -> (b,s,512)；参数对所有位置共享、层与层之间不共享（§3.3）


# ---------- 2) 残差 + post-LN 包装：式 (6.4) LayerNorm(x + Sublayer(x)) ----------
class PostLNSublayer(nn.Module):
    """式 (6.4)：§5.4 的 dropout 施加在子层输出上、加残差并归一化之前。
    输入 x: (b,s,d_model)；sublayer: (b,s,d_model)->(b,s,d_model)；输出: (b,s,d_model)。"""
    def __init__(self, d_model=D_MODEL, p_drop=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)    # Ba et al. 式 (6.3)，γ/β 可学习；逐位置作用，形状不变
        self.dropout = nn.Dropout(p_drop)
    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))  # (b,s,512) + (b,s,512) -> (b,s,512)，式 (6.4)


def split_heads(t):
    """(b, s, d_model) → (b, h, s, d_k)：先按头切片再换轴，总元素数不变。"""
    b, s, _ = t.shape
    return t.view(b, s, H, D_MODEL // H).transpose(1, 2)


class MiniMHA(nn.Module):
    """内联最小多头注意力（完整拆装见第 3 章；此处为自包含复用）。
    输入 x: (b, s_q, d_model)；cross 时另给 kv: (b, s_kv, d_model)；输出: (b, s_q, d_model)。"""
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.in_proj = nn.Linear(d_model, 3 * d_model)  # W^Q W^K W^V 沿 dim0 打包
        self.out = nn.Linear(d_model, d_model)          # W^O
    def forward(self, x, kv=None, mask=None):
        if kv is None:                                  # 自注意力：Q/K/V 同源
            q, k, v = self.in_proj(x).chunk(3, dim=-1)  # (b,s,512) -> 3 × (b,s,512)
        else:                                           # cross 注意力：Q 来自解码器，K/V 来自编码器输出
            q = self.in_proj(x).chunk(3, dim=-1)[0]     # Q: (b, s_q, 512)
            k, v = self.in_proj(kv).chunk(3, dim=-1)[1:]  # K, V: (b, s_kv, 512)
        q, k, v = split_heads(q), split_heads(k), split_heads(v)  # 各 (b, h, s, 64)
        scores = q @ k.transpose(-2, -1) / math.sqrt(D_MODEL // H)   # (b,h,n,64)×(b,h,64,m) -> (b,h,n,m)，缩放点积见第 2 章式 (2.1)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        ctx = torch.softmax(scores, dim=-1) @ v         # (b,h,n,m) @ (b,h,m,64) -> (b,h,n,64)
        b, _, s, _ = ctx.shape
        return self.out(ctx.transpose(1, 2).reshape(b, s, D_MODEL))  # 拼回 (b,s,512)，W^O 后仍 (b,s,512)


class EncoderLayerPostLN(nn.Module):
    """§3.1 编码器层：自注意力 + FFN 两个子层，各套残差 + post-LN。
    输入/输出 x: (b, s, d_model)，每子层进出形状不变（残差的维度约束）。"""
    def __init__(self, d_model=D_MODEL, d_ff=D_FF, p_drop=0.0):
        super().__init__()
        self.self_attn = MiniMHA(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.attn_block = PostLNSublayer(d_model, p_drop)
        self.ffn_block = PostLNSublayer(d_model, p_drop)
    def forward(self, x, mask=None):
        x = self.attn_block(x, lambda t: self.self_attn(t, mask=mask))  # (b,s,512) -> (b,s,512)
        return self.ffn_block(x, self.ffn)                              # (b,s,512) -> (b,s,512)


class DecoderLayerPostLN(nn.Module):
    """§3.1 解码器层：掩码自注意力 + cross 注意力 + FFN 三个子层，各套残差 + post-LN。
    输入 x: (b, s_tgt, d_model)、memory: (b, s_src, d_model)；输出: (b, s_tgt, d_model)。"""
    def __init__(self, d_model=D_MODEL, d_ff=D_FF, p_drop=0.0):
        super().__init__()
        self.self_attn = MiniMHA(d_model)
        self.cross_attn = MiniMHA(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.sub = nn.ModuleList(PostLNSublayer(d_model, p_drop) for _ in range(3))
    def forward(self, x, memory, causal_mask):
        x = self.sub[0](x, lambda t: self.self_attn(t, mask=causal_mask))  # 掩码自注意力，scores (b,h,s_tgt,s_tgt)
        x = self.sub[1](x, lambda t: self.cross_attn(t, kv=memory))        # cross：Q (b,s_tgt,512)，K/V (b,s_src,512)
        return self.sub[2](x, self.ffn)                                    # (b,s_tgt,512) -> (b,s_tgt,512)


def sinusoidal_pe(seq_len, d_model=D_MODEL):  # 第 5 章式 (5.1)/(5.2)，此处只作数据流入口；返回 (seq_len, d_model)
    pos = torch.arange(seq_len).unsqueeze(1)
    angle = pos / torch.pow(10000, torch.arange(0, d_model, 2) / d_model)
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1).reshape(seq_len, d_model)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    # 全书对拍规范：eval + 固定种子 + 锁 SDPA 的 MATH 后端（实验一二各自进入上下文）
    from torch.nn.attention import SDPBackend, sdpa_kernel
    # ---- 实验一：post-LN 顺序的数值证明（官方子块手排 vs 官方整层）----
    x = torch.randn(2, 10, D_MODEL)
    el = nn.TransformerEncoderLayer(D_MODEL, H, D_FF, dropout=0.0, batch_first=True).eval()
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        y = el.norm1(x + el._sa_block(x, None, None))   # 先残差求和、后 LayerNorm
        y = el.norm2(y + el._ff_block(y))
        ok1 = torch.allclose(y, el(x), atol=1e-6)
    print(f"[对拍 1] 官方子块手排 post-LN vs nn.TransformerEncoderLayer(x)：allclose={ok1} (atol=1e-6)")

    # ---- 实验二：本书手写层（同权重搬运）vs 官方层 ----
    mine = EncoderLayerPostLN().eval()
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        el.self_attn.in_proj_weight.copy_(mine.self_attn.in_proj.weight)
        el.self_attn.in_proj_bias.copy_(mine.self_attn.in_proj.bias)
        el.self_attn.out_proj.weight.copy_(mine.self_attn.out.weight)
        el.self_attn.out_proj.bias.copy_(mine.self_attn.out.bias)
        el.linear1.weight.copy_(mine.ffn.w1.weight); el.linear1.bias.copy_(mine.ffn.w1.bias)
        el.linear2.weight.copy_(mine.ffn.w2.weight); el.linear2.bias.copy_(mine.ffn.w2.bias)
        el.norm1.weight.copy_(mine.attn_block.norm.weight); el.norm1.bias.copy_(mine.attn_block.norm.bias)
        el.norm2.weight.copy_(mine.ffn_block.norm.weight); el.norm2.bias.copy_(mine.ffn_block.norm.bias)
        ok2 = torch.allclose(mine(x), el(x), atol=1e-5)
    print(f"[对拍 2] 手写 EncoderLayerPostLN vs 官方层（同权重）：allclose={ok2} (atol=1e-5)")

    # ---- 实验三：post-LN 输出的逐位置统计（LN 的「恒等尺度」眼见为实）----
    with torch.no_grad():
        out = mine(x)
        m_absmax = out.mean(-1).abs().max().item()
        s_min, s_max = out.std(-1, unbiased=False).min().item(), out.std(-1, unbiased=False).max().item()
    print(f"[统计] post-LN 输出逐位置 |均值| 最大 {m_absmax:.2e}，标准差范围 [{s_min:.4f}, {s_max:.4f}]")

    # ---- 实验四：一条 token 的数据流形状表（src 10 词、tgt 9 词）----
    emb = nn.Embedding(VOCAB, D_MODEL)
    enc = nn.ModuleList(EncoderLayerPostLN() for _ in range(N_LAYER))
    dec = nn.ModuleList(DecoderLayerPostLN() for _ in range(N_LAYER))
    src, tgt = torch.randint(0, VOCAB, (1, 10)), torch.randint(0, VOCAB, (1, 9))
    causal = torch.triu(torch.ones(9, 9, dtype=torch.bool), diagonal=1)
    rows = []
    rec = lambda name, t: rows.append((name, tuple(t.shape)))
    with torch.no_grad():
        rec("src 词元 id", src)
        h = emb(src); rec("查表 embedding", h)
        s_before = h.std().item()
        h = h * math.sqrt(D_MODEL); rec("× √512", h)
        print(f"[尺度] embedding std={s_before:.4f}，× √512 后 std={h.std().item():.2f}"
              f"（正弦 PE 各维幅值上界为 1）")
        h = h + sinusoidal_pe(10); rec("+ 正弦 PE", h)
        q, k, v = enc[0].self_attn.in_proj(h).chunk(3, dim=-1)
        rec("编码器第1层 Q/K/V（拼接前）", q)
        qs = split_heads(q); rec("按头 reshape (b,h,s,dk)", qs)
        sc = qs @ split_heads(k).transpose(-2, -1) / math.sqrt(D_MODEL // H)
        rec("注意力 scores", sc)
        ctx = torch.softmax(sc, dim=-1) @ split_heads(v)   # (1,8,10,10) @ (1,8,10,64) -> (1,8,10,64)
        rec("softmax 加权求和 ctx", ctx)
        back = enc[0].self_attn.out(ctx.transpose(1, 2).reshape(1, 10, D_MODEL))  # 拼回 (1,10,512) 过 W^O
        rec("拼回 + W^O 投影", back)
        h1 = enc[0].attn_block(h, lambda t: enc[0].self_attn(t))
        rec("注意力子层 残差+post-LN 后", h1)
        rec("FFN 升维隐藏层 2048", enc[0].ffn.w1(h1))
        rec("FFN 降维回 512", enc[0].ffn.w2(F.relu(enc[0].ffn.w1(h1))))
        h2 = enc[0].ffn_block(h1, enc[0].ffn)
        rec("FFN 子层 残差+post-LN 后", h2)
        for layer in enc:
            h = layer(h)
        rec("编码器输出 memory（6 层后）", h)
        d = emb(tgt) * math.sqrt(D_MODEL) + sinusoidal_pe(9)
        rec("tgt embedding×√512+PE", d)
        for layer in dec:
            d = layer(d, h, causal)
        rec("解码器输出", d)
        logits = d @ emb.weight.t()
        rec("logits（共享 pre-softmax 线性）", logits)
    print("[形状表] 一条 token 的数据流（base 配置，batch=1，src=10，tgt=9）")
    for name, shape in rows:
        print(f"  {name:<28s} {shape}")

    # ---- 实验五：参数占比账本（公式复算）----
    attn_p, ffn_p, ln_p = n_params(MiniMHA()), n_params(PositionwiseFFN()), 2 * D_MODEL
    enc_l = attn_p + ffn_p + 2 * ln_p
    dec_l = 2 * attn_p + ffn_p + 3 * ln_p
    emb_p = VOCAB * D_MODEL
    total = 6 * enc_l + 6 * dec_l + emb_p
    print("[参数账本] base 各部件参数量（按公式复算，词表 37000）")
    print(f"  注意力子层（QKVO+偏置） {attn_p:>10,} | FFN 子层 {ffn_p:>10,} | LN（γ+β） {ln_p:>6,}")
    print(f"  编码器层小计 {enc_l:>12,} ×6 = {6*enc_l:>10,}")
    print(f"  解码器层小计 {dec_l:>12,} ×6 = {6*dec_l:>10,}")
    print(f"  词嵌入（三处共享） {emb_p:>12,}")
    print(f"  合计 {total:,} ≈ {total/1e6:.1f}M（Table 3 报 65M，差 {(65e6-total)/1e6:.1f}M）")
    print(f"  FFN 家族占比 {12*ffn_p/total:.1%}，注意力家族占比 {18*attn_p/total:.1%}，"
          f"嵌入占比 {emb_p/total:.1%}，LN 家族占比 {(6*2+6*3)*ln_p/total:.3%}")


if __name__ == "__main__":
    main()
