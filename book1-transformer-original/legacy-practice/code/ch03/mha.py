# 用途：第 3 章多头注意力——手写 MHA（不用 nn.MultiheadAttention）+ in_proj 拆包对拍 + 多头分工构造演示
# 所属章节：Book1《Transformer 原典》第 3 章（drafts/Book1-Transformer原典/ch03-多头注意力.md）
# 运行方式：source env.sh && python code/ch03/mha.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 1. 手写多头注意力（论文 §3.2.2：投影 -> 切头 -> 按头缩放点积 -> 拼回 -> W^O） ----------

class MyMultiHeadAttention(nn.Module):
    """多头注意力，输入约定 (batch, seq, d_model)——等价 torch 官方的 batch_first=True。

    I/O 形状：forward(q, k, v)
        q: (bs, nq, d_model)，k / v: (bs, nk, d_model)（自注意力时 nq=nk、三路同源）
        返回 (out, attn)：out (bs, nq, d_model)，attn (bs, h, nq, nk)（全头权重，不沿头维平均）

    手写注意力一律 matmul+softmax（不用 SDPA），SDPA 仅在第 2 章作对照。
    """

    def __init__(self, d_model: int, h: int, bias: bool = True):
        super().__init__()
        assert d_model % h == 0, "d_model 必须能被头数 h 整除（d_k=d_v=d_model/h）"
        self.d_model, self.h, self.d_k = d_model, h, d_model // h
        # 三个投影各打包 h 份：W^Q 沿输出维按头排成 (d_model, h*d_k)=(512,512)，与单头全维同参数量
        self.w_q = nn.Linear(d_model, d_model, bias=bias)   # 权重 (d_model, d_model)=(512,512)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)  # W^O ∈ R^{h*d_v × d_model}，权重 (512,512)

    def forward(self, q, k, v, mask=None):
        """前向：投影 -> 切头 -> 按头缩放点积 -> 拼回 -> W^O。

        输入 q (bs,nq,512)、k/v (bs,nk,512) -> 返回 out (bs,nq,512)、attn (bs,h,nq,nk)。
        mask：bool 张量，True=屏蔽（与官方 nn.MultiheadAttention 同语义），可广播到 (bs,h,nq,nk)。
        """
        bs, nq, _ = q.shape
        nk = k.shape[1]
        # 投影后切头：(bs,n,512) -> (bs,n,h,64) -> (bs,h,n,64)，把每头送入自己的 64 维子空间
        Q = self.w_q(q).view(bs, nq, self.h, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(bs, nk, self.h, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(bs, nk, self.h, self.d_k).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (bs,h,nq,nk)：h 个小注意力一次算完
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))    # (bs,h,nq,nk) 形状不变
        attn = F.softmax(scores, dim=-1)          # (bs,h,nq,nk)：逐行归一，每个 query 对全部 key 的分布
        head_out = attn @ V                       # (bs,h,nq,64)：按头加权求和
        concat = head_out.transpose(1, 2).reshape(bs, nq, self.h * self.d_k)  # 拼回 (bs,n,512)
        return self.w_o(concat), attn             # out (bs,nq,512)，attn (bs,h,nq,nk)


# ---------- 2. 形状流打印（base 配置：d_model=512, h=8, batch=2, n=10） ----------

def shape_flow(d_model=512, h=8, bs=2, n=10):
    """打印一次前向的 8 行形状流（正文表 3.1 的数据源）。

    I/O 形状：x (bs,n,d_model)=(2,10,512) -> out (2,10,512)；中间量见各行打印，
    核心四维口径：切头后 (2,8,10,64)、打分/权重 (2,8,10,10)。
    """
    m = MyMultiHeadAttention(d_model, h).eval()
    x = torch.randn(bs, n, d_model)
    d_k = d_model // h
    with torch.no_grad():
        proj_q, proj_v = m.w_q(x), m.w_v(x)      # (2,10,512)：h 份 64 维投影沿输出维打包成一块
        Q = proj_q.view(bs, n, h, d_k).transpose(1, 2)   # (2,10,8,64) -> (2,8,10,64)
        K = m.w_k(x).view(bs, n, h, d_k).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (2,8,10,64)x(2,8,64,10) -> (2,8,10,10)
        attn = F.softmax(scores, dim=-1)          # (2,8,10,10) 形状不变
        head_out = attn @ proj_v.view(bs, n, h, d_k).transpose(1, 2)  # (2,8,10,64)
        concat = head_out.transpose(1, 2).reshape(bs, n, d_model)     # (2,10,512)
        out = m.w_o(concat)                       # (2,10,512)
    steps = [("input (Q=K=V=x)", x), ("x·W^Q (h 份按头打包)", proj_q),
             ("view+transpose 切头", Q), ("scores = QK^T/√d_k", scores),
             ("attn = softmax(scores)", attn), ("head output = attn·V", head_out),
             ("concat h 个头", concat), ("output = concat·W^O", out)]
    for name, t in steps:
        print(f"  {name:<26} {str(tuple(t.shape))}")


# ---------- 2b. §3.3 对拍：官方权重整块搬入手写 MHA，同输入比前向 ----------

def compare_weight_transfer(d_model=512, h=8, bs=2, n=10):
    """把官方 in_proj_weight/bias 按 chunk(3) 整块搬入本书 w_q/w_k/w_v、out_proj 搬入 w_o，
    同输入对拍前向输出；并实测官方 need_weights=True 沿头维平均返回的权重形状。

    I/O 形状：x (bs,n,d_model)=(2,10,512)；两侧输出均为 (2,10,512)；
    in_proj_weight (3*d_model, d_model)=(1536,512) 沿 dim0 切三段、各 (512,512) 整块直搬；
    官方第二返回值 (bs,n,n)=(2,10,10)（h 个头沿头维平均），本书 attn (bs,h,nq,nk)=(2,8,10,10)。
    """
    torch.manual_seed(3)
    official = nn.MultiheadAttention(d_model, h, batch_first=True).eval()  # batch_first 必须显式传
    mine = MyMultiHeadAttention(d_model, h).eval()
    with torch.no_grad():
        Wq, Wk, Wv = official.in_proj_weight.chunk(3, dim=0)  # (1536,512) -> 三段各 (512,512)，torch 两侧权重同为 (out,in)，整块直搬
        bq, bk, bv = official.in_proj_bias.chunk(3, dim=0)    # (1536,) -> 三段各 (512,)
        mine.w_q.weight.copy_(Wq), mine.w_q.bias.copy_(bq)
        mine.w_k.weight.copy_(Wk), mine.w_k.bias.copy_(bk)
        mine.w_v.weight.copy_(Wv), mine.w_v.bias.copy_(bv)
        mine.w_o.weight.copy_(official.out_proj.weight)       # W^O
        mine.w_o.bias.copy_(official.out_proj.bias)
        x = torch.randn(bs, n, d_model)
        # need_weights=True 走 bmm+softmax 慢路径（最贴近论文公式，也避开推理 fast path）
        out_official, w_official = official(x, x, x, need_weights=True)
        out_mine, attn_mine = mine(x, x, x)
    diff = (out_mine - out_official).abs().max().item()
    ok = torch.allclose(out_mine, out_official, rtol=0, atol=1e-6)
    print(f"  官方输出 vs 权重搬入手写 MHA：max|Δ| = {diff:.2e}，allclose(atol=1e-6) = {ok}")
    print(f"  官方 need_weights=True 第二返回值形状 {tuple(w_official.shape)}（{h} 个头沿头维平均；"
          f"本书实现保留全量 (bs,h,nq,nk) = {tuple(attn_mine.shape)}）")
    return ok


# ---------- 3. in_proj 之谜：拆包官方权重，逐头投影对拍 ----------

def compare_with_torch(d_model=512, h=8, bs=2, n=10):
    """官方 in_proj_weight (3d,d) 沿 dim0 打包 W^Q,W^K,W^V；拆包后逐头投影，与官方输出 allclose。

    I/O 形状：x (bs,n,d_model)=(2,10,512)；in_proj_weight (1536,512)；
    每头小投影 x (2,10,512) @ W_i (512,64) -> qi/ki/vi (2,10,64)；
    每头注意力 a (2,10,10)、输出 (2,10,64)；h 份沿末维拼回 (2,10,512)，过 out_proj 后仍 (2,10,512)。
    """
    torch.manual_seed(3)
    official = nn.MultiheadAttention(d_model, h, batch_first=True).eval()  # batch_first 必须显式传
    x = torch.randn(bs, n, d_model)
    with torch.no_grad():
        # need_weights=True 走 bmm+softmax 慢路径（最贴近论文公式，也避开推理 fast path）
        out_official, _ = official(x, x, x, need_weights=True)
        # torch 存 (out,in) 且沿 dim0 打包；先转置成论文记法 (in=512, out=512)，再按列块取每头
        Wq, Wk, Wv = (w.T for w in official.in_proj_weight.chunk(3, dim=0))
        bq, bk, bv = official.in_proj_bias.chunk(3, dim=0)
        d_k = d_model // h
        heads = []
        for i in range(h):
            cols = slice(i * d_k, (i + 1) * d_k)  # 第 i 头 = W 的第 i 个 64 列列块
            qi = x @ Wq[:, cols] + bq[cols]        # (2,10,512)x(512,64) -> (2,10,64)；即 torch 权重的第 i 个行块（两者互为转置）
            ki = x @ Wk[:, cols] + bk[cols]        # (2,10,64)
            vi = x @ Wv[:, cols] + bv[cols]        # (2,10,64)
            a = F.softmax(qi @ ki.transpose(-2, -1) / math.sqrt(d_k), dim=-1)  # (2,10,10)
            heads.append(a @ vi)                   # (2,10,64)
        out_manual = official.out_proj(torch.cat(heads, dim=-1))  # 拼回 (2,10,512) 后过 W^O
    diff = (out_manual - out_official).abs().max().item()
    ok = torch.allclose(out_manual, out_official, rtol=0, atol=1e-6)
    print(f"  官方输出 vs 拆包逐头投影：max|Δ| = {diff:.2e}，allclose(atol=1e-6) = {ok}")
    return ok


# ---------- 4. 投影参数量算术：(A) 行 h 扫描不变；(B) 行缩 d_k 下降并核回论文 60M/58M ----------

def projection_arithmetic(d_model=512, h_base=8):
    """数参数量（标量计数，无张量流转）：(A) 行证 h·d_k=d_model 下参数量与 h 无关；
    (B) 行证只有 W^Q/W^K 随 d_k 变瘦，逐层扣减核回论文 Table 3 的 60M/58M。"""
    print("  (A) 行头数扫描：d_k=d_v=d_model/h，h·d_k 恒等于 d_model -> 投影参数量与 h 无关")
    for h in (1, 4, 8, 16, 32):
        n_par = sum(p.numel() for p in MyMultiHeadAttention(d_model, h).parameters())
        print(f"    h={h:>2}  d_k=d_v={d_model // h:>3}  单个 MHA 参数量 = {n_par:,}")
    print("  (B) 行缩 d_k（h 固定 8，d_v=64 不变）：只有 W^Q/W^K 变瘦")
    base = None
    for d_k in (64, 32, 16):
        n_par = (2 * h_base * d_model * d_k          # W^Q 与 W^K：h 份 (512,d_k)
                 + h_base * d_model * 64             # W^V：d_v 不随 d_k 缩
                 + d_model * d_model                 # W^O：随 h·d_v，不变
                 + 2 * h_base * d_k + h_base * 64 + d_model)  # 偏置
        base = base or n_par
        print(f"    d_k={d_k:>2}  单个 MHA 参数量 = {n_par:,}（较 base 少 {base - n_par:,}）")
    n_attn = 6 * 1 + 6 * 2  # base 有 18 个注意力子层：编码器每层 1 个 + 解码器每层 2 个（见第 4 章）
    for d_k, paper in ((32, "60M"), (16, "58M")):
        drop = (2 * h_base * d_model * (64 - d_k) + 2 * h_base * (64 - d_k)) * n_attn
        print(f"    d_k={d_k:>2}: 65M - {drop / 1e6:.2f}M = {(65e6 - drop) / 1e6:.1f}M，论文 Table 3 记 {paper}")


# ---------- 5. 多头分工构造演示（合成输入 + 手工构造投影；是构造演示，非训练所得） ----------

SENT = ["<s>", "the", "old", "house", "that", "stood", "here", "was", "built", "in", "nineteen", "."]
TYPE = ["S", "D", "A", "N", "D", "V", "ADV", "V", "V", "P", "NUM", "S"]  # 词类（D/N/V 等成类重现）
TYPE_ID = {t: i for i, t in enumerate(["S", "D", "A", "N", "V", "ADV", "P", "NUM"])}
VERBNESS = {"<s>": .10, "the": .10, "old": .10, "house": .10, "that": .10, "stood": .45,
            "here": .10, "was": .45, "built": 1.0, "in": .10, "nineteen": .10, ".": .10}


def build_demo(d_model=64, h=4):
    """合成输入 x（常数维/动词度/词类 one-hot/正弦 PE 四块）+ 四个手工构造的 W^Q/W^K。

    I/O 形状：返回 (m, x)；x (n,d_model)=(12,64)；WQ/WK (h,d_model,d_k)=(4,64,16)，
    即每头 W_i (64,16)；前向时 x 加批维 (1,12,64)，注意力权重 (1,4,12,12)。
    增益 g 用于补偿 1/√d_k 缩放、使模式可见；真实模型中投影尺度由训练习得。
    W^V/W^O 保持随机初始化（演示只看注意力权重，不看输出）。
    """
    n, d_k = len(SENT), d_model // h
    x = torch.zeros(n, d_model)                                     # (12,64)
    x[:, 0] = 1.0                                                   # dim0：常数锚点维
    x[:, 1] = torch.tensor([VERBNESS[t] for t in SENT])             # dim1：动词度
    for j, t in enumerate(TYPE):                                    # dim2..9：词类 one-hot
        x[j, 2 + TYPE_ID[t]] = 1.0
    pos = torch.arange(n, dtype=torch.float).unsqueeze(1)           # dim16..63：正弦 PE（48 维）
    div = torch.exp(torch.arange(0, 48, 2, dtype=torch.float) * (-math.log(10000.0) / 48))
    x[:, 16:64:2] = torch.sin(pos * div)                            # 偶维 sin，奇维 cos
    x[:, 17:64:2] = torch.cos(pos * div)

    m = MyMultiHeadAttention(d_model, h, bias=False).eval()
    g = 4.0
    WQ, WK = torch.zeros(h, d_model, d_k), torch.zeros(h, d_model, d_k)
    # 头 0 动词锚点（内容子空间）：score ∝ (1+s_i)·s_j，所有 query 都盯住 built
    WQ[0, 0, 0], WQ[0, 1, 0], WK[0, 1, 0] = g, g, g
    # 头 1 前一词（位置子空间）：两对 PE 频率（f=1 与 0.464）叠加破除单频周期混叠，
    # K 侧各旋 f -> score ∝ cos(i-j-1) + cos(0.464·(i-j-1))，峰在 j=i-1
    for m_, p_ in ((0, 0), (1, 2)):
        f_ = 10000 ** (-2 * p_ / 48)
        WQ[1, 16 + 2 * p_, 2 * m_] = 3.5
        WQ[1, 17 + 2 * p_, 2 * m_ + 1] = 3.5
        WK[1, 16 + 2 * p_, 2 * m_] = 3.5 * math.cos(f_)
        WK[1, 17 + 2 * p_, 2 * m_] = 3.5 * math.sin(f_)
        WK[1, 16 + 2 * p_, 2 * m_ + 1] = -3.5 * math.sin(f_)
        WK[1, 17 + 2 * p_, 2 * m_ + 1] = 3.5 * math.cos(f_)
    # 头 2 句首锚点（位置子空间）：q 取常数维，k 取低频对（f=0.1·... 取第 3 对）cos 分量 -> 峰在 j=0
    p = 3
    WQ[2, 0, 0], WK[2, 16 + 2 * p + 1, 0] = 1.5 * g, 1.5 * g
    # 头 3 同词类（内容子空间）：q、k 都取词类 one-hot -> score ∝ [type_i == type_j]
    for c in range(8):
        WQ[3, 2 + c, c], WK[3, 2 + c, c] = g, g
    with torch.no_grad():
        # 论文记法打包：第 i 头的 W 放进列块 [i*d_k,(i+1)*d_k)，再转置成 torch 的 (out,in) 装填
        m.w_q.weight.copy_(WQ.transpose(0, 1).reshape(d_model, h * d_k).T)
        m.w_k.weight.copy_(WK.transpose(0, 1).reshape(d_model, h * d_k).T)
    return m, x


def run_demo():
    """跑构造演示并打印各头关注质量与行熵。I/O 形状：x (12,64) -> (1,12,64) 入前向，
    返回 A (h,n,n)=(4,12,12)；打印的四组质量/行熵均由 A 沿行归约而来。"""
    m, x = build_demo()
    with torch.no_grad():
        _, attn = m(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))  # (1,4,12,12)
    A = attn[0]                                   # (h, n, n) = (4,12,12)
    n, built = A.shape[-1], SENT.index("built")
    same = torch.tensor([[TYPE[a] == TYPE[b] for b in range(n)] for a in range(n)])
    metrics = {"verb-anchor": A[0][:, built].mean(),
               "prev-token": A[1][torch.arange(1, n), torch.arange(n - 1)].mean(),
               "start-anchor": A[2][:, 0].mean(),
               "same-class": (A[3] * same).sum(-1).mean()}
    print(f"  均匀分布的行熵为 ln({n}) = {math.log(n):.3f}")
    for i, (name, val) in enumerate(metrics.items()):
        ent = -(A[i] * A[i].clamp_min(1e-12).log()).sum(-1).mean()
        print(f"  头 {i} {name:<13} 关注质量 {val.item():.3f}  行熵 {ent.item():.3f}")
    return A


if __name__ == "__main__":
    print("== 1) 手写 MHA 形状流（d_model=512, h=8, batch=2, n=10）==")
    shape_flow()
    print("== 2) 官方权重整块搬入手写 MHA 对拍（§3.3）==")
    compare_weight_transfer()
    print("== 3) in_proj 拆包逐头投影对拍 nn.MultiheadAttention（§3.4）==")
    compare_with_torch()
    print("== 4) 投影参数量算术 ==")
    projection_arithmetic()
    print("== 5) 多头分工构造演示（合成输入+构造投影，非训练所得）==")
    run_demo()
