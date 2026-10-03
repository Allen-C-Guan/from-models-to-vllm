# 用途：Book2 ch4「四项手术」最小对照演示——①单塔折叠（cross-attn 参数税）②post→pre-LN（主干尺度对比）
#       ③学习式位置编码（1024 查表 vs 正弦闭合公式）④tied embeddings（124M 逐项账本+解绑代价），并输出形状流转表。
# 所属章节：Book2 第 4 章 §4.3-§4.7（块实现直接复用 warmup_ablation.py 的 Block/GPT——单轨依赖链 ch4→ch8 的族工厂前身）
# 运行方式：cd /Users/allen/Code/model_analysis && source env.sh && python code/ch04/surgery.py
# 产物：全部打印到 stdout（正文引用的数字均出自本脚本固定种子运行；CPU 秒级）

import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from warmup_ablation import GPT, Block  # noqa: E402  参数化块 = ch8 族模型工厂前身

SEED = 20261002  # 全书统一种子


# ---------------- 手术①：单塔折叠——给 Book1 式解码块把 cross-attention 装回去，量它的价 ----------------
class CrossAttention(nn.Module):
    """Book1 式解码块第二子层（cross-attention）的最小实现：Q 来自解码侧，K/V 来自 encoder 输出。

    输入 x (B,n,C) 与 memory (B,m,C)；输出 (B,n,C)。HF GPT2Attention 的 cross 分支同构
    （q_attn C→C + c_attn C→2C + c_proj C→C + ln_cross），此处按同口径计数。
    """

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.q_attn = nn.Linear(n_embd, n_embd)     # (B,n,C) -> (B,n,C)   Q 投影（来自解码侧）
        self.c_kv = nn.Linear(n_embd, 2 * n_embd)   # (B,m,C) -> (B,m,2C)  K/V 投影（来自 encoder）
        self.c_proj = nn.Linear(n_embd, n_embd)     # (B,n,C) -> (B,n,C)   输出投影

    def forward(self, x, memory):
        B, n, C = x.shape                                              # x (B,n,C), memory (B,m,C)
        m = memory.size(1)
        q = self.q_attn(x).view(B, n, self.n_head, C // self.n_head).transpose(1, 2)   # (B,h,n,dk)
        k, v = self.c_kv(memory).split(C, dim=2)
        k = k.view(B, m, self.n_head, C // self.n_head).transpose(1, 2)                # (B,h,m,dk)
        v = v.view(B, m, self.n_head, C // self.n_head).transpose(1, 2)                # (B,h,m,dk)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // self.n_head)                  # (B,h,n,m) ← m 维来自 encoder
        y = F.softmax(att, dim=-1) @ v                                 # (B,h,n,m)·(B,h,m,dk) -> (B,h,n,dk)
        y = y.transpose(1, 2).contiguous().view(B, n, C)               # (B,h,n,dk) -> (B,n,C)
        return self.c_proj(y)


def surgery_1_folding():
    """手术①对照：GPT 块（无 cross）vs Book1 式解码块（含 cross）；量参数税、演示 (B,h,n,m) 的 m 维消失。"""
    print("=" * 72)
    print("手术① 单塔折叠：Book1 双塔的解码块 vs GPT 单塔块（C=384, h=6 缩微档）")
    print("=" * 72)
    C, H = 384, 6
    torch.manual_seed(SEED)
    block_gpt = Block(C, H, block_size=64, ln_position="pre")     # GPT 式块：自注意力 + MLP，无 cross
    cross = CrossAttention(C, H)
    ln_cross = nn.LayerNorm(C)
    n_block = sum(p.numel() for p in block_gpt.parameters())
    n_cross = sum(p.numel() for p in cross.parameters()) + sum(p.numel() for p in ln_cross.parameters())
    print(f"  GPT 块（attn+mlp，无 cross）      : {n_block:,} 参数")
    print(f"  Book1 式解码块的 cross 子层       : {n_cross:,} 参数（q_attn+c_kv+c_proj+ln_cross）")
    print(f"  折叠省下（缩微档每层）            : {n_cross:,} 参数 = {n_cross / n_block * 100:.1f}% 的 GPT 块")
    # 124M 档换算（768 维、12 层——config 级算术，与 HF cross 分支字段一一对应）
    C768 = 768
    cross768 = 4 * C768 * C768 + 6 * C768             # q C·C + kv 2·C·C + proj C·C = 4C² 权重 + 4C 偏置（q/kv2/proj）+ 2C LN，与缩微档实测 592,128 同式
    print(f"  若给 124M 每层装回 cross 接口     : {cross768:,}/层 × 12 = {cross768 * 12:,} ≈ {cross768 * 12 / 1e6:.1f}M 参数")
    # m 维演示：折叠后注意力分数只剩 (B,h,n,n)
    x = torch.randn(2, 16, C)      # (B,n,C) 解码侧
    mem = torch.randn(2, 32, C)    # (B,m,C) encoder 侧（折叠后此输入不存在）
    att_cross = cross(x, mem)      # (B,n,C) —— 内部经过 (B,h,n,m=32)
    att_self, _ = block_gpt.attn(x), None  # (B,n,C) —— 内部只有 (B,h,n,n=16)
    print(f"  cross 分支注意力分数形状 (B,h,n,m) = (2,{H},16,32)；折叠后 (B,h,n,n) = (2,{H},16,16) —— m 维消失")
    return n_block, n_cross, cross768 * 12


# ---------------- 手术②：post→pre-LN——同一初始化流下，主干尺度的两种命运 ----------------
def surgery_2_preln():
    """手术②对照：post-LN 每站钉死 std≈1；pre-LN 残差主干无归一、std 逐层增长，最后由 ln_f 一次收口。"""
    print()
    print("=" * 72)
    print("手术② post→pre-LN：6 层主干逐层 std（输入 x~N(0,1)，(1,64,384)，固定种子）")
    print("=" * 72)
    L, C, H = 6, 384, 6
    torch.manual_seed(SEED)
    blocks_post = nn.ModuleList([Block(C, H, 64, ln_position="post") for _ in range(L)])
    torch.manual_seed(SEED)  # 同一 RNG 流：两种接法参数形状全同、初始值逐位一致
    blocks_pre = nn.ModuleList([Block(C, H, 64, ln_position="pre") for _ in range(L)])
    ln_f = nn.LayerNorm(C)
    x0 = torch.randn(1, 64, C)
    xp, xs = x0, x0
    print("  层号 | post-LN 出口 std | pre-LN 出口 std")
    print("  -----|------------------|-----------------")
    for i in range(L):
        xp = blocks_post[i](xp)     # post：每站 LN 钉回，出口 std≈1
        xs = blocks_pre[i](xs)      # pre：残差主干只加不归，std 逐层长大
        print(f"   {i + 1:2d}  |       {xp.std().item():.4f}      |      {xs.std().item():.4f}")
    print(f"  pre-LN 末尾 ln_f 收口后 std = {ln_f(xs).std().item():.4f}（手术②的两半：挪 LN + 补 ln_f）")
    return None


# ---------------- 手术③：学习式位置编码——1024 位查表 vs 正弦闭合公式 ----------------
def sinusoidal_pe(T: int, C: int) -> torch.Tensor:
    """Book1 式正弦位置编码（原论文式 5-8 的实现）：返回 (T,C)，零参数、任意 T 可算。"""
    pos = torch.arange(T).unsqueeze(1).float()                       # (T,1)
    i = torch.arange(0, C, 2).float()                                # (C/2,)
    div = torch.exp(i * -(math.log(10000.0) / (C - 2 + (C % 2))))    # 原文 10000^(2i/d)
    pe = torch.zeros(T, C)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: (C + 1) // 2])
    return pe


def surgery_3_pe():
    print()
    print("=" * 72)
    print("手术③ 学习式位置编码：wpe 查表 vs 正弦闭合公式（GPT-2 口径 C=768）")
    print("=" * 72)
    C = 768
    torch.manual_seed(SEED)
    wpe = nn.Embedding(1024, C)      # GPT-2 口径：1024 位查表，stddev=0.01 初始化（openai model.py）
    with torch.no_grad():
        nn.init.normal_(wpe.weight, std=0.01)
    ids = torch.arange(1024)
    lookup = wpe(ids)                                                # (1024,768) —— 查表
    sine = sinusoidal_pe(1024, C)                                    # (1024,768) —— 闭合公式
    print(f"  pos=0..1023 查表 wpe(ids) 形状 {tuple(lookup.shape)}，初始 std={lookup.std().item():.4f}")
    print(f"  正弦 PE 同位形状 {tuple(sine.shape)}，幅值范围 [{sine.min().item():.3f}, {sine.max().item():.3f}]")
    # 硬上限演示：pos=1024
    try:
        wpe(torch.tensor([1024]))
        print("  pos=1024 查表成功（不应发生）")
    except IndexError as e:
        print(f"  pos=1024 查表 -> IndexError（{e}）：1024 位之外没有向量可用，硬上限")
    beyond = sinusoidal_pe(4096, C)                                  # 正弦路线同一公式直接算到 4096
    print(f"  正弦 PE pos=4096 仍可计算（{tuple(beyond.shape)}）——「结构上可算」≠「训练后好用」")
    # 形状：进入主干前与词嵌入广播相加
    wte = nn.Embedding(50257, C)
    tok = wte(torch.randint(0, 50257, (2, 300)))                     # (B,n,C)
    x = tok + wpe(torch.arange(300))                                 # (B,n,C)+(n,C) 广播 -> (B,n,C)
    print(f"  词嵌入 {tuple(tok.shape)} + 位置嵌入 (n,C) 广播相加 -> {tuple(x.shape)}")
    return None


# ---------------- 手术④：tied embeddings——124M 逐项账本与解绑代价 ----------------
def ledger(vocab: int, ctx: int, n_layer: int = 12, n_embd: int = 768) -> dict:
    """参数逐项账本（config 级算术，与 HF Conv1D (in,out) 排布一一对应）。返回各项参数量。"""
    inner = 4 * n_embd
    per_ln = 2 * n_embd
    c_attn = n_embd * 3 * n_embd + 3 * n_embd
    c_proj = n_embd * n_embd + n_embd
    c_fc = n_embd * inner + inner
    c_proj2 = inner * n_embd + n_embd
    per_block = 2 * per_ln + c_attn + c_proj + c_fc + c_proj2
    return {
        "wte": vocab * n_embd, "wpe": ctx * n_embd, "per_block": per_block,
        "blocks": per_block * n_layer, "ln_f": per_ln,
    }


def surgery_4_tied():
    print()
    print("=" * 72)
    print("手术④ tied embeddings：124M 逐项账本 + 解绑代价 + 论文 117M 口径复算")
    print("=" * 72)
    d = ledger(50257, 1024)
    total = d["wte"] + d["wpe"] + d["blocks"] + d["ln_f"]
    print(f"  wte (50257×768, 输入=输出共享一份)  : {d['wte']:,}")
    print(f"  wpe (1024×768)                      : {d['wpe']:,}")
    print(f"  每块（ln×2+c_attn+c_proj+c_fc+proj）: {d['per_block']:,}  ×12 = {d['blocks']:,}")
    print(f"  ln_f                                : {d['ln_f']:,}")
    print(f"  合计（tied，lm_head 复用 wte 不计） : {total:,}")
    untied = total + d["wte"]
    print(f"  若解开绑定（另买一张输出矩阵）      : {untied:,}，多付 {d['wte']:,} = 总量的 {d['wte'] / total * 100:.1f}%")
    print(f"  嵌入合计占模型：{(d['wte'] + d['wpe']) / total * 100:.1f}%（wte+wpe / 全模型）")
    # meta 实例化对拍：账本算术 vs numel 实数
    with torch.device("meta"):
        m = GPT(n_layer=12, n_embd=768, n_head=12, vocab=50257, block_size=1024, ln_position="pre")
    assert m.n_params() == total, "账本与实例化不一致"
    print(f"  meta 实例化 numel 对拍              : {m.n_params():,} == 账本合计 ✓（逐位一致）")
    # 论文 117M 口径复算（GPT-1 词表 40478 / 上下文 512 —— 117M 考据框的算术支点）
    d1 = ledger(40478, 512)
    t1 = d1["wte"] + d1["wpe"] + d1["blocks"] + d1["ln_f"]
    with torch.device("meta"):
        m1 = GPT(n_layer=12, n_embd=768, n_head=12, vocab=40478, block_size=512, ln_position="pre")
    print(f"  GPT-1 口径（vocab 40478/ctx 512）   : {t1:,} ≈ 论文 Table 2 的「117M」（差 {d['wte'] - d1['wte'] + d['wpe'] - d1['wpe']:,} = 词表与上下文之差）")
    assert m1.n_params() == t1
    # medium 档同法复核（24/1024——正文 4.2 考据框「344.3M 对 345M / 实发 354.8M」的复现支点）
    d1m = ledger(40478, 512, n_layer=24, n_embd=1024)
    t1m = d1m["wte"] + d1m["wpe"] + d1m["blocks"] + d1m["ln_f"]
    d2m = ledger(50257, 1024, n_layer=24, n_embd=1024)
    t2m = d2m["wte"] + d2m["wpe"] + d2m["blocks"] + d2m["ln_f"]
    with torch.device("meta"):
        m1m = GPT(n_layer=24, n_embd=1024, n_head=16, vocab=40478, block_size=512, ln_position="pre")
        m2m = GPT(n_layer=24, n_embd=1024, n_head=16, vocab=50257, block_size=1024, ln_position="pre")
    assert m1m.n_params() == t1m and m2m.n_params() == t2m
    print(f"  medium 档 GPT-1 口径（24×1024）     : {t1m:,} ≈ 论文「345M」；实发口径（50257/1024）: {t2m:,} ≈ 实发 355M")
    return None


# ---------------- 形状流转表：GPT-2 Block 与整机（正文表 4.4 的数据源） ----------------
def shape_flow_table():
    print()
    print("=" * 72)
    print("形状流转表：GPT-2 单个 Block（pre-LN，B=2, n=64, C=384, h=6）")
    print("=" * 72)
    torch.manual_seed(SEED)
    blk = Block(384, 6, 64, ln_position="pre")
    x = torch.randn(2, 64, 384)
    rows = [
        ("输入 x", "-", "(2,64,384)", "(2,64,384)"),
        ("ln_1 归一", "LayerNorm(C)", "(2,64,384)", "(2,64,384)"),
        ("c_attn 投影", "Linear C→3C", "(2,64,384)", "(2,64,1152)"),
        ("切分 Q/K/V", "split(C,dim=2)", "(2,64,1152)", "各 (2,64,384)"),
        ("分头", "view+transpose", "(2,64,384)", "各 (2,6,64,64)"),
        ("注意力分数", "q@kᵀ/√64", "(2,6,64,64)", "(2,6,64,64)"),
        ("因果掩码+softmax", "tril 屏蔽", "(2,6,64,64)", "(2,6,64,64)"),
        ("加权 V", "att@v", "(2,6,64,64)", "(2,6,64,64)"),
        ("并头", "transpose+reshape", "(2,6,64,64)", "(2,64,384)"),
        ("c_proj", "Linear C→C", "(2,64,384)", "(2,64,384)"),
        ("残差相加", "x+attn(ln_1(x))", "(2,64,384)", "(2,64,384)"),
        ("ln_2→MLP", "ln_2; C→4C→GELU→C", "(2,64,384)", "(2,64,384)"),
        ("残差相加", "x+mlp(ln_2(x))", "(2,64,384)", "(2,64,384)"),
    ]
    for name, op, inp, out in rows:
        print(f"  {name:<10s} | {op:<18s} | {inp:<16s} -> {out}")
    print("  （整机再叠 12 块 → ln_f → h·wteᵀ：正文 §4.3 形状主干 / §4.7 解绑账）")


def main():
    print(f"[surgery.py] 固定种子 {SEED}，CPU 运行，数字供 ch4 正文引用\n")
    surgery_1_folding()
    surgery_2_preln()
    surgery_3_pe()
    surgery_4_tied()
    shape_flow_table()
    print("\n[完成] 四项手术对照全部输出；停药实验曲线数据见 log/book2-ch04/warmup_full__*.csv")


if __name__ == "__main__":
    main()
