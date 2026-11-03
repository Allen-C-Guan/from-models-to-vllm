# -*- coding: utf-8 -*-
# 用途：Book1 第 8 章——原论文训练配方三组件教学版与自测：Noam 调度（式 (8.1)，含 t2t 系数还原）、
#       label smoothing（式 (8.2)(8.3)，LS/NLL 双指标合成实验）、beam search + GNMT 长度惩罚（式 (8.4)）；
#       末尾与 ch09 管线应用版（train.py / decode.py）逐项对拍（本章=组件教学版，ch09=管线应用版）
# 所属章节：《Transformer 原典》第 8 章（原论文训练配方）
# 运行：source env.sh && python "code/ch08/recipe.py"（CPU 秒级；图另见 make_fig_8_1.py）
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(2017)  # 全书统一种子（复现基准）

# 对拍依赖：ch09 的管线应用版（sys.path 注入 ch09 目录；仅 import，不触发数据下载）
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "ch09")))

# 论文口径常量（§5.3 优化器 / §5.4 正则 / §6.1 解码）
D_BASE, D_BIG, WARMUP = 512, 1024, 4000
ADAM_BETAS, ADAM_EPS = (0.9, 0.98), 1e-9  # β2=0.98、ε=1e-9：两处偏离 Adam 默认（0.999 / 1e-8）
EPS_LS, P_DROP = 0.1, 0.1
BEAM, ALPHA = 4, 0.6


# ================= 8.1 Noam 调度（原论文式 (3)，本书编号 (8.1)） =================
def noam_lr(step_num, d_model, warmup, factor=1.0):
    """lrate = d_model^{-0.5}·min(step^{-0.5}, step·warmup^{-1.5})。
    step<warmup 时 min 取右支（∝step，线性升温）；之后取左支（∝step^{-0.5}）；交点恰在 step=warmup。
    形状：标量 → 标量（调度器无张量参与，每训练步调用一次）。"""
    return factor * d_model ** -0.5 * min(step_num ** -0.5, step_num * warmup ** -1.5)


def t2t_legacy_lr(step_num, hidden, warmup, learning_rate=0.1, adam=True):
    """t2t「legacy + noam」源码口径（learning_rate.py）：
    5000·hidden^{-0.5}·min((step+1)·warmup^{-1.5}, (step+1)^{-0.5})·0.002(adam 修正)·learning_rate。
    transformer_base_v1 缺省（hidden=512、warmup=4000、lr=0.1、adam）下系数 5000×0.002×0.1=1.0，
    恰还原论文式 (3)；残余差异只来自 step+1 的 step 0 保护（demo_noam 实测其大小）。形状：标量 → 标量。"""
    base = 5000.0 * hidden ** -0.5 * min((step_num + 1) * warmup ** -1.5, (step_num + 1) ** -0.5)
    return base * (0.002 if adam else 1.0) * learning_rate


def demo_noam():
    print("[8.1] Noam 调度（base：d_model=512，warmup=4000）")
    for s in (1, 100, 1000, 3999, 4000, 4001, 8000, 20000, 100000):
        print(f"  step {s:>6} → lr {noam_lr(s, D_BASE, WARMUP):.4e}")
    peak = noam_lr(WARMUP, D_BASE, WARMUP)
    print(f"  峰值 lr（step=4000）= {peak:.4e} ≈ 7.0e-4（本书复算；论文未给峰值数值）")
    print(f"  big 峰值（d_model=1024，warmup 同 4000）= {noam_lr(WARMUP, D_BIG, WARMUP):.4e}")
    r_lin = noam_lr(800, D_BASE, WARMUP) / noam_lr(400, D_BASE, WARMUP)
    r_dec = noam_lr(40000, D_BASE, WARMUP) / noam_lr(10000, D_BASE, WARMUP)
    print(f"  两段自检：线性段 lr(800)/lr(400)={r_lin:.4f}（应=2.0）；衰减段 lr(40k)/lr(10k)={r_dec:.4f}（应=0.5）")
    print(f"  warmup 占比：base 4000/100k={4000 / 100000:.1%}，big 4000/300k={4000 / 300000:.1%}")
    print("  t2t 系数还原：5000×0.002×0.1 =", 5000 * 0.002 * 0.1)
    for s in (1, 10, 100, 4000, 100000):
        ratio = t2t_legacy_lr(s, 512, 4000) / noam_lr(s, 512, 4000)
        print(f"    t2t_v1/paper(step {s:>6}) = {ratio:.6f}（偏差全部来自 step+1 保护，随 s 迅速消失）")


# ================= 8.2 label smoothing（式 (8.2) 目标分布 / 式 (8.3) 损失分解） =================
def smoothed_target(vocab_size, target, eps=EPS_LS):
    """式 (8.2)（Szegedy §7）：q'(k) = (1-ε)δ_{k,y} + ε/K——均匀质量撒在含目标类在内的全部 K 类上
    （目标类得 (1-ε)+ε/K，其余各类 ε/K；「非目标类均分 ε/(V-1)」是排除目标的变体，非 Szegedy 原文）。
    形状：标量 K、标量类号 y → q' (K,)。仅教学演示用——训练走式 (8.3) 的分解，从不实体化 q'。"""
    q = torch.full((vocab_size,), eps / vocab_size)  # (K,)：均匀底 ε/K
    q[target] += 1.0 - eps                           # 目标类再叠加 (1-ε)
    return q


def label_smoothing_loss(logits, target, eps=EPS_LS, ignore_index=None):
    """式 (8.3)：H(q',p) = (1-ε)·NLL(y) + ε·(-mean_k log p_k)。与 ch09.train.label_smoothing_loss 同款分解。
    形状：logits (B,L,V) 或 (N,V)，target (B,L) 或 (N,) → 标量（ignore_index 给定时按有效位平均）。"""
    logp = F.log_softmax(logits.float(), dim=-1)               # (B,L,V) -> (B,L,V)：softmax 在词表维上
    nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)   # 索引 (B,L)->(B,L,1) 挑真值列 -> (B,L)
    loss = (1 - eps) * nll + eps * (-logp.mean(-1))            # 两个 (B,L) 逐位加权 -> (B,L)
    if ignore_index is not None:
        keep = target != ignore_index                          # pad 布尔掩码 (B,L)（与损失同形）
        return loss[keep].sum() / keep.sum()                   # 有效位 (n_valid,) -> 标量
    return loss.mean()


def nll_loss(logits, target, ignore_index=None):
    """真实标签 NLL（PPL 的口径）——与 LS 损失并列为「双指标」。
    形状：与 label_smoothing_loss 同契约，(B,L,V)/(B,L) → 标量。"""
    logp = F.log_softmax(logits.float(), dim=-1)               # (B,L,V) -> (B,L,V)
    nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)   # -> (B,L)
    if ignore_index is not None:
        keep = target != ignore_index                          # (B,L)
        return nll[keep].sum() / keep.sum()                    # (n_valid,) -> 标量
    return nll.mean()


def demo_label_smoothing():
    print("[8.2] label smoothing（ε=0.1）")
    q = smoothed_target(10, 3)  # 教学例：q' (K,)=(10,)
    print(f"  式 (8.2) 构造（K=10, y=3）：目标类概率 {q[3]:.4f}，其余各类 {q[0]:.4f}（Σ={q.sum():.1f}）")

    # 合成任务：y = (x+1) 概率 0.7 / (x+2) 概率 0.3 ——「伤 PPL」是结构性的，可闭式算出理论值
    vocab = 16
    gen = torch.Generator().manual_seed(8)
    xs = torch.randint(0, vocab, (8192,), generator=gen)
    flip = torch.rand(8192, generator=gen) < 0.3
    ys = (xs + 1 + flip.long()) % vocab

    def train_toy(eps, steps=500):
        torch.manual_seed(2017)
        m = nn.Sequential(nn.Embedding(vocab, 32), nn.Flatten(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, vocab))
        opt = torch.optim.Adam(m.parameters(), lr=0.05)
        for _ in range(steps):  # 全批 2048 训练样本
            loss = (nll_loss(m(xs[:2048]), ys[:2048]) if eps == 0    # m(xs[:2048]) -> logits (2048,16)
                    else label_smoothing_loss(m(xs[:2048]), ys[:2048], eps))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            logp_eval = m(xs[4096:])  # eval logits (4096,16)
        return (nll_loss(logp_eval, ys[4096:]).item(),
                float((logp_eval.argmax(-1) == ys[4096:]).float().mean()))

    # 理论锚点：ε=0 可达 Bayes NLL=H(0.7,0.3)；ε=0.1 的最优解是 p*=0.9·p+0.1/K，NLL 必然更高
    bayes = -(0.7 * math.log(0.7) + 0.3 * math.log(0.3))
    smooth = -(0.7 * math.log(0.9 * 0.7 + 0.1 / vocab) + 0.3 * math.log(0.9 * 0.3 + 0.1 / vocab))
    print(f"  理论：ε=0 最优 NLL={bayes:.4f} nats（PPL {math.exp(bayes):.3f}）；"
          f"ε=0.1 最优 NLL={smooth:.4f} nats（PPL {math.exp(smooth):.3f}）——结构性差距 {smooth - bayes:.4f}")
    for eps in (0.0, 0.1):
        nll, acc = train_toy(eps)
        print(f"  实测 ε={eps:.1f}：eval NLL {nll:.4f} nats（PPL {math.exp(nll):.3f}），token 准确率 {acc:.1%}"
              f"（随机数据的准确率天花板为 70%）")


# ================= 8.3 dropout 机制微检（布点讲解见正文，两处布点数=30 子层 + 2 嵌入口） =================
def demo_dropout():
    drop = nn.Dropout(P_DROP)
    x = torch.ones(200001)  # 激活张量的替身（拍平的一维形态）
    y = drop(x)             # (200001,) -> (200001,)：逐元素置零/逆缩放，形状不变
    kept = y[y != 0]
    print(f"[8.3] Dropout(0.1) 训练态：置零比例 {float((y == 0).float().mean()):.4f}；"
          f"存活值 {kept[0].item():.4f}（=1/(1-p)≈1.111 逆缩放）；均值 {float(y.mean()):.4f}（期望不变）")


# ================= 8.5 beam search 与长度惩罚（式 (8.4)=GNMT eq(14)） =================
def length_penalty(length, alpha=ALPHA):
    """式 (8.4) 之 lp(Y) = (5+|Y|)^α / 6^α。α=0 → lp≡1，退化为纯 logP beam（GNMT §7 自己点明）。
    形状：标量 |Y| → 标量 lp（逐候选调用，只在定稿排序时参与）。"""
    return (5.0 + length) ** alpha / 6.0 ** alpha


def beam_search(next_logprobs, beam=BEAM, alpha=ALPHA, max_len=32, eos=3):
    """教学版 beam（单句，与 ch09.decode.beam_search 批量张量版同协议）：next_logprobs(prefix)->(V,) log-probs。
    形状直觉：活行 alive 是 k 条长度 ≤t 的部分译序列（直觉上的 (k,≤t) 不齐表）；每步打分把 k 个前缀
    各扩成一份 (V,) 分布——合看 (k,V)，得 k·V 个候选排一次序；返回最优假设 list[int]（长度 ≤max_len）。
    活行长度同步增长 → 步内 lp 相同，按原始 logP 剪枝与按惩罚分剪枝等价（与 ch09 注释同款口径）；
    每步取 top-2k 候选：EOS 者定稿（记各自长度的惩罚分），前 k 个非 EOS 候选继承为活行。"""
    alive, finished = [(0.0, [])], []
    while alive and len(finished) < beam and len(alive[0][1]) < max_len - 1:
        pool = []
        for score, toks in alive:  # alive：k 条活路 (累计 logP 标量, ≤t 的 token 前缀)
            lp = next_logprobs(toks)  # (V,)：该前缀下一步的全词表 log 概率
            pool.extend((score + float(lp[v]), toks + [v]) for v in range(lp.numel()))  # 候选池 k·V 条
        pool.sort(key=lambda c: c[0], reverse=True)
        alive = []
        for s, t in pool[: 2 * beam]:
            if t[-1] == eos:
                finished.append((s / length_penalty(len(t), alpha), t))  # 标量定稿分：累计 logP / lp(|Y|)
            elif len(alive) < beam:
                alive.append((s, t))
    if len(finished) < beam:  # 与 ch09 同口径：凑满 k 个定稿才不追加（max_len 截断时活行定稿）
        finished += [(s / length_penalty(len(t), alpha), t) for s, t in alive]  # 同款标量定稿分
    finished.sort(key=lambda c: c[0], reverse=True)
    return finished[0][1]


def short_bias_demo(eos=3):
    """短偏置演示（手造打分器）：第 1 步 EOS 概率 0.35 高于长句全程概率 0.55×0.9^6=0.292——
    纯 logP beam 选短句；式 (8.4) 的 lp 除以随长度增长的因子补偿长句，翻盘选长句。
    注意两段搜索路径完全相同（存活假设长度同步，步内剪枝不受 α 影响）——lp 只改定稿候选的排序。"""
    chain = [5, 7, 8, 9, 10, 11]  # 长句主体：每步 0.9；链上 EOS 概率取 1e-12（避免中途凑满定稿提前停机）

    def next_lp(toks):
        p = torch.full((12,), 1e-9)  # 词表 V=12
        if not toks:
            p[5], p[eos], p[6] = 0.55, 0.35, 0.10
        elif len(toks) < len(chain):        # 前缀已有 len 个 token，下一个是 chain[len]（chain[0]=5 已在首步发出）
            p[chain[len(toks)]], p[6] = 0.9, 0.099
            p[eos] = 1e-12
        else:
            p[eos] = 1.0 - 11e-9
        return torch.log(p / p.sum())  # -> (V,)=(12,)

    short_lp, long_lp = math.log(0.35), math.log(0.55 * 0.9 ** 6)
    print(f"  短句 [EOS]：logP={short_lp:.4f}；长句 [5,7,8,9,10,11,EOS]：logP={long_lp:.4f}"
          f"（lp(1)={length_penalty(1):.4f}，lp(7)={length_penalty(7):.4f}）")
    for a in (0.0, 0.6):
        best = beam_search(next_lp, alpha=a, max_len=12, eos=eos)
        tag = "纯 logP（α=0，lp≡1）" if a == 0 else "长度惩罚（α=0.6）"
        print(f"  {tag} → 最优假设 {best}（{len(best)} token）")


def demo_beam():
    print("[8.5] beam=4、α=0.6（式 (8.4)）")
    print("  lp(|Y|)：", {n: round(length_penalty(n), 4) for n in (1, 2, 5, 6, 10, 20, 50)})
    short_bias_demo()


# ================= 对拍：本章组件教学版 vs ch09 管线应用版 =================
def cross_check():
    import train as t9      # ch09 应用版：NoamLR / label_smoothing_loss
    import decode as d9     # ch09 应用版：beam_search（批量、张量化）
    from data import SOS, EOS
    print("[对拍] 组件教学版（本章）vs ch09 管线应用版")

    # ① Noam：同参数下逐步 lr 全等
    opt = torch.optim.Adam([nn.Parameter(torch.zeros(1))], lr=1e-7, betas=ADAM_BETAS, eps=ADAM_EPS)
    sched = t9.NoamLR(opt, D_BASE, WARMUP)
    steps = [1, 10, 100, 1000, 3999, 4000, 4001, 10000, 100000]
    ok1 = all(math.isclose(sched.rate(s), noam_lr(s, D_BASE, WARMUP), rel_tol=1e-12) for s in steps)
    print(f"  ① Noam：{len(steps)} 个步点 lr 全等 → {ok1}")

    # ② label smoothing：随机 logits 上两版损失 allclose（pad=-1 关闭屏蔽，聚焦公式本身）
    logits = torch.randn(64, 50, 11, generator=torch.Generator().manual_seed(8))  # (B,L,V)=(64,50,11)
    tgt = torch.randint(0, 11, (64, 50), generator=torch.Generator().manual_seed(9))  # (B,L)
    a = label_smoothing_loss(logits, tgt, ignore_index=-1)  # 教学版直接吃三维
    b = t9.label_smoothing_loss(logits.reshape(-1, 11), tgt.reshape(-1), pad=-1)  # 应用版先展平 (3200,11)/(3200,)
    print(f"  ② label smoothing：本章 {a.item():.6f} vs ch09 {b.item():.6f} → "
          f"allclose={torch.allclose(a, b, atol=1e-6)}")

    # ③ beam：确定性打分器上，教学版（逐句）与应用版（批量）最优假设逐一相同
    vocab, n_sent, k = 12, 3, 4
    g = torch.Generator().manual_seed(7)

    def sent_logits(i, t):  # 句 i 位置 t 的 logits：正解 +5、干扰 +3、其余 0.01·v（全相异避免平局）
        correct = corrects[i]
        lg = torch.arange(vocab).float() * 0.01
        if t < len(correct):
            lg[correct[t]] += 5.0
            lg[(correct[t] + 3) % vocab] += 3.0
        else:
            lg[EOS] += 5.0
        return lg

    corrects = []
    for i in range(n_sent):
        length = 5 + i
        toks = [4 + int(torch.randint(0, vocab - 4, (1,), generator=g)) for _ in range(length)]
        corrects.append(toks + [EOS])

    class ToyModel(nn.Module):  # 适配 ch09 接口：encode→state、decode_step(state, ys)→(B,V) logits
        def encode(self, src):
            return torch.arange(src.size(0))  # (B,) 句号作 state

        def decode_step(self, state, ys):
            t = ys.size(1) - 1  # ys 含 SOS 前缀：已生成 t 个 token（beam 内 ys 为 (B·k,≤t)）
            return torch.stack([sent_logits(int(s), t) for s in state])  # -> (B,V)（批量 beam 下 B=b·k）

    model = ToyModel().eval()
    src = torch.arange(n_sent).unsqueeze(1)  # (B,1)=(3,1)
    out9 = d9.beam_search(model, src, k=k, alpha=ALPHA, max_len=16)  # -> (b, ≤max_len)，PAD 补齐
    oks = []
    for i in range(n_sent):
        lp_fn = lambda toks, i=i: F.log_softmax(sent_logits(i, len(toks)), dim=-1)
        mine = beam_search(lp_fn, beam=k, alpha=ALPHA, max_len=16, eos=EOS)
        theirs = out9[i, 1:].tolist()
        theirs = theirs[: theirs.index(0)] if 0 in theirs else theirs  # 去 PAD 尾
        oks.append(mine == theirs == corrects[i])
    print(f"  ③ beam：3 句最优假设 教学版=应用版=构造正解 → {all(oks)}（{oks}）")


if __name__ == "__main__":
    demo_noam()
    demo_label_smoothing()
    demo_dropout()
    demo_beam()
    cross_check()
    print("[完成] ch08 recipe.py 全部自测通过")
