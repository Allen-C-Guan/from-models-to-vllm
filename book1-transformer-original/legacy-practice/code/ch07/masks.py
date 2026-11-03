# -*- coding: utf-8 -*-
# 用途：Book1 第 7 章「mask 全景」全部实验——因果/填充/组合三类 mask 的构造与语义、
#       bool/float 两种形态与 MHA 内部「合并即相加」的实测验证、masked_fill(-inf) 静默
#       变 1.0 的陷阱、全屏蔽注意力行在 CPU 与 MPS 双设备的行为实测（销项 D8）；
#       附图 fig-7-1「mask 三态热图」。
# 所属章节：《Transformer 原典》第 7 章（整机组装与 mask 全景；原论文 v7 §3.2.3/Fig 2）
# 运行：source env.sh && python "code/ch07/masks.py"（CPU 秒级；MPS 段需 Apple Silicon，无 MPS 自动跳过）
# 注：torch 2.13 已移除 F.generate_square_subsequent_mask（本文件自写 causal_mask 替代，
#     并与仅存的 nn.Transformer.generate_square_subsequent_mask 静态方法对拍）。
# 形状口径（与正文表 7.1 一致）：因果掩码 (L,L)、填充掩码 (B,L)、组合掩码 (B,1,L,L)（bool 形态）。
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# 本书印刷图规格（_写作规范.md 第 4 节）：蓝主色、灰系坐标文字、hairline 网格
BLUE, DARK, INK, NOTE, GRID, AX_GREY = "#2a78d6", "#0d366b", "#0b0b0b", "#52514e", "#e1e0d9", "#898781"
FIG_DIR = Path(__file__).resolve().parents[3] / "drafts" / "Book1-Transformer原典" / "figures"


# ---------- 1. 因果掩码（自写：torch 2.13 已无 F.generate_square_subsequent_mask） ----------

def causal_mask(sz: int, device=None) -> torch.Tensor:
    """因果掩码（float 形态）：上三角（不含对角线）为 -inf，其余 0。sz -> (sz,sz)，即 (L,L)。
    §3.2.3 "masking out (setting to -∞) ... input of the softmax"——位置 i 只许看 j<=i。"""
    return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)


def causal_mask_bool(sz: int, device=None) -> torch.Tensor:
    """因果掩码（bool 形态，MHA 语义：True=屏蔽）。sz -> (sz,sz)。与 float 形态数值等价（实验 3 验证）。"""
    return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)


def padding_mask(lengths, sz: int, device=None) -> torch.Tensor:
    """填充掩码（bool，True=该 key 位置是 padding，不许任何人看）。lengths (B,) 各句实长 -> (B,sz)，即 (B,L)。右填充口径。"""
    ls = torch.as_tensor(lengths, device=device)
    return torch.arange(sz, device=device).unsqueeze(0) >= ls.unsqueeze(1)


# ---------- 2. 实验一：mask 构造与 helper 存废核对 ----------

def exp_constructors():
    L = 5
    print("== 1) 因果/填充掩码构造（L=5）==")
    print(f"  自写 causal_mask(5) float：\n{causal_mask(L)}")
    ok = torch.allclose(causal_mask(L), nn.Transformer.generate_square_subsequent_mask(L))
    print(f"  与 nn.Transformer.generate_square_subsequent_mask(5) allclose = {ok}")
    import torch.nn.functional as Fn
    print(f"  hasattr(F, 'generate_square_subsequent_mask') = {hasattr(Fn, 'generate_square_subsequent_mask')}"
          f"（torch 2.13 已移除，旧教程代码会 AttributeError）")
    kpm = padding_mask([3, 5], L)                              # (B,L)=(2,5) bool：样本 0 末 2 位是 padding
    print(f"  右填充 padding_mask(lengths=[3,5], L=5)（True=padding）：\n{kpm.to(torch.long)}")


# ---------- 3. 实验二：masked_fill(-inf) 陷阱——bool 容器装不下 -inf ----------

def exp_trap():
    print("== 2) masked_fill 陷阱：bool 张量直接 fill(-inf) 静默变 True，相加时成 +1.0 偏置 ==")
    kpm = padding_mask([3, 5], 5)                            # (B,L)=(2,5) bool，True=padding
    wrong = torch.zeros_like(kpm).masked_fill(kpm, float("-inf"))   # 陷阱写法：zeros_like 继承 bool dtype，(2,5) 仍 bool
    right = torch.zeros(kpm.shape, dtype=torch.float32).masked_fill(kpm, float("-inf"))  # (2,5) float：-inf 存得住
    causal = causal_mask(5)                                  # (L,L)=(5,5) float
    combined_wrong = causal + wrong.unsqueeze(1)             # (5,5)+(2,1,5) -> (2,5,5)：bool(-inf 已丢失) 参与加法
    combined_right = causal + right.unsqueeze(1)             # (2,5,5)：float 合并偏置（B,L,L）
    print(f"  陷阱写法产物 dtype={wrong.dtype}，masked 位置的值 = {wrong[kpm][0].item()}"
          f"（-inf 被静默替换为 True）")
    r, c = 4, 3                                              # 行 4（因果允许看 j<=4）× 样本 0 的 padding 列 3
    print(f"  样本 0 查询位 4 看 key 位 3（因果放行、padding 应屏蔽）：")
    print(f"    正确合并（先转 float）score 偏置 = {combined_right[0, r, c].item():.0f}（-inf，屏蔽生效）")
    print(f"    陷阱合并（bool 丢 -inf）  score 偏置 = {combined_wrong[0, r, c].item():+.1f}"
          f"（本应 -inf 的位置变成 +1.0 的『加分项』——掩码失效且不报错）")
    return combined_right


# ---------- 4. 实验三：组合等价性——合并传入 vs 分开传给 MHA（内部合并=加法的实证） ----------

def exp_combine(d_model=64, h=8, B=2, L=5):
    print("== 3) 组合掩码：合并传入 vs 分开传 attn_mask + key_padding_mask ==")
    torch.manual_seed(7)
    mha = nn.MultiheadAttention(d_model, h, batch_first=True).eval()   # batch_first 必须显式传
    x = torch.randn(B, L, d_model)                            # (B,L,d_model)，Q=K=V 同源
    kpm = padding_mask([3, 5], L)                             # (B,L) bool：样本 0 末 2 位是 padding
    with torch.no_grad():
        # 写法 A：手工合并成 float (B,1,L,L)（先转 float 再相加），再展开到 (B·h, L, L) 作 attn_mask
        # （MHA 的 3D attn_mask 形状严格要求 (N·num_heads, L, S)，传 4D 直接 RuntimeError）
        pad_f = torch.zeros(B, 1, L, L, dtype=x.dtype).masked_fill(kpm[:, None, None, :], float("-inf"))
        combined = (causal_mask(L).unsqueeze(0) + pad_f).expand(B, h, L, L).reshape(B * h, L, L)
        out_a, _ = mha(x, x, x, attn_mask=combined, need_weights=True)
        # 写法 B：分开传 float 因果 + bool padding（官方推荐的省事口径）
        out_b, _ = mha(x, x, x, attn_mask=causal_mask(L), key_padding_mask=kpm, need_weights=True)
        # 写法 C：bool 因果 + bool padding（MHA 语义 True=屏蔽）
        out_c, _ = mha(x, x, x, attn_mask=causal_mask_bool(L), key_padding_mask=kpm, need_weights=True)
    print(f"  A(手工 float 合并) vs B(分开 float+bool)：max|Δ| = {(out_a - out_b).abs().max():.2e}"
          f"  allclose(atol=1e-5) = {torch.allclose(out_a, out_b, atol=1e-5)}")
    print(f"  A vs C(全 bool 分开传)          ：max|Δ| = {(out_a - out_c).abs().max():.2e}"
          f"  allclose(atol=1e-5) = {torch.allclose(out_a, out_c, atol=1e-5)}")
    print("  → MHA 内部把两种 mask 统一转 float 后相加；bool 因果∪填充 的手工 OR 合并与之数值一致")
    # SDPA 反语义对照：bool True=参与（与 MHA 相反），迁移须取反
    q = torch.randn(B, h, L, d_model // h)                   # SDPA 直接吃 4D：(B,h,L,dk)
    k, v = torch.randn_like(q), torch.randn_like(q)
    with torch.no_grad():
        out_f = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask(L))
        out_b2 = F.scaled_dot_product_attention(q, k, v, attn_mask=~causal_mask_bool(L))
    print(f"  SDPA 侧：float(-inf) vs ~bool（True=参与，与 MHA 反语义）allclose = "
          f"{torch.allclose(out_f, out_b2, atol=1e-5)}")


# ---------- 5. 实验四：全屏蔽行行为（CPU/MPS 双设备实测，销项 D8） ----------

def _all_masked_once(dev):
    """在指定设备上跑五条路径，返回 {路径名: 样本0 第一个 query 输出的前 3 维}。"""
    torch.manual_seed(7)
    E, H, L, N = 64, 8, 4, 2
    dk = E // H
    q = torch.randn(N, L, E, device=dev)
    kpm = torch.zeros(N, L, dtype=torch.bool, device=dev)
    kpm[0] = True                                             # 样本 0：全部 key 被 padding 屏蔽
    res = {}
    # 路径 1：数学真相——softmax 对全 -inf 行（0/0 未定义）
    row = torch.full((1, L), float("-inf"), device=dev)
    res["softmax(全 -inf 行)"] = torch.softmax(row, dim=-1)[0, :3]
    # 路径 2：手写 matmul+softmax（本书正文实现口径）
    gen = torch.Generator(device="cpu").manual_seed(7)
    wq, wk, wv = (torch.randn(E, E, generator=gen).to(dev) for _ in range(3))
    fm = torch.zeros(N, 1, 1, L, device=dev).masked_fill(kpm[:, None, None, :], float("-inf"))
    fm = fm.expand(N, H, L, L)                                # (N,H,L,L)：每头同一条屏蔽行
    split = lambda t: t.reshape(N, L, H, dk).transpose(1, 2)  # (N,L,E)->(N,H,L,dk)
    Q, K, V = split(q @ wq), split(q @ wk), split(q @ wv)     # 各 (N,H,L,dk)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(dk) + fm     # (N,H,L,dk)x(N,H,dk,L) -> (N,H,L,L)
    res["手写 matmul+softmax"] = (torch.softmax(scores, dim=-1) @ V)[0, 0, 0, :3]
    # 路径 3/4：nn.MultiheadAttention 两条内部路径（掩码分开传，内部转 float 相加合并）
    mha = nn.MultiheadAttention(E, H, batch_first=True).eval().to(dev)
    with torch.no_grad():
        out3, _ = mha(q, q, q, key_padding_mask=kpm, need_weights=True)   # bmm+softmax 慢路径
        res["MHA need_weights=True(bmm)"] = out3[0, 0, :3]
        out4, _ = mha(q, q, q, key_padding_mask=kpm, need_weights=False)  # 内部走 SDPA
        res["MHA need_weights=False(内部SDPA)"] = out4[0, 0, :3]
    # 路径 5：直接 F.scaled_dot_product_attention + 4D float 掩码（可触发 CPU 融合核）
    q4 = torch.randn(N, H, L, dk, device=dev)
    k4, v4 = torch.randn_like(q4), torch.randn_like(q4)
    with torch.no_grad():
        res["SDPA(4D float 掩码)"] = F.scaled_dot_product_attention(
            q4, k4, v4, attn_mask=fm.to(q4.dtype))[0, 0, 0, :3]
    return res


def exp_all_masked():
    print("== 4) 全屏蔽行行为：CPU 与 MPS 双设备实测（样本 0 全部 key 被 padding 屏蔽）==")
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    tables = {d: _all_masked_once(d) for d in devices}
    for name in tables["cpu"]:
        cells = []
        for d in devices:
            v = tables[d][name]
            tag = "NaN" if torch.isnan(v).any().item() else ("全 0" if (v == 0).all().item() else "非零")
            cells.append(f"{d}={tag}")
        print(f"  {name:<26s} {'  '.join(cells)}")
    # SDPA 的 3D 掩码语义陷阱：MHA 收 (N·h,L,S)，SDPA 却按 (N,h,L,S) 的子维解释，直接传会报错
    q4 = torch.randn(2, 8, 4, 8)
    k4, v4 = torch.randn_like(q4), torch.randn_like(q4)
    fm3 = torch.full((16, 4, 4), 0.0)
    try:
        with torch.no_grad():
            F.scaled_dot_product_attention(q4, k4, v4, attn_mask=fm3)
        print("  SDPA 3D (N·h,L,S) 掩码：可运行（与 MHA 3D 语义相同）")
    except RuntimeError as e:
        print(f"  SDPA 3D (N·h,L,S) 掩码：RuntimeError（{str(e)[:42]}…）——SDPA 的 3D 掩码按"
              f"(N,h,L,S) 子维解释，与 MHA 的 (N·num_heads,L,S) 不同，跨 API 搬掩码必失配")
    print("  → 数学上 0/0 未定义；实测：bmm 路径双设备皆 NaN；MHA 内部 SDPA 在 CPU 走数学后端得 NaN、")
    print("    在 MPS 得 0 向量；直接 SDPA(4D 掩码)双设备皆 0（CPU 融合核显式处理全屏蔽行）。")
    print("    同一份数学、五条实现路径、三种输出——这就是「实现定义行为」；CUDA/融合注意力内核（后世）")
    print("    fp16 行为本机未测（防御法：保证每个 query 至少一个可见位置，如右填充+因果天然满足）")
    # 加测：输入布局也会换路径——同一模型/数据/掩码，仅 batch_first 不同，CPU 输出一个 NaN 一个 0
    q_b = torch.randn(2, 4, 64)
    q_l = q_b.transpose(0, 1).contiguous()                     # 同一批数据，序列维在前
    kpm2 = torch.zeros(2, 4, dtype=torch.bool); kpm2[0] = True
    tag = {}
    for bf, q in ((True, q_b), (False, q_l)):
        m = nn.MultiheadAttention(64, 8, batch_first=bf).eval()
        with torch.no_grad():
            o, _ = m(q, q, q, key_padding_mask=kpm2, need_weights=False)
        tag[bf] = "NaN" if torch.isnan(o[0, 0]).any().item() else "0"
    print(f"  加测（仅 CPU）：同一数据与掩码，batch_first=True 走原生 fused 路径得 {tag[True]}、"
          f"batch_first=False 走 functional+SDPA 得 {tag[False]}——连输入布局都能换实现路径")


# ---------- 6. 图 7.1：mask 三态热图 ----------

def fig_7_1(L=6, lengths=(4, 6)):
    causal = causal_mask_bool(L)                               # (L,L) True=屏蔽
    kpm = padding_mask(list(lengths), L)                       # (N,L) True=padding
    combined = causal | kpm[0][:, None]                        # 样本 0 的组合（因果∪填充，行=查询）
    panels = [("causal (float, (T,T))", causal),
              ("key padding (bool, (B,S))", kpm),
              ("combined = causal ∪ padding (sample 0, (B,T,T))", combined)]
    fig, axes = plt.subplots(1, 3, figsize=(8, 4), dpi=300,
                             gridspec_kw={"width_ratios": [1, 0.75, 1]})
    for ax, (title, m) in zip(axes, panels):
        ax.imshow(m.to(torch.uint8), cmap=matplotlib.colors.ListedColormap(["#f4f3ef", DARK]),
                  vmin=0, vmax=1, aspect="equal", interpolation="nearest")
        ax.set_title(title, color=INK, fontsize=8)
        ax.set_xlabel("key position j", color=AX_GREY, fontsize=7)
        ax.tick_params(colors=AX_GREY, labelsize=7)
        for s in ax.spines.values():
            s.set_color(GRID)
        rows, cols = m.shape
        for i in range(rows):
            for j in range(cols):
                blocked = bool(m[i, j])
                ax.text(j, i, "−∞" if blocked else "0", ha="center", va="center",
                        fontsize=7, color="#f4f3ef" if blocked else NOTE)
        ax.set_xticks(range(cols), [f"{j}" for j in range(cols)])
        ax.set_yticks(range(rows), [f"{i}" for i in range(rows)])
    axes[0].set_ylabel("query position i", color=AX_GREY, fontsize=7)
    axes[1].set_ylabel("sample / position", color=AX_GREY, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig-7-1-mask-three-states.png", facecolor="white")
    plt.close(fig)
    print(f"== 图 7.1 已保存：{FIG_DIR / 'fig-7-1-mask-three-states.png'} ==")


if __name__ == "__main__":
    exp_constructors()   # 1) 构造与 helper 存废
    exp_trap()           # 2) masked_fill 陷阱
    exp_combine()        # 3) 组合等价性（合并 vs 分开；bool/float；SDPA 反语义）
    exp_all_masked()     # 4) 全屏蔽行：CPU/MPS 双设备
    fig_7_1()            # 图 7.1
