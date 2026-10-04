# gqa.py —— Book3 ch6 教学重写件：GQA 注意力插槽的手写实现、两种写法与三重对拍
# 用途：①给出可 import 的 GroupedQueryAttention(cfg).forward(x, cos, sin)（插槽 import 契约，ch10 整机将
#           import 本件；与 llama_slots 正身同名同签名，CPU fp32 逐位一致）；
#       ②「分组视图 vs repeat 视图」两种等价写法（K/V 从不复制 vs 广播撑回 h 份——同一数学的两种记账）；
#       ③增量解码 demo：手写逐层 KV cache（每层一对 (B,h_kv,n,d_k)）逐 token 生成 vs 全量重算逐位对拍
#         （6.2 增量算法的机器验证：因果性 => 旧 K/V 不变 => 缓存合法）；
#       ④三重对拍：与 llama_slots.GroupedQueryAttention 逐位一致 / 手写 repeat_kv 与 HF 5.18.0
#         modeling_llama.repeat_kv 逐位一致（含 n_rep=1 直通分支）/ repeat_kv 的写穿透实测（expand+reshape
#         真复制——省的是常驻账本不是瞬时工位）。
# 所属章节：Book3 第 6 章 §6.2（增量算法验证）/§6.5（repeat 两种写法）/§6.7（改装第四例）。
# 运行方式：cd 工作区根目录 && source env.sh && python code/ch06/gqa.py
#   （CPU fp32，秒级，无命令行参数；产物全部打印 stdout，正文引用的自测数字来源，不写文件不落 log）
# 依赖链：ch04/rope.py 的 build_rope_cache / apply_rope（插槽契约：cos/sin 由第 4 章位置插槽供给；
#           ch04 件未就位时回退 00-feasibility/llama_slots 正身——两者同名同参逐位一致）+ transformers 5.18.0（仅对照）。
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "00-feasibility")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
try:
    from ch04.rope import apply_rope, build_rope_cache     # 契约主路径：第 4 章位置插槽重写件
except ImportError:                                        # ch04/rope.py 未就位时的回退（正身同名同参）
    from llama_slots import apply_rope, build_rope_cache   # noqa: F811
from llama_slots import LLaMAConfig                        # noqa: E402  config 鸭子类型正身
from llama_slots import GroupedQueryAttention as SlotGQA   # noqa: E402  组件库正身（对拍锚）

SEED = 20261002                                            # 全书统一种子


# ---------------- 教学重写件（与 llama_slots.GroupedQueryAttention 逐位一致；注释按讲解顺序展开） ----------------
def repeat_kv(x, n_rep):
    """repeat 视图：KV 头沿头维撑回 h 份。x (B,h_kv,n,hd) -> (B,h_kv*n_rep,n,hd) = (B,h,n,hd)。

    相邻的 n_rep 个查询头共享同一组 KV（组内连号共享：第 i 个 Q 头配第 i//n_rep 个 KV 头——HF 同款）。
    n_rep==1 直通返回（= MHA 的零开销快路径：GQA-h 退化成 MHA 的代码形态）。
    expand 只给视图、reshape 落盘真复制——写穿透实测见 __main__ 探针 4。
    """
    if n_rep == 1:
        return x
    B, h_kv, n, hd = x.shape
    x = x[:, :, None, :, :].expand(B, h_kv, n_rep, n, hd)  # (B,h_kv,1,n,hd) 视图，零复制
    return x.reshape(B, h_kv * n_rep, n, hd)                # (B,h,n,hd)：reshape 触发真复制


class GroupedQueryAttention(nn.Module):
    """GQA 注意力插槽：q/k/v/o 四投影（无 bias）+ RoPE + 因果掩码。输入 (B,n,C) -> 输出 (B,n,C)。

    计算图（对应正文形状流转表，g = h/h_kv 为组内查询头数）：
        x (B,n,C)
          -> q_proj            (B,n,h*hd)   -> view/transpose  (B,h,n,hd)     # 查询头：全宽
          -> k_proj / v_proj   (B,n,h_kv*hd)-> view/transpose  (B,h_kv,n,hd)  # KV 头：窄投影（账本省在这）
          -> apply_rope        仅 q/k，形状不变（v 不转，第 4 章）
          -> repeat_kv         (B,h_kv,n,hd) -> (B,h,n,hd)                    # repeat 视图（或分组视图等价）
          -> 打分 * scaling    (B,h,n,hd)·(B,h,hd,n) -> (B,h,n,n)
          -> 因果掩码 + softmax (B,h,n,n)
          -> 加权求和          (B,h,n,n)·(B,h,n,hd) -> (B,h,n,hd)
          -> 拼头 + o_proj     (B,n,h*hd) -> (B,n,C)
    数值口径（对拍五约定之三）：softmax 在 fp32 中做，结束转回原 dtype。
    """

    def __init__(self, cfg: LLaMAConfig):
        super().__init__()
        C, hd = cfg.hidden_size, cfg.head_dim
        self.n_head = cfg.num_attention_heads               # h：查询头数（全宽）
        self.n_kv_head = cfg.num_key_value_heads            # h_kv：KV 头数（=h 即 MHA，=1 即 MQA）
        self.n_rep = self.n_head // self.n_kv_head          # g = h/h_kv：组内查询头数
        self.head_dim = hd
        self.scaling = hd ** -0.5
        self.q_proj = nn.Linear(C, self.n_head * hd, bias=False)     # (B,n,C) -> (B,n,h*hd)
        self.k_proj = nn.Linear(C, self.n_kv_head * hd, bias=False)  # (B,n,C) -> (B,n,h_kv*hd)  窄
        self.v_proj = nn.Linear(C, self.n_kv_head * hd, bias=False)  # (B,n,C) -> (B,n,h_kv*hd)  窄
        self.o_proj = nn.Linear(self.n_head * hd, C, bias=False)     # (B,n,h*hd) -> (B,n,C)
        self._mask_cache = {}                               # n -> causal mask（惰性构建）

    def _causal_mask(self, n, device):
        if n not in self._mask_cache or self._mask_cache[n].device != device:
            self._mask_cache[n] = torch.tril(torch.ones(n, n, device=device)).view(1, 1, n, n)
        return self._mask_cache[n]

    def forward(self, x, cos, sin):
        B, n, C = x.shape                                                       # (B,n,C)
        hd, h, h_kv = self.head_dim, self.n_head, self.n_kv_head
        q = self.q_proj(x).view(B, n, h, hd).transpose(1, 2)                    # (B,h,n,hd)
        k = self.k_proj(x).view(B, n, h_kv, hd).transpose(1, 2)                 # (B,h_kv,n,hd)
        v = self.v_proj(x).view(B, n, h_kv, hd).transpose(1, 2)                 # (B,h_kv,n,hd)
        q, k = apply_rope(q, k, cos, sin)                                       # 只旋 q/k，形状不变
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)               # -> (B,h,n,hd)
        att = (q @ k.transpose(-2, -1)) * self.scaling                          # (B,h,n,hd)·(B,h,hd,n)->(B,h,n,n)
        att = att.masked_fill(self._causal_mask(n, x.device)[:, :, :n, :n] == 0, float("-inf"))
        att = F.softmax(att.float(), dim=-1).to(q.dtype)                        # fp32 softmax（对拍约定）
        y = att @ v                                                             # (B,h,n,n)·(B,h,n,hd)->(B,h,n,hd)
        y = y.transpose(1, 2).contiguous().view(B, n, h * hd)                   # (B,n,h*hd)
        return self.o_proj(y)                                                   # (B,n,C)


# ---------------- 分组视图：同一数学的另一种记账（K/V 从不复制） ----------------
def grouped_view_attention(q, k, v, n_rep, scaling, mask):
    """分组视图打分+加权：不 materialize h 份 KV。

    q (B,h,n,hd) -> (B,h_kv,g,n,hd)（连号分组）；k/v (B,h_kv,n,hd) -> (B,h_kv,1,n,hd) 广播——
    广播不复制，缓存里始终只有 h_kv 份（对照 repeat_kv 的写穿透）。输出 (B,h,n,hd)，与 repeat 路径逐位一致。
    """
    B, h, n, hd = q.shape
    h_kv = h // n_rep
    qg = q.view(B, h_kv, n_rep, n, hd)                      # (B,h_kv,g,n,hd)
    kg = k.view(B, h_kv, 1, n, hd)                          # (B,h_kv,1,n,hd)
    vg = v.view(B, h_kv, 1, n, hd)
    att = (qg @ kg.transpose(-2, -1)) * scaling             # 广播 -> (B,h_kv,g,n,n)
    att = att.masked_fill(mask[:, :, :n, :n] == 0, float("-inf"))
    att = F.softmax(att.float(), dim=-1).to(q.dtype)
    y = att @ vg                                            # (B,h_kv,g,n,hd)
    return y.reshape(B, h, n, hd)                           # (B,h,n,hd)


# ---------------- 增量解码 demo：手写 KV cache vs 全量重算（6.2 的机器验证） ----------------
def incremental_vs_full(attn: GroupedQueryAttention, x, cos, sin):
    """逐 token 增量生成 vs 整段全量前向，逐位对拍。x (B,n,C)，cos/sin (1,n,hd)。

    单层 cache = 一对 (B,h_kv,t,hd)。第 t 步只做三件事：新 token 的三次投影（各 1 个位置）、
    位置 t 的 RoPE 行、q_t 对全部 t+1 个键的注意力——因果性使旧 K/V 永不改变，拼接即可；
    增量步的键恰好就是可见前缀，连因果掩码都不需要（禁区在「没进 cache」里，不在表上）。
    返回 (out_incremental (B,n,C), max|Δ| vs 全量)。
    """
    B, n, C = x.shape
    hd, h, h_kv = attn.head_dim, attn.n_head, attn.n_kv_head
    out_full = attn(x, cos, sin)                            # (B,n,C) 全量重算路径（Book2 口径）
    k_cache = x.new_zeros(B, h_kv, 0, hd)                   # (B,h_kv,0,hd) 空缓存起步
    v_cache = x.new_zeros(B, h_kv, 0, hd)
    outs = []
    for t in range(n):
        xt = x[:, t:t + 1]                                  # (B,1,C) 只取新 token
        q = attn.q_proj(xt).view(B, 1, h, hd).transpose(1, 2)      # (B,h,1,hd)
        kt = attn.k_proj(xt).view(B, 1, h_kv, hd).transpose(1, 2)  # (B,h_kv,1,hd)
        vt = attn.v_proj(xt).view(B, 1, h_kv, hd).transpose(1, 2)  # (B,h_kv,1,hd)
        q, kt = apply_rope(q, kt, cos[:, t:t + 1], sin[:, t:t + 1])  # 只取位置 t 的行 (1,1,hd)
        k_cache = torch.cat([k_cache, kt], dim=2)           # (B,h_kv,t+1,hd) 旧账不动、只追加
        v_cache = torch.cat([v_cache, vt], dim=2)
        k_rep = repeat_kv(k_cache, attn.n_rep)              # (B,h,t+1,hd)
        v_rep = repeat_kv(v_cache, attn.n_rep)
        att = (q @ k_rep.transpose(-2, -1)) * attn.scaling  # (B,h,1,t+1)——无需掩码：键=可见前缀
        att = F.softmax(att.float(), dim=-1).to(q.dtype)
        y = (att @ v_rep).transpose(1, 2).reshape(B, 1, h * hd)    # (B,1,h*hd)
        outs.append(attn.o_proj(y))                         # (B,1,C)
    out_inc = torch.cat(outs, dim=1)                        # (B,n,C)
    return out_inc, (out_full - out_inc).abs().max().item()


# ---------------- 自测：三重对拍 + 写穿透 + 退化档 + 投影宽度账（CPU fp32，秒级） ----------------
if __name__ == "__main__":
    torch.manual_seed(SEED)
    B, n = 2, 64                                            # 小形状，肉眼可核

    # ---- 探针 1：与 llama_slots 正身逐位一致（h_kv 三档：MHA / GQA / MQA）----
    print("=" * 72)
    print("探针 1：教学重写件 vs llama_slots 正身（CPU fp32，同 config 同权重同输入）")
    print("=" * 72)
    for h, h_kv, tag in [(8, 8, "MHA（h_kv=h）"), (8, 4, "GQA-4（h_kv=4）"), (8, 1, "MQA（h_kv=1）")]:
        cfg = LLaMAConfig(vocab_size=512, hidden_size=256, intermediate_size=704,
                          num_hidden_layers=2, num_attention_heads=h, num_key_value_heads=h_kv,
                          max_position_embeddings=128)
        mine, ref = GroupedQueryAttention(cfg), SlotGQA(cfg)
        mine.load_state_dict(ref.state_dict(), strict=True)  # HF 键位同名，strict 直搬
        x = torch.randn(B, n, cfg.hidden_size)
        cos, sin = build_rope_cache(n, cfg.head_dim, cfg.rope_theta, x.device)
        d = (mine(x, cos, sin) - ref(x, cos, sin)).abs().max().item()
        assert d == 0.0, f"对拍失败 {tag}: max|Δ|={d}"
        print(f"[1] {tag:14s} h={h} h_kv={h_kv}  max|Δ| = {d:.1e}  （逐位一致）")

    # ---- 探针 2：手写 repeat_kv vs HF transformers 5.18.0 repeat_kv 逐位一致 ----
    print("-" * 72)
    from transformers.models.llama.modeling_llama import repeat_kv as hf_repeat_kv
    print("探针 2：手写 repeat_kv vs HF modeling_llama.repeat_kv（torch.equal）")
    for h_kv, n_rep in [(8, 1), (4, 2), (2, 4), (1, 8)]:
        x = torch.randn(B, h_kv, n, 32)
        same = torch.equal(repeat_kv(x, n_rep), hf_repeat_kv(x, n_rep))
        assert same, f"HF 对拍失败 h_kv={h_kv} n_rep={n_rep}"
        print(f"[2] h_kv={h_kv} n_rep={n_rep}：repeat 后 {tuple(repeat_kv(x, n_rep).shape)}"
              f" == HF，torch.equal = True（含 n_rep=1 直通分支）")

    # ---- 探针 3：分组视图 vs repeat 视图（同一数学的两种记账）----
    print("-" * 72)
    print("探针 3：分组视图（不复制 KV）vs repeat 视图（撑回 h 份），输出逐位一致")
    cfg = LLaMAConfig(vocab_size=512, hidden_size=256, intermediate_size=704,
                      num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
                      max_position_embeddings=128)
    attn = GroupedQueryAttention(cfg)
    x = torch.randn(B, n, cfg.hidden_size)
    cos, sin = build_rope_cache(n, cfg.head_dim, cfg.rope_theta, x.device)
    q = attn.q_proj(x).view(B, n, 8, 32).transpose(1, 2)         # (B,8,n,32)
    k = attn.k_proj(x).view(B, n, 2, 32).transpose(1, 2)         # (B,2,n,32)
    v = attn.v_proj(x).view(B, n, 2, 32).transpose(1, 2)
    q, k = apply_rope(q, k, cos, sin)
    mask = attn._causal_mask(n, x.device)
    y_rep = (F.softmax(((q @ repeat_kv(k, 4).transpose(-2, -1)) * attn.scaling)
                       .masked_fill(mask == 0, float("-inf")).float(), dim=-1).to(q.dtype)
             @ repeat_kv(v, 4))                                  # repeat 路径 (B,8,n,32)
    y_grp = grouped_view_attention(q, k, v, 4, attn.scaling, mask)  # 分组路径 (B,8,n,32)
    d = (y_rep - y_grp).abs().max().item()
    assert d == 0.0, f"两种写法不一致: {d}"
    print(f"[3] h=8 h_kv=2（g=4）：max|Δ| = {d:.1e}（广播路径全程未复制 K/V）")

    # ---- 探针 4：写穿透实测——expand+reshape 会真复制 ----
    print("-" * 72)
    print("探针 4：repeat_kv 的写穿透（expand 视图经 reshape 落盘 = 真复制）")
    x = torch.randn(B, 2, n, 32)
    y = repeat_kv(x, 4)                                          # (B,8,n,32)
    y[0, 0, 0, 0] = 12345.0                                      # 往「视图」里写
    untouched = x[0, 0, 0, 0].item() != 12345.0
    copied = y.data_ptr() != x.data_ptr()
    assert untouched and copied
    print(f"[4] 改 y 不动 x = {untouched}（缓存里的 h_kv 份安全）；两存储独立 = {copied}"
          f"——瞬时工位是 h 份、常驻账本仍 h_kv 份")

    # ---- 探针 5：增量解码 vs 全量重算（6.2 增量算法的机器验证）----
    print("-" * 72)
    print("探针 5：手写 KV cache 逐 token 生成 vs 整段全量重算（Book2 的笨办法）")
    cos2, sin2 = build_rope_cache(n, cfg.head_dim, cfg.rope_theta, x.device)
    d = incremental_vs_full(attn, torch.randn(B, n, cfg.hidden_size), cos2, sin2)[1]
    n_pos_full, n_pos_inc = n * (n + 1) // 2, n                  # 全量重算累计 n(n+1)/2 个位置
    print(f"[5] n={n}：全量重算累计前向位置 {n_pos_full} 个 vs 增量 {n_pos_inc} 个"
          f"（{n_pos_full / n_pos_inc:.1f} 倍）；输出 max|Δ| = {d:.2e}（因果性 => 旧 K/V 不变）")

    # ---- 探针 6：投影宽度即账本（207M cand1 档的注意力参数逐项手算）----
    print("-" * 72)
    print("探针 6：207M cand1（d=1024,h=16,h_kv=8,hd=64）投影宽度与参数账")
    cfg = LLaMAConfig(vocab_size=32000, hidden_size=1024, intermediate_size=2816,
                      num_hidden_layers=12, num_attention_heads=16, num_key_value_heads=8)
    attn = GroupedQueryAttention(cfg)
    w = {n_: tuple(p.shape) for n_, p in attn.named_parameters()}
    hand = (1024 * 1024) * 2 + (1024 * 512) * 2                  # q/o 全宽 + k/v 窄投影
    got = sum(p.numel() for p in attn.parameters())
    assert got == hand == 3_145_728
    print(f"[6] q {w['q_proj.weight']} / k {w['k_proj.weight']} / v {w['v_proj.weight']}"
          f" / o {w['o_proj.weight']}"
          f" -> 每层 {got:,} 参数（手算=实测）；KV 投影宽度砍半，Q/O 原封——账本省在 k/v 两格")
    print("gqa.py 自测全部通过")
