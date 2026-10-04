# llama_slots.py —— Book3 插槽组件库：RMSNorm / SwiGLU / RoPE / GQA / LLaMA 整机（手写版）
# 用途：为 00-feasibility 各探针（模块档/长窗/对拍/外推）提供统一的 LLaMA 式组件实现；
#       参数命名完全对齐 HF transformers 5.18.0 的 LlamaForCausalLM（model.layers.i.self_attn.q_proj...），
#       因此 state_dict 可 load_state_dict(strict=True) 直搬——这是 05_probe_e_parity.py 对拍链路的基础。
# 所属章节：Book3 ch2（RMSNorm）/ ch3（SwiGLU）/ ch4（RoPE）/ ch6（GQA）；00-feasibility 全体探针共用。
# 运行方式：作为模块被探针 import；自测（秒级，CPU）：python llama_slots.py
#   —— 自测内容：215M 候选 config 参数逐项账 assert + mini config 前向 shape 检查。
# 数值口径：对齐 HF modeling_llama.py（eager 路径）——RMSNorm 方差在 fp32 计算；softmax 在 fp32；
#       RoPE 用 rotate_half 约定（与 RoPE 论文 2104.09864 的分块旋转在数学上等价、排列不同，
#       对拍锚定 HF 实现，论文记号差异在正文考据框处理）。
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- 215M 候选 config（ch10 大项目目标档；手算账见 __main__ 自测） ----------------
# d=1024 / L=12 / h=16 / h_kv=8 / d_ff=2816=11x256 / V=32000 untied / RoPE（无 wpe）
# 逐项账（无 bias，RMSNorm 只有 weight）：
#   注意力每层: q 1024*1024 + k 1024*512 + v 1024*512 + o 1024*1024        = 3,145,728
#   SwiGLU 每层: 3 * 1024 * 2816                                            = 8,650,752
#   每层小计 + 2*RMSNorm(1024)                                              = 11,798,528
#   12 层                                                                   = 141,582,336
#   wte 32000*1024 + lm_head 32000*1024（untied）+ 末层 RMSNorm              = 65,537,024
#   总计                                                                    = 207,119,360 ≈ 207.1M（落在 205-225M 目标带）
LLAMA_215M = dict(vocab_size=32000, hidden_size=1024, intermediate_size=2816,
                  num_hidden_layers=12, num_attention_heads=16, num_key_value_heads=8,
                  max_position_embeddings=8192, rope_theta=10000.0, rms_norm_eps=1e-5)


class LLaMAConfig:
    """手写 LLaMA 配置。字段名对齐 HF LlamaConfig 的驼峰风格，便于正文对照。"""

    def __init__(self, vocab_size=32000, hidden_size=1024, intermediate_size=2816,
                 num_hidden_layers=12, num_attention_heads=16, num_key_value_heads=None,
                 max_position_embeddings=8192, rope_theta=10000.0, rms_norm_eps=1e-5,
                 initializer_range=0.02):
        assert hidden_size % num_attention_heads == 0
        self.vocab_size = vocab_size                    # wte 行数（LLaMA1/2=32000，LLaMA3=128256）
        self.hidden_size = hidden_size                  # 残差流宽度 C
        self.intermediate_size = intermediate_size      # SwiGLU 门控维 d_ff（2/3*4C 取整到 256 倍数）
        self.num_hidden_layers = num_hidden_layers      # 层数 L
        self.num_attention_heads = num_attention_heads  # 查询头数 h（head_dim = C/h）
        self.num_key_value_heads = num_key_value_heads or num_attention_heads  # KV 头数 h_kv（GQA；=h 即 MHA，=1 即 MQA）
        assert num_attention_heads % self.num_key_value_heads == 0
        self.head_dim = hidden_size // num_attention_heads
        self.max_position_embeddings = max_position_embeddings  # 仅作训练窗设定，RoPE 本身不设硬上限（对照 wpe）
        self.rope_theta = rope_theta                    # RoPE 基底 theta（10000 为 LLaMA1/2；LLaMA3=500000）
        self.rms_norm_eps = rms_norm_eps    # 1e-5=大纲 215M 定版与 LLaMA2/3 出厂口径（HF LlamaConfig 默认是 1e-6——一律从 config 读，防 1e-6/1e-5 分叉）
        self.initializer_range = initializer_range


class RMSNorm(nn.Module):
    """L2 归一化 + 逐维缩放（无偏置、无均值中心化）。输入 (B,n,C) -> 输出 (B,n,C)。

    对齐 HF LlamaRMSNorm：方差在 fp32 计算（数值稳定），乘 weight 后转回原 dtype。
    eps 只加在方差上（rsqrt(mean(x^2)+eps)），与 LayerNorm 的分母结构对照见 ch2。
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))   # (C,)
        self.variance_epsilon = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()                                          # (B,n,C) fp32
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype)                       # (C,)*(B,n,C) -> (B,n,C)


class SwiGLU(nn.Module):
    """门控前馈：down( silu(gate(x)) * up(x) )。输入 (B,n,C) -> 输出 (B,n,C)。

    三投影无 bias；d_ff = intermediate_size（8/3C 规则的落地，见 ch3）。
    参数量 3*C*d_ff，对照 GPT-2 MLP 的 2*C*4C——「多一条门控分支、缩 2/3 维」是同一枚硬币的两面。
    """

    def __init__(self, cfg: LLaMAConfig):
        super().__init__()
        C, d_ff = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(C, d_ff, bias=False)        # (B,n,C) -> (B,n,d_ff)
        self.up_proj = nn.Linear(C, d_ff, bias=False)          # (B,n,C) -> (B,n,d_ff)
        self.down_proj = nn.Linear(d_ff, C, bias=False)        # (B,n,d_ff) -> (B,n,C)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))  # (B,n,C)


def build_rope_cache(seq_len, head_dim, theta, device):
    """RoPE 的 cos/sin 查表。返回 (cos, sin)，形状 (1, n, head_dim)，fp32。

    inv_freq_i = 1 / theta^(2i/head_dim)，i=0..head_dim/2-1   —— 低维转得快、高维转得慢
    freqs[t, i] = t * inv_freq_i；emb = cat(freqs, freqs)（rotate_half 约定需要整维表）
    注意：n 可以任意大——这正是 ch7 外推实验「不需要改代码就能喂长窗」的原因（对照 wpe 查表越界即崩）。
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))  # (hd/2,)
    t = torch.arange(seq_len, dtype=torch.float32, device=device)             # (n,)
    freqs = torch.outer(t, inv_freq)                                          # (n, hd/2)
    emb = torch.cat((freqs, freqs), dim=-1)                                   # (n, hd)
    return emb.cos()[None], emb.sin()[None]                                   # (1,n,hd) 各一


def rotate_half(x):
    """(…, hd) -> (…, hd)：后一半取负放前、前一半放后。旋转矩阵的分块排列实现（对齐 HF）。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """对 q/k 施加旋转（v 不转）。q (B,h,n,hd)、k (B,h_kv,n,hd)；cos/sin (1,n,hd) 广播。
    q' = q*cos + rotate_half(q)*sin —— 复数乘法 (x+iy)(cos+i·sin) 的实数化写法。"""
    cos, sin = cos.to(q.dtype), sin.to(q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)      # (B,h,n,hd)
    k_embed = (k * cos) + (rotate_half(k) * sin)      # (B,h_kv,n,hd)
    return q_embed, k_embed


def repeat_kv(x, n_rep):
    """GQA 的 KV 头扩展：x (B,h_kv,n,hd) -> (B,h_kv*n_rep,n,hd) = (B,h,n,hd)。
    相邻的 n_rep 个查询头共享同一组 KV（分组共享，非全局共享——MQA 才是 n_rep=h 的极端）。"""
    if n_rep == 1:
        return x
    B, h_kv, n, hd = x.shape
    x = x[:, :, None, :, :].expand(B, h_kv, n_rep, n, hd)
    return x.reshape(B, h_kv * n_rep, n, hd)


class GroupedQueryAttention(nn.Module):
    """GQA 注意力插槽：q/k/v/o 四投影（无 bias）+ RoPE + 因果掩码。输入 (B,n,C) -> 输出 (B,n,C)。

    k_proj/v_proj 只输出 h_kv*hd 维（KV 头少、每头维度不变）——KV cache 账本的省法在投影端，
    不在注意力计算端（repeat_kv 后注意力照常按 h 个头算，见 ch6）。
    """

    def __init__(self, cfg: LLaMAConfig):
        super().__init__()
        C, hd = cfg.hidden_size, cfg.head_dim
        self.n_head = cfg.num_attention_heads
        self.n_kv_head = cfg.num_key_value_heads
        self.n_rep = self.n_head // self.n_kv_head
        self.head_dim = hd
        self.scaling = hd ** -0.5
        self.q_proj = nn.Linear(C, self.n_head * hd, bias=False)    # (B,n,C) -> (B,n,h*hd)
        self.k_proj = nn.Linear(C, self.n_kv_head * hd, bias=False)  # (B,n,C) -> (B,n,h_kv*hd)
        self.v_proj = nn.Linear(C, self.n_kv_head * hd, bias=False)
        self.o_proj = nn.Linear(self.n_head * hd, C, bias=False)    # (B,n,h*hd) -> (B,n,C)
        self._mask_cache = {}                                       # n -> causal mask（惰性构建，越过 max_position 也能用）

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
        q, k = apply_rope(q, k, cos, sin)                                       # 只旋转 q/k
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)               # -> (B,h,n,hd)
        att = (q @ k.transpose(-2, -1)) * self.scaling                          # (B,h,n,hd)·(B,h,hd,n) -> (B,h,n,n)
        att = att.masked_fill(self._causal_mask(n, x.device)[:, :, :n, :n] == 0, float("-inf"))
        att = F.softmax(att.float(), dim=-1).to(q.dtype)                        # fp32 softmax（对齐 HF）
        y = att @ v                                                             # (B,h,n,n)·(B,h,n,hd) -> (B,h,n,hd)
        y = y.transpose(1, 2).contiguous().view(B, n, h * hd)                   # (B,n,h*hd)
        return self.o_proj(y)                                                   # (B,n,C)


class LLaMADecoderLayer(nn.Module):
    """pre-RMSNorm 残差块（拓扑同 Book2 三插槽 Block，两处插槽全换）。输入/输出 (B,n,C)。

        x = x + attn(input_layernorm(x))
        x = x + mlp(post_attention_layernorm(x))
    """

    def __init__(self, cfg: LLaMAConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)      # (B,n,C)
        x = x + self.mlp(self.post_attention_layernorm(x))             # (B,n,C)
        return x


class LLaMABackbone(nn.Module):
    """embed_tokens + L 层 decoder layer + norm。命名对齐 HF LlamaModel（model.*）。"""

    def __init__(self, cfg: LLaMAConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)      # (V,C)
        self.layers = nn.ModuleList([LLaMADecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, idx):
        B, n = idx.shape                                                       # (B,n)
        cfg = self.cfg
        x = self.embed_tokens(idx)                                             # (B,n,C)
        cos, sin = build_rope_cache(n, cfg.head_dim, cfg.rope_theta, idx.device)  # (1,n,hd) x2
        for layer in self.layers:                                              # L x (B,n,C)
            x = layer(x, cos, sin)
        return self.norm(x)                                                    # (B,n,C)


class LLaMA(nn.Module):
    """LLaMA 整机（手写版）：wte + L x [pre-RMSNorm -> GQA | pre-RMSNorm -> SwiGLU] + norm + untied lm_head。

    前向接口与 Book2 GPT2 一致：idx (B,n) -> logits (B,n,V)；给 targets 时返回逐 token 交叉熵。
    state_dict 键与 HF LlamaForCausalLM 完全同名（model.embed_tokens.weight / model.layers.i.../ lm_head.weight），
    可 strict 直搬——05_probe_e_parity.py 的验收管道即建立在此之上。
    与 GPT-2 的四点对照（ch10 整合清单）：LN->RMSNorm / GELU MLP->SwiGLU / wpe->RoPE / MHA+tied->GQA+untied。
    """

    def __init__(self, cfg: LLaMAConfig = None, init_std: float | None = None):
        super().__init__()
        cfg = cfg or LLaMAConfig(**LLAMA_215M)
        self.cfg = cfg
        self.model = LLaMABackbone(cfg)                       # 命名对齐 HF（state_dict 键 model.*）
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)  # untied：独立输出投影
        std = init_std or cfg.initializer_range
        self.apply(lambda m: self._init(m, std))

    @staticmethod
    def _init(module, std):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def n_params(self, non_embedding=False):
        """参数量。non_embedding=True 时扣 wte 与 lm_head（untied 各算一份）。"""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.model.embed_tokens.weight.numel() + self.lm_head.weight.numel()
        return n

    def forward(self, idx, targets=None):
        B, n = idx.shape                                                    # (B,n)
        x = self.model(idx)                                                 # (B,n,C)（backbone 含 RoPE 构建与全部层）
        logits = self.lm_head(x)                                            # (B,n,C)·(C,V) -> (B,n,V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),   # (B*n,V) fp32（对齐 HF loss 口径）
                                  targets.reshape(-1))                             # (B*n,)
        return logits, loss


# ---------------- 自测：215M 参数逐项账 assert + 前向 shape 检查（秒级） ----------------
if __name__ == "__main__":
    torch.manual_seed(20261002)  # 全书统一种子

    # 215M 候选 config 参数账（逐项手算，assert 锁死）
    cfg = LLaMAConfig(**LLAMA_215M)
    m = LLaMA(cfg)
    attn_per_layer = cfg.hidden_size ** 2 * 2 + cfg.hidden_size * cfg.head_dim * cfg.num_key_value_heads * 2
    mlp_per_layer = 3 * cfg.hidden_size * cfg.intermediate_size
    per_layer = attn_per_layer + mlp_per_layer + 2 * cfg.hidden_size
    hand = per_layer * cfg.num_hidden_layers + cfg.vocab_size * cfg.hidden_size * 2 + cfg.hidden_size
    total = m.n_params()
    print(f"[215M 候选] 手算 = {hand:,}   实测 = {total:,}   non-emb = {m.n_params(True):,}")
    assert total == hand == 207_119_360, f"参数逐项对账失败：{total} != {hand} != 207,119,360"
    assert 205e6 <= total <= 225e6, "未落在 205-225M 目标带"

    # mini config 前向 shape 检查（外推：n 超过 max_position_embeddings 也能前向——RoPE 无查表上限）
    mini = LLaMA(LLaMAConfig(vocab_size=512, hidden_size=256, intermediate_size=704,
                             num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=128))
    x = torch.randint(0, 512, (2, 300))                    # n=300 > max_position=128：GPT-2 会 assert 崩，这里照常
    logits, loss = mini(x, x)
    assert logits.shape == (2, 300, 512) and loss is not None and torch.isfinite(loss)
    print(f"[mini d=256/L=4] 前向 OK：logits {tuple(logits.shape)}，loss={loss.item():.4f}（理论初始 ~ln512+0.5*0.02^2*256/2≈{math.log(512):.3f}+）")
    print("llama_slots 自测全部通过")
