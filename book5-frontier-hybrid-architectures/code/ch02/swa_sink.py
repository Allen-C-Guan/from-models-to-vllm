# swa_sink.py —— Book5 ch2 教学件：SlidingWindowSinkAttention（banded mask 滑窗 + 每头可学 sink 标量）
# 用途（大纲 ch2 代码交付物正身）：
#   ① banded_causal_mask：滑窗因果掩码 (n,n) bool（2.2 节机制——例 2.1 的代码对照）；
#   ② SlidingWindowSinkAttention(cfg)：滑窗注意力 + 每头 1 个可学 sink 标量，gpt-oss 语义
#      （sink 作为第 n+1 个 logit 拼进 softmax 分母、无 value、概率质量在加权前丢弃——
#       transformers 5.18.0 modeling_gpt_oss.py L251-259：cat([attn_weights, sinks]) → softmax →
#       probs[..., :-1]，2.3 节公式 (2.2) 的实现）；
#   ③ 三重对拍：a) 自研 banded mask vs HF masking_utils.create_sliding_window_causal_mask 逐元素
#      （探针 08 已验路径正式收编）；b) sink 的「cat-softmax-drop」写法 vs 闭式分母写法数值等价；
#      c) 整层输出 vs HF GptOssAttention eager（同 config 同权重 strict 直搬）；
#   ④ 分类型 KV 账本第一算（2.5 节表 2.2）：5 机型 config 整数精确复算 + 与调研底单（papers/01 E.2）
#      逐位互证——「滑窗层封顶、全局层无界」的数字正身。
# 所属章节：Book5 第 2 章（2.2/2.3/2.5；needle_probe.py 内嵌同款插槽 SwaSinkAttn——两件行为一致）
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch02/swa_sink.py" [--out-name run1]
# 产物：log/book5-ch02/swa_sink_{out-name}.json（对拍数字与账本表——正文引用源，CPU fp32 秒级）
import argparse
import json
import os
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book5-ch02")
SEED = 20261002


def banded_causal_mask(n, window, device="cpu"):
    """banded causal mask：(n,n) bool，True=可见。第 i 个 query 只看 j<=i 且 i-j<window 的 key。

    输入 n:int, window:int → 输出 (n,n) bool——例 2.1（w=3, n=4）的手画矩阵即此函数的 4x4 特例。
    与 HF masking_utils 同式：sliding_window_overlay 的 kv_idx > q_idx - window AND causal 的 j <= i。
    """
    i = torch.arange(n, device=device)[:, None]                                  # (n,1)
    j = torch.arange(n, device=device)[None, :]                                  # (1,n)
    m = j <= i
    if window is not None:
        m = m & (i - j < window)
    return m


@dataclass
class SWACfg:
    """SlidingWindowSinkAttention 的配置（cfg 参数正身——大纲插槽契约）。"""
    d: int = 256            # 残差宽（hidden_size）
    n_head: int = 8         # 查询头数 h
    n_kv_head: int = 4      # KV 头数 h_kv（GQA；h/h_kv=每组头数）
    d_k: int = 64           # 头维
    window: int = 16        # 滑窗宽 W：每 token 只回看最近 W 个
    use_sink: bool = True   # 每头 1 个可学 sink 标量（gpt-oss 语义）
    use_rope: bool = False  # 本章实验关 rope（隔离 mask 变量）；True 时 forward 需给 cos/sin


class SlidingWindowSinkAttention(nn.Module):
    """滑窗注意力 + 每头可学 sink（forward: (B,n,d) → (B,n,d)）。

    机制（逐运算形状，正文 2.2/2.3 与图 2.1 的代码对照）：
      x (B,n,d) → q/k/v 投影 → (B,h,n,d_k)/(B,h_kv,n,d_k)×2 → GQA 展开 →
      打分 S=q·kᵀ/√d_k (B,h,n,n) → banded mask 填 -inf → cat sink b_h (B,h,n,n+1) →
      softmax → 丢 sink 位 (B,h,n,n) → 加权 V (B,h,n,d_k) → o_proj → (B,n,d)。
    sink 数学（式 (2.2)）：α_j = e^{s_j} / (e^{b_h} + Σ_j e^{s_j})——分母多一项 e^{b_h}，
    真实 token 的权重整体乘 (1-α_sink)、窗内相对比例不变；b_h≡0 时即 softmax₁（off-by-one）。
    """

    def __init__(self, cfg: SWACfg):
        super().__init__()
        self.cfg = cfg
        self.h, self.h_kv, self.d_k = cfg.n_head, cfg.n_kv_head, cfg.d_k
        self.scale = cfg.d_k ** -0.5
        self.q_proj = nn.Linear(cfg.d, cfg.n_head * cfg.d_k, bias=False)         # (B,n,d)→(B,n,h·d_k)
        self.k_proj = nn.Linear(cfg.d, cfg.n_kv_head * cfg.d_k, bias=False)
        self.v_proj = nn.Linear(cfg.d, cfg.n_kv_head * cfg.d_k, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * cfg.d_k, cfg.d, bias=False)
        if cfg.use_sink:
            self.sinks = nn.Parameter(torch.zeros(cfg.n_head))                   # 每头 1 个标量，零初始化

    def forward(self, x, cos=None, sin=None):
        """x (B,n,d) → (B,n,d)；cos/sin (1,1,n,d_k) 可选（use_rope=True 时施加）。"""
        cfg, B, n = self.cfg, *x.shape[:2]
        q = self.q_proj(x).view(B, n, self.h, self.d_k).transpose(1, 2)          # (B,h,n,d_k)
        k = self.k_proj(x).view(B, n, self.h_kv, self.d_k).transpose(1, 2)       # (B,h_kv,n,d_k)
        v = self.v_proj(x).view(B, n, self.h_kv, self.d_k).transpose(1, 2)
        if cfg.use_rope and cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.h // self.h_kv
        k = k.repeat_interleave(rep, dim=1)                                      # (B,h,n,d_k) GQA 展开
        v = v.repeat_interleave(rep, dim=1)
        att = (q @ k.transpose(-2, -1)) * self.scale                             # (B,h,n,n) 打分矩阵
        att = att.masked_fill(~banded_causal_mask(n, cfg.window, x.device),
                              float("-inf"))                                     # 带外 -inf（静态几何）
        if cfg.use_sink:
            sink = self.sinks.view(1, self.h, 1, 1).expand(B, self.h, n, 1)      # (B,h,n,1) 虚拟 sink logit
            att = torch.cat([att, sink], dim=-1)                                 # (B,h,n,n+1) sink 进分母
            att = att - att.max(dim=-1, keepdim=True).values                     # 数值稳定（HF 同款，非原实现）
            p = F.softmax(att.float(), dim=-1)[..., :n].to(x.dtype)              # 丢 sink 概率质量 (B,h,n,n)
        else:
            p = F.softmax(att.float(), dim=-1).to(x.dtype)
        y = (p @ v).transpose(1, 2).contiguous().view(B, n, self.h * self.d_k)   # (B,n,h·d_k)
        return self.o_proj(y)                                                    # (B,n,d)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    """x (B,h,n,d_k) · cos/sin (1,1,n,d_k) → (B,h,n,d_k)。"""
    return x * cos + rotate_half(x) * sin


# ---------------- ① mask 对拍：自研 banded mask vs HF masking_utils（逐元素） ----------------
def parity_mask():
    from transformers.models.gemma3.configuration_gemma3 import Gemma3Config
    from transformers.masking_utils import create_sliding_window_causal_mask

    out = {}
    for n, w in [(96, 32), (64, 16), (50, 7)]:                                   # 含例 2.1 几何与奇数窗
        g3 = Gemma3Config(vocab_size=256, hidden_size=64, num_hidden_layers=2,
                          sliding_window=w, _attn_implementation="eager")
        emb = torch.zeros(1, n, 64)                                              # (B,q_len,hidden) 只取形状
        hf = create_sliding_window_causal_mask(g3, emb, None, None,
                                               allow_is_causal_skip=False)       # 强制物化（滑窗≠纯因果）
        block = hf[0, 0]                                                         # (n,n) additive：可见=0
        mine = banded_causal_mask(n, w)
        dense = torch.zeros(n, n).masked_fill(~mine, float("-inf"))
        out[f"n{n}_w{w}"] = {"elementwise_equal": bool(torch.equal(dense == 0, block == 0)),
                             "hf_shape": list(block.shape)}
    return out


# ---------------- ② sink 写法等价：cat-softmax-drop vs 闭式分母（式 (2.2) 两边对拍） ----------------
def parity_sink_closed_form():
    torch.manual_seed(SEED)
    B, h, n = 2, 4, 12
    att = torch.randn(B, h, n, n) * 3.0
    att = att.masked_fill(~banded_causal_mask(n, 5), float("-inf"))
    sinks = torch.randn(h)                                                       # 任意可学值
    # 写法 A（实现语义，gpt-oss L251-259）：cat → softmax → 丢 sink 位
    cat = torch.cat([att, sinks.view(1, h, 1, 1).expand(B, h, n, 1)], dim=-1)    # (B,h,n,n+1)
    pa = F.softmax(cat.float(), dim=-1)[..., :n]
    # 写法 B（闭式，式 (2.2)）：α_j = e^{s_j}/(e^{b_h}+Σ e^{s_j})
    num = torch.exp(att.float())                                                 # (B,h,n,n)，带外 e^{-inf}=0
    den = num.sum(dim=-1, keepdim=True) + torch.exp(sinks).view(1, h, 1, 1)      # (B,h,n,1)
    pb = num / den
    sink_mass = 1.0 - pa.sum(dim=-1)                                             # α_sink（被丢掉的那份）
    return {"max_abs_diff": float((pa - pb).abs().max()),
            "sink_mass_range": [float(sink_mass.min()), float(sink_mass.max())]}


# ---------------- ③ 整层对拍：SlidingWindowSinkAttention vs HF GptOssAttention（同权重 strict） ----------------
def parity_gptoss_layer():
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention

    torch.manual_seed(SEED)
    n, w, d, h, h_kv, d_k = 24, 8, 64, 4, 2, 16
    cfg_hf = GptOssConfig(hidden_size=d, num_hidden_layers=2, num_attention_heads=h,
                          num_key_value_heads=h_kv, head_dim=d_k, sliding_window=w,
                          attention_bias=False, _attn_implementation="eager")
    hf = GptOssAttention(cfg_hf, layer_idx=0)                                    # layer 0 = sliding_attention
    mine = SlidingWindowSinkAttention(SWACfg(d=d, n_head=h, n_kv_head=h_kv, d_k=d_k, window=w))
    missing, unexpected = mine.load_state_dict(hf.state_dict(), strict=True), None
    x = torch.randn(2, n, d)                                                     # (B,n,d)
    cos = torch.ones(1, n, d_k // 2)                                             # 恒等 rope（打平位置变量：
    sin = torch.zeros(1, n, d_k // 2)                                            #   gpt-oss 系 cos/sin 宽=d_k/2——chunk 半旋转）
    mask4 = torch.zeros(1, 1, n, n).masked_fill(~banded_causal_mask(n, w), torch.finfo(torch.float32).min)
    y_hf, _ = hf(x, (cos, sin), mask4)                                           # (B,n,d)——eager 正身
    y_mine = mine(x)                                                             # (B,n,d)——use_rope=False ≡ 恒等
    return {"strict_load": "ok", "max_abs_diff": float((y_hf - y_mine).abs().max().detach()),
            "note": "同 config 同权重（q/k/v/o_proj+sinks 全同名）；GptOssConfig attention_bias=False 对齐"}


# ---------------- ④ 分类型 KV 账本第一算（2.5 节表 2.2；config 整数精确） ----------------
def kv_account():
    """KV(n) = 2·d_k·h_kv · (L_g·n + L_s·min(n,W))（元素数；bf16 字节=×2/1e9=GB 十进制）。
    全局层无界（随 n 线性）、滑窗层封顶 min(n,W)——config 依据与期望值=调研底单 papers/01 E.2（互证）。"""
    M = [  # (机型@上下文, L, L_g, per_tok=2·h_kv·d_k, n, W, 期望合计, 期望全满)
        ("Gemma2-9B@8192",        42, 21, 2 * 8 * 256,   8192,  4096, 1_056_964_608, 1_409_286_144),
        ("Gemma3-27B@131072",     62, 10, 2 * 16 * 128,  131072, 1024, 5_586_812_928, 33_285_996_544),
        ("Gemma3-27B@1M(假想外推)", 62, 10, 2 * 16 * 128, 1_000_000, 1024, 41_178_103_808, 253_952_000_000),
        ("gpt-oss-20b@131072",    24, 12, 2 * 8 * 64,    131072, 128, 1_612_185_600, 3_221_225_472),
        ("gpt-oss-120b@131072",   36, 18, 2 * 8 * 64,    131072, 128, 2_418_278_400, 4_831_838_208),
        ("Gemma3-1B@32768",       26, 4,  2 * 1 * 256,   32768,  512, 72_876_032, 436_207_616),
    ]
    rows = []
    for name, L, Lg, per_tok, n, W, exp_mix, exp_full in M:
        Ls = L - Lg
        g_elems = Lg * per_tok * n                                               # 全局层：无界，随 n 线性
        s_elems = Ls * per_tok * min(n, W)                                       # 滑窗层：封顶 min(n,W)
        mix, full = g_elems + s_elems, L * per_tok * n
        assert (mix, full) == (exp_mix, exp_full), (name, mix, full)             # 与调研底单逐位互证
        rows.append({"model": name, "L": L, "L_global": Lg, "per_tok": per_tok,
                     "global_elems": g_elems, "global_GB": round(g_elems * 2 / 1e9, 2),
                     "sliding_elems": s_elems, "sliding_MB": round(s_elems * 2 / 1e6, 1),
                     "mixed_total": mix, "full_total": full,
                     "saving_pct": round((1 - mix / full) * 100, 1),
                     "sliding_share_pct": round(s_elems / mix * 100, 2),
                     "asym_slope_pct": round(Lg / L * 100, 1)})                   # 渐近斜率=L_g/L（n→∞）
    return rows


def main():
    ap = argparse.ArgumentParser(description="ch2 教学件：滑窗+sink 组件与三重对拍+分类型账本")
    ap.add_argument("--out-name", default="run1", help="产物后缀（防覆写）")
    args = ap.parse_args()
    torch.manual_seed(SEED)

    res = {"seed": SEED, "date": "2026-10-05", "device": "cpu(fp32)",
           "mask_parity_vs_hf": parity_mask(),
           "sink_impl_vs_closed_form": parity_sink_closed_form(),
           "layer_parity_vs_gptoss_eager": parity_gptoss_layer(),
           "kv_account_typed": kv_account()}
    # 例 2.1 自检：w=3, n=4 的 banded mask 打印（正文手画矩阵的同源实物）
    m = banded_causal_mask(4, 3).int()
    res["toy_mask_w3_n4"] = ["".join(str(int(v)) for v in row) for row in m]

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"swa_sink_{args.out_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f"[① mask 对拍] {res['mask_parity_vs_hf']}")
    print(f"[② sink 闭式] max|Δ|={res['sink_impl_vs_closed_form']['max_abs_diff']:.2e} | "
          f"sink 质量范围 {res['sink_impl_vs_closed_form']['sink_mass_range']}")
    print(f"[③ 整层对拍] {res['layer_parity_vs_gptoss_eager']}")
    print(f"[④ 分类型账本]（{len(res['kv_account_typed'])} 机型，断言全过=与调研底单逐位一致）")
    for r in res["kv_account_typed"]:
        print(f"    {r['model']:<26} 混合 {r['mixed_total']:>13,} vs 全满 {r['full_total']:>13,}"
              f" | 省 {r['saving_pct']:>5}% | 滑窗层占 {r['sliding_share_pct']:>6}% | 斜率 {r['asym_slope_pct']}%")
    print(f"[例2.1 自检] w=3,n=4 mask = {res['toy_mask_w3_n4']}")
    print(f"[产物] {path}")


if __name__ == "__main__":
    main()
