# llama215.py —— Book3 ch10 大项目整合件：LLaMA 式 207M 整机（四插槽 import 自第 2/3/4/6 章）
# 用途：本册单轨代码主线的收官形态——「改装史写在 import 语句里」：
#       归一化插槽   import 自 ch02/rmsnorm.py::RMSNorm(hidden_size, eps)
#       前馈插槽     import 自 ch03/swiglu.py::SwiGLUMLP(cfg)
#       位置插槽     import 自 ch04/rope.py::build_rope_cache(seq_len, head_dim, theta, device)
#                     与 apply_rope(q, k, cos, sin)（ch06/gqa.py 内部同样依赖本件）
#       注意力插槽   import 自 ch06/gqa.py::GroupedQueryAttention(cfg).forward(x, cos, sin)
#       四件均承 00-feasibility/llama_slots.py 正身的类名/签名/HF 键位（插槽 import 契约），
#       本文件只做「组装」：DecoderLayer / Backbone / 整机 + untied lm_head + 参数逐项账 assert。
# config 定版 cand1：d=1024 / L=12 / h=16 / h_kv=8 / d_ff=2816 / V=32000 untied /
#       RoPE theta=10000（无 wpe）/ rms_norm_eps=1e-5 / 无 bias —— 总参数 207,119,360。
# 所属章节：Book3 第 10 章（10.2 整机实现 / 10.3 冒烟与对拍）；DESIGN-215M.md 的架构正身。
# 运行方式（三档，全 CPU 可跑除注明外）：
#   自测（秒级）        ：source env.sh && python llama215.py
#                        —— 参数逐项账 assert 207,119,360 + 随机 token 初始 loss 自检（ln V + sigma^2/2）
#   HF 结构对拍（分钟级）：python llama215.py --parity
#                        —— 同 config 随机权重 strict 直搬 HF LlamaForCausalLM，CPU fp32 对拍 logits
#   TinyLlama 实权重对拍：python llama215.py --tinyllama
#                        —— 载入 TinyLlama_v1.1（Apache-2.0，HF 缓存实物）真权重到本类，对拍 + 贪心续写
# 产物：对拍 JSON 落 log/book3-ch10/{llama215_parity, llama215_tinyllama}_{out}.json（不入库）。
# 依赖链：ch02/ch03/ch04/ch06 四章教学重写件（各自与 llama_slots 正身逐位一致——对拍已在前章验收）
#       ＋ 00-feasibility/llama_slots.py（LLaMAConfig 鸭子类型正身）——五个目录见文件内 sys.path 段；
#       transformers 5.18.0 仅在对拍段 import。
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
BOOK3 = os.path.abspath(os.path.join(HERE, ".."))
for sub in ("ch02", "ch03", "ch04", "ch06", "00-feasibility"):      # 插槽 import 契约的物理落点
    sys.path.insert(0, os.path.join(BOOK3, sub))

from rmsnorm import RMSNorm                                # noqa: E402  归一化插槽（第 2 章）
from swiglu import SwiGLUMLP                               # noqa: E402  前馈插槽（第 3 章）
from rope import apply_rope, build_rope_cache              # noqa: E402  位置插槽（第 4 章）
from gqa import GroupedQueryAttention                       # noqa: E402  注意力插槽（第 6 章）
from llama_slots import LLaMAConfig                        # noqa: E402  config 鸭子类型正身（组件库）

REPO_ROOT = os.path.abspath(os.path.join(BOOK3, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book3-ch10")
SEED = 20261002                                            # 全书统一种子

# ---------------- 215M 定版 config（cand1；逐字段依据见 DESIGN-215M.md §3） ----------------
LLAMA_215M = dict(vocab_size=32000, hidden_size=1024, intermediate_size=2816,
                  num_hidden_layers=12, num_attention_heads=16, num_key_value_heads=8,
                  max_position_embeddings=8192, rope_theta=10000.0, rms_norm_eps=1e-5)


class Llama215DecoderLayer(nn.Module):
    """pre-RMSNorm 残差块——拓扑仍是 Book2 三插槽 Block，两个插槽里装的是本册新件。

        x = x + attn(input_layernorm(x))      RMSNorm(ch2) -> GQA(ch6, 内旋 RoPE(ch4))
        x = x + mlp(post_attention_layernorm(x))   RMSNorm(ch2) -> SwiGLU(ch3)
    输入/输出 (B,n,C)——层内四件套换装，残差河恒宽不变（图 0.1 的插座①②③④全在这 12 行里）。
    """

    def __init__(self, cfg: LLaMAConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)          # 插槽①（第 2 章）
        self.self_attn = GroupedQueryAttention(cfg)                                # 插槽④（第 6 章）
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)  # 插槽①（第 2 章）
        self.mlp = SwiGLUMLP(cfg)                                                  # 插槽②（第 3 章）

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)   # (B,n,C) 残差河
        x = x + self.mlp(self.post_attention_layernorm(x))          # (B,n,C)
        return x


class Llama215Backbone(nn.Module):
    """embed_tokens + L 层 DecoderLayer + 终态 RMSNorm。命名对齐 HF LlamaModel（model.*）。

    入口无位置查表（wpe 已在第 4 章拆除）：位置信息由每层注意力内的 apply_rope 现做，
    cos/sin 表在 forward 里按当前窗长现算——n 任意大，无查表上限。
    """

    def __init__(self, cfg: LLaMAConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)            # (V,C)
        self.layers = nn.ModuleList([Llama215DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)                       # 出口前最后一枚

    def forward(self, idx):
        B, n = idx.shape                                                     # (B,n)
        cfg = self.cfg
        x = self.embed_tokens(idx)                                           # (B,n) -> (B,n,C)
        cos, sin = build_rope_cache(n, cfg.head_dim, cfg.rope_theta, idx.device)  # (1,n,hd) x2（第 4 章）
        for layer in self.layers:                                            # L x (B,n,C)
            x = layer(x, cos, sin)
        return self.norm(x)                                                  # (B,n,C)


class Llama215(nn.Module):
    """LLaMA 式整机（207M 定版）：wte + L x [pre-RMSNorm -> GQA | pre-RMSNorm -> SwiGLU]
    + 终态 RMSNorm + untied lm_head。前向接口与 Book2 GPT2 一致：idx (B,n) -> logits (B,n,V)；
    给 targets 时返回逐 token 交叉熵（fp32 口径，对齐 HF）。

    state_dict 键与 HF LlamaForCausalLM 完全同名（model.embed_tokens.weight /
    model.layers.i.self_attn.q_proj.weight / ... / model.norm.weight / lm_head.weight），
    strict 直搬即结构对拍（--parity）；换 cfg 实例化即载 TinyLlama 等同型真权重（--tinyllama）。
    cfg 缺省 = 215M 定版；init_std 缺省 = config initializer_range 0.02（Book2 式 (10.3) 的纪律）。
    """

    def __init__(self, cfg: LLaMAConfig = None, init_std: float | None = None):
        super().__init__()
        cfg = cfg or LLaMAConfig(**LLAMA_215M)
        self.cfg = cfg
        self.model = Llama215Backbone(cfg)                                    # 命名对齐 HF（model.*）
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)  # untied（第 5 章附加一刀）
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
        """参数量。non_embedding=True 时扣 wte 与 lm_head（untied 各扣一份）。"""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.model.embed_tokens.weight.numel() + self.lm_head.weight.numel()
        return n

    def forward(self, idx, targets=None):
        B, n = idx.shape                                               # (B,n)
        x = self.model(idx)                                            # (B,n,C)（backbone 含 RoPE 现算）
        logits = self.lm_head(x)                                       # (B,n,C)·(C,V) -> (B,n,V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),   # (B*n,V) fp32
                                  targets.reshape(-1))                             # (B*n,)
        return logits, loss


def hand_account(cfg: LLaMAConfig):
    """参数逐项手算（无 bias、untied、无 wpe）。返回 (明细 dict, 总数)——assert 的对账另一侧。"""
    d, L, V, f = cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size, cfg.intermediate_size
    attn = d * d * 2 + d * cfg.head_dim * cfg.num_key_value_heads * 2      # q/o 全宽 + k/v 按 h_kv 缩（GQA）
    mlp = 3 * d * f                                                        # gate/up/down 三投影（SwiGLU）
    per_layer = attn + mlp + 2 * d                                         # + 两枚 RMSNorm（各只有 weight）
    total = per_layer * L + V * d * 2 + d                                  # untied：wte 与 lm_head 各一份 + 终态 norm
    detail = {"attn_per_layer": attn, "mlp_per_layer": mlp, "two_norms": 2 * d,
              "per_layer": per_layer, "layers": per_layer * L,
              "wte": V * d, "lm_head_untied": V * d, "final_norm": d, "total": total}
    return detail, total


# ---------------- 自测：参数逐项账 assert + 初始 loss 的数学自检 ----------------
def self_test():
    torch.manual_seed(SEED)
    cfg = LLaMAConfig(**LLAMA_215M)
    m = Llama215(cfg)
    detail, hand = hand_account(cfg)
    total = m.n_params()
    print("[config] cand1:", {k: LLAMA_215M[k] for k in LLAMA_215M})
    for k, v in detail.items():
        print(f"  {k:>18} = {v:,}")
    print(f"[参数账] 手算 = {hand:,}   实测 = {total:,}   non-emb = {m.n_params(True):,}")
    assert total == hand == 207_119_360, f"参数逐项对账失败：{total} != {hand} != 207,119,360"

    # 初始 loss 自检（随机 token，CPU fp32，next-token 目标）：untied 下 E[L0] = ln V + sigma^2/2，
    # sigma^2 = d*s^2（末态 RMSNorm 后 ||h||=sqrt(d)、每维 RMS=1，lm_head 行 N(0,s^2) → logit 方差
    # = s^2*||h||^2 = d*s^2 = 1024*0.02^2 = 0.4096）。绑定头的自泄漏路径（Book2 式 (10.3) 的
    # 「复读机」方向：E[z_target] > 0 偷走概率质量）在 untied 独立初始化下不存在——E[z_t] 实测归零。
    # 注：窗长取 256（训练窗 1024 的缩小版）——窗太短时 E[z_t] 的抽样噪声变大，方差账不变。
    g = torch.Generator().manual_seed(SEED)
    x = torch.randint(0, cfg.vocab_size, (2, 256), generator=g)         # (2,256)
    xp, yp = x[:, :-1], x[:, 1:]                                        # next-token 目标（训练口径）
    with torch.no_grad():
        logits, loss = m(xp, yp)                                        # (2,255,V)
        z_t = logits.gather(-1, yp.unsqueeze(-1)).squeeze(-1)           # (2,255) 目标位 logit
        var = logits.var().item()
    ln_v = math.log(cfg.vocab_size)
    sig2_pred = cfg.hidden_size * cfg.initializer_range ** 2
    print(f"[初始 loss] 实测 {loss.item():.4f} | ln V = {ln_v:.4f} | 预言 ln V + d*s^2/2 = "
          f"{ln_v + sig2_pred / 2:.4f}（sigma^2 = {sig2_pred:.4f}，实测 logit 方差 {var:.4f}）"
          f" | E[z_target] = {z_t.mean().item():+.4f}（泄漏项，untied 下应为 0）")
    assert abs(loss.item() - (ln_v + sig2_pred / 2)) < 0.05, "初始 loss 偏离 ln V + d*s^2/2 的预言"
    assert abs(z_t.mean().item()) < 0.05, "E[z_target] 显著非零——untied 不该有自泄漏"
    assert loss.item() - ln_v < 1.0, "初始 loss 距 ln V 超过 1 nat——初始化或前向有病"
    print("llama215 自测全部通过（参数逐位 207,119,360；初始 loss = ln V + d*s^2/2 千分位咬合）")


# ---------------- HF 结构对拍：同 config 随机权重 strict 直搬（探针 E 链路，207M 档） ----------------
def parity_hf(out_name="run1"):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(SEED)
    ours = Llama215()                                                   # 215M 定版
    c = LLAMA_215M
    hf_cfg = LlamaConfig(vocab_size=c["vocab_size"], hidden_size=c["hidden_size"],
                         intermediate_size=c["intermediate_size"],
                         num_hidden_layers=c["num_hidden_layers"],
                         num_attention_heads=c["num_attention_heads"],
                         num_key_value_heads=c["num_key_value_heads"],
                         max_position_embeddings=c["max_position_embeddings"],
                         rope_theta=c["rope_theta"], rms_norm_eps=c["rms_norm_eps"],
                         tie_word_embeddings=False, attention_bias=False, mlp_bias=False,
                         attention_dropout=0.0)
    hf = LlamaForCausalLM(hf_cfg).eval()
    hf.load_state_dict(ours.state_dict(), strict=True, assign=True)     # 键不一致直接抛错——本身就是验收
    print(f"[搬运] ours {ours.n_params():,} -> HF LlamaForCausalLM（strict=True 通过）")

    g = torch.Generator().manual_seed(SEED)
    x = torch.randint(0, c["vocab_size"], (2, 128), generator=g)        # (2,128)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits_ours, loss_ours = ours(x, x)                             # (2,128,V)
        logits_hf = hf(x).logits                                        # (2,128,V)
    delta = (logits_ours - logits_hf).abs().max().item()
    argmax_agree = (logits_ours.argmax(-1) == logits_hf.argmax(-1)).float().mean().item()
    top5_agree = (logits_ours.topk(5, -1).indices.sort(-1).values
                  == logits_hf.topk(5, -1).indices.sort(-1).values).float().mean().item()
    loss_hf = F.cross_entropy(logits_hf.reshape(-1, c["vocab_size"]).float(), x.reshape(-1)).item()

    # sanity：两个独立随机初始化的模型 logits 应显著不同（证明搬运真的发生了）
    torch.manual_seed(SEED + 1)
    other = Llama215()
    with torch.no_grad():
        sanity = (other(x)[0] - logits_hf).abs().max().item()

    report = {"seed": SEED, "config": LLAMA_215M, "transformers_version": __import__("transformers").__version__,
              "parity": {"ours_params": ours.n_params(),
                         "max_abs_diff_logit": delta, "argmax_agreement": argmax_agree,
                         "top5_agreement": top5_agree, "loss_abs_diff": abs(loss_ours.item() - loss_hf),
                         "sanity_unshared_max_abs_diff": sanity,
                         "wall_sec": round(time.perf_counter() - t0, 1)},
              "verdict": "PASS" if (delta < 1e-4 and argmax_agree == 1.0 and sanity > 0.1) else "FAIL"}
    print(f"[对拍] max|Δlogits| = {delta:.3e} | argmax 一致率 = {argmax_agree:.4f} | "
          f"top-5 一致率 = {top5_agree:.4f} | |Δloss| = {abs(loss_ours.item() - loss_hf):.3e} | "
          f"sanity(未搬运对照) = {sanity:.3f}")
    print(f"[对拍] 判定 = {report['verdict']}")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"llama215_parity_{out_name}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


# ---------------- TinyLlama_v1.1 实权重对拍（Apache-2.0；HF 缓存实物，条件性） ----------------
def parity_tinyllama(out_name="run1"):
    """真权重三步：① HF 加载缓存实物；② state_dict strict 搬进本类（同型不同档）；
    ③ 同输入对拍 logits + 本类贪心续写一段——「四插槽整机跑真权重」的存在性证明。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    hf = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama_v1.1", dtype=torch.float32).eval()
    print(f"[载入] TinyLlama_v1.1（HF 缓存实物，fp32，{time.perf_counter() - t0:.1f}s，"
          f"{sum(p.numel() for p in hf.parameters()):,} 参数）")
    hc = hf.config
    theta = getattr(hc, "rope_theta", None)
    if theta is None:                                   # transformers 5.x 把 rope 参数收进 rope_parameters 子对象
        theta = hc.rope_parameters.get("rope_theta", 10000.0)
    cfg = LLaMAConfig(vocab_size=hc.vocab_size, hidden_size=hc.hidden_size,
                      intermediate_size=hc.intermediate_size, num_hidden_layers=hc.num_hidden_layers,
                      num_attention_heads=hc.num_attention_heads,
                      num_key_value_heads=hc.num_key_value_heads,
                      max_position_embeddings=hc.max_position_embeddings,
                      rope_theta=theta, rms_norm_eps=hc.rms_norm_eps)
    ours = Llama215(cfg).eval()
    ours.load_state_dict(hf.state_dict(), strict=True)                  # 201 张量 strict 直搬
    print(f"[搬运] HF state_dict -> Llama215({cfg.hidden_size}/{cfg.num_hidden_layers}层/"
          f"h={cfg.num_attention_heads}/h_kv={cfg.num_key_value_heads}) strict=True 通过")

    tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids                    # (1,n)
    with torch.no_grad():
        lo = ours(ids)[0]                                               # (1,n,V)
        lh = hf(ids).logits                                             # (1,n,V)
    delta = (lo - lh).abs().max().item()
    argmax_agree = (lo.argmax(-1) == lh.argmax(-1)).float().mean().item()

    # 本类贪心续写（无 KV cache 全量重算，12 token——教学演示不计性能）
    gen = ids.clone()
    for _ in range(12):
        with torch.no_grad():
            nxt = ours(gen)[0][:, -1].argmax(-1, keepdim=True)          # (1,1)
        gen = torch.cat([gen, nxt], dim=1)
    cont = tok.decode(gen[0, ids.shape[1]:])

    report = {"model": "TinyLlama/TinyLlama_v1.1", "license": "Apache-2.0",
              "config": {"d": cfg.hidden_size, "L": cfg.num_hidden_layers, "h": cfg.num_attention_heads,
                         "h_kv": cfg.num_key_value_heads, "d_ff": cfg.intermediate_size,
                         "V": cfg.vocab_size, "theta": cfg.rope_theta, "eps": cfg.rms_norm_eps},
              "parity": {"max_abs_diff_logit": delta, "argmax_agreement": argmax_agree,
                         "prompt": prompt, "greedy_continuation_ours": cont,
                         "wall_sec": round(time.perf_counter() - t0, 1)},
              "verdict": "PASS" if (delta < 1e-4 and argmax_agree == 1.0) else "FAIL"}
    print(f"[对拍] max|Δlogits| = {delta:.3e} | argmax 一致率 = {argmax_agree:.4f}")
    print(f"[贪心] {prompt!r} -> {cont!r}（v1.1 为 3T/3.9T 中途 base 检查点，greedy 退化属正常；对拍只看数值）")
    print(f"[对拍] 判定 = {report['verdict']}")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"llama215_tinyllama_{out_name}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LLaMA 式 207M 整机：自测 / HF 结构对拍 / TinyLlama 实权重对拍")
    ap.add_argument("--parity", action="store_true", help="HF 结构对拍（CPU fp32，分钟级）")
    ap.add_argument("--tinyllama", action="store_true", help="TinyLlama_v1.1 实权重对拍（CPU fp32）")
    ap.add_argument("--out-name", default="run1", help="产物 JSON 后缀（防覆写）")
    args = ap.parse_args()
    if args.parity:
        parity_hf(args.out_name)
    if args.tinyllama:
        parity_tinyllama(args.out_name)
    if not (args.parity or args.tinyllama):
        self_test()
