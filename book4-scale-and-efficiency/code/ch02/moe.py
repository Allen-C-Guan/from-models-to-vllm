# moe.py —— Book4 ch2 教学件：Mixtral 式稀疏 MoE 块（router + top-k + 融合 3D 专家）
# 用途：正文 2.7 节的代码正身——把 2.2-2.5 讲的机构（打分 / 硬选择 / top-k 内重归一化 /
#       gather-scatter 派发）写成一个可直接运行、可对拍的块；三方对拍（CPU fp32）：
#       ①教学件 vs 正身 moe_mla_slots.HFMixtralMoE（同权重前向，期望逐位一致 max|Δ|=0）；
#       ②MiniMixtralLM（Book3 组件 + 本件装 FFN 插槽）vs HF transformers 5.18.0
#         MixtralForCausalLM（小 config 随机权重 strict 直搬，探针 G 同口径复跑）；
#       ③自证三条机构性质：展平不变性（B=1 与 B=2 同 token 流输出全等）/ 路由可微
#       （梯度经 softmax 门控值流回 router 权重）/ 负载计数 Σ=N·k。
# 所属章节：Book4 第 2 章（MoE 基本机构：路由器、top-k 与容量）。
# 运行方式：cd 工作区根目录 && source env.sh && python code/ch02/moe.py --out-name run1
#   （CPU fp32，秒级；产物 log/book4-ch02/moe_parity_run1.json，不入库）
# 契约（plan 卷级大纲「插槽 import 契约 v1.1」）：类名 SparseMoE；HF Mixtral 键位
#   （mlp.gate.weight + mlp.experts.gate_up_proj/down_proj 融合 3D 裸 Parameter）；无共享专家
#   （共享专家是 ch5 DeepSeekMoE 的机构，不在本件）；forward 返回 hidden + 可选 router_logits。
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "00-feasibility")))
from moe_mla_slots import SEED, HFMixtralMoE, bootstrap_book3  # noqa: E402  正身（组件库）

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book4-ch02")


def _field(cfg, *names):
    """按别名族读 config 字段（n_expert|num_local_experts 等——HF attribute_map 同款别名）。"""
    for nm in names:
        v = getattr(cfg, nm, None)
        if v is not None:
            return v
    raise AttributeError(f"cfg 缺少字段（试过 {list(names)}）")


# ---------------- 机构三件：路由器 / 专家库 / MoE 块 ----------------
class MixtralTopKRouter(nn.Module):
    """教学版路由器：打分 → 全体 softmax(fp32) → top-k 硬选择 → top-k 内重归一化。

    键位：weight (E,d) 裸 Parameter —— state_dict 键 gate.weight（与 4.x 的 nn.Linear 同名巧合）。
    forward(h): h (N,d) 已展平的 token 流 → (logits (N,E), topk_w (N,k), topk_idx (N,k))。
    数值口径：softmax 强制 fp32（对齐 HF MixtralTopKRouter——路由是「选路」决策，数值要稳）。
    """

    def __init__(self, hidden_size, n_expert, top_k):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_expert, hidden_size))    # (E,d)
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.n_expert, self.top_k = n_expert, top_k

    def forward(self, h):
        logits = F.linear(h, self.weight)                        # (N,d)·(E,d)ᵀ -> (N,E) 逐 token 打分
        probs = logits.float().softmax(dim=-1)                   # (N,E) fp32 全体 softmax
        topk_w, topk_idx = torch.topk(probs, self.top_k, dim=-1)  # (N,k)+(N,k) 硬选择（落选者出局）
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)       # (N,k) top-k 内重归一化 Σg=1
        return logits, topk_w, topk_idx


class ExpertBank(nn.Module):
    """专家库：融合 3D 权重 + gather/scatter 派发（5.18.0 键位，无 .weight 后缀的裸 Parameter）。

    权重：gate_up_proj (E,2w,d)——gate 投影与前段 / up 投影在后，行拼接成一枚 3D 张量；
          down_proj (E,d,w)。
    forward(h, topk_w, topk_idx): (N,d),(N,k),(N,k) -> (out (N,d), counts (E,))。
    counts = 本批各专家实收 token 数——负载分布的第一手素材（ch3 直方图的数据源）。
    """

    def __init__(self, hidden_size, n_expert, expert_dim):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(n_expert, 2 * expert_dim, hidden_size))  # (E,2w,d)
        self.down_proj = nn.Parameter(torch.empty(n_expert, hidden_size, expert_dim))         # (E,d,w)
        for p in (self.gate_up_proj, self.down_proj):
            nn.init.normal_(p, mean=0.0, std=0.02)
        self.n_expert, self.expert_dim = n_expert, expert_dim

    def forward(self, h, topk_w, topk_idx):
        out = torch.zeros_like(h)                                # (N,d) 输出累加器
        counts = torch.zeros(self.n_expert, dtype=torch.long)    # (E,) 负载计数
        for e in range(self.n_expert):
            tok, slot = torch.nonzero(topk_idx == e, as_tuple=True)   # 选中 e 的 (token,槽位) 对
            if tok.numel() == 0:
                continue                                          # 本批无人问津的专家直接跳过
            xe = h[tok]                                           # gather：(N,d) -> (n_e,d)
            gate, up = F.linear(xe, self.gate_up_proj[e]).chunk(2, dim=-1)  # (n_e,2w) -> (n_e,w)×2
            y = F.linear(F.silu(gate) * up, self.down_proj[e])    # (n_e,w) -> (n_e,d) SwiGLU 专家
            out.index_add_(0, tok, y * topk_w[tok, slot].unsqueeze(1))      # scatter：加权写回 (N,d)
            counts[e] = tok.numel()
        return out, counts


class SparseMoE(nn.Module):
    """Mixtral 式稀疏 MoE 块（教学重写件，正文 2.7 的代码正身）。

    组装：gate = MixtralTopKRouter（路由器）+ experts = ExpertBank（E 个窄 SwiGLU 专家）。
    键位与 HF 5.18.0 MixtralSparseMoeBlock 严格同名：mlp.gate.weight / mlp.experts.gate_up_proj /
    mlp.experts.down_proj（5.x 融合 3D；4.x 的 experts.N.w1/w2/w3 ModuleList 键已废）。
    构造双形态：SparseMoE(cfg)（字段别名族 n_expert|num_local_experts、top_k|num_experts_per_tok、
    expert_dim|intermediate_size）或 SparseMoE(hidden_size=…, n_expert=…, top_k=…, expert_dim=…)。
    forward(x, out_router_logits=False)：x (B,n,d) -> hidden (B,n,d)；out_router_logits=True 时
    返回 (hidden, router_logits (B,n,E))——对齐 MixtralForCausalLM 的输出约定（aux loss 吃它）。
    """

    def __init__(self, cfg=None, hidden_size=None, n_expert=None, top_k=None, expert_dim=None, **_):
        super().__init__()
        if cfg is not None:
            hidden_size = _field(cfg, "hidden_size")
            n_expert = _field(cfg, "n_expert", "num_local_experts", "num_experts")
            top_k = _field(cfg, "top_k", "num_experts_per_tok")
            expert_dim = _field(cfg, "expert_dim", "intermediate_size", "moe_intermediate_size")
        assert all(v is not None for v in (hidden_size, n_expert, top_k, expert_dim)), \
            "缺几何参数：hidden_size/n_expert/top_k/expert_dim"
        self.gate = MixtralTopKRouter(hidden_size, n_expert, top_k)
        self.experts = ExpertBank(hidden_size, n_expert, expert_dim)
        self.n_expert, self.top_k, self.expert_dim = n_expert, top_k, expert_dim
        self.hidden_size = hidden_size

    def forward(self, x, out_router_logits=False):
        shape = x.shape                                           # (B,n,d)
        h = x.reshape(-1, shape[-1])                              # (B,n,d) -> (N,d)，N=B·n：先展平！
        logits, topk_w, topk_idx = self.gate(h)                   # (N,E) / (N,k) / (N,k)
        out, _counts = self.experts(h, topk_w, topk_idx)          # (N,d)（各专家加权合并）
        out = out.view(*shape)                                    # (N,d) -> (B,n,d) 还原残差河宽度
        if out_router_logits:
            return out, logits.view(*shape[:-1], self.n_expert)   # (B,n,E) HF 输出口径
        return out

    @torch.no_grad()
    def expert_load(self, x):
        """本批各专家实收 token 数 (E,)——只跑路由不算专家（负载探针，ch3 第一手素材）。"""
        h = x.reshape(-1, x.shape[-1])                            # (N,d)
        _, _, topk_idx = self.gate(h)
        return torch.bincount(topk_idx.reshape(-1), minlength=self.n_expert)  # (E,)


# ---------------- 整机对拍壳：Book3 组件 + 本件装 FFN 插槽 ----------------
class _MiniDecoderLayer(nn.Module):
    """LLaMA 式解码层：①RMSNorm→④GQA→①RMSNorm→②SparseMoE。

    注意力 / 归一化 / RoPE 全部 import 自 Book3 llama_slots（改装底座契约，不复制代码）——
    「第一刀只动 FFN 插槽」的代码形态。键名与 HF MixtralDecoderLayer 同名。"""

    def __init__(self, slots, cfg, n_expert, top_k, expert_dim):
        super().__init__()
        self.input_layernorm = slots.RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)      # 插槽①
        like = type("GQA_cfg", (), {})()                                             # 鸭喙 config（Book3 GQA 只读四件）
        like.hidden_size, like.head_dim = cfg.hidden_size, cfg.head_dim
        like.num_attention_heads = cfg.num_attention_heads
        like.num_key_value_heads = cfg.num_key_value_heads
        self.self_attn = slots.GroupedQueryAttention(like)                           # 插槽④（原样）
        self.post_attention_layernorm = slots.RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SparseMoE(hidden_size=cfg.hidden_size, n_expert=n_expert,         # 插槽②：动刀处
                             top_k=top_k, expert_dim=expert_dim)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)    # (B,n,d) 残差河
        x = x + self.mlp(self.post_attention_layernorm(x))           # (B,n,d)
        return x


class _MiniBackbone(nn.Module):
    """backbone 壳（键前缀 model.*，对齐 HF MixtralModel）：embed → L×层 → 终态 RMSNorm。"""

    def __init__(self, slots, cfg, n_expert, top_k, expert_dim):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)           # (B,n)->(B,n,d)
        self.layers = nn.ModuleList([_MiniDecoderLayer(slots, cfg, n_expert, top_k, expert_dim)
                                     for _ in range(cfg.num_hidden_layers)])
        self.norm = slots.RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, idx):
        B, n = idx.shape
        x = self.embed_tokens(idx)                                  # (B,n,d)
        cos, sin = bootstrap_book3().build_rope_cache(n, self.cfg.head_dim,
                                                      self.cfg.rope_theta, idx.device)
        for layer in self.layers:
            x = layer(x, cos, sin)                                  # (B,n,d)
        return self.norm(x)                                         # (B,n,d)


class MiniMixtralLM(nn.Module):
    """mini Mixtral 整机：model.backbone（embed → L×层 → RMSNorm）+ lm_head(untied)。

    组件全部来自 Book3 llama_slots，唯一新零件 = mlp 插槽里的 SparseMoE——
    state_dict 键与 MixtralForCausalLM 同构（model.layers.N.mlp.* 前缀），可 strict 直搬对拍。"""

    def __init__(self, cfg, n_expert, top_k, expert_dim):
        super().__init__()
        slots = bootstrap_book3()
        self.cfg, self.slots = cfg, slots
        self.model = _MiniBackbone(slots, cfg, n_expert, top_k, expert_dim)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)        # untied
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=cfg.initializer_range)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=cfg.initializer_range)

    def forward(self, idx, targets=None):
        x = self.model(idx)                                         # (B,n,d)
        logits = self.lm_head(x)                                    # (B,n,V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------- 自测：三方对拍 + 三条机构性质 ----------------
def main(out_name):
    device = torch.device("cpu")                                    # 数值对拍一律 CPU fp32（纪律）
    t0 = time.perf_counter()
    os.makedirs(OUT_DIR, exist_ok=True)
    slots = bootstrap_book3()
    report = {"seed": SEED, "device": "cpu fp32", "torch": torch.__version__}

    # ---- ① 教学件 vs 正身 HFMixtralMoE：同权重前向，期望逐位一致 ----
    d, E, k, w = 64, 4, 2, 128                                      # 小而全几何（parity_all 同款）
    torch.manual_seed(SEED)
    ref = HFMixtralMoE(d, E, k, w).eval()
    torch.manual_seed(SEED)
    teach = SparseMoE(hidden_size=d, n_expert=E, top_k=k, expert_dim=w).eval()
    missing, unexpected = teach.load_state_dict(ref.state_dict(), strict=False)
    assert not missing and not unexpected, f"键位不符 missing={missing} unexpected={unexpected}"
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(2, 32, d, generator=g)                          # (2,32,64)
    with torch.no_grad():
        yt, rl = teach(x, out_router_logits=True)                 # (2,32,64) + (2,32,4)
        yh = ref(x)                                               # (2,32,64) 正身对应件
    diff_ref = (yt - yh).abs().max().item()
    print(f"[① 教学件 vs 正身] strict 直搬通过 | max|Δhidden| = {diff_ref:.2e}"
          f"（{'逐位一致' if diff_ref == 0.0 else '数值一致'}）")
    print(f"    router_logits 形状 {tuple(rl.shape)} = (B,n,E)✓ | 键位 {[kk for kk in teach.state_dict()]}")
    report["teach_vs_ref"] = {"max_abs_diff": diff_ref, "keys": sorted(teach.state_dict()),
                              "verdict": "PASS" if diff_ref < 1e-6 else "FAIL"}

    # ---- ② 机构性质三条：展平不变性 / 路由可微 / 负载守恒 ----
    with torch.no_grad():                                           # (a) 展平不变性：路由决策粒度=token
        y_a = teach(x.reshape(1, 64, d))                            # (1,64,d) —— B=1
        y_b = teach(x.reshape(2, 32, d))                            # (2,32,d) —— B=2，token 流相同
        flat_diff = (y_a.reshape(2, 32, d) - y_b).abs().max().item()
    teach.zero_grad(set_to_none=True)
    out = teach(x).sum()                                            # (b) 可微性：梯度经 softmax 门控值反传
    out.backward()
    gw = teach.gate.weight.grad
    grad_ok = gw is not None and gw.abs().sum().item() > 0
    load = teach.expert_load(x)                                     # (c) 负载守恒：Σ counts = N·k
    print(f"[② 机构性质] 展平不变 max|Δ| = {flat_diff:.2e} | router 权重梯度非零 = {grad_ok}"
          f" | 负载 Σ={int(load.sum())}（N·k={x.numel() // d * k}）分布 {load.tolist()}")
    report["mechanism_checks"] = {"flatten_invariance_maxabs": flat_diff,
                                  "router_grad_nonzero": grad_ok,
                                  "load_counts": load.tolist(), "load_sum": int(load.sum()),
                                  "verdict": "PASS" if flat_diff == 0.0 and grad_ok
                                  and int(load.sum()) == x.numel() // d * k else "FAIL"}

    # ---- ③ MiniMixtralLM vs HF MixtralForCausalLM：小 config 随机权重 strict 直搬 ----
    from transformers import MixtralConfig, MixtralForCausalLM
    hcfg = slots.LLaMAConfig(vocab_size=512, hidden_size=256, intermediate_size=256,
                             num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=4,
                             max_position_embeddings=512, rope_theta=10000.0, rms_norm_eps=1e-5)
    torch.manual_seed(SEED)
    ours = MiniMixtralLM(hcfg, n_expert=4, top_k=2, expert_dim=256).eval()
    mcfg = MixtralConfig(vocab_size=512, hidden_size=256, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=4,
                         num_local_experts=4, num_experts_per_tok=2, max_position_embeddings=512,
                         rope_theta=10000.0, rms_norm_eps=1e-5, tie_word_embeddings=False,
                         attention_bias=False, mlp_bias=False, attention_dropout=0.0)
    hf = MixtralForCausalLM(mcfg).eval()
    hf.load_state_dict(ours.state_dict(), strict=True, assign=True)   # strict 直搬（键位全对齐）
    gi = torch.Generator().manual_seed(SEED)
    xi = torch.randint(0, 512, (2, 128), generator=gi)                # (2,128)
    with torch.no_grad():
        lo, _ = ours(xi)
        lh = hf(xi).logits                                            # (2,128,512)
    delta = (lo - lh).abs().max().item()
    argmax_agree = (lo.argmax(-1) == lh.argmax(-1)).float().mean().item()
    print(f"[③ vs HF 5.18.0] strict 直搬通过 {ours.n_params():,} 参 | max|Δlogits| = {delta:.2e}"
          f" | argmax 一致率 = {argmax_agree:.4f}")
    report["vs_hf_mixtral"] = {"params": ours.n_params(), "max_abs_diff_logit": delta,
                               "argmax_agreement": argmax_agree,
                               "verdict": "PASS" if delta < 1e-4 and argmax_agree == 1.0 else "FAIL"}

    report["wall_sec"] = round(time.perf_counter() - t0, 1)
    ok = all(report[s]["verdict"] == "PASS" for s in ("teach_vs_ref", "mechanism_checks", "vs_hf_mixtral"))
    report["verdict"] = "PASS" if ok else "FAIL"
    path = os.path.join(OUT_DIR, f"moe_parity_{out_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[判定] {report['verdict']} | 产物 {path} | 墙钟 {report['wall_sec']} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch2 教学件 moe.py 自测：三方对拍 + 机构性质（CPU fp32）")
    ap.add_argument("--out-name", default="run1", help="产物 JSON 后缀（防覆写）")
    args = ap.parse_args()
    main(args.out_name)
