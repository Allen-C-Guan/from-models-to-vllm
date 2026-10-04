# deepseek_moe.py —— Book4 ch5 教学件：DeepSeekMoE 式 MoE 块（细粒度路由专家 + 共享专家 + noaux 可选）
# 用途：正文 5.2-5.6 的代码正身——把 DeepSeekMoE 的三件机构写成一个可直接运行、可对拍的块：
#       ①细粒度路由专家（复用 ch02 的融合 3D 专家件 ExpertBank——SwiGLU 同构，E 个窄专家）；
#       ②共享专家（每 token 无条件激活，宽度 n_shared·w 的一个标准 SwiGLU，键位 shared_experts.*）；
#       ③noaux 免辅助损失路由（e_score_correction_bias Buffer：bias 只进 top-k 选择、不进门值；
#         步后按 b_i ← b_i + u·sign(e_i) 更新——2408.15664 Alg 1 / DSV3 §2.1.2 正身）。
#       打分代际可切换：scoring="softmax"（DSV2 Eq.22，greedy/不归一×rsf）| "sigmoid"（DSV3 Eq.15，
#       选中归一 norm_topk_prob×rsf）；组限制选择 n_group/topk_group（device-limited routing 的 config 形态）。
# 三方对拍（CPU fp32，种子 20261002）：
#   ① 教学件 vs 正身 moe_mla_slots.HFDeepseekMoE（同权重前向，期望逐位一致 max|Δ|=0）；
#   ② vs HF transformers 5.18.0 DeepseekV2Moe 块级对拍（小 config 随机权重 strict 直搬——探针 G 路径）；
#   ③ noaux 三段对拍 vs HF DeepseekV3TopkRouter（打分/选择/权重三段逐段比对；n_group=1 隔离组选择
#      + n_group=2 档过组选择路径）+ 机制自检三条（bias 只影响选择不进门值 / 负反馈方向 / 参数账双口径）。
# 所属章节：Book4 第 5 章（DeepSeekMoE：细粒度、共享专家与免损失的均衡）。
# 运行方式：cd 工作区根目录 && source env.sh && python code/ch05/deepseek_moe.py --out-name run1
#   （CPU fp32，秒级；产物 log/book4-ch05/deepseek_moe_parity_run1.json，不入库）
# 契约（plan 卷级大纲「插槽 import 契约 v1.1」）：类名 DeepSeekMoE；DSV2 键位（mlp.gate.weight +
#   mlp.experts.gate_up_proj/down_proj 融合 3D + mlp.shared_experts.{gate,up,down}_proj.weight）；
#   noaux 时多一枚 Buffer 键 mlp.gate.e_score_correction_bias（对齐 HF DeepseekV3TopkRouter 落地）；
#   forward 返回 hidden + 可选 router_logits——ch9 两刀整机 import 本件作 FFN 插槽（DSV2 键位为对拍正身）。
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
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "ch02")))
from moe import ExpertBank                     # noqa: E402  ch02 专家件复用（SwiGLU 同构，融合 3D 键位）
from moe_mla_slots import SEED, ExpertSwiGLU, HFDeepseekMoE   # noqa: E402  正身（组件库）

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book4-ch05")


def _field(cfg, *names):
    """按别名族读 config 字段（n_expert|n_routed_experts 等——HF attribute_map 同款别名）。"""
    for nm in names:
        v = getattr(cfg, nm, None)
        if v is not None:
            return v
    raise AttributeError(f"cfg 缺少字段（试过 {list(names)}）")


class DeepseekRouter(nn.Module):
    """DeepSeek 式路由器：打分 → 选择 → 权重 三段（DSV2 与 DSV3 两档打分代际）。

    打分段：fp32 线性打分 logits=(N,E)（对齐 HF——路由是选路决策，数值要稳，ch3 3.4 节）；
      scoring="softmax" → scores=softmax(logits)（DSV2 Eq.22）；"sigmoid" → scores=sigmoid(logits)（DSV3 Eq.15）。
    选择段：可选 noaux 偏置（choice = scores + e_score_correction_bias）与可选组限制
      （n_group>1：先选 topk_group 个组再组内 top-k——device-limited routing 的 config 形态，
      正文 5.4；组代表在 scores+bias 上取——对齐 HF DeepseekV3TopkRouter）。整段只决定「选谁」——
      bias 与组限制都不进门值。
    权重段：w = scores.gather(idx)（原始打分，不带 bias）；softmax 档不归一直接 ×routed_scaling_factor
      （DSV2 口径）；sigmoid 档 norm_topk_prob=True 时选中集内归一再 ×routed_scaling_factor（DSV3 口径）。
    键位：weight (E,d) 裸 Parameter（gate.weight）；noaux 时多一枚 Buffer e_score_correction_bias (E,)。
    forward(h): h (N,d) 已展平 token 流 → (logits (N,E), topk_w (N,k), topk_idx (N,k))。
    """

    def __init__(self, hidden_size, n_expert, top_k, scoring="softmax", n_group=1, topk_group=1,
                 norm_topk_prob=False, routed_scaling_factor=1.0, use_bias=False, bias_u=0.001):
        super().__init__()
        assert scoring in ("softmax", "sigmoid")
        self.weight = nn.Parameter(torch.empty(n_expert, hidden_size))     # (E,d) —— HF 键名 gate.weight
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.n_expert, self.top_k, self.scoring = n_expert, top_k, scoring
        self.n_group, self.topk_group = n_group, topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.bias_u = bias_u
        if use_bias:
            self.register_buffer("e_score_correction_bias", torch.zeros(n_expert))   # (E,) Buffer 非 Parameter

    def forward(self, h):
        logits = F.linear(h.float(), self.weight.float())                  # (N,d)·(E,d)ᵀ -> (N,E) fp32
        scores = logits.softmax(dim=-1) if self.scoring == "softmax" else logits.sigmoid()
        choice = scores
        if hasattr(self, "e_score_correction_bias"):
            choice = choice + self.e_score_correction_bias                 # bias 只进选择段（先加——组代表也吃它，对齐 HF）
        if self.n_group > 1:                                               # 组限制选择（先选组再选专家）
            g = choice.view(-1, self.n_group, self.n_expert // self.n_group)
            rep = (g.topk(2, dim=-1)[0].sum(dim=-1) if self.scoring == "sigmoid"
                   else g.max(dim=-1).values)                              # 组代表：V3=top2 和 / V2=组内 max
            gidx = torch.topk(rep, self.topk_group, dim=-1, sorted=False)[1]
            mask = torch.zeros_like(rep).scatter_(1, gidx, 1.0)
            keep = (mask.unsqueeze(-1).expand_as(g)).reshape_as(choice).bool()
            choice = choice.masked_fill(~keep, float("-inf"))              # 组外出局（-inf 不影响 sigmoid 分）
        _, idx = torch.topk(choice, self.top_k, dim=-1, sorted=False)      # (N,k)
        w = scores.gather(1, idx)                                          # 门值 = 原始打分（不带 bias）
        if self.norm_topk_prob:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)                  # 选中集内归一（DSV3）
        w = w * self.routed_scaling_factor                                 # DSV2/V3 共有：×rsf
        return logits, w, idx

    @torch.no_grad()
    def step_bias(self, counts, u=None):
        """noaux 步后更新：b_i ← b_i + u·sign(e_i)，e_i = c̄_i − c_i（理想均值 − 实收计数）。

        方向（2408.15664 Alg 1 逐字——欠载 e_i>0 → bias 抬升多被选，过载 → bias 压低）：以比例版
        e_i ∝ 1/E − frac_i 实现（frac_i = counts_i / Σcounts）。本方法不进计算图（no_grad），
        bias 是 Buffer——优化器不训练它，损失图也看不见它。
        counts: (E,) 本步（整 batch）各专家实收选择数。
        """
        u = self.bias_u if u is None else u
        frac = counts.float() / counts.float().sum()
        e = 1.0 / self.n_expert - frac                                     # (E,) 欠载为正
        self.e_score_correction_bias += u * torch.sign(e)


class DeepSeekMoE(nn.Module):
    """DeepSeekMoE 式 MoE 块（教学重写件，正文 5.5/5.6 的代码正身）。

    组装：gate = DeepseekRouter + experts = ExpertBank（ch02 复用，E 个窄 SwiGLU 融合 3D）
    + shared_experts = ExpertSwiGLU（宽度 n_shared·w，每 token 无条件激活）。
    前向：y = Σ_{e∈top-k} g_e(x)·Expert_e(x) + Shared(x)——路由专家细而多、共享专家常驻。
    键位与 HF 5.18.0 DeepseekV2Moe 严格同名：mlp.gate.weight / mlp.experts.{gate_up,down}_proj /
    mlp.shared_experts.{gate,up,down}_proj.weight（noaux 档另有一枚 gate.e_score_correction_bias）。
    构造双形态：DeepSeekMoE(cfg)（别名族 n_expert|n_routed_experts、top_k|num_experts_per_tok、
    expert_dim|moe_intermediate_size、n_shared|n_shared_experts）或关键字参数形态。
    forward(x, out_router_logits=False)：x (B,n,d) -> hidden (B,n,d)。
    """

    def __init__(self, cfg=None, hidden_size=None, n_expert=None, top_k=None, expert_dim=None,
                 n_shared=1, scoring="softmax", n_group=1, topk_group=1, norm_topk_prob=False,
                 routed_scaling_factor=1.0, use_bias=False, bias_u=0.001, **_):
        super().__init__()
        if cfg is not None:
            hidden_size = _field(cfg, "hidden_size")
            n_expert = _field(cfg, "n_expert", "n_routed_experts", "num_experts")
            top_k = _field(cfg, "top_k", "num_experts_per_tok")
            expert_dim = _field(cfg, "expert_dim", "moe_intermediate_size", "intermediate_size")
            n_shared = _field(cfg, "n_shared", "n_shared_experts")
        assert all(v is not None for v in (hidden_size, n_expert, top_k, expert_dim)), \
            "缺几何参数：hidden_size/n_expert/top_k/expert_dim"
        self.gate = DeepseekRouter(hidden_size, n_expert, top_k, scoring=scoring, n_group=n_group,
                                   topk_group=topk_group, norm_topk_prob=norm_topk_prob,
                                   routed_scaling_factor=routed_scaling_factor,
                                   use_bias=use_bias, bias_u=bias_u)
        self.experts = ExpertBank(hidden_size, n_expert, expert_dim)       # ch02 件复用（E,2w,d)/(E,d,w)
        self.shared_experts = ExpertSwiGLU(hidden_size, n_shared * expert_dim)   # (B,n,d)->(B,n,d)
        self.n_expert, self.top_k, self.expert_dim, self.n_shared = n_expert, top_k, expert_dim, n_shared
        self.hidden_size = hidden_size

    def forward(self, x, out_router_logits=False):
        shape = x.shape                                                    # (B,n,d)
        h = x.reshape(-1, shape[-1])                                       # (B,n,d) -> (N,d)，N=B·n
        logits, topk_w, topk_idx = self.gate(h)                            # (N,E) / (N,k) / (N,k)
        out, _counts = self.experts(h, topk_w, topk_idx)                   # (N,d) 路由专家加权合并
        out = out.view(*shape)                                             # (N,d) -> (B,n,d)
        out = out + self.shared_experts(x)                                 # 共享专家：全 token 无条件激活
        if out_router_logits:
            return out, logits.view(*shape[:-1], self.n_expert)            # (B,n,E)
        return out

    def hand_params(self):
        """参数账双口径（本块，正文 5.2/5.7 配臂公式的代码版）：
        总参 = E·3dw（路由专家）+ n_shared·3dw（共享）+ E·d（路由门）；
        激活参 = k·3dw + n_shared·3dw + E·d（每 token 只激活 k 个路由专家，路由门与共享专家常开）。"""
        d, w = self.hidden_size, self.expert_dim
        total = (self.n_expert + self.n_shared) * 3 * d * w + self.n_expert * d
        active = (self.top_k + self.n_shared) * 3 * d * w + self.n_expert * d
        return {"total": total, "active": active}

    @torch.no_grad()
    def expert_load(self, x):
        """本批各路由专家实收选择数 (E,)——只跑路由不算专家（负载探针口径）。"""
        h = x.reshape(-1, x.shape[-1])                                     # (N,d)
        _, _, idx = self.gate(h)
        return torch.bincount(idx.reshape(-1), minlength=self.n_expert)    # (E,)


# ---------------- 自测：三方对拍 + noaux 机制自检 ----------------
def main(out_name):
    device = torch.device("cpu")                                           # 数值对拍一律 CPU fp32（纪律）
    t0 = time.perf_counter()
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {"seed": SEED, "device": "cpu fp32", "torch": torch.__version__}

    # ---- ① 教学件 vs 正身 HFDeepseekMoE：同权重前向，期望逐位一致 ----
    d, E, k, w, s = 64, 4, 2, 128, 1                                       # 小而全几何（共享 1）
    torch.manual_seed(SEED)
    ref = HFDeepseekMoE(d, E, k, w, n_shared=s).eval()
    torch.manual_seed(SEED)
    teach = DeepSeekMoE(hidden_size=d, n_expert=E, top_k=k, expert_dim=w,
                        n_shared=s, scoring="softmax").eval()
    missing, unexpected = teach.load_state_dict(ref.state_dict(), strict=False)
    assert not missing and not unexpected, f"键位不符 missing={missing} unexpected={unexpected}"
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(2, 32, d, generator=g)                                 # (2,32,64)
    with torch.no_grad():
        yt, rl = teach(x, out_router_logits=True)                          # (2,32,64) + (2,32,4)
        yh = ref(x)                                                        # (2,32,64)
    diff_ref = (yt - yh).abs().max().item()
    print(f"[① 教学件 vs 正身] strict 直搬通过 | max|Δhidden| = {diff_ref:.2e}"
          f"（{'逐位一致' if diff_ref == 0.0 else '数值一致'}）")
    print(f"    router_logits 形状 {tuple(rl.shape)} = (B,n,E)✓ | 键位 {sorted(teach.state_dict())}")
    report["teach_vs_ref"] = {"max_abs_diff": diff_ref, "keys": sorted(teach.state_dict()),
                              "verdict": "PASS" if diff_ref == 0.0 else "FAIL"}

    # ---- ② vs HF DeepseekV2Moe 块级对拍（探针 G 路径：小 config 随机权重 strict 直搬） ----
    from transformers import DeepseekV2Config
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Moe
    hcfg = DeepseekV2Config(hidden_size=64, hidden_act="silu", mlp_bias=False,
                            n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=128,
                            n_shared_experts=1, topk_method="greedy", n_group=1, topk_group=1,
                            routed_scaling_factor=1.0)
    torch.manual_seed(SEED)
    ours = DeepSeekMoE(hidden_size=64, n_expert=4, top_k=2, expert_dim=128, n_shared=1,
                       scoring="softmax").eval()
    hf = DeepseekV2Moe(hcfg).eval()
    hf.load_state_dict(ours.state_dict(), strict=True, assign=True)        # strict 直搬（键位全对齐）
    with torch.no_grad():
        y_o, rl_o = ours(x, out_router_logits=True)                        # (2,32,64) + (2,32,4)
        y_h = hf(x)                                                        # (2,32,64)
    delta = (y_o - y_h).abs().max().item()
    n_par = sum(p.numel() for p in ours.parameters())
    print(f"[② vs HF 5.18.0 DeepseekV2Moe] strict 直搬通过 {n_par:,} 参 | max|Δhidden| = {delta:.2e}")
    report["vs_hf_deepseekv2moe"] = {"params": n_par, "max_abs_diff_hidden": delta,
                                     "router_logits_shape": list(rl_o.shape),
                                     "verdict": "PASS" if delta < 1e-5 else "FAIL"}

    # ---- ③ noaux 三段对拍 vs HF DeepseekV3TopkRouter（打分 / 选择 / 权重逐段） ----
    #  两档：n_group=1（隔离组选择，只验 noaux 本体）；n_group=2/topk_group=1（过组选择路径）。
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3TopkRouter
    h2 = x.reshape(-1, d)                                                  # (N,d)=(64,64)
    tri = {}
    for tag, (ng, tg) in (("n_group=1", (1, 1)), ("n_group=2", (2, 1))):
        torch.manual_seed(SEED)
        gate = DeepseekRouter(d, E, k, scoring="sigmoid", n_group=ng, topk_group=tg,
                              norm_topk_prob=True, routed_scaling_factor=2.5, use_bias=True).eval()
        with torch.no_grad():                                              # 演示性 bias：让选择段真的被扰动
            gate.e_score_correction_bias.copy_(torch.tensor([0.30, -0.10, 0.05, -0.02]))
        from transformers import DeepseekV3Config
        v3cfg = DeepseekV3Config(hidden_size=64, num_local_experts=4, num_experts_per_tok=2,
                                 n_group=ng, topk_group=tg, norm_topk_prob=True,
                                 routed_scaling_factor=2.5)
        hf3 = DeepseekV3TopkRouter(v3cfg).eval()
        hf3.load_state_dict(gate.state_dict(), strict=True, assign=True)
        with torch.no_grad():
            lo, lw, li = gate(h2)                                          # 教学件三段输出
            ho, hw, hi = hf3(h2)                                           # HF 三段输出
        d_sc, d_w, d_i = (lo - ho).abs().max().item(), (lw - hw).abs().max().item(), \
            (li != hi).float().mean().item() * k * h2.shape[0]
        print(f"[③ noaux 三段对拍 {tag}] max|Δlogits| = {d_sc:.2e} | max|Δweights| = {d_w:.2e} | "
              f"idx 不一致数 = {int(d_i)}")
        tri[tag] = {"max_abs_diff_logits": d_sc, "max_abs_diff_weights": d_w,
                    "idx_mismatch_count": int(d_i),
                    "verdict": "PASS" if d_sc == 0.0 and d_w == 0.0 and d_i == 0 else "FAIL"}
    report["noaux_vs_hf_v3router"] = tri

    # ---- ④ 机制自检三条 ----
    checks = {}
    # (a) bias 只影响选择、不进门值：换 bias 后，选择未变的 token 权重逐位不变
    torch.manual_seed(SEED)
    gate = DeepseekRouter(d, E, k, scoring="sigmoid", norm_topk_prob=True,
                          routed_scaling_factor=2.5, use_bias=True).eval()
    with torch.no_grad():
        _, w0, i0 = gate(h2)
        gate.e_score_correction_bias.copy_(torch.tensor([0.25, 0.10, -0.05, -0.30]))
        _, w1, i1 = gate(h2)
        same = (i0 == i1).all(dim=-1)                                      # (N,) 选择集未变的 token
        w_diff_on_same = (w0[same] - w1[same]).abs().max().item() if same.any() else 0.0
        n_changed = int((~same).sum())
    checks["bias_gate_value_invariance"] = {
        "tokens_selection_changed": n_changed, "max|Δw| on unchanged tokens": w_diff_on_same,
        "verdict": "PASS" if w_diff_on_same == 0.0 and n_changed > 0 else "FAIL"}
    # (b) 负反馈方向：过载专家 bias 降、欠载专家 bias 升（Alg 1 的 e_i = c̄_i − c_i 方向）
    counts = torch.tensor([6, 1, 0, 1])                                    # 均值 2：专家 0 过载、1/2/3 欠载
    before = gate.e_score_correction_bias.clone()
    gate.step_bias(counts, u=0.001)
    after = gate.e_score_correction_bias
    ok_dir = after[0] < before[0] and after[2] > before[2] and after[1] > before[1]
    checks["bias_feedback_direction"] = {
        "counts": counts.tolist(), "delta_bias": (after - before).tolist(),
        "verdict": "PASS" if ok_dir else "FAIL"}
    # (c) 参数账双口径：手算 vs 实测（含路由门口径——正文 5.7 配臂公式的口径）
    hand = teach.hand_params()
    n_all = sum(p.numel() for p in teach.experts.parameters()) \
        + sum(p.numel() for p in teach.shared_experts.parameters()) \
        + teach.gate.weight.numel()
    checks["hand_params"] = {"hand_total": hand["total"], "counted_total": n_all,
                             "hand_active": hand["active"],
                             "verdict": "PASS" if hand["total"] == n_all else "FAIL"}
    print(f"[④ 机制自检] bias 不进门值：{n_changed} token 换了选择、未变 token max|Δw| = "
          f"{w_diff_on_same:.2e} | 负反馈方向 {'对' if ok_dir else '错'}（Δbias="
          f"{[round(float(x), 4) for x in (after - before)]}）| 参数账 {hand['total']:,} 逐位")
    report["mechanism_checks"] = checks

    report["wall_sec"] = round(time.perf_counter() - t0, 1)
    ok = (report["teach_vs_ref"]["verdict"] == "PASS"
          and report["vs_hf_deepseekv2moe"]["verdict"] == "PASS"
          and all(v["verdict"] == "PASS" for v in tri.values())
          and all(v["verdict"] == "PASS" for v in checks.values()))
    report["verdict"] = "PASS" if ok else "FAIL"
    path = os.path.join(OUT_DIR, f"deepseek_moe_parity_{out_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[判定] {report['verdict']} | 产物 {path} | 墙钟 {report['wall_sec']} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch5 教学件 deepseek_moe.py 自测：三方对拍 + noaux 机制自检（CPU fp32）")
    ap.add_argument("--out-name", default="run1", help="产物 JSON 后缀（防覆写）")
    args = ap.parse_args()
    main(args.out_name)
