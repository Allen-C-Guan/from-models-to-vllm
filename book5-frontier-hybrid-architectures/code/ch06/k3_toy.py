# k3_toy.py —— Book5 ch6 正式件：K3 玩具替身整机（K3Toy）+ 参数账双口径 assert + MXFP4 钩子
# 用途：ch6 6.7「K3 玩具替身·注意力侧」的整机装配正身——
#   ① 血统写在 import 语句里：Book3 RMSNorm/RoPE（经 Book4 moe_mla_slots 的 SlotLLaMA 底座）+
#      Book4 MLA 件（MLAAttention）+ 本册 ch05 GDN 件（GatedDeltaNet——替身插槽）+ 本册 ch06 KDA 件
#      （KimiDeltaAttention——K3 化正身插槽）+ Book4 SparseMoE——四代组件一台机器；
#   ② 层排布 [kda×3, mla]×2（full_attention_interval=4 缩微；末层全局——K3「末层必全局」缩微）；
#   ③ 参数账双口径 assert（config 公式整数精确算 vs 模型实测 n_params 逐位对账）：
#      gdn_sub 臂（ch05 GatedDeltaNet 替身插槽）= toyA 定版账 **43,905,888 / 激活非嵌入 11,662,176**；
#      kda 臂（K3 化正身插槽：per-channel 下界门 + 低秩遗忘门 + 满秩输出门）= K3 化换装账
#      **44,277,936 / 12,034,224**（差值 +372,048 = 6 层 × 62,008/层——「K3 化单点替换的参数代价」
#      逐项报，主账仍 toyA——大纲口径「换档另报参数账，主账仍 toyA」）；
#   ④ MXFP4 钩子：路由专家权重 STE 模拟量化（复用 Book4 ch07 fp8_sim.mxfp4_rt——块 32/E8M0；
#      只量路由专家 12 Linear/层，router/共享专家/注意力不量——K3 侧口径）；
#   ⑤ KDA 遗忘门分布采集（collect_g_stats——「每通道学出了不同的遗忘速度」第一手，图 6.4 数据源）。
# 【替换件几何（toyA 定版；大纲 v1.1 §K3 玩具 config）】
#   V=8192 / d=512 / L=8；KDA 插槽 h_k=h_v=8、d_k=d_v=64、conv 4、chunk 64（gdn_sub）/16（kda——
#   K3 报告 16-token 数字链口径）；MLA 插槽 h=8、kv_lora=128、q_lora=None、qk_nope=64、qk_rope=32、
#   v_head=64（承 Book4 ch06 件几何）；MoE E=64/top4/共享 1/w_e=32（稀疏度 16:1——K3 896:16=56:1 缩微）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch06/k3_toy.py" [--out-name run1]
#   CPU 参数账 assert + MPS 3 步冒烟；预计 <2 分钟。
# 产物：log/book5-ch06/k3_toy_{out}.json（不入库）
import argparse
import importlib.util
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book5-ch06")
SEED = 20261002

# ---------------- 三候选 bootstrap（hybrid409 `_B4_CH09_CANDS` 同款；书仓布局同款可命中） ----------------
_B4_FEAS_CANDS = [
    os.path.join(REPO_ROOT, "code", "Book4-规模与效率", "00-feasibility"),              # 工作区布局
    os.path.join(REPO_ROOT, "from-models-to-vllm", "book4-scale-and-efficiency", "code", "00-feasibility"),
    os.path.join(REPO_ROOT, "book4-scale-and-efficiency", "code", "00-feasibility"),    # 书仓裸布局
]
_B4_CH07_CANDS = [
    os.path.join(REPO_ROOT, "code", "Book4-规模与效率", "ch07"),
    os.path.join(REPO_ROOT, "from-models-to-vllm", "book4-scale-and-efficiency", "code", "ch07"),
    os.path.join(REPO_ROOT, "book4-scale-and-efficiency", "code", "ch07"),
]


def _first(cands, need):
    """三候选里第一个含 need 文件的目录（工作区/书仓 monorepo/书仓裸布局）。"""
    for p in cands:
        if os.path.exists(os.path.join(p, need)):
            return p
    raise FileNotFoundError(need, cands)


def _fp8_path():
    return os.path.join(_first(_B4_CH07_CANDS, "fp8_sim.py"), "fp8_sim.py")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bootstrap():
    """加载 Book4 moe_mla_slots（SlotLLaMA 底座/MLA 件/训练环）、fp8_sim（MXFP4）、本册 ch05/ch06 件。"""
    feas = _first(_B4_FEAS_CANDS, "moe_mla_slots.py")
    if feas not in sys.path:
        sys.path.insert(0, feas)
    import moe_mla_slots as m4
    fp8 = _load(_fp8_path(), "fp8_sim")
    gdn = _load(os.path.join(HERE, "..", "ch05", "gdn.py"), "gdn_ch05")   # 本册件 HERE 相对——双布局同构
    kda = _load(os.path.join(HERE, "kda.py"), "kda_ch06")
    return m4, fp8, gdn, kda


def save_json(name, obj, out_name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}_{out_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[产物] {path}")
    return path


# ---------------- 手算账（整数精确——与 n_params() 双向逐位 assert） ----------------
MLA_GEOM = dict(kv_lora_rank=128, qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64,
                q_lora_rank=None)


def hand_account(V, d, pattern, E, top_k, w_e, n_shared, kda_impl):
    """参数逐项手算。pattern: "kda"|"mla" 逐层。返回 {total, activated_non_emb, detail}。"""
    h_k = h_v = 8
    d_k = d_v = 64
    conv_k = 4
    # gdn_sub 插槽（ch05 GatedDeltaNet 教学骨架——toyA 主账正身）
    conv_dim = 2 * h_k * d_k + h_v * d_v
    gdn_attn = (d * (2 * h_k * d_k + 2 * h_v * d_v)                              # in_proj_qkvz
                + d * 2 * h_v                                                    # in_proj_ba
                + conv_dim * conv_k                                              # conv1d
                + h_v + h_v                                                      # A_log + dt_bias
                + h_v * d_v * d)                                                 # out_proj（norm 无权）
    # kda 插槽（ch06 KimiDeltaAttention K3 化正身——K3 报告 Table 1 侧算同构）
    rank = d_k                                                                   # 低秩秩数 = head_dim
    kda_attn = (3 * d * (h_k * d_k)                                              # q/k/v 投影 ×3
                + 3 * (h_k * d_k) * conv_k                                       # 短卷积
                + d * h_v                                                        # β 投影
                + d * rank + rank * (h_v * d_v) + h_v * d_v + h_v                # 遗忘门低秩+dt_bias+A_log
                + d * d                                                          # 满秩输出门 W_g（512×512 同构缩微）
                + d_v                                                            # o_norm 权重（头间共享）
                + (h_v * d_v) * d)                                               # o_proj
    attn_of = {"gdn_sub": gdn_attn, "kda": kda_attn}[kda_impl]
    m = MLA_GEOM
    h = 8
    mla_attn = (d * h * (m["qk_nope_head_dim"] + m["qk_rope_head_dim"])          # q_proj
                + d * (m["kv_lora_rank"] + m["qk_rope_head_dim"]) + m["kv_lora_rank"]   # kv_a + LN
                + m["kv_lora_rank"] * h * (m["qk_nope_head_dim"] + m["v_head_dim"])    # kv_b
                + h * m["v_head_dim"] * d)                                       # o
    moe = E * d + E * 3 * d * w_e + n_shared * 3 * d * w_e                       # 门 + 路由专家 + 共享
    per_layer, detail = [], {"attn": attn_of, "mla_attn": mla_attn, "moe": moe,
                             "kda_impl": kda_impl}
    for kind in pattern:
        norms = 2 * d                                                            # 层内双 norm（MLA 潜向量 LN 已计入 mla_attn）
        per_layer.append((attn_of if kind == "kda" else mla_attn) + moe + norms)
    emb = 2 * V * d + d                                                          # untied 两份 + 终态 norm
    total = sum(per_layer) + emb
    act_body = sum((attn_of if k == "kda" else mla_attn) + 3 * d * (top_k * w_e + n_shared * w_e)
                   + 2 * d for k in pattern) + d
    return {"total": total, "activated_non_emb": act_body, "detail": detail}


# toyA 定版账（大纲 v1.1）：gdn_sub 与 kda 两口径——assert 正身
TOY_A_ASSERTS = {
    "gdn_sub": {"total": 43_905_888, "activated_non_emb": 11_662_176},
    "kda": {"total": 44_277_936, "activated_non_emb": 12_034_224},
}


def make_pattern(L=8, interval=4):
    """[kda×(interval−1), mla] 重复——3:1；末层必全局（interval 整除处为 mla，末层落在整除点）。"""
    return ["mla" if (i + 1) % interval == 0 else "kda" for i in range(L)]


# ---------------- K3 玩具整机 ----------------
class K3Toy(nn.Module):
    """K3 玩具替身整机：SlotLLaMA 底座 + 注意力插槽逐层换装（kda/mla）+ 超稀疏 MoE。

    cfg 字段：V/d/L/E/top_k/w_e/n_shared/kda_impl("gdn_sub"|"kda")/gate_mode
    （kda 臂的 per-token 替身粒度）/chunk_kda。forward 与 SlotLLaMA 一致 (idx,targets)->(logits,loss)。
    """

    def __init__(self, cfg):
        super().__init__()
        m4, _, gdn, kda = bootstrap()
        self.m4, self.cfg = m4, cfg
        pattern = cfg.get("pattern") or make_pattern(cfg["L"])
        self.pattern = pattern
        slot_cfg = m4.SlotConfig(vocab_size=cfg["V"], hidden_size=cfg["d"],
                                 num_hidden_layers=cfg["L"], num_attention_heads=8,
                                 num_key_value_heads=4, intermediate_size=1408,
                                 moe=dict(n_expert=cfg["E"], top_k=cfg["top_k"],
                                          expert_dim=cfg["w_e"], n_shared=cfg["n_shared"],
                                          mode="mixtral", variant="teach"))
        model = m4.SlotLLaMA(slot_cfg)
        for i, kind in enumerate(pattern):
            if kind == "kda":
                if cfg.get("kda_impl", "gdn_sub") == "kda":
                    model.model.layers[i].self_attn = kda.KimiDeltaAttention(
                        cfg["d"], 8, 64, 64, conv_kernel=4,
                        chunk=cfg.get("chunk_kda", 16),
                        gate_mode=cfg.get("gate_mode", "per_channel"))
                else:
                    model.model.layers[i].self_attn = gdn.GatedDeltaNet(
                        cfg["d"], 8, 8, 64, 64, conv_kernel=4, chunk=cfg.get("chunk_gdn", 64))
            else:
                model.model.layers[i].self_attn = m4.MLAAttention(
                    cfg["d"], 8, MLA_GEOM["kv_lora_rank"], MLA_GEOM["qk_nope_head_dim"],
                    MLA_GEOM["qk_rope_head_dim"], MLA_GEOM["v_head_dim"],
                    q_lora_rank=MLA_GEOM["q_lora_rank"], rope_theta=10000.0)
        self.inner = model                                                        # 底座整机（唯一注册——参数不重复计）

    @property
    def model(self):
        return self.inner.model

    def n_params(self, non_embedding=False):
        return self.inner.n_params(non_embedding)

    def forward(self, idx, targets=None):
        return self.inner(idx, targets)

    def collect_g_stats(self, x):
        """KDA 遗忘门分布采集：给一批 token id 或隐状态 x，返回各 KDA 层 g 的逐通道均值与直方图。

        g ∈ (g_min,0) per-channel——「每通道学出了不同的遗忘速度」的实测（图 6.4 数据源）。
        实现：临时以实例属性遮蔽各 kda 层的 forget_gate 抓输出（finally 还原，不侵入 kda.py）。
        gdn_sub 臂（无 per-channel 门）返回 None。
        """
        kda_layers = [i for i, k in enumerate(self.pattern) if k == "kda"]
        if not kda_layers or not hasattr(self.model.layers[kda_layers[0]].self_attn,
                                         "forget_gate"):
            return None
        captured = {}

        def wrap(layer_idx, orig):
            def hooked(xi):
                g = orig(xi)
                captured[layer_idx] = g.detach().float().cpu()
                return g
            return hooked

        slots = [self.model.layers[i].self_attn for i in kda_layers]
        try:
            for i, slot in zip(kda_layers, slots):
                slot.forget_gate = wrap(i, slot.forget_gate)                      # 实例属性遮蔽类方法
            with torch.no_grad():
                if x.dtype == torch.long:
                    self.inner(x)
                else:
                    h = x
                    for layer in self.model.layers:
                        h = layer(h, None, None)
        finally:
            for slot in slots:
                del slot.forget_gate                                              # 还原类方法
        stats = []
        for i in kda_layers:
            if i in captured:
                g = captured[i]                                                   # (B,S,h,d_k)
                per_channel = g.mean(dim=(0, 1))                                  # (h,d_k) 逐头逐通道均值
                hist = torch.histogram(g.flatten(), bins=50, range=(-5.0, 0.0))
                stats.append({"layer": i, "per_channel_mean": per_channel.tolist(),
                              "per_channel_std_across_channels": float(per_channel.std()),
                              "hist_counts": hist.hist.tolist(),
                              "bin_edges": hist.bin_edges.tolist(),
                              "g_min_bound": -5.0})
        return stats


# ---------------- MXFP4 钩子（复用 Book4 ch07 fp8_sim.mxfp4_rt） ----------------
class MXFP4Linear(nn.Module):
    """MXFP4 模拟量化线性层（STE 直通）：前向用块量化权重（OCP 口径 32 元素块 + E8M0 共享指数），
    反传对原权重恒等。weight 与被替换的 nn.Linear 共享同一 Parameter（优化器无缝）。"""

    def __init__(self, lin, block=32, fp8=None):
        super().__init__()
        self.weight = lin.weight                                                 # 共享 Parameter（不复制）
        self.block = block
        self.fp8 = fp8 or _load(_fp8_path(), "fp8_sim")

    def forward(self, x):
        w_q, _ = self.fp8.mxfp4_rt(self.weight.data, self.block)                  # (out,in) 量化-反量化
        w = self.weight + (w_q - self.weight).detach()                           # STE：前向 w_q、反传恒等
        return F.linear(x, w)


def apply_mxfp4_to_experts(model):
    """把所有 SparseMoE 路由专家的 Linear 换成 MXFP4Linear（门/共享专家/注意力不量——K3 侧口径）。

    model 可以是 K3Toy 或 SlotLLaMA（k3_toy 底座与 ch9 三刀机共用本钩子）。
    """
    n_swapped = 0
    for layer in model.model.layers:
        mlp = layer.mlp
        if hasattr(mlp, "experts") and isinstance(mlp.experts, (nn.ModuleList, list)):
            for e in mlp.experts:
                e.gate_proj = MXFP4Linear(e.gate_proj)
                e.up_proj = MXFP4Linear(e.up_proj)
                e.down_proj = MXFP4Linear(e.down_proj)
                n_swapped += 3
    return n_swapped


def toy_a_cfg(kda_impl="gdn_sub", gate_mode="per_channel"):
    return dict(V=8192, d=512, L=8, pattern=make_pattern(8), E=64, top_k=4, w_e=32, n_shared=1,
                kda_impl=kda_impl, gate_mode=gate_mode, chunk_gdn=64, chunk_kda=16)


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="Book5 ch6 K3 玩具整机：参数账 assert + 冒烟")
    ap.add_argument("--out-name", default="run1")
    args = ap.parse_args()
    m4 = bootstrap()[0]
    device = m4.pick_device("mps")
    results = {"seed": SEED, "device": str(device)}

    for kda_impl in ("gdn_sub", "kda"):
        cfg = toy_a_cfg(kda_impl=kda_impl)
        hand = hand_account(cfg["V"], cfg["d"], cfg["pattern"], cfg["E"], cfg["top_k"],
                            cfg["w_e"], cfg["n_shared"], kda_impl)
        torch.manual_seed(SEED)
        model = K3Toy(cfg)
        actual = model.n_params()
        want = TOY_A_ASSERTS[kda_impl]
        assert hand["total"] == want["total"], f"{kda_impl}: 手算 {hand['total']} != 定版 {want['total']}"
        assert actual == hand["total"], f"{kda_impl}: 实测 {actual} != 手算 {hand['total']}"
        assert hand["activated_non_emb"] == want["activated_non_emb"]
        print(f"[{kda_impl}] 总参 {actual:,}（手算逐位 ✓ 定版账 ✓）| "
              f"激活非嵌入 {hand['activated_non_emb']:,} ✓ | 明细 {hand['detail']}")
        results[kda_impl] = {"total": actual, "activated_non_emb": hand["activated_non_emb"],
                             "detail": hand["detail"]}

    # 冒烟：kda 臂 3 步（MPS bf16 autocast）+ MXFP4 钩子量/不量参数账不变 assert
    torch.manual_seed(SEED)
    model = K3Toy(toy_a_cfg(kda_impl="kda")).to(device)
    idx = torch.randint(0, 8192, (2, 128), device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        _, loss = model(idx[:, :-1], idx[:, 1:])
    loss.backward()
    gnorm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"[kda 冒烟] loss {loss.item():.3f} | 梯度范数 {gnorm:.2f}（有限）")
    n_before = model.n_params()
    n_swapped = apply_mxfp4_to_experts(model)
    assert model.n_params() == n_before, "量化钩子不得改参数账"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        _, loss_q = model(idx[:, :-1], idx[:, 1:])
    print(f"[MXFP4 钩子] 换 {n_swapped} 个 Linear | 参数账不变 ✓ | loss {loss.item():.3f}→{loss_q.item():.3f}")
    results["smoke"] = {"loss": float(loss.item()), "grad_norm": float(gnorm),
                        "mxfp4_swapped": n_swapped, "loss_with_mxfp4": float(loss_q.item())}

    save_json("k3_toy", results, args.out_name)
    print("ch06 k3_toy.py 完成。")


if __name__ == "__main__":
    main()
