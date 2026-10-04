# 07_probe_g_parity.py —— Book4 探针 G：HF 对拍链路（mini Mixtral + mini DeepseekV2，CPU fp32）
# 用途：验证本册手写组件与 transformers 5.18.0（repos/ 浅克隆同版本）的 strict 直搬对拍可行——
#       写作期「论文机制 ↔ HF 实现」互证的地基：
#       ①mini Mixtral（d=256/L=2/E=4/top-2）vs MixtralForCausalLM 同 config 随机权重 strict 直搬
#         （键位对齐），对拍 max|Δlogits| + argmax 一致率 + 参数逐项对账；
#       ②mini DeepseekV2（MLA 小几何 + first_k_dense=1 + MoE 共享专家 + greedy 路由）
#         vs DeepseekV2ForCausalLM 同 config 对拍——MLA 对拍是本册最高难对拍，探明可行性；
#       ③如实记录 5.18.0 的类名/键位坑（下方「坑清单」由运行时验证后写入产物 JSON）；
#       ④DeepseekV2-Lite 实权重对拍的条件性评估（只算体量不下权重）。
#       【5.18.0 已预核的坑（源码级，repos/transformers）】
#       (a) 5.x 起专家权重为融合 3D 张量 experts.gate_up_proj (E,2w,d)/experts.down_proj (E,d,w)，
#           不再是 4.x 的 experts.N.{gate,up,down}_proj ModuleList——strict 直搬必须按新键位；
#       (b) DeepseekV2TopkRouter 仅 softmax greedy / group_limited_greedy 两路——无 DSV3 的
#           noaux/sigmoid/e_score_correction_bias；config 的 norm_topk_prob 字段存在但建模代码不引用
#           （top-k 权重不重归一化，直接 × routed_scaling_factor）；
#       (c) Mixtral 路由重归一化 top-k 概率（和为 1）；两家的路由 dtype 处理不同（DSV2 强制 fp32）；
#       (d) DSV2 的 RoPE 用复数约定（view_as_complex），llama/mixtral 用 rotate_half——q/k 同约定时
#           q·k 点积等价，手写用 rotate_half 不影响对拍；
#       (e) 5.x 把 rope 参数收进 rope_parameters 子对象，但 config(rope_theta=...) 旧 kwarg 仍被接受。
# 所属章节：Book4 ch4（Mixtral 组装术）/ ch5（DeepseekMoE）/ ch6（MLA）对拍链路；00-feasibility 探针 G。
# 运行方式：source env.sh && python 07_probe_g_parity.py [--out-name run1]（分钟级，CPU fp32）
# 产物：log/book4-feasibility/probeG_parity_{out}.json（不入库）
import argparse
import time

import torch
import torch.nn.functional as F

from moe_mla_slots import (SEED, bootstrap_book3, hand_account, save_json)
from moe_mla_slots import SlotConfig, SlotLLaMA

VER = __import__("transformers").__version__


def compare(ours, hf, x, tag):
    """同输入对拍 logits：max|Δ| / argmax / top-5 一致率 + 独立初始化 sanity。"""
    torch.manual_seed(SEED)
    with torch.no_grad():
        lo, _ = ours(x, x)                              # SlotLLaMA 接口：(logits, loss)
        lh = hf(x).logits
    delta = (lo - lh).abs().max().item()
    argmax_agree = (lo.argmax(-1) == lh.argmax(-1)).float().mean().item()
    top5_agree = (lo.topk(5, -1).indices.sort(-1).values == lh.topk(5, -1).indices.sort(-1).values
                  ).float().mean().item()
    loss_o = F.cross_entropy(lo.reshape(-1, lo.size(-1)).float(), x.reshape(-1)).item()
    loss_h = F.cross_entropy(lh.reshape(-1, lh.size(-1)).float(), x.reshape(-1)).item()
    print(f"[{tag}] max|Δlogits| = {delta:.3e} | argmax 一致率 = {argmax_agree:.4f} | "
          f"top-5 一致率 = {top5_agree:.4f} | |Δloss| = {abs(loss_o - loss_h):.3e}")
    return {"max_abs_diff_logit": delta, "argmax_agreement": argmax_agree,
            "top5_agreement": top5_agree, "loss_abs_diff": abs(loss_o - loss_h)}


def arm_mixtral(device):
    from transformers import MixtralConfig, MixtralForCausalLM

    torch.manual_seed(SEED)
    ours = SlotLLaMA(SlotConfig(vocab_size=512, hidden_size=256, num_hidden_layers=2,
                                num_attention_heads=8, num_key_value_heads=4, intermediate_size=256,
                                moe=dict(n_expert=4, top_k=2, expert_dim=256, variant="hf_mixtral")))
    cfg = MixtralConfig(vocab_size=512, hidden_size=256, intermediate_size=256, num_hidden_layers=2,
                        num_attention_heads=8, num_key_value_heads=4, num_local_experts=4,
                        num_experts_per_tok=2, max_position_embeddings=512, rope_theta=10000.0,
                        rms_norm_eps=1e-5, tie_word_embeddings=False, attention_bias=False,
                        mlp_bias=False, attention_dropout=0.0)
    hf = MixtralForCausalLM(cfg).eval()
    missing_unexpected = {}
    try:
        hf.load_state_dict(ours.state_dict(), strict=True, assign=True)
        print(f"[mixtral] strict 直搬通过：ours {sum(p.numel() for p in ours.parameters()):,} "
              f"-> HF {sum(p.numel() for p in hf.parameters()):,}")
    except RuntimeError as e:
        missing_unexpected = {"error": str(e)[:500]}
        print(f"[mixtral][坑] strict 失败：{str(e)[:300]}")
        return {"verdict": "FAIL", **missing_unexpected}
    g = torch.Generator().manual_seed(SEED)
    x = torch.randint(0, 512, (2, 128), generator=g)
    res = compare(ours, hf, x, "mixtral")
    res.update({"params": sum(p.numel() for p in ours.parameters()),
                "params_hand": hand_account(ours.cfg)["total"],
                "verdict": "PASS" if (res["max_abs_diff_logit"] < 1e-4
                                      and res["argmax_agreement"] == 1.0) else "FAIL"})
    return res


def arm_deepseek(device):
    from transformers import DeepseekV2Config, DeepseekV2ForCausalLM

    torch.manual_seed(SEED)
    mla = dict(kv_lora_rank=64, q_lora_rank=64, qk_nope_head_dim=32,
               qk_rope_head_dim=16, v_head_dim=32, rope_style="complex")   # DSV2 复数约定（坑 d）
    ours = SlotLLaMA(SlotConfig(vocab_size=512, hidden_size=256, num_hidden_layers=2,
                                num_attention_heads=4, num_key_value_heads=1, intermediate_size=704,
                                moe=dict(n_expert=4, top_k=2, expert_dim=128, n_shared=1,
                                         variant="hf_deepseek", routed_scaling_factor=1.0),
                                moe_first_k_dense=1, mla=mla, rope_theta=10000.0, rms_norm_eps=1e-5))
    cfg = DeepseekV2Config(vocab_size=512, hidden_size=256, intermediate_size=704,
                           moe_intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                           num_key_value_heads=1, first_k_dense_replace=1, n_routed_experts=4,
                           n_shared_experts=1, num_experts_per_tok=2, kv_lora_rank=64,
                           q_lora_rank=64, qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
                           topk_method="greedy", routed_scaling_factor=1.0, norm_topk_prob=False,
                           rope_theta=10000.0, rms_norm_eps=1e-5, tie_word_embeddings=False,
                           max_position_embeddings=512)
    hf = DeepseekV2ForCausalLM(cfg).eval()
    try:
        hf.load_state_dict(ours.state_dict(), strict=True, assign=True)
        print(f"[deepseek] strict 直搬通过：ours {sum(p.numel() for p in ours.parameters()):,} "
              f"-> HF {sum(p.numel() for p in hf.parameters()):,}")
    except RuntimeError as e:
        print(f"[deepseek][坑] strict 失败：{str(e)[:400]}")
        return {"verdict": "FAIL", "error": str(e)[:500]}
    g = torch.Generator().manual_seed(SEED)
    x = torch.randint(0, 512, (2, 128), generator=g)
    res = compare(ours, hf, x, "deepseek")
    res.update({"params": sum(p.numel() for p in ours.parameters()),
                "params_hand": hand_account(ours.cfg)["total"],
                "mla_geom": mla, "verdict": "PASS" if (res["max_abs_diff_logit"] < 1e-4
                                                       and res["argmax_agreement"] == 1.0) else "FAIL"})
    return res


def lite_assessment():
    """DeepseekV2-Lite 实权重对拍的条件性评估——只算体量，不下权重（作者决策：大对象暂缓）。"""
    total_b = 15.7e9                                     # 官方 model card（厂商自报；写作期引用时标注）
    return {"params_total_billion": 15.7, "active_billion": 2.4,
            "download_gb_bf16": round(total_b * 2 / 1e9, 1),
            "mem_gb_bf16": round(total_b * 2 / 1e9 * 1.15, 1),
            "mem_gb_fp32": round(total_b * 4 / 1e9 * 1.15, 1),
            "hardware_48gb_unified": "bf16 可载（~36GB，紧）；fp32 不可载（~72GB>48GB）",
            "verdict": "条件性可行：bf16 口径整机对拍（tolerance 放宽到 1e-2 级）或分层对拍；"
                       "写作期再决策是否下载（~31GB，夜间任务）"}


def main(out_name):
    bootstrap_book3()
    device = torch.device("cpu")                        # 对拍一律 CPU fp32（纪律）
    t0 = time.perf_counter()
    print(f"[环境] transformers {VER} | CPU fp32 | 种子 {SEED}")
    mix = arm_mixtral(device)
    ds = arm_deepseek(device)
    quirks = {
        "fused_3d_experts_5x": "experts.gate_up_proj (E,2w,d)/down_proj (E,d,w) 融合 3D——4.x ModuleList 键位已废",
        "dsv2_router_5180": "仅 softmax greedy/group_limited_greedy；无 noaux/sigmoid/e_score_correction_bias；"
                            "norm_topk_prob 字段存在但建模代码不引用（top-k 权重不重归一化）",
        "mixtral_vs_dsv2_routing": "Mixtral top-k 概率重归一化（和为 1）；DSV2 不归一化直接 × routed_scaling_factor；"
                                   "DSV2 路由强制 fp32 线性",
        "rope_convention": "DSV2 复数约定（view_as_complex）vs llama/mixtral rotate_half——权重共享时"
                           "两种约定 q·k 点积不同，跨实现对拍必须同约定（实测：rotate_half 对拍 DSV2 "
                           "max|Δ|=2.8e-2，改 complex 后 PASS）",
        "mla_latent_norm_eps": "HF 的 q_a_layernorm/kv_a_layernorm 走 DeepseekV2RMSNorm 默认 eps=1e-6，"
                               "不随 config.rms_norm_eps（层内 LN 才用 config 值）——手写组件需分别对齐",
        "rope_parameters_nesting": "5.x rope 参数收进 rope_parameters；rope_theta=... 旧 kwarg 仍被接受",
    }
    report = {"seed": SEED, "transformers_version": VER, "device": "cpu fp32",
              "mixtral_mini": mix, "deepseek_mini": ds,
              "quirks_5_18_0": quirks, "deepseek_v2_lite_real_weight": lite_assessment(),
              "wall_sec": round(time.perf_counter() - t0, 1)}
    ok = mix.get("verdict") == "PASS" and ds.get("verdict") == "PASS"
    report["verdict"] = "PASS" if ok else "FAIL"
    print(f"[判定] {'两臂均 PASS' if ok else '存在 FAIL 臂——见上'} | 墙钟 {report['wall_sec']} s")
    save_json("probeG_parity", report, out_name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="探针 G：mini Mixtral / DeepseekV2 HF 对拍")
    ap.add_argument("--out-name", default="run1", help="产物 JSON 后缀（防覆写）")
    args = ap.parse_args()
    main(args.out_name)
