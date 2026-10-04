# 06_probe_f_config.py —— Book4 探针 F：单轨两刀整机候选 config 手算账 + 10 步实测（ch9 设计文档底数）
# 用途：Book3 207M 骨架两刀方案（FFN→MoE + GQA→MLA）的候选定档——ch9 大项目「整合短训」的底数：
#       三个候选（参数逐项手算账 + 模型实测 assert 逐位）：
#       cand1 等激活·重 MLA：MoE(E=8/top2/共享1, 专家宽 w=938——k·w+w_shared=3×938=2814≈dense 2816
#             即「等激活 FFN 宽」自算公式：w = d_ff/(k+n_shared) 向下取整) + MLA(d_c=512/d_c'=64,
#             d_nope=128/d_v=128, r_q=512——旗舰比例的等比缩小)；
#       cand2 等激活·轻 MLA：同 MoE + MLA(d_c=256/d_c'=64, d_nope=64/d_v=64, r_q=256)——
#             注意力参数近 GQA、KV −68.75%；
#       cand3 等总参·轻 MLA：MLA 同 cand2，MoE 专家宽 w 扫描取「总参最逼近 207,119,360」的值
#             （等总参 FFN 自算公式：3·d·d_ff_dense = 3·d·(E+n_shared)·w + E·d（路由门）→ 解 w）。
#       每候选：MPS bf16（fp32 权重 autocast，Book3 同口径）前向+反向 10 步 sec/step + 内存峰值；
#       与 207M 稠密（Book3 底座）同批（tokens32k，B=8/T=1024）3 步中位 wall-clock 对照。
#       全部数字=ch9 设计文档「预算表」与「整合短训档位」的实测底数。
# 所属章节：Book4 ch9（DSV2→DSV3 整机拆解 + 单轨两刀整合）；00-feasibility 探针 F。
# 运行方式：source env.sh && python 06_probe_f_config.py [--out-name run1]（约 5-8 min）
# 产物：log/book4-feasibility/probeF_config_{out}.json（不入库）
import argparse
import time

import numpy as np
import torch

from moe_mla_slots import (SEED, TOKENS_SP32K, TokenStream, activated_account, bootstrap_book3,
                           hand_account, kv_per_token_elements, pick_device, save_json)
from moe_mla_slots import SlotConfig, SlotLLaMA

BASE_TOTAL = 207_119_360          # Book3 215M 定版（对照锚）
BT = (8, 1024)                    # Book3 基线口径 B/T（1.702 s/step @ MPS bf16）

MLA_HEAVY = dict(kv_lora_rank=512, q_lora_rank=512, qk_nope_head_dim=128,
                 qk_rope_head_dim=64, v_head_dim=128)      # 旗舰比例缩小（d_c=512/d_c'=64）
MLA_LITE = dict(kv_lora_rank=256, q_lora_rank=256, qk_nope_head_dim=64,
                qk_rope_head_dim=64, v_head_dim=64)        # 小尺度友好（d_c=256/d_c'=64）


def moe_spec(w, mode="mixtral"):
    return dict(n_expert=8, top_k=2, expert_dim=w, n_shared=1, variant="teach", mode=mode)


def build_cfg(mla, w):
    return SlotConfig(vocab_size=32000, hidden_size=1024, num_hidden_layers=12,
                      num_attention_heads=16, num_key_value_heads=8,
                      intermediate_size=2816, moe=moe_spec(w), mla=mla,
                      rope_theta=10000.0, rms_norm_eps=1e-5)


def solve_w_eq_total(mla, target=BASE_TOTAL):
    """等总参专家宽求解：总参(w) 单调 → 线性扫 w 取 |总参−target| 最小（粗扫+邻域细扫）。"""
    best = None
    for w in range(280, 380):
        t = hand_account(build_cfg(mla, w))["total"]
        if best is None or abs(t - target) < abs(best[1] - target):
            best = (w, t)
    return best


def run_10step(cfg, tag, device, steps=10):
    """10 步前向+反向（含 AdamW+clip，Book3 同配方；MPS bf16 autocast）。"""
    torch.manual_seed(SEED)
    model = SlotLLaMA(cfg).to(device)
    stream = TokenStream(TOKENS_SP32K, *BT)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
    secs, losses, mem = [], [], []
    for step in range(1, steps + 1):
        x, y = stream.batch(step, device)
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.mps.synchronize() if device.type == "mps" else None
        secs.append(time.perf_counter() - t0)
        losses.append(loss.item())
        if device.type == "mps":
            mem.append(torch.mps.current_allocated_memory() / 1024 / 1024)
    return {"sec_per_step_steady": round(float(np.median(secs[1:])), 3),
            "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4),
            "mps_peak_mb": round(max(mem), 1) if mem else None, "losses": [round(v, 4) for v in losses]}


def main(out_name):
    device = pick_device("mps")
    slots = bootstrap_book3()
    ln_v = float(np.log(32000))

    # 等激活专家宽：w = floor(2816/(k+n_shared)) = floor(2816/3) = 938
    w_eq_active = 2816 // (2 + 1)
    w_eq_total, t_eq_total = solve_w_eq_total(MLA_LITE)
    print(f"[定宽] 等激活 w = {w_eq_active}（3×{w_eq_active}={3*w_eq_active} ≈ dense 2816）| "
          f"等总参 w = {w_eq_total}（cand3 总参 {t_eq_total:,}，|Δtarget| = {abs(t_eq_total-BASE_TOTAL):,}）")

    cands = {
        "cand1_eqactive_mla512": build_cfg(MLA_HEAVY, w_eq_active),
        "cand2_eqactive_mla256": build_cfg(MLA_LITE, w_eq_active),
        "cand3_eqtotal_mla256": build_cfg(MLA_LITE, w_eq_total),
    }
    report = {"seed": SEED, "device": str(device), "bt": f"B={BT[0]}/T={BT[1]}", "candidates": {}}
    for tag, cfg in cands.items():
        torch.manual_seed(SEED)
        model = SlotLLaMA(cfg)
        hand = hand_account(cfg)
        assert model.n_params() == hand["total"], f"{tag}: {model.n_params()} != {hand['total']}"
        act, act_ne = activated_account(cfg)
        kv_e, kv_b = kv_per_token_elements(cfg)
        perf = run_10step(cfg, tag, device)
        row = {"config": {"d": 1024, "L": 12, "h": 16, "E": 8, "top_k": 2, "n_shared": 1,
                          "w_expert": cfg.moe["expert_dim"], "d_ff_dense_baseline": 2816,
                          "mla": cfg.mla, "V": 32000},
               "params_total": model.n_params(), "params_total_hand": hand["total"],
               "params_active": act, "params_active_nonemb": act_ne,
               "kv_elems_per_token": kv_e, "kv_bytes_per_token_bf16": kv_b,
               "kv_saving_vs_gqa_base": round(1 - kv_e / (2 * 12 * 8 * 64), 4),
               "perf_mps_bf16": perf, "hand_detail": {k: v for k, v in hand.items() if k != "mlp_moe"}}
        report["candidates"][tag] = row
        del model
        print(f"[{tag}] 总参 {row['params_total']:,} | 激活参 {act:,}（non-emb {act_ne:,}）| "
              f"KV/token {kv_e} elems（−{row['kv_saving_vs_gqa_base']*100:.1f}% vs GQA 基线）| "
              f"{perf['sec_per_step_steady']} s/step | loss {perf['loss_first']}→{perf['loss_last']} "
              f"| 峰值 {perf['mps_peak_mb']} MB")

    # 207M 稠密对照（Book3 底座，同批同口径 3 步中位）
    torch.manual_seed(SEED)
    base = slots.LLaMA(slots.LLaMAConfig(**slots.LLAMA_215M)).to(device)
    stream = TokenStream(TOKENS_SP32K, *BT)
    opt = torch.optim.AdamW(base.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
    secs, mem = [], []
    for step in range(1, 4):
        x, y = stream.batch(step, device)
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, loss = base(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
        opt.step()
        torch.mps.synchronize()
        secs.append(time.perf_counter() - t0)
        mem.append(torch.mps.current_allocated_memory() / 1024 / 1024)
    base_perf = {"sec_per_step_median3": round(float(np.median(secs)), 3),
                 "mps_peak_mb": round(max(mem), 1), "params": BASE_TOTAL,
                 "book3_reference_sec_per_step": 1.702, "probeA_remeasure": 1.907}
    report["base_215m_dense"] = base_perf
    print(f"[对照] 207M 稠密 {base_perf['sec_per_step_median3']} s/step（Book3 记录 1.702 / 探针 A 复测 1.907）"
          f"| 峰值 {base_perf['mps_peak_mb']} MB")

    # 整合短训档位建议（ch9 设计文档底数）：≤30 min 能跑多少步
    rec = {}
    for tag, row in report["candidates"].items():
        s = row["perf_mps_bf16"]["sec_per_step_steady"]
        rec[tag] = {"steps_per_30min": int(1800 / s),
                    "tokens_per_30min_M": round(1800 / s * BT[0] * BT[1] / 1e6, 2)}
    report["integration_run_recommendation"] = rec
    print("[短训档位]（≤30 min）", rec)
    save_json("probeF_config", report, out_name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="探针 F：两刀整机候选账 + 10 步实测")
    ap.add_argument("--out-name", default="run1", help="产物 JSON 后缀（防覆写）")
    args = ap.parse_args()
    main(args.out_name)
