# 04_probe_d_dense_vmoe.py —— Book4 探针 D：MoE vs 等预算稠密 fast 冒烟（300 步三臂）
# 用途：为 ch4/ch5 的「MoE 划不划算」消融定对照口径——两个等预算口径各配一臂 dense：
#       ① MoE 臂（E=8/top-2、专家 d_ff=704；总参 65.0M、激活参 26.1M——探针 B 同臂）；
#       ② dense-等激活臂：d_ff = k×w_e = 2×704 = 1408（每 token 激活的 FFN 宽度与 MoE 相同；
#           自算公式——等激活 dense d_ff = top_k × expert_dim，若 MoE 有共享专家则再加 n_shared×w_e）；
#       ③ dense-等总参臂：d_ff = E×w_e = 8×704 = 5632（FFN 槽总参数与 MoE 相同——
#           3·d·d_ff = 3·d·E·w_e，忽略路由门 E·d 的 ~0.05% 差额）。
#       三臂同 token 预算（B=8/T=512 × 300 步 = 12.3M token）、同配方同种子；
#       只验「管道跑通 + 方向初读」（MoE 是否在两个口径下都不输），正式消融（多种子/长训/eval 曲线）
#       归写作期。等激活臂=同 FLOPs 多参数，等总参臂=同参数多 FLOPs——ch1 三口径账本的现场对照。
# 所属章节：Book4 ch4（Mixtral 组装术的对照逻辑）/ ch5（细粒度专家的消融设计）；00-feasibility 探针 D。
# 运行方式：source env.sh && python 04_probe_d_dense_vmoe.py [--out-name run1] [--steps 300]
#   （三臂合计约 5-8 min；若投影超 25 min 自动降档并留痕）
# 产物：log/book4-feasibility/probeD_dense_vmoe_{out}.json（不入库）
import argparse
import time

import numpy as np
import torch

from moe_mla_slots import (SEED, TOKENS_SP8K, activated_account, eval_loss, hand_account,
                           pick_device, save_json, train_loop)
from moe_mla_slots import SlotConfig, SlotLLaMA

LR = 1e-3


def run_arm(tag, cfg, steps, batch, block, device):
    torch.manual_seed(SEED)
    model = SlotLLaMA(cfg).to(device)
    hand = hand_account(cfg)
    assert model.n_params() == hand["total"]
    act, act_ne = activated_account(cfg)
    print(f"[{tag}] 总参 {model.n_params():,} | 激活参 {act:,} | d_ff/e="
          f"{cfg.moe['expert_dim'] if cfg.moe else cfg.intermediate_size}")
    t0 = time.perf_counter()
    secs, losses, _ = train_loop(model, steps, batch, block, device, TOKENS_SP8K, peak_lr=LR)
    ev = eval_loss(model, device)
    steady = secs[1:]
    res = {"tag": tag, "params_total": model.n_params(), "params_active": act,
           "params_active_nonemb": act_ne, "steps_run": steps,
           "sec_per_step_steady": round(float(np.median(steady)), 3),
           "wall_sec": round(time.perf_counter() - t0, 1),
           "loss_first20_mean": round(float(np.mean(losses[:20])), 4),
           "loss_last20_mean": round(float(np.mean(losses[-20:])), 4),
           "eval_loss": round(ev, 4),
           "tokens_seen": steps * batch * block}
    print(f"[{tag}] {res['sec_per_step_steady']} s/step | loss {res['loss_first20_mean']}→"
          f"{res['loss_last20_mean']} | eval {ev:.4f} | 墙钟 {res['wall_sec']} s")
    return res


def main(out_name, steps):
    device = pick_device("mps")
    common = dict(vocab_size=8192, hidden_size=512, num_hidden_layers=6,
                  num_attention_heads=8, num_key_value_heads=4)
    E, k, w = 8, 2, 704
    arms = {
        "moe": SlotConfig(**common, intermediate_size=1408,
                          moe=dict(n_expert=E, top_k=k, expert_dim=w, n_shared=0, variant="teach")),
        "dense_eq_active": SlotConfig(**common, intermediate_size=k * w),      # 1408 = k×w_e
        "dense_eq_total": SlotConfig(**common, intermediate_size=E * w),       # 5632 = E×w_e
    }
    order = ["dense_eq_active", "moe", "dense_eq_total"]                       # 由快到慢
    results = {tag: run_arm(tag, arms[tag], steps, 8, 512, device) for tag in order}

    da, mo, dt = results["dense_eq_active"], results["moe"], results["dense_eq_total"]
    reading = {
        "等激活口径 (MoE vs dense_eq_active)": f"eval {mo['eval_loss']} vs {da['eval_loss']} "
        f"（Δ={round(mo['eval_loss']-da['eval_loss'],4)} nat；同激活参/FLOPs，MoE 多 2.5× 总参）",
        "等总参口径 (MoE vs dense_eq_total)": f"eval {mo['eval_loss']} vs {dt['eval_loss']} "
        f"（Δ={round(mo['eval_loss']-dt['eval_loss'],4)} nat；同总参，dense 多 ~2.5× FLOPs/步）",
        "提示": "300 步 fast 冒烟只读方向；正式结论归写作期消融（多种子+长训+曲线）",
    }
    for kk, vv in reading.items():
        print(f"[初读] {kk}: {vv}")
    save_json("probeD_dense_vmoe", {"seed": SEED, "device": str(device), "steps": steps,
                                    "arms": results, "first_reading": reading}, out_name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="探针 D：MoE vs 等激活/等总参稠密三臂冒烟")
    ap.add_argument("--out-name", default="run1", help="产物 JSON 后缀（防覆写）")
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    main(args.out_name, args.steps)
