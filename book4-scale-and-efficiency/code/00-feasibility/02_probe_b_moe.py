# 02_probe_b_moe.py —— Book4 探针 B：mini MoE 模块档（第一刀的 mini 预演）
# 用途：在 d=512/L=6 档（26.1M）验证「FFN 插槽换 MoE」在本机可训，并为 ch2-5 的模块实验定档：
#       ① dense 基线（SwiGLU d_ff=1408）vs MoE 版（E=8/top-2、各专家 d_ff=704=dense 的一半——
#       等激活参数口径：每 token 激活 2×704=1408=dense 的 d_ff；总参 26.1M→65.0M，激活参不变）；
#       ②各 150-200 步真实语料短训（log/book2-ch08/tokens.bin sp-8k，306M token 流复用），
#       报 sec/step、MPS 内存峰值（逐步采样口径）、loss 可训性；
#       ③专家负载分布：router top-2 选择的逐专家计数表（随机初始化 vs 训练后——负载不均的现场素材，
#       ch3 训练难题的第一手数据）；
#       ④给「一个 ≤30 分钟 MoE 模块实验 = 多少步」定档建议（写作期实验预算的底数）。
# 变体实现：import llama_slots 组件（RMSNorm/GQA/RoPE 整机骨架不动），仅 Block 的 mlp 插槽
#       换 SparseMoE（moe_mla_slots.SparseMoE，教学版 ModuleList 专家）——本册第一刀的代码预演。
# 所属章节：Book4 ch2（MoE 基本机构）/ ch3（负载不均现场）；00-feasibility 探针 B。
# 运行方式：source env.sh && python 02_probe_b_moe.py [--out-name run1] [--steps 200] [--batch 8] [--block 512]
#   （dense 约 1-2 min、MoE 约 3-6 min——若投影超 25 min 自动降档并留痕）
# 产物：log/book4-feasibility/probeB_moe_{out}.json（不入库）
import argparse
import time

import numpy as np
import torch

from moe_mla_slots import (EVAL_SP8K, SEED, TOKENS_SP8K, TokenStream, bootstrap_book3,
                           hand_account, activated_account, pick_device, save_json)
from moe_mla_slots import SlotConfig, SlotLLaMA

LR = 1e-3                                            # 模块档学习率（26M 小模型，Book2 ch8 口径量级）
WARMUP = 20


def lr_at(step, total):
    return LR * min(1.0, step / WARMUP)


def eval_loss(model, device, batches=8):
    """heldout eval loss（eval.bin 独立文件，B=8/T=512，8 批 fp32 口径）。"""
    model.eval()
    arr = np.memmap(EVAL_SP8K, dtype=np.uint16, mode="r")
    offs = np.linspace(0, len(arr) - 8 * 512 - 1, batches).astype(int)
    losses = []
    with torch.no_grad():
        for o in offs:
            span = np.asarray(arr[o: o + 8 * 512 + 1]).astype(np.int64)
            x = torch.from_numpy(span[:-1].reshape(8, 512)).to(device)
            y = torch.from_numpy(span[1:].reshape(8, 512)).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "mps" \
                    else torch.autocast("cpu", enabled=False):
                _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def train_arm(tag, cfg, steps, batch, block, device):
    """一个训练臂：Book3 同配方（AdamW 0.9/0.95/wd0.1 + clip1.0 + warmup，MPS bf16 autocast）。"""
    torch.manual_seed(SEED)
    model = SlotLLaMA(cfg).to(device)
    hand = hand_account(cfg)
    assert model.n_params() == hand["total"], f"{tag} 参数账失败 {model.n_params()} != {hand['total']}"
    act, act_ne = activated_account(cfg)
    stream = TokenStream(TOKENS_SP8K, batch, block)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    print(f"[{tag}] 总参 {model.n_params():,} | 激活参 {act:,}（non-emb {act_ne:,}）| {steps} 步 × "
          f"{batch * block} tok = {steps * batch * block / 1e6:.2f}M token")

    hist0 = expert_hist(model, device)                # 随机初始化时的负载分布
    secs, losses, mem_trace = [], [], []
    t_wall = time.perf_counter()
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps)
        x, y = stream.batch(step, device)
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "mps" \
                else torch.autocast("cpu", enabled=False):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device.type == "mps":
            torch.mps.synchronize()                   # MPS 异步——显式同步后才是真实每步墙钟
        secs.append(time.perf_counter() - t0)
        losses.append(loss.item())
        if device.type == "mps":
            mem_trace.append(torch.mps.current_allocated_memory() / 1024 / 1024)
        if step == 4 and sum(secs) / 4 * steps > 1500:      # 25 min 守卫 → 降档留痕
            new_steps = max(60, int(1500 / (sum(secs) / 4)))
            print(f"[降档] {tag}: 单步 {sum(secs)/4:.2f}s 投影超 25 min → 减至 {new_steps} 步（如实记录）")
            steps = new_steps

    hist1 = expert_hist(model, device)                # 短训后的负载分布
    ev = eval_loss(model, device)
    steady = secs[1:]
    res = {"tag": tag, "params_total": model.n_params(), "params_active": act, "params_active_nonemb": act_ne,
           "steps_run": steps, "sec_per_step_steady": round(float(np.median(steady)), 3),
           "wall_sec": round(time.perf_counter() - t_wall, 1),
           "loss_curve_every20": [round(losses[i], 4) for i in range(0, steps, max(1, steps // 10))],
           "loss_first10_mean": round(float(np.mean(losses[:10])), 4),
           "loss_last10_mean": round(float(np.mean(losses[-10:])), 4),
           "eval_loss": round(ev, 4),
           "mps_peak_mb": round(max(mem_trace), 1) if mem_trace else None,
           "expert_hist_init": hist0, "expert_hist_trained": hist1}
    print(f"[{tag}] {res['sec_per_step_steady']} s/step | train loss {res['loss_first10_mean']}→"
          f"{res['loss_last10_mean']} | eval {ev:.4f} | 峰值 {res['mps_peak_mb']} MB | 墙钟 {res['wall_sec']} s")
    return res


@torch.no_grad()
def expert_hist(model, device, tokens=8192):
    """各层 router 的 top-2 专家选择直方图（喂 tokens 个真实 token，返回 L×E 计数表）。"""
    model.eval()
    stream = TokenStream(TOKENS_SP8K, 4, tokens // 4)
    x, _ = stream.batch(999, device)
    hists = []
    hooks = []

    def hook(mod, inp, out, hists=hists):
        hists.append(mod.expert_selection_counts(inp[0].detach()).tolist())

    for layer in model.model.layers:
        if hasattr(layer.mlp, "expert_selection_counts"):
            hooks.append(layer.mlp.register_forward_hook(hook))
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "mps" \
            else torch.autocast("cpu", enabled=False):
        model(x)
    for hk in hooks:
        hk.remove()
    model.train()
    return hists


def load_imbalance(hist):
    """负载不均度量：单层内 max/min 与 max/期望（1.0=完全均衡）。"""
    out = []
    for row in hist:
        arr = np.array(row, dtype=float)
        exp = arr.sum() / len(arr)
        out.append({"max_over_min": round(float(arr.max() / max(arr.min(), 1e-9)), 2),
                    "max_over_mean": round(float(arr.max() / exp), 2)})
    return out


def main(out_name, steps, batch, block):
    device = pick_device("mps")
    print(f"[设备] {device} | 语料 tokens.bin（V=8192，Book2 ch08 遗产）| 种子 {SEED}")
    common = dict(vocab_size=8192, hidden_size=512, num_hidden_layers=6,
                  num_attention_heads=8, num_key_value_heads=4, intermediate_size=1408)
    dense_cfg = SlotConfig(**common, moe=None)
    moe_cfg = SlotConfig(**common, moe=dict(n_expert=8, top_k=2, expert_dim=704, n_shared=0,
                                            variant="teach", mode="mixtral"))

    dense = train_arm("dense", dense_cfg, steps, batch, block, device)
    moe = train_arm("moe", moe_cfg, steps, batch, block, device)

    sec30 = 1800.0
    rec = {"seed": SEED, "device": str(device), "bt": f"B={batch}/T={block}",
           "config": {k: v for k, v in common.items()},
           "moe_spec": {"E": 8, "top_k": 2, "expert_dim": 704,
                        "等激活口径": "每 token 激活 2×704=1408 = dense d_ff；总参 3.75× dense FFN 槽"},
           "dense": dense, "moe": moe,
           "load_imbalance_init": load_imbalance(moe["expert_hist_init"]),
           "load_imbalance_trained": load_imbalance(moe["expert_hist_trained"]),
           "steps_per_30min": {"dense": int(sec30 / dense["sec_per_step_steady"]),
                               "moe": int(sec30 / moe["sec_per_step_steady"])}}
    save_json("probeB_moe", rec, out_name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="探针 B：mini MoE 模块档 vs dense 基线")
    ap.add_argument("--out-name", default="run1", help="产物 JSON 后缀（防覆写）")
    ap.add_argument("--steps", type=int, default=200, help="每臂训练步数（默认 200）")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--block", type=int, default=512)
    args = ap.parse_args()
    main(args.out_name, args.steps, args.batch, args.block)
