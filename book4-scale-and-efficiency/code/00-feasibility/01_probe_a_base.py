# 01_probe_a_base.py —— Book4 探针 A：Book3 207M 底座可用性与基线 sec/step
# 用途：本册两刀改装（FFN→MoE、GQA→MLA）的底座是 Book3 的 207M LLaMA 式骨架（ch10 llama215 /
#       00-feasibility llama_slots）。本探针验证底座在本机仍可复现：①跨册双候选 bootstrap 引入
#       llama_slots 并自建同 config 整机（LLAMA_215M）；②前向+反向 10 步（MPS bf16 autocast 与
#       CPU fp32 各报）——底座可用性与基线 sec/step，对照 Book3 实测口径（B=8/T=1024：
#       MPS bf16 1.702 s/step、峰值 7,118 MB；CPU fp32 10.400 s/step）；
#       ③顺带采一份真实残差流激活样本（中间层输出）存 npz，供探针 E 的 FP8 量化误差实验做
#       「真实激活分布」输入（随机高斯 vs 真实分布的量化误差对照）。
# 所属章节：Book4 ch0（底座契约）与 ch1（效率悖论账本的基线数字）；00-feasibility 探针 A。
# 运行方式：source env.sh && python 01_probe_a_base.py [--out-name run1] [--cpu-steps 10]
#   （MPS 10 步约 30 s；CPU fp32 10 步约 2 分钟——若单步 >90 s 自动降档并如实记录）
# 产物：log/book4-feasibility/probeA_base_{out}.json + act_sample.npz（不入库）
import argparse
import os
import time

import numpy as np
import torch

from moe_mla_slots import (OUT_DIR, SEED, TOKENS_SP32K, TokenStream, bootstrap_book3,
                           pick_device, save_json)


def run_arm(model, device, steps, batch, block, use_autocast, tag):
    """一个训练臂：AdamW(0.9,0.95,wd0.1)+clip1.0（Book3 train_215 同配方），返回逐步耗时与 loss。"""
    torch.manual_seed(SEED)
    stream = TokenStream(TOKENS_SP32K, batch, block)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
    secs, losses, mem_trace = [], [], []
    for step in range(1, steps + 1):
        x, y = stream.batch(step, device)
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16) \
                if (use_autocast and device.type == "mps") else torch.autocast("cpu", enabled=False):
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
            # torch.mps 无 reset_peak API——峰值口径 = 每步后采样 current_allocated_memory 取 max
            mem_trace.append(torch.mps.current_allocated_memory() / 1024 / 1024)
        # CPU 臂降档守卫：单步 >90 s 且投影总时长 >15 min → 减半步数（降档留痕）
        if device.type == "cpu" and step == 2 and sum(secs) / 2 * steps > 900:
            new_steps = max(5, int(900 / (sum(secs) / 2)))
            print(f"[降档] {tag}: CPU 单步 {sum(secs)/2:.1f}s，投影 {steps} 步超 15 min → 减至 {new_steps} 步")
            steps = new_steps
    peak = max(mem_trace) if mem_trace else None
    steady = secs[1:]                                  # 丢首步（惰性编译/缓存预热）
    return {"device": str(device), "dtype": "bf16-autocast" if use_autocast else "fp32",
            "steps": steps, "batch": batch, "block": block,
            "sec_per_step_steady": round(float(np.median(steady)), 3),
            "sec_per_step_mean": round(float(np.mean(steady)), 3),
            "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4),
            "mps_peak_mb": round(peak, 1) if peak else None}


def capture_activation_sample(model):
    """中间层（第 6 层输出）残差流样本：真实激活分布（供探针 E）。"""
    torch.manual_seed(SEED)
    model = model.to("cpu").eval()
    grabbed = {}
    model.model.layers[6].register_forward_hook(lambda m, i, o: grabbed.__setitem__("h", o.detach()))
    with torch.no_grad():
        g = torch.Generator().manual_seed(SEED)
        x = torch.randint(0, model.cfg.vocab_size, (2, 256), generator=g)
        model(x)
    h = grabbed["h"].float().numpy()                  # (2,256,1024)
    return h


def main(out_name, cpu_steps):
    slots = bootstrap_book3()
    print(f"[bootstrap] llama_slots 自：{os.path.dirname(slots.__file__)}")
    torch.manual_seed(SEED)
    model = slots.LLaMA(slots.LLaMAConfig(**slots.LLAMA_215M))     # 207M 定版（Book3 ch10 同 config）
    total = model.n_params()
    assert total == 207_119_360, f"底座参数账失败：{total}"
    print(f"[底座] LLaMA 207,119,360 参数逐位一致（llama_slots.LLAMA_215M）")

    report = {"seed": SEED, "base_params": total, "corpus": "log/book3-ch10/tokens32k.bin (V=32000)",
              "baseline_book3": {"mps_bf16_sec_per_step": 1.702, "mps_peak_mb": 7118,
                                 "cpu_fp32_sec_per_step": 10.400, "bt": "B=8/T=1024"}}

    # ① MPS bf16（autocast，fp32 权重——Book3 同口径）
    t0 = time.perf_counter()
    mps = run_arm(slots.LLaMA(slots.LLaMAConfig(**slots.LLAMA_215M)), pick_device("mps"),
                  10, 8, 1024, use_autocast=True, tag="mps")
    mps["wall_sec"] = round(time.perf_counter() - t0, 1)
    print(f"[MPS bf16] {mps['sec_per_step_steady']} s/step（稳态中位）| loss {mps['loss_first']}→"
          f"{mps['loss_last']} | 峰值 {mps['mps_peak_mb']} MB | 总墙钟 {mps['wall_sec']} s")
    report["mps_bf16"] = mps

    # ② CPU fp32
    t0 = time.perf_counter()
    cpu = run_arm(slots.LLaMA(slots.LLaMAConfig(**slots.LLAMA_215M)), torch.device("cpu"),
                  cpu_steps, 8, 1024, use_autocast=False, tag="cpu")
    cpu["wall_sec"] = round(time.perf_counter() - t0, 1)
    print(f"[CPU fp32] {cpu['sec_per_step_steady']} s/step | loss {cpu['loss_first']}→{cpu['loss_last']} "
          f"| 总墙钟 {cpu['wall_sec']} s")
    report["cpu_fp32"] = cpu

    # ③ 真实激活分布样本（中间层残差流）
    h = capture_activation_sample(slots.LLaMA(slots.LLaMAConfig(**slots.LLAMA_215M)))
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, "act_sample.npz"), resid_layer6=h,
                        stats=np.array([h.mean(), h.std(), *np.percentile(h, [1, 50, 99])]))
    print(f"[激活样本] resid_layer6 形状 {h.shape} | mean {h.mean():.4f} std {h.std():.4f} "
          f"| p1/p50/p99 {np.percentile(h, [1, 50, 99]).round(4).tolist()} -> act_sample.npz")

    report["activation_sample"] = {"shape": list(h.shape), "mean": float(h.mean()), "std": float(h.std()),
                                   "p1_p50_p99": np.percentile(h, [1, 50, 99]).round(4).tolist()}
    report["verdict"] = ("底座可用" if abs(mps["sec_per_step_steady"] - 1.702) < 1.0 else "可用但偏离口径，需查")
    save_json("probeA_base", report, out_name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="探针 A：Book3 207M 底座验证与基线 sec/step")
    ap.add_argument("--out-name", default="run1", help="产物 JSON 后缀（防覆写）")
    ap.add_argument("--cpu-steps", type=int, default=10, help="CPU 臂步数（默认 10，自动降档）")
    args = ap.parse_args()
    main(args.out_name, args.cpu_steps)
