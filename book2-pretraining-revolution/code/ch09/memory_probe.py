# memory_probe.py —— Book2 ch9 §9.2/§9.4/§9.5 显存账本探针：124M 模型状态理论账 vs 实测峰值；bf16 / 梯度检查点开关对照
# 所属章节：Book2 第 9 章（原理章）。模型工厂复用 ch04/warmup_ablation.GPT（单轨依赖链 ch4→ch8→ch9）；124M 全参数 124,439,808。
# 运行方式：source env.sh && python "code/ch09/memory_probe.py"   （分钟级，MPS；产物落 log/book2-ch09/memory_probe.json）
#   可选 --out-name NAME 写独立产物文件（防覆写纪律：正式定版产物建议另存一份不可变副本，如 memory_probe_final.json）。
# 批三修订 2026-10-03：三配置改为**独立子进程**逐个实跑——同进程连跑时 MPS 分配器池跨配置残留，曾把 bf16 峰值抬到
#   fp32 之上（池残留伪象）；每配置一个干净进程 + 跑前 empty_cache，峰值才可比。主进程只做理论账与汇总。

import json
import os
import subprocess
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ch04"))
from warmup_ablation import GPT  # noqa: E402  124M 工厂：n_layer=12/n_embd=768/n_head=12/vocab=50257/block=1024

SEED = 20261002  # 全书统一种子
OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
                   "log", "book2-ch09")


def peak_sampler(fn, interval=0.02):
    """跑 fn() 期间后台线程轮询 torch.mps.current_allocated_memory() 取峰值（全书对拍规范口径）。"""
    import torch.mps as mps
    peak, stop = [0.0], threading.Event()

    def poll():
        while not stop.is_set():
            try:
                peak[0] = max(peak[0], mps.current_allocated_memory() / 1024**2)
            except Exception:
                pass
            stop.wait(interval)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    t0, out = time.perf_counter(), fn()
    torch.mps.synchronize()
    return out, round((time.perf_counter() - t0), 2), round(peak[0], 1), stop


def bench(B=4, n=1024, steps=6, autocast=False, ckpt=False):
    """完整训练步（前向+反向+AdamW step）×steps：返回 sec/step 与峰值 MiB。x/y 均为 (B,n) 随机 id。
    本函数只在 --single 子进程模式下调用：每配置一个干净进程，峰值不含上一配置的池残留。"""
    torch.mps.empty_cache()                                 # 跑前清池（子进程本就干净，双保险）
    torch.manual_seed(SEED)
    model = GPT(n_layer=12, n_embd=768, n_head=12, vocab=50257, block_size=n,
                ln_position="pre").to("mps")
    if ckpt:  # 梯度检查点：只保留段边界激活（每块一段），段内反传时重算
        for blk in model.blocks:
            f = blk.forward
            blk.forward = lambda x, f=f: torch.utils.checkpoint.checkpoint(f, x, use_reentrant=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
    idx = torch.randint(0, 50257, (B, n + 1), device="mps")
    xs, ys = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()  # (B,n) 输入 / (B,n) 右移标签

    def run():
        loss = None
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            if autocast:
                with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                    _, loss = model(xs, ys)           # logits (B,n,50257) -> 标量 loss
            else:
                _, loss = model(xs, ys)
            loss.backward()
            opt.step()
        return round(loss.item(), 3)

    (loss, sec, peak, stop) = peak_sampler(run)
    stop.set()
    del model, opt
    torch.mps.empty_cache()
    return {"sec_per_step": round(sec / steps, 4), "peak_mib": peak, "loss": loss}


def single(cfg):
    """子进程入口：跑单配置，结果打到 stdout（一行 JSON）供主进程汇总。"""
    kw = {"fp32": {}, "bf16": {"autocast": True}, "ckpt_fp32": {"ckpt": True}}[cfg]
    r = bench(**kw)
    print("RESULT " + json.dumps({"cfg": cfg, **r}))


def main():
    os.makedirs(OUT, exist_ok=True)
    args = sys.argv[1:]
    out_name = "memory_probe.json"
    if "--out-name" in args:
        out_name = args[args.index("--out-name") + 1]
    if "--single" in args:                                  # 子进程模式
        single(args[args.index("--single") + 1])
        return

    torch.manual_seed(SEED)                                 # 理论账只在 CPU 上数参数，不占 MPS 池
    m = GPT(n_layer=12, n_embd=768, n_head=12, vocab=50257, block_size=1024, ln_position="pre")
    psi, psi_ne = sum(p.numel() for p in m.parameters()), m.n_params(non_embedding=True)
    del m  # 参数计数用（124,439,808 / 非嵌入 85,056,000）
    B, n, d, h, L, V = 4, 1024, 768, 12, 12, 50257  # 探针口径：b4×1024，fp32 记账
    results = {}
    for cfg in ("fp32", "bf16", "ckpt_fp32"):               # 每配置独立子进程（干净 MPS 分配器）
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--single", cfg],
                           capture_output=True, text=True, check=True)
        line = next(l for l in p.stdout.splitlines() if l.startswith("RESULT "))
        r = json.loads(line[len("RESULT "):])
        results[r.pop("cfg")] = r
        print(f"[{cfg}] {r}")
    fp32, bf16, ckpt = results["fp32"], results["bf16"], results["ckpt_fp32"]
    report = {
        "seed": SEED, "device": "mps", "batch": B, "seq": n, "steps": 6,
        "timing_note": "共享机器非独占基准；sec/step 仅作同批配置间对比口径；"
                       "三配置各在独立子进程实跑（防 MPS 池跨配置残留，批三修订 2026-10-03）",
        "theory": {  # 记账口径（保留集）：模型状态 + 激活两式 + 输出侧宽矩阵
            "params_total": psi, "params_non_emb": psi_ne,
            "state_gib": round(16 * psi / 1024**3, 3),                    # 16B/参数（两口径同 16）
            "act_linear_gib": round(12 * B * n * d * L * 4 / 1024**3, 3),  # ≈12 份 (B,n,d)/层（ZeRO 规则）
            "act_attn_gib": round(B * h * n * n * L * 4 / 1024**3, 3),     # 注意力分数 (B,h,n,n)/层
            "logits_gib": round(2 * B * n * V * 4 / 1024**3, 3),           # logits + log_softmax 两份 (B,n,V)
        },
        "fp32": fp32,
        "bf16": {**bf16, "speedup_vs_fp32": round(fp32["sec_per_step"] / bf16["sec_per_step"], 2)},
        "ckpt_fp32": {**ckpt,
                      "time_overhead_pct": round((ckpt["sec_per_step"] / fp32["sec_per_step"] - 1) * 100, 1),
                      "peak_delta_gib": round((ckpt["peak_mib"] - fp32["peak_mib"]) / 1024, 2)},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    with open(os.path.join(OUT, out_name), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[产物] {os.path.join(OUT, out_name)}")


if __name__ == "__main__":
    main()
