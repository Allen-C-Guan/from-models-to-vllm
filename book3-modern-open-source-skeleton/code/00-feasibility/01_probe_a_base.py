# 01_probe_a_base.py —— 探针 A：Book2 改装底座可用性验证
# 用途：sys.path 引入 Book2 ch10 的 model.py（GPT-2 三插槽 Block，本册改装契约的锚点），
#       构造模块档小 config（d=512/L=6/h=8/V=8192 ≈ 23.6M），前向+反向 10 步，
#       CPU 与 MPS 双报告 sec/step 与 MPS 内存峰值——确认「底座可直接 import 复用」并给模块实验的基线速度。
# 所属章节：Book3 00-feasibility（为 ch2-ch6 模块实验与 ch10 整合提供基线底数）。
# 运行方式：cd code/00-feasibility && python 01_probe_a_base.py [--out-name NAME]
#   产物：log/book3-feasibility/{out-name}.json（不 commit）
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                                   # llama_slots（本探针暂不用，路径先备好）
for _base in (os.path.join(HERE, "..", "..", "Book2-预训练革命", "ch10"),                              # 工作区布局
              os.path.join(HERE, "..", "..", "..", "book2-pretraining-revolution", "code", "ch10")):  # 书仓快照布局
    if os.path.isdir(_base):
        sys.path.insert(0, os.path.abspath(_base))  # Book2 ch10 GPT-2 定版底座
from model import GPT2, GPT2Config                                          # noqa: E402  Book2 ch10 定版底座

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book3-feasibility")
SEED = 20261002                                                            # 全书统一种子


def sync(device):
    """计时前同步（MPS 是异步队列，不同步会量到假速度）。"""
    if device.type == "mps":
        torch.mps.synchronize()


def mps_alloc_mb():
    """MPS 统一内存口径：torch.mps.current_allocated_memory（探针统一内存口径）。"""
    return torch.mps.current_allocated_memory() / 2**20 if torch.backends.mps.is_available() else None


def run_10steps(device, model, x, y, steps=10):
    """同一训练步（fwd+bwd+AdamW.step）重复 steps 次，返回 (逐步耗时, 内存峰值增量, 末步 loss)。"""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    x, y = x.to(device), y.to(device)
    base = mps_alloc_mb() or 0.0
    peak = base
    times = []
    loss = None
    for _ in range(steps):
        t0 = time.perf_counter()
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sync(device)
        times.append(time.perf_counter() - t0)
        peak = max(peak, mps_alloc_mb() or 0.0)
    return times, peak - base, float(loss.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="probe_a_base", help="产物文件名（防覆写）")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    cfg = GPT2Config(vocab_size=8192, n_positions=1024, n_layer=6, n_embd=512, n_head=8)
    model = GPT2(cfg)
    n_par = model.n_params()
    print(f"[config] d=512/L=6/h=8/V=8192  参数 = {n_par:,}（模块档 20-30M）")

    x = torch.randint(0, 8192, (16, 512))   # (B=16, n=512)
    y = torch.randint(0, 8192, (16, 512))

    result = {"seed": SEED, "config": {"n_embd": 512, "n_layer": 6, "n_head": 8, "vocab": 8192,
                                       "batch": 16, "block": 512, "params": n_par}, "devices": {}}
    for dev_name in (["mps", "cpu"] if torch.backends.mps.is_available() else ["cpu"]):
        torch.manual_seed(SEED)                              # 每设备同种子重建（权重与数据序列一致）
        device = torch.device(dev_name)
        m = GPT2(cfg).to(device)
        with torch.no_grad():                                # 初始 loss 合理性检查：应 ≈ ln(8192) = 9.01
            _, l0 = m(x.to(device), y.to(device))
        times, peak_mb, last_loss = run_10steps(device, m, x, y)
        steady = times[1:]                                   # 丢首步（MPS 首步含算子编译/缓存预热）
        result["devices"][dev_name] = {
            "initial_loss": round(l0.item(), 4),
            "sec_step_all": [round(t, 4) for t in times],
            "sec_step_steady_mean": round(sum(steady) / len(steady), 4),
            "mem_peak_delta_mb": round(peak_mb, 1) if peak_mb else None,
            "final_loss_on_fixed_batch": round(last_loss, 4),
        }
        print(f"[{dev_name}] 初始 loss={l0.item():.3f}（ln8192=9.011）  稳态 sec/step="
              f"{result['devices'][dev_name]['sec_step_steady_mean']:.4f}  峰值内存增量={peak_mb:.0f} MB  "
              f"10步后(固定批)={last_loss:.3f}")
        del m
        torch.mps.empty_cache() if dev_name == "mps" else None

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{args.out_name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"产物已写：{out}")


if __name__ == "__main__":
    main()
