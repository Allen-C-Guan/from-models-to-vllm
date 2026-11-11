# memory_probe.py —— Book2第9章：GPT-2 124M理论内存账与MPS测量。
# 工作区根目录激活环境后运行：python code/ch09/memory_probe.py --theory-only
# MPS测量去掉--theory-only；默认6步、无预热，含首次状态分配；已有结果不覆盖。
# 三配置各用独立进程；current为张量占用，driver还含缓存及驱动相关分配。
# 当前代码新增有限性检查，且复用修订后的ch04初始化，不承诺复现历史步时和峰值。
import argparse
import json
import math
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch04"))
from warmup_ablation import GPT

SEED = 20261002
OUT = Path.cwd() / "log/book2-ch09"
CONFIGS = ("fp32", "bf16", "ckpt_fp32")


def theory(batch=4, seq=1024):
    """固定上下文表1024；只建meta张量计数，不分配124M实体参数。返回GiB估算。"""
    if batch <= 0 or not 1 <= seq <= 1024:
        raise ValueError("batch必须正数，seq须为1—1024")
    with torch.device("meta"):
        model = GPT(n_layer=12, n_embd=768, n_head=12, vocab=50257,
                    block_size=1024, ln_position="pre")
    psi = sum(p.numel() for p in model.parameters())
    return {
        "params_total": psi, "params_non_emb": model.n_params(non_embedding=True),
        "state_gib": 16 * psi / 1024**3,
        "act_linear_gib": 12 * 4 * batch * seq * 768 * 12 / 1024**3,
        "act_attn_gib": 4 * batch * 12 * seq**2 * 12 / 1024**3,
        "logits_gib": 2 * 4 * batch * seq * 50257 / 1024**3,
        "note": "fp32教学保留集，不是各张量生命周期的精确峰值；临时工作区另计",
    }


def peak_sampler(fn, interval=0.02, query=None, synchronize=None):
    """执行fn并轮询(current,driver)字节值；异常也结束线程，不吞采样错误。

    首尾同步采样，期间不强制同步；短峰值仍可能漏采。两指标各取最大，不相加。
    query/synchronize用于局部测试注入；正式测量默认调用torch.mps。
    """
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("interval必须为有限正数")
    query = query or (lambda: (torch.mps.current_allocated_memory(),
                              torch.mps.driver_allocated_memory()))
    synchronize = synchronize or torch.mps.synchronize
    stop, errors = threading.Event(), []
    peaks, samples = [0, 0], [0]
    def sample():
        values = query()
        if len(values) != 2 or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("无效内存读数")
        for i, value in enumerate(values):
            peaks[i] = max(peaks[i], value)
        samples[0] += 1
    def poll():
        try:
            while not stop.wait(interval):
                sample()
        except BaseException as exc:
            errors.append(exc)
            stop.set()
    synchronize()
    sample()
    thread = threading.Thread(target=poll, name="book2-memory-sampler", daemon=True)
    t0 = time.perf_counter()
    thread.start()
    try:
        value = fn()
        synchronize()
    finally:
        stop.set()
        thread.join()
    sample()  # 已join，不与后台线程并发更新峰值
    elapsed = time.perf_counter() - t0
    if errors:
        raise RuntimeError("内存采样失败，拒绝生成有效测量") from errors[0]
    return value, elapsed, {
        "peak_current_mib": peaks[0] / 1024**2,
        "peak_driver_mib": peaks[1] / 1024**2,
        "samples": samples[0], "interval_s": interval,
    }


def check_finite_before_step(loss, parameters):
    """更新前检查标量loss及所有梯度；检查产生的同步/临时量属于本探针开销。"""
    if not bool(torch.isfinite(loss.detach()).all()):
        raise FloatingPointError("loss非有限，未执行优化器更新")
    maxima = [p.grad.detach().abs().amax() for p in parameters if p.grad is not None]
    if maxima and not bool(torch.isfinite(torch.stack(maxima)).all()):
        raise FloatingPointError("gradient非有限，未执行优化器更新")


def bench(B=4, n=1024, steps=6, autocast=False, ckpt=False,
          warmup_steps=0, interval=0.02):
    """(B,n)输入/右移标签→标量loss；仅MPS子进程运行，默认无预热。"""
    if B <= 0 or not 1 <= n <= 1024 or steps <= 0 or warmup_steps < 0:
        raise ValueError("批量/步数须正数，预热非负，序列长为1—1024")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS不可用；可用--theory-only检查理论账")
    torch.mps.empty_cache()
    torch.manual_seed(SEED)
    model = GPT(n_layer=12, n_embd=768, n_head=12, vocab=50257,
                block_size=1024, ln_position="pre").to("mps").train()
    if ckpt:
        for block in model.blocks:
            forward = block.forward
            block.forward = lambda x, f=forward: checkpoint(f, x, use_reentrant=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
    ids = torch.randint(0, 50257, (B, n + 1), device="mps")
    xs, ys = ids[:, :-1].contiguous(), ids[:, 1:].contiguous()  # (B,n)，标签右移一位
    def run(count):
        loss = None
        for _ in range(count):
            opt.zero_grad(set_to_none=True)
            ctx = torch.autocast(device_type="mps", dtype=torch.bfloat16) if autocast else nullcontext()
            with ctx:
                _, loss = model(xs, ys)  # (B,n,V) logits与标量loss
            # 先查loss，避免已知非有限值进入反向；再查梯度，防止坏更新。
            if not bool(torch.isfinite(loss.detach()).all()):
                raise FloatingPointError("loss非有限，未反向或更新")
            loss.backward()
            check_finite_before_step(loss, model.parameters())
            opt.step()
        return loss.item()
    if warmup_steps:
        run(warmup_steps)
    loss, seconds, memory = peak_sampler(lambda: run(steps), interval=interval)
    return {"sec_per_step": seconds / steps, "loss": loss, **memory}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--out-name", default="memory_probe.json")
    p.add_argument("--theory-only", action="store_true")
    p.add_argument("--single", choices=CONFIGS, help=argparse.SUPPRESS)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--interval", type=float, default=0.02)
    return p


def main():
    args = parser().parse_args()
    if (args.steps <= 0 or args.warmup_steps < 0
            or not math.isfinite(args.interval) or args.interval <= 0):
        raise ValueError("步数/采样间隔须正，预热须非负")
    ledger = theory(args.batch, args.seq)
    if args.single:
        kw = {"autocast": args.single == "bf16", "ckpt": args.single == "ckpt_fp32"}
        r = bench(args.batch, args.seq, args.steps, warmup_steps=args.warmup_steps,
                  interval=args.interval, **kw)
        print("RESULT " + json.dumps({"cfg": args.single, **r}, allow_nan=False))
        return
    if Path(args.out_name).name != args.out_name or args.out_name in (".", "..", ""):
        raise ValueError("out-name只接受单个文件名")
    target = args.out_dir / args.out_name
    if target.exists():
        raise FileExistsError(f"已有产物不覆盖：{target}")
    report = {
        "schema_version": 2, "kind": "theory" if args.theory_only else "measurement",
        "seed": SEED, "device": None if args.theory_only else "mps",
        "torch_version": str(torch.__version__), "batch": args.batch, "seq": args.seq,
        "context_table": 1024, "steps": args.steps, "warmup_steps": args.warmup_steps,
        "theory": ledger,
        "measurement_note": "每配置独立进程；计时含有限性检查；无预热时含首次状态分配。"
                            "current不含未用缓存，driver范围更广；轮询不保证捕获短峰值。",
    }
    if not args.theory_only:
        if not torch.backends.mps.is_available():
            raise RuntimeError("本机无MPS；未生成实测报告，请用--theory-only")
        for cfg in CONFIGS:
            command = [sys.executable, str(Path(__file__).resolve()), "--single", cfg,
                       "--batch", str(args.batch), "--seq", str(args.seq),
                       "--steps", str(args.steps), "--warmup-steps", str(args.warmup_steps),
                       "--interval", str(args.interval)]
            child = subprocess.run(command, capture_output=True, text=True)
            if child.returncode:
                raise RuntimeError(f"{cfg}测量失败：\n{child.stderr}")
            rows = [line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT ")]
            if len(rows) != 1:
                raise RuntimeError(f"{cfg}未返回唯一测量")
            value = json.loads(rows[0])
            if value.pop("cfg") != cfg:
                raise RuntimeError("测量配置不匹配")
            report[cfg] = value
    # 先严格序列化；最终x模式防止两次运行互相覆盖。
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with target.open("x") as f:
        f.write(payload)
    print(payload)
    print(f"[产物] {target}")


if __name__ == "__main__":
    main()
