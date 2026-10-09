# train_k3toy.py —— Book5 ch6 正式件：K3 玩具短训 CLI（门粒度双臂 + KDA 门控分布采集 + ckpt）
# 用途：ch6 6.7「数百步短训」——3:1 混合可训性 + KDA 遗忘门分布实测 + GDN/KDA 门粒度双臂：
#   --arm kda  → KimiDeltaAttention gate_mode="per_channel"（K3 正身：每头每步 d_k 维通道遗忘门）
#   --arm gdn  → 同一 KimiDeltaAttention gate_mode="per_token"（GDN 粒度替身臂：z 沿通道取均值
#                后广播——每头每步一个标量；**单变量**：与 kda 臂同结构同参数（44,277,936 逐位
#                相同），只有门的读出粒度不同——判据预注册见 notes/07 §3）
#   其余：--steps/--out-name/--batch/--block/--mxfp4（路由专家权重 STE 量化钩子）/--ckpt-dir。
# 口径：MPS bf16 autocast（fp32 权重）；AdamW(0.9,0.95,wd0.1)+clip1.0+warmup（Book3 train_215
#   同配方——m4.train_loop 复用）；种子 20261002；loss 曲线/sec-per-step p50/峰值内存全量落 JSON。
# 运行方式：cd 工作区根目录 && source env.sh &&
#   python "code/ch06/train_k3toy.py" --arm kda --steps 300 --out-name fast
# 产物：log/book5-ch06/train_k3toy_{arm}_{out}.json（+ 可选 ckpt）——不入库
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

sys.path.insert(0, HERE)
import importlib.util


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


k3_toy = _load(os.path.join(HERE, "k3_toy.py"), "k3_toy_mod")
m4 = k3_toy.bootstrap()[0]


def main():
    ap = argparse.ArgumentParser(description="Book5 ch6 K3 玩具短训（门粒度双臂）")
    ap.add_argument("--arm", choices=["kda", "gdn"], required=True,
                    help="kda=per-channel 门（K3 正身）；gdn=per-token 门（GDN 粒度替身——单变量臂）")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--peak-lr", type=float, default=1e-3)
    ap.add_argument("--out-name", default="fast")
    ap.add_argument("--mxfp4", action="store_true", help="路由专家权重 MXFP4 STE 钩子")
    ap.add_argument("--ckpt-dir", default=None, help="给路径则存 state_dict ckpt（续跑/复用）")
    ap.add_argument("--resume", default=None, help="从 ckpt 恢复权重（继续训练不恢复优化器）")
    ap.add_argument("--seed", type=int, default=20261002)
    args = ap.parse_args()

    gate_mode = "per_channel" if args.arm == "kda" else "per_token"
    device = m4.pick_device("mps")
    cfg = k3_toy.toy_a_cfg(kda_impl="kda", gate_mode=gate_mode)
    torch.manual_seed(args.seed)
    model = k3_toy.K3Toy(cfg)
    total = model.n_params()
    assert total == k3_toy.TOY_A_ASSERTS["kda"]["total"], \
        f"参数账 {total} != {k3_toy.TOY_A_ASSERTS['kda']['total']}（两臂必须同参——单变量）"
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
        print(f"[续跑] 自 {args.resume} 恢复权重")
    model = model.to(device)
    n_swapped = k3_toy.apply_mxfp4_to_experts(model) if args.mxfp4 else 0
    if args.mxfp4:
        assert model.n_params() == total, "量化钩子不得改参数账"

    print(f"[臂 {args.arm}] gate_mode={gate_mode} | 总参 {total:,} | MXFP4 换 {n_swapped} Linear | "
          f"{args.steps} 步 @ {device}（batch {args.batch} × block {args.block}）", flush=True)
    m4.mps_peak_mb(reset=True)
    secs, losses, _ = m4.train_loop(model, steps=args.steps, batch=args.batch, block=args.block,
                                    device=device, tokens_path=m4.TOKENS_SP8K,
                                    peak_lr=args.peak_lr, warmup=max(5, args.steps // 15),
                                    log_every=max(1, args.steps // 10))
    peak_mb = m4.mps_peak_mb()

    # KDA 遗忘门分布（两臂都采：kda 臂 per-channel 直方图；gdn 臂每头标量门在同一接口下也可读）
    x_probe, _ = m4.TokenStream(m4.TOKENS_SP8K, args.batch, args.block).batch(1, device)
    g_stats = model.collect_g_stats(x_probe)

    out = {
        "arm": args.arm, "gate_mode": gate_mode, "seed": args.seed,
        "steps": args.steps, "batch": args.batch, "block": args.block,
        "peak_lr": args.peak_lr, "mxfp4_swapped": n_swapped,
        "total_params": total,
        "sec_per_step_p50": float(np.median(secs)), "sec_per_step_mean": float(np.mean(secs)),
        "loss_first": losses[0], "loss_last": losses[-1],
        "loss_min": min(losses), "losses": losses,
        "mps_peak_mb": peak_mb,
        "g_stats": g_stats,
    }
    if args.ckpt_dir:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        ck = os.path.join(args.ckpt_dir, f"k3toy_{args.arm}_{args.out_name}.pt")
        torch.save(model.state_dict(), ck)
        out["ckpt"] = ck
        print(f"[ckpt] {ck}")
    path = k3_toy.save_json(f"train_k3toy_{args.arm}", out, args.out_name)
    print(f"[完成] {args.arm} 臂：loss {losses[0]:.3f}→{losses[-1]:.3f}（min {min(losses):.3f}）| "
          f"{np.median(secs):.3f} s/step | 产物 {path}", flush=True)
    return out


if __name__ == "__main__":
    main()
