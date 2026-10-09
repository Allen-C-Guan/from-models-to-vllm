# quant_sim.py —— Book6 ch11 代码件：量化推理的误差账模拟（RTN/分组方向/对角缩放迁移）
# 用途：①RTN（round-to-nearest）逐权重量化：per-tensor/per-channel/per-token 三口径的误差对照
#   ——ch3 KIVI 例 3.1 的同构代码化（权重侧视角）；②对角缩放迁移的同一恒等式两方向
#   （SmoothQuant 式 W8A8 闭式 vs AWQ 式网格搜索）——误差搬家前后的激活/权重各自误差账；
#   ③显存账：2/4/8 bit 权重的字节与 decode 带宽下限变化。
# 所属章节：Book6 ch11；设计书=plan/Book6-推理系统导论.md ch11
# 运行：cd <workspace> && source env.sh && python code/Book6-推理系统导论/ch11/quant_sim.py [--out-name base]
# 产物：log/book6-ch11/quant_sim_<out-name>.json
import argparse
import json
import os

import torch


def rtn_quant(x, axis=None, bits=4, group=32):
    """RTN 量化-反量化 round-trip。axis=None per-tensor、0 per-channel（输出维）、1 per-token。"""
    qmax = 2 ** (bits - 1) - 1
    if axis is None:
        s = x.abs().max() / qmax
    else:
        s = x.abs().amax(dim=axis, keepdim=True) / qmax
    return torch.round(x / s.clamp_min(1e-12)) * s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="base")
    a = ap.parse_args()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "log", "book6-ch11")
    os.makedirs(out_dir, exist_ok=True)
    g = torch.Generator().manual_seed(20261002)

    # ① 三口径 RTN 误差（带 1 个离群通道的权重——同 ch3 例 3.1 的结构）
    W = torch.randn(256, 128, generator=g) * 0.02
    W[:, 3] = W[:, 3] * 80                        # 离群通道
    errs = {}
    for bits in (4, 2):
        for name, ax in (("per_tensor", None), ("per_channel", 0), ("per_token", 1)):
            xq = rtn_quant(W, axis=ax, bits=bits)
            errs[f"{name}_b{bits}"] = round(float((xq - W).abs().mean() / W.abs().mean()), 4)
    # ② 对角缩放迁移（同一恒等式两方向）：Y=(X·diag(s)^-1)(diag(s)·W)
    X = torch.randn(64, 128, generator=g)
    X[:, 3] = X[:, 3] * 40                        # 激活也有离群通道（SmoothQuant 的观察）
    s = X.abs().amax(dim=0).pow(0.5) / W.abs().amax(dim=0).pow(0.5)   # α=0.5 的闭式（难度对半分）
    Xs, Ws = X / s, W * s                         # 迁移后：激活变温和、权重变尖
    scale_demo = dict(
        identity_check=round(float(((X / s) @ (W * s).T - X @ W.T).abs().max()), 8),
        act_max_before=round(float(X.abs().max()), 2), act_max_after=round(float(Xs.abs().max()), 2),
        w_max_before=round(float(W.abs().max()), 4), w_max_after=round(float(Ws.abs().max()), 4),
        note="恒等式保证乘积不变；最大值账：激活的尖搬给了权重（α 控制搬家比例）")
    # 迁移前后的量化误差对照（迁移后 per-tensor 量化激活侧更准、权重侧更差——两方向的取舍）
    err_act_before = float((rtn_quant(X, bits=8) - X).abs().mean())
    err_act_after = float((rtn_quant(Xs, bits=8) - Xs).abs().mean())
    scale_demo["act_quant_err_before_after"] = [round(err_act_before, 5), round(err_act_after, 5)]

    # ③ 显存与带宽账（207M 口径）
    N = 207_119_360
    mem = {f"w{b}": dict(MB=round(N * b / 8 / 1e6), decode_floor_ms=round(N * b / 8 / 1.19e12 * 1e3, 3))
           for b in (16, 8, 4)}
    res = dict(rtn_errors=errs, scale_transfer=scale_demo, memory_bandwidth=mem,
               note="误差相对口径：|Δ|均值/|x|均值；RTN 无校准集（对 GPTQ 式校准法的对照见正文）")
    p = os.path.join(out_dir, f"quant_sim_{a.out_name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1)[:900])
    print(f"[产物] {p}")


if __name__ == "__main__":
    main()
