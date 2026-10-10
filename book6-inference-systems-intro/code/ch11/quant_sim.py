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

    # ③ GPTQ 式逐列量化+二阶补偿（toy Hessian）：与同口径 RTN 在**输出空间**对账
    #    逐层重建目标 min ||(W'-W)X||_F：H = 2XX^T + 阻尼；逐列量化后按 H^{-1} 的列摊派误差给余列
    #    （GPTQ §4 Algorithm 1 的单块版——固定顺序替代贪心、全行共用一个 H）
    Xc = torch.randn(128, 96, generator=g) * 1.0        # 校准集：96 条伪样本的层输入
    Xc[:, 3] = Xc[:, 3] * 8                              # 顺带带一个相关结构（离群输入通道）
    H = 2 * Xc @ Xc.T
    damp = 0.01 * H.diag().mean()
    H += damp * torch.eye(128)
    Hinv = torch.linalg.inv(H)
    W0 = W.clone()                                       # 与 ① 同一只 W（带离群通道）
    qmax = 2 ** 3                                        # 4bit 有符号档（15 档）用于拉开差距

    def col_quant(w):                                    # per-输入通道（逐列）RTN
        sc = w.abs().amax(dim=0, keepdim=True) / qmax
        return torch.round(w / sc.clamp_min(1e-12)) * sc

    Q_rtn = col_quant(W0.clone())                        # 臂一：RTN（逐列、无补偿）
    Wg = W0.clone(); Qg = torch.zeros_like(Wg)
    for j in range(Wg.shape[1]):                         # 臂二：GPTQ 单块版
        Qg[:, j] = col_quant(Wg[:, j:j+1]).squeeze(1)
        e = (Wg[:, j] - Qg[:, j]) / Hinv[j, j]           # 该列误差（Hinv 对角归一）
        if j + 1 < Wg.shape[1]:                          # 摊派给余列（块内更新；128 列单块跑全量）
            Wg[:, j+1:] -= e.unsqueeze(1) * Hinv[j, j+1:].unsqueeze(0)
    def rel_out_err(Q):
        return float(((Q - W0) @ Xc).norm() / (W0 @ Xc).norm())
    gptq_demo = dict(
        rtn_out_err=round(rel_out_err(Q_rtn), 4),
        gptq_out_err=round(rel_out_err(Qg), 4),
        note="输出空间相对误差 ||(W'-W)X||_F/||WX||_F——同一只 W（含离群通道）、同一 4bit 逐列量化器；GPTQ 的补偿按 Hinv 把误差摊给后续列（toy Hessian=2XX^T+1%阻尼，96 条校准样本）")

    # ④ 显存与带宽账（207M 口径）
    N = 207_119_360
    mem = {f"w{b}": dict(MB=round(N * b / 8 / 1e6), decode_floor_ms=round(N * b / 8 / 1.19e12 * 1e3, 3))
           for b in (16, 8, 4)}
    res = dict(rtn_errors=errs, scale_transfer=scale_demo, gptq_sim=gptq_demo, memory_bandwidth=mem,
               note="误差相对口径：|Δ|均值/|x|均值（权重空间）；gptq_sim 为输出空间口径")
    p = os.path.join(out_dir, f"quant_sim_{a.out_name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1)[:900])
    print(f"[产物] {p}")


if __name__ == "__main__":
    main()
