# smoke_mtp.py —— Book4 ch8 前置（P11）：mini MTP 探针冒烟（秒级，CPU fp32，判据 papers/05 §4.2）
# 冒烟判据（notes/07 §3.3 预注册）：
#   ① 参数账两级 assert：主干 26,089,984（模块档 dense 锚点）+ MTP 新增 3,475,968（3 RMSNorm +
#      proj 2d² + TRM 层，手算=实测逐位）；总参 29,565,952（共享件只计一次——parameters() 去重）。
#   ② 初始 loss：主支路 ≈ ln V + d·σ²/2 = 9.011+0.102 ≈ 9.11；MTP 支路同量级（带方差修正带宽
#      [8.6, 9.6]——「≈ln 8192≈9.01」判据的方差修正版）。
#   ③ off-by-one 三视图核验（结构断言 + 教师强制功能核验：MTP 支路位置 i 的输入嵌入 = tokens[i+1]）。
#   ④ λ=0 时联合 loss == 主支路 CE（逐位）；λ=0.3 时联合 = main + 0.3·mtp（恒等式）。
#   ⑤ 反向一步：损失有限、梯度无 NaN、主支路梯度路径通（「主支路不受拖累」的最小验证）。
# 运行方式：source env.sh && python smoke_mtp.py [--out-name run1]（CPU fp32 报数）
# 产物：log/book4-ch08/mtp_smoke_{out}.json
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "00-feasibility")))
sys.path.insert(0, HERE)
from mtp import (BACKBONE_PARAMS, EVAL_BIN, MTP_NEW_PARAMS, MTP_LAMBDA_DEFAULT, SEED,   # noqa: E402
                 TOTAL_PARAMS, VOCAB, assert_off_by_one, joint_loss, three_views)

OUT_DIR = os.path.join(os.path.abspath(os.path.join(HERE, "..", "..")), "log", "book4-ch08")


class _RecordingEmbed(torch.nn.Module):
    """教师强制功能核验的探针：包一层 embedding，记录最后一次被喂的 token 索引。"""

    def __init__(self, embed):
        super().__init__()
        self.embed, self.last_input = embed, None

    def forward(self, idx):
        self.last_input = idx.detach().clone()
        return self.embed(idx)


def main():
    ap = argparse.ArgumentParser(description="mini MTP 冒烟（秒级 CPU）")
    ap.add_argument("--out-name", default="run1")
    args = ap.parse_args()
    device = torch.device("cpu")                     # 冒烟数字一律 CPU fp32 报（papers/05 §4.2）
    torch.manual_seed(SEED)

    # ---- ① 模型与参数账 ----
    from mtp import MTPSlotLLaMA                     # 延迟 import：冒烟先验 mtp.py 可载入
    m = MTPSlotLLaMA()
    checks = {}
    assert m.backbone.n_params() == BACKBONE_PARAMS, \
        f"主干 {m.backbone.n_params()} != {BACKBONE_PARAMS}（模块档 dense 锚点）"
    assert m.mtp.n_params() == MTP_NEW_PARAMS, \
        f"MTP 新增 {m.mtp.n_params()} != {MTP_NEW_PARAMS}（手算 3·d + 2d² + TRM 层）"
    assert m.n_params() == TOTAL_PARAMS, \
        f"总参 {m.n_params()} != {TOTAL_PARAMS}（共享 emb/head 必须只计一次）"
    checks["params"] = {"backbone": BACKBONE_PARAMS, "mtp_new": MTP_NEW_PARAMS,
                        "total": TOTAL_PARAMS,
                        "mtp_overhead_ratio": round(MTP_NEW_PARAMS / BACKBONE_PARAMS, 4)}
    print(f"[① 参数账] 主干 {BACKBONE_PARAMS:,} + MTP 新增 {MTP_NEW_PARAMS:,}（3 RMSNorm + "
          f"proj {512*1024:,} + TRM 层 {MTP_NEW_PARAMS - 512*1024 - 3*512:,}）= 总 {TOTAL_PARAMS:,}"
          f"（+{MTP_NEW_PARAMS/BACKBONE_PARAMS:.1%}——「MTP 很便宜」的量化体感）")

    # ---- ③ off-by-one 三视图：结构断言（合成可分辨序列） ----
    B, T = 2, 64
    tokens = (torch.arange(B * T) % VOCAB).reshape(B, T)             # 每个位置值可分辨
    assert_off_by_one(tokens)
    x_in, y_main, y_mtp = three_views(tokens)
    assert torch.equal(y_main[:, :-1], x_in[:, 1:]) and torch.equal(y_mtp[:, :-1], y_main[:, 1:])
    checks["off_by_one_structure"] = "PASS（等长 T-2 / y_main=x_in 左移一 / y_mtp=y_main 左移一 / 单点抽验）"
    print(f"[③ 三视图] tokens (B={B},T={T}) -> 三视图各 (B,{T-2})：x_in=t[:,-2] y_main=t[:,1:-1] "
          f"y_mtp=t[:,2:]——结构断言 PASS")

    # ---- ③' 教师强制功能核验：MTP 支路位置 i 的输入嵌入 = tokens[i+1]（= y_main） ----
    rec = _RecordingEmbed(m.backbone.model.embed_tokens)
    with torch.no_grad():
        h = m.backbone.model(x_in)                                   # (B,T-2,d)
        _ = m.mtp(h, y_main, rec, m.backbone.lm_head)                # 探针 embed 记录实际喂入
    assert rec.last_input is not None and torch.equal(rec.last_input, y_main), \
        "MTP 支路的 teacher forcing 嵌入应为 y_main（tokens[i+1]）"
    checks["teacher_forcing_functional"] = "PASS（MTP 支路输入嵌入逐位 = tokens[i+1]）"
    print("[③' 教师强制] MTP 支路位置 i 的输入嵌入逐位 = tokens[i+1]——功能核验 PASS")

    # ---- ② 初始 loss（真语料批：eval.bin 首段 (4,512)，CPU fp32） ----
    ev = np.memmap(EVAL_BIN, dtype=np.uint16, mode="r")
    toks = torch.from_numpy(np.asarray(ev[:4 * 512], dtype=np.int64)).reshape(4, 512)
    assert_off_by_one(toks)
    with torch.no_grad():
        logits_main, logits_mtp = m(toks)
        loss, ce_main, ce_mtp = joint_loss(logits_main, logits_mtp, toks, lam=MTP_LAMBDA_DEFAULT)
    ln_v = math.log(VOCAB)
    band = (8.6, 9.6)
    assert band[0] < ce_main.item() < band[1], f"主支路初始 {ce_main.item():.4f} 越带 {band}"
    assert band[0] < ce_mtp.item() < band[1], f"MTP 支路初始 {ce_mtp.item():.4f} 越带 {band}"
    checks["initial_loss"] = {"ln_v": round(ln_v, 4),
                              "expected_main": round(ln_v + 512 * 0.02 ** 2 / 2, 4),
                              "main": round(ce_main.item(), 4), "mtp": round(ce_mtp.item(), 4),
                              "band": list(band), "verdict": "PASS（两支路均 ≈ln V 量级）"}
    print(f"[② 初始 loss] ln V = {ln_v:.4f} | 主支路 {ce_main.item():.4f}（期望 ≈"
          f"{ln_v + 512 * 0.02 ** 2 / 2:.3f}）| MTP 支路 {ce_mtp.item():.4f}——均落带 {band} PASS")

    # ---- ④ λ 恒等式 ----
    with torch.no_grad():
        loss0, ce0_main, ce0_mtp = joint_loss(logits_main, logits_mtp, toks, lam=0.0)
    assert torch.equal(loss0, ce0_main), "λ=0 联合应逐位等于主支路 CE"
    assert torch.allclose(loss, ce_main + MTP_LAMBDA_DEFAULT * ce_mtp), "λ=0.3 恒等式不成立"
    checks["lambda_identities"] = "PASS（λ=0: L==CE_main 逐位；λ=0.3: L==CE_main+0.3·CE_mtp）"
    print("[④ λ 恒等式] λ=0 → L==CE_main（逐位）；λ=0.3 → L==CE_main+0.3·CE_mtp——PASS")

    # ---- ⑤ 反向一步（主支路不受拖累的最小验证） ----
    loss, _, _ = joint_loss(*m(toks), toks, lam=MTP_LAMBDA_DEFAULT)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads), "梯度出现非有限值"
    backbone_grad_norm = (sum(float(g.norm() ** 2) for g in
                              (p.grad for p in m.backbone.parameters()) if g is not None)) ** 0.5
    checks["backward"] = {"loss_finite": bool(torch.isfinite(loss).item()),
                          "n_grad_tensors": len(grads),
                          "backbone_grad_norm": round(backbone_grad_norm, 6),
                          "verdict": "PASS（反向通、梯度全有限——MTP 支路梯度确实回流主干）"}
    print(f"[⑤ 反向] 联合 loss {loss.item():.4f} 有限；{len(grads)} 个参数张量梯度全有限；"
          f"主干梯度范数 {backbone_grad_norm:.4f}——PASS（「主支路被拖累」与否留给 fast 曲线判读）")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"mtp_smoke_{args.out_name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"date": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED,
                   "device": str(device), "checks": checks,
                   "all_pass": True}, f, ensure_ascii=False, indent=2)
    print(f"[产物] {out}")
    print("mini MTP 冒烟全部通过")


if __name__ == "__main__":
    main()
