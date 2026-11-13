# fp8_sim.py —— Book4第7章：FP8/块FP4往返误差，CPU fp32数值模拟
# 运行：工作区根目录 source ./env.sh 后运行本文件 --out-name run1。
# 实验：格式格距、离群主体SNR、缩放粒度、OCP建议MXFP4与历史ceil策略对照。
# 不执行FP8/FP4矩阵乘，不测训练质量或硬件速度。仅接受有限浮点输入。
# FP8使用ties-to-even与有限值饱和；不模拟NaN/Inf算术。
# MXFP4默认依OCP MX v1.0 §6.3建议取scale=2^(floor(log2(amax))-2)；
# 历史ceil(log2(amax/6))另列为legacy_ceil，二者均用32元素块与E2M1。
# E8M0指数范围限制为[-127,127]；零块选scale=1；负零保留。
# 输入真实残差流缺失时使用合成替代并标注；历史已发表数据不被本次模拟覆盖。
# 输出log/book4-ch07独立JSON；BOOK4_ACTIVATION_NPZ可指定网络样本文件，存在不等于来自训练后分布。
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "00-feasibility")))
from moe_mla_slots import SEED  # noqa: E402  全书统一种子 20261002（单轨纪律：不另立种子）

REPO_ROOT = os.getcwd()
OUT_DIR = os.path.join(REPO_ROOT, "log", "book4-ch07")
ACT_NPZ = os.environ.get("BOOK4_ACTIVATION_NPZ", os.path.join(REPO_ROOT, "log", "book4-feasibility", "act_sample.npz"))

E4M3 = dict(name="E4M3", mant_bits=3, exp_bits=4, bias=7, fmax=448.0)      # 前向权重/激活：刻度细、尺子短
E5M2 = dict(name="E5M2", mant_bits=2, exp_bits=5, bias=15, fmax=57344.0)   # 梯度：尺子长、刻度粗
E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])       # MXFP4 块内元素格点（正半轴）


def finite_input(x):
    if x.dtype not in (torch.float32, torch.float64) or x.numel() == 0 or not torch.isfinite(x).all():
        raise ValueError("需要非空有限FP32或FP64输入")


def groups(x, size):
    finite_input(x)
    if not isinstance(size, int) or size < 1 or x.ndim == 0 or x.shape[-1] % size:
        raise ValueError("组大小须为正整数且整除最后一维；不得跨行拼组")
    return x.reshape(-1, size)


def fake_fp8_quant(x, fmt):
    """按指定位宽做量化-反量化（round-trip）。x 已预除 scale（|x|≲fmax）。

    实现：符号保留；a=|x| 截到 fmax；e=floor(log2 a)；尾数规范化到 [1,2) 后 round 到
    2^mant_bits 级格点；反解 frac_q·2^e（进位到 2.0 时数学上无缝并入下一指数）。
    subnormal 建模：e 截到 e_min=1-bias；小值用固定绝对步长 2^e_min/2^mant——
    per-tensor 大离群把 scale 撑大后小值掉进次正规数区的失真正是要演示的现场。
    输入 (…)，输出 (…) 同形。
    """
    finite_input(x)
    mant_bits = fmt["mant_bits"]
    e_min = 1 - fmt["bias"]
    a = x.abs().clamp(max=fmt["fmax"])
    e = torch.floor(torch.log2(a.clamp_min(1e-38))).clamp_min(e_min)      # 指数（次正规区截到 e_min）
    frac = a / torch.exp2(e)                                              # 正常区 [1,2)；次正规区 [0,1)
    levels = 2 ** mant_bits
    frac_q = torch.round(frac * levels) / levels                          # 格点舍入：torch.round为ties-to-even
    return torch.copysign(frac_q * torch.exp2(e), x)


def quantize_rt(x, fmt, group_size=None):
    """FP8 量化-反量化 round-trip。group_size=None → per-tensor（全张量一个 scale）；
    int → 最后维按组各一个 scale（DSV3 激活 1×128 口径）。scale=amax/fmax。返回 (x_hat, scale)。"""
    finite_input(x)
    if group_size is None:
        s = (x.abs().amax() / fmt["fmax"]).clamp(min=1e-30)
        return fake_fp8_quant(x / s, fmt) * s, s
    x2 = groups(x, group_size)                                        # (N, g) 每组一行
    s = (x2.abs().amax(dim=1, keepdim=True) / fmt["fmax"]).clamp(min=1e-30)  # (N,1)
    return fake_fp8_quant(x2 / s, fmt).mul(s).reshape(x.shape), s


def e2m1_round(x):
    """有限E2M1最近舍入，编码最低位为偶数者打破中点平局；越界饱和到±6。"""
    finite_input(x)
    levels = E2M1_LEVELS.to(device=x.device, dtype=x.dtype)
    mid = (levels[1:] + levels[:-1]) / 2
    magnitude = x.abs()
    idx = torch.bucketize(magnitude.contiguous(), mid)
    tie = (idx < len(mid)) & (magnitude == mid[idx.clamp_max(len(mid)-1)])
    idx = idx + (tie & (idx.remainder(2) == 1)).long()
    return torch.copysign(levels[idx], x)


def mxfp4_rt(x, block=32, scale_mode="ocp"):
    """32元素E2M1+E8M0缩放模拟。ocp为v1.0 §6.3建议；legacy_ceil为旧无溢出选法。
    两者均有有限指数范围与ties-to-even；未调用硬件打包或低精度GEMM。
    """
    if block != 32:
        raise ValueError("本函数的MXFP4块固定为32个元素")
    x2 = groups(x, block)
    amax = x2.abs().amax(dim=1, keepdim=True)
    logmax = torch.log2(amax.clamp_min(torch.finfo(x.dtype).tiny))
    if scale_mode == "ocp":
        exponent = torch.floor(logmax) - 2
    elif scale_mode == "legacy_ceil":
        exponent = torch.ceil(torch.log2(amax.double().clamp_min(torch.finfo(torch.float64).tiny) / 6)).to(x.dtype)
    else:
        raise ValueError("scale_mode须为ocp或legacy_ceil")
    exponent = torch.where(amax == 0, torch.zeros_like(exponent), exponent).clamp(-127,127)
    scale = torch.exp2(exponent)
    return (e2m1_round(x2 / scale) * scale).reshape(x.shape), scale


def e2m1_tensor_rt(x):
    """全张量单个实数scale=amax/6的E2M1对照；仍有scale，不能叫无缩放。"""
    finite_input(x)
    scale = (x.abs().amax() / 6).clamp_min(1e-30)
    return e2m1_round(x / scale) * scale, scale


def rel_l2(x, x_hat):
    """相对 L2 误差（P10 口径：本章一律 rel-L2% 报数，dB 只用于 E4M3 断崖主实验）。"""
    error, norm = (x - x_hat).norm().item(), x.norm().item()
    return error / norm if norm else (0.0 if error == 0 else float("inf"))


def snr_db(x, x_hat, mask=None):
    """SNR（dB）；mask 给定时只统计子集（bulk 指标——全体 SNR 会被离群能量稀释）。"""
    if mask is not None:
        x, x_hat = x[mask], x_hat[mask]
    err = (x - x_hat).norm().item()
    return round(20 * np.log10(x.norm().item() / max(err, 1e-30)), 2)


def main(out_name):
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    report = {"seed": SEED, "device": "cpu fp32 simulation", "torch": torch.__version__,
              "caveats": ["只接受有限浮点输入；FP8有限饱和与ties-to-even，不模拟特殊值算术",
                          "只测往返重建误差，不含FP8/FP4 GEMM与累加或训练质量",
                          "MXFP4: OCP v1.0建议scale与历史legacy_ceil分列；E8M0指数[-127,127]",
                          "实数scale设置1e-30数值下限；整体尺度不变性仅在未触及该下限时讨论"]}

    # ---------- 输入分布 ----------
    inputs = {"gauss_4096x512": torch.randn(4096, 512, generator=g)}
    if os.path.exists(ACT_NPZ):                                           # 探针 A 真实残差流
        h = torch.from_numpy(np.load(ACT_NPZ)["resid_layer6"]).float()    # (2,256,1024)
        inputs["real_act_512x1024"] = h.reshape(-1, 1024)
        src = f"{ACT_NPZ}（网络采样；探针A默认随机权重与随机token，非训练后语料激活）"
    else:
        inputs["real_act_512x1024"] = torch.randn(512, 1024, generator=g)
        src = "act_sample.npz 缺失——退化为高斯合成（非真实激活！）"
    report["activation_input"] = {"source": src, "from_sample_file": os.path.exists(ACT_NPZ),
        "sha256": hashlib.sha256(open(ACT_NPZ,"rb").read()).hexdigest() if os.path.exists(ACT_NPZ) else None,
        "legacy_key": "real_act_512x1024仅为兼容旧键，实际来源以is_real为准"}
    # 通道离群合成样：512 通道中 5 个 ×100 / ×1000（LLM 激活通道离群的结构化近似）
    ol_cols = torch.randperm(512, generator=g)[:5]
    ol_mask = torch.ones(512, dtype=torch.bool)
    ol_mask[ol_cols] = False                                              # 非离群通道掩码
    for mag, tag in ((100.0, "outlier_x100"), (1000.0, "outlier_x1000")):
        ol = torch.randn(4096, 512, generator=g)
        ol[:, ol_cols] *= mag
        inputs[f"{tag}_4096x512"] = ol
    print(f"[输入] {src}")

    # ---------- 实验① 格式刻度 + FP 尺度自相似 ----------
    xg = inputs["gauss_4096x512"]
    exp1 = {}
    for fmt in (E4M3, E5M2):
        xh, _ = quantize_rt(xg, fmt)
        exp1[fmt["name"] + "_pertensor_relL2"] = round(rel_l2(xg, xh) * 100, 3)
    exp1["one_mantissa_bit_ratio"] = round(
        exp1["E5M2_pertensor_relL2"] / exp1["E4M3_pertensor_relL2"], 2)   # 1 位尾数差 → ~2×
    selfsim = []
    for c in (0.011, 0.088, 0.88):                                        # 三种整体 scale 同一矩阵
        xh, _ = quantize_rt(xg * c, E4M3)
        selfsim.append(round(rel_l2(xg * c, xh) * 100, 3))
    exp1["fp_scale_selfsimilarity_relL2_at_0.011/0.088/0.88"] = selfsim
    report["exp1_format"] = exp1
    print(f"[①格式] E4M3 {exp1['E4M3_pertensor_relL2']}% | E5M2 {exp1['E5M2_pertensor_relL2']}% "
          f"(比 {exp1['one_mantissa_bit_ratio']}×) | 尺度自相似 {selfsim}")

    # ---------- 实验② bulk 断崖（主实验）：离群强度扫描 ----------
    exp2 = {}
    for fac, frac, tag in ((30.0, 0.001, "mild_x30"), (300.0, 0.001, "strong_x300"),
                           (30000.0, 0.0005, "extreme_x30000")):
        base = torch.randn(4, 512, 1024, generator=g)                     # (4,512,1024) 2.1M 元素
        flat = base.flatten()
        k = round(flat.numel() * frac)
        idx = torch.randperm(flat.numel(), generator=g)[:k]
        flat[idx] *= fac                                                  # 离群点：随机散点 ×fac
        x = flat.reshape(base.shape)
        bulk = torch.ones_like(flat, dtype=torch.bool)
        bulk[idx] = False
        bulk = bulk.reshape(base.shape)
        s_pt = (x.abs().amax() / E4M3["fmax"]).item()
        med_after = x[bulk].abs().median().item() / s_pt                  # bulk 中位数缩后落点
        row = {"outlier_frac": frac, "ratio_max_to_median": round(x.abs().max().item()
                                                                  / x.abs().median().item(), 1),
               "bulk_median_after_pertensor_scale": round(med_after, 5),
               "below_E4M3_min_normal_2^-6=0.0156": bool(med_after < 2 ** -6)}
        for gs, gname in ((None, "per_tensor"), (128, "per_group128"), (32, "per_group32")):
            xh, _ = quantize_rt(x, E4M3, gs)
            row[f"E4M3_{gname}"] = {"overall_db": snr_db(x, xh),
                                    "bulk_only_db": snr_db(x, xh, bulk)}
        exp2[tag] = row
        print(f"[②bulk] {tag}: max/med {row['ratio_max_to_median']:.0f} | bulk 中位缩后 "
              f"{med_after:.5f} | per-tensor 全体 {row['E4M3_per_tensor']['overall_db']} dB / "
              f"bulk {row['E4M3_per_tensor']['bulk_only_db']} dB | pg128 bulk "
              f"{row['E4M3_per_group128']['bulk_only_db']} dB")
    report["exp2_bulk_cliff"] = exp2

    # ---------- 实验③ 分组收益条件性：平滑 vs 通道离群 ----------
    exp3 = {}
    for fmt in (E4M3, E5M2):
        for in_name, x in inputs.items():
            pt, _ = quantize_rt(x, fmt)
            pg, _ = quantize_rt(x, fmt, 128)
            rpt, rpg = rel_l2(x, pt), rel_l2(x, pg)
            exp3[f"{fmt['name']}/{in_name}"] = {
                "pertensor_relL2_pct": round(rpt * 100, 3), "pergroup128_relL2_pct": round(rpg * 100, 3),
                "group_gain_x": round(rpt / max(rpg, 1e-12), 2)}
    report["exp3_group_gain"] = exp3
    key = lambda f, i: exp3[f"{f}/{i}"]["group_gain_x"]                    # noqa: E731
    print(f"[③分组] E4M3 增益：平滑 gauss {key('E4M3','gauss_4096x512')}× / 激活样本（来源见上） "
          f"{key('E4M3','real_act_512x1024')}× / 通道离群×100 "
          f"{key('E4M3','outlier_x100_4096x512')}× / ×1000 {key('E4M3','outlier_x1000_4096x512')}×")

    # ---------- 实验④ MXFP4 块模拟（扩展口）----------
    exp4 = {"mechanism": "最后维 32 元素/块 · 共享 E8M0（2 的幂）· 块内 E2M1 格点 "
                         "{0,.5,1,1.5,2,3,4,6} · 4.25 bits/param = 4+8/32（算术）"}
    for in_name in ("gauss_4096x512", "real_act_512x1024", "outlier_x100_4096x512"):
        x = inputs[in_name]
        xb, _ = mxfp4_rt(x)
        legacy, _ = mxfp4_rt(x, scale_mode="legacy_ceil")                                               # 块共享指数版
        xt, _ = e2m1_tensor_rt(x)                                         # per-tensor E2M1 对照
        exp4[in_name] = {"mxfp4_block32_relL2_pct": round(rel_l2(x, xb) * 100, 2),
                         "scale_policy": "OCP v1.0 §6.3 recommended",
                         "legacy_ceil_block32_relL2_pct": round(rel_l2(x, legacy) * 100, 2),
                         "e2m1_pertensor_relL2_pct": round(rel_l2(x, xt) * 100, 2),
                         "block_scale_gain_x": round(rel_l2(x, xt) / max(rel_l2(x, xb), 1e-12), 2)}
    report["exp4_mxfp4"] = exp4
    print("[④MXFP4] " + " | ".join(f"{k.split('_4096')[0].split('_512')[0]}: "
          f"{v['mxfp4_block32_relL2_pct']}%（块 scale 增益 {v['block_scale_gain_x']}×）"
          for k, v in exp4.items() if isinstance(v, dict)))

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"fp8_sim_{out_name}.json")
    with open(path, "x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"[产物] {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch7 FP8/MXFP4 量化误差模拟（CPU fp32 秒级）")
    ap.add_argument("--out-name", default="run1", help="产物 JSON 后缀（防覆写）")
    args = ap.parse_args()
    main(args.out_name)
