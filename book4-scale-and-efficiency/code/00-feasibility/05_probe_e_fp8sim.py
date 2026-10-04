# 05_probe_e_fp8sim.py —— Book4 探针 E：FP8 量化误差模拟（E4M3/E5M2 × per-tensor/per-group128）
# 用途：ch7 训练侧低精度（FP8）的机制实验底座——MPS 无原生 FP8，本探针用「指数/尾数位截断」
#       的 fake-quant 在 CPU fp32 上模拟 E4M3（4 位指数/3 位尾数，max=448）与 E5M2（5/2，max=57344）
#       的量化-反量化 round-trip 误差，对照两种缩放粒度：
#       ①per-tensor：全矩阵一个 scale（s = amax / FORMAT_MAX）；
#       ②per-group(128)：最后一维每 128 元素一组各一个 scale（DeepSeek FP8 训练的 128 分组口径）。
#       输入两种分布：随机高斯矩阵 + 探针 A 采的真实残差流激活（log/book4-feasibility/act_sample.npz；
#       缺失时自动退化为合成 RMSNorm 样分布并如实标注）。指标：相对 L2 误差与 SNR(dB)。
#       【模拟口径声明】非原生 FP8：①subnormal 按固定步长 2^e_min/2^mant 建模、饱和只做 max 截断
#       （无 NaN/Inf 算术）；②torch.round 为四舍五入（非严格 RN-even）；③只测「权重/激活存储量化」
#       的重建误差，不含 FP8 矩阵乘的累加误差——这三条在正文写作时必须显式标注。
# 所属章节：Book4 ch7（FP8 per-group scaling 的直觉来源：为什么分组缩放救了动态范围）。
# 运行方式：source env.sh && python 05_probe_e_fp8sim.py [--out-name run1]（秒级，CPU fp32）
# 产物：log/book4-feasibility/probeE_fp8sim_{out}.json（不入库）
import argparse
import os

import numpy as np
import torch

from moe_mla_slots import OUT_DIR, SEED, save_json

E4M3 = dict(name="E4M3", mant_bits=3, exp_bits=4, bias=7, fmax=448.0)     # E4M3FN：动态范围小、精度高（权重/前向）
E5M2 = dict(name="E5M2", mant_bits=2, exp_bits=5, bias=15, fmax=57344.0)  # E5M2：范围大、精度低（梯度/溢出容忍）


def fake_fp8_quant(x, fmt):
    """按指定位宽做量化-反量化（round-trip）。x 已预除 scale（|x|≲FORMAT_MAX）。

    实现：符号保留；a=|x| 截到 fmax；e=floor(log2 a)；尾数规范化到 [1,2) 后 round 到 2^mant_bits
    级格点；反解 frac_q·2^e。进位使 frac_q 可能到 2.0（=下一指数的 1.0），数学上无缝。
    subnormal 建模：e 截到 e_min=1-bias；a < 2^e_min 的值用固定绝对步长 2^e_min/2^mant
    （尾数位随幅度递减——per-tensor 大离群值把 scale 撑大后小值掉进 subnormal 区的失真，
    正是 FP 训练里分组缩放要救的现场；E5M2 的 e_min=-14 远低于 E4M3 的 -6，范围代价换精度）。
    """
    mant_bits = fmt["mant_bits"]
    e_min = 1 - fmt["bias"]
    sign = torch.sign(x)
    a = x.abs().clamp(max=fmt["fmax"])
    e = torch.floor(torch.log2(a.clamp_min(1e-38))).clamp_min(e_min)      # 指数（subnormal 区截到 e_min）
    frac = a / torch.exp2(e)                                              # 正常区 [1,2)；subnormal 区 [0,1)
    levels = 2 ** mant_bits
    frac_q = torch.round(frac * levels) / levels                          # 尾数格点化（RN 近似）
    return sign * frac_q * torch.exp2(e)


def quantize_rt(x, fmt, group_size=None):
    """量化-反量化 round-trip。group_size=None → per-tensor；int → 最后一维按组缩放。
    scale = amax/FORMAT_MAX（组内或全张量），反量化后乘回。返回 (x_hat, scale)。"""
    if group_size is None:
        s = x.abs().amax() / fmt["fmax"]
        s = torch.clamp(s, min=1e-30)
        return fake_fp8_quant(x / s, fmt) * s, s
    x2 = x.reshape(-1, group_size)                                        # (N, g)
    s = x2.abs().amax(dim=1, keepdim=True) / fmt["fmax"]                  # (N,1) 每组一个 scale
    s = torch.clamp(s, min=1e-30)
    out = fake_fp8_quant(x2 / s, fmt) * s
    return out.reshape(x.shape), s


def metrics(x, x_hat):
    """相对 L2 误差与 SNR（dB）——量化噪声的功/信比。"""
    err = (x - x_hat).norm().item()
    sig = x.norm().item()
    rel = err / sig
    snr = 20 * np.log10(sig / max(err, 1e-30))
    return round(rel, 6), round(snr, 2)


def main(out_name):
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)

    # 两种输入分布
    inputs = {"gauss_4096x512": torch.randn(4096, 512, generator=g)}      # 随机权重样
    act_path = os.path.join(OUT_DIR, "act_sample.npz")
    if os.path.exists(act_path):                                          # 探针 A 的真实残差流
        h = torch.from_numpy(np.load(act_path)["resid_layer6"]).float()   # (2,256,1024)
        inputs["real_act_512x1024"] = h.reshape(-1, 1024)
        src = "probeA act_sample.npz（207M 第 6 层残差流）"
    else:
        inputs["real_act_512x1024"] = torch.randn(512, 1024, generator=g)
        inputs["real_act_512x1024"] *= 1.0                                # 退化：合成样（如实标注）
        src = "act_sample.npz 缺失——退化为高斯合成（非真实激活！）"
    # 离群值合成样：少量通道放大 ×100 / ×1000（LLM 激活通道离群的结构化近似——per-group 救的就是这种分布）
    ol = torch.randn(4096, 512, generator=g)
    ol_cols = torch.randperm(512, generator=g)[:5]
    ol[:, ol_cols] *= 100.0                                               # 512 通道中 5 个通道 ×100
    inputs["outlier_synth_4096x512"] = ol
    ol_mask = torch.ones(512, dtype=torch.bool)
    ol_mask[ol_cols] = False                                              # 非离群通道掩码（附加指标用）
    ol2 = torch.randn(4096, 512, generator=g)
    ol2[:, ol_cols] *= 1000.0                                             # ×1000：per-tensor scale 撑大 10×，
    inputs["outlier1k_synth_4096x512"] = ol2                              # E4M3 的正常通道掉进 subnormal 区
    print(f"[输入] {src}")
    for k, v in inputs.items():
        print(f"  {k}: shape {tuple(v.shape)} | mean {v.mean():+.4f} std {v.std():.4f} "
              f"| amax {v.abs().max():.3f} | p99 {torch.quantile(v.flatten()[::16].abs(), 0.99):.3f}")

    rows = []
    for fmt in (E4M3, E5M2):
        for gs, gs_name in ((None, "per-tensor"), (128, "per-group(128)")):
            for in_name, x in inputs.items():
                x_hat, _ = quantize_rt(x, fmt, gs)
                rel, snr = metrics(x, x_hat)
                row = {"fmt": fmt["name"], "scaling": gs_name, "input": in_name,
                       "rel_l2_err": rel, "snr_db": snr}
                if in_name.startswith("outlier"):
                    # 附加指标：只看非离群通道——全局 L2 被大通道能量主导，会掩盖小通道的失真
                    sub = x[:, ol_mask]
                    sub_rel, sub_snr = metrics(sub, x_hat[:, ol_mask])
                    row["rel_l2_err_normal_channels"] = sub_rel
                    row["snr_db_normal_channels"] = sub_snr
                    print(f"[{fmt['name']:>5}] {gs_name:>15} | {in_name:>20} | rel_err {rel:.5f} | "
                          f"SNR {snr:6.2f} dB | 仅非离群通道: rel {sub_rel:.5f} / SNR {sub_snr:.2f} dB")
                else:
                    print(f"[{fmt['name']:>5}] {gs_name:>15} | {in_name:>20} | rel_err {rel:.5f} | SNR {snr:6.2f} dB")
                rows.append(row)

    # 汇总读数：分组缩放的增益倍数（per-group vs per-tensor 的 rel_err 比）
    summary = {}
    for fmt in (E4M3, E5M2):
        for in_name in inputs:
            pt = next(r["rel_l2_err"] for r in rows if r["fmt"] == fmt["name"]
                      and r["scaling"] == "per-tensor" and r["input"] == in_name)
            pg = next(r["rel_l2_err"] for r in rows if r["fmt"] == fmt["name"]
                      and r["scaling"] == "per-group(128)" and r["input"] == in_name)
            summary[f"{fmt['name']}/{in_name}"] = round(pt / max(pg, 1e-12), 2)
    print("[增益] per-group(128) 相对 per-tensor 的误差下降倍数：", summary)

    report = {"seed": SEED, "device": "cpu fp32（MPS 无原生 FP8——模拟口径，声明见文件头）",
              "input_source": src, "formats": {"E4M3": "4exp/3mant max=448", "E5M2": "5exp/2mant max=57344"},
              "rows": rows, "group_gain_x": summary,
              "caveats": ["subnormal 已按固定步长建模、饱和只做 max 截断（无 NaN/Inf 算术）",
                          "torch.round 非 RN-even",
                          "只测存储量化重建误差，不含 FP8 GEMM 累加误差"]}
    save_json("probeE_fp8sim", report, out_name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="探针 E：FP8 fake-quant round-trip 误差对照")
    ap.add_argument("--out-name", default="run1", help="产物 JSON 后缀（防覆写）")
    args = ap.parse_args()
    main(args.out_name)
