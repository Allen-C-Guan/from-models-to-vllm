# kv_probe_mla.py —— Book4 ch6 KV 账本 probe 正身：多几何手算=实测 + 三口径红线算术
# 用途：正文 6.2（手算例）/6.7（三代同表与三口径）/6.8（字节实测与多几何节省表）的全部数字来源：
#       Part 0  d=8→r=2 低秩手算例（确定性断言：c=[4,8] 两数装箱、K/V 同源两读法、缓存 −81.25%）
#       Part 1  KV 字节实测=公式逐位（7 case：真实 bf16 缓存布局张量构造 vs 手算公式 assert）
#       Part 2  多几何每 token KV 元素对照表（probe 档 / 207M 两刀档 / MHA 反事实 / V2-Lite /
#               DSV2 236B / DSV3 / DeepSeek 67B）——Book3 ch6 kv_probe 协议的 MLA 续本
#       Part 3  三口径红线算术：架构口径 −98.24% / 部署口径 −93.3%（复算 93.34%）/ 跨机 bf16
#               −82.24%；金句复算「=GQA with only 2.25 groups」（576/(2·128)）；MLA vs GQA-8
#               每层参数账（公式）；Lite 几何 14.06% vs 论文 Table 9 的 14%。
# 所属章节：Book4 第 6 章（MLA：把 KV 装进小箱子）。
# 运行方式：cd 工作区根目录 && source env.sh && python code/ch06/kv_probe_mla.py
#           [--out-name run1]（秒级，CPU，确定性——Part 0/3 无随机，Part 1 只构造零张量）
# 产物：log/book4-ch06/kv_probe_mla_{out}.json（不入库）
# 口径纪律：KV 账单位=元素数（DSV2 Table 1 口径）；MLA 无 K/V 各记一遍的因子 2——bf16 下出现的
#           2 是字节数。公式：MHA/GQA/MQA 每层 2·h_kv·d_k 元素（承 Book3 式 (6.2)）；MLA 每层
#           d_c+d_h^R 元素（c_KV 一份 + 解耦键 k_R 一份，上投影不进缓存）。
import argparse
import json
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book4-ch06")

BF16 = 2  # 每元素字节数


def part0_hand_example():
    """d=8 → r=2 手算例（论文 6.2 节正身；确定性、无随机）。"""
    h = torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])               # (8,)
    W_DKV = torch.tensor([[1.0, 0, 1, 0, 1, 0, 1, 0],
                          [0, 1.0, 0, 1, 0, 1, 0, 1]])                        # (2,8) 下投影：装箱
    c = W_DKV @ h                                                            # (2,) = [4, 8]
    W_UK = torch.tensor([[0.25, 0], [0, 0.125], [0.25, 0], [0, 0.125],
                         [0.25, 0], [0, 0.125], [0.25, 0], [0, 0.125]])       # (8,2) K 的拆箱镜
    W_UV = torch.tensor([[1.0, 0], [0, -1.0], [1.0, 0], [0, -1.0],
                         [1.0, 0], [0, -1.0], [1.0, 0], [0, -1.0]])           # (8,2) V 的拆箱镜
    k, v = W_UK @ c, W_UV @ c                                                # (8,) 两读法
    W_KR = torch.tensor([1.0, 1, 0, 0, 0, 0, 0, 0])                           # (8,) 位置通道
    k_r = W_KR @ h                                                           # 标量 3
    assert torch.allclose(c, torch.tensor([4.0, 8.0]))
    assert torch.allclose(k, torch.ones(8)) and torch.allclose(v, torch.tensor([4.0, -8, 4, -8, 4, -8, 4, -8.0]))
    assert k_r.item() == 3.0
    assert torch.allclose(k, (W_UK @ W_DKV) @ h) and torch.allclose(v, (W_UV @ W_DKV) @ h)
    cache = (2 + 1) / 16                                                     # (d_c + d_h^R)/d
    print(f"[Part0 手算例] h=[1,2,1,2,...] → c=[4,8]（外加 k_R=3）；K 读法全 1、V 读法 4/-8 交替"
          f"——与满秩真值逐位一致；缓存 {2+1}/16 = {cache:.4f}（−{(1-cache)*100:.2f}%）")
    return {"c": [4.0, 8.0], "k_r": 3.0, "cache_frac": 3 / 16, "saving_pct": (1 - 3 / 16) * 100}


def part1_bytes_measured():
    """7 case：构造真实布局的 bf16 缓存张量，实测字节 vs 手算公式逐位 assert。
    GQA/MHA 布局：(1,L,n,h_kv,d_k)×2（K、V 各一份）；MLA 布局：(1,L,n,d_c)+(1,L,n,d_r)。"""
    n = 1024
    cases = [  # (名称, L, h_kv, d_k, d_c, d_r)
        ("GQA 207M 两刀档基线（L=12,h_kv=8,d_k=64）", 12, 8, 64, None, None),
        ("MLA 207M 两刀档（d_c=256,d_h^R=64）", 12, None, None, 256, 64),
        ("MHA 反事实 207M 档（h_kv=16,d_k=64）", 12, 16, 64, None, None),
        ("GQA probe 档（L=6,h_kv=4,d_k=64）", 6, 4, 64, None, None),
        ("MLA probe 档（d_c=256,d_h^R=64）", 6, None, None, 256, 64),
        ("MLA DSV2-Lite 官方几何（L=27,d_c=512,d_h^R=64）", 27, None, None, 512, 64),
        ("MLA DSV2-236B 几何（L=60,d_c=512,d_h^R=64）", 60, None, None, 512, 64),
    ]
    rows, n_pass = [], 0
    for name, L, h_kv, d_k, d_c, d_r in cases:
        if d_c is None:   # GQA/MHA：K、V 各一份
            tensors = (torch.zeros(1, L, n, h_kv, d_k, dtype=torch.bfloat16),
                       torch.zeros(1, L, n, h_kv, d_k, dtype=torch.bfloat16))
            formula = 2 * L * n * h_kv * d_k * BF16
            elems = 2 * L * h_kv * d_k
        else:             # MLA：c_KV 一份 + k_R 一份（无 K/V 因子）
            tensors = (torch.zeros(1, L, n, d_c, dtype=torch.bfloat16),
                       torch.zeros(1, L, n, d_r, dtype=torch.bfloat16))
            formula = L * n * (d_c + d_r) * BF16
            elems = L * (d_c + d_r)
        measured = sum(t.element_size() * t.numel() for t in tensors)
        assert measured == formula, (name, measured, formula)
        n_pass += 1
        rows.append({"case": name, "elems_per_token": elems,
                     "bytes_measured": measured, "bytes_formula": formula, "match": True})
        print(f"[Part1 实测=公式] {name}: {elems:,} 元素/token | {measured:,} B = 公式逐位（"
              f"{measured/1024/1024:.2f} MiB @n=1024）")
    print(f"[Part1 判定] {n_pass}/{len(cases)} case 手算=实测逐位一致")
    return {"cases": rows, "n_pass": n_pass, "n_total": len(cases)}


def part2_geometry_table():
    """多几何每 token KV 元素对照（公式口径；「对照口径」列标分母是 GQA 还是同构 MHA 反事实）。"""
    def mla(L, d_c, d_r):
        return L * (d_c + d_r)

    rows = [
        {"geometry": "probe 档（L=6）", "baseline": "GQA h_kv=4/d_k=64", "baseline_elems": 2 * 6 * 4 * 64,
         "mla_elems": mla(6, 256, 64), "saving_pct": round((1 - mla(6, 256, 64) / (2 * 6 * 4 * 64)) * 100, 1)},
        {"geometry": "207M 两刀档（L=12）", "baseline": "GQA h_kv=8/d_k=64（Book3 定版）",
         "baseline_elems": 2 * 12 * 8 * 64, "mla_elems": mla(12, 256, 64),
         "saving_pct": round((1 - mla(12, 256, 64) / (2 * 12 * 8 * 64)) * 100, 1)},
        {"geometry": "207M 档 MHA 反事实", "baseline": "MHA h=16/d_k=64", "baseline_elems": 2 * 12 * 16 * 64,
         "mla_elems": mla(12, 256, 64), "saving_pct": round((1 - mla(12, 256, 64) / (2 * 12 * 16 * 64)) * 100, 1)},
        {"geometry": "DSV2-Lite 官方几何（L=27, n_h=16）", "baseline": "同构 MHA 2·16·128·27",
         "baseline_elems": 2 * 27 * 16 * 128, "mla_elems": mla(27, 512, 64),
         "saving_pct": round((1 - mla(27, 512, 64) / (2 * 27 * 16 * 128)) * 100, 1)},
        {"geometry": "DSV2-236B（L=60, n_h=128）", "baseline": "同构 MHA 2·128·128·60",
         "baseline_elems": 2 * 60 * 128 * 128, "mla_elems": mla(60, 512, 64),
         "saving_pct": round((1 - mla(60, 512, 64) / (2 * 60 * 128 * 128)) * 100, 1)},
    ]
    for r in rows:
        print(f"[Part2 多几何] {r['geometry']}: 对照 {r['baseline']} {r['baseline_elems']:,} vs MLA "
              f"{r['mla_elems']:,} 元素/token → 节省 {r['saving_pct']}%")
    # 逐机型绝对账（每 token 元素与 bf16 字节）
    per_model = [
        ("DeepSeek 67B（GQA-8, d_h=128, L=95）", 2 * 8 * 128 * 95),
        ("DSV2-236B MLA（L=60）", 576 * 60),
        ("DSV2-Lite MLA（L=27）", 576 * 27),
        ("DSV3-671B MLA（L=61）", 576 * 61),
        ("DSV2 同构 MHA 反事实（n_h=128, L=60）", 32768 * 60),
    ]
    for name, e in per_model:
        print(f"[Part2 机型账] {name}: {e:,} 元素/token = {e*BF16:,} B（{e*BF16/1024:.1f} KiB, bf16）")
    return {"vs_baseline": rows, "per_model": [{"name": n, "elems": e, "bf16_bytes": e * BF16}
                                               for n, e in per_model]}


def part3_three_calibers():
    """三口径红线算术 + 金句复算 + 每层参数账（全部整数精确算）。"""
    ml, mh, l2, l1 = 576, 32768, 60, 95                      # MLA/同构MHA 每层元素；V2/67B 层数
    gqa67 = 2 * 8 * 128                                       # 67B GQA-8 每层 2048
    arch = 1 - ml / mh                                        # 架构口径：同构 MHA 反事实
    cross = 1 - (ml * l2) / (gqa67 * l1)                      # 跨机 bf16：vs DeepSeek 67B
    deploy = 1 - (ml * l2 * 6) / (gqa67 * l1 * 16)            # 部署口径：V2 侧 6bit KV、67B 侧 bf16
    groups = 576 / (2 * 128)                                  # 金句「2.25 组 GQA」复算
    lite_frac = (576 * 27) / (2 * 16 * 128 * 27)              # Lite 几何 vs 同构 MHA
    # 每层参数账（DSV2-236B 维度，公式）：MLA vs GQA-8
    d, nh, dh, dv, dc, dcr, dr = 5120, 128, 128, 128, 512, 1536, 64
    mla_p = (d * dcr + dcr + dcr * nh * (dh + dr)             # q 侧低秩 + LN_q
             + d * (dc + dr) + dc                             # [W^DKV;W^KR] + LN_c
             + dc * nh * (dh + dv)                            # [W^UK;W^UV]
             + nh * dv * d)                                   # W^O
    gqa_p = d * nh * dh * 2 + d * dh * 8 * 2                  # q/o 全宽（各 16384×5120）+ k/v 8 份
    print(f"[Part3 三口径] 架构口径（同构 MHA 反事实）= 1−576/32768 = {arch*100:.2f}%  ← 本章主数字")
    print(f"[Part3 三口径] 跨机 bf16（vs DeepSeek 67B GQA-8）= 1−34,560/194,560 = {cross*100:.2f}%")
    print(f"[Part3 三口径] 部署口径（V2 侧 MLA+6bit KV、67B 侧 bf16）= 1−(576·60·6)/(2048·95·16)"
          f" = {deploy*100:.2f}%  ← 论文摘要 93.3% 的复算（复合：压缩×层数差×精度）")
    print(f"[Part3 金句复算] MLA 每层 576 元素 = 2·128·{groups} → 等效 {groups} 组的 GQA（论文口径 2.25 ✓）")
    print(f"[Part3 Lite 对 Table 9] {576*27:,}/{2*16*128*27:,} = {lite_frac*100:.2f}% ≈ 论文 App D.2 的 14%")
    print(f"[Part3 每层参数账] MLA {mla_p:,} vs GQA-8 {gqa_p:,}（DSV2 维度，省 {(1-mla_p/gqa_p)*100:.1f}%）")
    assert round(arch * 100, 2) == 98.24 and round(cross * 100, 2) == 82.24
    assert round(deploy * 100, 2) == 93.34 and groups == 2.25
    assert round(lite_frac * 100, 2) == 14.06
    return {"arch_pct": round(arch * 100, 2), "cross_pct": round(cross * 100, 2),
            "deploy_pct": round(deploy * 100, 2), "gqa_equiv_groups": groups,
            "lite_frac_pct": round(lite_frac * 100, 2),
            "mla_params_per_layer": mla_p, "gqa8_params_per_layer": gqa_p,
            "mla_param_saving_pct": round((1 - mla_p / gqa_p) * 100, 1)}


def main(out_name):
    print(f"[环境] CPU | KV 账单位=元素数（DSV2 Table 1 口径）| bf16 字节数仅在标注处出现")
    report = {"part0_hand_example": part0_hand_example(),
              "part1_bytes_measured": part1_bytes_measured(),
              "part2_geometry_table": part2_geometry_table(),
              "part3_three_calibers": part3_three_calibers()}
    report["verdict"] = "PASS"
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"kv_probe_mla_{out_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[产物] {path}")
    print("[判定] PASS（全部 assert 通过）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch6 KV 账本 probe：手算=实测 + 三口径红线算术")
    ap.add_argument("--out-name", default="run1", help="产物 JSON 后缀（防覆写）")
    args = ap.parse_args()
    main(args.out_name)
