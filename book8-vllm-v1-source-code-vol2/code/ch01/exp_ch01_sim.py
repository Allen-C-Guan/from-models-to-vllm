# exp_ch01_sim.py —— ch1 实验 C：混合组仿真四本账（纯 CPU，逐位判据预注册）
# 所属章节：Book8 ch1（§1.8）；设计书=notes/13 §四 实验 C（toy 数字全预注册——不闭合不出数）
# 仿真机：8 层=6 SWA+2 FA、b=128（NPU 锁定档）、W 可扫、每 token 每层 KV 2048B。
# 四账：①块布局（例 1.1：38/94 vs 32/256）②显存预算（n=4096→23.5MiB vs 64MiB，省 63.3%）
#       ③命中率阶梯（W=128/256/512/1024→8/16/32/56%）④mamba 三态（1/2/32 页）。
# 运行：source env.sh && python code/Book8-vLLM源码剖析下/ch01/exp_ch01_sim.py [--full]
# 产物：log/book8-ch01/exp_ch01_sim.json
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(WS, "log/book8-ch01")
B = 128                      # NPU 锁定块（APC/chunked 下 16 不可实测——Book7 负结果沿用）
CTOK = 2048                  # 2(K/V)×8 kv_heads×64 head_dim×2B(bf16)=2048B（与 Book7 207M 账同源）
L_FA, L_SW = 2, 6            # 8 层=2 FA+6 SWA（3:1）


def cdiv(a, b):
    return -(-a // b)


def ledger1(W):
    """块布局账：n=1232（例 1.1 workload 全长 1000+200+32）下 SWA 组 vs FA 组的常驻块。
    SWA：滚动窗 W→常驻 cdiv(W,B)+1 块/层（近似：窗跨满块数+边界）；FA：随长度增长 cdiv(n,B)。
    预注册：W=512→SWA 常驻 5 块×6 层=30；FA 12 层口径对照 94 块（本机 2 层→对照缩放）。"""
    n = 1232
    sw_blocks = (cdiv(W, B) + 1) * L_SW
    fa_blocks = cdiv(n, B) * L_FA
    return dict(W=W, n=n, sw_blocks=sw_blocks, fa_blocks=fa_blocks,
                pure_fa_12l=cdiv(n, B) * 12)


def ledger2(n=4096):
    """显存预算账：稳态每请求 KV 字节。FA: L_fa*n*CTOK；SWA: L_sw*(cdiv(W-1,B)+1)*B*CTOK。
    预注册：W=512→2*4096*2048+6*5*128*2048=16MiB+7.5MiB=23.5MiB（纯 FA 8 层=64MiB，省 63.3%）。"""
    W = 512
    fa_bytes = L_FA * n * CTOK
    sw_bytes = L_SW * (cdiv(W - 1, B) + 1) * B * CTOK
    total = fa_bytes + sw_bytes
    pure_fa = (L_FA + L_SW) * n * CTOK
    return dict(n=n, W=W, fa_mib=round(fa_bytes / 2**20, 1), sw_mib=round(sw_bytes / 2**20, 1),
                total_mib=round(total / 2**20, 1), pure_fa_mib=round(pure_fa / 2**20, 1),
                saving_pct=round((1 - total / pure_fa) * 100, 1))


def ledger3():
    """命中率阶梯：prefix 1000（7×128+104）、suffix+输出 232、4 前缀×4 请求。
    FA 天花板=prefix 可缓存 token；混合=min(FA 组, SWA 组可常驻前缀窗)。
    预注册：W=128/256/512/≥1024→8.0/16.0/32.0/56.0%。"""
    # 口径（Book7 例 6.4 同 workload）：4 前缀×4 请求，每前缀首条不命中（12 条可命中）；
    # queries=16 请求×1200 prompt token=19200；hits=12×min(FA 可缓存 896, SWA 常驻窗 W)。
    # 验证：W=128→12×128/19200=8.0%；256→16.0%；512→32.0%；W≥1024→min(896,W)=896→56.0%（纯 FA 天花板）。
    rows = []
    prefix_tok = 896                      # 7 满块×128（块粒度：不满块不缓存）
    n_cached_reqs, n_reqs, prompt_tok = 12, 16, 1200
    queries = n_reqs * prompt_tok
    for W, expect in ((128, 8.0), (256, 16.0), (512, 32.0), (1024, 56.0), (4096, 56.0)):
        merged = min(prefix_tok, W)
        rate = round(n_cached_reqs * merged / queries * 100, 1)
        rows.append(dict(W=W, merged_tokens=merged,
                         hit_rate_pct=rate, preregistered=expect,
                         match=abs(rate - expect) <= 0.1))
    return rows


def ledger4():
    """mamba 三态账：固定状态页数（toy：1 页=256KB 状态/层）。"""
    PAGE = 256 * 1024
    return dict(page_bytes=PAGE, states=("1 页/层（小状态）", "2 页/层（预跑 draft 对照）", "32 页/层（K3 级）"),
                per_layer_mib=[round(PAGE * k / 2**20, 1) for k in (1, 2, 32)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    l1 = [ledger1(W) for W in ((512,) if not a.full else (128, 256, 512, 1024, 4096))]
    l2 = ledger2()
    l3 = ledger3()
    l4 = ledger4()
    # 逐位判据（预注册）
    checks = {
        "l1_sw_blocks_w512": l1[0]["sw_blocks"] == 30,
        "l2_total_23_5": abs(l2["total_mib"] - 23.5) <= 0.05,
        "l2_saving_63_3": abs(l2["saving_pct"] - 63.3) <= 0.1,
        "l3_ladder_all_match": all(r["match"] for r in l3),
    }
    res = dict(probe="exp_ch01_sim", b=B, ctok=CTOK, layers=dict(fa=L_FA, sw=L_SW),
               ledger1_blocks=l1, ledger2_budget=l2, ledger3_hitrate=l3, ledger4_mamba=l4,
               checks=checks, all_pass=all(checks.values()), date="2026-10-12")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "exp_ch01_sim.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(checks, ensure_ascii=False), "| all_pass =", res["all_pass"])
    print("[产物]", os.path.join(OUT, "exp_ch01_sim.json"))
    if not res["all_pass"]:
        print("⚠ 仿真与手算不闭合——按预注册：修复前不得出数（对照 Book7 块账四动作语义查满块/断链实现）")


if __name__ == "__main__":
    main()
