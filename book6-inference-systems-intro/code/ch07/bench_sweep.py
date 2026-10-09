# bench_sweep.py —— Book6 ch7 代码件：四版引擎压测收官（吞吐-延迟总曲线 + 增益分解）
# 用途：①汇总 v1-v4 各自 JSON 正式档读数（同 workload 可比）成对照总表与增益分解瀑布；
#   ②batch 扫描（v1/v2 在 B∈{1,2,4,8} 下的吞吐曲线——批处理红利随并发的变化）；
#   ③（可选 --with-vllm）vllm bench 趋势级对照：运行栈跑同量级小模型的单点读数——绝对数
#     不可比（模型不同、实现成熟度不同），只标「工业水位线」供趋势定位（黑盒对拍纪律）。
# 所属章节：Book6 ch7 §7.5（全书压测大戏）；设计书=plan/Book6-推理系统导论.md ch7
# 运行：cd <workspace> && source env.sh && ASCEND_RT_VISIBLE_DEVICES=<卡> \
#      python code/Book6-推理系统导论/ch07/bench_sweep.py [--fast-scan --out-name base]
# 产物：log/book6-ch07/bench_sweep_<out-name>.json + 增益分解表
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(ROOT, "log", "book6-ch07")
SEED = 20261002


def run_engine(script, extra, out_name):
    """调 v1/v2 引擎（同 workload 种子=可比）并读回 JSON。"""
    env = dict(os.environ)
    cmd = [sys.executable, script, "--out-name", out_name] + extra
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    tag = os.path.basename(script).replace("engine_", "").replace(".py", "")
    p = os.path.join(ROOT, "log", f"book6-ch0{4 if tag in ('v1','v2') else 5}", f"engine_{tag}_{out_name}.json")
    if not os.path.isfile(p):
        print(f"[warn] {tag} 无产物：{r.stdout[-300:]} {r.stderr[-300:]}")
        return None
    return json.load(open(p, encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="四版引擎压测收官")
    ap.add_argument("--fast-scan", action="store_true", help="只跑 B∈{1,4} 的扫描点（fast 档）")
    ap.add_argument("--with-vllm", action="store_true", help="加跑 vllm bench 趋势级单点（分钟级）")
    ap.add_argument("--out-name", default="base")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ① 汇总四版正式档（已在盘——同 workload 可比）
    summary = {}
    for tag, ch in (("v1", "book6-ch04"), ("v2", "book6-ch04"), ("v3", "book6-ch05"), ("v4", "book6-ch06")):
        p = os.path.join(ROOT, "log", ch, f"engine_{tag}_base.json")
        if os.path.isfile(p):
            summary[tag] = json.load(open(p, encoding="utf-8"))
        else:
            p2 = os.path.join(ROOT, "log", ch, f"engine_{tag}_base2.json")
            if os.path.isfile(p2):
                summary[tag] = json.load(open(p2, encoding="utf-8"))

    # ② batch 扫描（v1/v2——v3/v4 的调度同 v2，单点已有）
    scan = {}
    bs_list = [1, 4] if a.fast_scan else [1, 2, 4, 8]
    for B in bs_list:
        for tag in ("v1", "v2"):
            script = os.path.join(ROOT, "code", "Book6-推理系统导论", "ch04", f"engine_{tag}.py")
            flag = "--batch" if tag == "v1" else "--max-batch"
            r = run_engine(script, [flag, str(B)], f"sweep_b{B}")
            if r:
                scan[f"{tag}_B{B}"] = dict(throughput=r["throughput_tok_s"],
                                           tpot_ms=r.get("per_req_tpot_ms_avg") or r.get("per_req_tpot_ms"),
                                           waste=r.get("waste_ratio_pct", 0))

    # ③ 增益分解（以同 workload 的 v1→v2→v3→v4 读数串成「还账瀑布」）
    falls = []
    if "v1" in summary and "v2" in summary:
        falls.append(dict(step="v1→v2 连续批处理+KV cache", metric="浪费位",
                          v1=summary["v1"].get("waste_ratio_pct"), v2=0,
                          note=f"吞吐 {summary['v1']['throughput_tok_s']}→{summary['v2']['throughput_tok_s']} tok/s"
                               f"（口径注：v2 手写执行层含 python 开销——吞吐对比见扫描档而非绝对值）"))
    if "v2" in summary and "v3" in summary:
        falls.append(dict(step="v2→v3 分页 KV", metric="KV 显存",
                          v2=f"{summary['v2']['kv_bytes_total']/2**20:.0f} MiB",
                          v3=f"{summary['v3']['kv_account_v3_paged_bytes']/2**20:.0f} MiB",
                          note=f"省 {summary['v3']['kv_saving_pct']}%（碎片限最后一块内）"))
    if "v4" in summary:
        falls.append(dict(step="v3→v4 前缀缓存", metric="TTFT（多轮负载）",
                          v4=f"首轮 {summary['v4']['ttft_first_round_ms']} ms → 末轮 {summary['v4']['ttft_last_round_ms']} ms",
                          note=f"命中率 {summary['v4']['hit_rate_pct']}%"))

    out = {"summary": {k: {kk: v[kk] for kk in ("engine", "throughput_tok_s", "useful_tokens",
                                                 "wasted_slots", "peak_mem_MiB") if kk in v}
                      for k, v in summary.items()},
           "batch_scan": scan, "gain_waterfall": falls, "seed": SEED}
    p = os.path.join(OUT_DIR, f"bench_sweep_{a.out_name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[四版对照]")
    for k, v in summary.items():
        if k == "v4":
            print(f"  v4: 命中率 {v['hit_rate_pct']}% | TTFT {v['ttft_first_round_ms']}→{v['ttft_last_round_ms']} ms（多轮负载）")
            continue
        print(f"  {k}: 吞吐 {v.get('throughput_tok_s', '-')} tok/s | 浪费位 {v.get('wasted_slots', '-')}"
              f" | 峰值 {v.get('peak_mem_MiB', '-')} MiB")
    print("[增益瀑布]")
    for fw in falls:
        print(f"  {fw['step']}: {fw['metric']} {fw.get('v1') or fw.get('v2') or ''} -> {fw.get('v2') or fw.get('v3') or fw.get('v4')}"
              f" | {fw['note']}")
    print(f"[产物] {p}")


if __name__ == "__main__":
    main()
