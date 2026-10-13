# alpha_recon.py —— ch2 实验 C：α 理论对账后处理（Book6 α 账的引擎侧兑现；CPU 可跑）
# 所属章节：Book8 ch2 §2.5；设计书=notes/11 §四（实验 C）——消费 A/B 底档，不新跑负载。
# 判据预注册（结果分支）：α̂<0.3 → Book6 负结果主线；0.3≤α̂≤0.6 → 中间带（两层账分读）；
#   α̂>0.6 → 正结果（加速比若仍<1 → 归因 B=1 小批固有开销）；残差>15% 逐行归因。
# 口径：mean_acceptance_length=1+num_accepted/num_drafts（含 bonus）；
#   α̂ 数值解 (1-α̂^{k+1})/(1-α̂)=τ_obs；ĉ=(t_spec_step−t_base_step)/k 从 bench ITL 反解（粗估标注）；
#   speedup_theory=τ_obs/(k·ĉ+1)。
# 运行：cd <workspace> && source env.sh && python code/Book8-vLLM源码剖析下/ch02/alpha_recon.py [--tier fast]
# 产物：log/book8-ch02/c_alpha_recon.json（表 2.2/例 2.4 对账的数据源）
import argparse
import json
import re
import os
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(WS, "log/book8-ch02")


def parse_metrics(path: str) -> dict:
    """从 /metrics 快照取三个 spec counter（ exposition 带 _total 后缀）。"""
    vals = {}
    if not os.path.exists(path):
        return vals
    for line in open(path, encoding="utf-8"):
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("vllm:spec_decode_"):
            key = parts[0].split("{")[0].replace("vllm:", "").replace("_total", "")
            try:
                vals[key] = float(parts[1])
            except ValueError:
                pass
    return vals


def parse_speclog(path: str) -> list:
    """周期日志行原文（τ/per-pos 的直接口径）。"""
    if not os.path.exists(path):
        return []
    return [l.rstrip() for l in open(path, encoding="utf-8") if "Mean acceptance" in l]


def solve_alpha(tau_obs: float, k: int):
    """数值解 (1-α^{k+1})/(1-α)=τ_obs（τ∈(1, k+1] 有解；边界夹到求解域内）。"""
    tau_c = min(max(tau_obs, 1.0 + 1e-9), k + 1 - 1e-9)
    f = lambda a: (1 - a ** (k + 1)) / (1 - a) - tau_c
    try:
        return round(brentq(f, 1e-9, 0.9999), 4)
    except Exception:
        return None


def bench_itl(path: str):
    """bench JSON 的平均 ITL/TPOT（ms）——ĉ 反解用（粗估口径：ITL≈每输出 token 均时）。"""
    if not os.path.exists(path):
        return None, None
    d = json.load(open(path, encoding="utf-8"))
    itl = d.get("mean_itl_ms") or d.get("itl_mean", d.get("mean_itl_ms"))
    tpot = d.get("mean_tpot_ms") or d.get("output_throughput") and None
    keys = {k: v for k, v in d.items() if "itl" in k.lower() or "tpot" in k.lower() or "throughput" in k.lower()}
    return itl, keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="fast")
    a = ap.parse_args()

    arms = []
    # 实验 A（ngram）：k/窗口从臂名解析
    for fn in sorted(os.listdir(OUT)):
        if not (fn.startswith("a_") and fn.endswith("_metrics.txt")):
            continue
        arm = fn[2:-len("_metrics.txt")]
        if arm in ("off_echo",):
            continue
        m = parse_metrics(os.path.join(OUT, fn))
        if not m:
            continue
        k = int(arm[1]) if arm[:1] == "k" and arm[1:2].isdigit() else 1   # A 臂名 k3_w28_echo
        drafts, draft_tok, accepted = m.get("spec_decode_num_drafts", 0), m.get("spec_decode_num_draft_tokens", 0), m.get("spec_decode_num_accepted_tokens", 0)
        if not drafts:
            continue
        tau = 1 + accepted / drafts
        arms.append(dict(route="ngram", arm=arm, k=k, num_drafts=drafts,
                         num_draft_tokens=draft_tok, num_accepted=accepted,
                         mean_acceptance_length=round(tau, 3),
                         alpha_hat=solve_alpha(tau, k)))
    # 实验 B（draft_model）
    for fn in sorted(os.listdir(OUT)):
        if not (fn.startswith("b_") and fn.endswith("_metrics.txt")):
            continue
        arm = fn[2:-len("_metrics.txt")]
        if arm == "b_off":
            continue
        m = parse_metrics(os.path.join(OUT, fn))
        if not m:
            continue
        k = int(arm[3]) if arm.startswith("b_k") and arm[3:4].isdigit() else 1   # B 臂名 b_k3
        drafts, draft_tok, accepted = m.get("spec_decode_num_drafts", 0), m.get("spec_decode_num_draft_tokens", 0), m.get("spec_decode_num_accepted_tokens", 0)
        if not drafts:
            continue
        tau = 1 + accepted / drafts
        arms.append(dict(route="draft_model", arm=arm, k=k, num_drafts=drafts,
                         num_draft_tokens=draft_tok, num_accepted=accepted,
                         mean_acceptance_length=round(tau, 3),
                         alpha_hat=solve_alpha(tau, k)))

    # 加速比粗账（若有 bench JSON 的吞吐）：以 off 臂为基线
    def thr(path):
        try:
            d = json.load(open(path, encoding="utf-8"))
            for key in ("request_throughput", "output_throughput", "total_token_throughput"):
                if key in d:
                    return d[key]
        except Exception:
            pass
        return None

    base_a = thr(os.path.join(OUT, "a_off_echo.json"))   # 分路线基线——跨实验不混算
    base_b = thr(os.path.join(OUT, "b_b_off.json"))
    for r in arms:
        prefix = "a_" if r["route"] == "ngram" else "b_"
        base = base_a if r["route"] == "ngram" else base_b
        t = thr(os.path.join(OUT, f"{prefix}{r['arm']}.json"))
        if t and base:
            r["speedup_meas"] = round(t / base, 3)
            k = r["k"]
            if r["alpha_hat"] is not None:
                # ĉ 反解：speedup=τ/(k·ĉ+1) → ĉ=(τ/speedup−1)/k（加速比>1 时可解）
                s = max(r["speedup_meas"], 1e-6)
                c_hat = (r["mean_acceptance_length"] / s - 1) / k if r["mean_acceptance_length"] / s > 1 else None
                r["c_hat"] = round(c_hat, 4) if c_hat is not None and c_hat > 0 else None

    res = dict(probe="c_alpha_recon", tier=a.tier, baseline_arm="off",
               criterion="α̂<0.3 负结果主线 / 0.3-0.6 中间带 / >0.6 正结果；残差>15% 逐行归因",
               rows=arms, date="2026-10-12")
    path = os.path.join(OUT, "c_alpha_recon.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1)[:1500])
    print("[产物]", path)


if __name__ == "__main__":
    main()
