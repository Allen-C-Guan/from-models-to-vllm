# fit_scaling.py —— Book2 ch8 Kaplan 单曲线拟合：L(N) = L∞ + A·N^(-α) 双拟合 + bootstrap CI
# 所属章节：Book2 第 8 章 §8.3（ch08-ScalingLaws.md 表 8.3 数据源）
# 设计书：notes/01-实验可行性.md §4「双拟合方案（含/不含不可约项 E）」。
#   * 拟合 A（E=L∞ 自由，3 参数）：scipy curve_fit（有界 trf）；
#   * 拟合 B（固定 L∞=0，2 参数）：log-log 线性 OLS（ln L = ln A - α·ln N）；
#   * r² 一律在 L 尺度上算（两拟合同尺可比）；bootstrap 500 次（case 重采样，抽签退化则跳过并计数）；
#   * 判定「E 在本动态范围可否辨识」：L∞ 的 bootstrap CI 是否含 0/贴边界、Δr² 是否边际、α 两拟合是否漂移。
# 运行方式：python fit_scaling.py [--in family_fast.json] [--out fit_kaplan] [--bootstrap 500]
# 产物：log/book2-ch08/{fit_kaplan.json, fit_curves.csv}

import argparse
import json
import os

import numpy as np
from scipy import optimize

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book2-ch08")
SEED = 20261002


def kaplan(N, Linf, A, alpha):
    return Linf + A * np.asarray(N, dtype=float) ** (-alpha)


def r2_on_L(y, yhat):
    y = np.asarray(y, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_free(N, L):
    """拟合 A：L∞/A/α 三参数自由（有界）。返回 (params, r2) 或 (None, None)。"""
    try:
        p, cov = optimize.curve_fit(
            kaplan, N, L, p0=[min(L) * 0.9, 10.0, 0.3],
            bounds=([0.0, 1e-8, 0.01], [10.0, 1e4, 5.0]), maxfev=200000)
        return p, r2_on_L(L, kaplan(N, *p))
    except Exception:
        return None, None


def fit_fixed0(N, L):
    """拟合 B：固定 L∞=0 的 log-log OLS。返回 (params[Linf=0, A, alpha], r2_on_L)。"""
    lnN, lnL = np.log(N), np.log(L)
    b1, b0 = np.polyfit(lnN, lnL, 1)          # ln L = b1*ln N + b0
    A, alpha = float(np.exp(b0)), float(-b1)
    return np.array([0.0, A, alpha]), r2_on_L(L, kaplan(N, 0.0, A, alpha))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="family_fast.json")
    ap.add_argument("--out", default="fit_kaplan")
    ap.add_argument("--bootstrap", type=int, default=500)
    args = ap.parse_args()

    with open(os.path.join(OUT_DIR, args.inp), encoding="utf-8") as f:
        report = json.load(f)
    pts = sorted(report["points"], key=lambda p: p["N"])
    N = np.array([p["N"] for p in pts], dtype=float)
    L = np.array([p["eval_final"] for p in pts], dtype=float)
    names = [p["name"] for p in pts]
    assert len(pts) >= 3, "点数不足以拟合"
    print("拟合输入（eval_final，fp32，401,408 token/点）：")
    for n, nn, ll in zip(names, N, L):
        print(f"  {n:>3s}  N={nn:>10,.0f}  L={ll:.4f}")

    # ---- 两个拟合 ----
    pA, r2A = fit_free(N, L)
    pB, r2B = fit_fixed0(N, L)
    residA = L - kaplan(N, *pA)
    residB = L - kaplan(N, *pB)
    # 相邻点局部 log-log 斜率（描述性：随 N 增大幂律是否变平——L∞ 存在的指纹）
    local_slope = (np.diff(np.log(L)) / np.diff(np.log(N))).tolist()

    # ---- bootstrap（case 重采样；退化抽签跳过并计数） ----
    rng = np.random.default_rng(SEED)
    draws = {"alpha_A": [], "Linf_A": [], "A_A": [], "alpha_B": [], "A_B": [],
             "skipped_A": 0, "skipped_B": 0, "unstable_A_bounds": 0, "n_draws": args.bootstrap}
    for _ in range(args.bootstrap):
        idx = rng.integers(0, len(N), len(N))
        uN, uL = N[idx], L[idx]
        uniq = np.unique(uN)
        if len(uniq) >= 3:
            p, _ = fit_free(uN, uL)
            if p is None:
                draws["skipped_A"] += 1
            else:
                draws["alpha_A"].append(p[2]); draws["Linf_A"].append(p[0]); draws["A_A"].append(p[1])
                if p[0] <= 1e-6 or p[2] <= 0.0101 or p[0] >= 9.999:   # 贴参数边界 = 不稳定
                    draws["unstable_A_bounds"] += 1
        else:
            draws["skipped_A"] += 1
        if len(uniq) >= 2:
            p, _ = fit_fixed0(uN, uL)
            draws["alpha_B"].append(p[2]); draws["A_B"].append(p[1])
        else:
            draws["skipped_B"] += 1

    def ci(vals, lo=2.5, hi=97.5):
        if not vals:
            return None, None, None
        v = np.array(vals)
        return (float(np.percentile(v, lo)), float(np.percentile(v, hi)), float(np.median(v)))

    aA_lo, aA_hi, aA_med = ci(draws["alpha_A"])
    e_lo, e_hi, e_med = ci(draws["Linf_A"])
    AA_lo, AA_hi, AA_med = ci(draws["A_A"])
    aB_lo, aB_hi, aB_med = ci(draws["alpha_B"])
    AB_lo, AB_hi, AB_med = ci(draws["A_B"])

    # ---- E 可辨识性判定 ----
    span = float(L.max() - L.min())
    e_ci_width = (e_hi - e_lo) if e_hi is not None else float("nan")
    checks = {
        "Linf_CI_contains_zero": e_lo is not None and e_lo <= 0.0,
        "Linf_CI_width_vs_L_span": e_ci_width / span if span > 0 else float("nan"),
        "delta_r2_free_minus_fixed": r2A - r2B,
        "alpha_shift_free_minus_fixed": pA[2] - pB[2],
    }
    identifiable = (
        (not checks["Linf_CI_contains_zero"])
        and checks["Linf_CI_width_vs_L_span"] < 0.5
        and checks["delta_r2_free_minus_fixed"] > 0.01
    )
    verdict = ("可辨识" if identifiable else
               "不可辨识（本动态范围 N 仅跨 1.4 个数量级、L 降幅有限：L∞ 与 (A,α) 强相关，"
               "三参数拟合自由度不足——与 Kaplan 原文需 6+ 数量级 N 支撑 L∞ 的观察同因）")

    out = {
        "meta": {"date": report["meta"]["date"], "seed": SEED,
                 "input": args.inp, "loss_def": "每点最终一次 eval loss（401,408 token，fp32 前向）",
                 "fit_domain": "N=非嵌入参数（6ND 口径同族）", "bootstrap": "case 重采样 500 次（5 点有放回）"},
        "points": [{"name": n, "N": float(nn), "L": float(ll)}
                   for n, nn, ll in zip(names, N, L)],
        "fit_free": {"params": {"L_inf": pA[0], "A": pA[1], "alpha": pA[2]},
                     "r2_on_L": r2A, "residuals": residA.tolist(),
                     "bootstrap_CI": {"alpha": [aA_lo, aA_hi], "alpha_median": aA_med,
                                      "L_inf": [e_lo, e_hi], "L_inf_median": e_med,
                                      "A": [AA_lo, AA_hi], "A_median": AA_med,
                                      "skipped": draws["skipped_A"],
                                      "hit_bounds": draws["unstable_A_bounds"]}},
        "fit_fixed0": {"params": {"L_inf": 0.0, "A": pB[1], "alpha": pB[2]},
                       "r2_on_L": r2B, "residuals": residB.tolist(),
                       "bootstrap_CI": {"alpha": [aB_lo, aB_hi], "alpha_median": aB_med,
                                        "A": [AB_lo, AB_hi], "A_median": AB_med,
                                        "skipped": draws["skipped_B"]}},
        "local_loglog_slopes": local_slope,
        "identifiability": {"checks": checks, "verdict": verdict},
    }
    out_json = os.path.join(OUT_DIR, f"{args.out}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ---- 曲线数据（ch8 作图直接可用）：log 网格 + 两拟合曲线 + 实测点 ----
    grid = np.logspace(np.log10(N.min() * 0.9), np.log10(N.max() * 1.1), 120)
    csv = os.path.join(OUT_DIR, f"{args.out}_curves.csv")
    with open(csv, "w", encoding="utf-8") as f:
        f.write("kind,N,L\n")
        for nn, ll in zip(N, L):
            f.write(f"measured,{nn:.0f},{ll:.6f}\n")
        for nn in grid:
            f.write(f"fit_free,{nn:.1f},{kaplan(nn, *pA):.6f}\n")
            f.write(f"fit_fixed0,{nn:.1f},{kaplan(nn, *pB):.6f}\n")

    print(f"\n拟合 A（L∞ 自由）：L∞={pA[0]:.4f}  A={pA[1]:.4f}  α={pA[2]:.4f}  r²={r2A:.5f}")
    print(f"  bootstrap α 95% CI [{aA_lo:.4f}, {aA_hi:.4f}]（跳过 {draws['skipped_A']}/500，贴界 {draws['unstable_A_bounds']}）")
    print(f"  bootstrap L∞ 95% CI [{e_lo:.4f}, {e_hi:.4f}]")
    print(f"拟合 B（L∞=0）：A={pB[1]:.4f}  α={pB[2]:.4f}  r²={r2B:.5f}")
    print(f"  bootstrap α 95% CI [{aB_lo:.4f}, {aB_hi:.4f}]（跳过 {draws['skipped_B']}/500）")
    print(f"局部 log-log 斜率：{[round(s, 3) for s in local_slope]}")
    print(f"E 可辨识性：{verdict}")
    print(f"checks: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in checks.items()})}")
    print(f" -> {out_json} / {csv}")


if __name__ == "__main__":
    main()
