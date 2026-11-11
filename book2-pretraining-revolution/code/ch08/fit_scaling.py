# fit_scaling.py —— Book2 ch8 Kaplan 单曲线拟合：L(N) = L∞ + A·N^(-α) 双拟合 + bootstrap CI
# 所属章节：Book2 第8章8.5节（drafts/Book2-预训练革命/ch08-ScalingLaws.md 表 8.3 数据源）
# 设计书：research/Book2-预训练革命/notes/01-实验可行性.md §4「双拟合方案（含/不含不可约项 E）」。
#   * 拟合 A（E=L∞ 自由，3 参数）：scipy curve_fit（有界 trf）；
#   * 拟合 B（固定 L∞=0，2 参数）：log-log 线性 OLS（ln L = ln A - α·ln N）；
#   * r² 一律在 L 尺度上算（两拟合同尺可比）；bootstrap 500 次（case 重采样，抽签退化则跳过并计数）；
#   * 诊断下界稳定性：报告贴边比例、条件重采样区间与参数变化，不以启发式阈值证明可辨识性。
# 运行方式：python fit_scaling.py [--in family_fast.json] [--out fit_kaplan] [--bootstrap 500]
# 产物：log/book2-ch08/{fit_kaplan.json, fit_curves.csv}

import argparse
import json
import os
from pathlib import Path
import hashlib

import numpy as np
from scipy import optimize

REPO_ROOT = os.getcwd()
OUT_DIR = os.path.join(REPO_ROOT, "log", "book2-ch08")
SEED = 20261002


def kaplan(N, Linf, A, alpha):
    return Linf + A * np.asarray(N, dtype=float) ** (-alpha)


def r2_on_L(y, yhat):
    y = np.asarray(y, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else None


def fit_free(N, L):
    """拟合 A：L∞/A/α 三参数自由（有界）。返回 (params, r2) 或 (None, None)。"""
    try:
        p, cov = optimize.curve_fit(
            kaplan, N, L, p0=[min(L) * 0.9, 10.0, 0.3],
            bounds=([0.0, 1e-8, 0.01], [10.0, 1e4, 5.0]), maxfev=200000)
        return p, r2_on_L(L, kaplan(N, *p))
    except (RuntimeError, ValueError, FloatingPointError):
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
    ap.add_argument("--log-dir", default=OUT_DIR)
    args = ap.parse_args()
    if args.bootstrap < 0:
        ap.error("--bootstrap不能为负数")
    if Path(args.out).name != args.out or args.out in ("", ".", ".."):
        ap.error("--out须为一个名称，不含目录")
    out_dir = os.path.abspath(args.log_dir)
    input_path = Path(args.inp)
    if not input_path.is_absolute():
        input_path = Path(out_dir) / input_path
    out_json = os.path.join(out_dir, f"{args.out}.json")
    csv = os.path.join(out_dir, f"{args.out}_curves.csv")
    if any(Path(path).exists() for path in (out_json, csv)):
        raise FileExistsError("拟合产物已存在，请指定新的--out名称")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    with input_path.open(encoding="utf-8") as f:
        report = json.load(f)
    pts = sorted(report["points"], key=lambda p: p["N"])
    bad = [p.get("name","unknown") for p in pts
           if p.get("diverged") or p.get("completed") is False or p.get("eval_final") is None
           or (p.get("steps_planned") is not None and p.get("steps") != p["steps_planned"])]
    if bad:
        raise ValueError(f"未完成或无效终点不进入拟合：{bad}")
    N = np.array([p["N"] for p in pts], dtype=float)
    L = np.array([p["eval_final"] for p in pts], dtype=float)
    names = [p["name"] for p in pts]
    if (len(pts) < 3 or len(np.unique(N)) < 3 or not np.isfinite(N).all()
            or not np.isfinite(L).all() or (N <= 0).any() or (L <= 0).any()):
        raise ValueError("需要至少三个不同、有限、正的参数量，以及有限正的损失")
    if len(np.unique(N)) != len(N):
        raise ValueError("本脚本每个尺寸需一个终点；重复尺寸须先明确选择或聚合规则")
    if np.ptp(L) == 0:
        raise ValueError("损失完全相同，无法报告有意义的R²与下界稳定性")
    print("拟合输入：最终eval_loss；设备、评估量与实现条件应查看输入报告，不预设固定协议")
    for n, nn, ll in zip(names, N, L):
        print(f"  {n:>3s}  N={nn:>10,.0f}  L={ll:.4f}")

    # ---- 两个拟合 ----
    pA, r2A = fit_free(N, L)
    pB, r2B = fit_fixed0(N, L)
    if pA is None or not np.isfinite(pA).all() or not np.isfinite(pB).all():
        raise RuntimeError("拟合未得到有限参数，未写出结果")
    residA = L - kaplan(N, *pA)
    residB = L - kaplan(N, *pB)
    # 相邻尺寸的描述性log-log斜率；不能单凭它证明正下界或单模型收敛。
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
                if (p[0] <= 1e-6 or p[2] <= 0.0101 or p[0] >= 9.999
                        or p[1] <= 1.01e-8 or p[1] >= 9999. or p[2] >= 4.999):
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

    # ---- 下界稳定性诊断；不将阈值启发式当作可辨识性证明 ----
    span = float(L.max() - L.min())
    e_ci_width = (e_hi - e_lo) if e_hi is not None else None
    checks = {
        "Linf_interval_touches_lower_bound": bool(e_lo <= 1e-6) if e_lo is not None else None,
        "Linf_interval_width_vs_L_span": e_ci_width / span if e_ci_width is not None else None,
        "delta_r2_free_minus_fixed": r2A - r2B,
        "alpha_shift_free_minus_fixed": float(pA[2] - pB[2]),
        "log10_N_span": float(np.log10(N.max()/N.min())),
        "loss_span": span,
    }
    if not draws["Linf_A"]:
        verdict = "未得到自由下界的有效重采样区间；不能据此判断下界稳定"
    elif checks["Linf_interval_touches_lower_bound"] or checks["Linf_interval_width_vs_L_span"] >= .5:
        verdict = "存在贴边或宽区间信号：此组尺寸点不能稳定约束损失下界"
    else:
        verdict = "本次启发式诊断未见上述不稳信号；不等于证明下界可辨识或外推可靠"

    out = {
        "meta": {"date": report.get("meta",{}).get("date"), "seed": SEED,
                 "input": str(input_path), "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                 "loss_def": "输入报告中的最终eval_loss；原协议元数据一并保留",
                 "source_meta": report.get("meta",{}),
                 "fit_domain": "N沿用输入定义，本脚本不推断是否排除了嵌入",
                 "fit_objectives": {"free": "raw-loss squared residual",
                                    "fixed0": "log-loss squared residual"},
                 "bootstrap": f"case重采样{args.bootstrap}次，每次从{len(N)}个尺寸点有放回抽取",
                 "uncertainty_scope": "条件于当前尺寸点；不包含训练重跑或新语料的不确定性"},
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
    with open(out_json, "x", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)

    # ---- 曲线数据（ch8 作图直接可用）：log 网格 + 两拟合曲线 + 实测点 ----
    grid = np.logspace(np.log10(N.min() * 0.9), np.log10(N.max() * 1.1), 120)
    with open(csv, "x", encoding="utf-8") as f:
        f.write("kind,N,L\n")
        for nn, ll in zip(N, L):
            f.write(f"measured,{nn:.0f},{ll:.6f}\n")
        for nn in grid:
            f.write(f"fit_free,{nn:.1f},{kaplan(nn, *pA):.6f}\n")
            f.write(f"fit_fixed0,{nn:.1f},{kaplan(nn, *pB):.6f}\n")

    def interval(lo, hi):
        return "未估计" if lo is None or hi is None else f"[{lo:.4f}, {hi:.4f}]"
    print(f"拟合A（原loss目标）：E={pA[0]:.4f} A={pA[1]:.4f} alpha={pA[2]:.4f} r²={r2A:.5f}")
    print(f"  条件case重采样alpha区间 {interval(aA_lo,aA_hi)}；"
          f"跳过{draws['skipped_A']}/{args.bootstrap}，贴界{draws['unstable_A_bounds']}")
    print(f"  下界区间 {interval(e_lo,e_hi)}")
    print(f"拟合B（log-loss目标，E=0）：A={pB[1]:.4f} alpha={pB[2]:.4f} r²={r2B:.5f}")
    print(f"  条件case重采样alpha区间 {interval(aB_lo,aB_hi)}；跳过{draws['skipped_B']}/{args.bootstrap}")
    print(f"局部log-log斜率：{[round(v,3) for v in local_slope]}")
    print("下界稳定性诊断：", verdict)
    print("checks:", json.dumps(checks,ensure_ascii=False))
    print(f" -> {out_json} / {csv}")


if __name__ == "__main__":
    main()
