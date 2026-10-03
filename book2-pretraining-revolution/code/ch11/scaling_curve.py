# scaling_curve.py —— Book2 ch11 损失-算力曲线：自家点汇编 + Kaplan/Chinchilla 坐标换算 + 招牌图（11.4 节）
# 用途：把 ch08 Kaplan 族 5 点、ch11 iso-FLOP 网格 15 点、ch11 124M fast 轨迹汇到同一张损失-算力图上：
#   ①全部点换算 per-byte 口径（跨词表唯一合法口径：L_tok/(B/token)——50257 与 8192 词表不可直接比）；
#   ②compute-optimal 斜率 γ 三口径 band 与文献指数的同型对照（Kaplan 0.73 / Chinchilla iso-FLOP 0.49-0.51）；
#   ③124M@50M 的 compute-optimal 自诊（N* band / D* band / D:N 比——「官方架构在本预算下远离最优」的定量自证）；
#   ④ch9 换算盒兑现：全部点画进 PF-days / (C, N*) 论文坐标系（图 11.4 招牌图，混排图双源标注）。
# 数据源（只读，不重训）：log/book2-ch11/{iso_flop.json, train124_fast.json, data50k_meta.json}
#                          + log/book2-ch08/{family_fast.json, data_meta.json}
# 所属章节：Book2 第 11 章 11.4 节（图 11.4/11.5、表 11.2）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch11/scaling_curve.py"
# 产物：log/book2-ch11/{scaling_curve.json, scaling_points.csv}
#       + figures/{fig-11-4-loss-compute.png, fig-11-5-isoflop-valleys.png}
import csv
import json
import math
import os

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CH11 = os.path.join(REPO_ROOT, "log", "book2-ch11")
CH08 = os.path.join(REPO_ROOT, "log", "book2-ch08")
FIG_DIR = os.path.join(REPO_ROOT, "figures")
PF_DAY = 8.64e19                     # 1 PF-day = 10^15 FLOP/s × 86400 s（ch8 换算盒）
KAPLAN_GAMMA = 0.73                  # Kaplan N*(C)∝C^0.73（摘要/§6.1/Table 5，03-引用复核清单 警-4 已复核）
CHINCHILLA_GAMMA = 0.50              # Chinchilla iso-FLOP 法 0.49/0.51（§3.2/Table 2，警-2 已复核）


def load_all():
    with open(os.path.join(CH11, "iso_flop.json"), encoding="utf-8") as f:
        iso = json.load(f)
    with open(os.path.join(CH11, "train124_fast.json"), encoding="utf-8") as f:
        t124 = json.load(f)
    with open(os.path.join(CH11, "data50k_meta.json"), encoding="utf-8") as f:
        m50k = json.load(f)
    with open(os.path.join(CH08, "family_fast.json"), encoding="utf-8") as f:
        fam = json.load(f)
    with open(os.path.join(CH08, "data_meta.json"), encoding="utf-8") as f:
        m8k = json.load(f)
    return iso, t124, m50k, fam, m8k


def build_points(iso, t124, m50k, fam, m8k):
    """全自家点 -> 统一行（C, N, D, L_tok, B_per_token, L_per_byte）。跨词表红线：只用 per-byte 并比。"""
    bpt8k = m8k["encoding"]["bytes_per_token"]            # 3.820（sp-8k 全语料）
    bpt50k = m50k["encoding"]["bytes_per_gpt2_token"]     # 4.398（gpt2 50257，56k 篇）
    rows = []
    for p in fam["points"]:                               # ch08 Kaplan 族（固定 D=40M）
        D = p["d_tokens_processed"]                        # 39,993,344（ch08 字段名）
        C = 6.0 * p["N"] * D
        rows.append(dict(family="ch08_kaplan", name=p["name"], C=C, N=p["N"], D=D,
                         L_tok=p["eval_final"], bpt=bpt8k, L_by=p["eval_final"] / bpt8k))
    for p in iso["points"]:                               # ch11 iso-FLOP 网格（D=C/6N）
        rows.append(dict(family=f"iso_{p['level']}", name=p["name"], C=p["C_actual"], N=p["N"],
                         D=p["D_tokens"], L_tok=p["eval_final"], bpt=bpt8k, L_by=p["eval_final"] / bpt8k))
    r = t124["run"]                                       # ch11 124M fast（gpt2 50257 词表）
    C124 = 6.0 * r["model"]["params_non_embedding"] * r["D_tokens"]
    rows.append(dict(family="gpt2_124m_fast", name="124m@50M", C=C124, N=r["model"]["params_non_embedding"],
                     D=r["D_tokens"], L_tok=r["eval_final"], bpt=bpt50k, L_by=r["eval_final"] / bpt50k))
    return rows


def optimal_analysis(iso, t124):
    """γ 三口径 band + 124M@50M 的 compute-optimal 自诊 + c2a vs ch08-20m 对比。"""
    sf = iso["slope_fit"]
    gamma = sf["slope_band"]                              # [0.855, 1.014]（argmin 下界-混合上界）
    verts = {a["level"]: a for a in sf["anchors"]}
    c3C, c3N = 10 ** verts["c3"]["log10C"], 10 ** verts["c3"]["vertex"]
    r = t124["run"]
    C = 6.0 * r["model"]["params_non_embedding"] * r["D_tokens"]
    lo, hi = gamma
    n_star = [c3N * (C / c3C) ** lo, c3N * (C / c3C) ** hi]
    d_star = [C / (6.0 * n) for n in n_star]
    return {
        "gamma_band": gamma,
        "gamma_views": {k: sf[k]["slope"] for k in ("slope_argmin_all", "slope_ingrid_vertices", "slope_mixed")},
        "lit_compare": {"kaplan": KAPLAN_GAMMA, "chinchilla_isoflop": (0.49, 0.51),
                        "stance": "同型对照不比大小：本实验 C 跨 1 个数量级、N<=23M、OWT/sp-8k 缩小版；"
                                  "日程形状跨点不一致 + 未逐点调 lr 两项混淆如实挂账"},
        "D_over_N": sf["D_over_N_at_optimum"]["ratios_tokens_per_param"],
        "L_star_power_in_C": sf.get("L_star_power_in_C_ingrid"),
        "diag_124m": {"C": C, "N_actual": r["model"]["params_non_embedding"],
                      "N_star_band": [round(n, 0) for n in n_star],
                      "D_star_band_tok": [round(d, 0) for d in d_star],
                      "D_over_N_star": [round(d / n, 1) for d, n in zip(d_star, n_star)],
                      "verdict": "官方 124M（非嵌入 85.06M）在本预算的 compute-optimal N*（band 21-26M）之上 3.2-4 倍"
                                 "——D:N=0.6:1 严重欠数据，compute-limited 演示口径的定量自证"},
        "c2a_vs_20m": {"iso_c2a": next(p["eval_final"] for p in iso["points"] if p["name"] == "c2a"),
                       "c2a_C": next(p["C_actual"] for p in iso["points"] if p["name"] == "c2a"),
                       "ch08_20m": 4.50131, "ch08_20m_C": 6.0 * 19_519_872 * 39_993_344,
                       "note": "c2a 用不到一半算力拿到更低 loss（同 sp-8k 同 eval 窗集，直接可比）——"
                               "ch8 固定 D=40M 堆 N 的路线远离 compute-optimal 的定量实锤（mini 版 Chinchilla 打 Kaplan）"},
    }


def convert_box(rows):
    """ch9 换算盒兑现：自家点 -> 论文坐标（PF-days；与 GPT-3 的比值）。
    批三修订 2026-10-03：原键 iso_total_C 误把「iso 15 点 + ch8 族 5 点」都计入 iso 合计
    （= C_all − 124M 一点），正文按字面误读——现拆为 iso15_only_* / ch08_family_* / family_total_* 三键。"""
    c124 = next(r for r in rows if r["name"] == "124m@50M")
    iso_c = sum(r["C"] for r in rows if r["family"].startswith("iso_"))
    fam_c = sum(r["C"] for r in rows if r["family"] == "ch08_kaplan")
    return {"pf_day": PF_DAY,
            "iso15_only_C": iso_c, "iso15_only_pf_days": iso_c / PF_DAY,
            "ch08_family_C": fam_c, "ch08_family_pf_days": fam_c / PF_DAY,
            "family_total_C": iso_c + fam_c, "family_total_pf_days": (iso_c + fam_c) / PF_DAY,
            "run124_C": c124["C"], "run124_pf_days": c124["C"] / PF_DAY,
            "run124_vs_gpt3": c124["C"] / 3.14e23,
            "note": "GPT-3 3.14e23 FLOPs = 3.64e3 PF-days（2005.14165 v4 Table D.1 + 本书算术）；"
                    "family_total = iso 15 点 + 第 8 章族 5 点（不含 124M 一跑）；"
                    "自家全部点画进 Kaplan/Chinchilla 的 (C, N*) 坐标系——坐标同一套，位置在左下角"}


def dump(rows, ana, box, iso):
    out = {"meta": {"date": "2026-10-03", "note": "scaling_curve.py 只汇编不重训；数字源=iso_flop.json/"
                                                 "train124_fast.json/family_fast.json（单种子，MPS bf16/fp32 eval）"},
           "optimal": ana, "convert_box": box,
           "level_fits": [{k: f[k] for k in ("level", "C_nominal", "n_points", "N_star", "N_star_inside_grid",
                                             "N_range", "eval_finals", "quad")} for f in iso["level_fits"]]}
    with open(os.path.join(CH11, "scaling_curve.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(os.path.join(CH11, "scaling_points.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["family", "name", "C_flops", "C_pf_days", "N_nonemb", "D_tokens", "L_tok_nats",
                    "B_per_token", "L_per_byte"])
        for r in rows:
            w.writerow([r["family"], r["name"], f"{r['C']:.4e}", f"{r['C'] / PF_DAY:.3e}", r["N"],
                        r["D"], r["L_tok"], r["bpt"], round(r["L_by"], 5)])
    print(f"[完成] scaling_curve.json + scaling_points.csv（{len(rows)} 点）")
    for r in rows:
        print(f"  {r['family']:<16} {r['name']:<10} C={r['C']:.3e} N={r['N']:>10,} L_tok={r['L_tok']:.4f} "
              f"L/byte={r['L_by']:.4f}")


def make_fig_11_5(rows, ana, iso):
    """招牌图（图 11.5，按正文出现顺序编号）：左=per-byte 损失-算力曲线（全自家点）；右=N*(C) 画进 Kaplan/Chinchilla 坐标系。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})
    BLUE, ORANGE, TEAL, YELLOW, GRID, TICK, DARK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e1e0d9", "#898781", "#0b0b0b"
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8, 4), dpi=300)

    # ---- A：损失-算力（per-byte） ----
    fam = [r for r in rows if r["family"] == "ch08_kaplan"]
    axA.plot([r["C"] for r in fam], [r["L_by"] for r in fam], "o--", color=BLUE, ms=5, lw=1.6,
             label="ch08 family (fixed D=40M)", zorder=3)
    for lvl, color in (("iso_c1", ORANGE), ("iso_c2", TEAL), ("iso_c3", YELLOW)):
        pts = sorted([r for r in rows if r["family"] == lvl], key=lambda r: r["N"])
        axA.plot([r["C"] for r in pts], [r["L_by"] for r in pts], "s", color=color, ms=5,
                 label=f"iso-FLOP {lvl[-2:].upper()} ({pts[0]['C']:.0e})".replace("e+0", "e"), zorder=4)
        # 每档谷点连线（compute-optimal 包络）
    opt_pts = []
    for lvl in ("iso_c1", "iso_c2", "iso_c3"):
        pts = [r for r in rows if r["family"] == lvl]
        best = min(pts, key=lambda r: r["L_tok"])
        opt_pts.append(best)
    axA.plot([r["C"] for r in opt_pts], [r["L_by"] for r in opt_pts], "^-", color=DARK, ms=6, lw=1.4,
             label="compute-optimal envelope", zorder=5)
    star = next(r for r in rows if r["name"] == "124m@50M")
    axA.plot([star["C"]], [star["L_by"]], "*", color="#8f3cc9", ms=13, label="GPT-2 124M @ 50M tok", zorder=6)
    # c2a vs ch08-20m 教学箭头
    c2a = next(r for r in rows if r["name"] == "c2a")
    f20 = next(r for r in rows if r["name"] == "20m")
    axA.annotate("half compute, lower loss", xy=(c2a["C"], c2a["L_by"]), xytext=(f20["C"] * 0.28, f20["L_by"] + 0.045),
                 fontsize=7, color="#52514e",
                 arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=0.9,
                                 connectionstyle="arc3,rad=-0.25"))
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xlabel("compute C = 6ND (FLOPs)", fontsize=8, color=DARK)
    axA.set_ylabel("loss per byte (nats/B, cross-vocab scale)", fontsize=8, color=DARK)
    axA.grid(color=GRID, lw=0.5, zorder=0); axA.tick_params(colors=TICK)
    axA.legend(fontsize=6.2, frameon=False, loc="upper right")
    axA.set_title("(A) loss vs compute (our points, per-byte)", fontsize=8.5, color=DARK, loc="left")

    # ---- B：N*(C) 与文献斜率 ----
    sf = iso["slope_fit"]
    verts = {a["level"]: a for a in sf["anchors"]}
    c2x, c2y = verts["c2"]["log10C"], verts["c2"]["vertex"]
    xs = np.linspace(c2x - 1.15, c2x + 0.75, 50)
    for g, color, ls, label in ((KAPLAN_GAMMA, "#898781", "--", f"Kaplan slope {KAPLAN_GAMMA}"),
                                (CHINCHILLA_GAMMA, "#b0aea6", ":", f"Chinchilla iso-FLOP slope 0.50")):
        axB.plot(10 ** xs, 10 ** (c2y + g * (xs - c2x)), ls, color=color, lw=1.6, label=label, zorder=2)
    lo, hi = ana["gamma_band"]
    axB.fill_between(10 ** xs, 10 ** (c2y + lo * (xs - c2x)), 10 ** (c2y + hi * (xs - c2x)),
                     color=TEAL, alpha=0.18, zorder=1, label=f"our band gamma in [{lo:.2f},{hi:.2f}]")
    # c2/c3 顶点 + CI
    for lvl, color in (("c2", TEAL), ("c3", YELLOW)):
        a = verts[lvl]
        ci = next(f.get("bootstrap_main", {}).get("N_star_ci95") for f in iso["level_fits"] if f["level"] == lvl)
        ci = (ci[0], ci[-1])          # bootstrap 分位数组=[2.5%, 50%, 97.5%]，取两端
        x, y = 10 ** a["log10C"], 10 ** a["vertex"]
        axB.errorbar([x], [y], yerr=[[y - ci[0]], [ci[1] - y]], fmt="o", color=color, ms=6,
                     capsize=3, lw=1.4, zorder=4,
                     label=f"{lvl} vertex N*={y/1e6:.2f}M" + (" (CI tight)" if lvl == "c2" else " (near-tie)"))
    # c1 只报界（上界=argmin、下界=一次通过池）
    a1 = verts["c1"]
    x1 = 10 ** a1["log10C"]
    axB.errorbar([x1], [10 ** a1["argmin"]], yerr=[[10 ** a1["argmin"] - 0.33e6], [0]], fmt="v", color=ORANGE,
                 ms=6, capsize=3, lw=1.4, zorder=4, label="c1: N*<0.60M (pool bound 0.33M)")
    axB.set_xscale("log"); axB.set_yscale("log")
    axB.set_xlabel("compute C (FLOPs)", fontsize=8, color=DARK)
    axB.set_ylabel("compute-optimal N* (non-emb params)", fontsize=8, color=DARK)
    axB.grid(color=GRID, lw=0.5, zorder=0); axB.tick_params(colors=TICK)
    axB.legend(fontsize=6.2, frameon=False, loc="upper left")
    axB.set_title("(B) N*(C): our band vs paper slopes", fontsize=8.5, color=DARK, loc="left")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-11-5-loss-compute.png")
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"[figure] -> {out}")


def make_fig_11_4(iso):
    """iso-FLOP 谷底拟合三联图（图 11.4）：c1（单调降，谷在界外）/ c2（干净 U）/ c3（近平局）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})
    BLUE, ORANGE, TEAL, GRID, TICK, DARK = "#2a78d6", "#eb6834", "#1baf7a", "#e1e0d9", "#898781", "#0b0b0b"
    fig, axes = plt.subplots(1, 3, figsize=(8, 3.1), dpi=300, sharey=False)
    for ax, f in zip(axes, iso["level_fits"]):
        pts = sorted([p for p in iso["points"] if p["level"] == f["level"]], key=lambda p: p["N"])
        x = np.log10(np.array([p["N"] for p in pts], dtype=float))
        y = np.array([p["eval_final"] for p in pts], dtype=float)
        ax.plot(10 ** x, y, "o", color=BLUE, ms=5, zorder=3)
        xs = np.linspace(x.min() - 0.12, x.max() + 0.12, 160)
        a, b, c = f["quad"]["a"], f["quad"]["b"], f["quad"]["c"]
        ax.plot(10 ** xs, a * xs ** 2 + b * xs + c, "-", color=ORANGE, lw=1.6, zorder=2)
        if f.get("N_star_inside_grid"):
            xv = math.log10(f["N_star"])
            ax.axvline(f["N_star"], color=TEAL, lw=1.2, ls="--", zorder=1)
            ci = f.get("bootstrap_main", {}).get("N_star_ci95")
            if ci:
                ax.axvspan(ci[0], ci[-1], color=TEAL, alpha=0.15, zorder=0)   # CI 数组已在线性尺度
            ax.text(0.97, 0.06, f"N*={f['N_star']/1e6:.2f}M", transform=ax.transAxes, fontsize=7,
                    color=TEAL, ha="right")
        else:
            ax.text(0.97, 0.88, "monotone: valley\noff-grid left;\nN*<0.60M only",
                    transform=ax.transAxes, fontsize=7, color=ORANGE, ha="right", va="top")
        if f["level"] == "c3":                     # 近平局阴影（c3a-c3b 差 0.020 nat≈5×eval se）
            ax.axvspan(min(p["N"] for p in pts), next(p["N"] for p in pts if p["name"] == "c3b"),
                       color=GRID, alpha=0.8, zorder=0)
            ax.text(0.97, 0.06, "near-tie floor", transform=ax.transAxes, fontsize=7, color="#52514e", ha="right")
        ax.set_xscale("log")
        ax.set_xlabel(f"N (non-emb), C={f['C_nominal']:.0e}".replace("e+0", "e"), fontsize=7.5, color=DARK)
        ax.grid(color=GRID, lw=0.5, zorder=0); ax.tick_params(colors=TICK, labelsize=7)
    axes[0].set_ylabel("final eval loss (nats/tok, sp-8k)", fontsize=8, color=DARK)
    fig.suptitle("iso-FLOP valleys: parabola fits per compute level (log N axis)", fontsize=9, color=DARK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(FIG_DIR, "fig-11-4-isoflop-valleys.png")
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"[figure] -> {out}")


def main():
    iso, t124, m50k, fam, m8k = load_all()
    rows = build_points(iso, t124, m50k, fam, m8k)
    ana = optimal_analysis(iso, t124)
    box = convert_box(rows)
    dump(rows, ana, box, iso)
    make_fig_11_4(iso)
    make_fig_11_5(rows, ana, iso)
    print(json.dumps({"gamma": ana["gamma_band"], "diag_124m": ana["diag_124m"],
                      "c2a_vs_20m": {k: v for k, v in ana["c2a_vs_20m"].items() if k != "note"},
                      "pf_days": {k: v for k, v in box.items() if "pf_days" in k or k == "run124_vs_gpt3"}},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
