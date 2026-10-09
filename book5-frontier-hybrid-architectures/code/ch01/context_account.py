# context_account.py —— Book5 ch1 账本立式件：打分矩阵账 + 分类型 KV/状态账本公式 + 5 机型速算 + 层类型词表实物 dump
# 用途：本章两本账的「正身算盘」——
#   ① 打分矩阵账（1.2 节）：每层每头打分元素 = n^2（因果上三角 ≈ 其半），n=2048/8192/131072/1M 四档
#      与 1M/8k 放大倍数（P1 探针数字的正式复算档——探针升格判据：入正文数字一律本脚本重跑落 JSON）；
#   ② 分类型 KV/状态账本（1.6 节）：三类层的三条公式（全局无界 / 滑窗有界 / 线性固定）逐机型整数速算，
#      五机型 = Gemma3-27B、gpt-oss-20b（路线一）、GLM-5（路线二）、Qwen3-Next-80B-A3B、Kimi K3（路线三）；
#      交叉点 n* = 线性层固定状态总量 ÷ 全局层每 token KV 增量（Qwen3-Next 1536 / K3 自算）；
#   ③ 层类型词表实物（1.5 节）：transformers 5.18.0 masking_utils 的 LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING
#      源码摘录（三候选 bootstrap 定位源码根）+ GlmMoeDsaConfig 按真实 GLM-5 config 快照生成的
#      layer_types 78 项 dump（78 层全部 indexed_attention——「工程已收编」的实物正身）。
# 证据口径：全部整数精确算（config 公式，无浮点误差）；种子按全书纪律置 20261002（纯算术、无随机性）。
# 所属章节：Book5 第 1 章（1.2 / 1.5 / 1.6；图 1.1 数据同源于本脚本的 score_matrix_account）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch01/context_account.py" [--out-name run1]
# 产物：log/book5-ch01/context_account_{out}.json（不入库）
import argparse
import json
import os
import sys

SEED = 20261002
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book5-ch01")
GLM5_SNAP = os.path.join(REPO_ROOT, "log", "book5-feasibility", "glm5_config.json")

# ---------------- 三候选 bootstrap：transformers 5.18.0 源码根（工作区 / 书仓嵌套 / 书仓裸布局；末位回退安装包） ----------------
_HF_SRC_CANDS = [
    os.path.join(REPO_ROOT, "repos", "transformers"),
    os.path.join(REPO_ROOT, "from-models-to-vllm", "repos", "transformers"),
    os.path.join(REPO_ROOT, "repos", "transformers"),  # 裸布局占位（与首位同路径，保持候选数口径一致）
]


def find_hf_src():
    """定位 transformers 源码根：优先本地浅克隆，失败回退安装包（版本须 5.18.x）。"""
    for root in _HF_SRC_CANDS:
        mu = os.path.join(root, "src", "transformers", "masking_utils.py")
        if os.path.isfile(mu):
            return mu, "本地浅克隆 repos/transformers"
    import transformers
    from transformers import masking_utils as _mu
    assert transformers.__version__.startswith("5.18"), f"安装包版本 {transformers.__version__} ≠ 5.18"
    return _mu.__file__, f"安装包 transformers {transformers.__version__}"


def vocab_excerpt():
    """实物 1.1 素材：LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING 原样源码行（带行号）。"""
    path, src_desc = find_hf_src()
    lines = open(path, encoding="utf-8").readlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING"))
    end = next(i for i in range(start, len(lines)) if lines[i].rstrip() == "}")
    excerpt = [f"L{i+1}: {lines[i].rstrip()}" for i in range(start, end + 1)]
    return excerpt, src_desc, path


def glm5_layer_types():
    """实物 1.1 素材：真实 GLM-5 config 快照实例化 GlmMoeDsaConfig → layer_types 78 项。

    GLM-5 config 无显式 layer_types，由 __post_init__ 生成规则填成全机 indexed_attention（DSA）。
    快照缺失时降级：用 num_hidden_layers=78 的默认构造（状态行注明「快照缺失，默认构造」）。
    """
    status = "真实 config 快照"
    kwargs = {"num_hidden_layers": 78}
    if os.path.isfile(GLM5_SNAP):
        raw = json.load(open(GLM5_SNAP, encoding="utf-8"))
        skip = {"architectures", "model_type", "torch_dtype", "transformers_version", "quantization_config", "dtype"}
        kwargs = {k: v for k, v in raw.items() if k not in skip}
    else:
        status = "快照缺失，默认构造"
    try:
        from transformers.models.glm_moe_dsa import GlmMoeDsaConfig
        cfg = GlmMoeDsaConfig(**kwargs)
        lt = list(cfg.layer_types)
        return {"status": f"OK（{status}）", "num_hidden_layers": len(lt),
                "unique": sorted(set(lt)), "counts": {t: lt.count(t) for t in sorted(set(lt))},
                "excerpt": lt[:6] + ["..."] + lt[-2:]}, f"transformers GlmMoeDsaConfig @ {GLM5_SNAP}"
    except Exception as e:  # noqa: BLE001 —— 条件性降级
        return {"status": f"降级（{type(e).__name__}: {e}）"}, "-"


# ---------------- ① 打分矩阵账：每层每头打分元素 = n^2（整矩阵口径；因果上三角 = n(n+1)/2 ≈ 其半） ----------------
def score_matrix_account():
    ns = [2048, 8192, 131072, 1_000_000]
    rows = [{"n": n, "n2_elements_per_layer_head": n * n, "causal_upper_triangle": n * (n + 1) // 2} for n in ns]
    for r in rows:
        r["ratio_vs_8192"] = round(r["n2_elements_per_layer_head"] / (8192 * 8192), 1)
    return rows


# ---------------- ② 分类型 KV/状态账本（本册新账，单位=元素数，整数精确；BF16 字节 = 元素×2，GB 取十进制） ----------------
# 通式（正文式 (1.2)）：Ledger(n) = c_kv·(L_g·n + L_s·min(n,W)) + L_l·s_state
#   c_kv = 每 token 每层 KV 增量——GQA/MQA 机型 = 2·h_kv·d_k；MLA 机型 = 压缩维+共享绳位（K3/GLM-5 均为 576）。
def kv_global(L_g, h_kv, d_k, n):
    """全局（满注意力）层：KV 存量 = c_kv·L_g·n（此处 GQA 形 c_kv=2·h_kv·d_k）—— 随 n 线性无界。"""
    return 2 * L_g * h_kv * d_k * n


def kv_sliding(L_s, h_kv, d_k, W, n):
    """滑窗层：KV 存量 = c_kv·L_s·min(n, W)（GQA 形）—— 过窗即封顶（有界）。"""
    return 2 * L_s * h_kv * d_k * min(n, W)


def state_linear(L_l, state_per_layer):
    """线性层：状态 = L_l ×（每层固定状态元素数）—— 与 n 无关（常数）。"""
    return L_l * state_per_layer


def cross_point(L_l, state_per_layer, L_g, c_kv):
    """交叉点 n* = L_l·s_state / (c_kv·L_g)：全局层无界 KV 追平线性层固定状态所需 token 数。"""
    return (L_l * state_per_layer) / (L_g * c_kv)


# 五机型 config 账（来源：各章调研底单——① papers/01 §B/§E（Gemma3/gpt-oss config+源码规则）；
# ③ papers/03 §D（Qwen3-Next config 逆向）；④ papers/04 §1.3/§2.2（K3 config）；
# GLM-5 = 本地 config 快照（批一 #5 销项件））。ctx = max_position_embeddings（满档口径）。
MACHINES = [
    dict(name="Gemma3-27B", route=1, L=62, L_g=10, L_s=52, L_l=0, h_kv=16, d_k=128, W=1024, ctx=131072,
         note="5:1（52 滑窗+10 全局），窗 1024"),
    dict(name="gpt-oss-20b", route=1, L=24, L_g=12, L_s=12, L_l=0, h_kv=8, d_k=64, W=128, ctx=131072,
         note="1:1（12 滑窗+12 全局），窗 128"),
    dict(name="GLM-5", route=2, L=78, L_g=78, L_s=0, L_l=0, h_kv=None, d_k=None, W=None, ctx=202752,
         latent=576, note="78 层全 DSA（indexed_attention）；KV=MLA latent 576/token/层，打分参与 top-2048（计算侧有界）"),
    dict(name="Qwen3-Next-80B-A3B", route=3, L=48, L_g=12, L_s=0, L_l=36, h_kv=2, d_k=256, W=None, ctx=262144,
         gdn_state=32 * 128 * 128, gdn_conv=24576, note="3:1（36 GDN+12 全局）；GDN 状态 524,288/层+conv 24,576/层"),
    dict(name="Kimi K3", route=3, L=93, L_g=24, L_s=0, L_l=69, h_kv=None, d_k=None, W=None, ctx=1048576,
         latent=576, kda_state=96 * 128 * 128, note="3:1（69 KDA+24 Gated MLA）；MLA latent 576/token/层，KDA 状态 1,572,864/层"),
]


def machine_row(m):
    """逐机型满档账：三类层各自存量 + 混合合计 + 全满对照 + 交叉点（路线三）。"""
    n = m["ctx"]
    row = dict(name=m["name"], route=m["route"], ctx=n, note=m["note"])
    if m["route"] == 1:
        per = 2 * m["h_kv"] * m["d_k"]
        g = kv_global(m["L_g"], m["h_kv"], m["d_k"], n)
        s = kv_sliding(m["L_s"], m["h_kv"], m["d_k"], m["W"], n)
        full = kv_global(m["L"], m["h_kv"], m["d_k"], n)
        row.update(per_token_per_layer=per, kv_global_unbounded=g, kv_sliding_bounded=s,
                   mixed_total=g + s, if_all_full=full, saving_pct=round(100 * (1 - (g + s) / full), 1),
                   sliding_share_pct=round(100 * s / (g + s), 1))
    elif m["route"] == 2:
        per = m["latent"]  # MLA latent（512 压缩 + 64 共享绳位）每 token 每层元素
        g = per * m["L_g"] * n  # DSA 层 KV 仍逐 token 存（无界）；索引键另计（ch4 展开）
        row.update(per_token_per_layer=per, kv_global_unbounded=g, mixed_total=g, if_all_full=g,
                   note=m["note"] + f"；打分参与 top-2048 → 每层每头打分元素 ≈ h·n·2048")
    else:
        per = 2 * m["h_kv"] * m["d_k"] if m.get("h_kv") else m["latent"]
        st = m.get("gdn_state") or m.get("kda_state")
        g = per * m["L_g"] * n
        tot = state_linear(m["L_l"], st) + (m["L_l"] * m["gdn_conv"] if m.get("gdn_conv") else 0)
        n_star = cross_point(m["L_l"], st, m["L_g"], per)
        st_tot = state_linear(m["L_l"], st)
        row.update(per_token_per_layer=per, kv_global_unbounded=g, linear_state_fixed=tot,
                   state_core_only=st_tot, n_star=round(n_star, 1),
                   mixed_total=g + tot, if_all_full=per * m["L"] * n,
                   linear_share_pct=round(100 * st_tot / (g + st_tot), 2),
                   linear_vs_global_pct=round(100 * st_tot / g, 2))
    for k in ("mixed_total", "if_all_full", "kv_global_unbounded", "kv_sliding_bounded", "linear_state_fixed"):
        if k in row:
            row[k + "_gb_bf16"] = round(row[k] * 2 / 1e9, 2)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="run1", help="产物文件名后缀（防覆写）")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    scores = score_matrix_account()
    rows = [machine_row(m) for m in MACHINES]

    print("=" * 88)
    print("① 打分矩阵账（每层每头打分元素，整矩阵口径；因果上三角≈其半）")
    for r in scores:
        print(f"  n={r['n']:>9,} → {r['n2_elements_per_layer_head']:>15,} 元素/层/头"
              f"（对 8k 放大 ×{r['ratio_vs_8192']:,.1f}）")
    print(f"  1M/8k 精确放大 = 1e12 / 8192^2 = {1e12 / 8192 ** 2:,.1f}×")

    print("-" * 88)
    print("② 分类型 KV/状态账本·五机型满档速算（元素数；括号内 BF16 GB=十进制）")
    for r in rows:
        line = f"  {r['name']:<20} @{r['ctx']:>8,} | 全局层 {r.get('kv_global_unbounded', 0):>15,}"
        if r.get("kv_sliding_bounded") is not None:
            line += f" | 滑窗层 {r['kv_sliding_bounded']:>12,}"
        if r.get("linear_state_fixed") is not None:
            line += f" | 线性层 {r['linear_state_fixed']:>12,}"
        print(line + f"  （{r['mixed_total_gb_bf16']:.2f} GB）")
        extra = []
        if "saving_pct" in r:
            extra.append(f"全满对照 {r['if_all_full_gb_bf16']:.2f} GB → 省 {r['saving_pct']}%")
        if "sliding_share_pct" in r:
            extra.append(f"滑窗层占比 {r['sliding_share_pct']}%")
        if "linear_share_pct" in r:
            extra.append(f"线性状态占总账 {r['linear_share_pct']}%（对全局层 KV 之比 {r['linear_vs_global_pct']}%）")
        if "n_star" in r:
            extra.append(f"交叉点 n*={r['n_star']:,.0f} token")
        if extra:
            print("      " + "；".join(extra))

    print("-" * 88)
    print("③ 层类型词表实物（transformers 5.18.0）")
    excerpt, src_desc, src_path = vocab_excerpt()
    for ln in excerpt:
        print("  " + ln)
    print(f"  [源码出处 {src_desc} | {src_path}]")
    lt, lt_src = glm5_layer_types()
    print(f"  GLM-5 layer_types（{lt_src}）：{lt}")

    out = dict(seed=SEED, score_matrix_account=scores, machines=rows,
               layer_types_vocab=dict(excerpt=excerpt, source=src_desc, path=src_path),
               glm5_layer_types=dict(info=lt, source=lt_src))
    path = os.path.join(OUT_DIR, f"context_account_{args.out_name}.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("=" * 88)
    print(f"[产物] {path}")


if __name__ == "__main__":
    main()
