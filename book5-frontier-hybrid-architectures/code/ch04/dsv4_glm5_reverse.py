# dsv4_glm5_reverse.py —— Book5 ch4 正式件：DSV4 / GLM-5(.2) config 逆向 + 1M 分类型账本手算 + meta 结构对拍 + 层模式图
# 用途（大纲 ch4「路线二续：DSV4 与 GLM-5 的 1M 上下文」，案例章豁免⑥主场）：
#   ① config 逆向：本地 config 快照零下载（log/book5-feasibility/{dsv4flash,dsv4pro,glm5,glm52}_config.json，
#      【证据等级：config 逆向（checkpoint 原文快照）】）——DSV4 层表从 compress_ratios 重建（滑窗/CSA/HCA 计数、
#      尾项 MTP 占位断言）、双 rope 组读数（main 1e4 无缩放 / compress 1.6e5+YaRN16、attention_factor=1.0 无 mscale，
#      C-60 定谳的 config 正身）；GLM-5.2 indexer_types 周期 4 断言（前 3 full + 每 4 层 1 full）；
#   ② 结构对拍双口径（v1.1 钉死，防 48GB 统一内存 OOM）：全尺寸旗舰（284B/1.6T/744B/753B 档）一律
#      meta device（init_empty_weights）构造——只建模块树、dump state_dict 键名与参数账（零显存零内存分配）；
#      随机权重实跑仅限 6-8 层微缩 config（CPU fp32，秒级）——meta 手算 = checkpoint 亲算 = 论文数三方对账；
#   ③ 1M 分类型账本手算（元素口径）：DSV4-Flash/Pro（滑窗层有界 + CSA n/m + HCA n/128 条目账）、GLM-5.2
#      （MLA 576/token 无界 ×78 + 索引器 K 缓存 21 层）、K3（24 MLA + 69 KDA 固定状态）、Qwen3.5-35B（10 GQA + 30 GDN）；
#      附 DSV3.2 对比复算（厂商自报 1M KV 10%/7% 的数量级复核，假设已注明）；
#   ④ 出图：--plot-layer-map=GLM-5.2 IndexShare 层模式图（图 4.2 产物，config 自动绘制）；
#      --plot-dsv4-map=DSV4-Flash 整机结构推断图（图 4.1 产物，证据等级实线/虚线）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch04/dsv4_glm5_reverse.py" [--out-name run1] [--plot-layer-map] [--plot-dsv4-map]
#   CPU 秒-分钟级；零下载（config 快照与 params_audit 产物已落 log/）。
# 产物：log/book5-ch04/dsv4_glm5_reverse_{out-name}.json（不入库）；
#       figures/fig-4-1-dsv4-teardown.png、fig-4-2-glm52-indexshare.png
import argparse
import json
import math
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FEAS_DIR = os.path.join(REPO_ROOT, "log", "book5-feasibility")
AUDIT_DIR = os.path.join(REPO_ROOT, "log", "book5-ch11", "params_audit")
OUT_DIR = os.path.join(REPO_ROOT, "log", "book5-ch04")
FIG_DIR = os.path.join(REPO_ROOT, "book5-frontier-hybrid-architectures", "figures")
SEED = 20261002

N1M = 1_048_576  # 1M token 上下文账本统一口径
V32_LAYERS, V32_MLA_PER_TOK = 61, 512 + 64  # DSV3.2 参照机：61 层 MLA，latent 512 + rope 64 = 576 元素/token/层


def load_raw(name):
    with open(os.path.join(FEAS_DIR, f"{name}_config.json")) as f:
        return json.load(f)


# ---------------- ① config 逆向：层表与 rope 组 ----------------
def reverse_dsv4(raw, tag):
    """DSV4 config 逆向：compress_ratios 逐层重建（0=滑窗 4=CSA 128=HCA，论文 §4.2.1 同口径）。

    输入 raw dict；输出 dict（层表计数、尾项 MTP 占位断言、双 rope 组读数）。
    """
    ratio2type = {0: "SWA", 4: "CSA", 128: "HCA"}
    ratios = raw["compress_ratios"]
    n_layers, n_mtp = raw["num_hidden_layers"], raw["num_nextn_predict_layers"]
    assert len(ratios) == n_layers + n_mtp, f"{tag}: compress_ratios 长度 {len(ratios)} != L{n_layers}+MTP{n_mtp}"
    types = [ratio2type[r] for r in ratios[:n_layers]]  # HF __post_init__ 同款截断：尾项 MTP 占位不入层表
    counts = {t: types.count(t) for t in ("SWA", "CSA", "HCA")}
    rp = raw
    rope_main = {"theta": rp["rope_theta"], "scaling": "none"}
    rope_comp = {"theta": rp["compress_rope_theta"], "scaling": rp["rope_scaling"]}
    return {
        "tag": tag, "layers": n_layers, "layer_types_head": types[:6], "counts": counts,
        "mtp_tail_slot": ratios[-1], "max_pos": raw["max_position_embeddings"],
        "rope_main": rope_main, "rope_comp": rope_comp,
        "sliding_window": raw["sliding_window"], "index_topk": raw["index_topk"],
        "hidden": raw["hidden_size"], "kv_heads": raw["num_key_value_heads"], "head_dim_c": raw["head_dim"],
        "moe": f"{raw['n_routed_experts']}+{raw['n_shared_experts']}sh top{raw['num_experts_per_tok']}",
    }


def reverse_glm(raw, tag):
    """GLM-5.2 config 逆向：indexer_types 逐层（full/shared）、周期断言、索引器层参数公式。

    每层索引器参数 = q_lora*(h_I*d_I) + d*d_I + 2*d_I + d*h_I（wq_b+wk+k_norm+weights_proj，
    与 HF GlmMoeDsaIndexer 四个 Linear 的形状逐项一致）。
    """
    it = raw["indexer_types"] if raw.get("indexer_types") else ["full"] * raw["num_hidden_layers"]
    n_full, n_shared = it.count("full"), it.count("shared")
    pos_full = [i for i, v in enumerate(it) if v == "full"]
    offset, freq = raw.get("index_skip_topk_offset", 0), raw.get("index_topk_freq", 0)
    # HF 展开：i<offset 全 full；此后 (i-offset+1)%freq==0 处 full（即 offset+freq-1 起每 freq 层一个）
    expect = list(range(0, offset)) + list(range(offset + freq - 1, raw["num_hidden_layers"], max(freq, 1)))
    assert pos_full == expect, f"{tag}: full 层位号 {pos_full[:8]}... 与 offset{offset}/freq{freq} 展开不符"
    d, q_lora = raw["hidden_size"], raw["q_lora_rank"]
    h_i, d_i = raw["index_n_heads"], raw["index_head_dim"]
    per_layer = q_lora * (h_i * d_i) + d * d_i + 2 * d_i + d * h_i
    return {"tag": tag, "layers": raw["num_hidden_layers"], "n_full": n_full, "n_shared": n_shared,
            "pos_full_head": pos_full[:8], "offset": offset, "freq": freq,
            "index_topk": raw["index_topk"], "per_layer_indexer_params": per_layer,
            "shared_drop_total": n_shared * per_layer,
            "max_pos": raw["max_position_embeddings"], "rope_theta": raw["rope_parameters"]["rope_theta"],
            "mla_per_tok": raw["kv_lora_rank"] + raw["qk_rope_head_dim"]}


# ---------------- ② 结构对拍双口径：meta 全尺寸 + CPU 微缩实跑 ----------------
def meta_account(name, cfg_cls, model_cls, raw):
    """meta device 口径（零分配）：全尺寸旗舰只建模块树，参数账 + 索引器携带层数 + 代表性键名。

    输入 config 类与原始 dict；输出 dict。与 log/book5-ch11/params_audit 的 safetensors 亲算对账。
    """
    from accelerate import init_empty_weights
    kw = {k: v for k, v in raw.items() if k not in ("architectures", "dtype", "torch_dtype",
                                                    "quantization_config", "expert_dtype")}
    cfg = cfg_cls(**kw)
    with init_empty_weights():
        model = model_cls(cfg)
    total = sum(p.numel() for p in model.parameters())  # meta 张量 numel 可读、无数据
    keys = list(model.state_dict().keys())
    idx_layers = sorted({int(k.split(".")[2]) for k in keys if ".indexer." in k})
    compr_layers = sorted({int(k.split(".")[2]) for k in keys if ".compressor." in k})
    return {"total_B": round(total / 1e9, 3), "n_state_keys": len(keys),
            "indexer_layers_meta": len(idx_layers), "indexer_pos_head": idx_layers[:8],
            "compressor_layers_meta": len(compr_layers),
            "key_samples": [k for k in keys if ".indexer." in k][:4] or [k for k in keys if "compressor" in k][:4],
            "layer_types_len": len(getattr(cfg, "layer_types", []) or [])}


def mini_dsv4():
    """微缩 DSV4（6 层，CPU fp32 随机权重实跑）：2 滑窗 + CSA/HCA 交错——Flash 层模式的同构缩微。"""
    from transformers import DeepseekV4Config, DeepseekV4ForCausalLM
    torch.manual_seed(SEED)
    lt = ["sliding_attention", "sliding_attention", "compressed_sparse_attention",
          "heavily_compressed_attention", "compressed_sparse_attention", "heavily_compressed_attention"]
    cfg = DeepseekV4Config(
        hidden_size=256, num_hidden_layers=6, vocab_size=512, num_attention_heads=4,
        head_dim=64, qk_rope_head_dim=16, q_lora_rank=128, o_groups=2, o_lora_rank=64,
        index_n_heads=4, index_head_dim=32, index_topk=8,  # n=64/m=4 -> 16 条目，top-8
        n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1, moe_intermediate_size=128,
        num_hash_layers=0, sliding_window=8, hc_mult=1, layer_types=lt,
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 16},  # m=4 / m'=16（缩 128）
        max_position_embeddings=128)
    model = DeepseekV4ForCausalLM(cfg).to(torch.float32)
    x = torch.randint(0, 512, (2, 64))  # (B=2, n=64)
    logits = model(input_ids=x).logits  # (2, 64, 512)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 512).float(), x[:, 1:].reshape(-1))
    return {"params": sum(p.numel() for p in model.parameters()), "logits_shape": list(logits.shape),
            "loss": round(loss.item(), 3), "ln_vocab": round(math.log(512), 3),
            "layer_types": cfg.layer_types, "partial_rope": cfg.partial_rotary_factor}


def mini_glm():
    """微缩 GLM-5.2（8 层，CPU fp32 随机权重实跑）：F,F,F,S,S,S,F,S——IndexShare 周期 4 的同构缩微。

    shared 层无索引器模块（权重级缺席的缩微正身）：携带索引器的层 = config full 层位号 {0,1,2,6}。
    """
    from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
    torch.manual_seed(SEED)
    it = ["full", "full", "full", "shared", "shared", "shared", "full", "shared"]
    cfg = GlmMoeDsaConfig(
        hidden_size=256, num_hidden_layers=8, vocab_size=512, num_attention_heads=4, num_key_value_heads=4,
        qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64, kv_lora_rank=128, q_lora_rank=256,
        index_n_heads=4, index_head_dim=64, index_topk=16, indexer_types=it,
        first_k_dense_replace=1, intermediate_size=512, n_routed_experts=8, num_experts_per_tok=2,
        n_shared_experts=1, moe_intermediate_size=128, rope_theta=10000,
        max_position_embeddings=128, num_nextn_predict_layers=0)
    model = GlmMoeDsaForCausalLM(cfg).to(torch.float32)
    idx_layers = sorted({int(k.split(".")[2]) for k in model.state_dict() if ".indexer." in k})
    per_idx = sum(p.numel() for n, p in model.named_parameters() if ".indexer." in n) // len(idx_layers)
    x = torch.randint(0, 512, (2, 64))
    logits = model(input_ids=x).logits  # (2, 64, 512)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 512).float(), x[:, 1:].reshape(-1))
    return {"params": sum(p.numel() for p in model.parameters()), "logits_shape": list(logits.shape),
            "loss": round(loss.item(), 3), "ln_vocab": round(math.log(512), 3),
            "indexer_layers_mini": idx_layers, "shared_layers_no_indexer": 8 - len(idx_layers),
            "per_layer_indexer_params": per_idx}


def cross_check_glm52_weights(res_glm52):
    """534M 差值的精确闭合：57 个 shared 层 × 每层索引器参数 = GLM-5 与 5.2 的 checkpoint 全量差。

    params_audit 产物（log/book5-ch11/params_audit/*.json）为 safetensors 分片头亲算（不下载权重）。
    """
    out = {"shared_drop_total": res_glm52["shared_drop_total"]}
    try:
        with open(os.path.join(AUDIT_DIR, "GLM-5.json")) as f1, open(os.path.join(AUDIT_DIR, "GLM-5_2.json")) as f2:
            a5, a52 = json.load(f1), json.load(f2)
        raw5, raw52 = a5["_raw"]["total"], a52["_raw"]["total"]
        audit_diff = raw5 - raw52
        out.update({"audit_total_glm5": raw5, "audit_total_glm52": raw52,
                    "audit_diff": audit_diff, "diff_equals_57x": audit_diff == res_glm52["shared_drop_total"]})
    except (OSError, KeyError):
        out["audit"] = "params_audit 产物缺席（脚本仍可用算术闭合自查）"
    return out


# ---------------- ③ 1M 分类型账本手算（元素口径） ----------------
def ledger_1m():
    """1M token 的 KV/状态账（单位=元素数；含 batch=1）：逐机型分类型账。

    DSV4 条目账：滑窗层 min(n,W) 条 × c；CSA 层 n/m 条 + W 条；HCA 层 n/m' 条 + W 条（条目=共享 K=V 的 c 维向量，
    每条目一份、非 K/V 两份——HF 源码 kv_proj 单张量兼任）；索引器 K 缓存（d_I=128/条目）单列为小项。
    GLM/K3 MLA 账：(d_c + d_r)/token/层；KDA/GDN 固定状态：头数×d_k×d_v（形状出处=第 5 章）。
    """
    n, w, c = N1M, 128, 512
    d = {}
    # DSV4-Flash：2 SWA + 21 CSA(m=4) + 20 HCA(m'=128)；DSV4-Pro：30 CSA + 31 HCA
    swa_l = w * c
    csa_l = (n // 4) * c + w * c
    hca_l = (n // 128) * c + w * c
    flash = 2 * swa_l + 21 * csa_l + 20 * hca_l
    pro = 30 * csa_l + 31 * hca_l
    d["DSV4-Flash"] = {"per_layer": {"SWA(2)": swa_l, "CSA(21)": csa_l, "HCA(20)": hca_l},
                       "total": flash, "indexer_K_note": "索引器 K 仅 CSA 层：21*(n/4)*128（正文单列）"}
    d["DSV4-Pro"] = {"per_layer": {"CSA(30)": csa_l, "HCA(31)": hca_l}, "total": pro,
                     "indexer_K_note": "索引器 K 仅 CSA 层：30*(n/4)*128（正文单列）"}
    v32 = V32_LAYERS * V32_MLA_PER_TOK * n
    d["DSV3.2_ref"] = {"total": v32, "note": "61 层 MLA × 576/token（对比基准）",
                       "flash_vs_v32": round(flash / v32 * 100, 1), "pro_vs_v32": round(pro / v32 * 100, 1),
                       "claim": "预览版自报 KV 10%/7%（Pro/Flash）——同档数量级复核（口径=只计压缩+滑窗条目）"}
    # GLM-5.2：78 层 MLA 无界 + 21 full 层索引器 K 缓存（128/token）
    glm = 78 * 576 * n
    d["GLM-5.2"] = {"mla_per_token": 78 * 576, "total": glm, "gb_bf16": round(glm * 2 / 1e9, 1),
                    "indexer_K_full_layers": 21 * 128 * n, "indexer_pct": round(21 * 128 / (78 * 576) * 100, 1)}
    # K3：24 层 Gated MLA（576/token）+ 69 层 KDA 固定状态（96 头×128×128）
    k3_mla, k3_kda = 24 * 576 * n, 69 * (96 * 128 * 128)
    d["K3"] = {"mla_total": k3_mla, "mla_gb_bf16": round(k3_mla * 2 / 1e9, 1),
               "kda_fixed": k3_kda, "kda_mb_bf16": round(k3_kda * 2 / 1e6, 1),
               "kda_vs_mla_pct": round(k3_kda / k3_mla * 100, 2)}
    # Qwen3.5-35B-A3B：10 层 GQA 16:2（2×2×256=1024/token）+ 30 层 GDN 固定状态（32V×128×128）
    q_full, q_gdn = 10 * 1024 * n, 30 * (32 * 128 * 128)
    d["Qwen3.5-35B"] = {"gqa_total_1m": q_full, "gqa_gb_bf16": round(q_full * 2 / 1e9, 1),
                        "gdn_fixed": q_gdn, "gdn_mb_bf16": round(q_gdn * 2 / 1e6, 1),
                        "note": "1M=托管口径（原生 262K）；GDN 状态形状出处=第 5 章"}
    return d


# ---------------- ④ 出图（matplotlib 印刷规格：300dpi/分类色/图内英文） ----------------
BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK = "#e1e0d9", "#898781", "#0b0b0b"


def plot_layer_map():
    """图 4.2：GLM-5.2 IndexShare 层模式图（config indexer_types 自动绘制；21 full + 57 shared，周期 4）。"""
    import matplotlib.pyplot as plt
    raw = load_raw("glm52")
    it = raw["indexer_types"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 3.4), dpi=300,
                                  gridspec_kw={"height_ratios": [1.0, 0.9]}, layout="tight")
    for i, t in enumerate(it):
        ax.add_patch(plt.Rectangle((i, 0), 0.92, 1, facecolor=BLUE if t == "full" else "#f7d9cb",
                                   edgecolor="none"))
    ax.set_xlim(-0.5, len(it) + 8); ax.set_ylim(-0.1, 1.35); ax.axis("off")
    ax.set_title("GLM-5.2  indexer_types  (78 layers: 21 full + 57 shared, period 4)",
                 fontsize=10, color=DARK, pad=6)
    ax.annotate("first 3 full", xy=(1.4, 1.08), xytext=(1.4, 1.22), ha="center", fontsize=8, color=DARK,
                arrowprops=dict(arrowstyle="-", color=TICK, lw=0.8))
    ax.annotate("layer 6 = next full (offset 3, freq 4)", xy=(6.4, 1.08), xytext=(24, 1.24), ha="center",
                fontsize=8, color=DARK, arrowprops=dict(arrowstyle="-", color=TICK, lw=0.8))
    ax.annotate("MTP (layer 78, off-map)\nkeeps its indexer", xy=(80.2, 0.5), fontsize=7.5, color="#52514e",
                va="center")
    zoom = it[:12]
    for i, t in enumerate(zoom):
        ax2.add_patch(plt.Rectangle((i, 0), 0.92, 1, facecolor=BLUE if t == "full" else "#f7d9cb",
                                    edgecolor="none"))
        ax2.text(i + 0.46, 0.5, t[0].upper(), ha="center", va="center", fontsize=7.5,
                 color="white" if t == "full" else "#b35633")
        ax2.text(i + 0.46, -0.34, str(i), ha="center", va="center", fontsize=6.5, color=TICK)
    ax2.set_xlim(-0.6, 12); ax2.set_ylim(-0.75, 1.3); ax2.axis("off")
    ax2.set_title("zoom: layers 0-11   F=full (own indexer)   S=shared (reuse last F top-k)", fontsize=8.5,
                  color="#52514e", pad=3)
    path = os.path.join(FIG_DIR, "fig-4-2-glm52-indexshare.png")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def plot_dsv4_map():
    """图 4.1：DSV4-Flash 整机结构推断图（示意级；实线=预览版论文明文，虚线=config/源码-only）。"""
    import matplotlib.pyplot as plt

    def box(ax, x, y, w, h, text, solid=True, fc="white", fs=7.5, ec=None):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec or DARK,
                                   lw=1.4 if solid else 1.0, linestyle="-" if solid else (0, (4, 3))))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=DARK)

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=300, layout="tight")
    ax.axis("off"); ax.set_xlim(0, 16); ax.set_ylim(0, 10.4)
    # 左列：整机堆叠（论文实线为主）
    box(ax, 0.3, 8.9, 3.4, 0.8, "embed  (V=129280)\n-> (B, n, 4096)", True)
    box(ax, 0.3, 7.5, 3.4, 1.0, "layers 0-1: SWA\nwindow 128  (B,n,4096)", True, fc="#eaf2fc")
    box(ax, 0.3, 5.9, 3.4, 1.3, "layers 2-42:\nCSA | HCA interleave\n(21 CSA + 20 HCA)", True, fc="#eaf2fc")
    box(ax, 0.3, 4.5, 3.4, 1.0, "MoE every layer\n(3 hash-routing first)\n256+1 experts, top-6", True, fc="#fdf1ea")
    box(ax, 0.3, 3.1, 3.4, 1.0, "mHC residual\nn_hc=4, Sinkhorn 20", True, fc="#fdf1ea")
    box(ax, 0.3, 1.7, 3.4, 1.0, "norm -> lm_head\n(B, n, 4096) -> (B, n, V)", True)
    box(ax, 0.3, 0.4, 3.4, 0.9, "MTP depth 1\n(config slot only, HF not built)", False, fc="white")
    ax.annotate("", xy=(2.0, 8.85), xytext=(2.0, 8.55), arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))
    ax.annotate("", xy=(2.0, 7.45), xytext=(2.0, 7.15), arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))
    ax.annotate("", xy=(2.0, 5.85), xytext=(2.0, 5.55), arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))
    ax.annotate("", xy=(2.0, 4.45), xytext=(2.0, 4.15), arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))
    ax.annotate("", xy=(2.0, 3.05), xytext=(2.0, 2.75), arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))
    ax.annotate("", xy=(2.0, 1.65), xytext=(2.0, 1.35), arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))
    # 右上：CSA 层内部（论文实线；数值 config 互证）
    ax.text(4.5, 9.95, "CSA layer (compressed + select)", fontsize=9, color=DARK, weight="bold")
    box(ax, 4.5, 8.5, 5.0, 1.1, "compressor  m=4: overlap windows\nKV (B,n,512) -> entries (B, n/4, 512)\nshared K = V (one kv_proj)", True, fc="#eaf2fc")
    box(ax, 10.0, 8.5, 5.6, 1.1, "lightning indexer: 64 heads x 128\nQK path in FP4  |  I in (B, n/4)", True, fc="#eaf2fc")
    box(ax, 4.5, 7.1, 5.0, 1.0, "top-k = 512 entries\n(sel. from n/4 compressed)", True, fc="#eaf2fc")
    box(ax, 10.0, 7.1, 5.6, 1.0, "core attn: 64 heads, c=512\npartial RoPE 64/512 + sink", True, fc="#eaf2fc")
    box(ax, 4.5, 5.9, 11.1, 0.9, "+ sliding branch: n_win = 128 raw KV  (B,128,512)  |  grouped out-proj g=8 x 1024", True, fc="#eaf2fc")
    ax.annotate("", xy=(7.0, 8.15), xytext=(7.0, 8.45), arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))
    # 右下：HCA 层内部
    ax.text(4.5, 5.35, "HCA layer (heavily compressed, dense)", fontsize=9, color=DARK, weight="bold")
    box(ax, 4.5, 4.1, 5.0, 1.0, "compressor  m'=128: no overlap\nentries (B, n/128, 512)", True, fc="#eaf2fc")
    box(ax, 10.0, 4.1, 5.6, 1.0, "dense attention over entries\nno indexer, no top-k", True, fc="#eaf2fc")
    box(ax, 4.5, 2.9, 11.1, 0.9, "same shared-KV MQA + window 128 branch + grouped out-proj", True, fc="#eaf2fc")
    # 底注：证据等级与 rope 双组
    box(ax, 4.5, 1.35, 11.1, 1.25, "dual RoPE groups (C-60):\nmain: theta=1e4, NO scaling (window layers/branch)\ncompress: theta=1.6e5 + YaRN x16 (from 64K, no mscale)", False, fc="white")
    ax.text(4.5, 0.75, "solid = paper-stated (preview 2606.19348)   dashed = config/source-only", fontsize=7.5, color="#52514e")
    ax.text(4.5, 0.3, "1M = trained ladder 4K->16K->64K->1M (dense first 1T, sparse from 64K)", fontsize=7.5, color="#52514e")
    path = os.path.join(FIG_DIR, "fig-4-1-dsv4-teardown.png")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="Book5 ch4: DSV4/GLM-5(.2) config 逆向 + 1M 账本 + meta 对拍")
    ap.add_argument("--out-name", default="run1", help="产物 JSON 后缀（防覆写）")
    ap.add_argument("--plot-layer-map", action="store_true", help="出图 4.2（GLM-5.2 层模式图）")
    ap.add_argument("--plot-dsv4-map", action="store_true", help="出图 4.1（DSV4 整机结构推断图）")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    from transformers import (DeepseekV4Config, DeepseekV4ForCausalLM,
                              GlmMoeDsaConfig, GlmMoeDsaForCausalLM)

    out = {"seed": SEED}
    # ① config 逆向
    out["dsv4_flash_rev"] = reverse_dsv4(load_raw("dsv4flash"), "V4-Flash")
    out["dsv4_pro_rev"] = reverse_dsv4(load_raw("dsv4pro"), "V4-Pro")
    out["glm52_rev"] = reverse_glm(load_raw("glm52"), "GLM-5.2")
    # ② meta 结构对拍（全尺寸、零分配）+ CPU 微缩实跑
    out["meta_dsv4_flash"] = meta_account("dsv4flash", DeepseekV4Config, DeepseekV4ForCausalLM, load_raw("dsv4flash"))
    out["meta_dsv4_pro"] = meta_account("dsv4pro", DeepseekV4Config, DeepseekV4ForCausalLM, load_raw("dsv4pro"))
    out["meta_glm5"] = meta_account("glm5", GlmMoeDsaConfig, GlmMoeDsaForCausalLM, load_raw("glm5"))
    out["meta_glm52"] = meta_account("glm52", GlmMoeDsaConfig, GlmMoeDsaForCausalLM, load_raw("glm52"))
    out["mini_dsv4"] = mini_dsv4()
    out["mini_glm"] = mini_glm()
    out["glm52_weight_closure"] = cross_check_glm52_weights(out["glm52_rev"])
    # ③ 1M 分类型账本
    out["ledger_1m"] = ledger_1m()

    if args.plot_layer_map:
        out["fig_layer_map"] = plot_layer_map()
    if args.plot_dsv4_map:
        out["fig_dsv4_map"] = plot_dsv4_map()

    path = os.path.join(OUT_DIR, f"dsv4_glm5_reverse_{args.out_name}.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    # 控制台摘录（正文引用数字的来源读数）
    print("=" * 72)
    for k in ("dsv4_flash_rev", "dsv4_pro_rev", "glm52_rev"):
        print(k, "->", json.dumps(out[k], ensure_ascii=False))
    print("-" * 72)
    for k in ("meta_dsv4_flash", "meta_dsv4_pro", "meta_glm5", "meta_glm52"):
        v = out[k]
        print(k, f"total={v['total_B']}B compressor_layers={v['compressor_layers_meta']} "
                 f"indexer_layers={v['indexer_layers_meta']}")
    print("-" * 72)
    print("mini_dsv4 ->", json.dumps(out["mini_dsv4"], ensure_ascii=False))
    print("mini_glm  ->", json.dumps(out["mini_glm"], ensure_ascii=False))
    print("glm52_weight_closure ->", json.dumps(out["glm52_weight_closure"], ensure_ascii=False))
    print("-" * 72)
    print("ledger_1m ->", json.dumps(out["ledger_1m"], ensure_ascii=False))
    print("=" * 72)
    print("saved:", path)


if __name__ == "__main__":
    main()
