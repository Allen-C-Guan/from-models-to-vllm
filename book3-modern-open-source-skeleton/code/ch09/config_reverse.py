# config_reverse.py —— Book3 ch9：Llama 4 双镜像 config 逆向（「无论文模型怎么读」五步全程演练）
# 用途：下载 unsloth/RedHatAI 双镜像 config.json（KB 级，走 env.sh 的 HF_HOME 缓存；下载失败自动回落
#       调研期缓存镜像并在产物 JSON 标来源）→ STEP1 认家族 → STEP2 结构推断表打印（表 9.1 数字源）
#       → STEP3 参数手算 assert（式 9.1 手算 vs transformers meta 建模 vs 第三方转述口径；总账+活账两本）
#       → STEP4 外围 token id 锚点（词表构成推断）→ STEP5 证据分级表；
#       另绘 图 9.1 Llama 4 Scout 结构推断图（实线=config 逆向，虚线=转述级标注）。
# 所属章节：Book3 第 9 章 §9.2-9.4（正文表 9.1/9.2 与式 9.1 的机器可读版；ch5 五步法的 2025 档升级）
# 运行方式：cd 工作区根目录 && source env.sh && python code/ch09/config_reverse.py \
#           [--model scout|maverick] [--out-name run1] [--no-figure]
# 产物：log/book3-ch09/config_reverse_{model}_{out-name}.json（不入库）；
#       figures/fig-9-1-llama4-inferred-structure.png（默认 --model scout 时绘制）
# 耗时：CPU 秒级（config 下载含重试；meta 建模需 transformers/torch，缺库自动跳过该列）。
# 注：本章无随机过程（纯 config 读取与确定性算术），种子 20261002 仅按全书纪律占位。
import argparse
import json
import os

SEED = 20261002
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO, "log", "book3-ch09")
FIG_DIR = os.path.join(REPO, "figures")
FALLBACK = {  # 调研期落盘的镜像缓存（2026-10-03 抓取；HF 不可达时回落，产物 JSON 标来源）
    ("scout", "unsloth"): "log/book3-research/hf-llama4-scout.json",
    ("scout", "RedHatAI"): "log/book3-research/hf-llama4-scout-redhat.json",
    ("maverick", "unsloth"): "log/book3-research/hf-llama4-maverick.json",
}
MIRRORS = {  # meta-llama 原仓 gated 401 —— 一律走社区镜像双源交叉
    "scout": [("unsloth", "unsloth/Llama-4-Scout-17B-16E-Instruct"),
              ("RedHatAI", "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic")],
    "maverick": [("unsloth", "unsloth/Llama-4-Maverick-17B-128E-Instruct")],
}
RELAY = {"scout": dict(total=109e9, active=17e9, ctx="10M"),
         "maverick": dict(total=400e9, active=17e9, ctx="1M")}  # 第三方转述官方 blog（原页已下线）
CORE_FIELDS = ["hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
               "head_dim", "num_local_experts", "num_experts_per_tok", "intermediate_size",
               "intermediate_size_mlp", "max_position_embeddings", "vocab_size", "rope_theta",
               "rms_norm_eps", "tie_word_embeddings"]


def fetch(model):
    """双镜像下载 config.json（KB 级）；失败回落调研期缓存。返回 [(源名, config dict, 来源标注)]。"""
    got = []
    for src, repo_id in MIRRORS[model]:
        cfg, tag = None, ""
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id, "config.json")
            cfg, tag = json.load(open(path, encoding="utf-8")), f"HF 在线下载：{repo_id}"
        except Exception as e:  # 网络阻断/缺库 —— 回落本地缓存镜像（来源降级如实标注）
            fb = os.path.join(REPO, FALLBACK[(model, src)])
            if os.path.exists(fb):
                cfg = json.load(open(fb, encoding="utf-8"))
                tag = f"调研期缓存镜像 {FALLBACK[(model, src)]}（在线失败：{type(e).__name__}）"
        if cfg is not None:
            got.append((src, cfg, tag))
            print(f"[fetch] {src:8s} {tag}")
    assert got, "双镜像均不可得（在线与缓存皆失败）"
    return got


def effective(cfg):
    """经 transformers 5.18.0 的 Llama4Config 读『生效值』：空 no_rope_layers 会被默认 3:1 接管。"""
    from transformers.models.llama4 import Llama4Config
    return Llama4Config.from_dict(cfg).text_config


def infer_rows(t):
    """STEP2 结构推断表：字段 → 值 → 部件（表 9.1 的机器可读版）。"""
    moe_n = len(t.moe_layers)
    return [
        ("architectures", "Llama4ForConditionalGeneration", "原生多模态整机（含视觉塔与投影）"),
        ("hidden_size / layers", f"{t.hidden_size} / {t.num_hidden_layers}", "残差流宽 d / Block 数 L"),
        ("heads / KV heads / head_dim", f"{t.num_attention_heads} / {t.num_key_value_heads} / {t.head_dim}",
         f"GQA 分组沿用：h×d_k = {t.num_attention_heads * t.head_dim} = d"),
        ("num_local_experts / per_tok", f"{t.num_local_experts} / {t.num_experts_per_tok}",
         f"前馈插槽→专家群：{t.num_local_experts} 份里每 token 取 {t.num_experts_per_tok} 份"),
        ("moe_layers", f"{moe_n}/{t.num_hidden_layers} 层（step {t.interleave_moe_layer_step}）",
         "专家层布局：Scout 全层 / Maverick 奇数层（偶数层稠密）"),
        ("intermediate_size / _mlp", f"{t.intermediate_size} / {t.intermediate_size_mlp}",
         "专家宽 f / 稠密层宽 f_mlp（Scout 的稠密宽闲置）"),
        ("no_rope_layers", f"{sum(t.no_rope_layers)} NoPE + {len(t.no_rope_layers) - sum(t.no_rope_layers)} RoPE",
         "位置编码插槽 3:1 交错：每 4 层仅第 4 层旋转；NoPE 层=chunked_attention"),
        ("attention_chunk_size", str(t.attention_chunk_size), "NoPE 层的分块注意力窗（块内注意、块间不看）"),
        ("rope_theta / rope_scaling",
         (f"{t.rope_parameters.get('rope_theta'):.0f} / llama3 x{t.rope_parameters.get('factor'):g}"
          if t.rope_parameters.get("rope_type") != "default"
          else f"{t.rope_parameters.get('rope_theta'):.0f} / null"),
         "θ 继承 Llama3；llama3 型分段缩放（快针不动、慢针压 1/factor）——Scout 有 / Maverick 无"),
        ("max_position_embeddings", f"{t.max_position_embeddings:,}", "声明上下文（Scout 10M / Maverick 1M）"),
        ("vocab_size / tie", f"{t.vocab_size:,} / {t.tie_word_embeddings}", "出入口：嵌入与 lm_head 各一份（untied 沿用）"),
        ("use_qk_norm", str(t.use_qk_norm), "RoPE 层 Q/K 过一道无参数 L2 归一化（源码级，不入参数账）"),
        ("attn_temperature_tuning / floor_scale / attn_scale",
         f"{t.attn_temperature_tuning} / {t.floor_scale} / {t.attn_scale}",
         "推理期注意力温度缩放的 config 事实（与第 7 章温度配方同族旋钮——只登记不展开机制）"),
        ("rms_norm_eps / attention_bias", f"{t.rms_norm_eps} / {t.attention_bias}", "pre-RMSNorm 与无偏置：LLaMA 遗产未动"),
    ]


def param_account(t):
    """STEP3 参数手算（式 9.1/9.2）——总账与活账两本，逐项整数账；混排机型按层型分册。"""
    d, L, hkv, dk = t.hidden_size, t.num_hidden_layers, t.num_key_value_heads, t.head_dim
    V, tied = t.vocab_size, t.tie_word_embeddings
    E, k, f, f_mlp = t.num_local_experts, t.num_experts_per_tok, t.intermediate_size, t.intermediate_size_mlp
    attn = 2 * d * d + 2 * d * hkv * dk                      # q/o 全宽 + k/v 按 KV 组缩（式 5.2 同款）
    moe_ffn = d * E + 3 * d * f * (E + 1)                    # router + E 份路由专家 + 1 份共享（同宽 f，源码级）
    dense_ffn = 3 * d * f_mlp                                # 稠密层门控前馈（SwiGLU 三矩阵沿用）
    n_moe = len(t.moe_layers)
    text = V * d * (2 - tied) + L * (attn + 2 * d) + n_moe * moe_ffn + (L - n_moe) * dense_ffn + d
    act_moe = attn + 2 * d + d * E + 3 * d * f * (1 + k)     # 专家层每 token：注意力+双 norm+router+共享+k 份专家
    act_dense = attn + 2 * d + 3 * d * f_mlp                 # 稠密层每 token：无 router，整份稠密前馈（式 9.2 层型分册）
    active = (n_moe * act_moe + (L - n_moe) * act_dense      # 活账按层型分账——router 只长在专家层上
              + V * d * (2 - tied) + d)                      # 活账含出入口两份（对齐官方 17B 口径）
    return dict(attn_per_layer=attn, moe_ffn_per_layer=moe_ffn, dense_ffn_per_layer=dense_ffn,
                n_moe_layers=n_moe, text_total=text, moe_layer_active=act_moe,
                dense_layer_active=act_dense, layer_active=act_moe,
                active_total=active, total_over_active=round(text / active, 2))


def meta_count(cfg):
    """transformers meta 设备建模计数（不下载权重、不占内存）——手算的独立对拍方。"""
    import torch
    torch.manual_seed(SEED)
    from transformers.models.llama4 import Llama4Config, Llama4ForConditionalGeneration
    with torch.device("meta"):
        m = Llama4ForConditionalGeneration(Llama4Config.from_dict(cfg))
    vis = sum(p.numel() for n, p in m.named_parameters() if "vision" in n or "multi_modal" in n)
    tot = sum(p.numel() for p in m.parameters())
    return dict(text_total=tot - vis, vision_total=vis, grand_total=tot)


def index_total(model):
    """safetensors index 的 total_size/2（只下载 KB-MB 级索引、不碰权重）——第三方实物对拍方。"""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(MIRRORS[model][0][1], "model.safetensors.index.json")
    ts = json.load(open(p, encoding="utf-8"))["metadata"]["total_size"]
    return ts // 2  # 全 bf16 落盘（Llama 4 检查点无 F32 缓冲混装——与 Llama-2-7b 的 2,048 幻影对照）


def draw_fig9_1(t, acct, meta, path):
    """图 9.1：Llama 4 Scout 结构推断图（自绘示意级；实线=config 逆向双镜像，虚线=转述级标注）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    BLUE, ORANGE, TEAL, GRID, TICK, DARK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.size": 8.5, "text.color": DARK, "font.family": "DejaVu Sans"})

    def box(ax, x, y, w, h, text, ec=DARK, fc="white", lw=1.1, fs=7.2, ls="-", tc=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=lw, linestyle=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc or DARK)

    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=300)
    ax.set_xlim(0, 17); ax.set_ylim(0, 10.6); ax.axis("off")
    ax.text(8.5, 10.25, "Llama 4 Scout: structure inferred from config.json (no paper shipped)",
            ha="center", fontsize=9.5, fontweight="bold")

    # ---- 主列：文本整机（config 逆向，实线）；视觉通路在左侧汇入输入侧 ----
    box(ax, 4.6, 9.0, 7.6, 0.95, "Embedding  V=202,048 x d=5120  (untied, tie=false)", ec=BLUE, fc="#f2f7fd")
    box(ax, 4.6, 6.35, 7.6, 2.25, "", ec=BLUE, fc="#f2f7fd")
    ax.text(4.85, 8.32, "x 48 blocks", fontsize=7.4, fontweight="bold", color=DARK)
    ax.text(4.85, 7.15, "RMSNorm -> Attention: 40 Q heads / 8 KV heads, head_dim 128\n"
                        "     q:(B,40,n,128)  k,v:(B,8,n,128)  ->  o:(B,n,5120)\n"
                        "     every 4th layer: RoPE (theta=5e5, llama3-type x16) + L2 qk-norm (no params)\n"
                        "     other 3 of 4: no positional rotation, chunked attention, chunk=8192\n"
                        "RMSNorm -> FFN slot = router (16 -> 1) + shared expert + 16 experts, f=8192\n"
                        "     (B,n,5120) -> gate/up (B,n,8192) x 2 -> * -> down (B,n,5120)",
            fontsize=6.5, color=DARK, va="center")
    box(ax, 4.6, 5.15, 7.6, 0.8, "Final RMSNorm  (d=5120, eps=1e-5)", ec=BLUE, fc="#f2f7fd")
    box(ax, 4.6, 3.85, 7.6, 0.9, "LM head  5120 -> 202,048  (untied)", ec=BLUE, fc="#f2f7fd")
    for y0, y1 in [(9.0, 8.6), (6.35, 5.95), (5.15, 4.75)]:
        ax.annotate("", xy=(8.4, y1), xytext=(8.4, y0), arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))
    ax.text(8.75, 8.8, "text tokens (B, n, 5120)", fontsize=6.6, color=MUTED, va="center")

    # ---- 左列：视觉通路（config 逆向，实线），在输入侧汇入 ----
    box(ax, 0.4, 8.55, 3.6, 1.1, "Vision tower (config: vision_config)\n34 layers, d=1408, patch 14, 336px", ec=TEAL, fc="#eefaf5")
    box(ax, 0.4, 7.35, 3.6, 0.8, "Multi-modal projector\n4096 -> 5120", ec=TEAL, fc="#eefaf5")
    ax.annotate("", xy=(2.2, 7.35), xytext=(2.2, 8.55), arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))
    ax.annotate("", xy=(7.2, 8.78), xytext=(4.0, 7.75),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4,
                                connectionstyle="arc3,rad=0.22"))
    ax.text(2.6, 6.95, "image tokens (B, n_img, 5120)\njoin the token stream", fontsize=6.6, color=MUTED, ha="center")

    # ---- 右列：转述级标注（虚线） —— 证据等级降级 ----
    relay = [
        (9.0, "~109B total params"),
        (7.9, "17B activated params"),
        (6.8, "~30T training tokens"),
        (5.7, "10M token context"),
        (4.6, "\"iRoPE\": interleaved attention\nlayers without positional\nembeddings + full attention\nwith RoPE (official blog, relayed)"),
    ]
    for y, txt in relay:
        box(ax, 13.1, y - 0.42, 3.6, 0.95, txt, ec=ORANGE, ls="--", lw=1.2, fs=6.6, tc="#9c4413")
    ax.text(14.9, 3.55, "{third-party relay of the\ndelisted official blog}", ha="center", fontsize=6.4, color=MUTED)

    # ---- 图例 ----
    box(ax, 4.6, 2.3, 3.6, 0.72, "solid = config reverse\n(dual mirror, unsloth+RedHatAI)", ec=BLUE, fs=6.6)
    box(ax, 8.6, 2.3, 3.6, 0.72, "dashed = relayed claims\n(evidence level downgraded)", ec=ORANGE, ls="--", fs=6.6, tc="#9c4413")
    ax.text(8.5, 1.5, f"hand-computed from config: text {acct['text_total']:,} + vision {meta['vision_total']:,}"
                      f" = {meta['grand_total']:,}  |  per-token active {acct['active_total']:,}",
            ha="center", fontsize=7.0, color=MUTED)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["scout", "maverick"], default="scout")
    ap.add_argument("--out-name", default="run1")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    mirrors = fetch(args.model)
    base = mirrors[0][1]
    t = effective(base)

    print("\n== STEP1 认家族 ==")
    print(f"model_type={base['model_type']}  architectures={base['architectures']}  "
          f"transformers_version={base.get('transformers_version')}")
    if len(mirrors) > 1:  # 双镜像交叉：核心字段逐项比对 + 分歧字段的默认语义验证
        other = effective(mirrors[1][1])
        diff = [k for k in CORE_FIELDS if getattr(t, k, None) != getattr(other, k, None)]
        raw_empty = mirrors[1][1]["text_config"].get("no_rope_layers") == []
        same = t.no_rope_layers == other.no_rope_layers and t.layer_types == other.layer_types
        print(f"双镜像核心字段分歧：{diff or '无（逐项一致）'}")
        print(f"镜像B no_rope_layers 原始值为空表：{raw_empty} → 经 config 类默认接管后与镜像A生效值一致：{same}")

    print("\n== STEP2 结构推断表（字段 → 值 → 部件） ==")
    for f, v, part in infer_rows(t):
        print(f"  {f:30s} {v:40s} {part}")

    print("\n== STEP3 参数账 assert（式 9.1：总账 + 活账） ==")
    acct = param_account(t)
    for kk, vv in acct.items():
        print(f"  {kk:20s} {vv:,.2f}" if isinstance(vv, float) else f"  {kk:20s} {vv:,d}")
    meta = None
    try:
        meta = meta_count(base)
        print(f"  {'meta建模(对拍方)':18s} text={meta['text_total']:,} vision={meta['vision_total']:,} "
              f"grand={meta['grand_total']:,}")
        assert acct["text_total"] == meta["text_total"], "手算与 meta 建模不一致！"
        print("  [assert] 手算 = meta 建模：逐位一致 ✓")
    except ImportError:
        print("  [skip] 无 transformers/torch，meta 对拍列跳过")
    try:
        idx = index_total(args.model)
        meta_match = (meta is None) or (idx == meta["grand_total"])
        print(f"  {'index实物(对拍方)':18s} total_size/2 = {idx:,}"
              f"{'  == 手算+视觉：三方逐位一致 ✓' if idx == acct['text_total'] + (meta['vision_total'] if meta else 0) and meta_match else '  ≠ 手算（需归因）'}")
    except Exception as e:
        print(f"  [skip] index 远程读取失败（{type(e).__name__}），实物对拍列跳过")
    r = RELAY[args.model]
    vis_b = meta["vision_total"] / 1e9 if meta else 0.87
    print(f"  第三方转述：总量 ~{r['total'] / 1e9:g}B / 激活 ~{r['active'] / 1e9:g}B —— 对照手算 "
          f"{acct['text_total'] / 1e9:.1f}B(text) + {vis_b:.2f}B(vision) 与活账 {acct['active_total'] / 1e9:.2f}B"
          f"（转述仅两位有效数字，四舍五入可容）")

    print("\n== STEP4 外围 token id 锚点（词表构成推断） ==")
    tc = base["text_config"]
    for kk in ["bos_token_id", "pad_token_id", "boi_token_index", "eoi_token_index", "image_token_index"]:
        print(f"  {kk:18s} {base.get(kk) if kk in base else tc.get(kk)}")
    print("  → 特殊/多模态 token 集中在 200,000+ 区段：基础词表 ≈200,000 + 预留 ≈2,048（config 逆向推断）")

    print("\n== STEP5 证据分级 ==")
    print("  config 逆向(双镜像) > 官方源码(transformers 5.18.0 参考) > 第三方转述官方 blog > 厂商自报 > 社区")

    step3 = dict(acct)
    if meta:
        step3.update(meta_text=meta["text_total"], vision_total=meta["vision_total"],
                     grand_total=meta["grand_total"])
    try:
        step3["index_total_size_half"] = index_total(args.model)
    except Exception:
        pass
    out = dict(model=args.model, mirrors=[{"src": s, "source": tag} for s, _, tag in mirrors],
               step1=dict(model_type=base["model_type"], arch=base["architectures"],
                          version=base.get("transformers_version")),
               step2=infer_rows(t), step3=step3, step3_relay=r)
    out_path = os.path.join(OUT_DIR, f"config_reverse_{args.model}_{args.out_name}.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(f"\n[产物] {out_path}")

    if not args.no_figure and meta is not None:
        os.makedirs(FIG_DIR, exist_ok=True)
        draw_fig9_1(t, acct, meta, os.path.join(FIG_DIR, "fig-9-1-llama4-inferred-structure.png"))
        print("[图] figures/fig-9-1-llama4-inferred-structure.png")


if __name__ == "__main__":
    main()
