# config_infer.py —— Book3 ch5：HF config 反推架构法（五步程序 + 参数账 assert）
# 用途：把「拿到没读过的模型先读 config」演练成一段可复跑的程序（本章方法论遗产，ch9 整章复用）：
#   STEP1 认家族    —— model_type / architectures / transformers_version（版本即年代学；字段缺席也是信息）
#   STEP2 重建图纸  —— 字段→结构部件映射表（正文表 5.3 的机器可读版）
#   STEP3 算账对拍  —— 参数公式 N = V·d·(2-1_tied) + L·(2d^2 + 2d·h_kv·d_k + 3df + 2d) + d
#                      七档 LLaMA + GPT-2 124M + 本册 207M 全量手算；
#                      Llama3-8B「四方逐位」assert（手算 = index total_size/2 = 远程 header 和 = meta 建模）；
#                      Llama2-7B 差额逐字节归因（32 个 inv_freq 缓冲 × 64 维 F32 = 2,048 个浮点）
#   STEP4 读外围三件 —— tokenizer 构成实测：32k = 3 特殊 + 256 byte + 31,741 合并；
#                      128256 = 128,000 基础 + 256 特殊位（id 128000-128255）
#   STEP5 证据分级  —— config 反推的产出不是架构图，是「带证据等级的架构图」
# 另产出两张账表（正文数字来源）：tie 四拨账（GPT-2 绑 31.0% → L1/L2 解 3.89% → L3-8B 回升 13.1%
#                      → L3.2-1B 回绑 21.3%）与「固定骨架换词表」表（d=768/L=12/f=2048 tied，四档 V）。
# config 来源：优先读本地 HF 缓存实物（log/huggingface，env.sh 已重定向；调研期经镜像下载并与论文
#              表格交叉锚定）；缓存缺失的档位用内嵌 canonical 值（证据等级 = config/源码，镜像源注明）。
# 运行方式：cd 工作区根目录 && source env.sh && python code/ch05/config_infer.py [--out-name run1]
# 产物：log/book3-ch05/config_infer_{out-name}.json（不入库）；stdout 打印五步报告
# 耗时：CPU 秒级（STEP3 的 meta 建模需要 transformers/torch，缺库自动跳过该列）
import argparse
import glob
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HF_HUB = os.path.join(REPO, "log", "huggingface", "hub")
OUT_DIR = os.path.join(REPO, "log", "book3-ch05")

# ---------------- 内嵌 canonical config（{config/源码}；镜像源见调研 papers/03 来源清单） ----------------
# 字段缺席也是信息：LLaMA1/2 时代 config 不写 rope_theta / num_key_value_heads —— 缺席即默认值
# （10000 / =H 即 MHA）；transformers 5.18.0 的 LlamaConfig 默认 tie_word_embeddings=False。
CONFIGS = {
    "gpt2_124M": dict(model_type="gpt2", architectures=["GPT2LMHeadModel"], vocab_size=50257,
                      n_embd=768, n_layer=12, n_head=12, n_positions=1024,
                      tie_word_embeddings=True, _src="本地缓存 openai-community/gpt2（Book2 遗产）",
                      _repo="models--openai-community--gpt2"),
    "llama1_7B": dict(model_type="llama", architectures=["LlamaForCausalLM"], vocab_size=32000,
                      hidden_size=4096, intermediate_size=11008, num_hidden_layers=32,
                      num_attention_heads=32, tie_word_embeddings=False, _transformers_version="(4.28 时代)",
                      _src="镜像 huggyllama/llama-7b + 论文 Table 2 交叉锚定", _repo="models--huggyllama--llama-7b"),
    "llama1_13B": dict(model_type="llama", architectures=["LlamaForCausalLM"], vocab_size=32000,
                       hidden_size=5120, intermediate_size=13824, num_hidden_layers=40,
                       num_attention_heads=40, tie_word_embeddings=False,
                       _src="论文 Table 2（13.0B 档）——缓存无实物，内嵌 canonical"),
    "llama1_33B": dict(model_type="llama", architectures=["LlamaForCausalLM"], vocab_size=32000,
                       hidden_size=6656, intermediate_size=17920, num_hidden_layers=60,
                       num_attention_heads=52, tie_word_embeddings=False,
                       _src="论文 Table 2（32.5B 档）——缓存无实物，内嵌 canonical"),
    "llama1_65B": dict(model_type="llama", architectures=["LlamaForCausalLM"], vocab_size=32000,
                       hidden_size=8192, intermediate_size=22016, num_hidden_layers=80,
                       num_attention_heads=64, tie_word_embeddings=False,
                       _src="镜像 huggyllama/llama-65b + 论文 Table 2 交叉锚定", _repo="models--huggyllama--llama-65b"),
    "llama2_7B": dict(model_type="llama", architectures=["LlamaForCausalLM"], vocab_size=32000,
                      hidden_size=4096, intermediate_size=11008, num_hidden_layers=32,
                      num_attention_heads=32, tie_word_embeddings=False,
                      _src="镜像 NousResearch/Llama-2-7b-hf（snapshot 8efe6c9b）",
                      _repo="models--NousResearch--Llama-2-7b-hf"),
    "llama2_70B": dict(model_type="llama", architectures=["LlamaForCausalLM"], vocab_size=32000,
                       hidden_size=8192, intermediate_size=28672, num_hidden_layers=80,
                       num_attention_heads=64, num_key_value_heads=8, tie_word_embeddings=False,
                       _src="镜像 NousResearch/Llama-2-70b-hf + 论文 Table 1（GQA=34B/70B 双档）"),
    "llama3_8B": dict(model_type="llama", architectures=["LlamaForCausalLM"], vocab_size=128256,
                      hidden_size=4096, intermediate_size=14336, num_hidden_layers=32,
                      num_attention_heads=32, num_key_value_heads=8, rope_theta=500000.0,
                      max_position_embeddings=8192, tie_word_embeddings=False,
                      _src="镜像 NousResearch/Meta-Llama-3-8B + 论文 Table 3",
                      _repo="models--NousResearch--Meta-Llama-3-8B"),
    "llama3_405B": dict(model_type="llama", architectures=["LlamaForCausalLM"], vocab_size=128256,
                        hidden_size=16384, intermediate_size=53248, num_hidden_layers=126,
                        num_attention_heads=128, num_key_value_heads=8, rope_theta=500000.0,
                        tie_word_embeddings=False,
                        _src="论文 Table 3（405B 档）——缓存无实物，内嵌 canonical"),
    "llama32_1B": dict(model_type="llama", architectures=["LlamaForCausalLM"], vocab_size=128256,
                       hidden_size=2048, intermediate_size=8192, num_hidden_layers=16,
                       num_attention_heads=32, num_key_value_heads=8, tie_word_embeddings=True,
                       _src="镜像 unsloth + NousResearch 双镜像一致（2024-09，冻结点外——仅 tie 级 config 事实）",
                       _repo="models--NousResearch--Llama-3.2-1B"),
    # 本册 207M 定版（plan/Book3 卷级大纲 cand1；llama_slots.LLAMA_215M 同款）
    "book3_207M": dict(model_type="llama(手写)", architectures=["(本册单轨整机)"], vocab_size=32000,
                       hidden_size=1024, intermediate_size=2816, num_hidden_layers=12,
                       num_attention_heads=16, num_key_value_heads=8, rope_theta=10000.0,
                       tie_word_embeddings=False, _src="本册定版 config（untied 出厂口径）"),
}

# 官方对拍锚点（证据等级见各条；调研 papers/03 §3.2 逐条落盘）
ANCHORS = {
    "gpt2_124M": 124_439_808,      # 本地 checkpoint 计数（Book2 ch4 已核，124M 档）
    "llama1_7B": 6_738_415_616,     # 论文 6.7B / HF 卡 6.74B 的精确底数（index total_size 含缓冲，见下）
    "llama1_13B": 13_015_864_320,   # 论文 13.0B
    "llama1_33B": 32_528_943_616,   # 论文 32.5B
    "llama1_65B": 65_285_660_672,   # 论文 65.2B（四舍五入档位名 65B）
    "llama2_7B": 6_738_415_616,
    "llama2_70B": 68_976_648_192,   # 名义 70B（四舍五入档位）
    "llama3_8B": 8_030_261_248,     # 四方逐位：手算=index/2=远程 header 和=meta 建模
    "llama3_405B": 405_853_388_800, # 论文「405B trainable parameters」的精确底数
    "llama32_1B": 1_235_814_400,    # 名义 1B（tied：词表账只计一份）
    "book3_207M": 207_119_360,      # 本册定版（llama_slots 自测同数）
}

# 调研期远程读数（HTTP Range 读 safetensors header，不下载权重；papers/03 §3.2）
REMOTE_ANCHORS = {
    "llama2_7B_index_total_size": 13_476_839_424,   # bytes（bf16 权重 + 32 个 F32 inv_freq 缓冲）
    "llama3_8B_index_total_size": 16_060_522_496,   # bytes（291 张量，纯 bf16 参数）
    "llama3_8B_remote_header_sum": 8_030_261_248,   # 远程逐张量 numel 求和
}


def llama_params(V, d, f, L, H, h_kv, tied, head_dim=None):
    """LLaMA 式参数账（无 bias；RMSNorm 只有 weight）。

    N = V·d·(2-1_tied) + L·(2d^2 + 2d·h_kv·d_k + 3df + 2d) + d
    其中 d_k = d/H（或显式 head_dim）；MHA 时 h_kv·d_k = d，注意力项退化为 4d^2。
    """
    d_k = (d // H) if head_dim is None else head_dim
    vocab_acct = V * d * (1 if tied else 2)                 # 词表账：embed_tokens（+ lm_head 若解绑）
    attn = 2 * d * d + 2 * d * h_kv * d_k                   # q/o 全宽 + k/v 按 KV 头缩
    mlp = 3 * d * f                                          # SwiGLU 三矩阵
    per_layer = attn + mlp + 2 * d                           # + 每层两枚 RMSNorm
    total = vocab_acct + L * per_layer + d                   # + 末层 RMSNorm
    return {"V": V, "d": d, "f": f, "L": L, "H": H, "h_kv": h_kv, "tied": tied, "d_k": d_k,
            "vocab_acct": vocab_acct, "attn_per_layer": attn, "mlp_per_layer": mlp,
            "per_layer": per_layer, "total": total,
            "vocab_share": round(vocab_acct / total, 6)}


def gpt2_params(V, C, L, H, n_ctx):
    """GPT-2 124M 对照账（带 bias + wpe 查表 + LayerNorm 双参——与 LLaMA 式的口径差异本身是教学点）。"""
    wte = V * C
    wpe = n_ctx * C
    c_attn = C * 3 * C + 3 * C          # QKV 合并投影 + bias
    c_proj = C * C + C
    c_fc = C * 4 * C + 4 * C       # mlp.c_fc: (C -> 4C)，bias 长 4C
    c_proj2 = 4 * C * C + C        # mlp.c_proj: (4C -> C)，bias 长 C
    ln = 2 * 2 * C                      # 每块两枚 LN(weight+bias)
    block = c_attn + c_proj + c_fc + c_proj2 + ln
    total = wte + wpe + L * block + 2 * C   # ln_f 双参
    return {"wte": wte, "wpe": wpe, "block": block, "total": total,
            "wte_share_tied": round(wte / total, 6), "untie_extra": wte,
            "untie_extra_pct": round(wte / total, 6)}


def cached_config(name):
    """STEP1/3 用：优先读本地 HF 缓存的真 config.json（实物口径）；缺则返回 None。"""
    repo = CONFIGS[name].get("_repo")
    if not repo:
        return None
    hits = glob.glob(os.path.join(HF_HUB, repo, "snapshots", "*", "config.json"))
    if not hits:
        return None
    with open(hits[0], encoding="utf-8") as f:
        return json.load(f)


def cached_file(name, fname):
    repo = CONFIGS[name].get("_repo")
    if not repo:
        return None
    hits = glob.glob(os.path.join(HF_HUB, repo, "snapshots", "*", fname))
    return hits[0] if hits else None


def step1_family(report):
    print("=" * 78)
    print("STEP 1  认家族：model_type / architectures / transformers_version（字段缺席也是信息）")
    print("=" * 78)
    rows = {}
    for name in ["gpt2_124M", "llama1_7B", "llama2_7B", "llama3_8B", "llama32_1B"]:
        emb = CONFIGS[name]
        live = cached_config(name)
        row = {
            "model_type": emb["model_type"], "architectures": emb["architectures"][0],
            "tie_word_embeddings": emb["tie_word_embeddings"],
            "config_source": ("本地缓存实物" if live else emb["_src"]),
        }
        if live is not None:
            # 实物与内嵌 canonical 交叉核（词表/宽度/层数三字段必须逐位一致）
            for k in ("vocab_size", "hidden_size", "num_hidden_layers", "tie_word_embeddings"):
                if k in live:
                    assert live[k] == emb[k], f"{name} 字段 {k} 缓存 {live[k]} != canonical {emb[k]}"
            row["transformers_version"] = live.get("transformers_version", "(缺席=更早期)")
        # 字段缺席考据（LLaMA1/2 时代不写这两个字段）
        absent = [k for k in ("rope_theta", "num_key_value_heads") if k not in emb]
        row["absent_fields_note"] = ("、".join(absent) + " 缺席 → 默认值（10000 / =H 即 MHA）"
                                     if absent else "全字段在册")
        rows[name] = row
        print(f"  {name:12s} type={row['model_type']:6s} arch={row['architectures']:18s} "
              f"tie={row['tie_word_embeddings']!s:5s} src={row['config_source']}")
        print(f"               {row['absent_fields_note']}")
    report["step1_family"] = rows


FIELD_MAP = [
    # (字段, LLaMA1-7B 实值, Llama3-8B 实值, → 结构部件, 读法)
    ("vocab_size", "32000", "128256", "embed_tokens 行数 = lm_head 行数", "×hidden_size = 词表账（本章主角）"),
    ("hidden_size", "4096", "4096", "残差流宽度 d", "÷num_attention_heads = head_dim（除不尽即 config 错）"),
    ("intermediate_size", "11008", "14336", "SwiGLU 三矩阵宽 f", "名义 8d/3 取整 256 倍数；后世另有 3.5d 档"),
    ("num_hidden_layers", "32", "32", "Block 重复数 L", "参数账的大头乘数"),
    ("num_attention_heads", "32", "32", "Q 头数 h", "d_k = d/h"),
    ("num_key_value_heads", "(缺席)=h", "8", "KV 头数 h_kv（GQA）", "缺席或 =h → MHA；<h → KV 共享组（ch6 正菜）"),
    ("max_position_embeddings", "2048", "8192", "训练上下文出厂值", "≠推理硬上限（ch7 外推的起点）"),
    ("rope_theta", "(缺席)=10000", "500000.0", "RoPE 基频 θ", "缺席即默认 10000（ch4/ch7 供料）"),
    ("tie_word_embeddings", "false", "false", "输入输出嵌入绑/解", "本章主角；true 时 checkpoint 只存一份"),
    ("attention_bias / mlp_bias", "(缺席)=false", "false", "六个投影无偏置", "缺席即 false；手写 nn.Linear 默认带 bias 是对拍坑"),
    ("rms_norm_eps", "(LLaMA1-7B) 1e-06", "1e-05", "RMSNorm 分母 ε", "代际差异：LLaMA1 小档 1e-6，2/3 全系 1e-5"),
    ("bos/eos/pad_token_id", "1 / 2 / 0", "128000 / 128001 / —", "特殊 token id", "与 tokenizer_config 互核（pad 错位案例见 STEP4）"),
]


def step2_blueprint(report):
    print()
    print("=" * 78)
    print("STEP 2  重建图纸：字段 → 结构部件映射（embed → L×[RMSNorm→attn→RMSNorm→SwiGLU] → norm → head）")
    print("=" * 78)
    for row in FIELD_MAP:
        print(f"  {row[0]:28s} {row[1]:>14s} -> {row[2]:>8s}   {row[3]}")
    report["step2_field_map"] = [dict(zip(["field", "llama1_7b", "llama3_8b", "component", "how_to_read"], r))
                                 for r in FIELD_MAP]


def step3_accounts(report):
    print()
    print("=" * 78)
    print("STEP 3  算账对拍：N = V·d·(2-1_tied) + L·(2d^2 + 2d·h_kv·d_k + 3df + 2d) + d")
    print("=" * 78)
    accounts, ok_all = {}, True
    for name in ["llama1_7B", "llama1_13B", "llama1_33B", "llama1_65B", "llama2_7B", "llama2_70B",
                 "llama3_8B", "llama3_405B", "llama32_1B", "book3_207M"]:
        c = CONFIGS[name]
        acc = llama_params(c["vocab_size"], c["hidden_size"], c["intermediate_size"],
                           c["num_hidden_layers"], c["num_attention_heads"],
                           c.get("num_key_value_heads", c["num_attention_heads"]),
                           c["tie_word_embeddings"])
        anchor = ANCHORS[name]
        assert acc["total"] == anchor, f"{name} 手算 {acc['total']:,} != 锚点 {anchor:,}"
        accounts[name] = acc
        share = f"{acc['vocab_share'] * 100:.2f}%"
        print(f"  {name:12s} 手算 {acc['total']:>15,}  == 锚点 {anchor:,}  ✓   词表账占比 {share}"
              f"（{'tied 一份' if c['tie_word_embeddings'] else 'untied 两份'}）")
        ok_all &= True

    g2 = gpt2_params(50257, 768, 12, 12, 1024)
    assert g2["total"] == ANCHORS["gpt2_124M"]
    accounts["gpt2_124M"] = g2
    print(f"  {'gpt2_124M':12s} 手算 {g2['total']:>15,}  == 锚点 {ANCHORS['gpt2_124M']:,}  ✓   "
          f"wte 占比 {g2['wte_share_tied'] * 100:.2f}%（tied；解绑要另付 {g2['untie_extra']:,} = "
          f"+{g2['untie_extra_pct'] * 100:.1f}%）")

    # ---- Llama3-8B「四方逐位」：手算 = index total_size/2 = 远程 header 和 = meta 建模 ----
    l3 = accounts["llama3_8B"]["total"]
    four = {"hand": l3}
    idx_path = cached_file("llama3_8B", "model.safetensors.index.json")
    if idx_path:
        with open(idx_path, encoding="utf-8") as f:
            idx = json.load(f)
        four["index_total_size_div2"] = idx["metadata"]["total_size"] // 2
        four["index_tensors"] = len(idx["weight_map"])
    four["remote_header_sum"] = REMOTE_ANCHORS["llama3_8B_remote_header_sum"]
    try:
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        c3 = CONFIGS["llama3_8B"]
        with torch.device("meta"):
            m = LlamaForCausalLM(LlamaConfig(
                vocab_size=c3["vocab_size"], hidden_size=c3["hidden_size"],
                intermediate_size=c3["intermediate_size"], num_hidden_layers=c3["num_hidden_layers"],
                num_attention_heads=c3["num_attention_heads"],
                num_key_value_heads=c3["num_key_value_heads"],
                tie_word_embeddings=c3["tie_word_embeddings"]))
        four["meta_modeling"] = sum(p.numel() for p in m.parameters())
    except Exception as e:  # 缺库/版本变动时降级：三方仍有两方本地实读
        four["meta_modeling"] = f"SKIP（{type(e).__name__}）"
    count_keys = ["hand", "index_total_size_div2", "remote_header_sum", "meta_modeling"]
    vals = [four[k] for k in count_keys if isinstance(four.get(k), int)]
    assert len(vals) >= 3 and len(set(vals)) == 1, f"Llama3-8B 四方不逐位一致：{four}"
    print(f"\n  Llama3-8B 四方逐位：{l3:,}")
    for k, v in four.items():
        print(f"    {k:22s} = {v:,}" if isinstance(v, int) else f"    {k:22s} = {v}")
    report["step3_llama3_8b_four_way"] = four

    # ---- Llama2-7B 差额逐字节归因：32 个 inv_freq 缓冲（[64] F32）----
    l2 = accounts["llama2_7B"]["total"]
    total_size = REMOTE_ANCHORS["llama2_7B_index_total_size"]
    inv_elements = 32 * 64          # 32 层 × rotary_emb.inv_freq[64]（head_dim 128 → d/2=64）
    inv_bytes = inv_elements * 4    # F32
    div2 = total_size // 2
    assert l2 * 2 + inv_bytes == total_size, "Llama2-7B 逐字节对账失败"
    recon = {"arch_params": l2, "index_total_size": total_size, "total_size_div2": div2,
             "div2_phantom_extra": div2 - l2, "inv_freq_tensors": 32, "inv_freq_shape": [64],
             "inv_freq_elements": inv_elements, "inv_freq_dtype_bytes": 4, "inv_freq_bytes": inv_bytes,
             "byte_check": f"{l2:,}×2B + {inv_bytes:,}B = {l2 * 2 + inv_bytes:,}B == total_size ✓"}
    print(f"\n  Llama2-7B 差额归因：total_size/2 = {div2:,} 比架构真参多 {div2 - l2:,}（除 2 口径的幻影参数）")
    print(f"    真相：旧检查点多存 {inv_elements:,} 个 F32 浮点（32 张量 × [64] inv_freq，4.31 时代落盘、现代按需重算）")
    print(f"    逐字节：{l2:,}×2B(bf16) + {inv_bytes:,}B(F32 缓冲) = {total_size:,}B = index total_size 分毫不差")
    report["step3_llama2_7b_invfreq"] = recon

    # ---- tie 四拨账（正文表 5.1 数字源）----
    tie = [
        {"era": "GPT-2 124M（2019）", "tie": True, "V": 50257, "d": 768,
         "vocab_acct": g2["wte"], "total": g2["total"], "share": g2["wte_share_tied"]},
        {"era": "LLaMA1/2 7B（2023）", "tie": False, "V": 32000, "d": 4096,
         "vocab_acct": accounts["llama1_7B"]["vocab_acct"], "total": accounts["llama1_7B"]["total"],
         "share": accounts["llama1_7B"]["vocab_share"]},
        {"era": "Llama 3 8B（2024-07）", "tie": False, "V": 128256, "d": 4096,
         "vocab_acct": accounts["llama3_8B"]["vocab_acct"], "total": accounts["llama3_8B"]["total"],
         "share": accounts["llama3_8B"]["vocab_share"]},
        {"era": "Llama 3.2 1B（2024-09）", "tie": True, "V": 128256, "d": 2048,
         "vocab_acct": accounts["llama32_1B"]["vocab_acct"], "total": accounts["llama32_1B"]["total"],
         "share": accounts["llama32_1B"]["vocab_share"]},
    ]
    untied_hypo = llama_params(128256, 2048, 8192, 16, 32, 8, tied=False)
    tie[3]["untie_extra_pct_if_untied"] = round((untied_hypo["total"] / accounts["llama32_1B"]["total"] - 1), 4)
    print("\n  tie 四拨账（嵌入+输出层占参数总量比）：")
    for r in tie:
        print(f"    {r['era']:24s} tie={str(r['tie']):5s} V={r['V']:>6d}  词表账 {r['vocab_acct']:>12,}  "
              f"占比 {r['share'] * 100:5.2f}%")
    print(f"    Llama3.2-1B 若解绑要多付 +{tie[3]['untie_extra_pct_if_untied'] * 100:.1f}%——回绑的账本理由")
    report["step3_tie_switch"] = tie

    # ---- 固定骨架换词表（d=768/L=12/f=2048 tied 玩具骨架——「128k 装进小机身」的账）----
    toy = []
    for label, V in [("sp8k（Book2/3 自训）", 8192), ("LLaMA 32k", 32000),
                     ("GPT-2 50257", 50257), ("Llama3 128256", 128256)]:
        acc = llama_params(V, 768, 2048, 12, 12, 12, tied=True)
        toy.append({"label": label, "V": V, "embed": acc["vocab_acct"],
                    "tied_total": acc["total"], "embed_share": acc["vocab_share"],
                    "untie_extra_pct": acc["vocab_share"]})
    print("\n  固定骨架换词表（d=768/L=12/f=2048，tied；解绑按占比同额加价）：")
    for r in toy:
        print(f"    {r['label']:22s} V={r['V']:>6d}  嵌入 {r['embed']:>11,}  总参 {r['tied_total']:>11,}  "
              f"占比 {r['embed_share'] * 100:5.2f}%")
    report["step3_skeleton_vocab_table"] = toy
    report["step3_accounts"] = accounts
    report["step3_all_asserts_passed"] = ok_all


def step4_periphery(report):
    print()
    print("=" * 78)
    print("STEP 4  读外围三件：tokenizer_config / tokenizer 本体 / index（构成实测，本地缓存实物）")
    print("=" * 78)
    out = {}
    sp_path = cached_file("llama2_7B", "tokenizer.model")
    if sp_path:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(sp_path)
        n = sp.GetPieceSize()
        byte_ids = [sp.PieceToId(f"<0x{i:02X}>") for i in range(256)]
        specials = {"<unk>": sp.PieceToId("<unk>"), "<s>": sp.PieceToId("<s>"), "</s>": sp.PieceToId("</s>")}
        pad_id = sp.PieceToId("<pad>")   # 无 <pad>：返回 0 = unk 的 id（config 里 pad_token_id 却写 0——错位案例）
        merged = n - 3 - 256
        assert n == 32000 and merged == 31741   # 32000 = 3 特殊 + 256 byte + 31741 BPE 合并（实测直读）
        assert min(byte_ids) == 3 and max(byte_ids) == 258
        out["llama2_sp"] = {"piece_size": n, "specials": specials, "pad_piece_to_id": pad_id,
                            "byte_token_ids": "3-258（256 个）", "bpe_merged_pieces": merged,
                            "note": "无 <pad>（PieceToId 返回 0=unk）；[INST] 非特殊 token（普通文本切多片）"}
        print(f"  LLaMA2 sp：GetPieceSize={n}；特殊 3 个（unk=0/s=1//s=2）；byte token id 3-258 共 256 个；"
              f"其余 BPE 合并 {merged} 个")
        print(f"            无 <pad>——PieceToId('<pad>') 返回 {pad_id}（=unk）；而 config.pad_token_id=0：字段互相打架的现成案例")
    else:
        out["llama2_sp"] = "SKIP（本地缓存无 tokenizer.model）"
        print("  LLaMA2 sp：SKIP（本地缓存无 tokenizer.model）")

    tok3_path = cached_file("llama3_8B", "tokenizer.json")
    if tok3_path:
        import tokenizers
        tok = tokenizers.Tokenizer.from_file(tok3_path)
        v3 = tok.get_vocab_size(with_added_tokens=True)
        added = tok.get_added_tokens_decoder() if hasattr(tok, "get_added_tokens_decoder") else None
        out["llama3_tokenizers"] = {"vocab_size": v3,
                                    "added_tokens": {str(i): getattr(t, "content", str(t))
                                                     for i, t in (added or {}).items()} if added
                                    else "（API 不可用，见 tokenizer_config）"}
        added_names = {i: getattr(t, "content", str(t)) for i, t in (added or {}).items()}
        reserved = sum(1 for name in added_names.values() if "reserved" in name)
        non_reserved = {i: name for i, name in added_names.items() if "reserved" not in name}
        assert v3 == 128256 and len(added or {}) == 256
        out["llama3_tokenizers"]["decomposition"] = f"128000 基础 + 256 特殊（id 128000-128255）"
        out["llama3_tokenizers"]["reserved_count"] = reserved
        out["llama3_tokenizers"]["non_reserved"] = {str(i): t for i, t in non_reserved.items()}
        print(f"  Llama3 tokenizers：vocab={v3:,} = 128,000 基础 + 256 特殊位（id 128000-128255）")
        if added:
            print(f"            非保留位仅 {len(non_reserved)} 个：" +
                  "、".join(f"{t}({i})" for i, t in sorted(non_reserved.items())) +
                  f"；其余 {reserved} 个为 <|reserved_special_token_N|> 占位")
    else:
        out["llama3_tokenizers"] = "SKIP（本地缓存无 tokenizer.json）"
        print("  Llama3 tokenizers：SKIP（本地缓存无 tokenizer.json）")
    report["step4_periphery"] = out


def step5_evidence(report):
    print()
    print("=" * 78)
    print("STEP 5  证据分级：config 反推的产出 = 带证据等级的架构图")
    print("=" * 78)
    ladder = [
        ("config/源码（本缓存实物直读）", "字段值本身：vocab_size=128256、tie=false、h_kv=8"),
        ("官方论文（表格交叉锚定）", "名义参数档位：8B/70B/405B、128K 词表声明"),
        ("config/源码（镜像，无官方 repo 逐字节）", "meta-llama gated——全部经 NousResearch/unsloth 镜像，值已与论文表格对上"),
        ("第三方复测（远程 header，调研期读数）", "index total_size、逐张量 numel 和（HTTP Range，不下载权重）"),
        ("{二手推断}", "tying 的『为什么』：三篇 LLaMA 论文全部沉默（grep 负结论），社区解释只能标二手推断"),
    ]
    for lv, ex in ladder:
        print(f"  [{lv}]  {ex}")
    report["step5_evidence_ladder"] = [dict(zip(["level", "example"], r)) for r in ladder]


def main():
    ap = argparse.ArgumentParser(description="Book3 ch5 config 反推五步法（CPU 秒级）")
    ap.add_argument("--out-name", default="run1")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {"meta": {"script": "code/ch05/config_infer.py",
                       "note": "STEP1-3 的 assert 全部通过才落盘本 JSON；canonical 值与锚点出处见脚本注释"}}
    step1_family(report)
    step2_blueprint(report)
    step3_accounts(report)
    step4_periphery(report)
    step5_evidence(report)
    out_path = os.path.join(OUT_DIR, f"config_infer_{args.out_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[完成] 全部 assert 通过 -> {out_path}")


if __name__ == "__main__":
    main()
