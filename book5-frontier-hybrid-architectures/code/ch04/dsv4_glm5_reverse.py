# dsv4_glm5_reverse.py —— Book5 第4章：发布配置、meta主模型、微缩前向与逻辑缓存账。
# source env.sh && python code/ch04/dsv4_glm5_reverse.py --out-name verify
# configs内为公开配置和少量checkpoint元数据，默认离线，不下载或实例化全尺寸实权重。
# 全尺寸全部参数/buffer断言meta；主模型参数与state_dict缓冲区分开，不含MTP。
# 微缩前向CPU FP32，不是质量实验；V4微缩hc_mult=1，mHC数学另作局部验证。
# 账本统一元素数并分列2byte换算与混精载荷假设，不等于HF eager实际内存。
# --plot-layer-map / --plot-dsv4-map 写入独立log子目录；JSON及图目录均拒覆盖。
import argparse
import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone
import transformers

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = str(next((p for p in Path(__file__).resolve().parents if (p / "env.sh").is_file()), Path.cwd()))
CONFIG_DIR = Path(HERE)/"configs"
AUDIT_DIR = os.path.join(REPO_ROOT, "log", "book5-ch11", "params_audit")
OUT_DIR = os.path.join(REPO_ROOT, "log", "book5-ch04")
FIG_DIR = os.path.join(REPO_ROOT, "book5-frontier-hybrid-architectures", "figures")
SEED = 20261002

N1M = 1_048_576  # 1M token 上下文账本统一口径
V32_LAYERS, V32_MLA_PER_TOK = 61, 512 + 64  # DSV3.2 参照机：61 层 MLA，latent 512 + rope 64 = 576 元素/token/层


def load_raw(name):
    with (CONFIG_DIR/f"{name}-config.json").open() as f:
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
    kw = {k: v for k, v in raw.items() if k not in ("architectures", "dtype", "torch_dtype",
                                                    "quantization_config", "expert_dtype")}
    cfg = cfg_cls(**kw)
    with torch.device("meta"):
        model = model_cls(cfg)
    assert all(p.device.type=="meta" for p in model.parameters())
    assert all(b.device.type=="meta" for b in model.buffers())
    total = sum(p.numel() for p in model.parameters())  # meta 张量 numel 可读、无数据
    keys = list(model.state_dict().keys())
    idx_layers = sorted({int(k.split(".")[2]) for k in keys if ".indexer." in k})
    compr_layers = sorted({int(k.split(".")[2]) for k in keys if ".compressor." in k})
    return {"total": total, "total_B": round(total / 1e9, 3), "n_state_keys": len(keys),
            "state_elements_including_buffers": sum(v.numel() for v in model.state_dict().values()),
            "scope": "当前HF主模型；不含MTP；参数数与state_dict的buffer分开",
            "indexer_layers_meta": len(idx_layers), "indexer_pos_head": idx_layers[:8],
            "compressor_layers_meta": len(compr_layers),
            "key_samples": [k for k in keys if ".indexer." in k][:4] or [k for k in keys if "compressor" in k][:4],
            "layer_types_len": len(getattr(cfg, "layer_types", []) or [])}


@torch.no_grad()
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
        max_position_embeddings=128, _attn_implementation="eager")
    model = DeepseekV4ForCausalLM(cfg).to(torch.float32).eval()
    x = torch.randint(0, 512, (2, 64))  # (B=2, n=64)
    logits = model(input_ids=x, use_cache=False).logits  # (2, 64, 512)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 512).float(), x[:, 1:].reshape(-1))
    return {"params": sum(p.numel() for p in model.parameters()), "logits_shape": list(logits.shape),
            "loss": round(loss.item(), 3), "ln_vocab": round(math.log(512), 3),
            "layer_types": cfg.layer_types, "partial_rope": cfg.partial_rotary_factor}


@torch.no_grad()
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
        max_position_embeddings=128, num_nextn_predict_layers=0, _attn_implementation="eager")
    model = GlmMoeDsaForCausalLM(cfg).to(torch.float32).eval()
    idx_layers = sorted({int(k.split(".")[2]) for k in model.state_dict() if ".indexer." in k})
    per_idx = sum(p.numel() for n, p in model.named_parameters() if ".indexer." in n) // len(idx_layers)
    x = torch.randint(0, 512, (2, 64))
    logits = model(input_ids=x, use_cache=False).logits  # (2, 64, 512)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 512).float(), x[:, 1:].reshape(-1))
    return {"params": sum(p.numel() for p in model.parameters()), "logits_shape": list(logits.shape),
            "loss": round(loss.item(), 3), "ln_vocab": round(math.log(512), 3),
            "indexer_layers_mini": idx_layers, "shared_layers_no_indexer": 8 - len(idx_layers),
            "per_layer_indexer_params": per_idx}


def cross_check_glm52_weights(res_glm52):
    """534M 差值的精确闭合：57 个 shared 层 × 每层索引器参数 = GLM-5 与 5.2 的 checkpoint 全量差。

    params_audit 产物（log/book5-ch11/params_audit/*.json）为 safetensors 分片头亲算（不下载权重）。
    """
    fixture=json.loads((CONFIG_DIR/"checkpoint-metadata.json").read_text())
    total5=fixture["glm5"]["aggregate"]["total"]
    total52=fixture["glm52"]["aggregate"]["total"]
    diff=total5-total52
    expected=res_glm52["shared_drop_total"]
    assert diff==expected
    return dict(aggregate_total_glm5=total5,aggregate_total_glm52=total52,
                diff=diff,shared_drop_total=expected,diff_equals_57x=True,
                indexer_layers_declared=fixture["glm52"]["indexer_layers"],
                note="随书官方元数据+索引清单摘要；不是本次CLI重新联网读权重")



# ---------------- ③ 1M 分类型账本手算（元素口径） ----------------
def ledger_1m(n=N1M):
    """逻辑数据载荷（batch=1），不等于当前HF eager实际缓存或峰值内存。

    V4主记录=压缩条目+最多W条局部KV，K=V一份；索引键另列。
    混精列仅算448 FP8+64 BF16=576bytes/记录，尚未含scale、压缩暂存及对齐。
    KDA/GDN计状态矩阵与最小k-1卷积历史，真实实现可分配k槽并用FP32状态。
    """
    if n < 1:
        raise ValueError("n必须为正")
    out={}
    for tag,name in [("DSV4-Flash","dsv4flash"),("DSV4-Pro","dsv4pro")]:
        raw=load_raw(name)
        ratios=raw["compress_ratios"][:raw["num_hidden_layers"]]
        width=raw["head_dim"];rope=raw["qk_rope_head_dim"];window=raw["sliding_window"]
        compressed=sum(n//r for r in ratios if r)
        local=len(ratios)*min(n,window)
        entries=compressed+local
        index=sum(n//r for r in ratios if r==4)*raw["index_head_dim"]
        out[tag]=dict(main_compressed_elements=compressed*width,
                      main_local_elements=local*width,total=entries*width,
                      gb_at_2bytes=entries*width*2/1e9,
                      mixed_payload_gb=entries*(width-rope+2*rope)/1e9,
                      indexer_elements=index,indexer_gb_at_2bytes=index*2/1e9,
                      excluded="压缩器未闭合/重叠暂存、量化scale、对齐和临时计算")
    v32=V32_LAYERS*V32_MLA_PER_TOK*n
    out["DSV3.2_ref"]=dict(total=v32,
        flash_vs_v32_main_elements=out["DSV4-Flash"]["total"]/v32,
        pro_vs_v32_main_elements=out["DSV4-Pro"]["total"]/v32,
        note="仅主记录元素比；非论文部署字节比，双方索引键另计")
    glm=load_raw("glm52");layers=glm["num_hidden_layers"]
    per=glm["kv_lora_rank"]+glm["qk_rope_head_dim"]
    full=sum(t=="full" for t in glm["indexer_types"])
    main=layers*per*n;index=full*glm["index_head_dim"]*n
    out["GLM-5.2"]=dict(main_elements=main,indexer_elements=index,
                       total=main+index,main_gb_at_2bytes=main*2/1e9,
                       total_gb_at_2bytes=(main+index)*2/1e9,
                       note="MLA吸收形态的逻辑账；HF eager会展开每头K/V")
    # K3配置几何见第1章；此处将最小卷积历史与状态矩阵分开。
    main=24*576*n;core=69*96*128*128;conv=69*3*96*128*3
    out["K3"]=dict(main_elements=main,state_core_elements=core,
                   conv_history_min_elements=conv,total=main+core+conv,
                   total_gb_at_2bytes=(main+core+conv)*2/1e9,
                   core_vs_main=core/main)
    raw=load_raw("qwen35-35b")["text_config"]
    g=raw["layer_types"].count("full_attention");l=raw["layer_types"].count("linear_attention")
    main=g*2*raw["num_key_value_heads"]*raw["head_dim"]*n
    dk=raw["linear_key_head_dim"];dv=raw["linear_value_head_dim"]
    hk=raw["linear_num_key_heads"];hv=raw["linear_num_value_heads"]
    core=l*hv*dk*dv
    conv=l*(2*hk*dk+hv*dv)*(raw["linear_conv_kernel_dim"]-1)
    out["Qwen3.5-35B"]=dict(main_elements=main,state_core_elements=core,
                           conv_history_min_elements=conv,total=main+core+conv,
                           total_gb_at_2bytes=(main+core+conv)*2/1e9,
                           evaluation_length=n,declared_native_length=raw["max_position_embeddings"],
                           card_extension_length=1_010_000,
                           note="在1048576处只作同长度算术比较，超出卡面1010000；状态实际可用FP32")
    return out


# ---------------- ④ 教学图（独立输出目录；不写正式书稿目录） ----------------
def plot_layer_map():
    from make_figures import draw_glm
    path=Path(FIG_DIR)/"fig-4-2-glm52-indexshare.png"
    draw_glm(path,load_raw("glm52"));return str(path)

def plot_dsv4_map():
    from make_figures import draw_dsv4
    path=Path(FIG_DIR)/"fig-4-1-dsv4-teardown.png"
    draw_dsv4(path);return str(path)

# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="Book5 ch4: DSV4/GLM-5(.2) config 逆向 + 1M 账本 + meta 对拍")
    ap.add_argument("--out-name", default="run1", help="产物 JSON 后缀（防覆写）")
    ap.add_argument("--plot-layer-map", action="store_true", help="出图 4.2（GLM-5.2 层模式图）")
    ap.add_argument("--plot-dsv4-map", action="store_true", help="出图 4.1（DSV4 整机结构推断图）")
    args = ap.parse_args()
    path = os.path.join(OUT_DIR, f"dsv4_glm5_reverse_{args.out_name}.json")
    if os.path.exists(path):
        raise FileExistsError(path)
    global FIG_DIR
    FIG_DIR=os.path.join(OUT_DIR,f"figures_{args.out_name}")
    if args.plot_layer_map or args.plot_dsv4_map:
        Path(FIG_DIR).mkdir(parents=True,exist_ok=False)
    os.makedirs(OUT_DIR, exist_ok=True)

    from transformers import (DeepseekV4Config, DeepseekV4ForCausalLM,
                              GlmMoeDsaConfig, GlmMoeDsaForCausalLM)

    out = {"seed": SEED,"date":datetime.now(timezone.utc).isoformat(),
           "torch":torch.__version__,"transformers":transformers.__version__,"device":"CPU/meta"}
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
    with open(path, "x") as f:
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
