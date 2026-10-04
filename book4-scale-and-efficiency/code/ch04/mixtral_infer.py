# mixtral_infer.py —— Book4 ch4 config 反推实习：Mixtral 8x7B 上第一次独立走完五步法
# 用途：正文 4.2/4.3 的代码正身。①认家族（model_type/architectures/transformers_version 年代学）
#       ②重建图纸（字段→部件映射表，含 head_dim 缺席注记）③算账对拍（参数账公式整数精确算
#       +assert 逐位：总参 46,702,792,704 / 激活参单份 12,748,853,248 / 双份 12,879,925,248
#       ——论文 47B/13B=取整口径、HF 模型卡 46.7B/12.9B 的精确底数）④读外围（词表 32k / 32k 原生
#       上下文 / sliding_window=null）⑤证据分级与默认类差异检测（checkpoint vs transformers
#       5.18.0 MixtralConfig() 默认两处差异：max_position_embeddings 32768 vs 131072、
#       router_aux_loss_coef 0.02 vs 0.001；head_dim 两边都缺席→按 d/h=128 派生）。
# 所属章节：Book4 第 4 章（Mixtral：开源 MoE 组装术的实证）。
# 运行方式：cd 工作区根目录 && source env.sh && \
#           python code/ch04/mixtral_infer.py --out-name run1
#   （CPU fp32，秒级；产物 log/book4-ch04/mixtral_infer_run1.json，不入库）
# config 来源：mistralai/Mixtral-8x7B-v0.1 raw config.json（2026-10-04 curl 直读复核，
#   与调研期 papers/02 §1.2 逐字段一致——P6 销项口径；head_dim 字段不存在是该复核的新发现）。
import argparse
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book4-ch04")

# ---- 第①步的原料：checkpoint raw config.json 逐字段（禁手抄数字入正文，一切以此为准） ----
CHECKPOINT_CONFIG = {
    "architectures": ["MixtralForCausalLM"],
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 14336,
    "max_position_embeddings": 32768,
    "model_type": "mixtral",
    "num_attention_heads": 32,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "num_local_experts": 8,
    "output_router_logits": False,
    "rms_norm_eps": 1e-05,
    "rope_theta": 1000000.0,
    "router_aux_loss_coef": 0.02,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "vocab_size": 32000,
}
# （torch_dtype=bfloat16 / transformers_version="4.36.0.dev0" / use_cache 三项为外围元数据，
#   不入结构账；transformers_version 只作第①步的年代学证据。）

# ---- 第③步的锚点：期望值全部逐位（ch1 account.py 同源公式；论文/HF 卡口径对照） ----
EXPECT = {
    "attn_per_layer": 41_943_040,          # q/o 全宽 + k/v 按 8 KV 头缩（head_dim=128 派生）
    "expert": 176_160_768,                 # 3·d·I = 3×4096×14336（SwiGLU 三矩阵）
    "moe_per_layer": 1_409_318_912,        # 8 专家 + router 4096×8
    "total": 46_702_792_704,               # 论文 47B 的精确底数
    "active_single": 12_748_853_248,       # 含 lm_head、不含输入 embedding lookup（本册口径）
    "active_double": 12_879_925_248,       # 再加输入 embedding 一份（HF 卡 12.9B 的底数）
}


def step1_family(rep):
    """认家族：model_type / architectures / transformers_version 年代学。"""
    cfg = CHECKPOINT_CONFIG
    rows = [("model_type", cfg["model_type"], "HF 家族注册名（mixtral 族）"),
            ("architectures", cfg["architectures"][0], "入仓时对应的实现类"),
            ("transformers_version", "4.36.0.dev0", "config 写入时的库版本——2023-12/2024-01 年代")]
    print("== ① 认家族 ==")
    for k, v, note in rows:
        print(f"  {k:<22} {str(v):<28} {note}")
    rep["step1_family"] = {k: v for k, v, _ in rows}


def step2_blueprint(rep):
    """重建图纸：字段→部件映射（表 4.1 的数据源）。head_dim 的缺席在这一步就暴露。"""
    c = CHECKPOINT_CONFIG
    d, h, h_kv = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"]
    fields = [
        ("hidden_size", c["hidden_size"], "残差流宽 d（每层输入输出宽度）"),
        ("num_hidden_layers", c["num_hidden_layers"], "层数 L（每层都换 MoE——无 first_k_dense 排布）"),
        ("num_attention_heads", c["num_attention_heads"], "查询头数 h"),
        ("num_key_value_heads", c["num_key_value_heads"], f"KV 头数 h_kv（GQA {h // h_kv}:1）"),
        ("(head_dim)", f"缺席 → 派生 d/h = {d // h}", "raw config 无此字段；有效头维 128（32×128=4096=d）"),
        ("intermediate_size", c["intermediate_size"], "每专家 SwiGLU 宽 w（全宽专家，非切细）"),
        ("vocab_size", c["vocab_size"], "词表 V（Llama 1/2 同款 32k SentencePiece BPE）"),
        ("num_local_experts", c["num_local_experts"], "专家数 E=8（每层各一套，32 层共 256 个专家参数组）"),
        ("num_experts_per_tok", c["num_experts_per_tok"], "每 token 激活专家数 k=2（top-2）"),
        ("max_position_embeddings", c["max_position_embeddings"], "RoPE 缓存上限=原生 32k 上下文"),
        ("rope_theta", c["rope_theta"], "RoPE 底数 θ=1e6（长窗出厂档；Book3 第 7 章对照点）"),
        ("sliding_window", "null", "无滑动窗字段——32k 全稠密注意力（config 事实，机制归后续册）"),
        ("hidden_act / rms_norm_eps", "silu / 1e-5", "SwiGLU + RMSNorm（四插槽里的两格）"),
        ("router_aux_loss_coef", c["router_aux_loss_coef"], "辅助损失系数——社区转换默认，非论文值（4.5 节）"),
        ("tie_word_embeddings", False, "untied：入口 embedding 与出口 lm_head 两份（Book3 第 5 章）"),
    ]
    print("\n== ② 重建图纸：字段 → 部件 ==")
    for k, v, note in fields:
        print(f"  {k:<28} {str(v):<30} {note}")
    rep["step2_blueprint"] = [{"field": k, "value": v, "component": n} for k, v, n in fields]
    rep["head_dim_absent_in_raw_config"] = True


def step3_account(rep):
    """算账对拍：config 公式整数精确算（承第 1 章式 (1.1)/(1.2)，无浮点误差）。"""
    c = CHECKPOINT_CONFIG
    d, L, V = c["hidden_size"], c["num_hidden_layers"], c["vocab_size"]
    h, h_kv = c["num_attention_heads"], c["num_key_value_heads"]
    dh = d // h                                   # head_dim 派生：4096/32 = 128
    E, k, I = c["num_local_experts"], c["num_experts_per_tok"], c["intermediate_size"]

    attn = d * (h * dh) + 2 * d * (h_kv * dh) + (h * dh) * d     # q/o 全宽 + k/v 按 KV 头缩
    expert = 3 * d * I                                            # 单专家 SwiGLU 三矩阵
    moe = E * expert + d * E                                      # 专家群 + 路由门 W_g (E,d)
    norms = 2 * d                                                 # 两枚 RMSNorm
    emb_head = 2 * V * d + d                                      # untied 两份 + 终态 norm
    total = L * (attn + moe + norms) + emb_head
    active_single = L * (attn + k * expert + d * E + norms) + V * d + d    # 含 lm_head 不含输入 lookup
    active_double = active_single + V * d
    expert_share = L * E * expert / total

    acc = {"head_dim_derived": dh, "attn_per_layer": attn, "expert": expert,
           "moe_per_layer": moe, "total": total,
           "active_single": active_single, "active_double": active_double,
           "expert_share_of_total": expert_share}
    assert attn == EXPECT["attn_per_layer"], (attn, EXPECT["attn_per_layer"])
    assert expert == EXPECT["expert"], (expert, EXPECT["expert"])
    assert moe == EXPECT["moe_per_layer"], (moe, EXPECT["moe_per_layer"])
    assert total == EXPECT["total"], (total, EXPECT["total"])
    assert active_single == EXPECT["active_single"], (active_single, EXPECT["active_single"])
    assert active_double == EXPECT["active_double"], (active_double, EXPECT["active_double"])
    acc["assert"] = "全部逐位通过（6/6）"

    print("\n== ③ 算账对拍（公式整数精确算，assert 逐位） ==")
    print(f"  注意力/层 A        = {attn:,}（q/o 4096×4096×2 + k/v 4096×1024×2）")
    print(f"  单专家 3·d·I       = {expert:,}（≈176.16M）")
    print(f"  MoE 槽/层          = {moe:,}（8 专家 + router 4096×8）")
    print(f"  总参               = {total:,}（论文 47B=取整 | HF 卡 46.7B）")
    print(f"  激活参·单份        = {active_single:,}（含 lm_head 不含输入 lookup——本册口径）")
    print(f"  激活参·双份        = {active_double:,}（HF 卡 12.9B | 论文 13B=取整）")
    print(f"  专家占总参         = {expert_share:.4%}（≈96.6%）")
    # 「8×7B」名字的参数几何：Mixtral = Mistral-7B 同构稠密机 + 7 份 FFN 复件 + 路由门（逐位）
    dense_eq = L * (attn + expert + norms) + emb_head                    # 同几何稠密机（≈Mistral 7B）
    assert total == dense_eq + (E - 1) * L * expert + L * d * E, "加法分解失败"
    acc["decomposition"] = {"dense_backbone": dense_eq,
                            "extra_ffn_copies": (E - 1) * L * expert,
                            "router_total": L * d * E,
                            "note": "46.7B = 7.24B 骨架 + 7×5.64B FFN 复件 + 0.001B 路由门（8×7=56B 的"
                            "名字夸大：注意力与出入口只装一份）"}
    print(f"  加法分解           = {dense_eq:,}（≈Mistral 7B 骨架）+ {(E - 1) * L * expert:,}"
          f"（7 份 FFN 复件）+ {L * d * E:,}（路由门）")
    rep["step3_account"] = acc


def step4_periphery(rep):
    """读外围三件：词表 / 上下文 / 无滑动窗字段。"""
    c = CHECKPOINT_CONFIG
    print("\n== ④ 读外围 ==")
    print(f"  vocab_size=32000：Llama 1/2 同款 SentencePiece BPE 词表（Book3 第 5 章已拆到字节）")
    print(f"  max_position_embeddings=32768：原生 32k 上下文（论文摘要 'Trained with a 32k-token"
          f" context'——非外推；Book3 第 7 章外推链的对照点）")
    print(f"  sliding_window=null：32k 全稠密注意力（config 事实；机制与收益的唯一归属在后续册）")
    rep["step4_periphery"] = {"vocab": c["vocab_size"], "ctx": c["max_position_embeddings"],
                              "sliding_window": c["sliding_window"]}


def step5_diff(rep):
    """证据分级收尾：checkpoint 实值 vs transformers 5.18.0 默认类——两处差异 + head_dim 缺席。"""
    from transformers import MixtralConfig
    default = MixtralConfig()
    compare = ["hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
               "intermediate_size", "vocab_size", "num_local_experts", "num_experts_per_tok",
               "max_position_embeddings", "sliding_window", "hidden_act", "rms_norm_eps",
               "router_aux_loss_coef", "tie_word_embeddings"]
    diffs = {}
    for f in compare:
        ck = CHECKPOINT_CONFIG[f]
        df = getattr(default, f, None)
        if ck != df:
            diffs[f] = {"checkpoint": ck, "default_class": df}
    # rope_theta 在 5.x 被折进 rope_parameters 对象（读法换代的演示）
    cfg5 = MixtralConfig(rope_theta=1000000.0)
    rope_read = {"rope_parameters": dict(cfg5.rope_parameters)}
    try:
        _ = cfg5.rope_theta
        rope_read["direct_attr_access"] = "可用"
    except AttributeError:
        rope_read["direct_attr_access"] = "AttributeError（5.x 一律读 config.rope_parameters）"

    print("\n== ⑤ 默认类差异检测（checkpoint 为准——两处差异 + 一处双方缺席） ==")
    for f, v in diffs.items():
        print(f"  ⚠ {f}: checkpoint={v['checkpoint']} vs 5.18.0 默认={v['default_class']}")
    print(f"  ⚠ head_dim: checkpoint 与默认类均无此字段（默认 None）→ 按派生 d/h=128")
    print(f"  rope 读法演示: {rope_read}")
    print(f"  红线提醒: router_aux_loss_coef=0.02 是社区转换默认——Mixtral 论文零披露训练配方，")
    print(f"            不得写成『Mistral 训练用了 0.02 的辅助损失』（正文 4.5 节）")
    rep["step5_diff"] = {"diffs": diffs, "head_dim": "absent both -> derived 128",
                         "rope_read": rope_read}
    rep["verdict"] = "PASS" if len(diffs) == 2 else f"CHECK（预期 2 处差异，实测 {len(diffs)}）"


def main(out_name):
    t0 = time.perf_counter()
    os.makedirs(OUT_DIR, exist_ok=True)
    rep = {"seed": None, "device": "cpu", "config_source":
           "mistralai/Mixtral-8x7B-v0.1 raw config.json（2026-10-04 curl 直读复核，P6 销项口径）",
           "expect": EXPECT}
    step1_family(rep)
    step2_blueprint(rep)
    step3_account(rep)
    step4_periphery(rep)
    step5_diff(rep)
    rep["wall_sec"] = round(time.perf_counter() - t0, 1)
    path = os.path.join(OUT_DIR, f"mixtral_infer_{out_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"\n[判定] {rep['verdict']} | 产物 {path} | 墙钟 {rep['wall_sec']} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch4 config 反推实习：Mixtral 8x7B 五步法（CPU 秒级）")
    ap.add_argument("--out-name", default="run1", help="产物 JSON 后缀（防覆写）")
    args = ap.parse_args()
    main(args.out_name)
