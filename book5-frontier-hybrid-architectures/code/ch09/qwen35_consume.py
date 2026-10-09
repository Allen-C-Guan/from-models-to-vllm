# qwen35_consume.py —— Book5 ch9 正式件：真权重①路消费点（Qwen3.5-4B 加载对账+分类型账+生成样例）
# 用途：ch9 9.2 谱系表的「真身压舱实物」——config 承诺（layer_types 24 linear + 8 full）与真权重
#       逐层互证：加载模型 → 逐层 isinstance 对账 → 分类型 KV/状态账逐层实测 → 两条生成样例。
# 【条件性降级链（大纲钉死）】Qwen3.5-4B（8.69 GiB，MPS 10-20 min）加载失败时依次降级：
#   小档真权重（若在盘）→ 随机权重同 config 构造（layer_types 对账仍成立、生成样例放弃）。
#   本轮主路径一次通过，降级链未触发（如实登记）。
# 【层号口径】config layer_types 为 0 起索引；行文人读 1 起（第 4/8/12/…/32 层全局）。
# 运行方式：cd 工作区根目录 && source env.sh &&
#   python "code/ch09/qwen35_consume.py" --out-name run1
# 产物：log/book5-ch09/qwen35_consume_{out}.json（不入库）
import argparse
import json
import os
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book5-ch09")
MODEL_ID = "Qwen/Qwen3.5-4B"          # Apache-2.0；批一夜间预取（8.69 GiB 在盘）


def typed_ledger(cfg):
    """分类型 KV/状态账（元素数口径，承第 1 章式 (1.3)）——逐层实测自真 config 字段。

    full_attention 层：每 token 每层 2·h_kv·head_dim 元素（K+V 各一份），随 n 线性（无界）；
    linear_attention 层（GDN）：固定 h_v·d_k·d_v（递归状态 S）+ (k_conv−1)·conv_dim（因果短卷积滞留），
        conv_dim = 2·h_k·d_k + h_v·d_v（q/k/v 三段都过卷积——ch5 gdn.py 同款），与 n 无关。
    """
    types = list(cfg.layer_types)
    h_kv, hd = cfg.num_key_value_heads, cfg.head_dim
    h_k, h_v = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    d_k, d_v = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    conv = cfg.linear_conv_kernel_dim
    per_tok_full = 2 * h_kv * hd
    conv_dim = 2 * h_k * d_k + h_v * d_v
    fixed_linear = h_v * d_k * d_v + (conv - 1) * conv_dim
    n_full, n_lin = types.count("full_attention"), types.count("linear_attention")
    per_tok = n_full * per_tok_full
    fixed_total = n_lin * fixed_linear
    return {"layer_types_count": {"full_attention": n_full, "linear_attention": n_lin},
            "full_per_token_per_layer": per_tok_full,
            "full_per_token_all_layers": per_tok,
            "linear_fixed_per_layer": {"recurrent_state": h_v * d_k * d_v,
                                       "conv_state": (conv - 1) * conv_dim, "total": fixed_linear},
            "linear_fixed_all_layers": fixed_total,
            "crossover_n": fixed_total / per_tok if per_tok else None}


def main(out_name):
    os.makedirs(OUT_DIR, exist_ok=True)
    from transformers import AutoConfig, AutoTokenizer    # 5.18.0（本地浅克隆同版本）
    t0 = time.perf_counter()
    report = {"model_id": MODEL_ID, "seed": 20261002, "date": "2026-10-08"}

    # ① config 零下载直读（在盘 snapshot；真实权重消费点的前半步）
    cfg = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    tcfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
    led = typed_ledger(tcfg)
    interval = getattr(tcfg, "full_attention_interval", None)
    if interval is None:  # HF config 对象未映射该字段——由 layer_types 现推（首个 full 层 0 起位次+1）
        interval = list(tcfg.layer_types).index("full_attention") + 1
    print(f"[config] layer_types：{led['layer_types_count']}（full_attention_interval="
          f"{interval}）| 全局层每 token "
          f"{led['full_per_token_all_layers']:,} 元素（无界）| 线性层固定 "
          f"{led['linear_fixed_all_layers']:,}（每层 {led['linear_fixed_per_layer']['total']:,}，与 n 无关）"
          f" | 交叉 n* = {led['crossover_n']:.0f}", flush=True)
    report["config_ledger"] = led
    report["full_attention_interval"] = interval

    # ② 真权重加载（MPS bf16；失败降级链见文件头）
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, local_files_only=True, dtype=torch.bfloat16).to(device).eval()
        report["load"] = {"device": str(device), "dtype": "bfloat16", "ok": True}
    except Exception as e:                                   # 降级链触发登记（本轮未触发）
        report["load"] = {"device": str(device), "ok": False, "error": repr(e)}
        print(f"[降级] 真权重加载失败：{e!r}——按条件性降级链登记，账本仍以 config 为正身", flush=True)
        with open(os.path.join(OUT_DIR, f"qwen35_consume_{out_name}.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return
    t_load = time.perf_counter() - t0
    mem = torch.mps.current_allocated_memory() / 1024**3 if device.type == "mps" else None
    print(f"[加载] {t_load:.0f} s | 权重驻留 {mem:.2f} GiB（{device}）", flush=True)
    report["load"].update({"wall_sec": round(t_load, 1), "resident_gib": round(mem, 2) if mem else None})

    # ③ layer_types 逐层对账（真权重侧 block_type 属性 + isinstance 双证 vs config 承诺）
    from transformers.models.qwen3_5.modeling_qwen3_5 import (Qwen3_5GatedDeltaNet,
                                                              Qwen3_5Attention)
    layers = model.model.language_model.layers if hasattr(model.model, "language_model") \
        else model.model.layers
    got, cls = [], []
    for i, layer in enumerate(layers):
        # HF 实现：linear 层挂 linear_attn、full 层挂 self_attn——两类属性互斥，逐层双证
        if hasattr(layer, "linear_attn"):
            got.append("linear_attention")
            cls.append(isinstance(layer.linear_attn, Qwen3_5GatedDeltaNet))
        else:
            got.append("full_attention")
            cls.append(isinstance(layer.self_attn, Qwen3_5Attention))
    assert all(cls), "层型与类名 isinstance 双证失败"
    cfg_types = list(cfg.text_config.layer_types if hasattr(cfg, "text_config") else cfg.layer_types)
    match = got == cfg_types
    full_idx_human = [i + 1 for i, t in enumerate(got) if t == "full_attention"]
    print(f"[对账] 32 层 isinstance 与 config layer_types 逐层{'一致' if match else '不一致！'}"
          f" | 全局层（人读 1 起）={full_idx_human}", flush=True)
    report["layer_recon"] = {"match": match, "got_types": got,
                             "full_layers_human_1idx": full_idx_human}
    assert match, "真权重层型与 config 对账失败"

    # ④ 两条生成样例（贪心解码；文本路——视觉塔不用）
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    prompts = ["用一句话解释：为什么混合架构要在大部分层用线性注意力、少量层保留全局注意力？",
               "Write one sentence about the sea."]
    samples, t_gen0 = [], time.perf_counter()
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        ids = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=96, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        samples.append({"prompt": p, "generation": gen.strip()})
        print(f"[样例] {p}\n  -> {gen.strip()[:180]}", flush=True)
    report["samples"] = samples
    report["gen_wall_sec"] = round(time.perf_counter() - t_gen0, 1)

    report["wall_sec"] = round(time.perf_counter() - t0, 1)
    path = os.path.join(OUT_DIR, f"qwen35_consume_{out_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[判定] PASS | wall {report['wall_sec']} s | 产物 {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch9 真权重①路消费点：Qwen3.5-4B 加载对账+分类型账+生成样例")
    ap.add_argument("--out-name", default="run1")
    args = ap.parse_args()
    main(args.out_name)
