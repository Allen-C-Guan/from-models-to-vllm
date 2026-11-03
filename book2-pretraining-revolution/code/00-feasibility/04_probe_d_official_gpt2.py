# 04_probe_d_official_gpt2.py —— 探针(d)：openai-community/gpt2 官方权重全链路实测（ch11 对拍基线）
# transformers 5.18.0 下加载 GPT2LMHeadModel+tokenizer，记录真实 API 形态与模块路径；
# 固定 prompt 集（短/长/刁钻）CPU 前向取 logits 并存盘（数值对拍一律 CPU 口径）。
# 用法：python 04_probe_d_official_gpt2.py；产物落 log/book2-feasibility/{probe_d_official_gpt2.json, gpt2_logits_baseline.pt}

import json
import os
import time

import torch

SEED = 20261002
OUT_DIR = os.path.join("log", "book2-feasibility")
MODEL_ID = "openai-community/gpt2"

# 固定 prompt 集（ch11 权重对拍基线；覆盖短事实/长段落/重复诱导/unicode/代码风/算术/续写）
PROMPTS = [
    "The capital of France is",
    "The quick brown fox jumps over the lazy dog. The quick brown fox",
    "In 1776, the Second Continental Congress declared the independence of the colonies as the United States. Historians agree that",
    "Der Begriff Nähe bedeutet im Deutschen",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "2 + 2 = 4, 3 + 3 = 6, 4 + 4 =",
    "a",
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    report = {"meta": {"date": "2026-10-02", "model_id": MODEL_ID,
                       "transformers": __import__("transformers").__version__,
                       "torch": torch.__version__}}

    import transformers
    from transformers import AutoTokenizer, GPT2LMHeadModel

    # ---- 加载（缓存命中）与 API 形态记录 ----
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    load_tok_sec = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter()
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID)  # 默认 dtype 与类名形态记入报告
    load_model_sec = round(time.perf_counter() - t0, 2)

    cfg = model.config
    cfg_fields = {k: getattr(cfg, k, None) for k in
                  ["n_layer", "n_head", "n_embd", "vocab_size", "n_positions",
                   "layer_norm_epsilon", "activation_function"]}
    # transformers 5.x 的统一命名也可能并存——探测两种字段名
    for k in ["num_hidden_layers", "hidden_size", "num_attention_heads"]:
        if hasattr(cfg, k):
            cfg_fields[f"[v5命名]{k}"] = getattr(cfg, k)

    # 模块路径与 tied 权重验证（v5 的顶层结构可能与旧版 transformer. 前缀不同，如实记录）
    top_modules = [name for name, _ in model.named_children()]
    wte_path = [n for n, _ in model.named_modules() if n.endswith("wte")]
    lm_head_path = [n for n, _ in model.named_modules() if n.endswith("lm_head")]
    emb_w = dict(model.named_parameters())
    tied = None
    pnames = list(emb_w.keys())
    wte_name = [n for n in pnames if n.endswith("wte.weight")][0]
    head_name = [n for n in pnames if "lm_head" in n and n.endswith("weight")]
    if head_name:
        tied = emb_w[wte_name] is emb_w[head_name[0]]
    else:
        # v5 里 tied lm_head 可能不作为独立参数出现在 named_parameters——用模块属性直查
        import re as _re
        lm_mod = [m for n, m in model.named_modules() if n.endswith("lm_head")]
        wte_mod = [m for n, m in model.named_modules() if n.endswith("wte")]
        if lm_mod and wte_mod and hasattr(lm_mod[0], "weight") and hasattr(wte_mod[0], "weight"):
            tied = lm_mod[0].weight is wte_mod[0].weight
    tied_cfg = getattr(cfg, "tie_word_embeddings", None)

    n_params = sum(p.numel() for p in model.parameters())
    # tied 时 lm_head 与 wte 共享存储：按 data_ptr 去重后的真实参数量
    n_params_dedup = sum(p.numel() for p in {p.data_ptr(): p for p in model.parameters()}.values())
    dtypes = {str(p.dtype) for p in model.parameters()}

    report["api"] = {
        "load_tokenizer_sec": load_tok_sec, "load_model_sec_cache_hit": load_model_sec,
        "tokenizer_class": type(tok).__name__, "model_class": type(model).__name__,
        "config_fields": cfg_fields,
        "top_level_modules": top_modules,
        "wte_module": wte_path, "lm_head_module": lm_head_path,
        "lm_head_tied_to_wte": tied,
        "config_tie_word_embeddings": tied_cfg,
        "param_names_sample": pnames[:6] + [n for n in pnames if "lm_head" in n][:3],
        "param_dtype": sorted(dtypes),
        "n_params_sum": n_params,
        "n_params_dedup": n_params_dedup,
        "vocab_size_tokenizer": tok.vocab_size,
        "tokenizer_roundtrip": {
            s: tok.decode(tok.encode(s)) == s
            for s in ["Nähe café 🚀", "  double  spaces\n\nnewline\ttab", "距离很近"]
        },
    }
    print(json.dumps(report["api"], ensure_ascii=False, indent=1)[:900], flush=True)

    # ---- CPU logits 基线（数值对拍一律 CPU） ----
    model.eval()
    baseline = {}
    logits_store = {}
    topk = []
    with torch.no_grad():
        for i, p in enumerate(PROMPTS):
            enc = tok(p, return_tensors="pt")
            out = model(**enc)
            last = out.logits[0, -1, :]  # (50257,)
            probs = torch.softmax(last, dim=-1)
            t5v, t5i = torch.topk(probs, 5)
            topk.append({"prompt": p,
                         "n_prompt_tokens": enc["input_ids"].shape[1],
                         "top5": [{"token": tok.decode([int(i_)]) if i_ not in (198,) else "\\n",
                                   "id": int(i_), "p": round(float(v), 5)}
                                  for v, i_ in zip(t5v, t5i)]})
            baseline[f"p{i}"] = {"argmax_id": int(last.argmax()),
                                 "mean_logit": round(float(last.mean()), 6),
                                 "logit_l2": round(float(last.norm()), 6),
                                 "sum_logprob": round(float(
                                     torch.log_softmax(
                                         out.logits[0, :-1, :], dim=-1
                                     ).gather(1, enc["input_ids"][0, 1:].unsqueeze(1)).sum()), 4)}
            logits_store[f"p{i}"] = last.clone()
    torch.save(logits_store, os.path.join(OUT_DIR, "gpt2_logits_baseline.pt"))

    # 贪婪生成 20 token（CPU，直观验证权重加载正确）
    gen_examples = []
    with torch.no_grad():
        for p in [PROMPTS[0], PROMPTS[5]]:
            ids = tok(p, return_tensors="pt")
            g = model.generate(**ids, max_new_tokens=20, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            gen_examples.append({"prompt": p, "gen": tok.decode(g[0][ids["input_ids"].shape[1]:])})

    # ---- wall-clock：同一 prompt 批 CPU vs MPS 各一次前向（仅计时，不做数值对拍） ----
    timing = {}
    tok.pad_token = tok.eos_token  # GPT-2 无 pad token，标准做法用 eos 代位（配合 attention_mask）
    with torch.no_grad():
        enc = tok(PROMPTS, return_tensors="pt", padding=True)
        for dev in ["cpu", "mps"]:
            m = model.to(dev)
            inps = {k: v.to(dev) for k, v in enc.items()}
            for _ in range(2):  # 预热
                m(**inps)
            if dev == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                m(**inps)
            if dev == "mps":
                torch.mps.synchronize()
            timing[f"fwd_batch7_{dev}_sec"] = round((time.perf_counter() - t0) / 5, 4)
        model.to("cpu")

    report["topk"] = topk
    report["baseline_summary"] = baseline
    report["greedy_gen_examples"] = gen_examples
    report["timing"] = timing
    report["download_facts"] = {
        "required_files_mb": {
            "model.safetensors": 522.7, "tokenizer.json": 1.3, "vocab+merges+configs": 2.3},
        "note": "gpt2 仓全量 26 文件≈1.6GB（含 TF/Flax/Rust/ONNX 重复权重格式）；只取 PT 必需集 ≈527MB。"
                "首次下载经未鉴权 HF 限速，实测出现两次挂起（bert 262MB 处、gpt2 辅助 blob 处），"
                "重试+断点续传可完成；完整 model.safetensors 在 21:17-21:51 首次尝试内完成。",
    }

    out_path = os.path.join(OUT_DIR, "probe_d_official_gpt2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("top5[0]:", json.dumps(topk[0], ensure_ascii=False))
    print("timing:", timing)
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
