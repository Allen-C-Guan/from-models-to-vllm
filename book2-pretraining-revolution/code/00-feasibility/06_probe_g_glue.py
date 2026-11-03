# 06_probe_g_glue.py —— 探针(g)：bert-base 与 bert-tiny 在 SST-2 上微调可行性（ch3 mini-GLUE 底座）
# datasets 5.0.1 实测 GLUE 加载 API；两档模型各微调 ~100 步：sec/step、MPS 内存峰值、loss 趋势；
# MRPC 全量(3.7k)以 SST-2 实测 sec/step 外推一步预算；以 Book1 的 11.76M=43.5ms/step 锚做交叉核对。
# 用法：python 06_probe_g_glue.py [--steps 100]；产物落 log/book2-feasibility/probe_g_glue.json

import argparse
import json
import os
import time

import torch

from importlib.util import spec_from_file_location, module_from_spec

_here = os.path.dirname(os.path.abspath(__file__))
_spec = spec_from_file_location("probe_a", os.path.join(_here, "01_probe_a_gpt2_124m.py"))
probe_a = module_from_spec(_spec)
_spec.loader.exec_module(probe_a)  # 复用 MpsMemSampler

SEED = 20261002
OUT_DIR = os.path.join("log", "book2-feasibility")


def load_glue():
    """datasets 5.0.1 实测 API：nyu-mll/glue（旧名 glue 会重定向）。"""
    from datasets import load_dataset
    sst2 = load_dataset("nyu-mll/glue", "sst2")
    mrpc = load_dataset("nyu-mll/glue", "mrpc")
    lic = str(getattr(sst2["train"].info, "license", ""))
    return sst2, mrpc, lic


def tokenize_ds(ds, tok, max_len=128):
    """SST-2/MRPC 句对统一编码为 (input_ids, attention_mask, label)。"""
    def _enc(batch):
        if "sentence2" in batch:  # MRPC 句对
            return tok(batch["sentence1"], batch["sentence2"], truncation=True,
                       max_length=max_len, padding="max_length")
        return tok(batch["sentence"], truncation=True, max_length=max_len,
                   padding="max_length")

    cols = ["input_ids", "attention_mask"]
    if "token_type_ids" in ds.column_names:
        pass  # tokenizer 自带返回
    return ds.map(_enc, batched=True, remove_columns=[c for c in ds.column_names
                                                      if c not in ("label", "idx")])


def finetune_bert(model_id, enc_train, n_steps, device="mps", batch=32, lr=2e-5):
    """HF 编码器 + 随机初始化分类头微调 n_steps；返回 sec/step、内存峰值、loss 曲线。"""
    from transformers import AutoModelForSequenceClassification
    torch.manual_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_encoder = n_params - model.classifier.weight.numel() - model.classifier.bias.numel()

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    cols = [c for c in ["input_ids", "attention_mask", "token_type_ids"] if c in enc_train.column_names]
    dl = torch.utils.data.DataLoader(
        enc_train.with_format("torch", columns=cols + ["label"]), batch_size=batch, shuffle=True)
    it = iter(dl)
    losses = []
    mem = probe_a.MpsMemSampler()
    mem.start()
    model.train()
    for _ in range(2):  # 预热（shader 编译/优化器懒分配）
        b = next(it)
        out = model(**{k: v.to(device) for k, v in b.items() if k != "label"},
                    labels=b["label"].to(device))
        out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    for step in range(n_steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl); b = next(it)
        out = model(**{k: v.to(device) for k, v in b.items() if k != "label"},
                    labels=b["label"].to(device))
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        opt.step()
        losses.append(out.loss.item())
    torch.mps.synchronize()
    sec_total = time.perf_counter() - t0
    m = mem.stop_and_report()
    del model, opt
    torch.mps.empty_cache()
    return {"model_id": model_id, "n_params": n_params, "n_params_encoder": n_encoder,
            "batch": batch, "max_len": 128, "steps": n_steps,
            "sec_per_step": round(sec_total / n_steps, 4),
            "loss_first10": round(sum(losses[:10]) / 10, 4),
            "loss_last10": round(sum(losses[-10:]) / 10, 4),
            "mem_peak_mib": m["peak_current_mib"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    from transformers import AutoTokenizer
    sst2, mrpc, lic = load_glue()
    report = {"meta": {"date": "2026-10-02", "seed": SEED, "device": "mps",
                       "datasets_api": "load_dataset('nyu-mll/glue', 'sst2'|'mrpc')",
                       "sst2_splits": {k: len(v) for k, v in sst2.items()},
                       "mrpc_splits": {k: len(v) for k, v in mrpc.items()},
                       "glue_license_field": lic or "(info.license 为空——许可证见 HF 卡片，见调研笔记)"}}

    # bert-base（110M）：SST-2 微调 100 步
    tok_base = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    enc = tokenize_ds(sst2["train"], tok_base)
    r_base = finetune_bert("google-bert/bert-base-uncased", enc, args.steps)
    report["bert_base_sst2"] = r_base
    print("bert-base:", json.dumps(r_base, ensure_ascii=False), flush=True)

    # bert-tiny（4.4M）：同协议 100 步（给「课堂快档」口径）
    tok_tiny = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
    enc_t = tokenize_ds(sst2["train"], tok_tiny)
    r_tiny = finetune_bert("google/bert_uncased_L-2_H-128_A-2", enc_t, args.steps)
    report["bert_tiny_sst2"] = r_tiny
    print("bert-tiny:", json.dumps(r_tiny, ensure_ascii=False), flush=True)

    # MRPC 全量预算：3668 例 / batch32 = 115 步/epoch；用 bert-base 实测 sec/step 外推 3 epoch
    n_mrpc = len(mrpc["train"])
    steps_per_epoch = -(-n_mrpc // 32)
    report["mrpc_full_budget"] = {
        "n_train": n_mrpc, "steps_per_epoch_b32": steps_per_epoch,
        "est_sec_3epoch_bert_base_mps": round(steps_per_epoch * 3 * r_base["sec_per_step"], 1),
        "est_sec_3epoch_bert_tiny_mps": round(steps_per_epoch * 3 * r_tiny["sec_per_step"], 1),
    }

    # Book1 锚交叉核对：11.76M 模型 43.5 ms/step（MPS）——与 bert-tiny(4.4M) 实测同量级则口径一致
    report["book1_anchor_check"] = {
        "book1": "11.76M 模型 = 43.5 ms/step (MPS, Book1 实测)",
        "bert_tiny_measured_ms": round(r_tiny["sec_per_step"] * 1000, 1),
        "note": "同量级即口径自洽（序列长/词表/优化器差异会造成倍率级偏差）",
    }

    out_path = os.path.join(OUT_DIR, "probe_g_glue.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
