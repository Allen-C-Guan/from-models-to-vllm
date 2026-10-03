# -*- coding: utf-8 -*-
# 用途：mini-GLUE 微调实验——{bert-tiny, bert-base} × {SST-2 单句, MRPC 句对} 四格微调：
#       加载预训练 BERT → 分类头随机初始化 → AdamW 微调（论文口径 batch 32 / lr 2e-5 / 3 epochs）
#       → 逐 epoch 验证集准确率 vs 多数类基线；实测 sec/step、MPS 内存峰值、逐步 loss 曲线。
#       HF transformers/datasets 为借用工具（对拍身份，见正文 3.6 借用框），算法不在此展开。
# 所属章节：《预训练革命》第 3 章 3.6 节（mini-GLUE 实验）
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch03/glue_finetune.py" --model tiny --task sst2
#       档位：fast = --model tiny --task sst2（3 epoch 约 1 分钟）/ full = --model base --task sst2 --epochs 1（约 15 分钟）
#            （--model base --task mrpc 约 2.5 分钟；全部产物落 log/book2-ch03/，数据集与权重不入库）

import argparse
import json
import os
import threading
import time

import torch

SEED = 20261002
MODELS = {  # model 键 → HF 模型 id（均为 uncased，对齐 BERT 论文「除 NER 外全部任务用 uncased」口径）
    "tiny": "google/bert_uncased_L-2_H-128_A-2",  # L=2/H=128/A=2 ≈ 4.4M：Google 官方缩小版（2019-05 随 Turc et al. 家族发布）
    "base": "google-bert/bert-base-uncased",      # L=12/H=768/A=12 ≈ 109.5M：BERT_BASE 同源权重
}
OUT_DIR = os.path.join("log", "book2-ch03")


class MpsMemSampler(threading.Thread):
    """后台线程采样 MPS 分配内存峰值（全书统一口径：torch.mps.current_allocated_memory，MiB）。"""

    def __init__(self, interval=0.03):
        super().__init__(daemon=True)
        import torch.mps as mps
        self._mps, self.interval, self.peak = mps, interval, 0.0
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                self.peak = max(self.peak, self._mps.current_allocated_memory() / 1024**2)
            except Exception:
                pass
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set(); self.join(timeout=1.0)
        return round(self.peak, 1)


def load_task(task):
    """加载 GLUE 子任务（datasets 5.0.1，nyu-mll/glue）。SST-2=单句情感 / MRPC=句对释义。"""
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", task)
    keys = ["sentence1", "sentence2"] if "sentence1" in ds["train"].column_names else ["sentence"]
    return ds, keys


def tokenize(ds, tok, keys, max_len=128):
    """编码为 input_ids/attention_mask/token_type_ids。
    单句: [CLS] x1..xn [SEP]；句对: [CLS] A [SEP] B [SEP]（token_type_ids 自动 0/1 两段）——
    即预训练 NSP 段对拼包格式的微调复用（正文 3.3/3.5 的「打包即接口」）。"""
    def _enc(b):
        if len(keys) == 2:
            return tok(b[keys[0]], b[keys[1]], truncation=True, max_length=max_len, padding="max_length")
        return tok(b[keys[0]], truncation=True, max_length=max_len, padding="max_length")
    return ds.map(_enc, batched=True, remove_columns=[c for c in ds.column_names if c not in ("label",)])


def evaluate(model, enc, device, batch=64):
    """验证集准确率：encoder (B,n)→(B,n,H) 取 [CLS] 池化后分类，argmax 对比标签。"""
    cols = [c for c in ("input_ids", "attention_mask", "token_type_ids") if c in enc.column_names]
    dl = torch.utils.data.DataLoader(enc.with_format("torch", columns=cols + ["label"]), batch_size=batch)
    n_hit = n_all = 0
    model.eval()
    with torch.no_grad():
        for b in dl:
            logits = model(**{k: v.to(device) for k, v in b.items() if k != "label"}).logits  # (B,2)
            n_hit += (logits.argmax(-1) == b["label"].to(device)).sum().item()
            n_all += b["label"].numel()
    model.train()
    return 100.0 * n_hit / n_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="tiny")
    ap.add_argument("--task", choices=["sst2", "mrpc"], default="sst2")
    ap.add_argument("--epochs", type=int, default=3, help="论文 §4.1 GLUE 口径=3")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5, help="论文 §4.1 搜索集 {5e-5..2e-5} 中 SST-2/MRPC 常用值")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--run-tag", default="", help="输出文件名后缀（防同种子多次运行互相覆盖；留空=沿用基准文件名，make_figs.py 读基准名）")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    ds, keys = load_task(args.task)
    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    enc_train = tokenize(ds["train"], tok, keys, args.max_len)
    enc_val = tokenize(ds["validation"], tok, keys, args.max_len)
    # 多数类基线（训练/验证各算一份）：微调「及格线」= 显著超过它
    maj_train = 100.0 * max(ds["train"]["label"].count(l) for l in set(ds["train"]["label"])) / len(ds["train"])
    maj_val = 100.0 * max(ds["validation"]["label"].count(l) for l in set(ds["validation"]["label"])) / len(ds["validation"])

    model = AutoModelForSequenceClassification.from_pretrained(MODELS[args.model], num_labels=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_head = model.classifier.weight.numel() + model.classifier.bias.numel()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    cols = [c for c in ("input_ids", "attention_mask", "token_type_ids") if c in enc_train.column_names]
    g = torch.Generator().manual_seed(SEED)
    dl = torch.utils.data.DataLoader(enc_train.with_format("torch", columns=cols + ["label"]),
                                     batch_size=args.batch, shuffle=True, generator=g)
    it, losses, accs = iter(dl), [], []
    mem = MpsMemSampler() if device == "mps" else None
    if mem:
        mem.start()
    model.train()
    t0, n_steps = None, 0
    for epoch in range(args.epochs):
        for _ in range(len(dl)):
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl); b = next(it)
            out = model(**{k: v.to(device) for k, v in b.items() if k != "label"}, labels=b["label"].to(device))
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            opt.step()
            losses.append(round(out.loss.item(), 4))
            n_steps += 1
            if n_steps == 3:  # 前 3 步为编译/懒分配预热，不计入计时
                torch.mps.synchronize() if device == "mps" else None
                t0 = time.perf_counter()
        accs.append(round(evaluate(model, enc_val, device), 1))
        print(f"[{args.model}/{args.task}] epoch {epoch + 1}/{args.epochs}: val acc {accs[-1]}%", flush=True)
    if device == "mps":
        torch.mps.synchronize()
    sec_total = time.perf_counter() - t0 if t0 else 0.0
    mem_peak = mem.stop() if mem else None

    report = {
        "meta": {"date": "2026-10-03", "seed": SEED, "device": device, "model": args.model,
                 "model_id": MODELS[args.model], "task": args.task, "epochs": args.epochs,
                 "batch": args.batch, "lr": args.lr, "max_len": args.max_len,
                 "n_train": len(ds["train"]), "n_val": len(ds["validation"])},
        "params": {"total": n_params, "classifier_head_new": n_head},
        "majority_baseline": {"train_pct": round(maj_train, 1), "val_pct": round(maj_val, 1)},
        "train": {"steps": n_steps, "sec_per_step": round(sec_total / max(n_steps - 3, 1), 4),
                  "loss_first10": round(sum(losses[:10]) / 10, 4), "loss_last10": round(sum(losses[-10:]) / 10, 4),
                  "mem_peak_mib": mem_peak, "loss_curve": losses},
        "val_acc_per_epoch": accs,
    }
    suffix = f"_{args.run_tag}" if args.run_tag else ""
    out = os.path.join(OUT_DIR, f"finetune_{args.model}_{args.task}{suffix}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k != "train"} | {"loss_curve_tail": losses[-3:]},
                     ensure_ascii=False), "\nsaved ->", out)


if __name__ == "__main__":
    main()
