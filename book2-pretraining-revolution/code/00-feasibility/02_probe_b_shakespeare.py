# 02_probe_b_shakespeare.py —— 探针(b)：tiny shakespeare 字符级 ~10M 模型短训可行性
# 自写 GPT（非 nanoGPT 复制）：6L/384d/6H ≈ 10.65M 参数，char-level vocab≈65，MPS 上短训并记录
# sec/step 与 train/val loss 曲线；对照社区基线 val loss≈1.48（nanoGPT shakespeare_char 口径）。
# 用法：python 02_probe_b_shakespeare.py [--steps 1000]；产物落 log/book2-feasibility/probe_b_shakespeare.json

import argparse
import json
import os
import time

import torch

from importlib.util import spec_from_file_location, module_from_spec

# 复用探针(a)的手写 GPT 实现（独立运行前提：同目录下存在 01_probe_a_gpt2_124m.py）
_here = os.path.dirname(os.path.abspath(__file__))
_spec = spec_from_file_location("probe_a", os.path.join(_here, "01_probe_a_gpt2_124m.py"))
probe_a = module_from_spec(_spec)
_spec.loader.exec_module(probe_a)

SEED = 20261002
DATA_PATH = os.path.join("log", "book2-feasibility", "data", "tiny_shakespeare.txt")
OUT_DIR = os.path.join("log", "book2-feasibility")


def load_data():
    """读入全文 -> 字符级 vocab -> 张量化的 id 流；90/10 切 train/val。"""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    vocab = len(chars)  # 预期 ≈65
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(len(ids) * 0.9)
    return ids[:n], ids[n:], vocab, chars


def get_batch(split_ids, block, batch, device):
    """随机采样一个 (B,T) 训练批：x 与 y 错位一格。"""
    ix = torch.randint(len(split_ids) - block - 1, (batch,))
    x = torch.stack([split_ids[i:i + block] for i in ix]).to(device)
    y = torch.stack([split_ids[i + 1:i + block + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def eval_loss(model, val_ids, block, batch, device, n_batches=20):
    """val loss：固定种子采 20 批取均值（口径稳定，可复算）。"""
    model.eval()
    losses = []
    gen = torch.Generator().manual_seed(SEED)
    for _ in range(n_batches):
        ix = torch.randint(len(val_ids) - block - 1, (batch,), generator=gen)
        x = torch.stack([val_ids[i:i + block] for i in ix]).to(device)
        y = torch.stack([val_ids[i + 1:i + block + 1] for i in ix]).to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    torch.manual_seed(SEED)
    train_ids, val_ids, vocab, chars = load_data()
    device = "mps"

    # shakespeare_char 配置：6 层/384 维/6 头 -> ~10.65M 参数（nanoGPT 社区口径，便于基线可比）
    model = probe_a.GPT2(n_layer=6, n_embd=384, n_head=6, vocab=vocab,
                         block_size=256).to(device)
    n_params = model.n_params()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95),
                            weight_decay=0.1)

    block, batch = 256, 64
    curve = []
    t0 = time.perf_counter()
    val0 = eval_loss(model, val_ids, block, batch, device)
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_ids, block, batch, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # nanoGPT 式梯度裁剪（保持与社区基线同口径）
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            vl = eval_loss(model, val_ids, block, batch, device)
            curve.append({"step": step, "train_loss": round(loss.item(), 4),
                          "val_loss": round(vl, 4)})
            print(f"step {step:5d} train {loss.item():.4f} val {vl:.4f}", flush=True)
    torch.mps.synchronize()
    sec_total = time.perf_counter() - t0
    sec_per_step = sec_total / args.steps

    report = {
        "meta": {"date": "2026-10-02", "seed": SEED, "device": device,
                 "corpus_bytes": os.path.getsize(DATA_PATH),
                 "vocab": vocab, "n_params_total": n_params,
                 "n_params_non_embedding": model.n_params(non_embedding=True),
                 "block": block, "batch": batch, "steps": args.steps,
                 "lr": 1e-3, "optimizer": "AdamW(0.9,0.95,wd=0.1)+grad_clip1.0",
                 "val_eval_protocol": "固定种子 20 批 x (64,256) 取均值"},
        "sec_per_step": round(sec_per_step, 4),
        "sec_total": round(sec_total, 1),
        "val_loss_init": round(val0, 4),
        "curve": curve,
        "community_baseline_val_loss": 1.48,
        "baseline_note": "nanoGPT shakespeare_char 收敛口径 val loss≈1.48（数万步/3e-4 cosine）",
    }
    out_path = os.path.join(OUT_DIR, "probe_b_shakespeare.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("sec/step:", sec_per_step, "-> saved", out_path)


if __name__ == "__main__":
    main()
