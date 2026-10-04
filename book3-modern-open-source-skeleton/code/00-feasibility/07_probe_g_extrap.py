# 07_probe_g_extrap.py —— 探针 G：外推管道冒烟（ch7 问题现场的复现底座）
# 用途：(1) 训一个 2k 窗 RoPE mini 模型（d=256/L=4/h=4/V=8192，B=8/T=2048/600 步 ≈ 6 分钟，
#           语料 tokens.bin 前 40M 段，Book2 五件套配方）；
#       (2) 直接在 8k 窗算 ppl（eval.bin heldout 段）——验证「RoPE 直接外推会崩」可复现，
#           与 2k 窗（in-window）ppl 对照，产出 ch7 开篇「问题现场」素材；
#       (3) passkey 生成协议冒烟：合成 ~4k token 的 needle 上下文（真实 filler + 中段藏一个 passkey），
#           贪心解码、字符串包含判定，in-window(1.5k) 与 out-window(4k) 双口径准确率。
# 预期管理：600 步的 4 层模型本身很弱，passkey 准确率可能双零——协议跑通即达标（口径可复用即可），
#           如实记录即 Book3 的诚实口径。
# 所属章节：Book3 00-feasibility（ch7 外推实验全案的预算与协议底数）。
# 运行方式：cd code/00-feasibility && python 07_probe_g_extrap.py [--steps 600] [--out-name NAME]
#   产物：log/book3-feasibility/{out-name}.json + {out-name}_train_curve.csv（不 commit）
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from llama_slots import LLaMA, LLaMAConfig   # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book3-feasibility")
TOKENS_BIN = os.path.join(REPO_ROOT, "log", "book2-ch08", "tokens.bin")
EVAL_BIN = os.path.join(REPO_ROOT, "log", "book2-ch08", "eval.bin")            # 1.2M heldout token（流尾部）
SP_MODEL = os.path.join(REPO_ROOT, "log", "book2-ch08", "sp8k_owt.model")     # sp-8k 分词器
TRAIN_POOL_CAP = 40_000_000    # Book2 口径：训练只读流首部 ≤40M token（与 eval 段互斥）
SEED = 20261002


def get_lr(step, total, base=1e-3, warmup=50, min_ratio=0.1):
    """Book2 五件套学习率：线性 warmup + 余弦退火到 0.1x 峰值。"""
    if step < warmup:
        return base * step / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + np.cos(np.pi * p)))


@torch.no_grad()
def eval_ppl(model, eval_ids, n, device, batch=8, max_windows=None, pos_buckets=None):
    """heldout 段不重叠窗一次通过：token 加权平均 loss -> ppl。n 可大于训练窗（外推口径）。

    pos_buckets 给定时同时返回「窗内相对位置分桶」的逐桶 loss——ch7 问题现场的强口径：
    同一个 8k/16k 窗里，前 2k 位置（训练见过）与 2k 之后位置（RoPE 旋转角超界）的 loss 对照。
    """
    model.eval()
    nwin = (len(eval_ids) - 1) // n
    if max_windows:
        nwin = min(nwin, max_windows)
    losses, ntok = 0.0, 0
    bucket_sum = [0.0] * (len(pos_buckets) - 1) if pos_buckets else None
    bucket_cnt = [0] * (len(pos_buckets) - 1) if pos_buckets else None
    for b0 in range(0, nwin, batch):
        xs = [eval_ids[i * n:(i + 1) * n] for i in range(b0, min(b0 + batch, nwin))]
        ys = [eval_ids[i * n + 1:(i + 1) * n + 1] for i in range(b0, min(b0 + batch, nwin))]
        x = torch.from_numpy(np.stack(xs).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack(ys).astype(np.int64)).to(device)
        logits, loss = model(x, y)
        k = x.numel()
        losses += loss.item() * k
        ntok += k
        if pos_buckets:
            pos_loss = torch.nn.functional.cross_entropy(          # (B,n) 逐位置 loss
                logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="none").view(x.shape)
            pos_loss = pos_loss.mean(dim=0).cpu()                   # (n,) 批内平均 -> 单窗逐位置
            nb = x.shape[0]                                         # 本批窗数（尾批可能不满）
            for bi in range(len(bucket_sum)):
                lo, hi = pos_buckets[bi], min(pos_buckets[bi + 1], n)
                if hi <= lo:
                    continue
                bucket_sum[bi] += pos_loss[lo:hi].sum().item() * nb
                bucket_cnt[bi] += (hi - lo) * nb
    model.train()
    avg = losses / ntok
    buckets = None
    if pos_buckets:
        buckets = {f"pos[{pos_buckets[i]}:{pos_buckets[i+1]}]": round(bucket_sum[i] / max(1, bucket_cnt[i]), 4)
                   for i in range(len(bucket_sum))}
    return avg, buckets


@torch.no_grad()
def greedy_decode(model, ids, n_new, device):
    """贪心解码（全量重算 forward，无 KV cache——探针从简，ch7 正文实现时换增量式）。"""
    model.eval()
    ctx = ids.tolist()
    for _ in range(n_new):
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits, _ = model(x)
        ctx.append(int(logits[0, -1].argmax()))
    return ctx[len(ids):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--out-name", default="probe_g_extrap")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(SEED)
    cfg = LLaMAConfig(vocab_size=8192, hidden_size=256, intermediate_size=704,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=2048)            # 训练窗 2k
    model = LLaMA(cfg).to(device)
    print(f"[config] 2k 窗 RoPE mini：参数 {model.n_params():,}（d=256/L=4/h=4/V=8192）")

    tokens = np.memmap(TOKENS_BIN, dtype=np.uint16, mode="r")
    eval_ids = np.fromfile(EVAL_BIN, dtype=np.uint16)
    data_rng = torch.Generator().manual_seed(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)

    # ---- (1) 训练 2k 窗（B=8/T=2048；采样限制在前 40M 训练池） ----
    B, T = 8, 2048
    curve_path = os.path.join(OUT_DIR, f"{args.out_name}_train_curve.csv")
    t0 = time.perf_counter()
    train_log = []
    with open(curve_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["step", "loss", "lr"])
        for step in range(args.steps):
            lr = get_lr(step, args.steps)
            for gparam in opt.param_groups:
                gparam["lr"] = lr
            ix = torch.randint(0, TRAIN_POOL_CAP - T - 1, (B,), generator=data_rng)
            x = torch.from_numpy(np.stack(tokens[ix.numpy()[:, None] + np.arange(T)]).astype(np.int64)).to(device)
            y = torch.from_numpy(np.stack(tokens[ix.numpy()[:, None] + 1 + np.arange(T)]).astype(np.int64)).to(device)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 100 == 0 or step == args.steps - 1:
                writer.writerow([step, round(loss.item(), 4), round(lr, 6)])
                train_log.append({"step": step, "loss": round(loss.item(), 4)})
                print(f"[train] step {step:4d}  loss {loss.item():.4f}  lr {lr:.2e}", flush=True)
    train_sec = time.perf_counter() - t0
    print(f"[train] {args.steps} 步完成，{train_sec:.0f}s（{train_sec/args.steps:.3f} s/step）")

    # ---- (2) 外推崩塌复现：2k 窗 ppl vs 8k/16k 窗 ppl + 位置分桶（同 heldout 段） ----
    loss_2k, _ = eval_ppl(model, eval_ids, 2048, device)
    loss_8k, b8 = eval_ppl(model, eval_ids, 8192, device, batch=4,
                           pos_buckets=[0, 512, 1024, 2048, 4096, 8192])
    loss_16k, b16 = eval_ppl(model, eval_ids, 16384, device, batch=2, max_windows=32,
                             pos_buckets=[0, 1024, 2048, 4096, 8192, 16384])
    print(f"[eval] in-window(2k)   loss = {loss_2k:.4f}  ppl = {np.exp(loss_2k):.2f}")
    print(f"[eval] out-window(8k)  loss = {loss_8k:.4f}  ppl = {np.exp(loss_8k):.2f}   <- 直接外推的代价")
    print(f"[eval] out-window(16k) loss = {loss_16k:.4f}  ppl = {np.exp(loss_16k):.2f}")
    print(f"[分桶] 8k 窗内逐段 loss = {b8}")
    print(f"[分桶] 16k 窗内逐段 loss = {b16}")

    # ---- (3) passkey 生成协议冒烟（合成 needle 上下文；sp-8k 分词器编码 needle） ----
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(SP_MODEL)
    passkey_rec = {"protocol": "filler + needle('The passkey is XXXXX. Remember it.') + filler + query('The passkey is')",
                   "decode": "贪心 12 token", "judge": "解码文本字符串包含 5 位 passkey", "trials": 8}
    for label, total_len in [("in-window 1.5k", 1500), ("out-window 4k", 4000)]:
        hits, examples = 0, []
        for trial in range(8):
            rng = np.random.default_rng(SEED + trial)
            key = int(rng.integers(10000, 99999))
            needle = f" The passkey is {key}. Remember it. "
            needle_ids = sp.EncodeAsIds(needle)
            nfill = total_len - len(needle_ids) - 8          # 8 token 留给 query 前缀
            start = int(rng.integers(0, len(eval_ids) - nfill - 1))
            fill_a = eval_ids[start:start + nfill // 2]
            fill_b = eval_ids[start + nfill // 2:start + nfill]
            query_ids = sp.EncodeAsIds(" The passkey is")
            ctx = np.concatenate([fill_a, np.array(needle_ids, dtype=np.uint16), fill_b,
                                  np.array(query_ids, dtype=np.uint16)]).astype(np.uint16)
            out = greedy_decode(model, ctx, 12, device)
            text = sp.DecodeIds([int(t) for t in out])
            ok = str(key) in text
            hits += ok
            if trial < 3:
                examples.append({"key": key, "ctx_len": len(ctx) + len(query_ids), "decoded": text[:60]})
        passkey_rec[label] = {"accuracy": hits / 8, "hits": hits}
        print(f"[passkey] {label}: 准确率 {hits}/8")
    passkey_rec["examples"] = examples

    # ---- 保存 checkpoint（后续 ch7 章节实验可免重训直接复用） ----
    ckpt_path = os.path.join(OUT_DIR, "probe_g_mini_2k.pt")
    torch.save({"model": model.state_dict(), "config": {k: getattr(cfg, k) for k in
               ["vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "max_position_embeddings"]},
                "steps": args.steps, "seed": SEED}, ckpt_path)
    print(f"[ckpt] {ckpt_path}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{args.out_name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "device": str(device), "steps": args.steps,
                   "train_sec": round(train_sec, 1), "sec_per_step": round(train_sec / args.steps, 4),
                   "params": model.n_params(), "train_log": train_log,
                   "eval": {"loss_2k": round(loss_2k, 4), "ppl_2k": round(float(np.exp(loss_2k)), 3),
                            "loss_8k": round(loss_8k, 4), "ppl_8k": round(float(np.exp(loss_8k)), 3),
                            "loss_16k": round(loss_16k, 4), "ppl_16k": round(float(np.exp(loss_16k)), 3),
                            "bucket_loss_8k": b8, "bucket_loss_16k": b16},
                   "passkey": passkey_rec}, f, ensure_ascii=False, indent=2)
    print(f"产物已写：{out} / {curve_path}")


if __name__ == "__main__":
    main()
