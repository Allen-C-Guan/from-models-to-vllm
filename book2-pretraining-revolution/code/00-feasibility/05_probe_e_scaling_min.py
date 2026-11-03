# 05_probe_e_scaling_min.py —— 探针(e)：1M-20M scaling 族实验的最小管道验证（只跑 1 个点）
# 固定 8k 自训 BPE 词表（复用探针(c)的 sentencepiece 模型）+ 拼块数据管道；1M 级模型训 1000 步，
# 验证 loss 曲线可拟合、eval 协议噪声可测（per-token std -> 达到 se<0.005 所需 eval tokens）。
# 产出供 ch8/ch11 的双实验设计书（Kaplan 单曲线 + iso-FLOP 网格）取实测底数。
# 用法：python 05_probe_e_scaling_min.py [--steps 1000]；产物落 log/book2-feasibility/probe_e_scaling_min.json

import argparse
import json
import os
import time

import sentencepiece as spm
import torch

from importlib.util import spec_from_file_location, module_from_spec

_here = os.path.dirname(os.path.abspath(__file__))
_spec = spec_from_file_location("probe_a", os.path.join(_here, "01_probe_a_gpt2_124m.py"))
probe_a = module_from_spec(_spec)
_spec.loader.exec_module(probe_a)

SEED = 20261002
OUT_DIR = os.path.join("log", "book2-feasibility")
SP_MODEL = os.path.join(OUT_DIR, "c_sp_8k_cov1.model")  # 探针(c)产物：8k BPE，coverage=1.0
DATA_PATH = os.path.join(OUT_DIR, "data", "tiny_shakespeare.txt")


def encode_corpus():
    """语料 -> token 流：自训 8k BPE 编码（分词器训练语料与 LM 语料在此冒烟档未分离——
    正式族实验须分离，见设计书）。"""
    sp = spm.SentencePieceProcessor(model_file=SP_MODEL)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    return torch.tensor(sp.encode(text, out_type=int), dtype=torch.long), sp.get_piece_size()


@torch.no_grad()
def eval_chunks(model, heldout, block, batch, device, n_chunks=4, seed=SEED):
    """held-out 均分为 n_chunks 个互斥块，逐块算 eval loss：块间 std = 语料异质性+噪声的上界。
    每块内贪婪组批（块可能不足一个整批，用 min(batch, 块内序列数)）。"""
    model.eval()
    n_tok = (len(heldout) // block) * block
    stream = heldout[:n_tok - 1]
    chunks = stream.chunk(n_chunks)
    losses = []
    per_tok_sample = None
    for ch in chunks:
        nb = len(ch) // block  # 块内可用序列数
        if nb < 1:
            continue
        tot, cnt = 0.0, 0
        i = 0
        while i < nb:
            b_eff = min(batch, nb - i)
            x = torch.stack([ch[(i + j) * block:(i + j + 1) * block] for j in range(b_eff)]).to(device)
            y = torch.stack([ch[(i + j) * block + 1:(i + j + 1) * block + 1] for j in range(b_eff)]).to(device)
            logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)),
                                                     y.reshape(-1))
            tot += loss.item(); cnt += 1
            if per_tok_sample is None:
                lp = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)),
                                                       y.reshape(-1), reduction="none")
                per_tok_sample = lp.cpu()
            i += b_eff
        if cnt:
            losses.append(tot / cnt)
    import statistics as st
    chunk_std = st.pstdev(losses) if len(losses) > 1 else 0.0
    tok_std = float(per_tok_sample.std()) if per_tok_sample is not None and len(per_tok_sample) > 1 else None
    # 均值标准误 se = tok_std / sqrt(n)；达到 se<0.005 所需 token 数（正式协议的关键底数）
    need_tokens = int((tok_std / 0.005) ** 2) if tok_std else None
    model.train()
    return {"chunk_losses": [round(l, 4) for l in losses],
            "chunk_std": round(chunk_std, 4),
            "per_token_std_first_batch": round(tok_std, 4) if tok_std else None,
            "per_token_n_sampled": len(per_tok_sample) if per_tok_sample is not None else 0,
            "tokens_needed_for_se_lt_0.005": need_tokens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    ids, vocab = encode_corpus()
    n_train = int(len(ids) * 0.9)
    train_ids, heldout = ids[:n_train], ids[n_train:]
    device = "mps"

    # 1M 级最小点：4L/128d/4H（非 embedding ≈1.19M，总量级 2M+ 含 8k 词表 embedding）
    model = probe_a.GPT2(n_layer=4, n_embd=128, n_head=4, vocab=vocab,
                         block_size=256).to(device)
    n_total = model.n_params()
    n_nonemb = model.n_params(non_embedding=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    block, batch = 256, 32  # 8192 tokens/step
    gen = torch.Generator().manual_seed(SEED)

    def get_batch():
        ix = torch.randint(len(train_ids) - block - 1, (batch,), generator=gen)
        x = torch.stack([train_ids[i:i + block] for i in ix]).to(device)
        y = torch.stack([train_ids[i + 1:i + block + 1] for i in ix]).to(device)
        return x, y

    losses = []
    t0 = time.perf_counter()
    for step in range(args.steps):
        x, y = get_batch()
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    torch.mps.synchronize()
    sec_total = time.perf_counter() - t0
    sec_per_step = sec_total / args.steps
    flops_per_step = 6 * n_nonemb * block * batch  # 6ND 口径（N=非 embedding 参数）
    tflops = flops_per_step / sec_per_step / 1e12

    # MA-50 平滑曲线（每 50 步一个点）
    ma = [round(sum(losses[max(0, i - 49):i + 1]) / len(losses[max(0, i - 49):i + 1]), 4)
          for i in range(49, args.steps, 50)]

    # 尾段幂律拟合检查：log(L - L_min) vs log(step) 的斜率与 R²（管道可拟合性指标）
    import math
    tail = ma[-8:]
    lmin = min(ma) - 1e-3
    xs = [math.log(50 * (i + 1) * 2) for i in range(len(tail))]  # 近似步数轴
    ys = [math.log(max(t - lmin, 1e-6)) for t in tail]
    n_ = len(xs)
    mx, my = sum(xs) / n_, sum(ys) / n_
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    ss_res = sum((y - (my + slope * (x - mx))) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    ev = eval_chunks(model, heldout, block, batch, device)

    report = {
        "meta": {"date": "2026-10-02", "seed": SEED, "device": device,
                 "vocab": vocab, "tokenizer": "sentencepiece BPE 8k（探针c产物）",
                 "corpus_tokens_total": len(ids), "train_tokens": n_train,
                 "heldout_tokens": len(ids) - n_train,
                 "epochs_equivalent": round(args.steps * block * batch / n_train, 1),
                 "n_params_total": n_total, "n_params_non_embedding": n_nonemb,
                 "block": block, "batch": batch, "steps": args.steps},
        "sec_per_step": round(sec_per_step, 4),
        "sec_total": round(sec_total, 1),
        "effective_tflops_6ND": round(tflops, 2),
        "flops_per_step": flops_per_step,
        "train_loss_first10": round(sum(losses[:10]) / 10, 3),
        "train_loss_last10": round(sum(losses[-10:]) / 10, 3),
        "ma50_curve": ma,
        "tail_fit": {"slope": round(slope, 3), "r2": round(r2, 3)},
        "eval_protocol": ev,
        "note": "冒烟档：分词器与 LM 语料未分离、epoch 数高（语料太小）——正式族实验按设计书改用大语料",
    }
    out_path = os.path.join(OUT_DIR, "probe_e_scaling_min.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "ma50_curve"},
                     ensure_ascii=False, indent=1))
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
