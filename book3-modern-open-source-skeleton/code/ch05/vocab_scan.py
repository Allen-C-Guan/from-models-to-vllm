# vocab_scan.py —— Book3 ch5 词表扫描：V ∈ {8192,16384,32768,49152} 四臂（分词器训练 + 语料重切 + 短训）
# 用途：同一骨架（模块档 d=512/L=6/h=8，GPT-2 底座 tied embeddings，B=16/T=512）换词表大小四臂短训，
#       产出「三权衡」实测：①嵌入占比（V×d/总参，tied 单份）②压缩率（同文本 bytes/token、tokens/word）
#       ③速度（sec/step）+ 短训质量。跨臂质量比较的预注册口径 = bpb（bits per byte，按各臂压缩率
#       归一）——nats/token 跨词表不可比（V 越大每 token 覆盖字节越多，loss 天然越高）。
# 分词器纪律（沿用 Book2 ch8）：分词器只在独立分片 owt_tokenizer_corpus.txt（6000 篇/29MB，与 LM
#       语料互斥）上训练；sp-BPE / byte_fallback / coverage=1.0，与 sp-8k 出身同配方，只变 vocab_size。
# 各臂语料：v8192 直接复用 log/book2-ch08/{tokens.bin,eval.bin}（Book2 定稿遗产）；其余自切——
#       owt_docs.jsonl 流首部逐篇编码（doc 间插 EOS）直至 ≥ 42M+1.2M token，尾 1.2M 留 eval；
#       四臂同 token 预算=同算力（预注册比较轴），文本起点同为流首部。
# 运行方式：python vocab_scan.py [--steps 1000] [--out-name fast] [--cells v8192,v16384,v32768,v49152]
#           [--eval-every 500] [--workers 8]；本实验 fast=1000 步/臂（卷级大纲 ch5 契约，无 full 档）。
# 产物（log/book3-ch05/，不入库）：vocab_scan_{out}.json + curve_v{V}_{out}.csv + ckpt_v{V}_{out}.pt
#           + sp{V}_owt.model/.vocab（分词器）+ stream_v{V}_{train,eval}.bin（自切流）+ slice_meta_v{V}.json。
import argparse
import csv
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import sentencepiece as spm
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for _base in (os.path.join(HERE, "..", "..", "Book2-预训练革命", "ch10"),                              # 工作区布局
              os.path.join(HERE, "..", "..", "..", "book2-pretraining-revolution", "code", "ch10")):  # 书仓快照布局
    if os.path.isdir(_base):
        sys.path.insert(0, os.path.abspath(_base))  # Book2 ch10 GPT-2 定版底座
from model import GPT2, GPT2Config      # noqa: E402  Book2 ch10 定版底座（tied embeddings，ch5 主线）

# ---------------- 固定口径 ----------------
SEED = 20261002
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book3-ch05")
TOK_TXT = os.path.join(REPO_ROOT, "log", "book2-scaling", "owt_tokenizer_corpus.txt")  # 分词器独立分片
DOCS_JSONL = os.path.join(REPO_ROOT, "log", "book2-scaling", "owt_docs.jsonl")         # LM 主语料
BOOK2_DIR = os.path.join(REPO_ROOT, "log", "book2-ch08")

ARMS = {"v8192": 8192, "v16384": 16384, "v32768": 32768, "v49152": 49152}   # 臂名 -> V
# 49152 = 192×256（256 倍数口径的「GPT-2 粗档」≈50257；papers/03 §2 谱系锚 GPT-2 50257）
TRAIN_TARGET = 42_000_000     # 自切训练池目标（≥5000 步×8192 tok 的余量，本实验 fast 只用前 8.2M）
EVAL_TOKENS = 1_200_000       # 自切 eval 池（对齐 Book2 ch08 eval.bin 体量）
SAMPLE_DOCS = 2000            # 压缩率固定样本（首 2000 篇文本，四臂同文本才可比）

BATCH, BLOCK = 16, 512        # B/T（统一模块档；8192 tok/步）
TPB = BATCH * BLOCK
EVAL_BATCHES = 49
PEAK_LR, MIN_LR_RATIO, WARMUP, WDECL, TAIL = 1e-3, 0.1, 100, 0.1, 300
N_LAYER, N_EMBD, N_HEAD, N_POSITIONS = 6, 512, 8, 1024

_SP = None      # worker 内持有分词器


def _init_worker(model_path):
    global _SP
    _SP = spm.SentencePieceProcessor()
    _SP.Load(model_path)


def _encode_range(job):
    """编码 jsonl 行区间 [start,end) -> uint16 part 文件（doc 间插 EOS，Book2 ch8 同口径）。"""
    start, end, part_path = job
    sp = _SP
    buf, n_tok, n_unk, n_doc = [], 0, 0, 0
    with open(DOCS_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= end:
                break
            line = line.strip()
            if not line:
                continue
            ids = sp.encode(json.loads(line)["text"], out_type=int, add_eos=True)  # EOS 结尾
            buf.extend(ids)
            n_tok += len(ids)
            n_unk += sum(1 for t in ids if t == 0)
            n_doc += 1
            if len(buf) > 4_000_000:                       # 分批落盘控内存
                with open(part_path, "ab") as g:
                    g.write(np.asarray(buf, dtype=np.uint16).tobytes())
                buf = []
    if buf:
        with open(part_path, "ab") as g:
            g.write(np.asarray(buf, dtype=np.uint16).tobytes())
    return part_path, n_tok, n_unk, n_doc


def ensure_tokenizer(V):
    """分词器训练（存在即复用）：sp-BPE / byte_fallback / coverage=1.0——sp-8k 同配方只变 V。"""
    prefix = os.path.join(OUT_DIR, f"sp{V}_owt")
    model_path = prefix + ".model"
    if os.path.exists(model_path):
        return model_path
    print(f"[分词器] 训练 sp-{V}（独立分片 {TOK_TXT}，bpe/byte_fallback/coverage=1.0）…", flush=True)
    t0 = time.perf_counter()
    spm.SentencePieceTrainer.Train(
        input=TOK_TXT, model_prefix=prefix, vocab_size=V,
        model_type="bpe", character_coverage=1.0, byte_fallback=True,
    )
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    assert sp.GetPieceSize() == V, f"词表 {sp.GetPieceSize()} != {V}"
    print(f"[分词器] sp-{V} 完成（{time.perf_counter()-t0:.1f}s）", flush=True)
    return model_path


def estimate_docs_needed(sp_path, target_tokens):
    """用首 200 篇估计 tokens/doc，算覆盖 target（+20% 余量）所需行数。"""
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_path)
    n_tok, n_doc = 0, 0
    with open(DOCS_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 200:
                break
            line = line.strip()
            if not line:
                continue
            n_tok += len(sp.encode(json.loads(line)["text"], out_type=int, add_eos=True))
            n_doc += 1
    per_doc = n_tok / max(1, n_doc)
    return int(target_tokens / per_doc * 1.2) + 1000


def slice_stream(V, sp_path, workers):
    """自切 LM 流：编码流首部直至 ≥ TRAIN_TARGET+EVAL_TOKENS，尾 EVAL_TOKENS 留 eval。"""
    train_bin = os.path.join(OUT_DIR, f"stream_v{V}_train.bin")
    eval_bin = os.path.join(OUT_DIR, f"stream_v{V}_eval.bin")
    meta_path = os.path.join(OUT_DIR, f"slice_meta_v{V}.json")
    if os.path.exists(train_bin) and os.path.exists(eval_bin) and os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

    target = TRAIN_TARGET + EVAL_TOKENS
    parts_dir = os.path.join(OUT_DIR, f"tmp_parts_v{V}")
    if os.path.exists(parts_dir):                     # 清理旧 part（断点重跑口径：全量重算，编码确定性）
        for p in os.listdir(parts_dir):
            os.remove(os.path.join(parts_dir, p))
    os.makedirs(parts_dir, exist_ok=True)
    t0 = time.perf_counter()
    need_from = 0
    need_lines = estimate_docs_needed(sp_path, target)
    chunks, total_tok, total_unk, total_doc = [], 0, 0, 0
    while True:   # 分块推进直至达标（估不准时自动续块）
        jobs = []
        w = workers
        nchunk = max(1, (need_lines - need_from + w - 1) // w)
        for k in range(need_from, need_lines, nchunk):
            jobs.append((k, min(k + nchunk, need_lines), os.path.join(parts_dir, f"part_{k:08d}.bin")))
        with Pool(len(jobs), initializer=_init_worker, initargs=(sp_path,)) as pool:
            results = pool.map(_encode_range, jobs)
        chunks.extend(results)
        total_tok += sum(r[1] for r in results)
        total_unk += sum(r[2] for r in results)
        total_doc += sum(r[3] for r in results)
        print(f"  [切流 v{V}] 行 [0:{need_lines}) 完成，累计 {total_tok:,} tok（{total_doc:,} 篇）",
              flush=True)
        if total_tok >= target:
            break
        need_from, need_lines = need_lines, need_lines + estimate_docs_needed(sp_path,
                                                                              target - total_tok)
    # 按行序拼接（与 Book2 ch8 的 order-preserving 拼接同法）
    stream = np.empty(total_tok, dtype=np.uint16)
    pos = 0
    for part_path, n_tok, _, _ in chunks:
        with open(part_path, "rb") as f:
            arr = np.frombuffer(f.read(), dtype=np.uint16)
        stream[pos:pos + n_tok] = arr
        pos += n_tok
    assert pos == total_tok
    eval_arr = stream[-EVAL_TOKENS:].copy()
    train_arr = stream[:-EVAL_TOKENS].copy()
    train_arr.tofile(train_bin)
    eval_arr.tofile(eval_bin)
    for part_path, _, _, _ in chunks:
        os.remove(part_path)
    os.rmdir(parts_dir)
    # ---- 切片参数校验（任务书「参数带校验」）----
    assert int(train_arr.max()) < V and int(eval_arr.max()) < V, "切片校验失败：id 越界词表"
    assert len(train_arr) >= TRAIN_TARGET, f"切片校验失败：训练池 {len(train_arr):,} < {TRAIN_TARGET:,}"
    assert len(eval_arr) == EVAL_TOKENS
    meta = {"V": V, "tokenizer": sp_path, "docs_encoded": total_doc, "unk_tokens": total_unk,
            "unk_rate": total_unk / total_tok, "train_tokens": int(len(train_arr)),
            "eval_tokens": int(len(eval_arr)), "train_bin": train_bin, "eval_bin": eval_bin,
            "protocol": ("owt_docs.jsonl 流首部逐篇编码（EOS 分隔），尾 1.2M 留 eval；"
                         "四臂同 token 预算=同算力，文本起点同为流首部"),
            "wallclock_sec": round(time.perf_counter() - t0, 1)}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  [切流 v{V}] train {len(train_arr):,} tok / eval {len(eval_arr):,} tok / "
          f"unk {total_unk}（{meta['wallclock_sec']}s）", flush=True)
    return meta


def measure_compression(sp_path):
    """固定样本压缩率（首 SAMPLE_DOCS 篇，四臂同文本）：bytes/token 与 tokens/word。"""
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_path)
    n_tok, n_bytes, n_words = 0, 0, 0
    with open(DOCS_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= SAMPLE_DOCS:
                break
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)["text"]
            n_tok += len(sp.encode(text, out_type=int, add_eos=True))
            n_bytes += len(text.encode("utf-8"))
            n_words += len(text.split())
    return {"sample_docs": SAMPLE_DOCS, "sample_bytes": n_bytes, "sample_tokens": n_tok,
            "bytes_per_token": n_bytes / n_tok, "tokens_per_word": n_tok / max(1, n_words)}


def lr_at(step, total):
    if step <= WARMUP:
        return PEAK_LR * step / WARMUP
    prog = (step - WARMUP) / max(1, total - WARMUP)
    return PEAK_LR * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * prog)))


class TrainStream:
    """tokens 流首部按步号顺序切块（一次通过；四臂按步号取同 token 数的批）。"""

    def __init__(self, path):
        self.arr = np.memmap(path, dtype=np.uint16, mode="r")
        self.max_step = (len(self.arr) - 1) // TPB

    def batch(self, step, device):
        off = (step - 1) * TPB
        span = np.asarray(self.arr[off:off + TPB + 1], dtype=np.int64)       # (8193,)
        x = torch.from_numpy(span[:-1].reshape(BATCH, BLOCK)).to(device)     # (16,512)
        y = torch.from_numpy(span[1:].reshape(BATCH, BLOCK)).to(device)      # (16,512)
        return x, y


def build_eval_batches(eval_bin, device):
    """固定 eval 窗集：各臂自己的 eval 池内固定种子抽 49 批 (16,512)（窗位协议跨臂一致）。"""
    ev = np.memmap(eval_bin, dtype=np.uint16, mode="r")
    gen = torch.Generator().manual_seed(SEED)
    offs = torch.randint(0, len(ev) - BLOCK - 1, (EVAL_BATCHES * BATCH,), generator=gen).numpy()
    xs, ys = [], []
    for o in offs:
        xs.append(np.asarray(ev[o:o + BLOCK], dtype=np.int64))
        ys.append(np.asarray(ev[o + 1:o + BLOCK + 1], dtype=np.int64))
    x = torch.from_numpy(np.stack(xs).reshape(EVAL_BATCHES, BATCH, BLOCK)).to(device)  # (49,16,512)
    y = torch.from_numpy(np.stack(ys).reshape(EVAL_BATCHES, BATCH, BLOCK)).to(device)
    return x, y


@torch.no_grad()
def eval_loss(model, ex, ey):
    model.eval()
    losses = []
    for i in range(EVAL_BATCHES):
        _, loss = model(ex[i], ey[i])
        losses.append(loss.item())
    model.train()
    mean = sum(losses) / len(losses)
    se = (sum((l - mean) ** 2 for l in losses) / (len(losses) - 1)) ** 0.5 / math.sqrt(len(losses))
    return mean, se


def build_arm(V):
    """四臂模型：同种子建底座，只变 vocab_size（wte 行数；tied head 同表）。"""
    torch.manual_seed(SEED)
    cfg = GPT2Config(vocab_size=V, n_positions=N_POSITIONS,
                     n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD)
    m = GPT2(cfg)
    hand = 23_633_920 + (V - 8192) * N_EMBD      # 模块档基线（ch2 定账）+ 词表行差量（tied 单份）
    n_total = m.n_params()
    assert n_total == hand, f"参数对账失败：{n_total} != {hand}（V={V}）"
    return m


def run_arm(arm, V, steps, eval_every, device, stream, ex, ey, report, out_json):
    ckpt_path = os.path.join(OUT_DIR, f"ckpt_{arm}_{report['meta']['out_name']}.pt")
    m = build_arm(V).to(device)
    n_total = m.n_params()
    n_nonemb = n_total - m.wte.weight.numel() - m.wpe.weight.numel()
    opt = torch.optim.AdamW(m.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=WDECL)

    train_losses, lrs, times, evals = [], [], [], []
    start_step, elapsed_train, elapsed_eval = 0, 0.0, 0.0
    resumed_from = None
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        m.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        train_losses, lrs, times = ck["train_losses"], ck["lrs"], ck["times"]
        evals = ck["evals"]
        start_step, elapsed_train, elapsed_eval = ck["step"], ck["elapsed_train"], ck["elapsed_eval"]
        resumed_from = start_step
        print(f"  [{arm} 续跑] 自 step {start_step + 1}", flush=True)

    m.train()
    base_alloc = torch.mps.current_allocated_memory() / 2**20 if device.type == "mps" else 0.0
    peak = base_alloc
    diverged_at, reason = None, None
    t_arm = time.perf_counter()
    for step in range(start_step + 1, steps + 1):
        lr = lr_at(step, steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = stream.batch(step, device)
        t0 = time.perf_counter()
        with torch.autocast(device_type="mps", dtype=torch.bfloat16) if device.type == "mps" \
                else torch.autocast("cpu", enabled=False):
            _, loss = m(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if device.type == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
        elapsed_train += times[-1]
        lv = loss.item()
        train_losses.append(lv)
        lrs.append(lr)
        if device.type == "mps":
            peak = max(peak, torch.mps.current_allocated_memory() / 2**20)
        if not math.isfinite(lv) or lv > 100.0:
            diverged_at, reason = step, ("non-finite" if not math.isfinite(lv) else f"loss={lv:.1f}>100")
            print(f"  [{arm} 发散] step {step}: {reason}", flush=True)
            break
        if step % 100 == 0 or step == 1:
            print(f"  [{arm:7s}] step {step:5d}/{steps} train {lv:.4f} lr {lr:.2e} "
                  f"({elapsed_train/(step - start_step + 1):.3f}s/step)", flush=True)
        if step % eval_every == 0 or step == steps:
            t0 = time.perf_counter()
            el, se = eval_loss(m, ex, ey)
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed_eval += time.perf_counter() - t0
            evals.append({"step": step, "eval_loss": round(el, 5), "eval_se": round(se, 5),
                          "n_tokens": EVAL_BATCHES * TPB})
            print(f"  [{arm:7s}] [eval] step {step}: eval loss {el:.4f} ± {se:.4f}", flush=True)
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": step,
                        "train_losses": train_losses, "lrs": lrs, "times": times, "evals": evals,
                        "elapsed_train": elapsed_train, "elapsed_eval": elapsed_eval}, ckpt_path)

    steps_done = len(train_losses)
    steady = times[20:] if len(times) > 20 else times
    tail = train_losses[-min(TAIL, steps_done):]
    emb_params = V * N_EMBD                     # tied：单份
    rec = {
        "arm": arm, "V": V,
        "arm_def": f"GPT-2 底座（模块档，tied embeddings）只变 vocab_size={V}",
        "config": {"d": N_EMBD, "L": N_LAYER, "h": N_HEAD, "V": V, "B": BATCH, "T": BLOCK,
                   "n_positions": N_POSITIONS, "tied": True},
        "params_total": n_total, "params_non_embedding": n_nonemb,
        "embedding_params": emb_params,
        "embedding_share": round(emb_params / n_total, 4),
        "steps_done": steps_done, "steps_planned": steps, "resumed_from_step": resumed_from,
        "tail300_train_loss": round(sum(tail) / len(tail), 5),
        "last_step_loss": round(train_losses[-1], 5) if train_losses else None,
        "min_train_loss": round(min(train_losses), 5) if train_losses else None,
        "evals": evals,
        "eval_final": evals[-1]["eval_loss"] if evals else None,
        "eval_final_se": evals[-1]["eval_se"] if evals else None,
        "sec_per_step_steady": round(sum(steady) / len(steady), 4),
        "sec_step_std": round(float(np.std(steady)), 4),
        "mem_peak_delta_mb": round(peak - base_alloc, 1),
        "wallclock_sec": round(time.perf_counter() - t_arm, 1),
        "diverged": diverged_at is not None, "diverged_at_step": diverged_at, "divergence_reason": reason,
        "precision": "train bf16 autocast / eval fp32",
    }
    curve_path = os.path.join(OUT_DIR, f"curve_{arm}_{report['meta']['out_name']}.csv")
    with open(curve_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["step", "train_loss", "lr"])
        for i, (lv, lr) in enumerate(zip(train_losses, lrs), 1):
            w.writerow([i, f"{lv:.6g}", f"{lr:.6g}"])
    if steps_done >= steps and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    report["arms"][arm] = rec
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  [{arm} 完成] steps {steps_done} | tail300 {rec['tail300_train_loss']} | "
          f"eval_final {rec['eval_final']} | {rec['sec_per_step_steady']}s/step", flush=True)
    del m, opt
    if device.type == "mps":
        torch.mps.empty_cache()
    return rec


def add_tradeoff_summary(report):
    """三权衡汇总：嵌入占比 / 压缩率 / 速度 + bpb 质量排序（预注册跨臂口径）。"""
    rows = []
    for arm, r in report["arms"].items():
        comp = report["meta"]["compression"][arm]
        if r["eval_final"] is None:
            continue
        bpb = r["eval_final"] / (math.log(2) * comp["bytes_per_token"])   # nats/tok -> bits/byte
        rows.append({"arm": arm, "V": r["V"], "embedding_share": r["embedding_share"],
                     "bytes_per_token": round(comp["bytes_per_token"], 4),
                     "tokens_per_word": round(comp["tokens_per_word"], 4),
                     "eval_nats_per_token": r["eval_final"],
                     "eval_bpb": round(bpb, 5), "sec_per_step": r["sec_per_step_steady"]})
    report["tradeoff_table"] = rows
    report["tradeoff_note"] = ("跨臂质量唯一可比口径 = eval_bpb（按各臂同文本 bytes/token 归一）；"
                               "eval_nats_per_token 跨词表不可比；四臂同 token 预算=同算力（预注册比较轴）")


def main():
    ap = argparse.ArgumentParser(description="ch5 词表扫描：V=8k/16k/32k/49k 四臂（三权衡实测）")
    ap.add_argument("--steps", type=int, default=1000, help="每臂步数（本实验 fast=1000，无 full 档）")
    ap.add_argument("--out-name", type=str, default=None)
    ap.add_argument("--cells", type=str, default="v8192,v16384,v32768,v49152")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8, help="语料编码并行 worker 数")
    args = ap.parse_args()
    out_name = args.out_name or ("fast" if args.steps <= 1500 else "full")
    arms = [c.strip() for c in args.cells.split(",")]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"未知臂 {a}（可选 {'/'.join(ARMS)}）")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- 阶段 1：分词器 + 自切流 + 压缩率（存在即复用）----
    tokenizers, slices, compression = {}, {}, {}
    for arm in arms:
        V = ARMS[arm]
        if arm == "v8192":
            tokenizers[arm] = os.path.join(BOOK2_DIR, "sp8k_owt.model")     # Book2 遗产直接复用
            slices[arm] = {"train_bin": os.path.join(BOOK2_DIR, "tokens.bin"),
                           "eval_bin": os.path.join(BOOK2_DIR, "eval.bin"),
                           "train_tokens": 306_117_199, "eval_tokens": 1_200_000,
                           "protocol": "Book2 ch08 定稿遗产直接复用（同切片协议出身）"}
        else:
            tokenizers[arm] = ensure_tokenizer(V)
            slices[arm] = slice_stream(V, tokenizers[arm], args.workers)
        compression[arm] = measure_compression(tokenizers[arm])
        print(f"[压缩率] {arm}: {compression[arm]['bytes_per_token']:.3f} B/tok, "
              f"{compression[arm]['tokens_per_word']:.3f} tok/word（首 {SAMPLE_DOCS} 篇同文本）",
              flush=True)

    out_json = os.path.join(OUT_DIR, f"vocab_scan_{out_name}.json")
    if os.path.exists(out_json):
        with open(out_json, encoding="utf-8") as f:
            report = json.load(f)
        report.setdefault("arms", {})
        print(f"[续跑] 已有报告载入：臂 {sorted(report['arms'])}", flush=True)
    else:
        report = {"arms": {}}
    report["meta"] = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "device": str(device),
        "torch": torch.__version__, "out_name": out_name,
        "experiment": "ch5 词表扫描：同骨架（tied GPT-2 模块档）换词表大小四臂——三权衡实测",
        "module_tier": f"d={N_EMBD}/L={N_LAYER}/h={N_HEAD}，B={BATCH}/T={BLOCK}（8192 tok/步）",
        "vocab_arms": {a: ARMS[a] for a in arms},
        "vocab_note": ("49152=192×256（256 倍数口径的 GPT-2 粗档 ≈50257，papers/03 §2 谱系锚）；"
                       "分词器全部在独立分片训练（bpe/byte_fallback/coverage=1.0，sp-8k 同配方）"),
        "corpus": ("v8192=Book2 ch08 遗产；其余=owt_docs.jsonl 流首部自切（尾 1.2M eval，"
                   f"训练池目标 {TRAIN_TARGET:,} tok 留 5000 步余量）"),
        "slices": slices, "compression": compression,
        "optimizer": f"AdamW(betas=0.9,0.95,wd={WDECL}) + grad_clip 1.0（Book2 五件套）",
        "schedule": f"lr {PEAK_LR} 余弦降至 {PEAK_LR*MIN_LR_RATIO:.0e}，warmup={WARMUP}；四臂统一日程",
        "data_protocol": "各臂自己的流首部顺序分块一次通过（同 token 预算=同算力，预注册比较轴）",
        "eval_protocol": (f"每 {args.eval_every} 步+终步：各臂 eval 池固定 49 窗 (16,512) fp32；"
                          "跨臂质量比较口径 = bpb（见 tradeoff_note）"),
        "precision": "train bf16 autocast / eval fp32",
        "ckpt": "每 eval 边界存 ckpt，臂完成即删；中断后重跑同命令自动续跑",
        "argv": " ".join(sys.argv[1:]),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    t_all = time.perf_counter()
    for arm in arms:
        done = report["arms"].get(arm)
        if done and done["steps_done"] >= args.steps:
            print(f"[跳过] 臂 {arm} 已完成（{done['steps_done']} 步）", flush=True)
            continue
        stream = TrainStream(slices[arm]["train_bin"])
        if args.steps > stream.max_step:
            raise SystemExit(f"{arm} 步数 {args.steps} 超出一次通过上限 {stream.max_step}")
        ex, ey = build_eval_batches(slices[arm]["eval_bin"], device)
        print(f"\n===== 臂 {arm}（V={ARMS[arm]}，{args.steps} 步） =====", flush=True)
        run_arm(arm, ARMS[arm], args.steps, args.eval_every, device, stream, ex, ey, report, out_json)
        del stream, ex, ey
    add_tradeoff_summary(report)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[全部完成] 总耗时 {(time.perf_counter()-t_all)/60:.1f} min -> {out_json}", flush=True)
    for row in report.get("tradeoff_table", []):
        print(f"  {row['arm']:7s} V={row['V']:5d} 嵌入占比={row['embedding_share']*100:5.2f}%  "
              f"{row['bytes_per_token']:.3f}B/tok  bpb={row['eval_bpb']:.4f}  "
              f"{row['sec_per_step']}s/step", flush=True)


if __name__ == "__main__":
    main()
