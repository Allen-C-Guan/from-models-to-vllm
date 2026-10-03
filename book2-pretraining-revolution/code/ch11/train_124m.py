# train_124m.py —— Book2 ch11 实验二：GPT-2 124M 官方口径 fast 训练（50M token，MPS bf16）
# 所属章节：Book2 第 11 章 §11.3（ch11-大项目II.md 124M 训练与图 11.4 数据源）
# 设计书：notes/01-实验可行性.md §1/§4/§7（124M 诚实条款=compute-limited 演示口径）。
#   * 词表口径：与官方 gpt2 逐位一致 → 50257（byte-level BPE，tiktoken gpt2 编码器——ch2/bpe.py 已对拍
#     tiktoken≡HF GPT2Tokenizer id 全等）。sp-8k 流不能直接喂 50257 模型，故用同一 OWT 文档流
#     （log/book2-scaling/owt_docs.jsonl，ch08/data_prep.py 同源）重新编码为 50257 词表流；
#   * 模型：12L/768d/12H、wte50257、wpe1024（GPT-2 官方 n_positions）、tied head——
#     assert 总参 124,439,808 / 非嵌入 85,056,000（6ND 口径，ch08 族同款 GPT 工厂直接复用）；
#   * 训练：b32×s256（8192 token/步，与 ch08/ch11-iso 同步长；任务原文写 b8×s256 但其「≈6100 步」
#     只在 8192 token/步下成立——两种口径 50M token 总时间相同，取与全族可比的 b32×s256，在案澄清）、
#     6104 步 = 50,003,968 token 一次通过、warmup 200、余弦 1e-3→1e-4、AdamW(0.9,0.95,wd=0.1)+clip1.0、
#     bf16 autocast 训练 / fp32 eval（每 500 步 + 终步；49 固定窗×8192=401,408 token，与 ch08 同协议
#     ——eval 池为 50257 流末尾 600k token 重编码留出，与训练段间隔 ~9.4M token 互斥）；
#   * 产物：log/book2-ch11/train124_fast.json（loss 曲线 + wallclock + 末段均值 + 轨迹幂律双拟合
#     + 与 ch8 族的 per-byte 换算衔接）。诚实条款：50M token 远未收敛（官方口径多 GPU 数周），
#     本实验只要求曲线健康与幂律可拟合，不与官方模型比绝对 PPL。
# 运行方式：python train_124m.py [--prep-only] [--skip-prep] [--analyze-only] [--device mps]
# 产物：log/book2-ch11/{tokens50k.bin, eval50k.bin, data50k_meta.json, train124_fast.json,
#        curve_124m.csv/.npy, ckpt_124m.pt(跑完即删)}

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

# ---------------- 固定口径 ----------------
SEED = 20261002
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book2-ch11")
DOCS_JSONL = os.path.join(REPO_ROOT, "log", "book2-scaling", "owt_docs.jsonl")   # ch08 同源文档流
SP8K_MODEL = os.path.join(REPO_ROOT, "log", "book2-ch08", "sp8k_owt.model")      # sp8k（统计映射用）

VOCAB = 50257             # GPT-2 官方词表（byte-level BPE，tiktoken gpt2）
EOT = 50256               # <|endoftext|>：文档间分隔（GPT-2 预训练拼接流同口径）
TARGET_TOKENS = 60_000_000   # 重编码目标：60M gpt2 token（≈训练 50M + eval 0.6M + 间隔 ~9.4M）
TRAIN_STEPS = 6104        # 6104×8192 = 50,006,528 token ≈ 50M（fast 档）
EVAL_POOL = 600_000       # eval 池（≥40 万协议留裕量；窗协议与 ch08 同）
BLOCK = 256               # 训练窗长 s（wpe=1024 只用前 256 个位置——fast 档短窗口径，如实声明）
BATCH = 32
TPB = BATCH * BLOCK
EVAL_EVERY = 500
EVAL_BATCHES = 49         # 49×8192 = 401,408 token
PEAK_LR = 1e-3
MIN_LR_RATIO = 0.1
WARMUP = 200
WDECL = 0.1
N_LAYER, N_EMBD, N_HEAD = 12, 768, 12
WPE = 1024                # GPT-2 官方 n_positions
EXPECT_TOTAL = 124_439_808
EXPECT_NONEMB = 85_056_000

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ch04"))
from warmup_ablation import GPT  # noqa: E402  官方同构工厂（ch04 手术②，notes/01 §1 对拍 124,439,808）

# 若未重定向 tiktoken 缓存则落 log/（缓存红线：env.sh 未覆盖 tiktoken，此处兜底）
os.environ.setdefault("TIKTOKEN_CACHE_DIR", os.path.join(REPO_ROOT, "log", "cache", "tiktoken"))
os.makedirs(os.environ["TIKTOKEN_CACHE_DIR"], exist_ok=True)

_ENC = None
_SP = None


def _init_worker():
    global _ENC, _SP
    import tiktoken
    import sentencepiece as spm
    _ENC = tiktoken.get_encoding("gpt2")
    _SP = spm.SentencePieceProcessor()
    _SP.Load(SP8K_MODEL)


def _encode_docs(lines):
    """编码一批文档行 -> (uint16 数组, n_doc, n_tok, n_unk50k 无此概念/0, n_bytes, n_words, n_sp8k_tok)。

    每篇：gpt2 ids + [50256]（EOT 结尾，与 ch08/data_prep 的 EOS 拼接流同型）；
    disallowed_special=() 把文本中字面 <|endoftext|> 当普通文本（Web 文本可能含此字面量，不设会抛异常）。
    同时统计 sp8k token 数（同一批文档）——只为「60M gpt2 token ↔ 多少 sp8k token」的映射统计。
    """
    global _ENC, _SP
    ids = []
    n_doc = n_tok = n_bytes = n_words = n_sp8k = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        text = json.loads(line)["text"]
        enc = _ENC.encode(text, disallowed_special=()) + [EOT]
        ids.extend(enc)
        n_doc += 1
        n_tok += len(enc)
        n_bytes += len(text.encode("utf-8"))
        n_words += len(text.split())
        n_sp8k += len(_SP.encode(text, out_type=int))   # 不加 EOS——映射统计口径即可
    return np.asarray(ids, dtype=np.uint16), n_doc, n_tok, n_bytes, n_words, n_sp8k


# ---------------- 预备：50257 重编码（一次，~分钟级） ----------------
def prep_50k(workers=6, chunk_lines=4000):
    """OWT 文档流（ch08 同源）按原顺序重编码为 50257 词表流，直到 ≥60M token。

    产物：tokens50k.bin（全量累计流，训练只读首 50,006,529 token）、eval50k.bin（末尾 600k，与训练段
    间隔 ~9.4M token 互斥）、data50k_meta.json（含 gpt2↔sp8k↔byte 映射统计）。
    """
    t0 = time.perf_counter()
    import tiktoken
    _ = tiktoken.get_encoding("gpt2")     # 主进程预热（触发缓存落 log/）

    def line_chunks():
        buf = []
        with open(DOCS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                buf.append(line)
                if len(buf) >= chunk_lines:
                    yield buf
                    buf = []
            if buf:
                yield buf

    stream_parts, total = [], 0
    stats = {"docs": 0, "gpt2_tokens": 0, "sp8k_tokens": 0, "bytes": 0, "words": 0}
    with Pool(workers, initializer=_init_worker) as pool:
        for ids, n_doc, n_tok, n_bytes, n_words, n_sp8k in pool.imap(_encode_docs, line_chunks(), chunksize=1):
            stream_parts.append(ids)
            total += n_tok
            stats["docs"] += n_doc; stats["gpt2_tokens"] += n_tok
            stats["sp8k_tokens"] += n_sp8k; stats["bytes"] += n_bytes; stats["words"] += n_words
            if total >= TARGET_TOKENS:
                pool.terminate()
                break
    stream = np.concatenate(stream_parts)[:TARGET_TOKENS]
    assert stream.max() <= EOT, f"id 越界 {stream.max()} > {EOT}"
    train_bin = stream   # 训练只读首 TRAIN_STEPS*TPB+1（下方 assert 保证与 eval 互斥）
    eval_bin = stream[-EVAL_POOL:].copy()
    assert TRAIN_STEPS * TPB + 1 + EVAL_POOL <= len(stream), "训练段与 eval 池间隔不足"
    train_bin.tofile(os.path.join(OUT_DIR, "tokens50k.bin"))
    eval_bin.tofile(os.path.join(OUT_DIR, "eval50k.bin"))

    # 频率谱与熵（首 10M token，与 ch08 data_meta 同口径）
    uniq, counts = np.unique(stream[:10_000_000], return_counts=True)
    p = counts / counts.sum()
    ent = float(-(p * np.log(p)).sum())
    bpt = stats["bytes"] / stats["gpt2_tokens"]
    meta = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED,
        "source": {"path": DOCS_JSONL, "docs_used": stats["docs"],
                   "note": "OpenWebText 子采样（CC0 包装层），与 ch08/data_prep.py 同一文档流同顺序"},
        "tokenizer": {"type": "tiktoken gpt2（byte-level BPE，vocab=50257）", "eot_id": EOT,
                      "verified_by": "ch2/bpe.py：tiktoken≡HF GPT2Tokenizer 受控样例+Multi30K 前 2000 行 id 全等",
                      "doc_separator": "每篇末尾 append EOT(id=50256)；无 BOS"},
        "encoding": {
            "dtype": "uint16", "total_tokens": int(len(stream)),
            "train_pool_cap_tokens": TRAIN_STEPS * TPB,
            "eval_pool_tokens": int(len(eval_bin)),
            "eval_heldout_from": "50257 流末尾 600k（与训练段间隔 ~9.4M token，互斥）",
            "bytes_per_gpt2_token": bpt,
            "sp8k_tokens_same_docs": stats["sp8k_tokens"],
            "gpt2_vs_sp8k_token_ratio": stats["gpt2_tokens"] / stats["sp8k_tokens"],
            "bytes_per_sp8k_token_ref_ch08": 3.82,
            "tokens_per_word": stats["gpt2_tokens"] / max(1, stats["words"]),
            "token_entropy_nats_first10M": ent,
            "vocab_usage_first10M": int((counts > 0).sum()), "vocab_size": VOCAB,
            "id_max": int(stream.max()),
            "eot_count_first10M": int((stream[:10_000_000] == EOT).sum()),
        },
        "wallclock_sec": round(time.perf_counter() - t0, 1),
    }
    with open(os.path.join(OUT_DIR, "data50k_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta["encoding"], ensure_ascii=False, indent=2), flush=True)
    print(f"[prep 完成] {len(stream):,} gpt2 token（{stats['docs']:,} 篇 / {stats['bytes']:,} B）"
          f" -> tokens50k.bin + eval50k.bin，wall {meta['wallclock_sec']}s", flush=True)


# ---------------- 日程/数据/eval（与 ch11/iso_flop.py 同式，vocab 不同） ----------------
def lr_at(step, peak, warmup, total, min_ratio=MIN_LR_RATIO):
    if warmup > 0 and step <= warmup:
        return peak * step / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


class TrainStream:
    def __init__(self, path, cap_steps):
        self.arr = np.memmap(path, dtype=np.uint16, mode="r")
        self.max_step = min((len(self.arr) - 1) // TPB, cap_steps)

    def batch(self, step, device):
        off = (step - 1) * TPB
        span = np.asarray(self.arr[off:off + TPB + 1], dtype=np.int64)
        x = torch.from_numpy(span[:-1].reshape(BATCH, BLOCK)).to(device)
        y = torch.from_numpy(span[1:].reshape(BATCH, BLOCK)).to(device)
        return x, y


def build_eval_batches(device):
    """固定 eval 批集：50257 流 eval 池（末尾 600k）固定种子 49 窗——协议与 ch08 逐位同型（窗位随池不同）。"""
    ev = np.memmap(os.path.join(OUT_DIR, "eval50k.bin"), dtype=np.uint16, mode="r")
    gen = torch.Generator().manual_seed(SEED)
    offs = torch.randint(0, len(ev) - BLOCK - 1, (EVAL_BATCHES * BATCH,), generator=gen).numpy()
    xs, ys = [], []
    for o in offs:
        xs.append(np.asarray(ev[o:o + BLOCK], dtype=np.int64))
        ys.append(np.asarray(ev[o + 1:o + BLOCK + 1], dtype=np.int64))
    x = torch.from_numpy(np.stack(xs).reshape(EVAL_BATCHES, BATCH, BLOCK)).to(device)
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
    return sum(losses) / len(losses)


# ---------------- 训练 ----------------
def train_124m(device):
    ckpt_path = os.path.join(OUT_DIR, "ckpt_124m.pt")
    torch.manual_seed(SEED)
    model = GPT(n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD, vocab=VOCAB, block_size=WPE).to(device)
    n_total = model.n_params()
    n_nonemb = model.n_params(non_embedding=True)
    assert n_total == EXPECT_TOTAL, f"总参 {n_total:,} != {EXPECT_TOTAL:,}"
    assert n_nonemb == EXPECT_NONEMB, f"非嵌入 {n_nonemb:,} != {EXPECT_NONEMB:,}"
    print(f"[模型] 12L×768d×12H wte{VOCAB} wpe{WPE} tied：总参 {n_total:,} / 非嵌入 {n_nonemb:,}"
          f"（与官方 gpt2 逐位一致，notes/01 §1/§3 对拍口径）", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=WDECL)
    ex, ey = build_eval_batches(device)
    stream = TrainStream(os.path.join(OUT_DIR, "tokens50k.bin"), TRAIN_STEPS)
    assert TRAIN_STEPS <= stream.max_step

    train_losses, lrs, evals, peak_mib = [], [], [], 0.0
    start_step, elapsed_train, elapsed_eval = 0, 0.0, 0.0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        train_losses, lrs, evals = ck["train_losses"], ck["lrs"], ck["evals"]
        start_step, elapsed_train, elapsed_eval = ck["step"], ck["elapsed_train"], ck["elapsed_eval"]
        peak_mib = ck.get("peak_mib", 0.0)
        print(f"[续跑] 自 step {start_step + 1}（已完成 {start_step} 步）", flush=True)

    t_point = time.perf_counter()
    for step in range(start_step + 1, TRAIN_STEPS + 1):
        lr = lr_at(step, PEAK_LR, WARMUP, TRAIN_STEPS)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = stream.batch(step, device)
        t0 = time.perf_counter()
        with torch.autocast(device_type="mps", dtype=torch.bfloat16) if device == "mps" else torch.autocast("cpu", enabled=False):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device == "mps":
            torch.mps.synchronize()
        elapsed_train += time.perf_counter() - t0
        lv = loss.item()
        train_losses.append(lv)
        lrs.append(lr)
        if not math.isfinite(lv) or lv > 100.0:
            print(f"  [发散] step {step}: {lv}", flush=True)
            break
        if step % 100 == 0 or step == 1:
            print(f"  step {step:6d}/{TRAIN_STEPS} train {lv:.4f} lr {lr:.2e} "
                  f"({elapsed_train / step:.3f}s/step 累计)", flush=True)
        if step % EVAL_EVERY == 0 or step == TRAIN_STEPS:
            t0 = time.perf_counter()
            el = eval_loss(model, ex, ey)
            if device == "mps":
                torch.mps.synchronize()
                peak_mib = max(peak_mib, torch.mps.current_allocated_memory() / 2 ** 20)
            elapsed_eval += time.perf_counter() - t0
            evals.append({"step": step, "eval_loss": round(el, 5), "n_tokens": EVAL_BATCHES * TPB})
            print(f"  [eval] step {step}: eval loss {el:.4f}（401,408 token，fp32）", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "train_losses": train_losses, "lrs": lrs, "evals": evals,
                        "elapsed_train": elapsed_train, "elapsed_eval": elapsed_eval,
                        "peak_mib": peak_mib}, ckpt_path)

    wallclock = time.perf_counter() - t_point
    steps_done = len(train_losses)
    tail = train_losses[-min(100, steps_done):]
    result = {
        "model": {"n_layer": N_LAYER, "n_embd": N_EMBD, "n_head": N_HEAD, "vocab": VOCAB,
                  "wpe": WPE, "tied": True, "params_total": n_total, "params_non_embedding": n_nonemb},
        "steps": steps_done, "steps_planned": TRAIN_STEPS, "D_tokens": steps_done * TPB,
        "batch_block": f"b{BATCH}×s{BLOCK}（{TPB} token/步）",
        "schedule": f"lr {PEAK_LR} 余弦降至 {PEAK_LR*MIN_LR_RATIO:.0e} + warmup {WARMUP}（ch08/ch11-iso 同日程）",
        "initial_train_loss": round(train_losses[0], 4) if train_losses else None,
        "initial_loss_check": f"ln({VOCAB})={math.log(VOCAB):.4f}（std=0.02 初始化自检：初始 loss 应≈此值+小量）",
        "final_train_loss": round(sum(tail) / len(tail), 5),
        "final_train_loss_def": "最后 100 步 train loss 均值",
        "eval_losses": evals,
        "eval_final": evals[-1]["eval_loss"] if evals else None,
        "eval_tokens_per_eval": EVAL_BATCHES * TPB,
        "wallclock": round(wallclock, 1),
        "wallclock_sec_train_only": round(elapsed_train, 1),
        "wallclock_sec_eval_only": round(elapsed_eval, 1),
        "sec_per_step": round(elapsed_train / max(1, steps_done), 4),
        "tokens_per_sec_train": round(steps_done * TPB / max(1e-9, elapsed_train), 1),
        "effective_tflops_6nd": round(6.0 * n_nonemb * steps_done * TPB / max(1e-9, elapsed_train) / 1e12, 2),
        "peak_mps_mib": round(peak_mib, 1),
        "precision": "train bf16 autocast / eval fp32",
    }

    # 曲线落盘
    stem = os.path.join(OUT_DIR, "curve_124m")
    with open(stem + ".csv", "w", encoding="utf-8") as f:
        f.write("step,train_loss,lr\n")
        for i, (lv, lr) in enumerate(zip(train_losses, lrs), 1):
            f.write(f"{i},{lv:.6g},{lr:.6g}\n")
    np.save(stem + ".npy", np.column_stack([np.arange(1, steps_done + 1), train_losses, lrs]))

    out_json = os.path.join(OUT_DIR, "train124_fast.json")
    report = None
    if os.path.exists(out_json):
        with open(out_json, encoding="utf-8") as f:
            report = json.load(f)
    if report is None:
        report = {"meta": {}, "run": {}, "analysis": {}}
    report["run"] = result
    report["analysis"] = analyze_trajectory(result)
    report["meta"] = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "device": device,
        "torch": torch.__version__,
        "experiment": "GPT-2 124M 官方口径 fast 训练：50257 词表（tiktoken gpt2 重编码 OWT 同源流）+ 50M token 一次通过",
        "honesty_clause": ("compute-limited 演示口径（notes/01 §7）：50M token 远未收敛（官方训练算力未随论文公布，"
                           "常见复述多 GPU 数周量级属二手转述）；本实验只要求曲线健康、幂律可拟合、与 ch8 族按"
                           " per-byte 口径衔接，不与官方模型比绝对 PPL"),
        "batch_note": ("任务原文 b8×s256 与「≈6100 步」不自洽（b8×s256=2048 tok/步 → 50M token 是 24,414 步）；"
                       "取 8192 token/步（b32×s256）与全族可比且总时间相同（吞吐口径 12k tok/s，notes/01 §1）"),
        "data": "tokens50k.bin（data50k_meta.json 详统计）；eval=eval50k.bin 末尾 600k，与训练段互斥",
        "argv": " ".join(sys.argv[1:]),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"\n[完成] steps {steps_done} | final_train {result['final_train_loss']} | "
          f"eval_final {result['eval_final']} | {result['sec_per_step']}s/step | "
          f"wall {wallclock / 60:.1f} min -> {out_json}", flush=True)
    return result


# ---------------- 分析：轨迹幂律双拟合 + ch8 族 per-byte 衔接 ----------------
def _plaw(D, E, B, beta):
    return E + B * np.asarray(D, dtype=float) ** (-beta)


def _r2_on_L(y, yhat):
    y = np.asarray(y, dtype=float)
    return 1.0 - float(np.sum((y - yhat) ** 2)) / float(np.sum((y - y.mean()) ** 2))


def analyze_trajectory(result):
    """对 eval 轨迹（L vs D）做 ch8/fit_scaling.py 同型双拟合：E 自由 vs E=0。

    红线：这是「单条余弦退火轨迹」上的幂律（早期点在高 lr 下、末点在退火末端），与 Kaplan/Chinchilla
    固定 lr 大网格口径不是同一对象——只作曲线特征描述（可拟合性），不当 scaling 指数引用。
    """
    from scipy import optimize
    ev = result["eval_losses"]
    D = np.array([e["step"] * TPB for e in ev], dtype=float)
    L = np.array([e["eval_loss"] for e in ev], dtype=float)
    out = {"fit_domain": "eval 轨迹 L(D)（每 500 步一个点，fp32，401,408 token/点）",
           "red_line": "单条余弦退火轨迹的幂律≠固定 lr 网格的 L(D) scaling 读数——只描述可拟合性，不当指数引用"}

    def fit_free(d, l):
        try:
            p, _ = optimize.curve_fit(_plaw, d, l, p0=[min(l) * 0.9, 20.0, 0.3],
                                      bounds=([0.0, 1e-8, 0.01], [12.0, 1e4, 5.0]), maxfev=200000)
            return p, _r2_on_L(l, _plaw(d, *p))
        except Exception:
            return None, None

    def fit_fixed0(d, l):
        b1, b0 = np.polyfit(np.log(d), np.log(l), 1)
        B, beta = float(np.exp(b0)), float(-b1)
        return np.array([0.0, B, beta]), _r2_on_L(l, _plaw(d, 0.0, B, beta))

    pA, r2A = fit_free(D, L)
    pB, r2B = fit_fixed0(D, L)
    out["fit_free"] = ({"params": {"E": pA[0], "B": pA[1], "beta": pA[2]}, "r2_on_L": r2A}
                       if pA is not None else None)
    out["fit_fixed0"] = {"params": {"E": 0.0, "B": pB[1], "beta": pB[2]}, "r2_on_L": r2B}
    out["local_loglog_slopes"] = (np.diff(np.log(L)) / np.diff(np.log(D))).tolist()

    # bootstrap（case 重采样 500 次；fit_scaling.py 同法）
    rng = np.random.default_rng(SEED)
    beta_A, beta_B, E_A, skip = [], [], [], 0
    for _ in range(500):
        idx = rng.integers(0, len(D), len(D))
        if len(np.unique(D[idx])) < 3:
            skip += 1
            continue
        p, _ = fit_free(D[idx], L[idx])
        if p is None:
            skip += 1
        else:
            beta_A.append(p[2]); E_A.append(p[0])
        p2, _ = fit_fixed0(D[idx], L[idx])
        beta_B.append(p2[2])
    out["bootstrap_CI"] = {
        "beta_free": [float(np.percentile(beta_A, 2.5)), float(np.percentile(beta_A, 97.5))] if beta_A else None,
        "E_free": [float(np.percentile(E_A, 2.5)), float(np.percentile(E_A, 97.5))] if E_A else None,
        "beta_fixed0": [float(np.percentile(beta_B, 2.5)), float(np.percentile(beta_B, 97.5))] if beta_B else None,
        "skipped": skip,
    }

    # 与 ch8 族衔接（跨词表红线：nats/token 不可直接比，须 per-byte）
    dmeta_path = os.path.join(OUT_DIR, "data50k_meta.json")
    if os.path.exists(dmeta_path):
        with open(dmeta_path, encoding="utf-8") as f:
            bpt = json.load(f)["encoding"]["bytes_per_gpt2_token"]
        ef = result["eval_final"]
        out["ch8_bridge"] = {
            "bytes_per_gpt2_token": bpt,
            "bytes_per_sp8k_token_ch08": 3.82,
            "red_line": ("50257 与 sp-8k 词表不同 → nats/token 绝对值不可直接比（ln50257=10.82 vs ln8192=9.01）；"
                         "唯一同尺度口径=per-byte loss：L_tok / (B/token)"),
            "this_run_eval_per_byte": round(ef / bpt, 5),
            "ch08_20m_eval_per_byte_ref": round(4.50131 / 3.82, 5),
            "ch08_20m_eval_ref": 4.50131,
            "note": ("ch8 20m=D=40M sp8k 一次通过；本跑 124M=D=50M gpt2 一次通过。若本跑 per-byte 更低，"
                     "说明「更大模型×更多数据×更细词表」三变量合力——不能拆功给单一变量"),
        }
    return out


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="ch11 GPT-2 124M fast 训练（50257 词表重编码 + 50M token）")
    ap.add_argument("--prep-only", action="store_true", help="只做 50257 重编码（tokens50k.bin 等）")
    ap.add_argument("--skip-prep", action="store_true", help="跳过重编码（文件已存在时）")
    ap.add_argument("--analyze-only", action="store_true", help="只重跑分析（从已有 train124_fast.json+曲线）")
    ap.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.analyze_only:
        with open(os.path.join(OUT_DIR, "train124_fast.json"), encoding="utf-8") as f:
            report = json.load(f)
        report["analysis"] = analyze_trajectory(report["run"])
        with open(os.path.join(OUT_DIR, "train124_fast.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(json.dumps(report["analysis"], ensure_ascii=False, indent=2), flush=True)
        return

    if not args.skip_prep and not os.path.exists(os.path.join(OUT_DIR, "tokens50k.bin")):
        prep_50k()
    else:
        print("[prep] tokens50k.bin 已存在，跳过重编码", flush=True)
    if args.prep_only:
        return
    train_124m(args.device)


if __name__ == "__main__":
    main()
