# train_124m.py —— Book2第11章：GPT-2词表重编码与指定预算训练。
# 默认6104×8192=50,003,968目标token；bf16仅在MPS训练启用，eval为fp32。
# 同一字节流中的位置分离不证明文本去重；完整文档计数不等于截断流或eval窗的分母。
# 工作区激活env.sh后：python code/ch11/train_124m.py --help
# 所有产物在工作目录log；已有数据核哈希，训练/分析独立命名，保留断点供采样。
# current只在eval结束后采样，不称训练峰值；纯训练时长与本次调用时间分别报告。
import argparse
import hashlib
from pathlib import Path
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
REPO_ROOT = os.getcwd()
OUT_DIR = os.path.join(REPO_ROOT, "log", "book2-ch11")
DOCS_JSONL = os.path.join(REPO_ROOT, "log", "book2-scaling", "owt_docs.jsonl")   # ch08 同源文档流
SP8K_MODEL = os.path.join(REPO_ROOT, "log", "book2-ch08", "sp8k_owt.model")      # sp8k（统计映射用）

VOCAB = 50257             # GPT-2 官方词表（byte-level BPE，tiktoken gpt2）
EOT = 50256               # 本项目在文档后添加的EOT；不据推理代码断言原始完整训练管道
TARGET_TOKENS = 60_000_000   # 重编码目标：60M gpt2 token（≈训练 50M + eval 0.6M + 间隔 ~9.4M）
TRAIN_STEPS = 6104        # 6104×8192 = 50,003,968 token ≈ 50M（fast 档）
EVAL_POOL = 600_000       # 600k位置留出池；抽样窗可重叠，401,408被评分位置不等于独立样本数
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


def _init_worker(sp_model=None):
    global _ENC, _SP
    import tiktoken
    import sentencepiece as spm
    _ENC = tiktoken.get_encoding("gpt2")
    _SP = spm.SentencePieceProcessor()
    _SP.Load(sp_model or SP8K_MODEL)
    if _SP.eos_id()<0:raise ValueError("SP模型缺EOS")


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
        n_sp8k += len(_SP.encode(text,out_type=int,add_eos=True))  # 与GPT2一样逐篇计一个边界词元
    return np.asarray(ids, dtype=np.uint16), n_doc, n_tok, n_bytes, n_words, n_sp8k


# ---------------- 预备：50257 重编码（一次，~分钟级） ----------------
def prep_50k(workers=6, chunk_lines=4000):
    """OWT 文档流（ch08 同源）按原顺序重编码为 50257 词表流，直到 ≥60M token。

    产物：tokens50k.bin（全量累计流，训练只读首 50,003,969 token）、eval50k.bin（末尾 600k，与训练段
    间隔 ~9.4M token 互斥）、data50k_meta.json（含 gpt2↔sp8k↔byte 映射统计）。
    """
    os.makedirs(OUT_DIR,exist_ok=True)
    for name in ("tokens50k.bin","eval50k.bin","data50k_meta.json"):
        if Path(OUT_DIR,name).exists():raise FileExistsError("不覆盖已有准备产物："+name)
    if workers<=0 or chunk_lines<=0 or TARGET_TOKENS<=EVAL_POOL:
        raise ValueError("workers/chunk须正，目标流须大于eval池")
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
    with Pool(workers,initializer=_init_worker,initargs=(SP8K_MODEL,)) as pool:
        for ids, n_doc, n_tok, n_bytes, n_words, n_sp8k in pool.imap(_encode_docs, line_chunks(), chunksize=1):
            stream_parts.append(ids)
            total += n_tok
            stats["docs"] += n_doc; stats["gpt2_tokens"] += n_tok
            stats["sp8k_tokens"] += n_sp8k; stats["bytes"] += n_bytes; stats["words"] += n_words
            if total >= TARGET_TOKENS:
                pool.terminate()
                break
    if not stream_parts or total<TARGET_TOKENS:
        raise ValueError("语料不足以生成指定流长度")
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
                   "identity":file_identity(DOCS_JSONL),
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
            "bytes_per_sp8k_token_same_docs":stats["bytes"]/stats["sp8k_tokens"],
            "count_scope":"最后一个完整文档批次的全部文本；不等于截断后60M流或实际eval窗",
            "boundary_protocol":"两编码每篇均计一个EOS/EOT，净文本UTF8字节不含人工边界词元",
            "tokens_per_word": stats["gpt2_tokens"] / max(1, stats["words"]),
            "token_entropy_nats_first10M": ent,
            "vocab_usage_first10M": int((counts > 0).sum()), "vocab_size": VOCAB,
            "id_max": int(stream.max()),
            "eot_count_first10M": int((stream[:10_000_000] == EOT).sum()),
        },
        "wallclock_sec": round(time.perf_counter() - t0, 1),
        "prepared_files":{name:file_identity(Path(OUT_DIR,name)) for name in ("tokens50k.bin","eval50k.bin")},
        "tokenizer_model_identity":file_identity(SP8K_MODEL),
        "full_documents_statistics":stats,
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
        self.path=str(path)
        if os.path.getsize(path)%2:raise ValueError("uint16文件字节数非偶数")
        self.arr = np.memmap(path, dtype=np.uint16, mode="r")
        self.max_step = min((len(self.arr) - 1) // TPB, cap_steps)

    def batch(self, step, device):
        if not 1<=step<=self.max_step:raise ValueError("step越界")
        off = (step - 1) * TPB
        span = np.asarray(self.arr[off:off + TPB + 1], dtype=np.int64)
        x = torch.from_numpy(span[:-1].reshape(BATCH, BLOCK)).to(device)
        y = torch.from_numpy(span[1:].reshape(BATCH, BLOCK)).to(device)
        return x, y


def build_eval_batches(device):
    """固定 eval 批集：50257 流 eval 池（末尾 600k）固定种子49批×32窗；抽样方法与ch08同型，窗位和词表不同。"""
    ev = np.memmap(os.path.join(OUT_DIR, "eval50k.bin"), dtype=np.uint16, mode="r")
    gen = torch.Generator().manual_seed(SEED)
    if len(ev)<BLOCK+1:raise ValueError("eval池至少需BLOCK+1个token")
    offs = torch.randint(0,len(ev)-BLOCK,(EVAL_BATCHES*BATCH,),generator=gen).numpy()
    xs, ys = [], []
    for o in offs:
        xs.append(np.asarray(ev[o:o + BLOCK], dtype=np.int64))
        ys.append(np.asarray(ev[o + 1:o + BLOCK + 1], dtype=np.int64))
    x = torch.from_numpy(np.stack(xs).reshape(EVAL_BATCHES, BATCH, BLOCK)).to(device)
    y = torch.from_numpy(np.stack(ys).reshape(EVAL_BATCHES, BATCH, BLOCK)).to(device)
    return x, y


@torch.no_grad()
def eval_loss(model,ex,ey):
    """fp32等长窗口均值；保存/恢复原模式，非有限值不进入拟合。"""
    if len(ex)==0 or ex.shape!=ey.shape:raise ValueError("无效评估窗口")
    before=model.training;model.eval()
    try:
        values=[float(model(ex[i],ey[i])[1]) for i in range(len(ex))]
        if not all(math.isfinite(v) for v in values):raise FloatingPointError("非有限eval")
        return sum(values)/len(values)
    finally:model.train(before)


def file_identity(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return {"bytes":os.path.getsize(path),"sha256":h.hexdigest()}


def save_json(path,value):
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    with open(str(path)+".tmp","w") as f:f.write(text)
    os.replace(str(path)+".tmp",path)


def run_identity(device,steps):
    return {"schema":2,"seed":SEED,"steps":steps,"batch":BATCH,"block":BLOCK,"vocab":VOCAB,
            "model":[N_LAYER,N_EMBD,N_HEAD,WPE],"device":device,"torch":str(torch.__version__),
            "lr":PEAK_LR,"warmup":WARMUP,"min_ratio":MIN_LR_RATIO,"weight_decay":WDECL,
            "eval_every":EVAL_EVERY,"eval_batches":EVAL_BATCHES,"eval_sampler":"all_legal_starts_v2",
            "train":file_identity(Path(OUT_DIR,"tokens50k.bin")),"eval":file_identity(Path(OUT_DIR,"eval50k.bin")),
            "factory":file_identity(Path(__file__).resolve().parents[1]/"ch04/warmup_ablation.py"),
            "runner":file_identity(__file__)}


# ---------------- 训练 ----------------
def train_124m(device,out_name="train124_fast",steps=None,bootstrap_draws=500):
    steps=TRAIN_STEPS if steps is None else steps
    if steps<=0 or not 0<=WARMUP<steps or EVAL_EVERY<=0 or bootstrap_draws<0:
        raise ValueError("需要0<=warmup<steps，eval间隔正，bootstrap非负")
    if Path(out_name).name!=out_name or out_name in ("",".",".."):raise ValueError("out-name只接受文件名")
    os.makedirs(OUT_DIR,exist_ok=True)
    identity=run_identity(device,steps)
    out_json=os.path.join(OUT_DIR,out_name+".json")
    if os.path.exists(out_json):
        old=json.loads(Path(out_json).read_text())
        if old.get("meta",{}).get("run_identity")!=identity:
            raise ValueError("已有报告身份不同或为旧版，另用out-name")
        if old.get("run",{}).get("completed"):
            return old["run"]
    ckpt_path = os.path.join(OUT_DIR, f"ckpt_{out_name}.pt")
    if not os.path.exists(ckpt_path) and any(Path(OUT_DIR,f"curve_{out_name}{ext}").exists() for ext in (".csv",".npy")):
        raise FileExistsError("同名曲线已存在但无可核续跑断点，请另选out-name")
    torch.manual_seed(SEED)
    model = GPT(n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD, vocab=VOCAB, block_size=WPE).to(device)
    n_total = model.n_params()
    n_nonemb = model.n_params(non_embedding=True)
    assert n_total == EXPECT_TOTAL, f"总参 {n_total:,} != {EXPECT_TOTAL:,}"
    assert n_nonemb == EXPECT_NONEMB, f"非嵌入 {n_nonemb:,} != {EXPECT_NONEMB:,}"
    print(f"[模型] {N_LAYER}L×{N_EMBD}d×{N_HEAD}H wte{VOCAB} wpe{WPE} tied：总参 {n_total:,} / 非嵌入 {n_nonemb:,}"
          "（参数计数已与当前配置断言核对；此处是随机初始化，不是加载官方权重）", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=WDECL)
    ex, ey = build_eval_batches(device)
    stream = TrainStream(os.path.join(OUT_DIR, "tokens50k.bin"), steps)
    if steps>stream.max_step:raise ValueError("训练步数超出流容量")
    if steps*TPB+1+EVAL_POOL>len(stream.arr):raise ValueError("训练目标范围与尾部eval池重合")

    train_losses, lrs, evals, peak_mib = [], [], [], 0.0
    start_step, elapsed_train, elapsed_eval = 0, 0.0, 0.0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path,map_location=device,weights_only=True)
        if ck.get("run_identity")!=identity:raise ValueError("检查点身份不同或缺失，另用out-name")
        model.load_state_dict(ck["model"])
        torch.set_rng_state(ck["cpu_rng_state"].cpu())
        opt.load_state_dict(ck["opt"])
        train_losses, lrs, evals = ck["train_losses"], ck["lrs"], ck["evals"]
        start_step, elapsed_train, elapsed_eval = ck["step"], ck["elapsed_train"], ck["elapsed_eval"]
        peak_mib = ck.get("peak_mib", 0.0)
        print(f"[续跑] 自 step {start_step + 1}（已完成 {start_step} 步）", flush=True)

    if not 0<=start_step<=steps or len(train_losses)!=start_step or len(lrs)!=start_step:
        raise ValueError("断点步数与曲线不一致")
    failure=None
    t_point = time.perf_counter()
    for step in range(start_step + 1, steps + 1):
        lr = lr_at(step, PEAK_LR, WARMUP, steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = stream.batch(step, device)
        t0 = time.perf_counter()
        with torch.autocast(device_type="mps", dtype=torch.bfloat16) if device == "mps" else torch.autocast("cpu", enabled=False):
            _, loss = model(x, y)
        lv=float(loss.detach())
        if not math.isfinite(lv) or lv>100:
            failure={"step":step,"reason":"非有限或超本实验阈值loss，未更新"};break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0))
        if not math.isfinite(grad_norm):
            failure={"step":step,"reason":"非有限梯度，未更新"};break
        opt.step()
        if device == "mps":
            torch.mps.synchronize()
        elapsed_train += time.perf_counter() - t0
        lv = loss.item()
        train_losses.append(lv)
        lrs.append(lr)
        if step % 100 == 0 or step == 1:
            print(f"  step {step:6d}/{steps} train {lv:.4f} lr {lr:.2e} "
                  f"({elapsed_train / step:.3f}s/step 累计)", flush=True)
        if step % EVAL_EVERY == 0 or step == steps:
            t0 = time.perf_counter()
            el = eval_loss(model, ex, ey)
            if device == "mps":
                torch.mps.synchronize()
                peak_mib = max(peak_mib, torch.mps.current_allocated_memory() / 2 ** 20)
            elapsed_eval += time.perf_counter() - t0
            evals.append({"step": step, "eval_loss": round(el, 5), "n_tokens": EVAL_BATCHES * TPB})
            print(f"  [eval] step {step}: eval loss {el:.4f}（{EVAL_BATCHES*TPB:,} token，fp32）", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "train_losses": train_losses, "lrs": lrs, "evals": evals,
                        "elapsed_train": elapsed_train, "elapsed_eval": elapsed_eval,
                        "peak_mib":peak_mib,"memory_scope":"eval结束后的current离散读数",
                        "run_identity":identity,"cpu_rng_state":torch.get_rng_state(),
                        "config":{"n_layer":N_LAYER,"n_embd":N_EMBD,"n_head":N_HEAD,"vocab":VOCAB,"block_size":WPE}},
                       ckpt_path+".tmp")
            os.replace(ckpt_path+".tmp",ckpt_path)

    wallclock = time.perf_counter() - t_point
    steps_done = len(train_losses)
    tail=train_losses[-min(100,steps_done):] if steps_done else []
    completed=failure is None and steps_done==steps and bool(evals) and evals[-1]["step"]==steps
    result = {
        "model": {"n_layer": N_LAYER, "n_embd": N_EMBD, "n_head": N_HEAD, "vocab": VOCAB,
                  "wpe": WPE, "tied": True, "params_total": n_total, "params_non_embedding": n_nonemb},
        "steps":steps_done,"steps_planned":steps,"completed":completed,"failure":failure,
        "checkpoint":ckpt_path,"run_identity":identity, "D_tokens": steps_done * TPB,
        "batch_block": f"b{BATCH}×s{BLOCK}（{TPB} token/步）",
        "schedule": f"lr {PEAK_LR} 余弦降至 {PEAK_LR*MIN_LR_RATIO:.0e} + warmup {WARMUP}（ch08/ch11-iso 同日程）",
        "initial_train_loss": round(train_losses[0], 4) if train_losses else None,
        "initial_loss_check": f"ln({VOCAB})={math.log(VOCAB):.4f}（std=0.02 初始化自检：初始 loss 应≈此值+小量）",
        "final_train_loss":round(sum(tail)/len(tail),5) if tail else None,
        "final_train_loss_def": "最后min(100,已完成步数)步train loss均值",
        "eval_losses": evals,
        "eval_final":evals[-1]["eval_loss"] if completed else None,
        "eval_tokens_per_eval": EVAL_BATCHES * TPB,
        "wallclock":round(wallclock,1),"wallclock_scope":"本次调用片段，累计纯训练/评估时间另列",
        "wallclock_sec_train_only": round(elapsed_train, 1),
        "wallclock_sec_eval_only": round(elapsed_eval, 1),
        "sec_per_step": round(elapsed_train / max(1, steps_done), 4),
        "tokens_per_sec_train": round(steps_done * TPB / max(1e-9, elapsed_train), 1),
        "effective_tflops_6nd": round(6.0 * n_nonemb * steps_done * TPB / max(1e-9, elapsed_train) / 1e12, 2),
        "max_current_mib_after_eval":round(peak_mib,1) if device=="mps" else None,
        "memory_note":"eval结束后离散current读数最大值，不是训练峰值",
        "precision":"train bf16 autocast / eval fp32" if device=="mps" else "train/eval fp32",
    }

    # 曲线落盘
    stem = os.path.join(OUT_DIR, "curve_"+out_name)
    with open(stem + ".csv", "w", encoding="utf-8") as f:
        f.write("step,train_loss,lr\n")
        for i, (lv, lr) in enumerate(zip(train_losses, lrs), 1):
            f.write(f"{i},{lv:.6g},{lr:.6g}\n")
    np.save(stem + ".npy", np.column_stack([np.arange(1, steps_done + 1), train_losses, lrs]))

    report={"meta":{"date_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                    "run_identity":identity,"seed":SEED,"device":device,"torch":str(torch.__version__),
                    "experiment":"指定GPT2规格的有限预算训练；位置一次通过不证明内容无重复",
                    "data_note":"token流与eval尾池位置互斥，跨文本/词表不能直接比较损失"},
            "run":result,"analysis":analyze_trajectory(result,bootstrap_draws) if completed else None}
    save_json(out_json,report)
    # 完成仍保留断点，可直接交采样脚本；失败不删除最后的有效状态。
    print(f"[{'完成' if completed else '未完成'}] {steps_done}/{steps} -> {out_json}",flush=True)
    return result


# ---------------- 分析：轨迹双拟合；缺同范围byte分母时不跨词表换算 ----------------
def _plaw(D, E, B, beta):
    return E + B * np.asarray(D, dtype=float) ** (-beta)


def _r2_on_L(y, yhat):
    y = np.asarray(y, dtype=float)
    den=float(np.sum((y-y.mean())**2))
    return 1-float(np.sum((y-yhat)**2))/den if den>0 else None


def analyze_trajectory(result,bootstrap_draws=500):
    """对 eval 轨迹（L vs D）做 ch8/fit_scaling.py 同型双拟合：E 自由 vs E=0。

    红线：这是「单条余弦退火轨迹」上的幂律（早期点在高 lr 下、末点在退火末端），与 Kaplan/Chinchilla
    多次独立训练终点不是同一对象；原论文也采用学习率日程，只作本轨迹特征描述。
    """
    from scipy import optimize
    ev=result["eval_losses"]
    if bootstrap_draws<0:raise ValueError("bootstrap次数须非负")
    if result.get("completed") is False:raise ValueError("未完成运行不拟合最终曲线")
    if len(ev)<2:return {"status":"不足两个评估点，不拟合"}
    if any(e["step"]<=0 or not math.isfinite(e["eval_loss"]) or e["eval_loss"]<=0 for e in ev):
        raise ValueError("幂律拟合需要正且有限的D/L")
    if any(ev[i]["step"]>=ev[i+1]["step"] for i in range(len(ev)-1)):
        raise ValueError("评估步数须严格递增")
    D = np.array([e["step"] * TPB for e in ev], dtype=float)
    L = np.array([e["eval_loss"] for e in ev], dtype=float)
    out = {"fit_domain": "eval 轨迹 L(D)（每 500 步一个点，fp32，401,408 token/点）",
           "red_line":"单轨迹共享优化历史，不等于多次独立训练终点；原论文也使用学习率日程",
           "residual_objectives":"自由E拟合原损失残差，固定E=0拟合log损失残差，不能称纯单变量比较"}

    def fit_free(d, l):
        if len(np.unique(d))<3:return None,None
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
    for _ in range(bootstrap_draws):
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
        "skipped":skip,"draws":bootstrap_draws,
        "interpretation":"同条相关轨迹的case重采样，仅作条件敏感性，非独立重训CI",
    }

    out["cross_tokenizer_comparison"]={
        "per_byte":None,
        "reason":"缺同一实际评分范围的原文byte分母；旧ch08 3.82是JSONL包装字节，不能与净文本4.398混用"}
    return out


# ---------------- 主流程 ----------------
def main():
    global OUT_DIR,DOCS_JSONL,SP8K_MODEL,SEED
    ap=argparse.ArgumentParser(description="GPT2 124M准备/训练/独立分析；保留断点供采样")
    ap.add_argument("--prep-only",action="store_true")
    ap.add_argument("--skip-prep",action="store_true")
    ap.add_argument("--analyze-only",action="store_true")
    ap.add_argument("--input",type=Path)
    ap.add_argument("--out-dir",default=OUT_DIR)
    ap.add_argument("--out-name",default="train124_fast")
    ap.add_argument("--docs",default=DOCS_JSONL)
    ap.add_argument("--sp-model",default=SP8K_MODEL)
    ap.add_argument("--steps",type=int,default=TRAIN_STEPS)
    ap.add_argument("--bootstrap-draws",type=int,default=500)
    ap.add_argument("--seed",type=int,default=SEED)
    ap.add_argument("--workers",type=int,default=6)
    ap.add_argument("--device",choices=["cpu","mps"],default="mps" if torch.backends.mps.is_available() else "cpu")
    a=ap.parse_args();OUT_DIR=a.out_dir;DOCS_JSONL=a.docs;SP8K_MODEL=a.sp_model;SEED=a.seed
    os.makedirs(OUT_DIR,exist_ok=True)
    name=a.out_name[:-5] if a.out_name.endswith(".json") else a.out_name
    if Path(name).name!=name or name in ("",".",".."):raise ValueError("out-name只接受文件名")
    if a.analyze_only:
        source=a.input or Path(OUT_DIR,name+".json")
        target=Path(OUT_DIR,name+"_analysis.json")
        if target.exists() or source.resolve()==target.resolve():raise FileExistsError("分析另存，不覆盖")
        old=json.loads(source.read_text())
        save_json(target,{"source":str(source),"source_identity":file_identity(source),
                          "analysis":analyze_trajectory(old["run"],a.bootstrap_draws)})
        print(target);return
    token_path=Path(OUT_DIR,"tokens50k.bin")
    if not token_path.exists():
        if a.skip_prep:raise FileNotFoundError("要求skip-prep但数据流不存在")
        prep_50k(workers=a.workers)
    meta_path=Path(OUT_DIR,"data50k_meta.json")
    if not meta_path.exists():raise ValueError("数据旁件不完整，禁止仅凭tokens文件复用")
    meta=json.loads(meta_path.read_text())
    for key in ("tokens50k.bin","eval50k.bin"):
        if meta.get("prepared_files",{}).get(key)!=file_identity(Path(OUT_DIR,key)):
            raise ValueError("准备数据缺新版身份或内容已变，使用新out-dir，不覆盖旧流")
    if a.prep_only:return
    result=train_124m(a.device,name,a.steps,a.bootstrap_draws)
    if not result["completed"]:raise RuntimeError("训练未完成；有效断点已保留")


if __name__ == "__main__":
    main()
