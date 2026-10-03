# train.py —— Book2 ch10 大项目 I 定版训练循环（CLI）
# 所属章节：Book2 第 10 章 §10.3/10.5（ch10-大项目I.md 图 10.2/10.3 数据源）
# 单轨依赖链：Book1 ch9 训练循环 -> ch4/ch8 参数化前身（warmup_ablation.py / run_family.py）-> 本文件定版 -> ch11 复用。
# 配方：AdamW(0.9, 0.95, wd 0.1) + grad clip 1.0 + cosine 退火（warmup 线性升、降到 0.1x 峰值）+ 可选 bf16 autocast；
#   eval 协议 = held-out 段「一次通过、窗口不重叠」（第 4 章 47 遍记忆化教训的修正）。
# 任务：--task char（6L/384d/6H 字符级热身，nanoGPT shakespeare_char 同口径，10.65M 非嵌入）
#       --task smoke124（124M 冒烟：tiny shakespeare 自训 sp-8k 编码——分词器在独立前 10% 分片上训练、与 LM 语料
#                        分离；id<8192 天然落在 50257 词表内，参数 assert 不变；只验循环健康不追指标）
#       --parity-hf（验收②：随机权重同 config 对拍 HF，CPU fp32）
# 运行方式：python train.py --task char --steps 1000 | --task smoke124 --steps 150 --bf16 | --parity-hf
# 产物：log/book2-ch10/{out-name}.json（meta/曲线/计时/内存峰值/语料一元账 corpus_stats）
import argparse
import json
import os
import threading
import time
import urllib.request

import numpy as np
import torch

from model import GPT2, GPT2Config, GPT2_124M

SEED = 20261002                       # 全书统一种子
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book2-ch10")
DATA_PATH = os.path.join(OUT_DIR, "data", "tiny_shakespeare.txt")
FALLBACK_DATA = os.path.join(REPO_ROOT, "log", "book2-feasibility", "tiny_shakespeare.txt")
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

TASKS = {  # 各任务默认档（均可被 CLI 覆盖）
    "char": dict(n_layer=6, n_embd=384, n_head=6, block=256, batch=64, lr=1e-3,
                 warmup=100, steps=1000, schedule_total=5000, eval_every=250),
    "smoke124": dict(**GPT2_124M, block=256, batch=8, lr=1e-4,
                     warmup=10, steps=150, schedule_total=150, eval_every=50),
}


def ensure_corpus() -> str:
    """tiny shakespeare（公版莎剧子集；char-rnn 仓库代码 MIT、数据文件无专门授权条款）。"""
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        if os.path.exists(FALLBACK_DATA):                   # 复用探针/第 4 章已下载副本（拷贝，不动原文件）
            import shutil
            shutil.copyfile(FALLBACK_DATA, DATA_PATH)
        else:
            print("下载 tiny shakespeare …", flush=True)
            urllib.request.urlretrieve(SHAKESPEARE_URL, DATA_PATH)
    return DATA_PATH


def load_char_ids():
    """字符级编码：全文 -> 排序去重字符表 -> id 流；位置切分 train 90% / val 10%。"""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(len(ids) * 0.9)
    return ids[:n], ids[n:], len(chars), {"kind": "char", "vocab": len(chars)}


def load_sp8k_ids():
    """sp-8k 编码：分词器只在「前 10% 文本」分片上训练（与 LM 语料分离，写入运行元数据）；
    LM train = 10%~90%，val = 90%~100%。id < 8192，天然落在 50257 词表内。"""
    import sentencepiece as spm
    tok_prefix = os.path.join(OUT_DIR, "sp8k_shakespeare")   # 分词器分片训练，缓存复用
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    n_tok = int(len(text) * 0.1)
    tok_model = tok_prefix + ".model"
    if not os.path.exists(tok_model):
        with open(tok_prefix + ".txt", "w", encoding="utf-8") as f:
            f.write(text[:n_tok])
        spm.SentencePieceTrainer.train(input=tok_prefix + ".txt", model_prefix=tok_prefix,
                                       model_type="bpe", vocab_size=8192,
                                       byte_fallback=True, character_coverage=1.0)
    sp = spm.SentencePieceProcessor()
    sp.Load(tok_model)
    lm_text = text[n_tok:]
    ids = torch.tensor(sp.EncodeAsIds(lm_text), dtype=torch.long)
    n_train = int(len(ids) * 0.9)                          # 按 token 数切 train/val（位置切分）
    tok_info = {"kind": "sp-8k(bpe, byte_fallback)", "trained_on": "前 10% 分片（与 LM 语料分离）",
                "vocab": sp.GetPieceSize()}
    return ids[:n_train], ids[n_train:], sp.GetPieceSize(), tok_info


def get_batch(split_ids, block, batch, device):
    """训练批：随机采样窗口，x 与 y 错位一格（因果 LM 目标）。x (B,n) / y (B,n)。"""
    ix = torch.randint(len(split_ids) - block - 1, (batch,))
    x = torch.stack([split_ids[i:i + block] for i in ix]).to(device)
    y = torch.stack([split_ids[i + 1:i + block + 1] for i in ix]).to(device)
    return x, y


def corpus_stats(train_ids, val_ids, vocab):
    """语料一元分布账（10.5 节「地板」数字的产物落盘）：train/val token 数、
    训练段活跃 token 种数（去重）、词频熵 H=-Σp·ln p（nat）——「只背词频」的地板。"""
    cnt = torch.bincount(train_ids, minlength=vocab).double()
    p = cnt / cnt.sum()
    p = p[p > 0]
    return {"train_tokens": int(train_ids.numel()), "val_tokens": int(val_ids.numel()),
            "train_active_types": int((cnt > 0).sum()),
            "train_unigram_entropy_nat": round(float(-(p * p.log()).sum()), 2),
            "uniform_entropy_lnV": round(float(torch.log(torch.tensor(float(vocab)))), 2)}


@torch.no_grad()
def eval_once_through(model, val_ids, block, batch, device, bf16=False):
    """定版 eval 协议：val 段「一次通过、窗口不重叠」，token 加权平均——确定性、可复算、
    无采样噪声（第 4 章「固定种子 20 批」协议的升级；随机采样会把 eval 段也重复抽查多遍）。
    逐窗组批前向：xb (batch,n) 由连续不重叠窗口堆叠而成。"""
    model.eval()
    total_loss, total_tok = 0.0, 0
    xs, ys = [], []
    for i in range(0, len(val_ids) - block - 1, block):
        xs.append(val_ids[i:i + block])                    # (n,) 窗口
        ys.append(val_ids[i + 1:i + block + 1])            # 错位一格的目标
        if len(xs) == batch or i + 2 * block >= len(val_ids) - 1:
            xb = torch.stack(xs).to(device)                # (B,n)
            yb = torch.stack(ys).to(device)
            ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if bf16 else torch.enable_grad()
            with ctx:
                _, loss = model(xb, yb)
            total_loss += loss.item() * xb.numel()
            total_tok += xb.numel()
            xs, ys = [], []
    model.train()
    return total_loss / total_tok


class MpsMemSampler(threading.Thread):
    """MPS 内存峰值采样（口径=轮询 current_allocated_memory，30ms；全书对拍规范）。"""

    def __init__(self, interval=0.03):
        super().__init__(daemon=True)
        self.interval, self.peak = interval, 0.0
        self._stop_evt = threading.Event()   # 命名避开 Thread._stop 内部属性

    def run(self):
        while not self._stop_evt.is_set():
            try:
                self.peak = max(self.peak, torch.mps.current_allocated_memory() / 1024**2)
            except Exception:
                pass
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set(); self.join(timeout=1.0)
        return round(self.peak, 1)


def lr_at(step, peak, warmup, total, min_ratio=0.1):
    """lr 日程：warmup 线性升到 peak，之后余弦降到 min_ratio*peak（ch4/ch8 同一函数的定版）。"""
    if warmup > 0 and step <= warmup:
        return peak * step / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_ratio * peak + (1 - min_ratio) * peak * 0.5 * (1 + np.cos(np.pi * min(t, 1.0)))


def run_task(args):
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    cfgd = TASKS[args.task]
    steps = args.steps or cfgd["steps"]
    eval_every = args.eval_every or cfgd["eval_every"]
    total = args.schedule_total or cfgd["schedule_total"]
    ensure_corpus()
    train_ids, val_ids, vocab, tok_info = load_char_ids() if args.task == "char" else load_sp8k_ids()
    cfg = GPT2Config(vocab_size=GPT2_124M["vocab_size"] if args.task == "smoke124" else vocab,
                     n_positions=1024 if args.task == "smoke124" else cfgd["block"],
                     n_layer=cfgd["n_layer"], n_embd=cfgd["n_embd"], n_head=cfgd["n_head"])
    model = GPT2(cfg).to(device)
    if args.task == "smoke124":
        assert model.n_params() == 124_439_808, "124M 参数逐位对账失败"
    opt = torch.optim.AdamW(model.parameters(), lr=cfgd["lr"], betas=(0.9, 0.95), weight_decay=0.1)

    # 初始自检：随机初始化 loss 应 ≈ ln(vocab)（初始化警示的运行时版）
    x0, y0 = get_batch(val_ids, cfgd["block"], min(cfgd["batch"], 8), device)
    with torch.no_grad():
        _, l0 = model(x0, y0)
    mem = MpsMemSampler() if device == "mps" else None
    if mem:
        mem.start()
    curve, run_sum, run_n = [], 0.0, 0
    t0 = time.perf_counter()
    t_train = 0.0                       # 纯训练步计时（不含 eval），与探针口径可比
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfgd["lr"], cfgd["warmup"], total)
        ts = time.perf_counter()
        x, y = get_batch(train_ids, cfgd["block"], cfgd["batch"], device)
        ac = torch.autocast(device_type=device, dtype=torch.bfloat16) if args.bf16 else torch.enable_grad()
        with ac:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device == "mps":
            torch.mps.synchronize()
        t_train += time.perf_counter() - ts
        run_sum += loss.item(); run_n += 1
        if step % eval_every == 0 or step == steps:
            vl = eval_once_through(model, val_ids, cfgd["block"], 8, device, args.bf16)
            curve.append({"step": step, "train_loss": round(run_sum / run_n, 4), "val_loss": round(vl, 4)})
            print(f"step {step:5d} train {run_sum/run_n:.4f} val {vl:.4f}", flush=True)
            run_sum, run_n = 0.0, 0
    if device == "mps":
        torch.mps.synchronize()
    sec = time.perf_counter() - t0
    peak = mem.stop() if mem else None

    report = {"meta": {"date": "2026-10-03", "seed": args.seed, "task": args.task, "device": device,
                       "dtype": "bf16-autocast" if args.bf16 else "fp32",
                       "torch": torch.__version__, "corpus_bytes": os.path.getsize(DATA_PATH),
                       "tokenizer": tok_info, "model_vocab": cfg.vocab_size,
                       "n_params_total": model.n_params(),
                       "n_params_non_embedding": model.n_params(non_embedding=True),
                       "block": cfgd["block"], "batch": cfgd["batch"], "lr": cfgd["lr"],
                       "warmup": cfgd["warmup"], "schedule_total": total,
                       "optimizer": "AdamW(0.9,0.95,wd=0.1)+clip1.0+cosine->0.1x",
                       "eval_protocol": "val 段一次通过、窗口不重叠、token 加权",
                       "init_loss_check": round(l0.item(), 4),
                       "init_loss_expected_lnV": round(float(torch.log(torch.tensor(float(cfg.vocab_size)))), 4)},
              "corpus_stats": corpus_stats(train_ids, val_ids, vocab),
              "sec_per_step": round(sec / steps, 4), "sec_total": round(sec, 1),
              "sec_train_only_per_step": round(t_train / steps, 4),
              "mps_peak_mib": peak, "curve": curve}
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{args.out_name or args.task}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"sec/step {sec/steps:.4f}  total {sec:.1f}s  peak {peak} MiB  -> {out}")


def parity_hf():
    """验收②：随机权重同 config 对拍 HF（CPU fp32）——GPT2LMHeadModel(GPT2Config()) 随机初始化
    （不下权重），state_dict 按键映射表搬进手写模型（Conv1D 转置 / tied 补绑 / 前缀剥掉），
    同输入对拍 logits：权重逐位相同 + logits 一致 => 两实现前向数学等价（比加载官方权重更早可用）。"""
    from transformers import GPT2Config as HFCfg, GPT2LMHeadModel
    torch.manual_seed(SEED)
    ours = GPT2().eval()                                   # 124M 默认 config
    cfg_hf = HFCfg()
    cfg_hf._attn_implementation = "eager"                  # v5：注意力实现挂在 config（默认 sdpa；eager 与手写路径同型）
    hf = GPT2LMHeadModel(cfg_hf).eval()
    n_ours, n_hf = ours.n_params(), sum(p.numel() for p in hf.parameters())
    print(f"参数量：本书 {n_ours:,}  vs  HF {n_hf:,}   一致={n_ours == n_hf == 124_439_808}")

    sd, n_t = {}, 0
    for k, v in hf.state_dict().items():
        if k.split(".")[-2:] == ["attn", "bias"] or k.startswith("lm_head."):
            continue                                        # 遗留 mask buffer（attn.bias）与 tied 头（lm_head=wte 同存储，checkpoint 只存一份）
        k2 = k.replace("transformer.", "").replace("h.", "blocks.")
        if k2.endswith(("c_attn.weight", "c_proj.weight", "c_fc.weight")):
            v = v.t()                                       # Conv1D (in,out) -> Linear (out,in)
            n_t += 1
        sd[k2] = v
    sd["head.weight"] = sd["wte.weight"]                    # tied：加载时显式补绑（ch11 高危点 B 同款）
    ours.load_state_dict(sd, strict=True)
    print(f"键映射搬移完成（转置 {n_t} 处 Conv1D 权重；strict=True 无缺键/多键即通过）")

    g = torch.Generator().manual_seed(SEED)                # 固定输入（随机 token；真实文本对拍归 ch11）
    ids = torch.randint(0, 50257, (4, 128), generator=g)
    with torch.no_grad():
        lg_ours, _ = ours(ids)
        lg_hf = hf(input_ids=ids).logits
    d = (lg_ours - lg_hf).abs()
    top5_o = lg_ours.topk(5, dim=-1).indices
    top5_h = lg_hf.topk(5, dim=-1).indices
    top5_hit = (top5_o.unsqueeze(-1) == top5_h.unsqueeze(-2)).any(-1).float().mean().item()
    am = (lg_ours.argmax(-1) == lg_hf.argmax(-1)).float().mean().item()
    print(f"logits 对拍：max|Δ| = {d.max().item():.3e}   mean|Δ| = {d.mean().item():.3e}")
    print(f"argmax 一致率 = {am:.4f}   top-5 命中率 = {top5_hit:.4f}")
    print(f"验收判定（atol 1e-4）：{'PASS' if d.max().item() < 1e-4 and am == 1.0 else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default="char")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--schedule-total", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--parity-hf", action="store_true")
    args = ap.parse_args()
    if args.eval_every:
        TASKS[args.task]["eval_every"] = args.eval_every
    if args.parity_hf:
        parity_hf()
    else:
        run_task(args)


if __name__ == "__main__":
    main()
