# warmup_ablation.py —— Book2 ch4 §4.5「停药实验」：{post-LN, pre-LN}×{warmup 0, 200} 四配置 2×2 短训对照
# 手术背景：GPT-1 术后现状 = post-LN + 2000 步 warmup（§4.1 原配方，仍在吃药）；GPT-2 §2.3 = pre-LN + ln_f +
#   残差初始化 1/√N（术后停药）。判据二居其一：post-LN×warmup0 「绝对发散」或「相对自伤」（同配置加 warmup 明显更优）；
#   两者皆否 => 6 层档可停药（与 Nguyen&Salazar 1910.05895「6 层 post-LN 即 warmup-free、掉点在更大规模」一致，同为好叙事）。
# 所属章节：Book2 第 4 章 §4.5（ch04-GPT单塔与差异清单.md 表 4.5/图 4.3 数据源）
# 运行方式：python warmup_ablation.py --steps 3000 --cells all --out-name warmup_full [--depth 6|12 --lr 1e-3|3e-3]
#   冒烟：--steps 500 --cells post_ln__w0,pre_ln__w0,post_ln__w200 --out-name warmup_smoke
# 产物：log/book2-ch04/{out-name}.json + 每配置曲线 {out-name}__{cell}.csv / .npy（step, train_loss, ema, grad_norm, lr）

import argparse
import json
import math
import os
import shutil
import sys
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 固定口径 ----------------
SEED = 20261002  # 全书统一种子（与 00-feasibility 探针同源，可复算）；--seed 可换种子复核稳健性
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book2-ch04")
DATA_PATH = os.path.join(OUT_DIR, "data", "tiny_shakespeare.txt")
FALLBACK_DATA = os.path.join(REPO_ROOT, "log", "book2-feasibility", "data", "tiny_shakespeare.txt")
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# 四格命名：ln 位置 × warmup 步数（w0 = 停药，w200 = nanoGPT 同型对照；GPT-1 原配方 2000 步见章内考据）
CELLS = {
    "post_ln__w0": ("post", 0),
    "post_ln__w200": ("post", 200),
    "pre_ln__w0": ("pre", 0),
    "pre_ln__w200": ("pre", 200),
}


# ---------------- 语料（char 级 tiny shakespeare，落 log/ 缓存红线内） ----------------
def ensure_corpus() -> str:
    """语料三级兜底：本目录已有 -> 复用探针(b)缓存 -> 官方 URL 下载，全部落在 log/ 下。"""
    if os.path.exists(DATA_PATH):
        return DATA_PATH
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if os.path.exists(FALLBACK_DATA):
        shutil.copy(FALLBACK_DATA, DATA_PATH)
        return DATA_PATH
    print(f"[语料] 下载 {SHAKESPEARE_URL} -> {DATA_PATH}", flush=True)
    urllib.request.urlretrieve(SHAKESPEARE_URL, DATA_PATH)
    return DATA_PATH


def load_ids():
    """全文 -> 排序字符表 -> id 张量，90/10 切 train/val（探针(b)同口径）。返回 (train_ids, val_ids, vocab)。"""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(len(ids) * 0.9)
    return ids[:n], ids[n:], len(chars)


def get_batch(ids, block, batch, device, gen=None):
    """随机采样一个批：x,y 均为 (B,T)，y 与 x 错位一格（next-char 预测）。gen 给定则用固定生成器（评测口径）。"""
    ix = torch.randint(len(ids) - block - 1, (batch,), generator=gen)
    x = torch.stack([ids[i:i + block] for i in ix]).to(device)      # (B,T) 整数 id
    y = torch.stack([ids[i + 1:i + block + 1] for i in ix]).to(device)  # (B,T) 右移一位的标签
    return x, y


# ---------------- 模型：块实现参数化（ch8 族模型工厂的前身） ----------------
class Block(nn.Module):
    """可配置 LN 位置的 Transformer 解码块——同一组参数形状，两种接法（手术②的最小实现）。

    ln_position="pre"（GPT-2 式）:  x = x + attn(ln_1(x)); x = x + mlp(ln_2(x))   —— LN 在子层入口
    ln_position="post"（GPT-1 式）: x = ln_1(x + attn(x)); x = ln_2(x + mlp(x))   —— LN 在残差和之后
    forward 输入/输出均为 (B,T,C)。n_embd/n_head/block_size 由构造参数决定，供上层 GPT 按
    n_layer 组装任意深度/宽度——ch8 的 0.5M~25M 族工厂直接复用本类（单轨依赖链：ch4 -> ch8）。
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, ln_position: str = "pre"):
        super().__init__()
        assert ln_position in ("pre", "post")
        assert n_embd % n_head == 0
        self.ln_position = ln_position
        self.n_head = n_head
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        B, T, C = x.shape                                                  # (B,T,C)
        if self.ln_position == "pre":   # pre-LN：先归一再进子层，残差主干上是干净的加法
            x = x + self.attn(self.ln_1(x))                                # (B,T,C) -> (B,T,C)
            x = x + self.mlp(self.ln_2(x))                                 # (B,T,C) -> (B,T,C)
        else:                           # post-LN：先加残差再归一，梯度须经 LN 双口进出每个子层
            x = self.ln_1(x + self.attn(x))                                # (B,T,C) -> (B,T,C)
            x = self.ln_2(x + self.mlp(x))                                 # (B,T,C) -> (B,T,C)
        return x


class CausalSelfAttention(nn.Module):
    """多头因果自注意力（合并式 QKV，nanoGPT 口径）。输入 (B,T,C) -> 输出 (B,T,C)。"""

    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)     # (B,T,C) -> (B,T,3C)
        self.c_proj = nn.Linear(n_embd, n_embd)         # (B,T,C) -> (B,T,C)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))  # (1,1,T,T) 下三角因果掩码

    def forward(self, x):
        B, T, C = x.shape                                                # (B,T,C)
        q, k, v = self.c_attn(x).split(C, dim=2)                         # 各 (B,T,C)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)                # (B,h,T,hd)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)                  # (B,h,T,T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        y = F.softmax(att, dim=-1) @ v                                   # (B,h,T,hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)                 # (B,h,T,hd) -> (B,T,C)
        return self.c_proj(y)


class MLP(nn.Module):
    """GPT-2 前馈：C -> 4C -> 4C -> C（GELU）。输入 (B,T,C) -> 输出 (B,T,C)。"""

    def __init__(self, n_embd: int):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class GPT(nn.Module):
    """参数化小 GPT：词嵌入 + 位置嵌入 + n_layer 个 Block +（仅 pre-LN）ln_f + tied lm_head。

    ln_position="pre" 时按 GPT-2 §2.3 在最后一个块之后加 ln_f（手术②的两半：挪 LN + 补 ln_f）；
    "post" 时与 GPT-1/Book1 同型、无 ln_f。p_drop 默认 0（小语料短训不设 dropout，参数保留给 ch8 族）。
    前向：idx (B,T) -> logits (B,T,V)。初始化 GPT-2 口径 N(0,0.02)，残差流出端投影再乘 0.02/sqrt(2*n_layer)
    ——两种 ln_position 用同一初始化流（参数形状全同、RNG 消耗一致），保证四格唯一的差异变量就是 LN 接法与 warmup。
    """

    def __init__(self, n_layer=6, n_embd=384, n_head=6, vocab=65, block_size=256,
                 ln_position="pre", p_drop=0.0):
        super().__init__()
        self.block_size = block_size
        self.ln_position = ln_position
        self.wte = nn.Embedding(vocab, n_embd)          # (V,C)
        self.wpe = nn.Embedding(block_size, n_embd)     # (T_max,C) 学习式位置嵌入
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, ln_position) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) if ln_position == "pre" else nn.Identity()
        self.head = nn.Linear(n_embd, vocab, bias=False)
        self.head.weight = self.wte.weight              # tied embeddings：输出投影与词嵌入共享
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
        assert p_drop == 0.0, "教学实现未含 dropout（ch8 族工厂扩展点）"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape                                                  # (B,T)
        pos = torch.arange(T, device=idx.device)                          # (T,)
        x = self.wte(idx) + self.wpe(pos)                                 # (B,T,C) + (T,C) -> (B,T,C)
        for block in self.blocks:
            x = block(x)                                                  # (B,T,C)
        x = self.ln_f(x)                                                  # (B,T,C)
        logits = self.head(x)                                             # (B,T,V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),   # (B*T,V)
                                  targets.reshape(-1))                    # (B*T,) -> 标量
        return logits, loss

    def n_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        return n - self.wte.weight.numel() - self.wpe.weight.numel() if non_embedding else n


# ---------------- 训练单格 ----------------
def lr_at(step, peak, warmup, total, min_ratio=0.1):
    """lr 日程：warmup 段线性升到 peak，之后余弦降到 min_ratio*peak（nanoGPT 同型；warmup=0 即从 peak 起步）。"""
    if warmup > 0 and step <= warmup:
        return peak * step / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


@torch.no_grad()
def eval_val_loss(model, val_ids, block, batch, device, n_batches=20):
    """val loss：固定种子 20 批均值（探针(b)同口径，跨格可比）。"""
    model.eval()
    gen = torch.Generator().manual_seed(20261002)   # val 批序固定（与探针 b 同源，不随 --seed 变）
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(val_ids, block, batch, device, gen)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def run_cell(cell, ln_position, warmup, args, train_ids, val_ids, vocab, device):
    """跑一格：固定种子重建模型与数据批序（四格初始权重与批序列完全一致），逐 step 记录曲线。"""
    print(f"\n===== 格 {cell}：ln={ln_position}, warmup={warmup}, L={args.depth}, "
          f"lr={args.lr}, steps={args.steps}, seed={args.seed} =====", flush=True)
    torch.manual_seed(args.seed)          # 每格重置：init 与批序跨格一致，唯一变量 = LN 接法 × warmup
    model = GPT(n_layer=args.depth, n_embd=384, n_head=6, vocab=vocab, block_size=256,
                ln_position=ln_position).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    curve = {"step": [], "train_loss": [], "ema": [], "grad_norm": [], "lr": []}
    ema = None
    diverged_at, reason = None, None
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        lr = lr_at(step, args.lr, warmup, args.steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(train_ids, 256, 64, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        lv = loss.item()
        ema = lv if ema is None else 0.99 * ema + 0.01 * lv
        if not math.isfinite(lv) or lv > 100.0:   # 发散判据：非有限值，或 loss 冲破 100（ln65≈4.17 起步）
            diverged_at, reason = step, ("non-finite" if not math.isfinite(lv) else f"loss={lv:.1f}>100")
            curve["step"].append(step); curve["train_loss"].append(lv); curve["ema"].append(ema)
            curve["grad_norm"].append(gn); curve["lr"].append(lr)
            print(f"  [发散] step {step}: {reason}", flush=True)
            break
        curve["step"].append(step); curve["train_loss"].append(lv); curve["ema"].append(ema)
        curve["grad_norm"].append(gn); curve["lr"].append(lr)
        if step % 50 == 0 or step == 1:
            print(f"  step {step:5d} train {lv:.4f} ema {ema:.4f} |g| {gn:.2f} lr {lr:.2e}", flush=True)
    if device == "mps":
        torch.mps.synchronize()
    sec = time.perf_counter() - t0

    n_done = len(curve["step"])
    end_n = max(50, args.steps // 10)                 # 末段 = 最后 10% 步（至少 50 步）
    tail = curve["train_loss"][-min(end_n, n_done):]
    result = {
        "cell": cell, "ln_position": ln_position, "warmup": warmup,
        "n_layer": args.depth, "n_embd": 384, "n_head": 6,
        "steps_planned": args.steps, "steps_done": n_done,
        "diverged": diverged_at is not None, "diverged_at_step": diverged_at, "divergence_reason": reason,
        "n_params_total": model.n_params(), "n_params_non_embedding": model.n_params(non_embedding=True),
        "end_mean_train_loss": round(sum(tail) / len(tail), 4),
        "end_segment": [max(1, n_done - min(end_n, n_done) + 1), n_done],
        "min_train_loss": round(min(curve["train_loss"]), 4),
        "final_ema": round(curve["ema"][-1], 4),
        "max_grad_norm_first200": round(max(curve["grad_norm"][:200]), 3),
        "batch": 64, "block": 256, "peak_lr": args.lr,
        "sec_per_step": round(sec / n_done, 4), "sec_total": round(sec, 1),
        "final_val_loss": (round(eval_val_loss(model, val_ids, 256, 64, device), 4)
                           if n_done >= args.steps // 2 else None),  # 中途发散的格不给 val（无意义）
    }
    # 曲线落盘：csv + npy（图 4.3 自产数据源）
    stem = os.path.join(OUT_DIR, f"{args.out_name}__{cell}")
    with open(stem + ".csv", "w", encoding="utf-8") as f:
        f.write("step,train_loss,ema,grad_norm,lr\n")
        for row in zip(curve["step"], curve["train_loss"], curve["ema"],
                       curve["grad_norm"], curve["lr"]):
            f.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row) + "\n")
    np.save(stem + ".npy", np.column_stack([curve["step"], curve["train_loss"],
                                            curve["ema"], curve["grad_norm"], curve["lr"]]))
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  [格完成] end_mean {result['end_mean_train_loss']}, diverged={result['diverged']}, "
          f"{result['sec_per_step']}s/step -> {stem}.csv", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description="ch4 §4.5 停药实验：post/pre-LN × warmup 0/200 四格对照")
    ap.add_argument("--steps", type=int, default=3000, help="每格总步数（默认 3000）")
    ap.add_argument("--cells", type=str, default="all",
                    help="逗号分隔的格名（post_ln__w0/post_ln__w200/pre_ln__w0/pre_ln__w200）或 all")
    ap.add_argument("--depth", type=int, default=6, help="层数 n_layer（默认 6；分化不足时 12）")
    ap.add_argument("--lr", type=float, default=1e-3, help="peak 学习率（默认 1e-3；保分化可 3e-3）")
    ap.add_argument("--out-name", type=str, default="warmup_ablation", help="产物名（JSON/CSV 前缀）")
    ap.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=SEED, help="训练种子（init+批序；默认全书 20261002）")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    cell_names = list(CELLS) if args.cells == "all" else [c.strip() for c in args.cells.split(",")]
    for c in cell_names:
        assert c in CELLS, f"未知格名 {c}"

    ensure_corpus()
    train_ids, val_ids, vocab = load_ids()
    device = args.device
    print(f"[实验] steps={args.steps} depth={args.depth} lr={args.lr} cells={cell_names} "
          f"device={device} vocab={vocab}", flush=True)

    results = [run_cell(c, *CELLS[c], args, train_ids, val_ids, vocab, device) for c in cell_names]

    report = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": args.seed, "device": device,
            "torch": torch.__version__, "corpus": "tiny_shakespeare(char)",
            "vocab": vocab, "steps": args.steps, "peak_lr": args.lr,
            "optimizer": "AdamW(betas=0.9,0.95,wd=0.1)+grad_clip1.0, cosine->0.1*peak",
            "schedule": "warmup 线性升 + 余弦降（min_ratio=0.1）；w0 即无 warmup 直取 peak",
            "init": "N(0,0.02), 残差 c_proj 再乘 0.02/sqrt(2N)（四格同初始化流，单变量=LN 接法×warmup）",
            "protocol": "batch 64 x block 256；每格重置种子（init/批序跨格一致）；val=固定种子 20 批均值",
            "argv": " ".join(sys.argv[1:]),
        },
        "cells": results,
    }
    # 判据速览：post-LN 两格自比（停药 vs 吃药）+ pre-LN 停药格参照
    by = {r["cell"]: r for r in results}
    summary = {}
    if "post_ln__w0" in by and "post_ln__w200" in by:
        gap = by["post_ln__w0"]["end_mean_train_loss"] - by["post_ln__w200"]["end_mean_train_loss"]
        summary["post_selfharm_gap(end_mean_w0_minus_w200)"] = round(gap, 4)
        summary["post_w0_diverged"] = by["post_ln__w0"]["diverged"]
        summary["verdict_hint"] = ("diverged=绝对发散" if by["post_ln__w0"]["diverged"] else
                                   ("gap>0.05=相对自伤" if gap > 0.05 else "两者皆否=6层档可停药(仍需看规模变量)"))
    report["summary"] = summary
    out_json = os.path.join(OUT_DIR, f"{args.out_name}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[汇总] {json.dumps(summary, ensure_ascii=False)} -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
