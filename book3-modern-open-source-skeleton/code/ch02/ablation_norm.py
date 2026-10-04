# ablation_norm.py —— Book3 ch2 模块消融：LayerNorm vs RMSNorm（同数据同种子两臂短训）
# 用途：统一模块档（d=512/L=6/h=8/V=8192，B=16/T=512）下跑两臂——base（Book2 ch10 model.py 的
#       GPT-2 底座，import 复用零改动）vs +rmsnorm（llama_slots.RMSNorm 换装 ln_1/ln_2/ln_f 共 13 处，
#       pre 位置不变、无 bias、eps=1e-5）——产出每臂 train loss 曲线 CSV、末 300 步均值、固定窗 eval
#       loss 与秒步速度；判据预注册见同目录 preregistered.md（随书快照，含执行记录）。
# 运行方式：cd code/ch02 && python ablation_norm.py [--steps 1000] [--out-name fast]
#           [--cells base,rmsnorm] [--eval-every 500]；两档位：fast=1000 步 / full=5000 步（卷级大纲 ch2 契约）。
# 产物（log/book3-ch02/，不入库）：ablation_norm_{out}.json（总报告）+ curve_{臂}_{out}.csv（每臂曲线）
#           + ckpt_{臂}_{out}.pt（断点续跑用，臂完成即删）。
# 协议口径（= Book2 ch8 五件套，run_family.py 同款）：AdamW(betas 0.9/0.95, wd=0.1) + grad clip 1.0
#           + lr 1e-3 余弦降至 1e-4 + warmup 100 步；train bf16 autocast / eval fp32；
#           训练数据 tokens.bin 顺序连续分块一次通过（第 k 步消费流首部 (k-1)*8192 起 8192+1 token，
#           两臂按步号取同一批——「同数据批序列」由构造保证，且天然断点续跑安全）；
#           eval 固定窗：eval.bin 内固定种子抽 49 个 (16,512) 窗 = 401,408 token（跨臂跨步完全同集）。
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "00-feasibility")))
for _base in (os.path.join(HERE, "..", "..", "Book2-预训练革命", "ch10"),                              # 工作区布局
              os.path.join(HERE, "..", "..", "..", "book2-pretraining-revolution", "code", "ch10")):  # 书仓快照布局
    if os.path.isdir(_base):
        sys.path.insert(0, os.path.abspath(_base))  # Book2 ch10 GPT-2 定版底座
from model import GPT2, GPT2Config              # noqa: E402  Book2 ch10 定版底座（改装对象，import 复用）
from llama_slots import RMSNorm                 # noqa: E402  本册插槽组件库（ch2 换装件）

# ---------------- 固定口径 ----------------
SEED = 20261002          # 全书统一种子（模型初始化；数据由步号决定，无需 RNG）
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book3-ch02")
TOKENS_BIN = os.path.join(REPO_ROOT, "log", "book2-ch08", "tokens.bin")   # 306M token 训练池（sp-8k）
EVAL_BIN = os.path.join(REPO_ROOT, "log", "book2-ch08", "eval.bin")       # 1.2M token eval 池（与训练段互斥）

VOCAB = 8192             # sp-8k 词表
BATCH = 16               # B：批大小
BLOCK = 512              # T：序列长（wpe 查表上限 1024，留外推余量）
TPB = BATCH * BLOCK      # 8192 token/步（与 Book2 ch8 b32×s256 同吞吐口径）
EVAL_BATCHES = 49        # 49×8192 = 401,408 token ≥ 40 万（Book2 ch8 eval 协议，se<0.005）
PEAK_LR = 1e-3           # 峰值学习率
MIN_LR_RATIO = 0.1       # 余弦降到 0.1×peak = 1e-4
WARMUP = 100             # warmup 步数（Book2 五件套口径）
WDECL = 0.1              # weight decay
TAIL = 300               # 末段均值窗口（末 300 步）
N_LAYER, N_EMBD, N_HEAD = 6, 512, 8              # 统一模块档（探针 A/B 定档：23.6M）


def lr_at(step, total):
    """lr 日程（Book2 ch8 同式）：warmup 段线性升，之后余弦降到 min_ratio*peak。"""
    if step <= WARMUP:
        return PEAK_LR * step / WARMUP
    prog = (step - WARMUP) / max(1, total - WARMUP)
    return PEAK_LR * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * prog)))


class TrainStream:
    """训练数据：tokens.bin 流首部按步号顺序切块（一次通过，构造上无重复 epoch）。

    第 k 步（1 起）消费 tokens[(k-1)*8192 : (k-1)*8192+8193]：x=前 8192 reshape (16,512)，y=左移一格。
    两臂按步号取到完全相同的批（同数据批序列）；步号确定性使断点续跑天然安全。
    """

    def __init__(self, path):
        self.arr = np.memmap(path, dtype=np.uint16, mode="r")     # 只读 memmap，306M token 不进内存
        self.max_step = (len(self.arr) - 1) // TPB

    def batch(self, step, device):
        off = (step - 1) * TPB
        span = np.asarray(self.arr[off:off + TPB + 1], dtype=np.int64)       # (8193,)
        x = torch.from_numpy(span[:-1].reshape(BATCH, BLOCK)).to(device)     # (16,512)
        y = torch.from_numpy(span[1:].reshape(BATCH, BLOCK)).to(device)      # (16,512) 目标=左移一格
        return x, y


def build_eval_batches(device):
    """固定 eval 窗集：eval 池内固定种子抽 49 批 (16,512)（跨臂/跨步/跨档完全同集，可比）。

    返回 (ex, ey)，形状 (49,16,512)；构建后常驻显存，eval 时逐批前向。
    """
    ev = np.memmap(EVAL_BIN, dtype=np.uint16, mode="r")
    gen = torch.Generator().manual_seed(SEED)      # 窗位固定（Book2 ch8 同口径）
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
    """固定窗 eval loss：49 批 fp32 前向（无 autocast，报数用全精度），返回 (均值, 批间 se 估计)。

    每批 (16,512) 前向得标量 loss；401,408 token 池化。se = std(批loss)/sqrt(49) 为粗口径
    采样误差估计（Book2 ch8 tok_std=2.50 实测底数下 se≈0.004）。
    """
    model.eval()
    losses = []
    for i in range(EVAL_BATCHES):
        _, loss = model(ex[i], ey[i])              # (16,512) -> 标量
        losses.append(loss.item())
    model.train()
    mean = sum(losses) / len(losses)
    se = (sum((l - mean) ** 2 for l in losses) / (len(losses) - 1)) ** 0.5 / math.sqrt(len(losses))
    return mean, se


def build_arm(arm):
    """构造两臂模型：同种子先建 GPT-2 底座，rmsnorm 臂再做插槽换装。

    底座构造消耗的 RNG 抽样序列两臂完全一致（LayerNorm/RMSNorm 的 ones/zeros 初始化不耗 RNG），
    因此 attn/mlp/wte/wpe 初始权重逐位相同——唯一变量 = 归一化零件（单插槽改装原则）。
    换装发生在初始化之后：RMSNorm.weight 保持全 1（与 LayerNorm 初始 weight=1/bias=0 对齐）。
    形状口径：两臂前向均 idx (16,512) -> logits (16,512,8192)。
    """
    torch.manual_seed(SEED)
    m = GPT2(GPT2Config(vocab_size=VOCAB, n_positions=1024,
                        n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD))
    if arm == "base":
        pass                                        # GPT-2 底座原样（pre-LN + nn.LayerNorm）
    elif arm == "rmsnorm":
        for blk in m.blocks:                        # 6×2 处块内归一化：pre 位置不变，只换零件
            blk.ln_1 = RMSNorm(N_EMBD, eps=1e-5)    # (16,512,512) -> (16,512,512)，无 bias、eps=1e-5
            blk.ln_2 = RMSNorm(N_EMBD, eps=1e-5)
        m.ln_f = RMSNorm(N_EMBD, eps=1e-5)          # 末尾 ln_f 同步换装（共 13 处）
    else:
        raise ValueError(arm)
    return m


def run_arm(arm, steps, eval_every, device, stream, ex, ey, report, out_json):
    """单臂训练（含断点续跑）：训练循环 = Book2 五件套；每 eval 边界存 ckpt，臂完成即删。"""
    ckpt_path = os.path.join(OUT_DIR, f"ckpt_{arm}_{report['meta']['out_name']}.pt")
    m = build_arm(arm).to(device)
    n_total = m.n_params()
    n_nonemb = m.n_params(non_embedding=True)
    opt = torch.optim.AdamW(m.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=WDECL)

    train_losses, lrs, times, evals = [], [], [], []
    start_step, elapsed_train, elapsed_eval = 0, 0.0, 0.0
    resumed_from = None
    if os.path.exists(ckpt_path):   # ---- 断点续跑：从最近 eval 边界恢复 ----
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        m.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        train_losses, lrs, times = ck["train_losses"], ck["lrs"], ck["times"]
        evals = ck["evals"]
        start_step, elapsed_train, elapsed_eval = ck["step"], ck["elapsed_train"], ck["elapsed_eval"]
        resumed_from = start_step
        print(f"  [{arm} 续跑] 自 step {start_step + 1}（已完成 {start_step} 步，eval {len(evals)} 次）", flush=True)

    m.train()
    base_alloc = torch.mps.current_allocated_memory() / 2**20 if device.type == "mps" else 0.0
    peak = base_alloc
    diverged_at, reason = None, None
    t_arm = time.perf_counter()
    for step in range(start_step + 1, steps + 1):
        lr = lr_at(step, steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = stream.batch(step, device)           # (16,512)×2（按步号定位，两臂同批）
        t0 = time.perf_counter()
        with torch.autocast(device_type="mps", dtype=torch.bfloat16) if device.type == "mps" \
                else torch.autocast("cpu", enabled=False):
            _, loss = m(x, y)                       # (16,512) -> 标量 train loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)   # 五件套：梯度裁剪 1.0
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
        if not math.isfinite(lv) or lv > 100.0:     # 发散守门（Book2 ch8 同款）
            diverged_at, reason = step, ("non-finite" if not math.isfinite(lv) else f"loss={lv:.1f}>100")
            print(f"  [{arm} 发散] step {step}: {reason}", flush=True)
            break
        if step % 100 == 0 or step == 1:
            print(f"  [{arm:8s}] step {step:5d}/{steps} train {lv:.4f} lr {lr:.2e} "
                  f"({elapsed_train/(step - start_step + 1):.3f}s/step)", flush=True)
        if step % eval_every == 0 or step == steps:   # ---- eval 边界：报数 + 存 ckpt ----
            t0 = time.perf_counter()
            el, se = eval_loss(m, ex, ey)
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed_eval += time.perf_counter() - t0
            evals.append({"step": step, "eval_loss": round(el, 5), "eval_se": round(se, 5),
                          "n_tokens": EVAL_BATCHES * TPB})
            print(f"  [{arm:8s}] [eval] step {step}: eval loss {el:.4f} ± {se:.4f}"
                  f"（401,408 token，fp32）", flush=True)
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": step,
                        "train_losses": train_losses, "lrs": lrs, "times": times, "evals": evals,
                        "elapsed_train": elapsed_train, "elapsed_eval": elapsed_eval}, ckpt_path)

    steps_done = len(train_losses)
    steady = times[20:] if len(times) > 20 else times    # 丢前 20 步（编译/缓存预热）
    tail = train_losses[-min(TAIL, steps_done):]
    rec = {
        "arm": arm,
        "arm_def": ("GPT-2 底座（Book2 ch10 model.py，pre-LN + nn.LayerNorm eps=1e-5）" if arm == "base"
                    else "13 处 LayerNorm -> llama_slots.RMSNorm（pre 位置不变/无 bias/eps=1e-5）"),
        "config": {"d": N_EMBD, "L": N_LAYER, "h": N_HEAD, "V": VOCAB, "B": BATCH, "T": BLOCK},
        "params_total": n_total, "params_non_embedding": n_nonemb,
        "steps_done": steps_done, "steps_planned": steps, "resumed_from_step": resumed_from,
        "tail300_train_loss": round(sum(tail) / len(tail), 5),
        "tail300_def": f"最后 {min(TAIL, steps_done)} 步 train loss 均值（bf16 autocast 训练损失）",
        "last_step_loss": round(train_losses[-1], 5) if train_losses else None,
        "min_train_loss": round(min(train_losses), 5) if train_losses else None,
        "evals": evals,
        "eval_final": evals[-1]["eval_loss"] if evals else None,
        "eval_final_se": evals[-1]["eval_se"] if evals else None,
        "sec_per_step_steady": round(sum(steady) / len(steady), 4),
        "sec_step_std": round(float(np.std(steady)), 4),
        "sec_per_step_all": round(elapsed_train / max(1, steps_done - start_step), 4),
        "mem_peak_delta_mb": round(peak - base_alloc, 1),
        "wallclock_sec": round(time.perf_counter() - t_arm, 1),
        "eval_overhead_sec": round(elapsed_eval, 1),
        "diverged": diverged_at is not None, "diverged_at_step": diverged_at, "divergence_reason": reason,
        "precision": "train bf16 autocast / eval fp32",
    }
    # 曲线落盘 + 清 ckpt（臂已完成，结果自含）
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
          f"eval_final {rec['eval_final']} | {rec['sec_per_step_steady']}s/step（稳态） | "
          f"曲线 {curve_path}", flush=True)
    del m, opt
    if device.type == "mps":
        torch.mps.empty_cache()
    return rec


def add_pair_summary(report):
    """双臂对照汇总：eval 差 / 末段差 / 速度差（判据预注册的读数字段，判读规则见 preregistered.md）。"""
    a, b = report["arms"].get("base"), report["arms"].get("rmsnorm")
    if not (a and b):
        return
    report["pair_summary"] = {
        "eval_final_diff_base_minus_rmsnorm": round(a["eval_final"] - b["eval_final"], 5)
        if a["eval_final"] is not None and b["eval_final"] is not None else None,
        "tail300_diff_base_minus_rmsnorm": round(a["tail300_train_loss"] - b["tail300_train_loss"], 5),
        "sec_per_step_diff": round(a["sec_per_step_steady"] - b["sec_per_step_steady"], 4),
        "speedup_base_over_rmsnorm": round(a["sec_per_step_steady"] / b["sec_per_step_steady"], 4),
        "note": "正=base 更高；判据：|eval 差|≤0.02 nat 视为等价成立（噪声带口径见 preregistered.md），速度差单独报",
    }


def main():
    ap = argparse.ArgumentParser(description="ch2 模块消融：LayerNorm vs RMSNorm（同数据同种子）")
    ap.add_argument("--steps", type=int, default=1000, help="每臂步数（fast=1000 / full=5000）")
    ap.add_argument("--out-name", type=str, default=None,
                    help="输出名（缺省按步数自动取 fast(≤1500)/full，防覆写正式文件）")
    ap.add_argument("--cells", type=str, default="base,rmsnorm",
                    help="臂选择，逗号分隔（base / rmsnorm；默认双臂全跑）")
    ap.add_argument("--eval-every", type=int, default=500, help="eval 间隔步数（默认 500，Book2 ch8 口径）")
    args = ap.parse_args()
    out_name = args.out_name or ("fast" if args.steps <= 1500 else "full")
    arms = [c.strip() for c in args.cells.split(",")]
    for a in arms:
        if a not in ("base", "rmsnorm"):
            raise SystemExit(f"未知臂 {a}（可选 base / rmsnorm）")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    stream = TrainStream(TOKENS_BIN)
    if args.steps > stream.max_step:
        raise SystemExit(f"步数 {args.steps} 超出一次通过上限 {stream.max_step}（不许重复 epoch）")
    ex, ey = build_eval_batches(device)
    out_json = os.path.join(OUT_DIR, f"ablation_norm_{out_name}.json")

    # ---- 载入已有报告（跨进程续跑：已完成且步数足够的臂直接跳过） ----
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
        "corpus": "log/book2-ch08/tokens.bin（OpenWebText 子采样，sp-8k V=8192，306M token 流）",
        "tokenizer": "sp-8k BPE byte_fallback（Book2 ch8 遗产，分词器与 LM 语料分离）",
        "experiment": "ch2 模块消融：LayerNorm（base）vs RMSNorm（+rmsnorm）——单插槽改装，其余零改动",
        "module_tier": f"d={N_EMBD}/L={N_LAYER}/h={N_HEAD}/V={VOCAB}（统一模块档，{BATCH}×{BLOCK}，8192 tok/步）",
        "optimizer": f"AdamW(betas=0.9,0.95,wd={WDECL}) + grad_clip 1.0（Book2 五件套）",
        "schedule": f"lr {PEAK_LR} 余弦降至 {PEAK_LR*MIN_LR_RATIO:.0e}，warmup={WARMUP}；两臂统一日程",
        "data_protocol": "顺序连续分块一次通过（第 k 步消费流首部 (k-1)*8192 起 token）；两臂按步号取同批",
        "eval_protocol": f"每 {args.eval_every} 步 + 终步：eval.bin 固定 49 窗 {EVAL_BATCHES*TPB:,} token，fp32 前向",
        "precision": "train bf16 autocast / eval fp32",
        "ckpt": "每 eval 边界存 ckpt_{臂}_{out}.pt，臂完成即删；中断后重跑同命令自动续跑",
        "argv": " ".join(sys.argv[1:]),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    t_all = time.perf_counter()
    for arm in arms:
        done = report["arms"].get(arm)
        if done and done["steps_done"] >= args.steps:
            print(f"[跳过] 臂 {arm} 已完成（{done['steps_done']} 步 ≥ {args.steps}）", flush=True)
            continue
        print(f"\n===== 臂 {arm}（{args.steps} 步） =====", flush=True)
        run_arm(arm, args.steps, args.eval_every, device, stream, ex, ey, report, out_json)
    add_pair_summary(report)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[全部完成] 总耗时 {(time.perf_counter()-t_all)/60:.1f} min -> {out_json}", flush=True)
    for a, r in report["arms"].items():
        print(f"  {a:8s} params={r['params_total']:,} tail300={r['tail300_train_loss']} "
              f"eval_final={r['eval_final']} {r['sec_per_step_steady']}s/step", flush=True)


if __name__ == "__main__":
    main()
