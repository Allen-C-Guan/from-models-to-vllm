# ablation_moe.py —— Book4 ch2 消融：dense（d_ff=1408）vs MoE（E8/top2/w=704）等激活基线
# 用途：统一模块档（探针 B 已验配置 d=512/L=6/h=8/h_kv=4/V=8192，B=16/T=512）两臂同数据同种子短训，
#       单变量 = FFN 插槽形态：dense SwiGLU（d_ff=1408）vs 稀疏 MoE（E=8/top-2/每专家 w=704）——
#       等激活口径逐位相同（每 token 激活 2×704=1408=dense 的 d_ff；激活参两臂同为 26,089,984，
#       MoE 总参 65,042,944=2.49×）。「等激活参数下 MoE 是否反超」的 ch2 核心读数，
#       与探针 B 初读（200 步 eval MoE 反超 0.084 nat，单种子趋势级）做方向一致性检查；
#       sec/step 单独报（MoE 派发开销 ≈2.3×，如实标注——质量读数与速度读数分开念）。
# 协议口径（Book2 五件套，承 Book3 ch2-6 消融同款）：AdamW(0.9/0.95, wd=0.1) + clip 1.0
#       + lr 1e-3 余弦降至 1e-4 + warmup 100；train bf16 autocast / eval fp32；
#       tokens.bin 顺序分块一次通过（两臂按步号取同批）；eval 固定 49 窗 (16,512)=401,408 token。
# 运行方式：source env.sh && python ablation_moe.py [--steps 2000] [--out-name fast|full] [--arms dense,moe]
#           两档位：fast=200 步 / full=2000 步（晚间档）。
# 产物（log/book4-ch02/，不入库）：ablation_moe_{out}.json + curve_{arm}_{out}.csv + ckpt_{arm}_{out}.pt
#           （ckpt 每 eval 边界落盘、臂完成即删；中断后重跑同命令自动续跑）。
# 模块来源：不私搭模型——整机/手算账 import 自正身 moe_mla_slots（其内部 bootstrap 引 Book3 llama_slots）。
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
from moe_mla_slots import (SEED, SlotConfig, SlotLLaMA, TokenStream, activated_account,   # noqa: E402
                           hand_account)

# ---------------- 固定口径 ----------------
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book4-ch02")
TOKENS_BIN = os.path.join(REPO_ROOT, "log", "book2-ch08", "tokens.bin")
EVAL_BIN = os.path.join(REPO_ROOT, "log", "book2-ch08", "eval.bin")

VOCAB, D_MODEL, N_LAYER, N_HEAD, N_KV = 8192, 512, 6, 8, 4        # 探针 B 已验模块档
D_DENSE, N_EXPERT, TOP_K, D_EXPERT = 1408, 8, 2, 704              # 等激活：2×704=1408=d_ff
BATCH, BLOCK = 16, 512
TPB = BATCH * BLOCK
EVAL_BATCHES = 49                                                  # 固定 49 窗 = 401,408 token
PEAK_LR, MIN_LR_RATIO, WARMUP, WDECL, TAIL = 1e-3, 0.1, 100, 0.1, 300
DENSE_PARAMS = 26_089_984                                          # 探针 B 逐位锚点（等激活口径）
MOE_PARAMS = 65_042_944
ARMS = ("dense", "moe")


def lr_at(step, total):
    """Book2 五件套日程：线性 warmup 100 步 + 余弦 1e-3 → 1e-4（两臂统一）。"""
    if step <= WARMUP:
        return PEAK_LR * step / WARMUP
    prog = (step - WARMUP) / max(1, total - WARMUP)
    return PEAK_LR * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * prog)))


def build_eval_batches(device):
    """固定 eval 窗集：eval 池固定种子抽 49 批 (16,512)（跨臂/跨步完全同集）。"""
    ev = np.memmap(EVAL_BIN, dtype=np.uint16, mode="r")
    gen = torch.Generator().manual_seed(SEED)
    offs = torch.randint(0, len(ev) - BLOCK - 1, (EVAL_BATCHES * BATCH,), generator=gen).numpy()
    xs, ys = [], []
    for o in offs:
        xs.append(np.asarray(ev[o:o + BLOCK], dtype=np.int64))
        ys.append(np.asarray(ev[o + 1:o + BLOCK + 1], dtype=np.int64))
    x = torch.from_numpy(np.stack(xs).reshape(EVAL_BATCHES, BATCH, BLOCK)).to(device)   # (49,16,512)
    y = torch.from_numpy(np.stack(ys).reshape(EVAL_BATCHES, BATCH, BLOCK)).to(device)
    return x, y


@torch.no_grad()
def eval_loss(model, ex, ey):
    """49 窗 fp32 eval（无 autocast——权重本就 fp32）；返回 (均值, 标准误)。"""
    model.eval()
    losses = []
    for i in range(EVAL_BATCHES):
        _, loss = model(ex[i], ey[i])               # (16,512) -> 标量
        losses.append(loss.item())
    model.train()
    mean = sum(losses) / len(losses)
    se = (sum((l - mean) ** 2 for l in losses) / (len(losses) - 1)) ** 0.5 / math.sqrt(len(losses))
    return mean, se


def build_arm(arm):
    """构造臂：同种子建整机（SlotLLaMA 正身），dense=moe 插槽留空 / moe=teach 版 SparseMoE。"""
    common = dict(vocab_size=VOCAB, hidden_size=D_MODEL, num_hidden_layers=N_LAYER,
                  num_attention_heads=N_HEAD, num_key_value_heads=N_KV, intermediate_size=D_DENSE)
    if arm == "dense":
        cfg = SlotConfig(**common, moe=None)
    else:
        cfg = SlotConfig(**common, moe=dict(n_expert=N_EXPERT, top_k=TOP_K, expert_dim=D_EXPERT,
                                            n_shared=0, variant="teach", mode="mixtral"))
    torch.manual_seed(SEED)
    m = SlotLLaMA(cfg)
    # ---- 参数账 assert（手算 = 实测逐位；等激活口径锚点）----
    hand = hand_account(cfg)
    assert m.n_params() == hand["total"], f"{arm} 总参对账失败 {m.n_params()} != {hand['total']}"
    act, act_ne = activated_account(cfg)
    assert act == DENSE_PARAMS, f"{arm} 激活参对账失败 {act} != {DENSE_PARAMS}（等激活口径被破坏）"
    expect_total = DENSE_PARAMS if arm == "dense" else MOE_PARAMS
    assert m.n_params() == expect_total, f"{arm} 总参 {m.n_params()} != 探针 B 锚点 {expect_total}"
    return m, cfg


def run_arm(arm, steps, eval_every, device, stream, ex, ey, report, out_json):
    """一个臂的五件套训练环：ckpt 断点续跑（eval 边界存档），发散守卫，曲线 CSV。"""
    ckpt_path = os.path.join(OUT_DIR, f"ckpt_{arm}_{report['meta']['out_name']}.pt")
    m, cfg = build_arm(arm)
    m = m.to(device)
    n_total = m.n_params()
    act, act_ne = activated_account(cfg)
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
        x, y = stream.batch(step, device)                    # (16,512)×2（两臂同批）
        t0 = time.perf_counter()
        with torch.autocast(device_type="mps", dtype=torch.bfloat16) if device.type == "mps" \
                else torch.autocast("cpu", enabled=False):
            _, loss = m(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if device.type == "mps":
            torch.mps.synchronize()                          # MPS 异步——同步后才是真实每步墙钟
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
            print(f"  [{arm:5s}] step {step:5d}/{steps} train {lv:.4f} lr {lr:.2e} "
                  f"({elapsed_train / (step - start_step):.3f}s/step)", flush=True)
        if step % eval_every == 0 or step == steps:
            t0 = time.perf_counter()
            el, se = eval_loss(m, ex, ey)
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed_eval += time.perf_counter() - t0
            evals.append({"step": step, "eval_loss": round(el, 5), "eval_se": round(se, 5),
                          "n_tokens": EVAL_BATCHES * TPB})
            print(f"  [{arm:5s}] [eval] step {step}: eval loss {el:.4f} ± {se:.4f}", flush=True)
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": step,
                        "train_losses": train_losses, "lrs": lrs, "times": times, "evals": evals,
                        "elapsed_train": elapsed_train, "elapsed_eval": elapsed_eval}, ckpt_path)

    steps_done = len(train_losses)
    steady = times[20:] if len(times) > 20 else times          # 丢前 20 步（惰性编译/缓存预热）
    tail = train_losses[-min(TAIL, steps_done):]
    rec = {
        "arm": arm,
        "arm_def": ("dense SwiGLU（d_ff=1408）" if arm == "dense" else
                    f"稀疏 MoE（E={N_EXPERT}/top-{TOP_K}/w={D_EXPERT}，等激活 2×{D_EXPERT}={2*D_EXPERT}）"),
        "config": {"d": D_MODEL, "L": N_LAYER, "h": N_HEAD, "h_kv": N_KV, "V": VOCAB,
                   "B": BATCH, "T": BLOCK, "d_ff_dense": D_DENSE,
                   **({} if arm == "dense" else {"n_expert": N_EXPERT, "top_k": TOP_K,
                                                 "expert_dim": D_EXPERT})},
        "params_total": n_total, "params_active": act, "params_active_nonemb": act_ne,
        "params_total_ratio_vs_dense": round(n_total / DENSE_PARAMS, 3),
        "steps_done": steps_done, "steps_planned": steps, "resumed_from_step": resumed_from,
        "tail300_train_loss": round(sum(tail) / len(tail), 5),
        "last_step_loss": round(train_losses[-1], 5) if train_losses else None,
        "min_train_loss": round(min(train_losses), 5) if train_losses else None,
        "evals": evals,
        "eval_final": evals[-1]["eval_loss"] if evals else None,
        "eval_final_se": evals[-1]["eval_se"] if evals else None,
        "sec_per_step_steady": round(sum(steady) / len(steady), 4),
        "sec_per_step_median": round(float(np.median(steady)), 4),
        "sec_step_std": round(float(np.std(steady)), 4),
        "mem_peak_delta_mb": round(peak - base_alloc, 1),
        "elapsed_train_sec": round(elapsed_train, 1), "elapsed_eval_sec": round(elapsed_eval, 1),
        "wallclock_sec": round(time.perf_counter() - t_arm, 1),
        "diverged": diverged_at is not None, "diverged_at_step": diverged_at,
        "divergence_reason": reason,
        "precision": "train bf16 autocast / eval fp32",
    }
    curve_path = os.path.join(OUT_DIR, f"curve_{arm}_{report['meta']['out_name']}.csv")
    with open(curve_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["step", "train_loss", "lr"])
        for i, (lv, lr) in enumerate(zip(train_losses, lrs), 1):
            w.writerow([i, f"{lv:.6g}", f"{lr:.6g}"])
    if steps_done >= steps and os.path.exists(ckpt_path):
        os.remove(ckpt_path)                                  # 臂完成即删（Book3 消融同款）
    report["arms"][arm] = rec
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  [{arm} 完成] steps {steps_done} | tail300 {rec['tail300_train_loss']} | "
          f"eval_final {rec['eval_final']} | {rec['sec_per_step_steady']}s/step", flush=True)
    del m, opt
    if device.type == "mps":
        torch.mps.empty_cache()
    return rec


def add_pair_summary(report):
    """dense vs moe 两读数分开念：质量（eval 差）与速度（派发开销倍数）。"""
    arms = report["arms"]
    a, b = arms.get("dense"), arms.get("moe")
    if not (a and b and a["eval_final"] is not None and b["eval_final"] is not None):
        return
    report["pair_summary"] = {
        "eval_final_diff_dense_minus_moe": round(a["eval_final"] - b["eval_final"], 5),
        "eval_diff_read": "正 = MoE 更低（反超 dense）；负 = dense 更低——判据预注册见 notes/06",
        "sec_per_step_ratio_moe_over_dense": round(b["sec_per_step_steady"] / a["sec_per_step_steady"], 2),
        "sec_per_step_diff": round(b["sec_per_step_steady"] - a["sec_per_step_steady"], 4),
        "probeB_reference": {"eval_diff": 0.084, "sec_ratio": 2.32,
                             "note": "探针 B run2：200 步 eval 6.465 vs 6.549 / 0.434 vs 0.187 s/step"
                                     "（B=8/T=512 口径；本实验 B=16/T=512）"},
    }


def main():
    ap = argparse.ArgumentParser(description="ch2 消融：dense vs MoE 等激活基线（同数据同种子）")
    ap.add_argument("--steps", type=int, default=2000, help="每臂步数（fast=200 / full=2000）")
    ap.add_argument("--out-name", type=str, default=None)
    ap.add_argument("--arms", type=str, default="dense,moe", help="臂选择（dense/moe，逗号分隔）")
    ap.add_argument("--eval-every", type=int, default=None, help="eval 间隔（默认 max(100, steps//4)）")
    args = ap.parse_args()
    out_name = args.out_name or ("fast" if args.steps <= 500 else "full")
    eval_every = args.eval_every or max(100, args.steps // 4)
    arms = [a.strip() for a in args.arms.split(",")]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"未知臂 {a}（可选 {'/'.join(ARMS)}）")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    stream = TokenStream(TOKENS_BIN, BATCH, BLOCK)
    if args.steps > stream.max_step:
        raise SystemExit(f"步数 {args.steps} 超出一次通过上限 {stream.max_step}")
    ex, ey = build_eval_batches(device)
    out_json = os.path.join(OUT_DIR, f"ablation_moe_{out_name}.json")

    if os.path.exists(out_json):
        with open(out_json, encoding="utf-8") as f:
            report = json.load(f)
        report.setdefault("arms", {})          # 保留已完成的未请求臂（防 --arms 子集覆写丢记录）
        print(f"[续跑] 已有报告载入：臂 {sorted(report['arms'])}", flush=True)
    else:
        report = {"arms": {}}
    report["meta"] = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "device": str(device),
        "torch": torch.__version__, "out_name": out_name,
        "experiment": "ch2 等激活消融：dense(d_ff=1408) vs MoE(E8/top2/w=704)——单变量 = FFN 插槽形态",
        "module_tier": f"d={D_MODEL}/L={N_LAYER}/h={N_HEAD}/h_kv={N_KV}/V={VOCAB}，B={BATCH}/T={BLOCK}"
                       f"（{TPB:,} tok/步，探针 B 已验配置）",
        "equal_activation": f"两臂激活参同为 {DENSE_PARAMS:,}（每 token 2×{D_EXPERT}={2*D_EXPERT}"
                            f"={D_DENSE}=dense d_ff）；MoE 总参 {MOE_PARAMS:,}",
        "corpus": "log/book2-ch08/tokens.bin（sp-8k，306M token 流）",
        "optimizer": f"AdamW(betas=0.9,0.95,wd={WDECL}) + grad_clip 1.0（Book2 五件套）",
        "schedule": f"lr {PEAK_LR} 余弦降至 {PEAK_LR*MIN_LR_RATIO:.0e}，warmup={WARMUP}；两臂统一日程",
        "data_protocol": "顺序连续分块一次通过（moe_mla_slots.TokenStream）；两臂按步号取同批",
        "eval_protocol": f"每 {eval_every} 步+终步：eval.bin 固定 49 窗 {EVAL_BATCHES*TPB:,} token，fp32",
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
        print(f"\n===== 臂 {arm}（{args.steps} 步，eval_every={eval_every}） =====", flush=True)
        run_arm(arm, args.steps, eval_every, device, stream, ex, ey, report, out_json)
    add_pair_summary(report)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[全部完成] 总耗时 {(time.perf_counter()-t_all)/60:.1f} min -> {out_json}", flush=True)
    for a, r in report["arms"].items():
        print(f"  {a:5s} params={r['params_total']:,} active={r['params_active']:,} "
              f"eval_final={r['eval_final']} {r['sec_per_step_steady']}s/step", flush=True)


if __name__ == "__main__":
    main()
