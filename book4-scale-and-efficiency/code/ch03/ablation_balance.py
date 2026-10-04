# ablation_balance.py —— Book4 ch3 均衡干预消融：none vs Shazeer 式 load-loss 两臂（负载崩塌现场 + 干预）
# 用途：统一模块档（探针 B 已验配置 d=512/L=6/h=8/h_kv=4/V=8192，B=16/T=512）两臂同数据同种子短训，
#       单变量 = 负载均衡干预：none 臂（无辅助损失，预期复现赢者通吃/负载崩塌——探针 B 现场为参照：
#       200 步训后第 0 层 max/min=327.8、多轮出现死专家）vs aux 臂（Shazeer 式 load-loss
#       L_load = w·CV(Load)²，w=0.1——papers/01 §A.4 R3 正身公式、LM 档权重口径）。
#       产出每层每专家路由计数直方图 CSV（训前 init + 训中各 eval 边界——ch3 正文第一手现场素材）
#       与均衡度量（max/min、CV、MaxVio——noaux 论文 R35 的 MaxVio 口径）。
# 正身改编说明（诚实条款）：Shazeer 的 Load 用噪声门控下的 Φ 概率代理（Eq 8-11）；本实验门为确定性
#       softmax top-k（moe_mla_slots.TopkGate，无 W_noise 项），noise→0 时 Φ 代理退化为硬计数不可导，
#       故 aux 臂的 Load 用 softmax 概率质量 Σ_x P(x,i) 作平滑代理（可导、方向一致）；
#       度量环节一律用硬计数（离散路由的真实现场），损失环节用概率代理——两者分开念。
# 协议口径（Book2 五件套，与 ch02/ablation_moe.py 同款）：AdamW(0.9/0.95, wd=0.1) + clip 1.0
#       + lr 1e-3 余弦降至 1e-4 + warmup 100；train bf16 autocast / eval fp32；
#       tokens.bin 顺序分块一次通过（两臂按步号取同批）；eval 固定 49 窗 (16,512)=401,408 token。
# 运行方式：source env.sh && python ablation_balance.py [--steps 2000] [--out-name fast|full] [--arms none,aux]
#           两档位：fast=300 步 / full=2000 步（晚间档）。
# 产物（log/book4-ch03/，不入库）：ablation_balance_{out}.json + curve_{arm}_{out}.csv
#       + routing_{arm}_{out}.csv（长表：phase/step/layer/expert/count/frac）+ ckpt_{arm}_{out}.pt
#       （ckpt 每 eval 边界落盘、臂完成即删；中断后重跑同命令自动续跑）。
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
OUT_DIR = os.path.join(REPO_ROOT, "log", "book4-ch03")
TOKENS_BIN = os.path.join(REPO_ROOT, "log", "book2-ch08", "tokens.bin")
EVAL_BIN = os.path.join(REPO_ROOT, "log", "book2-ch08", "eval.bin")

VOCAB, D_MODEL, N_LAYER, N_HEAD, N_KV = 8192, 512, 6, 8, 4        # 探针 B 已验模块档
N_EXPERT, TOP_K, D_EXPERT = 8, 2, 704                             # MoE 臂 = ch02 moe 臂同配置
AUX_W = 0.1                                                       # w_load（R3 LM 档口径）
BATCH, BLOCK = 16, 512
TPB = BATCH * BLOCK
EVAL_BATCHES = 49                                                  # 固定 49 窗 = 401,408 token
PROBE_B, PROBE_T = 4, 2048                                         # 路由探针：8192 token（eval.bin 首段）
PEAK_LR, MIN_LR_RATIO, WARMUP, WDECL, TAIL = 1e-3, 0.1, 100, 0.1, 300
MOE_PARAMS = 65_042_944                                            # 探针 B 逐位锚点
ARMS = ("none", "aux")


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
    """两臂同为 MoE 整机（E8/top2/w=704，= ch02 moe 臂）；单变量 = 是否加 load-loss。"""
    cfg = SlotConfig(vocab_size=VOCAB, hidden_size=D_MODEL, num_hidden_layers=N_LAYER,
                     num_attention_heads=N_HEAD, num_key_value_heads=N_KV, intermediate_size=1408,
                     moe=dict(n_expert=N_EXPERT, top_k=TOP_K, expert_dim=D_EXPERT,
                              n_shared=0, variant="teach", mode="mixtral"))
    torch.manual_seed(SEED)
    m = SlotLLaMA(cfg)
    hand = hand_account(cfg)
    assert m.n_params() == hand["total"] == MOE_PARAMS, \
        f"参数对账失败 {m.n_params()} != {MOE_PARAMS}（探针 B 锚点）"
    act, _ = activated_account(cfg)
    assert act == 26_089_984, f"等激活口径被破坏：{act}"
    return m, cfg


def attach_gate_hooks(model, stash):
    """在每层 router（TopkGate）上挂前向钩子，捕 router logits (N,E)——aux 臂的损失接口。

    不改 moe_mla_slots 正身一行：干预从外面包夹（教学的「加一项损失」最小侵入形态）。
    """
    hooks = []

    def hook(mod, inp, out):
        stash.append(out[0])                        # router logits (N,E)，带梯度图

    for layer in model.model.layers:
        hooks.append(layer.mlp.gate.register_forward_hook(hook))
    return hooks


def load_cv2_terms(stash):
    """L_load 正身（改编版）：每层 Load_i = Σ_x P(x,i)（softmax 概率质量），CV=std/mean，
    返回各层 CV² 列表（fp32；梯度经 logits 流回 gate.weight）。"""
    terms = []
    for logits in stash:
        probs = logits.float().softmax(dim=-1)      # (N,E)
        load = probs.sum(dim=0)                     # (E,)  Σ_i Load_i = N
        cv = load.std(unbiased=False) / load.mean()
        terms.append(cv ** 2)
    return terms


# ---------------- 路由探针：每层每专家硬计数直方图 + 均衡度量 ----------------
def build_probe_batch(device):
    """固定路由探针：eval.bin 首段 8192 token (4,2048)——跨臂/跨快照完全同批（held-out）。"""
    ev = np.memmap(EVAL_BIN, dtype=np.uint16, mode="r")
    span = np.asarray(ev[:PROBE_B * PROBE_T], dtype=np.int64)
    return torch.from_numpy(span.reshape(PROBE_B, PROBE_T)).to(device)     # (4,2048)


@torch.no_grad()
def routing_snapshot(model, probe_x, device):
    """喂探针 token，返回 L×E 硬计数表 + 逐层/全局度量（max/min、CV、MaxVio、死专家数）。

    度量口径（计数为离散路由的真实现场）：
      frac_li = count_li / (N_tok·k)（每层选择总数归一，均匀 = 1/E）
      max_over_min = max(count_l) / max(min(count_l), 1)   ← 死专家（count=0）时下限取 1，防 ∞
      CV_l = std(count_l)/mean(count_l)（总体标准差）
      MaxVio_l = max_i |frac_li − 1/E|（noaux R35 的 MaxVio 口径的逐层版）
      MaxVio_global = 跨层聚合每专家计数后再取 max_i |frac_i − 1/E|
    """
    model.eval()
    stash = []
    hooks = attach_gate_hooks(model, stash)
    with torch.autocast(device_type="mps", dtype=torch.bfloat16) if device.type == "mps" \
            else torch.autocast("cpu", enabled=False):
        model(probe_x)
    for hk in hooks:
        hk.remove()
    counts = []
    for logits in stash:                            # (N,E) —— 与训练同语义：softmax→top-k
        _, idx = logits.float().softmax(dim=-1).topk(TOP_K, dim=-1)
        counts.append(torch.bincount(idx.reshape(-1), minlength=N_EXPERT).tolist())
    model.train()
    n_sel = PROBE_B * PROBE_T * TOP_K               # 每层总选择数 = 16384
    metrics, agg = [], [0] * N_EXPERT
    for row in counts:
        arr = np.array(row, dtype=float)
        frac = arr / n_sel
        for i, c in enumerate(row):
            agg[i] += c
        metrics.append({"counts": row,
                        "frac": [round(float(f), 5) for f in frac],
                        "max_over_min": round(float(arr.max() / max(arr.min(), 1.0)), 2),
                        "cv": round(float(arr.std() / arr.mean()), 4),
                        "max_vio": round(float(np.abs(frac - 1.0 / N_EXPERT).max()), 5),
                        "n_zero_experts": int((arr == 0).sum())})
    agg_arr = np.array(agg, dtype=float)            # 跨层聚合（L 层 × 每专家）
    agg_frac = agg_arr / (n_sel * N_LAYER)
    return {"counts": counts,
            "per_layer": metrics,
            "global": {"counts": agg,
                       "frac": [round(float(f), 5) for f in agg_frac],
                       "cv": round(float(agg_arr.std() / agg_arr.mean()), 4),
                       "max_vio": round(float(np.abs(agg_frac - 1.0 / N_EXPERT).max()), 5),
                       "n_zero_cells": int(sum(m["n_zero_experts"] for m in metrics))},
            "layer_mean_max_over_min": round(float(np.mean([m["max_over_min"] for m in metrics])), 2),
            "layer_mean_cv": round(float(np.mean([m["cv"] for m in metrics])), 4)}


def append_routing_csv(arm, out_name, phase, step, snap):
    """长表追加写：phase/step/layer/expert/count/frac（ch3 正文直方图素材的标准形态）。"""
    path = os.path.join(OUT_DIR, f"routing_{arm}_{out_name}.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        if new:
            w.writerow(["phase", "step", "layer", "expert", "count", "frac"])
        for li, m in enumerate(snap["per_layer"]):
            for ei, (c, f) in enumerate(zip(m["counts"], m["frac"])):
                w.writerow([phase, step, li, ei, c, f])


def run_arm(arm, steps, eval_every, device, stream, ex, ey, probe_x, report, out_json):
    """一个臂的五件套训练环：aux 臂在 CE 外加 L_load=w·CV(Load)²（层均值），断点续跑，快照直方图。"""
    ckpt_path = os.path.join(OUT_DIR, f"ckpt_{arm}_{report['meta']['out_name']}.pt")
    m, cfg = build_arm(arm)
    m = m.to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=WDECL)
    stash = []
    hooks = attach_gate_hooks(m, stash) if arm == "aux" else []

    train_losses, aux_losses, lrs, times, evals, snaps = [], [], [], [], [], []
    start_step, elapsed_train, elapsed_eval = 0, 0.0, 0.0
    resumed_from = None
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        m.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        train_losses, aux_losses = ck["train_losses"], ck["aux_losses"]
        lrs, times = ck["lrs"], ck["times"]
        evals, snaps = ck["evals"], ck["snaps"]
        start_step, elapsed_train, elapsed_eval = ck["step"], ck["elapsed_train"], ck["elapsed_eval"]
        resumed_from = start_step
        print(f"  [{arm} 续跑] 自 step {start_step + 1}", flush=True)
    else:
        snap0 = routing_snapshot(m, probe_x, device)             # 训前（随机初始化）负载直方图
        snaps.append({"step": 0, "phase": "init", **snap0})
        append_routing_csv(arm, report["meta"]["out_name"], "init", 0, snap0)
        print(f"  [{arm}] init 快照：max/min 均值 {snap0['layer_mean_max_over_min']} | "
              f"CV 均值 {snap0['layer_mean_cv']} | MaxVio_global {snap0['global']['max_vio']}", flush=True)

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
        stash.clear()
        with torch.autocast(device_type="mps", dtype=torch.bfloat16) if device.type == "mps" \
                else torch.autocast("cpu", enabled=False):
            _, ce = m(x, y)
        if arm == "aux":
            aux = AUX_W * torch.stack(load_cv2_terms(stash)).mean()   # L_load = w·层均值 CV²
            loss = ce + aux
        else:
            aux = None
            loss = ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if device.type == "mps":
            torch.mps.synchronize()                          # MPS 异步——同步后才是真实每步墙钟
        times.append(time.perf_counter() - t0)
        elapsed_train += times[-1]
        lv = ce.item()
        train_losses.append(lv)
        aux_losses.append(round(aux.item(), 6) if aux is not None else 0.0)
        lrs.append(lr)
        if device.type == "mps":
            peak = max(peak, torch.mps.current_allocated_memory() / 2**20)
        if not math.isfinite(lv) or lv > 100.0:
            diverged_at, reason = step, ("non-finite" if not math.isfinite(lv) else f"loss={lv:.1f}>100")
            print(f"  [{arm} 发散] step {step}: {reason}", flush=True)
            break
        if step % 100 == 0 or step == 1:
            msg = f"  [{arm:4s}] step {step:5d}/{steps} train {lv:.4f}"
            if arm == "aux":
                msg += f" aux {aux_losses[-1]:.5f}"
            print(msg + f" lr {lr:.2e} ({elapsed_train / (step - start_step):.3f}s/step)", flush=True)
        if step % eval_every == 0 or step == steps:
            t0 = time.perf_counter()
            el, se = eval_loss(m, ex, ey)
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed_eval += time.perf_counter() - t0
            evals.append({"step": step, "eval_loss": round(el, 5), "eval_se": round(se, 5),
                          "n_tokens": EVAL_BATCHES * TPB})
            snap = routing_snapshot(m, probe_x, device)      # 训中快照：直方图随训练漂移的现场
            snaps.append({"step": step, "phase": "train", **snap})
            append_routing_csv(arm, report["meta"]["out_name"], "train", step, snap)
            print(f"  [{arm:4s}] [eval] step {step}: eval {el:.4f} ± {se:.4f} | "
                  f"max/min 均值 {snap['layer_mean_max_over_min']} | MaxVio_global "
                  f"{snap['global']['max_vio']} | 死专家 {snap['global']['n_zero_cells']}/{N_LAYER*N_EXPERT}",
                  flush=True)
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": step,
                        "train_losses": train_losses, "aux_losses": aux_losses, "lrs": lrs,
                        "times": times, "evals": evals, "snaps": snaps,
                        "elapsed_train": elapsed_train, "elapsed_eval": elapsed_eval}, ckpt_path)

    for hk in hooks:
        hk.remove()
    steps_done = len(train_losses)
    steady = times[20:] if len(times) > 20 else times          # 丢前 20 步（惰性编译/缓存预热）
    tail = train_losses[-min(TAIL, steps_done):]
    rec = {
        "arm": arm,
        "arm_def": ("none：纯 CE 训练（无均衡干预——赢者通吃现场）" if arm == "none" else
                    f"aux：CE + {AUX_W}·层均值 CV(Load)²（Shazeer R3 正身公式，Load=softmax 概率质量代理）"),
        "config": {"d": D_MODEL, "L": N_LAYER, "h": N_HEAD, "h_kv": N_KV, "V": VOCAB,
                   "B": BATCH, "T": BLOCK, "n_expert": N_EXPERT, "top_k": TOP_K,
                   "expert_dim": D_EXPERT, "aux_w": AUX_W if arm == "aux" else 0.0},
        "params_total": m.n_params(), "params_active": 26_089_984,
        "steps_done": steps_done, "steps_planned": steps, "resumed_from_step": resumed_from,
        "tail300_train_loss": round(sum(tail) / len(tail), 5),
        "evals": evals,
        "eval_final": evals[-1]["eval_loss"] if evals else None,
        "eval_final_se": evals[-1]["eval_se"] if evals else None,
        "aux_loss_first10_mean": (round(sum(aux_losses[:10]) / 10, 6) if aux_losses else None),
        "aux_loss_last10_mean": (round(sum(aux_losses[-10:]) / 10, 6) if aux_losses else None),
        "snapshots": snaps,
        "balance_init": snaps[0]["global"] if snaps else None,
        "balance_final": snaps[-1]["global"] if snaps else None,
        "layer_mean_max_over_min": {"init": snaps[0]["layer_mean_max_over_min"] if snaps else None,
                                    "final": snaps[-1]["layer_mean_max_over_min"] if snaps else None},
        "layer_mean_cv": {"init": snaps[0]["layer_mean_cv"] if snaps else None,
                          "final": snaps[-1]["layer_mean_cv"] if snaps else None},
        "sec_per_step_steady": round(sum(steady) / len(steady), 4),
        "sec_per_step_median": round(float(np.median(steady)), 4),
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
        w.writerow(["step", "train_loss", "aux_loss", "lr"])
        for i, (lv, al, lr) in enumerate(zip(train_losses, aux_losses, lrs), 1):
            w.writerow([i, f"{lv:.6g}", f"{al:.6g}", f"{lr:.6g}"])
    if steps_done >= steps and os.path.exists(ckpt_path):
        os.remove(ckpt_path)                                  # 臂完成即删（Book3 消融同款）
    report["arms"][arm] = rec
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  [{arm} 完成] steps {steps_done} | eval_final {rec['eval_final']} | max/min 均值 "
          f"{rec['layer_mean_max_over_min']['init']}→{rec['layer_mean_max_over_min']['final']} | "
          f"{rec['sec_per_step_steady']}s/step", flush=True)
    del m, opt
    if device.type == "mps":
        torch.mps.empty_cache()
    return rec


def add_pair_summary(report):
    """none vs aux 的干预读数：max/min 与 MaxVio 的下降 + eval 差（判据预注册见 notes/06）。"""
    arms = report["arms"]
    a, b = arms.get("none"), arms.get("aux")
    if not (a and b and a["eval_final"] is not None and b["eval_final"] is not None):
        return
    report["pair_summary"] = {
        "eval_final_diff_none_minus_aux": round(a["eval_final"] - b["eval_final"], 5),
        "eval_diff_read": "正 = aux 更低（干预未伤质量）；|diff| ≤ 0.05 nat 视为不劣于带宽（判据见 notes/06）",
        "layer_mean_max_over_min": {"none": a["layer_mean_max_over_min"]["final"],
                                    "aux": b["layer_mean_max_over_min"]["final"],
                                    "ratio_none_over_aux": round(
                                        a["layer_mean_max_over_min"]["final"] /
                                        max(b["layer_mean_max_over_min"]["final"], 1e-9), 2)},
        "max_vio_global_final": {"none": a["balance_final"]["max_vio"] if a["balance_final"] else None,
                                 "aux": b["balance_final"]["max_vio"] if b["balance_final"] else None},
        "n_zero_cells_final": {"none": a["balance_final"]["n_zero_cells"] if a["balance_final"] else None,
                               "aux": b["balance_final"]["n_zero_cells"] if b["balance_final"] else None},
        "probeB_reference": {"layer0_max_over_min_after_200_steps": 327.8,
                             "note": "探针 B run2 现场参照（B=8/T=512，无干预）——本实验 none 臂应同向"},
    }


def main():
    ap = argparse.ArgumentParser(description="ch3 均衡干预消融：none vs Shazeer 式 load-loss（负载直方图）")
    ap.add_argument("--steps", type=int, default=2000, help="每臂步数（fast=300 / full=2000）")
    ap.add_argument("--out-name", type=str, default=None)
    ap.add_argument("--arms", type=str, default="none,aux", help="臂选择（none/aux，逗号分隔）")
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
    probe_x = build_probe_batch(device)
    out_json = os.path.join(OUT_DIR, f"ablation_balance_{out_name}.json")

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
        "experiment": "ch3 均衡干预：none vs aux（Shazeer 式 load-loss）——单变量 = 负载均衡干预",
        "module_tier": f"d={D_MODEL}/L={N_LAYER}/h={N_HEAD}/h_kv={N_KV}/V={VOCAB}，B={BATCH}/T={BLOCK}"
                       f"（{TPB:,} tok/步，探针 B 已验配置；MoE=E{N_EXPERT}/top{TOP_K}/w{D_EXPERT}）",
        "aux_loss": f"L_load = {AUX_W} · mean_layers CV(Load)²，Load_i = Σ_x softmax(x)_i"
                    "（R3 正身公式的确定性门改编：Φ 概率代理在 noise→0 退化为硬计数，故用概率质量）",
        "metrics_def": "度量用硬计数：frac=count/(N·k)；max/min 下限取 1（防死专家除零）；"
                       "CV=std/mean（总体 std）；MaxVio=max|frac−1/E|（noaux R35 口径，"
                       "global=跨层聚合后）",
        "corpus": "log/book2-ch08/tokens.bin（sp-8k，306M token 流）",
        "optimizer": f"AdamW(betas=0.9,0.95,wd={WDECL}) + grad_clip 1.0（Book2 五件套）",
        "schedule": f"lr {PEAK_LR} 余弦降至 {PEAK_LR*MIN_LR_RATIO:.0e}，warmup={WARMUP}；两臂统一日程",
        "data_protocol": "顺序连续分块一次通过（moe_mla_slots.TokenStream）；两臂按步号取同批",
        "eval_protocol": f"每 {eval_every} 步+终步：eval.bin 固定 49 窗 {EVAL_BATCHES*TPB:,} token，fp32",
        "routing_probe_protocol": f"eval.bin 首段 {PROBE_B}×{PROBE_T}={PROBE_B*PROBE_T:,} token 固定探针，"
                                  "init + 每 eval 边界快照；长表 routing_{arm}_{out}.csv",
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
        run_arm(arm, args.steps, eval_every, device, stream, ex, ey, probe_x, report, out_json)
    add_pair_summary(report)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[全部完成] 总耗时 {(time.perf_counter()-t_all)/60:.1f} min -> {out_json}", flush=True)
    for a, r in report["arms"].items():
        print(f"  {a:4s} eval_final={r['eval_final']} max/min均值 "
              f"{r['layer_mean_max_over_min']['init']}→{r['layer_mean_max_over_min']['final']} "
              f"{r['sec_per_step_steady']}s/step", flush=True)


if __name__ == "__main__":
    main()
