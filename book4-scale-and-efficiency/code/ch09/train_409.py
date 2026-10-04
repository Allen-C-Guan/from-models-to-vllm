# train_409.py —— Book4 ch9 训练 CLI：两刀整机冒烟（smoke）/ 整合短训（integrated），双臂 cand2/cand3
# 用途：两刀机的点火健康检查与 ≤30 分钟整合短训（作者决策形态：全量长训暂缓——预算账与重启手册见
#   同目录 DESIGN-409M.md §2/§6；full 档不在本机执行，重启凭设计文档）：
#   --task smoke       20 步（MPS bf16，<1 min/臂）：初始 loss 健康检查（锚 ln 32000=10.3735；精确预言
#                      ln V + d*s^2/2 = 10.578——untied 无自泄漏）+ 有限性（无 NaN/Inf）+ 有限下降；
#   --task integrated  整合短训（管道验收口径：两刀协同无冲突的真凭据，非效果训练）：cand2 主臂默认
#                      540 步（按探针 F 实测 3.007 s/step 规划 ≈27 min——v1.1 降档线：预留 +12% 复测漂移
#                      余量，超窗即 ckpt 断点续跑补齐、判据不降）；cand3 对照臂默认 512 步（3.514 s/step
#                      ≈30 min；窗口紧张时 --steps 300 降档留痕）。两刀协同无 NaN、eval 曲线下降为判据。
#   --arm cand2|cand3  config 双臂（llama409.CANDS 定版；cand1 只留参数账教学点，不设训练档）。
# 配方（Book2 ch10 五件套原样继承，与 Book3 train_215 同宗）：AdamW(0.9,0.95,wd=0.1)+clip 1.0+
#   cosine 1e-3 -> 1e-4（终 lr=峰值 10%）+ warmup 100 + 训练 bf16 autocast / eval fp32。
# 语料：log/book3-ch10/tokens32k.bin（sp-32000，Book3 P-03 重切，56,284,791 tok 训练池 + 1.2M eval
#   独立池）——启动时七件套参数带校验沿用（piece/id 范围/池深/字节数/EOS 数/首 3 篇回读）。
# 训练读法（Book3 train_215 同款）：第 k 步消费 [(k-1)*8192, (k+1)*8192)——x=前 8192、y=后 8192，
#   顺序一次通过；eval = eval 池固定种子抽 49 窗 (8,1024)（401,408 token，fp32）——与 Book3 207M
#   短训曲线同协议跨章可比（cand3 等参对照的读数意义正在于此）。
# 断点续跑：每 eval 边界写 ckpt（模型+优化器+步号）；中断后重跑同命令自动续（超窗补齐判据不降）；
#   curve CSV 增量落盘、续跑并入既有窗口。
# 运行方式：source env.sh && python train_409.py --task smoke --arm cand2 --out-name s2
#           python train_409.py --task integrated --arm cand2 --out-name run1   （≈27 min，独占 MPS 窗口）
#           python train_409.py --task integrated --arm cand3 --out-name run1   （≈30 min，第二窗）
# 产物（log/book4-ch09/，不入库）：train409_{task}_{arm}_{out}.json（运行元数据）+
#   curve_{task}_{arm}_{out}.csv（逐步 train loss / eval 边界）+ ckpt_{task}_{arm}_{out}.pt（保留）。
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
sys.path.insert(0, HERE)
from llama409 import (CANDS, BASE_TOTAL_207M, TwoKnifeLM, activated_account,   # noqa: E402
                      hand_account, kv_per_token_elements, make_cfg)

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book4-ch09")
TOKENS_BIN = os.path.join(REPO_ROOT, "log", "book3-ch10", "tokens32k.bin")
EVAL_BIN = os.path.join(REPO_ROOT, "log", "book3-ch10", "eval32k.bin")
SP_MODEL = os.path.join(REPO_ROOT, "log", "book3-ch10", "sp32000_owt.model")
SLICE_META = os.path.join(REPO_ROOT, "log", "book3-ch10", "slice_meta_32k.json")
DOCS_JSONL_CANDS = [os.path.join(REPO_ROOT, "log", "book2-scaling", "owt_docs.jsonl"),
                    os.path.join(REPO_ROOT, "log", "book3-ch10", "owt_docs.jsonl")]

SEED = 20261002                       # 全书统一种子
BATCH, BLOCK = 8, 1024                # B/T（探针 F 预算档：8192 tok/步）
TPB = BATCH * BLOCK
EVAL_BATCHES = 49                     # 49 x (8,1024) = 401,408 token fp32 eval（Book3 同口径）
PEAK_LR, MIN_LR_RATIO, WARMUP, WDECL = 1e-3, 0.1, 100, 0.1
LN_V = math.log(32000)                # 10.3735——初始 loss 的地板（均匀分布的熵）
SIG2 = 1024 * 0.02 ** 2               # d*s^2 = 0.4096——初始 logit 方差（untied 无泄漏项）

# 整合短训步数（v1.1 降档线）：cand2 主臂 540 步（598 步/30min 的 0.9 档，防复测漂移 +12% 超窗）；
# cand3 对照臂 512 步（窗口紧张时 --steps 300 降档留痕——降档须写入元数据）
TASKS = {"smoke": dict(steps=20, eval_every=10),
         "integrated": dict(steps={"cand2": 540, "cand3": 512}, eval_every=50)}


def corpus_checks():
    """七件套参数带校验（Book3 prep_sp32k 的复检版——重切过的语料重跑训练前必须全过）：

    ①piece_size==32000 ②id∈[0,32000) ③train≥42,000,000 ④eval==1,200,000
    ⑤磁盘字节==2×token 数（两池） ⑥EOS 计数==编码篇数 ⑦流首部与首 3 篇重编码逐 id 一致。
    """
    checks = {}
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(SP_MODEL)
    checks["piece_size == 32000"] = sp.GetPieceSize() == 32000
    train = np.memmap(TOKENS_BIN, dtype=np.uint16, mode="r")
    ev = np.memmap(EVAL_BIN, dtype=np.uint16, mode="r")
    checks["0 <= id < 32000"] = (int(train.max()) < 32000 and int(train.min()) >= 0
                                 and int(ev.max()) < 32000)
    checks["train_tokens >= 42,000,000"] = len(train) >= 42_000_000
    checks["eval_tokens == 1,200,000"] = len(ev) == 1_200_000
    checks["磁盘字节数 == 2 x token 数"] = (os.path.getsize(TOKENS_BIN) == 2 * len(train)
                                           and os.path.getsize(EVAL_BIN) == 2 * len(ev))
    eos_id = sp.piece_to_id("</s>")
    n_eos = int((train == eos_id).sum()) + int((ev == eos_id).sum())
    docs = json.load(open(SLICE_META, encoding="utf-8"))["docs_encoded"]
    checks["EOS 计数 == 编码篇数"] = n_eos == docs
    docs_path = next((p for p in DOCS_JSONL_CANDS if os.path.exists(p)), None)
    if docs_path:
        head = []
        with open(docs_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                head.extend(sp.encode(json.loads(line)["text"], out_type=int, add_eos=True))
        checks["首 3 篇重编码一致"] = list(np.concatenate([train, ev])[:len(head)]) == head
    else:                                                    # 语料原文缺席：如实降级为 meta 对账
        checks["首 3 篇重编码一致（降级：原文缺席，改对 slice_meta 计数）"] = n_eos == docs
    ok = all(checks.values())
    for k, v in checks.items():
        print(f"  [{'x' if v else ' '}] {k}")
    assert ok, f"语料七件套校验失败：{checks}"
    return {"checks": checks, "train_tokens": int(len(train)), "eval_tokens": int(len(ev)),
            "eos_count": n_eos, "docs": docs, "id_max": int(train.max())}


def corpus_stats(n_tokens):
    """训练段一元分布账（Book2/3 同源算法）：活跃 token 种数与词频熵（nat）——短训曲线的地板参照。"""
    arr = np.memmap(TOKENS_BIN, dtype=np.uint16, mode="r")
    seg = np.asarray(arr[:n_tokens])
    cnt = np.bincount(seg, minlength=32000)
    p = cnt.astype(np.float64)
    p = p / p.sum()
    p = p[p > 0]
    return {"tokens": int(n_tokens), "active_types": int((cnt > 0).sum()),
            "unigram_entropy_nat": round(float(-(p * np.log(p)).sum()), 4)}


def lr_at(step, total):
    """五件套 lr 日程：线性 warmup -> 余弦降至峰值 10%（Book2 ch10 同款）。"""
    if step <= WARMUP:
        return PEAK_LR * step / WARMUP
    prog = (step - WARMUP) / max(1, total - WARMUP)
    return PEAK_LR * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * prog)))


class TrainStream:
    """tokens32k 流首部按步号顺序切块（一次通过，第 k 步消费 [(k-1)*TPB, (k+1)*TPB)）。"""

    def __init__(self, path):
        self.arr = np.memmap(path, dtype=np.uint16, mode="r")
        self.max_step = (len(self.arr) - 1) // TPB

    def batch(self, step, device):
        off = (step - 1) * TPB
        span = np.asarray(self.arr[off:off + TPB + 1], dtype=np.int64)           # (8193,)
        x = torch.from_numpy(span[:-1].reshape(BATCH, BLOCK)).to(device)         # (8,1024)
        y = torch.from_numpy(span[1:].reshape(BATCH, BLOCK)).to(device)          # (8,1024)
        return x, y


def build_eval_batches(device):
    """eval 池内固定种子抽 49 批 (8,1024)（窗位协议与 Book3 train_215 一致，跨章可比）。"""
    ev = np.memmap(EVAL_BIN, dtype=np.uint16, mode="r")
    gen = torch.Generator().manual_seed(SEED)
    offs = torch.randint(0, len(ev) - BLOCK - 1, (EVAL_BATCHES * BATCH,), generator=gen).numpy()
    xs = np.stack([ev[o:o + BLOCK] for o in offs]).astype(np.int64)
    ys = np.stack([ev[o + 1:o + BLOCK + 1] for o in offs]).astype(np.int64)
    x = torch.from_numpy(xs.reshape(EVAL_BATCHES, BATCH, BLOCK)).to(device)      # (49,8,1024)
    y = torch.from_numpy(ys.reshape(EVAL_BATCHES, BATCH, BLOCK)).to(device)
    return x, y


@torch.no_grad()
def eval_loss(model, ex, ey):
    model.eval()
    losses = []
    for i in range(EVAL_BATCHES):
        _, loss = model(ex[i], ey[i])                 # fp32 口径（autocast 只包训练段）
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def load_curve_rows(path, upto):
    """读既有 curve CSV 中 step <= upto 的行（续跑并入前一窗口，按 step 去重保末值）。"""
    if not os.path.exists(path):
        return []
    by_step = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("step") and r.get("train_loss") and int(r["step"]) <= upto:
                row = {"step": int(r["step"]), "train_loss": float(r["train_loss"])}
                for k in ("eval_loss", "lr", "sec_step"):
                    if r.get(k):
                        row[k] = float(r[k])
                by_step[row["step"]] = row
    return [by_step[s] for s in sorted(by_step)]


CURVE_FIELDS = ["step", "train_loss", "eval_loss", "lr", "sec_step", "final"]


def _rewrite_curve(path, curve):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CURVE_FIELDS)
        w.writeheader()
        for row in curve:
            w.writerow({k: row.get(k, "") for k in w.fieldnames})


def run(task, arm, out_name, steps=None, device_pref="mps"):
    assert arm in ("cand2", "cand3"), "训练档只开 cand2/cand3（cand1 留参数账教学点）"
    spec = TASKS[task]
    steps = steps or (spec["steps"] if isinstance(spec["steps"], int) else spec["steps"][arm])
    eval_every = spec["eval_every"]
    print(f"[语料] tokens32k 七件套校验（Book3 P-03 产物复检）", flush=True)
    corpus = corpus_checks()
    device = torch.device(device_pref if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(SEED)

    cfg = make_cfg(arm)
    model = TwoKnifeLM(cfg)
    hand = hand_account(cfg)
    total = model.n_params()
    assert total == hand["total"] == CANDS[arm]["expect_total"], f"参数账失败 {total}"
    act, kv = activated_account(cfg), kv_per_token_elements(cfg)
    model = model.to(device)
    stream = TrainStream(TOKENS_BIN)
    assert steps <= stream.max_step, f"训练池不够 {steps} 步（{stream.max_step} 步可用）"
    ex, ey = build_eval_batches(device)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=WDECL)

    tag = f"{task}_{arm}_{out_name}"
    ckpt_path = os.path.join(OUT_DIR, f"ckpt_{tag}.pt")
    json_path = os.path.join(OUT_DIR, f"train409_{tag}.json")
    start_step, init_loss0, logit_var0, zt_mean0, healthy = 1, None, None, None, None
    if os.path.exists(ckpt_path):                        # 断点续跑（超窗补齐：判据不降）
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck["step"] >= steps and os.path.exists(json_path):
            print(f"[跳过] {tag} 已完成，产物在 {json_path}；重跑请换 --out-name", flush=True)
            return
        if ck["step"] >= steps:
            os.remove(ckpt_path)
        else:
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            start_step = ck["step"] + 1
            init_loss0, logit_var0, zt_mean0 = ck["init_loss"], ck["logit_var"], ck["zt_mean"]
            print(f"[续跑] 自 ckpt 第 {ck['step']} 步起（{ckpt_path}）——超窗补齐口径，判据不降", flush=True)
    print(f"[任务] {task} | 臂 {arm}（{CANDS[arm]['note']}）| 设备 {device} | {steps} 步 x {TPB} tok = "
          f"{steps * TPB / 1e6:.1f}M token | 参数 {total:,}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    curve_path = os.path.join(OUT_DIR, f"curve_{tag}.csv")
    if start_step == 1 and os.path.exists(curve_path):
        os.remove(curve_path)
    prev_curve = load_curve_rows(curve_path, upto=start_step - 1)
    curve, flushed = list(prev_curve), len(prev_curve)
    if start_step > 1:
        _rewrite_curve(curve_path, curve)
    if prev_curve:
        print(f"[曲线] 并入既有窗口 {prev_curve[0]['step']}-{prev_curve[-1]['step']} 共 {len(prev_curve)} 行",
              flush=True)
    t_start, sec_steps, mem_peak = time.perf_counter(), [], 0.0
    for step in range(start_step, steps + 1):
        lr = lr_at(step, steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = stream.batch(step, device)
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "mps" \
                else torch.autocast("cpu", enabled=False):
            logits, loss = model(x, y)                   # (8,1024,V) / 标量
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device.type == "mps":
            torch.mps.synchronize()                      # 逐步同步——与探针 F 秒/步同口径（含完成）
            mem_peak = max(mem_peak, torch.mps.current_allocated_memory())
        sec_steps.append(time.perf_counter() - t0)

        if step == 1:                                    # 初始 loss 自检（第一步更新前的前向）
            with torch.no_grad():
                init_loss0 = loss.item()
                logit_var0 = logits.var().item()
                zt_mean0 = logits.gather(-1, y.unsqueeze(-1)).squeeze(-1).float().mean().item()
            healthy = (abs(init_loss0 - (LN_V + SIG2 / 2)) < 0.06 and math.isfinite(init_loss0))
            print(f"[冒烟自检] 初始 loss {init_loss0:.4f} | ln V {LN_V:.4f} | 预言 ln V + d*s^2/2 "
                  f"{LN_V + SIG2 / 2:.4f} | logit 方差 {logit_var0:.4f}（预言 {SIG2:.4f}）| "
                  f"E[z_t] {zt_mean0:+.4f} | {'健康' if healthy else '异常'}", flush=True)

        curve.append({"step": step, "train_loss": round(loss.item(), 5), "lr": lr,
                      "sec_step": round(sec_steps[-1], 3)})
        if step % 10 == 0 or step == steps:
            print(f"  step {step:>5}/{steps}  train {loss.item():.4f}  lr {lr:.2e}  "
                  f"{sum(sec_steps[-10:]) / min(10, len(sec_steps)):.2f} s/step", flush=True)
        if step % eval_every == 0 or step == steps:
            ev = eval_loss(model, ex, ey)
            curve[-1]["eval_loss"] = round(ev, 5)
            print(f"  [eval @{step:>4}] {ev:.4f}（fp32，49 窗 401,408 tok）", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "init_loss": init_loss0, "logit_var": logit_var0, "zt_mean": zt_mean0,
                        "eval_loss": ev}, ckpt_path)
            need_header = not os.path.exists(curve_path) or os.path.getsize(curve_path) == 0
            with open(curve_path, "a", newline="", encoding="utf-8") as f:   # eval 边界增量落盘
                w = csv.DictWriter(f, fieldnames=CURVE_FIELDS)
                if need_header:
                    w.writeheader()
                for row in curve[flushed:]:
                    w.writerow({k: row.get(k, "") for k in w.fieldnames})
            flushed = len(curve)

    wall = time.perf_counter() - t_start
    final_ev = eval_loss(model, ex, ey)
    steady = sum(sec_steps[3:]) / max(1, len(sec_steps) - 3)                  # 丢首 3 步预热（探针同口径）
    curve.append({"step": steps, "eval_loss": round(final_ev, 5), "final": True})

    prev_ss = [r["sec_step"] for r in prev_curve if r.get("sec_step")]
    windows = []
    if prev_curve:
        prev_steady = round(sum(prev_ss[3:]) / max(1, len(prev_ss) - 3), 4) if len(prev_ss) > 3 else None
        windows.append({"steps": [prev_curve[0]["step"], prev_curve[-1]["step"]],
                        "steady_sec_step": prev_steady, "rows": len(prev_curve),
                        "note": "续跑前窗口（曲线经 eval 边界增量落盘保全）"})
    windows.append({"steps": [start_step, steps], "steady_sec_step": round(steady, 4),
                    "wall_sec": round(wall, 1)})

    smoke = {"init_loss": init_loss0, "ln_V": LN_V, "predict_lnV_plus_var_half": LN_V + logit_var0 / 2,
             "logit_var": logit_var0, "zt_mean": zt_mean0,
             "finite": all(math.isfinite(c["train_loss"]) for c in curve if "train_loss" in c),
             "dropped": next((c["train_loss"] for c in reversed(curve) if "train_loss" in c), None),
             "healthy": healthy if init_loss0 is not None else None}
    finite_ok = smoke["finite"] and final_ev < curve[0].get("train_loss", math.inf)
    meta = {"task": task, "arm": arm, "arm_note": CANDS[arm]["note"], "out_name": out_name,
            "seed": SEED, "device": str(device),
            "params": total, "params_active": act, "kv_per_token": kv,
            "config": {"d": 1024, "L": 12, "h": 16, "V": 32000, "untied": True, "theta": 10000,
                       "eps": 1e-5, "E": cfg.n_expert, "top_k": cfg.top_k, "n_shared": cfg.n_shared,
                       "w_expert": cfg.w_expert, "mla": cfg.mla,
                       "first_k_dense_replace": cfg.first_k_dense_replace,
                       "scoring": cfg.scoring, "routed_scaling_factor": cfg.routed_scaling_factor},
            "recipe": "AdamW(0.9,0.95,wd=0.1)+clip1.0+cosine 1e-3->1e-4+warmup100（Book2 五件套）",
            "precision": "train bf16 autocast / eval fp32（MPS 逐步 sync 计时）",
            "steps": steps, "tokens": steps * TPB,
            "note_integrated": "管道验收口径非效果训练（两刀协同无冲突的真凭据；距 Chinchilla 档差三个数量级）",
            "data": {"train_bin": TOKENS_BIN, "eval_bin": EVAL_BIN, "vocab": 32000,
                     **corpus, "corpus_stats": corpus_stats((steps + 1) * TPB)},
            "account": hand,
            "resume_from_step": start_step if start_step > 1 else None,
            "windows": windows,
            "steady_sec_step": round(steady, 4), "wall_sec": round(wall, 1),
            "wall_note": "独占 MPS 窗口、单进程单实验（计时纪律）" if device.type == "mps" else "CPU 口径",
            "mps_peak_mb": round(mem_peak / 1024 / 1024, 1) if mem_peak else None,
            "smoke_check": smoke,
            "final_eval": round(final_ev, 5),
            "first_train_loss": curve[0].get("train_loss"),
            "last_train_loss": next((c["train_loss"] for c in reversed(curve) if "train_loss" in c), None),
            "verdict": "PASS" if (finite_ok and smoke["healthy"]) else "CHECK"}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    _rewrite_curve(curve_path, curve)
    print(f"[完成] wall {wall / 60:.1f} min | 稳态 {steady:.3f} s/step | 初始 {init_loss0:.4f} -> "
          f"末 train {meta['last_train_loss']} | final eval {final_ev:.4f} | 判定 {meta['verdict']}", flush=True)
    print(f"[产物] {OUT_DIR}/train409_{tag}.json + curve_{tag}.csv + ckpt_{tag}.pt", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="两刀整机训练 CLI：冒烟 / 整合短训（cand2 主臂 / cand3 等参对照臂；"
                                            "全量长训暂缓——重启凭 DESIGN-409M.md）")
    ap.add_argument("--task", choices=list(TASKS), default="smoke")
    ap.add_argument("--arm", choices=["cand2", "cand3"], default="cand2")
    ap.add_argument("--steps", type=int, default=None,
                    help="覆盖默认步数（smoke 20 / integrated：cand2 540、cand3 512——降档留痕用）")
    ap.add_argument("--out-name", default=None, help="产物后缀（防覆写；默认 = task_arm）")
    ap.add_argument("--device", default="mps", help="mps（默认，回退 cpu）")
    args = ap.parse_args()
    run(args.task, args.arm, args.out_name or "run1", args.steps, args.device)
