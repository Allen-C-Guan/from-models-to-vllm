# train_215.py —— Book3 ch10 训练 CLI：冒烟（smoke）/ 整合短训（integrated）/ 全量（full）
# 用途：207M 整机的点火、管道验收与全量重启档（本机作者决策：全量长训暂缓——预算账见
#   DESIGN-215M.md §2/§6；full 档供高算力机器凭文档重启，默认按 Chinchilla 4.3B 折算步数）：
#   --task smoke      20 步（MPS bf16，<1 min）：初始 loss 健康检查（锚 ln 32000=10.3735；
#                     精确预言 ln V + d*s^2/2 = 10.578——untied 无自泄漏，见 llama215.py 自测）+
#                     有限性（无 NaN/Inf）+ 有限下降。产物与正式产物分文件（--smoke 产物不覆写正式件）。
#   --task integrated 1000 步（MPS bf16，B=8/T=1024=8192 tok/步，≈28 min）：管道验收口径——
#                     四插槽协同无冲突的真凭据（eval 曲线下降），非效果训练（8.2M token，
#                     距 Chinchilla 配方 4.3B 差三个数量级，正文与设计文档均写明）。
#   --task full       全量档（重启用，DESIGN §7 验收⑤/§9 第 7 步）：默认 524,902 步 =
#                     Chinchilla 4.3B token（--steps 可覆盖）；前置 = 按 DESIGN §4 重切
#                     ≥4.3B token 语料（本机 56M 池会被训练池断言拒跑）、批量/lr 按 §5
#                     随机器重定并写入元数据；eval 边界加宽到 1000 步。
# 配方（Book2 ch10 五件套原样继承）：AdamW(0.9, 0.95, wd=0.1) + grad clip 1.0 +
#   cosine 1e-3 -> 1e-4（终 lr = 峰值 10%）+ warmup 100 + bf16 autocast（train）/ fp32（eval）。
# 语料：log/book3-ch10/tokens32k.bin（sp-32000，P-03 重切；读法=notes/08 §1.2——
#   np.memmap uint16 无头裸流 56,284,791 tok；eval32k.bin 1.2M tok 尾切独立池）。
# 训练读法（与 ch05 TrainStream 同款）：第 k 步消费 [(k-1)*8192, (k+1)*8192)——x=前 8192、
#   y=后 8192，顺序一次通过；eval = eval 池内固定种子抽 49 窗 x (8,1024)（401,408 token，fp32）。
# 断点续跑：每 eval 边界写 ckpt（模型+优化器+步号+lr+流位置）；中断后重跑同命令自动续。
#   curve CSV 同步在 eval 边界增量落盘、续跑时把既有窗口并入（同 out-name 不再覆盖丢曲线），
#   运行元数据以 windows 数组分窗记计时账（首窗秒/步不因截断丢失）。
# 运行方式：source env.sh && python train_215.py --task smoke --out-name s1
#           python train_215.py --task integrated --out-name run1     （≈28 min，独占 MPS 窗口）
#           python train_215.py --task full --out-name f1             （重启档；语料按 DESIGN §4 先行）
# 产物（log/book3-ch10/，不入库）：train215_{task}_{out}.json（运行元数据）+
#   curve_{task}_{out}.csv（逐步 train loss / eval 边界 eval loss；多窗合并）+
#   ckpt_{task}_{out}.pt（保留——integrated/full 档 ckpt 是 Book6 消费线登记对象，DESIGN §10）。
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
from llama215 import LLAMA_215M, Llama215, hand_account   # noqa: E402  整机正身（四插槽 import 组装）

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book3-ch10")
TOKENS_BIN = os.path.join(OUT_DIR, "tokens32k.bin")
EVAL_BIN = os.path.join(OUT_DIR, "eval32k.bin")

SEED = 20261002                       # 全书统一种子
BATCH, BLOCK = 8, 1024                # B/T（探针 C 预算档：8192 tok/步）
TPB = BATCH * BLOCK
EVAL_BATCHES = 49                     # 49 x (8,1024) = 401,408 token fp32 eval（ch2-6 消融同口径）
PEAK_LR, MIN_LR_RATIO, WARMUP, WDECL = 1e-3, 0.1, 100, 0.1
LN_V = math.log(32000)                # 10.3735——初始 loss 的地板（均匀分布的熵）
SIG2 = 1024 * 0.02 ** 2               # d*s^2 = 0.4096——初始 logit 方差（untied 无泄漏项）

TASKS = {"smoke": dict(steps=20, eval_every=10),
         "integrated": dict(steps=1000, eval_every=100),
         "full": dict(steps=524_902, eval_every=1000)}   # Chinchilla 4.3B/8192（重启档；语料按 DESIGN §4 重切）


def corpus_stats(n_tokens):
    """训练段一元分布账（Book2 ch10 corpus_stats 同源算法）：活跃 token 种数与词频熵（nat）。

    冒烟/短训曲线的第一层「地板」——loss 降到它之下，说明模型已越过「背词频」、
    开始兑现窗内局部搭配（正文 10.3 节引用）。
    """
    seg = np.asarray(TrainStream(TOKENS_BIN).arr[:n_tokens])
    cnt = np.bincount(seg, minlength=32000)
    p = cnt.astype(np.float64)
    p = p / p.sum()
    p = p[p > 0]
    return {"tokens": int(n_tokens), "active_types": int((cnt > 0).sum()),
            "unigram_entropy_nat": round(float(-(p * np.log(p)).sum()), 4)}


def load_curve_rows(path, upto):
    """读既有 curve CSV 中 step <= upto 的行（续跑并入前一窗口——同 out-name 续跑不覆盖丢曲线）。

    同步去重：ckpt 落盘点之后、截断点之前可能残留越界行，按 step 保留最后一次记录。
    """
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


def flush_curve(path, curve, from_idx):
    """curve[from_idx:] 追加落盘（eval 边界调用）——中途截断的窗口也留得下曲线。"""
    need_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "train_loss", "eval_loss", "lr", "sec_step", "final"])
        if need_header:
            w.writeheader()
        for row in curve[from_idx:]:
            w.writerow({k: row.get(k, "") for k in w.fieldnames})


CURVE_FIELDS = ["step", "train_loss", "eval_loss", "lr", "sec_step", "final"]


def _rewrite_curve(path, curve):
    """整文件重写为 curve 行（续跑起点截断 / 收尾规范化共用）。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CURVE_FIELDS)
        w.writeheader()
        for row in curve:
            w.writerow({k: row.get(k, "") for k in w.fieldnames})


def lr_at(step, total):
    """五件套的 lr 日程：线性 warmup -> 余弦降至峰值 10%（Book2 ch10 同款）。"""
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
        span = np.asarray(self.arr[off:off + TPB + 1], dtype=np.int64)       # (8193,)
        x = torch.from_numpy(span[:-1].reshape(BATCH, BLOCK)).to(device)     # (8,1024)
        y = torch.from_numpy(span[1:].reshape(BATCH, BLOCK)).to(device)      # (8,1024)
        return x, y


def build_eval_batches(device):
    """eval 池内固定种子抽 49 批 (8,1024)（窗位协议与 ch2-6 消融一致，跨章可比）。"""
    ev = np.memmap(EVAL_BIN, dtype=np.uint16, mode="r")
    gen = torch.Generator().manual_seed(SEED)
    offs = torch.randint(0, len(ev) - BLOCK - 1, (EVAL_BATCHES * BATCH,), generator=gen).numpy()
    xs = np.stack([ev[o:o + BLOCK] for o in offs]).astype(np.int64)
    ys = np.stack([ev[o + 1:o + BLOCK + 1] for o in offs]).astype(np.int64)
    x = torch.from_numpy(xs.reshape(EVAL_BATCHES, BATCH, BLOCK)).to(device)  # (49,8,1024)
    y = torch.from_numpy(ys.reshape(EVAL_BATCHES, BATCH, BLOCK)).to(device)
    return x, y


@torch.no_grad()
def eval_loss(model, ex, ey):
    model.eval()
    losses = []
    for i in range(EVAL_BATCHES):
        _, loss = model(ex[i], ey[i])          # fp32 口径（autocast 只包训练段）
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def run(task, out_name, steps=None, device_pref="mps"):
    steps = steps or TASKS[task]["steps"]
    eval_every = TASKS[task]["eval_every"]
    assert os.path.exists(TOKENS_BIN) and os.path.exists(EVAL_BIN), \
        "缺 tokens32k.bin / eval32k.bin——先跑 prep_sp32k.py（P-03，<30 min）"
    device = torch.device(device_pref if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(SEED)

    model = Llama215()                                   # 207M 定版（四插槽 import 整机）
    detail, hand = hand_account(model.cfg)
    total = model.n_params()
    assert total == hand == 207_119_360, f"参数账失败 {total}"
    model = model.to(device)
    stream = TrainStream(TOKENS_BIN)
    assert steps <= stream.max_step, f"训练池不够 {steps} 步（{stream.max_step} 步可用）"
    ex, ey = build_eval_batches(device)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=WDECL)

    ckpt_path = os.path.join(OUT_DIR, f"ckpt_{task}_{out_name}.pt")
    json_path = os.path.join(OUT_DIR, f"train215_{task}_{out_name}.json")
    start_step, init_loss0, logit_var0, zt_mean0, healthy = 1, None, None, None, None
    if os.path.exists(ckpt_path):                        # 断点续跑（Book2 教训 1/4）
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck["step"] >= steps and os.path.exists(json_path):
            print(f"[跳过] {task}_{out_name} 已完成，产物在 {json_path}；重跑请换 --out-name", flush=True)
            return
        if ck["step"] >= steps:                          # 完成但 JSON 缺失（写盘前中断）——清掉重跑
            os.remove(ckpt_path)
        else:
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            start_step = ck["step"] + 1
            init_loss0, logit_var0, zt_mean0 = ck["init_loss"], ck["logit_var"], ck["zt_mean"]
            print(f"[续跑] 自 ckpt 第 {ck['step']} 步起（{ckpt_path}）", flush=True)
    print(f"[任务] {task} | 设备 {device} | {steps} 步 x {TPB} tok = {steps * TPB / 1e6:.1f}M token | "
          f"参数 {total:,}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    curve_path = os.path.join(OUT_DIR, f"curve_{task}_{out_name}.csv")
    if start_step == 1 and os.path.exists(curve_path):
        os.remove(curve_path)                 # 全新起跑：清旧 curve 防重复行；续跑则保留并入
    prev_curve = load_curve_rows(curve_path, upto=start_step - 1)   # 既有窗口的曲线行
    curve = list(prev_curve)
    flushed = len(curve)                      # 已在盘上的行数（增量追加起点）
    if start_step > 1:
        _rewrite_curve(curve_path, curve)     # 截断到续跑点（防 ckpt 后的残留行重复追加）
    if prev_curve:
        print(f"[曲线] 并入既有窗口 {prev_curve[0]['step']}-{prev_curve[-1]['step']} 共 {len(prev_curve)} 行",
              flush=True)
    t_start = time.perf_counter()
    sec_steps = []
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
        sec_steps.append(time.perf_counter() - t0)

        if step == 1:                                    # 初始 loss 自检（第一步更新前的前向）
            with torch.no_grad():
                init_loss0 = loss.item()
                logit_var0 = logits.var().item()
                zt_mean0 = logits.gather(-1, y.unsqueeze(-1)).squeeze(-1).float().mean().item()
            healthy = (abs(init_loss0 - (LN_V + SIG2 / 2)) < 0.05 and math.isfinite(init_loss0))
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
            flush_curve(curve_path, curve, flushed)         # eval 边界增量落盘（截断不丢曲线）
            flushed = len(curve)

    wall = time.perf_counter() - t_start
    final_ev = eval_loss(model, ex, ey)
    steady = sum(sec_steps[3:]) / max(1, len(sec_steps) - 3)      # 丢首 3 步预热（探针 C 同口径）
    curve.append({"step": steps, "eval_loss": round(final_ev, 5), "final": True})

    # 分窗计时账：续跑时首窗秒/步从并入的 curve 行复原（截断窗口的元数据不丢）
    prev_ss = [r["sec_step"] for r in prev_curve if r.get("sec_step")]
    prev_steady = round(sum(prev_ss[3:]) / max(1, len(prev_ss) - 3), 4) if len(prev_ss) > 3 else None
    windows = []
    if prev_curve:
        windows.append({"steps": [prev_curve[0]["step"], prev_curve[-1]["step"]],
                        "steady_sec_step": prev_steady, "rows": len(prev_curve),
                        "note": "续跑前窗口（曲线行经 eval 边界增量落盘保全）"})
    windows.append({"steps": [start_step, steps], "steady_sec_step": round(steady, 4),
                    "wall_sec": round(wall, 1)})

    # 冒烟判据（v1.1 口径：只锚 ln V——untied 无绑定头的自泄漏路径；方差项 ½ds² 如实报）
    smoke = {"init_loss": init_loss0, "ln_V": LN_V, "predict_lnV_plus_var_half": LN_V + logit_var0 / 2,
             "logit_var": logit_var0, "zt_mean": zt_mean0,
             "finite": all(math.isfinite(c["train_loss"]) for c in curve if "train_loss" in c),
             "dropped": curve[-1]["train_loss"] if "train_loss" in curve[-1] else curve[-2]["train_loss"],
             "healthy": healthy if init_loss0 is not None else None}
    meta = {"task": task, "out_name": out_name, "seed": SEED, "device": str(device),
            "params": total, "config": LLAMA_215M,
            "recipe": "AdamW(0.9,0.95,wd=0.1)+clip1.0+cosine 1e-3->1e-4+warmup100（Book2 五件套）",
            "precision": "train bf16 autocast / eval fp32",
            "steps": steps, "tokens": steps * TPB,
            "note_integrated": "管道验收口径非效果训练（8.2M token 距 Chinchilla 4.3B 差三数量级）",
            "data": {"train_bin": TOKENS_BIN, "eval_bin": EVAL_BIN, "vocab": 32000,
                     "train_tokens": int(len(stream.arr)), "eval_tokens": int(EVAL_BATCHES * TPB),
                     "corpus_stats": corpus_stats((steps + 1) * TPB)},   # 实耗流段 [0,(steps+1)*TPB)
            "account": detail,
            "resume_from_step": start_step if start_step > 1 else None,
            "windows": windows,
            "steady_sec_step": round(steady, 4), "wall_sec": round(wall, 1),
            "wall_note": "独占 MPS 窗口、单进程单实验（计时纪律）" if device.type == "mps" else "CPU 口径",
            "smoke_check": smoke,
            "final_eval": round(final_ev, 5),
            "first_train_loss": curve[0]["train_loss"] if curve else None,
            "last_train_loss": next((c["train_loss"] for c in reversed(curve) if "train_loss" in c), None)}

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    _rewrite_curve(curve_path, curve)         # 收尾整文件重写=规范化（含并入窗口与 final 行）
    # ckpt 保留（不删）：integrated/full 档 ckpt 是 Book6 消费线登记对象（DESIGN-215M.md §10）
    print(f"[完成] wall {wall / 60:.1f} min | 稳态 {steady:.3f} s/step | 初始 {init_loss0:.4f} -> "
          f"末 train {meta['last_train_loss']} | final eval {final_ev:.4f}", flush=True)
    print(f"[产物] {OUT_DIR}/train215_{task}_{out_name}.json + curve_{task}_{out_name}.csv"
          f"{' + ' + os.path.basename(ckpt_path) if os.path.exists(ckpt_path) else ''}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="207M 整机训练 CLI：冒烟 / 整合短训 / 全量重启档（本机作者决策："
                                            "全量暂缓，full 档凭 DESIGN-215M.md 重启执行）")
    ap.add_argument("--task", choices=list(TASKS), default="smoke")
    ap.add_argument("--steps", type=int, default=None,
                    help="覆盖默认步数（smoke 20 / integrated 1000 / full 524,902=Chinchilla 4.3B）")
    ap.add_argument("--out-name", default=None, help="产物后缀（防覆写；默认 = task 名）")
    ap.add_argument("--device", default="mps", help="mps（默认，回退 cpu）")
    args = ap.parse_args()
    run(args.task, args.out_name or args.task, args.steps, args.device)
