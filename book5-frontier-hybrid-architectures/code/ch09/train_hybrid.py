# train_hybrid.py —— Book5 ch9 训练 CLI：三刀整机冒烟（smoke）/ 整合短训（integrated）
# 用途：三刀机（两刀底座 + 9 层 GDN + w=810 等总参补偿）的点火健康检查与 ≤30 分钟整合短训
#   （作者决策形态：全量长训暂缓——预算账与重启手册归 DESIGN-3TO1.md，写作期归写手）：
#   --task smoke       20 步（MPS bf16，分钟级）：初始 loss 健康检查（锚 ln 32000=10.3735；精确预言
#                      ln V + d·σ²/2 = 10.578——untied 无自泄漏，承 Book3/4 判据）+ 有限性 + 有限下降；
#   --task integrated  整合短训（管道验收口径：三刀协同无冲突的真凭据，非效果训练）：默认 100 步
#                      ——大纲推算「2-3 s/step → 600-900 步」经 smoke s1 实测 11.715 s/step 作废
#                      （GDN chunk=64 块循环在 MPS 无融合 kernel——ch5 scaling 已实录 MPS 慢 ~25×；
#                      9 层 GDN fwd+bwd 实测 6.1 s + 其余 ~5.6 s）。一窗 30 min 降档定版 100 步
#                      ≈19.5 min 训练 + 2 次 49 窗 fp32 eval ≈6.6 min（留 12% 漂移余量贴窗）；
#                      超窗 ckpt 断点续跑补齐、判据不降；降档留痕入 JSON。无 NaN、eval 下降为判据。
# 配方（Book2 ch10 五件套原样继承，与 Book4 train_409 同宗——跨册 import 复用不复制）：
#   AdamW(0.9,0.95,wd=0.1)+clip 1.0+cosine 1e-3→1e-4（终 lr=峰值 10%）+warmup 100+训练 bf16 autocast/eval fp32。
# 语料：log/book3-ch10/tokens32k.bin（sp-32000，Book3 P-03 重切，56,284,791 tok 训练池 + 1.2M eval 独立池）
#   ——与 Book4 两刀机同一语料与 eval 协议（49 窗 401,408 tok fp32），跨章可比。
# 断点续跑：每 eval 边界写 ckpt（模型+优化器+步号）；中断后重跑同命令自动续；curve CSV 增量落盘。
# 运行方式：source env.sh && python "code/ch09/train_hybrid.py" --task smoke --out-name s1
#           python "code/ch09/train_hybrid.py" --task integrated --steps 600 --out-name run1
# 产物（log/book5-ch09/，不入库）：trainhyb_{task}_{out}.json（运行元数据）+ curve_{task}_{out}.csv
#   （逐步 train loss / eval 边界）+ ckpt_{task}_{out}.pt（保留）。
import argparse
import csv
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# —— 跨册 import 链（三刀改装史延续到训练 CLI：整机件=本册 hybrid409，训练五件套=Book4 train_409 ——）
from hybrid409 import (EXPECT_TOTAL, HYBRID_W_EXPERT, LAYER_PATTERN, ThreeKnifeLM,   # noqa: E402
                       hand_account, make_hybrid_cfg, typed_state_account, _B4_CH09_CANDS)

# Book4 ch09 显式三候选（此前仅靠 hybrid409.bootstrap() 的 sys.path 副作用偶然命中——去偶然性）
_B4_CH09 = next((p for p in _B4_CH09_CANDS
                 if os.path.exists(os.path.join(p, "train_409.py"))), None)
if _B4_CH09 is None:
    raise FileNotFoundError([os.path.join(p, "train_409.py") for p in _B4_CH09_CANDS])
sys.path.insert(0, _B4_CH09)
from train_409 import (BATCH, BLOCK, CURVE_FIELDS, EVAL_BATCHES, LN_V, MIN_LR_RATIO,  # noqa: E402
                       PEAK_LR, SEED, SIG2, TPB, WARMUP, WDECL, TrainStream,
                       _rewrite_curve, build_eval_batches, corpus_checks, corpus_stats,
                       eval_loss, lr_at, load_curve_rows)

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book5-ch09")

# smoke 20 步（分钟级）；integrated 100 步（smoke s1 实测 11.715 s/step 的一窗降档定版——见文件头留痕）
TASKS = {"smoke": dict(steps=20, eval_every=10),
         "integrated": dict(steps=100, eval_every=50)}


def run(task, out_name, steps=None, device_pref="mps"):
    spec = TASKS[task]
    steps = steps or spec["steps"]
    eval_every = spec["eval_every"]
    print("[语料] tokens32k 七件套校验（Book3 P-03 产物复检——与两刀机同一语料口径）", flush=True)
    corpus = corpus_checks()
    device = torch.device(device_pref if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(SEED)

    cfg = make_hybrid_cfg()
    model = ThreeKnifeLM(cfg)
    hand = hand_account(cfg)
    total = model.n_params()
    assert total == hand["total"] == EXPECT_TOTAL, f"参数账失败 {total}"
    state_ledger = typed_state_account(cfg)
    model = model.to(device)
    stream = TrainStream(os.path.join(REPO_ROOT, "log", "book3-ch10", "tokens32k.bin"))
    assert steps <= stream.max_step, f"训练池不够 {steps} 步（{stream.max_step} 步可用）"
    ex, ey = build_eval_batches(device)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=WDECL)

    tag = f"{task}_{out_name}"
    ckpt_path = os.path.join(OUT_DIR, f"ckpt_{tag}.pt")
    json_path = os.path.join(OUT_DIR, f"trainhyb_{tag}.json")
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
    print(f"[任务] {task} | 三刀机 hybrid409（层模式 {LAYER_PATTERN}，MLA@3/7/11（0 起）= 人读 4/8/12 层）| "
          f"设备 {device} | {steps} 步 × {TPB} tok = {steps * TPB / 1e6:.1f}M token | 参数 {total:,}", flush=True)
    print(f"[状态账] MLA 3 层 {state_ledger['mla_per_token_elems']} 元素/token（无界）+ GDN 9 层固定 "
          f"{state_ledger['gdn_fixed_total_elems']:,}（与 n 无关）| w_expert {HYBRID_W_EXPERT}（938→810 等总参补偿）",
          flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    curve_path = os.path.join(OUT_DIR, f"curve_{tag}.csv")
    if start_step == 1 and os.path.exists(curve_path):
        os.remove(curve_path)
    prev_curve = load_curve_rows(curve_path, upto=start_step - 1)
    curve, flushed = list(prev_curve), len(prev_curve)
    if start_step > 1:
        _rewrite_curve(curve_path, curve)
    if prev_curve:
        print(f"[曲线] 并入既有窗口 {prev_curve[0]['step']}-{prev_curve[-1]['step']} 共 {len(prev_curve)} 行", flush=True)
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
            torch.mps.synchronize()                      # 逐步同步——与 train_409 秒/步同口径（含完成）
            mem_peak = max(mem_peak, torch.mps.current_allocated_memory())
        sec_steps.append(time.perf_counter() - t0)

        if step == 1:                                    # 初始 loss 自检（第一步更新前的前向）
            with torch.no_grad():
                init_loss0 = loss.item()
                logit_var0 = logits.var().item()
                zt_mean0 = logits.gather(-1, y.unsqueeze(-1)).squeeze(-1).float().mean().item()
            healthy = (abs(init_loss0 - (LN_V + SIG2 / 2)) < 0.06 and math.isfinite(init_loss0))
            print(f"[冒烟自检] 初始 loss {init_loss0:.4f} | ln V {LN_V:.4f} | 预言 ln V + d·σ²/2 "
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
    # 末步恰为 eval 边界时复用该边界 eval（同权重同协议——省一次 49 窗 fp32 前向；JSON 留痕）
    last_boundary = next((c["eval_loss"] for c in reversed(curve)
                          if c.get("eval_loss") is not None and c.get("step") == steps), None)
    if last_boundary is not None:
        final_ev, final_ev_note = last_boundary, "复用末步边界 eval（同权重同协议，未重跑）"
    else:
        final_ev, final_ev_note = eval_loss(model, ex, ey), "实跑"
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
    meta = {"task": task, "model": "hybrid409 三刀机", "out_name": out_name,
            "seed": SEED, "device": str(device),
            "params": total, "vs_cand2_gap": total - 409_115_648,
            "typed_state_account": state_ledger,
            "config": {"d": 1024, "L": 12, "h": 16, "V": 32000, "untied": True, "theta": 10000,
                       "eps": 1e-5, "E": cfg.n_expert, "top_k": cfg.top_k, "n_shared": cfg.n_shared,
                       "w_expert": cfg.w_expert, "mla": cfg.mla, "gdn": cfg.gdn,
                       "layer_types_config0": cfg.layer_types,
                       "mla_human_reading": "第 4/8/12 层（1 起）= 3/7/11（0 起）",
                       "scoring": cfg.scoring, "routed_scaling_factor": cfg.routed_scaling_factor},
            "recipe": "AdamW(0.9,0.95,wd=0.1)+clip1.0+cosine 1e-3->1e-4+warmup100（Book2 五件套，train_409 同宗）",
            "precision": "train bf16 autocast / eval fp32（MPS 逐步 sync 计时）",
            "steps": steps, "tokens": steps * TPB,
            "note_integrated": "管道验收口径非效果训练（三刀协同无冲突的真凭据；距 Chinchilla 档差三个数量级）",
            "downgrade_trace": ("大纲推算 2-3 s/step→600-900 步作废：smoke s1 实测稳态 11.715 s/step"
                                "（9 层 GDN chunk=64 块循环 MPS 无融合 kernel）→ 一窗 30 min 降档 100 步留痕"),
            "data": {"train_bin": os.path.join(REPO_ROOT, "log", "book3-ch10", "tokens32k.bin"),
                     "vocab": 32000, **corpus, "corpus_stats": corpus_stats((steps + 1) * TPB)},
            "account": hand,
            "resume_from_step": start_step if start_step > 1 else None,
            "windows": windows,
            "steady_sec_step": round(steady, 4), "wall_sec": round(wall, 1),
            "wall_note": "独占 MPS 窗口、单进程单实验（计时纪律）" if device.type == "mps" else "CPU 口径",
            "mps_peak_mb": round(mem_peak / 1024 / 1024, 1) if mem_peak else None,
            "smoke_check": smoke,
            "final_eval": round(final_ev, 5),
            "final_eval_note": final_ev_note,
            "first_train_loss": curve[0].get("train_loss"),
            "last_train_loss": next((c["train_loss"] for c in reversed(curve) if "train_loss" in c), None),
            "verdict": "PASS" if (finite_ok and smoke["healthy"]) else "CHECK"}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    _rewrite_curve(curve_path, curve)
    print(f"[完成] wall {wall / 60:.1f} min | 稳态 {steady:.3f} s/step | 初始 {init_loss0:.4f} -> "
          f"末 train {meta['last_train_loss']} | final eval {final_ev:.4f} | 判定 {meta['verdict']}", flush=True)
    print(f"[产物] {OUT_DIR}/trainhyb_{tag}.json + curve_{tag}.csv + ckpt_{tag}.pt", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="三刀整机训练 CLI：冒烟 / 整合短训（全量长训暂缓——重启凭 DESIGN-3TO1.md）")
    ap.add_argument("--task", choices=list(TASKS), default="smoke")
    ap.add_argument("--steps", type=int, default=None,
                    help="覆盖默认步数（smoke 20 / integrated 100——降档/贴窗调档留痕用）")
    ap.add_argument("--out-name", default=None, help="产物后缀（防覆写；默认 = task）")
    ap.add_argument("--device", default="mps", help="mps（默认，回退 cpu）")
    args = ap.parse_args()
    run(args.task, args.out_name or "run1", args.steps, args.device)
