# iso_flop.py —— Book2第11章：固定名义6ND的模型/数据分配实验。
# 原历史网格7+5+3共15点；每点位置一次通过，内容未必无重复，形状/日程也随点改变。
# 假设高斯扰动区间与三种锚点敏感性，不是重训置信区间，也不证明连续最优边界。
# 工作区激活env.sh后：python code/ch11/iso_flop.py --help
# --smoke前三点300步独立日程，不构成iso-FLOP证据；与正式实验的所有产物分名保存。
# 同名续跑须配置/语料/代码身份一致；失败不列完成，断点保留。
import argparse
import hashlib
from pathlib import Path
import json
import math
import os
import sys
import time

import numpy as np
import torch

# ---------------- 固定口径 ----------------
SEED = 20261002          # 全书统一种子
REPO_ROOT = os.getcwd()
OUT_DIR = os.path.join(REPO_ROOT, "log", "book2-ch11")
CH08_DIR = os.path.join(REPO_ROOT, "log", "book2-ch08")
TOKENS_BIN = os.path.join(CH08_DIR, "tokens.bin")   # 训练池（ch08/data_prep.py 产物，流首部）
EVAL_BIN = os.path.join(CH08_DIR, "eval.bin")       # eval 池（流末尾留出，与训练段互斥；与 ch08 同窗集）

VOCAB = 8192             # sp-8k（与 ch08 族同分词器）
BLOCK = 256              # 序列长 s
BATCH = 32               # 批大小 b（b32×s256=8192 token/步，与 ch08 同）
TPB = BATCH * BLOCK
EVAL_EVERY = 500
EVAL_BATCHES = 49        # 49×8192=401,408位置；0.0039来自独立同分布近似，不是实测标准误
PEAK_LR = 1e-3
MIN_LR_RATIO = 0.1
WARMUP = 200             # 批二教训：统一 warmup 200（notes/07 §3.1）
WDECL = 0.1
BOOT_SE = 0.004          # 条件扰动的假设标准差；不是重训测得的eval标准误
BOOT_SE_SENS = 0.008     # 加倍扰动的敏感性设置；不保证涵盖全部误差
BOOT_DRAWS = 2000

# 三档预算的离散形状网格，实际N由模型清点，D由整步预算反算。
# N=L*(12d²+13d)+2d；宽度/头维离散约束使N并非严格对数等距。
# 2026-10-03初版C1从3.1M到9.9M的损失随N增加而升高，故向小模型方向补点。
# 这支持扩展观测范围，不证明连续最优小于3.1M；最终C1最佳已测点0.595M仍在边界。
# C2/C3范围随之下移。完整实测N/D见正文表11.2，避免沿用初始计划的舍入数。
# 约306M位置池限定一次通过的可行N范围，不提供最优N的统计界。
LEVELS = [
    ("c1", 6e14, [
        ("c1a", 7, 192, 3),
        ("c1b", 7, 256, 4),
        ("c1c", 8, 320, 5),
        ("c1d", 6, 144, 2),
        ("c1e", 5, 128, 2),
        ("c1f", 4, 128, 2),
        ("c1g", 3, 128, 2),
    ]),
    ("c2", 2e15, [
        ("c2a", 6, 160, 2),
        ("c2b", 7, 192, 3),
        ("c2c", 9, 192, 3),
        ("c2d", 10, 224, 4),
        ("c2e", 11, 256, 4),
    ]),
    ("c3", 6e15, [
        ("c3a", 8, 208, 4),
        ("c3b", 8, 320, 5),
        ("c3c", 11, 384, 6),
    ]),
]

# ---------------- 模型：复用 ch04/warmup_ablation.py 的 GPT（ch08 同款单轨依赖链） ----------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ch04"))
from warmup_ablation import GPT, Block  # noqa: E402,F401  族模型工厂（Block 备行级引用/调试）


# ---------------- 日程与数据（与 ch08/run_family.py 同式，单独成文件便于行级引用） ----------------
def lr_at(step, peak, warmup, total, min_ratio=MIN_LR_RATIO):
    """lr 日程（ch04/ch08 同式）：warmup 段线性升，之后余弦降到 min_ratio*peak。"""
    if warmup > 0 and step <= warmup:
        return peak * step / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


class TrainStream:
    """第k步输入tokens[a:a+TPB]，目标tokens[a+1:a+TPB+1]，均重排(BATCH,BLOCK)。
    各点前缀长度不同；位置不重复不证明内容不重复，也不证明经验分布相同。
    """

    def __init__(self, path):
        self.path = str(path)
        if os.path.getsize(path)%2:
            raise ValueError("uint16训练文件长度须为偶数字节")
        self.arr = np.memmap(path, dtype=np.uint16, mode="r")
        self.max_step = (len(self.arr) - 1) // TPB

    def batch(self, step, device):
        if not 1 <= step <= self.max_step:
            raise ValueError("step超出训练池可消费范围")
        off = (step - 1) * TPB
        span = np.asarray(self.arr[off:off + TPB + 1], dtype=np.int64)      # (8193,)
        x = torch.from_numpy(span[:-1].reshape(BATCH, BLOCK)).to(device)    # (32,256)
        y = torch.from_numpy(span[1:].reshape(BATCH, BLOCK)).to(device)     # (32,256)
        return x, y


def build_eval_batches(device):
    """固定 eval 批集：与 ch08/run_family.py 完全同窗（同种子同 eval.bin）——ch8/ch11 数字可直接同图。"""
    ev = np.memmap(EVAL_BIN, dtype=np.uint16, mode="r")
    gen = torch.Generator().manual_seed(SEED)
    if len(ev)<BLOCK+1:
        raise ValueError("eval池至少需BLOCK+1个token")
    offs = torch.randint(0, len(ev) - BLOCK, (EVAL_BATCHES * BATCH,), generator=gen).numpy()
    xs, ys = [], []
    for o in offs:
        xs.append(np.asarray(ev[o:o + BLOCK], dtype=np.int64))
        ys.append(np.asarray(ev[o + 1:o + BLOCK + 1], dtype=np.int64))
    x = torch.from_numpy(np.stack(xs).reshape(EVAL_BATCHES, BATCH, BLOCK)).to(device)  # (49,32,256)
    y = torch.from_numpy(np.stack(ys).reshape(EVAL_BATCHES, BATCH, BLOCK)).to(device)
    return x, y


@torch.no_grad()
def eval_loss(model, ex, ey):
    """等长固定窗均值；关闭梯度、保持原模式，不推断token独立性。"""
    if len(ex)==0 or ex.shape!=ey.shape:raise ValueError("评估批为空或标签形状不同")
    before=model.training;model.eval()
    try:
        values=[float(model(ex[i],ey[i])[1]) for i in range(len(ex))]
        if not all(math.isfinite(v) for v in values):raise FloatingPointError("eval非有限")
        return sum(values)/len(values)
    finally:model.train(before)


def file_identity(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return {"bytes":os.path.getsize(path),"sha256":h.hexdigest()}


def save_json(path,value):
    payload=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    temp=str(path)+".tmp"
    with open(temp,"w") as f:f.write(payload)
    os.replace(temp,path)


def run_identity(device,stream):
    return {"schema":2,"seed":SEED,"device":device,"torch":str(torch.__version__),
            "batch":BATCH,"block":BLOCK,"vocab":VOCAB,"warmup":WARMUP,
            "peak_lr":PEAK_LR,"min_lr_ratio":MIN_LR_RATIO,"weight_decay":WDECL,
            "eval_batches":EVAL_BATCHES,"eval_sampler":"all_legal_starts_v2",
            "factory":file_identity(Path(__file__).resolve().parents[1]/"ch04/warmup_ablation.py"),
            "runner":file_identity(__file__),"train":file_identity(stream.path),"eval":file_identity(EVAL_BIN)}


# ---------------- 单点训练（含断点续跑；ch08/run_family.py 同骨架） ----------------
def run_point(name, level, c_nominal, n_layer, n_embd, n_head, device, train_stream,
              steps_override=None, eval_every=EVAL_EVERY, out_name="iso_flop"):
    if Path(out_name).name!=out_name or out_name in ("",".",".."):
        raise ValueError("out_name必须为单个文件名")
    os.makedirs(OUT_DIR,exist_ok=True)
    ckpt_path = os.path.join(OUT_DIR, f"ckpt_{out_name}__{name}.pt")
    if not os.path.exists(ckpt_path) and any(Path(OUT_DIR,f"curve_{out_name}__{name}{ext}").exists() for ext in (".csv",".npy")):
        raise FileExistsError("同名曲线已存在但无可核续跑断点，请另选out-name")
    torch.manual_seed(SEED)   # 同seed不使不同形状模型的逐参数初始化相同。
    model = GPT(n_layer=n_layer, n_embd=n_embd, n_head=n_head,
                vocab=VOCAB, block_size=BLOCK).to(device)
    n_total = model.n_params()
    n_nonemb = model.n_params(non_embedding=True)
    # iso-FLOP 步数：D=C/(6N) 取整到步；反报 D_actual 与 C_actual（与名义 C 偏差 <0.1% 量级，如实登记）
    if steps_override is not None:
        steps = steps_override
    else:
        steps = int(round(c_nominal / (6.0 * n_nonemb) / TPB))
    if steps<=0 or not 0<=WARMUP<steps or eval_every<=0:
        raise ValueError("需要0<=warmup<steps，评估间隔须正")
    point_identity={**run_identity(device,train_stream),"point":[name,level,n_layer,n_embd,n_head],
                    "steps":steps,"eval_every":eval_every,"C_nominal":c_nominal}
    d_tokens = steps * TPB
    c_actual = 6.0 * n_nonemb * d_tokens
    if steps_override is None:
        assert steps >= 400, f"{name}: 步数 {steps} 过短（warmup 200 后余量不足）"
    assert steps <= train_stream.max_step, \
        f"{name}: 步数 {steps} 超出训练池一次通过上限 {train_stream.max_step}"
    print(f"\n===== 点 {name}（档 {level}, C={c_nominal:.1e}）: {n_layer}L×{n_embd}d×{n_head}h "
          f"（head_dim={n_embd // n_head}）非嵌入 N={n_nonemb:,} / 总参 {n_total:,} =====\n"
          f"      steps={steps:,} -> D={d_tokens:,} tok, C_actual=6ND={c_actual:.6e} "
          f"（名义偏差 {c_actual / c_nominal - 1:+.4%}）", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=WDECL)
    ex, ey = build_eval_batches(device)

    train_losses, lrs, evals = [], [], []
    start_step, elapsed_train, elapsed_eval = 0, 0.0, 0.0
    if os.path.exists(ckpt_path):   # ---- 断点续跑：从最近 eval 边界恢复 ----
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        if ck.get("run_identity")!=point_identity:
            raise ValueError("断点的配置/语料/代码与本次不一致，或缺少身份；另用out-name")
        model.load_state_dict(ck["model"])
        torch.set_rng_state(ck["cpu_rng_state"].cpu())
        opt.load_state_dict(ck["opt"])
        train_losses, lrs, evals = ck["train_losses"], ck["lrs"], ck["evals"]
        start_step, elapsed_train, elapsed_eval = ck["step"], ck["elapsed_train"], ck["elapsed_eval"]
        if not 0<=start_step<=steps or len(train_losses)!=start_step or len(lrs)!=start_step:
            raise ValueError("断点步数/曲线不一致")
        print(f"  [续跑] 自 step {start_step + 1}（已完成 {start_step} 步，eval {len(evals)} 次）", flush=True)

    diverged_at, reason = None, None
    t_point = time.perf_counter()
    for step in range(start_step + 1, steps + 1):
        lr = lr_at(step, PEAK_LR, WARMUP, steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = train_stream.batch(step, device)
        t0 = time.perf_counter()
        with torch.autocast(device_type="mps", dtype=torch.bfloat16) if device == "mps" else torch.autocast("cpu", enabled=False):
            _, loss = model(x, y)
        lv=float(loss.detach())
        if not math.isfinite(lv) or lv>100:
            diverged_at,reason=step,"非有限或超过本实验阈值的loss，未执行更新"
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0))
        if not math.isfinite(grad_norm):
            diverged_at,reason=step,"非有限梯度，未执行更新"
            break
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
        if step % eval_every == 0 or step == steps:
            t0 = time.perf_counter()
            el = eval_loss(model, ex, ey)
            if device == "mps":
                torch.mps.synchronize()
            elapsed_eval += time.perf_counter() - t0
            evals.append({"step": step, "eval_loss": round(el, 5),
                          "n_tokens": EVAL_BATCHES * TPB})
            print(f"  [eval] step {step}: eval loss {el:.4f}（{EVAL_BATCHES*TPB:,} token，fp32）", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "train_losses": train_losses, "lrs": lrs, "evals": evals,
                        "elapsed_train": elapsed_train, "elapsed_eval": elapsed_eval,
                        "config": {"n_layer": n_layer, "n_embd": n_embd, "n_head": n_head,
                                   "vocab":VOCAB,"block_size":BLOCK},
                        "run_identity":point_identity,"cpu_rng_state":torch.get_rng_state()},
                       ckpt_path+".tmp")
            os.replace(ckpt_path+".tmp",ckpt_path)

    wallclock = time.perf_counter() - t_point
    steps_done = len(train_losses)
    tail = train_losses[-min(100, steps_done):] if steps_done else []
    completed=diverged_at is None and steps_done==steps and bool(evals) and evals[-1]["step"]==steps
    result = {
        "name": name, "level": level, "C_nominal": c_nominal, "C_actual": 6.0*n_nonemb*steps_done*TPB, "C_planned":c_actual,
        "N": n_nonemb, "N_total": n_total,
        "n_layer": n_layer, "n_embd": n_embd, "n_head": n_head, "head_dim": n_embd // n_head,
        "steps": steps_done, "steps_planned": steps,
        "completed":completed,"budget_matched":steps_override is None,
        "run_identity":point_identity,"checkpoint":ckpt_path,
        "D_tokens": steps_done * TPB,
        "one_pass":"目标位置不重复，各点复用嵌套前缀；不是内容去重或严格固定经验分布",
        "final_train_loss": round(sum(tail) / len(tail), 5) if tail else None,
        "final_train_loss_def": "最后min(100,已完成步数)步train loss均值",
        "eval_losses": evals,
        "eval_final": evals[-1]["eval_loss"] if completed else None,
        "eval_tokens_per_eval": EVAL_BATCHES * TPB,
        "diverged": diverged_at is not None, "diverged_at_step": diverged_at, "divergence_reason": reason,
        "wallclock": round(wallclock, 1), "wallclock_scope":"本次调用片段，累计训练/评估时间另列",
        "wallclock_sec_train_only": round(elapsed_train, 1),
        "wallclock_sec_eval_only": round(elapsed_eval, 1),
        "sec_per_step": round(elapsed_train / max(1, steps_done), 4),
        "precision":"train bf16 autocast / eval fp32" if device=="mps" else "train/eval fp32",
    }
    stem = os.path.join(OUT_DIR, f"curve_{out_name}__{name}")
    with open(stem + ".csv", "w", encoding="utf-8") as f:
        f.write("step,train_loss,lr\n")
        for i, (lv, lr) in enumerate(zip(train_losses, lrs), 1):
            f.write(f"{i},{lv:.6g},{lr:.6g}\n")
    np.save(stem + ".npy", np.column_stack([np.arange(1, steps_done + 1), train_losses, lrs]))
    # 成功也保留最近断点，失败保留最后一个有效eval边界，不用重训才能取得权重。
    print(f"  [{'点完成' if completed else '点未完成'}] steps {steps_done} | final_train {result['final_train_loss']} | "
          f"eval_final {result['eval_final']} | {result['sec_per_step']}s/step | "
          f"wall {wallclock / 60:.1f} min", flush=True)
    del model, opt, ex, ey
    if device == "mps":
        torch.mps.empty_cache()
    return result


# ---------------- 分析：每档抛物线谷底 N*(C) + 斜率拟合 ----------------
def _vertex(x,y):
    """正曲率二次拟合的有限顶点；非正曲率并不证明真实曲线单调。"""
    if len(x)<3 or len(np.unique(x))<3:return None
    a,b,_=np.polyfit(x,y,2)
    v=-b/(2*a) if a>0 else None
    return float(v) if v is not None and math.isfinite(v) and -300<v<300 else None


def usable_point(p):
    """旧报告须给完整步数及末步eval；不能把失败/未完成点纳入拟合。"""
    v=p.get("eval_final")
    return (not p.get("diverged",False) and p.get("completed",True)
            and p.get("steps",0)>0 and p.get("steps")==p.get("steps_planned")
            and isinstance(v,(int,float)) and math.isfinite(v) and v>=0
            and p.get("N",0)>0 and p.get("C_actual",0)>0
            and bool(p.get("eval_losses")) and p["eval_losses"][-1]["step"]==p["steps"])


def fit_level(level_name, c_nominal, pts):
    """二次形状拟合、假设高斯扰动和LOO敏感性；不将条件区间称作重训CI。"""
    pts=sorted(pts,key=lambda p:p["N"])
    if len(pts)<3 or len({p["N"] for p in pts})!=len(pts):
        raise ValueError("二次拟合至少需要3个不同N")
    if not all(usable_point(p) and p.get("budget_matched",True) for p in pts):
        raise ValueError("只接受有限、完整、末步已评估的点")
    x = np.log10(np.array([p["N"] for p in pts], dtype=float))
    y = np.array([p["eval_final"] for p in pts], dtype=float)
    a, b, c = np.polyfit(x, y, 2)
    out = {
        "level": level_name, "C_nominal": c_nominal,
        "C_actual_mean": float(np.mean([p["C_actual"] for p in pts])),
        "n_points": len(pts), "points": [p["name"] for p in pts],
        "N_range": [float(10 ** x.min()), float(10 ** x.max())],
        "quad": {"a": float(a), "b": float(b), "c": float(c)},
        "eval_finals": [p["eval_final"] for p in pts],
        "eval_final_monotone_decreasing_in_N": bool(all(y[i] > y[i + 1] for i in range(len(y) - 1))),
    }
    xv=_vertex(x,y)
    if xv is not None:
        out["log10_N_star"] = float(xv)
        out["N_star"] = float(10 ** xv)
        out["L_star_pred"] = float(a * xv * xv + b * xv + c)
        out["N_star_inside_grid"] = bool(x.min() <= xv <= x.max())
    else:
        out["log10_N_star"] = None
        out["N_star"] = None
        out["L_star_pred"] = None
        out["N_star_inside_grid"] = False
        out["note"] = "拟合没有可报告的有限正曲率顶点；不据此断言真实损失单调"

    # ① 参数协方差（≥4 点才有自由度）：delta 法传到 x*
    if len(pts) >= 4:
        try:
            coef, cov = np.polyfit(x, y, 2, cov=True)
            a_, b_ = coef[0], coef[1]
            if a_ > 0:
                g = np.array([b_ / (2 * a_ * a_), -1.0 / (2 * a_), 0.0])   # x* 对 (a,b,c) 的梯度
                out["log10_N_star_se_cov"] = float(math.sqrt(max(0.0, g @ cov @ g)))
        except Exception as e:  # noqa: BLE001
            out["cov_error"] = str(e)

    # ② 条件参数扰动：假设独立高斯标准差0.004/0.008，非实测重训SE
    rng = np.random.default_rng(SEED)
    for se, tag in ((BOOT_SE, "main"), (BOOT_SE_SENS, "sens")):
        xs = []
        no_valley = 0
        for _ in range(BOOT_DRAWS):
            yb = y + rng.normal(0.0, se, len(y))
            xv = _vertex(x, yb)
            if xv is None:
                no_valley += 1
            else:
                xs.append(xv)
        xs = np.asarray(xs)
        entry = {"assumed_se":se,"draws":BOOT_DRAWS,
                 "no_valley_frac":no_valley/BOOT_DRAWS if BOOT_DRAWS else None,
                 "interpretation":"固定二次形状和独立高斯噪声下的条件敏感性，非重训置信区间"}
        if len(xs):
            q = np.percentile(xs, [2.5, 50, 97.5])
            entry.update({"log10_N_star_ci95": [float(v) for v in q],
                          "N_star_ci95": [float(10 ** v) for v in q],
                          "n_valid_draws": int(len(xs)),
                          "note": "分位数只反映所设噪声；3点无残差自由度，未覆盖种子/日程/形状误差"})
        else:
            entry["note"] = f"有效抽取仅 {len(xs)}，谷底不可定位"
        out[f"bootstrap_{tag}"] = entry

    # ③ 逐点剔除（LOO，≥4 点）
    if len(pts) >= 4:
        loo = []
        for i in range(len(pts)):
            keep = [j for j in range(len(pts)) if j != i]
            xv = _vertex(x[keep], y[keep])
            loo.append({"dropped": pts[i]["name"],
                        "log10_N_star": None if xv is None else float(xv),
                        "N_star": None if xv is None else float(10 ** xv)})
        out["leave_one_out"] = loo
    return out


def fit_slope(level_fits,points=None):
    """不同锚点选择下的描述性斜率；没有连续最优界或置信带。"""
    out={"slope_def":"所选log N锚点对log C的OLS描述",
         "lit_compare":{"kaplan":.73,"chinchilla_isoflop":.5,
                        "note":"平移截距后只比斜率形状，不能据本小网格确认或推翻文献定律"}}
    anchors=[]
    for fit in level_fits:
        pts=[p for p in (points or []) if p["level"]==fit["level"] and usable_point(p)]
        if not pts:continue
        best=min(pts,key=lambda p:p["eval_final"])
        anchors.append({"level":fit["level"],"log10C":math.log10(fit["C_actual_mean"]),
                        "vertex":fit["log10_N_star"],"vertex_in_grid":fit["N_star_inside_grid"],
                        "argmin":math.log10(best["N"]),"argmin_name":best["name"]})
    out["anchors"]=anchors
    def ols(pairs):
        if len(pairs)<2 or len({p[0] for p in pairs})<2:return None
        slope,intercept=np.polyfit([p[0] for p in pairs],[p[1] for p in pairs],1)
        return {"slope":float(slope),"intercept":float(intercept)}
    inside=[a for a in anchors if a["vertex"] is not None and a["vertex_in_grid"]]
    out["slope_ingrid_vertices"]=ols([(a["log10C"],a["vertex"]) for a in inside])
    out["slope_argmin_all"]=ols([(a["log10C"],a["argmin"]) for a in anchors])
    out["slope_mixed"]=ols([(a["log10C"],a["vertex"] if a in inside else a["argmin"]) for a in anchors])
    slopes=[out[k]["slope"] for k in ("slope_ingrid_vertices","slope_argmin_all","slope_mixed") if out[k]]
    out["anchor_sensitivity_range"]=[min(slopes),max(slopes)] if slopes else None
    out["interpretation"]="三种选择的变化范围，非上下界、非置信区间；in-grid不保证真实谷底"
    out["extrapolated_vertex_levels"]=[a["level"] for a in anchors if a["vertex"] is not None and not a["vertex_in_grid"]]
    losses=[(math.log10(f["C_actual_mean"]),math.log10(f["L_star_pred"])) for f in level_fits
            if f["N_star_inside_grid"] and f["L_star_pred"] is not None and f["L_star_pred"]>0]
    fit=ols(losses)
    out["L_star_power_in_C_ingrid"]=fit["slope"] if fit else None
    out["D_over_N_candidates"]={f["level"]:f["C_actual_mean"]/(6*f["N_star"]**2)
        for f in level_fits if f["N_star_inside_grid"] and f["N_star"] is not None}
    return out


def analyze(report):
    """部分网格只列观测；每档至少3个合法不同N后才作条件拟合。"""
    if report.get("meta",{}).get("run_identity",{}).get("smoke"):
        report["level_fits"]=[];report["slope_fit"]={}
        report["analysis_note"]="smoke不是iso-FLOP网格，不拟合最优规模";return report
    pts=[p for p in report["points"] if usable_point(p) and p.get("budget_matched",True)]
    report["excluded_points"]=[p.get("name") for p in report["points"]
                              if not usable_point(p) or not p.get("budget_matched",True)]
    fits=[];partial=[]
    for name,budget,cfgs in LEVELS:
        group=sorted([p for p in pts if p["level"]==name],key=lambda p:p["N"])
        if len(group)<3:
            if group:partial.append({"level":name,"n_points":len(group),"status":"不足3点，不作二次拟合"})
            continue
        fit=fit_level(name,budget,group);fit["complete"]=len(group)==len(cfgs);fits.append(fit)
    report["level_fits"]=fits;report["partial_levels"]=partial
    report["slope_fit"]=fit_slope(fits,pts)
    return report


# ---------------- 主流程 ----------------
def main():
    global OUT_DIR,CH08_DIR,TOKENS_BIN,EVAL_BIN,BOOT_DRAWS,SEED
    ap=argparse.ArgumentParser(description="iso-FLOP网格；同名断点仅在完整身份一致时恢复")
    ap.add_argument("--levels",default="all")
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--analyze-only",action="store_true")
    ap.add_argument("--input",type=Path,help="只分析时的原报告")
    ap.add_argument("--out-dir",default=OUT_DIR)
    ap.add_argument("--data-dir",default=CH08_DIR)
    ap.add_argument("--out-name")
    ap.add_argument("--bootstrap-draws",type=int,default=BOOT_DRAWS)
    ap.add_argument("--seed",type=int,default=SEED)
    ap.add_argument("--device",choices=["cpu","mps"],default="mps" if torch.backends.mps.is_available() else "cpu")
    a=ap.parse_args()
    if a.bootstrap_draws<0:raise ValueError("bootstrap次数不能为负")
    BOOT_DRAWS=a.bootstrap_draws;SEED=a.seed;OUT_DIR=a.out_dir;CH08_DIR=a.data_dir
    TOKENS_BIN=os.path.join(CH08_DIR,"tokens.bin");EVAL_BIN=os.path.join(CH08_DIR,"eval.bin")
    name=a.out_name or ("iso_flop_analysis" if a.analyze_only else "iso_flop_smoke" if a.smoke else "iso_flop")
    if Path(name).name!=name or name in ("",".",".."):raise ValueError("out-name只接受文件名")
    name=name[:-5] if name.endswith(".json") else name
    if name in ("",".",".."):raise ValueError("无效产物名称")
    target=Path(OUT_DIR)/(name+".json");target.parent.mkdir(parents=True,exist_ok=True)
    if a.analyze_only:
        source=a.input or Path(OUT_DIR)/"iso_flop.json"
        if target.exists() or source.resolve()==target.resolve():raise FileExistsError("分析结果不得覆盖输入或旧输出")
        report=json.loads(source.read_text());analyze(report)
        report["analysis_source"]={"path":str(source),"identity":file_identity(source)}
        save_json(target,report);print(target);return
    names=[x[0] for x in LEVELS] if a.levels=="all" else a.levels.split(",")
    if a.smoke:names=["c1"]
    if not names or any(n not in {x[0] for x in LEVELS} for n in names):raise ValueError("未知算力档")
    stream=TrainStream(TOKENS_BIN)
    identity={**run_identity(a.device,stream),"smoke":a.smoke,"grid":LEVELS}
    # JSON把tuple转list；标准化后可跨进程精确比较。
    identity=json.loads(json.dumps(identity))
    if target.exists():
        report=json.loads(target.read_text())
        if report.get("meta",{}).get("run_identity")!=identity:
            raise ValueError("已有报告身份不同或为旧版，请使用新的out-name")
    else:
        report={"meta":{"run_identity":identity,"precision":"bf16/fp32" if a.device=="mps" else "fp32",
                        "protocol":"位置一次通过；各点不同N/D/日程/形状，不能称严格单因素",
                        "smoke_note":"300步独立日程，不是iso-FLOP点" if a.smoke else None},"points":[]}
    save_json(target,report)
    done={p["name"] for p in report["points"] if usable_point(p)}
    for level,budget,cfgs in LEVELS:
        if level not in names:continue
        for point,layers,width,heads in (cfgs[:3] if a.smoke else cfgs):
            if point in done:continue
            result=run_point(point,level,budget,layers,width,heads,a.device,stream,
                             steps_override=300 if a.smoke else None,
                             eval_every=100 if a.smoke else EVAL_EVERY,out_name=name)
            report["points"]=[p for p in report["points"] if p["name"]!=point]+[result]
            report["points"].sort(key=lambda p:(p["level"],p["N"]))
            if not a.smoke:analyze(report)
            save_json(target,report)
            if not result["completed"]:raise RuntimeError("本点未完成，保留断点；不继续假报全网格成功")
    print(f"[本次所选范围结束] {target}")


if __name__ == "__main__":
    main()
