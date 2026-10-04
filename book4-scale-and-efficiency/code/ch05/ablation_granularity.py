# ablation_granularity.py —— Book4 ch5 消融：细粒度+共享+均衡手段四臂（②③ 实跑，①④ 引用复用）
# 用途：统一模块档（探针 B 已验配置 d=512/L=6/h=8/h_kv=4/V=8192，B=16/T=512）四臂同数据同种子，
#       大纲 v1.1 钉死四臂（ch5 §5.7）：
#         ① 粗粒度 E8/top2/无共享+aux —— 复用批一 ch3 aux 臂 full 产物（引用登记，不重跑）；
#         ② 细粒度 E16/top4/共享 1+aux（等激活：w=⌊1408/(4+1)⌋=281，每 token 激活 5×281=1405≈dense 1408）；
#         ③ = ② 换 noaux（bias 更新 b_i+=u·sign(e_i)、u=0.001 per-step——R34/2408.15664 Alg1 正身，
#            e_i=1/E−本批该层专家 i 被选比例（=c̄_i−c_i 的比例版，欠载抬/过载压——方向金证 papers/02 L181）；
#            bias 只进选择不进门值；mini 档不做 γ 衰减）；
#         ④ 等激活 dense 对照 —— 复用批一 ch2 dense 臂 full 产物（引用登记，不重跑）。
#       实跑 = ②③ 两臂；②vs③ 单变量 = 均衡手段（损失式 vs 机制式），②vs① 单变量束 = 几何粗细。
#       产出：eval 曲线 + 负载度量（MaxVio/max/min/死专家，口径同 ch3 ablation_balance——硬计数）+
#       E16 档 sec/step 单独实测（「派发开销随 E 增长」实验产物，与 ch1 开销梯度呼应）。
# 协议口径（Book2 五件套，与 ch02/ch03 逐字同款）：AdamW(0.9/0.95, wd=0.1) + clip 1.0
#       + lr 1e-3 余弦降至 1e-4 + warmup 100；train bf16 autocast / eval fp32；
#       tokens.bin 顺序分块一次通过（两臂按步号取同批）；eval 固定 49 窗 (16,512)=401,408 token。
# 运行方式：source env.sh && python ablation_granularity.py [--steps 2000] [--out-name fast|full]
#           [--arms fine_aux,fine_noaux]。两档位：fast=300 步 / full=2000 步（晚间档）。
# 产物（log/book4-ch05/，不入库）：ablation_granularity_{out}.json + curve_{arm}_{out}.csv
#       + routing_{arm}_{out}.csv（长表：phase/step/layer/expert/count/frac）+ ckpt_{arm}_{out}.pt
#       （ckpt 每 eval 边界落盘、臂完成即删；中断后重跑同命令自动续跑）。
# 引用臂（不重跑）：①=log/book4-ch03/ablation_balance_full.json 的 arms.aux；④=log/book4-ch02/
#       ablation_moe_full.json 的 arms.dense——脚本启动时读文件核对锚点（文件缺失则用内嵌副本，注明）。
# 模块来源：不私搭模型——整机/手算账/TopkGate import 自正身 moe_mla_slots（其内部 bootstrap 引 Book3
#       llama_slots）。noaux 干预=本脚本 NoauxGate 子类（与正身 TopkGate 同权重对象，零新增 RNG），
#       「不改 moe_mla_slots 正身一行」纪律承 ch3 钩子先例。
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "00-feasibility")))
from moe_mla_slots import (SEED, SlotConfig, SlotLLaMA, TokenStream, TopkGate,   # noqa: E402
                           activated_account, hand_account)

# ---------------- 固定口径 ----------------
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book4-ch05")
TOKENS_BIN = os.path.join(REPO_ROOT, "log", "book2-ch08", "tokens.bin")
EVAL_BIN = os.path.join(REPO_ROOT, "log", "book2-ch08", "eval.bin")
CH02_FULL_JSON = os.path.join(REPO_ROOT, "log", "book4-ch02", "ablation_moe_full.json")
CH03_FULL_JSON = os.path.join(REPO_ROOT, "log", "book4-ch03", "ablation_balance_full.json")

VOCAB, D_MODEL, N_LAYER, N_HEAD, N_KV = 8192, 512, 6, 8, 4        # 探针 B 已验模块档
D_DENSE = 1408                                                    # ④ dense 臂 d_ff（引用臂的等激活锚）
N_EXPERT, TOP_K, D_EXPERT, N_SHARED = 16, 4, 281, 1                # ②③：E16/top4/w=⌊1408/5⌋/共享 1
AUX_W = 0.1                                                       # ② aux w_load（与 ch3 逐字同款）
NOAUX_U = 0.001                                                   # ③ bias 步长 u（R34 正身；无 γ 衰减）
BATCH, BLOCK = 16, 512
TPB = BATCH * BLOCK
EVAL_BATCHES = 49                                                  # 固定 49 窗 = 401,408 token
PROBE_B, PROBE_T = 4, 2048                                         # 路由探针：8192 token（eval.bin 首段）
PEAK_LR, MIN_LR_RATIO, WARMUP, WDECL, TAIL = 1e-3, 0.1, 100, 0.1, 300
FINE_PARAMS = 57_187_840                                           # E16/w281/共享1 手算锚点
FINE_ACTIVE = 26_062_336                                           # 激活参锚点（vs dense 26,089,984）
ARMS = ("fine_aux", "fine_noaux")

# 引用臂锚点（批一 full 实测；读文件核对，缺失/错版退回内嵌副本）——只锚 JSON 顶层实有字段
REF_COARSE_AUX = {  # ① ch3 aux 臂（E8/top2/w704/无共享 + 0.1·CV²）
    "source": CH03_FULL_JSON, "eval_final": 4.85354, "eval_final_se": 0.01326,
    "sec_per_step_steady": 0.6135,
    "max_vio_global_final": 0.015, "layer_mean_max_over_min_final": 2.13,
    "layer_mean_cv_final": 0.2397, "n_zero_cells_final": 0,
}
REF_DENSE = {      # ④ ch2 dense 臂（d_ff=1408）
    "source": CH02_FULL_JSON, "eval_final": 4.93375, "eval_final_se": 0.01295,
    "sec_per_step_steady": 0.3543,
}
REF_E8_MOE_SEC = 0.7142        # ch2 moe 臂 sec/step（E8 派发开销梯的中间档，只作同机跨表趋势）


def lr_at(step, total):
    """Book2 五件套日程：线性 warmup 100 步 + 余弦 1e-3 → 1e-4（两臂统一）。"""
    if step <= WARMUP:
        return PEAK_LR * step / WARMUP
    prog = (step - WARMUP) / max(1, total - WARMUP)
    return PEAK_LR * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * prog)))


def build_eval_batches(device):
    """固定 eval 窗集：eval 池固定种子抽 49 批 (16,512)（跨臂/跨步完全同集，与 ch2/ch3 同款）。"""
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
    """②③ 同几何整机（E16/top4/w281/共享1，teach 版 SparseMoE 正身）；单变量 = 均衡手段。

    noaux 臂：每层 gate 换成 NoauxGate（同一 weight Parameter 对象——初始化 RNG 流与 aux 臂逐位
    相同，两臂初始权重严格一致）；bias 为 buffer 不进优化器，由训练环每步手工更新。
    """
    cfg = SlotConfig(vocab_size=VOCAB, hidden_size=D_MODEL, num_hidden_layers=N_LAYER,
                     num_attention_heads=N_HEAD, num_key_value_heads=N_KV,
                     intermediate_size=D_DENSE,
                     moe=dict(n_expert=N_EXPERT, top_k=TOP_K, expert_dim=D_EXPERT,
                              n_shared=N_SHARED, variant="teach", mode="mixtral"))
    torch.manual_seed(SEED)
    m = SlotLLaMA(cfg)
    if arm == "fine_noaux":
        for layer in m.model.layers:
            layer.mlp.gate = NoauxGate.from_gate(layer.mlp.gate, u=NOAUX_U)   # 权重共享、零 RNG 消耗
    hand = hand_account(cfg)
    assert m.n_params() == hand["total"] == FINE_PARAMS, \
        f"参数对账失败 {m.n_params()} != {hand['total']} != {FINE_PARAMS}（E16 手算锚点）"
    act, _ = activated_account(cfg)
    assert act == FINE_ACTIVE, f"等激活口径被破坏：{act} != {FINE_ACTIVE}"
    return m, cfg


# ---------------- ③ noaux：路由门子类（R34 正身的教学落地） ----------------
class NoauxGate(TopkGate):
    """noaux 免辅助损失路由门（Wang et al. 2024, arXiv:2408.15664 / DSV3 §2.1.2）。

    与正身 TopkGate 的唯一差异 = 选择规则：top-k 在 (softmax 概率 + bias) 上做；
    门值仍取无 bias 的 softmax 概率（选中集内归一化）——「bias 只影响路由选择，不进门值」
    （2408.15664 原句 + DSV3 Eq.16 两文一致）。bias 为 buffer（不进优化器、不进损失图），
    训练环每步后按 b += u·sign(e) 手工更新（e_i=1/E−被选比例：欠载抬、过载压——Alg 1 方向）。键名
    e_score_correction_bias 对齐 HF DeepseekV2/3 的落地 Buffer（ch5 §5.6「HF 落地」教学点的本机同型件）。
    """

    def __init__(self, hidden_size, n_expert, top_k, u=0.001, mode="mixtral"):
        super().__init__(hidden_size, n_expert, top_k, mode=mode)
        self.register_buffer("e_score_correction_bias", torch.zeros(n_expert))   # (E,)
        self.u = u

    @classmethod
    def from_gate(cls, gate, u=0.001):
        """从正身 gate 构造：复用同一 weight Parameter（参数身份不变，优化器无感）。"""
        new = cls(gate.weight.shape[1], gate.weight.shape[0], gate.top_k, u=u, mode=gate.mode)
        new.weight = gate.weight
        return new

    def forward(self, x):
        h = x.reshape(-1, x.shape[-1])                                        # (B,n,d)->(N,d)
        logits = F.linear(h, self.weight)                                     # (N,E)
        probs = logits.float().softmax(dim=-1)                                # (N,E) fp32（与 ② 同口径）
        sel = probs + self.e_score_correction_bias                            # bias 进选择
        _, idx = torch.topk(sel, self.top_k, dim=-1, sorted=False)            # (N,k)
        w = probs.gather(1, idx)                                              # 门值 = 无 bias 概率
        w = w / w.sum(dim=-1, keepdim=True)                                   # 选中集内归一（与 ② 一致）
        return logits, w, idx


def attach_gate_hooks(model, stash):
    """在每层 router 上挂前向钩子，捕完整输出 (logits, w, idx)——aux 臂的损失接口 + 度量用的真实现场。"""
    hooks = []

    def hook(mod, inp, out):
        stash.append(out)                         # ((N,E) 带梯度, (N,k), (N,k))

    for layer in model.model.layers:
        hooks.append(layer.mlp.gate.register_forward_hook(hook))
    return hooks


def load_cv2_terms(stash):
    """② 的 L_load 正身（与 ch3 逐字一致）：Load_i = Σ_x P(x,i)（softmax 概率质量），返回各层 CV²。"""
    terms = []
    for out in stash:
        probs = out[0].float().softmax(dim=-1)    # (N,E)
        load = probs.sum(dim=0)                   # (E,)  Σ_i Load_i = N
        cv = load.std(unbiased=False) / load.mean()
        terms.append(cv ** 2)
    return terms


def noaux_bias_update(model, stash):
    """③ 的 R34 正身更新：b_i ← b_i + u·sign(e_i)，逐层；e_i = 1/E − 本批该层专家 i 被选比例。

    方向勘正（2408.15664 Alg 1 逐字，papers/02 L181 金证）：违规量 e_i = c̄_i − c_i
    （理想均值 token 数 − 实收 token 数）——**欠载 e_i>0 → bias 升（多被选），过载 → bias 降**。
    与 HF 落地自洽：选择用 score+bias（DeepseekV3TopkRouter scores_for_choice=scores+bias），
    故更新必须给欠载专家抬 bias 才构成负反馈。首版实现误取 e_i=f_i−1/E（正反馈——欠载专家
    被越压越死，72/96 死专家贯穿全程；该次跑留档 fine_noaux_signflip 作反向对照，见 notes/07 B.4）。
    只用本步（最后一个）训练 batch 的路由计数——「无 EMA、只看上一 batch」（Alg 1）。
    """
    n_sel = BATCH * BLOCK * TOP_K                 # 每层总选择数 = N·k
    for layer, out in zip(model.model.layers, stash):
        idx = out[2].detach().reshape(-1)
        counts = torch.bincount(idx, minlength=N_EXPERT).float()
        e = 1.0 / N_EXPERT - counts / n_sel       # (E,) 欠载为正——bias 抬升被压制方
        layer.mlp.gate.e_score_correction_bias += NOAUX_U * torch.sign(e)


# ---------------- 路由探针：每层每专家硬计数直方图 + 均衡度量（口径同 ch3） ----------------
def build_probe_batch(device):
    """固定路由探针：eval.bin 首段 8192 token (4,2048)——跨臂/跨快照完全同批（held-out）。"""
    ev = np.memmap(EVAL_BIN, dtype=np.uint16, mode="r")
    span = np.asarray(ev[:PROBE_B * PROBE_T], dtype=np.int64)
    return torch.from_numpy(span.reshape(PROBE_B, PROBE_T)).to(device)     # (4,2048)


@torch.no_grad()
def routing_snapshot(model, probe_x, device):
    """喂探针 token，返回 L×E 硬计数表 + 逐层/全局度量（max/min、CV、MaxVio、死专家数）。

    度量口径（计数为离散路由的真实现场；与 ch3 ablation_balance 同款）：
      frac_li = count_li / (N_tok·k)（每层选择总数归一，均匀 = 1/E）
      max_over_min = max(count_l) / max(min(count_l), 1)   ← 死专家（count=0）时下限取 1，防 ∞
      CV_l = std(count_l)/mean(count_l)（总体标准差）
      MaxVio_l = max_i |frac_li − 1/E|（noaux R35 的 MaxVio 口径的逐层版）
      MaxVio_global = 跨层聚合每专家计数后再取 max_i |frac_i − 1/E|
    计数直捕钩子 idx（noaux 臂 = 含 bias 的实际选择；aux 臂 = 同一钩子口径，两臂同法可比）。
    """
    model.eval()
    stash = []
    hooks = attach_gate_hooks(model, stash)
    with torch.autocast(device_type="mps", dtype=torch.bfloat16) if device.type == "mps" \
            else torch.autocast("cpu", enabled=False):
        model(probe_x)
    for hk in hooks:
        hk.remove()
    counts = [torch.bincount(out[2].detach().reshape(-1), minlength=N_EXPERT).tolist()
              for out in stash]                    # 每层 (E,)——真实路由（含 noaux bias）
    model.train()
    n_sel = PROBE_B * PROBE_T * TOP_K               # 每层总选择数 = 32768
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


def bias_snapshot(model):
    """③ 的 bias 现场留档：每层 e_score_correction_bias 列表（L×E，浮点小表）。"""
    return [[round(float(b), 5) for b in layer.mlp.gate.e_score_correction_bias]
            for layer in model.model.layers]


def append_routing_csv(arm, out_name, phase, step, snap):
    """长表追加写：phase/step/layer/expert/count/frac（ch5 正文直方图素材的标准形态）。"""
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
    """一个臂的五件套训练环：② CE+aux / ③ CE+bias 更新；断点续跑，快照直方图。"""
    ckpt_path = os.path.join(OUT_DIR, f"ckpt_{arm}_{report['meta']['out_name']}.pt")
    m, cfg = build_arm(arm)
    m = m.to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=WDECL)
    stash = []
    hooks = attach_gate_hooks(m, stash)            # 两臂同钩子：aux 取 logits，度量取 idx

    train_losses, aux_losses, lrs, times, evals, snaps, biases = [], [], [], [], [], [], []
    start_step, elapsed_train, elapsed_eval = 0, 0.0, 0.0
    resumed_from = None
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        m.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        train_losses, aux_losses = ck["train_losses"], ck["aux_losses"]
        lrs, times = ck["lrs"], ck["times"]
        evals, snaps, biases = ck["evals"], ck["snaps"], ck["biases"]
        start_step, elapsed_train, elapsed_eval = ck["step"], ck["elapsed_train"], ck["elapsed_eval"]
        resumed_from = start_step
        print(f"  [{arm} 续跑] 自 step {start_step + 1}", flush=True)
    else:
        snap0 = routing_snapshot(m, probe_x, device)             # 训前（随机初始化）负载直方图
        snaps.append({"step": 0, "phase": "init", **snap0})
        append_routing_csv(arm, report["meta"]["out_name"], "init", 0, snap0)
        if arm == "fine_noaux":
            biases.append({"step": 0, "bias": bias_snapshot(m)})  # 全零档
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
        if arm == "fine_aux":
            aux = AUX_W * torch.stack(load_cv2_terms(stash)).mean()   # 与 ch3 逐字同款
            loss = ce + aux
        else:
            aux = None
            loss = ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if arm == "fine_noaux":
            noaux_bias_update(m, stash)                      # R34：步后按本批计数更新 bias
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
            msg = f"  [{arm:10s}] step {step:5d}/{steps} train {lv:.4f}"
            if arm == "fine_aux":
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
            if arm == "fine_noaux":
                biases.append({"step": step, "bias": bias_snapshot(m)})
            print(f"  [{arm:10s}] [eval] step {step}: eval {el:.4f} ± {se:.4f} | "
                  f"max/min 均值 {snap['layer_mean_max_over_min']} | MaxVio_global "
                  f"{snap['global']['max_vio']} | 死专家 {snap['global']['n_zero_cells']}"
                  f"/{N_LAYER*N_EXPERT}", flush=True)
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": step,
                        "train_losses": train_losses, "aux_losses": aux_losses, "lrs": lrs,
                        "times": times, "evals": evals, "snaps": snaps, "biases": biases,
                        "elapsed_train": elapsed_train, "elapsed_eval": elapsed_eval}, ckpt_path)

    for hk in hooks:
        hk.remove()
    steps_done = len(train_losses)
    steady = times[20:] if len(times) > 20 else times          # 丢前 20 步（惰性编译/缓存预热）
    tail = train_losses[-min(TAIL, steps_done):]
    rec = {
        "arm": arm,
        "arm_def": (f"② 细粒度 E{N_EXPERT}/top-{TOP_K}/w{D_EXPERT}+共享{N_SHARED}+aux："
                    f"CE + {AUX_W}·层均值 CV(Load)²（与 ch3 aux 臂逐字同款）" if arm == "fine_aux" else
                    f"③ 细粒度 E{N_EXPERT}/top-{TOP_K}/w{D_EXPERT}+共享{N_SHARED}+noaux："
                    f"纯 CE + bias 更新 b+=u·sign(e)，u={NOAUX_U}/步（R34 正身，无 γ 衰减）"),
        "config": {"d": D_MODEL, "L": N_LAYER, "h": N_HEAD, "h_kv": N_KV, "V": VOCAB,
                   "B": BATCH, "T": BLOCK, "n_expert": N_EXPERT, "top_k": TOP_K,
                   "expert_dim": D_EXPERT, "n_shared": N_SHARED,
                   "aux_w": AUX_W if arm == "fine_aux" else 0.0,
                   "noaux_u": NOAUX_U if arm == "fine_noaux" else None},
        "params_total": m.n_params(), "params_active": FINE_ACTIVE,
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
        "bias_snapshots": biases if arm == "fine_noaux" else None,
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
          f"MaxVio_global {rec['balance_final']['max_vio'] if rec['balance_final'] else None} | "
          f"{rec['sec_per_step_steady']}s/step", flush=True)
    del m, opt
    if device.type == "mps":
        torch.mps.empty_cache()
    return rec


def load_reference(path, arm, anchors):
    """读批一 JSON 引用臂并核对锚点（防引用文件被覆盖/错版）；缺失则退回内嵌副本。"""
    try:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)["arms"][arm]
        anchor_keys = ("eval_final", "eval_final_se", "sec_per_step_steady")   # JSON 顶层实有字段
        for k in anchor_keys:
            if abs(float(rec[k]) - float(anchors[k])) > 1e-9:
                raise ValueError(f"锚点不符 {k}: {rec[k]} != {anchors[k]}")
        return {"source": path, "from_file": True, **{k: rec[k] for k in
                (*anchor_keys, "evals")}}
    except (OSError, KeyError, ValueError) as e:
        print(f"[引用臂退回内嵌副本] {path}: {e}", flush=True)
        return {"source": path, "from_file": False, "fallback": "内嵌批一副本", **anchors}


def add_pair_summary(report):
    """四臂主对照：③vs②（均衡手段）+ ②vs①（几何粗细方向）+ ②③vs④（等激活差）+ E16 派发开销梯。"""
    arms = report["arms"]
    a, b = arms.get("fine_aux"), arms.get("fine_noaux")
    refs = report.get("references", {})
    if a and b and a["eval_final"] is not None and b["eval_final"] is not None:
        report["pair_summary"] = {
            "eval_final_diff_aux_minus_noaux": round(a["eval_final"] - b["eval_final"], 5),
            "eval_diff_read": "正 = noaux 更低（机制式均衡未伤质量）；|diff| ≤ 0.05 nat 视为不劣于带宽"
                              "（≤0.03 按噪声口径念）——判据预注册见 notes/07 §3.1",
            "max_vio_global_final": {"fine_aux": a["balance_final"]["max_vio"] if a["balance_final"] else None,
                                     "fine_noaux": b["balance_final"]["max_vio"] if b["balance_final"] else None},
            "layer_mean_max_over_min_final": {"fine_aux": a["layer_mean_max_over_min"]["final"],
                                              "fine_noaux": b["layer_mean_max_over_min"]["final"]},
            "n_zero_cells_final": {"fine_aux": a["balance_final"]["n_zero_cells"] if a["balance_final"] else None,
                                   "fine_noaux": b["balance_final"]["n_zero_cells"] if b["balance_final"] else None},
            "granularity_read": {
                "note": "② vs ①（E8 无共享 aux，批一 ch3 引用）：细粒度+共享的方向判读，判据见 notes/07 §3.2",
                "eval_final_diff_fine_minus_coarse": (
                    round(a["eval_final"] - refs["coarse_aux"]["eval_final"], 5)
                    if refs.get("coarse_aux") else None)},
            "equal_activation_read": {
                "note": "② vs ④（dense 引用）：与 ch2 moe−dense=−0.0377 并排念",
                "eval_final_diff_fine_minus_dense": (
                    round(a["eval_final"] - refs["dense"]["eval_final"], 5)
                    if refs.get("dense") else None)},
            "dispatch_overhead_ladder": {
                "note": "「派发开销随 E 增长」实验产物（同机跨表趋势，不同 run 不混表；"
                        "dense/E8 来自批一 full，E16 本档实测）",
                "dense_ref_sec_per_step": REF_DENSE["sec_per_step_steady"],
                "e8_moe_ref_sec_per_step": REF_E8_MOE_SEC,
                "e16_sec_per_step": {"fine_aux": a["sec_per_step_steady"],
                                     "fine_noaux": b["sec_per_step_steady"]},
                "e16_over_e8_ratio": round(a["sec_per_step_steady"] / REF_E8_MOE_SEC, 2),
                "e16_over_dense_ratio": round(a["sec_per_step_steady"] / REF_DENSE["sec_per_step_steady"], 2)},
        }


def main():
    ap = argparse.ArgumentParser(description="ch5 四臂消融：细粒度 E16/top4/共享1——aux vs noaux（②③ 实跑）")
    ap.add_argument("--steps", type=int, default=2000, help="每臂步数（fast=300 / full=2000）")
    ap.add_argument("--out-name", type=str, default=None)
    ap.add_argument("--arms", type=str, default="fine_aux,fine_noaux",
                    help="臂选择（fine_aux/fine_noaux，逗号分隔）")
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
    out_json = os.path.join(OUT_DIR, f"ablation_granularity_{out_name}.json")

    if os.path.exists(out_json):
        with open(out_json, encoding="utf-8") as f:
            report = json.load(f)
        report.setdefault("arms", {})          # 保留已完成的未请求臂（防 --arms 子集覆写丢记录）
        print(f"[续跑] 已有报告载入：臂 {sorted(report['arms'])}", flush=True)
    else:
        report = {"arms": {}}
    report["references"] = {                   # ①④ 引用臂登记（不重跑）
        "coarse_aux": load_reference(CH03_FULL_JSON, "aux", REF_COARSE_AUX),
        "dense": load_reference(CH02_FULL_JSON, "dense", REF_DENSE),
    }
    report["meta"] = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "device": str(device),
        "torch": torch.__version__, "out_name": out_name,
        "experiment": "ch5 四臂消融（大纲 v1.1）：①粗E8aux(引用ch3) ②细E16/top4/共享1+aux(实跑) "
                      "③=②换noaux(实跑) ④等激活dense(引用ch2)——实跑=②③",
        "module_tier": f"d={D_MODEL}/L={N_LAYER}/h={N_HEAD}/h_kv={N_KV}/V={VOCAB}，B={BATCH}/T={BLOCK}"
                       f"（{TPB:,} tok/步，探针 B 已验配置；MoE=E{N_EXPERT}/top{TOP_K}/w{D_EXPERT}"
                       f"/共享{N_SHARED}）",
        "equal_activation": f"w=⌊{D_DENSE}/(top{TOP_K}+n_shared{N_SHARED})⌋={D_EXPERT}，每 token 激活 "
                            f"{(TOP_K + N_SHARED) * D_EXPERT}≈dense {D_DENSE}；激活参 {FINE_ACTIVE:,} vs "
                            f"dense {26_089_984:,}（差 {26_089_984 - FINE_ACTIVE:,}=0.11% 取整缺口，如实）",
        "balance_interventions": {
            "fine_aux": f"L_load = {AUX_W} · mean_layers CV(Load)²（与 ch3 aux 臂逐字同款，"
                        "Load=softmax 概率质量代理）",
            "fine_noaux": f"b_i ← b_i + {NOAUX_U}·sign(e_i) per-step（2408.15664 Alg1 正身；"
                          "e_i=1/E−本批该层被选比例（c̄_i−c_i，欠载抬/过载压）；bias 只进选择不进门值；"
                          "mini 档无 γ 衰减）"},
        "metrics_def": "度量用硬计数（钩子直捕 top-k idx——noaux 臂含 bias 的实际选择）：frac=count/(N·k)；"
                       "max/min 下限取 1（防死专家除零）；CV=std/mean（总体 std）；"
                       "MaxVio=max|frac−1/E|（noaux R35 口径，global=跨层聚合后）",
        "corpus": "log/book2-ch08/tokens.bin（sp-8k，306M token 流）",
        "optimizer": f"AdamW(betas=0.9,0.95,wd={WDECL}) + grad_clip 1.0（Book2 五件套）",
        "schedule": f"lr {PEAK_LR} 余弦降至 {PEAK_LR*MIN_LR_RATIO:.0e}，warmup={WARMUP}；两臂统一日程",
        "data_protocol": "顺序连续分块一次通过（moe_mla_slots.TokenStream）；两臂按步号取同批",
        "eval_protocol": f"每 {eval_every} 步+终步：eval.bin 固定 49 窗 {EVAL_BATCHES*TPB:,} token，fp32",
        "routing_probe_protocol": f"eval.bin 首段 {PROBE_B}×{PROBE_T}={PROBE_B*PROBE_T:,} token 固定探针，"
                                  "init + 每 eval 边界快照；长表 routing_{arm}_{out}.csv",
        "precision": "train bf16 autocast / eval fp32",
        "ckpt": "每 eval 边界存 ckpt（含 noaux bias buffer），臂完成即删；中断后重跑同命令自动续跑",
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
        print(f"  {a:10s} eval_final={r['eval_final']} max/min均值 "
              f"{r['layer_mean_max_over_min']['init']}→{r['layer_mean_max_over_min']['final']} "
              f"MaxVio_global={r['balance_final']['max_vio'] if r['balance_final'] else None} "
              f"{r['sec_per_step_steady']}s/step", flush=True)


if __name__ == "__main__":
    main()
