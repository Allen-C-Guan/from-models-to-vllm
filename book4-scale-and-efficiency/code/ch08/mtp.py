# mtp.py —— Book4 ch8 前置（P11）：DSV3 式 mini 多 token 预测（MTP）模块与训练件
# 用途：papers/05-MTP与LatentMoE.md §4.2 设计正身的落地——「让 26M 模块档主干多想一步棋」：
#       主干 = 模块档 dense 几何（d=512/L=6/h=8/h_kv=4/V=8192，探针 B 已验——26.1M）；
#       MTP 模块（D=1，DSV3 §2.2 Eq 21-23 教学式）= 共享 embedding + 共享 output head
#       + 两枚入模 RMSNorm（hnorm/enorm）+ M_1 投影 Linear(2d,d) + 一个与主干同型的 TRM 块
#       + 出口 RMSNorm——新增约 3.5M 档（实测 3,475,968，见参数账）。
#       联合损失（Eq 24-25，D=1）：L = CE_main + λ·CE_mtp，λ=0.3（DSV3 §4.2 前段；冒烟不做调度）。
#       off-by-one 三视图（本实验第一陷阱，正文显式教）：x_in=tokens[:,:-2]、y_main=tokens[:,1:-1]
#       （主支路 target=t+1）、y_mtp=tokens[:,2:]（MTP 支路 target=t+2），三者等长 T-2——
#       MTP 支路在位置 i 的输入嵌入 = tokens[i+1]（teacher forcing 真值，即 y_main[:,i]）。
# 结构对照（工业镜像）：与 vLLM deepseek_mtp.py 的 DeepSeekMultiTokenPredictorLayer 同型
#       （enorm/hnorm + 拼接投影 + TRM + 词表头）；mini用GQA+dense及末norm后主干状态，不能直载官方权重：
#       ① 拼接序本件取论文 Eq 21 的 [RMSNorm(h); RMSNorm(Emb)]，vLLM 实现为 [Emb; h]；
#       ② 本件共享 = 严格共享主干 embed_tokens/lm_head 两件（DSV3 §3.2.3 共享思想的最小版）。
# 协议口径（与 ch02/03/05 消融同款 Book2 五件套）：AdamW(0.9/0.95,wd=0.1)+clip 1.0
#       + lr 1e-3 余弦降 1e-4 + warmup 100；MPS train/eval bf16 autocast + fp32 CE；CPU fp32；tokens.bin 顺序分块；
#       eval固定49批×16窗，原序列512；本章每窗510个标签，不能与dense的512标签直接等同。
# 运行方式：source env.sh && python mtp.py --steps 500 --out-name fast [--lambdas 0.3,0.0]
#           （冒烟请用 smoke_mtp.py；本件 __main__ 为训练档：fast=500 步）。
# 产物（log/book4-ch08/，不入库）：mtp_{out}.json + curve_{arm}_{out}.csv + ckpt_{arm}_{out}.pt
#       （ckpt 每 eval 边界落盘、臂完成即删；中断后重跑同命令自动续跑）。
# 模块来源：不私搭模型——主干 SlotLLaMA/SlotDecoderLayer/TokenStream import 自正身 moe_mla_slots
#       （其内部 bootstrap 引 Book3 llama_slots：RMSNorm/build_rope_cache）。
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "00-feasibility")))
from moe_mla_slots import (SEED, SlotConfig, SlotDecoderLayer, SlotLLaMA, TokenStream,   # noqa: E402
                           bootstrap_book3, hand_account)

# ---------------- 固定口径（模块档 dense 几何） ----------------
REPO_ROOT = os.getcwd()  # 从工作区根目录运行，所有数据与产物统一落log/
OUT_DIR = os.path.join(REPO_ROOT, "log", "book4-ch08")
TOKENS_BIN = os.path.join(REPO_ROOT, "log", "book2-ch08", "tokens.bin")
EVAL_BIN = os.path.join(REPO_ROOT, "log", "book2-ch08", "eval.bin")

VOCAB, D_MODEL, N_LAYER, N_HEAD, N_KV, D_FF = 8192, 512, 6, 8, 4, 1408
BATCH, BLOCK = 16, 512
EVAL_BATCHES = 49                                                  # 固定49批×16窗，每支路399,840个评测标签
PEAK_LR, MIN_LR_RATIO, WARMUP, WDECL, TAIL = 1e-3, 0.1, 100, 0.1, 300
BACKBONE_PARAMS = 26_089_984                                       # 模块档 dense 锚点（探针 B）
MTP_LAMBDA_DEFAULT = 0.3                                           # DSV3 §4.2 前段（Eq 25，D=1）


def mtp_new_params():
    """MTP 模块新增参数手算账：3 枚 RMSNorm + M_1 投影(d·2d) + 一个同型 TRM 层。"""
    layer_cfg = SlotConfig(vocab_size=VOCAB, hidden_size=D_MODEL, num_hidden_layers=1,
                           num_attention_heads=N_HEAD, num_key_value_heads=N_KV,
                           intermediate_size=D_FF)
    trm = hand_account(layer_cfg)["layers"]                        # attn+mlp+层内 2 norm = 2,950,144
    return 3 * D_MODEL + D_MODEL * (2 * D_MODEL) + trm             # = 3,475,968（≈3.5M 档）


MTP_NEW_PARAMS = mtp_new_params()
TOTAL_PARAMS = BACKBONE_PARAMS + MTP_NEW_PARAMS                    # = 29,565,952


def three_views(tokens):
    """三视图切片（off-by-one 正身）：tokens (B,T) -> x_in/y_main/y_mtp 各 (B,T-2)。

    x_in   = tokens[:, :-2]   位置 0..T-3：主干输入（主干只见「已确定」的上下文）
    y_main = tokens[:, 1:-1]  主支路目标（位置 i 的 target = t_{i+1}）
    y_mtp  = tokens[:, 2:]    MTP 支路目标（位置 i 的 target = t_{i+2}——多想的那一步棋）
    """
    if tokens.ndim != 2 or tokens.shape[0] < 1 or tokens.shape[1] < 3:
        raise ValueError("tokens必须为非空(B,T)，且T>=3")
    return tokens[:, :-2], tokens[:, 1:-1], tokens[:, 2:]


def assert_off_by_one(tokens):
    """三视图核验（内置断言，冒烟/训练前必过）：等长 + 互相恰错一位 + 单点抽样对齐。"""
    x_in, y_main, y_mtp = three_views(tokens)
    B, T = tokens.shape
    assert x_in.shape == y_main.shape == y_mtp.shape == (B, T - 2), "三视图不等长（应各为 T-2）"
    assert torch.equal(y_main[:, :-1], x_in[:, 1:]), "y_main 应为 x_in 左移一位"
    assert torch.equal(y_mtp[:, :-1], y_main[:, 1:]), "y_mtp 应为 y_main 左移一位"
    i = min(T // 2, T - 3)                                                     # 单点抽验：位置 i 的两支路目标
    assert bool(y_main[:, i].eq(tokens[:, i + 1]).all()), "y_main[i] 应 = tokens[i+1]"
    assert bool(y_mtp[:, i].eq(tokens[:, i + 2]).all()), "y_mtp[i] 应 = tokens[i+2]"
    return x_in, y_main, y_mtp


class MTPModule(nn.Module):
    """DSV3 §2.2 Eq 21-23 的教学式 MTP 模块（D=1，深度维只此一个——「串行保因果链」的最浅档）。

    数据流（形状流转表，n=T-2）：
      h (B,n,d) 主干终态（已含终态 RMSNorm）
      e = Emb(y_main) (B,n,d)        teacher forcing 真值 t_{i+1} 的嵌入（共享主干 embedding）
      cat([hnorm(h); enorm(e)]) (B,n,2d)   Eq 21 的拼接（序 = 论文原式 [h; Emb]）
      proj M_1: (2d)->(d)  (B,n,d)
      TRM 块（同型 SlotDecoderLayer，含 RoPE/因果 mask）(B,n,d)   Eq 22
      出口 RMSNorm -> 共享 lm_head (B,n,V)                        Eq 23
    共享件零新增：embedding 与 lm_head 由 forward 传入（主干对象本体——共享写在调用点，可审计）。
    """

    def __init__(self, hidden_size=D_MODEL):
        super().__init__()
        slots = bootstrap_book3()
        d = hidden_size
        if d % N_HEAD or (d // N_HEAD) % 2:
            raise ValueError("hidden_size须使每头维度为偶数")
        self.head_dim = d // N_HEAD
        self.hnorm = slots.RMSNorm(d, 1e-5)                        # RMSNorm(h_i)：主干表征入模归一
        self.enorm = slots.RMSNorm(d, 1e-5)                        # RMSNorm(Emb(t_{i+1}))：真值嵌入归一
        self.proj = nn.Linear(2 * d, d, bias=False)                # M_1 ∈ R^{d×2d}（Eq 21 投影）
        layer_cfg = SlotConfig(vocab_size=VOCAB, hidden_size=d, num_hidden_layers=1,
                               num_attention_heads=N_HEAD, num_key_value_heads=N_KV,
                               intermediate_size=D_FF)
        self.trm = SlotDecoderLayer(layer_cfg, 0)                  # TRM_1：与主干同型的插槽层
        self.norm = slots.RMSNorm(d, 1e-5)                         # 出口 RMSNorm（Head 前的最后一 norm）
        self._rope_cache = {}

        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        self.apply(_init)

    def _rope(self, n, device):
        if (n, str(device)) not in self._rope_cache:
            self._rope_cache[(n, str(device))] = bootstrap_book3().build_rope_cache(
                n, self.head_dim, 10000.0, device)             # (1,1,n,head_dim) ×2
        return self._rope_cache[(n, str(device))]

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, h, next_tok, embed, lm_head):
        """h (B,n,d) 主干终态；next_tok (B,n) = y_main（teacher forcing 的 t_{i+1}）；
        embed/lm_head = 主干共享件。返回 logits_mtp (B,n,V)。"""
        if h.ndim != 3 or next_tok.shape != h.shape[:2] or h.shape[-1] != self.hnorm.weight.numel():
            raise ValueError("h与next_tok的批次、长度或隐藏维不匹配")
        e = embed(next_tok)                                        # (B,n,d) 共享 embedding
        x = torch.cat([self.hnorm(h), self.enorm(e)], dim=-1)      # (B,n,2d) Eq 21 拼接
        x = self.proj(x)                                           # (B,n,d)
        cos, sin = self._rope(x.shape[1], x.device)
        x = self.trm(x, cos, sin)                                  # (B,n,d) Eq 22（TRM 内含因果 mask）
        return lm_head(self.norm(x))                               # (B,n,V) Eq 23（共享 head）


class MTPSlotLLaMA(nn.Module):
    """主干（模块档 dense SlotLLaMA 正身）+ 一个 MTP 模块的联合整机。

    前向返回 (logits_main, logits_mtp)，各 (B,T-2,V)；两支路目标由 three_views 切出。
    共享件：backbone.model.embed_tokens / backbone.lm_head 同一对象进出两支路（零新增参数）。
    """

    def __init__(self):
        super().__init__()
        cfg = SlotConfig(vocab_size=VOCAB, hidden_size=D_MODEL, num_hidden_layers=N_LAYER,
                         num_attention_heads=N_HEAD, num_key_value_heads=N_KV,
                         intermediate_size=D_FF)
        self.backbone = SlotLLaMA(cfg)
        self.mtp = MTPModule(D_MODEL)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())           # parameters() 去重——共享件只计一次

    def forward(self, tokens):
        x_in, y_main, _ = three_views(tokens)                      # (B,T-2) ×2
        h = self.backbone.model(x_in)                              # (B,T-2,d)（含终态 RMSNorm）
        logits_main = self.backbone.lm_head(h)                     # (B,T-2,V)
        embed, lm_head = self.backbone.model.embed_tokens, self.backbone.lm_head
        logits_mtp = self.mtp(h, y_main, embed, lm_head)           # (B,T-2,V)
        return logits_main, logits_mtp


def joint_loss(logits_main, logits_mtp, tokens, lam=MTP_LAMBDA_DEFAULT):
    """L = CE_main + λ·CE_mtp（Eq 24-25，D=1；λ=0 即 λ 对照臂——CE_mtp 仍记录不进损失）。"""
    _, y_main, y_mtp = three_views(tokens)
    if not math.isfinite(lam) or lam < 0:
        raise ValueError("lambda须有限且非负")
    if logits_main.ndim != 3 or logits_main.shape != logits_mtp.shape or logits_main.shape[:2] != y_main.shape:
        raise ValueError("两支路logits必须与三视图逐位对齐")
    vocab = logits_main.shape[-1]
    ce_main = F.cross_entropy(logits_main.reshape(-1, vocab).float(), y_main.reshape(-1))
    ce_mtp = F.cross_entropy(logits_mtp.reshape(-1, vocab).float(), y_mtp.reshape(-1))
    return ce_main + lam * ce_mtp, ce_main, ce_mtp


# ---------------- 训练环（Book2 五件套，与 ch02/03/05 同款） ----------------
def lr_at(step, total):
    if step <= WARMUP:
        return PEAK_LR * step / WARMUP
    prog = (step - WARMUP) / max(1, total - WARMUP)
    return PEAK_LR * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * prog)))


def build_eval_batches(device):
    """沿用相同起点采样种子；每窗只保留510个监督位置，不能当作dense的完整评测集。"""
    ev = np.memmap(EVAL_BIN, dtype=np.uint16, mode="r")
    gen = torch.Generator().manual_seed(SEED)
    offs = torch.randint(0, len(ev) - BLOCK - 1, (EVAL_BATCHES * BATCH,), generator=gen).numpy()
    xs = []
    for o in offs:
        xs.append(np.asarray(ev[o:o + BLOCK], dtype=np.int64))
    return torch.from_numpy(np.stack(xs).reshape(EVAL_BATCHES, BATCH, BLOCK)).to(device)   # (49,16,512)


@torch.no_grad()
def eval_loss(model, ex):
    """返回主/MTP批均值及批间标准误；MPS前向BF16、CE为FP32，不表示种子方差。"""
    was_training = model.training
    model.eval()
    mains, mtps = [], []
    for i in range(len(ex)):
        with torch.autocast(device_type=ex.device.type, dtype=torch.bfloat16) \
                if ex.device.type == "mps" else torch.autocast("cpu", enabled=False):
            logits_main, logits_mtp = model(ex[i])
        _, ce_main, ce_mtp = joint_loss(logits_main, logits_mtp, ex[i], lam=0.0)
        mains.append(ce_main.item())
        mtps.append(ce_mtp.item())
    model.train(was_training)

    def stat(v):
        m = sum(v) / len(v)
        se = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 / math.sqrt(len(v)) if len(v) > 1 else 0.0
        return round(m, 5), round(se, 5)

    return stat(mains), stat(mtps)


def run_arm(arm, lam, steps, eval_every, device, stream, ex, report, out_json):
    """一个 λ 臂的五件套训练环：ckpt 断点续跑（eval 边界存档），双支路曲线 CSV。"""
    ckpt_path = os.path.join(OUT_DIR, f"ckpt_{arm}_{report['meta']['out_name']}.pt")
    torch.manual_seed(SEED)                                        # 两 λ 臂同种子同构序——初始权重逐位相同
    m = MTPSlotLLaMA().to(device)
    assert m.n_params() == TOTAL_PARAMS, f"参数对账失败 {m.n_params()} != {TOTAL_PARAMS}"
    assert m.backbone.n_params() == BACKBONE_PARAMS, "主干应=模块档 dense 锚点"
    assert m.mtp.n_params() == MTP_NEW_PARAMS, "MTP 模块新增参数与手算不符"
    opt = torch.optim.AdamW(m.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=WDECL)

    main_losses, mtp_losses, lrs, times, evals = [], [], [], [], []
    start_step, elapsed_train, elapsed_eval = 0, 0.0, 0.0
    resumed_from = None
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck.get("run_contract") != report["run_contract"] or ck.get("arm") != arm or ck.get("lambda") != lam:
            raise ValueError("checkpoint的日程、数据、实现或lambda与本次不符")
        m.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        main_losses, mtp_losses = ck["main_losses"], ck["mtp_losses"]
        lrs, times, evals = ck["lrs"], ck["times"], ck["evals"]
        start_step, elapsed_train, elapsed_eval = ck["step"], ck["elapsed_train"], ck["elapsed_eval"]
        resumed_from = start_step
        print(f"  [{arm} 续跑] 自 step {start_step + 1}", flush=True)

    m.train()
    diverged_at, reason = None, None
    t_arm = time.perf_counter()
    for step in range(start_step + 1, steps + 1):
        lr = lr_at(step, steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = stream.batch(step, device)                          # (16,512)×2
        tokens = torch.cat([x[:, :1], y], dim=1)                   # (B,BLOCK+1)，每支路BLOCK-1个监督位
        assert_off_by_one(tokens)                                  # O(BT)检查；同步开销未计入下方步时
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "mps" \
                else torch.autocast("cpu", enabled=False):
            logits_main, logits_mtp = m(tokens)
            loss, ce_main, ce_mtp = joint_loss(logits_main, logits_mtp, tokens, lam=lam)
        if not torch.isfinite(loss) or ce_main.item() > 100:
            raise FloatingPointError("损失非有限或超过预设上限，停止更新")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        if device.type == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
        elapsed_train += times[-1]
        lv, mv = ce_main.item(), ce_mtp.item()
        main_losses.append(lv)
        mtp_losses.append(mv)
        lrs.append(lr)
        if not math.isfinite(lv) or not math.isfinite(mv) or lv > 100.0:
            diverged_at, reason = step, (f"non-finite(main={lv}, mtp={mv})" if not math.isfinite(lv)
                                         else f"loss={lv:.1f}>100")
            print(f"  [{arm} 发散] step {step}: {reason}", flush=True)
            break
        if step % 100 == 0 or step == 1:
            print(f"  [{arm:6s}] step {step:5d}/{steps} main {lv:.4f} mtp {mv:.4f} lr {lr:.2e} "
                  f"({elapsed_train / (step - start_step):.3f}s/step)", flush=True)
        if step % eval_every == 0 or step == steps:
            t0 = time.perf_counter()
            (em, em_se), (et, et_se) = eval_loss(m, ex)
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed_eval += time.perf_counter() - t0
            evals.append({"step": step, "eval_main": em, "eval_main_se": em_se,
                          "eval_mtp": et, "eval_mtp_se": et_se,
                          "n_tokens": EVAL_BATCHES * BATCH * (BLOCK - 2)})   # 每窗有效监督位 = 510
            print(f"  [{arm:6s}] [eval] step {step}: main {em:.4f} ± {em_se:.4f} | "
                  f"mtp {et:.4f} ± {et_se:.4f}", flush=True)
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": step,
                        "run_contract": report["run_contract"], "arm": arm, "lambda": lam,
                        "main_losses": main_losses, "mtp_losses": mtp_losses, "lrs": lrs,
                        "times": times, "evals": evals,
                        "elapsed_train": elapsed_train, "elapsed_eval": elapsed_eval}, ckpt_path)

    steps_done = len(main_losses)
    steady = times[20:] if len(times) > 20 else times
    tail = main_losses[-min(TAIL, steps_done):]
    rec = {
        "arm": arm,
        "arm_def": f"λ={lam}（L = CE_main + {lam}·CE_mtp；λ=0 为对照臂——MTP 支路前向与记录保留、"
                   "梯度权重为零）",
        "config": {"d": D_MODEL, "L": N_LAYER, "h": N_HEAD, "h_kv": N_KV, "V": VOCAB,
                   "B": BATCH, "T_train": BLOCK + 1, "T_eval": BLOCK,
                   "T_eff_train": BLOCK - 1, "T_eff_eval": BLOCK - 2, "lambda": lam, "mtp_depth": 1,
                   "mtp_module": "shared emb + shared head + {hnorm,enorm,proj 2d→d, TRM, exit norm}"},
        "params_total": TOTAL_PARAMS, "params_backbone": BACKBONE_PARAMS,
        "params_mtp_new": MTP_NEW_PARAMS, "params_mtp_overhead_ratio": round(MTP_NEW_PARAMS / BACKBONE_PARAMS, 4),
        "steps_done": steps_done, "steps_planned": steps, "resumed_from_step": resumed_from,
        "tail300_main_loss": round(sum(tail) / len(tail), 5),
        "evals": evals,
        "eval_main_final": evals[-1]["eval_main"] if evals else None,
        "eval_mtp_final": evals[-1]["eval_mtp"] if evals else None,
        "main_loss_first10_mean": round(sum(main_losses[:10]) / len(main_losses[:10]), 5) if main_losses else None,
        "main_loss_last10_mean": round(sum(main_losses[-10:]) / len(main_losses[-10:]), 5) if main_losses else None,
        "mtp_loss_first10_mean": round(sum(mtp_losses[:10]) / len(mtp_losses[:10]), 5) if mtp_losses else None,
        "mtp_loss_last10_mean": round(sum(mtp_losses[-10:]) / len(mtp_losses[-10:]), 5) if mtp_losses else None,
        "sec_per_step_steady": round(sum(steady) / len(steady), 4),
        "elapsed_train_sec": round(elapsed_train, 1), "elapsed_eval_sec": round(elapsed_eval, 1),
        "wallclock_sec": round(time.perf_counter() - t_arm, 1),
        "diverged": diverged_at is not None, "diverged_at_step": diverged_at,
        "divergence_reason": reason,
        "precision": "MPS train/eval bf16 autocast + fp32 CE；CPU fp32",
    }
    curve_path = os.path.join(OUT_DIR, f"curve_{arm}_{report['meta']['out_name']}.csv")
    with open(curve_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["step", "main_loss", "mtp_loss", "lr"])
        for i, (lv, mv, lr) in enumerate(zip(main_losses, mtp_losses, lrs), 1):
            w.writerow([i, f"{lv:.6g}", f"{mv:.6g}", f"{lr:.6g}"])
    report["arms"][arm] = rec
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if steps_done >= steps and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"  [{arm} 完成] steps {steps_done} | eval_main_final {rec['eval_main_final']} | "
          f"eval_mtp_final {rec['eval_mtp_final']} | {rec['sec_per_step_steady']}s/step", flush=True)
    del m, opt
    if device.type == "mps":
        torch.mps.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser(description="P11 mini MTP 探针：DSV3 Eq21-23 教学式（D=1，λ 两臂）")
    ap.add_argument("--steps", type=int, default=500, help="每臂步数（fast=500）")
    ap.add_argument("--out-name", type=str, default=None)
    ap.add_argument("--lambdas", type=str, default="0.3,0.0", help="λ 臂（主臂 0.3 + 对照 0.0）")
    ap.add_argument("--eval-every", type=int, default=None, help="eval 间隔（默认 max(100, steps//5)）")
    args = ap.parse_args()
    if args.steps < 1 or (args.eval_every is not None and args.eval_every < 1):
        ap.error("steps和eval-every必须为正")
    lambdas = [float(s.strip()) for s in args.lambdas.split(",")]
    if not lambdas or any(not math.isfinite(v) or v < 0 for v in lambdas):
        ap.error("lambdas必须有限且非负")
    arms = [f"lam{int(round(v * 100)):02d}" for v in lambdas]
    if len(arms) != len(set(arms)):
        ap.error("lambda生成了重复结果名；请分别运行并用独立out-name")
    out_name = args.out_name or ("fast" if args.steps <= 1000 else "full")
    if os.path.basename(out_name) != out_name or out_name in {".", ".."}:
        ap.error("out-name须为单个文件名后缀")
    eval_every = args.eval_every or max(100, args.steps // 5)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    stream = TokenStream(TOKENS_BIN, BATCH, BLOCK)
    if args.steps > stream.max_step:
        raise SystemExit(f"步数 {args.steps} 超出一次通过上限 {stream.max_step}")
    ex = build_eval_batches(device)
    out_json = os.path.join(OUT_DIR, f"mtp_{out_name}.json")

    def file_id(path):
        st = os.stat(path)
        return {"path": os.path.realpath(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    contract = {"version": 1, "steps": args.steps, "eval_every": eval_every,
                "seed": SEED, "batch": BATCH, "block": BLOCK, "eval_batches": EVAL_BATCHES,
                "geometry": [VOCAB,D_MODEL,N_LAYER,N_HEAD,N_KV,D_FF],
                "lambdas": lambdas, "train_data": file_id(TOKENS_BIN), "eval_data": file_id(EVAL_BIN),
                "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
                "core_sha256": hashlib.sha256(open(sys.modules[SlotLLaMA.__module__].__file__, "rb").read()).hexdigest(),
                "slots_sha256": hashlib.sha256(open(bootstrap_book3().__file__, "rb").read()).hexdigest(),
                "torch": torch.__version__, "device": str(device)}
    if os.path.exists(out_json):
        with open(out_json, encoding="utf-8") as f:
            report = json.load(f)
        if report.get("run_contract") != contract:
            raise ValueError("已有报告与本次运行约定不符，请换out-name")
        report.setdefault("arms", {})
        print(f"[续跑] 已有报告载入：臂 {sorted(report['arms'])}", flush=True)
    else:
        if any(os.path.exists(os.path.join(OUT_DIR, f"{kind}_{arm}_{out_name}.{ext}"))
               for arm in arms for kind,ext in [("curve","csv"),("ckpt","pt")]):
            raise FileExistsError("曲线或断点存在但报告缺失，请换out-name")
        report = {"arms": {}, "run_contract": contract}
    report["meta"] = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "device": str(device),
        "torch": torch.__version__, "out_name": out_name,
        "experiment": "P11 前置：mini MTP 探针（DSV3 §2.2 Eq21-23 教学式，D=1）——"
                      "联合训练对主支路的影响（历史判据预注册notes/07 §3.3；不作跨论文规模曲线验证）",
        "backbone": f"模块档 dense 几何 d={D_MODEL}/L={N_LAYER}/h={N_HEAD}/h_kv={N_KV}/V={VOCAB}"
                    f"（探针 B 已验；总参 {BACKBONE_PARAMS:,}）",
        "mtp_module": f"新增 {MTP_NEW_PARAMS:,}（3 枚 RMSNorm {3 * D_MODEL} + proj {D_MODEL * 2 * D_MODEL}"
                      f" + TRM 层 {MTP_NEW_PARAMS - 3 * D_MODEL - D_MODEL * 2 * D_MODEL}）= "
                      f"+{MTP_NEW_PARAMS / BACKBONE_PARAMS:.1%}；共享主干 embedding 与 lm_head",
        "off_by_one": "x_in=tokens[:,:-2] / y_main=tokens[:,1:-1] / y_mtp=tokens[:,2:]（各 T-2）；"
                      "MTP 支路位置 i 的输入嵌入 = tokens[i+1]（teacher forcing）；断言内置每步训练前",
        "corpus": "log/book2-ch08/tokens.bin（sp-8k，306M token 流）；每步每序列511个监督位；评测每窗510个",
        "optimizer": f"AdamW(betas=0.9,0.95,wd={WDECL}) + grad_clip 1.0（Book2 五件套）",
        "schedule": f"lr {PEAK_LR} 余弦降至 {PEAK_LR*MIN_LR_RATIO:.0e}，warmup={WARMUP}；两 λ 臂统一日程",
        "eval_protocol": f"每 {eval_every} 步+终步：eval.bin固定49批×16窗、每窗510标签，"
                         "MPS前向bf16/CE fp32，CPU全fp32，主/MTP双支路",
        "references": {"dense_full_eval_at_500": 5.7858, "dense_full_eval_final": 4.93375,
                       "note": "ch2 dense 臂 full 轨迹参照（不同日程长度，只作量级参照不作判据）；"
                               "判据用同日程 λ=0 对照臂（notes/07 §3.3）"},
        "precision": "MPS train/eval bf16 autocast + fp32 CE；CPU fp32",
        "ckpt": "每 eval 边界存 ckpt，臂完成即删；中断后重跑同命令自动续跑",
        "argv": " ".join(sys.argv[1:]),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    t_all = time.perf_counter()
    for lam, arm in zip(lambdas, arms):
        done = report["arms"].get(arm)
        if done and done["steps_done"] >= args.steps:
            print(f"[跳过] 臂 {arm} 已完成（{done['steps_done']} 步）", flush=True)
            continue
        print(f"\n===== 臂 {arm}（λ={lam}，{args.steps} 步，eval_every={eval_every}） =====", flush=True)
        run_arm(arm, lam, args.steps, eval_every, device, stream, ex, report, out_json)
    a, b = report["arms"].get("lam30"), report["arms"].get("lam00")
    if a and b and a["config"]["lambda"] == 0.3 and b["config"]["lambda"] == 0.0 \
            and a["eval_main_final"] is not None and b["eval_main_final"] is not None:
        report["pair_summary"] = {
            "eval_main_diff_lam03_minus_lam00": round(a["eval_main_final"] - b["eval_main_final"], 5),
            "read": "正 = 挂 MTP 拖累主支路；0.05 nat为排障阈值，不是统计显著性或跨论文规模反转判据；"
                    ">0.05 才排障（notes/07 §3.3 预注册）",
            "mtp_branch_above_main_always": None,   # 正文核验用：曲线 CSV 里逐点核
        }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[全部完成] 总耗时 {(time.perf_counter()-t_all)/60:.1f} min -> {out_json}", flush=True)
    for a_name, r in report["arms"].items():
        print(f"  {a_name:6s} eval_main={r['eval_main_final']} eval_mtp={r['eval_mtp_final']} "
              f"{r['sec_per_step_steady']}s/step", flush=True)


if __name__ == "__main__":
    main()
