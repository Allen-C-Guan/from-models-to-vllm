# hybrid_recall.py —— Book5 ch9 正式件：混合比例 recall 消融（全全局 vs 3:1 vs 全线性，等参三臂）
# 用途：ch9 9.6「线性:全局 3:1 的收益」正式实验——P18 管道（探针07 验证：hybrid31 臂 L=4，
#   7,750,936 参，300 步 loss 9.09→4.18、0.212 s/step）升格为等参三臂 6000 步正档。
# 【等参对齐口径（本件定版，FFN 宽度粒度内最近对齐）】
#   参考臂 = P18 hybrid31 探针臂原封不动（d=256/L=4/d_ff=704/vocab 8192/FullAttn(h=8,h_kv=4,d_k=64,
#   无 rope)+GDNLayer(h_k=h_v=4,d_k=d_v=64)）→ 总参 7,750,936（锚）。
#   补偿方向（探针07 caveat 勘误：本几何下 FullAttn 插槽 393,216 > GDN 插槽 332,808——
#   「all_global 的 FullAttn < GDN」为探针期笔误，正式件以逐位枚举为准）：
#     all_global：注意力多 4×(393,216−332,808)−(393,216−332,808)=+181,224 → FFN 减 59 单位（704→645）
#                 → 7,750,912（−24 = −0.0003%）
#     all_linear：注意力少 60,408 → FFN 加 20 单位（704→724）→ 7,751,968（+1,032 = +0.013%）
#   FFN 宽度粒度 = 3·d = 768 参/单位（×4 层 = 3,072）；三臂最大互差 1,056 参 = 0.0136%（assert 上限）。
#   等激活说明：三臂均无 MoE（FFN 稠密全激活、注意力全激活）——同口径活跃参数近似相等，但计算量和时延不一定相等（probe07 caveat 闭合）。
# 【判据预注册（跑前写定 2026-10-08，防事后择优；负结论照报——notes/08 同文留档）】
#   任务：MQAR needle-recall（探针02 v4 定版：T=96/B=32/监督 8 查询位/needle 训练分布 50% 远端
#   [16,60)+50% 窗内 [62,78)）；配方：AdamW(0.9,0.95,wd0.1)+clip1.0+lr 3e-3+warmup 50+cosine，
#   6000 步（大纲 6000-10000 下限——保底降档线），seed 20261002 全书统一种子。
#   主判据：
#     J1 all_global 学会全位置召回：五位置（18/42/58/62/74）acc 全 ≥ 0.7；
#     J2 hybrid31 凭 1 个全局层提供可达性也学会：远端三位置（18/42/58）acc 全 ≥ 0.7
#        （只检验本玩具配方，不验证旗舰）；
#     J3 all_linear远端三位置均值比hybrid31低至少0.3；达标本身不证明容量/衰减为原因。
#   结果分支（预登记，均不算管道失败）：
#     B1 若 hybrid31 意外垫底（五位置均值 < all_linear）→记录异常排序；须再查优化/任务/层位置，不据此确认机制；
#     B2 若 all_linear 远端三位置也 ≥ 0.7 → 本任务中达到阈值，不证明任意容量均足够
#        （候选解释：容量/优化/数据均需检查；chunkwise与递推计算同一函数，不额外增加记忆能力）；
#     B3 若 all_global 未达 J1 → 6000 步尚未达标，预算不足只是候选原因，补 10000 步档再判（不据此下结论）。
#   ruler-mini（次判据，长度外推苗头——训练只见 T=96，外推读数不下终局结论）：
#     T ∈ {96,128,160,224} × needle 位 frac ∈ {0.2,0.5,0.8}（usable [16,T-18) 内取位），报各臂 acc 表；
#     预期苗头：无 rope 下 full 臂随 T 缓降、GDN 臂衰减更快（卷积局部性）——只报方向。
# 运行方式：cd 工作区根目录 && source env.sh &&
#   python "code/ch09/hybrid_recall.py" --out-name run1          # 三臂串行正式档
#   python "code/ch09/hybrid_recall.py" --arm hybrid31 --steps 300 --out-name pipe2
#   python "code/ch09/hybrid_recall.py" --eval ruler-mini --out-name run1
# 产物（log/book5-ch09/，不入库）：recall3_{arm}_{out}.json（逐臂，跑完即写——中断不丢已完臂）+
#   recall3_{out}.json（总表）+ ckpt recall_ckpt_{arm}_{out}.pt（每 1000 步——单臂贴 25 min 窗分段续跑）。
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = str(next((p for p in Path(__file__).resolve().parents if (p/"env.sh").is_file()),Path.cwd()))
FEAS = os.path.join(HERE, "..", "00-feasibility")       # 本册件 HERE 相对——工作区/书仓双布局同构
OUT_DIR = os.path.join(REPO_ROOT, "log", "book5-ch09")
SEED = 20261002                       # 同种子协议；不同结构消耗随机数不同，不表示共用权重

_B4_FEAS_CANDS = [str(p/"book4-scale-and-efficiency/code/00-feasibility") for p in Path(__file__).resolve().parents]
_B4_FEAS_CANDS += [
    os.path.join(REPO_ROOT,"from-models-to-vllm/book4-scale-and-efficiency/code/00-feasibility"),
    os.path.join(REPO_ROOT,"code/Book4-规模与效率/00-feasibility"),
]


def bootstrap():
    """挂 Book4 moe_mla_slots（RMSNorm/SwiGLU 底座）+ 本册 00-feasibility 探针件（02 管道/05 GDN 插槽）。"""
    for p in _B4_FEAS_CANDS:
        if os.path.exists(os.path.join(p, "moe_mla_slots.py")):
            if p not in sys.path:
                sys.path.insert(0, p)
            break
    else:
        raise FileNotFoundError(_B4_FEAS_CANDS)
    import moe_mla_slots as m4
    return m4


def load_feas(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(FEAS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod


# ---------------- 等参三臂构建（FFN 宽度粒度内最近对齐；逐位报数） ----------------
D, L, VOCAB = 256, 4, 8192
FULL_PARAMS = 393_216                  # FullAttn(d=256,h=8,h_kv=4,d_k=64) 插槽逐位枚举
GDN_PARAMS = 332_808                   # GDNLayer(d=256,h_k=h_v=4,d_k=d_v=64) 插槽逐位枚举
REF_TOTAL = 7_750_936                  # P18 hybrid31 探针臂总参（锚——原封不动）
FFN_UNIT = 3 * D                       # 768 参/宽度单位；4 层同改 → 3,072 参/单位
MAX_INTER_GAP = 1_056                  # 三臂最大互差上限（0.0136%）——assert

ARM_SPECS = {                          # kind -> (注意力布局描述, d_ff 由下方 solve 定版)
    "all_global": "L=4 全 FullAttn（无 rope）",
    "hybrid31": "3 GDN + 1 FullAttn@层3（index 0 起——interval=4 同构 P18）",
    "all_linear": "L=4 全 GDN",
}


def arm_attention_params(kind):
    n_full = {"all_global": 4, "hybrid31": 1, "all_linear": 0}[kind]
    return n_full * FULL_PARAMS + (L - n_full) * GDN_PARAMS


def solve_d_ff(kind):
    """FFN 宽度最近对齐：总参(d_ff) = BASE + 4·768·d_ff；BASE = 嵌入双份+终态 norm+4 层双 norm(8D)+注意力。"""
    base = VOCAB * D * 2 + D + 8 * D + arm_attention_params(kind)
    return round((REF_TOTAL - base) / (4 * FFN_UNIT))


def build_arm(p02, p05, kind, device):
    """构建一等参臂（同 seed 同初始化协议：MiniLM 构造器内 std=0.02 统一）。"""
    d_ff = solve_d_ff(kind)
    attns = []
    for i in range(L):
        if kind == "all_global" or (kind == "hybrid31" and (i + 1) % 4 == 0):
            attns.append(p02.FullAttn(D, 8, 4, 64, use_rope=False))
        else:
            attns.append(p05.GDNLayer(D, h_k=4, h_v=4, d_k=64, d_v=64))
    m = p02.MiniLM(d=D, n_layer=L, d_ff=d_ff, vocab=VOCAB, attn_modules=attns).to(device)
    n = sum(p.numel() for p in m.parameters())
    return m, n, d_ff


# ---------------- ruler-mini 评估（长度外推——训练分布 T=96 之外的读数） ----------------
@torch.no_grad()
def ruler_mini_eval(p02, model, device, n_seq=60, seed=SEED + 999):
    """RULER-mini：T×frac 网格 needle 召回。usable 区 [16, T-18) 内按 frac 取 needle 位；
    acc = argmax(logits[:, T-2]) == 应答 token（与 needle_eval 同判位）。"""
    model.eval()
    rng = np.random.default_rng(seed)
    out = {}
    for T in (96, 128, 160, 224):
        for frac in (0.2, 0.5, 0.8):
            pos = 16 + int(frac * (T - 18 - 16))
            accs = []
            for _ in range(n_seq // 12):
                x, _ = p02.make_needle_batch(12, T, rng, p_min=pos, device=device)
                pred = model(x)[0][:, T - 2].argmax(dim=-1)          # (12,) 查询位预测
                accs.append((pred == x[:, T - 1]).float().mean().item())
            out[f"T{T}_f{frac}_p{pos}"] = float(np.mean(accs))
    model.train()
    return out


# ---------------- 训练环（P18 train_mini 同配方 + cosine + ckpt 续跑） ----------------
def train_arm(p02, model, kind, steps, device, lr=3e-3, warmup=50, out_name='run1', resume=False):
    """一臂训练：AdamW(0.9,0.95,wd0.1)+clip1.0+warmup+cosine；每 1000 步 ckpt（单臂贴 25 min 窗分段续跑）。"""
    ckpt_path = os.path.join(OUT_DIR, f"recall_ckpt_{kind}_{out_name}.pt")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    rng = np.random.default_rng(SEED + 7)
    start_step = 1
    if os.path.exists(ckpt_path) and not resume:raise FileExistsError("已有checkpoint；显式--resume或换输出名")
    if resume and not os.path.exists(ckpt_path):raise FileNotFoundError(ckpt_path)
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck.get("protocol")!="book5-r2-recall-v1" or ck.get("kind")!=kind:
            raise ValueError("checkpoint来源/版本不符；历史checkpoint不得静默混用")
        if ck.get("target_steps")!=steps:
            print(f"  [{kind}] 目标由{ck.get('target_steps')}改为{steps}；重新计算cosine，不等于原长日程")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        if "torch_rng" in ck:torch.set_rng_state(ck["torch_rng"].cpu())
        rng.bit_generator.state = ck["rng"]
        if ck["step"] < steps:
            start_step = ck["step"] + 1
            print(f"  [{kind}] 续跑自 ckpt 第 {ck['step']} 步", flush=True)
        else:
            print(f"  [{kind}] ckpt 已达 {ck['step']} 步 ≥ 目标 {steps}（复用末态）", flush=True)
            if ck["step"]>steps:raise ValueError("checkpoint超出目标步数")
            return ck.get("losses",[]),ck.get("secs",[]),ck["step"]+1
    model.train()
    losses, secs = [], []
    for step in range(start_step, steps + 1):
        sched = min(1.0, step / warmup)
        sched *= 0.5 * (1 + math.cos(math.pi * step / steps))        # cosine 至 0
        opt.param_groups[0]["lr"] = lr * sched
        x, p = p02.make_needle_batch(32, 96, rng, device=device)
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "mps" \
                else torch.autocast("cpu", enabled=False):
            loss = p02.needle_loss(model, x, p, 96)
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):raise FloatingPointError(f"非有限loss step={step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        if device.type == "mps":
            torch.mps.synchronize()
        secs.append(time.perf_counter() - t0)
        losses.append(loss.item())
        if step % 200 == 0:
            print(f"    [{kind}] step {step}/{steps} loss {loss.item():.4f} "
                  f"({np.median(secs[-50:]) * 1000:.0f} ms/step)", flush=True)
        if step % 1000 == 0 or step == steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "rng": rng.bit_generator.state, "losses": losses, "secs": secs,
                        "torch_rng":torch.get_rng_state(), "protocol":"book5-r2-recall-v1",
                        "kind":kind, "target_steps":steps}, ckpt_path)
    return losses, secs, start_step


def run_arm(p02, p05, kind, steps, device, out_name, resume=False):
    saved=Path(OUT_DIR)/f"recall3_{kind}_{out_name}_s{steps}.json"
    if saved.exists():
        if not resume:raise FileExistsError(saved)
        prior=json.loads(saved.read_text())
        if prior.get("protocol")!="book5-r2-recall-v1" or prior["kind"]!=kind or prior["steps"]!=steps:
            raise ValueError("已有单臂记录版本/范围不符")
        return prior
    t0 = time.perf_counter()
    torch.manual_seed(SEED)
    model, n_par, d_ff = build_arm(p02, p05, kind, device)
    ref_dev = n_par - REF_TOTAL
    print(f"  [{kind}] {ARM_SPECS[kind]} | d_ff={d_ff} | 总参 {n_par:,}（vs 锚 {ref_dev:+d}）", flush=True)
    losses, secs, resumed = train_arm(p02, model, kind, steps, device, out_name=out_name,
        resume=resume and (Path(OUT_DIR)/f"recall_ckpt_{kind}_{out_name}.pt").exists())
    positions = [18, 42, 58, 62, 74]                                  # P18 五位置（probe07 同款）
    scan = p02.needle_eval(model, 96, positions, device=device)
    ruler = ruler_mini_eval(p02, model, device)
    res = {"protocol":"book5-r2-recall-v1", "kind": kind, "spec": ARM_SPECS[kind], "d_ff": d_ff, "params": n_par,
           "params_vs_ref": ref_dev, "seed": SEED, "steps": steps, "T": 96,
           "resumed_from_step": resumed if resumed > 1 else None,
           "train": {"loss_first": losses[0] if losses else None,
                     "loss_last_20_mean": float(np.mean(losses[-20:])) if losses else None,
                     "sec_per_step_p50": float(np.median(secs)) if secs else None},
           "recall_acc_by_pos": {str(k): v for k, v in scan.items()},
           "far_mean_18_42_58": float(np.mean([scan[p] for p in (18, 42, 58)])),
           "all_pos_mean": float(np.mean(list(scan.values()))),
           "ruler_mini": ruler,
           "wall_sec": round(time.perf_counter() - t0, 1)}
    far = res["far_mean_18_42_58"]
    print(f"  [{kind}] train {res['train']['loss_first']:.3f}->{res['train']['loss_last_20_mean']:.3f} | "
          f"{res['train']['sec_per_step_p50'] * 1000:.0f} ms/step | 五位置 "
          f"{ {k: round(v, 2) for k, v in scan.items()} } | 远端均值 {far:.2f} | wall {res['wall_sec'] / 60:.1f} min",
          flush=True)
    with open(os.path.join(OUT_DIR, f"recall3_{kind}_{out_name}_s{steps}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return res


def eval_only(p02, p05, kinds, steps, device, out_name):
    """--eval ruler-mini：从各臂末态 ckpt 只跑评估（不重训——写作期复评口径）。"""
    out = {}
    for kind in kinds:
        ckpt_path = os.path.join(OUT_DIR, f"recall_ckpt_{kind}_{out_name}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        torch.manual_seed(SEED)
        model, n_par, d_ff = build_arm(p02, p05, kind, device)
        if ck.get("protocol")!="book5-r2-recall-v1" or ck.get("kind")!=kind:
            raise ValueError("checkpoint来源/版本不符")
        model.load_state_dict(ck["model"])
        scan = p02.needle_eval(model, 96, [18, 42, 58, 62, 74], device=device)
        ruler = ruler_mini_eval(p02, model, device)
        out[kind] = {"ckpt_step": ck["step"], "params": n_par, "d_ff": d_ff,
                     "recall_acc_by_pos": {str(k): v for k, v in scan.items()},
                     "ruler_mini": ruler}
        print(f"  [{kind}] ckpt@{ck['step']} 五位置 { {k: round(v, 2) for k, v in scan.items()} }", flush=True)
    path = os.path.join(OUT_DIR, f"recall3_eval_{out_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "steps_target": steps, "arms": out}, f, ensure_ascii=False, indent=2)
    print(f"[产物] {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="ch9 混合比例 recall 消融：等参三臂（判据预注册见文件头）")
    ap.add_argument("--out-name", default="run1")
    ap.add_argument("--arm", default=None, choices=list(ARM_SPECS), help="只跑一臂（默认三臂串行）")
    ap.add_argument("--steps", type=int, default=6000, help="正式档 6000（大纲 6000-10000 下限保底线）")
    ap.add_argument("--eval", default=None, help="评估模式：ruler-mini（从 ckpt 只跑评估不重训）")
    ap.add_argument("--resume",action="store_true",help="明确续接同out-name的当前协议checkpoint")
    args = ap.parse_args()
    if args.steps<1:raise ValueError("steps必须为正")
    if not args.out_name or Path(args.out_name).name!=args.out_name or args.out_name in (".",".."):
        raise ValueError("out-name必须为单个文件名")
    if args.eval not in (None,"ruler-mini"):raise ValueError("未知eval模式")
    report_path=Path(OUT_DIR)/(f"recall3_eval_{args.out_name}.json" if args.eval else f"recall3_{args.out_name}_s{args.steps}.json")
    if report_path.exists():
        if args.eval or not args.resume:raise FileExistsError(report_path)
        previous=json.loads(report_path.read_text())
        if previous.get("protocol")!="book5-r2-recall-v1" or previous.get("steps")!=args.steps:
            raise ValueError("已有总表版本/范围不符")
    os.makedirs(OUT_DIR, exist_ok=True)
    m4 = bootstrap()
    p02 = load_feas("02_probe_swa_sink")
    from types import SimpleNamespace
    sys.path.insert(0,os.path.abspath(os.path.join(HERE,"..","ch05")))
    from gdn import GatedDeltaNet
    p05 = SimpleNamespace(GDNLayer=GatedDeltaNet)
    device = m4.pick_device("mps")
    print(f"[bootstrap] moe_mla_slots + 历史探针02任务/已核ch05 GDN| device {device} | "
          f"steps {args.steps} | seed {SEED}", flush=True)

    # 等参对齐核验（跑前三臂参数账逐位报数 + 插槽常数对枚举 + assert 互差上限）
    plan = {}
    for kind in ARM_SPECS:
        torch.manual_seed(SEED)
        model, n_par, d_ff = build_arm(p02, p05, kind, torch.device("cpu"))
        attn_params = sum(p.numel() for p in model.attns.parameters()) // L
        if kind == "all_global":
            assert attn_params == FULL_PARAMS, (attn_params, FULL_PARAMS)
        if kind == "all_linear":
            assert attn_params == GDN_PARAMS, (attn_params, GDN_PARAMS)
        plan[kind] = {"d_ff": d_ff, "params": n_par, "vs_ref": n_par - REF_TOTAL}
        print(f"[等参账] {kind:11s} d_ff={d_ff:4d} 总参 {n_par:,}（vs 锚 {n_par - REF_TOTAL:+d}）", flush=True)
        del model
    tots = [v["params"] for v in plan.values()]
    assert max(tots) - min(tots) <= MAX_INTER_GAP, f"三臂互差 {max(tots) - min(tots)} 超上限 {MAX_INTER_GAP}"
    assert plan["hybrid31"]["params"] == REF_TOTAL, "锚臂必须与 P18 探针臂逐位一致"

    kinds = [args.arm] if args.arm else list(ARM_SPECS)
    if args.eval == "ruler-mini":
        eval_only(p02, p05, kinds, args.steps, device, args.out_name)
        return

    results = {"protocol":"book5-r2-recall-v1", "seed": SEED, "date": datetime.now(timezone.utc).isoformat(), "torch": torch.__version__, "steps": args.steps, "T": 96, "B": 32,
               "recipe": "AdamW(0.9,0.95,wd0.1)+clip1.0+lr3e-3+warmup50+cosine（P18 管道升格）",
               "equal_param_plan": plan, "ref_total": REF_TOTAL,
               "prereg": "见文件头 J1-J3 / B1-B3（notes/08 同文留档）", "arms": {}}
    if report_path.exists():
        results["arms"].update(previous.get("arms",{}))
    for kind in kinds:
        results["arms"][kind] = run_arm(p02, p05, kind, args.steps, device, args.out_name, args.resume)
        with open(os.path.join(OUT_DIR, f"recall3_{args.out_name}_s{args.steps}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)     # 逐臂落盘——中断不丢已完臂

    # ---- 判据结算（预注册 J1-J3 / B1-B3） ----
    if len(results["arms"]) == 3:
        g, h, l = (results["arms"][k] for k in ("all_global", "hybrid31", "all_linear"))
        verdict = {
            "J1_all_global_all_positions": all(v >= 0.7 for v in g["recall_acc_by_pos"].values()),
            "J2_hybrid31_far_positions": all(h["recall_acc_by_pos"][str(p)] >= 0.7 for p in (18, 42, 58)),
            "J3_all_linear_far_gap": (h["far_mean_18_42_58"] - l["far_mean_18_42_58"]) >= 0.3,
            "B1_hybrid31_bottom": h["all_pos_mean"] < l["all_pos_mean"],
            "B2_all_linear_far_learned": all(l["recall_acc_by_pos"][str(p)] >= 0.7 for p in (18, 42, 58)),
            "B3_budget_short": not all(v >= 0.7 for v in g["recall_acc_by_pos"].values()),
        }
        results["prereg_verdict"] = verdict
        print(f"[判据结算] {verdict}", flush=True)
        with open(os.path.join(OUT_DIR, f"recall3_{args.out_name}_s{args.steps}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    print("hybrid_recall 完成。", flush=True)


if __name__ == "__main__":
    main()
