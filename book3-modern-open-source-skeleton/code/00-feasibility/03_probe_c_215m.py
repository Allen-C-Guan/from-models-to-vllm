# 03_probe_c_215m.py —— 探针 C：215M 候选 config 探针 + 全册训练预算外推表
# 用途：(1) 215M 候选 config 的逐项手算参数账（微调 h_kv / d_ff 使总参数落在 205-225M）；
#       (2) 前向+反向 10-12 步实测 sec/step（MPS bf16 / MPS fp32 / CPU fp32 三口径）与 MPS 内存峰值；
#       (3) 输出预算外推表：100M/1B/10B/15T-tokens 各档 wall-clock——ch10 大项目设计文档的量化底数，
#           也是「215M 全量长训暂缓（作者决策 2026-10-03）」的论证依据。
# 所属章节：Book3 00-feasibility（ch10 大项目设计文档底数）。
# 运行方式：cd code/00-feasibility && python 03_probe_c_215m.py [--out-name NAME]
#   产物：log/book3-feasibility/{out-name}.json（含手算账、三口径实测、外推表）
# 手算逐项账（d=1024/L=12/h=16/h_kv=8/d_ff=2816/V=32000 untied，无 bias，RMSNorm 只有 weight）：
#   注意力每层 = q 1024*1024 + k 1024*512 + v 1024*512 + o 1024*1024     = 3,145,728
#   SwiGLU 每层 = 3 * 1024 * 2816                                        = 8,650,752
#   每层 RMSNorm 2*1024；小计                                          = 11,798,528
#   12 层                                                              = 141,582,336
#   wte + lm_head（untied）= 2 * 32000 * 1024；末层 RMSNorm 1024          = 65,537,024
#   总计 = 207,119,360 ≈ 207.1M   （候选② h_kv=4/d_ff=2816 → 200.8M 低于带；
#                                  候选③ h_kv=4/d_ff=3328 → 219.7M 亦在带内——本探针取①为主 config）
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from llama_slots import LLaMA, LLaMAConfig, LLAMA_215M   # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book3-feasibility")
SEED = 20261002


def hand_count(d, L, h, h_kv, d_ff, V):
    """LLaMA 式参数逐项手算（无 bias、untied、无 wpe）。返回 (明细 dict, 总数)。"""
    attn = d * d * 2 + d * (d // h) * h_kv * 2                      # q/o 全维 + k/v 低维（GQA 省在 k/v）
    mlp = 3 * d * d_ff                                               # gate/up/down 三投影
    per_layer = attn + mlp + 2 * d
    total = per_layer * L + V * d * 2 + d                            # untied：wte 与 lm_head 各一份
    detail = {"attn_per_layer": attn, "mlp_per_layer": mlp, "per_layer": per_layer,
              "layers": per_layer * L, "wte+lm_head(untied)": V * d * 2, "final_norm": d, "total": total}
    return detail, total


def timed_steps(model, device, steps, batch, block, bf16=False):
    """固定 B/T 的 fwd+bwd+AdamW 逐步计时（丢首 3 步预热），返回 (稳态均值, 逐步列表, 峰值增量 MB)。"""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
    g = torch.Generator().manual_seed(SEED)
    x = torch.randint(0, model.cfg.vocab_size, (batch, block), generator=g).to(device)
    y = torch.randint(0, model.cfg.vocab_size, (batch, block), generator=g).to(device)
    base = torch.mps.current_allocated_memory() / 2**20 if device.type == "mps" else 0.0
    peak, times = base, []
    for i in range(steps):
        t0 = time.perf_counter()
        if bf16:
            with torch.autocast("mps", dtype=torch.bfloat16):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if device.type == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
        if device.type == "mps":
            peak = max(peak, torch.mps.current_allocated_memory() / 2**20)
    return sum(times[3:]) / len(times[3:]), [round(t, 3) for t in times], round(peak - base, 1), float(loss.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="probe_c_215m")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--block", type=int, default=1024)
    args = ap.parse_args()

    torch.manual_seed(SEED)

    # ---- (1) 候选 config 手算账：微调 h_kv / d_ff 落进 205-225M ----
    candidates = {
        "cand1_hkv8_dff2816 (主)": dict(d=1024, L=12, h=16, h_kv=8, d_ff=2816, V=32000),
        "cand2_hkv4_dff2816": dict(d=1024, L=12, h=16, h_kv=4, d_ff=2816, V=32000),
        "cand3_hkv4_dff3328": dict(d=1024, L=12, h=16, h_kv=4, d_ff=3328, V=32000),
    }
    hand = {}
    for name, c in candidates.items():
        detail, total = hand_count(**c)
        hand[name] = {"config": c, "detail": detail, "total": total,
                      "in_band": bool(205e6 <= total <= 225e6)}
        print(f"[手算] {name}: total = {total:,}  in 205-225M band = {hand[name]['in_band']}")

    cfg = LLaMAConfig(**LLAMA_215M)          # 主 config = cand1
    model = LLaMA(cfg)
    total_real = model.n_params()
    _, total_hand = hand_count(**candidates["cand1_hkv8_dff2816 (主)"])
    assert total_real == total_hand == 207_119_360, "参数逐项对账失败"
    print(f"[对账] 215M 主 config 实测参数 = {total_real:,}（与手算逐位一致）")

    # ---- (2) 三口径实测 ----
    measurements = {}
    device_specs = []
    if torch.backends.mps.is_available():
        device_specs += [("mps", True, 12), ("mps", False, 12)]
    device_specs += [("cpu", False, 6)]
    for dev_name, bf16, steps in device_specs:
        device = torch.device(dev_name)
        torch.manual_seed(SEED)
        m = LLaMA(cfg)
        tag = f"{dev_name}{'_bf16' if bf16 else '_fp32'}"
        try:
            sec, all_t, peak, last_loss = timed_steps(m, device, steps, args.batch, args.block, bf16)
            measurements[tag] = {"sec_step": round(sec, 4), "steps": steps, "all_times": all_t,
                                 "mem_peak_delta_mb": peak, "final_loss": round(last_loss, 3)}
            print(f"[{tag}] sec/step = {sec:.4f}  峰值增量 = {peak} MB  末步 loss = {last_loss:.3f}")
        except RuntimeError as e:                                   # OOM 等情况：如实记录后继续
            measurements[tag] = {"error": str(e)[:200]}
            print(f"[{tag}] 失败：{e}")
        del m
        torch.mps.empty_cache() if dev_name == "mps" else None

    # ---- (3) 预算外推表（ch10 设计文档底数；全部标注为「本机实测外推」证据等级） ----
    # 口径：sec/step 按每步 FLOPs ∝ N 缩放（同 B/T）；tokens/step = batch*block。
    # Chinchilla 配方 = 20 tokens/param；15T = LLaMA3 口径（2407.21783，正文引用处另行核对）。
    ref_tag = "mps_bf16" if "mps_bf16" in measurements and "sec_step" in measurements.get("mps_bf16", {}) else \
              (next((t for t, v in measurements.items() if "sec_step" in v), None))
    tok_per_step = args.batch * args.block
    extrap = {"reference": {"tag": ref_tag, "sec_step": measurements.get(ref_tag, {}).get("sec_step"),
                            "tokens_per_step": tok_per_step, "params_ref": total_real},
              "scaling_assumption": "sec/step ∝ N（同 B/T 固定；MPS 大模型效率变化的偏差未计入，外推仅作量级论证）",
              "rows": []}
    if ref_tag:
        sec_ref = measurements[ref_tag]["sec_step"]
        for label, N, tokens in [
            ("215M @ Chinchilla 4.3B tok", 207_119_360, 4_300_000_000),
            ("215M @ 15T tok（LLaMA3 口径，暂缓档）", 207_119_360, 15_000_000_000_000),
            ("100M @ Chinchilla 2B tok", 100_000_000, 2_000_000_000),
            ("1B   @ Chinchilla 20B tok", 1_000_000_000, 20_000_000_000),
            ("10B  @ Chinchilla 200B tok", 10_000_000_000, 200_000_000_000),
        ]:
            sec_step = sec_ref * N / total_real
            steps_needed = tokens / tok_per_step
            sec_total = steps_needed * sec_step
            days = sec_total / 86400
            extrap["rows"].append({
                "case": label, "assumed_params": N, "tokens": tokens,
                "sec_step_extrap": round(sec_step, 3), "steps_needed": int(steps_needed),
                "wall_clock_days_extrap": round(days, 2),
                "verdict": "本机可行（<1 天）" if days < 1 else ("本机勉强（1-3 天）" if days < 3 else "本机不可行——暂缓/换高算力机"),
            })
            print(f"[外推] {label}: sec/step≈{sec_step:.3f}  步数={int(steps_needed):,}  wall-clock≈{days:.2f} 天")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{args.out_name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "batch": args.batch, "block": args.block,
                   "hand_account": hand, "measured": measurements, "extrapolation": extrap},
                  f, ensure_ascii=False, indent=2)
    print(f"产物已写：{out}")


if __name__ == "__main__":
    main()
