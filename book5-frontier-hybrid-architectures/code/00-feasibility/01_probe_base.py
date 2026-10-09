# 01_probe_base.py —— Book5 底座验证探针：Book4 moe_mla_slots 双布局引入 + MPS 基线 sec/step
# 用途：全册实验预算与档位的底座——验证 Book4 插槽组件库（SlotConfig/SlotLLaMA/三口径账本/
#       train_loop）能被 Book5 以「跨册双候选 bootstrap」引入（工作区 code/ 与书仓
#       from-models-to-vllm/ 两布局，llama409.py 头部写法的先例），并：
#       ① CPU 秒级：dense/MoE 两档参数手算账 vs 实测逐位 assert（Book4 资产在本册无损可用）；
#       ② MPS：dense 小档（26.1M）与 MoE 小档（59.2M）前向+反向各 10 步，量出基线 sec/step
#          与内存峰值（torch.mps.current_allocated_memory 口径）——后续各探针的开销对照锚。
# 所属章节：Book5 00-feasibility（全册实验档位底座；结论进 notes/01-实验可行性.md）
# 运行方式：cd 工作区根目录 && source env.sh && python "code/00-feasibility/01_probe_base.py" [--out-name run1]
# 产物：log/book5-feasibility/probe01_base_{out}.json（不入库）；预计 3-5 分钟（MPS）。
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))   # 工作区根 / 书仓根两布局同式
OUT_DIR = os.path.join(REPO_ROOT, "log", "book5-feasibility")
SEED = 20261002                                                     # 全书统一种子

# ---------------- 跨册双候选 bootstrap：Book4 00-feasibility 的物理落点，存在即引入 ----------------
# 三候选与 llama409.py 的 _B3_CANDS 同构：工作区布局 / 书仓布局（from-models-to-vllm/<slug>/code/）
# / 书仓裸布局（<slug>/code/——探针自身被快照进书仓后 REPO_ROOT 即书仓根的形态）。
_B4_FEAS_CANDS = [
    os.path.join(REPO_ROOT, "code", "Book4-规模与效率", "00-feasibility"),
    os.path.join(REPO_ROOT, "from-models-to-vllm", "book4-scale-and-efficiency", "code", "00-feasibility"),
    os.path.join(REPO_ROOT, "book4-scale-and-efficiency", "code", "00-feasibility"),
]


def bootstrap_book4():
    """引入 Book4 的 moe_mla_slots（组件正身：SlotConfig/SlotLLaMA/SparseMoE/MLAAttention/
    hand_account/activated_account/kv_per_token_elements/train_loop）。返回 (模块, 命中路径表)。"""
    hits = []
    for p in _B4_FEAS_CANDS:
        if os.path.exists(os.path.join(p, "moe_mla_slots.py")):
            hits.append(p)
    if not hits:
        raise FileNotFoundError(f"三候选布局均未找到 moe_mla_slots.py：{_B4_FEAS_CANDS}")
    sys.path.insert(0, hits[0])                    # 工作区优先（Book3 终校「存在即插入」先例）
    import moe_mla_slots as m4
    return m4, hits


def save_json(name, obj, out_name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}_{out_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[产物] {path}")
    return path


def run_10_steps(m4, cfg, tag, device, batch=8, block=512):
    """小档 10 步前向+反向（train_loop 同配方：AdamW + warmup + clip + MPS bf16 autocast）。
    返回 sec/step 列表、首末 loss、MPS 内存峰值（MB）。"""
    torch.manual_seed(SEED)
    model = m4.SlotLLaMA(cfg).to(device)
    if device.type == "mps":
        torch.mps.empty_cache()                    # torch 2.13 无 reset_peak_memory_stats：清缓存后取分配水位
        mem_floor = torch.mps.current_allocated_memory()
    t0 = time.perf_counter()
    secs, losses, _ = m4.train_loop(model, steps=10, batch=batch, block=block, device=device,
                                    tokens_path=m4.TOKENS_SP8K, peak_lr=1e-3, warmup=2, log_every=5)
    wall = time.perf_counter() - t0
    # 口径：current_allocated_memory 水位差（近似峰值增量；torch 2.13 无官方峰值 API，如实标注）
    peak = (torch.mps.current_allocated_memory() - mem_floor) / 1024 / 1024 if device.type == "mps" else None
    n_par = model.n_params()
    del model
    return {"tag": tag, "params": n_par,
            "sec_per_step": {"mean": float(np.mean(secs)), "p50": float(np.median(secs)),
                             "last3_mean": float(np.mean(secs[-3:]))},
            "loss_first": losses[0], "loss_last": losses[-1],
            "wall_total_s": wall, "mps_peak_mb": peak}


def main():
    ap = argparse.ArgumentParser(description="Book5 底座验证探针")
    ap.add_argument("--out-name", default="run1", help="产物 JSON 后缀（防覆写）")
    args = ap.parse_args()
    torch.manual_seed(SEED)

    m4, hits = bootstrap_book4()
    print(f"[bootstrap] Book4 00-feasibility 命中：{hits[0]}")
    print(f"[bootstrap] 全部存在候选（{len(hits)}/3）：{hits}")

    # ① CPU 秒级：Book4 两档参数账逐位 assert（资产可用性）
    dense_cfg = m4.SlotConfig(vocab_size=8192, hidden_size=512, num_hidden_layers=6,
                              num_attention_heads=8, num_key_value_heads=4, intermediate_size=1408)
    moe_cfg = m4.SlotConfig(vocab_size=8192, hidden_size=512, num_hidden_layers=6,
                            num_attention_heads=8, num_key_value_heads=4, intermediate_size=1408,
                            moe=dict(n_expert=8, top_k=2, expert_dim=704, n_shared=0, variant="teach"))
    cpu_checks = {}
    for tag, c in (("dense_26M", dense_cfg), ("moe_65M", moe_cfg)):
        m = m4.SlotLLaMA(c)
        hand = m4.hand_account(c)
        assert m.n_params() == hand["total"], f"{tag}: {m.n_params()} != {hand['total']}"
        act, act_ne = m4.activated_account(c)
        kv_elems, _ = m4.kv_per_token_elements(c)
        cpu_checks[tag] = {"total": m.n_params(), "activated": act, "activated_non_emb": act_ne,
                           "kv_per_token_elems": kv_elems}
        print(f"[CPU 账本] {tag}: 总参 {m.n_params():,}（手算逐位一致）| 激活 {act:,} | KV/token {kv_elems} 元素")
        del m

    # ② MPS 基线：dense / MoE 各 10 步
    device = m4.pick_device("mps")
    print(f"[device] {device}（PYTORCH_ENABLE_MPS_FALLBACK=1）")
    runs = [run_10_steps(m4, dense_cfg, "dense_26M", device),
            run_10_steps(m4, moe_cfg, "moe_65M", device)]
    for r in runs:
        print(f"[MPS 基线] {r['tag']}: {r['sec_per_step']['p50']:.3f} s/step (p50) | "
              f"loss {r['loss_first']:.3f}→{r['loss_last']:.3f} | 峰值 {r['mps_peak_mb']:.0f} MB")

    save_json("probe01_base", {
        "seed": SEED, "date": "2026-10-05",
        "bootstrap_hits": hits, "bootstrap_candidates": _B4_FEAS_CANDS,
        "device": str(device), "cpu_account_checks": cpu_checks, "mps_runs": runs,
    }, args.out_name)
    print("探针 01 完成：底座可用，基线已量。")


if __name__ == "__main__":
    main()
