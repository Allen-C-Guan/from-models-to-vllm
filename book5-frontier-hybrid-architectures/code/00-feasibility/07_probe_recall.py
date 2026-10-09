# 07_probe_recall.py —— Book5 ch9 前置探针：混合比例 recall 消融管道（全全局 vs 3:1 vs 全线性）
# 用途：ch9（2026 混合配方谱系）「线性:全局 3:1 的收益」消融实验的管道验证——
#   三臂（同 seed 同任务）：all_global（L=4 全 FullAttn）/ hybrid31（L=4：3 GDN + 1 全局，全局层在第 3 层
#   ——interval=4 的层型模式）/ all_linear（L=4 全 GDN）；任务 = 02 探针的 MQAR needle-recall（T=96）。
# 【判据预注册（管道验证档，非终局结论）】
#   - 管道可用 = 三臂 loss 有限且下降 + needle eval 跑通 + 各臂输出形状/耗时可复现；
#   - 终局假设（正式实验 6000-10000 步验证，本档 300 步只看方向苗头）：all_global 学会全位置召回；
#     hybrid31 凭 1 个全局层提供可达性也能学会（Qwen3-Next/K3 配方的存在性证据）；all_linear
#     对远端 needle 预期失败或显著劣化（固定状态容量 + 衰减 + 卷积局部性）。
#   - 参数口径如实：all_global 是 L=4、两混合臂是 L=4 但 GDN 插槽参数多于 FullAttn（本档不做等参对齐，
#     正式实验需等参或至少等激活对齐——见 JSON 口径栏）。
#   本探针默认只跑 hybrid31 一臂（fast 300 步）验管道；--arms all 跑全三臂。
# 所属章节：Book5 第 9 章（混合比例消融的管道底座）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/00-feasibility/07_probe_recall.py" [--arms one|all] [--steps 300] [--out-name run1]
# 产物：log/book5-feasibility/probe07_recall_{out}.json（不入库）
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book5-feasibility")
SEED = 20261002

_B4_FEAS_CANDS = [
    os.path.join(REPO_ROOT, "code", "Book4-规模与效率", "00-feasibility"),
    os.path.join(REPO_ROOT, "from-models-to-vllm", "book4-scale-and-efficiency", "code", "00-feasibility"),
    os.path.join(REPO_ROOT, "book4-scale-and-efficiency", "code", "00-feasibility"),
]


def bootstrap_book4():
    hits = [p for p in _B4_FEAS_CANDS if os.path.exists(os.path.join(p, "moe_mla_slots.py"))]
    if not hits:
        raise FileNotFoundError(_B4_FEAS_CANDS)
    sys.path.insert(0, hits[0])
    import moe_mla_slots as m4
    return m4, hits


def load_sibling(name):
    p = os.path.join(HERE, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save_json(name, obj, out_name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}_{out_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[产物] {path}")
    return path


def build_arm(p02, p05, kind, d=256, h=8, h_kv=4, d_k=64, d_ff=704, L=4):
    """三臂构建：attention 插槽列表——full / GDN（05 探针的 GDNLayer，h_k=h_v=4, d=64 小档）。"""
    attns = []
    for i in range(L):
        if kind == "all_global" or (kind == "hybrid31" and (i + 1) % 4 == 0):
            attns.append(p02.FullAttn(d, h, h_kv, d_k, use_rope=False))
        else:
            attns.append(p05.GDNLayer(d, h_k=4, h_v=4, d_k=d_k, d_v=d_k))
    return p02.MiniLM(d=d, n_layer=L, d_ff=d_ff, attn_modules=attns)


def main():
    ap = argparse.ArgumentParser(description="Book5 混合比例 recall 消融管道探针")
    ap.add_argument("--out-name", default="run1")
    ap.add_argument("--arms", default="one", choices=["one", "all"], help="one=只跑 hybrid31 验管道")
    ap.add_argument("--steps", type=int, default=300, help="fast 档 300 步；正式实验 6000-10000")
    args = ap.parse_args()
    m4, hits = bootstrap_book4()
    p02 = load_sibling("02_probe_swa_sink")
    p05 = load_sibling("05_probe_hybrid")
    p05.p04 = load_sibling("04_probe_gdn")
    device = m4.pick_device("mps")
    print(f"[bootstrap] moe_mla_slots 自 {hits[0]} | 复用 02（MiniLM/needle）/05（GDNLayer）| device {device}")
    torch.manual_seed(SEED)

    T = 96
    arms = ["hybrid31"] if args.arms == "one" else ["all_global", "hybrid31", "all_linear"]
    results = {"seed": SEED, "date": "2026-10-05", "T": T, "steps": args.steps, "arms": {}}
    positions = [18, 42, 58, 62, 74]                                             # 02 探针同款（cliff=62 对全局臂无意义）
    for kind in arms:
        torch.manual_seed(SEED)
        m = build_arm(p02, p05, kind).to(device)
        n_par = sum(p.numel() for p in m.parameters())
        bf = p02.NeedleBatch(32, T, SEED + 7, device)
        res = p02.train_mini(m, args.steps, bf, device, lr=3e-3, warmup=50, tag=kind)
        acc = p02.needle_eval(m, T, positions, device=device)
        results["arms"][kind] = {"params": n_par, "train": res, "recall_acc_by_pos": acc}
        print(f"  [{kind}] {n_par/1e6:.2f}M | train {res['loss_first']:.3f}→{res['loss_last_20_mean']:.3f} "
              f"| {res['sec_per_step_p50']*1000:.0f} ms/step | acc { {k: round(v,2) for k,v in acc.items()} }")
        del m
    results["caveats"] = ("①参数口径：三臂不等参（all_global 的 FullAttn 插槽 < GDN 插槽）——正式实验需等参/等激活对齐；"
                          "②fast 300 步只验管道与方向苗头，终局结论需 6000-10000 步（02 探针试点：MQAR 类任务"
                          "归纳头电路需千步级才成形）。")
    save_json("probe07_recall", results, args.out_name)
    print("探针 07 完成。")


if __name__ == "__main__":
    main()
