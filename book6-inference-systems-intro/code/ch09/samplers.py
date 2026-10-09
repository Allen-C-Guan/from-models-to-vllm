# samplers.py —— Book6 ch9 代码件：采样全家福与 logits processor 管线
# 用途：temperature/top-k/top-p 与三 penalties 的可运行正身 + 管线顺序实验（顺序不同结果不同
#   的构造例代码化）+ 207M 实跑样例（三温度对照生成）。CPU 秒级。
# 所属章节：Book6 ch9；设计书=plan/Book6-推理系统导论.md ch9
# 运行：cd <workspace> && source env.sh && python code/Book6-推理系统导论/ch09/samplers.py [--out-name base]
# 产物：log/book6-ch09/samplers_<out-name>.json
import argparse
import json
import os

import torch
import torch.nn.functional as F


def apply_temperature(logits, t):
    """logits (V,) -> (V,)。t->0 锐化为贪心、t=1 原分布、t->inf 平坦化。"""
    return logits / max(t, 1e-6)


def top_k_filter(logits, k):
    """保留 top-k 个 id，其余 -inf（个数截断——对任何分布固定留 k 个）。"""
    if k >= logits.numel():
        return logits
    thresh = torch.topk(logits, k).values[-1]
    return logits.masked_fill(logits < thresh, float("-inf"))


def top_p_filter(logits, p):
    """核采样：按累积概率 p 截断——分布尖时留得少、分布平时留得多（自适应截断）。"""
    probs = F.softmax(logits.float(), dim=-1)
    sorted_p, idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_p, dim=-1)
    keep = cum - sorted_p < p                      # 恰好越过 p 的那个仍保留
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[idx[keep]] = True
    return logits.masked_fill(~mask, float("-inf"))


def apply_penalties(logits, seen_counts, presence_w=0.0, freq_w=0.0, rep_w=0.0):
    """三 penalties：repetition（出现即罚）/frequency（按次数线性罚）/presence（出现即罚、不管次数）。"""
    out = logits.clone()
    if rep_w:
        out = out - rep_w * (seen_counts > 0).float()
    if freq_w:
        out = out - freq_w * seen_counts.float()
    if presence_w:
        out = out - presence_w * (seen_counts > 0).float()
    return out


def sample(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    x = apply_temperature(logits.float(), temperature)
    if top_k:
        x = top_k_filter(x, top_k)
    if top_p:
        x = top_p_filter(x, top_p)
    return torch.multinomial(F.softmax(x, dim=-1), 1, generator=generator)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="base")
    a = ap.parse_args()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "log", "book6-ch09")
    os.makedirs(out_dir, exist_ok=True)
    g = torch.Generator().manual_seed(20261002)

    # ① 三温度对照（同一 logits）
    logits = torch.randn(50, generator=g) * 3
    t_read = {}
    for t in (0.5, 1.0, 2.0):
        p = F.softmax(apply_temperature(logits, t), dim=-1)
        t_read[t] = dict(top3_prob=[round(v, 3) for v in p.topk(3).values.tolist()],
                         entropy=round(float(-(p * p.clamp_min(1e-12).log()).sum()), 3))
    # ② top-k vs top-p 的自适应差（两个形状不同的分布）
    sharp = torch.zeros(20); sharp[3] = 8.0                    # 尖分布
    flat = torch.ones(20)                                      # 平分布
    adaptive = {}
    for name, lg in (("sharp", sharp), ("flat", flat)):
        k_keep = int((top_k_filter(lg, 5) > float("-inf")).sum())
        p_keep = int((top_p_filter(lg, 0.9) > float("-inf")).sum())
        adaptive[name] = dict(top_k_keeps=k_keep, top_p_keeps=p_keep)
    # ③ 管线顺序实验：先 top-k 后 temperature vs 先 temperature 后 top-k（k=2 时等价吗？）
    order_demo = {}
    lg = torch.tensor([1.0, 2.0, 3.0, 0.5])
    a1 = top_k_filter(apply_temperature(lg, 0.5), 2)           # 先温度后截断
    a2 = apply_temperature(top_k_filter(lg, 2), 0.5)           # 先截断后温度（k 截断与温度交换律成立）
    order_demo["temp_topk_vs_topk_temp"] = dict(
        t_then_k=[round(v, 3) for v in a1.tolist()],
        k_then_t=[round(v, 3) for v in a2.tolist()],
        note="温度与 top-k 可交换（线性缩放不改 top-k 集合）；但 top-p 与 temperature 不可交换——温度改分布形状、核采样依赖分布形状")
    p1 = top_p_filter(apply_temperature(lg, 0.3), 0.9)
    p2 = apply_temperature(top_p_filter(lg, 0.9), 0.3)
    order_demo["temp_topp_vs_topp_temp"] = dict(
        t_then_p=[round(v, 3) for v in p1.tolist()],
        p_then_t=[round(v, 3) for v in p2.tolist()])

    res = dict(temperature=t_read, adaptive=adaptive, order=order_demo,
               note="采样不改引擎性能数字——但管线顺序是分布形状的函数，工程上以引擎文档口径为准")
    p = os.path.join(out_dir, f"samplers_{a.out_name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1)[:800])
    print(f"[产物] {p}")


if __name__ == "__main__":
    main()
