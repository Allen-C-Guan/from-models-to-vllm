# samplers.py —— Book6 ch9 代码件：采样全家福与 logits processor 管线
# 用途：temperature/top-k/top-p 与三 penalties 的可运行正身 + 管线顺序实验（顺序不同结果不同
#   的构造例代码化）+ 207M ckpt 实跑样例（两段 prompt × 三温度对照生成，--ckpt-gen 档）。
#   penalties 口径：repetition 为 CTRL 乘式正身（正 logit 除以 w、负 logit 乘以 w）；
#   frequency/presence 为加式通行口径（OpenAI API 文档级）——正文 9.4 与代码一一对应。
# 所属章节：Book6 ch9；设计书=plan/Book6-推理系统导论.md ch9
# 运行：cd <workspace> && source env.sh && python code/Book6-推理系统导论/ch09/samplers.py [--out-name base]
#           [--ckpt-gen]（207M 生成档；NPU 快/CPU 亦可，先 npu-smi 选卡）
# 产物：log/book6-ch09/samplers_<out-name>.json（含三温度/自适应/顺序实验）
#       log/book6-ch09/samplers_ckptgen_<out-name>.json（生成样例）
import argparse
import importlib.util
import json
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(ROOT, "log", "book6-ch09")
CKPT = os.path.join(ROOT, "log", "book3-ch10", "ckpt_integrated_npu1.pt")
TOK = os.path.join(ROOT, "log", "book3-ch10", "tokens32k.bin")
SEED = 20261002


def apply_temperature(logits, t):
    """logits (V,) -> (V,)。t->0 锐化为贪心、t=1 原分布、t->inf 平坦化。"""
    return logits / max(t, 1e-6)


def top_k_filter(logits, k):
    """保留恰 k 个 id（按 top-k 索引取——平局时按索引序破平，严格 k 个）。"""
    if k >= logits.numel():
        return logits
    keep = torch.zeros_like(logits, dtype=torch.bool)
    keep[torch.topk(logits, k).indices] = True
    return logits.masked_fill(~keep, float("-inf"))


def top_p_filter(logits, p):
    """核采样：按累积概率 p 截断——分布尖时留得少、分布平时留得多（自适应截断）。"""
    probs = F.softmax(logits.float(), dim=-1)
    sorted_p, idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_p, dim=-1)
    keep = cum - sorted_p < p                      # 恰好越过 p 的那个仍保留
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[idx[keep]] = True
    return logits.masked_fill(~mask, float("-inf"))


def apply_penalties(logits, seen_counts, rep_w=1.0, freq_w=0.0, presence_w=0.0):
    """三 penalties（logits (V,), seen_counts (V,) -> (V,)）：
    repetition=CTRL 乘式正身：正 logit 除以 w、负 logit 乘以 w（对称压向 0）；
    frequency=按出现次数线性减（加式）；presence=出现即减、不管次数（加式）。"""
    out = logits.clone().float()
    if rep_w != 1.0:
        seen = seen_counts > 0
        out[seen] = torch.where(out[seen] > 0, out[seen] / rep_w, out[seen] * rep_w)
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


def run_basics(out_name):
    g = torch.Generator().manual_seed(SEED)
    # ① 三温度对照（同一 logits）
    logits = torch.randn(50, generator=g) * 3
    t_read = {}
    for t in (0.5, 1.0, 2.0):
        p = F.softmax(apply_temperature(logits, t), dim=-1)
        t_read[t] = dict(top3_prob=[round(v, 3) for v in p.topk(3).values.tolist()],
                         entropy=round(float(-(p * p.clamp_min(1e-12).log()).sum()), 3))
    # ② top-k vs top-p 的自适应差（尖/平两个分布；top-k 恰 k 个——平局按索引破平）
    sharp = torch.zeros(20); sharp[3] = 8.0
    flat = torch.ones(20)
    adaptive = {}
    for name, lg in (("sharp", sharp), ("flat", flat)):
        k_keep = int((top_k_filter(lg, 5) > float("-inf")).sum())
        p_keep = int((top_p_filter(lg, 0.9) > float("-inf")).sum())
        adaptive[name] = dict(top_k_keeps=k_keep, top_p_keeps=p_keep)
    # ③ 管线顺序实验：温度×top-k 可交换；温度×top-p 不可交换（保集不同）
    order_demo = {}
    lg = torch.tensor([1.0, 2.0, 3.0, 0.5])
    a1 = top_k_filter(apply_temperature(lg, 0.5), 2)
    a2 = apply_temperature(top_k_filter(lg, 2), 0.5)
    order_demo["temp_topk_vs_topk_temp"] = dict(
        t_then_k=[round(v, 3) for v in a1.tolist()],
        k_then_t=[round(v, 3) for v in a2.tolist()],
        note="温度与 top-k 可交换（线性缩放不改 top-k 集合）；温度与 top-p 不可交换——温度改分布形状、核采样依赖分布形状")
    p1 = top_p_filter(apply_temperature(lg, 0.3), 0.9)
    p2 = apply_temperature(top_p_filter(lg, 0.9), 0.3)
    keep1 = [i for i, v in enumerate(p1.tolist()) if v > float("-inf")]
    keep2 = [i for i, v in enumerate(p2.tolist()) if v > float("-inf")]
    order_demo["temp_topp_vs_topp_temp"] = dict(
        t_then_p=[round(v, 3) for v in p1.tolist()], t_then_p_keep=keep1,
        p_then_t=[round(v, 3) for v in p2.tolist()], p_then_t_keep=keep2)
    # ④ penalties 手算例（正文例 9.x 的数据源）：某 token 已现 2 次的 logits 变化
    lg2 = torch.tensor([2.0, 1.0, 0.5])
    cnt = torch.tensor([2.0, 0.0, 1.0])
    penalties = {
        "ctrl_rep_w1.3": [round(v, 3) for v in apply_penalties(lg2, cnt, rep_w=1.3).tolist()],
        "freq_w0.5": [round(v, 3) for v in apply_penalties(lg2, cnt, freq_w=0.5).tolist()],
        "presence_w0.5": [round(v, 3) for v in apply_penalties(lg2, cnt, presence_w=0.5).tolist()],
    }
    res = dict(temperature=t_read, adaptive=adaptive, order=order_demo, penalties=penalties,
               note="采样不改引擎性能数字——但管线顺序是分布形状的函数，工程上以引擎文档口径为准")
    p = os.path.join(OUT_DIR, f"samplers_{out_name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[basics] 温度/自适应/顺序/penalties 四组读数 -> {p}")


def run_ckpt_gen(out_name):
    """207M 实跑：两段 prompt × 三温度（T=0.6/1.0/1.5），各 32 token，种子固定。"""
    spec = importlib.util.spec_from_file_location(
        "llama215", os.path.join(ROOT, "code", "Book3-现代开源骨架", "ch10", "llama215.py"))
    l215 = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.join(ROOT, "code", "Book3-现代开源骨架", "ch10", "llama215.py")))
    spec.loader.exec_module(l215)
    device = "npu" if torch.npu.is_available() else "cpu"
    model = l215.Llama215()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(device=device, dtype=torch.bfloat16)

    import numpy as np
    stream = np.fromfile(TOK, dtype=np.uint16, count=2048)
    # 两段 prompt：一段事实型（语料开头=叙述文本）、一段开放型（语料后段）
    prompts = [stream[:96].tolist(), stream[1024:1024 + 96].tolist()]
    res = dict(device=device, dtype="bfloat16", seed=SEED, gen_len=32, samples=[])
    for pi, prompt in enumerate(prompts):
        for t in (0.6, 1.0, 1.5):
            g = torch.Generator().manual_seed(SEED + pi * 10)
            ctx = list(prompt)
            with torch.no_grad():
                for _ in range(32):
                    logits = model(torch.tensor([ctx[-256:]], device=device, dtype=torch.long))[0][0, -1]
                    nxt = sample(logits.float().cpu(), temperature=t, generator=g)
                    ctx.append(int(nxt))
            res["samples"].append(dict(prompt_i=pi, temperature=t,
                                       prompt_ids=prompt[:8], gen_ids=ctx[len(prompt):]))
            print(f"[ckpt-gen] prompt{pi} T={t}: 首 8 生成 id {ctx[len(prompt):len(prompt)+8]}")
    p = os.path.join(OUT_DIR, f"samplers_ckptgen_{out_name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[产物] {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="base")
    ap.add_argument("--ckpt-gen", action="store_true", help="207M 三温度生成档（实物 9.1 数据源）")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if a.ckpt_gen:
        run_ckpt_gen(a.out_name)
    else:
        run_basics(a.out_name)


if __name__ == "__main__":
    main()
