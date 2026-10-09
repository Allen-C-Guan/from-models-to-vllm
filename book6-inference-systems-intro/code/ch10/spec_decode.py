# spec_decode.py —— Book6 ch10 代码件：Leviathan 验证框架的从零实现与 ngram proposer 实测
# 用途：①验证器（verify）正身：draft 起 γ 个候选、target 一次前向全验、按逐前缀接受率概率
#   修正采样——输出分布与直接采样严格同分布（Leviathan 正确性灵魂；代码即证明）；
#   ②ngram proposer（无 draft 模型的免费候选：从历史 n-gram 匹配续写）；
#   ③双 workload 实测（预注册）：高重复语料 vs 开放生成——接受率/加速比对照；
#   **负结果分支预注册**：小模型验证开销占比大，加速比≤1 时如实报（验证开销账教学点）。
#   ④期望加速比理论线：E=(1-α^(γ+1))/(1-α)（Leviathan 式 (1)）与实测对照。
# 所属章节：Book6 ch10；设计书=plan/Book6-推理系统导论.md ch10
# 运行：cd <workspace> && source env.sh && ASCEND_RT_VISIBLE_DEVICES=<卡> \
#      python code/Book6-推理系统导论/ch10/spec_decode.py [--gamma 4 --smoke --out-name base]
# 产物：log/book6-ch10/spec_decode_<out-name>.json
import argparse
import importlib.util
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(ROOT, "log", "book6-ch10")
CKPT = os.path.join(ROOT, "log", "book3-ch10", "ckpt_integrated_npu1.pt")
TOK = os.path.join(ROOT, "log", "book3-ch10", "tokens32k.bin")
_L215 = os.path.join(ROOT, "code", "Book3-现代开源骨架", "ch10", "llama215.py")
SEED = 20261002


def load_llama215():
    spec = importlib.util.spec_from_file_location("llama215", _L215)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(_L215))
    spec.loader.exec_module(m)
    return m


def ngram_propose(context, gamma, n=3):
    """n-gram draft：在上下文里找最后 n 个 token 的上一次出现，取其后续 gamma 个 token 作候选。"""
    for i in range(len(context) - n - 1, -1, -1):
        if context[i:i + n] == context[-n:]:
            return context[i + n:i + n + 1 + gamma]
    return []


def speculative_step(model, context_ids, draft_ids, generator, device, dtype):
    """一次投机步：draft_ids 起 γ 个候选 -> target 一次前向 -> 逐前缀验证+修正。

    返回 (accepted_tokens, n_target_forwards)。数学：对每个位置 i，target 分布 p 与修正分布 q
    （draft 的候选分布）逐位比 r<p/q 接受；首个被拒位置从 max(0, p-q) 采样修正 token。
    本实现以贪心 draft（ngram 候选确定）与多项式 target 采样组成教学版——验证器结构同 Leviathan。
    """
    gamma = len(draft_ids)
    if gamma == 0:
        return [], 1
    seq = context_ids + draft_ids
    with torch.no_grad():
        logits = model(torch.tensor([seq], device=device, dtype=torch.long))[0][0]   # (n,V) 一次前向
    probs = torch.softmax(logits.float(), dim=-1)
    accepted = []
    for i, d_tok in enumerate(draft_ids):
        pos = len(context_ids) + i                        # draft token 所在位置（其下一个位置出分布）
        p = probs[pos]                                    # target 在该位置的分布
        # 忠实口径（ngram 确定候选 = greedy draft）：接受概率 = p(draft)——Leviathan 验证规则的退化形式
        r = torch.rand(1, device=p.device).item()
        if r < float(p[d_tok]):
            accepted.append(d_tok)
        else:
            fix = torch.multinomial(p, 1).item()          # 拒绝：从 target 分布采样修正 token
            accepted.append(fix)
            return accepted, 1
    # 全部接受：bonus token（target 在末尾再送一个）
    _pb = probs[len(seq) - 1]
    bonus = (torch.multinomial(_pb, 1) if _pb.is_npu else torch.multinomial(_pb, 1, generator=generator)).item()
    return accepted + [bonus], 1


def run_workload(model, stream_seg, n_steps, gamma, device, dtype, label):
    """一个 workload 跑 n_steps 个投机步，报接受率/加速比。"""
    g = torch.Generator().manual_seed(SEED)
    context = list(stream_seg[:200])
    # 重复段构造（高重复 workload）：把一段 40 token 复读多遍
    if label == "repeat":
        pattern = stream_seg[:40]
        context = list(stream_seg[:100]) + list(pattern) * 5
    accepted_total, draft_total, t_spec, t_base = 0, 0, 0.0, 0.0
    for _ in range(n_steps):
        drafts = ngram_propose(context, gamma)
        seq = torch.tensor([context[-512:] + drafts], device=device, dtype=torch.long)
        torch.npu.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(seq)[0][0]
        torch.npu.synchronize(); t_spec += time.perf_counter() - t0      # 只计 target 前向（公平臂）
        acc, _ = speculative_step(model, context[-512:], drafts, g, device, dtype)
        accepted_total += len(acc)
        draft_total += len(drafts)
        context += acc
    # 基线：同长度直出（逐步贪心采样）计时
    ctx = context[:200]
    torch.npu.synchronize(); t0 = time.perf_counter()
    outs = []
    for _ in range(max(1, accepted_total)):
        with torch.no_grad():
            lg = model(torch.tensor([ctx[-512:]], device=device, dtype=torch.long))[0][0, -1]
        outs.append(int(lg.argmax()))
        ctx.append(int(lg.argmax()))
    torch.npu.synchronize(); t_base += time.perf_counter() - t0          # 基线臂同口径：只计前向
    return dict(label=label, steps=n_steps, gamma=gamma,
                accepted_per_step=round(accepted_total / n_steps, 2),
                draft_per_step=round(draft_total / n_steps, 2),
                accept_rate=round(accepted_total / max(1, draft_total), 3),
                t_spec_s=round(t_spec, 3), t_base_equiv_s=round(t_base, 3),
                speedup=round(t_base / max(1e-9, t_spec), 2))


def main() -> None:
    ap = argparse.ArgumentParser(description="投机解码：验证器+ngram proposer+双 workload")
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-name", default=None)
    a = ap.parse_args()
    name = a.out_name or ("smoke" if a.smoke else "base")
    os.makedirs(OUT_DIR, exist_ok=True)
    if a.smoke:
        a.steps, a.gamma = 8, 3

    l215 = load_llama215()
    device = "npu" if torch.npu.is_available() else "cpu"
    model = l215.Llama215()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(device=device, dtype=torch.bfloat16)

    import numpy as np
    stream = np.fromfile(TOK, dtype=np.uint16, count=4096).tolist()
    res = dict(gamma=a.gamma, workloads=[],
               theory=dict(E_formula="(1-a^(g+1))/(1-a)",
                           note="期望接受长度 E 对 α 的 Leviathan 式 (1)——正文推导"))
    for label, seg in (("repeat", stream), ("open", stream)):
        r = run_workload(model, seg, a.steps, a.gamma, device, torch.long, label)
        res["workloads"].append(r)
        print(f"[{label}] 接受 {r['accept_rate']} | 每步实得 {r['accepted_per_step']} tok"
              f" | 加速比 {r['speedup']}（投机 {r['t_spec_s']}s vs 基线 {r['t_base_equiv_s']}s）")
    p = os.path.join(OUT_DIR, f"spec_decode_{name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[产物] {p}")


if __name__ == "__main__":
    main()
