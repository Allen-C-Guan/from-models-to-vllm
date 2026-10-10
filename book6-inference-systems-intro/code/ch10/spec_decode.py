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
            return context[i + n:i + n + gamma]
    return []


def speculative_step(model, context_ids, draft_ids, generator, device, dtype, timer=None):
    """一次投机步：draft_ids 起 γ 个候选 -> target 一次前向 -> 逐前缀验证+修正。

    返回 (accepted_tokens, n_target_forwards)。数学：对每个位置 i，target 分布 p 与修正分布 q
    （draft 的候选分布）逐位比 r<p/q 接受；首个被拒位置从 max(0, p-q) 采样修正 token。
    本实现以贪心 draft（ngram 候选确定）与多项式 target 采样组成教学版——验证器结构同 Leviathan。
    timer 非 None 时只计**本函数内部的 target 前向**（投机臂的真实成本=这次前向，一次一步）。
    """
    gamma = len(draft_ids)
    if gamma == 0:
        return [], 1
    seq = context_ids + draft_ids
    if timer is not None:
        torch.npu.synchronize() if device == "npu" else None
        t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(torch.tensor([seq], device=device, dtype=torch.long))[0][0]   # (n,V) 一次前向
    if timer is not None:
        torch.npu.synchronize() if device == "npu" else None
        timer.append(time.perf_counter() - t0)
    probs = torch.softmax(logits.float(), dim=-1)
    accepted = []
    for i, d_tok in enumerate(draft_ids):
        # 条件分布正身：logits[pos-1] 由前缀+d_0..d_{i-1} 算出——正是「评判 d_i 的 target 分布」
        # （用 probs[pos] 会把被评判的 token 自身混进条件——「d 之后再出 d」的自重复概率，错位；
        #  bonus 行的 probs[len(seq)-1] 与此处同约：位置 j 的分布预测 j+1）
        p = probs[len(context_ids) + i - 1]
        # 忠实口径（ngram 确定候选 = greedy draft）：接受概率 = p(draft)——Leviathan 验证规则的退化形式
        r = torch.rand(1, device=p.device).item()
        if r < float(p[d_tok]):
            accepted.append(d_tok)
        else:
            # 拒绝：从残差分布采样修正 token——max(0, p−q) 的贪心 draft 特例：
            # q 是 one-hot(d)，残差 = p 去掉 d 处的质量后归一（Leviathan 修正项；
            # 从完整 p 采样会破坏分布不变性——self_test 的对拍专门抓这个错）
            residual = p.clone()
            residual[d_tok] = 0.0
            fix = torch.multinomial(residual, 1).item()
            accepted.append(fix)
            return accepted, 1
    # 全部接受：bonus token（target 在末尾再送一个）
    bonus = torch.multinomial(probs[len(seq) - 1], 1).item()
    return accepted + [bonus], 1


def run_workload(model, stream_seg, n_steps, gamma, device, dtype, label):
    """一个 workload 跑 n_steps 个投机步，报接受率/加速比。

    计时口径（双臂同构、只计 target 前向、起始上下文同源）：投机臂=每步 speculative_step
    内部的一次 (n+γ) 前向（timer 计入函数内部——draft 的 ngram 匹配为纯 python 字典级操作，
    不计）；基线臂=从**同一份起始上下文**出发逐步贪心，每产一个 token 一次前向、共
    accepted_total 次（两臂前向长度随各自增长同源演化——长度口径对齐，勿混入长度差的假开销）。
    两臂前向实现同款（全量重算无 KV）。
    """
    g = torch.Generator().manual_seed(SEED)
    context = list(stream_seg[:200])
    # 重复段构造（高重复 workload）：把一段 40 token 复读多遍
    if label == "repeat":
        pattern = stream_seg[:40]
        context = list(stream_seg[:100]) + list(pattern) * 5
    ctx0 = list(context)                                      # 两臂共同的起始上下文（对齐长度口径）
    accepted_total, draft_total = 0, 0
    t_spec_l = []
    productive, empty_proposals = 0, 0
    while productive < n_steps:
        drafts = ngram_propose(context, gamma)
        if not drafts:
            empty_proposals += 1
            if label == "repeat":
                context = context + list(stream_seg[:40])      # 续种：模拟持续到达的复读内容
                continue
            if empty_proposals >= n_steps:                     # open：负载确无重复，如实收账
                break
            context = context + [stream_seg[len(context) % len(stream_seg)]]
            continue
        acc, _ = speculative_step(model, context[-512:], drafts, g, device, dtype, timer=t_spec_l)
        accepted_total += len(acc)
        draft_total += len(drafts)
        context += acc
        productive += 1
    t_spec = sum(t_spec_l)
    # 基线：同起始上下文、同量 token 逐步贪心（每次一条前向）。
    # 计时对称（与投机臂同款）：逐步 forward+sync——两臂的 sync 边界数相同，勿让基线靠
    # 整环免 sync 的流水占便宜（真实 decode 循环每步也有同步边界）。
    ctx = list(ctx0)
    t_base_l = []
    for _ in range(max(1, accepted_total)):
        torch.npu.synchronize() if device == "npu" else None
        tb = time.perf_counter()
        with torch.no_grad():
            lg = model(torch.tensor([ctx[-512:]], device=device, dtype=torch.long))[0][0, -1]
        nxt = int(lg.argmax())
        torch.npu.synchronize() if device == "npu" else None
        t_base_l.append(time.perf_counter() - tb)
        ctx.append(nxt)
    t_base = sum(t_base_l)
    n_eff = max(1, productive)
    return dict(label=label, steps=n_steps, productive_steps=productive,
                empty_proposals=empty_proposals, gamma=gamma,
                accepted_per_step=round(accepted_total / n_eff, 2),
                draft_per_step=round(draft_total / n_eff, 2),
                accept_rate=round(accepted_total / max(1, draft_total), 3),
                t_spec_s=round(t_spec, 3), t_base_equiv_s=round(t_base, 3),
                t_base_per_token_ms=round(t_base / max(1, accepted_total) * 1e3, 2),
                t_spec_per_step_ms=round(t_spec / n_eff * 1e3, 2),
                speedup=(round(t_base / max(1e-9, t_spec), 2) if t_spec > 0 else None))


def self_test():
    """序列级分布对拍（CPU fp32）——投机验证器 vs 逐 token 重算（规范 §8 预登记口径）。

    玩具 target=bigram stub（logits[pos]=W[seq[pos]]——分布只依赖上一个 token），于是
    speculative_step 的每个输出位置都有**解析正身** softmax(W[prev])；对拍两条：
    ① 首位输出频率 vs softmax(W[ctx[-1]])（覆盖 i=0 的条件分布索引——probs[pos-1] 正身位）；
    ② d0 被接受条件下的第二位 vs softmax(W[d0])（覆盖 i=1 的索引与回滚后续）。
    20000 试验，两位最大频率偏差 <0.012 视为通过。"""
    torch.manual_seed(SEED)
    V, GAMMA = 8, 3
    W = torch.randn(V, V) * 1.5

    class _Bigram:
        def __call__(self, ids):                     # ids (1,n) -> ((1,n,V),)——与 llama215 返回元组同构
            return (W[ids[0]].unsqueeze(0),)

    stub = _Bigram()
    ctx = [2, 5, 3]
    p1 = torch.softmax(W[ctx[-1]], dim=-1)           # 首位解析正身：P(·|ctx)
    d0 = int(p1.argmax())                            # 首选候选取 p1 峰值（保证条件样本量足够）
    drafts = [d0, (d0 + 3) % V, (d0 + 5) % V]        # 固定候选（贪心 draft 忠实口径）
    p2 = torch.softmax(W[drafts[0]], dim=-1)         # 第二位解析正身：P(·|ctx+d0)
    n_trial = 20000
    first = torch.zeros(V)
    second = torch.zeros(V)
    n_acc1 = 0
    for _ in range(n_trial):
        out, _ = speculative_step(stub, ctx, drafts, None, "cpu", torch.float32)
        first[out[0]] += 1
        if out[0] == drafts[0]:                      # d0 落地 → out[1] 应 ~ p2
            n_acc1 += 1
            second[out[1]] += 1
    dev1 = float((first / n_trial - p1).abs().max())
    dev2 = float((second / n_acc1 - p2).abs().max())
    print(f"[self_test] 首位分布 vs 解析正身 softmax(W[ctx[-1]])：max|freq-p|={dev1:.4f}"
          f"（{n_trial} 试验）")
    print(f"[self_test] 第二位（d0 接受条件）vs softmax(W[d0])：max|freq-p|={dev2:.4f}"
          f"（{n_acc1} 试验）")
    assert dev1 < 0.012 and dev2 < 0.012, "序列级分布对拍未过"
    print("[self_test] 通过——投机解码不改输出分布（speculative_step 本体的逐位实测证据）")


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
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
