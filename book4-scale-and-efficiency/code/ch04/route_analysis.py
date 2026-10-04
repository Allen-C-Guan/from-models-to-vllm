# route_analysis.py —— Book4 ch4 自家版路由统计：在自训 mini MoE 上复现 Mixtral §5 的三件观察
# 用途：正文 4.7 的实验正身。训练一台 ch3 aux 臂同款 mini MoE（E8/top2/w704 + load-loss w=0.1，
#       Book2 五件套、500 步 fast 档——真实旗舰必带某种均衡，路由统计要在「健康」的 router 上做），
#       然后在 held-out 探针（eval.bin 首段 16,384 token）上量三件指标，对照 Mixtral §5 口径：
#       ① 专家分工矩阵：token 类型（词/数字/标点/空白/混合）× 专家的首选项分布——Mixtral 的
#          域轴（ArXiv/PubMed/PhilPapers）在 OWT 子采样语料上无标签，改用 token 类型作内容轴
#          （与其 Fig 8 的句法轴同族、更细；诚实条款见 JSON meta）；
#       ② 位置局部性：逐层「连续 token 首选项重复率」——Mixtral Table 5 同口径（基线 1/E=12.5%）；
#       ③ 层间重复率：同 token 在相邻两层的首选项是否同专家——本书补充口径（论文未报；
#          预期≈基线：每层专家群独立，「8 个专家」不是全模型共享 8 个）。
#       init（训练前）与 final（训练后）各拍一张快照——重复率是「训出来的还是初始化就有的」
#       由两线对照回答。
# 所属章节：Book4 第 4 章（4.7 节 + 图 4.2 数据源）。
# 运行方式：cd 工作区根目录 && source env.sh && \
#           python code/ch04/route_analysis.py --steps 500 --out-name run1
#   （MPS bf16 训练 ≈5-6 min + CPU fp32 统计秒级；--no-train 复用已存模型档只重跑统计；
#    产物 log/book4-ch04/route_analysis_{out}.json + route_model_{out}.pt，不入库）
# 模块来源：不私搭模型——整机/训练数据 import 自正身 moe_mla_slots（其内部 bootstrap 引
#           Book3 llama_slots）；aux 损失承 ch3 式 (3.6) 的概率质量改编（改编说明同 ch3 脚本头注）。
import argparse
import json
import math
import os
import sys
import time
import unicodedata

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "00-feasibility")))
from moe_mla_slots import (SEED, SlotConfig, SlotLLaMA, TokenStream, activated_account,  # noqa: E402
                           hand_account)

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book4-ch04")
TOKENS_BIN = os.path.join(REPO_ROOT, "log", "book2-ch08", "tokens.bin")   # sp-8k，306M token 流
EVAL_BIN = os.path.join(REPO_ROOT, "log", "book2-ch08", "eval.bin")
SP_MODEL = os.path.join(REPO_ROOT, "log", "book2-ch08", "sp8k_owt.model")  # 词表标签（仅类型分类用）

VOCAB, D_MODEL, N_LAYER, N_HEAD, N_KV = 8192, 512, 6, 8, 4     # 探针 B 已验模块档（ch2/ch3 同款）
N_EXPERT, TOP_K, D_EXPERT = 8, 2, 704                          # E8/top2——与 Mixtral 同几何（无共享）
AUX_W = 0.1                                                    # load-loss 权重（ch3 aux 臂同款）
BATCH, BLOCK = 16, 512
PEAK_LR, MIN_LR_RATIO, WARMUP = 1e-3, 0.1, 100
PROBE_B, PROBE_T = 8, 2048                                     # 探针 16,384 token（eval.bin 首段）
MOE_PARAMS = 65_042_944                                        # 探针 B 逐位锚点


def lr_at(step, total):
    """Book2 五件套日程：线性 warmup 100 步 + 余弦 1e-3 → 1e-4（ch3 同款）。"""
    if step <= WARMUP:
        return PEAK_LR * step / WARMUP
    prog = (step - WARMUP) / max(1, total - WARMUP)
    return PEAK_LR * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * prog)))


def build_model():
    """ch3 aux 臂同款整机（E8/top2/w=704 Mixtral 式）；参数账 assert 逐位。"""
    cfg = SlotConfig(vocab_size=VOCAB, hidden_size=D_MODEL, num_hidden_layers=N_LAYER,
                     num_attention_heads=N_HEAD, num_key_value_heads=N_KV, intermediate_size=1408,
                     moe=dict(n_expert=N_EXPERT, top_k=TOP_K, expert_dim=D_EXPERT,
                              n_shared=0, variant="teach", mode="mixtral"))
    torch.manual_seed(SEED)
    m = SlotLLaMA(cfg)
    assert m.n_params() == hand_account(cfg)["total"] == MOE_PARAMS, "参数对账失败（探针 B 锚点）"
    act, _ = activated_account(cfg)
    assert act == 26_089_984, f"等激活口径被破坏：{act}"
    return m, cfg


def attach_aux_hook(model):
    """load-loss 接口（承 ch3 式 (3.6) 改编）：前向钩子捕每层 router logits (N,E)，
    返回 (hooks, stash, loss_fn)。Load_i = Σ_x softmax(x)_i 概率质量（可导代理），
    损失 = w·mean_layers CV(Load)²——度量环节另用硬计数，两者分开念（ch3 诚实条款）。"""
    stash, hooks = [], []

    def hook(mod, inp, out):
        stash.append(out[0])                        # (N,E) router logits，带梯度图

    for layer in model.model.layers:
        hooks.append(layer.mlp.gate.register_forward_hook(hook))

    def loss_fn():
        terms = []
        for logits in stash:
            probs = logits.float().softmax(dim=-1)  # (N,E)
            load = probs.sum(dim=0)                 # (E,)
            terms.append((load.std(unbiased=False) / load.mean()) ** 2)
        return AUX_W * sum(terms) / len(terms)

    return hooks, stash, loss_fn


def token_type_table():
    """sp-8k 词表的 id→类型表：sentencepiece 解码 piece 后按字符类分类。
    类：word（纯字母）/ digit（纯数字）/ punct（标点符号）/ space（▁ 空白件）/ mixed（其余）。
    special/byte-fallback 件罕见，并入 mixed（JSON meta 登记）。"""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=SP_MODEL)
    types = []
    for tid in range(VOCAB):
        piece = sp.id_to_piece(tid)
        if piece.startswith("<") and piece.endswith(">"):   # <unk>/<s>/</s>/<0xXX>
            types.append("mixed")
            continue
        content = piece[1:] if piece.startswith("▁") else piece
        if content == "":
            types.append("space")
        elif all(unicodedata.category(ch)[0] in "PS" for ch in content):
            types.append("punct")
        elif content.isdigit():
            types.append("digit")
        elif content.isalpha():
            types.append("word")
        else:
            types.append("mixed")
    return types


@torch.no_grad()
def capture_routing(model, probe_x):
    """喂探针批，钩每层 gate 捕 top-k 索引。返回：
    idx0 (L,B,T) 首选项专家号（topk 有序，第 0 槽=第一选择） + counts (L,E) 硬计数。"""
    idx_snap, hooks = [], []

    def hook(mod, inp, out):
        idx_snap.append(out[2].detach())            # (N,k) top-k 索引

    for layer in model.model.layers:
        hooks.append(layer.mlp.gate.register_forward_hook(hook))
    model.eval()
    model(probe_x)                                   # 前向一次（lm_head 也算，无所谓）
    for h in hooks:
        h.remove()
    L = len(idx_snap)
    idx0 = torch.stack([s[:, 0].view(PROBE_B, PROBE_T) for s in idx_snap])   # (L,B,T) 首选项
    counts = torch.stack([torch.bincount(s.reshape(-1), minlength=N_EXPERT) for s in idx_snap])
    return idx0, counts                              # (L,B,T) / (L,E)


def routing_stats(idx0):
    """三件指标（口径见文件头）。idx0 (L,B,T) 首选项专家号。"""
    L, B, T = idx0.shape
    stats = {}
    # ② 逐层连续 token 首选项重复率（Mixtral Table 5 口径；基线 1/E）
    rep_layer = []
    for l in range(L):
        same = (idx0[l, :, 1:] == idx0[l, :, :-1])   # (B,T-1) 相邻位置同专家？
        rep_layer.append(same.float().mean().item())
    stats["repetition_per_layer"] = rep_layer
    stats["baseline_uniform"] = 1.0 / N_EXPERT       # = 0.125
    # ③ 层间重复率（本书补充口径）：同 token 相邻层首选项是否同专家
    cross = []
    for l in range(L - 1):
        same = (idx0[l] == idx0[l + 1])              # (B,T)
        cross.append(same.float().mean().item())
    stats["cross_layer_repetition"] = cross
    return stats


def assignment_matrix(idx0, type_ids, probe_x):
    """① 专家分工矩阵：token 类型 × 专家 的首选项计数（全层累计）→ 行归一化分布。

    type_ids (V,) int：每个词表 id 的类型号（token_type_table 的输出转号）；
    probe_x (B,T)：探针 token id——各位置的类型号按它查表（idx0 是专家号，别混用）。"""
    type_names = ["word", "digit", "punct", "space", "mixed"]
    n_type = len(type_names)
    L = idx0.shape[0]
    probe_types = type_ids[probe_x.reshape(-1)]                      # (B·T,) 各位置 token 的类型号
    mat = np.zeros((n_type, N_EXPERT), dtype=np.int64)
    flat_sel = idx0.reshape(L, -1)                                   # (L, B·T) 首选项专家号
    for l in range(L):
        sel = torch.bincount(probe_types * N_EXPERT + flat_sel[l], minlength=n_type * N_EXPERT)
        mat += sel.reshape(n_type, N_EXPERT).numpy()
    frac = mat / mat.sum(axis=1, keepdims=True)
    dev = {type_names[i]: float(np.max(np.abs(frac[i] - 1.0 / N_EXPERT))) for i in range(n_type)}
    return {"type_names": type_names, "counts": mat.tolist(),
            "frac": frac.tolist(), "max_dev_from_uniform": dev,
            "probe_token_counts": {type_names[i]: int(mat[i].sum()) for i in range(n_type)}}


def main(steps, out_name, no_train):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    model_path = os.path.join(OUT_DIR, f"route_model_{out_name}.pt")
    json_path = os.path.join(OUT_DIR, f"route_analysis_{out_name}.json")

    ev = np.memmap(EVAL_BIN, dtype=np.uint16, mode="r")
    span = np.asarray(ev[:PROBE_B * PROBE_T], dtype=np.int64)
    probe_x = torch.from_numpy(span.reshape(PROBE_B, PROBE_T))       # (8,2048) held-out

    model, cfg = build_model()
    rep = {"seed": SEED, "device": str(device), "steps": steps,
           "config": {"d": D_MODEL, "L": N_LAYER, "E": N_EXPERT, "k": TOP_K, "w": D_EXPERT,
                      "V": VOCAB, "total_params": MOE_PARAMS, "active_params": 26_089_984,
                      "aux": f"load-loss w={AUX_W}（ch3 aux 臂同款——健康 router 上做统计）"},
           "probe": {"tokens": PROBE_B * PROBE_T, "source": "eval.bin 首段 held-out"},
           "meta": {"metric1": "token 类型×专家 首选项分布（Mixtral 域轴在 OWT 无标签→改 token 类型轴）",
                    "metric2": "逐层连续 token 首选项重复率（Mixtral Table 5 同口径，基线 1/E）",
                    "metric3": "层间重复率=本书补充口径（论文未报，预期≈基线）"}}

    if no_train and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"[--no-train] 复用模型档 {model_path}")
        if os.path.exists(json_path):                     # init/train 记录从旧 JSON 续（同种子 CPU 口径可复现）
            with open(json_path, encoding="utf-8") as f:
                old = json.load(f)
            rep["init"], rep["train"] = old.get("init"), old.get("train")
    else:
        # ---- init 快照（训练前）----
        model_cpu = model.to("cpu").float()
        idx0, counts = capture_routing(model_cpu, probe_x)
        rep["init"] = routing_stats(idx0)
        rep["init"]["assignment_matrix"] = None                      # init 不做分工矩阵（正文只看 final）
        rep["init"]["load_max_over_min"] = [round(float(c.max() / max(c.min(), 1)), 1)
                                            for c in counts]
        print(f"[init] 连续重复率 {['%.3f' % r for r in rep['init']['repetition_per_layer']]}")

        # ---- 训练（aux 臂协议；与后台下载共机时计时仅作工期参考、不入正文） ----
        model = model_cpu.to(device)
        hooks, stash, loss_fn = attach_aux_hook(model)
        opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=0.1)
        stream = TokenStream(TOKENS_BIN, BATCH, BLOCK)
        model.train()
        t0 = time.perf_counter()
        losses = []
        for step in range(1, steps + 1):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, steps)
            stash.clear()
            x, y = stream.batch(step, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16) \
                    if device.type == "mps" else torch.autocast("cpu", enabled=False):
                _, loss_ce = model(x, y)
                loss = loss_ce + loss_fn()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss_ce.item())
            if step % 100 == 0:
                print(f"  step {step}/{steps} train CE {np.mean(losses[-100:]):.4f}", flush=True)
        for h in hooks:
            h.remove()
        wall = time.perf_counter() - t0
        print(f"[train] {steps} 步完成，墙钟 {wall / 60:.1f} min（含共机干扰，不作正文数字）")
        rep["train"] = {"steps": steps, "ce_first10": float(np.mean(losses[:10])),
                        "ce_last10": float(np.mean(losses[-10:])), "wall_min": round(wall / 60, 1)}
        model_cpu = model.to("cpu").float()
        torch.save(model_cpu.state_dict(), model_path)

    # ---- final 快照（训练后；CPU fp32 统计口径） ----
    model_cpu = model.to("cpu").float()
    idx0, counts = capture_routing(model_cpu, probe_x)
    rep["final"] = routing_stats(idx0)
    type_ids = torch.tensor([["word", "digit", "punct", "space", "mixed"].index(t)
                             for t in token_type_table()])           # (V,) 类型号表
    rep["final"]["assignment_matrix"] = assignment_matrix(idx0, type_ids, probe_x)
    rep["final"]["load_max_over_min"] = [round(float(c.max() / max(c.min(), 1)), 1) for c in counts]
    rep["final"]["load_counts_layer0"] = counts[0].tolist()

    print("\n[final] 三件指标读数：")
    print(f"  ② 连续 token 首选项重复率（基线 1/E={1 / N_EXPERT:.3f}）: "
          f"{['%.3f' % r for r in rep['final']['repetition_per_layer']]}")
    print(f"  ③ 层间重复率（本书补充口径）: "
          f"{['%.3f' % r for r in rep['final']['cross_layer_repetition']]}")
    print(f"  ① 分工矩阵行分布对均匀的最大偏离: "
          f"{ {k: round(v, 3) for k, v in rep['final']['assignment_matrix']['max_dev_from_uniform'].items()} }")
    print(f"  负载健康度 max/min（各层）: {rep['final']['load_max_over_min']}")

    rep["verdict"] = "DONE"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"[产物] {json_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch4 自家版路由统计（mini MoE 自训 + Mixtral §5 三指标）")
    ap.add_argument("--steps", type=int, default=500, help="训练步数（fast 档 500）")
    ap.add_argument("--out-name", default="run1", help="产物后缀（防覆写）")
    ap.add_argument("--no-train", action="store_true", help="复用已存模型档只重跑统计")
    args = ap.parse_args()
    main(args.steps, args.out_name, args.no_train)
