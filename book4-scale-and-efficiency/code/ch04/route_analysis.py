# route_analysis.py —— Book4 第4章4.6节：小型MoE的类型与位置路由统计
# 用途：采用ch3概率质量辅助项训练mini MoE，比较初始化与训练后的路由。
# 类型轴不替代Mixtral的领域轴；1/E只作均匀独立参照；跨层编号不等于语义对应。
# 运行：工作区根目录 source ./env.sh 后运行本文件 --steps 500 --out-name run1。
# 历史MPS bf16训练约5—6分钟；统计为CPU fp32。新运行时间以实际设备为准。
# --no-train要求已有模型及配套JSON，重算结果另存；产物均在log/book4-ch04。
# 模型与数据流复用moe_mla_slots；均衡项是ch3式(3.6)的概率质量改编。
import argparse
import hashlib
from pathlib import Path
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

REPO_ROOT = os.getcwd()
OUT_DIR = os.path.join(REPO_ROOT, "log", "book4-ch04")
TOKENS_BIN = os.path.join(REPO_ROOT, "log", "book2-ch08", "tokens.bin")   # sp-8k，306M token 流
EVAL_BIN = os.path.join(REPO_ROOT, "log", "book2-ch08", "eval.bin")
SP_MODEL = os.path.join(REPO_ROOT, "log", "book2-ch08", "sp8k_owt.model")  # 词表标签（仅类型分类用）

VOCAB, D_MODEL, N_LAYER, N_HEAD, N_KV = 8192, 512, 6, 8, 4     # 探针 B 已验模块档（ch2/ch3 同款）
N_EXPERT, TOP_K, D_EXPERT = 8, 2, 704                          # 与Mixtral同专家数/top-k/无共享，宽度与层数不同
AUX_W = 0.1                                                    # load-loss 权重（ch3 aux 臂同款）
BATCH, BLOCK = 16, 512
PEAK_LR, MIN_LR_RATIO, WARMUP = 1e-3, 0.1, 100
PROBE_B, PROBE_T = 8, 2048                                     # 探针 16,384 token（eval.bin 首段）
ACTIVE_PARAMS = 26_089_984  # 历史双表、不含router
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
    assert act == ACTIVE_PARAMS, f"等激活口径被破坏：{act}"
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
    was_training=model.training
    model.eval()
    try:
        model(probe_x)
    finally:
        for h in hooks:h.remove()
        model.train(was_training)
    L = len(idx_snap)
    idx0 = torch.stack([s[:, 0].view(*probe_x.shape) for s in idx_snap])   # (L,B,T) 首选项
    counts = torch.stack([torch.bincount(s.reshape(-1), minlength=N_EXPERT) for s in idx_snap])
    return idx0, counts                              # (L,B,T) / (L,E)


def routing_stats(idx0):
    """(L,B,T)首选索引：观察值、逐层边际和独立基线；编号相同不代表同一语义专家。"""
    if idx0.ndim!=3 or min(idx0.shape[:2])<1 or idx0.shape[2]<2:
        raise ValueError("需要非空(L,B,T)，T至少为2")
    if idx0.min()<0 or idx0.max()>=N_EXPERT:
        raise ValueError("专家索引越界")
    L,B,T=idx0.shape
    marginal=[];left=[];right=[]
    for row in idx0:
        marginal.append(torch.bincount(row.flatten(),minlength=N_EXPERT).double()/(B*T))
        left.append(torch.bincount(row[:,:-1].flatten(),minlength=N_EXPERT).double()/(B*(T-1)))
        right.append(torch.bincount(row[:,1:].flatten(),minlength=N_EXPERT).double()/(B*(T-1)))
    return {"repetition_per_layer":(idx0[:,:,1:]==idx0[:,:,:-1]).double().mean((1,2)).tolist(),
        "baseline_uniform":1/N_EXPERT,
        "first_choice_marginal_per_layer":[p.tolist() for p in marginal],
        "baseline_independent_per_layer":[float((p*q).sum()) for p,q in zip(left,right)],
        "baseline_shuffle_per_layer":[float(((p*B*T)*(p*B*T-1)).sum()/(B*T*(B*T-1))) for p in marginal],
        "cross_layer_repetition":(idx0[1:]==idx0[:-1]).double().mean((1,2)).tolist(),
        "cross_layer_independent_baseline":[float((p*q).sum()) for p,q in zip(marginal[:-1],marginal[1:])]}


def assignment_matrix(idx0,type_ids,probe_x):
    """保留旧全层合并矩阵供对照，另给L×type×E计数；缺失类别返回null而非NaN。"""
    names=["word","digit","punct","space","mixed"];L=idx0.shape[0]
    probe_types=type_ids[probe_x.reshape(-1)].cpu()
    per=[]
    for row in idx0.cpu().reshape(L,-1):
        count=torch.bincount(probe_types*N_EXPERT+row,minlength=len(names)*N_EXPERT)
        per.append(count.reshape(len(names),N_EXPERT).numpy())
    mat=np.stack(per).sum(0);totals=mat.sum(1)
    frac=[];dev={}
    for name,row,total in zip(names,mat,totals):
        if total:
            prob=row/total;frac.append(prob.tolist());dev[name]=float(np.abs(prob-1/N_EXPERT).max())
        else:frac.append([None]*N_EXPERT);dev[name]=None
    return {"type_names":names,"counts":mat.tolist(),"frac":frac,"max_dev_from_uniform":dev,
        "per_layer_counts":np.stack(per).tolist(),
        "probe_token_counts":dict(zip(names,map(int,totals))), # 兼容旧键，含L次重复计数
        "type_position_counts":dict(zip(names,torch.bincount(probe_types,minlength=len(names)).tolist())),
        "aggregation_note":"全层同编号合并仅为描述；不同层的同编号不是同一专家"}


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(steps, out_name, no_train):
    if steps<1:raise ValueError("steps必须为正整数")
    if not out_name or Path(out_name).name != out_name or out_name in {".", ".."}:
        raise ValueError("out-name须为单个文件名后缀")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    model_path = os.path.join(OUT_DIR, f"route_model_{out_name}.pt")
    json_path = os.path.join(OUT_DIR, f"route_analysis_{out_name}.json")
    if no_train and not os.path.exists(model_path):
        raise FileNotFoundError("--no-train要求已有模型，缺失时不会退回训练")
    if not no_train and (os.path.exists(model_path) or os.path.exists(json_path)):
        raise FileExistsError("已有同名结果，请使用新的out-name")
    result_path=json_path if not no_train else os.path.join(OUT_DIR,f"route_analysis_{out_name}_restat.json")
    if os.path.exists(result_path):raise FileExistsError(result_path)

    old = None
    if no_train:
        with open(json_path, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("steps") != steps or old.get("train", {}).get("steps") != steps:
            raise ValueError("--steps与配套训练记录不符")
        if old.get("model_sha256") != file_sha(model_path):
            raise ValueError("模型与配套记录不匹配，或旧记录缺少模型哈希")
    ev = np.memmap(EVAL_BIN, dtype=np.uint16, mode="r")
    if len(ev) < PROBE_B * PROBE_T:
        raise ValueError("评估流不足一个完整探针批")
    span = np.asarray(ev[:PROBE_B * PROBE_T], dtype=np.int64)
    probe_x = torch.from_numpy(span.reshape(PROBE_B, PROBE_T))       # (8,2048) held-out

    probe_identity = {"shape": list(probe_x.shape),
                      "tokens_sha256": hashlib.sha256(span.tobytes()).hexdigest(),
                      "tokenizer_sha256": file_sha(SP_MODEL)}
    model, cfg = build_model()
    rep = {"seed": SEED, "device": str(device), "steps": steps,
           "config": {"d": D_MODEL, "L": N_LAYER, "E": N_EXPERT, "k": TOP_K, "w": D_EXPERT,
                      "V": VOCAB, "total_params": MOE_PARAMS, "active_params": ACTIVE_PARAMS,
                      "active_convention":"双表、不含router",
                      "aux": f"概率质量CV²，w={AUX_W}"},
           "probe": {"tokens": PROBE_B * PROBE_T, "source": "eval.bin 首段 held-out", **probe_identity},
           "meta": {"metric1": "token 类型×专家 首选项分布（Mixtral 域轴在 OWT 无标签→改 token 类型轴）",
                    "metric2": "逐层相邻位置首选编号重复率；1/E仅为均匀独立参照",
                    "metric3": "相邻层当前编号匹配率；不直接度量语义对应"}}

    if no_train:
        if old.get("seed") != SEED or old.get("config") != rep["config"] or old.get("probe") != rep["probe"]:
            raise ValueError("模型配置、种子或探针与配套记录不符")
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        rep["init"], rep["train"] = old["init"], old["train"]
        rep["device"] = old["device"]  # 原训练设备，不能改写为本次重算所在设备
        rep["restat_device"] = "cpu"
        print(f"[--no-train] 复用模型档 {model_path}")
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
        if steps>stream.max_step:raise ValueError("步数超过当前数据流可用范围")
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
            if not torch.isfinite(loss):
                raise FloatingPointError("非有限损失，停止此次运行")
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
    print(f"  ② 连续 token 首选项重复率（均匀独立参照 1/E={1 / N_EXPERT:.3f}）: "
          f"{['%.3f' % r for r in rep['final']['repetition_per_layer']]}")
    print(f"  ③ 层间重复率（本书补充口径）: "
          f"{['%.3f' % r for r in rep['final']['cross_layer_repetition']]}")
    print(f"  ① 分工矩阵行分布对均匀的最大偏离: "
          f"{ {k: (round(v, 3) if v is not None else None) for k, v in rep['final']['assignment_matrix']['max_dev_from_uniform'].items()} }")
    print(f"  负载计数 max/min（各层）: {rep['final']['load_max_over_min']}")

    rep["model_sha256"] = file_sha(model_path)
    rep["statistics_dtype"] = "float32"
    rep["torch_version"] = torch.__version__
    rep["verdict"] = "DONE"
    with open(result_path, "x", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"[产物] {result_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch4 自家版路由统计（mini MoE 自训 + Mixtral §5 三指标）")
    ap.add_argument("--steps", type=int, default=500, help="训练步数（fast 档 500）")
    ap.add_argument("--out-name", default="run1", help="产物后缀（防覆写）")
    ap.add_argument("--no-train", action="store_true", help="复用已存模型档只重跑统计")
    args = ap.parse_args()
    main(args.steps, args.out_name, args.no_train)
