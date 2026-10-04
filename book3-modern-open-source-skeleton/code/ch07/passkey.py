# passkey.py —— Book3 ch7 passkey 检索协议（ppl 之外的第二口径；KV cache 增量解码的正身实现）
# 用途：对 extrapolate.py 训好的 2k 窗 mini ckpt 做「大海捞针」检索评测——真实 filler（eval.bin heldout
#       段）中段藏一个 5 位 passkey（needle），末尾提问（query），贪心解码 12 token，解码文本包含
#       passkey 即命中。双口径：in-window（1.5k < 训练窗 2k）与 out-window（4k / 8k，2×/4× 越窗）；
#       四干预（none/PI/NTK/YaRN-lite）× 三窗长逐一跑——「ppl 高 ≠ 检索失效」的实验落地
#      （papers/05 §1.3：YaRN 附录 B.2 "perplexity may not be a great indicator…"）。
#       解码用 KV cache 增量式（ch6 增量生成算法的第一次实战：prefill 一次 + 每步 O(t) 单 query），
#       启动前跑「缓存解码 vs 全量重算解码」的自检（同 token 序列 assert，防缓存实现错）。
# 协议细节：needle = " The passkey is {key}. Remember it. "，query = " The passkey is"；
#       同一 (窗长, 深度, trial) 的上下文跨四干预完全复用（配对比较，压方差）；命中判定 = 精确
#       包含 5 位数字串。
# 运行方式：python passkey.py [--ckpt log/book3-ch07/mini2k_full.pt] [--out-name full]
#           [--trials 8] [--depths 0.25,0.5,0.75] [--lengths 1536,4096,8192]
#           fast 档：--trials 4 --depths 0.5（协议跑通，数字仅协议验证）；full 档：全套。
# 产物（log/book3-ch07/，不入库）：passkey_{out}.json（准确率矩阵 + 示例 + 自检记录）。
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import sentencepiece as spm
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "00-feasibility")))
from llama_slots import LLaMA, LLaMAConfig, apply_rope, repeat_kv     # noqa: E402
from extrapolate import INTERVENTIONS, forward_with_rope, rope_cache_for   # noqa: E402  四干预正身（同目录）

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book3-ch07")
EVAL_BIN = os.path.join(REPO_ROOT, "log", "book2-ch08", "eval.bin")     # filler 来源（heldout，与训练互斥）
SP_MODEL = os.path.join(REPO_ROOT, "log", "book2-ch08", "sp8k_owt.model")
SEED = 20261002
DECODE_N = 12       # 贪心解码 token 数
QUERY = " The passkey is"


def layer_step(layer, x, pos_lo, caches, cos, sin):
    """单层前进一步（prefill 或解码），带 KV cache。x (1,t,C)，pos_lo = 本步起始绝对位置。

    k/v 追加进 caches[layer_idx]（(1,h_kv,t_cache,hd)）；注意力 t_query x t_cache——解码步
    t=1 时全历史可见（因果性），prefill 步用 tril 掩码。返回 (1,t,C)。
    """
    attn = layer.self_attn
    B, t, C = x.shape
    h, h_kv, hd = attn.n_head, attn.n_kv_head, attn.head_dim
    h_in = layer.input_layernorm(x)                                        # (1,t,C)
    q = attn.q_proj(h_in).view(B, t, h, hd).transpose(1, 2)                # (1,h,t,hd)
    k = attn.k_proj(h_in).view(B, t, h_kv, hd).transpose(1, 2)             # (1,h_kv,t,hd)
    v = attn.v_proj(h_in).view(B, t, h_kv, hd).transpose(1, 2)             # (1,h_kv,t,hd)
    cos_t = cos[:, pos_lo:pos_lo + t]                                      # (1,t,hd) 本步位置行
    sin_t = sin[:, pos_lo:pos_lo + t]
    q, k = apply_rope(q, k, cos_t, sin_t)                                  # 只旋本步 q/k
    if caches[layer._idx] is not None:
        k = torch.cat([caches[layer._idx][0], k], dim=2)                   # (1,h_kv,t_cache+t,hd)
        v = torch.cat([caches[layer._idx][1], v], dim=2)
    caches[layer._idx] = (k, v)
    k_rep, v_rep = repeat_kv(k, attn.n_rep), repeat_kv(v, attn.n_rep)      # -> (1,h,t_cache+t,hd)
    att = (q @ k_rep.transpose(-2, -1)) * attn.scaling                     # (1,h,t,t_cache+t)
    tc = k_rep.shape[2]
    if t > 1:                                                              # prefill：因果掩码（历史列全可见）
        hist = tc - t                                                      # 已缓存列（右侧新块才需 tril）
        mask = torch.ones(t, tc, device=x.device)
        mask[:, hist:] = torch.tril(torch.ones(t, t, device=x.device))
        att = att.masked_fill(mask.view(1, 1, t, tc) == 0, float("-inf"))
    att = torch.softmax(att.float(), dim=-1).to(q.dtype)                   # fp32 softmax（对齐 HF）
    y = (att @ v_rep).transpose(1, 2).contiguous().view(B, t, C)           # (1,t,C)
    x = x + attn.o_proj(y)                                                 # 残差一
    x = x + layer.mlp(layer.post_attention_layernorm(x))                   # 残差二
    return x


@torch.no_grad()
def greedy_decode(model, ctx_ids, n_new, intervention, device):
    """KV cache 贪心解码（正身）：prefill 一次保留 caches，之后每步喂 1 个 token。

    ctx_ids (n,) -> 新增 n_new 个 token id。干预通过 rope_cache_for 注入（免重训，权重不动）。
    位置口径：prefill 覆盖绝对位置 0..n-1；第 j 个新 token（j≥1）落在绝对位置 n+j-1——
    RoPE 的 cos/sin 行号必须用绝对位置（此处曾出 bug：从 0 起数，自检拦截）。
    """
    n = len(ctx_ids)
    cos, sin = rope_cache_for(intervention, n + n_new, device)             # (1,n+n_new,hd)
    caches = [None] * len(model.model.layers)
    x = torch.tensor([ctx_ids], dtype=torch.long, device=device)           # (1,n)
    h = model.model.embed_tokens(x)                                        # (1,n,C)
    for layer in model.model.layers:                                       # prefill（逐层带 cache，位置 0..n-1）
        h = layer_step(layer, h, 0, caches, cos, sin)
    logits = model.lm_head(model.model.norm(h[:, -1:, :]))                 # (1,1,V) 取末位
    out = [int(logits[0, -1].argmax())]
    pos = n                                                                # 下一个被喂入 token 的绝对位置
    for _ in range(n_new - 1):
        tok = torch.tensor([[out[-1]]], dtype=torch.long, device=device)   # (1,1)
        h = model.model.embed_tokens(tok)                                  # (1,1,C)
        for layer in model.model.layers:                                   # 解码步：单 query O(t)
            h = layer_step(layer, h, pos, caches, cos, sin)
        pos += 1
        logits = model.lm_head(model.model.norm(h))                        # (1,1,V)
        out.append(int(logits[0, -1].argmax()))
    return out


@torch.no_grad()
def greedy_decode_full_recompute(model, ctx_ids, n_new, intervention, device):
    """全量重算贪心（探针 G 口径）——仅自检用：每步对整段 ctx 重新前向。"""
    ctx = list(ctx_ids)
    for _ in range(n_new):
        x = torch.tensor([ctx], dtype=torch.long, device=device)           # (1,t)
        cos, sin = rope_cache_for(intervention, len(ctx), device)
        logits, _ = forward_with_rope(model, x, cos, sin)
        ctx.append(int(logits[0, -1].argmax()))
    return ctx[len(ctx_ids):]


def self_check(model, device):
    """协议守门：缓存解码 vs 全量重算解码，同一小上下文上 token 序列必须逐位一致。"""
    rng = np.random.default_rng(SEED)
    ev = np.memmap(EVAL_BIN, dtype=np.uint16, mode="r")
    start = int(rng.integers(0, len(ev) - 512))
    ctx = np.asarray(ev[start:start + 320], dtype=int).tolist()
    rec = {}
    for iv in INTERVENTIONS:
        a = greedy_decode(model, ctx, 8, iv, device)
        b = greedy_decode_full_recompute(model, ctx, 8, iv, device)
        rec[iv] = {"cached": a, "full": b, "identical": a == b}
        assert a == b, f"缓存解码自检失败（{iv}）：{a} != {b}"
    rec["note"] = "320 token 上下文 × 8 步 × 四干预，缓存路径与全量重算逐 token 一致"
    return rec


def build_contexts(length, depth, trials, sp):
    """构造 (length, depth) 的 trials 个上下文（同种子确定性；跨干预复用同一批上下文）。

    结构：filler_a + needle + filler_b + query；filler 取 eval.bin 随机段（真实文本），
    needle 深度 = filler_a 占比 depth。返回 [{ctx, key, ctx_len}]。
    """
    ev = np.memmap(EVAL_BIN, dtype=np.uint16, mode="r")
    needle_tpl = " The passkey is {key}. Remember it. "
    query_ids = sp.EncodeAsIds(QUERY)
    out = []
    for trial in range(trials):
        rng = np.random.default_rng(SEED + trial * 1000 + length + int(depth * 100))
        key = int(rng.integers(10000, 99999))
        needle_ids = sp.EncodeAsIds(needle_tpl.format(key=key))
        nfill = length - len(needle_ids) - len(query_ids)
        start = int(rng.integers(0, len(ev) - nfill - 1))
        fill_a = ev[start:start + int(nfill * depth)]
        fill_b = ev[start + int(nfill * depth):start + nfill]
        ctx = np.concatenate([fill_a, np.array(needle_ids, dtype=np.uint16), fill_b,
                              np.array(query_ids, dtype=np.uint16)]).astype(np.int64)
        out.append({"ctx": ctx, "key": key, "ctx_len": int(len(ctx))})
    return out


def main():
    ap = argparse.ArgumentParser(description="ch7 passkey 检索：四干预 × in/out-window 双口径")
    ap.add_argument("--ckpt", type=str, default=os.path.join(OUT_DIR, "mini2k_full.pt"))
    ap.add_argument("--out-name", type=str, default=None)
    ap.add_argument("--trials", type=int, default=8, help="每 (窗长,深度) 试验数（fast=4 / full=8）")
    ap.add_argument("--depths", type=str, default="0.5", help="needle 深度列表（fast=0.5 / full=0.25,0.5,0.75）")
    ap.add_argument("--lengths", type=str, default="1536,4096,8192",
                    help="上下文长（1536=in-window 基线；4096/8192=out-window）")
    args = ap.parse_args()
    out_name = args.out_name or ("fast" if args.trials <= 4 else "full")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = LLaMAConfig(**{**{k: ck["config"][k] for k in
                            ["vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                             "num_attention_heads", "num_key_value_heads", "max_position_embeddings"]},
                        "rope_theta": 10000.0})
    model = LLaMA(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    for i, layer in enumerate(model.model.layers):     # 层号标记（caches 索引用）
        layer._idx = i
    sp = spm.SentencePieceProcessor()
    sp.Load(SP_MODEL)

    lengths = [int(x) for x in args.lengths.split(",")]
    depths = [float(x) for x in args.depths.split(",")]
    print(f"[passkey] ckpt={args.ckpt}（{ck.get('steps')} 步）| lengths={lengths} | depths={depths} | "
          f"trials={args.trials} | 四干预={list(INTERVENTIONS)}", flush=True)

    # ---- 自检：缓存解码 = 全量重算解码 ----
    t0 = time.perf_counter()
    check = self_check(model, device)
    print(f"[自检] 缓存解码 vs 全量重算：四干预全部逐 token 一致（{time.perf_counter()-t0:.1f}s）",
          flush=True)

    # ---- 主循环：四干预 × 窗长 × 深度 × trials（同上下文跨干预复用） ----
    results = {iv: {} for iv in INTERVENTIONS}
    t0 = time.perf_counter()
    for length in lengths:
        for depth in depths:
            cases = build_contexts(length, depth, args.trials, sp)
            cell = f"len{length}@d{depth}"
            for iv in INTERVENTIONS:
                hits, examples = 0, []
                for c in cases:
                    out_ids = greedy_decode(model, c["ctx"].tolist(), DECODE_N, iv, device)
                    text = sp.DecodeIds([int(t) for t in out_ids])
                    ok = str(c["key"]) in text
                    hits += ok
                    if len(examples) < 2:
                        examples.append({"key": c["key"], "ctx_len": c["ctx_len"],
                                         "decoded": text[:60], "hit": ok})
                results[iv][cell] = {"accuracy": hits / args.trials, "hits": hits,
                                     "trials": args.trials,
                                     "mean_ctx_len": round(sum(c["ctx_len"] for c in cases)
                                                           / len(cases), 1),
                                     "examples": examples}
                print(f"[passkey] {cell:16s} {iv:9s}  {hits}/{args.trials}", flush=True)
    wall = time.perf_counter() - t0

    # ---- 汇总：逐干预 × 窗长的跨深度平均（窗长级读数） ----
    summary = {}
    for iv in INTERVENTIONS:
        summary[iv] = {}
        for length in lengths:
            cells = [v for k, v in results[iv].items() if k.startswith(f"len{length}@")]
            if cells:
                summary[iv][str(length)] = round(sum(c["accuracy"] for c in cells) / len(cells), 4)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, f"passkey_{out_name}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "device": str(device),
            "ckpt": args.ckpt, "ckpt_steps": ck.get("steps"),
            "protocol": ("真实 filler（eval.bin heldout 随机段）+ needle \" The passkey is {key}. "
                         "Remember it. \"（深度 = filler_a 占比）+ query \" The passkey is\"；"
                         f"贪心 {DECODE_N} token（KV cache 增量解码）；命中 = 解码文本含 5 位 passkey"),
            "dual_criteria": {"in-window": "len1536（< 训练窗 2048）",
                              "out-window": "len4096/8192（2×/4× 越窗）"},
            "interventions": {k: v["def"] for k, v in INTERVENTIONS.items()},
            "paired_design": "同一 (窗长,深度,trial) 上下文跨四干预完全复用（配对比较）",
            "trials": args.trials, "depths": depths, "lengths": lengths,
            "decode": "KV cache 增量（prefill 一次 + 单 query 步进）——ch6 增量算法的实战首演",
            "argv": " ".join(sys.argv[1:]),
        }, "self_check": check, "results": results, "by_length_summary": summary,
            "wallclock_sec": round(wall, 1)}, f, ensure_ascii=False, indent=2)
    print(f"\n[完成] {wall:.0f}s -> {out_json}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
