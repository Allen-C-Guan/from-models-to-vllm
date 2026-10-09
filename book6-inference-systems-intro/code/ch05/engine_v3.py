# engine_v3.py —— Book6 大项目 v3：分页 KV 引擎（BlockTable + BlockFreeList——虚拟内存思想进缓存管理）
# 用途：ch5 正身。对 v2 的改造：KV 从「每请求一段连续预分配 buffer」换成「全局物理块池 + 每请求块表」——
#   ①BlockTable：逻辑 token 位置 -> (物理块号, 块内 slot)（论文概念身份——vLLM 源码类名刻意区分，红线⑭）；
#   ②BlockFreeList：物理块池（自有类名），按需分配/即用即还——v2 的 n_cap 买断浪费（请求短于上限时
#     预留段全闲）被消解，碎片被压缩到「每请求最后一块之内」；
#   ③碎片实测：同一变长 workload 下 v2 连续布局 vs v3 分页布局的 KV 峰值显存与分配账对照（章稿核心读数）。
# 执行层与调度逻辑承 v2（Stepper 复用 engine_v2 的实现——import 复用不复制）；本件只替换 KV 层。
# 所属章节：Book6 ch5 §5.7；设计书=plan/Book6-推理系统导论.md ch5
# 运行：cd <workspace> && source env.sh && ASCEND_RT_VISIBLE_DEVICES=<卡> \
#      python code/Book6-推理系统导论/ch05/engine_v3.py [--block-size 32 --smoke --out-name s1]
# 产物：log/book6-ch05/engine_v3_<out-name>.json（吞吐/KV 峰值/碎片账 v2v3 对照）
import argparse
import importlib.util
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code", "Book6-推理系统导论", "ch04"))
OUT_DIR = os.path.join(ROOT, "log", "book6-ch05")
CKPT = os.path.join(ROOT, "log", "book3-ch10", "ckpt_integrated_npu1.pt")
TOK = os.path.join(ROOT, "log", "book3-ch10", "tokens32k.bin")
SEED = 20261002


def load_mod(name, relpath):
    for c in [os.path.join(ROOT, relpath),
              os.path.join(ROOT, "from-models-to-vllm", "book6-inference-systems-intro", "code", relpath),
              os.path.join(ROOT, "from-models-to-vllm", "code", relpath)]:
        if os.path.isfile(c):
            spec = importlib.util.spec_from_file_location(name, c)
            m = importlib.util.module_from_spec(spec)
            sys.path.insert(0, os.path.dirname(c))
            spec.loader.exec_module(m)
            return m
    raise FileNotFoundError(relpath)


class BlockFreeList:
    """物理块池：块张量堆 (num_blocks, L, 2, h_kv, bs, hd) + 空闲块号栈。自有类名（红线⑭）。"""

    def __init__(self, cfg, num_blocks, bs, device, dtype):
        self.blocks = torch.zeros(num_blocks, cfg.num_hidden_layers, 2,
                                  cfg.num_key_value_heads, bs, cfg.head_dim,
                                  device=device, dtype=dtype)
        self.free = list(range(num_blocks))
        self.bs = bs
        self.num_blocks = num_blocks

    def alloc(self):
        return self.free.pop() if self.free else None

    def release(self, blk):
        self.free.append(blk)

    @property
    def used_bytes(self):
        return (self.num_blocks - len(self.free)) * self.blocks[0].numel() * self.blocks.element_size()


class BlockTable:
    """逻辑位置 -> 物理块号的映射表（论文数据结构身份；docstring 标地址流转）。"""

    def __init__(self):
        self.blocks = []          # 逻辑块序 -> 物理块号

    def append_block(self, phys):
        self.blocks.append(phys)

    def locate(self, pos):        # 逻辑 token 位置 -> (物理块号, slot)
        return self.blocks[pos // self_bs], pos % BS_PLACEHOLDER

    def gather_layer(self, pool, li, n):
        """收集前 n 个逻辑 token 在第 li 层的 K：按块拼 (h_kv, n, hd)。"""
        outs = []
        for lb in range((n + pool.bs - 1) // pool.bs):
            phys = self.blocks[lb]
            take = min(pool.bs, n - lb * pool.bs)
            outs.append(pool.blocks[phys, li, 0, :, :take])
        return torch.cat(outs, dim=1)

    def gather_layer_v(self, pool, li, n):
        outs = []
        for lb in range((n + pool.bs - 1) // pool.bs):
            phys = self.blocks[lb]
            take = min(pool.bs, n - lb * pool.bs)
            outs.append(pool.blocks[phys, li, 1, :, :take])
        return torch.cat(outs, dim=1)

    def write(self, pool, li, pos, k, v):        # 单 token 写入（k/v: (h_kv, hd)）
        lb, slot = pos // pool.bs, pos % pool.bs
        phys = self.blocks[lb]
        pool.blocks[phys, li, 0, :, slot] = k
        pool.blocks[phys, li, 1, :, slot] = v

    def ensure_capacity(self, pool, n):
        while len(self.blocks) * pool.bs < n:
            blk = pool.alloc()
            assert blk is not None, "物理块池耗尽——容量规划问题（教学件显式失败）"
            self.append_block(blk)

    def release_all(self, pool):
        for b in self.blocks:
            pool.release(b)
        self.blocks = []


class PagedStepper:
    """v3 执行层：复用 v2 Stepper 的投影/FFN/RoPE 逻辑，KV 读写换成块池接口。"""

    def __init__(self, v2stepper, pool):
        self.s = v2stepper
        self.pool = pool

    def prefill(self, ids, table):               # ids (1,n0) -> (V,)
        st, pool = self.s, self.pool
        n = ids.shape[1]
        x = st.m.embed_tokens(ids)
        cos, sin = st._rope_cache(n, st.hd, st.cfg.rope_theta, st.device)
        d2 = st.hd // 2
        cr, csi = cos[0].float()[:, :d2], sin[0].float()[:, d2:]
        rot = lambda t: torch.cat([t[..., :d2] * cr - t[..., d2:] * csi,
                                   t[..., :d2] * csi + t[..., d2:] * cr], dim=-1)
        mask = torch.tril(torch.ones(n, n, device=st.device, dtype=torch.bool))
        table.ensure_capacity(pool, n)
        for li, layer in enumerate(st.m.layers):
            a = layer.self_attn
            xn = layer.input_layernorm(x)
            q = a.q_proj(xn).view(n, st.h, st.hd).transpose(0, 1)
            k = a.k_proj(xn).view(n, st.h_kv, st.hd).transpose(0, 1)
            v = a.v_proj(xn).view(n, st.h_kv, st.hd).transpose(0, 1)
            q, k = rot(q).to(st.dtype), rot(k).to(st.dtype)
            for pos in range(n):                  # 逐 token 入块（教学版；生产版按块批写）
                table.write(pool, li, pos, k[:, pos], v[:, pos])
            kb = table.gather_layer(pool, li, n).repeat_interleave(st.n_rep, 0)
            vb = table.gather_layer_v(pool, li, n).repeat_interleave(st.n_rep, 0)
            att = (q @ kb.transpose(-2, -1)) * st.scale
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att.float(), dim=-1).to(st.dtype)
            o = (att @ vb).transpose(0, 1).reshape(1, n, st.h * st.hd)
            x = x + a.o_proj(o)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        table.n = n
        return st.model.lm_head(st.m.norm(x[:, -1:]))[0, 0]

    def step_batch(self, last_idx, tables):
        """批化 decode 一步：各请求各自块表读写；注意力 pad+mask 向量化（承 v2 语义）。"""
        st, pool = self.s, self.pool
        B = last_idx.shape[0]
        pos = [t.n for t in tables]
        n_max = max(pos) + 1
        cos_t, sin_t = st._rope_cache(n_max, st.hd, st.cfg.rope_theta, st.device)
        d2 = st.hd // 2
        cos_b = cos_t[0, :].float()[torch.tensor(pos, device=st.device)]
        sin_b = sin_t[0, :].float()[torch.tensor(pos, device=st.device)]
        cr, csi = cos_b[:, :d2], sin_b[:, d2:]

        def rot(t):                               # (B,heads,1,hd)
            tf = t.float()
            cr4 = cr.unsqueeze(1).unsqueeze(1)
            csi4 = csi.unsqueeze(1).unsqueeze(1)
            return torch.cat([tf[..., :d2] * cr4 - tf[..., d2:] * csi4,
                              tf[..., :d2] * csi4 + tf[..., d2:] * cr4], dim=-1).to(st.dtype)

        x = st.m.embed_tokens(last_idx)
        attn_mask = torch.zeros(B, 1, 1, n_max, dtype=torch.bool, device=st.device)
        for b, n in enumerate(pos):
            attn_mask[b, 0, 0, :n + 1] = True
        for li, layer in enumerate(st.m.layers):
            a = layer.self_attn
            xn = layer.input_layernorm(x)
            q = a.q_proj(xn).view(B, st.h, 1, st.hd)
            k = a.k_proj(xn).view(B, st.h_kv, 1, st.hd)
            v = a.v_proj(xn).view(B, st.h_kv, 1, st.hd)
            q, k = rot(q), rot(k)
            kb_full = torch.zeros(B, st.h_kv, n_max, st.hd, device=st.device, dtype=st.dtype)
            vb_full = torch.zeros_like(kb_full)
            for b, t in enumerate(tables):
                t.ensure_capacity(pool, t.n + 1)
                t.write(pool, li, t.n, k[b, :, 0], v[b, :, 0])
                kk = t.gather_layer(pool, li, t.n + 1)
                vv = t.gather_layer_v(pool, li, t.n + 1)
                kb_full[b, :, :kk.shape[1]] = kk
                vb_full[b, :, :vv.shape[1]] = vv
            kb = kb_full.repeat_interleave(st.n_rep, 1)
            vb = vb_full.repeat_interleave(st.n_rep, 1)
            att = (q @ kb.transpose(-2, -1)) * st.scale
            att = att.masked_fill(~attn_mask, float("-inf"))
            att = F.softmax(att.float(), dim=-1).to(st.dtype)
            o = (att @ vb).transpose(1, 2).reshape(B, 1, st.h * st.hd)
            x = x + a.o_proj(o)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        for t in tables:
            t.n += 1
        return st.model.lm_head(st.m.norm(x))[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser(description="v3 分页 KV 引擎（BlockTable+BlockFreeList）")
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--p-len", type=int, default=128)
    ap.add_argument("--max-span", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-name", default=None)
    a = ap.parse_args()
    name = a.out_name or ("smoke" if a.smoke else "base")
    os.makedirs(OUT_DIR, exist_ok=True)
    if a.smoke:
        a.max_batch, a.p_len, a.max_span, a.block_size = 4, 64, 16, 16

    ev2 = load_mod("engine_v2", os.path.join("code", "Book6-推理系统导论", "ch04", "engine_v2.py"))
    l215 = ev2.load_llama215()
    device = "npu" if torch.npu.is_available() else "cpu"
    torch.manual_seed(SEED)
    model = l215.Llama215()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(device=device, dtype=torch.bfloat16)
    st = ev2.Stepper(model)
    cfg = model.cfg

    # 数值对拍：v3 分页路径 vs v2 连续路径 vs 全量重算（CPU fp32，一次）
    model_c = l215.Llama215(); model_c.load_state_dict(ck["model"], strict=True); model_c.eval()
    stc = ev2.Stepper(model_c)
    ids = torch.randint(0, 32000, (1, 20))
    pool = BlockFreeList(cfg, 64, 8, "cpu", torch.float32)
    tb = BlockTable()
    ps = PagedStepper(stc, pool)
    lg = ps.prefill(ids, tb)
    ref = model_c(ids)[0][0, -1]
    d = (lg - ref).abs().max().item()
    print(f"[对拍] v3 分页 prefill vs 全量重算 max|Δ|={d:.2e}（CPU fp32）")
    assert d < 1e-4

    # workload（承 v2 同款——可比性）
    import numpy as np
    stream = np.fromfile(TOK, dtype=np.uint16, count=a.p_len + 8192)
    prompts = [torch.tensor(stream[i * 8: i * 8 + a.p_len].tolist()) for i in range(a.max_batch)]
    g = torch.Generator().manual_seed(SEED)
    hi, lo = (16, 4) if a.smoke else (a.max_span, a.max_span // 4)
    max_new = torch.randint(lo, hi + 1, (a.max_batch,), generator=g).tolist()

    pool = BlockFreeList(cfg, 4096, a.block_size, device, torch.bfloat16)
    pst = PagedStepper(st, pool)
    pending = list(range(a.max_batch))
    active, finished = {}, {}
    t0 = time.perf_counter()
    if device == "npu":
        torch.npu.reset_peak_memory_stats()

    while pending or active:
        while pending and len(active) < a.max_batch:
            batch_in = []
            while pending and len(active) + len(batch_in) < a.max_batch:
                batch_in.append(pending.pop(0))
            tables = [BlockTable() for _ in batch_in]
            with torch.no_grad():
                logits = pst.prefill_batch if False else None
                outs = []
                for i, rid in enumerate(batch_in):      # prefill 逐条（分页写入教学版）
                    outs.append(pst.prefill(prompts[rid].unsqueeze(0).to(device), tables[i]))
            for i, rid in enumerate(batch_in):
                active[rid] = dict(table=tables[i], last=outs[i], produced=0, t_pref=time.perf_counter())
        rids = list(active.keys())
        if not rids:
            continue
        with torch.no_grad():
            nxt = torch.multinomial(
                torch.softmax(torch.stack([active[r]["last"] for r in rids]).float(), -1), 1)
            logits = pst.step_batch(nxt.to(device), [active[r]["table"] for r in rids])
        for i, rid in enumerate(rids):
            s = active[rid]
            s["produced"] += 1
            s["last"] = logits[i]
            if s["produced"] >= max_new[rid]:
                finished[rid] = dict(produced=s["produced"], n_final=s["table"].n,
                                     blocks=len(s["table"].blocks))
                s["table"].release_all(pool)             # 请求退出——物理块即刻归还
                del active[rid]
    wall = time.perf_counter() - t0
    peak = torch.npu.max_memory_allocated() / 2**20 if device == "npu" else 0

    useful = sum(f["produced"] for f in finished.values())
    # 碎片账：v2 连续布局的理论预留（每请求 n_cap 买断） vs v3 实际占用（块数和）
    n_cap = a.p_len + max(max_new) + 8
    v2_kv = a.max_batch * cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * n_cap * cfg.head_dim * 2
    v3_blocks_used_peak = max(f["blocks"] for f in finished.values()) * len(finished)  # 粗估峰值
    v3_kv = sum((a.p_len + f["produced"] + a.block_size - 1) // a.block_size * a.block_size
                for f in finished.values()) * cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * cfg.head_dim * 2
    out = {
        "engine": "v3_paged", "device": device, "dtype": "bfloat16", "block_size": a.block_size,
        "workload": {"batch": a.max_batch, "p_len": a.p_len, "max_new": max_new},
        "wall_s": round(wall, 3), "throughput_tok_s": round(useful / wall, 2),
        "useful_tokens": useful, "wasted_slots": 0,
        "peak_mem_MiB": round(peak, 1),
        "kv_account_v2_contiguous_bytes": v2_kv,
        "kv_account_v3_paged_bytes": v3_kv,
        "kv_saving_pct": round((1 - v3_kv / v2_kv) * 100, 1),
        "fragment_note": f"v3 碎片=每请求最后一块内的空 slot（块大小 {a.block_size} 时最大 {a.block_size-1} token/请求）",
    }
    p = os.path.join(OUT_DIR, f"engine_v3_{name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[v3] {a.max_batch} 条 | wall {wall:.2f}s | 吞吐 {out['throughput_tok_s']} tok/s"
          f" | KV 账 v2 {v2_kv/2**20:.0f} MiB -> v3 {v3_kv/2**20:.0f} MiB（省 {out['kv_saving_pct']}%）")
    print(f"[产物] {p}")


if __name__ == "__main__":
    main()
