# engine_v2.py —— Book6 大项目 v2：连续批处理引擎（iteration 级调度 + KV cache）
# 用途：ch4 §4.5-4.6 正身。对 v1 三宗罪的偿还：①iteration 级调度（每步重组活跃批——完成即出、
#   腾位即入，浪费位归零）；②KV cache（手写带缓存的步进前向——prefill 全段一次入缓存、
#   decode 每步只算新 token，O(n²) 账清偿）；③selective batching 教学实现：投影/FFN 段真拼批
#   (B,1,C)；注意力段 pad+mask 向量化异长共存（各请求 K/V pad 到 n_max、softmax 掩掉 pad 位）
#   ——Orca 正身（attention 按请求各自算）的合法向量化降档，头注声明版本契约。
#   KV 存放=每请求一段连续预分配 buffer (L,2,h_kv,n_cap,hd)——v3 分页改造的对象。
# 执行层复用 llama215 全部权重组件（HF 键位 q/k/v/o_proj、RMSNorm、SwiGLUMLP），
#   RoPE 单位置现算（Book3 ch04 build_rope_cache）——「改装史写在 import 语句里」的引擎版。
# 数值口径：与 llama215.forward 全量重算对拍（CPU fp32 allclose——见文件尾 self_test）。
# 所属章节：Book6 ch4 §4.5-4.6；设计书=plan/Book6-推理系统导论.md ch4
# 运行：cd <workspace> && source env.sh && ASCEND_RT_VISIBLE_DEVICES=<卡> \
#      python code/Book6-推理系统导论/ch04/engine_v2.py [--max-batch 8 --smoke --out-name s1]
# 产物：log/book6-ch04/engine_v2_<out-name>.json（吞吐/TPOT/显存/KV 账）
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
OUT_DIR = os.path.join(ROOT, "log", "book6-ch04")
CKPT = os.path.join(ROOT, "log", "book3-ch10", "ckpt_integrated_npu1.pt")
TOK = os.path.join(ROOT, "log", "book3-ch10", "tokens32k.bin")
_L215_CANDS = [
    os.path.join(ROOT, "code", "Book3-现代开源骨架", "ch10", "llama215.py"),
    os.path.join(ROOT, "from-models-to-vllm", "book3-modern-open-source-skeleton", "code", "ch10", "llama215.py"),
    os.path.join(ROOT, "from-models-to-vllm", "code", "ch10", "llama215.py"),
]
SEED = 20261002


def load_llama215():
    for c in _L215_CANDS:
        if os.path.isfile(c):
            sys.path.insert(0, os.path.dirname(c))
            spec = importlib.util.spec_from_file_location("llama215", c)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("llama215.py 三候选均未命中")


def _rot_single(q, k, cos, sin, hd, dtype):
    """单位置 RoPE：q/k (h,1,hd)；cos/sin (hd,)——与 Book3 apply_rope 数值一致。"""
    d2 = hd // 2
    cr, csi = cos[:d2], sin[:d2]
    f = lambda t: torch.cat([t[..., :d2] * cr - t[..., d2:] * csi,
                             t[..., :d2] * csi + t[..., d2:] * cr], dim=-1)
    return f(q.float()).to(dtype), f(k.float()).to(dtype)


class KVCache:
    """每请求一份连续 KV buffer：(L, 2, h_kv, n_cap, hd)。v2=整段预分配（v3 分页改造对象）。"""

    def __init__(self, cfg, n_cap, device, dtype):
        self.buf = torch.zeros(cfg.num_hidden_layers, 2, cfg.num_key_value_heads,
                               n_cap, cfg.head_dim, device=device, dtype=dtype)
        self.n = 0
        self.n_cap = n_cap
        self.bytes = self.buf.numel() * self.buf.element_size()

    def append(self, li, k, v):                 # k/v: (h_kv,1,hd) 写入第 li 层
        self.buf[li, 0, :, self.n] = k[:, 0]
        self.buf[li, 1, :, self.n] = v[:, 0]

    def k(self, li):                            # (h_kv,n,hd)
        return self.buf[li, 0, :, :self.n + 1]

    def v(self, li):
        return self.buf[li, 1, :, :self.n + 1]


class Stepper:
    """带 KV cache 的步进执行层（单请求粒度；拼批语义=权重共享的逐请求循环——见文件头注）。"""

    def __init__(self, model):
        self.model = model
        self.m = model.model
        cfg = model.cfg
        self.cfg = cfg
        self.hd, self.h, self.h_kv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
        self.n_rep = self.h // self.h_kv
        self.scale = self.hd ** -0.5
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        from rope import build_rope_cache       # Book3 ch04 插槽（llama215 已带 sys.path）
        self._rope_cache = build_rope_cache
        self._cos_sin = {}

    def _cs(self, pos):                          # (hd,) 惰性缓存
        if pos not in self._cos_sin:
            c, s = self._rope_cache(pos + 1, self.hd, self.cfg.rope_theta, self.device)
            self._cos_sin[pos] = (c[0, pos].float(), s[0, pos].float())
        return self._cos_sin[pos]

    def prefill(self, ids, cache):               # ids (1,n0) -> logits (V)；KV 全量入缓存
        n = ids.shape[1]
        x = self.m.embed_tokens(ids)             # (1,n,C)
        cos, sin = self._rope_cache(n, self.hd, self.cfg.rope_theta, self.device)
        d2 = self.hd // 2
        cr, csi = cos[0].float()[:, :d2], sin[0].float()[:, d2:]   # (n,d2)
        rot = lambda t: torch.cat([t[..., :d2] * cr - t[..., d2:] * csi,
                                   t[..., :d2] * csi + t[..., d2:] * cr], dim=-1)
        mask = torch.tril(torch.ones(n, n, device=self.device, dtype=torch.bool))
        for li, layer in enumerate(self.m.layers):
            a = layer.self_attn
            xn = layer.input_layernorm(x)                                  # (1,n,C)
            q = a.q_proj(xn).view(n, self.h, self.hd).transpose(0, 1)      # (h,n,hd)
            k = a.k_proj(xn).view(n, self.h_kv, self.hd).transpose(0, 1)   # (h_kv,n,hd)
            v = a.v_proj(xn).view(n, self.h_kv, self.hd).transpose(0, 1)   # (h_kv,n,hd)
            q, k = rot(q).to(self.dtype), rot(k).to(self.dtype)
            cache.buf[li, 0, :, :n], cache.buf[li, 1, :, :n] = k, v
            kb = k.repeat_interleave(self.n_rep, 0)                        # (h,n,hd)
            vb = v.repeat_interleave(self.n_rep, 0)
            att = (q @ kb.transpose(-2, -1)) * self.scale                  # (h,n,n)
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att.float(), dim=-1).to(self.dtype)
            o = (att @ vb).transpose(0, 1).reshape(1, n, self.h * self.hd)  # (1,n,h*hd)
            x = x + a.o_proj(o)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        cache.n = n
        return self.model.lm_head(self.m.norm(x[:, -1:]))[0, 0]           # (V,)

    def step_batch(self, last_idx, caches):
        """批化 decode 一步（selective batching 真身）：投影/FFN 段真拼批；注意力段 pad+mask
        向量化处理异长 cache（各请求位置=各自 cache.n）。last_idx (B,1) -> logits (B,V)。"""
        B = last_idx.shape[0]
        pos = torch.tensor([c.n for c in caches], device=self.device)       # (B,) 各请求位置
        n_now = [c.n for c in caches]
        n_max = max(n_now) + 1
        cos_t, sin_t = self._rope_cache(n_max, self.hd, self.cfg.rope_theta, self.device)
        d2 = self.hd // 2
        # 各请求按自己位置的 cos/sin：(B,hd)
        cos_b = cos_t[0, :].float()[pos]                                    # (B,hd)
        sin_b = sin_t[0, :].float()[pos]
        cr, csi = cos_b[:, :d2], sin_b[:, d2:]                              # (B,d2)

        def rot_batch(t):                                                   # t: (B, heads, 1, hd)
            tf = t.float()
            cr4, csi4 = cr.unsqueeze(1).unsqueeze(1), csi.unsqueeze(1).unsqueeze(1)   # (B,1,1,d2)
            r = tf[..., :d2] * cr4 - tf[..., d2:] * csi4
            i = tf[..., :d2] * csi4 + tf[..., d2:] * cr4
            return torch.cat([r, i], dim=-1).to(self.dtype)

        x = self.m.embed_tokens(last_idx)                                   # (B,1,C)
        attn_mask = torch.zeros(B, 1, 1, n_max, dtype=torch.bool, device=self.device)
        for b, n in enumerate(n_now):
            attn_mask[b, 0, 0, :n + 1] = True                               # 各自真长度内可见
        for li, layer in enumerate(self.m.layers):
            a = layer.self_attn
            xn = layer.input_layernorm(x)                                   # (B,1,C)
            q = a.q_proj(xn).view(B, self.h, 1, self.hd)                    # (B,h,1,hd)
            k = a.k_proj(xn).view(B, self.h_kv, 1, self.hd)                 # (B,h_kv,1,hd)
            v = a.v_proj(xn).view(B, self.h_kv, 1, self.hd)                 # (B,h_kv,1,hd)
            q = rot_batch(q)
            k = rot_batch(k)
            kb_list, vb_list = [], []
            for b in range(B):                                              # append 各 cache + 收集
                caches[b].append(li, k[b], v[b])
                kb_list.append(caches[b].k(li))
                vb_list.append(caches[b].v(li))
            # pad 到 n_max：(B,h_kv,n_max,hd)
            kb = torch.zeros(B, self.h_kv, n_max, self.hd, device=self.device, dtype=self.dtype)
            vb = torch.zeros_like(kb)
            for b, (kk, vv) in enumerate(zip(kb_list, vb_list)):
                kb[b, :, :kk.shape[1]] = kk
                vb[b, :, :vv.shape[1]] = vv
            kb = kb.repeat_interleave(self.n_rep, 1)                        # (B,h,n_max,hd)
            vb = vb.repeat_interleave(self.n_rep, 1)
            att = (q @ kb.transpose(-2, -1)) * self.scale                   # (B,h,1,n_max)
            att = att.masked_fill(~attn_mask, float("-inf"))
            att = F.softmax(att.float(), dim=-1).to(self.dtype)
            o = (att @ vb)                                                  # (B,h,1,hd)
            o = o.transpose(1, 2).reshape(B, 1, self.h * self.hd)           # (B,1,h*hd)
            x = x + a.o_proj(o)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        for b in range(B):
            caches[b].n += 1
        return self.model.lm_head(self.m.norm(x))[:, 0]                      # (B,V)

    def prefill_batch(self, ids, caches):        # ids (B,n0) 等长 -> logits (B,V)；KV 全量入各缓存
        B, n = ids.shape
        x = self.m.embed_tokens(ids)                                        # (B,n,C)
        cos, sin = self._rope_cache(n, self.hd, self.cfg.rope_theta, self.device)
        d2 = self.hd // 2
        cr, csi = cos[0].float()[:, :d2], sin[0].float()[:, d2:]             # (n,d2)
        rot = lambda t: torch.cat([t[..., :d2] * cr - t[..., d2:] * csi,
                                   t[..., :d2] * csi + t[..., d2:] * cr], dim=-1)
        mask = torch.tril(torch.ones(n, n, device=self.device, dtype=torch.bool))
        for li, layer in enumerate(self.m.layers):
            a = layer.self_attn
            xn = layer.input_layernorm(x)                                   # (B,n,C)
            q = a.q_proj(xn).view(B, n, self.h, self.hd).permute(0, 2, 1, 3)   # (B,h,n,hd)
            k = a.k_proj(xn).view(B, n, self.h_kv, self.hd).permute(0, 2, 1, 3)
            v = a.v_proj(xn).view(B, n, self.h_kv, self.hd).permute(0, 2, 1, 3)
            q, k = rot(q).to(self.dtype), rot(k).to(self.dtype)
            for b in range(B):
                caches[b].buf[li, 0, :, :n], caches[b].buf[li, 1, :, :n] = k[b], v[b]
                caches[b].n = n
            kb = k.repeat_interleave(self.n_rep, 1)                         # (B,h,n,hd)
            vb = v.repeat_interleave(self.n_rep, 1)
            att = (q @ kb.transpose(-2, -1)) * self.scale                   # (B,h,n,n)
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att.float(), dim=-1).to(self.dtype)
            o = (att @ vb).permute(0, 2, 1, 3).reshape(B, n, self.h * self.hd)
            x = x + a.o_proj(o)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.model.lm_head(self.m.norm(x[:, -1:]))[:, 0]             # (B,V)


def self_test():
    """数值对拍：prefill+2 步 step 的末位 logits vs llama215.forward 全量重算（CPU fp32）。"""
    l215 = load_llama215()
    torch.manual_seed(SEED)
    model = l215.Llama215()
    model.eval()                                       # CPU fp32
    st = Stepper(model)
    ids = torch.randint(0, 32000, (1, 12))
    cache = KVCache(model.cfg, 32, "cpu", torch.float32)
    lg0 = st.prefill(ids, cache)
    ref0 = model(ids)[0][0, -1]
    d0 = (lg0 - ref0).abs().max().item()
    seq = [torch.tensor([123]), torch.tensor([4567])]
    c1 = KVCache(model.cfg, 32, "cpu", torch.float32); c1.buf.copy_(cache.buf[:, :, :12]); c1.n = 12
    lg1 = st.step_batch(seq[0].view(1, 1), [c1])
    ref1 = model(torch.cat([ids, seq[0].view(1, 1)], 1))[0][0, -1]
    d1 = (lg1[0] - ref1).abs().max().item()
    lg2 = st.step_batch(seq[1].view(1, 1), [c1])
    ref2 = model(torch.cat([ids, seq[0].view(1, 1), seq[1].view(1, 1)], 1))[0][0, -1]
    d2 = (lg2[0] - ref2).abs().max().item()
    # 异长批对拍：cache 长度 12（prefill 态）与 13（已含 seq[0]）各喂下一步 token，两路都到 14 长态
    c_a = KVCache(model.cfg, 32, "cpu", torch.float32); c_a.buf.copy_(cache.buf[:, :, :12]); c_a.n = 12
    c_b = KVCache(model.cfg, 32, "cpu", torch.float32); c_b.buf.copy_(c1.buf); c_b.n = 13
    lb = st.step_batch(torch.tensor([[seq[0][0]], [seq[1][0]]]), [c_a, c_b])
    d3 = max((lb[0] - ref1).abs().max().item(), (lb[1] - ref2).abs().max().item())
    print(f"[self_test] 异长批(12,13) 双路 max|Δ|={d3:.2e}（lb[0] 对拍 ref1、lb[1] 对拍 ref2）")
    print(f"[self_test] prefill max|Δ|={d0:.2e} step1 {d1:.2e} step2 {d2:.2e}（CPU fp32，应 <1e-4）")
    assert max(d0, d1, d2, d3) < 1e-4, "步进前向与全量重算不一致"


def main() -> None:
    ap = argparse.ArgumentParser(description="v2 连续批处理引擎（iteration 级调度+KV cache）")
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--p-len", type=int, default=128)
    ap.add_argument("--max-span", type=int, default=64)
    ap.add_argument("--n-cap", type=int, default=256)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-name", default=None)
    a = ap.parse_args()
    name = a.out_name or ("smoke" if a.smoke else "base")
    os.makedirs(OUT_DIR, exist_ok=True)
    if a.smoke:
        a.max_batch, a.p_len, a.max_span, a.n_cap = 4, 64, 16, 96

    l215 = load_llama215()
    device = "npu" if torch.npu.is_available() else "cpu"
    torch.manual_seed(SEED)
    model = l215.Llama215()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(device=device, dtype=torch.bfloat16)
    st = Stepper(model)

    import numpy as np
    stream = np.fromfile(TOK, dtype=np.uint16, count=a.p_len + 8192)
    prompts = [torch.tensor(stream[i * 8: i * 8 + a.p_len].tolist()) for i in range(a.max_batch)]
    g = torch.Generator().manual_seed(SEED)
    hi, lo = (16, 4) if a.smoke else (a.max_span, a.max_span // 4)
    max_new = torch.randint(lo, hi + 1, (a.max_batch,), generator=g).tolist()

    pending = list(range(a.max_batch))
    active, finished = {}, {}
    t0 = time.perf_counter()
    if device == "npu":
        torch.npu.reset_peak_memory_stats()

    while pending or active:
        while pending and len(active) < a.max_batch:          # 腾位即入（iteration 级；首拍整批入）
            batch_in = []
            while pending and len(active) + len(batch_in) < a.max_batch:
                batch_in.append(pending.pop(0))
            caches = [KVCache(model.cfg, a.n_cap, device, torch.bfloat16) for _ in batch_in]
            with torch.no_grad():
                logits = st.prefill_batch(
                    torch.stack([prompts[r] for r in batch_in]).to(device), caches)
            for i, rid in enumerate(batch_in):
                active[rid] = dict(cache=caches[i], last=logits[i], produced=0,
                                   t_pref=time.perf_counter())
        rids = list(active.keys())
        if not rids:
            continue
        with torch.no_grad():                                  # 批化一步 decode（真拼批+异长 mask）
            nxt = torch.multinomial(
                torch.softmax(torch.stack([active[r]["last"] for r in rids]).float(), -1), 1)
            logits = st.step_batch(nxt.to(device), [active[r]["cache"] for r in rids])
        for i, rid in enumerate(rids):
            s = active[rid]
            s["produced"] += 1
            s["last"] = logits[i]
            if s["produced"] >= max_new[rid]:                   # 完成即出（iteration 级）
                finished[rid] = dict(produced=s["produced"],
                                     t_total=time.perf_counter() - s["t_pref"])
                del active[rid]
    wall = time.perf_counter() - t0
    peak = torch.npu.max_memory_allocated() / 2**20 if device == "npu" else 0

    useful = sum(f["produced"] for f in finished.values())
    kv_bytes = a.max_batch * KVCache(model.cfg, a.n_cap, device, torch.bfloat16).bytes
    out = {
        "engine": "v2_continuous", "device": device, "dtype": "bfloat16",
        "workload": {"batch": a.max_batch, "p_len": a.p_len, "max_new": max_new, "n_cap": a.n_cap},
        "wall_s": round(wall, 3), "throughput_tok_s": round(useful / wall, 2),
        "useful_tokens": useful, "wasted_slots": 0,
        "per_req_tpot_ms_avg": round(sum(f["t_total"] / f["produced"] for f in finished.values())
                                     / a.max_batch * 1e3, 3),
        "peak_mem_MiB": round(peak, 1), "kv_bytes_total": kv_bytes,
        "kv": f"per-request 连续 buffer (L,2,h_kv,{a.n_cap},hd) 共 {kv_bytes/2**20:.0f} MiB",
        "repayments": ["iteration 级调度（浪费位=0）", "KV cache（O(n²) 清偿）", "完成即出/腾位即入"],
    }
    p = os.path.join(OUT_DIR, f"engine_v2_{name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[v2] {a.max_batch} 条 | wall {wall:.2f}s | 吞吐 {out['throughput_tok_s']} tok/s"
          f" | TPOT 均 {out['per_req_tpot_ms_avg']} ms | 峰值 {out['peak_mem_MiB']} MiB | 浪费位 0")
    print(f"[产物] {p}")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
