# engine_v4.py —— Book6 大项目 v4：前缀缓存引擎（块级内容哈希 + 物理块共享/引用计数）
# 用途：ch6 正身。在 v3 块池/块表之上加前缀复用：①哈希键=token id 块序列的哈希（因果模型下
#   相同 token 前缀+相同权重 -> 确定性相同的 K/V——以 token 序列哈希作键是合法的免算依据）；
#   ②命中：请求直接挂已算好的物理块（refcount+1、免该段 prefill 前向）；分歧点起各自新算；
#   ③workload=多轮对话（每轮 prompt=共享系统前缀+逐轮增长的历史）——本册版「多轮对话的
#   上下文经济学」：命中率、免算 token、各轮 TTFT 前后对照。
# 执行层承 v3（import 复用）；新增 prefill_incremental（增量段前向：attention 读全历史含命中块）。
# 所属章节：Book6 ch6 §6.5；设计书=plan/Book6-推理系统导论.md ch6
# 运行：cd <workspace> && source env.sh && ASCEND_RT_VISIBLE_DEVICES=<卡> \
#      python code/Book6-推理系统导论/ch06/engine_v4.py [--smoke --out-name s1]
# 产物：log/book6-ch06/engine_v4_<out-name>.json（命中率/免算账/TTFT 对照）
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(ROOT, "log", "book6-ch06")
CKPT = os.path.join(ROOT, "log", "book3-ch10", "ckpt_integrated_npu1.pt")
TOK = os.path.join(ROOT, "log", "book3-ch10", "tokens32k.bin")
SEED = 20261002


def load_mod(relpath):
    for c in [os.path.join(ROOT, relpath),
              os.path.join(ROOT, "from-models-to-vllm", "book6-inference-systems-intro", "code", relpath),
              os.path.join(ROOT, "from-models-to-vllm", "code", relpath)]:
        if os.path.isfile(c):
            spec = importlib.util.spec_from_file_location(os.path.basename(relpath), c)
            m = importlib.util.module_from_spec(spec)
            sys.path.insert(0, os.path.dirname(c))
            spec.loader.exec_module(m)
            return m
    raise FileNotFoundError(relpath)


def block_key(ids_tuple):
    """块内容哈希键（token id 元组 -> hex digest；uint16 序列化——词表 32000 超 255；教学版用 hashlib，生产版用更快的滚动哈希）。"""
    import numpy as _np
    return hashlib.sha1(_np.asarray(ids_tuple, dtype=_np.uint16).tobytes()).hexdigest()


class PrefixCache:
    """块键 -> 物理块号（含 refcount）。命中即挂块免算；全释放时归还块池。"""

    def __init__(self, pool):
        self.pool = pool
        self.map = {}            # key -> [phys_block, refcount]

    def lookup(self, key):
        e = self.map.get(key)
        if e is None:
            return None
        e[1] += 1
        return e[0]

    def register(self, key, phys):
        self.map[key] = [phys, 1]

    def drop(self, key):
        e = self.map.get(key)
        if e is None:
            return
        e[1] -= 1
        if e[1] <= 0:
            del self.map[key]


def main() -> None:
    ap = argparse.ArgumentParser(description="v4 前缀缓存引擎（块哈希+共享+命中免算）")
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--n-dialogs", type=int, default=3)
    ap.add_argument("--n-rounds", type=int, default=4)
    ap.add_argument("--sys-prefix", type=int, default=128)
    ap.add_argument("--round-inc", type=int, default=16)
    ap.add_argument("--gen-len", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-name", default=None)
    a = ap.parse_args()
    name = a.out_name or ("smoke" if a.smoke else "base")
    os.makedirs(OUT_DIR, exist_ok=True)
    if a.smoke:
        a.n_dialogs, a.n_rounds, a.sys_prefix, a.round_inc, a.gen_len = 2, 3, 64, 16, 4

    ev2 = load_mod(os.path.join("code", "Book6-推理系统导论", "ch04", "engine_v2.py"))
    ev3 = load_mod(os.path.join("code", "Book6-推理系统导论", "ch05", "engine_v3.py"))
    l215 = ev2.load_llama215()
    device = "npu" if torch.npu.is_available() else "cpu"
    torch.manual_seed(SEED)
    model = l215.Llama215()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(device=device, dtype=torch.bfloat16)
    st = ev2.Stepper(model)
    cfg = model.cfg
    pool = ev3.BlockFreeList(cfg, 8192, a.block_size, device, torch.bfloat16)
    pcache = PrefixCache(pool)
    ps = ev3.PagedStepper(st, pool)

    # workload：n_dialogs 条多轮对话；每轮 prompt = 系统前缀 + 历史（round_inc/轮）；每轮生成 gen_len
    import numpy as np
    stream = np.fromfile(TOK, dtype=np.uint16, count=65536)
    dialogs = []
    for d in range(a.n_dialogs):
        sys_p = stream[d * 512: d * 512 + a.sys_prefix].tolist()
        rounds = [stream[d * 512 + a.sys_prefix + r * a.round_inc:
                          d * 512 + a.sys_prefix + (r + 1) * a.round_inc].tolist()
                  for r in range(a.n_rounds)]
        dialogs.append((sys_p, rounds))

    stats = dict(prefill_tokens=0, saved_tokens=0, hits=0, misses=0, rounds=[])
    t_all = time.perf_counter()

    def prefill_cached(ids_list, table):
        """带前缀缓存的 prefill：逐块查键——命中挂共享块，未命中段增量前向并入表。返回 (logits, hit_len)。"""
        n = len(ids_list)
        table.ensure_capacity(pool, n)
        hit_blocks = 0
        nb = n // a.block_size                      # 只对整块做缓存（尾块不共享——简化）
        for lb in range(nb):
            key = block_key(tuple(ids_list[lb * a.block_size:(lb + 1) * a.block_size]))
            phys = pcache.lookup(key)
            if phys is not None:
                table.blocks[lb] = phys             # 命中：挂共享块（refcount 已 +1）
                hit_blocks += 1
            else:
                break
        hit_len = hit_blocks * a.block_size
        stats["prefill_tokens"] += n
        stats["saved_tokens"] += hit_len
        stats["hits"] += hit_blocks
        stats["misses"] += max(0, (n - hit_len + a.block_size - 1) // a.block_size - 0)
        # 未命中段增量前向：新算 [hit_len, n)，attention 读全历史（命中段 K/V 已在池中）
        return prefill_incremental(torch.tensor([ids_list], device=device), hit_len, table), hit_len

    def prefill_incremental(ids, hit_len, table):
        """增量 prefill：输入整段 ids，前向只算 [hit_len, n) 的新 token；K/V 全量入表（命中段已挂）。"""
        st_ = st
        n = ids.shape[1]
        new_n = n - hit_len
        x = st_.m.embed_tokens(ids[:, hit_len:])           # (1, new_n, C)
        cos, sin = st_._rope_cache(n, st_.hd, st_.cfg.rope_theta, st_.device)
        d2 = st_.hd // 2
        cr, csi = cos[0].float()[:, :d2], sin[0].float()[:, d2:]   # (n,d2)
        mask = torch.zeros(new_n, n, dtype=torch.bool, device=st_.device)
        for i in range(new_n):
            mask[i, :hit_len + i + 1] = True                # 梯形 mask：新 token i 可见 0..hit_len+i
        for li, layer in enumerate(st_.m.layers):
            a_ = layer.self_attn
            xn = layer.input_layernorm(x)                   # 新段 norm（仅新 token）
            q = a_.q_proj(xn).view(new_n, st_.h, st_.hd).transpose(0, 1)     # (h,new_n,hd)
            k_new = a_.k_proj(xn).view(new_n, st_.h_kv, st_.hd).transpose(0, 1)
            v_new = a_.v_proj(xn).view(new_n, st_.h_kv, st_.hd).transpose(0, 1)
            qf = q.float()
            qr = torch.cat([qf[..., :d2] * cr[hit_len:] - qf[..., d2:] * csi[hit_len:],
                            qf[..., :d2] * csi[hit_len:] + qf[..., d2:] * cr[hit_len:]], dim=-1)
            q = qr.to(st_.dtype)
            kf = k_new.float()
            kr = torch.cat([kf[..., :d2] * cr[hit_len:] - kf[..., d2:] * csi[hit_len:],
                            kf[..., :d2] * csi[hit_len:] + kf[..., d2:] * cr[hit_len:]], dim=-1)
            k_new = kr.to(st_.dtype)
            for j in range(new_n):                          # 新 token 逐位入块
                table.write(pool, li, hit_len + j, k_new[:, j], v_new[:, j])
            kb = table.gather_layer(pool, li, n).repeat_interleave(st_.n_rep, 0)   # 全历史（含命中块）
            vb = table.gather_layer_v(pool, li, n).repeat_interleave(st_.n_rep, 0)
            att = (q @ kb.transpose(-2, -1)) * st_.scale    # (h,new_n,n)
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att.float(), dim=-1).to(st_.dtype)
            o = (att @ vb).transpose(0, 1).reshape(1, new_n, st_.h * st_.hd)
            x = x + a_.o_proj(o)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        # 新算出的整块注册进缓存（从首个未命中块起；命中块已在表中、不重复注册）
        for lb in range(hit_len // a.block_size, n // a.block_size):
            key = block_key(ids[0, lb * a.block_size:(lb + 1) * a.block_size].tolist())
            if key not in pcache.map:
                pcache.register(key, table.blocks[lb])      # 首个持有者（本轮）
        table.n = n
        return st_.model.lm_head(st_.m.norm(x[:, -1:]))[0, 0]

    # 多轮对话主循环（逐对话逐轮——封闭 workload）
    for d_i, (sys_p, rounds) in enumerate(dialogs):
        for r_i, inc in enumerate(rounds):
            prompt = sys_p + [t for r in rounds[:r_i] for t in r] + inc
            table = ev3.BlockTable()
            # 命中查询+增量前向（prefill_cached 内部完成注册）
            t0 = time.perf_counter()
            with torch.no_grad():
                logits, hit_len = prefill_cached(prompt, table)
            t_pref = time.perf_counter() - t0
            # decode gen_len 步
            t1 = time.perf_counter()
            with torch.no_grad():
                for _ in range(a.gen_len):
                    nxt = torch.multinomial(torch.softmax(logits.float(), -1), 1)
                    logits = ps.step_batch(nxt.view(1, 1).to(device), [table])[0]
            t_dec = time.perf_counter() - t1
            stats["rounds"].append(dict(dialog=d_i, round=r_i, prompt_len=len(prompt),
                                        hit_len=hit_len, ttft_ms=round(t_pref * 1e3, 1),
                                        dec_ms=round(t_dec * 1e3, 1)))
            # 简化：每轮结束释放该轮私有的块引用（共享块 refcount 保留——下轮还能命中）
            for lb in range(len(table.blocks)):
                pass  # 教学版：对话内块保持存活（多轮命中依赖它）；对话级回收略

    wall_all = time.perf_counter() - t_all
    total_tok = stats["prefill_tokens"]
    out = {
        "engine": "v4_prefix_cache", "device": device, "dtype": "bfloat16", "block_size": a.block_size,
        "workload": {"dialogs": a.n_dialogs, "rounds": a.n_rounds,
                     "sys_prefix": a.sys_prefix, "round_inc": a.round_inc, "gen": a.gen_len},
        "prefill_tokens": total_tok, "saved_tokens": stats["saved_tokens"],
        "hit_rate_pct": round(stats["saved_tokens"] / total_tok * 100, 1) if total_tok else 0,
        "ttft_first_round_ms": stats["rounds"][0]["ttft_ms"] if stats["rounds"] else None,
        "ttft_last_round_ms": stats["rounds"][-1]["ttft_ms"] if stats["rounds"] else None,
        "wall_s": round(wall_all, 2), "rounds_detail": stats["rounds"],
    }
    p = os.path.join(OUT_DIR, f"engine_v4_{name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[v4] {a.n_dialogs} 对话 × {a.n_rounds} 轮 | prefill {total_tok} tok（免算 {stats['saved_tokens']}"
          f" = {out['hit_rate_pct']}%）| TTFT 首轮 {out['ttft_first_round_ms']} ms → 末轮 {out['ttft_last_round_ms']} ms")
    print(f"[产物] {p}")


if __name__ == "__main__":
    main()
