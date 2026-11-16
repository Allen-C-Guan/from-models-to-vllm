# needle_probe.py —— Book5 第2章：位置可控的合成检索，两臂共享初始公共权重。
# 运行：source env.sh && python code/ch02/needle_probe.py --arm full --steps 300 --out-name smoke
# 正文历史档：T64 / w16 / L2 / d256 / h8 / h_kv4 / d_k64 / 无RoPE；8k 与20k采用各自余弦日程。
# 本任务目标键固定，干扰键有放回抽样；目标召回不代表一般 MQAR 能力，平均查询损失还可能受歧义影响。
# 两层查询62的输入依赖最早到32，目标值p+1可达需p>=31；旧单层阈值46已撤回。
# full 与 swa_sink 同时改变掩码及sink，不能用两臂独立判断sink收益。
# 冒烟只检管道；截断运行不按完整预算验收。JSON/CSV独立命名且拒绝覆盖，checkpoint支持相同日程续跑。
# 续跑：同参数、同 --out-name 加 --resume；不得把8k日程的checkpoint续成20k而冒称同一实验。
# 产物 log/book5-ch02/needle_probe_<name>_<arm>.json 与对应 CSV/checkpoint。
import argparse
import csv
import json
import math
import os
import sys
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = str(next((p for p in Path(__file__).resolve().parents if (p / "env.sh").is_file()), Path(__file__).resolve().parents[3]))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book5-ch02")
SEED = 20261002

# 跨册三候选 bootstrap（llama409 `_B3_CANDS` 同款已验证模式：工作区 / 书仓嵌套 / 书仓裸布局）
_B4_FEAS_CANDS = [
    str(p / "book4-scale-and-efficiency/code/00-feasibility")
    for p in Path(__file__).resolve().parents
] + [
    os.path.join(REPO_ROOT, "from-models-to-vllm", "book4-scale-and-efficiency", "code", "00-feasibility"),
    os.path.join(REPO_ROOT, "code", "Book4-规模与效率", "00-feasibility"),
]


def bootstrap_book4():
    hits = [p for p in _B4_FEAS_CANDS if os.path.exists(os.path.join(p, "moe_mla_slots.py"))]
    if not hits:
        raise FileNotFoundError(_B4_FEAS_CANDS)
    sys.path.insert(0, hits[0])
    import moe_mla_slots as m4
    return m4, hits


# ---------------- 位置件（MiniLM 接口恒传 cos/sin；needle 任务关 rope，只留管线） ----------------
def rope_cache(n, d_k, theta, device):
    """cos/sin (1,1,n,d_k)：freqs=outer(pos, inv_freq) 后 cat(freqs,freqs)——llama 系同构。"""
    inv = 1.0 / (theta ** (torch.arange(0, d_k // 2, dtype=torch.float32, device=device) / (d_k // 2)))
    ang = torch.outer(torch.arange(n, dtype=torch.float32, device=device), inv)              # (n,d_k/2)
    freqs = torch.cat((ang, ang), dim=-1)                                                    # (n,d_k)
    return freqs.cos()[None, None], freqs.sin()[None, None]


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    """x (B,h,n,d_k) · cos/sin (1,1,n,d_k) → (B,h,n,d_k)。"""
    return x * cos + rotate_half(x) * sin


def band_causal_mask(n, window, device):
    """banded causal mask (n,n) bool，True=可看：j ≤ i 且（window 给定时）i-j < window。"""
    i = torch.arange(n, device=device)[:, None]
    j = torch.arange(n, device=device)[None, :]
    m = j <= i
    if window is not None:
        m = m & (i - j < window)
    return m


# ---------------- 注意力插槽：full / swa_sink（forward(x, cos, sin) → (B,n,d)） ----------------
class FullAttn(nn.Module):
    """满因果注意力（对照组）：GQA 四投影 + 可选 rope + causal；fp32 softmax。"""

    def __init__(self, d, n_head, n_kv_head, d_k, use_rope=True):
        super().__init__()
        self.h, self.h_kv, self.d_k = n_head, n_kv_head, d_k
        self.use_rope = use_rope
        self.scale = d_k ** -0.5
        self.q_proj = nn.Linear(d, n_head * d_k, bias=False)                     # (B,n,d)→(B,n,h·d_k)
        self.k_proj = nn.Linear(d, n_kv_head * d_k, bias=False)
        self.v_proj = nn.Linear(d, n_kv_head * d_k, bias=False)
        self.o_proj = nn.Linear(n_head * d_k, d, bias=False)

    def forward(self, x, cos, sin):
        B, n, d = x.shape
        q = self.q_proj(x).view(B, n, self.h, self.d_k).transpose(1, 2)          # (B,h,n,d_k)
        k = self.k_proj(x).view(B, n, self.h_kv, self.d_k).transpose(1, 2)       # (B,h_kv,n,d_k)
        v = self.v_proj(x).view(B, n, self.h_kv, self.d_k).transpose(1, 2)
        if self.use_rope:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.h // self.h_kv
        k = k.repeat_interleave(rep, dim=1)                                      # (B,h,n,d_k) GQA 展开
        v = v.repeat_interleave(rep, dim=1)
        att = (q @ k.transpose(-2, -1)) * self.scale                             # (B,h,n,n)
        att = att.masked_fill(~band_causal_mask(n, None, x.device), float("-inf"))
        p = F.softmax(att.float(), dim=-1).to(v.dtype)
        y = (p @ v).transpose(1, 2).contiguous().view(B, n, self.h * self.d_k)   # (B,n,h·d_k)
        return self.o_proj(y)                                                    # (B,n,d)


class SwaSinkAttn(nn.Module):
    """滑窗注意力 + 每头可学 sink 标量（gpt-oss 语义，transformers 5.18.0 L251-259）：
    sink 作为第 n+1 个 logit 拼进 softmax、无 value、概率质量在加权前丢弃——
    数学上 = 分母多一项 exp(s_h)，输出仍是 Σ p_j v_j（sink 只重分配窗内权重）。"""

    def __init__(self, d, n_head, n_kv_head, d_k, window, use_sink=True, use_rope=True):
        super().__init__()
        self.h, self.h_kv, self.d_k, self.window = n_head, n_kv_head, d_k, window
        self.use_sink, self.use_rope = use_sink, use_rope
        self.scale = d_k ** -0.5
        self.q_proj = nn.Linear(d, n_head * d_k, bias=False)
        self.k_proj = nn.Linear(d, n_kv_head * d_k, bias=False)
        self.v_proj = nn.Linear(d, n_kv_head * d_k, bias=False)
        self.o_proj = nn.Linear(n_head * d_k, d, bias=False)
        if use_sink:
            self.sinks = nn.Parameter(torch.zeros(n_head))                       # 每头 1 个标量，零初始化

    def forward(self, x, cos, sin):
        B, n, d = x.shape
        q = self.q_proj(x).view(B, n, self.h, self.d_k).transpose(1, 2)          # (B,h,n,d_k)
        k = self.k_proj(x).view(B, n, self.h_kv, self.d_k).transpose(1, 2)       # (B,h_kv,n,d_k)
        v = self.v_proj(x).view(B, n, self.h_kv, self.d_k).transpose(1, 2)
        if self.use_rope:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.h // self.h_kv
        k = k.repeat_interleave(rep, dim=1)                                      # (B,h,n,d_k)
        v = v.repeat_interleave(rep, dim=1)
        att = (q @ k.transpose(-2, -1)) * self.scale                             # (B,h,n,n)
        att = att.masked_fill(~band_causal_mask(n, self.window, x.device), float("-inf"))
        if self.use_sink:
            sink = self.sinks.view(1, self.h, 1, 1).expand(B, self.h, n, 1)      # (B,h,n,1) 虚拟 sink logit
            att = torch.cat([att, sink], dim=-1)                                 # (B,h,n,n+1)
            att = att - att.max(dim=-1, keepdim=True).values                     # 数值稳定（HF 同款）
            p = F.softmax(att.float(), dim=-1)[..., :n].to(v.dtype)              # 丢 sink 概率质量 (B,h,n,n)
        else:
            p = F.softmax(att.float(), dim=-1).to(v.dtype)
        y = (p @ v).transpose(1, 2).contiguous().view(B, n, self.h * self.d_k)   # (B,n,h·d_k)
        return self.o_proj(y)                                                    # (B,n,d)


# ---------------- MiniLM 骨架（Book2/3 拓扑自包含版；ch03 nsa_block.py 复用） ----------------
class MiniLM(nn.Module):
    """x = x + attn(norm1(x)); x = x + swiglu(norm2(x))——残差河恒宽；attn 插槽逐层注入。"""

    def __init__(self, vocab=8192, d=256, n_layer=2, d_ff=704, eps=1e-5, attn_modules=None):
        super().__init__()
        import importlib
        slots_mod = importlib.import_module("moe_mla_slots").bootstrap_book3()
        self.embed = nn.Embedding(vocab, d)                                      # (B,n)→(B,n,d)
        self.attns = nn.ModuleList(attn_modules)
        self.norm1 = nn.ModuleList([slots_mod.RMSNorm(d, eps) for _ in range(n_layer)])
        self.norm2 = nn.ModuleList([slots_mod.RMSNorm(d, eps) for _ in range(n_layer)])
        self.ffns = nn.ModuleList([slots_mod.SwiGLU(type("S", (), {"hidden_size": d,
                                                                  "intermediate_size": d_ff})())
                                   for _ in range(n_layer)])
        self.d_k = self.attns[0].d_k
        self.theta = 10000.0
        self.norm_f = slots_mod.RMSNorm(d, eps)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, n = idx.shape
        x = self.embed(idx)                                                      # (B,n,d)
        cos, sin = rope_cache(n, self.d_k, self.theta, idx.device)                # (1,1,n,d_k)
        for a, n1, n2, f in zip(self.attns, self.norm1, self.norm2, self.ffns):
            x = x + a(n1(x), cos, sin)                                           # (B,n,d)
            x = x + f(n2(x))                                                     # (B,n,d)
        logits = self.lm_head(self.norm_f(x))                                    # (B,n,V)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))
        return logits, loss


# ---------------- 合成 needle-recall 任务（MQAR 式；设计承探针 02 v4，T=64 定版） ----------------
# 词表分区：filler∈[0,4000) / 值∈[4000,4064) / needle 键 K=8099 / 干扰键∈[8100,8180)。
# 序列布局（T=64）：[0,16) 8 个干扰 (k,v) 对；needle 对 (K,v) 在受控位置 p；[48,64) 查询区——
#   7 个干扰查询（键值流内紧随应答——紧邻答案在未来，但键值区已有答案）+ 末位 needle 查询
#   q=K@62、应答 a=v@63；监督只放 8 个查询位（查询-only，防自拷贝捷径——试点教训②）。
NEEDLE_K = 8099
N_PAIRS, N_QUERIES = 8, 8
# 训练分布：50% 远端 [16,42) + 50% 可达位 {46}（p+1=47 ≤ 47 且不撞查询区 [48,64)）。
FAR_BAND, NEAR_BAND = (16, 42), (46, 47)


def make_needle_batch(B, T, rng, p_min=None, device="cpu"):
    """p_min 给定时钉死 needle 位置（eval 扫描）；否则 50/50 远端/可达混合（训练）。
    返回 (x, p)：x (B,T) int64，p (B,) needle 位置。随机 CE 基线 ln(64)=4.16。"""
    if B < 1 or T < 64 or (p_min is not None and not 16 <= p_min < T - 17):
        raise ValueError("需要 B>0、T>=64，目标键值对不得覆盖干扰区或查询区")
    x = torch.from_numpy(rng.integers(0, 4000, size=(B, T)).astype(np.int64))    # filler (B,T)
    dk = rng.integers(8100, 8180, size=(B, N_PAIRS))                            # 干扰键 (B,8)
    dv = rng.integers(4000, 4064, size=(B, N_PAIRS))                            # 干扰值 (B,8)
    v = torch.from_numpy(rng.integers(4000, 4064, size=(B,)).astype(np.int64))  # needle 值 (B,)
    x[:, 0:16:2] = torch.from_numpy(dk.astype(np.int64))                        # 对区键
    x[:, 1:16:2] = torch.from_numpy(dv.astype(np.int64))                        # 对区值
    if p_min is None:                                                           # 训练：50/50 混合
        far = rng.integers(FAR_BAND[0], FAR_BAND[1], size=B)
        near = rng.integers(NEAR_BAND[0], NEAR_BAND[1], size=B)
        p_np = np.where(rng.random(B) < 0.5, far, near)
    else:                                                                       # eval：钉死位置
        p_np = np.full(B, p_min)
    p = torch.from_numpy(p_np.astype(np.int64))
    rows = torch.arange(B)
    x[rows, p] = NEEDLE_K                                                       # needle 键
    x[rows, p + 1] = v                                                          # needle 值
    perm = np.argsort(rng.random((B, N_PAIRS)), axis=1)[:, :N_QUERIES - 1]      # (B,7) 每行不同的 7 个
    x[:, T - 16: T - 2: 2] = torch.from_numpy(np.take_along_axis(dk, perm, axis=1).astype(np.int64))
    x[:, T - 15: T - 1: 2] = torch.from_numpy(np.take_along_axis(dv, perm, axis=1).astype(np.int64))
    x[:, T - 2] = NEEDLE_K                                                      # q8 = needle 键
    x[:, T - 1] = v                                                             # a8 = needle 值（eval 目标）
    return x.to(device), p.to(device)


def needle_loss(model, x, p, T):
    """MQAR 损失：8 个查询位（T-16,T-14,...,T-2）的 CE，目标=流内紧随的应答 token。"""
    logits, _ = model(x)                                                        # (B,T,V)
    qpos = torch.arange(T - 16, T, 2, device=x.device)                          # (8,)
    return F.cross_entropy(logits[:, qpos].float().reshape(-1, logits.size(-1)),
                           x[:, qpos + 1].reshape(-1))


@torch.no_grad()
def needle_eval(model, T, positions, n_seq=60, seed=SEED + 999, device="cpu"):
    """位置扫描：各位置 acc = argmax(logits[:,T-2]) == x[:,T-1]（needle 应答）的比例。"""
    if n_seq < 1:
        raise ValueError("n_seq 必须为正")
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    out = {}
    try:
        for pp in positions:
            hits = 0
            for start in range(0, n_seq, 12):
                x, _ = make_needle_batch(min(12, n_seq-start), T, rng, p_min=pp, device=device)
                pred = model(x)[0][:, T-2].argmax(dim=-1)
                hits += int((pred == x[:, T-1]).sum())
            out[int(pp)] = hits / n_seq
    finally:
        model.train(was_training)
    return out


def train_mini(model, steps, batch_fn, device, lr=1e-3, warmup=50, tag="",
               cosine=False, time_budget_min=0.0, log_every=200,
               checkpoint_path=None, resume=False, checkpoint_every=200, stop_after=None):
    """共用短训环（AdamW 0.9/0.95 wd0.1 + clip1.0 + warmup[+cosine]；MPS bf16 autocast）。
    time_budget_min>0 时超窗主动截断（返回 truncated=True——超窗降档留痕条款，见文件头）。"""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    if steps < 1 or warmup < 1 or checkpoint_every < 1:
        raise ValueError("steps/warmup/checkpoint_every 必须为正")
    contract = dict(steps=steps, lr=lr, warmup=warmup, cosine=cosine,
                    batch_type=type(batch_fn).__name__, B=getattr(batch_fn,"B",None),
                    T=getattr(batch_fn,"T",None), device=str(device), torch=torch.__version__,
                    shapes={k:list(v.shape) for k,v in model.state_dict().items()},
                    attentions=[{key:getattr(a,key,None) for key in
                                 ("window","block","n_selected","use_sink","use_rope","h","h_kv","d_k")}
                                for a in getattr(model,"attns",[])],
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if checkpoint_path and Path(checkpoint_path).exists() and not resume:
        raise FileExistsError(checkpoint_path)
    if resume and (not checkpoint_path or not Path(checkpoint_path).exists()):
        raise FileNotFoundError(checkpoint_path)
    model.train()
    t_start, secs, losses, truncated = time.perf_counter(), [], [], False
    ran = 0
    if resume:
        # 仅加载本脚本自产、受信任的本地 checkpoint。
        ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if ck["contract"] != contract:
            raise ValueError("checkpoint 与当前模型、批配置、总步数或日程不一致")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        ran, losses, secs = ck["step"], ck["losses"], ck["secs"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        if ck.get("batch_rng") is not None:
            batch_fn.rng.bit_generator.state = ck["batch_rng"]
        if ck.get("mps_rng") is not None:
            torch.mps.set_rng_state(ck["mps_rng"].cpu())
    def save_checkpoint():
        if not checkpoint_path:
            return
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ck = dict(contract=contract, model=model.state_dict(), optimizer=opt.state_dict(),
                  step=ran, losses=losses, secs=secs, torch_rng=torch.get_rng_state(),
                  batch_rng=batch_fn.rng.bit_generator.state if hasattr(batch_fn,"rng") else None,
                  mps_rng=torch.mps.get_rng_state() if device.type=="mps" else None)
        tmp = path.with_suffix(path.suffix+".tmp")
        torch.save(ck, tmp)
        os.replace(tmp, path)
    for step in range(ran + 1, steps + 1):
        sched = min(1.0, step / warmup)
        if cosine:
            sched *= 0.5 * (1 + math.cos(math.pi * step / steps))
        opt.param_groups[0]["lr"] = lr * sched
        x, y = batch_fn(step)
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "mps" \
                else torch.autocast("cpu", enabled=False):
            loss = batch_fn.loss(model, x, y)
        if not torch.isfinite(loss):
            save_checkpoint()
            raise FloatingPointError(f"非有限 loss，step={step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        if device.type == "mps":
            torch.mps.synchronize()
        secs.append(time.perf_counter() - t0)
        losses.append(loss.item())
        ran = step
        if step % log_every == 0 or step == steps:
            el = time.perf_counter() - t_start
            print(f"    [{tag}] step {step}/{steps} loss {loss.item():.4f} "
                  f"| elapsed {el:.0f}s | median {np.median(secs)*1e3:.0f} ms/step", flush=True)
        if step % checkpoint_every == 0:
            save_checkpoint()
        if (stop_after is not None and step >= stop_after) or (time_budget_min > 0 and (time.perf_counter() - t_start) > time_budget_min * 60):
            print(f"    [{tag}] 保存停止点 @step {step}/{steps}（时间或局部测试步数上限）",
                  flush=True)
            truncated = step < steps
            break
    save_checkpoint()
    return {"sec_per_step_p50": float(np.median(secs)), "loss_first": losses[0],
            "loss_last_20_mean": float(np.mean(losses[-20:])), "steps_ran": ran,
            "wall_clock_sec": round(time.perf_counter() - t_start, 1), "truncated": truncated}


class NeedleBatch:
    """合成 needle 训练批（同 seed 同序列——两臂吃到完全相同的批序列，受控对照）。"""
    def __init__(self, B, T, seed, device):
        self.rng, self.B, self.T, self.device = np.random.default_rng(seed), B, T, device
    def __call__(self, step):
        x, p = make_needle_batch(self.B, self.T, self.rng, device=self.device)
        return (x, p), None
    def loss(self, model, batch, _):
        (x, p) = batch
        return needle_loss(model, x, p, self.T)


class SpBatch:
    """sp-8k 语料批（TokenStream 确定性顺序批）——ch03 质量对照复用。"""
    def __init__(self, m4, batch, block):
        self.B, self.T = batch, block
        self.stream = m4.TokenStream(m4.TOKENS_SP8K, batch, block)
        self.device = m4.pick_device("mps")
    def __call__(self, step):
        return self.stream.batch(step, self.device)
    def loss(self, model, x, y):
        logits, _ = model(x, y)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))


def build_arm(kind, cfg, state_dict=None):
    """两臂构建：full / swa_sink。同初始化：state_dict 由 full 臂（干净种子下构建）灌注，
    形状全同；sink 为 swa_sink 臂独有参数，零初始化是教学选择。"""
    if kind not in {"full", "swa", "swa_sink"}:
        raise ValueError(kind)
    d, L, h, h_kv, d_k, d_ff = cfg["d"], cfg["L"], cfg["h"], cfg["h_kv"], cfg["d_k"], cfg["d_ff"]
    w, use_rope = cfg.get("window"), cfg.get("use_rope", True)
    attns = [FullAttn(d, h, h_kv, d_k, use_rope) if kind == "full"
             else SwaSinkAttn(d, h, h_kv, d_k, w, use_sink=(kind == "swa_sink"), use_rope=use_rope)
             for _ in range(L)]
    m = MiniLM(d=d, n_layer=L, d_ff=d_ff, attn_modules=attns)
    if state_dict is not None:
        missing, unexpected = m.load_state_dict(state_dict, strict=False)
        assert not unexpected, unexpected
        assert all("sinks" in k for k in missing), missing                    # 只允许 sink 缺省（零初始化）
    return m


def upsert_csv(arm, steps, acc_by_pos, out_name="run"):
    """保留函数名兼容调用；新产物按运行隔离，绝不覆盖旧曲线。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"needle_scan_{out_name}_{arm}.csv")
    with open(path, "x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["arm", "steps", "position", "acc"])
        writer.writeheader()
        for pp, acc in sorted(acc_by_pos.items()):
            writer.writerow(dict(arm=arm, steps=steps, position=pp, acc=f"{acc:.6f}"))
    return path


def main():
    ap = argparse.ArgumentParser(description="Book5 ch2 needle 位置扫描正式件（两臂：full / swa_sink）")
    ap.add_argument("--arm", choices=["full", "swa_sink"], required=True)
    ap.add_argument("--steps", type=int, default=300,
                    help="训练步数（档位：冒烟 300 / fast 8000 / full 20000）")
    ap.add_argument("--out-name", default=None, help="产物后缀（默认 {arm}_s{steps}）")
    ap.add_argument("--time-budget-min", type=float, default=0.0,
                    help="单臂训练 wall-clock 上限（分钟；超窗主动截断留痕，0=不设限）")
    ap.add_argument("--resume", action="store_true", help="同总步数、日程与参数恢复自产checkpoint")
    args = ap.parse_args()
    m4, hits = bootstrap_book4()
    device = m4.pick_device("mps")
    out_name = args.out_name or f"{args.arm}_s{args.steps}"
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Path(OUT_DIR) / f"needle_probe_{out_name}_{args.arm}"
    ckpath = str(base)+".pt"
    # 截断报告以实际步数独立保存；恢复后最终报告仍写 base.json。
    if base.with_suffix(".json").exists() or (Path(ckpath).exists() and not args.resume):
        raise FileExistsError(base)
    print(f"[bootstrap] moe_mla_slots 自：{hits[0]} | device {device} | arm {args.arm} | {out_name}",
          flush=True)

    cfg = dict(d=256, L=2, h=8, h_kv=4, d_k=64, d_ff=704, window=16, use_rope=False)  # 正档几何（文件头）
    T, cliff = 64, 31                                                            # 值可达边界：T-2-L*(w-1)-1
    positions = [16, 22, 28, 34, 40, 46]                                         # 全部避开查询区 [48,64)

    # 同初始化：干净种子下建 full 臂取 state_dict，再灌入目标臂（跨进程可复现）
    torch.manual_seed(SEED)
    sd = build_arm("full", cfg).state_dict()
    torch.manual_seed(SEED)
    m = build_arm(args.arm, cfg, sd).to(device)
    n_par = sum(p_.numel() for p_ in m.parameters())
    print(f"[init] {n_par/1e6:.2f}M 参数 | cliff={cliff} | eval 位置 {positions}", flush=True)

    bf = NeedleBatch(32, T, SEED + 7, device)
    res = train_mini(m, args.steps, bf, device, lr=3e-3, warmup=50, cosine=True,
                     tag=f"{args.arm}/{out_name}", time_budget_min=args.time_budget_min,
                     checkpoint_path=ckpath, resume=args.resume)
    acc = needle_eval(m, T, positions, device=device)
    print(f"[{args.arm}] train {res['loss_first']:.3f}→{res['loss_last_20_mean']:.3f} "
          f"| {res['sec_per_step_p50']*1e3:.0f} ms/step | acc@pos "
          f"{ {k: round(v, 2) for k, v in acc.items()} }", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = f"_partial_s{res['steps_ran']}" if res["truncated"] else ""
    jpath = str(base)+suffix+".json"   # 文件名恒带臂名防覆写
    with open(jpath, "x", encoding="utf-8") as f:
        json.dump({"arm": args.arm, "steps": args.steps, "seed": SEED, "date": datetime.now(timezone.utc).isoformat(),
                   "device": str(device), "torch": torch.__version__,
                   "evaluation": "complete_budget" if not res["truncated"] and args.steps>=8000 else "smoke_or_partial",
                   "cfg": cfg, "T": T, "reach_cliff": cliff, "positions": positions,
                   "far_band": list(FAR_BAND), "near_band": list(NEAR_BAND),
                   "train": res, "recall_acc_by_pos": acc,
                   "revised_checks": {
                       "A_full_all_pos": all(v >= (0.8 if args.steps >= 20000 else 0.7) for v in acc.values())
                           if args.arm=="full" and args.steps>=8000 and not res["truncated"] else None,
                       "unreachable_positions": [p for p in positions if p<cliff],
                       "reachable_positions": [p for p in positions if p>=cliff],
                       "sink_independent_effect": "not_identifiable_from_two_arms",
                       "legacy_cliff_46": "invalid_for_two_layer_model"},
                   "references": {"gpt_oss_sink_semantics":
                                  "历史 5.18.0 行号见冻结稿；当前数学语义 cat-softmax-drop，由 swa_sink.py 对拍",
                                  "design_record": "00-feasibility/02_probe_swa_sink.py（探针，T=64 试点 acc 0.83@8k）"},
                   }, f, ensure_ascii=False, indent=2)
    print(f"[产物] {jpath}", flush=True)
    upsert_csv(args.arm, res["steps_ran"], acc, out_name+suffix)
    print(f"{args.arm}/{out_name} 完成。", flush=True)


if __name__ == "__main__":
    main()
