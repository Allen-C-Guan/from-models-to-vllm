# needle_probe.py —— Book5 ch2 正式实验件：lost-in-the-middle mini 复现（needle 位置扫描，两臂对照）
# 用途（大纲 ch2 2.6「本机复现 lost-in-the-middle」正身实验；探针 02 升格——升格判据：凡入正文
#   数字=本脚本重跑+JSON 落 log/book5-ch02/；探针 02 JSON 只作设计记录，其 T=64 试点 acc 0.83
#   （fast 8k，位置 18/28/38）为档位选择依据，不直接入正文）：
#   合成 needle-recall 任务（MQAR 式）上训练两臂——full（满因果注意力）vs swa_sink（滑窗+每头可学
#   sink 标量，gpt-oss 语义：cat([attn_weights, sinks])→softmax→probs[...,:-1] 丢 sink 位，
#   transformers 5.18.0 modeling_gpt_oss.py L251-259），然后做 needle 位置扫描 eval：
#   每个 eval 位置的 acc = argmax 预测命中 needle 值的比例——「needle 放得越远越记不住」的本机读数。
# 【正档几何（探针 02 实测定版，与试点记录一致）】T=64 / w=16 / L=2 / d=256 / h=8 / h_kv=4 /
#   d_k=64 / d_ff=704 / 无 rope（隔离 mask 变量）/ B=32 / lr 3e-3 + warmup 50 + cosine；
#   词表分区：filler∈[0,4000) / 值∈[4000,4064)（64 个，随机 CE 基线 ln 64=4.16、acc 基线 ≈1/64≈0.016）/
#   needle 键 K=8099 / 干扰键∈[8100,8180)；干扰 (k,v) 对占 [0,16)，查询区占 [T-16,T)=[48,64)，
#   needle 可放位置 p∈[16,46]（p+1≤47 不撞查询区——试点教训④：撞查询区产生位置伪影）。
#   训练分布 50/50：远端带 [16,42) ∪ 可达位 {46}（唯一干净可达位——见判据 B）。
#   可达性（cliff）：needle 查询在 T-2=62，窗口 16 覆盖 [47,62]；归纳头电路需「值位 p+1」可见
#   → p+1 ≥ 47 → cliff = 46（探针 02 同式：cliff = T-2-(w-1)-1）。
# 【判据预注册（跑前写定，防事后择优；以 full 20k 档为正档结算；负结论照报）】
#   A. full 臂·任意位置召回：fast 8k 所有位置 acc ≥ 0.7；full 20k 所有位置 acc ≥ 0.8
#      （试点复现带：acc 0.83@8k）。不达 = 「满注意力在小任务上也欠训」照实报，档位延长而非换几何。
#   B. swa_sink 臂·可达性决定论：不可达位置（16/22/28/34/40，均 < cliff=46）acc ≤ 0.15
#      （随机基线 0.016 + 干扰值混淆余量）。可达位 46 两分支预注册：
#      B1 acc ≥ 0.7 → 「窗内召回成立」：滑窗把可达范围内召回学到与 full 相当（正文主线写法）；
#      B2 acc < 0.7 → 「窗内召回不成立」：即使训练分布 50% 钉在唯一可达位、监督信号充分，
#      滑窗仍学不会 → 正文按可达性决定论的更强表述如实写（SWA 在该任务几何下的可学性边界；
#      「不成立」是合法结果，不算失败、不追加预算翻案）。
#   C. sink 语义核对（两臂制下读法）：sink 是 softmax 分母的可学质量锚，不传递远端内容——
#      swa_sink 臂不可达位置仍在随机带（≤0.15，含在 B 内）即 C 成立；若不可达位置 acc ≥ 0.3
#      = 反例信号（sink 侧信道），如实追查。与探针 02 三臂制（swa vs swa_sink）结论互证：
#      sink 不救远端、不伤窗内。
#   结算规则：冒烟 300 只验管道（loss 有限下降 + eval 可跑），不参与判据；fast 8k = 写作期参照；
#   full 20k = 正文正档。单种子 20261002（纪律：模块实验单种子+JSON 存档+趋势判据+结果分支预注册）。
# 【超窗降档留痕条款】full 20k 预计 ≈8 min/臂（T=64 L=2 实测 ≈24 ms/步）。若单臂 wall-clock 超
#   --time-budget-min（full 档运行命令统一给 25 min ≈ 预计 3×，MPS 降频/后台争用兜底），训练环
#   主动截断：JSON 落 "truncated": true 与已跑步数，正文降档改用 fast 8k 作正档、full 只报趋势；
#   降档事实与原因必须留痕写进 notes/06（不许静默丢弃）。
# 运行方式（三档；单臂单进程，两臂分两次跑——nohup 链式串行见 notes/06）：
#   冒烟（≈10 s/臂）  ：source env.sh && python "code/ch02/needle_probe.py" --arm full  --steps 300   --out-name smoke300
#   fast 8k（≈3.5 min/臂）：... --arm full  --steps 8000  --out-name fast8k   （另一臂 --arm swa_sink）
#   full 20k（≈8 min/臂） ：... --arm swa_sink --steps 20000 --out-name full20k --time-budget-min 25
# 产物：log/book5-ch02/needle_probe_{out-name}_{arm}.json（单臂数字，文件名恒带臂名防覆写）+
#   log/book5-ch02/needle_scan.csv（两臂位置扫描合并曲线：arm,steps,position,acc 四列，按
#   (arm,steps) upsert——两臂跑完即得「full vs 滑窗+sink」对照曲线；ch2 图 2.3 的数据源）。
# 升格注：本件内嵌 FullAttn/SwaSinkAttn 插槽与 MiniLM 骨架自包含（ch03 nsa_block.py 复用）；
#   ch2 写作期交付 swa_sink.py::SlidingWindowSinkAttention(cfg) 后，可将内嵌插槽对齐/替换为
#   import（行为须保持：banded mask、sink 语义 L251-259、零初始化、同初始化灌注接口）。
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book5-ch02")
SEED = 20261002

# 跨册三候选 bootstrap（llama409 `_B3_CANDS` 同款已验证模式：工作区 / 书仓嵌套 / 书仓裸布局）
_B4_FEAS_CANDS = [
    os.path.join(REPO_ROOT, "code", "Book4-规模与效率", "00-feasibility"),
    os.path.join(REPO_ROOT, "from-models-to-vllm", "book4-scale-and-efficiency", "code", "00-feasibility"),
    os.path.join(REPO_ROOT, "book4-scale-and-efficiency", "code", "00-feasibility"),
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
        p = F.softmax(att.float(), dim=-1).to(x.dtype)
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
            p = F.softmax(att.float(), dim=-1)[..., :n].to(x.dtype)              # 丢 sink 概率质量 (B,h,n,n)
        else:
            p = F.softmax(att.float(), dim=-1).to(x.dtype)
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
#   7 个干扰查询（键值流内紧随应答——因果上查询位尚未见答案，必须回检索）+ 末位 needle 查询
#   q=K@62、应答 a=v@63；监督只放 8 个查询位（查询-only，防自拷贝捷径——试点教训②）。
NEEDLE_K = 8099
N_PAIRS, N_QUERIES = 8, 8
# 训练分布：50% 远端 [16,42) + 50% 可达位 {46}（p+1=47 ≤ 47 且不撞查询区 [48,64)）。
FAR_BAND, NEAR_BAND = (16, 42), (46, 47)


def make_needle_batch(B, T, rng, p_min=None, device="cpu"):
    """p_min 给定时钉死 needle 位置（eval 扫描）；否则 50/50 远端/可达混合（训练）。
    返回 (x, p)：x (B,T) int64，p (B,) needle 位置。随机 CE 基线 ln(64)=4.16。"""
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
    model.eval()
    rng = np.random.default_rng(seed)
    out = {}
    for pp in positions:
        accs = []
        for _ in range(n_seq // 12):
            x, _ = make_needle_batch(12, T, rng, p_min=pp, device=device)
            pred = model(x)[0][:, T - 2].argmax(dim=-1)                          # (12,) 查询位预测
            accs.append((pred == x[:, T - 1]).float().mean().item())
        out[int(pp)] = float(np.mean(accs))
    model.train()
    return out


def train_mini(model, steps, batch_fn, device, lr=1e-3, warmup=50, tag="",
               cosine=False, time_budget_min=0.0, log_every=200):
    """共用短训环（AdamW 0.9/0.95 wd0.1 + clip1.0 + warmup[+cosine]；MPS bf16 autocast）。
    time_budget_min>0 时超窗主动截断（返回 truncated=True——超窗降档留痕条款，见文件头）。"""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    model.train()
    t_start, secs, losses, truncated = time.perf_counter(), [], [], False
    ran = 0
    for step in range(1, steps + 1):
        sched = min(1.0, step / warmup)
        if cosine:
            sched *= 0.5 * (1 + math.cos(math.pi * step / steps))
        opt.param_groups[0]["lr"] = lr * sched
        x, y = batch_fn(step)
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "mps" \
                else torch.autocast("cpu", enabled=False):
            loss = batch_fn.loss(model, x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        if time_budget_min > 0 and (time.perf_counter() - t_start) > time_budget_min * 60:
            print(f"    [{tag}] 超窗截断 @step {step}（>{time_budget_min} min）——降档留痕条款触发",
                  flush=True)
            truncated = True
            break
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
        self.stream = m4.TokenStream(m4.TOKENS_SP8K, batch, block)
        self.device = m4.pick_device("mps")
    def __call__(self, step):
        return self.stream.batch(step, self.device)
    def loss(self, model, x, y):
        logits, _ = model(x, y)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))


def build_arm(kind, cfg, state_dict=None):
    """两臂构建：full / swa_sink。同初始化：state_dict 由 full 臂（干净种子下构建）灌注，
    形状全同；sink 为 swa_sink 臂独有参数、零初始化（gpt-oss 语义试点教训⑤）。"""
    d, L, h, h_kv, d_k, d_ff = cfg["d"], cfg["L"], cfg["h"], cfg["h_kv"], cfg["d_k"], cfg["d_ff"]
    w, use_rope = cfg.get("window"), cfg.get("use_rope", True)
    attns = [FullAttn(d, h, h_kv, d_k, use_rope) if kind == "full"
             else SwaSinkAttn(d, h, h_kv, d_k, w, use_sink=(kind != "full"), use_rope=use_rope)
             for _ in range(L)]
    m = MiniLM(d=d, n_layer=L, d_ff=d_ff, attn_modules=attns)
    if state_dict is not None:
        missing, unexpected = m.load_state_dict(state_dict, strict=False)
        assert not unexpected, unexpected
        assert all("sinks" in k for k in missing), missing                    # 只允许 sink 缺省（零初始化）
    return m


def upsert_csv(arm, steps, acc_by_pos):
    """合并位置扫描 CSV：arm,steps,position,acc 四列；按 (arm,steps,position) upsert——
    两臂（各档）跑完后 needle_scan.csv 即图 2.3 的两臂对照曲线数据源。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "needle_scan.csv")
    rows = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows[(r["arm"], int(r["steps"]), int(r["position"]))] = r
    for pp, acc in acc_by_pos.items():
        rows[(arm, steps, int(pp))] = {"arm": arm, "steps": str(steps),
                                       "position": str(int(pp)), "acc": f"{acc:.4f}"}
    with open(path, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=["arm", "steps", "position", "acc"])
        wtr.writeheader()
        for key in sorted(rows, key=lambda k: (k[1], k[0], k[2])):
            wtr.writerow(rows[key])
    print(f"[产物] {path}（{len(rows)} 行）")
    return path


def main():
    ap = argparse.ArgumentParser(description="Book5 ch2 needle 位置扫描正式件（两臂：full / swa_sink）")
    ap.add_argument("--arm", choices=["full", "swa_sink"], required=True)
    ap.add_argument("--steps", type=int, default=300,
                    help="训练步数（档位：冒烟 300 / fast 8000 / full 20000）")
    ap.add_argument("--out-name", default=None, help="产物后缀（默认 {arm}_s{steps}）")
    ap.add_argument("--time-budget-min", type=float, default=0.0,
                    help="单臂训练 wall-clock 上限（分钟；超窗主动截断留痕，0=不设限）")
    args = ap.parse_args()
    m4, hits = bootstrap_book4()
    device = m4.pick_device("mps")
    out_name = args.out_name or f"{args.arm}_s{args.steps}"
    print(f"[bootstrap] moe_mla_slots 自：{hits[0]} | device {device} | arm {args.arm} | {out_name}",
          flush=True)

    cfg = dict(d=256, L=2, h=8, h_kv=4, d_k=64, d_ff=704, window=16, use_rope=False)  # 正档几何（文件头）
    T, cliff = 64, 46                                                            # cliff = T-2-(w-1)-1
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
                     tag=f"{args.arm}/{out_name}", time_budget_min=args.time_budget_min)
    acc = needle_eval(m, T, positions, device=device)
    print(f"[{args.arm}] train {res['loss_first']:.3f}→{res['loss_last_20_mean']:.3f} "
          f"| {res['sec_per_step_p50']*1e3:.0f} ms/step | acc@pos "
          f"{ {k: round(v, 2) for k, v in acc.items()} }", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    jpath = os.path.join(OUT_DIR, f"needle_probe_{out_name}_{args.arm}.json")   # 文件名恒带臂名防覆写
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump({"arm": args.arm, "steps": args.steps, "seed": SEED, "date": "2026-10-05",
                   "cfg": cfg, "T": T, "reach_cliff": cliff, "positions": positions,
                   "far_band": list(FAR_BAND), "near_band": list(NEAR_BAND),
                   "train": res, "recall_acc_by_pos": acc,
                   "prereg_verdict_this_arm": {
                       "A_full_all_pos": all(v >= (0.8 if args.steps >= 20000 else 0.7)
                                             for v in acc.values()) if args.arm == "full" else None,
                       "B_unreachable_random": all(acc[p] <= 0.15 for p in positions if p < cliff)
                       if args.arm == "swa_sink" else None,
                       "B_branch_at_cliff": ("B1_窗内召回成立" if acc[cliff] >= 0.7
                                             else "B2_窗内召回不成立→可达性决定论如实写")
                       if args.arm == "swa_sink" else None},
                   "references": {"gpt_oss_sink_semantics":
                                  "transformers 5.18.0 modeling_gpt_oss.py L251-259",
                                  "design_record": "00-feasibility/02_probe_swa_sink.py（探针，T=64 试点 acc 0.83@8k）"},
                   }, f, ensure_ascii=False, indent=2)
    print(f"[产物] {jpath}", flush=True)
    upsert_csv(args.arm, args.steps, acc)
    print(f"{args.arm}/{out_name} 完成。", flush=True)


if __name__ == "__main__":
    main()
