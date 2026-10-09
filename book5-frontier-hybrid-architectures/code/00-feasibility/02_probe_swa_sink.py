# 02_probe_swa_sink.py —— Book5 ch2 实验可行性探针：滑窗注意力 + attention sink + lost-in-the-middle mini
# 用途：ch2（路线一 SWA 与 attention sink）的两件正身实验在本机（Apple M5 Pro 48GB, MPS）是否成立——
#   ① 机制件：banded causal mask 滑窗注意力 + 每头可学 sink 标量（gpt-oss 语义：sink 作为附加注意力
#      logit 参与 softmax、无 value、概率质量被丢弃——对齐 transformers 5.18.0 modeling_gpt_oss.py
#      L251-259：cat([attn_weights, sinks]) → softmax → probs[..., :-1] → @V）；逐运算形状注释；
#   ② 可训性：full vs 滑窗+sink 同初始化小档 200 步（sp-8k 语料）——「窗口这么窄也不塌」的对照起点；
#   ③ lost-in-the-middle mini 复现：合成 needle-recall 任务，needle 位置扫描（首/中/尾 + 窗沿），
#      三臂对照：full / 滑窗 / 滑窗+sink。
# 【判据预注册】（跑前写定，防事后择优；负结论照报）：
#   A. full 臂：所有 needle 位置 recall acc ≥ 0.7（6000 步 + cosine 内学会任意位置召回）；
#   B. 滑窗臂（w=32, T=96, cliff p≥62）：可达位置（62/74）acc ≥ 0.7 而不可达位置（18/42/58）
#      acc ≈ 随机（≤0.1）——「位置可达性悬崖」；
#   C. 滑窗+sink 臂：不可达位置 acc 与滑窗臂差 < 0.1（sink 不传递远端内容——sink 存的是注意力
#      质量锚，不是远端记忆）；可达位置 acc 不低于滑窗臂 0.1 以上（sink 不损害可达范围内召回）。
#   若 C 出现「sink 使不可达位置 acc 升高 ≥0.1」则推翻预注册（如实报告并追查）。
# 【试点记录（run1-run3 + 四轮交互试点，防重复踩坑）】①单 needle 稀疏监督 200 步学不会
#   （loss 停随机基线）②绑定监督（预测位置 p+1 自己的 token）退化为自拷贝捷径——皆弃；
#   ③MQAR 化后信号成立但 4L/rope/600 步仍欠训（full 4.16→2.09 仍下降、swa 全钉在随机）；
#   ④两化简定版：L=2（归纳头最小电路，24ms/step）+ 无 rope（隔离 mask 变量）+ 8000 步
#   试点 full 臂 acc 0.83（远端均匀）；⑤T=64 尾档 0.67/0.0 为 needle 撞查询区伪影——T=96
#   且 needle 上界 78 排除。Part A（sp-8k 可训性）run1 起即有效：full eval 6.608 /
#   滑窗+sink eval 6.525（T=512 w=128，小档短上下文两者相当——如实报，不渲染差异）。
# 所属章节：Book5 第 2 章（机制唯一深讲处的实验底座）；02/07 的 needle 生成器与 MiniLM 骨架被 07 复用。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/00-feasibility/02_probe_swa_sink.py" [--out-name run1]
#   全程 MPS bf16 autocast（数值对拍口径：本探针无跨实现互证，训练口径即可）；预计 3-4 分钟。
# 产物：log/book5-feasibility/probe02_swa_sink_{out}.json（不入库）
import argparse
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
OUT_DIR = os.path.join(REPO_ROOT, "log", "book5-feasibility")
SEED = 20261002

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


def save_json(name, obj, out_name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}_{out_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[产物] {path}")
    return path


# ---------------- 位置件：自建 rope（rotate_half 约定，llama 系同构） ----------------
def rope_cache(n, d_k, theta, device):
    """cos/sin：(1,1,n,d_k)。freqs=outer(pos, inv_freq) 后 cat(freqs,freqs)——与 (i, i+d/2) 配对自洽。"""
    inv = 1.0 / (theta ** (torch.arange(0, d_k // 2, dtype=torch.float32, device=device) / (d_k // 2)))
    ang = torch.outer(torch.arange(n, dtype=torch.float32, device=device), inv)   # (n, d_k/2)
    freqs = torch.cat((ang, ang), dim=-1)                                        # (n, d_k)
    return freqs.cos()[None, None], freqs.sin()[None, None]


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]                # (…,d_k/2) x2
    return torch.cat((-x2, x1), dim=-1)                                          # (…,d_k)


def apply_rope(x, cos, sin):
    """x (B,h,n,d_k) · cos/sin (1,1,n,d_k) → (B,h,n,d_k)。"""
    return x * cos + rotate_half(x) * sin


# ---------------- 注意力插槽：full / 滑窗(+sink)。forward(x, cos, sin) → (B,n,d) ----------------
def band_causal_mask(n, window, device):
    """banded causal mask：(n,n) bool，True=可看。window=None → 全因果；否则 j ≤ i 且 i-j < window。"""
    i = torch.arange(n, device=device)[:, None]                                  # (n,1)
    j = torch.arange(n, device=device)[None, :]                                  # (1,n)
    m = j <= i
    if window is not None:
        m = m & (i - j < window)                                                 # 滑窗带
    return m


class FullAttn(nn.Module):
    """满注意力（对照组）：GQA 四投影 + 可选 rope（needle 任务关掉——隔离 mask 变量）+ causal。"""

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
        v = self.v_proj(x).view(B, n, self.h_kv, self.d_k).transpose(1, 2)       # (B,h_kv,n,d_k)
        if self.use_rope:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.h // self.h_kv
        k = k.repeat_interleave(rep, dim=1)                                      # (B,h,n,d_k) GQA 展开
        v = v.repeat_interleave(rep, dim=1)                                      # (B,h,n,d_k)
        att = (q @ k.transpose(-2, -1)) * self.scale                             # (B,h,n,n)
        att = att.masked_fill(~band_causal_mask(n, None, x.device), float("-inf"))
        p = F.softmax(att.float(), dim=-1).to(x.dtype)                           # fp32 softmax
        y = (p @ v).transpose(1, 2).contiguous().view(B, n, self.h * self.d_k)   # (B,n,h·d_k)
        return self.o_proj(y)                                                    # (B,n,d)


class SwaSinkAttn(nn.Module):
    """滑窗注意力 + 可选 attention sink（gpt-oss 语义，见文件头）。

    sink = 每头一个可学标量，作为第 n+1 个注意力 logit 拼进 softmax；该虚拟位没有 value，
    其概率质量在加权前被丢弃——数学上 = 分母多一项 exp(s_h)，输出仍是 Σ p_j v_j。
    """

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
            self.sinks = nn.Parameter(torch.zeros(n_head))                       # 每头 1 个标量（gpt-oss 同形）

    def forward(self, x, cos, sin):
        B, n, d = x.shape
        q = self.q_proj(x).view(B, n, self.h, self.d_k).transpose(1, 2)          # (B,h,n,d_k)
        k = self.k_proj(x).view(B, n, self.h_kv, self.d_k).transpose(1, 2)       # (B,h_kv,n,d_k)
        v = self.v_proj(x).view(B, n, self.h_kv, self.d_k).transpose(1, 2)       # (B,h_kv,n,d_k)
        if self.use_rope:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.h // self.h_kv
        k = k.repeat_interleave(rep, dim=1)                                      # (B,h,n,d_k)
        v = v.repeat_interleave(rep, dim=1)                                      # (B,h,n,d_k)
        att = (q @ k.transpose(-2, -1)) * self.scale                             # (B,h,n,n)
        att = att.masked_fill(~band_causal_mask(n, self.window, x.device), float("-inf"))
        if self.use_sink:
            sink = self.sinks.view(1, self.h, 1, 1).expand(B, self.h, n, 1)      # (B,h,n,1) 虚拟 sink logit
            att = torch.cat([att, sink], dim=-1)                                 # (B,h,n,n+1)
            att = att - att.max(dim=-1, keepdim=True).values                     # 数值稳定（HF 同款）
            p = F.softmax(att.float(), dim=-1)[..., :n].to(x.dtype)              # 丢 sink 概率质量 (B,h,n,n)
        else:
            p = F.softmax(att.float(), dim=-1).to(x.dtype)                       # (B,h,n,n)
        y = (p @ v).transpose(1, 2).contiguous().view(B, n, self.h * self.d_k)   # (B,n,h·d_k)
        return self.o_proj(y)                                                    # (B,n,d)


# ---------------- MiniLM 骨架：可插注意力插槽的三插槽 Block（Book2/3 拓扑，自包含版） ----------------
class MiniLM(nn.Module):
    """x = x + attn(norm1(x)); x = x + swiglu(norm2(x))——残差河恒宽。
    attn_modules 逐层传入（full / 滑窗 / 混排皆可）——07 探针复用本骨架做三臂消融。"""

    def __init__(self, vocab=8192, d=256, n_layer=4, d_ff=704, eps=1e-5, attn_modules=None):
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
            return logits, None                                                  # 无目标：探针内部口径
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               targets.reshape(-1))
        return logits, loss


# ---------------- 合成 needle-recall 任务（MQAR 式；07 探针同款复用） ----------------
# 词表分区：filler∈[0,4000) / 值∈[4000,4064)（64 个）/ needle 键 K=8099 / 干扰键∈[8100,8180)。
# 设计定版 v4（run1-3 + 四轮试点教训，见文件头试点记录）：
#   [0,16) 8 个干扰 (k_i,v_i) 对；needle 对 (K,v) 在受控位置 p；尾部 [T-16,T) 8 个
#   (查询键, 应答) 位（应答在流内紧随查询——因果上查询位尚未看见，必须从对区/needle 检索——
#   MQAR 标准口径）；监督只放 8 个查询位。q8=K（T-2），a8=v（T-1）。
#   默认几何 T=96 / w=32 / L=2（归纳头最小电路）/ 无 rope（隔离 mask 变量——位置调制不进小任务）。
#   needle 训练分布：50% 远端 [16,60) + 50% 窗内 [62,78)（三臂同分布保公平；查询区在 [80,96)，
#   needle 上界 78 保证不与查询区碰撞——v3 试点 0.67/0.0 尾档即碰撞伪影，已排除）。
#   可达性：query 在 T-2=94，w=32 → needle 可达 iff p+1 ≥ 94-31 = 63，即 p ≥ 62（cliff=62）。
NEEDLE_K = 8099
N_PAIRS, N_QUERIES = 8, 8


def make_needle_batch(B, T, rng, p_min=None, device="cpu"):
    """p_min 给定时钉死 needle 位置（eval 扫描用）；否则 50/50 远端/窗内混合（训练用）。
    返回 (x, p)。值空间 64——随机 CE 基线 ln(64)=4.16。"""
    x = torch.from_numpy(rng.integers(0, 4000, size=(B, T)).astype(np.int64))    # filler (B,T)
    dk = rng.integers(8100, 8180, size=(B, N_PAIRS))                            # 干扰键 (B,8)
    dv = rng.integers(4000, 4064, size=(B, N_PAIRS))                            # 干扰值 (B,8)
    v = torch.from_numpy(rng.integers(4000, 4064, size=(B,)).astype(np.int64))  # needle 值
    x[:, 0:16:2] = torch.from_numpy(dk.astype(np.int64))                        # 对区键
    x[:, 1:16:2] = torch.from_numpy(dv.astype(np.int64))                        # 对区值
    if p_min is None:                                                           # 训练：50/50 混合
        far = rng.integers(16, 60, size=B)
        near = rng.integers(62, 78, size=B)
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
    logits, _ = model(x)                                                         # (B,T,V)
    qpos = torch.arange(T - 16, T, 2, device=x.device)                          # (8,) 查询位
    return F.cross_entropy(logits[:, qpos].float().reshape(-1, logits.size(-1)),
                           x[:, qpos + 1].reshape(-1))


@torch.no_grad()
def needle_eval(model, T, positions, n_seq=60, seed=SEED + 999, device="cpu"):
    """needle 位置扫描：各位置 acc = argmax(logits[:, T-2]) == x[:, T-1]（needle 应答）的比例。"""
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


def train_mini(model, steps, batch_fn, device, lr=1e-3, warmup=20, tag="", cosine=False):
    """共用短训环（AdamW 0.9/0.95 wd0.1 + clip1.0 + warmup[+cosine]；MPS bf16 autocast）——Book4 口径。"""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    model.train()
    secs, losses = [], []
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
        if step % 50 == 0:
            print(f"    [{tag}] step {step}/{steps} loss {loss.item():.4f}", flush=True)
    return {"sec_per_step_p50": float(np.median(secs)), "loss_first": losses[0],
            "loss_last_20_mean": float(np.mean(losses[-20:]))}


class SpBatch:
    """sp-8k 语料批（TokenStream 确定性顺序批）。"""
    def __init__(self, m4, batch, block):
        self.stream = m4.TokenStream(m4.TOKENS_SP8K, batch, block)
        self.device = m4.pick_device("mps")
    def __call__(self, step):
        return self.stream.batch(step, self.device)
    def loss(self, model, x, y):
        logits, _ = model(x, y)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))


class NeedleBatch:
    """合成 needle 批（位置随机化——训练分布覆盖全序列）。"""
    def __init__(self, B, T, seed, device):
        self.rng, self.B, self.T, self.device = np.random.default_rng(seed), B, T, device
    def __call__(self, step):
        x, p = make_needle_batch(self.B, self.T, self.rng, device=self.device)
        return (x, p), None
    def loss(self, model, batch, _):
        (x, p) = batch
        return needle_loss(model, x, p, self.T)


def build_arm(kind, cfg, state_dict=None):
    """三臂构建：full / swa / swa_sink。同初始化：先建 full 取 state_dict，再灌入两滑窗臂
    （形状全同；sink 是滑窗臂独有参数，置零初始化——见文件头 gpt-oss 语义）。"""
    d, L, h, h_kv, d_k, d_ff = cfg["d"], cfg["L"], cfg["h"], cfg["h_kv"], cfg["d_k"], cfg["d_ff"]
    w, use_rope = cfg.get("window"), cfg.get("use_rope", True)
    attns = [FullAttn(d, h, h_kv, d_k, use_rope) if kind == "full"
             else SwaSinkAttn(d, h, h_kv, d_k, w, use_sink=(kind == "swa_sink"), use_rope=use_rope)
             for _ in range(L)]
    m = MiniLM(d=d, n_layer=L, d_ff=d_ff, attn_modules=attns)
    if state_dict is not None:
        missing, unexpected = m.load_state_dict(state_dict, strict=False)
        assert not unexpected, unexpected
        assert all("sinks" in k for k in missing), missing                        # 只允许 sink 缺省（零初始化）
    return m


def main():
    ap = argparse.ArgumentParser(description="Book5 ch2 滑窗+sink 可行性探针")
    ap.add_argument("--out-name", default="run1")
    ap.add_argument("--steps", type=int, default=6000, help="Part B 训练步数（fast=6000 / full=10000）")
    args = ap.parse_args()
    m4, hits = bootstrap_book4()
    device = m4.pick_device("mps")
    print(f"[bootstrap] moe_mla_slots 自：{hits[0]} | device {device}")
    torch.manual_seed(SEED)

    cfg = dict(d=256, L=4, h=8, h_kv=4, d_k=64, d_ff=704, window=128)
    T = 512

    # ---- Part A：sp-8k 可训性（full vs 滑窗+sink，同初始化 200 步） ----
    print("[Part A] sp-8k 可训性：full vs 滑窗+sink（w=128）各 200 步")
    torch.manual_seed(SEED)
    full0 = build_arm("full", cfg)
    sd = full0.state_dict()
    partA = {}
    for kind in ("full", "swa_sink"):
        torch.manual_seed(SEED)
        m = build_arm(kind, cfg, sd).to(device)
        n_par = sum(p.numel() for p in m.parameters())
        bf = SpBatch(m4, 8, T)
        res = train_mini(m, 200, bf, device, tag=f"A-{kind}")
        ev = m4.eval_loss(m, device)                                             # sp-8k heldout
        partA[kind] = {"params": n_par, **res, "eval_loss": ev}
        print(f"  [{kind}] {n_par/1e6:.1f}M | train {res['loss_first']:.3f}→{res['loss_last_20_mean']:.3f} "
              f"| eval {ev:.4f} | {res['sec_per_step_p50']:.3f} s/step")
        del m

    # ---- Part B：needle 位置扫描三臂（MQAR v4 定版：T=96 / w=32 / L=2 / 无 rope / cosine） ----
    print("[Part B] needle-recall：full / 滑窗 / 滑窗+sink 三臂（T=96, w=32, L=2, 6000 步 fast 档）")
    Tb, steps = 96, args.steps
    cfgB = dict(d=256, L=2, h=8, h_kv=4, d_k=64, d_ff=704, window=32, use_rope=False)
    cliff = Tb - 2 - (cfgB["window"] - 1) - 1                                    # 可达边界：p ≥ cliff（=62）
    positions = [18, 42, 58, 62, 74]                                             # 全部避开查询区 [80,96)
    print(f"  [预注册] 可达边界 p ≥ {cliff}（query 在 T-2=94，窗口 {cfgB['window']}）")
    sdB = build_arm("full", cfgB).state_dict()
    partB = {"T": Tb, "steps": steps, "reach_cliff": cliff, "positions": positions, "arms": {}}
    for kind in ("full", "swa", "swa_sink"):
        torch.manual_seed(SEED)
        m = build_arm(kind, cfgB, sdB).to(device)
        bf = NeedleBatch(32, Tb, SEED + 7, device)
        res = train_mini(m, steps, bf, device, lr=3e-3, warmup=50, cosine=True, tag=f"B-{kind}")
        acc = needle_eval(m, Tb, positions, device=device)
        partB["arms"][kind] = {"train": res, "recall_acc_by_pos": acc}
        print(f"  [{kind}] train {res['loss_first']:.3f}→{res['loss_last_20_mean']:.3f} | "
              f"acc@pos { {k: round(v,2) for k,v in acc.items()} }")
        del m

    # ---- 判据预注册结算 ----
    verdict = {}
    full_acc = partB["arms"]["full"]["recall_acc_by_pos"]
    swa_acc = partB["arms"]["swa"]["recall_acc_by_pos"]
    sink_acc = partB["arms"]["swa_sink"]["recall_acc_by_pos"]
    verdict["A_full_learns_all_positions"] = all(v >= 0.8 for v in full_acc.values())
    verdict["B_swa_cliff"] = all((swa_acc[p] >= 0.8) == (p >= cliff) for p in positions)
    unreachable = [p for p in positions if p < cliff]
    verdict["C_sink_no_remote_rescue"] = (max(sink_acc[p] - swa_acc[p] for p in unreachable) < 0.1
                                          if unreachable else None)
    verdict["C_sink_no_window_harm"] = all(sink_acc[p] >= swa_acc[p] - 0.1
                                           for p in positions if p >= cliff)
    print(f"[判据结算] {verdict}")

    save_json("probe02_swa_sink", {
        "seed": SEED, "date": "2026-10-05", "cfgA": cfg, "T_A": T, "cfgB": cfgB,
        "partA_sp8k_trainability": partA, "partB_needle_scan": partB,
        "prereg_verdict": verdict,
        "references": {"gpt_oss_sink_semantics": "transformers 5.18.0 modeling_gpt_oss.py L251-259"},
    }, args.out_name)
    print("探针 02 完成。")


if __name__ == "__main__":
    main()
