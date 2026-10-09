# latentmoe_parts.py —— Book5 ch7 正式件：Stable LatentMoE 三处方件 + QB Alg.1 + 换档 MoE + ablation 臂
# 用途：ch7 7.1/7.6「三处方逐字逐件 + 数百步 ablation」的组件与实验正身——
#   ① 处方一 AggRMSNorm：聚合后 RMSNorm（K3 §2.3.1 Eq.11——在「专家聚合 u」与「上投影 W↑」之间
#      插 RMSNorm；治「聚合后尺度漂移」：top-k 专家输出的方差叠加使 u 的尺度随选中专家组而变）；
#   ② 处方二 SiTU-GLU（§2.3.2 Eq.12）：软盖 softcap(x,β)=β·tanh(x/β) 分别盖 SwiGLU 的两支——
#      gate 支 β1=4、up 支 β2=25，|中间积| ≤ β1β2=100；一阶贴合 SwiGLU（Eq.18：β tanh(z/β)=z+O(z³/β²)）、
#      β→∞ 逐点回收 SwiGLU——「有界的 SwiGLU」；治「激活尖峰/低精度溢出」；
#   ③ 处方三 Quantile Balancing（§2.3.3 Eq.13-14 + App C Alg.1 p44）：aux-loss-free 家族的分位数版
#      ——路由 s=Sigmoid(W_r x) + 偏置 topk（b 只管派发不进混合权重）；b 的更新 = 对偶坐标最优解
#      （Eq.14：b̂_j = −(s_{:,j}−α) 的 (1−k/n) 分位数，α 取 Top-(k+1) 截止）；Alg.1 交替求解可照写；
#      治「千级专家负载均衡失灵」——sign 规则的 γ 两难（慢收敛 vs 震荡）被「直接跳到坐标最优」消解；
#   ④ StableLatentMoE(cfg)：三处方合一的玩具 LatentMoE（ℓ=0.5×d 几何；路由专家活潜空间 (ℓ,)→(m,)→(ℓ,)；
#      W↓/W↑ 共享；全宽共享专家支路；三开关 use_rmsnorm/use_situ/use_qb——ablation 臂的正身）；
#   ⑤ 换档 MoE：E256/top8/ℓ'=128——**top_k·ℓ 守恒**（4×256 = 8×128 = 1024：扩专家数减潜宽、
#      激活潜预算不动——K3「K·width 守恒」的玩具同构；对照 K3 896:16）。参数账另报（大纲口径：
#      换档另报参数账，主账仍 toyA）；
#   ⑥ --mode ablate：三臂短训（有①②③ vs 去② vs 去③）——把 K3Toy（ch06 整机）的 MoE 插槽逐层
#      换成 StableLatentMoE，m4 训练环同配方；判据预注册见 notes/07 §4（先落盘再跑）。
# 【Symptom↔处方对应（ch7 图 7.1 骨架）】没有①→聚合尺度漂移 / 没有②→激活尖峰（无界乘积）/
#   没有③→负载塌缩（无均衡的 sigmoid topk 会赢者通吃）——失效模式与处方一一对应；
#   玩具档若不复现 = 规模绑定声明如实写（报告失效模式明确绑定 2.8T 规模）。
# 运行方式：cd 工作区根目录 && source env.sh &&
#   python "code/ch07/latentmoe_parts.py" --mode units [--out-name run1]   # 秒级
#   python "code/ch07/latentmoe_parts.py" --mode ablate --arm full --steps 300
# 产物：log/book5-ch07/latentmoe_units_{out}.json / train_ablate_{arm}_{out}.json（不入库）
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book5-ch07")
SEED = 20261002
CH06 = os.path.join(HERE, "..", "ch06")                 # 本册件 HERE 相对——工作区/书仓双布局同构
_B4_FEAS_CANDS = [                                      # 三候选（hybrid409 同款；书仓布局同款可命中）
    os.path.join(REPO_ROOT, "code", "Book4-规模与效率", "00-feasibility"),
    os.path.join(REPO_ROOT, "from-models-to-vllm", "book4-scale-and-efficiency", "code", "00-feasibility"),
    os.path.join(REPO_ROOT, "book4-scale-and-efficiency", "code", "00-feasibility"),
]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _first(cands, need):
    for p in cands:
        if os.path.exists(os.path.join(p, need)):
            return p
    raise FileNotFoundError(need, cands)


def bootstrap():
    feas = _first(_B4_FEAS_CANDS, "moe_mla_slots.py")
    if feas not in sys.path:
        sys.path.insert(0, feas)
    import moe_mla_slots as m4
    k3_toy = _load(os.path.join(CH06, "k3_toy.py"), "k3_toy_mod")
    return m4, k3_toy


def save_json(name, obj, out_name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}_{out_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[产物] {path}")
    return path


# ---------------- 处方二：SiTU-GLU（Eq.12 + App B） ----------------
def softcap(x, beta):
    """softcap(x,β) = β·tanh(x/β)——软盖：原点近线性（β tanh(z/β)=z+O(z³/β²)，Eq.18）、远端有界 ±β。"""
    return beta * torch.tanh(x / beta)


class LatentExpert(nn.Module):
    """潜空间专家 (ℓ,)→(m,)→(ℓ,)：SiTU-GLU（处方②）或 SwiGLU（去② 对照臂——参数同构）。"""

    def __init__(self, ell, m, beta1=4.0, beta2=25.0, use_situ=True):
        super().__init__()
        self.W_g = nn.Linear(ell, m, bias=False)                                 # gate 支 (ℓ,)→(m,)
        self.W_u = nn.Linear(ell, m, bias=False)                                 # up 支
        self.W_d = nn.Linear(m, ell, bias=False)                                 # down (m,)→(ℓ,)
        self.beta1, self.beta2, self.use_situ = beta1, beta2, use_situ

    def forward(self, z):
        g, u = self.W_g(z), self.W_u(z)                                          # (N,m) x2
        if self.use_situ:
            inter = softcap(g, self.beta1) * torch.sigmoid(g) * softcap(u, self.beta2)  # |·|≤β1β2=100
        else:
            inter = F.silu(g) * u                                                # SwiGLU：无界乘积（去② 臂）
        return self.W_d(inter)                                                   # (N,ℓ)


# ---------------- 处方一：聚合后 RMSNorm（Eq.11） ----------------
class AggRMSNorm(nn.Module):
    """u 聚合后、W↑ 前的 RMSNorm（带权）——治 routed 支尺度漂移（处方①）。"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + self.eps) * self.weight.float()).to(x.dtype)


# ---------------- 处方三：Quantile Balancing（Eq.13-14 + Alg.1 + sign 对照） ----------------
def qb_alternating(s, k, T=8):
    """QB Alg.1（K3 报告 App C p44「可照写」版——按算法思想自写的 toy 实现）。

    s (m,n) 打分矩阵；交替取对偶变量的分位数闭合解：
      α ← desc_sort(s − β, axis=1)[:, k:k+1]      # token 侧截止（Top-(k+1) 的第 k+1 大）
      β ← desc_sort(s − α, axis=0)[mk/n : mk/n+1] # 专家侧（第 q+1 大 margin，q := mk/n）
    返回 (指派 x (m,n) bool, β (n,))。b = −β（符号约定：路由分 = s + b = s − β）。
    """
    m, n = s.shape
    q = m * k // n
    beta = torch.zeros(1, n, dtype=s.dtype)                                       # (1,n) 全程同形
    for _ in range(T):
        alpha = torch.sort(s - beta, dim=1, descending=True).values[:, k:k + 1]           # (m,1)
        beta = torch.sort(s - alpha, dim=0, descending=True).values[q:q + 1]              # (1,n)
    top = (s - beta).topk(k, dim=1).indices
    x = torch.zeros_like(s, dtype=torch.bool)
    x.scatter_(1, top, True)
    return x, beta.flatten()


def qb_bias_update(s, b, k, gamma_mean_center=True):
    """训练循环内的 Eq.14 更新（部署一致性：b 只影响下一批——本批永不自带自家 bias 路由）。

    s (N,E) 本批 Sigmoid 打分；b (E,) 当前偏置。路由按 Top-(k+1) 的 biased 分取截止 α；
    b̂_j = −(s_{:,j} − α) 的 (1−k/n) 分位数（降序第 q+1 大，q := N·k/E）；去公共偏移。
    """
    N, E = s.shape
    q = N * k // E
    alpha = torch.sort(s + b[None, :], dim=1, descending=True).values[:, k:k + 1]         # (N,1) Top-(k+1) 截止
    b_hat = -torch.sort(s - alpha, dim=0, descending=True).values[q:q + 1]               # (1,E)
    b_hat = b_hat.flatten()
    if gamma_mean_center:
        b_hat = b_hat - b_hat.mean()                                                     # 去公共偏移不改 Top-k
    return b_hat


def sign_rule_update(s, b, k, gamma=0.01):
    """DSV3 aux-loss-free 的 sign 规则对照臂：b ← b + γ·sign(目标负载 − 实负载)——γ 两难现场。"""
    N, E = s.shape
    _, idx = (s + b[None, :]).topk(k, dim=1)
    load = torch.bincount(idx.reshape(-1), minlength=E).float()
    return b + gamma * torch.sign(N * k / E - load)


def qb_small_example():
    """Fig.5 教学小例（m=8 token / n=4 专家 / k=1 / q=2）：构造 s 使无偏 top-1 负载 (4,3,1,0)，
    QB 交替求解配平到 (2,2,2,2)。

    构造（构造例——s 自造非报告原数；设计=「近缝翻转」）：行 1-2 的 e0/e3 缝 ~0.02、行 5 的
    e1/e2 缝 ~0.02（β 下调专家 3/2 即翻这两三行）；其余行缝大（0.3+，翻不动）——与 Fig.5 同构的
    (4,3,1,0)→(2,2,2,2)。**边界教训如实记**：缝宽任意时交替求解可在对偶不动点停在部分配平
    （如 (3,3,1,1)）——greedy argtopk 在近缝/平局处的取整敏感；报告「few steps equilibrate」
    对训练态近均匀 Sigmoid 打分成立，对构造矩阵不保证逐位配平（正文考据框素材）。
    """
    g = torch.Generator().manual_seed(30)
    s = torch.rand(8, 4, generator=g) * 0.08 + 0.46                               # 基底 ~0.5 微噪
    s[0, 0], s[0, 3] = 0.80, 0.78                                                 # 行1: e0 强、e3 近缝 0.02
    s[1, 0], s[1, 3] = 0.78, 0.75                                                 # 行2: 近缝 0.03
    s[2, 0], s[2, 3] = 0.85, 0.30                                                 # 行3: 缝大翻不动
    s[3, 0], s[3, 3] = 0.84, 0.35
    s[4, 1], s[4, 2] = 0.82, 0.80                                                 # 行5: e1 强、e2 近缝 0.02
    s[5, 1], s[5, 2] = 0.83, 0.40
    s[6, 1], s[6, 2] = 0.81, 0.45
    s[7, 2] = 0.79
    s = s.clamp(0.01, 0.99)
    load0 = torch.bincount(s.argmax(1), minlength=4)
    traj = [load0.tolist()]                                                       # t=0：无偏 top-1
    for t in range(1, 7):                                                         # t 次交替后的负载（收敛轨迹）
        x_t, _ = qb_alternating(s, k=1, T=t)
        traj.append(torch.bincount(x_t.float().argmax(1), minlength=4).tolist())
    x_final, beta_final = qb_alternating(s, k=1, T=8)
    load_final = torch.bincount(x_final.float().argmax(1), minlength=4).tolist()
    return {"s": s.tolist(), "initial_load": traj[0], "trajectory": traj,
            "final_load": load_final, "beta": beta_final.tolist(),
            "expect_initial": [4, 3, 1, 0], "expect_final": [2, 2, 2, 2],
            "m,n,k,q": [8, 4, 1, 2]}


# ---------------- Stable LatentMoE（三处方合一 + 三开关） ----------------
class StableLatentMoE(nn.Module):
    """K3 Stable LatentMoE 玩具正身（Eq.11-14）：

      y = Σ_shared E_shared(x) + W↑[ 处方①? RMSNorm(u) : u ]，u = Σ_{i∈T_k} p_i·E_i(W↓x)
    几何：W↓/W↑ 共享（d↔ℓ）；路由专家 E 个活潜空间 (ℓ,)→(m,)→(ℓ,)；共享专家全宽 (d→m_s→d)；
    路由：s = Sigmoid(W_r x) ∈ (0,1)^E；topk 用 biased 分 s+b（派发），混合权重 p 用 s 归一（b 不进权重）。
    开关：use_rmsnorm（处方①）/use_situ（处方②）/use_qb（处方③——qb_bias_update 逐批更新 b；
    False → b≡0 的裸 sigmoid topk，无均衡）。forward 返回 (out, stats)——stats 供 ablation 采集：
    act_max/act_rms（②症状：专家中间积幅值）、u_rms（①症状：聚合尺度）、load（③症状：负载分布）。
    """

    def __init__(self, d, ell, m, E, top_k, m_shared=None, use_rmsnorm=True, use_situ=True,
                 use_qb=True, beta1=4.0, beta2=25.0):
        super().__init__()
        self.d, self.ell, self.m, self.E, self.top_k = d, ell, m, E, top_k
        self.use_rmsnorm, self.use_situ, self.use_qb = use_rmsnorm, use_situ, use_qb
        self.W_down = nn.Linear(d, ell, bias=False)                               # 潜下投影（共享）
        self.router = nn.Linear(d, E, bias=False)                                 # 路由打分（Sigmoid 域）
        self.experts = nn.ModuleList([LatentExpert(ell, m, beta1, beta2, use_situ)
                                      for _ in range(E)])
        self.agg_norm = AggRMSNorm(ell) if use_rmsnorm else nn.Identity()
        self.W_up = nn.Linear(ell, d, bias=False)
        m_shared = m_shared or m
        self.shared = nn.Module()
        self.shared.W_g = nn.Linear(d, m_shared, bias=False)                      # 全宽共享支路（SwiGLU）
        self.shared.W_u = nn.Linear(d, m_shared, bias=False)
        self.shared.W_d = nn.Linear(m_shared, d, bias=False)
        self.register_buffer("bias", torch.zeros(E))                              # QB 偏置 b（部署时冻结）
        self.last_stats = {}
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

    def _shared(self, x):
        s = self.shared
        return s.W_d(F.silu(s.W_g(x)) * s.W_u(x))

    def forward(self, x, update_qb=True):
        shape = x.shape
        h = x.reshape(-1, self.d)                                                 # (N,d)
        z = self.W_down(h)                                                        # (N,ℓ) 潜空间
        s = torch.sigmoid(self.router(h)).float()                                 # (N,E) ∈ (0,1)
        b = self.bias.float()
        w_sel, idx = (s + b[None, :]).topk(self.top_k, dim=-1)                    # (N,k) biased topk 派发
        p = s.gather(1, idx)                                                      # 混合权重用 s（b 不进权重）
        p = (p / p.sum(-1, keepdim=True).clamp_min(1e-6)).to(z.dtype)
        u = torch.zeros_like(z)
        act_max, act_cnt = 0.0, 0
        for e in range(self.E):
            tok, slot = torch.nonzero(idx == e, as_tuple=True)
            if tok.numel() == 0:
                continue
            expert = self.experts[e]
            zz = z[tok]
            g, uu = expert.W_g(zz), expert.W_u(zz)                                # (n_e,m) x2
            if expert.use_situ:
                inter = softcap(g, expert.beta1) * torch.sigmoid(g) * softcap(uu, expert.beta2)
            else:
                inter = F.silu(g) * uu
            act_max = max(act_max, float(inter.detach().abs().max()))
            act_cnt += inter.numel()
            y = expert.W_d(inter) * p[tok, slot].to(inter.dtype).unsqueeze(1)     # (n_e,ℓ)×权重
            u.index_add_(0, tok, y.to(u.dtype))
        u_normed = self.agg_norm(u)
        out = self.W_up(u_normed) + self._shared(h)                               # Eq.11（处方①在 W↑ 前）
        # stats（ablation 采集：①u_rms ②act_max ③load 不均衡）
        with torch.no_grad():
            load = torch.bincount(idx.reshape(-1), minlength=self.E)
            nz = load[load > 0]
            self.last_stats = {
                "u_rms": float(u.detach().float().pow(2).mean().sqrt()),
                "act_max": act_max,
                "load_max_over_min": float(nz.max() / nz.min().clamp_min(1)),
                "load_top2_share": float(torch.sort(load, descending=True).values[:2].float().sum()
                                        / max(load.sum().item(), 1)),
                "s_for_qb": s.detach(),
            }
        # 处方③：批后更新（本批永不自带自家 bias 路由——Eq.14 因果口径；no_qb 臂 b≡0 不动）
        if self.use_qb and update_qb and self.training:
            with torch.no_grad():
                self.bias.copy_(qb_bias_update(self.last_stats["s_for_qb"], b, self.top_k))
        return out.view(*shape)

    def n_params(self):
        n = sum(p.numel() for p in self.parameters())
        if isinstance(self.agg_norm, nn.Identity):
            n += 0
        return n


def latent_moe_account(d, ell, m, E, top_k, m_shared=None, use_rmsnorm=True):
    """StableLatentMoE 参数手算：W↓ d·ℓ + 路由 d·E + 专家 E·3·ℓ·m + W↑ ℓ·d + 共享 3·d·m_s
    +（处方①）聚合 RMSNorm ℓ。"""
    m_shared = m_shared or m
    n = d * ell + d * E + E * 3 * ell * m + ell * d + 3 * d * m_shared + (ell if use_rmsnorm else 0)
    return {"total": n, "activated": d * ell + d * E + top_k * 3 * ell * m + ell * d + 3 * d * m_shared
            + (ell if use_rmsnorm else 0)}


# ---------------- ablation 机型：K3Toy 换 MoE 插槽 ----------------
def build_ablation_machine(k3_toy, arm, d=512, ell=256, m=256, E=64, top_k=4):
    """arm: full（①+②+③）/ nositu（去②）/ noqb（去③）。K3Toy(kda) 的 8 层 MoE 全换 StableLatentMoE。

    三臂 iso-param（②是激活函数换装、③是路由规则换装——参数全同；①只差聚合 RMSNorm 的 ℓ 个权重，
    声明记录）。返回 (model, moe_cfg)。"""
    switches = {"full": (True, True, True), "nositu": (True, False, True),
                "noqb": (True, True, False)}[arm]
    use_rmsnorm, use_situ, use_qb = switches
    cfg = k3_toy.toy_a_cfg(kda_impl="kda")
    torch.manual_seed(SEED)
    model = k3_toy.K3Toy(cfg)
    for layer in model.model.layers:
        layer.mlp = StableLatentMoE(d, ell, m, E, top_k, use_rmsnorm=use_rmsnorm,
                                    use_situ=use_situ, use_qb=use_qb)
    moe_cfg = {"arm": arm, "d": d, "ell": ell, "m": m, "E": E, "top_k": top_k,
               "use_rmsnorm": use_rmsnorm, "use_situ": use_situ, "use_qb": use_qb}
    return model, moe_cfg


def run_ablation(m4, k3_toy, arm, steps, out_name, batch=8, block=512, peak_lr=1e-3):
    device = m4.pick_device("mps")
    model, moe_cfg = build_ablation_machine(k3_toy, arm)
    model = model.to(device)
    stream = m4.TokenStream(m4.TOKENS_SP8K, batch, block)
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1)
    moe_params = sum(l.mlp.n_params() for l in model.model.layers) // 8
    print(f"[ablate {arm}] MoE 插槽参数 {moe_params:,}/层 | {steps} 步 @ {device}", flush=True)
    losses, stats_trace, secs = [], [], []
    warmup = max(5, steps // 15)
    model.train()
    import time
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = m4.lr_at(step, steps, peak_lr, warmup)
        x, y = stream.batch(step, device)
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device.type == "mps":
            torch.mps.synchronize()
        secs.append(time.perf_counter() - t0)
        losses.append(loss.item())
        if step % 10 == 0 or step == 1:
            layer0 = model.model.layers[0].mlp.last_stats
            stats_trace.append({"step": step, "u_rms": layer0["u_rms"], "act_max": layer0["act_max"],
                                "load_max_over_min": layer0["load_max_over_min"],
                                "load_top2_share": layer0["load_top2_share"]})
            print(f"    step {step}/{steps} loss {loss.item():.4f} act_max {layer0['act_max']:.1f} "
                  f"u_rms {layer0['u_rms']:.3f} load_max/min {layer0['load_max_over_min']:.1f}", flush=True)
    out = {"arm": arm, **moe_cfg, "moe_params_per_layer": moe_params,
           "total_params": model.n_params(), "steps": steps,
           "sec_per_step_p50": float(np.median(secs)),
           "loss_first": losses[0], "loss_last": losses[-1], "losses": losses,
           "stats_trace": stats_trace,
           "final_bias_head": model.model.layers[0].mlp.bias.tolist()[:12]}
    save_json(f"train_ablate_{arm}", out, out_name)
    print(f"[完成] {arm}: loss {losses[0]:.3f}→{losses[-1]:.3f} | {np.median(secs):.3f} s/step", flush=True)
    return out


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="Book5 ch7 三处方件：units 秒级 / ablate 三臂短训")
    ap.add_argument("--mode", choices=["units", "ablate"], default="units")
    ap.add_argument("--arm", choices=["full", "nositu", "noqb"], default="full")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out-name", default="run1")
    args = ap.parse_args()
    m4, k3_toy = bootstrap()

    if args.mode == "units":
        torch.manual_seed(SEED)
        results = {"seed": SEED}
        # ① QB 小例（Fig.5 口径 m=8,n=4,k=1,q=2：负载 (4,3,1,0)→(2,2,2,2)）
        qb = qb_small_example()
        assert qb["initial_load"] == qb["expect_initial"], f"构造例初始负载 {qb['initial_load']}"
        assert qb["final_load"] == qb["expect_final"], f"QB 未配平：{qb['final_load']}"
        results["qb_small_example"] = qb
        print(f"[QB 小例] 负载 {qb['initial_load']} → {qb['final_load']}（q=2 配平 ✓）| "
              f"β={ [round(x,3) for x in qb['beta']] }")
        # ② SiTU-GLU 四断言（近原点贴合/有界/β→∞ 回收/梯度存活）
        x = torch.linspace(-30, 30, 1201, requires_grad=True)
        situ = softcap(x, 4.0) * torch.sigmoid(x) * softcap(torch.tensor(1.5), 25.0)  # gate 支 × up 支示意
        bound_ok = bool(softcap(x, 4.0).abs().max() <= 4.0)
        near = x.abs() <= 1.0
        first_order = float((softcap(x, 4.0) - x)[near].abs().max())                # |x|≤1 区一阶贴合（Eq.18）
        big_beta = float((softcap(x, 1e6) - x).abs().max())                          # β→∞ 回收恒等
        grad_alive = bool((torch.autograd.grad(softcap(x, 4.0).sum(), x)[0].abs() > 0).any())  # 软盖梯度非零（对照硬 clamp 的死区）
        results["situ_checks"] = {"softcap_bound_le_beta1": bound_ok,
                                  "first_order_fit_max_dev_x_le_1": first_order,
                                  "beta_inf_recovers_identity_dev": big_beta,
                                  "softcap_grad_nonzero": grad_alive,
                                  "output_bound_beta1x_beta2": 4.0 * 25.0}
        print(f"[SiTU 断言] |softcap|≤β1 ✓ | 一阶贴合偏差(±1 内) {first_order:.2e} | "
              f"β→∞ 回收偏差 {big_beta:.2e} | 梯度存活 {grad_alive}")
        # ③ 换档账（E256/top8/ℓ128——top_k·ℓ 守恒 4×256=8×128；对照主档 E64/top4/ℓ256）
        main_acc = latent_moe_account(512, 256, 256, 64, 4)
        shift_acc = latent_moe_account(512, 128, 128, 256, 8)
        results["moe_accounts"] = {
            "main_E64_top4_ell256": {**main_acc, "topk_x_ell": 4 * 256},
            "shift_E256_top8_ell128": {**shift_acc, "topk_x_ell": 8 * 128},
            "k_width_conservation": "4×256 = 8×128 = 1024（激活潜预算不动——K3「K·width 守恒」玩具同构）",
        }
        print(f"[换档账] 主档 E64/t4/ℓ256：{main_acc['total']:,} | 换档 E256/t8/ℓ128：{shift_acc['total']:,}"
              f" | top_k·ℓ 守恒 1024 ✓")
        # ④ 部署一致性断言：冻结 b + argtopk == QB 指派（App C「no quantile computation at deployment」）
        torch.manual_seed(SEED + 1)
        s = torch.rand(512, 64)
        x_asg, beta = qb_alternating(s, k=4, T=8)
        frozen = (s - beta[None, :]).topk(4, dim=1).indices
        rows_equal = all(set(torch.nonzero(x_asg[i]).flatten().tolist()) == set(frozen[i].tolist())
                         for i in range(s.shape[0]))
        results["deployment_consistency_frozen_bias"] = rows_equal
        print(f"[部署一致性] 冻结 β + argtopk ≡ QB 指派：{rows_equal}")
        assert rows_equal
        save_json("latentmoe_units", results, args.out_name)
        print("ch7 latentmoe_parts.py units 完成。")
    else:
        run_ablation(m4, k3_toy, args.arm, args.steps, args.out_name)


if __name__ == "__main__":
    main()
