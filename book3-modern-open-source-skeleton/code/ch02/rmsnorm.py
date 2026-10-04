# rmsnorm.py —— Book3 ch02 教学重写件：RMSNorm 的手写实现、性质验证与双对拍
# 用途：①给出可 import 的 RMSNorm(hidden_size, eps)（插槽 import 契约，ch10 整机将 import 本件）；
#       ②四个性质探针（平移/缩放不变性、零均值下与 LayerNorm 恒等、RMS^2 = sigma^2 + mu^2 手算算例）；
#       ③双对拍：与 00-feasibility/llama_slots.RMSNorm（组件库正身）CPU fp32 逐位一致 +
#         与 HF transformers 5.18.0 LlamaRMSNorm 逐位一致（eps 1e-6/1e-5 两档——对拍隐形开关）。
# 所属章节：Book3 第 2 章 §2.3（公式）/§2.5（改装第一例）/§2.7（eps 代际）
# 运行方式：cd code/ch02 && python rmsnorm.py     （CPU fp32，秒级，无命令行参数）
# 产物：全部打印到 stdout（正文引用的自测数字来源）；不写文件、不落 log。
# 依赖链：llama_slots.py（组件库正身）+ transformers 5.18.0（仅对照）；torch 2.13。
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "00-feasibility")))
from llama_slots import RMSNorm as SlotRMSNorm            # noqa: E402  组件库正身（对拍锚 1）

SEED = 20261002                                            # 全书统一种子


# ---------------- 教学重写件（与 llama_slots.RMSNorm 逐位一致；注释按讲解顺序展开） ----------------
class RMSNorm(nn.Module):
    """RMSNorm：只重缩放、不重中心化的层归一化。输入 (B,n,C) -> 输出 (B,n,C)。

    计算图（对应正文图 2.1 右半）：
        x (B,n,C)
          -> 平方 (B,n,C) -> 末维均值 (B,n,1)          # 均方（二阶矩），不减均值
          -> + eps -> rsqrt (B,n,1)                    # 分母 = 1/sqrt(mean(x^2)+eps)
          -> 逐元素除 (B,n,C)
          -> * weight(g) (C,) 广播                    # 唯一可学参数，无 bias
    数值纪律（对齐官方 Meta 实现与 HF LlamaRMSNorm）：统计量在 fp32 中计算，
    结束后转回输入 dtype 再乘 g——bf16 下 mean(x^2) 的累加误差是真实灾难，不是洁癖。
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))   # (C,) 增益 g，初始全 1
        self.variance_epsilon = eps                           # 只加在方差上：rsqrt(mean(x^2)+eps)

    def forward(self, x):
        dtype = x.dtype                                   # 记住原 dtype
        x32 = x.float()                                   # (B,n,C) 上抛 fp32（统计量精度）
        ms = x32.pow(2).mean(-1, keepdim=True)            # (B,n,C)->(B,n,1) 均方 = RMS^2
        x32 = x32 * torch.rsqrt(ms + self.variance_epsilon)   # (B,n,C)*(B,n,1)->(B,n,C) 重缩放
        return self.weight * x32.to(dtype)                # (C,)*(B,n,C)->(B,n,C)，回原 dtype 后乘 g


# ---------------- 性质探针（正文 §2.3 的「验证」步：直觉 -> 公式 -> 数字） ----------------
def probe_properties():
    print("=" * 72)
    print("探针 1-4：RMSNorm 四条性质（CPU fp32，种子 20261002）")
    print("=" * 72)
    torch.manual_seed(SEED)
    B, n, C = 2, 5, 8                                     # 小数字，肉眼可核
    x = torch.randn(B, n, C)                              # (2,5,8)
    ln = nn.LayerNorm(C, eps=1e-6, elementwise_affine=True)
    ln.bias.data.zero_()                                  # LN 置 beta=0，与 RMSNorm 只差算法本身
    rms = RMSNorm(C, eps=1e-6)

    # 探针 1：平移不变性——LN 有（re-centering），RMSNorm 没有（这正是砍掉的项）
    c = 3.0                                               # 平移量
    d_ln = (ln(x + c) - ln(x)).abs().max().item()
    d_rms = (rms(x + c) - rms(x)).abs().max().item()
    print(f"[1] 平移 +{c}:  max|LN(x+c)-LN(x)|   = {d_ln:.2e}  （=0：LN 对平移不敏感）")
    print(f"              max|RMS(x+c)-RMS(x)| = {d_rms:.3e}  （>0：RMSNorm 放弃了 re-centering）")

    # 探针 2：缩放不变性——两者都有（re-scaling）：整体乘 alpha 后输出不变
    alpha = 2.5
    d_ln_s = (ln(alpha * x) - ln(x)).abs().max().item()
    d_rms_s = (rms(alpha * x) - rms(x)).abs().max().item()
    print(f"[2] 缩放 x{alpha}:  max|LN(a·x)-LN(x)|   = {d_ln_s:.2e}")
    print(f"              max|RMS(a·x)-RMS(x)| = {d_rms_s:.2e}  （都 ~0：RMS 线性性 RMS(ax)=a·RMS(x)）")

    # 探针 3：零均值输入下 LN == RMSNorm（论文原话的数值版）
    x0 = x - x.mean(-1, keepdim=True)                     # (2,5,8) 强制每格 mu=0
    d_eq = (ln(x0) - rms(x0)).abs().max().item()
    print(f"[3] 零均值:    max|LN(x0)-RMS(x0)|  = {d_eq:.2e}  （mu=0 时 sigma=RMS，两式合流）")

    # 探针 4：恒等式 RMS^2 = sigma^2 + mu^2（逐格成立——「总幅度 = 漂移 + 波动」的代数面）
    ms = x.pow(2).mean(-1)                                # (2,5) RMS^2
    mu = x.mean(-1)                                       # (2,5)
    var = x.var(-1, unbiased=False)                       # (2,5) sigma^2（总体方差）
    gap = (ms - (var + mu ** 2)).abs().max().item()
    print(f"[4] 恒等式:    max|RMS^2-(sigma^2+mu^2)| = {gap:.2e}  （B*n 个格子全部成立）")

    # 手算算例（正文小数字例子）：a = [1,2,3,6]
    a = torch.tensor([1.0, 2.0, 3.0, 6.0])
    mu_a = a.mean()
    var_a = a.var(unbiased=False)
    ms_a = a.pow(2).mean()
    print(f"[例] a=[1,2,3,6]:  mu={mu_a:.1f}  sigma^2={var_a:.2f}  RMS^2={ms_a:.2f}"
          f"  ->  sigma^2+mu^2={var_a + mu_a ** 2:.2f}（={ms_a:.2f} ✓）")
    print(f"    LN(a)   = {torch.nn.functional.layer_norm(a, (4,), eps=1e-6).tolist()}")
    print(f"    RMS(a)  = {(a / torch.sqrt(ms_a + 1e-6)).tolist()}")
    a0 = a - mu_a
    print(f"    a-mu    = {a0.tolist()}（零均值化后 RMS^2={a0.pow(2).mean():.2f}=sigma^2，LN 与 RMS 合流）")


# ---------------- 双对拍（正文 §2.5 改装的验收；eps 两档=§2.7 的隐形开关） ----------------
def probe_parity():
    print()
    print("=" * 72)
    print("对拍：手写 RMSNorm vs llama_slots（正身） vs HF LlamaRMSNorm（CPU fp32 逐位）")
    print("=" * 72)
    from transformers.models.llama.modeling_llama import LlamaRMSNorm   # 5.18.0（对照锚 2）
    torch.manual_seed(SEED)
    B, n, C = 4, 64, 512
    x = torch.randn(B, n, C)                              # (4,64,512)
    for eps in (1e-6, 1e-5):                              # HF 默认 1e-6 / LLaMA2/3 全系 1e-5
        ours = RMSNorm(C, eps=eps)
        slot = SlotRMSNorm(C, eps=eps)
        hf = LlamaRMSNorm(C, eps=eps)
        w = torch.randn(C)                                # 同一份增益（默认全 1 测不出权重通路）
        for m in (ours, slot, hf):
            m.weight.data.copy_(w)
        with torch.no_grad():
            y_ours, y_slot, y_hf = ours(x), slot(x), hf(x)
        e1 = (y_ours - y_slot).abs().max().item()
        e2 = (y_ours - y_hf).abs().max().item()
        print(f"eps={eps:.0e}:  max|ours-slot| = {e1:.3e}  max|ours-HF| = {e2:.3e} "
              f" 逐位一致={torch.equal(y_ours, y_slot) and torch.equal(y_ours, y_hf)}")

    # bf16 不上抛 fp32 的反例（官方 .float() 纪律的理由；只演示、不进对拍口径）
    x16 = torch.randn(64, 256, dtype=torch.bfloat16)
    ms16 = x16.pow(2).mean(-1, keepdim=True)              # bf16 直接算均方
    ms32 = x16.float().pow(2).mean(-1, keepdim=True)       # 上抛 fp32 再算（官方路径）
    rel = ((ms16.float() - ms32).abs() / ms32).max().item()
    print(f"[bf16 反例] 均方统计 bf16 直算 vs fp32 上抛：最大相对偏差 {rel:.2e}"
          f"（rsqrt 会放大末位误差 -> 官方实现一律 .float()）")


if __name__ == "__main__":
    probe_properties()
    probe_parity()
