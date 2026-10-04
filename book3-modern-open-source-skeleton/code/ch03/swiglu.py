# swiglu.py —— Book3 ch3 教学重写件：门控前馈 SwiGLU-MLP（gate/up/down 三矩阵）
# 用途：ch3.5「改装第二例」的换装件正身。教学重写 = 不复制 llama_slots.SwiGLU 的代码，
#       而是按式 (3.4) 从零写一遍，再过两道对拍验收：
#       ① 与 00-feasibility/llama_slots.py 的 SwiGLU（组件库正身）CPU fp32 逐位一致；
#       ② 与 HF transformers 5.18.0 的 LlamaMLP 同 config 同权重 state_dict 直搬对拍
#         （HF 键位 gate_proj/up_proj/down_proj，strict=True 零缺零余）。
#       自测第三件：LLaMA1/2/3 七档 d_ff 用官方公式复算对账（表 3.2 的复算列）。
# 所属章节：Book3 ch3（3.4 参数账 / 3.5 改装第二例）；ch10 整机 import 本件作 MLP 插槽
#       （插槽 import 契约：ch03/swiglu.py::SwiGLUMLP(cfg)——类名/签名/HF 键位承 llama_slots 正身）。
# 运行方式：cd code/ch03 && python swiglu.py   （全 CPU 秒级，无命令行参数）
# 依赖：sys.path 两级 bootstrap（00-feasibility 的 llama_slots）；transformers 仅在对拍段 import。
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "00-feasibility")))
from llama_slots import LLaMAConfig, SwiGLU   # noqa: E402  组件库正身（对拍对象①）


class SwiGLUMLP(nn.Module):
    """门控前馈（式 (3.4)）：down_proj( silu(gate_proj(x)) ⊙ up_proj(x) )。

    输入 (B,n,C) -> 输出 (B,n,C)；d_ff = intermediate_size（8C/3 口径的落地，见 ch3.4）。
    三投影全部无 bias（Shazeer 2020 "Following the T5 codebase" 的选择，LLaMA 沿用）；
    命名对齐 HF LlamaMLP——state_dict 键 gate_proj.weight / up_proj.weight / down_proj.weight
    可与 HF / llama_slots strict 直搬（本文件的对拍②即验收这件事）。
    cfg 只读 hidden_size 与 intermediate_size 两个字段（LLaMAConfig 与 HF LlamaConfig 通用）；
    激活固定 silu（SiLU = Swish_beta 取 beta=1，torch 的 F.silu；HF config 的 hidden_act="silu"
    是同一件事的字符串口径——「读得出 silu、读不出 SwiGLU」的命名考据见正文 3.5）。
    """

    def __init__(self, cfg):
        super().__init__()
        C, d_ff = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(C, d_ff, bias=False)    # (B,n,C) -> (B,n,d_ff) 门支路：过 SiLU 后作门控系数
        self.up_proj = nn.Linear(C, d_ff, bias=False)      # (B,n,C) -> (B,n,d_ff) 值支路：候选内容，保持线性
        self.down_proj = nn.Linear(d_ff, C, bias=False)    # (B,n,d_ff) -> (B,n,C)  压回残差流宽度

    def forward(self, x):
        gate = self.gate_proj(x)          # (B,n,C) -> (B,n,d_ff)
        up = self.up_proj(x)              # (B,n,C) -> (B,n,d_ff)
        gated = F.silu(gate) * up         # (B,n,d_ff) * (B,n,d_ff) -> (B,n,d_ff)：SiLU 逐元素当门
        return self.down_proj(gated)      # (B,n,d_ff) -> (B,n,C)


def llama_d_ff(dim, multiple_of=256, ffn_dim_multiplier=None):
    """Meta 官方 d_ff 确定公式（meta-llama/llama model.py L331-335 逐行复刻，ch3.4 表 3.2 用）。

    初值 4*dim -> 折 2/3 成 8d/3 -> （L2-70B 起）乘 ffn_dim_multiplier -> 向上取整到 multiple_of
    的倍数（multiple_of=256 的注释原文："make SwiGLU hidden layer size multiple of large power of 2"）。
    """
    hidden_dim = int(2 * (4 * dim) / 3)                       # 4d 折 2/3 = 8d/3
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)     # L2-70B/L3 起为 1.3（社区引述口径，见正文）
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)   # 向上取整到 multiple_of 倍数


def check_seven_configs():
    """表 3.2 七档对账：官方公式复算 vs HF config 实测 intermediate_size。

    config 实测值来源：LLaMA-1 官方 params.json（社区转录+huggyllama 镜像双源）、Llama-2/3
    NousResearch 镜像 config.json（等级 {config 逆向}；L2-70B 的 multiple_of=4096 与 mult=1.3
    为社区引述官方 params.json——正文红线：引用须注明）。
    """
    rows = [  # (型号, d, multiplier, multiple_of, config 实测 d_ff)
        ("LLaMA-1 7B", 4096, None, 256, 11008),
        ("LLaMA-1 13B", 5120, None, 256, 13824),
        ("LLaMA-1 33B", 6656, None, 256, 17920),
        ("LLaMA-1 65B", 8192, None, 256, 22016),
        ("Llama-2 70B", 8192, 1.3, 4096, 28672),
        ("Llama-3 8B", 4096, 1.3, 256, 14336),
        ("Llama-3 70B", 8192, 1.3, 4096, 28672),
    ]
    print("== 表 3.2 七档 d_ff 复算对账（公式 mo*ceil(int(int(8d/3)*mult)/mo)）==")
    ok = 0
    for name, d, mult, mo, cfg_ff in rows:
        rec = llama_d_ff(d, multiple_of=mo, ffn_dim_multiplier=mult)
        hit = "OK" if rec == cfg_ff else "MISMATCH"
        ok += rec == cfg_ff
        print(f"  {name:12s} d={d:5d} mult={mult} mo={mo:4d} 复算={rec:6d} config={cfg_ff:6d} "
              f"比率={cfg_ff/d:.4f} {hit}")
    assert ok == len(rows), f"七档对账 {ok}/{len(rows)}——公式与 config 不符，正文表 3.2 不能引用"
    print(f"  七档全部复算吻合（{ok}/{len(rows)}）——「8/3 取整到 256 倍数」精确复现 LLaMA1 全系与 mult 档\n")
    return rows


def main():
    torch.manual_seed(20261002)          # 全书统一种子
    cfg = LLaMAConfig(vocab_size=512, hidden_size=768, intermediate_size=2048,   # d=768：8/3d=2048 整除零误差档
                      num_hidden_layers=2, num_attention_heads=12)
    B, n, C, d_ff = 4, 16, cfg.hidden_size, cfg.intermediate_size

    # ---- 对拍①：教学重写件 vs llama_slots 正身（CPU fp32 逐位一致） ----
    ours, ref = SwiGLUMLP(cfg), SwiGLU(cfg)
    missing, unexpected = ours.load_state_dict(ref.state_dict(), strict=True), None
    x = torch.randn(B, n, C)                              # (4,16,768)
    with torch.no_grad():
        y_ours, y_ref = ours(x), ref(x)                   # 各 (4,16,768)
    d1 = (y_ours - y_ref).abs().max().item()
    print(f"== 对拍① vs llama_slots.SwiGLU（CPU fp32，strict 直搬）==")
    print(f"  state_dict 键：{sorted(ours.state_dict())}（strict=True 零缺零余）")
    print(f"  max|Δ| = {d1:.3e}（逐位一致判据 =0）")
    assert d1 == 0.0, "与 llama_slots 正身不一致——教学重写件不得有任何数值分叉"

    # ---- 对拍②：vs HF transformers 5.18.0 LlamaMLP（同 config 同权重 strict 直搬） ----
    from transformers import LlamaConfig as HFLlamaConfig
    from transformers.models.llama.modeling_llama import LlamaMLP
    hf = LlamaMLP(HFLlamaConfig(hidden_size=C, intermediate_size=d_ff, hidden_act="silu"))
    hf.load_state_dict(ours.state_dict(), strict=True)    # HF 键 gate_proj/up_proj/down_proj 同名直搬
    with torch.no_grad():
        y_hf = hf(x)                                      # (4,16,768)
    d2 = (y_ours - y_hf).abs().max().item()
    agree = (y_ours.argmax(-1) == y_hf.argmax(-1)).float().mean().item()
    print(f"== 对拍② vs HF LlamaMLP（transformers 5.18.0，CPU fp32）==")
    print(f"  max|Δ| = {d2:.3e}；argmax 一致率 = {agree:.4f}（同一运算的两种实现只差浮点结合序）")
    assert d2 < 1e-5 and agree == 1.0

    # ---- 参数账（ch3.4 的手算在代码里的对应物） ----
    p_gpt2, p_swiglu = 2 * C * 4 * C, 3 * C * d_ff        # GPT-2 两矩阵 vs SwiGLU 三矩阵
    print(f"== 参数账（d=768 档）==")
    print(f"  GELU-4C 两矩阵 2·C·4C = {p_gpt2:,}；SwiGLU-8/3C 三矩阵 3·C·(8/3·C) = {p_swiglu:,}"
          f"（d_ff=2048 整除，零误差对齐 8C²）")
    assert p_gpt2 == p_swiglu == 8 * C ** 2

    # ---- 表 3.2 七档复算 ----
    check_seven_configs()
    print("swiglu.py 自测全部通过（对拍①逐位一致 / 对拍② HF 同构 / 参数账 / 七档 d_ff 对账）")


if __name__ == "__main__":
    main()
