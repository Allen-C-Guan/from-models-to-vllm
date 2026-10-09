# qat_sim.py —— Book5 ch7 正式件：MXFP4/MXFP8 STE 训练循环嵌入（QAT vs PTQ 双臂）
# 用途：ch7 7.2/7.6「MXFP4 QAT 从 SFT 起覆盖全后训练」的本机模拟正身——
#   ① 分区清单照 K3 §4.1.4 口径（B2）：**只压路由专家**——专家权重 MXFP4（STE）+ 专家输入激活
#      MXFP8（STE）；W↓/W↑ 等潜投影/router/共享专家/注意力/lm_head 全高精度（玩具 fp32/bf16 autocast）。
#      「分区本身就是实验的一问」——什么量什么不量是 K3 工程事实，不是我们拍脑袋；
#   ② 双臂谱系对照（ch7 表 7.2 的产物源）：
#      --arm qat  = K3 谱系：从第 1 步起前向走模拟量化（STE 反传恒等）——「SFT 起量化感知微调」
#                   的玩具模拟（数百步短训里，量化噪声全程在场）；
#      --arm ptq  = gpt-oss 谱系：训练全程高精度，训完一次性套量化评估（post-trained quantization
#                   ——Book4 ch7 已核 gpt-oss「post-trained … quantization」无 QAT 字样）；
#      读数：ptq 臂报「量化前后 eval loss 跳变」；qat 臂报「量化噪声下的 loss 曲线」；
#      官方掉点数字未披露（papers/05 未核实项 #4）——玩具结果无论方向都是
#      「本册自产数据 + 官方无数字」的诚实组合；
#   ③ MXFP8 格式声明：报告未指名 E4M3/E5M2（papers/05 未核实项 #5）——本件选 **E4M3**
#      （块 32 + E8M0 共享指数，与 MXFP4 同族；选择在此声明，正文考据框引用）。
# 【STE 口径（承 Book4 ch07）】前向 w_q = mxfp4_rt(w)（量化-反量化 round-trip），反传恒等：
#   w + (w_q − w).detach()——优化器更新原权重；激活同理。复用 Book4 fp8_sim.mxfp4_rt（块 32
#   E8M0）与 fake_fp8_quant（E4M3 格点）；MXFP8 的 E8M0 块 scale 本件自实现（fp8_sim 只有
#   per-tensor/实数 scale 的 FP8 路径）。
# 运行方式：cd 工作区根目录 && source env.sh &&
#   python "code/ch07/qat_sim.py" --arm qat --steps 300 --out-name fast
# 产物：log/book5-ch07/qat_sim_{arm}_{out}.json（不入库）
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
_B4_CH07_CANDS = [
    os.path.join(REPO_ROOT, "code", "Book4-规模与效率", "ch07"),
    os.path.join(REPO_ROOT, "from-models-to-vllm", "book4-scale-and-efficiency", "code", "ch07"),
    os.path.join(REPO_ROOT, "book4-scale-and-efficiency", "code", "ch07"),
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
    fp8 = _load(os.path.join(_first(_B4_CH07_CANDS, "fp8_sim.py"), "fp8_sim.py"), "fp8_sim")
    k3_toy = _load(os.path.join(CH06, "k3_toy.py"), "k3_toy_mod")
    return m4, fp8, k3_toy


def save_json(name, obj, out_name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}_{out_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[产物] {path}")
    return path


# ---------------- MXFP8 激活模拟（E4M3 + E8M0 块 scale——本件自实现） ----------------
def mxfp8_rt(x, block=32, fp8=None, e4m3_max=448.0):
    """MXFP8 块模拟：最后维每 block=32 元素一块、共享 E8M0 纯指数 scale（2 的幂，保证块内元素
    ≤E4M3 上限 448）；块内元素 round 到 E4M3 格点（fp8_sim.fake_fp8_quant）。量化-反量化 round-trip。"""
    x2 = x.reshape(-1, block).float()
    amax = x2.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    s = torch.exp2(torch.ceil(torch.log2(amax / e4m3_max)))                     # E8M0：2 的幂
    y = fp8.fake_fp8_quant(x2 / s, fp8.E4M3)                                    # E4M3 格点（声明见文件头）
    return (y * s).reshape(x.shape).to(x.dtype)


# ---------------- STE 件（权重 MXFP4 + 专家输入 MXFP8） ----------------
class MXFP8ActSte(torch.autograd.Function):
    """激活 STE：前向 mxfp8_rt、反传恒等。"""

    @staticmethod
    def forward(ctx, x, block, fp8):
        return mxfp8_rt(x, block=block, fp8=fp8)

    @staticmethod
    def backward(ctx, g):
        return g, None, None


def mxfp8_act(x, block=32, fp8=None):
    return MXFP8ActSte.apply(x, block, fp8)


class QuantExpertInput(nn.Module):
    """专家输入的 MXFP8 件：训练态走 STE（前向量化、反传恒等），eval 态直接量化-反量化。"""

    def __init__(self, block=32, fp8=None):
        super().__init__()
        self.block, self.fp8 = block, fp8

    def forward(self, x):
        if torch.is_grad_enabled():
            return mxfp8_act(x, self.block, self.fp8)
        return mxfp8_rt(x, self.block, self.fp8)


def _attach_input_quant(e, fp8):
    """给单个专家挂输入量化：注册 in_quant 子模块（随 .to()/.train() 搬运）+ 以实例方法遮蔽
    ExpertSwiGLU.forward（x 先过 in_quant 再进原前向）。"""
    import types
    e.in_quant = QuantExpertInput(fp8=fp8)
    orig = e.forward
    e._orig_forward = orig

    def forward(self, x, _orig=orig, _e=e):
        return _orig(_e.in_quant(x))
    e.forward = types.MethodType(forward, e)                                     # 实例方法遮蔽（带 self 签名）


def apply_qat_partition(model, k3_toy, fp8, weights=True, acts=True):
    """K3 分区落地：路由专家权重 → MXFP4Linear（k3_toy 件，键位不变）；路由专家输入 → MXFP8 STE。

    返回换装计数。weights/acts 开关留作分区消融（正文一问：只量权重 vs 权重+激活）。"""
    n_w, n_a = 0, 0
    for layer in model.model.layers:
        mlp = layer.mlp
        if not hasattr(mlp, "experts") or not isinstance(mlp.experts, (nn.ModuleList, list)):
            continue
        for e in mlp.experts:
            if weights:
                e.gate_proj = k3_toy.MXFP4Linear(e.gate_proj, fp8=fp8)
                e.up_proj = k3_toy.MXFP4Linear(e.up_proj, fp8=fp8)
                e.down_proj = k3_toy.MXFP4Linear(e.down_proj, fp8=fp8)
                n_w += 3
            if acts:
                _attach_input_quant(e, fp8)
                n_a += 1
    return {"weights_swapped": n_w, "expert_inputs_quantized": n_a}


def eval_loss(model, stream, device, n_batches=4):
    """eval 口径：no_grad；取流尾未训段（max_step 往回数 n_batches 批——训练只消费过前段）。"""
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(n_batches):
            x, y = stream.batch(stream.max_step - i, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "mps" \
                    else torch.autocast("cpu", enabled=False):
                _, loss = model(x, y)
            losses.append(loss.item())
    if was_training:
        model.train()
    return float(np.mean(losses))


def run_arm(m4, fp8, k3_toy, arm, steps, out_name, batch=8, block=512, peak_lr=1e-3):
    device = m4.pick_device("mps")
    cfg = k3_toy.toy_a_cfg(kda_impl="kda")
    torch.manual_seed(SEED)
    model = k3_toy.K3Toy(cfg)
    part = apply_qat_partition(model, k3_toy, fp8,
                               weights=(arm == "qat"), acts=(arm == "qat"))
    assert model.n_params() == k3_toy.TOY_A_ASSERTS["kda"]["total"], "量化钩子不得改参数账"
    model = model.to(device)
    print(f"[{arm} 臂] {steps} 步 @ {device} | 分区 {part}（ptq 臂训练全程高精度）", flush=True)
    stream = m4.TokenStream(m4.TOKENS_SP8K, batch, block)
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1)
    warmup = max(5, steps // 15)
    import time
    secs, losses = [], []
    model.train()
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
        if step % max(1, steps // 10) == 0:
            print(f"    step {step}/{steps} loss {loss.item():.4f}", flush=True)
    # eval 双读数：高精度路径 vs 量化路径（PTQ 臂=训后量化的跳变；QAT 臂=量化在场的收敛水平）
    out = {"arm": arm, "steps": steps, "partition": part, "seed": SEED,
           "sec_per_step_p50": float(np.median(secs)),
           "loss_first": losses[0], "loss_last": losses[-1], "losses": losses}
    if arm == "ptq":
        out["eval_loss_clean"] = eval_loss(model, stream, device)
        # 训后一次性套量化：权重 MXFP4 + 输入 MXFP8（同一分区，只是从评估时才生效）
        part2 = apply_qat_partition(model, k3_toy, fp8, weights=True, acts=True)
        out["eval_loss_quantized"] = eval_loss(model, stream, device)
        out["ptq_jump"] = out["eval_loss_quantized"] - out["eval_loss_clean"]
        print(f"[PTQ 读数] clean {out['eval_loss_clean']:.4f} → quantized "
              f"{out['eval_loss_quantized']:.4f}（跳变 {out['ptq_jump']:+.4f}）", flush=True)
    else:
        out["eval_loss_qat"] = eval_loss(model, stream, device)
        print(f"[QAT 读数] 量化在场 eval loss {out['eval_loss_qat']:.4f}", flush=True)
    save_json(f"qat_sim_{arm}", out, out_name)
    print(f"[完成] {arm}: loss {losses[0]:.3f}→{losses[-1]:.3f} | {np.median(secs):.3f} s/step", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="Book5 ch7 MXFP4/MXFP8 QAT vs PTQ 双臂")
    ap.add_argument("--arm", choices=["qat", "ptq"], required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out-name", default="fast")
    args = ap.parse_args()
    m4, fp8, k3_toy = bootstrap()
    run_arm(m4, fp8, k3_toy, args.arm, args.steps, args.out_name)


if __name__ == "__main__":
    main()
