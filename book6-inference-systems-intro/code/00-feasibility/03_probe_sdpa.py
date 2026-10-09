# 03_probe_sdpa.py —— Book6 本机探针 P1：F.scaled_dot_product_attention 在 NPU bf16 的可用性
#                        与后端枚举（ch8 attention 后端章的前置事实）
# 用途：
#   ① 后端名枚举：torch.nn.attention.SDPBackend 全成员名单（torch 2.10 本机实测）；
#   ② 逐后端强制：sdpa_kernel([B]) 在 npu/bf16 下逐个试跑（flash/math/mem-efficient/cudnn/...），
#      成功/失败皆如实记录——后端门控在 NPU 上是否生效是本探针要回答的事实；
#   ③ 数值对拍：npu/bf16 各跑法 vs CPU/fp32 math 参照（同输入），max|Δ| 落表；
#   ④ 默认分发（不设上下文）跑通性 + 对拍——「裸调 sdpa 在 NPU 上能不能用」的一手答案。
# 形状口径（ch8 前置，尺寸流转）：q,k,v (B,h,n,d_k) -> o (B,h,n,d_k)，两组 d_k（64/128）×
#   两种 mask（无 mask / is_causal=True），softmax(S/√d_k)V，S=(B,h,n,n)。
# 运行：cd <workspace> && source env.sh && npu-smi info（挑空闲卡）后
#      ASCEND_RT_VISIBLE_DEVICES=<卡号> python code/Book6-推理系统导论/00-feasibility/03_probe_sdpa.py
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import torch_npu  # noqa: F401 —— import 即注册 npu 后端

DEV = "npu"
CASES = [(2, 16, 512, 64), (2, 16, 512, 128)]     # (B, h, n, d_k)：d_k=64 与 128 各一档
MASKS = [False, True]                              # 无 mask / is_causal


def make_qkv(B, h, n, dk, device, dtype, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    qkv = [torch.randn(B, h, n, dk, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
           for _ in range(3)]                      # (B,h,n,dk) x3
    return qkv


def ref_cpu(q, k, v, causal):
    """CPU fp32 math 参照（强制 math 后端，确定性）。"""
    with sdpa_kernel([SDPBackend.MATH]):
        return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal)


def run_one(backend_list, q, k, v, causal):
    """给定后端（None=默认分发）跑一次 sdpa，返回 (ok, out_or_errmsg)。"""
    try:
        if backend_list is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)   # (B,h,n,dk)
        else:
            with sdpa_kernel(backend_list):
                out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return True, out
    except Exception as e:                                        # 失败也是事实，如实记录
        return False, f"{type(e).__name__}: {str(e)[:140]}"


if __name__ == "__main__":
    print("=" * 78)
    print("[P1 sdpa 可用性探针] torch", torch.__version__, "| torch_npu", torch_npu.__version__,
          "| device:", torch.npu.get_device_name(0))
    print("=" * 78)

    # ① 后端名枚举（torch 2.10 本机实测名单）
    members = list(SDPBackend.__members__)
    print("\n[① 后端枚举] SDPBackend 成员（torch %s）：%s" % (torch.__version__, members))
    probe_backends = [getattr(SDPBackend, b) for b in
                      ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH", "CUDNN_ATTENTION")
                      if hasattr(SDPBackend, b)]
    if hasattr(SDPBackend, "FLEX_ATTENTION"):
        probe_backends.append(SDPBackend.FLEX_ATTENTION)          # 若有 flex 枚举也一并试
    # 注：本机实测 SDPBackend 不可下标取值（SDPBackend["MATH"] 抛 TypeError）——枚举名单可读、
    # 成员经属性访问取得；这本身也是一条 torch 2.10+torch_npu 组合的事实，记录在案。

    # ②+③ 逐后端强制 × 默认分发：跑通性 + 与 CPU fp32 参照的 max|Δ| + 与默认分发的逐位一致性
    #    （逐位一致 = 同一 kernel 路径的强证据——门控在 NPU 上是否形同虚设的关键判定）
    print("\n[②③ 后端强制矩阵]  npu/bf16 强制单后端（失败=该门控下无可用 kernel，如实记录）")
    hdr = "%-22s %14s %10s %14s %16s %s" % ("backend", "case", "causal", "max|Δ|vsCPU", "逐位==默认?", "备注")
    print(hdr)
    for B, h, n, dk in CASES:
        for causal in MASKS:
            q, k, v = make_qkv(B, h, n, dk, DEV, torch.bfloat16, seed=20261009)
            qc, kc, vc = q.float().cpu(), k.float().cpu(), v.float().cpu()
            ref = ref_cpu(qc, kc, vc, causal)                     # (B,h,n,dk) fp32
            case = f"B{B}h{h}n{n}d{dk}"
            ok0, out0 = run_one(None, q, k, v, causal)            # 默认分发参照输出
            assert ok0
            rows = [(None, "默认分发(无上下文)")] + [(b, b.name) for b in probe_backends]
            for be, label in rows:
                ok, res = run_one([be] if be is not None else None, q, k, v, causal)
                if ok:
                    d = (res.float().cpu() - ref).abs().max().item()
                    bit_eq = torch.equal(res, out0)               # 与默认分发逐位比对
                    eq_str = "是" if be is None else ("是(同kernel)" if bit_eq else "否")
                    print("%-22s %14s %10s %14.3e %16s %14s" % (label, case, causal, d, eq_str, "跑通"))
                else:
                    print("%-22s %14s %10s %14s %16s %14s"
                          % (label, case, causal, "-", "-", "失败:" + res[:60]))
            del q, k, v, qc, kc, vc, ref, out0

    # ④ 小结占位（结论文字在 notes/04 定版；此处打印机器可读的判定）
    print("\n[④ 判定] 默认分发与逐后端结果见上表；数值对拍容差以 bf16 量级（~1e-2）为参考。")

    # ⑤ NPU 侧 attention API 面：torch_npu 私有 flash attention 算子是否在位（ch8/vllm-ascend 关联）
    #    torch_npu 2.10 无 torch_npu.nn 包（本机实测 ModuleNotFoundError）——算子走 torch.ops.npu.*
    #    与 torch_npu.npu.nn.functional.*（旧路径）两条，逐一探测。
    print("[⑤ API面] torch.ops.npu.npu_fusion_attention: %s"
          % ("在位" if hasattr(torch.ops.npu, "npu_fusion_attention") else "不在位"))
    print("[⑤ API面] torch.ops.npu.npu_scaled_dot_product_attention: %s"
          % ("在位" if hasattr(torch.ops.npu, "npu_scaled_dot_product_attention") else "不在位"))
    try:
        from torch_npu.npu.nn.functional import npu_fusion_attention  # noqa: F401
        print("[⑤ API面] torch_npu.npu.nn.functional.npu_fusion_attention: 在位")
    except ImportError as e:
        print("[⑤ API面] torch_npu.npu.nn.functional.npu_fusion_attention: 不在位（%s）" % e)
