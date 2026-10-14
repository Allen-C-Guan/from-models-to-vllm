# 用途：Book8 ch08 微探针——隔离 fp8 装载失败的算子模式。忠实复刻 fp8_block.py#L86-97
#       resolve_block_scales 的写入语句 `resolved[rs:re] = weight[rs:re].to(fp32)*row_scales`
#       在四种变体下的 NPU 行为（toy 尺寸/真实层尺寸/非原地写法/CPU 解块再上卡），
#       定位「561103 边界」到底住在哪一步——是数学不可行、还是特定算子路径不可行。
# 所属章节：code/Book8-vLLM源码剖析下/ch08/（notes/18 §五；正文 8.8 的机制隔离件）
# 运行方式：source env.sh && ASCEND_RT_VISIBLE_DEVICES=6 python exp_ch08_fp8_resolve_probe.py

import json
import os
import traceback

import torch
import torch_npu  # noqa: F401

DEV = "npu"
BLOCK_N, BLOCK_K = 128, 128
ROWS_PER_STEP = 1024 // BLOCK_N * BLOCK_N  # =1024（与 fp8_block.py#L86 同式）


def resolve_like_engine(out_f: int, in_f: int, inplace: bool, on_cpu: bool):
    """与 fp8_block.py#L86-97 同构的解块路径；inplace=True 复刻原行，False 为变体写法。"""
    dev = "cpu" if on_cpu else DEV
    g = torch.Generator(device="cpu").manual_seed(8)
    w = (torch.randn(out_f, in_f, generator=g) * 0.1).to(torch.float8_e4m3fn).to(dev)
    s = (torch.rand(out_f // BLOCK_N, in_f // BLOCK_K, generator=g) + 0.5).to(
        device=dev, dtype=torch.float32)
    resolved = torch.empty((out_f, in_f), dtype=torch.bfloat16,
                           device="cpu" if on_cpu else DEV)
    for rs in range(0, out_f, ROWS_PER_STEP):
        re = min(rs + ROWS_PER_STEP, out_f)
        row_scales = s[rs // BLOCK_N: (re + BLOCK_N - 1) // BLOCK_N].to(
            device=w.device, dtype=torch.float32)
        row_scales = row_scales.repeat_interleave(BLOCK_N, dim=0)[: re - rs]
        row_scales = row_scales.repeat_interleave(BLOCK_K, dim=1)[:, :in_f]
        if inplace:
            resolved[rs:re] = w[rs:re].to(torch.float32) * row_scales.to(DEV)
        else:
            resolved[rs:re] = (w[rs:re].to(torch.float32) * row_scales).to(torch.bfloat16)
    return resolved.to(DEV) if on_cpu else resolved


def decompose(out_f: int, in_f: int):
    """把 #L96 那一行拆成四个原子步，逐步定位 561103 住在哪个算子里。"""
    g = torch.Generator(device="cpu").manual_seed(8)
    w = (torch.randn(out_f, in_f, generator=g) * 0.1).to(torch.float8_e4m3fn).to(DEV)
    s = (torch.rand(1, 4, generator=g) + 0.5).to(DEV, torch.float32)
    steps = {}
    try:
        w32 = w[:256].to(torch.float32); steps["1_e4m3_to_fp32_slice"] = "OK"
    except Exception as e:  # noqa: BLE001
        steps["1_e4m3_to_fp32_slice"] = f"{type(e).__name__}: {str(e)[:120]}"
    try:
        rs = s.repeat_interleave(BLOCK_N, 0)[:256].repeat_interleave(BLOCK_K, 1)[:, :in_f]
        prod = w32 * rs; steps["2_fp32_mul_broadcast"] = f"OK {tuple(prod.shape)}"
    except Exception as e:  # noqa: BLE001
        steps["2_fp32_mul_broadcast"] = f"{type(e).__name__}: {str(e)[:120]}"
    try:
        resolved = torch.empty((out_f, in_f), dtype=torch.bfloat16, device=DEV)
        resolved[0:256] = prod; steps["3_slice_assign_fp32_into_bf16"] = "OK"
    except Exception as e:  # noqa: BLE001
        steps["3_slice_assign_fp32_into_bf16"] = f"{type(e).__name__}: {str(e)[:120]}"
    try:
        prod64 = prod.to(torch.bfloat16)
        resolved[0:256] = prod64; steps["4_slice_assign_bf16_into_bf16"] = "OK"
    except Exception as e:  # noqa: BLE001
        steps["4_slice_assign_bf16_into_bf16"] = f"{type(e).__name__}: {str(e)[:120]}"
    return steps


def main() -> None:
    cases = [
        ("A_toy_256x512_inplace", dict(out_f=256, in_f=512, inplace=True, on_cpu=False)),
        ("B_layer_12288x2048_inplace", dict(out_f=12288, in_f=2048, inplace=True, on_cpu=False)),
        ("C_layer_12288x2048_noninplace", dict(out_f=12288, in_f=2048, inplace=False, on_cpu=False)),
        ("D_layer_12288x2048_cpu_dequant", dict(out_f=12288, in_f=2048, inplace=False, on_cpu=True)),
    ]
    rec = {"exp": "book8-ch08 fp8 resolve_block_scales operator isolation probe",
           "hardware": "Atlas 910B3, torch 2.10.0 + torch-npu 2.10.0.post4 (CANN 9.1.0)",
           "replicates": "fp8_block.py#L86-97 (rows_per_step=1024, block 128x128, e4m3->fp32*scale -> bf16 slice-assign)",
           "cases": {}}
    for name, kw in cases:
        try:
            r = resolve_like_engine(**kw)
            rec["cases"][name] = {"ok": True,
                                  "dtype": str(r.dtype), "shape": list(r.shape),
                                  "sample": [round(float(r[0, 0]), 4), round(float(r[-1, -1]), 4)]}
        except Exception as e:  # noqa: BLE001
            rec["cases"][name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        print("[probe]", name, "->", "OK" if rec["cases"][name]["ok"] else rec["cases"][name]["error"][:120])

    rec["decompose_12288x2048"] = decompose(12288, 2048)
    for k, v in rec["decompose_12288x2048"].items():
        print("[probe:decompose]", k, "->", v[:110])

    # 三镜头 B-6 补档：转换拷贝的三个单算子变体（此前仅 Bash 内联跑过，未落 JSON）
    g = torch.Generator(device="cpu").manual_seed(8)
    wv = (torch.randn(256, 512, generator=g) * 0.1).to(torch.float8_e4m3fn).to(DEV)
    variants = {}
    for name, fn in [
        ("whole_to_fp32", lambda: wv.to(torch.float32)),
        ("whole_to_bf16", lambda: wv.to(torch.bfloat16)),
        ("slice_to_fp32", lambda: wv[:64].to(torch.float32)),
        ("cpu_to_fp32", lambda: wv.cpu().to(torch.float32)),
    ]:
        try:
            r = fn(); variants[name] = f"OK {tuple(r.shape)} {r.device}"
        except Exception as e:  # noqa: BLE001
            variants[name] = f"FAIL {type(e).__name__}: {str(e)[:90]}"
        print("[probe:variant]", name, "->", variants[name][:100])
    rec["single_op_variants_256x512"] = variants

    out = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "..", "..", "log", "book8-ch08",
                                       "exp_ch08_fp8_resolve_probe.json"))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("[probe] wrote", out)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
