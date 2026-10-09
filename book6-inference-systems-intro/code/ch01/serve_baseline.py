# serve_baseline.py —— Book6 ch1 代码件：服务前基线的延迟分解（「没有引擎的生活」正身测量）
# 用途：无 KV cache 的逐 token 全量重算——逐步手算 FLOPs/bytes 与 NPU 实测对账，
#   给 ch1（两阶段与 roofline）与 ch2（「没有引擎的价格」70-140 倍账）提供第一手读数：
#   ① 加载 Book3 终态 207M（消费线 ckpt，NPU 重训版）；② 贪心逐 token 生成，
#      每步 = 全量重算前向（O(n²) 注意力 + O(n) 逐层）；③ fp32 / bf16 双口径；
#   ④ 扫描 prompt 长 n0 ∈ {24, 128, 512}，各生成 16 token，报每 token 均时延与增长斜率。
# 所属章节：Book6 ch1 §1.5（服务前基线实测）；设计书=plan/Book6-推理系统导论.md ch1 节列表
# 运行：cd <workspace> && source env.sh && npu-smi info 挑空闲卡后
#      ASCEND_RT_VISIBLE_DEVICES=<卡号> python code/Book6-推理系统导论/ch01/serve_baseline.py [--smoke]
# 产物：log/book6-ch01/serve_baseline[_smoke].json（--smoke 产物分文件，防覆写）
import argparse
import importlib.util
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(ROOT, "log", "book6-ch01")
CKPT = os.path.join(ROOT, "log", "book3-ch10", "ckpt_integrated_npu1.pt")

# 三候选 bootstrap（工作区 / 书仓嵌套 / 书仓裸布局——Book3/4/5 已验证模式）
_L215_CANDS = [
    os.path.join(ROOT, "code", "Book3-现代开源骨架", "ch10", "llama215.py"),
    os.path.join(ROOT, "from-models-to-vllm", "book3-modern-open-source-skeleton", "code", "ch10", "llama215.py"),
    os.path.join(ROOT, "from-models-to-vllm", "code", "ch10", "llama215.py"),
]

# 手算常数（与 llama215.py cand1 config 对齐；运行时 assert 逐位）
N_PARAMS = 207_119_360          # 总参（含 untied lm_head）
N_ACT = 207_119_360             # 稠密模型：激活参=总参（MoE 才有分叉——ch2 展开）
D_MODEL = 1024
V = 32000


def _load_llama215():
    for c in _L215_CANDS:
        if os.path.isfile(c):
            spec = importlib.util.spec_from_file_location("llama215", c)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod, c
    raise FileNotFoundError("llama215.py 三候选均未命中")


def flops_forward(n: int) -> int:
    """一次前向（n token 全量重算）的 FLOPs 手算（matmul 乘加账：FLOPs≈2×激活参×n）。

    精确拆法（教学口径，ch1/ch2 展开逐层版）：每 token 过全部参数一次 ≈ 2N；
    n token 并行前向 = 2N·n。注意力分数/softmax 等非 matmul 项略计（<5%，正文注明）。
    """
    return 2 * N_ACT * n


def bytes_weights(dtype_bytes: int) -> int:
    """一次前向至少要把全部权重从 HBM 搬一遍（无 KV cache、无批处理：权重不摊薄）。"""
    return N_PARAMS * dtype_bytes


def main() -> None:
    ap = argparse.ArgumentParser(description="无 KV cache 服务前基线：手算 vs NPU 实测对账")
    ap.add_argument("--smoke", action="store_true", help="冒烟档：单 prompt/8 token（产物分文件）")
    ap.add_argument("--out-name", default=None, help="产物后缀（默认 base）")
    a = ap.parse_args()
    suffix = ("_" + (a.out_name or ("smoke" if a.smoke else "base")))
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"serve_baseline{suffix}.json")

    l215, src = _load_llama215()
    device = "npu" if torch.npu.is_available() else "cpu"
    dtype = torch.float32 if a.smoke else None  # 正式档跑 fp32+bf16 双口径

    # 语料 prompt：tokens32k.bin 首 512 token（真实语料——三候选）
    tok_path = os.path.join(ROOT, "log", "book3-ch10", "tokens32k.bin")
    n0_list = [24] if a.smoke else [24, 128, 512]
    n_gen = 8 if a.smoke else 16

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    assert sum(p.numel() for p in l215.Llama215().parameters()) == N_PARAMS  # 参数账 assert

    results = {"ckpt": os.path.basename(CKPT), "device": device, "llama215_src": src,
               "flops_const": {"N_params": N_PARAMS, "d_model": D_MODEL, "vocab": V}, "runs": []}

    for dtype_name in (["float32"] if a.smoke else ["float32", "bfloat16"]):
        dt = torch.float32 if dtype_name == "float32" else torch.bfloat16
        model = l215.Llama215()
        model.load_state_dict(ck["model"], strict=True)
        model.eval().to(device=device, dtype=dt)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == N_PARAMS, n_params

        for n0 in n0_list:
            if os.path.isfile(tok_path):
                import numpy as np
                ids0 = np.fromfile(tok_path, dtype=np.uint16, count=n0).tolist()
                src_note = f"tokens32k 前 {n0} token（真实语料）"
            else:
                g = torch.Generator().manual_seed(20261002)
                ids0 = torch.randint(0, V, (n0,), generator=g).tolist()
                src_note = f"随机 id（种子 20261002）"
            gen_ids = list(ids0)
            t_in = torch.tensor([gen_ids], device=device)

            torch.npu.synchronize() if device == "npu" else None
            t1 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_gen):
                    logits = model(t_in)[0]                     # (1, n, V) 全量重算
                    nxt = logits[:, -1].argmax(-1, keepdim=True)
                    t_in = torch.cat([t_in, nxt], dim=1)        # (1, n+1)
            torch.npu.synchronize() if device == "npu" else None
            dt_s = time.perf_counter() - t1

            n_final = t_in.shape[1]
            ms_tok = dt_s / n_gen * 1e3
            # 手算（以末步 n=n_final 为代表步；平均步 FLOPs 取积分中值 n_mid）
            n_mid = (n0 + n_final) // 2
            fl_mid = flops_forward(n_mid)
            bw = bytes_weights(4 if dtype_name == "float32" else 2)
            results["runs"].append({
                "dtype": dtype_name, "n0": n0, "n_gen": n_gen, "prompt_src": src_note,
                "ms_per_token": round(ms_tok, 3), "tok_per_s": round(n_gen / dt_s, 2),
                "hand": {"flops_per_tok_mid": fl_mid, "bytes_weights": bw,
                         "fwd_time_floor_by_bytes@1190GBps_s": round(bw / 1.19e12, 6)},
            })
            print(f"[{dtype_name} n0={n0:3d}] {ms_tok:7.2f} ms/token | {n_gen/dt_s:6.2f} tok/s "
                  f"| 手算 {fl_mid/1e9:.1f} GFLOP/步 · 权重 {bw/2**20:.0f} MiB", flush=True)
        del model
        if device == "npu":
            torch.npu.empty_cache()

    results["note"] = ("无 KV cache 全量重算口径（服务前基线）；FLOPs=2N·n 乘加账，"
                       "非 matmul 项<5% 未计；带宽下限按本机实测拷贝稳态 1190 GB/s（notes/04 P2）。")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"[产物] {out_path}")


if __name__ == "__main__":
    main()
