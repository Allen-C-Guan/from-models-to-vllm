# 05_probe_e_parity.py —— 探针 E：手写 LLaMA vs HF transformers 5.18.0 对拍链路
# 用途：打通「我们手写的 LLaMA 数学等价于 HF 官方实现」的验收管道——
#       mini-LLaMA（d=256/L=4/h=4/h_kv=2/d_ff=704/V=512，RoPE theta=10000）随机初始化后，
#       state_dict 按同名键直搬进 HF LlamaForCausalLM（llama_slots 的参数命名刻意与 HF 对齐），
#       CPU fp32 同输入对拍 logits：max|Δlogits| 与 argmax 一致率。
#       附加 sanity：两个独立随机初始化的模型 logits 应明显不同（证明搬运真的发生了）。
#       可选 --with-tinyllama：尝试下载 TinyLlama-1.1B（走 env.sh 的 HF 缓存；失败/超时只记录不算探针失败）。
# 所属章节：Book3 00-feasibility（ch10 整合验收管道；Book2 ch11 官方权重对拍的等价物）。
# 运行方式：cd code/00-feasibility && python 05_probe_e_parity.py [--with-tinyllama] [--out-name NAME]
#   产物：log/book3-feasibility/{out-name}.json（不 commit）
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from llama_slots import LLaMA, LLaMAConfig   # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book3-feasibility")
SEED = 20261002

# mini config：d=256/L=4/h=4/h_kv=2/d_ff=704(=2/3*4*256 取 64 倍数)/V=512/ctx=128
MINI = dict(vocab_size=512, hidden_size=256, intermediate_size=704,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6)


def parity_check():
    """结构对拍（免下载，CPU fp32）。"""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(SEED)
    ours = LLaMA(LLaMAConfig(**MINI))

    hf_cfg = LlamaConfig(vocab_size=512, hidden_size=256, intermediate_size=704,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=128, rms_norm_eps=1e-6, rope_theta=10000.0,
                         tie_word_embeddings=False, attention_bias=False, mlp_bias=False,
                         attention_dropout=0.0)
    hf = LlamaForCausalLM(hf_cfg).eval()

    # ---- 键映射搬运：ours.state_dict() 键与 HF 完全同名（llama_slots 命名对齐），strict 直搬 ----
    hf.load_state_dict(ours.state_dict(), strict=True, assign=True)   # 键不一致会直接抛错——本身就是验收
    print(f"[搬运] ours 参数 {ours.n_params():,} -> HF LlamaForCausalLM（strict=True 通过）")

    # ---- 同输入 CPU fp32 对拍 ----
    g = torch.Generator().manual_seed(SEED)
    x = torch.randint(0, 512, (2, 96), generator=g)          # (B=2, n=96)
    with torch.no_grad():
        logits_ours, loss_ours = ours(x, x)
        logits_hf = hf(x).logits
    delta = (logits_ours - logits_hf).abs().max().item()
    argmax_agree = (logits_ours.argmax(-1) == logits_hf.argmax(-1)).float().mean().item()
    loss_delta = abs(loss_ours.item() - torch.nn.functional.cross_entropy(
        logits_hf.reshape(-1, 512).float(), x.reshape(-1)).item())

    # ---- sanity：独立随机初始化（未搬运）应明显不同 ----
    torch.manual_seed(SEED + 1)
    other = LLaMA(LLaMAConfig(**MINI))
    with torch.no_grad():
        logits_other, _ = other(x, x)
    delta_unshared = (logits_other - logits_hf).abs().max().item()

    verdict = "PASS" if (delta < 1e-4 and argmax_agree == 1.0 and delta_unshared > 1e-1) else "FAIL"
    rec = {"ours_params": ours.n_params(),
           "max_abs_diff_logit": delta, "argmax_agreement": argmax_agree, "loss_abs_diff": round(loss_delta, 8),
           "sanity_unshared_max_abs_diff": delta_unshared,
           "verdict": verdict,
           "note": "阈值：max|Δ|<1e-4 且 argmax 一致率=100% 且未搬运对照差异显著（>0.1）"}
    print(f"[对拍] max|Δlogits| = {delta:.3e}  argmax 一致率 = {argmax_agree*100:.2f}%  "
          f"|Δloss| = {loss_delta:.2e}  [sanity 未搬运 |Δ| = {delta_unshared:.2f}]  -> {verdict}")
    return rec


def tinyllma_download(timeout_sec=360):
    """TinyLlama-1.1B 下载尝试（ch10 整合短训的「真模型对拍」候选）。走 env.sh 的 HF_HOME 缓存；
    失败/超时如实记录（体量与原因），不算探针失败。"""
    rec = {"repo_id": "TinyLlama/TinyLlama_v1.1"}
    try:
        from huggingface_hub import snapshot_download
        t0 = time.perf_counter()
        path = snapshot_download("TinyLlama/TinyLlama_v1.1",
                                 allow_patterns=["*.json", "*.safetensors"],
                                 max_workers=4)
        n_bytes = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(path) for fn in fns)
        rec.update({"status": "ok", "local_path": path, "bytes": n_bytes,
                    "sec": round(time.perf_counter() - t0, 1)})
        print(f"[tinyllama] 下载完成：{n_bytes/1e9:.2f} GB（{rec['sec']}s）-> {path}")
    except Exception as e:
        rec.update({"status": "failed_or_timeout", "error": str(e)[:300],
                    "expected_size_note": "model.safetensors 约 2.2 GB（bf16 1.1B）"})
        print(f"[tinyllma] 未能完成：{str(e)[:120]}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="probe_e_parity")
    ap.add_argument("--with-tinyllama", action="store_true", help="附带尝试下载 TinyLlama-1.1B")
    args = ap.parse_args()

    result = {"seed": SEED, "config": MINI, "transformers_version": __import__("transformers").__version__,
              "parity": parity_check()}
    if args.with_tinyllama:
        result["tinyllma"] = tinyllma_download()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{args.out_name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"产物已写：{out}")


if __name__ == "__main__":
    main()
