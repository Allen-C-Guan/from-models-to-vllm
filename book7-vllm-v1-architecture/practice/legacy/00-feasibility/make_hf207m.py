# make_hf207m.py —— Book7 00-feasibility 代码件：把 Book3 终态 207M ckpt 封装为 HF Llama 格式模型目录
# 用途：大项目（作者决策：大项目形态——注册进 vllm-ascend 与 mini 引擎对照）的模型资产前置件——
#       log/book3-ch10/ckpt_integrated_npu1.pt 的 model 态字典本就是 HF Llama 键位（Book3 插槽 import
#       契约），本件只做「抽取+封装」：写 config.json / model.safetensors / generation_config /
#       tokenizer（sp32000 sentencepiece → LlamaTokenizer）+ 两道 CPU sanity（HF 加载前向 + 可选
#       llama215 对拍）。产物目录随后供 vllm 运行栈直接加载（LlamaForCausalLM 内置注册路径）。
# 所属章节：Book7 ch10（权重加载与模型注册·大项目）；设计书=plan/Book7-vLLM源码剖析上.md
# 运行：cd <workspace> && source env.sh && python code/Book7-vLLM源码剖析上/00-feasibility/make_hf207m.py \
#        [--smoke]（只做 2 层小模型封装自检，秒级）[--parity]（附 llama215 逐位对拍，CPU fp32）
# 产物：log/book7-00/hf207m/（config.json、model.safetensors、generation_config.json、tokenizer.model、
#       tokenizer_config.json）+ log/book7-00/make_hf207m_report.json
# 资产正身：ckpt 键位=HF Llama（111 键）；V=32000 sp32k（unk 0 / bos 1 / eos 2 / 无 pad / byte_fallback）；
#       config 定版 cand1：d=1024 / L=12 / h=16 / h_kv=8 / d_ff=2816 / RoPE theta=10000 / eps=1e-5 / untied。
import argparse
import json
import os
import shutil

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
CKPT = os.path.join(REPO_ROOT, "log", "book3-ch10", "ckpt_integrated_npu1.pt")
SP_MODEL = os.path.join(REPO_ROOT, "log", "book3-ch10", "sp32000_owt.model")
OUT_DIR = os.path.join(REPO_ROOT, "log", "book7-00", "hf207m")
REPORT = os.path.join(REPO_ROOT, "log", "book7-00", "make_hf207m_report.json")

CFG = dict(  # 与 DESIGN-215M / llama215.py 定版逐项一致
    arch="LlamaForCausalLM", model_type="llama", hidden_size=1024, num_hidden_layers=12,
    num_attention_heads=16, num_key_value_heads=8, intermediate_size=2816, vocab_size=32000,
    rope_theta=10000.0, rms_norm_eps=1e-5, tie_word_embeddings=False,
    max_position_embeddings=2048,  # 服务侧安全窗（训练为文档流，无硬窗；RoPE 无外推设定）
    bos_token_id=1, eos_token_id=2, pad_token_id=0,  # pad 借 unk=0（sp 无 pad）
    torch_dtype="float32",
)
N_PARAMS_EXPECT = 207_119_360


def load_sd(smoke: bool):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["model"]
    if not smoke:
        return sd
    # smoke 档：只保留 layers.0/1（自检封装流程，不验参数量）
    keep = {k: v for k, v in sd.items() if not k.startswith("model.layers.") or k.split(".")[2] in ("0", "1")}
    cfg = dict(CFG, num_hidden_layers=2)
    return keep, cfg


def write_dir(sd, cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n_params = sum(v.numel() for v in sd.values())
    # config.json（LlamaConfig 全字段显式——教学口径：注册链上被读的每个字段都有主）
    config = {
        "architectures": [cfg["arch"]], "model_type": cfg["model_type"],
        "hidden_size": cfg["hidden_size"], "num_hidden_layers": cfg["num_hidden_layers"],
        "num_attention_heads": cfg["num_attention_heads"], "num_key_value_heads": cfg["num_key_value_heads"],
        "intermediate_size": cfg["intermediate_size"], "vocab_size": cfg["vocab_size"],
        "rope_theta": cfg["rope_theta"], "rms_norm_eps": cfg["rms_norm_eps"],
        "tie_word_embeddings": cfg["tie_word_embeddings"],
        "max_position_embeddings": cfg["max_position_embeddings"],
        "bos_token_id": cfg["bos_token_id"], "eos_token_id": cfg["eos_token_id"],
        "pad_token_id": cfg["pad_token_id"], "torch_dtype": cfg["torch_dtype"],
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=1)
    # generation_config.json
    with open(os.path.join(out_dir, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump({"bos_token_id": cfg["bos_token_id"], "eos_token_id": cfg["eos_token_id"],
                   "_from_model_config": True}, f, indent=1)
    # model.safetensors（fp32 原样——vllm 侧 dtype="bfloat16" 再铸）
    from safetensors.torch import save_file
    save_file(sd, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    # tokenizer：sp 模型原样拷贝 + LlamaTokenizer 配置
    shutil.copyfile(SP_MODEL, os.path.join(out_dir, "tokenizer.model"))
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "tokenizer_class": "LlamaTokenizer", "model_max_length": cfg["max_position_embeddings"],
            "legacy": True, "add_bos_token": False, "add_eos_token": False,
            "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>",
            "pad_token": None, "clean_up_tokenization_spaces": False,
        }, f, ensure_ascii=False, indent=1)
    return n_params


def sanity_forward(out_dir):
    """HF 链 sanity：AutoModel 加载 + 定种子 token 前向（CPU fp32）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(out_dir, dtype=torch.float32)
    model.eval()
    g = torch.Generator().manual_seed(20261010)
    ids = torch.randint(3, 32000, (1, 64), generator=g)
    with torch.no_grad():
        logits = model(ids).logits                      # (1, 64, 32000)
        loss = torch.nn.functional.cross_entropy(
            logits[0, :-1].float(), ids[0, 1:])          # (63, 32000) 对 (63,)
    tok = AutoTokenizer.from_pretrained(out_dir)
    roundtrip = tok.decode(tok.encode("The quick brown fox", add_special_tokens=False))
    # sentencepiece 直连对拍（Book3 语料协议同款：无 BOS、EOS 只在文档尾）
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(); sp.Load(SP_MODEL)
    enc = tok.encode("The quick brown fox", add_special_tokens=False)
    enc_sp = sp.encode("The quick brown fox", out_type=int)
    return dict(loss=round(loss.item(), 4), first_logits_head=[round(x, 4) for x in logits[0, -1, :5].tolist()],
                roundtrip_ok=(roundtrip == "The quick brown fox"), enc_match=(enc == enc_sp))


def parity_llama215(out_dir):
    """llama215 对拍：HF 目录权重 strict 搬进 llama215 整机（键位同名契约），CPU fp32 logits 对。"""
    import sys
    sys.path.insert(0, os.path.join(REPO_ROOT, "code", "Book3-现代开源骨架", "ch10"))
    import llama215
    from transformers import LlamaForCausalLM
    hf = LlamaForCausalLM.from_pretrained(out_dir, dtype=torch.float32).eval()
    m215 = llama215.Llama215()                       # 215M 定版（config 内建，参数账 assert 自带）
    m215.load_state_dict(hf.state_dict(), strict=True)  # 键不一致直接抛错——本身就是验收
    m215.eval()
    g = torch.Generator().manual_seed(20261010)
    ids = torch.randint(3, 32000, (1, 64), generator=g)
    with torch.no_grad():
        out = m215(ids)
        l215 = out[0] if isinstance(out, tuple) else out   # Llama215.forward 返回 (logits, loss?)
        lhf = hf(ids).logits
    dev = (l215 - lhf).abs().max().item()
    agree = (l215.argmax(-1) == lhf.argmax(-1)).float().mean().item()
    return dict(max_abs_diff=dev, argmax_agree=round(agree, 6),
                allclose=bool(torch.allclose(l215, lhf, atol=1e-4, rtol=1e-4)))


def main() -> None:
    ap = argparse.ArgumentParser(description="207M ckpt → HF Llama 格式封装")
    ap.add_argument("--smoke", action="store_true", help="2 层小模型封装自检（秒级）")
    ap.add_argument("--parity", action="store_true", help="附 llama215 逐位对拍")
    a = ap.parse_args()
    if a.smoke:
        sd, cfg = load_sd(True)
        out = os.path.join(REPO_ROOT, "log", "book7-00", "hf207m_smoke")
        n = write_dir(sd, cfg, out)
        print(f"[smoke] 2 层 {n:,} 参数封装于 {out}")
        return
    sd = load_sd(False)
    n_params = sum(v.numel() for v in sd.values())
    assert n_params == N_PARAMS_EXPECT, f"参数量 {n_params:,} ≠ 定版 {N_PARAMS_EXPECT:,}"
    n = write_dir(sd, CFG, OUT_DIR)
    assert n == N_PARAMS_EXPECT
    res = sanity_forward(OUT_DIR)
    report = dict(params=n, out_dir=OUT_DIR, sanity=res, ckpt=CKPT)
    if a.parity:
        report["parity"] = parity_llama215(OUT_DIR)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"[产物] {REPORT}")


if __name__ == "__main__":
    main()
