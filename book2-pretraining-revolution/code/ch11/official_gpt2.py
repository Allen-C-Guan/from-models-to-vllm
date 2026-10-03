# official_gpt2.py —— Book2 ch11 官方权重加载与对拍四步（11.2 节的全部数字来源）
# 用途：openai-community/gpt2 官方 checkpoint 加载进 ch10/model.py 的手写 GPT2，四步验收：
#   ①下载最小文件集（≈525MB，缓存命中即跳过）；②state_dict 键映射（三高危点：Conv1D 转置/tied 补绑/前缀剥除）；
#   ③参数逐项对账（总参 124,439,808 / 非嵌入 85,056,000，checkpoint 张量逐组核对）；
#   ④固定 7-prompt 集 logits 对拍（手写加载官方权重 vs HF 官方实现 vs 探针基线 gpt2_logits_baseline.pt，
#     CPU fp32 / eager 注意力，报 max|Δ| / argmax 一致率 / top-5 命中率）+ 贪心 sanity 生成两条教例。
# 所属章节：Book2 第 11 章 11.2 节（三高危点在 ch10 --parity-hf 随机权重上预演过，本章是官方权重的正片）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch11/official_gpt2.py"
# 产物：log/book2-ch11/official_parity.json + figures/fig-11-2-parity.png
import json
import os
import sys
import time

import numpy as np
import torch

# ---------------- 固定口径 ----------------
SEED = 20261002                      # 全书统一种子
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book2-ch11")
FIG_DIR = os.path.join(REPO_ROOT, "figures")
MODEL_ID = "openai-community/gpt2"   # 官方 124M（Modified MIT 许可——商用完整、生成内容免附注记）
BASELINE_PT = os.path.join(REPO_ROOT, "log", "book2-feasibility", "gpt2_logits_baseline.pt")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ch10"))
from model import GPT2  # noqa: E402  ch10 定版模型（三插槽 Block；本章对拍的对象就是它）

# 固定 7-prompt 集（与探针(d)完全同一组：短事实/重复诱导/长段落/德语/代码风/算术/单字符——
# ch11 对拍基线；覆盖 token 长度 1~31，跨语言与代码域）
PROMPTS = [
    "The capital of France is",
    "The quick brown fox jumps over the lazy dog. The quick brown fox",
    "In 1776, the Second Continental Congress declared the independence of the colonies as the United States. Historians agree that",
    "Der Begriff Nähe bedeutet im Deutschen",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "2 + 2 = 4, 3 + 3 = 6, 4 + 4 =",
    "a",
]


# ---------------- 步骤①：最小文件集下载（缓存命中即跳过） ----------------
def ensure_weights():
    """只取 PyTorch 必需文件集（safetensors + tokenizer + config ≈525MB，勿拉全仓 1.6GB）。"""
    from huggingface_hub import snapshot_download
    t0 = time.perf_counter()
    path = snapshot_download(MODEL_ID, allow_patterns=[
        "model.safetensors", "config.json", "generation_config.json",
        "vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json"])
    dt = time.perf_counter() - t0
    files = {}
    for fn in sorted(os.listdir(path)):
        fp = os.path.join(path, fn)
        if os.path.isfile(fp):
            files[fn] = round(os.path.getsize(fp) / 1024 / 1024, 1)
    total_mb = round(sum(files.values()), 1)
    return {"snapshot_dir": path, "files_mb": files, "total_mb": total_mb,
            "wallclock_sec": round(dt, 1), "note": "未鉴名 HF 下载实测出现过两次挂起（notes/01 §3）——"
            "重试+断点续传可完成；缓存命中时本步秒级"}


# ---------------- 步骤②③：键映射 + 参数逐项对账 ----------------
def load_official_into_ours(snap_dir):
    """官方 safetensors -> 手写 GPT2 的 state_dict 键映射（三高危点教学件）。

    高危点 A（Conv1D 转置）：checkpoint 的 c_attn/c_fc/c_proj 权重是 (in,out) 排布
      （Conv1D 的 addmm 语义 y=xW+b），nn.Linear 是 (out,in)（F.linear 语义 y=xW^T+b）——
      搬运须 .t()；其中 attn.c_proj 是 (768,768) 方阵，不转置不报错、静默输出垃圾（最阴的一处）。
    高危点 B（tied 缺键）：checkpoint 只存 wte 一份（160 键里没有 lm_head.weight），
      加载时须显式补 head.weight = wte.weight，否则 strict 加载缺键。
    高危点 C（无前缀）：raw checkpoint 顶层键无 transformer. 前缀（OpenAI TF 转换遗产），
      h.{i} 需改名 blocks.{i}；另有 12 个 h.{i}.attn.bias 遗留 causal-mask buffer 要跳过（v5 已废弃持久化）。
    """
    from safetensors.torch import load_file
    raw = load_file(os.path.join(snap_dir, "model.safetensors"))   # 160 键（148 参数 + 12 attn.bias buffer）
    sd, n_transpose, n_skip = {}, 0, 0
    for k, v in raw.items():
        if k.split(".")[-2:] == ["attn", "bias"]:
            n_skip += 1                                            # 高危点 C 的伴生件：遗留 mask buffer
            continue
        k2 = k.replace("h.", "blocks.")                            # 高危点 C：h.{i} -> blocks.{i}（无 transformer. 前缀可剥）
        if k2.endswith(("c_attn.weight", "c_proj.weight", "c_fc.weight")):
            v = v.t()                                              # 高危点 A：Conv1D (in,out) -> Linear (out,in)
            n_transpose += 1
        sd[k2] = v
    sd["head.weight"] = sd["wte.weight"]                           # 高危点 B：tied 补绑（同一存储的引用赋值）
    ours = GPT2().eval()                                           # dropout=0（对拍口径；官方 pdrop 0.1 仅训练态生效）
    ours.load_state_dict(sd, strict=True)                          # strict=True：无缺键/无多键即通过（高危点 B 的验收）
    # 步骤③：参数逐项对账（逐组张量 numel 与逐位相等校验）
    groups = {"wte": 0, "wpe": 0, "blocks": 0, "ln_f": 0}
    bitwise_equal = True
    for k, v in sd.items():
        g = k.split(".")[0]
        g = "blocks" if g == "blocks" else g
        groups[g] = groups.get(g, 0) + v.numel()
        ours_v = ours.state_dict()[k]
        if not torch.equal(ours_v, v):
            bitwise_equal = False
    n_total = sum(p.numel() for p in {p.data_ptr(): p for p in ours.parameters()}.values())
    n_nonemb = n_total - ours.wte.weight.numel() - ours.wpe.weight.numel()
    audit = {
        "checkpoint_keys_total": len(raw),
        "keys_mapped": len(sd) - 1, "conv1d_transposed": n_transpose, "attn_bias_skipped": n_skip,
        "strict_load": True,
        "per_group_params": groups,
        "params_total_dedup": n_total, "params_non_embedding_6ND": n_nonemb,
        "bitwise_equal_after_load": bitwise_equal,
        "asserts": {"total==124439808": n_total == 124_439_808,
                    "nonemb==85056000": n_nonemb == 85_056_000,
                    "wte==38597376": groups["wte"] == 38_597_376,
                    "wpe==786432": groups["wpe"] == 786_432,
                    "blocks==85054464": groups["blocks"] == 85_054_464,
                    "ln_f==1536": groups["ln_f"] == 1_536},
    }
    return ours, audit


# ---------------- 步骤④：7-prompt logits 对拍 + sanity 生成 ----------------
@torch.no_grad()
def parity_7prompts(ours, tok):
    """三向对拍：手写(官方权重) vs HF 官方实现（eager, CPU fp32）vs 探针基线（2026-10-02 存盘）。"""
    from transformers import GPT2LMHeadModel
    # 注意力路径与手写同型（eager）：v5 默认 sdpa 与 eager 差 ~1e-6，对拍取同型路径
    hf_eager = GPT2LMHeadModel.from_pretrained(MODEL_ID, attn_implementation="eager").eval()

    baseline = torch.load(BASELINE_PT, map_location="cpu") if os.path.exists(BASELINE_PT) else None
    rows = []
    for i, p in enumerate(PROMPTS):
        enc = tok(p, return_tensors="pt")
        ids = enc["input_ids"]                                  # (1, n)
        lg_ours, _ = ours(ids)                                  # (1,n,50257)
        lg_hf = hf_eager(input_ids=ids).logits                  # (1,n,50257)
        last_o, last_h = lg_ours[0, -1, :], lg_hf[0, -1, :]     # (50257,) 各取末位
        d_hf = (last_o - last_h).abs()
        t5o, t5h = torch.topk(last_o, 5), torch.topk(last_h, 5)
        hit5 = (t5o.indices.unsqueeze(-1) == t5h.indices.unsqueeze(-2)).any(-1).float().mean().item()
        row = {
            "prompt": p, "n_tokens": int(ids.shape[1]),
            "max_dlogits_vs_hf": float(d_hf.max()),
            "argmax_agree": bool(last_o.argmax() == last_h.argmax()),
            "argmax_token": tok.decode([int(last_o.argmax())]),
            "top5_hit": hit5,
            "sum_logprob_ours": round(float(torch.log_softmax(lg_ours[0, :-1, :], dim=-1)
                                            .gather(1, ids[0, 1:].unsqueeze(1)).sum()), 4),
        }
        if baseline is not None and f"p{i}" in baseline:
            row["max_dlogits_vs_probe_baseline"] = float((last_o - baseline[f"p{i}"]).abs().max())
        rows.append(row)
    del hf_eager
    return rows, baseline is not None


@torch.no_grad()
def greedy_generate(model, tok, prompt, n_new=20):
    """贪心自回归生成（全量重算口径：每步重跑整段前向，本册不引入增量缓存——欠账在 Book3/6）。
    ids (1,n) -> 逐步 append argmax -> (1,n+n_new)。"""
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    for _ in range(n_new):
        logits, _ = model(ids)                       # (1,n,V)
        ids = torch.cat([ids, logits[:, -1, :].argmax(-1, keepdim=True)], dim=1)   # (1,n) -> (1,n+1)
    return tok.decode(ids[0][ids.shape[1] - n_new:])


def make_figure(report):
    """图 11.2：对拍四步流程（左）+ 7-prompt 逐条 max|Δlogits|（右）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=300,
                                   gridspec_kw={"width_ratios": [1.05, 1]})
    BLUE, ORANGE, TEAL, GRID, TICK = "#2a78d6", "#eb6834", "#1baf7a", "#e1e0d9", "#898781"
    # 左：四步流程框线图
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 10); ax1.axis("off")
    steps = [
        ("step 1  download", "525 MB minimal set\n(safetensors+tokenizer+config)"),
        ("step 2  key mapping", "A Conv1D .t() x48\nB tied: head=wte\nC no prefix / skip attn.bias x12"),
        ("step 3  param audit", "total 124,439,808\nnon-emb 85,056,000\nbitwise equal"),
        ("step 4  logits parity", "7 prompts, CPU fp32\nmax|dl| / argmax / top-5"),
    ]
    for i, (t, s) in enumerate(steps):
        y = 8.6 - i * 2.55
        ax1.add_patch(plt.Rectangle((0.6, y - 0.95), 8.8, 2.1, fill=True, facecolor="white",
                                    edgecolor=BLUE, lw=1.4))
        ax1.text(5.0, y + 0.45, t, ha="center", va="center", fontsize=8.5, color="#0b0b0b", weight="bold")
        ax1.text(5.0, y - 0.35, s, ha="center", va="center", fontsize=7, color="#52514e")
        if i < 3:
            ax1.annotate("", xy=(5.0, y - 1.55), xytext=(5.0, y - 0.95),
                         arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2))
    ax1.set_title("official weights -> ours: 4-step parity", fontsize=9, color="#0b0b0b")
    # 右：逐 prompt max|Δlogits|（对数轴散点；严格 0 画在 1e-9 地板并用空心标记区分）
    rows = report["parity"]
    xs = np.arange(len(rows))
    vals = [max(r["max_dlogits_vs_hf"], 1e-9) for r in rows]
    zeros = [i for i, r in enumerate(rows) if r["max_dlogits_vs_hf"] == 0.0]
    ax2.plot(xs, vals, "o", color=BLUE, ms=6, zorder=4, label="ours(official wts) vs HF eager")
    ax2.plot(zeros, [1e-9] * len(zeros), "o", mfc="white", mec=BLUE, ms=6, zorder=5)
    ax2.text(0.99, 0.985, f"{len(zeros)}/7 exactly 0.0 (bitwise)", fontsize=6.6, color=BLUE,
             ha="right", va="top", transform=ax2.transAxes)
    if "max_dlogits_vs_probe_baseline" in rows[0]:
        ax2.plot(xs, [r["max_dlogits_vs_probe_baseline"] for r in rows], "o", color=ORANGE,
                 ms=5, zorder=3, label="vs probe baseline (sdpa path)")
    ax2.axhline(1e-4, color=TEAL, lw=1.4, ls="--", zorder=2)
    ax2.text(0.02, 1.4e-4, "atol 1e-4", fontsize=7, color=TEAL, va="bottom", transform=ax2.get_yaxis_transform())
    ax2.set_yscale("log"); ax2.set_ylim(3e-10, 1e-3)
    ax2.set_xticks(list(xs)); ax2.set_xticklabels([f"p{i}" for i in xs], fontsize=7, color=TICK)
    ax2.set_xlabel("fixed 7-prompt set", fontsize=8, color="#0b0b0b")
    ax2.set_ylabel("max |delta logits| (last position)", fontsize=8, color="#0b0b0b")
    ax2.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax2.tick_params(colors=TICK)
    am = sum(r["argmax_agree"] for r in rows)
    t5 = sum(r["top5_hit"] >= 1.0 for r in rows)
    ax2.set_title(f"argmax {am}/7 agree, top-5 {t5}/7 full hit", fontsize=9, color="#0b0b0b")
    ax2.legend(fontsize=6.4, frameon=False, loc="center left")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-11-2-parity.png")
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"[figure] -> {out}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    t0 = time.perf_counter()
    from transformers import AutoTokenizer
    report = {"meta": {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED,
                       "model_id": MODEL_ID, "device": "cpu", "dtype": "fp32",
                       "torch": torch.__version__,
                       "attention": "eager (both sides; sdpa-eager diff ~1e-6, parity uses same-type path)"}}

    print("===== 步骤① 下载/定位最小文件集 =====", flush=True)
    report["download"] = ensure_weights()
    print(json.dumps(report["download"]["files_mb"], ensure_ascii=False), flush=True)

    print("===== 步骤②③ 键映射 + 参数逐项对账 =====", flush=True)
    ours, audit = load_official_into_ours(report["download"]["snapshot_dir"])
    report["audit"] = audit
    print(json.dumps(audit["asserts"], ensure_ascii=False), "| bitwise_equal:", audit["bitwise_equal_after_load"])

    print("===== 步骤④ 7-prompt logits 对拍 + sanity 生成 =====", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    rows, has_baseline = parity_7prompts(ours, tok)
    report["parity"] = rows
    report["parity_note"] = ("三向：手写加载官方权重 vs HF eager 实现 vs 探针基线"
                             + ("（log/book2-feasibility/gpt2_logits_baseline.pt，2026-10-02 存盘）" if has_baseline else "（基线文件缺失）"))
    # sanity 生成两条教例（贪心 20 token）
    report["sanity_gen"] = []
    for p in [PROMPTS[0], PROMPTS[5]]:
        gen = greedy_generate(ours, tok, p, n_new=20)
        with torch.no_grad():
            ids = tok(p, return_tensors="pt")["input_ids"]
            logits, _ = ours(ids)
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            t5v, t5i = torch.topk(probs, 5)
        report["sanity_gen"].append({
            "prompt": p, "greedy_20tok": gen,
            "next_top5": [{"token": tok.decode([int(i_)]), "id": int(i_), "p": round(float(v), 5)}
                          for v, i_ in zip(t5v, t5i)]})
        print(f"  [{p[:36]}...] greedy -> {gen!r}", flush=True)

    report["meta"]["wallclock_sec"] = round(time.perf_counter() - t0, 1)
    out_json = os.path.join(OUT_DIR, "official_parity.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[完成] -> {out_json}")
    make_figure(report)


if __name__ == "__main__":
    main()
