# official_gpt2.py —— Book2第11章：从同一快照加载权重/配置/分词器并核对固定输入。
# 工作区激活env.sh后：python code/ch11/official_gpt2.py --help
# --snapshot-dir可读取已有快照；网络获取记录所解析的目录与文件哈希，不预设历史键数。
# 参数与映射采用真实断言；有限输入对拍不等于全输入数学等价证明。
# 输出log，图为当前运行核算图，不覆盖正文历史图。
import argparse
import hashlib
from pathlib import Path
import json
import os
import sys
import time

import numpy as np
import torch

# ---------------- 固定口径 ----------------
SEED = 20261002                      # 全书统一种子
REPO_ROOT = os.getcwd()
OUT_DIR = os.path.join(REPO_ROOT, "log", "book2-ch11")
FIG_DIR = os.path.join(OUT_DIR,"figures")
MODEL_ID = "openai-community/gpt2"   # 许可仅按公开LICENSE原文描述，正文不扩张解释。
BASELINE_PT = os.path.join(REPO_ROOT, "log", "book2-feasibility", "gpt2_logits_baseline.pt")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ch10"))
from model import GPT2,GPT2Config  # ch10教学模型；构造权重单元检查也可使用同结构小配置。

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
def file_sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def ensure_weights(revision=None,snapshot_dir=None):
    """取得一个确定快照；后续权重、HF参考和分词器全从该目录读取。"""
    t0=time.perf_counter()
    if snapshot_dir is None:
        from huggingface_hub import snapshot_download
        path=Path(snapshot_download(MODEL_ID,revision=revision,allow_patterns=[
            "model.safetensors","config.json","generation_config.json",
            "vocab.json","merges.txt","tokenizer.json","tokenizer_config.json"]))
    else:path=Path(snapshot_dir)
    for name in ("model.safetensors","config.json"):
        if not (path/name).is_file():raise FileNotFoundError(path/name)
    names=("model.safetensors","config.json","generation_config.json","vocab.json",
           "merges.txt","tokenizer.json","tokenizer_config.json")
    files={name:(path/name).stat().st_size for name in names if (path/name).is_file()}
    return {"snapshot_dir":str(path.resolve()),"requested_revision":revision,
            "resolved_snapshot_name":path.name,"file_bytes":files,
            "files_mib":{k:v/1024**2 for k,v in files.items()},"total_mib":sum(files.values())/1024**2,
            "model_sha256":file_sha(path/"model.safetensors"),"config_sha256":file_sha(path/"config.json"),
            "wallclock_sec":time.perf_counter()-t0,
            "note":"MiB按2^20 bytes；这些是本次文件实际信息，不默认等同历史160键/525.4MiB快照"}


# ---------------- 步骤②③：键映射 + 参数逐项对账 ----------------
def load_official_into_ours(snap_dir):
    """按明确GPT2配置映射有/无transformer前缀的权重，检查共享别名而非盲目丢键。"""
    from safetensors.torch import load_file
    path=Path(snap_dir);conf=json.loads((path/"config.json").read_text())
    cfg=GPT2Config(vocab_size=conf["vocab_size"],n_positions=conf["n_positions"],
                   n_layer=conf["n_layer"],n_embd=conf["n_embd"],n_head=conf["n_head"],dropout=0.)
    if (conf.get("activation_function","gelu_new")!="gelu_new"
            or conf.get("n_inner") not in (None,4*cfg.n_embd)
            or conf.get("layer_norm_epsilon",1e-5)!=1e-5
            or conf.get("add_cross_attention",False)
            or conf.get("scale_attn_by_inverse_layer_idx",False)
            or not conf.get("scale_attn_weights",True)
            or not conf.get("tie_word_embeddings",True)):
        raise ValueError("配置超出本教学GPT2实现支持范围")
    raw=load_file(str(path/"model.safetensors"))
    if any(v.is_floating_point() and v.dtype!=torch.float32 for k,v in raw.items()
           if not k.endswith((".attn.bias",".attn.masked_bias"))):
        raise ValueError("本对拍限定原权重为fp32；低精度文件不能经静默转换后称逐位拷贝")
    normalized={}
    for key,value in raw.items():
        name=key.removeprefix("transformer.")
        if name in normalized:raise ValueError("去前缀后键冲突："+name)
        normalized[name]=value
    if "wte.weight" not in normalized:raise ValueError("缺词嵌入")
    for alias in ("lm_head.weight","head.weight"):
        if alias in normalized and not torch.equal(normalized[alias],normalized["wte.weight"]):
            raise ValueError("共享输出头别名与词嵌入数值不一致")
    sd={};n_transpose=0;skipped=[]
    for key,value in normalized.items():
        if key in ("lm_head.weight","head.weight"):continue
        if key.endswith((".attn.bias",".attn.masked_bias")):
            skipped.append(key);continue
        name="blocks."+key[2:] if key.startswith("h.") else key
        if name.endswith(("c_attn.weight","c_proj.weight","c_fc.weight")):
            value=value.t();n_transpose+=1
        sd[name]=value
    sd["head.weight"]=sd["wte.weight"]  # 加载别名；构造器已建立参数共享
    ours=GPT2(cfg).eval();ours.load_state_dict(sd,strict=True)
    after=ours.state_dict()
    bitwise=all(after[k].dtype==v.dtype and torch.equal(after[k],v) for k,v in sd.items())
    groups={k:sum(v.numel() for name,v in sd.items() if name.startswith(k+"."))
            for k in ("wte","wpe","blocks","ln_f")}
    n_total=ours.n_params();n_non=ours.n_params(True);d=cfg.n_embd
    expected=cfg.vocab_size*d+cfg.n_positions*d+cfg.n_layer*(12*d*d+13*d)+2*d
    checks={"total_matches_config":n_total==expected,
            "nonembedding_matches_config":n_non==cfg.n_layer*(12*d*d+13*d)+2*d,
            "group_sum_matches_unique_total":sum(groups.values())==n_total,
            "all_loaded_tensors_equal":bitwise,"shared_parameter_object":ours.head.weight is ours.wte.weight,
            "four_transposes_per_block":n_transpose==4*cfg.n_layer}
    if not all(checks.values()):raise AssertionError(checks)
    audit={"checkpoint_keys_total":len(raw),"keys_mapped":len(sd)-1,"loading_keys":len(sd),
           "conv1d_transposed":n_transpose,"attn_bias_skipped":len(skipped),"skipped_keys":skipped,
           "strict_load":True,"per_group_params":groups,"head_is_alias_not_extra_parameters":True,
           "params_total_dedup":n_total,"params_non_embedding_6ND":n_non,
           "bitwise_equal_after_load":bitwise,"asserts":checks,"config":vars(cfg),
           "note":"键数属于本次文件，不预设160；别名不重复计入参数组"}
    return ours,audit


# ---------------- 步骤④：7-prompt logits 对拍 + sanity 生成 ----------------
@torch.no_grad()
def parity_7prompts(ours,tok,snapshot_dir,prompts=None):
    """同一快照的手写实现与HF eager固定输入对拍；不自动使用缺身份的历史探针。"""
    from transformers import GPT2LMHeadModel
    hf_eager=GPT2LMHeadModel.from_pretrained(str(snapshot_dir),attn_implementation="eager",
                                               local_files_only=True).float().eval()
    # 旧tensor-only探针文件缺快照/提示身份，不自动拿它当当前参考。
    baseline=None
    prompts=PROMPTS if prompts is None else prompts
    if not prompts:raise ValueError("对拍至少需要一个输入")
    rows = []
    for i,p in enumerate(prompts):
        enc = tok(p, return_tensors="pt")
        ids = enc["input_ids"]                                  # (1, n)
        lg_ours, _ = ours(ids)                                  # (1,n,50257)
        lg_hf = hf_eager(input_ids=ids,use_cache=False).logits                  # (1,n,50257)
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
        if not torch.isfinite(d_hf).all() or row["max_dlogits_vs_hf"]>=1e-4 or not row["argmax_agree"]:
            raise AssertionError({"prompt":p,"parity":row})
        rows.append(row)
    del hf_eager
    return rows, baseline is not None


@torch.no_grad()
def greedy_generate(model, tok, prompt, n_new=20):
    """贪心自回归生成（全量重算口径：每步重跑整段前向，本册不引入增量缓存——欠账在 Book3/6）。
    ids (1,n) -> 逐步 append argmax -> (1,n+n_new)。"""
    ids=tok(prompt,return_tensors="pt")["input_ids"]
    if n_new<0 or ids.shape[1]==0 or ids.shape[1]+max(0,n_new-1)>model.block_size:
        raise ValueError("无效生成长度或超过模型上下文")
    if n_new==0:return ""
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
        ("step 1  download", f"{report['download']['total_mib']:.1f} MiB this snapshot\n(safetensors + tokenizer + config)"),
        ("step 2  key mapping", f"A Conv1D .t() x{report['audit']['conv1d_transposed']}\nB tied alias; C normalize prefixes\nskipped masks: {report['audit']['attn_bias_skipped']}"),
        ("step 3  param audit", f"total {report['audit']['params_total_dedup']:,}\nnon-emb {report['audit']['params_non_embedding_6ND']:,}\nbitwise checked"),
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
    ax2.text(0.99, 0.985, f"{len(zeros)}/{len(rows)} exactly 0.0", fontsize=6.6, color=BLUE,
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
    ax2.set_title(f"argmax {am}/{len(rows)}, top-5 {t5}/{len(rows)}", fontsize=9, color="#0b0b0b")
    ax2.legend(fontsize=6.4, frameon=False, loc="center left")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out=os.path.join(FIG_DIR,"check-11-2-parity.png")
    if os.path.exists(out):raise FileExistsError(out)
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"[figure] -> {out}")


def main():
    global OUT_DIR,FIG_DIR
    ap=argparse.ArgumentParser()
    ap.add_argument("--snapshot-dir")
    ap.add_argument("--revision")
    ap.add_argument("--out-dir",default=OUT_DIR)
    ap.add_argument("--out-name",default="official_parity.json")
    a=ap.parse_args();OUT_DIR=a.out_dir;FIG_DIR=os.path.join(OUT_DIR,"figures",Path(a.out_name).stem)
    if Path(a.out_name).name!=a.out_name or a.out_name in ("",".",".."):raise ValueError("out-name只接受文件名")
    target=Path(OUT_DIR)/a.out_name
    if target.exists():raise FileExistsError(target)
    torch.manual_seed(SEED);t0=time.perf_counter()
    import transformers
    from transformers import AutoTokenizer
    report={"meta":{"date_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                    "seed":SEED,"device":"cpu","dtype":"fp32","torch":str(torch.__version__),
                    "transformers":transformers.__version__,"attention":"eager，同快照、use_cache=False"}}
    report["download"]=ensure_weights(a.revision,a.snapshot_dir)
    snapshot=report["download"]["snapshot_dir"]
    ours,audit=load_official_into_ours(snapshot);report["audit"]=audit
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    rows,_=parity_7prompts(ours,tok,snapshot);report["parity"]=rows
    report["parity_note"]="有限输入的当前数值证据；旧无身份探针基线未自动参与"
    report["sanity_gen"]=[]
    for prompt in (PROMPTS[0],PROMPTS[5]):
        text=greedy_generate(ours,tok,prompt,20)
        with torch.no_grad():
            ids=tok(prompt,return_tensors="pt")["input_ids"]
            probs=torch.softmax(ours(ids)[0][0,-1],-1)
            values,index=probs.topk(5)
        report["sanity_gen"].append({"prompt":prompt,"greedy_20tok":text,
            "next_top5":[{"token":tok.decode([int(i)]),"id":int(i),"p":float(v)} for v,i in zip(values,index)]})
    report["meta"]["wallclock_sec"]=time.perf_counter()-t0
    payload=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    target.parent.mkdir(parents=True,exist_ok=True)
    make_figure(report)
    with target.open("x") as f:f.write(payload)
    print(f"[固定输入对拍通过] {target}")


if __name__ == "__main__":
    main()
