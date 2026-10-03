# sample.py —— Book2 ch11 采样生成：贪心/temperature/top-k + 自训 vs 官方词法统计对照 + ICL 探针（11.3/11.5 节）
# 用途：①在官方权重与自训 124M 上对比四种解码（贪心 / T=1.0 / T=0.7 / top-k=40，k=40=GPT-2 附录口径）；
#   ②生成文本的词法统计对照（词长/空格率/型例比/高频 bigram 命中——诚实条款「生成具备词法统计特征」的实测件）；
#   ③自训模型在 WikiText-2 上的 token loss -> word PPL 换算盒演示（跨语料诚实口径）；
#   ④ICL 探针（选做）：官方权重上 2-3 个 few-shot 演示（本地前向，不调任何外部 API——冻结纪律）。
# 生成一律「全量重算」口径：每生成一个 token 重跑整段前向（本册不引入增量缓存；欠账见 Book3/6）。
# 所属章节：Book2 第 11 章 11.3/11.5 节。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch11/sample.py"
#   [--train-steps 6104]（自训模型步数，与 fast 档同配同种子 -> 复现 train124_fast 曲线并保留权重）
#   [--skip-train]（已有 self124m.pt 时直接加载）[--train-only]（只训练存档，不做采样件）
# 产物：log/book2-ch11/{self124m.pt, sampling.json} + figures/fig-11-3-sampling.png
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

SEED = 20261002
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book2-ch11")
FIG_DIR = os.path.join(REPO_ROOT, "figures")
SELF_PT = os.path.join(OUT_DIR, "self124m.pt")
DOCS_JSONL = os.path.join(REPO_ROOT, "log", "book2-scaling", "owt_docs.jsonl")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ch10"))
import train_124m as t12m          # noqa: E402  复用 fast 档的全部口径常量与数据件（单轨依赖链）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ch04"))
from official_gpt2 import ensure_weights, load_official_into_ours   # noqa: E402  对拍四步的①②③在此复用
from model import GPT2             # noqa: E402  ch10 定版 124M（随机初始化对照侧的机器）

os.environ.setdefault("TIKTOKEN_CACHE_DIR", os.path.join(REPO_ROOT, "log", "cache", "tiktoken"))

GEN_PROMPT = "The meaning of life is"     # 采样演示 prompt（与对拍 7-prompt 分开：这里要看的是开放续写）
GEN_TOKENS = 40                           # 每条生成的续写长度


def get_tok():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


# ---------------- 生成器（全量重算口径） ----------------
@torch.no_grad()
def generate(model, enc, prompt, n_new=GEN_TOKENS, temperature=1.0, top_k=0, generator=None, device="cpu"):
    """自回归生成：贪心（temperature=0）或截断采样。

    ids (1,n) -> 每步整段前向重算 logits (1,n,V)，取末位 (V,)；
    top_k>0 时先截断到前 k 个再按 temperature 加权重归一化采样（Fan 2017 的方案，GPT-2 取 k=40）。
    """
    ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)   # (1,n)
    out = []
    for _ in range(n_new):
        logits, _ = model(ids)                                  # (1,n,V) 全量重算（不引入增量缓存）
        last = logits[0, -1, :].float()                         # (V,)
        if temperature <= 0:
            nxt = int(last.argmax())
        else:
            last = last / temperature                           # (V,) 温度缩放（尖锐度旋钮）
            if top_k > 0:
                kth = torch.topk(last, top_k).values[-1]        # 第 k 大值
                last = last.masked_fill(last < kth, float("-inf"))   # 截断到前 k
            probs = torch.softmax(last, dim=-1)                 # (V,) 重归一化
            nxt = int(torch.multinomial(probs, 1, generator=generator))
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)    # (1,n) -> (1,n+1)
    return out


# ---------------- 自训侧：复现 fast 档并保留权重 ----------------
def train_self(steps, device):
    """与 train_124m.py 完全同配同种子的训练循环（区别仅一处：结束保留最终权重供采样对照——
    train_124m 跑完即删 checkpoint，故在此复跑；确定性同种子下曲线应与 train124_fast.json 一致，
    MPS 并行归约的非确定性差 ~0.004 nat 量级，如实登记）。"""
    from warmup_ablation import GPT     # 单轨依赖链：ch04 工厂（train_124m 同款）
    torch.manual_seed(SEED)
    model = GPT(n_layer=t12m.N_LAYER, n_embd=t12m.N_EMBD, n_head=t12m.N_HEAD,
                vocab=t12m.VOCAB, block_size=t12m.WPE).to(device)
    assert model.n_params() == 124_439_808 and model.n_params(non_embedding=True) == 85_056_000
    opt = torch.optim.AdamW(model.parameters(), lr=t12m.PEAK_LR, betas=(0.9, 0.95), weight_decay=t12m.WDECL)
    ex, ey = t12m.build_eval_batches(device)
    stream = t12m.TrainStream(os.path.join(OUT_DIR, "tokens50k.bin"), steps)
    assert steps <= stream.max_step
    evals, t0 = [], time.perf_counter()
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = t12m.lr_at(step, t12m.PEAK_LR, t12m.WARMUP, steps)
        x, y = stream.batch(step, device)
        with torch.autocast(device_type="mps", dtype=torch.bfloat16) if device == "mps" else torch.autocast("cpu", enabled=False):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device == "mps":
            torch.mps.synchronize()
        if step % 1000 == 0 or step == steps:
            el = t12m.eval_loss(model, ex, ey)
            evals.append({"step": step, "eval_loss": round(el, 5)})
            print(f"  [self-train] step {step}/{steps} train {loss.item():.4f} eval {el:.4f} "
                  f"({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
    torch.save(model.state_dict(), SELF_PT)
    wall = round(time.perf_counter() - t0, 1)
    return model, {"steps": steps, "eval_curve": evals, "wallclock_sec": wall,
                   "ref": "train124_fast.json 同配同种子（train_124m.py 跑完即删 ckpt，此处复跑并保留权重）"}


def load_self(device):
    """加载自训 124M。注意：self124m.pt 由 ch04 工厂 GPT 训练保存（其 attn.bias 是持久化 buffer，
    与 ch10 model.py 的非持久化 mask 不同）——加载须用同一工厂类，保持单轨依赖链口径一致。"""
    from warmup_ablation import GPT
    model = GPT(n_layer=t12m.N_LAYER, n_embd=t12m.N_EMBD, n_head=t12m.N_HEAD,
                vocab=t12m.VOCAB, block_size=t12m.WPE).to(device).eval()
    model.load_state_dict(torch.load(SELF_PT, map_location=device, weights_only=True))
    return model


# ---------------- 词法统计 ----------------
def lexical_stats(text, ref_bigrams):
    """生成文本的词法统计特征（诚实条款口径：词/空格/高频 bigram）。"""
    words = text.split()
    n_char = max(1, len(text))
    bigrams = list(zip(words, words[1:]))
    return {
        "n_words": len(words),
        "mean_word_len": round(sum(len(w) for w in words) / max(1, len(words)), 2),
        "space_rate": round(text.count(" ") / n_char, 3),
        "alpha_rate": round(sum(c.isalpha() for c in text) / n_char, 3),
        "type_token_ratio": round(len(set(words)) / max(1, len(words)), 3),
        "bigram_hit_rate": round(sum(1 for b in bigrams if b in ref_bigrams) / max(1, len(bigrams)), 3),
        "distinct_bigram_ratio": round(len(set(bigrams)) / max(1, len(bigrams)), 3),
    }


def build_ref_bigrams(n_docs=2000, max_words=300_000):
    """参考 bigram 集：OWT 前 n_docs 篇的 word bigram（词法统计的「语料侧」参照）。"""
    words = []
    with open(DOCS_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_docs or len(words) >= max_words:
                break
            words.extend(json.loads(line)["text"].split())
    return set(zip(words, words[1:]))


# ---------------- WikiText-2 换算盒演示（自训模型跨语料诚实口径） ----------------
@torch.no_grad()
def wt2_eval(model, enc, device, n_windows=24, block=256):
    """自训模型在 WikiText-2 valid 上的 token loss（fp32）+ word PPL 换算盒。"""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    text = "".join(t for t in ds["validation"]["text"])
    ids = enc.encode(text)
    n_words = len(text.split())
    losses = []
    model.eval()
    for i in range(n_windows):
        x = torch.tensor([ids[i * block:(i + 1) * block]], dtype=torch.long, device=device)      # (1,256)
        y = torch.tensor([ids[i * block + 1:(i + 1) * block + 1]], dtype=torch.long, device=device)
        _, loss = model(x, y)
        losses.append(loss.item())
    L = sum(losses) / len(losses)
    ntnw = (len(ids) / n_words)
    return {"n_tokens_eval": n_windows * block, "n_words_full_valid": n_words,
            "nt_over_nw_full_valid": round(ntnw, 4),
            "L_token_nats": round(L, 4), "PPL_token": round(math.exp(L), 1),
            "PPL_word_converted": round(math.exp(ntnw * L), 1),
            "note": "fp32 前向；换算盒 ln PPL_word=(n_t/n_w)*L——跨语料（OWT 训练/WT2 评测）只作诚实对照，非可比基线"}


# ---------------- ICL 探针（选做；官方权重本地前向，不调 API） ----------------
ICL_TASKS = [
    {"task": "capital few-shot",
     "zero": "The capital of Italy is",
     "few": "The capital of France is Paris. The capital of Japan is Tokyo. The capital of Italy is"},
    {"task": "antonym few-shot",
     "zero": "The opposite of hot is",
     "few": "The opposite of hot is cold. The opposite of up is down. The opposite of big is"},
    {"task": "EN->FR pair (GPT-2 paper style)",
     "zero": "English: cheese. French:",
     "few": "English: hello. French: bonjour.\nEnglish: cheese. French: fromage.\nEnglish: dog. French:"},
]


@torch.no_grad()
def icl_probe(model, enc, device):
    """每个任务取两个口径：zero-shot vs few-shot，各记（末位 top-1 token 及其概率）+（贪心续写 8 token）。
    结果无论成败都如实入书——124M 档的 ICL 本来就弱，这是 ch7「ICL 随规模涌现」的地面端读数。"""
    rows = []
    for t in ICL_TASKS:
        row = {"task": t["task"]}
        for tag, p in (("zero_shot", t["zero"]), ("few_shot", t["few"])):
            ids = torch.tensor([enc.encode(p)], dtype=torch.long, device=device)
            logits, _ = model(ids)
            last = logits[0, -1, :]
            nxt = int(last.argmax())
            p_val = float(torch.softmax(last, -1)[nxt])
            cont = generate(model, enc, p, n_new=8, temperature=0.0, device=device)
            row[tag] = {"prompt": p, "greedy_next": enc.decode([nxt]), "p": round(p_val, 4),
                        "greedy_8tok": enc.decode(cont)}
        rows.append(row)
        print(f"  [ICL] {t['task']}: zero -> {row['zero_shot']['greedy_8tok']!r} | "
              f"few -> {row['few_shot']['greedy_8tok']!r}", flush=True)
    return rows


# ---------------- 图 11.3 ----------------
def make_figure_real(rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})
    BLUE, ORANGE, TEAL, YELLOW, GRID, TICK, DARK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e1e0d9", "#898781", "#0b0b0b"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.2), dpi=300, gridspec_kw={"width_ratios": [1.15, 1]})
    # ---- 左：样例表（短标签；官方=青、自训=橙、随机=灰） ----
    short = {"greedy": "greedy", "temp=1.0": "T=1.0", "temp=0.7": "T=0.7", "top-k=40,T=1.0": "top-k40"}
    ax1.axis("off")
    ax1.set_title("generation samples (same prompt, same seed)", fontsize=9, color=DARK, loc="left")
    y = 0.995
    for r in rep["samples"]:
        color = TEAL if r["model"] == "official" else (ORANGE if r["model"] == "self" else TICK)
        txt = r["text"].replace("\n", " ")
        if len(txt) > 62:
            txt = txt[:62] + "…"
        tag = f"{r['model'].replace('random-init', 'rand-init')} {short.get(r['label'].split(' ', 1)[-1], '')}"
        ax1.text(0.0, y, tag, fontsize=6.9, color=color, family="monospace", va="top",
                 transform=ax1.transAxes)
        ax1.text(0.235, y, txt, fontsize=6.6, color=DARK, family="monospace", va="top",
                 transform=ax1.transAxes)
        y -= 0.098
    # ---- 右：词法统计（按解码分组同色系成对排布——解码间不可比、同解码内可比） ----
    rows = rep["lexical"]
    order = ["greedy", "T=1.0", "T=0.7", "top-k40"]
    groups = [[r for r in rows if r["label"].endswith(f" {k}")] for k in order]
    groups.append([r for r in rows if r["label"] == "random-init top-k=40"])
    groups.append([r for r in rows if r["label"] == "OWT corpus ref"])
    colors = {"greedy": BLUE, "T=1.0": TEAL, "T=0.7": YELLOW, "top-k40": ORANGE}
    xs, hs, cs, labs = [], [], [], []
    x = 0
    for gi, grp in enumerate(groups):
        for r in grp:
            key = r["label"].split(" ", 1)[-1] if " " in r["label"] else r["label"]
            key = {"top-k=40,T=1.0": "top-k40", "top-k=40": "top-k40"}.get(key, key)
            cs.append(colors.get(key, "#898781" if "random" in r["label"] else DARK))
            labs.append(r["label"].replace("top-k=40,T=1.0", "top-k40").replace(" top-k=40", " top-k40"))
            hs.append(r["bigram_hit_rate"])
            xs.append(x)
            x += 1
        x += 0.7      # 组间空隙
    ax2.bar(xs, hs, color=cs, width=0.8, zorder=3)
    for xi, r in zip(xs, [row for grp in groups for row in grp]):
        ax2.text(xi, r["bigram_hit_rate"] + 0.02, f"{r['mean_word_len']:.1f}|{r['type_token_ratio']:.2f}",
                 ha="center", fontsize=5.4, color="#52514e", rotation=90)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labs, rotation=42, ha="right", fontsize=5.8, color=TICK)
    ax2.set_ylabel("word-bigram hit rate vs OWT ref", fontsize=8, color=DARK)
    ax2.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax2.tick_params(colors=TICK)
    ax2.set_ylim(0, 1.18)
    ax2.set_title("lexical stats (labels: mean word len | TTR);\ncompare within a decode group only",
                  fontsize=8.5, color=DARK)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-11-3-sampling.png")
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"[figure] -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-steps", type=int, default=t12m.TRAIN_STEPS,
                    help="自训模型步数（默认 6104=fast 档同配，复现 train124_fast 曲线）")
    ap.add_argument("--skip-train", action="store_true", help="已有 self124m.pt 时直接加载")
    ap.add_argument("--train-only", action="store_true", help="只训练存档（长任务先跑）")
    ap.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    device = args.device
    enc = get_tok()
    gen_cpu = torch.Generator().manual_seed(SEED)      # 采样的固定随机源（生成在 CPU 前向，逐 token 可复算）

    # ---- 自训侧：训练（或加载） ----
    if os.path.exists(SELF_PT) and args.skip_train:
        self_model = load_self(device)
        train_meta = {"steps": None, "note": "加载已有 self124m.pt（--skip-train）"}
        print("[self] 加载已有 self124m.pt", flush=True)
    else:
        print(f"===== 自训 124M（{args.train_steps} 步，fast 档同配同种子）=====", flush=True)
        self_model, train_meta = train_self(args.train_steps, device)
    if args.train_only:
        with open(os.path.join(OUT_DIR, "self_train.json"), "w", encoding="utf-8") as f:
            json.dump(train_meta, f, ensure_ascii=False, indent=2)
        return

    rep = {"meta": {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "device": device,
                    "prompt": GEN_PROMPT, "gen_tokens": GEN_TOKENS,
                    "generation": "全量重算口径（每 token 整段前向重算；本册不引入增量缓存）",
                    "top_k_note": "k=40 = GPT-2 附录 Figure 5/Table 7-12 题注口径（Top-k random sampling with k = 40）；"
                                  "k=2 为其摘要特例；top-k 方案出自 Fan et al. 1805.04833",
                    "self_train": train_meta}}
    self_model.eval()

    # ---- 官方侧：对拍四步的①②③复用 ----
    dl = ensure_weights()
    official, _audit = load_official_into_ours(dl["snapshot_dir"])
    official.eval()

    # ---- 采样对照：官方/自训/随机初始化 × 贪心/T=1.0/T=0.7/top-k40 ----
    ref_bigrams = build_ref_bigrams()
    models = [("official", official), ("self", self_model)]
    samples, lexical = [], []
    for name, m in models:
        m_cpu = m.to("cpu")
        for label, kw in [("greedy", dict(temperature=0.0)),
                          ("temp=1.0", dict(temperature=1.0)),
                          ("temp=0.7", dict(temperature=0.7)),
                          ("top-k=40,T=1.0", dict(temperature=1.0, top_k=40))]:
            gen_cpu.manual_seed(SEED)
            ids = generate(m_cpu, enc, GEN_PROMPT, generator=gen_cpu, device="cpu", **kw)
            text = enc.decode(ids)
            tag = f"{name} {label}"
            samples.append({"model": name, "label": tag, "text": text, "kw": kw})
            lexical.append({"label": tag, **lexical_stats(text, ref_bigrams)})
        m.to(device)
    # 随机初始化对照（未训练的 124M：第 10 章的「出厂」状态）
    torch.manual_seed(SEED)
    rand_model = GPT2().eval()
    ids = generate(rand_model, enc, GEN_PROMPT, temperature=1.0, top_k=40, generator=gen_cpu, device="cpu")
    text = enc.decode(ids)
    samples.append({"model": "random-init", "label": "random-init top-k=40", "text": text})
    lexical.append({"label": "random-init top-k=40", **lexical_stats(text, ref_bigrams)})
    # 语料参照行
    with open(DOCS_JSONL, "r", encoding="utf-8") as f:
        ref_text = json.loads(f.readline())["text"][:1200]
    lexical.append({"label": "OWT corpus ref", **lexical_stats(ref_text, ref_bigrams)})
    rep["samples"] = samples
    rep["lexical"] = lexical

    # ---- WikiText-2 换算盒（自训模型跨语料诚实口径） ----
    print("===== WikiText-2 token loss -> word PPL 换算盒 =====", flush=True)
    rep["wt2_box"] = wt2_eval(self_model, enc, device)

    # ---- ICL 探针（官方权重本地前向） ----
    print("===== ICL 探针（few-shot，本地前向）=====", flush=True)
    rep["icl"] = icl_probe(official.to("cpu"), enc, "cpu")

    out_json = os.path.join(OUT_DIR, "sampling.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"[完成] -> {out_json}")
    make_figure_real(rep)


if __name__ == "__main__":
    main()
