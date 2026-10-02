# -*- coding: utf-8 -*-
# 用途：Book1 第 9 章出图脚本——图 9.1 三模型训练曲线 / 图 9.2 cross-attn 对齐热图 / 图 9.3 外推长度-BLEU 曲线
# 所属章节：《Transformer 原典》第 9 章（从零实现与小型翻译实验）
# 运行：source env.sh && python "code/ch09/make_figures.py" --device mps --figs 1,2,3
#   图 9.1 只读 log/Book1-ch09/*_log.json（秒级）；图 9.2 加载 checkpoint（约 10 s）；
#   图 9.3 含全 test 1000 句 × 三模型 beam 解码（MPS 约 2 分钟，CPU 约 6 倍），
#   另加载 ch10 消融网格的 learned PE checkpoint 补外推两点（约 10 s，依赖 log/Book1-ch10/grid_learnedpe_s0.pt）。
# 产物：figures/fig-9-{1,2,3}-*.png（300 dpi）；
#   图 9.3 的分桶数字落 log/Book1-ch09/fig_9_3_data.json（正文引用以此为准）。
import argparse
import json
import math
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap

from data import (LOG_DIR, ensure_bpe, load_raw, encode_sent, encode_pairs, batches,
                  EXTRAP_BAND)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "figures"))

# ---- 作图规范（丛书统一：分类色固定序 / 蓝色单色渐变热图 / hairline 网格置底层） ----
C = {"transformer": "#2a78d6", "gru_bahdanau": "#eb6834", "gru_noattn": "#1baf7a"}
LABEL = {"transformer": "Transformer (9.42M)", "gru_bahdanau": "GRU+Bahdanau (10.17M)",
         "gru_noattn": "GRU no-attn (10.89M)"}
GRID, TICK, INK, INK2 = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
CMAP = LinearSegmentedColormap.from_list("book_blue", ["#cde2fb", "#0d366b"])


def style_ax(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(TICK)
    ax.tick_params(colors=TICK, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(TICK)


# ---------------- 图 9.1 三模型训练曲线（train loss + val PPL 对数轴） ----------------
def fig_9_1():
    logs = {t: json.load(open(os.path.join(LOG_DIR, f"{t}_log.json"), encoding="utf-8"))
            for t in ("transformer", "gru_bahdanau", "gru_noattn")}
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    for t in ("transformer", "gru_bahdanau", "gru_noattn"):
        h = logs[t]["history"]
        ep = [r["epoch"] for r in h]
        a.plot(ep, [r["train_loss"] for r in h], color=C[t], lw=2, label=LABEL[t],
               marker="o", ms=3.5, mfc=C[t], mec="none")
        b.plot(ep, [r["val_ppl_piece"] for r in h], color=C[t], lw=2, label=LABEL[t],
               marker="o", ms=3.5, mfc=C[t], mec="none")
        # 终点直标（证据等级口径：final val PPL = 30.0 / 46.7 / 217.9）
        b.annotate(f'{h[-1]["val_ppl_piece"]:.1f}', xy=(ep[-1], h[-1]["val_ppl_piece"]),
                   xytext=(3, 0), textcoords="offset points", color=INK2, fontsize=9, va="center")
    # Transformer 最优验证点（epoch 19，PPL 26.9——表 9.2 的「不挑 best」口径在此标出对照）
    ht = logs["transformer"]["history"]
    best = min(ht, key=lambda r: r["val_ppl_piece"])
    b.plot([best["epoch"]], [best["val_ppl_piece"]], marker="o", ms=8, mfc="none",
           mec=C["transformer"], mew=1.6, zorder=5)
    b.annotate(f'best {best["val_ppl_piece"]:.1f} @ ep{best["epoch"]}',
               xy=(best["epoch"], best["val_ppl_piece"]), xytext=(-8, 14),
               textcoords="offset points", color=INK2, fontsize=9, ha="right")
    a.set_title("(a) Training loss (label-smoothed)", color=INK, fontsize=11)
    a.set_xlabel("Epoch")
    a.set_ylabel("loss / piece")
    a.set_xlim(0.5, 30.5)
    b.set_title("(b) Validation PPL (per-piece)", color=INK, fontsize=11)
    b.set_xlabel("Epoch")
    b.set_ylabel("PPL (piece)")
    b.set_yscale("log")
    b.set_xlim(0.5, 30.5)
    b.set_ylim(20, 1e9)
    b.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK2)
    for ax in (a, b):
        style_ax(ax)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-9-1-training-curves.png")
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 9.1] → {out}")


# ---------------- 图 9.2 真翻译模型的对齐热图（Transformer cross-attn vs GRU+Bahdanau） ----------------
SENT_IDX = 21  # test[21]：A teenager plays her trumpet on the field at a game.
#               德语参考把 Trompete 后置、bei einem Spiel 前移——非单调对齐（Bahdanau Fig.3 的真语料版）


def fig_9_2(device):
    from eval import load_ckpt
    sp = ensure_bpe()
    en, de = load_raw()["test"][SENT_IDX]
    src = torch.tensor([encode_sent(sp, en)], device=device)
    tgt = torch.tensor([encode_sent(sp, de)], device=device)

    tm, _ = load_ckpt("transformer", device)
    gm, _ = load_ckpt("gru_bahdanau", device)
    with torch.no_grad():
        a_t = tm.cross_attention_weights(src, tgt, layer=-2)   # 第 3/4 解码层、8 头平均
        a_g = gm.attention_weights(src, tgt)                   # Bahdanau α（逐步一个分布）
    spieces = ["<bos>"] + sp.encode(en, out_type=str) + ["<eos>"]
    tpieces = ["<bos>"] + sp.encode(de, out_type=str) + ["<eos>"]

    vmax = max(a_t.max().item(), a_g.max().item())  # 两面板同一色标——「摊平 vs 尖锐」可直接对读
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.4), dpi=300)
    for ax, A, title in ((axes[0], a_t, "(a) Transformer cross-attn (layer 3/4, head avg)"),
                         (axes[1], a_g, "(b) GRU+Bahdanau  α (additive)")):
        im = ax.imshow(A.numpy(), cmap=CMAP, aspect="auto", vmin=0.0, vmax=vmax)
        ax.set_xticks(range(len(spieces)), spieces, rotation=90, fontsize=7)
        ax.set_yticks(range(len(tpieces)), tpieces, fontsize=7)
        ax.set_xlabel("source pieces", fontsize=9)
        ax.set_ylabel("target pieces", fontsize=9)
        ax.set_title(title, color=INK, fontsize=10)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color(INK2)
        for s in ax.spines.values():
            s.set_color(TICK)
        ax.tick_params(length=2, colors=TICK)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("attention weight", color=INK2, fontsize=8)
        cb.ax.tick_params(labelsize=7, colors=TICK)
        cb.outline.set_color(TICK)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-9-2-alignment-heatmap.png")
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 9.2] → {out}（test[{SENT_IDX}] {en!r} / {de!r}）")


# ---------------- 图 9.3 外推长度-BLEU / PPL 曲线（三模型 × 源句 piece 分桶） ----------------
IN_BUCKETS = ((1, 8), (8, 12), (12, 16), (16, 24))  # 域内：test 天然句按源 piece 长分桶


def _bucket_pairs(sp, pairs, lo, hi):
    return [p for p in pairs if lo < len(sp.encode(p[0], out_type=int)) <= hi]


@torch.no_grad()
def _bucket_ppl(model, sp, pairs, device):
    """teacher-forced per-piece NLL → PPL（nll_loss 与 train.py 同式）。"""
    from train import nll_loss
    tot, n = 0.0, 0
    for src, tgt in batches(encode_pairs(sp, pairs), 64, shuffle=False):
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src, tgt[:, :-1])
        k = int((tgt[:, 1:] != 0).sum())
        tot += nll_loss(logits, tgt[:, 1:]).item() * k
        n += k
    return math.exp(tot / n)


def _load_learnedpe(device):
    """ch10 消融网格的 learned PE 行（18 位位置查表、超长截位复用末位向量）——图 9.3 的学习式外推对照点。"""
    ch10_dir = os.path.normpath(os.path.join(HERE, "..", "ch10"))
    sys.path.insert(0, ch10_dir)
    import ablation  # ch10 网格编排器：LearnedPETransformer/build_model 定义处（其内部自带 ch09 路径注入）
    ckpt = torch.load(os.path.normpath(os.path.join(HERE, "..", "..", "..", "log", "Book1-ch10",
                                                    "grid_learnedpe_s0.pt")),
                      map_location="cpu", weights_only=False)
    model = ablation.build_model(ckpt["cfg"], ensure_bpe().get_piece_size()).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def fig_9_3(device):
    import sacrebleu
    from eval import load_ckpt
    from decode import decode_corpus, BEAM, ALPHA
    from data import ids_to_text

    torch.manual_seed(2017)
    sp = ensure_bpe()
    raw = load_raw()
    test = raw["test"]

    # 外推集构造：与 eval.evaluate_extrap 完全同种子（拼接 286 对 + 天然长句 43 句）
    lo, hi = EXTRAP_BAND
    order = list(range(len(test)))
    random.Random(2017).shuffle(order)
    concat_pairs = []
    for i in range(0, len(order) - 1, 2):
        a, b = test[order[i]], test[order[i + 1]]
        src_s, ref_s = a[0] + " " + b[0], a[1] + " " + b[1]
        if lo <= len(sp.encode(src_s, out_type=int)) <= hi:
            concat_pairs.append((src_s, ref_s))
    natural_pairs = [p for p in test if lo <= len(sp.encode(p[0], out_type=int)) <= hi]
    assert (len(concat_pairs), len(natural_pairs)) == (286, 43), "外推集与 eval.py 口径不一致"

    sets = [(f"({a},{b}]", _bucket_pairs(sp, test, a, b)) for a, b in IN_BUCKETS]
    sets += [("concat[24,32]", concat_pairs), ("natural[24,32]", natural_pairs)]

    data = {"in_buckets": [f"({a},{b}]" for a, b in IN_BUCKETS], "sets": {},
            "models": {}}
    for name, pairs in sets:
        data["sets"][name] = {"n": len(pairs),
                              "mean_src_pieces": round(sum(len(sp.encode(p[0], out_type=int))
                                                           for p in pairs) / len(pairs), 2)}
    for tag in ("transformer", "gru_bahdanau", "gru_noattn"):
        model, _ = load_ckpt(tag, device)
        row_bleu, row_ppl = [], []
        for name, pairs in sets:
            hyp_ids, _ = decode_corpus(model, sp, pairs, device, k=BEAM, alpha=ALPHA)
            hyps = [ids_to_text(sp, ids) for ids in hyp_ids]
            refs = [de for _, de in pairs]
            row_bleu.append(round(sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a").score, 2))
            row_ppl.append(round(_bucket_ppl(model, sp, pairs, device), 2))
        data["models"][tag] = {"bleu_13a": row_bleu, "ppl_piece": row_ppl}
        print(f"[图 9.3] {tag}: BLEU={row_bleu}  PPL={row_ppl}")

    # 学习式位置编码（learned PE）同协议补测外推两点：与正弦并排收口（9.6；数字同 log/Book1-ch10/learnedpe_extrap.json）
    lp = _load_learnedpe(device)
    lp_bleu, lp_ppl = {}, {}
    for name in ("concat[24,32]", "natural[24,32]"):
        pairs = dict(sets)[name]
        hyp_ids, _ = decode_corpus(lp, sp, pairs, device, k=BEAM, alpha=ALPHA)
        hyps = [ids_to_text(sp, ids) for ids in hyp_ids]
        refs = [de for _, de in pairs]
        lp_bleu[name] = round(sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a").score, 2)
        lp_ppl[name] = round(_bucket_ppl(lp, sp, pairs, device), 2)
    ref_json = os.path.normpath(os.path.join(LOG_DIR, "..", "Book1-ch10", "learnedpe_extrap.json"))
    if os.path.exists(ref_json):  # 与 A1 补测 JSON 对账（同协议应逐位一致）
        rj = json.load(open(ref_json, encoding="utf-8"))
        assert (lp_bleu["concat[24,32]"], lp_bleu["natural[24,32]"]) == (
            rj["bleu_extrap"]["extrap_concat[24,32]"]["bleu_13a"],
            rj["bleu_extrap"]["extrap_natural[24,32]"]["bleu_13a"]), "learned PE 外推 BLEU 与 learnedpe_extrap.json 不一致"
    data["models"]["learnedpe_s0"] = {
        "note": "ch10 消融网格 learned PE 行（18×256 位置查表，超长截位复用末位向量）；仅评外推两点"
                "（拼接/天然，同协议同种子），域内四桶未评——学习式与正弦的域内之差见 ch10 表 10.2 判据⑥",
        "bleu_13a": lp_bleu, "ppl_piece": lp_ppl}
    print(f"[图 9.3] learnedpe_s0: BLEU={lp_bleu}  PPL={lp_ppl}")
    with open(os.path.join(LOG_DIR, "fig_9_3_data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    for tag in ("transformer", "gru_bahdanau", "gru_noattn"):
        m = data["models"][tag]
        xs = [data["sets"][n]["mean_src_pieces"] for n, _ in sets]
        for ax, vals, ylab in ((a, m["bleu_13a"], "BLEU (13a)"), (b, m["ppl_piece"], "PPL (per-piece)")):
            ax.plot(xs[:4], vals[:4], color=C[tag], lw=2, marker="o", ms=6, mec="none", label=LABEL[tag])
            ax.plot(xs[3:], vals[3:], color=C[tag], lw=2, ls="--", marker="D", ms=6,
                    mfc="white", mec=C[tag], mew=1.6)  # 空心 = 外推两口径（拼接 / 天然）
    # learned PE：点线 + 实心方点——与三模型实线、正弦的空心菱形虚线段双重区分
    C_LP = "#eda100"  # 丛书分类色序第四位（黄）
    xs_ex = [data["sets"]["concat[24,32]"]["mean_src_pieces"], data["sets"]["natural[24,32]"]["mean_src_pieces"]]
    for ax, vals in ((a, lp_bleu), (b, lp_ppl)):
        ax.plot(xs_ex, [vals["concat[24,32]"], vals["natural[24,32]"]], color=C_LP, lw=2, ls=":",
                marker="s", ms=6, mec="none", zorder=3,
                label="Transformer (learned PE)" if ax is a else None)
    for ax, title in ((a, "(a) BLEU vs source length"), (b, "(b) PPL vs source length")):
        ax.set_title(title, color=INK, fontsize=11)
        ax.set_xlabel("mean source length (BPE pieces)")
        style_ax(ax)
    a.set_ylabel("BLEU (13a)")
    b.set_ylabel("PPL (per-piece)")
    b.set_yscale("log")
    for ax in (a, b):
        ax.axvline(16, color=TICK, lw=1.2, ls=(0, (4, 3)))
        ax.axvspan(24, 32, color="#f0efec", zorder=0)
        ax.set_xlim(4, 31)
    a.annotate("train max len 16", xy=(16, a.get_ylim()[1]), xytext=(16.4, 0.92),
               textcoords=("data", "axes fraction"), color=INK2, fontsize=9)
    a.annotate("extrapolation band [24,32]\nopen = concat / natural\ndotted = learned PE (18-pos table)",
               xy=(25.5, 0.94), textcoords=("data", "axes fraction"), va="top", color=INK2, fontsize=8)
    a.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK2)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-9-3-extrapolation-length-bleu.png")
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[图 9.3] → {out}；分桶数字 → {os.path.join(LOG_DIR, 'fig_9_3_data.json')}")


def main():
    ap = argparse.ArgumentParser(description="ch9 出图脚本（图 9.1/9.2/9.3）")
    ap.add_argument("--figs", default="1,2,3", help="要出的图，逗号分隔（1 只需 log JSON；2/3 需 checkpoint）")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    figs = set(args.figs.split(","))
    if "1" in figs:
        fig_9_1()
    if "2" in figs:
        fig_9_2(torch.device(args.device))
    if "3" in figs:
        fig_9_3(torch.device(args.device))


if __name__ == "__main__":
    main()
