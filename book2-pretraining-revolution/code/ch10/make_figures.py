# 用途：Book2 ch10 三张图——图 10.1 三插槽 Block（自绘标形状）/ 图 10.2 热身曲线 / 图 10.3 124M 冒烟
# 所属章节：Book2 第 10 章 §10.2、§10.4、§10.5
# 运行方式：工作区根目录激活env.sh后，python code/ch10/make_figures.py
# 可选--char-report/--smoke-report指定当前完整运行报告；默认仅画结构图。
# 产物：log/book2-ch10/figures/check-10-*.png；不覆盖正式书图或旧日志。

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.getcwd()
FIG_DIR = os.path.join(REPO, "log", "book2-ch10", "figures")
LOG_DIR = os.path.join(REPO, "log", "book2-ch10")

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MUTED = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": TICK, "axes.labelcolor": DARK,
    "xtick.color": TICK, "ytick.color": TICK, "text.color": DARK,
    "axes.linewidth": 0.8, "font.family": "DejaVu Sans",
})


# ---------------- 图 10.1：三插槽 Block（自绘示意级，框线 + 形状标注；整机位置左侧小图） ----------------
def fig_10_1():
    """两残差支路和三类替换组件；形状明确，默认无需训练日志。"""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.axis("off")
    rows = [
        ["input", "X: (B,n,C), after token + position embeddings"],
        ["normalization + attention", "A = Attention(LN1(X)): (B,n,C)"],
        ["first residual", "U = X + A: (B,n,C)"],
        ["normalization + MLP", "F = Linear(GELU(Linear(LN2(U)))): C -> 4C -> C"],
        ["second residual", "Y = U + F: (B,n,C)"],
        ["whole model", "12 Blocks -> final LN -> tied projection: (B,n,V)"],
    ]
    table = ax.table(cellText=rows, colLabels=["location", "operation / shape"],
                     colWidths=[.28,.72], cellLoc="left", bbox=[0,.16,1,.72])
    table.auto_set_font_size(False);table.set_fontsize(8)
    for (i,j),cell in table.get_celld().items():
        cell.set_edgecolor(GRID);cell.set_facecolor("#edf3fa" if i==0 else "white")
    ax.set_title("GPT-2 Block: two residual branches, three component types")
    ax.text(0,.04,"Replaceable types: normalization, attention, MLP. Dropout omitted in this schematic.",
            fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR,"check-10-1-block.png"),facecolor="white")
    plt.close(fig)


# ---------------- 图 10.2：Shakespeare 热身曲线（自产数据） ----------------
def read_curve(path):
    with open(path,encoding="utf-8") as f:
        d=json.load(f)
    if d.get("status")!="complete" or not d.get("curve"):
        raise ValueError("需要完整的当前运行报告；未知旧汇总不补造曲线")
    previous=0
    for row in d["curve"]:
        if row["step"]<=previous or any(not math.isfinite(row[k]) for k in ("train_loss","val_loss")):
            raise ValueError("曲线次序或数值无效")
        previous=row["step"]
    return d


def fig_10_2(path):
    d=read_curve(path)
    if d["meta"]["task"]!="char":
        raise ValueError("热身图要求char报告")
    steps=[c["step"] for c in d["curve"]]
    tr=[c["train_loss"] for c in d["curve"]];val=[c["val_loss"] for c in d["curve"]]
    fig,ax=plt.subplots(figsize=(8,4.5),dpi=300)
    ax.plot(steps,tr,"o-",color=BLUE,lw=2,label="train: interval mean")
    ax.plot(steps,val,"s-",color=ORANGE,lw=2,label="validation: fixed full windows")
    ax.axhline(1.4697,color=YELLOW,ls="--",label="nanoGPT README: 1.4697 (different recipe)")
    best=min(range(len(val)),key=lambda i:val[i])
    ax.set_title(f"Character LM: best observed val {val[best]:.4f} at step {steps[best]}")
    ax.set_xlabel("completed training steps");ax.set_ylabel("nats / character")
    ax.grid(color=GRID,lw=.5);ax.set_axisbelow(True)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.18),fontsize=8,frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR,"check-10-2-char.png"),facecolor="white")
    plt.close(fig)


# ---------------- 图 10.3：124M实际运行报告；初始随机验证批与固定窗口分别画 ----------------
def fig_10_3(path):
    d=read_curve(path)
    if d["meta"]["task"]!="smoke124":
        raise ValueError("124M图要求smoke124报告")
    steps=[c["step"] for c in d["curve"]]
    tr=[c["train_loss"] for c in d["curve"]];val=[c["val_loss"] for c in d["curve"]]
    fig,ax=plt.subplots(figsize=(8,4.5),dpi=300)
    ax.plot(steps,tr,"o-",color=BLUE,lw=2,label="train: interval mean")
    ax.plot(steps,val,"s-",color=ORANGE,lw=2,label="validation: full windows")
    ax.scatter([0],[d["meta"]["init_loss_check"]],color=ORANGE,marker="x",
               label="initial validation random batch")
    ax.axhline(math.log(d["meta"]["model_vocab"]),color=TICK,ls="--",label="uniform prediction")
    entropy=d["corpus_stats"]["train_unigram_entropy_nat"]
    ax.axhline(entropy,color=TEAL,ls=":",label="training unigram entropy (reference)")
    ax.set_xlabel("completed training steps");ax.set_ylabel("nats / token")
    ax.set_title("124M smoke test: observed losses and separate reference values")
    ax.grid(color=GRID,lw=.5);ax.set_axisbelow(True)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.18),fontsize=8,frameon=False,ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR,"check-10-3-smoke.png"),facecolor="white")
    plt.close(fig)


def main():
    global FIG_DIR
    ap=argparse.ArgumentParser()
    ap.add_argument("--char-report")
    ap.add_argument("--smoke-report")
    ap.add_argument("--out-dir",default=FIG_DIR)
    args=ap.parse_args()
    FIG_DIR=args.out_dir
    names=["check-10-1-block.png"]
    if args.char_report:
        read_curve(args.char_report);names.append("check-10-2-char.png")
    if args.smoke_report:
        read_curve(args.smoke_report);names.append("check-10-3-smoke.png")
    for name in names:
        if os.path.exists(os.path.join(FIG_DIR,name)):
            raise FileExistsError("已有核算图不覆盖："+name)
    os.makedirs(FIG_DIR,exist_ok=True)
    fig_10_1()
    if args.char_report:fig_10_2(args.char_report)
    if args.smoke_report:fig_10_3(args.smoke_report)
    print("核算图输出：",FIG_DIR)


if __name__=="__main__":
    main()
