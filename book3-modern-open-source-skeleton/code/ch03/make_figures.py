# Book3第3章：前馈计算图、激活函数及真实日志曲线。默认仅画1/3，另存当前目录log。
# 第3.3图(d)是两投影都为恒等的特例：门SiLU(x)、值x、乘积x²sigmoid(x)，不代表一般SwiGLU。
# 第3.2图需真实报告/CSV；旧报告显式--historical，失败或不完整记录不得绘成成功结果。
import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = str(Path.cwd() / 'log/book3-ch03/figures')

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, TICK, DARK, MID = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9.5,
    "axes.edgecolor": TICK, "axes.labelcolor": DARK, "xtick.color": TICK,
    "ytick.color": TICK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "font.family": "sans-serif",
})


def style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------- 图 3.1：门控前馈权路（GPT-2 两矩阵 vs SwiGLU 三矩阵，标形状） ----------------
def fig_3_1():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")
    ax.grid(False)

    def tensor(x, y, w, h, label, sub=""):
        """张量框：浅灰圆角框 + 形状标注。"""
        r = plt.Rectangle((x, y), w, h, fc="#f4f3ef", ec=MID, lw=1.0, zorder=2)
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2 + (2.2 if sub else 0), label, ha="center", va="center",
                fontsize=8.5, color=DARK, zorder=3)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.4, sub, ha="center", va="center",
                    fontsize=7.5, color=MID, zorder=3)

    def mat(x, y, w, h, label, sub, color=BLUE):
        """矩阵/算子框：分类色描边。"""
        r = plt.Rectangle((x, y), w, h, fc="white", ec=color, lw=1.6, zorder=2)
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2 + (2.0 if sub else 0), label, ha="center", va="center",
                fontsize=8.5, color=DARK, zorder=3)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.2, sub, ha="center", va="center",
                    fontsize=7.5, color=MID, zorder=3)

    def arrow(x1, y1, x2, y2, label="", color=MID, dy=1.6, ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, linestyle=ls), zorder=1)
        if label:
            ax.text((x1 + x2) / 2, max(y1, y2) + dy, label, ha="center", va="bottom",
                    fontsize=7.5, color=MID)

    # ---- 上排：GPT-2 MLP（两矩阵，d_ff = 4C）----
    y1 = 36
    ax.text(2, y1 + 13.5, "GPT-2 MLP — 2 matrices, $d_{ff}=4C$  (Book2 ch10)", fontsize=10, color=DARK)
    tensor(1, y1, 9, 8, "x", "(B,n,C)")
    mat(13.5, y1, 10, 8, "c_fc", "C$\\to$4C")
    tensor(26, y1, 11, 8, "h", "(B,n,4C)")
    mat(39.5, y1, 10, 8, "GELU", "(tanh)", color=ORANGE)
    tensor(52, y1, 11, 8, "a", "(B,n,4C)")
    mat(65.5, y1, 10, 8, "c_proj", "4C$\\to$C")
    tensor(78, y1, 9, 8, "y", "(B,n,C)")
    arrow(10, y1 + 4, 13.5, y1 + 4)
    arrow(23.5, y1 + 4, 26, y1 + 4)
    arrow(37, y1 + 4, 39.5, y1 + 4)
    arrow(49.5, y1 + 4, 52, y1 + 4)
    arrow(63, y1 + 4, 65.5, y1 + 4)
    arrow(75.5, y1 + 4, 78, y1 + 4)
    ax.text(89, y1 + 4, "matrix weights\n$2C{\\cdot}4C=8C^2$", fontsize=8, color=MID, va="center")

    # ---- 下排：SwiGLU MLP（三矩阵，d_ff = 8C/3）----
    y2 = 6
    yg, yu = y2 + 14, y2      # gate 支路在上、up 支路在下
    ax.text(2, y2 + 26.5, "SwiGLU MLP — 3 matrices, $d_{ff}=\\frac{8}{3}C$  (this chapter)",
            fontsize=10, color=DARK)
    tensor(1, y2 + 5, 9, 8, "x", "(B,n,C)")
    mat(13.5, yg, 11, 8, "gate_proj", "C$\\to\\frac{8}{3}C$")
    mat(13.5, yu, 11, 8, "up_proj", "C$\\to\\frac{8}{3}C$")
    mat(27, yg, 8, 8, "SiLU", "", color=ORANGE)
    circ = plt.Circle((35.5, y2 + 9), 2.6, fc="white", ec=TEAL, lw=1.8, zorder=2)
    ax.add_patch(circ)
    ax.text(35.5, y2 + 9, "$\\odot$", ha="center", va="center", fontsize=11, color=DARK, zorder=3)
    tensor(41, y2 + 5, 12, 8, "gated", "(B,n,$\\frac{8}{3}$C)")
    mat(56.5, y2 + 5, 11, 8, "down_proj", "$\\frac{8}{3}C\\to$C")
    tensor(71, y2 + 5, 9, 8, "y", "(B,n,C)")
    arrow(10, y2 + 11, 13.5, yg + 4)            # x -> gate 支路
    arrow(10, y2 + 7, 13.5, yu + 4)             # x -> up 支路
    arrow(24.5, yg + 4, 27, yg + 4)
    arrow(35, yg + 4, 35.0, y2 + 10.5)
    ax.annotate("", xy=(33.4, y2 + 10.0), xytext=(24.5, yu + 4),
                arrowprops=dict(arrowstyle="-|>", color=MID, lw=1.3))  # up 支路汇入 ⊙
    ax.text(32.5, y2 + 22.5, "gate path: (B,n,8C/3)", fontsize=7.5, color=MID, ha="center")
    ax.text(28.0, yu - 1.9, "value path: (B,n,8C/3)", fontsize=7.5, color=MID, ha="center")
    arrow(37.6, y2 + 9.6, 41, y2 + 9)
    arrow(53, y2 + 9, 56.5, y2 + 9)
    arrow(67.5, y2 + 9, 71, y2 + 9)
    ax.text(82, y2 + 9, "matrix weights\n$3C{\\cdot}\\frac{8}{3}C=8C^2$\n(bias excluded)", fontsize=8,
            color=MID, va="center")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-3-1-gated-ffn-path.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig 3.1] {out}")


# ---------------- 图 3.3：激活函数四景（ReLU / GELU / SiLU / 门控乘积） ----------------
def fig_3_3():
    x = np.linspace(-4, 4, 801)

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    fig, axs = plt.subplots(2, 2, figsize=(8, 4.6), sharex=True, sharey=False)
    panels = [
        ("(a) ReLU: hard binary gate", np.maximum(0, x), None, BLUE),
        ("(b) GELU: x·Φ(x)", x * 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2))), None, BLUE),
        ("(c) SiLU = Swish (β=1): x·σ(x)", x * sigmoid(x), None, BLUE),
        ("(d) Identity projections: x × SiLU(x)", x * x * sigmoid(x), x * sigmoid(x), BLUE),
    ]
    for ax, (title, main, gate, color) in zip(axs.flat, panels):
        ax.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        if gate is not None:
            ax.plot(x, x, color="#b9b7b0", lw=1.2, label="value x", zorder=1)
            ax.plot(x, gate, color=ORANGE, lw=2, label="gate silu(x)", zorder=3)
            ax.plot(x, main, color=color, lw=2, label="x * silu(x)", zorder=2)
            ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        else:
            ax.plot(x, main, color=color, lw=2)
        ax.axhline(0, color=TICK, lw=0.8)
        ax.axvline(0, color=TICK, lw=0.8)
        ax.set_title(title, fontsize=9, color=DARK)
        ax.set_ylim((-4.6, 16.8) if gate is not None else (-.6, 4.3))
        style_axis(ax)
    for ax in axs[1]:
        ax.set_xlabel("x", fontsize=9)
    for ax in axs[:, 0]:
        ax.set_ylabel("f(x)", fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-3-3-activation-four-panels.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig 3.3] {out}")


# ---------------- 图 3.2：GELU-4C vs SwiGLU-8/3C 消融曲线（参数对齐档，自产） ----------------
def load_curves(path,historical=False):
    report=json.loads(path.read_text());name=report.get('meta',{}).get('out_name')
    if not name or Path(name).name!=name:raise ValueError('报告缺有效out_name')
    data=[]
    for arm in ('gelu4c','swiglu83'):
        rec=report.get('arms',{}).get(arm,{})
        if rec.get('completed') is False:raise ValueError('明确未完成的报告不绘图')
        if rec.get('completed') is not True and not historical:raise ValueError('旧报告需显式--historical')
        steps=rec.get('steps_done',0)
        if steps<=0 or steps!=rec.get('steps_planned') or rec.get('diverged'):raise ValueError('未完成或失败实验不能画成功曲线')
        values=[]
        with (path.parent/f'curve_{arm}_{name}.csv').open() as f:
            for row in csv.DictReader(f):
                if int(row['step'])!=len(values)+1:raise ValueError('CSV步数不连续')
                value=float(row['train_loss'])
                if not math.isfinite(value):raise ValueError('CSV非有限损失')
                values.append(value)
        if len(values)!=steps or not rec.get('evals') or rec['evals'][-1]['step']!=steps:raise ValueError('曲线/报告终点不完整')
        if not isinstance(rec.get('eval_final'),(int,float)) or not math.isfinite(rec['eval_final']):raise ValueError('缺有效最终eval')
        for event in rec['evals']:
            if not math.isfinite(event['eval_loss']):raise ValueError('eval损失非有限')
        data.append((arm,rec,values))
    return data



def fig_3_2(data):
    t_g=(np.arange(1,len(data[0][2])+1),np.asarray(data[0][2]))
    t_s=(np.arange(1,len(data[1][2])+1),np.asarray(data[1][2]))
    e={arm:rec['evals'] for arm,rec,values in data}
    labels={}
    for arm,rec,values in data:
        cfg=rec.get('config',{});key='d_ff_gelu' if arm=='gelu4c' else 'd_ff_swiglu'
        width=cfg.get('d_ff',cfg.get(key))
        labels[arm]=arm+(f' (d_ff={width})' if width is not None else '')
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    ax = axs[0]
    ax.plot(t_g[0], t_g[1], color=BLUE, lw=2, label=labels['gelu4c'])
    ax.plot(t_s[0], t_s[1], color=ORANGE, lw=2, label=labels['swiglu83'])
    ax.set_xlabel("step", fontsize=9)
    ax.set_ylabel("train loss (nat)", fontsize=9)
    ax.set_title("(a) Train loss", fontsize=9.5, color=DARK)
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)

    ax = axs[1]
    for arm, color in (("gelu4c", BLUE), ("swiglu83", ORANGE)):
        ev = e.get(arm, [])
        if not ev:
            continue
        s = np.array([d["step"] for d in ev])
        v = np.array([d["eval_loss"] for d in ev])
        errors=[d.get("eval_se") for d in ev]
        se=np.asarray(errors,dtype=float) if all(v is not None for v in errors) else None
        ax.errorbar(s, v, yerr=se, color=color, lw=2, marker="o", ms=5, capsize=2,
                    label=labels[arm])
    ax.set_xlabel("step", fontsize=9)
    ax.set_ylabel("fixed-window eval loss (nat)", fontsize=9)
    ax.set_title("(b) Fixed-window eval loss", fontsize=9.5, color=DARK)
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig-3-2-ablation-curves.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig 3.2] {out}（只使用给定报告及CSV）")


def main():
    global FIG_DIR
    ap=argparse.ArgumentParser(description='本章计算与函数图；训练曲线只从真实报告/CSV生成，另存log')
    ap.add_argument('--only',default='1,3',help='逗号分隔1,2,3；2需要报告及CSV')
    ap.add_argument('--report',type=Path);ap.add_argument('--historical',action='store_true')
    ap.add_argument('--out-dir',type=Path,default=Path(FIG_DIR));a=ap.parse_args()
    todo=a.only.split(',')
    if not todo or len(set(todo))!=len(todo) or any(x not in {'1','2','3'} for x in todo):raise ValueError('无效图号')
    names={'1':'fig-3-1-gated-ffn-path.png','2':'fig-3-2-ablation-curves.png','3':'fig-3-3-activation-four-panels.png'}
    for n in todo:
        if (a.out_dir/names[n]).exists():raise FileExistsError('不覆盖已有图：'+names[n])
    data=None
    if '2' in todo:
        if a.report is None:raise ValueError('训练曲线需要--report与相邻CSV')
        data=load_curves(a.report,a.historical)
    FIG_DIR=str(a.out_dir);a.out_dir.mkdir(parents=True,exist_ok=True)
    if '1' in todo:fig_3_1()
    if '2' in todo:fig_3_2(data)
    if '3' in todo:fig_3_3()


if __name__=='__main__':main()
