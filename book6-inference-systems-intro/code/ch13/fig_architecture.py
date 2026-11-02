"""第13章：按固定DeepSeek-V2-Lite配置自绘数据流，不复制论文图形。

source ./env.sh; python code/Book6-推理系统导论/ch13/fig_architecture.py
依据官方论文2405.04434与模型revision 604d5664的config；仅表示逻辑依赖。
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main():
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 11,
                         "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#ffffff")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    colors = {"plain": "#edf2f7", "attn": "#dceef5", "moe": "#e7efd9"}

    def box(x, y, w, h, label, color="plain", fontsize=11):
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                     boxstyle="round,pad=0.008,rounding_size=0.008",
                     linewidth=1, edgecolor="#65798b", facecolor=colors[color]))
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize)

    def arrow(a, b, dashed=False):
        ax.annotate("", xy=b, xytext=a,
                    arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#526576",
                                "linestyle": "--" if dashed else "-"})

    ax.text(.5, .975, "DeepSeek-V2-Lite：先认清数据，再讨论把它放到哪张卡", ha="center", fontsize=16)
    ax.text(.18, .91, "一个 MoE 层的主路径", ha="center", fontsize=12)
    labels = [("输入 X  (T, 2048)", "plain"), ("RMSNorm", "plain"),
              ("MLA attention", "attn"), ("加残差", "plain"),
              ("RMSNorm", "plain"), ("MoE 前馈网络", "moe"),
              ("加残差 → 下一层", "plain")]
    ys = [.83, .73, .63, .53, .43, .33, .23]
    for i, ((label, color), y) in enumerate(zip(labels, ys)):
        box(.18, y, .28, .064, label, color)
        if i:
            arrow((.18, ys[i-1]-.034), (.18, y+.034))
    ax.text(.18, .12, "共 27 层\n第 0 层是稠密 FFN\n其余 26 层采用 MoE", ha="center", va="center", fontsize=11)

    ax.text(.68, .91, "MLA：查询头与压缩缓存是不同对象", ha="center", fontsize=12)
    box(.68, .81, .52, .08, "当前 X(T,2048) → Q(T,16,192)\nLite 的 Q 直接投影，没有低秩 Q 压缩", "attn")
    box(.68, .68, .52, .08, "X → latent(T,512) + 位置键(T,64)\n压缩 KV：每层每历史 token 保存 576 个逻辑元素", "attn")
    box(.68, .55, .52, .08, "当前 Q 读取所需历史 KV → attention\n合头、输出投影 → (T,2048)", "attn")
    ax.plot([.95, .978, .978], [.81, .81, .55], color="#526576", lw=1.4)
    arrow((.978, .55), (.95, .55))
    arrow((.68, .635), (.68, .595))
    arrow((.325, .63), (.405, .63), True)

    ax.text(.68, .445, "MoE：全部专家要存，当前 token 只选其中一部分", ha="center", fontsize=12)
    box(.535, .34, .24, .10, "路由专家 64 个\n每 token 选择 6 个\n输入 (n_e,2048)", "moe")
    box(.825, .34, .24, .10, "共享专家 2 个\n每 token 均执行\n输入 (T,2048)", "moe")
    box(.68, .18, .52, .09, "路由输出按原权重归并 + 共享输出\n恢复原 token 顺序与 (T,2048) 形状", "moe")
    arrow((.535, .285), (.60, .23))
    arrow((.825, .285), (.76, .23))
    arrow((.325, .33), (.405, .33), True)
    ax.text(.68, .075, "T：本轮处理的 token 数；n_e：分配给专家 e 的任务数\n逻辑依赖示意；不表示物理卡数、通信 API 或运行时间", ha="center", fontsize=10, color="#526576")
    root = Path(__file__).resolve().parents[3]
    out = root / "drafts/Book6-推理系统导论/figures"
    if not out.exists():
        out = Path(__file__).resolve().parents[2] / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig-13-1-lite-dataflow.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
