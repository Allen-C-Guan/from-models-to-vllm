# fig_ch05.py —— Book5 ch5 制图件（图 5.1-5.4，本章专属 slug：gdn/qwen3next 前缀防跨册冲撞）
# 用途：ch05 四张图——
#   图 5.1 线性注意力 vs delta rule 的状态更新对照（例 5.1 同款数据：同键双写——累加 vs 覆写）；
#   图 5.2 GDN chunkwise 双形态（recurrent 逐步递推 vs chunkwise 块内 UT 并行+块间递推，张量框标形状）；
#   图 5.3 Qwen3-Next 3:1 层模式图（48 层=36 GDN+12 全局；partial RoPE 0.25/结构级 NoPE 标注）；
#   图 5.4 scaling 双对数曲线（图数据源=log/book5-ch05/gdn_run1.json 的 scaling.times——自产实物）。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch05/fig_ch05.py"
#   （图 5.4 需先跑 ch05/gdn.py --out-name run1 产出 JSON；缺文件时报错并给命令）
# 产物：figures/fig-5-{1..4}-<slug>.png（不入 git 的是 log/，figures 随书）
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "book5-frontier-hybrid-architectures", "figures")
RUN_JSON = os.path.join(REPO_ROOT, "log", "book5-ch05", "gdn_run1.json")

# 规范 §4 印刷规格常量（与 ch01-04 制图件同款）
BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
DARK, GREY, TICK, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
FILL_L, FILL_O = "#eaf2fc", "#fdf1ea"   # 浅蓝/浅橙底（ch04 同款）


def fig_5_1():
    """图 5.1：S 2×2 同键双写——线性注意力（累加，两代混叠 [8,9]）vs delta rule（覆写，最新值 [3,2]）。"""
    import matplotlib.pyplot as plt

    def grid(ax, x0, y0, S, title, sub, ec):
        # S: 2×2 数值阵（行=值轴 v1/v2，列=键轴 k1/k2）；每格一个数
        for i in range(2):
            for j in range(2):
                ax.add_patch(plt.Rectangle((x0 + j, y0 - i), 0.96, 0.96, facecolor="white",
                                           edgecolor=ec, lw=1.4))
                ax.text(x0 + j + 0.48, y0 - i + 0.5, f"{S[i][j]:g}", ha="center", va="center",
                        fontsize=13, color=DARK)
        ax.text(x0 + 1.0, y0 + 1.05, title, ha="center", fontsize=10.5, color=DARK, weight="bold")
        ax.text(x0 + 1.0, y0 - 1.75, sub, ha="center", fontsize=8.5, color=GREY)
        ax.text(x0 - 0.28, y0 - 0.5, r"$v_1$", ha="right", va="center", fontsize=9, color=TICK)
        ax.text(x0 - 0.28, y0 - 1.5, r"$v_2$", ha="right", va="center", fontsize=9, color=TICK)
        ax.text(x0 + 0.5, y0 - 2.12, r"key axis $k_1\,|\,k_2$", ha="center", fontsize=8, color=TICK)

    fig, ax = plt.subplots(figsize=(8, 3.9), dpi=300, layout="tight")
    ax.axis("off"); ax.set_xlim(-0.6, 11.6); ax.set_ylim(-0.4, 3.6)
    # 公共输入
    ax.text(0.0, 3.28, "same key written twice:  $k_1{=}(1,0),\\ v_1{=}(5,7)$   then   $k_2{=}k_1,\\ v_2{=}(3,2)$",
            fontsize=9.5, color=DARK)
    # 左：线性注意力（只加不减）
    grid(ax, 0.3, 2.35, [[8, 0], [9, 0]],
         "linear attention  $S \\leftarrow S + v\\,k^{\\top}$",
         "read $q{=}k_1$: $Sk_1=(8,9)$\nboth generations mixed (ghost of $v_1$)", ORANGE)
    # 右：delta rule（先删旧再写新）
    grid(ax, 5.6, 2.35, [[3, 0], [2, 0]],
         "delta rule  $S \\leftarrow S(I-\\beta kk^{\\top}) + \\beta vk^{\\top}$",
         "read $q{=}k_1$: $Sk_1=(3,2)$\nlatest value only ($\\beta{=}1$: full overwrite)", BLUE)
    ax.annotate("", xy=(5.45, 1.1), xytext=(4.05, 1.1),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.6))
    ax.text(4.75, 1.32, "add an erase step\nbefore each write", ha="center", fontsize=8.5, color=GREY)
    ax.text(5.75, -0.18, "state $S \\in \\mathbb{R}^{d_v \\times d_k}$ = associative memory "
                         "(rows = value axis, cols = key axis);  toy: $d_k{=}d_v{=}2$",
            fontsize=8.5, color=GREY)
    path = os.path.join(FIG_DIR, "fig-5-1-gdn-accumulate-vs-overwrite.png")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def fig_5_2():
    """图 5.2：GDN 双形态——recurrent（推理：逐步递推）vs chunkwise（训练：块内 UT 并行+块间递推）。"""
    import matplotlib.pyplot as plt

    def box(ax, x, y, w, h, text, fc="white", ec=DARK, fs=7.6, lw=1.3, dashed=False):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw,
                                   linestyle=(0, (4, 3)) if dashed else "-"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=DARK)

    def arrow(ax, x0, y0, x1, y1, color=DARK, lw=1.2):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300, layout="tight")
    ax.axis("off"); ax.set_xlim(0, 16); ax.set_ylim(0, 10.6)
    ax.text(2.3, 10.25, "form 1: recurrent (decode / inference)", fontsize=10, color=ORANGE,
            weight="bold", ha="center")
    box(ax, 0.3, 8.6, 4.0, 1.05, "per token t:  $q_t,k_t$: $(B,H,d_k)$   $v_t$: $(B,H,d_v)$\n"
                                 "$g_t,\\beta_t$: $(B,H)$ scalars per head", FILL_O)
    box(ax, 0.3, 7.0, 4.0, 1.2, "1) forget   $S \\leftarrow S\\cdot e^{g_t}$\n"
                                "$S$: $(B,\\,H,\\,d_k,\\,d_v)$ fixed state", FILL_O)
    box(ax, 0.3, 5.4, 4.0, 1.2, "2) predict   $\\bar v_t = S^{\\top} k_t$: $(B,H,d_v)$", FILL_O)
    box(ax, 0.3, 3.8, 4.0, 1.2, "3) write residual   $\\delta_t=\\beta_t(v_t-\\bar v_t)$\n"
                                "$S \\leftarrow S + k_t\\,\\delta_t^{\\top}$", FILL_O)
    box(ax, 0.3, 2.2, 4.0, 1.2, "4) read   $o_t = S^{\\top} q_t$: $(B,H,d_v)$", FILL_O)
    for y0, y1 in ((8.55, 8.25), (6.95, 6.65), (5.35, 5.05), (3.75, 3.45)):
        arrow(ax, 2.3, y0, 2.3, y1)
    ax.annotate("", xy=(0.15, 8.6), xytext=(0.15, 2.2),
                arrowprops=dict(arrowstyle="-", color=TICK, lw=0.9, linestyle=(0, (2, 2))))
    ax.text(0.02, 5.4, "loop over t = 1..n  (serial, O(1) state per step)", rotation=90,
            fontsize=7.5, color=TICK, va="center", ha="right")
    ax.text(2.3, 1.7, "same math, one token at a time\nno KV cache grows with n", ha="center",
            fontsize=8, color=GREY)

    ax.text(11.4, 10.25, "form 2: chunkwise (training / prefill)", fontsize=10, color=BLUE,
            weight="bold", ha="center")
    box(ax, 5.6, 8.6, 9.8, 1.05, "blocks of C=64 tokens:  $Q,K$: $(B,H,C,d_k)$   $V$: $(B,H,C,d_v)$   "
                                 "$G,\\beta$: $(B,H,C)$   ->  NC = n/C blocks", FILL_L)
    box(ax, 5.6, 6.7, 4.6, 1.5, "intra: UT triangular solve\n$(I+M)u = \\beta V$,  "
                                "$M=\\mathrm{tril}(\\beta KK^{\\top}\\odot \\hat D, -1)$\n"
                                "$u,\\tilde k$: $(B,H,C,d_v)/(B,H,C,d_k)$", FILL_L)
    box(ax, 10.8, 6.7, 4.6, 1.5, "intra-chunk attention\n$(QK^{\\top}\\odot\\hat D)\\,(u-\\tilde k S_w)$\n"
                                 "scores: $(B,H,C,C)$", FILL_L)
    box(ax, 5.6, 5.0, 9.8, 1.3, "inter: $\\mathrm{out}_w=(Q\\odot e^{c})S_w + \\mathrm{intra}$: $(B,H,C,d_v)$;"
                                "   state hand-off  $S_{w+1}=e^{c_C}S_w+k_{\\times}^{\\top}(u-\\tilde k S_w)$",
        FILL_L)
    box(ax, 5.6, 3.4, 9.8, 1.2, "loop over w = 1..NC blocks (linear scan over chunks — "
                                "\"O(n) lives here\")\n$S_w$: $(B,\\,H,\\,d_k,\\,d_v)$ passes block to block",
        "white", dashed=True)
    for y0, y1 in ((8.55, 8.25), (6.65, 6.35), (4.95, 4.65)):
        arrow(ax, 10.5, y0, 10.5, y1)
    arrow(ax, 10.25, 7.45, 10.75, 7.45)
    ax.annotate("", xy=(5.45, 4.0), xytext=(5.45, 4.6),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))
    ax.text(11.4, 2.85, "one chunk = a few big matmuls (parallel over C);\nC steps of delta updates "
                        "condensed by the UT solve", ha="center", fontsize=8, color=GREY)
    ax.text(8.0, 1.9, "two forms, one operator:  recurrent = chunkwise with C = 1 x n blocks"
                      "  —  parity max|Δ| = 1.17e-07 (Table 5.2 / sec 5.7)",
            ha="center", fontsize=8.5, color=DARK)
    ax.text(8.0, 0.9, "mirror of Book4 ch.6 MLA explicit/absorbed: training wants parallelism, "
                      "inference wants a small state", ha="center", fontsize=8.5, color=GREY)
    path = os.path.join(FIG_DIR, "fig-5-2-gdn-chunkwise-dualform.png")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def fig_5_3():
    """图 5.3：Qwen3-Next 80B-A3B 层模式图——48 层 = 12×(3 GDN + 1 full)；partial RoPE 0.25/结构级 NoPE。"""
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 3.6), dpi=300,
                                  gridspec_kw={"height_ratios": [1.0, 0.95]}, layout="tight")
    types = ["linear_attention" if (i + 1) % 4 else "full_attention" for i in range(48)]
    for i, t in enumerate(types):
        ax.add_patch(plt.Rectangle((i, 0), 0.9, 1, facecolor=BLUE if t == "linear_attention" else ORANGE,
                                   edgecolor="none"))
    ax.set_xlim(-0.6, 55.5); ax.set_ylim(-0.1, 1.6); ax.axis("off")
    ax.set_title("Qwen3-Next-80B-A3B   layer_types (48 layers = 12 x (3 GDN + 1 full attention), "
                 "interval 4)", fontsize=9.5, color=DARK, pad=6)
    ax.annotate("layer 3 = first full (0-based: (i+1) % 4 == 0)", xy=(3.4, 1.05), xytext=(9, 1.36),
                fontsize=7.5, color=GREY, arrowprops=dict(arrowstyle="-", color=TICK, lw=0.8))
    ax.text(49.6, 0.5, "36 GDN\n12 full", fontsize=8, color=DARK, va="center")
    zoom = types[:8]
    for i, t in enumerate(zoom):
        ax2.add_patch(plt.Rectangle((i, 0), 0.9, 1, facecolor=BLUE if t == "linear_attention" else ORANGE,
                                    edgecolor="none"))
        ax2.text(i + 0.45, 0.5, "G" if t == "linear_attention" else "F", ha="center", va="center",
                 fontsize=7.5, color="white")
        ax2.text(i + 0.45, -0.32, str(i), ha="center", va="center", fontsize=6.5, color=TICK)
    ax2.set_xlim(-0.6, 8.9); ax2.set_ylim(-0.8, 1.75); ax2.axis("off")
    ax2.text(-0.45, 0.5, "zoom:", fontsize=7.5, color=GREY, va="center", ha="right")
    ax2.text(4.2, 1.5, "G = Gated DeltaNet: structurally NoPE, fixed state 32x128x128 = 524,288 el/layer"
                       "   |   F = full: partial RoPE 64/256 dims (0.25), KV 1,024 el/token/layer unbounded",
             fontsize=7.5, color=GREY, ha="center")
    path = os.path.join(FIG_DIR, "fig-5-3-qwen3next-layermap.png")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def fig_5_4():
    """图 5.4：scaling 双对数曲线——chunkwise vs 满注意力（数据=gdn_run1.json 实测，斜率 1.00/1.85）。"""
    import matplotlib.pyplot as plt
    import numpy as np
    if not os.path.exists(RUN_JSON):
        raise SystemExit(f"缺 {RUN_JSON}——先跑: python \"code/ch05/gdn.py\" --out-name run1")
    with open(RUN_JSON) as f:
        sc = json.load(f)["scaling"]
    ns = np.array(sorted(int(k) for k in sc["times"]), dtype=float)
    chunk = np.array([sc["times"][str(int(n))]["chunk_gdn"] * 1000 for n in ns])
    full = np.array([sc["times"][str(int(n))]["full_attn"] * 1000 for n in ns])
    sl = sc["slopes"]
    fig, ax = plt.subplots(figsize=(6.4, 4), dpi=300, layout="tight")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(ns, chunk, "-o", color=BLUE, lw=2, ms=6, label=f"chunkwise GDN (slope {sl['chunk_gdn']:.2f})")
    ax.plot(ns, full, "-s", color=ORANGE, lw=2, ms=6, label=f"full softmax attention (slope {sl['full_attn']:.2f})")
    for n, c, f_ in zip(ns, chunk, full):
        ax.annotate(f"{c:.0f}", (n, c), textcoords="offset points", xytext=(0, 7), fontsize=7,
                    color=BLUE, ha="center")
        ax.annotate(f"{f_:.1f}", (n, f_), textcoords="offset points", xytext=(0, -12), fontsize=7,
                    color=ORANGE, ha="center")
    ax.set_xlabel("sequence length n (tokens)", fontsize=9, color=DARK)
    ax.set_ylabel("median wall-clock per forward (ms)", fontsize=9, color=DARK)
    ax.set_title("GDN chunkwise vs full attention, log-log (B=2, H=4, d=64, chunk=64, MPS)",
                 fontsize=9.5, color=DARK)
    ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
    ax.tick_params(colors=TICK, labelsize=8)
    ax.legend(fontsize=8, frameon=False)
    ax.text(0.03, 0.05, "teaching implementation (no fused kernels, ~25x slower than fused):\n"
                        "compare slopes only — ~1.0 linear vs ~1.85 quadratic",
            transform=ax.transAxes, fontsize=7.5, color=GREY)
    path = os.path.join(FIG_DIR, "fig-5-4-gdn-scaling-loglog.png")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    for fn in (fig_5_1, fig_5_2, fig_5_3, fig_5_4):
        print(f"[图] {fn.__name__} -> {fn()}")
    print("ch05 fig_ch05.py 完成。")
