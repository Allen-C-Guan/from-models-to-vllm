# 用途：生成 Book1 第 6 章图 6.1「一条 token 的数据流全景」（自绘重制，改绘自 Vaswani et al. 2017 Fig.1）
# 所属章节：Book1《Transformer 原典》第 6 章
# 运行：source env.sh && python "code/ch06/make_fig_6_1.py"
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, FancyArrowPatch

BLUE, CYAN, ORANGE, DARK = "#2a78d6", "#1baf7a", "#eb6834", "#0d366b"
BLUE_F, CYAN_F, ORNG_F = "#cde2fb", "#d8f3e8", "#fde8db"
GRAY, HAIR = "#898781", "#52514e"

fig, ax = plt.subplots(figsize=(7.2, 9.0), dpi=300)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

def box(x, y, w, h, label, ec=BLUE, fc=BLUE_F, fs=7.5, style="round,pad=0.006,rounding_size=0.008", tc="#0b0b0b"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style,
                                ec=ec, fc=fc, lw=1.3, mutation_scale=1))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=tc)
    return (x, y, w, h)

def arrow(p, q, color=DARK, lw=1.4, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11,
                                 color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))

def note(x, y, text, ha="left", fs=6.6, color=HAIR):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color)

EX, DX = 0.30, 0.71          # 编码器 / 解码器列中心
# ---------------- 编码器塔（左） ----------------
box(EX, 0.045, 0.21, 0.045, "Input token ids\n(shape 1×10)", fs=7)
box(EX, 0.125, 0.21, 0.048, "Input Embedding\n× √512")
ax.add_patch(Circle((EX, 0.196), 0.013, ec=DARK, fc="white", lw=1.1))
ax.text(EX, 0.196, "+", ha="center", va="center", fontsize=9, color=DARK)
box(EX - 0.155, 0.196, 0.13, 0.040, "Positional\nEncoding", fs=6.6)
arrow((EX - 0.09, 0.196), (EX - 0.016, 0.196))
arrow((EX, 0.070), (EX, 0.100)); arrow((EX, 0.150), (EX, 0.184))

enc = FancyBboxPatch((EX - 0.115, 0.235), 0.23, 0.40,
                     ec=DARK, fc="#f7fafe", lw=1.2, boxstyle="round,pad=0.006,rounding_size=0.008")
ax.add_patch(enc); ax.text(EX + 0.085, 0.625, "N ×6", ha="center", va="center",
                           fontsize=7.5, color=DARK, style="italic")
box(EX, 0.300, 0.19, 0.052, "Multi-Head\nSelf-Attention", fs=7)
box(EX, 0.372, 0.115, 0.032, "Add & Norm", ec=ORANGE, fc=ORNG_F, fs=6.8)
box(EX, 0.452, 0.19, 0.052, "Feed Forward\n(d_ff = 2048)", ec=CYAN, fc=CYAN_F, fs=7)
box(EX, 0.524, 0.115, 0.032, "Add & Norm", ec=ORANGE, fc=ORNG_F, fs=6.8)
box(EX, 0.596, 0.17, 0.040, "residual: x + Sublayer(x)", fc="white", fs=6.2, tc=HAIR)
arrow((EX, 0.212), (EX, 0.272)); arrow((EX, 0.328), (EX, 0.354))
arrow((EX, 0.390), (EX, 0.424)); arrow((EX, 0.480), (EX, 0.506)); arrow((EX, 0.542), (EX, 0.574))

mem = box(EX, 0.685, 0.20, 0.042, "encoder output\nmemory (1×10×512)", fc="white", fs=7)
arrow((EX, 0.640), (EX, 0.662))
# ---------------- 解码器塔（右） ----------------
box(DX, 0.045, 0.22, 0.045, "Output tokens\nshifted right (1×9)", fs=7)
box(DX, 0.125, 0.22, 0.048, "Output Embedding\n× √512 + PE")
arrow((DX, 0.070), (DX, 0.100)); arrow((DX, 0.150), (DX, 0.212))
dec = FancyBboxPatch((DX - 0.125, 0.235), 0.25, 0.44,
                     ec=DARK, fc="#f7fafe", lw=1.2, boxstyle="round,pad=0.006,rounding_size=0.008")
ax.add_patch(dec); ax.text(DX + 0.094, 0.665, "N ×6", ha="center", va="center",
                           fontsize=7.5, color=DARK, style="italic")
box(DX, 0.300, 0.19, 0.052, "Masked\nSelf-Attention", fs=7)
box(DX, 0.372, 0.115, 0.032, "Add & Norm", ec=ORANGE, fc=ORNG_F, fs=6.8)
box(DX, 0.452, 0.19, 0.052, "Cross-Attention\n(K, V = memory)", fs=7)
box(DX, 0.524, 0.115, 0.032, "Add & Norm", ec=ORANGE, fc=ORNG_F, fs=6.8)
box(DX, 0.596, 0.19, 0.052, "Feed Forward\n(d_ff = 2048)", ec=CYAN, fc=CYAN_F, fs=7)
box(DX, 0.668, 0.115, 0.032, "Add & Norm", ec=ORANGE, fc=ORNG_F, fs=6.8)
arrow((DX, 0.328), (DX, 0.354)); arrow((DX, 0.390), (DX, 0.424))
arrow((DX, 0.480), (DX, 0.506)); arrow((DX, 0.542), (DX, 0.568)); arrow((DX, 0.624), (DX, 0.650))
arrow((EX + 0.012, 0.685), (DX - 0.100, 0.452), color=GRAY, lw=1.2, rad=-0.25)
ax.text(0.505, 0.545, "K, V", ha="center", fontsize=6.6, color=GRAY, rotation=0)
# ---------------- 出口 ----------------
arrow((DX, 0.690), (DX, 0.735))
box(DX, 0.755, 0.20, 0.040, "Linear (shared W_E)", fc="white", fs=7)
arrow((DX, 0.777), (DX, 0.803))
box(DX, 0.825, 0.14, 0.038, "Softmax", fc="white", fs=7.5)
arrow((DX, 0.846), (DX, 0.872))
box(DX, 0.900, 0.22, 0.042, "output logits\n(1×9×37000)", fs=7)
# ---------------- 一条 token 的中轴（贯穿线 + 注意力扇出） ----------------
lane_x, lane_c = EX + 0.062, EX + 0.062
ax.plot([lane_x, lane_x], [0.175, 0.660], color=DARK, lw=1.1, ls=(0, (4, 3)), zorder=5)
ax.plot([DX - 0.062, DX - 0.062], [0.175, 0.872], color=DARK, lw=1.1, ls=(0, (4, 3)), zorder=5)
for (x0, y0, spread) in [(lane_c, 0.300, 1), (DX - 0.062, 0.300, 1),
                         (DX - 0.062, 0.452, 1)]:
    ax.add_patch(Polygon([[x0, y0], [x0 - 0.028, y0 - 0.020], [x0 + 0.028, y0 - 0.020]],
                         closed=True, fc=BLUE, ec="none", alpha=0.5, zorder=4))
ax.text(EX + 0.075, 0.155, "one token\n(512-d vector)", fontsize=6.4, color=DARK, ha="left")
note(EX - 0.128, 0.235, "sublayer shapes →", ha="right", fs=6.2)
# 形状标注（编码器，左缘）
note(EX - 0.130, 0.300, "(1,10,512) → Q,K,V (1,8,10,64)\nscores (1,8,10,10)", ha="right")
note(EX - 0.130, 0.452, "hidden (1,10,2048)", ha="right")
note(EX - 0.130, 0.372, "out (1,10,512)", ha="right")
note(EX - 0.130, 0.125, "(1,10,512)", ha="right")
# 形状标注（解码器，右缘）
note(DX + 0.140, 0.300, "Q,K,V (1,8,9,64)\nscores (1,8,9,9)", ha="left")
note(DX + 0.140, 0.452, "Q (1,9,512)\nK,V (1,10,512)", ha="left")
note(DX + 0.140, 0.596, "hidden (1,9,2048)", ha="left")
note(DX + 0.140, 0.125, "(1,9,512)", ha="left")
# 河宽不变量标注：主干（残差河道）每站末维恒为 d_model=512，2048 只在 FFN 车间内部出现
ax.text(0.505, 0.612, "trunk stays 512 wide\n(at every station;\n2048 only in FFN)",
        ha="center", va="center", fontsize=6.2, color=HAIR)
ax.set_title("One Token’s Data Flow through the Transformer (base: $d_{model}$=512, N=6, h=8)",
             fontsize=9, color="#0b0b0b", pad=10)
fig.tight_layout()
out = Path(__file__).resolve().parents[3] / "drafts" / "Book1-Transformer原典" / "figures" / "fig-6-1-token-dataflow.png"
fig.savefig(out, dpi=300, facecolor="white")
print("saved", out)
