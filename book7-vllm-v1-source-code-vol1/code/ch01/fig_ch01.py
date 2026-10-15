# fig_ch01.py —— Book7 ch01 制图：图 1.1 vLLM 三代引擎演进时间线
# 规格（notes/19 §3.2）：三色时代带（V0 蓝/V1 橙/MRV2 青，alpha 淡填充）；上链 V0→V1 六事件蓝圆点错高、
#   下链 MRV2 事件橙方点；黑虚线=本书阅读锚 v0.30.0；推算点空心+(est.)；图内英文、图注中文（在章稿）。
# 数据源：notes/04 §2.3 七时点 + notes/05 §1.3/§1.4（全部官方一手锚；#11 为节奏推算非官方）。
# 所属章节：Book7 ch01（§1.3）。运行：source env.sh && python code/Book7-vLLM源码剖析上/ch01/fig_ch01.py
# 产物：drafts/Book7-vLLM源码剖析上/figures/fig-1-1-timeline.png
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..",
                                      "drafts", "Book7-vLLM源码剖析上", "figures"))
BLUE, ORANGE, TEAL, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"

# (date, label, chain, stagger, est)  chain: 'up'=V0→V1 上链 / 'lo'=MRV2 下链
EVENTS = [
    (date(2023, 6, 20),  "2023-06 open-sourced\n+PagedAttention (SOSP'23)", "up", 0, False),
    (date(2024, 9, 5),   "v0.6.0: 2.7x (V0 peak)",                          "up", 1, False),
    (date(2025, 1, 27),  "V1 alpha\n(VLLM_USE_V1 opt-in)",                  "up", 2, False),
    (date(2025, 3, 18),  "v0.8.0: V1 default",                             "up", 0, False),
    (date(2025, 5, 22),  "RFC #18571:\nv0.9 = last V0 support",             "up", 1, False),
    (date(2025, 7, 24),  "v0.10.0: V0 removal begins",                      "up", 2, False),
    (date(2026, 3, 24),  "MRV2 blog\n(6,700 -> <1,300 lines)",              "lo", 0, False),
    (date(2026, 9, 9),   "v0.29.0: MRV2 default\nfor all models",           "lo", 1, False),
    (date(2026, 9, 22),  "v0.30.0 (ced6857)\nbook anchor",                  "ln", 0, False),
    (date(2026, 10, 5),  "v0.31.0: MRV2\nmigration complete",               "lo", 2, False),
    (date(2026, 10, 16), "v0.32 (est.):\nMRV1 removal",                     "lo", 3, True),
]
BANDS = [(date(2023, 5, 1), date(2025, 1, 27), BLUE, "V0 era"),
         (date(2025, 1, 27), date(2026, 3, 24), ORANGE, "V1 era"),
         (date(2026, 3, 24), date(2026, 11, 10), TEAL, "MRV2 era")]
Y_UP = [1.55, 1.15, 1.85]   # 上链三档错高
Y_LO = [0.62, 0.30, 0.62, 0.24]


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=300)
    for x0, x1, c, name in BANDS:
        ax.axvspan(x0, x1, color=c, alpha=0.08, zorder=0)
        ax.text(mdates.date2num(x0) + (mdates.date2num(x1) - mdates.date2num(x0)) / 2,
                2.28, name, ha="center", fontsize=8.5, color=c, weight="bold")
    i_up = i_lo = 0
    for d, label, chain, stagger, est in EVENTS:
        x = mdates.date2num(d)
        if chain == "ln":
            ax.axvline(x, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
            ax.text(x + 6, 0.06, "v0.30.0 (ced6857)\nbook anchor",
                    fontsize=6.8, color=INK, weight="bold", va="bottom")
            continue
        if chain == "up":
            y = Y_UP[stagger % 3]
            ax.plot([x], [y], "o", color=BLUE, ms=5.5, zorder=3,
                    mfc="none" if est else BLUE, mec=BLUE)
            ax.annotate(label, (x, y), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=6.0, color=BLUE)
            i_up += 1
        else:
            y = Y_LO[i_lo % 4]
            ax.plot([x], [y], "s", color=ORANGE, ms=5.0, zorder=3,
                    mfc="none" if est else ORANGE, mec=ORANGE)
            ax.annotate(label, (x, y), xytext=(0, -20), textcoords="offset points",
                        ha="center", fontsize=6.0,
                        color=MUTED if est else ORANGE,
                        style="italic" if est else "normal")
            i_lo += 1
    ax.set_ylim(-0.12, 2.46)
    ax.set_yticks([])
    ax.set_xlim(mdates.date2num(date(2023, 5, 1)), mdates.date2num(date(2026, 11, 10)))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(colors="#898781", labelsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#e1e0d9")
    ax.set_title("Three generations of the vLLM engine, 2023-2026 (upper chain: V0 -> V1; lower chain: MRV2)",
                 fontsize=8.2, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig-1-1-timeline.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print("[图 1.1] timeline")


if __name__ == "__main__":
    main()
