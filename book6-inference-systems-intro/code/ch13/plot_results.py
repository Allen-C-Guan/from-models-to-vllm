"""第13章实测曲线；输入results/summary.json，不生成模拟数字。

source ./env.sh; python code/Book6-推理系统导论/ch13/plot_results.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    here = Path(__file__).resolve().parent
    data = json.loads((here / "results/summary.json").read_text())["cases"]
    degrees = [1, 2, 4, 8]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    for i, (key, title) in enumerate([("ttft_ms", "Time to first token"), ("tpot_ms", "Time per output token")]):
        ax = axes[i]
        for index, color, label in [(0, "#2a78d6", "128 input tokens"), (1, "#eb6834", "4096 input tokens")]:
            values = [data[f"tp{p}"]["measurements"][index][key]["p50"] for p in degrees]
            ax.plot(degrees, values, marker="o", markersize=6, linewidth=2, color=color, label=label)
        ax.set_xticks(degrees)
        ax.set_xlabel("Tensor parallel degree")
        ax.set_ylabel("Client p50 (ms)")
        ax.set_title(title, fontsize=11)
        ax.set_axisbelow(True)
        ax.grid(color="#e1e0d9", linewidth=.5)
        ax.tick_params(colors="#52514e")
        ax.legend(fontsize=8)
    fig.suptitle("DeepSeek-V2-Lite / Ascend 910B2 / BF16 eager / concurrency 1", fontsize=10)
    fig.tight_layout()
    root = here.parents[2]
    output = root / "drafts/Book6-推理系统导论/figures/fig-13-19-tp-measured.png"
    if not output.parent.exists():
        # Standalone book layout.
        output = here.parents[1] / "figures/fig-13-19-tp-measured.png"
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
