# fig_ch09.py —— Book8 ch09 制图：图 9.1 指标族谱（五族+来源分色）
# 所属章节：Book8 ch9（§9.2）；数据源=log/book8-ch09/{off,on}_metrics.txt（2026-10-14 实测）
# 运行：cd <workspace> && source env.sh && python code/Book8-vLLM源码剖析下/ch09/fig_ch09.py
# 产物：drafts/Book8-vLLM源码剖析下/figures/fig-9-1-metric-families.png
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "drafts", "Book8-vLLM源码剖析下", "figures"))
BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#0b0b0b", "#52514e"


def fam_count(path, prefix):
    fams = set()
    try:
        for l in open(path, encoding="utf-8", errors="ignore"):
            if l.startswith("vllm:"):
                fams.add(l.split("{")[0].split()[0])
    except FileNotFoundError:
        pass
    return sum(1 for f in fams if f.startswith(prefix)), len(fams)


def fig_families():
    off_m = "log/book8-ch09/off_metrics.txt"
    on_m = "log/book8-ch09/on_metrics.txt"
    fams = set()
    for p in (off_m, on_m):
        try:
            for l in open(p, encoding="utf-8", errors="ignore"):
                if l.startswith("vllm:"):
                    fams.add(l.split("{")[0].split()[0])
        except FileNotFoundError:
            pass
    groups = [
        ("latency (ttft/tpot/itl/e2e)", lambda f: any(t in f for t in ("ttft", "tpot", "itl", "e2e", "time_to", "time_per"))),
        ("throughput / tokens", lambda f: any(t in f for t in ("generation_tokens", "request_success", "prompt_tokens", "tokens"))),
        ("cache (prefix/kv/offload)", lambda f: any(t in f for t in ("prefix_cache", "kv_cache", "kv_offload", "cache_config"))),
        ("scheduler / requests", lambda f: any(t in f for t in ("num_requests", "iteration", "preempt", "waiting", "running"))),
        ("lifecycle / spec / other", lambda f: True),
    ]
    counts = []
    for name, pred in groups[:-1]:
        c = sum(1 for f in fams if pred(f))
        counts.append((name, c))
    counted = sum(c for _, c in counts)
    counts.append((groups[-1][0], len(fams) - counted))

    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=300)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.6); ax.axis("off")
    ax.text(5, 5.3, f"/metrics family census: {len(fams)} families on one Qwen3-0.6B serve (off-arm snapshot; +15 kv_offload families with connector on)",
            ha="center", fontsize=8.2, color=INK, weight="bold")
    colors = [BLUE, ORANGE, TEAL, YELLOW, "#9b9a94"]
    x = 0.5
    total = sum(c for _, c in counts)
    for (name, c), col in zip(counts, colors):
        w = 9.0 * c / total
        ax.add_patch(FancyBboxPatch((x, 2.6), max(w, 0.35), 1.5, boxstyle="round,pad=0.02",
                                    ec="white", fc=col, lw=1.2, zorder=3))
        ax.text(x + max(w, 0.35) / 2, 3.35, str(c), ha="center", va="center",
                fontsize=10, color="white", weight="bold", zorder=4)
        ax.text(x + max(w, 0.35) / 2, 2.15, name, ha="center", va="top", fontsize=6.6, color=INK, wrap=True)
        x += w
    ax.text(0.5, 1.0, "reading: cache family carries the KV ledgers of ch1/ch7/ch8 (prefix_cache_hits/queries, gpu_kv_usage, +kv_offload_* when connector on);\n"
                      "spec_decode counters only register when speculative_config is set (ch2); labels = {engine, model_name}",
            fontsize=6.8, color=MUTED)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig-9-1-metric-families.png")
    fig.savefig(out, facecolor="white"); plt.close(fig)
    print("[fig]", out)


if __name__ == "__main__":
    fig_families()
