# Book2第9章：按实际探针报告画内存核算图，并整理训练异常记录。
# 运行：python code/ch09/make_figures.py [--probe log/book2-ch09/memory_probe_new.json]
# 默认仅生成记录表；不造历史曲线，不覆盖正文图；输出到工作目录log。
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def configure():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei"):
        if name in names:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False,
                        "text.color": "#0b0b0b", "axes.labelcolor": "#0b0b0b"})


def fig_9_1(report, target):
    """仅接受本章schema2真实测量；理论报告/未知旧口径不会冒充实测。"""
    if report.get("schema_version") != 2 or report.get("kind") != "measurement":
        raise ValueError("需要schema2实测报告；理论报告或旧汇总不能作为本图实测输入")
    labels = ["模型状态", "线性激活估算", "注意力矩阵估算", "输出侧估算"]
    keys = ("state_gib", "act_linear_gib", "act_attn_gib", "logits_gib")
    theory = [float(report["theory"][key]) for key in keys]
    cfgs = ("fp32", "ckpt_fp32", "bf16")
    peaks = [float(report[c]["peak_current_mib"]) / 1024 for c in cfgs]
    if any(not math.isfinite(v) or v <= 0 for v in theory + peaks):
        raise ValueError("内存值须为有限正数")
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=300)
    bottom = 0.0
    for name, value, color in zip(labels, theory, (BLUE, TEAL, ORANGE, YELLOW)):
        ax.bar(0, value, bottom=bottom, color=color, width=.55, label=name)
        bottom += value
    for x, value in enumerate(peaks, 1):
        ax.bar(x, value, color="#b5b4ad", width=.55, hatch="//", edgecolor="#898781")
    height = max(bottom, *peaks)
    for x, value in enumerate([bottom] + peaks):
        ax.text(x, value + height * .025, f"{value:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(4), ["fp32理论账", "fp32测量", "fp32＋检查点", "bf16测量"])
    ax.set_ylabel("GiB（实测列为current轮询峰值）")
    ax.set_ylim(0, height * 1.16)
    ax.set_title(f"GPT-2 124M内存核算：B={report['batch']}，n={report['seq']}")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=4, fontsize=8, frameon=False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(target, facecolor="white")
    plt.close(fig)


def fig_9_2(target):
    """三份原始记录的五行摘要；无模拟loss轨迹。"""
    rows = [
        ("BERT微调", "小数据集有时不稳定", "不同随机初始化/洗牌重启，选开发集最优"),
        ("PaLM预训练", "540B约20次尖峰，已启用裁剪", "回退约100步，跳过约200—500批"),
        ("PaLM归因", "较早状态重放同批未复现", "证据支持特定批次与参数状态共同作用"),
        ("OPT数值问题", "loss scale下降，末层激活范数尖峰", "回退、调学习率/裁剪，部分重置有效"),
        ("OPT硬件故障", "至少35次手动，估计70余次自动重启", "检查、隔离故障节点，再恢复训练"),
    ]
    fig, ax = plt.subplots(figsize=(10.2, 4), dpi=300)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["记录", "观察", "处置或推论"],
                     colWidths=[.15, .36, .49], cellLoc="left", bbox=[0, .17, 1, .7])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#e1e0d9")
        cell.set_facecolor("#edf3fa" if row == 0 else "white")
    ax.set_title("训练异常：区分观察、处置与因果解释", fontsize=12)
    ax.text(0, .04, "来源：BERT v2 §4.1/A.3；PaLM v5 §5.1；OPT v4 §2.5。\n"
            "不同任务与事件不合并为同一原因；正文另保留OPT逐日日志。", fontsize=8)
    fig.tight_layout()
    fig.savefig(target, facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", type=Path)
    p.add_argument("--out-dir", type=Path, default=Path.cwd() / "log/book2-ch09/figures")
    a = p.parse_args()
    targets = [a.out_dir / "check-9-2-instability-records.png"]
    if a.probe:
        targets.append(a.out_dir / "check-9-1-memory.png")
    for target in targets:
        if target.exists():
            raise FileExistsError(f"已有图不覆盖：{target}")
    report = json.loads(a.probe.read_text()) if a.probe else None
    configure()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    # 先验证可选报告，避免无效输入留下不完整的一组产物。
    if report is not None:
        fig_9_1(report, targets[1])
    fig_9_2(targets[0])
    print("\n".join(map(str, targets)))


if __name__ == "__main__":
    main()
