# shakespeare.py —— Book2 ch10 §10.4 热身配置（tiny shakespeare 字符级，参考nanoGPT shakespeare_char的主要尺寸）
# 所属章节：Book2 第 10 章 §10.4（drafts/Book2-预训练革命/ch10-大项目I.md 图 10.2 数据源）
# 口径对照（结构对照非照抄；nanoGPT MIT；本轮代码锚3adf61e）：6L/384d/6H（本书扣词嵌入/位置嵌入为10.65M，非nanoGPT默认计数口径）、
#   字符级词表 65、batch 64 × block 256、AdamW(0.9,0.95,wd 0.1)、grad clip 1.0、峰值 lr 1e-3、
#   warmup 100 步 + 余弦退火到 0.1x（日程总长 5000 步）、dropout 0（1M 字符级小语料口径）。
# 基线：nanoGPT README（A100 约 3 min 充分训练 best val 1.4697；MacBook CPU 档约 1.88）——官方仓库口径。
# 运行方式：
#   python shakespeare.py                  # fast 档：1000 步（=5000 步日程的前段，历史约6 min MPS fp32）
#   python shakespeare.py --steps 5000     # full 档：跑完整个余弦日程（历史约30 min MPS fp32）
#   python shakespeare.py --bf16 ...       # bf16 加速档（数值口径 fp32 对拍、bf16同时影响速度与数值舍入）
# 产物：log/book2-ch10/shakespeare-fast.json / shakespeare-full.json
import argparse
from pathlib import Path

import train


def main():
    ap = argparse.ArgumentParser(description="ch10 热身：tiny shakespeare 字符级训练（fast/full 档）")
    ap.add_argument("--steps", type=int, default=1000, help="1000=fast（默认），5000=full")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=train.SEED)
    ap.add_argument("--device", choices=["cpu","mps"], default=None)
    ap.add_argument("--out-dir", type=Path, default=Path(train.OUT_DIR))
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--data-path", type=Path, default=None)
    args = ap.parse_args()
    full = args.steps >= 5000
    train.run_task(argparse.Namespace(
        task="char", steps=args.steps, schedule_total=5000, eval_every=250,
        bf16=args.bf16, device=args.device, seed=args.seed,
        out_dir=args.out_dir, data_path=args.data_path,
        out_name=args.out_name or ("shakespeare-full" if full else "shakespeare-fast")))


if __name__ == "__main__":
    main()
