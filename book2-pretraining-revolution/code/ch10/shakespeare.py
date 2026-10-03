# shakespeare.py —— Book2 ch10 §10.4 热身配置（tiny shakespeare 字符级，nanoGPT shakespeare_char 同口径）
# 所属章节：Book2 第 10 章 §10.4（ch10-大项目I.md 图 10.2 数据源）
# 口径对照（结构对照非照抄；nanoGPT MIT 2022-12）：6L/384d/6H（非嵌入 10.65M 与其 shakespeare_char 同型）、
#   字符级词表 65、batch 64 × block 256、AdamW(0.9,0.95,wd 0.1)、grad clip 1.0、峰值 lr 1e-3、
#   warmup 100 步 + 余弦退火到 0.1x（日程总长 5000 步）、dropout 0（1M 字符级小语料口径）。
# 基线：nanoGPT README（A100 约 3 min 充分训练 best val 1.4697；MacBook CPU 档约 1.88）——官方仓库口径。
# 运行方式：
#   python shakespeare.py                  # fast 档：1000 步（=5000 步日程的前段，约 5 min MPS fp32）
#   python shakespeare.py --steps 5000     # full 档：跑完整个余弦日程（约 26 min MPS fp32）
#   python shakespeare.py --bf16 ...       # bf16 加速档（数值口径 fp32 对拍、bf16 只影响速度）
# 产物：log/book2-ch10/shakespeare-fast.json / shakespeare-full.json
import argparse

import train


def main():
    ap = argparse.ArgumentParser(description="ch10 热身：tiny shakespeare 字符级训练（fast/full 档）")
    ap.add_argument("--steps", type=int, default=1000, help="1000=fast（默认），5000=full")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=train.SEED)
    args = ap.parse_args()
    full = args.steps >= 5000
    train.run_task(argparse.Namespace(
        task="char", steps=args.steps, schedule_total=5000, eval_every=250,
        bf16=args.bf16, device=None, seed=args.seed,
        out_name="shakespeare-fast" if not full else "shakespeare-full"))


if __name__ == "__main__":
    main()
