# 用途：OpenWebText 流式子采样——为 ch8/ch11 scaling 族与 124M 训练预备语料（约 1.1GB 文本 + 独立分词器分片）
# 所属章节：Book2 00-feasibility 探针（ch11/data.py 将在此底座上正式化）
# 运行方式：cd 工作区根目录 && source env.sh && python code/00-feasibility/08_owt_subsample.py --target-mb 1150
# 说明：HF 流式读取（缓存经 env.sh 落 log/），产物写 log/book2-scaling/：owt_docs.jsonl（LM 主语料，从第 skip 篇文档起）
#       owt_tokenizer_corpus.txt（分词器分片，取主语料之前的文档——分词器语料与 LM 语料必须分离）

import argparse
import json
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parents[3] / "log" / "book2-scaling"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-mb", type=float, default=1150.0, help="LM 主语料目标体量（MB）")
    ap.add_argument("--tok-mb", type=float, default=30.0, help="分词器分片体量（MB）")
    ap.add_argument("--skip", type=int, default=6000, help="跳过前 N 篇（前 N 篇留作分词器分片池）")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    tok_path = OUT_DIR / "owt_tokenizer_corpus.txt"
    docs_path = OUT_DIR / "owt_docs.jsonl"
    n_tok_docs = 0
    if not tok_path.exists():  # 分词器分片：取前若干篇（不含 LM 段），写纯文本
        buf = []
        size = 0
        for i, ex in enumerate(ds):
            if i >= args.skip:
                break
            text = ex["text"]
            buf.append(text)
            size += len(text)
            n_tok_docs = i + 1
            if size >= args.tok_mb * 1e6:
                break
        tok_path.write_text("\n\n".join(buf), encoding="utf-8")
        print(f"分词器分片完成：{n_tok_docs} 篇 / {size / 1e6:.1f} MB -> {tok_path}")
        del buf
    else:
        print(f"分词器分片已存在：{tok_path}")

    # LM 主语料：jsonl 保留文档边界（ch11 两种切法对比需要）
    size = 0
    n = 0
    with docs_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if i < args.skip:
                continue
            text = ex["text"]
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            size += len(text)
            n += 1
            if n % 20000 == 0:
                print(f"  已写 {n} 篇 / {size / 1e6:.0f} MB ...")
            if size >= args.target_mb * 1e6:
                break
    print(f"LM 主语料完成：{n} 篇 / {size / 1e6:.1f} MB -> {docs_path}")


if __name__ == "__main__":
    main()
