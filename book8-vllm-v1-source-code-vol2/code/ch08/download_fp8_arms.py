# 用途：Book8 ch08 实验④（fp8 反量化服务观测）的双臂检查点下载
#       fp8 臂 = Qwen/Qwen3-1.7B-FP8（OCP fp8 检查点，Book7 ch10 §10.5 定档）
#       bf16 臂 = Qwen/Qwen3-1.7B（同模型原生精度对照臂）
# 所属章节：code/Book8-vLLM源码剖析下/ch08/
# 运行方式：source env.sh 后 python download_fp8_arms.py [--arm fp8|bf16|both]
#           （经 HF_ENDPOINT=hf-mirror.com 镜像；缓存落 log/huggingface——env.sh 已重定向）
# 说明：磁盘 98% 红线复查过（188G avail，本次 ~6.1G）；用后按写作规范 §1 清 blob 由
#       clean_note 登记在实验底档。

import sys

from huggingface_hub import snapshot_download


def fetch(repo_id: str, tag: str) -> None:
    # 只要 config/tokenizer/权重四类文件；排除原始 bin 备份与 README 减小断流面
    allow = ["*.json", "*.safetensors", "tokenizer*", "merges.txt", "vocab*"]
    path = snapshot_download(
        repo_id=repo_id,
        allow_patterns=allow,
        max_workers=4,
    )
    print(f"[{tag}] OK {repo_id} -> {path}", flush=True)


if __name__ == "__main__":
    arm = sys.argv[sys.argv.index("--arm") + 1] if "--arm" in sys.argv else "both"
    jobs = {
        "fp8": [("Qwen/Qwen3-1.7B-FP8", "fp8")],
        "bf16": [("Qwen/Qwen3-1.7B", "bf16")],
        "both": [("Qwen/Qwen3-1.7B-FP8", "fp8"), ("Qwen/Qwen3-1.7B", "bf16")],
    }[arm]
    for repo_id, tag in jobs:
        fetch(repo_id, tag)
