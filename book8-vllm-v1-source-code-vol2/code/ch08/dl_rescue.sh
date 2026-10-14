#!/bin/bash
# curl 断点续传抢救：hf_hub 的 xet 会话在 hf-mirror 上会卡死（新会话 0 字节停滞），
# 改 curl 直连 resolve→CDN 签名链，-C - 断点续传；LFS blob 名=文件 sha256，落位后
# snapshot_download 只做收尾（建快照 symlink，不重下）。
source /home/g00594152/worktree/model_analysis_project/env.sh >/dev/null 2>&1
cd /home/g00594152/worktree/model_analysis_project
B8=log/book8-ch08
B_FP8=log/huggingface/hub/models--Qwen--Qwen3-1.7B-FP8/blobs
B_BF16=log/huggingface/hub/models--Qwen--Qwen3-1.7B/blobs
mkdir -p "$B_BF16"

fetch() { # $1=url $2=part 路径 $3=期望字节数
  local url=$1 part=$2 want=$3 have i
  for i in $(seq 1 90); do
    have=0; [ -f "$part" ] && have=$(stat -c%s "$part")
    [ "$have" -ge "$want" ] && return 0
    echo "[rescue] $(basename "$part") at $have/$want (attempt $i) $(date +%H:%M:%S)"
    timeout 600 curl -sL -C - --max-time 590 -o "$part" "$url" || true
    sleep 3
  done
  have=0; [ -f "$part" ] && have=$(stat -c%s "$part"); [ "$have" -ge "$want" ]
}

# ---- fp8 臂：从旧 xet partial 接续（blob 名即期望 sha256）----
SHA_FP8=8a9a9202762899cfa7c69a59a4d950ba38b3779323b93085dd1971fdf1d2b85a
if [ ! -f "$B_FP8/$SHA_FP8" ]; then
  cp -n "$B_FP8/$SHA_FP8.59693d55.incomplete" "$B_FP8/$SHA_FP8.part" 2>/dev/null
  fetch "https://hf-mirror.com/Qwen/Qwen3-1.7B-FP8/resolve/main/model.safetensors" \
        "$B_FP8/$SHA_FP8.part" 2654425984 || { echo "[rescue] fp8 FAILED"; exit 1; }
  GOT=$(sha256sum "$B_FP8/$SHA_FP8.part" | cut -d' ' -f1)
  echo "[rescue] fp8 sha256=$GOT (expect $SHA_FP8)"
  if [ "$GOT" = "$SHA_FP8" ]; then
    mv "$B_FP8/$SHA_FP8.part" "$B_FP8/$SHA_FP8"
    rm -f "$B_FP8/$SHA_FP8".*.incomplete
    echo "[rescue] fp8 blob placed"
  else
    mv "$B_FP8/$SHA_FP8.part" "$B_FP8/$GOT"; echo "[rescue] fp8 placed under computed sha"
  fi
fi

# ---- bf16 臂：两 shard 下载→sha256 命名落位 ----
declare -A WANT=( [model-00001-of-00002.safetensors]=3441185608 [model-00002-of-00002.safetensors]=622329984 )
for name in model-00001-of-00002.safetensors model-00002-of-00002.safetensors; do
  tmp="$B_BF16/$name.rescue"
  if fetch "https://hf-mirror.com/Qwen/Qwen3-1.7B/resolve/main/$name" "$tmp" "${WANT[$name]}"; then
    GOT=$(sha256sum "$tmp" | cut -d' ' -f1)
    mv "$tmp" "$B_BF16/$GOT"
    echo "[rescue] bf16 $name -> blob $GOT"
  else
    echo "[rescue] bf16 $name FAILED"; exit 1
  fi
done

# ---- 收尾：hf_hub 只做快照 symlink 与小文件 ----
python code/Book8-vLLM源码剖析下/ch08/download_fp8_arms.py --arm both >> "$B8/download.log" 2>&1
grep "OK " "$B8/download.log" || echo "[rescue] snapshot finalize incomplete"
echo "[rescue] DONE"
