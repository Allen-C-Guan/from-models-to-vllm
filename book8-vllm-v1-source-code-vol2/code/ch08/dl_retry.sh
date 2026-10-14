#!/bin/bash
# 断点续传重试环：hf-mirror 断流时 timeout 强杀再续（snapshot_download 自动 resume）
source /home/g00594152/worktree/model_analysis_project/env.sh >/dev/null 2>&1
cd /home/g00594152/worktree/model_analysis_project
for i in $(seq 1 20); do
  n=$(grep -c "OK " log/book8-ch08/download.log 2>/dev/null || echo 0)
  [ "$n" -ge 2 ] && { echo "[retry] both arms present, exit"; exit 0; }
  echo "[retry] attempt $i $(date +%H:%M:%S)"
  timeout 900 python code/Book8-vLLM源码剖析下/ch08/download_fp8_arms.py --arm both >> log/book8-ch08/download.log 2>&1
  sleep 5
done
echo "[retry] gave up after 20 attempts"
