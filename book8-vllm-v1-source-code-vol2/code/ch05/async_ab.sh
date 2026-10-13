#!/usr/bin/env bash
# async_ab.sh —— ch5 实验：async scheduling on/off 对拍（判据 J1-J5 预注册见 notes/15 §四）
# 纪律：沿 ch2 a_sweep 模板（端口清场/一臂一进程/离线变量）；Qwen3-0.6B 无 spec——隔离 async 单因子。
set -u
cd "$(dirname "$0")/../../.." || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=log/book8-ch05
mkdir -p "$OUT"
MODEL="Qwen/Qwen3-0.6B"
PORT=18454
WORKLOAD="code/Book8-vLLM源码剖析下/ch02/wl_echo.jsonl"

run_arm() {
  local arm="$1"; shift
  pkill -f "port $PORT" 2>/dev/null; sleep 5
  echo "== [asyncAB] arm=$arm $(date +%H:%M:%S)"
  nohup vllm serve "$MODEL" --dtype bfloat16 --max-model-len 1024 --gpu-memory-utilization 0.4 \
    --seed 20261012 --enforce-eager --host 127.0.0.1 --port $PORT "$@" > "$OUT/${arm}_serve.log" 2>&1 &
  local pid=$!
  local code=000
  for i in $(seq 1 120); do sleep 2; code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null); [ "$code" = "200" ] && break; kill -0 "$pid" 2>/dev/null || break; done
  if [ "$code" != "200" ]; then echo "  🔴 $arm 起服失败"; tail -3 "$OUT/${arm}_serve.log" | grep -viE "WARNING|Triton" | head -2; echo "{\"arm\":\"$arm\",\"up\":false}" > "$OUT/${arm}.json"; kill -TERM "$pid" 2>/dev/null; sleep 8; return; fi
  grep -c "Async scheduling" "$OUT/${arm}_serve.log" | xargs echo "  async 行数:"
  vllm bench serve --model "$MODEL" --port $PORT --dataset-name custom --dataset-path "$WORKLOAD" --skip-chat-template \
    --num-prompts 32 --request-rate 8.0 --temperature 0 --seed 20261012 \
    --save-result --result-dir "$OUT" --result-filename "${arm}.json" --percentile-metrics ttft,tpot,itl > "$OUT/${arm}_bench.log" 2>&1
  sleep 5
  curl -s "http://127.0.0.1:$PORT/metrics" | grep -E "^vllm:(num_requests|prefix_cache)" > "$OUT/${arm}_metrics.txt"
  kill -TERM "$pid" 2>/dev/null; for i in $(seq 1 25); do sleep 2; kill -0 "$pid" 2>/dev/null || break; done; sleep 10
  echo "  $arm 完成"
}

run_arm async_off --no-async-scheduling
run_arm async_on  --async-scheduling
echo "async_ab done —— $OUT/"
