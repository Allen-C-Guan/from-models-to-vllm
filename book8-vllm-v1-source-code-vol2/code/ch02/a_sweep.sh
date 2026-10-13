#!/usr/bin/env bash
# a_sweep.sh —— ch2 实验 A：ngram 参数扫描驱动（serve 口径，一臂一进程）
# 所属章节：Book8 ch2；设计书=notes/11 §四（判据预注册 J1-J4 含结果分支；八条混淆源逐条处置见彼表）
# 纪律：127.0.0.1+高位口 18021；pid 落档；SIGTERM 干净关停；HBM 慢释放 30-60s；两臂显式
#       --no-async-scheduling + --temperature 0 + --enforce-eager + --no-enable-prefix-caching + --seed。
# 档位：fast=8 轮（off_echo/k3 缺省窗(2,8)/k1×{(2,4),(4,8)}/k3×{(2,4),(4,8)}/random/pr）；
#       full=12 轮（k∈{1,2,3,4}×{(2,4),(4,8)} 全网格+基线+random+pr）。
# 运行：cd <workspace> && source env.sh && ASCEND_RT_VISIBLE_DEVICES=<空闲卡> \
#        bash code/Book8-vLLM源码剖析下/ch02/a_sweep.sh [fast|full]
set -u
cd "$(dirname "$0")/../../.." || exit 1
HERE="code/Book8-vLLM源码剖析下/ch02"
OUT=log/book8-ch02
mkdir -p "$OUT"
MODEL=log/book7-00/hf207m
PORT=18021
TIER="${1:-fast}"

run_arm() {
  local arm="$1" specarg="$2" dsname="$3" dspath="$4"
  echo "== [$TIER] arm=$arm $(date +%H:%M:%S)"
  nohup vllm serve "$MODEL" --dtype bfloat16 --max-model-len 1024 --gpu-memory-utilization 0.4 \
    --enforce-eager --no-async-scheduling --no-enable-prefix-caching --seed 20261012 \
    --host 127.0.0.1 --port $PORT $specarg > "$OUT/a_${arm}_serve.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$OUT/serve.pid"
  local code=000
  for i in $(seq 1 60); do
    sleep 2
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
    [ "$code" = "200" ] && break
    kill -0 "$pid" 2>/dev/null || { echo "  🔴 serve 进程早死（arm=$arm）"; tail -3 "$OUT/a_${arm}_serve.log" 2>/dev/null || tail -3 "$OUT/b_${arm}_serve.log" 2>/dev/null; break; }
  done
  if [ "$code" != "200" ]; then
    echo "  🔴 serve 未就绪（arm=$arm）"; tail -5 "$OUT/a_${arm}_serve.log"
    kill -TERM "$pid" 2>/dev/null; return 1
  fi
  # J1 装配断言（spec 臂）：启动 config 行含 method='ngram' 与所配 k
  if [ -n "$specarg" ]; then
    grep -q "method='ngram'" "$OUT/a_${arm}_serve.log" && echo "  J1 ✓ method='ngram' 装配行在档" || echo "  J1 ✗ 未找到装配行"
    grep "Async scheduling not supported" "$OUT/a_${arm}_serve.log" | head -1 | sed 's/^/  互斥证据: /' || true
  fi
  vllm bench serve --model "$MODEL" --port $PORT \
    --dataset-name "$dsname" ${dspath:+--dataset-path "$dspath"} --skip-chat-template \
    --num-prompts 32 --request-rate 4.0 --temperature 0 --seed 20261012 \
    --save-result --result-dir "$OUT" --result-filename "a_${arm}.json" --percentile-metrics ttft,tpot,itl \
    > "$OUT/a_${arm}_bench.log" 2>&1
  # J2 观测面：spec counter 快照（spec-off 臂应为空——旁证）
  sleep 5  # 等最后一轮统计周期 flush
curl -s "http://127.0.0.1:$PORT/metrics" | grep -E "^vllm:spec_decode" > "$OUT/a_${arm}_metrics.txt"
  # 周期日志行（SpecDecoding metrics——τ/per-pos 原文）
  grep "SpecDecoding metrics" "$OUT/a_${arm}_serve.log" | tail -2 > "$OUT/a_${arm}_speclog.txt" || true
  # 干净关停
  kill -TERM "$pid" 2>/dev/null
  for i in $(seq 1 30); do sleep 2; kill -0 "$pid" 2>/dev/null || break; done
  sleep 8
  echo "  arm=$arm 完成（bench JSON: $OUT/a_${arm}.json）"
}

SPEC_OFF=""
K3_W28="--speculative-config {\"method\":\"ngram\",\"num_speculative_tokens\":3,\"prompt_lookup_min\":2,\"prompt_lookup_max\":8}"
k_w() { echo "--speculative-config {\"method\":\"ngram\",\"num_speculative_tokens\":$1,\"prompt_lookup_min\":$2,\"prompt_lookup_max\":$3}"; }

if [ "$TIER" = "fast" ]; then
  run_arm off_echo      "$SPEC_OFF"       custom "$HERE/wl_echo.jsonl"
  run_arm k3_w28_echo   "$K3_W28"         custom "$HERE/wl_echo.jsonl"
  run_arm k1_w24_echo   "$(k_w 1 2 4)"    custom "$HERE/wl_echo.jsonl"
  run_arm k1_w48_echo   "$(k_w 1 4 8)"    custom "$HERE/wl_echo.jsonl"
  run_arm k3_w24_echo   "$(k_w 3 2 4)"    custom "$HERE/wl_echo.jsonl"
  run_arm k3_w48_echo   "$(k_w 3 4 8)"    custom "$HERE/wl_echo.jsonl"
  run_arm k3_w28_random "$K3_W28"         custom "$HERE/wl_random.jsonl"
  run_arm k3_w28_pr     "$K3_W28"         prefix_repetition ""
else
  run_arm off_echo      "$SPEC_OFF"       custom "$HERE/wl_echo.jsonl"
  run_arm k3_w28_echo   "$K3_W28"         custom "$HERE/wl_echo.jsonl"
  for k in 1 2 3 4; do
    run_arm "k${k}_w24_echo" "$(k_w "$k" 2 4)" custom "$HERE/wl_echo.jsonl"
    run_arm "k${k}_w48_echo" "$(k_w "$k" 4 8)" custom "$HERE/wl_echo.jsonl"
  done
  run_arm k3_w28_random "$K3_W28"         custom "$HERE/wl_random.jsonl"
  run_arm k3_w28_pr     "$K3_W28"         prefix_repetition ""
fi

echo "== 孤儿复查:"; ps aux | grep -iE "vllm serve|EngineCore" | grep -v grep | grep -v defunct | head -3 || echo "  无孤儿"
echo "a_sweep($TIER) done —— 底档 $OUT/"
