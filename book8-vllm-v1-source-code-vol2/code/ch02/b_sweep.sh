#!/usr/bin/env bash
# b_sweep.sh —— ch2 实验 B：drafter 配对 k 扫描（hf207m target + hf_draft2m drafter；serve 口径，一臂一进程）
# 所属章节：Book8 ch2；设计书=notes/11 §四（J1-J4 预注册含结果分支：acceptance 区间 [1.5,2.5] 三分支）
# 纪律：同实验 A（--no-async-scheduling/--temperature 0/--enforce-eager/--no-enable-prefix-caching/--seed/--port 18031 段）
# 档位：fast=3 轮（off/k1/k3）；full=5 轮（off+k∈{1,2,3,4}）。workload=wl_echo.jsonl（与 A 同文件同 md5——跨路线可比）。
# 运行：cd <workspace> && source env.sh && ASCEND_RT_VISIBLE_DEVICES=<空闲卡> \
#        bash code/Book8-vLLM源码剖析下/ch02/b_sweep.sh [fast|full]
set -u
cd "$(dirname "$0")/../../.." || exit 1
HERE="code/Book8-vLLM源码剖析下/ch02"
OUT=log/book8-ch02
mkdir -p "$OUT"
MODEL=log/book7-00/hf207m
DRAFTER=log/book8-00/hf_draft2m
PORT=18031
TIER="${1:-fast}"

run_arm() {
  local arm="$1" specarg="$2"
  echo "== [B/$TIER] arm=$arm $(date +%H:%M:%S)"
  nohup vllm serve "$MODEL" --dtype bfloat16 --max-model-len 1024 --gpu-memory-utilization 0.4 \
    --enforce-eager --no-async-scheduling --no-enable-prefix-caching --seed 20261012 \
    --host 127.0.0.1 --port $PORT $specarg > "$OUT/b_${arm}_serve.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$OUT/serve_b.pid"
  local code=000
  for i in $(seq 1 75); do
    sleep 2
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
    [ "$code" = "200" ] && break
    kill -0 "$pid" 2>/dev/null || { echo "  🔴 serve 进程早死（arm=$arm）"; tail -3 "$OUT/a_${arm}_serve.log" 2>/dev/null || tail -3 "$OUT/b_${arm}_serve.log" 2>/dev/null; break; }
  done
  if [ "$code" != "200" ]; then
    echo "  🔴 serve 未就绪（arm=$arm）"; tail -5 "$OUT/b_${arm}_serve.log"
    kill -TERM "$pid" 2>/dev/null; return 1
  fi
  # J1 双模型装载断言
  if [ -n "$specarg" ]; then
    grep -q "Loading drafter model" "$OUT/b_${arm}_serve.log" && echo "  J1 ✓ Loading drafter 行在档" || echo "  J1 ✗ 缺 drafter 装载行"
    grep "Overriding draft model max model len" "$OUT/b_${arm}_serve.log" | head -1 | sed 's/^/  长度闸: /' || true
  fi
  vllm bench serve --model "$MODEL" --port $PORT \
    --dataset-name custom --dataset-path "$HERE/wl_echo.jsonl" --skip-chat-template \
    --num-prompts 32 --request-rate 4.0 --temperature 0 --seed 20261012 \
    --save-result --result-dir "$OUT" --result-filename "b_${arm}.json" --percentile-metrics ttft,tpot,itl \
    > "$OUT/b_${arm}_bench.log" 2>&1
  sleep 5  # 等最后一轮统计周期 flush
curl -s "http://127.0.0.1:$PORT/metrics" | grep -E "^vllm:spec_decode" > "$OUT/b_${arm}_metrics.txt"
  grep "SpecDecoding metrics" "$OUT/b_${arm}_serve.log" | tail -2 > "$OUT/b_${arm}_speclog.txt" || true
  kill -TERM "$pid" 2>/dev/null
  for i in $(seq 1 30); do sleep 2; kill -0 "$pid" 2>/dev/null || break; done
  sleep 8
  echo "  arm=$arm 完成"
}

dk() { echo "--speculative-config {\"method\":\"draft_model\",\"model\":\"$DRAFTER\",\"num_speculative_tokens\":$1}"; }

if [ "$TIER" = "fast" ]; then
  run_arm b_off ""
  run_arm b_k1 "$(dk 1)"
  run_arm b_k3 "$(dk 3)"
else
  run_arm b_off ""
  for k in 1 2 3 4; do run_arm "b_k$k" "$(dk "$k")"; done
fi

echo "== 孤儿复查:"; ps aux | grep -iE "vllm serve|EngineCore" | grep -v grep | grep -v defunct | head -3 || echo "  无孤儿"
echo "b_sweep($TIER) done —— 底档 $OUT/"
