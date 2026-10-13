#!/usr/bin/env bash
# combo_matrix.sh —— ch4 实验②：LoRA×APC×图模式×spec 四元对拍（6+1 臂；判据 J1-J5 预注册见 notes/14 §四）
# 纪律：一臂一进程（沿 ch2 a_sweep 模板）；全臂 --no-async-scheduling --temperature 0；M5 失败=守卫定谳读数（非实验失败）。
# 运行：cd <workspace> && source env.sh && ASCEND_RT_VISIBLE_DEVICES=<空卡> \
#        bash code/Book8-vLLM源码剖析下/ch04/combo_matrix.sh
set -u
cd "$(dirname "$0")/../../.." || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # bench 客户端 tokenizer 离线解析（SSL 教训 2026-10-13）
OUT=log/book8-ch04/combo_matrix
mkdir -p "$OUT"
MODEL="Qwen/Qwen3-0.6B"
LORA_DIR=log/book8-00/lora_toy
PORT=18444
WORKLOAD=log/book8-00/wl_echo.jsonl   # 复用 ch2 echo workload（md5 同源跨章可比）
[ -f "$WORKLOAD" ] || { cp "code/Book8-vLLM源码剖析下/ch02/wl_echo.jsonl" "$WORKLOAD"; WORKLOAD="code/Book8-vLLM源码剖析下/ch02/wl_echo.jsonl"; }

run_arm() {
  local arm="$1"; shift
  pkill -f "port $PORT" 2>/dev/null; sleep 5
  echo "== [combo] arm=$arm $(date +%H:%M:%S)"
  nohup vllm serve "$MODEL" --dtype bfloat16 --max-model-len 1024 --gpu-memory-utilization 0.4 \
    --seed 20261012 --no-async-scheduling --enforce-eager --host 127.0.0.1 --port $PORT \
    "$@" > "$OUT/${arm}_serve.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$OUT/serve.pid"
  local code=000
  for i in $(seq 1 180); do
    sleep 2
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
    [ "$code" = "200" ] && break
    kill -0 "$pid" 2>/dev/null || { echo "  🔴 serve 早死（arm=$arm）——守卫读数"; tail -3 "$OUT/${arm}_serve.log" | grep -viE "WARNING|Triton" | head -3; break; }
  done
  if [ "$code" != "200" ]; then
    echo "{\"arm\": \"$arm\", \"up\": false, \"guard\": \"serve_failed\"}" > "$OUT/${arm}.json"
    kill -TERM "$pid" 2>/dev/null; sleep 8; return 0
  fi
  vllm bench serve --model "$MODEL" --port $PORT \
    --dataset-name custom --dataset-path "$WORKLOAD" --skip-chat-template \
    --num-prompts 16 --request-rate 4.0 --temperature 0 --seed 20261012 \
    --save-result --result-dir "$OUT" --result-filename "${arm}.json" \
    --percentile-metrics ttft,tpot,itl > "$OUT/${arm}_bench.log" 2>&1
  sleep 5
  curl -s "http://127.0.0.1:$PORT/metrics" | grep -E "^vllm:(prefix_cache|spec_decode)" > "$OUT/${arm}_metrics.txt"
  kill -TERM "$pid" 2>/dev/null
  for i in $(seq 1 25); do sleep 2; kill -0 "$pid" 2>/dev/null || break; done
  sleep 8
  echo "  arm=$arm 完成"
}

# M0 锚：无任何附加
run_arm M0
# M1 perturb：LoRA 生效臂
run_arm M1 --enable-lora --max-lora-rank 8 --lora-modules "p=$LORA_DIR/toy_perturb"
# M1' noop：判锚臂
run_arm M1n --enable-lora --max-lora-rank 8 --lora-modules "n=$LORA_DIR/toy_noop"
# M2 APC
run_arm M2 --enable-prefix-caching
# M3 图模式（ACLGraph 默认档：非 eager——用 no_eager 变体去掉 --enforce-eager）
nohup vllm serve "$MODEL" --dtype bfloat16 --max-model-len 1024 --gpu-memory-utilization 0.4 \
    --seed 20261012 --no-async-scheduling --host 127.0.0.1 --port $PORT \
    > "$OUT/M3_serve.log" 2>&1 &
M3PID=$!
echo "== [combo] arm=M3（图模式档）pid=$M3PID $(date +%H:%M:%S)"
code3=000
for i in $(seq 1 90); do sleep 2; code3=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null); [ "$code3" = "200" ] && break; kill -0 $M3PID 2>/dev/null || break; done
if [ "$code3" = "200" ]; then
  vllm bench serve --model "$MODEL" --port $PORT --dataset-name custom --dataset-path "$WORKLOAD" --skip-chat-template \
    --num-prompts 16 --request-rate 4.0 --temperature 0 --seed 20261012 \
    --save-result --result-dir "$OUT" --result-filename "M3.json" --percentile-metrics ttft,tpot,itl > "$OUT/M3_bench.log" 2>&1
  sleep 5
  curl -s "http://127.0.0.1:$PORT/metrics" | grep -E "^vllm:(prefix_cache|spec_decode)" > "$OUT/M3_metrics.txt"
  kill -TERM $M3PID 2>/dev/null; for i in $(seq 1 25); do sleep 2; kill -0 $M3PID 2>/dev/null || break; done; sleep 8
else
  echo "  🔴 M3 图模式起服失败——守卫读数"; echo '{"arm": "M3", "up": false, "guard": "graph_mode_failed"}' > "$OUT/M3.json"
  kill -TERM $M3PID 2>/dev/null; sleep 8
fi
# M4 spec ngram（ch2 优选档）
run_arm M4 --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":10,"prompt_lookup_min":5}'
# M5 全组合
run_arm M5 --enable-lora --max-lora-rank 8 --lora-modules "p=$LORA_DIR/toy_perturb" \
  --enable-prefix-caching --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":10,"prompt_lookup_min":5}'

echo "== 孤儿复查:"; ps aux | grep -iE "vllm serve|EngineCore" | grep -v grep | grep -v defunct | head -3 || echo "  无"
echo "combo_matrix done —— $OUT/"
