# exp_sse_capture.sh —— ch11 实验①复现件：serve 起服 + SSE 流式原文采集
# 用途：复现《Book7 第 11 章》11.5 的 15 事件实测（max_tokens=12 → role+12 delta+usage+DONE）
# 所属章节：Book7 ch11 §11.5（读数底档 log/book7-ch11/sse_stream_raw.txt / serve_8021.log）
# 运行方式：source env.sh && ASCEND_RT_VISIBLE_DEVICES=<空闲卡> bash code/Book7-vLLM源码剖析上/ch11/exp_sse_capture.sh <out-name>
#   <out-name> 缺省 sse_repro（产物落 log/book7-ch11/<out-name>_raw.txt；不覆写底档 sse_stream_raw.txt）
# 验收行：grep -c '^data:' log/book7-ch11/<out-name>_raw.txt —— 应得 15（12 token + role + usage + DONE 各一事件）
# 注意：起服约 1-2 分钟；结束后 SIGTERM 收尾（127.0.0.1+高位端口纪律）；一臂一进程。
set -euo pipefail
OUT="${1:-sse_repro}"
LOGDIR=log/book7-ch11
mkdir -p "$LOGDIR"
PORT=8031
MODEL=log/huggingface/Qwen/Qwen3-0.6B   # 本地缓存路径（缓存红线：不经外网）

VLLM_LOGGING_LEVEL=INFO vllm serve "$MODEL" \
  --port "$PORT" --max-model-len 2048 --gpu-memory-utilization 0.4 --enforce-eager \
  > "$LOGDIR/${OUT}_serve.log" 2>&1 &
SERVE_PID=$!
trap 'kill -TERM $SERVE_PID 2>/dev/null || true' EXIT

# 就绪等待（轮询 /v1/models，最长 240 s）
for i in $(seq 1 120); do
  curl -s "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 2
done

# 流式请求：max_tokens=12 + include_usage（与底档同参数；temperature=0 贪心）
curl -sN "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"用一句话介绍杭州"}],"max_tokens":12,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}' \
  > "$LOGDIR/${OUT}_raw.txt"

echo "== 事件计数（验收=15）"
grep -c '^data:' "$LOGDIR/${OUT}_raw.txt"
echo "== 首末事件（对照底档 sse_stream_raw.txt 的 role 头与 [DONE]）"
grep '^data:' "$LOGDIR/${OUT}_raw.txt" | head -1 | head -c 200; echo
grep '^data:' "$LOGDIR/${OUT}_raw.txt" | tail -2 | head -1 | head -c 200; echo
