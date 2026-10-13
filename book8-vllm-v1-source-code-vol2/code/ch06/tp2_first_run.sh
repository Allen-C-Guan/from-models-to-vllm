#!/usr/bin/env bash
# tp2_first_run.sh —— ch6 实验①：TP 双卡首跑（判据 J1-J7 预注册见 notes/16 §四）
# 判据：J1 起服；J2 进程数=1 API+1 EngineCore+2 Worker（TP2）；J3 ps 表 vs 公式；J4 hccl 两行证据；
#       J5 config 行 TP=2；J6 正确性（贪心两轮一致）；J7 HBM 双卡各占。
set -u
cd "$(dirname "$0")/../../.." || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=log/book8-ch06
mkdir -p "$OUT"
PORT=18464
MODEL="Qwen/Qwen3-0.6B"

pkill -f "port $PORT" 2>/dev/null; sleep 5
echo "== [tp2] $(date +%H:%M:%S)"
nohup vllm serve "$MODEL" --dtype bfloat16 --max-model-len 1024 --gpu-memory-utilization 0.4 \
  --tensor-parallel-size 2 --enforce-eager --seed 20261012 \
  --host 127.0.0.1 --port $PORT > "$OUT/tp2_serve.log" 2>&1 &
PID=$!
echo "$PID" > "$OUT/serve.pid"

code=000
for i in $(seq 1 150); do sleep 2; code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null); [ "$code" = "200" ] && break; kill -0 "$PID" 2>/dev/null || break; done
echo "J1 http=$code"
if [ "$code" != "200" ]; then tail -5 "$OUT/tp2_serve.log" | grep -viE "WARNING|Triton" | head -4; kill -TERM "$PID"; exit 1; fi

echo "== J2/J3 进程拓扑（ps）:"
ps aux | grep -iE "vllm serve|EngineCore|Worker_TP" | grep -v grep | grep -v defunct | awk '{print $2, $11, $12}' | head -8 | tee "$OUT/tp2_ps.txt"
echo "== J4/J5 通信与 config 证据:"
grep -E "world_size|hccl|HCCL" "$OUT/tp2_serve.log" | grep -v WARNING | head -4 | tee "$OUT/tp2_evidence.txt"
grep -oE "tensor_parallel_size=[0-9]+" "$OUT/tp2_serve.log" | head -1 | tee -a "$OUT/tp2_evidence.txt"

echo "== J6 正确性（贪心两轮）:"
for r in 1 2; do
  curl -s "http://127.0.0.1:$PORT/v1/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"one two three four five six seven\",\"max_tokens\":24,\"temperature\":0,\"ignore_eos\":true}" > "$OUT/tp2_round$r.json"
done
python3 -c "
import json
r1 = json.load(open('$OUT/tp2_round1.json'))['choices'][0]['text']
r2 = json.load(open('$OUT/tp2_round2.json'))['choices'][0]['text']
print('J6 两轮一致:', r1 == r2, '| 输出头:', r1[:50])" 2>&1 | tee -a "$OUT/tp2_evidence.txt"

echo "== J7 双卡 HBM:"
npu-smi info | sed -n '/^| 0 /,+1p;/^| 1 /,+1p' | grep -oE "[0-9]+ / 65536" | head -2 | tee -a "$OUT/tp2_evidence.txt"

kill -TERM "$PID" 2>/dev/null
for i in $(seq 1 25); do sleep 2; kill -0 "$PID" 2>/dev/null || break; done
echo "tp2_first_run done"
