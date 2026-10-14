#!/bin/bash
# 用途：Book8 ch09 实验——kv_offload（NPUOffloadingSpec）on/off 两臂对拍 + /metrics 指标族谱抓取
#       （ch7 E1-b 死于 CPUOffloadingSpec 的 CUDA 断言；本实验换 spec_name=NPUOffloadingSpec——
#        kv_cache_cpu_offload.md 官方 NPU 路径，ch9 的「兑现面」实验）
# 所属：code/Book8-vLLM源码剖析下/ch09/（notes/19 §五）
# 运行：source env.sh && bash exp_ch09_offload_ab.sh   （内部自选卡：环境变量 CARD 可覆盖）
# 判据预注册：J1 off 臂起服（基线）；J2 on 臂起服（NPUOffloadingSpec 生效——失败则登记为
#   「文档在册路径不可行」负结果）；J3 两臂贪心输出非空一致口径；J4 /metrics 抓取两份
#   （vllm: 前缀指标族计数——表 9.1 数据源）；J5 offload 观测行（connector 创建行）。
set -u
cd "$(dirname "$0")/../../.."
source env.sh >/dev/null 2>&1
CARD="${CARD:-6}"
LOG=log/book8-ch09
mkdir -p "$LOG"

run_arm () {  # $1=arm名 $2=kv-transfer-config JSON 或 "-"
  local arm="$1" cfg="$2"
  echo "[$arm] starting on card $CARD $(date +%H:%M:%S)"
  if [ "$cfg" = "-" ]; then
    ASCEND_RT_VISIBLE_DEVICES=$CARD timeout 600 vllm serve Qwen/Qwen3-0.6B \
      --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.4 \
      --enforce-eager --no-async-scheduling --seed 20261013 \
      --host 127.0.0.1 --port 18091 > "$LOG/${arm}_serve.log" 2>&1 &
  else
    ASCEND_RT_VISIBLE_DEVICES=$CARD timeout 600 vllm serve Qwen/Qwen3-0.6B \
      --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.4 \
      --enforce-eager --no-async-scheduling --seed 20261013 \
      --host 127.0.0.1 --port 18091 \
      --kv-transfer-config "$cfg" > "$LOG/${arm}_serve.log" 2>&1 &
  fi
  local pid=$!
  echo $pid > "$LOG/${arm}.pid"
  # readiness 轮询
  for i in $(seq 1 90); do
    curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:18091/v1/models 2>/dev/null | grep -q 200 && break
    sleep 2
  done
  if ! curl -s http://127.0.0.1:18091/v1/models | grep -qi qwen; then
    echo "[$arm] SERVE FAILED"
    kill -TERM $pid 2>/dev/null; sleep 10; kill -9 $pid 2>/dev/null; sleep 40
    return 1
  fi
  echo "[$arm] serve up"
  # 负载：共享前缀 workload（4 前缀×4 请求——offload 有东西可卸）
  for j in 1 2 3 4; do
    for k in 1 2 3 4; do
      curl -s http://127.0.0.1:18091/v1/completions -H "Content-Type: application/json" \
        -d "{\"prompt\":\"The quick brown fox jumps over the lazy dog number $j. This is a shared prefix for offload testing, repeated text block $j.\",\"max_tokens\":24,\"temperature\":0}" \
        >> "$LOG/${arm}_gen.jsonl" 2>/dev/null
      echo >> "$LOG/${arm}_gen.jsonl"
    done
  done
  # /metrics 抓取（两份：负载后）
  curl -s http://127.0.0.1:18091/metrics > "$LOG/${arm}_metrics.txt"
  grep -c "^vllm:" "$LOG/${arm}_metrics.txt" | xargs echo "[$arm] vllm: metric lines:"
  grep -E "^vllm:(prefix_cache|spec_decode|num_requests|gpu_kv)" "$LOG/${arm}_metrics.txt" | head -12 >> "$LOG/${arm}_keymetrics.txt"
  # 关停
  kill -TERM $pid 2>/dev/null
  for i in $(seq 1 15); do kill -0 $pid 2>/dev/null || break; sleep 2; done
  kill -9 $pid 2>/dev/null
  sleep 45   # HBM 慢释放
  echo "[$arm] done $(date +%H:%M:%S)"
}

OFF_CFG='{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":4294967296,"blocks_per_chunk":8,"spec_name":"NPUOffloadingSpec","spec_module_path":"vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.npu"}}'

run_arm off "-" || true
run_arm on "$OFF_CFG" || true

python3 - <<'PYEOF'
import json, re, os
LOG='log/book8-ch09'
rec={"probe":"exp_ch09_offload_ab","runtime":"vllm 0.27.1+vllm-ascend 0.27.1rc1（Atlas 910B3）","date":"2026-10-14"}
for arm in ('off','on'):
    a={"serve_log":f"{LOG}/{arm}_serve.log"}
    lg=open(f"{LOG}/{arm}_serve.log",encoding='utf-8',errors='ignore').read()
    a["connector_line"]=next((l.strip()[:120] for l in lg.split('\n') if 'Creating v1 connector' in l), None)
    a["spec_line"]=next((l.strip()[:120] for l in lg.split('\n') if 'Creating offloading spec' in l or 'OffloadingSpec' in l), None)
    m=open(f"{LOG}/{arm}_metrics.txt",encoding='utf-8',errors='ignore').read() if os.path.exists(f"{LOG}/{arm}_metrics.txt") else ''
    fams={}
    for l in m.split('\n'):
        if l.startswith('vllm:'):
            name=l.split('{')[0].split()[0]
            fams[name]=fams.get(name,0)+1
    a["metric_families"]=len(fams)
    a["metric_lines"]=sum(fams.values())
    key={k:v for k,v in fams.items() if any(t in k for t in ('prefix','requests','gpu_kv','iteration'))}
    a["key_family_sample"]=dict(list(key.items())[:10])
    gen=open(f"{LOG}/{arm}_gen.jsonl",encoding='utf-8',errors='ignore').read() if os.path.exists(f"{LOG}/{arm}_gen.jsonl") else ''
    outs=re.findall(r'"text":\s*"([^"]{0,40})',gen)
    a["gen_count"]=len(outs); a["gen_head"]=outs[0][:40] if outs else None
    rec[arm]=a
json.dump(rec,open(f"{LOG}/exp_ch09_offload_ab.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
print("wrote",f"{LOG}/exp_ch09_offload_ab.json")
print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk in('connector_line','spec_line','metric_families','metric_lines','gen_count','gen_head')} for k,v in rec.items() if isinstance(v,dict)},ensure_ascii=False,indent=1))
PYEOF
echo "ALL DONE"
