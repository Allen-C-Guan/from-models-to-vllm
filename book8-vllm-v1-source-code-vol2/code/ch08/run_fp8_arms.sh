#!/bin/bash
# 用途：Book8 ch08 实验④驱动——等双臂下载完成后依序各起一进程跑 fp8/bf16 对拍
#       （一进程一 LLM 纪律；两臂之间留 HBM 慢释放窗口）
# 所属：code/Book8-vLLM源码剖析下/ch08/
# 运行：source env.sh && nohup bash run_fp8_arms.sh > log/book8-ch08/run_arms_driver.log 2>&1 &
set -u
cd "$(dirname "$0")/../../.."   # 回工作区根
source env.sh >/dev/null 2>&1

CARD="${CARD:-6}"
LOG=log/book8-ch08
mkdir -p "$LOG"

echo "[driver] waiting for download (both OK lines) ..."
for i in $(seq 1 360); do   # 最多等 3h（360×30s）
  n=$(grep -c "^\[fp8\] OK\|^\[bf16\] OK" "$LOG/download.log" 2>/dev/null || true)
  [ "$n" -ge 2 ] && break
  sleep 30
done
n=$(grep -c "OK " "$LOG/download.log" 2>/dev/null || true)
if [ "$n" -lt 2 ]; then echo "[driver] TIMEOUT waiting download"; exit 1; fi
echo "[driver] download done: $(grep 'OK ' "$LOG/download.log" | tr '\n' ' ')"

# 体积校验：两臂 safetensors 齐全才跑
python3 - <<'PYEOF'
import glob, json, os, sys
ok = True
for repo, tag in [("Qwen/Qwen3-1.7B-FP8","fp8"), ("Qwen/Qwen3-1.7B","bf16")]:
    base = f"log/huggingface/hub/models--{repo.replace('/','--')}/snapshots"
    snaps = glob.glob(base + "/*")
    st = []
    for s in snaps:
        st += glob.glob(s + "/*.safetensors")
    if not st:
        print(f"[verify] {tag}: NO safetensors"); ok = False; continue
    size = sum(os.path.getsize(p) for p in st)
    idx = glob.glob(base + "/*/model.safetensors.index.json")
    want = None
    if idx:
        want = json.load(open(idx[0]))["metadata"].get("total_size")
    print(f"[verify] {tag}: {len(st)} shards, {size/1e9:.3f} GB on disk, index total_size={want}")
    if want and abs(size - want) > want * 0.02:
        print(f"[verify] {tag}: SIZE MISMATCH"); ok = False
sys.exit(0 if ok else 2)
PYEOF
if [ $? -ne 0 ]; then echo "[driver] verify FAILED"; exit 2; fi

echo "[driver] run arm fp8 on card $CARD"
ASCEND_RT_VISIBLE_DEVICES=$CARD python code/Book8-vLLM源码剖析下/ch08/exp_ch08_fp8_dequant.py --arm fp8 \
  > "$LOG/run_fp8_arm.log" 2>&1
echo "[driver] arm fp8 exit=$? ; HBM slow-release wait 60s"
sleep 60

echo "[driver] run arm bf16 on card $CARD"
ASCEND_RT_VISIBLE_DEVICES=$CARD python code/Book8-vLLM源码剖析下/ch08/exp_ch08_fp8_dequant.py --arm bf16 \
  > "$LOG/run_bf16_arm.log" 2>&1
echo "[driver] arm bf16 exit=$?"

ls -la "$LOG"/*.json 2>/dev/null
echo "[driver] DONE"
