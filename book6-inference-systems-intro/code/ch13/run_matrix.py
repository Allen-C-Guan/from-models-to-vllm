"""第13章：依次启动、检查、测量、关闭本任务的服务，保留失败记录。

运行前 npu-smi info 选空闲卡；export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; source ./env.sh
python code/Book6-推理系统导论/ch13/run_matrix.py --model-path <snapshot> --cases tp1 tp2 --mode fast --out-name run1
每case独立目录；仅关闭本脚本创建的进程组。不安装、不修改上游。
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import requests

CASES = {
    "tp1": (1, 1, 1, 1, False, False), "tp2": (2, 1, 1, 1, False, False),
    "tp4": (4, 1, 1, 1, False, False), "tp8": (8, 1, 1, 1, False, False),
    "pp2": (1, 2, 1, 1, False, False), "tp2pp2": (2, 2, 1, 1, False, False),
    "tp2dcp2": (2, 1, 1, 2, False, False), "tp4dcp4": (4, 1, 1, 4, False, False),
    "dp2ep2": (1, 1, 2, 1, True, False), "dp2tp2ep4": (2, 1, 2, 1, True, False),
    "dp2tp2ep4sp": (2, 1, 2, 1, True, True), "replicas2": (1, 1, 1, 1, False, False),
}


def monitor(stop, dest):
    """Driver HBM samples include reserved KV and runtime; not allocator peaks."""
    with dest.open("w") as f:
        while not stop.is_set():
            proc = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=15)
            f.write(json.dumps({"time": time.time(), "stdout": proc.stdout}) + "\n")
            f.flush()
            stop.wait(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--cases", nargs="+", choices=CASES, default=["tp1", "tp2", "tp4", "tp8"])
    ap.add_argument("--mode", choices=["smoke", "fast", "full"], default="fast")
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--startup-timeout", type=int, default=900)
    ap.add_argument("--port", type=int, default=8130)
    ap.add_argument("--scheduling", choices=["auto", "sync", "async"], default="auto")
    a = ap.parse_args()
    cards = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
    if list(map(int, cards)) != sorted(set(map(int, cards))):
        raise ValueError("Device list must be unique and ascending")
    root = Path("log/book6-ch13") / a.out_name
    root.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    session.trust_env = False
    for case in a.cases:
        tp, pp, dp, dcp, ep, sp = CASES[case]
        replicas = 2 if case == "replicas2" else 1
        needed = tp * pp * dp * replicas
        if needed > len(cards):
            raise ValueError(f"{case} needs {needed} free devices")
        folder = root / case
        folder.mkdir()
        procs, files, commands, urls = [], [], [], []
        status = {"case": case, "mode": a.mode, "scheduling_requested": a.scheduling,
                  "state": "starting", "commands": commands}
        stop = threading.Event()
        sampler = threading.Thread(target=monitor, args=(stop, folder / "hbm-samples.jsonl"), daemon=True)
        sampler.start()
        try:
            for replica in range(replicas):
                port = a.port + replica
                # Refuse to probe or benchmark someone else's pre-existing server.
                with socket.socket() as port_check:
                    port_check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    port_check.bind(("127.0.0.1", port))
                url = f"http://127.0.0.1:{port}"
                urls.append(url)
                env = os.environ.copy()
                env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(cards[replica * tp * pp * dp:(replica + 1) * tp * pp * dp])
                env["OMP_NUM_THREADS"] = "4"
                env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
                cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", a.model_path,
                       "--served-model-name", "deepseek-v2-lite", "--host", "127.0.0.1", "--port", str(port),
                       "--dtype", "bfloat16", "--enforce-eager", "--max-model-len", "18432" if a.mode == "full" else "8192",
                       "--max-num-seqs", "16", "--max-num-batched-tokens", "2048",
                       "--gpu-memory-utilization", "0.8", "--no-enable-prefix-caching",
                       "--tensor-parallel-size", str(tp), "--pipeline-parallel-size", str(pp),
                       "--data-parallel-size", str(dp), "--decode-context-parallel-size", str(dcp), "--seed", "20261030"]
                if ep:
                    cmd += ["--enable-expert-parallel"]
                if a.scheduling != "auto":
                    cmd += ["--no-async-scheduling" if a.scheduling == "sync" else "--async-scheduling"]
                if sp:
                    cmd += ["--additional-config", '{"enable_flashcomm1": true}']
                commands.append({"argv": cmd, "devices": env["ASCEND_RT_VISIBLE_DEVICES"]})
                f = (folder / f"server-{replica}.log").open("w")
                files.append(f)
                procs.append(subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True))
            deadline = time.monotonic() + a.startup_timeout
            for url in urls:
                while True:
                    if any(p.poll() is not None for p in procs):
                        raise RuntimeError("server exited before ready; see server log")
                    try:
                        if session.get(url + "/health", timeout=2).status_code == 200:
                            break
                    except requests.RequestException:
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError("server startup timed out")
                    time.sleep(2)
            status["state"] = "ready"
            status["benchmark_start_unix"] = time.time()
            subprocess.run([sys.executable, str(Path(__file__).with_name("bench_serving.py")),
                            "--urls", *urls, "--model-path", a.model_path, "--mode", a.mode,
                            "--out-name", a.out_name + "_" + case], check=True, timeout=2400)
            status["benchmark_end_unix"] = time.time()
            status["state"] = "pass"
        except Exception as e:
            status.update(state="failed", error=f"{type(e).__name__}: {e}")
        finally:
            for p in procs:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            for p in procs:
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
            stop.set()
            sampler.join(timeout=20)
            for f in files:
                f.close()
            (folder / "status.json").write_text(json.dumps(status, indent=2))
            print(case, status["state"], status.get("error", ""), flush=True)
            time.sleep(3)
    session.close()


if __name__ == "__main__":
    main()
