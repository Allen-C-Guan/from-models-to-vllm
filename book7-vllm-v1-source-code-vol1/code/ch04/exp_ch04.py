# exp_ch04.py —— Book7 ch04 代码件：进程拓扑实测 + StartPlan 开关对拍
# 用途：ch04《EngineCore 进程模型》两实验——
#   实验① ps 拓扑：起 vllm serve（单卡），从进程表数 VLLM:: 前缀进程，对照公式 A+DP+N(+1)——
#     预注册判据（notes/13 §8.1 的 uniproc 洞见）：物理进程 = APIServer + EngineCore（worker 角色
#     被吸收进 EngineCore——world_size==1 时 distributed_executor_backend 自动 "uni"），
#     「公式数逻辑角色、ps 数物理进程」；
#   实验② StartPlan 对拍：三连跑 offline LLM 初始化——run1（=1 落 plan）/run2（=1 用 plan）/run3（=0 全量
#     profiling）——判据：run2 的 ready 时间 < run3（跳过 memory-profiling 与图捕获估计 pass）。
# 所属章节：Book7 ch04（§4.2/§4.5 实验）；设计书=notes/13 §8。
# 运行：cd <workspace> && source env.sh && npu-smi info 选卡 && ASCEND_RT_VISIBLE_DEVICES=<卡> \
#        python code/Book7-vLLM源码剖析上/ch04/exp_ch04.py [--skip-ps] [--skip-startplan]
# 产物：log/book7-ch04/{ps_topology.json, startplan_<n>.json, exp_ch04_summary.json}
# 纪律：__main__ 守卫；起服进程 terminate 收尾；--out-name 防覆写。
import argparse
import json
import os
import subprocess
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MODEL = os.path.join(WS, "log", "book7-00", "hf207m")
OUT = os.path.join(WS, "log", "book7-ch04")
PORT = 8023
CACHE_PLAN_DIR = os.path.join(WS, "log", "vllm", "startup_plan")


def ps_topology() -> dict:
    serve_log = os.path.join(OUT, "serve_topology.log")
    with open(serve_log, "w") as lf:
        proc = subprocess.Popen(["vllm", "serve", MODEL, "--port", str(PORT), "--max-model-len", "1024",
                                 "--gpu-memory-utilization", "0.4", "--enforce-eager"],
                                stdout=lf, stderr=subprocess.STDOUT, cwd=WS)
    try:
        t0 = time.perf_counter()
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                if proc.poll() is not None:
                    return dict(error=f"serve died, see {serve_log}")
                if time.perf_counter() - t0 > 420:
                    return dict(error="health timeout")
                time.sleep(3)
        load_s = round(time.perf_counter() - t0, 1)
        # ps 实测：抓 VLLM:: 前缀进程与 vllm 相关 python 进程
        ps_out = subprocess.run(["ps", "-eo", "pid,ppid,cmd"], capture_output=True, text=True).stdout
        vllm_procs = [ln.strip() for ln in ps_out.splitlines()
                      if ("VLLM::" in ln or "EngineCore" in ln or ("api_server" in ln and "python" in ln)
                          or ("vllm" in ln and "serve" in ln and "python" in ln))]
        # 进程标题：setproctitle 改的名从 /proc/<pid>/cmdline 直接可见
        titles = subprocess.run(["bash", "-c",
                                 r"grep -l 'VLLM::' /proc/[0-9]*/cmdline 2>/dev/null | while read f; do"
                                 r"  tr '\0' ' ' < $f | cut -c1-80; echo; done"],
                                capture_output=True, text=True).stdout.strip().splitlines()
        res = dict(load_s=load_s, proc_titles=titles, ps_lines=vllm_procs[:12],
                   n_titled=len(titles), expect="APIServer+EngineCore（worker 被 uniproc 吸收）",
                   runtime="vllm serve 0.27.1 + vllm-ascend（Atlas 910B3）")
        return res
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()


def one_offline(startplan: str, tag: str) -> dict:
    env = dict(os.environ)
    env["VLLM_ENABLE_STARTUP_PLAN"] = startplan
    env["VLLM_CACHE_ROOT"] = os.path.join(WS, "log", "vllm")   # plan 落 log/（缓存红线）
    log_p = os.path.join(OUT, f"startplan_{tag}.log")
    t0 = time.perf_counter()
    with open(log_p, "w") as lf:
        r = subprocess.run(["python", "-c",
                            "from vllm import LLM, SamplingParams\n"
                            f"llm = LLM(model=r'{MODEL}', dtype='bfloat16', max_model_len=1024,"
                            " gpu_memory_utilization=0.4, enforce_eager=True)\n"
                            "llm.generate(['hi'], SamplingParams(max_tokens=2))\n"
                            "print('READY')"],
                           env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=WS, timeout=600)
    wall = round(time.perf_counter() - t0, 1)
    log = open(log_p, encoding="utf-8", errors="ignore").read()
    return dict(tag=tag, startplan=startplan, wall_s=wall, exit=r.returncode,
                startup_plan_lines=[ln.strip() for ln in log.splitlines() if "startup" in ln.lower()][:4],
                profile_lines=[ln.strip()[:90] for ln in log.splitlines() if "rofil" in ln][:3])


def main() -> None:
    ap = argparse.ArgumentParser(description="ch04 进程拓扑与 StartPlan 实验")
    ap.add_argument("--skip-ps", action="store_true")
    ap.add_argument("--skip-startplan", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    if not a.skip_ps:
        summary["ps_topology"] = ps_topology()
        print("[ps]", json.dumps(summary["ps_topology"], ensure_ascii=False)[:600])
    if not a.skip_startplan:
        runs = []
        # 预先清掉旧 plan，保证 run1 是「首次落 plan」
        subprocess.run(["rm", "-rf", CACHE_PLAN_DIR])
        for startplan, tag in (("1", "run1_save"), ("1", "run2_reuse"), ("0", "run3_off")):
            r = one_offline(startplan, tag)
            runs.append(r)
            print(f"[startplan {tag}]", json.dumps(r, ensure_ascii=False)[:400])
        summary["startplan"] = runs
        r2, r3 = runs[1]["wall_s"], runs[2]["wall_s"]
        summary["startplan_verdict"] = dict(reuse_s=r2, off_s=r3, saved_s=round(r3 - r2, 1),
                                            reuse_faster=bool(r2 < r3))
    with open(os.path.join(OUT, "exp_ch04_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"[产物] {os.path.join(OUT, 'exp_ch04_summary.json')}")


if __name__ == "__main__":
    main()
