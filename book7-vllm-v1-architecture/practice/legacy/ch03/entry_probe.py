# entry_probe.py —— Book7 ch03 代码件：入口层双入口三档探针（判据预注册 J1-J7）
# 用途：ch03《入口层》实验——同一份 207M 权重经三条入口路径加载生成，验证「双入口单核心」与
#       分派/插件/request_id/tokenizer 的入口层事实。三档：
#   档 A 离线·默认多进程（VLLM_ENABLE_V1_MULTIPROCESSING 未设=1 → SyncMPClient+EngineCoreProc 子进程）；
#   档 B 离线·同进程（=0 → InprocClient）；
#   档 C 在线（vllm serve + /v1/completions，temperature=0）。
# 判据预注册（notes/12 §12）：J1 三档贪心 completion 逐字一致；J2 档 A 日志含 (EngineCore pid=、档 B 不含；
#   J3 各档激活行 ≥1（档 A 预期主+子进程两份）；J4 离线 request_id 形态 ^\d+-[0-9a-f]{8}$；
#   J5 在线响应 id 前缀（仅记录）；J6 档 A prompt_token_ids==[349,2487,290,4020,323]（引擎侧 tokenizer 复核）；
#   J7 档 C 起服日志含「Block size is set to 128」与 PYTORCH_NPU_ALLOC_CONF 行。
# 所属章节：Book7 ch03（§3.2/§3.6 实验）；设计书=notes/12 §12。
# 运行：cd <workspace> && source env.sh && npu-smi info 选卡，然后逐档（bash 外层重定向日志）：
#   ASCEND_RT_VISIBLE_DEVICES=<卡> python code/Book7-vLLM源码剖析上/ch03/entry_probe.py --arm A > log/book7-ch03/armA.log 2>&1
#   ASCEND_RT_VISIBLE_DEVICES=<卡> python … --arm B > …/armB.log 2>&1
#   ASCEND_RT_VISIBLE_DEVICES=<卡> python … --arm C > …/armC.log 2>&1
#   python … --analyze   # 汇总三档日志+产物 → log/book7-ch03/entry_probe_summary.json（J1-J7 判定）
# 产物：log/book7-ch03/entry_probe_<A|B|C>_raw.json + entry_probe_summary.json + armC_serve.log
# 纪律：__main__ 守卫（spawn）；档 B 的环境变量在 import vllm 前置位；--out-name 防覆写。
import argparse
import json
import os
import re
import subprocess
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MODEL = os.path.join(WS, "log", "book7-00", "hf207m")
OUT = os.path.join(WS, "log", "book7-ch03")
PORT = 8022

PROMPTS = ["The capital of France is", "Machine learning is a field of", "One, two, three, four,"]
TOY_IDS = [349, 2487, 290, 4020, 323]  # notes/12 §2.3 transformers 5.14.1 直测值（J6 基准）


def arm_offline(arm: str) -> None:
    if arm == "B":
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # 必须在 import vllm 前
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=0.4,
              enforce_eager=True, disable_log_stats=False)
    sp = SamplingParams(temperature=0.0, max_tokens=32)
    t0 = time.perf_counter()
    outs = llm.generate(PROMPTS, sp)
    gen_s = round(time.perf_counter() - t0, 2)
    rows = [dict(prompt=p, completion=o.outputs[0].text, request_id=o.request_id,
                 prompt_token_ids=list(o.prompt_token_ids or [])) for p, o in zip(PROMPTS, outs)]
    raw = dict(arm=arm, gen_s=gen_s, outputs=rows,
               runtime="vllm 0.27.1 + vllm-ascend 0.27.1rc1（Atlas 910B3）")
    with open(os.path.join(OUT, f"entry_probe_{arm}_raw.json"), "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=1)
    print(f"[arm {arm}] done, gen_s={gen_s}")


def arm_online() -> None:
    serve_log = os.path.join(OUT, "armC_serve.log")
    cmd = ["vllm", "serve", MODEL, "--port", str(PORT), "--max-model-len", "1024",
           "--gpu-memory-utilization", "0.4", "--enforce-eager"]
    with open(serve_log, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=WS)
    try:
        t0 = time.perf_counter()
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                if proc.poll() is not None:
                    raise RuntimeError(f"serve died, see {serve_log}")
                if time.perf_counter() - t0 > 420:
                    raise RuntimeError("health timeout")
                time.sleep(3)
        load_s = round(time.perf_counter() - t0, 1)
        rows, ttfts = [], []
        for p in PROMPTS:
            body = json.dumps({"model": MODEL, "prompt": p, "max_tokens": 32,
                               "temperature": 0.0, "stream": True}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            t1 = time.perf_counter()
            completion, rid = "", ""
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        break
                    chunk = json.loads(payload)
                    if not ttfts or len(ttfts) < len(rows) + 1:
                        pass
                    rid = chunk.get("id", rid)
                    for ch in chunk.get("choices", []):
                        completion += ch.get("text", "")
                        ttfts.append(time.perf_counter() - t1)
                        break
            rows.append(dict(prompt=p, completion=completion, response_id=rid,
                             ttft_s=round(ttfts[-1], 4) if ttfts else None))
        raw = dict(arm="C", load_s=load_s, outputs=rows,
                   runtime="vllm serve 0.27.1 + vllm-ascend 0.27.1rc1（Atlas 910B3）")
        with open(os.path.join(OUT, "entry_probe_C_raw.json"), "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=1)
        print(f"[arm C] done, load_s={load_s}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()


def analyze() -> None:
    def read_arm(a: str):
        p = os.path.join(OUT, f"entry_probe_{a}_raw.json")
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

    def log_text(name: str) -> str:
        p = os.path.join(OUT, name)
        return open(p, encoding="utf-8", errors="ignore").read() if os.path.exists(p) else ""

    A, B, C = read_arm("A"), read_arm("B"), read_arm("C")
    la, lb, lc = log_text("armA.log"), log_text("armB.log"), log_text("armC.log") + log_text("armC_serve.log")
    res = {}
    if A:
        res["J2_A_has_enginecore_pid"] = "(EngineCore pid=" in la
        res["J3_A_activation_lines"] = la.count("Platform plugin ascend is activated")
        ids = [r["request_id"] for r in A["outputs"]]
        res["J4_A_request_ids"] = ids
        res["J4_pass"] = all(re.fullmatch(r"\d+-[0-9a-f]{8}", i or "") for i in ids)
        ptids = A["outputs"][0]["prompt_token_ids"]
        res["J6_A_prompt_token_ids"] = ptids
        res["J6_pass"] = ptids == TOY_IDS
    if B:
        res["J2_B_has_enginecore_pid"] = "(EngineCore pid=" in lb
        res["J3_B_activation_lines"] = lb.count("Platform plugin ascend is activated")
    if C:
        res["J5_C_response_id_head"] = (C["outputs"][0].get("response_id") or "")[:20]
        res["J7_C_blocksize_line"] = "Block size is set to 128" in lc
        res["J7_C_alloc_line"] = "PYTORCH_NPU_ALLOC_CONF" in lc
        res["J3_C_activation_lines"] = lc.count("Platform plugin ascend is activated")
    if A and B and C:
        ta = [r["completion"] for r in A["outputs"]]
        res["J1_parity_AB"] = ta == [r["completion"] for r in B["outputs"]]
        res["J1_parity_AC"] = ta == [r["completion"] for r in C["outputs"]]
    res["raw"] = dict(A=A, B=B, C=C)
    with open(os.path.join(OUT, "entry_probe_summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "raw"}, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description="ch03 入口层三档探针")
    ap.add_argument("--arm", choices=["A", "B", "C"], help="单档运行")
    ap.add_argument("--analyze", action="store_true", help="汇总判据")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.analyze:
        analyze()
    elif a.arm == "A" or a.arm == "B":
        arm_offline(a.arm)
    elif a.arm == "C":
        arm_online()


if __name__ == "__main__":
    main()
