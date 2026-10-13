# vlm_first_run.py —— ch4 实验①：VLM 首跑（Qwen3-VL-2B 本机首验；判据 J1-J5 预注册见 notes/14 §四）
# 用途：起 serve→PIL 造图→图文+纯文本两发→usage 差分=视觉词元账（J3 三分支预注册）→LoRA×VLM 守卫观测（J5）。
# 运行：cd <workspace> && source env.sh && ASCEND_RT_VISIBLE_DEVICES=<空卡> \
#        python code/Book8-vLLM源码剖析下/ch04/vlm_first_run.py
# 产物：log/book8-ch04/vlm_first_run.json（+serve 日志）
import base64
import io
import json
import os
import subprocess
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(WS, "log/book8-ch04")
SNAP = os.path.join(WS, "log/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203")
PORT = 18434
BASE = f"http://127.0.0.1:{PORT}"


def make_image_b64(size=640):
    """确定性测试图（PIL 三色块，无随机）→ PNG base64。"""
    from PIL import Image
    img = Image.new("RGB", (size, size))
    for x in range(3):
        for y in range(3):
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][(x + y) % 3]
            for i in range(size // 3):
                for j in range(size // 3):
                    img.putpixel((x * (size // 3) + i, y * (size // 3) + j), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def chat(content, model="qwen3-vl-2b", max_tokens=48):
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    return d, (time.time() - t0) * 1000


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    serve_log = open(os.path.join(OUT, "vlm_serve.log"), "w")
    cmd = ["vllm", "serve", SNAP, "--host", "127.0.0.1", "--port", str(PORT),
           "--served-model-name", "qwen3-vl-2b", "--dtype", "bfloat16",
           "--max-model-len", "4096", "--gpu-memory-utilization", "0.5",
           "--mm-processor-cache-gb", "4",
           "--enable-lora", "--max-lora-rank", "8",
           "--lora-modules", "toy_perturb=" + os.path.join(WS, "log/book8-00/lora_toy/toy_perturb")]
    proc = subprocess.Popen(cmd, stdout=serve_log, stderr=subprocess.STDOUT)
    with open(os.path.join(OUT, "serve.pid"), "w") as f:
        f.write(str(proc.pid))
    code = "000"
    for _ in range(220):
        time.sleep(2)
        try:
            with urllib.request.urlopen(f"{BASE}/v1/models", timeout=5) as r:
                code = str(r.status)
            if code == "200":
                break
        except Exception:
            pass
    res = {"probe": "vlm_first_run", "j1_serve_up": code == "200", "http": code,
           "serve_cmd": " ".join(cmd[:6]) + " ...", "date": "2026-10-13"}
    if code != "200":
        res["j1_fail_tail"] = open(os.path.join(OUT, "vlm_serve.log"), encoding="utf-8").read()[-600:]
        with open(os.path.join(OUT, "vlm_first_run.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        proc.terminate()
        print(json.dumps(res, ensure_ascii=False)[:400])
        return

    img_b64 = make_image_b64(640)
    img_content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                   {"type": "text", "text": "Describe this image in one sentence."}]
    txt_content = [{"type": "text", "text": "Describe this image in one sentence."}]
    try:
        d_img, ms_img = chat(img_content)
        d_txt, ms_txt = chat(txt_content)
        pt_img = d_img["usage"]["prompt_tokens"]
        pt_txt = d_txt["usage"]["prompt_tokens"]
        res.update(j2_ok=True, img_ms=round(ms_img), txt_ms=round(ms_txt),
                   prompt_tokens_img=pt_img, prompt_tokens_txt=pt_txt,
                   visual_tokens=pt_img - pt_txt,
                   j3_formula_expect=400,
                   img_reply=d_img["choices"][0]["message"]["content"][:120],
                   txt_reply=d_txt["choices"][0]["message"]["content"][:80])
    except Exception as e:
        res.update(j2_ok=False, j2_err=str(e)[:300])
    # J5: LoRA×VLM 守卫观测（预期装载/请求报错——维度不匹配）
    try:
        d_lora, _ = chat(img_content, model="toy_perturb")
        res.update(j5_lora_vlm="跑通（意外——回查键名交集）",
                   lora_reply=d_lora["choices"][0]["message"]["content"][:80])
    except Exception as e:
        res.update(j5_lora_vlm=f"报错（守卫正身读数）: {str(e)[:220]}")
    # J4: 同图二发延迟（观测级）
    try:
        _, ms2 = chat(img_content)
        res["j4_same_img_second_ms"] = round(ms2)
    except Exception:
        pass
    with open(os.path.join(OUT, "vlm_first_run.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    proc.terminate()
    for _ in range(20):
        time.sleep(2)
        if proc.poll() is not None:
            break
    print(json.dumps({k: v for k, v in res.items() if k not in ("j1_fail_tail",)}, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()
