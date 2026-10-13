# make_workloads.py —— ch2 实验 A workload 构造（echo 主臂 + random 负结果锚；prefix_repetition 用引擎内建）
# 所属章节：Book8 ch2（投机解码落地·实验 A）；设计书=research/Book8-vLLM源码剖析下/notes/11-ch02-投机解码素材底单.md §四
# 判据预注册：notes/11 实验 A·J3——echo 臂预期 acceptance>1.2；random 臂预期 ≈1.0（负结果锚）。
# 构造（echo）：~64 token 英文段×2 遍+「Repeat the passage:」——续写大概率回抄 prompt→prompt-lookup 有得抄；
#               （random）：同长随机 token 解码文本——ngram 匹配面近零。
# 运行：cd <workspace> && source env.sh && python code/Book8-vLLM源码剖析下/ch02/make_workloads.py  （CPU 秒级）
# 产物：本目录 wl_echo.jsonl / wl_random.jsonl（32 条×output 64 tok，fast 档）+ log/book8-ch02/workload_md5.json
import hashlib
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))   # code/<book>/ch02 → 工作区根（三级上跳）
TOK_DIR = os.path.join(WS, "log/book7-00/hf207m")
OUT_LOG = os.path.join(WS, "log/book8-ch02")
N_PROMPTS = 32
PASSAGE_TOKENS = 64
OUT_TOKENS = 64
SEED = 20261012

_POOL = ("the cat sat on the mat a dog ran in park sun bright bird flew over tree river "
         "flow quiet town road long book read page story told night star light wind soft "
         "rain fall field green hill small house door window open close boy girl play game "
         "ball round table food bread milk morning evening warm cold day year time work rest "
         "hand head eye walk slow fast jump song sing word line begin end first last new old").split()


def _passage(rng: random.Random, tok) -> str:
    """构造 ~PASSAGE_TOKENS token 的确定性随机段（词池采样、tokenizer 计数截断）。"""
    words = [rng.choice(_POOL) for _ in range(200)]
    text = ""
    for w in words:
        cand = (text + " " + w).strip()
        if len(tok.encode(cand, add_special_tokens=False)) >= PASSAGE_TOKENS:
            return text
        text = cand
    return text


def main() -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    rng = random.Random(SEED)

    echo = []
    for i in range(N_PROMPTS):
        p = _passage(rng, tok)
        echo.append({"prompt": f"{p}\n{p}\nRepeat the passage: ", "output_tokens": OUT_TOKENS})
    rnd = []
    for i in range(N_PROMPTS):
        ids = [rng.randint(3, 31999) for _ in range(PASSAGE_TOKENS)]
        rnd.append({"prompt": tok.decode(ids) + " Continue:", "output_tokens": OUT_TOKENS})

    md5s = {}
    for name, rows in [("wl_echo", echo), ("wl_random", rnd)]:
        path = os.path.join(HERE, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        md5s[name] = hashlib.md5(open(path, "rb").read()).hexdigest()
        print(f"[workload] {name}.jsonl：{len(rows)} 条，md5={md5s[name]}")

    os.makedirs(OUT_LOG, exist_ok=True)
    with open(os.path.join(OUT_LOG, "workload_md5.json"), "w", encoding="utf-8") as f:
        json.dump(dict(seed=SEED, n_prompts=N_PROMPTS, passage_tokens=PASSAGE_TOKENS,
                       out_tokens=OUT_TOKENS, md5=md5s, date="2026-10-12"), f, ensure_ascii=False, indent=1)
    print("[产物]", os.path.join(OUT_LOG, "workload_md5.json"))


if __name__ == "__main__":
    main()
