# exp_template_diff.py —— ch11 实验②复现件：chat 模板两种模式的 token 差（CPU 可跑）
# 用途：复现《Book7 第 11 章》11.5 的 28↔32 读数——enable_thinking 开/关两种模板模式
#   的 token 数差：关思考反而多 4 个（模板注入空 <think></think> 占位块，ids 尾
#   [151667,271,151668,271]）——「关掉=省略」的直觉在模板世界里不成立。
# 所属章节：Book7 ch11 §11.5（读数底档 log/book7-ch11/chat_template_diff.json）
# 运行方式：source env.sh && python code/Book7-vLLM源码剖析上/ch11/exp_template_diff.py --out-name tpl_repro
#   CPU 可跑（只调 tokenizer 的 apply_chat_template，不起引擎）；--out-name 缺省 template_diff
#   （不覆写底档 chat_template_diff.json——破坏性覆写零容忍）。
# 验收行：delta=+4 且 off 尾部含空 think 占位块 [151667,271,151668,271]（绝对 token 数随 prompt 而变；
#   底档 28↔32 是成档 prompt 的绝对数——机制读数是「关思考反多 4 token」这一条）。
import argparse
import json
import os

MODEL_PATH = ("log/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
              "c1899de289a04d12100db370d81485cdf75e47ca")  # 本地缓存快照（缓存红线；快照哈希以本机为准）


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="template_diff", help="产物名（落 log/book7-ch11/）")
    args = ap.parse_args()

    from transformers import AutoTokenizer  # noqa: E402（import 在参数解析后，保持 --help 快速）

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=False)
    msgs = [{"role": "user", "content": "杭州有什么好玩的地方"}]

    out = {}
    for tag, thinking in (("on", True), ("off", False)):
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
        )
        ids = tok(text, add_special_tokens=False)["input_ids"]
        out[f"n_{tag}"] = len(ids)
        out[f"ids_tail_{tag}"] = ids[-6:]

    out["delta_off_minus_on"] = out["n_off"] - out["n_on"]
    out["expect"] = "on=28 / off=32 / delta=+4（底档 chat_template_diff.json，Qwen3-0.6B 同款）"

    os.makedirs("log/book7-ch11", exist_ok=True)
    path = f"log/book7-ch11/{args.out_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"on={out['n_on']} off={out['n_off']} delta={out['delta_off_minus_on']:+d} → {path}")
    print(f"off 尾部 ids: {out['ids_tail_off']}")


if __name__ == "__main__":
    main()
