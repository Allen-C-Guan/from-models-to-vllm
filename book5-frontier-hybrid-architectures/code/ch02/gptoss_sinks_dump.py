# gptoss_sinks_dump.py —— Book5 ch2 实物 2.1 素材件：gpt-oss-20b sinks 标量张量 dump（真权重④路消费点）
# 用途（大纲 ch2 实物 2.1 口径）：gpt-oss-20b 存量权重（log/huggingface 缓存，零下载）里逐层
#   `model.layers.{i}.self_attn.sinks` 张量的形状/数值摘要（每头一个标量——dump 每层前 8 个值
#   + min/max/mean），附 transformers 5.18.0 modeling_gpt_oss.py 的行级锚与原文摘录——
#   「sink 从现象变成参数」的实物：真权重里每层每头都有一个小标量在改 softmax 分母。
# 条件性条款（大纲：失败不算失败）：权重未落地/读取失败 → 降级为源码静态行摘录（同文件落盘，
#   状态行标「降级」），正文消费口径退回「源码行实物」。
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch02/gptoss_sinks_dump.py"
# 产物：log/book5-ch02/sinks_dump.txt（不入库）
import glob
import json
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
OUT = os.path.join(REPO_ROOT, "log", "book5-ch02", "sinks_dump.txt")
SNAP_GLOB = os.path.join(REPO_ROOT, "log", "huggingface", "hub",
                         "models--openai--gpt-oss-20b", "snapshots", "*")
HF_SRC = os.path.join(REPO_ROOT, "repos", "transformers", "src", "transformers",
                      "models", "gpt_oss", "modeling_gpt_oss.py")

SRC_ANCHORS = {  # 行级锚（transformers 5.18.0 本地浅克隆，与 conda env 同版本）
    "L251-252 sink 拼接": (251, 252), "L257-259 softmax 后丢 sink": (257, 259),
    "L293 参数定义": (293, 293), "L329 s_aux 透传": (329, 329), "L415 初始化": (415, 415),
}


def src_excerpt(lines_map):
    out = []
    try:
        with open(HF_SRC, encoding="utf-8") as f:
            src = f.readlines()
        for tag, (a, b) in lines_map.items():
            out.append(f"  [{tag}]")
            for i in range(a, b + 1):
                out.append(f"  L{i}: {src[i-1].rstrip()}")
    except Exception as e:  # noqa: BLE001
        out.append(f"  [源码摘录失败] {type(e).__name__}: {e}")
    return "\n".join(out)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    snaps = sorted(glob.glob(SNAP_GLOB))
    try:
        snap = snaps[0]
        sts = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
        assert sts, f"snapshot 无 safetensors：{snap}"
        total = sum(os.path.getsize(p) for p in sts)
        cfg = json.load(open(os.path.join(snap, "config.json"), encoding="utf-8"))
        from safetensors import safe_open
        sinks = {}
        for p in sts:
            with safe_open(p, framework="pt") as f:
                for key in f.keys():
                    if key.endswith("self_attn.sinks"):
                        sinks[int(key.split(".")[2])] = (key, f.get_tensor(key))
        assert sinks, "未找到 *self_attn.sinks 张量"
        lines = [
            "# gpt-oss-20b sinks 标量张量 dump（实物 2.1 素材；状态：OK——真权重 safe_open 惰性读，零下载）",
            f"# 日期 2026-10-05 | snapshot: {snap}",
            f"# 权重分片 {len(sts)} 件共 {total/2**30:.2f} GiB | config: "
            f"num_hidden_layers={cfg.get('num_hidden_layers')} "
            f"num_attention_heads={cfg.get('num_attention_heads')} "
            f"num_key_value_heads={cfg.get('num_key_value_heads')} head_dim={cfg.get('head_dim')} "
            f"sliding_window={cfg.get('sliding_window')}（gpt-oss 生效字段；window_size="
            f"{cfg.get('window_size')} 为空置键）",
            f"# layer_types: {cfg.get('layer_types')}",
            "# 键型：model.layers.{i}.self_attn.sinks —— 每层一个 (num_attention_heads,) 张量，每头 1 个标量",
            "# 读法：值=sink logit（进 softmax 分母的加性项）；负值=压低 sink 位吸走的概率质量，",
            "#       正值=主动多吸；与 HF 建模语义对照见下方源码行级锚。",
            "",
        ]
        for i in sorted(sinks):
            key, t = sinks[i]
            tf = t.float()
            vals = ", ".join(f"{x:+.4f}" for x in tf[:8].tolist())
            lines.append(
                f"L{i:02d} {key} shape={tuple(t.shape)} dtype={t.dtype} "
                f"sinks[:8]=[{vals}] min={tf.min().item():+.4f} max={tf.max().item():+.4f} "
                f"mean={tf.mean().item():+.4f}")
        lines += ["", "# HF 建模源码行级锚（transformers 5.18.0 modeling_gpt_oss.py，本地浅克隆同版本）：", ""]
        lines.append(src_excerpt(SRC_ANCHORS))
        status = "OK"
    except Exception as e:  # noqa: BLE001 —— 条件性：失败降级为源码静态行，不算失败
        lines = [
            "# gpt-oss-20b sinks dump（状态：降级——权重读取失败，按条件性条款退回源码静态行实物）",
            f"# 失败原因 {type(e).__name__}: {e}",
            "", "# HF 建模源码行级锚（transformers 5.18.0 modeling_gpt_oss.py，本地浅克隆同版本）：", "",
        ]
        lines.append(src_excerpt(SRC_ANCHORS))
        status = f"降级（{type(e).__name__}）"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[状态] {status} | [产物] {OUT}")
    print("\n".join(lines[:14]))


if __name__ == "__main__":
    main()
