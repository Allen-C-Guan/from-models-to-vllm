# gptoss_sinks_dump.py —— Book5 第2章：本地 sink 权重与已安装实现的源码定位。
# 运行：source env.sh && python code/ch02/gptoss_sinks_dump.py --out-name reread [--snapshot 本地目录]
# 不下载模型。无完整权重时只报告 source_only，不生成或沿用历史权重数值。
import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import transformers
from transformers.models.gpt_oss import modeling_gpt_oss

ROOT = next((p for p in Path(__file__).resolve().parents if (p/"env.sh").is_file()), Path.cwd())

def dump(snapshot):
    from safetensors import safe_open
    import torch
    cfg=json.loads((snapshot/"config.json").read_text())
    index=json.loads((snapshot/"model.safetensors.index.json").read_text())["weight_map"]
    missing=[name for name in set(index.values()) if not (snapshot/name).is_file()]
    if missing:
        raise FileNotFoundError(f"快照缺少 {len(missing)} 个分片")
    rows=[]
    for i in range(cfg["num_hidden_layers"]):
        key=f"model.layers.{i}.self_attn.sinks"
        with safe_open(snapshot/index[key],framework="pt",device="cpu") as f:
            t=f.get_tensor(key).float()
        assert tuple(t.shape)==(cfg["num_attention_heads"],) and torch.isfinite(t).all()
        rows.append(f"L{i:02d} {key} shape={tuple(t.shape)} first8={t[:8].tolist()} "
                    f"min={t.min().item():.4f} max={t.max().item():.4f} mean={t.mean().item():.4f}")
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-name",default="run1")
    ap.add_argument("--snapshot",type=Path)
    a=ap.parse_args()
    out=ROOT/"log/book5-ch02"/f"sinks_dump_{a.out_name}.txt"
    if out.exists():
        raise FileExistsError(out)
    src=Path(inspect.getsourcefile(modeling_gpt_oss))
    lines=[f"# {datetime.now(timezone.utc).isoformat()} transformers={transformers.__version__}",
           f"# 实际安装源码：{src}"]
    anchors=("self.sinks =","sinks = module.sinks","combined_logits =","scores = probs",
             "module.sinks.data")
    excerpt=[f"L{i}: {s.strip()}" for i,s in enumerate(src.read_text().splitlines(),1)
             if any(key in s for key in anchors)]
    assert len(excerpt)>=3,"实现变化：请重新检查源码，不沿用旧行号"
    lines+=excerpt
    snaps=[a.snapshot] if a.snapshot else sorted((ROOT/"log/huggingface/hub/models--openai--gpt-oss-20b/snapshots").glob("*"))
    status="source_only"
    for snap in snaps:
        try:
            rows=dump(snap)
        except (OSError,KeyError,ValueError,AssertionError) as err:
            lines.append(f"# 未读取该快照 {snap}: {type(err).__name__}: {err}")
            continue
        lines += [f"# 权重来源：{snap}",*rows]
        status="weights_and_source"
        break
    lines += [f"# status={status}",
              "# sink 质量由 b 与当前 logsumexp(logits) 共同决定，不能仅按 b 正负判断。"]
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x") as f:
        f.write("\n".join(lines)+"\n")
    print(f"{status}: {out}")

if __name__=="__main__":
    main()
