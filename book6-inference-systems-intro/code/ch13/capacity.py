"""第13章：读权重头与config，计算真实参数账和逻辑MLA KV账，不加载权重到NPU。

source ./env.sh; python .../capacity.py --model-path <snapshot> --out-name model
"""
import argparse
import json
import math
from pathlib import Path
from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out-name", required=True)
    a = ap.parse_args()
    root = Path(a.model_path)
    config = json.loads((root / "config.json").read_text())
    index = json.loads((root / "model.safetensors.index.json").read_text())
    groups, dtypes = {}, {}
    for filename in sorted(set(index["weight_map"].values())):
        with safe_open(root / filename, framework="pt", device="cpu") as weights:
            for key in weights.keys():
                view = weights.get_slice(key)
                count = math.prod(view.get_shape())
                group = ("routed_experts" if ".experts." in key else
                         "shared_experts" if ".shared_experts." in key else "other")
                groups[group] = groups.get(group, 0) + count
                dtype = view.get_dtype()
                dtypes[dtype] = dtypes.get(dtype, 0) + count
    per_token = config["num_hidden_layers"] * (config["kv_lora_rank"] + config["qk_rope_head_dim"]) * 2
    result = {"model_revision": root.name, "parameters": sum(groups.values()),
              "parameter_groups": groups, "parameter_dtypes": dtypes,
              "weight_payload_bytes": index["metadata"]["total_size"],
              "logical_bf16_mla_kv_bytes_per_token_all_layers": per_token,
              "logical_kv_bytes_8192_tokens": per_token * 8192,
              "note": "KV formula excludes padding, alignment, cache metadata and runtime replication"}
    out = Path("log/book6-ch13") / ("capacity_" + a.out_name + ".json")
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
