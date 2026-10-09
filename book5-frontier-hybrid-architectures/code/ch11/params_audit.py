#!/usr/bin/env python
"""params_audit.py——旗舰参数量口径自算裁决（GLM-5 家族 + Kimi K3）
=====================================================

Book5 ch11「读厂商自报数字五步法」第 4 步「复算可算的」的实测脚本（批一前置版，
批三扩写 K3 侧）。裁决口径：

  GLM-5 家族（批一，三口径）：
    A【论文 744B 】 GLM-5 论文 Table 10：含 MTP 层、不含词嵌入与输出层。
    B【HF 卡面 754B / 753B】 模型卡架构区的总参数自报数。
    C【safetensors 全量】 checkpoint 分片头逐张量求和的自算数。

  Kimi K3（批三，MXFP4 打包口径，checkpoint=moonshotai/Kimi-K3 remote code）：
    A【官方 Table 1】 总参 2.78T / 激活 104.2B（arXiv:2607.24653）。
    B【config 公式自算】 2,776,070,367,200 / 101,948,568,544（k3_account_out.txt）。
    C【safetensors 逻辑参数】 本脚本解码 MXFP4 打包后逐张量求和：
       compressed-tensors 把路由专家存成 `*.weight_packed`（U8，每字节 2 个 FP4）
       + `*.weight_scale`（U8，E8M0，group 32——量化元数据不计逻辑参数）；
       逻辑参数 = packed_numel×2 + 普通 dtype numel（BF16/F32 直数）。
    另报存储字节数（按 dtype 宽度）与 index.json total_size 对账。

方法：safetensors 文件头 = 前 8 字节小端 u64 头长度 N + N 字节 JSON 元数据
（张量名/dtype/shape）。对每个分片只发两个 HTTP Range 请求（8 字节 + 头 JSON），
不下载权重本体——K3 96 分片头合计约 77MB（vs 1.56TB 权重本体）。

用法：
    python params_audit.py [--repo zai-org/GLM-5] [--repo moonshotai/Kimi-K3] [--out DIR]
    （不传 --repo 时默认 GLM-5 + GLM-5.2 + Kimi-K3 三仓）

产物（写入 log/book5-ch11/params_audit/，缓存红线）：
    <repo slug>.json  逐分片/逐组件账本 + 口径裁决表

证据等级：{config/checkpoint 亲算}（safetensors 头为官方发布工件）。
"""

import argparse
import json
import os
import struct
import sys
import time
import urllib.request
from collections import defaultdict

HF = "https://huggingface.co"
UA = {"User-Agent": "book5-params-audit/0.1 (research; ranged header reads)"}


def http_get(url: str, rng: str | None = None, maxlen: int = 20_000_000,
             retries: int = 5) -> bytes:
    """GET（可选 Range），指数退避重试。safetensors 头对超大 MoE 可到 MB 级。"""
    headers = dict(UA)
    if rng:
        headers["Range"] = f"bytes={rng}"
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read(maxlen)
                if r.status not in (200, 206):
                    raise RuntimeError(f"{url} -> HTTP {r.status}")
                return data
        except Exception as e:  # noqa: BLE001 —— 网络/超时统一退避重试
            last = e
            wait = 2 ** attempt
            print(f"  [retry {attempt + 1}/{retries}] {url}: {e} — sleep {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"{url} failed after {retries} retries: {last}")


def list_shards(repo: str) -> list[str]:
    """枚举仓库根目录的 *.safetensors 分片（HF models API siblings）。"""
    api = f"{HF}/api/models/{repo}"
    with urllib.request.urlopen(
        urllib.request.Request(api, headers=UA), timeout=60
    ) as r:
        meta = json.load(r)
    shards = sorted(
        s["rfilename"]
        for s in meta.get("siblings", [])
        if s["rfilename"].endswith(".safetensors") and "/" not in s["rfilename"]
    )
    if not shards:
        raise RuntimeError(f"{repo}: no root-level safetensors found")
    return shards


def read_shard_header(repo: str, fname: str) -> dict[str, dict]:
    """Range 读单个分片头，返回 {tensor_name: {"shape":…, "dtype":…}}。"""
    url = f"{HF}/{repo}/resolve/main/{fname}"
    n = struct.unpack("<Q", http_get(url, rng="0-7"))[0]
    if n <= 0 or n > 100_000_000:
        raise RuntimeError(f"{fname}: implausible header length {n}")
    header = json.loads(http_get(url, rng=f"8-{8 + n - 1}"))
    return {k: {"shape": v["shape"], "dtype": v["dtype"]}
            for k, v in header.items() if k != "__metadata__"}


DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
               "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def list_shards_via_index(repo: str) -> tuple[list[str], int]:
    """经 model.safetensors.index.json 取分片清单与 total_size（K3 用，权威口径）。
    注意：大 MoE 仓的 index 把 50 万级张量名逐条映射，JSON 可达数十 MB——
    maxlen 必须放大（默认 20MB 会在 K3 上截断）。"""
    idx = json.loads(http_get(f"{HF}/{repo}/resolve/main/model.safetensors.index.json",
                              maxlen=400_000_000))
    return (sorted(set(idx["weight_map"].values())), idx["metadata"]["total_size"])


def numel(shape: list[int]) -> int:
    p = 1
    for d in shape:
        p *= d
    return p


def classify(name: str, n_layers: int) -> str:
    """张量名 -> 组件桶（GLM/DeepSeek 系命名约定）。"""
    if "embed_tokens" in name:
        return "embed_tokens(词嵌入)"
    if name.startswith("lm_head") or name.endswith(".lm_head.weight"):
        return "lm_head(输出层)"
    # MTP/nextn 层：GLM 惯例把 nextn 层放在 model.layers.<n_layers> 起的额外编号
    # （num_hidden_layers=78 时，layers.78.* 即 nextn/MTP 层；DeepSeek 同款约定）。
    if name.startswith("model.layers."):
        idx = int(name.split(".")[2])
        return f"layer[{idx}] MTP" if idx >= n_layers else "layer[0..77] 主干"
    if "nextn" in name or "mtp" in name.lower() or "draft" in name:
        return "MTP(显式命名)"
    return "其他(norm 等)"


def audit(repo: str, n_layers: int | None, cache_dir: str | None = None) -> dict:
    shards = list_shards(repo)
    if n_layers is None:  # 从 config.json 现场读层数
        cfg = json.loads(http_get(f"{HF}/{repo}/resolve/main/config.json"))
        n_layers = cfg["num_hidden_layers"]
    os.makedirs(cache_dir, exist_ok=True) if cache_dir else None
    total = 0
    by_comp: defaultdict[str, int] = defaultdict(int)
    layer_ids, mtp_layer_ids = set(), set()
    per_shard = []
    for s in shards:
        # 缓存键必须带 repo slug：GLM-5 与 GLM-5.2 的分片名同形，裸文件名会串仓
        cpath = (os.path.join(cache_dir, f"{repo.replace('/', '__')}__{s}.hdr.json")
                 if cache_dir else None)
        hdr = None
        if cpath and os.path.exists(cpath):  # 断点续跑：分片头落盘缓存
            with open(cpath) as f:
                hdr = json.load(f)
            # 旧版缓存（GLM 批一）只存 shape：升级为 {"shape","dtype"}
            if hdr and isinstance(next(iter(hdr.values())), list):
                hdr = {k: {"shape": v, "dtype": "BF16?"} for k, v in hdr.items()}
        if hdr is None:
            hdr = read_shard_header(repo, s)
            if cpath:
                with open(cpath, "w") as f:
                    json.dump(hdr, f)
        shard_numel = 0
        for name, t in hdr.items():
            m = numel(t["shape"])
            total += m
            shard_numel += m
            by_comp[classify(name, n_layers)] += m
            if name.startswith("model.layers."):
                idx = int(name.split(".")[2])
                (mtp_layer_ids if idx >= n_layers else layer_ids).add(idx)
        per_shard.append({"file": s, "tensors": len(hdr), "params": shard_numel})
        print(f"  {s}: {len(hdr):6d} tensors, {shard_numel/1e9:.3f}B", file=sys.stderr)

    # 三口径换算
    embed = sum(v for k, v in by_comp.items() if "词嵌入" in k or "输出层" in k)
    mtp = sum(v for k, v in by_comp.items() if "MTP" in k)
    backbone = total - mtp  # 主干 = 全量 − MTP（含嵌入/输出层）
    paper_caliber = total - embed  # 论文口径 = 全量 − 词嵌入 − 输出层（含 MTP）

    return {
        "repo": repo,
        "num_hidden_layers_config": n_layers,
        "layer_index_max_backbone": (max(layer_ids) if layer_ids else None),
        "mtp_layer_indices": sorted(mtp_layer_ids),
        "n_shards": len(shards),
        "per_shard": per_shard,
        "by_component": {k: {"params": v, "B": round(v / 1e9, 3)}
                         for k, v in sorted(by_comp.items(), key=lambda kv: -kv[1])},
        "calibers": {
            "A_total_safetensors(全量)": round(total / 1e9, 3),
            "B_paper(全量-嵌入-输出层, 含MTP)": round(paper_caliber / 1e9, 3),
            "C_backbone_only(全量-MTP)": round(backbone / 1e9, 3),
        },
        "_raw": {"total": total, "embed_lmhead": embed, "mtp": mtp},
    }


# ----------------------------------------------------------------------
# Kimi K3 侧（批三扩写）：remote code 仓 + compressed-tensors MXFP4 打包
# 键名事实（log/book5-feasibility/k3_remote_code/ 本地亲核 + 分片头实测）：
#   顶层前缀 language_model.（KimiK3ForConditionalGeneration 包文本骨架）；
#   MoE 模块名 block_sparse_moe（非 mlp——mlp 只在首层稠密 FFN 出现）；
#   专家三件 experts.N.w1/w2/w3.weight_packed（U8×2/FP4）+ .weight_scale（U8，g32）；
#   AttnRes=res_proj[1,7168]+res_norm；层号 checkpoint 0 起（config 层表 1 起）。
# ----------------------------------------------------------------------

K3_TOP = "language_model."


def k3_layer_kind(name: str) -> str | None:
    """按张量名签名判层型：MLA 层带 kv_a/kv_b/q_a/q_b 投影，KDA 层带 A_log/dt_bias。"""
    if ".self_attn.kv_a_proj_with_mqa." in name or ".self_attn.kv_b_proj." in name \
            or ".self_attn.q_a_proj." in name or ".self_attn.q_b_proj." in name:
        return "MLA"
    if ".self_attn.A_log" in name or ".self_attn.dt_bias" in name:
        return "KDA"
    return None


def audit_k3(repo: str, cache_dir: str | None = None) -> dict:
    import re
    shards, index_total_size = list_shards_via_index(repo)
    os.makedirs(cache_dir, exist_ok=True) if cache_dir else None
    # config 交叉：full_attn_layers（papers/04 明注 1 起）vs checkpoint 层签名
    cfg = json.loads(http_get(f"{HF}/{repo}/resolve/main/config.json"))
    lac = cfg["text_config"]["linear_attn_config"]
    full_attn_1idx, n_experts, top_k = (lac["full_attn_layers"], cfg["text_config"]["num_experts"],
                                        cfg["text_config"]["num_experts_per_token"])
    n_layers = cfg["text_config"]["num_hidden_layers"]

    def load_hdr(s: str) -> dict:
        cpath = (os.path.join(cache_dir, f"{repo.replace('/', '__')}__{s}.hdr.json")
                 if cache_dir else None)
        if cpath and os.path.exists(cpath):
            with open(cpath) as f:
                return json.load(f)
        hdr = read_shard_header(repo, s)
        if cpath:
            with open(cpath, "w") as f:
                json.dump(hdr, f)
        return hdr

    headers = [(s, load_hdr(s)) for s in shards]

    # ---- 第一遍：按签名建层型表（MLA=kv_a/kv_b/q_a/q_b；KDA=A_log/dt_bias）----
    kda_layers, mla_layers = set(), set()
    for _, hdr in headers:
        for name in hdr:
            kind = k3_layer_kind(name)
            m = re.match(r"language_model\.model\.layers\.(\d+)\.", name)
            if kind and m:
                (mla_layers if kind == "MLA" else kda_layers).add(int(m.group(1)))
    layer_kind = {i: "MLA" for i in mla_layers} | {i: "KDA" for i in kda_layers}

    logical = 0                      # 逻辑参数（packed×2；scale 不计）
    scale_elems = 0                  # 量化元数据元素（weight_scale）
    storage_bytes = 0                # 存储字节（按 dtype 宽度）
    by_comp: defaultdict[str, int] = defaultdict(int)   # 逻辑参数桶
    by_dtype: defaultdict[str, int] = defaultdict(int)  # 张量个数×dtype
    layer_expert_logical: defaultdict[int, int] = defaultdict(int)
    unmatched = defaultdict(int)
    per_shard = []
    n_tensors = 0

    def add(bucket: str, k: int):
        nonlocal logical
        by_comp[bucket] += k
        logical += k

    # ---- 第二遍：逐张量入桶 ----
    for s, hdr in headers:
        shard_logical = 0
        for name, t in hdr.items():
            n_tensors += 1
            dt, shp = t["dtype"], t["shape"]
            m = numel(shp)
            by_dtype[dt] += 1
            storage_bytes += m * DTYPE_BYTES.get(dt, 2)
            if name.endswith(".weight_packed"):        # MXFP4：每字节 2 个 FP4 权重
                k = m * 2
            elif name.endswith(".weight_scale"):       # E8M0 组标度（g32）——元数据
                scale_elems += m
                continue
            else:
                k = m
            shard_logical += k
            m_layer = re.match(r"language_model\.model\.layers\.(\d+)\.(.+)", name)
            idx, rest = (int(m_layer.group(1)), m_layer.group(2)) if m_layer else (None, name)
            if rest.startswith("self_attn."):
                bucket = f"层[{layer_kind.get(idx, '?')}].self_attn"
            elif rest.startswith("block_sparse_moe.experts."):
                bucket = "层.MoE.路由专家(w1w2w3)"
                layer_expert_logical[idx] += k
            elif rest.startswith("block_sparse_moe.gate."):
                bucket = "层.MoE.router(含bias)"
            elif rest.startswith("block_sparse_moe.shared_experts."):
                bucket = "层.MoE.共享专家"
            elif rest.startswith("block_sparse_moe.routed_expert_down_proj.") \
                    or rest.startswith("block_sparse_moe.routed_expert_up_proj."):
                bucket = "层.MoE.Wdn/Wup(潜升降)"
            elif rest.startswith("block_sparse_moe.routed_expert_norm."):
                bucket = "层.MoE.潜RMSNorm"
            elif rest.startswith("mlp."):
                bucket = "层.稠密FFN(首层)"
            elif "res_proj" in rest or "res_norm" in rest:
                bucket = "层.AttnRes(伪查询+范数)"
            elif "layernorm" in rest:
                bucket = "层.RMSNorm"
            elif name.startswith("vision_tower."):
                bucket = "vision_tower(MoonViT-V2)"
            elif name.startswith("mm_projector."):
                bucket = "mm_projector"
            elif name.endswith("lm_head.weight"):
                bucket = "lm_head(输出层)"
            elif name.startswith(K3_TOP + "model.embed_tokens"):
                bucket = "embed_tokens(词嵌入)"
            elif name.startswith(K3_TOP + "model.norm"):
                bucket = "final_norm"
            elif name.startswith(K3_TOP + "model.output_attn_res"):
                bucket = "model.output_attn_res(draft特征)"
            else:
                bucket = "其他"
                unmatched[name] += k
            add(bucket, k)
        per_shard.append({"file": s, "tensors": len(hdr), "logical": shard_logical})
        print(f"  {s}: {len(hdr):6d} tensors, {shard_logical/1e9:9.3f}B logical",
              file=sys.stderr)

    # ---- 层型交叉验证：config full_attn_layers（1 起）-1 vs checkpoint 签名 ----
    expect_mla = {x - 1 for x in full_attn_1idx}
    xcheck = {
        "config_full_attn_layers(1起)": full_attn_1idx,
        "checkpoint_MLA_signature_layers(0起)": sorted(mla_layers),
        "checkpoint_KDA_layers(0起)": sorted(kda_layers),
        "match": expect_mla == mla_layers and len(kda_layers) + len(mla_layers) == n_layers,
    }

    # ---- 激活参数（口径：全量−路由专家 + top_k×单专家×MoE 层数） ----
    expert_total = by_comp.get("层.MoE.路由专家(w1w2w3)", 0)
    n_moe_layers = len(layer_expert_logical)
    per_expert = next(iter(layer_expert_logical.values())) // n_experts if n_moe_layers else 0
    vit = by_comp.get("vision_tower(MoonViT-V2)", 0)
    proj = by_comp.get("mm_projector", 0)
    head = by_comp.get("lm_head(输出层)", 0)
    text_backbone = logical - vit - proj
    activated_text = text_backbone - expert_total + top_k * per_expert * n_moe_layers

    return {
        "repo": repo,
        "n_shards": len(shards),
        "n_tensors": n_tensors,
        "index_total_size_bytes": index_total_size,
        "sum_storage_bytes": storage_bytes,
        "scale_elements(not params)": scale_elems,
        "dtype_tensor_counts": dict(by_dtype),
        "layer_xcheck": xcheck,
        "by_component_logical": {k: {"params": v, "B": round(v / 1e9, 4)}
                                 for k, v in sorted(by_comp.items(), key=lambda kv: -kv[1])},
        "per_layer_logical": {
            "KDA_self_attn_per_layer": by_comp.get("层[KDA].self_attn", 0) // max(len(kda_layers), 1),
            "MLA_self_attn_per_layer": by_comp.get("层[MLA].self_attn", 0) // max(len(mla_layers), 1),
            "g_proj_each_layer_12288x7168": 12288 * 7168,   # 权重级正身（vs 报告 account 51.4M）
        },
        "unmatched_should_be_empty": dict(unmatched),
        "per_shard": per_shard,
        "activated_params": {
            "text_backbone_with_embed_head": activated_text,
            "B": round(activated_text / 1e9, 4),
            "text_minus_lm_head": activated_text - head,
            "B_minus_lm_head": round((activated_text - head) / 1e9, 4),
            "rule": "文本主干全量−路由专家 + top_k(16)×单专家×92 层；含嵌入+输出层+稠密FFN+AttnRes；不含 ViT/projector",
        },
        "calibers": {
            "A_index_total_size(存储TB≈)": round(index_total_size / 1e12, 3),
            "B_logical_total(全家族含ViT+projector)": round(logical / 1e9, 3),
            "C_logical_text_backbone(官方2.78T主数域)": round(text_backbone / 1e9, 3),
        },
        "_raw": {"logical": logical, "text_backbone": text_backbone,
                 "storage_bytes": storage_bytes, "vit": vit, "projector": proj,
                 "lm_head": head, "expert_total": expert_total,
                 "per_expert": per_expert},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", action="append", default=None,
                    help="HF 仓库 id，可多次；默认 GLM-5 / GLM-5.2 / Kimi-K3")
    ap.add_argument("--out", default="log/book5-ch11/params_audit",
                    help="JSON 产物目录（默认 log/book5-ch11/params_audit）")
    args = ap.parse_args()
    repos = args.repo or ["zai-org/GLM-5", "zai-org/GLM-5.2", "moonshotai/Kimi-K3"]

    results = {}
    for repo in repos:
        print(f"[audit] {repo}", file=sys.stderr)
        if repo == "moonshotai/Kimi-K3":
            results[repo] = audit_k3(repo, cache_dir=os.path.join(args.out, "cache"))
        else:
            results[repo] = audit(repo, n_layers=None,
                                  cache_dir=os.path.join(args.out, "cache"))

    # 裁决表（三个自报口径的对照；自报数来源见 papers/06 §2.6 / papers/04 §0）
    claims = {
        "zai-org/GLM-5": {"论文 Table 10（口径句见 papers/06 §2.6 勘异）": 744,
                          "HF 自动统计": 754},
        "zai-org/GLM-5.2": {"HF 自动统计（README 从未自报参数量）": 753},
        "moonshotai/Kimi-K3": {"官方 Table 1 总参 2.78T": 2780,
                               "config 公式自算 k3_account_out.txt": 2776.07},
    }
    print("\n==== 三口径裁决（单位 B=1e9）====")
    for repo, r in results.items():
        c = r["calibers"]
        if repo == "moonshotai/Kimi-K3":
            print(f"\n{repo}  (分片 {r['n_shards']} 个, 逻辑张量 {r['n_tensors']} 个)")
            for k, v in c.items():
                print(f"  {k:46s} {v:12.3f} B")
            for src, claimed in claims[repo].items():
                best = min(c.items(), key=lambda kv: abs(kv[1] - claimed))
                delta = claimed - best[1]
                print(f"  自报 {src} {claimed}B  <≈ 自算[{best[0]}] {best[1]:.3f}B "
                      f"(差 {delta:+.3f}B)")
            act = r["activated_params"]
            print(f"  激活参数自算 = {act['text_backbone_with_embed_head']/1e9:.3f} B "
                  f"vs 官方 104.2B (差 {104.2 - act['text_backbone_with_embed_head']/1e9:+.3f}B)"
                  f" | 去输出层 = {act['B_minus_lm_head']}B (差 {104.2 - act['B_minus_lm_head']:+.3f}B)")
            continue
        print(f"\n{repo}  (config 层数 {r['num_hidden_layers_config']}, "
              f"分片 {r['n_shards']} 个, MTP 层索引 {r['mtp_layer_indices']})")
        for k, v in c.items():
            print(f"  {k:42s} {v:10.3f} B")
        for src, claimed in claims.get(repo, {}).items():
            best = min(c.items(), key=lambda kv: abs(kv[1] - claimed))
            delta = claimed - best[1]
            print(f"  自报 {src} {claimed}B  <≈ 自算[{best[0]}] {best[1]:.3f}B "
                  f"(差 {delta:+.3f}B)")
    os.makedirs(args.out, exist_ok=True)
    for repo, r in results.items():
        slug = repo.split("/")[-1].replace(".", "_")
        path = os.path.join(args.out, f"{slug}.json")
        with open(path, "w") as f:
            json.dump(r, f, ensure_ascii=False, indent=1)
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
