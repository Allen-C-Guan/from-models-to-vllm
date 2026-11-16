#!/usr/bin/env python
"""params_audit.py——旗舰参数量口径自算裁决（GLM-5 家族 + Kimi K3）
=====================================================

Book5 ch11「读厂商自报数字五步法」第 4 步「复算可算的」的实测脚本（批一前置版，
批三扩写 K3 侧）。对照口径：

  GLM-5 家族（批一，三口径）：
    A【论文 744B 】 GLM-5 论文 Table 10：含 MTP 层、不含词嵌入与输出层。
    B【HF页面754B/753B】 页面自动统计，不等同作者自报架构口径。
    C【safetensors 全量】 checkpoint 分片头逐张量求和的自算数。

  Kimi K3（批三，MXFP4 打包口径，checkpoint=moonshotai/Kimi-K3 remote code）：
    A【官方 Table 1】 总参 2.78T / 激活 104.2B（arXiv:2607.24653）。
    B【旧版错误公式，保留查错对象】 2,776,070,367,200 / 101,948,568,544（k3_account_out.txt）。
    C【safetensors 逻辑元素】 本脚本解码 MXFP4 打包后逐张量求和：
       compressed-tensors 把路由专家存成 `*.weight_packed`（U8，每字节 2 个 FP4）
       + `*.weight_scale`（U8，E8M0，group 32——量化元数据不计逻辑权重元素）；
       逻辑元素 = packed_numel×2 + 普通 dtype numel（BF16/F32 直数）。
    另报存储字节数（按 dtype 宽度）与 index.json total_size 对账。

方法：safetensors 文件头 = 前 8 字节小端 u64 头长度 N + N 字节 JSON 元数据
（张量名/dtype/shape/data_offsets）。对每个分片只发两个 HTTP Range 请求（8 字节 + 头 JSON），
不下载权重本体——K3 96 分片头合计约 77MB（vs 1.56TB 权重本体）。

用法：
    python params_audit.py [--repo zai-org/GLM-5] [--repo moonshotai/Kimi-K3] [--out DIR]
    （显式--repo；--out-name隔离报告；--offline复用已有头快照）

产物（写入 log/book5-ch11/params_audit/，缓存红线）：
    <repo slug>_<out-name>.json  逐分片/逐组件账本 + 口径对照表

证据等级：{config/checkpoint 亲算}（safetensors 头为官方发布工件）。
"""

import argparse
from pathlib import Path
import hashlib,math,re
from datetime import datetime, timezone
import json
import os
import struct
import sys
import time
import urllib.request
from collections import defaultdict

HF = "https://huggingface.co"
UA = {"User-Agent": "book5-params-audit/0.1 (research; ranged header reads)"}


class Snapshot:
    """固定revision的只读权重头。缓存含model/config/index与*.header.json，不含权重正文。"""
    def __init__(self,repo,cache_dir,revision=None,offline=False,config_path=None):
        self.repo=repo;self.path=Path(cache_dir);self.path.mkdir(parents=True,exist_ok=True)
        self.offline=offline
        model=self.path/"model.json"
        if model.exists():meta=json.loads(model.read_text())
        else:
            if offline:raise FileNotFoundError(model)
            url=f"{HF}/api/models/{repo}"+(f"/revision/{revision}" if revision else "")
            raw=http_get(url,maxlen=4_000_000);meta=json.loads(raw);model.write_bytes(raw)
        self.revision=meta["sha"]
        if not re.fullmatch("[0-9a-f]{40}",self.revision):raise ValueError("API未返回固定commit SHA")
        if revision and revision!=self.revision:raise ValueError("缓存revision不同，请使用新目录")
        if meta.get("id","").lower()!=repo.lower():raise ValueError("缓存仓库身份不符")
        self.config=json.loads(Path(config_path).read_text()) if config_path else self.json_file("config.json","config.json",4_000_000)
        self.index=self.json_file("index.json","model.safetensors.index.json",200_000_000)
        self.shards=sorted(set(self.index["weight_map"].values()))
        if not self.shards:raise ValueError("index没有分片")
        self.manifest=[]

    def json_file(self,local,remote,limit):
        p=self.path/local
        if p.exists():return json.loads(p.read_text())
        if self.offline:raise FileNotFoundError(p)
        raw=http_get(f"{HF}/{self.repo}/resolve/{self.revision}/{remote}",maxlen=limit)
        value=json.loads(raw);p.write_bytes(raw);return value

    def header(self,file):
        if Path(file).name!=file or not file.endswith(".safetensors"):raise ValueError("仅支持根目录标准分片名")
        p=self.path/(file+".header.json")
        if p.exists():raw=p.read_bytes()
        else:
            if self.offline:raise FileNotFoundError(p)
            url=f"{HF}/{self.repo}/resolve/{self.revision}/{file}"
            length=struct.unpack("<Q",http_get(url+"?head=0-7",rng="0-7",maxlen=8))[0]
            if not 0<length<=100_000_000:raise ValueError("分片头长度异常")
            raw=http_get(url+f"?head=8-{7+length}",rng=f"8-{7+length}",maxlen=length)
            json.loads(raw);p.write_bytes(raw)
        hdr=json.loads(raw);hdr.pop("__metadata__",None)
        intervals=[]
        for name,t in hdr.items():
            if t["dtype"] not in DTYPE_BYTES:raise ValueError(f"未知dtype {t['dtype']}")
            if not all(isinstance(d,int) and d>=0 for d in t["shape"]):raise ValueError("非法形状")
            a,b=t["data_offsets"];expected=numel(t["shape"])*DTYPE_BYTES[t["dtype"]]
            if a<0 or b-a!=expected:raise ValueError(f"shape/dtype/offset不一致：{name}")
            intervals.append((a,b))
        end=0
        for a,b in sorted(intervals):
            if a!=end:raise ValueError("分片数据区存在空隙/重叠")
            end=b
        self.manifest.append(dict(file=file,header_sha256=hashlib.sha256(raw).hexdigest(),header_bytes=len(raw),data_bytes=end))
        return hdr

    def headers(self):
        seen={};result=[]
        for file in self.shards:
            hdr=self.header(file)
            for name in hdr:
                if name in seen:raise ValueError(f"重复张量{name}")
                seen[name]=file
            result.append((file,hdr))
        if seen!=self.index["weight_map"]:raise ValueError("分片头与index映射不一致")
        storage=sum(m["data_bytes"] for m in self.manifest)
        if storage!=self.index["metadata"]["total_size"]:raise ValueError("数据字节和与index不符")
        return result


def http_get(url,rng=None,maxlen=20_000_000,retries=3):
    """Range必须收到匹配的206；读取上限+1检测截断/超长，不接受整权重回包。"""
    for attempt in range(retries):
        try:
            headers=dict(UA)
            if rng:headers["Range"]="bytes="+rng
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=45) as response:
                if rng:
                    a,b=map(int,rng.split("-"))
                    if response.status!=206 or not response.headers.get("Content-Range","").startswith(f"bytes {a}-{b}/"):
                        raise ValueError("服务器未按Range返回分片头")
                    if b-a+1!=maxlen:raise ValueError("Range长度与读上限不同")
                elif response.status!=200:raise ValueError(f"HTTP {response.status}")
                raw=response.read(maxlen+1)
                if len(raw)>maxlen or (rng and len(raw)!=maxlen):raise ValueError("响应长度不符")
                return raw
        except Exception:
            if attempt==retries-1:raise
            time.sleep(attempt+1)


DTYPE_BYTES={"F64":8,"F32":4,"F16":2,"BF16":2,"I64":8,"I32":4,"I16":2,"I8":1,"U8":1,"BOOL":1}


def numel(shape):
    return math.prod(shape)


def classify(name,n_layers):
    """先判层范围，避免MTP自带嵌入与顶层嵌入混桶。"""
    match=re.match(r"model\.layers\.(\d+)\.",name)
    if match:return "MTP" if int(match[1])>=n_layers else "主干层"
    if "nextn" in name or "mtp" in name.lower():return "MTP"
    if "embed_tokens" in name:return "embed_tokens(词嵌入)"
    if name.startswith("lm_head") or name.endswith(".lm_head.weight"):return "lm_head(输出层)"
    return "其他(norm等)"


def audit(repo: str, n_layers: int | None=None, cache_dir: str | None=None, snapshot=None) -> dict:
    snap=snapshot or Snapshot(repo,cache_dir)
    shards=snap.shards
    if n_layers is None:n_layers=snap.config["num_hidden_layers"]
    headers=snap.headers()
    total = 0
    by_comp: defaultdict[str, int] = defaultdict(int)
    layer_ids, mtp_layer_ids = set(), set()
    per_shard = []
    for s,hdr in headers:
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
        "repo":repo,"revision":snap.revision,"header_manifest":snap.manifest,
        "scope":"checkpoint逻辑元素和，含非梯度buffer；不是全可学习参数",
        "num_hidden_layers_config":n_layers,
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


def audit_k3(repo: str, cache_dir: str | None=None, snapshot=None) -> dict:
    snap=snapshot or Snapshot(repo,cache_dir)
    shards=snap.shards;index_total_size=snap.index["metadata"]["total_size"];cfg=snap.config
    lac = cfg["text_config"]["linear_attn_config"]
    full_attn_1idx, n_experts, top_k = (lac["full_attn_layers"], cfg["text_config"]["num_experts"],
                                        cfg["text_config"]["num_experts_per_token"])
    n_layers = cfg["text_config"]["num_hidden_layers"]

    headers=snap.headers()

    # ---- 第一遍：按签名建层型表（MLA=kv_a/kv_b/q_a/q_b；KDA=A_log/dt_bias）----
    kda_layers, mla_layers = set(), set()
    for _, hdr in headers:
        for name in hdr:
            kind = k3_layer_kind(name)
            m = re.match(r"language_model\.model\.layers\.(\d+)\.", name)
            if kind and m:
                (mla_layers if kind == "MLA" else kda_layers).add(int(m.group(1)))
    if mla_layers & kda_layers or mla_layers|kda_layers!=set(range(n_layers)):
        raise ValueError("层型覆盖或互斥条件不成立")
    layer_kind = {i:"MLA" for i in mla_layers}|{i:"KDA" for i in kda_layers}

    logical = 0                      # 逻辑参数（packed×2；scale 不计）
    scale_elems = 0                  # 量化元数据元素（weight_scale）
    storage_bytes = 0                # 存储字节（按 dtype 宽度）
    by_comp: defaultdict[str, int] = defaultdict(int)   # 逻辑参数桶
    by_dtype: defaultdict[str, int] = defaultdict(int)  # 张量个数×dtype
    layer_expert_logical: defaultdict[int, int] = defaultdict(int)
    expert_sizes=defaultdict(int)
    gate_shapes=[]
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
            storage_bytes += m * DTYPE_BYTES[dt]
            if name.endswith(".weight_packed"):        # MXFP4：每字节 2 个 FP4 权重
                if dt!="U8":raise ValueError("K3 FP4打包须为U8")
                k = m * 2
            elif name.endswith(".weight_scale"):       # E8M0 组标度（g32）——元数据
                scale_elems += m
                continue
            else:
                k = m
            shard_logical += k
            m_layer = re.match(r"language_model\.model\.layers\.(\d+)\.(.+)", name)
            idx, rest = (int(m_layer.group(1)), m_layer.group(2)) if m_layer else (None, name)
            if name.startswith("vision_tower."):
                bucket="vision_tower(MoonViT-V2)"
            elif name.startswith("mm_projector."):
                bucket="mm_projector"
            elif rest.startswith("self_attn."):
                bucket = f"层[{layer_kind.get(idx, '?')}].self_attn"
            elif rest.startswith("block_sparse_moe.experts."):
                bucket = "层.MoE.路由专家(w1w2w3)"
                layer_expert_logical[idx] += k
                expert_id=int(rest.split(".")[2])
                expert_sizes[(idx,expert_id)]+=k
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
            elif m_layer and ("res_proj" in rest or "res_norm" in rest):
                bucket = "层.AttnRes(伪查询+范数)"
            elif m_layer and "layernorm" in rest:
                bucket = "层.RMSNorm"
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
            if m_layer and rest=="self_attn.g_proj.weight":gate_shapes.append(shp)
            add(bucket,k)
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
    if not n_moe_layers or any(v%n_experts for v in layer_expert_logical.values()):
        raise ValueError("路由专家账不能整除")
    if len(set(layer_expert_logical.values()))!=1:raise ValueError("本K3公式要求各MoE层同宽")
    per_expert=next(iter(layer_expert_logical.values()))//n_experts
    if set(expert_sizes)!={(i,j) for i in layer_expert_logical for j in range(n_experts)}:
        raise ValueError("专家ID不完整或越界")
    if set(expert_sizes.values())!={per_expert}:raise ValueError("专家实际宽度不一致")
    if len(gate_shapes)!=n_layers or any(s!=[12288,7168] for s in gate_shapes):
        raise ValueError("本K3门几何已变化，须先更新计数解释")
    vit = by_comp.get("vision_tower(MoonViT-V2)", 0)
    proj = by_comp.get("mm_projector", 0)
    head = by_comp.get("lm_head(输出层)", 0)
    text_backbone = logical - vit - proj
    embed=by_comp.get("embed_tokens(词嵌入)",0)
    activated_text = text_backbone - expert_total + top_k * per_expert * n_moe_layers
    if not xcheck["match"] or unmatched or storage_bytes!=index_total_size:
        raise ValueError("层型/分类/存储对账失败")

    return {
        "repo":repo,"revision":snap.revision,"header_manifest":snap.manifest,
        "scope":"FP4展开后的逻辑元素，含校正等buffer；活跃为结构计数约定",
        "n_shards":len(shards),
        "n_tensors": n_tensors,
        "index_total_size_bytes":index_total_size, "storage_TB_decimal":index_total_size/1e12,
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
            "text_minus_lm_head":activated_text-head,
            "active_without_embeddings":activated_text-head-embed,
            "active_with_output_only":activated_text-embed,
            "B_minus_lm_head": round((activated_text - head) / 1e9, 4),
            "rule": "文本主干全量−路由专家 + top_k(16)×单专家×92 层；含嵌入+输出层+稠密FFN+AttnRes；不含 ViT/projector",
        },
        "calibers": {
            "B_logical_total(全家族含ViT+projector)": round(logical / 1e9, 3),
            "C_logical_text_backbone(文本范围)": round(text_backbone / 1e9, 3),
        },
        "_raw": {"logical": logical, "text_backbone": text_backbone,
                 "storage_bytes": storage_bytes, "vit": vit, "projector": proj,
                 "lm_head":head,"embed":embed,"expert_total":expert_total,
                 "per_expert": per_expert},
    }


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo",action="append",required=True,choices=["zai-org/GLM-5","zai-org/GLM-5.2","moonshotai/Kimi-K3"])
    ap.add_argument("--out",default="log/book5-ch11/params_audit")
    ap.add_argument("--out-name",default="run1")
    ap.add_argument("--revision",help="固定commit SHA；单仓使用")
    ap.add_argument("--snapshot-dir",help="已有本工具格式分片头目录；单仓使用")
    ap.add_argument("--config",help="已有匹配revision的config文件；仅配snapshot-dir")
    ap.add_argument("--offline",action="store_true",help="缺任一小文件/分片头即报错")
    args=ap.parse_args()
    if not args.out_name or Path(args.out_name).name!=args.out_name or args.out_name in (".",".."):
        raise ValueError("out-name须为单个文件名")
    if (args.revision or args.snapshot_dir) and len(args.repo)!=1:raise ValueError("指定快照时逐仓运行")
    if args.config and not args.snapshot_dir:raise ValueError("--config仅用于已有快照")
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    for repo in args.repo:
        slug=repo.replace("/","__");dest=root/f"{slug}_{args.out_name}.json"
        if dest.exists():raise FileExistsError(dest)
        cache=Path(args.snapshot_dir) if args.snapshot_dir else root/"cache"/slug
        snap=Snapshot(repo,cache,args.revision,args.offline,args.config)
        result=audit_k3(repo,snapshot=snap) if repo=="moonshotai/Kimi-K3" else audit(repo,snapshot=snap)
        result["date"]=datetime.now(timezone.utc).isoformat()
        with dest.open("x") as out:json.dump(result,out,ensure_ascii=False,indent=2)
        print(json.dumps({"repo":repo,"revision":snap.revision,"calibers":result["calibers"],
                          "raw":result["_raw"],"active":result.get("activated_params"),"out":str(dest)},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
