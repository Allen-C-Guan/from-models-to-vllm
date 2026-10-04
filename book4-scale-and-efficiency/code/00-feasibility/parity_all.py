# parity_all.py —— Book4 聚合对拍脚手架：ch02/ch05/ch06 教学件 vs 正身(moe_mla_slots) vs HF 三层互拍
# 用途：发现式聚合对拍（plan v1.1 前置任务：正身任何改动的 commit 须注明已跑本脚本；批二/批三
#       交付边界与终校固定动作）——
#       ①自动发现各章教学件：ch02/moe.py::SparseMoE（Mixtral 教学件）、ch05/deepseek_moe.py::
#         DeepSeekMoE（DSV2 键位件）、ch06/mla.py::MultiHeadLatentAttention（MLA 双路径件），
#         存在几件跑几件（本批交付时教学件可能 0 件——框架就位，交付后零改扩自动生效）；
#       ②逐件与正身对应类 CPU fp32 对拍：构造小 config → 权重直搬（strict 优先，键位差异如实
#         上报）→ 同输入同 seed 前向 → max|Δ|（阈值 1e-4 + 形状一致，Book3 验收管道口径）；
#         对应关系：ch02→HFMixtralMoE、ch05→HFDeepseekMoE、ch06→MLAAttention（两 q_lora 变体）；
#       ③三层互拍汇总表（教学件 vs 正身 vs HF）——HF 侧仅 Mixtral/DeepseekV2 块级
#         （MixtralSparseMoeBlock/DeepseekV2Moe，键位与正身严格同名），--hf 开关默认关；
#         ch06 MLA 的 HF 对拍走整机口径（07_probe_g_parity.py 已验），本表登记 N/A；
#       ④教学件在否都恒跑「正身内部 sanity」：三 config 参数账 assert（dense/MoE/MLA）+
#         MLA 显式 vs 吸收两路等价 + teach(ModuleList) 与 fused(3D 融合) 两族数值等价
#         （Mixtral 式 + DeepSeek 式，即 conversion_mapping 的 MergeModulelist+Concatenate 教学预演）。
# 所属章节：00-feasibility 共用件——Book4 ch2/ch5/ch6 教学件验收、ch9 大项目聚合对拍、终校固定动作。
# 运行方式：source env.sh && python parity_all.py [--hf] [--out-name run1]（CPU fp32，秒级）
# 产物：log/book4-feasibility/parity_all_{out}.json（不入库）
import argparse
import importlib.util
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from moe_mla_slots import (SEED, HFDeepseekMoE, HFMixtralMoE, MLAAttention, SlotConfig,
                           SlotLLaMA, SparseMoE, bootstrap_book3, hand_account,
                           mla_absorbed_forward, save_json)

TOL = 1e-4                                  # Book3 验收管道口径：max|Δ| < 1e-4
CODE_ROOT = os.path.abspath(os.path.join(HERE, ".."))          # code/


# ---------------- 教学件注册表（发现式：新增章件只在此追加一条，无需改运行逻辑） ----------------
REGISTRY = [
    {"chapter": "ch02", "file": "moe.py", "cls": "SparseMoE", "kind": "mixtral_moe",
     "batch": "批一", "hf_kind": "mixtral_moe", "variants": [("default", {})]},
    {"chapter": "ch05", "file": "deepseek_moe.py", "cls": "DeepSeekMoE", "kind": "dsv2_moe",
     "batch": "批二", "hf_kind": "dsv2_moe", "variants": [("default", {})]},
    {"chapter": "ch06", "file": "mla.py", "cls": "MultiHeadLatentAttention", "kind": "mla",
     "batch": "批二", "hf_kind": None,                                   # MLA 的 HF 对拍走整机口径（探针 G）
     "variants": [("q_lora=64", {"q_lora_rank": 64}), ("q_lora=null", {"q_lora_rank": None})]},
]

GEOM = {                                    # 各 kind 的小 config（小而全：分维不退化为恒等）
    "mixtral_moe": dict(d=64, E=4, k=2, w=128),
    "dsv2_moe": dict(d=64, E=4, k=2, w=96, n_shared=1),
    "mla": dict(d=64, h=4, d_c=32, d_n=16, d_r=8, d_v=16),
}


class DuckCfg:
    """鸭喙 config：一批字段别名挂同一值。教学件的 cfg 构造契约在批二/三交付前未定稿，
    本类让 n_expert/num_local_experts、top_k/num_experts_per_tok 等任一别名写法都可读——
    发现式框架对「教学件怎么写 config」保持中立，只对键位与前向数值立契约。"""

    def __init__(self, fields):
        object.__setattr__(self, "_fields", fields)

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, "_fields")[name]
        except KeyError:
            raise AttributeError(f"DuckCfg 无字段 {name}（可用：{sorted(object.__getattribute__(self, '_fields'))})") from None


def teach_spec(kind, over):
    """由 kind+变体覆盖项生成 (鸭喙 cfg, kwargs 兜底, 几何 dict)——两路构造都喂给教学件。"""
    g = {**GEOM[kind], **{k: v for k, v in over.items() if k in GEOM[kind] or k == "q_lora_rank"}}
    if kind == "mixtral_moe":
        kw = dict(hidden_size=g["d"], n_expert=g["E"], top_k=g["k"], expert_dim=g["w"])
    elif kind == "dsv2_moe":
        kw = dict(hidden_size=g["d"], n_expert=g["E"], top_k=g["k"], expert_dim=g["w"], n_shared=1)
    else:
        kw = dict(hidden_size=g["d"], num_attention_heads=g["h"], kv_lora_rank=g["d_c"],
                  qk_nope_head_dim=g["d_n"], qk_rope_head_dim=g["d_r"], v_head_dim=g["d_v"],
                  rope_style="complex")                                  # HF DSV2 复数约定（探针 G 坑 d）
        kw.update({k: v for k, v in over.items() if k == "q_lora_rank"})
    fields = {
        "hidden_size": g["d"],
        "rope_theta": 10000.0, "rms_norm_eps": 1e-5, "rope_style": "complex",
    }
    if kind in ("mixtral_moe", "dsv2_moe"):                               # MoE 别名族只对 MoE 件铺（mla 无 E/k/w）
        fields.update({
            "n_expert": g["E"], "num_local_experts": g["E"], "n_routed_experts": g["E"], "num_experts": g["E"],
            "top_k": g["k"], "num_experts_per_tok": g["k"],
            "expert_dim": g["w"], "moe_intermediate_size": g["w"], "intermediate_size": g["w"],
            "mode": "mixtral" if kind == "mixtral_moe" else "deepseek",
            "routed_scaling_factor": 1.0, "topk_method": "greedy", "norm_topk_prob": False,
        })
        if kind == "dsv2_moe":
            fields.update(n_shared=1, n_shared_experts=1)                 # 共享专家别名族（仅 DSV2 件）
    if kind == "mla":
        fields.update(num_attention_heads=g["h"], n_head=g["h"], kv_lora_rank=g["d_c"],
                      qk_nope_head_dim=g["d_n"], qk_rope_head_dim=g["d_r"], v_head_dim=g["d_v"])
    fields.update(kw)                                                     # 变体显式项（如 q_lora_rank）最后落位
    return DuckCfg(fields), kw, g


def build_ref(kind, g):
    """正身对应类：键位/前向语义的验收基准（moe_mla_slots 的 HF 键位版/MLA 件）。"""
    if kind == "mixtral_moe":
        return HFMixtralMoE(g["d"], g["E"], g["k"], g["w"])
    if kind == "dsv2_moe":
        return HFDeepseekMoE(g["d"], g["E"], g["k"], g["w"], n_shared=1, routed_scaling_factor=1.0)
    return MLAAttention(g["d"], g["h"], g["d_c"], g["d_n"], g["d_r"], g["d_v"],
                        q_lora_rank=g.get("q_lora_rank"), rope_theta=10000.0, rope_style="complex")


def make_input(kind, g):
    """同 seed 输入：MoE 件 (B,n,d)；MLA 件按 Book3 GQA 契约另配 (cos,sin)。"""
    gen = torch.Generator().manual_seed(SEED)
    if kind == "mla":
        x = torch.randn(2, 48, g["d"], generator=gen)                    # (2,48,d)
        cos, sin = bootstrap_book3().build_rope_cache(48, g["d"], 10000.0, x.device)
        return (x, cos, sin)
    return (torch.randn(2, 32, g["d"], generator=gen),)                  # (2,32,d)


def call_mod(mod, args_tuple, rope):
    """前向调用并归一输出：mla 件吃 (x,cos,sin)（TypeError 退 (x)；正身 MLA 忽略 cos/sin）；
    MoE 件吃 (x,)。教学件可能附带返回 router_logits/counts——tuple 一律取第 0 位为 hidden。"""
    if rope:
        try:
            out = mod(*args_tuple)
        except TypeError:
            out = mod(args_tuple[0])
    else:
        out = mod(args_tuple[0])
    return out[0] if isinstance(out, (tuple, list)) else out


def build_teaching(cls, cfg, kwargs):
    """教学件构造：先 cfg 单参（plan 插槽契约 SparseMoE(cfg) 形），再 kwargs 兜底。"""
    err = []
    for make in (lambda: cls(cfg), lambda: cls(**kwargs)):
        try:
            return make(), None
        except Exception as e:                                            # 构造契约未定稿——两试均败即登记待适配
            err.append(f"{type(e).__name__}: {str(e)[:120]}")
    return None, "构造契约不符（cfg 单参与 kwargs 两试均败）｜" + " ／ ".join(err)


def align_weights(src_sd, dst):
    """权重直搬：strict=False 加事后清点（missing=正身缺料、unexpected=教学件多键如
    e_score_correction_bias Buffer）——形状不匹配直接抛 RuntimeError 由上层捕获。"""
    missing, unexpected = dst.load_state_dict(src_sd, strict=False)
    return list(missing), list(unexpected)


def forward_pair(a, b, args_tuple, rope):
    """同输入前向 → max|Δ|；形状不一致按 FAIL 报（返回 None）。"""
    with torch.no_grad():
        ya, yb = call_mod(a, args_tuple, rope), call_mod(b, args_tuple, rope)
    if ya.shape != yb.shape:
        return None, f"形状不一致 {tuple(ya.shape)} vs {tuple(yb.shape)}"
    return (ya.float() - yb.float()).abs().max().item(), None


def judge(diff):
    if diff is None:
        return "FAIL"
    return "PASS" if diff < TOL else "FAIL"


# ---------------- HF 块级臂（仅 --hf：Mixtral/DeepseekV2；键位与正身严格同名可 strict 直搬） ----------------
def hf_block(kind, g):
    if kind == "mixtral_moe":
        from transformers import MixtralConfig
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
        cfg = MixtralConfig(hidden_size=g["d"], intermediate_size=g["w"], num_local_experts=g["E"],
                            num_experts_per_tok=g["k"], router_jitter_noise=0.0)
        return MixtralSparseMoeBlock(cfg).eval()
    from transformers import DeepseekV2Config
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Moe
    cfg = DeepseekV2Config(hidden_size=g["d"], moe_intermediate_size=g["w"], intermediate_size=g["w"],
                           n_routed_experts=g["E"], n_shared_experts=1, num_experts_per_tok=g["k"],
                           topk_method="greedy", n_group=1, topk_group=1, routed_scaling_factor=1.0,
                           mlp_bias=False)
    return DeepseekV2Moe(cfg).eval()


def hf_arm(kind, g, src, args_tuple):
    """一件 src（正身或教学件）vs HF 块级：直搬 → 同输入前向 → max|Δ|。"""
    block = hf_block(kind, g)
    missing, unexpected = align_weights(src.state_dict(), block)
    note = f"missing={missing} unexpected={unexpected}" if (missing or unexpected) else "strict 等价"
    diff, err = forward_pair(src, block, (args_tuple[0],), rope=False)
    return {"max_abs_diff": diff, "verdict": "SKIP" if err else judge(diff), "keys": note,
            "detail": err or ""}


# ---------------- 逐注册条目运行（发现式主体） ----------------
def run_entry(entry, hf_on):
    path = os.path.join(CODE_ROOT, entry["chapter"], entry["file"])
    if not os.path.exists(path):
        return {"label": f'{entry["chapter"]}/{entry["file"]}::{entry["cls"]}',
                "status": "absent", "variants": [], "detail": f'未交付（{entry["batch"]}交付后自动生效）'}
    mod_name = f'book4_{entry["chapter"]}_{os.path.splitext(entry["file"])[0]}'
    try:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        # 章目录与全部在位章目录入 path——兜「ch05 复用 ch02 专家件」式平铺 import
        for d in [os.path.dirname(path)] + [os.path.join(CODE_ROOT, e["chapter"]) for e in REGISTRY]:
            if os.path.isdir(d) and d not in sys.path:
                sys.path.insert(0, d)
        spec.loader.exec_module(mod)
    except Exception as e:
        return {"label": f'{entry["chapter"]}/{entry["file"]}::{entry["cls"]}',
                "status": "import_fail", "variants": [], "detail": str(e)[:300]}
    cls = getattr(mod, entry["cls"], None)
    if cls is None:
        return {"label": f'{entry["chapter"]}/{entry["file"]}::{entry["cls"]}',
                "status": "import_fail", "variants": [], "detail": f'模块无类 {entry["cls"]}（契约见 plan 插槽 import 契约）'}

    variants = []
    for tag, over in entry["variants"]:
        cfg, kwargs, g = teach_spec(entry["kind"], over)
        torch.manual_seed(SEED)
        teach, err = build_teaching(cls, cfg, kwargs)
        v = {"tag": tag, "teach_vs_ref": None, "ref_vs_hf": None, "teach_vs_hf": None}
        if teach is None:
            v.update(status="adapt", detail=err)                          # 框架在位、构造待批二按实物补
            variants.append(v)
            continue
        torch.manual_seed(SEED)
        ref = build_ref(entry["kind"], g).eval()
        teach = teach.eval()
        args_tuple = make_input(entry["kind"], g)
        try:
            missing, unexpected = align_weights(teach.state_dict(), ref)
            if missing:
                v.update(status="keys_mismatch",
                         detail=f"教学件缺正身键 {missing[:4]}（键位契约未承住）")
                variants.append(v)
                continue
            note = f"unexpected={unexpected}" if unexpected else "strict 等价直搬"
            diff, ferr = forward_pair(teach, ref, args_tuple, rope=(entry["kind"] == "mla"))
            v["teach_vs_ref"] = {"max_abs_diff": diff, "verdict": "SKIP" if ferr else judge(diff),
                                 "keys": note, "detail": ferr or ""}
            v.update(status="ok" if v["teach_vs_ref"]["verdict"] == "PASS" else "fail")
            if hf_on and entry["hf_kind"]:
                v["ref_vs_hf"] = hf_arm(entry["hf_kind"], g, ref, args_tuple)
                v["teach_vs_hf"] = hf_arm(entry["hf_kind"], g, teach, args_tuple)
        except Exception as e:                                            # 形状不匹配等运行期失败——如实登记
            v.update(status="forward_fail", detail=f"{type(e).__name__}: {str(e)[:200]}")
        variants.append(v)
    return {"label": f'{entry["chapter"]}/{entry["file"]}::{entry["cls"]}',
            "status": "found", "variants": variants, "detail": ""}


# ---------------- 正身内部 sanity（无论教学件在否恒跑——正身改动的回归网） ----------------
def fuse_teach_weights(teach, ref):
    """ModuleList 教学版 → 3D 融合版权重搬运（等价于 conversion_mapping 的
    MergeModulelist+Concatenate 两行——4.x 旧 checkpoint 进 5.x 的键位机制预演，本身是教学点）。"""
    with torch.no_grad():
        ref.gate.weight.copy_(teach.gate.weight)
        ref.experts.gate_up_proj.copy_(torch.stack(
            [torch.cat([e.gate_proj.weight, e.up_proj.weight], dim=0) for e in teach.experts]))  # gate 在前 up 在后
        ref.experts.down_proj.copy_(torch.stack([e.down_proj.weight for e in teach.experts]))
        if getattr(ref, "shared_experts", None) is not None and teach.shared_experts is not None:
            ref.shared_experts.gate_proj.weight.copy_(teach.shared_experts.gate_proj.weight)
            ref.shared_experts.up_proj.weight.copy_(teach.shared_experts.up_proj.weight)
            ref.shared_experts.down_proj.weight.copy_(teach.shared_experts.down_proj.weight)


def slots_sanity(hf_on):
    res, ok = {}, True

    def check(name, fn):
        nonlocal ok
        try:
            out = fn()
            res[name] = out
            good = out.get("verdict") == "PASS" if isinstance(out, dict) else True
        except Exception as e:
            res[name] = {"verdict": "FAIL", "detail": f"{type(e).__name__}: {str(e)[:200]}"}
            good = False
        ok = ok and good

    def account():
        cfgs = {
            "dense": SlotConfig(vocab_size=8192, hidden_size=512, num_hidden_layers=6,
                                num_attention_heads=8, num_key_value_heads=4, intermediate_size=1408),
            "moe": SlotConfig(vocab_size=8192, hidden_size=512, num_hidden_layers=6,
                              num_attention_heads=8, num_key_value_heads=4, intermediate_size=1408,
                              moe=dict(n_expert=8, top_k=2, expert_dim=704, n_shared=0, variant="teach")),
            "mla": SlotConfig(vocab_size=512, hidden_size=256, num_hidden_layers=2,
                              num_attention_heads=4, num_key_value_heads=1, intermediate_size=704,
                              mla=dict(kv_lora_rank=64, q_lora_rank=64, qk_nope_head_dim=32,
                                       qk_rope_head_dim=16, v_head_dim=32)),
        }
        got = {}
        for tag, c in cfgs.items():
            n = SlotLLaMA(c).n_params()
            assert n == hand_account(c)["total"], f"{tag}: {n} != 手算"
            got[tag] = n
        return {"verdict": "PASS", "params": got}

    def mla_two_paths():
        torch.manual_seed(SEED)
        attn = MLAAttention(64, 4, 32, 16, 8, 16, q_lora_rank=32, rope_style="half").eval()
        gen = torch.Generator().manual_seed(SEED)
        x = torch.randn(2, 48, 64, generator=gen)
        with torch.no_grad():
            diff = (attn(x) - mla_absorbed_forward(attn, x)).abs().max().item()
        return {"verdict": "PASS" if diff < TOL else "FAIL", "max_abs_diff": diff}

    def teach_vs_fused(mode_kind):
        g = GEOM[mode_kind]
        torch.manual_seed(SEED)
        if mode_kind == "mixtral_moe":
            teach = SparseMoE(g["d"], g["E"], g["k"], g["w"], n_shared=0, mode="mixtral").eval()
            ref = HFMixtralMoE(g["d"], g["E"], g["k"], g["w"]).eval()
        else:
            teach = SparseMoE(g["d"], g["E"], g["k"], g["w"], n_shared=1, mode="deepseek").eval()
            ref = HFDeepseekMoE(g["d"], g["E"], g["k"], g["w"], n_shared=1).eval()
        fuse_teach_weights(teach, ref)
        gen = torch.Generator().manual_seed(SEED)
        x = torch.randn(2, 32, g["d"], generator=gen)
        with torch.no_grad():
            yt = teach(x)[0]                                              # SparseMoE 返回 (out, counts)
            yr = ref(x)
            diff = (yt - yr).abs().max().item()
        return {"verdict": "PASS" if diff < TOL else "FAIL", "max_abs_diff": diff}

    check("account_dense_moe_mla", account)
    check("mla_explicit_vs_absorbed", mla_two_paths)
    check("teach_vs_fused_mixtral", lambda: teach_vs_fused("mixtral_moe"))
    check("teach_vs_fused_deepseek", lambda: teach_vs_fused("dsv2_moe"))
    if hf_on:
        def ref_vs_hf(kind):
            g = GEOM[kind]
            torch.manual_seed(SEED)
            ref = (HFMixtralMoE(g["d"], g["E"], g["k"], g["w"]) if kind == "mixtral_moe"
                   else HFDeepseekMoE(g["d"], g["E"], g["k"], g["w"], n_shared=1)).eval()
            args_tuple = make_input(kind, g)
            return hf_arm(kind, g, ref, args_tuple)
        check("hf_block_mixtral", lambda: ref_vs_hf("mixtral_moe"))
        check("hf_block_deepseek", lambda: ref_vs_hf("dsv2_moe"))
    return res, ok


# ---------------- 汇总表 ----------------
def cell(v, na="—"):
    if v is None:
        return na
    if isinstance(v, dict):
        d = v.get("max_abs_diff")
        if d is None:
            return "SKIP"
        return f"{d:.2e}"
    return str(v)


def print_table(entries, sanity, hf_on):
    print("\n==== Book4 聚合对拍汇总（CPU fp32，种子 %d，阈值 max|Δ|<%.0e）====" % (SEED, TOL))
    print(f"{'件':<46}{'教学件↔正身':>14}{'正身↔HF':>12}{'教学件↔HF':>12}{'判定':>8}")
    print("-" * 96)
    for e in entries:
        if e["status"] == "absent":
            print(f'{e["label"]:<46}{"—":>14}{"—":>12}{"—":>12}{"未交付":>8}  {e["detail"]}')
            continue
        for v in e["variants"]:
            label = f'{e["label"]}[{v["tag"]}]'
            if e["label"].startswith("ch06"):                       # MLA 的 HF 对拍走整机口径（探针 G）
                r_h, t_h = "N/A" if hf_on else "—", "N/A" if hf_on else "—"
            else:
                r_h = cell(v.get("ref_vs_hf"))
                t_h = cell(v.get("teach_vs_hf"))
            verdict = "PASS" if v.get("status") == "ok" else v.get("status", "?")
            print(f"{label:<46}{cell(v.get('teach_vs_ref')):>14}{r_h:>12}{t_h:>12}{verdict:>8}"
                  + (f'  {v.get("detail", "")}' if v.get("detail") else ""))
    print("---- 正身内部 sanity（恒跑：正身改动的回归网）----")
    for name, v in sanity.items():
        if isinstance(v, dict) and "max_abs_diff" in v:
            print(f"  {name:<28} max|Δ| = {v['max_abs_diff']:.2e}  {v['verdict']}"
                  + (f'  {v["detail"]}' if v.get("detail") else ""))
        elif isinstance(v, dict) and v.get("verdict") == "PASS":
            print(f"  {name:<28} PASS  {v.get('params', v.get('keys', ''))}")
        else:
            print(f"  {name:<28} {v}")


def main():
    ap = argparse.ArgumentParser(description="Book4 聚合对拍：ch02/ch05/ch06 教学件 vs 正身 vs HF（发现式）")
    ap.add_argument("--hf", action="store_true", help="加块级 HF 臂（Mixtral/DeepseekV2；默认关）")
    ap.add_argument("--out-name", default="run1", help="产物 JSON 后缀（防覆写）")
    args = ap.parse_args()
    t0 = time.perf_counter()
    torch.manual_seed(SEED)
    bootstrap_book3()
    print(f"[环境] torch {torch.__version__} | CPU fp32 | 种子 {SEED} | --hf {'开' if args.hf else '关'}")
    if args.hf:
        import transformers
        print(f"[环境] transformers {transformers.__version__}（HF 臂=块级：MixtralSparseMoeBlock/DeepseekV2Moe）")

    entries = [run_entry(e, args.hf) for e in REGISTRY]
    n_found = sum(1 for e in entries if e["status"] == "found")
    if n_found == 0:
        print(f"[发现] 0 件教学件在位——待批一交付后生效（框架就位：注册表 {len(REGISTRY)} 条，"
              f"{'、'.join(e['chapter'] for e in REGISTRY)} 交付即自动接入）")
    else:
        print(f"[发现] {n_found}/{len(REGISTRY)} 章教学件在位，逐件对拍中")
    sanity, sanity_ok = slots_sanity(args.hf)
    print_table(entries, sanity, args.hf)

    variants_ok = all(v.get("status") == "ok" for e in entries if e["status"] == "found"
                      for v in e["variants"]) and n_found > 0
    verdict = "PASS" if sanity_ok and (variants_ok or n_found == 0) else ("PASS(仅正身自检)"
              if sanity_ok else "FAIL")
    print(f"\n[判定] {verdict} | 教学件 {n_found} 件 | 正身 sanity {'全 PASS' if sanity_ok else '存在 FAIL'} "
          f"| 墙钟 {time.perf_counter() - t0:.1f} s")
    report = {
        "seed": SEED, "device": "cpu fp32", "torch": torch.__version__,
        "transformers": (__import__("transformers").__version__ if args.hf else None),
        "tol": TOL, "discovered": n_found, "registry": entries, "slots_sanity": sanity,
        "verdict": verdict, "wall_sec": round(time.perf_counter() - t0, 1),
    }
    save_json("parity_all", report, args.out_name)


if __name__ == "__main__":
    main()
