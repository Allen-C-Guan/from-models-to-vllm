"""Book5第1章：逻辑配对数、按层状态载荷与当前库的掩码/配置读数。
在工作区source env.sh后运行本文件 --out-name run1；结果仅写log/book5-ch01。
不下载模型，不训练；统一2字节列是教学换算，不是实际运行显存。
"""
from pathlib import Path
import argparse,hashlib,json
HERE=Path(__file__).resolve().parent
REPO_ROOT=next((p for p in HERE.parents if (p/"env.sh").is_file()),Path.cwd())
OUT_DIR=REPO_ROOT/"log/book5-ch01"
GLM5_SNAP=HERE/"configs/glm5.json"
SEED=20261002

def config(name):
    path=HERE/"configs"/(name+".json")
    raw=json.loads(path.read_text())
    return raw.get("text_config",raw)

def find_hf_src():
    """只读取实际安装版本，避免把只读源码仓版本冒充运行版本。"""
    import transformers
    from transformers import masking_utils
    return masking_utils.__file__,f"transformers {transformers.__version__}"

def vocab_excerpt():
    path,source=find_hf_src()
    lines=Path(path).read_text().splitlines()
    start=next(i for i,s in enumerate(lines) if s.startswith("LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING"))
    end=next(i for i in range(start,len(lines)) if lines[i].strip()=="}")
    return [f"L{i+1}: {lines[i]}" for i in range(start,end+1)],source,path

def glm5_layer_types():
    """按随书真实配置构造配置对象；不实例化744B模型。"""
    import transformers
    from transformers.models.glm_moe_dsa import GlmMoeDsaConfig
    raw=config("glm5")
    skip={"architectures","model_type","torch_dtype","dtype","transformers_version","quantization_config"}
    cfg=GlmMoeDsaConfig(**{k:v for k,v in raw.items() if k not in skip})
    lt=list(cfg.layer_types)
    if len(lt)!=raw["num_hidden_layers"]:raise AssertionError("生成层数与快照不符")
    return dict(status="实际安装版本按随书快照构造",num_hidden_layers=len(lt),
        counts={k:lt.count(k) for k in sorted(set(lt))},excerpt=lt[:6]+["..."]+lt[-2:],
        sha256=hashlib.sha256(GLM5_SNAP.read_bytes()).hexdigest()),f"transformers {transformers.__version__}"

def score_matrix_account():
    """每样本、每层、每查询头的逻辑位置数；不是实际显存分配量。"""
    return [dict(n=n,n2_elements_per_layer_head=n*n,causal_valid_pairs=n*(n+1)//2,
                 ratio_vs_8192=round(n*n/8192**2,1)) for n in (2048,8192,131072,1000000)]

def kv_global(L_g,h_kv,d_k,n):
    """假定K/V同宽时的逐项历史载荷：2*Lg*hkv*dk*n。"""
    return 2*L_g*h_kv*d_k*n

def kv_sliding(L_s,h_kv,d_k,W,n):
    """至多保留W个可用位置的载荷估计；后端可能仅持久保存W−1个旧位置。"""
    return 2*L_s*h_kv*d_k*min(n,W)

def state_linear(L_l,state_per_layer):
    return L_l*state_per_layer

def cross_point(L_l,state_per_layer,L_g,c_kv):
    """同一模型的两类有效载荷交叉；无增长项时不存在有限追平点。"""
    return L_l*state_per_layer/(L_g*c_kv) if L_g*c_kv else None

def machines():
    gem=config("gemma3");oss=config("gptoss20");glm=config("glm5");qn=config("qwen3next");k3=config("k3")
    pattern=gem.get("sliding_window_pattern",6)
    ng=sum((i+1)%pattern==0 for i in range(gem["num_hidden_layers"]))
    oss_types=oss["layer_types"];qnl=qn["num_hidden_layers"]//qn["full_attention_interval"]
    kg=k3["linear_attn_config"];kglo=len(kg["full_attn_layers"]);klin=len(kg["kda_layers"])
    qconv=(2*qn["linear_num_key_heads"]*qn["linear_key_head_dim"]
           +qn["linear_num_value_heads"]*qn["linear_value_head_dim"])*(qn["linear_conv_kernel_dim"]-1)
    kconv=3*kg["num_heads"]*kg["head_dim"]*(kg["short_conv_kernel_size"]-1)
    return [
        dict(name="Gemma3-27B",route=1,L=gem["num_hidden_layers"],L_g=ng,L_s=gem["num_hidden_layers"]-ng,L_l=0,
             h_kv=gem["num_key_value_heads"],d_k=gem["head_dim"],W=gem["sliding_window"],ctx=gem["max_position_embeddings"]),
        dict(name="gpt-oss-20b",route=1,L=oss["num_hidden_layers"],L_g=oss_types.count("full_attention"),
             L_s=oss_types.count("sliding_attention"),L_l=0,h_kv=oss["num_key_value_heads"],
             d_k=oss["head_dim"],W=oss["sliding_window"],ctx=oss["max_position_embeddings"]),
        dict(name="GLM-5",route=2,L=glm["num_hidden_layers"],L_g=glm["num_hidden_layers"],L_s=0,L_l=0,
             latent=glm["kv_lora_rank"]+glm["qk_rope_head_dim"],index_key=glm["index_head_dim"],ctx=glm["max_position_embeddings"]),
        dict(name="Qwen3-Next-80B-A3B",route=3,L=qn["num_hidden_layers"],L_g=qnl,L_s=0,L_l=qn["num_hidden_layers"]-qnl,
             h_kv=qn["num_key_value_heads"],d_k=qn["head_dim"],ctx=qn["max_position_embeddings"],
             core=qn["linear_num_value_heads"]*qn["linear_key_head_dim"]*qn["linear_value_head_dim"],conv=qconv),
        dict(name="Kimi K3",route=3,L=k3["num_hidden_layers"],L_g=kglo,L_s=0,L_l=klin,
             latent=k3["kv_lora_rank"]+k3["qk_rope_head_dim"],ctx=k3["max_position_embeddings"],
             core=kg["num_heads"]*kg["head_dim"]**2,conv=kconv)]

MACHINES=machines()

def machine_row(m):
    """主体KV、矩阵状态、最小卷积历史和索引key分别列，不混成实际显存。"""
    n=m["ctx"];per=m.get("latent") or 2*m["h_kv"]*m["d_k"]
    g=per*m["L_g"]*n
    row=dict(name=m["name"],route=m["route"],ctx=n,per_token_per_layer=per,
             kv_global_unbounded=g,kv_sliding_bounded=0,state_core_only=0,conv_history_min=0,
             index_keys=0,mixed_total=g,if_all_full=per*m["L"]*n)
    if m["route"]==1:
        s=per*m["L_s"]*min(n,m["W"])
        row.update(kv_sliding_bounded=s,mixed_total=g+s,
            saving_pct=100*(1-(g+s)/row["if_all_full"]),sliding_share_pct=100*s/(g+s))
    elif m["route"]==2:
        row["index_keys"]=m["L"]*m["index_key"]*n
        row["with_index_keys"]=g+row["index_keys"]
        row["note"]="latent主体设计账；索引key另列。HF eager缓存展开K/V，不等于这里的潜状态布局。"
    else:
        st=m["L_l"]*m["core"];cv=m["L_l"]*m["conv"]
        row.update(state_core_only=st,conv_history_min=cv,linear_state_fixed=st+cv,mixed_total=g+st+cv,
            n_star=cross_point(m["L_l"],m["core"],m["L_g"],per),
            n_star_with_conv_min=cross_point(m["L_l"],m["core"]+m["conv"],m["L_g"],per),
            linear_vs_global_pct=100*st/g,
            note="卷积仅计k−1个历史位置；实现可能保存k个或更多。矩阵状态可用FP32，不能统一当BF16。")
    for key in ("mixed_total","if_all_full","kv_global_unbounded","kv_sliding_bounded",
                "state_core_only","conv_history_min","index_keys","with_index_keys"):
        if key in row:row[key+"_gb_at_2bytes"]=row[key]*2/1e9
    return row

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-name",default="run1",help="结果后缀；已有结果拒绝覆盖")
    args=ap.parse_args()
    if not args.out_name or Path(args.out_name).name!=args.out_name or args.out_name in (".",".."):
        ap.error("out-name须为单个文件名后缀")
    path=OUT_DIR/f"context_account_{args.out_name}.json"
    if path.exists():raise FileExistsError(path)
    scores=score_matrix_account();rows=[machine_row(m) for m in MACHINES]
    excerpt,source,src_path=vocab_excerpt();lt,lt_source=glm5_layer_types()
    report=dict(seed=SEED,score_matrix_account=scores,machines=rows,
        layer_types_vocab=dict(excerpt=excerpt,source=source,path=src_path),
        glm5_layer_types=dict(info=lt,source=lt_source),
        byte_assumption="所有2字节列为统一假设有效载荷，非实际dtype/分配/峰值；不含模型权重和前向临时量")
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    with path.open("x") as handle:json.dump(report,handle,ensure_ascii=False,indent=2)
    for row in scores:print(row)
    for row in rows:print(json.dumps(row,ensure_ascii=False))
    print(lt,source);print("产物：",path)
if __name__=="__main__":main()
