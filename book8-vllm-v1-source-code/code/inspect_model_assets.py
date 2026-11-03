"""读取固定模型快照配置和 safetensors 头部，不加载权重张量。"""
from pathlib import Path
import hashlib,json,struct
from observe_requests import MODEL,ROOT


def main():
    config=MODEL/'config.json';weight=MODEL/'model.safetensors'
    with weight.open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0];header=f.read(size)
    tensors=json.loads(header)
    names=['model.embed_tokens.weight']+[f'model.layers.0.self_attn.{p}.weight' for p in ['q_proj','k_proj','v_proj','o_proj']]+[f'model.layers.0.mlp.{p}.weight' for p in ['gate_proj','up_proj','down_proj']]
    result={'model_revision':MODEL.name,'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),'header_sha256':hashlib.sha256(header).hexdigest(),'config':json.loads(config.read_text()),'weights':{n:{k:tensors[n][k] for k in ['dtype','shape']} for n in names}}
    out=ROOT/'drafts/Book8-vLLM源码剖析/evidence/model-assets';out.mkdir(exist_ok=True)
    (out/'metadata.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result['weights'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
