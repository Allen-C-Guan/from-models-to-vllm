"""只检查平台插件与类定义来源；不实例化Worker、不加载模型。"""
from pathlib import Path
import hashlib,importlib.metadata,inspect,json
from vllm.platforms import current_platform
from vllm_ascend.worker.worker import NPUWorker
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionBackendImpl, AscendAttentionMetadataBuilder
ROOT=Path(__file__).resolve().parents[2]


def info(cls):
    path=Path(inspect.getfile(cls));parts=path.parts
    repo='vllm-ascend' if 'vllm_ascend' in parts else 'vllm';package='vllm_ascend' if repo=='vllm-ascend' else 'vllm'
    rel=Path(*parts[parts.index(package):]);fixed=ROOT/'repos'/repo/rel
    assert path.read_bytes()==fixed.read_bytes()
    return {'class':cls.__module__+'.'+cls.__name__,'mro':[c.__module__+'.'+c.__name__ for c in cls.__mro__],
        'repo':repo,'source':str(rel),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    r={'kind':'installed entry points and class introspection; no worker instance or model execution',
       'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'entry_points':[{'name':e.name,'group':e.group,'value':e.value} for e in importlib.metadata.distribution('vllm-ascend').entry_points],
       'platform':info(type(current_platform)),'worker':info(NPUWorker),'runner':info(NPUModelRunner),
       'attention_backend':info(AscendAttentionBackend),'attention_impl':info(AscendAttentionBackendImpl),
       'attention_builder':info(AscendAttentionMetadataBuilder)}
    out=ROOT/'drafts/Book8-vLLM源码剖析/evidence/platform';out.mkdir(exist_ok=True)
    (out/'metadata.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print('Platform class origins and MRO match fixed repositories; no model run')


if __name__=='__main__':main()
