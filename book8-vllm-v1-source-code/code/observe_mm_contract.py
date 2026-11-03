"""CPU多模态位置和encoder缓存管理实验；无视觉模型前向。"""
import hashlib, inspect, json
from pathlib import Path
import torch
from vllm.multimodal.inputs import MultiModalFeatureSpec,PlaceholderRange
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.request import Request
from vllm.sampling_params import SamplingParams
ROOT=Path(__file__).resolve().parents[2]

def main():
    def request(name,identifier):
        feature=MultiModalFeatureSpec(data=None,modality='image',identifier=identifier,mm_position=PlaceholderRange(2,4))
        return Request(name,[1,2,99,99,99,99,3],SamplingParams(max_tokens=2),None,mm_features=[feature])
    a,c,b=request('A-image','image-X'),request('C-image','image-X'),request('B-image','image-Y')
    manager=EncoderCacheManager(cache_size=4);states=[]
    def snap(event):states.append({'event':event,'free':manager.num_free_slots,'reclaimable':manager.num_freeable_slots,'cached':{k:sorted(v) for k,v in manager.cached.items()},'freeable':dict(manager.freeable)})
    assert manager.can_allocate(a,0,4,0);manager.allocate(a,0);snap('allocate A/X')
    assert manager.check_and_update_cache(c,0);snap('C hits X')
    manager.free(a);snap('free A')
    assert not manager.can_allocate(b,0,4,0);snap('B/Y blocked while C holds X')
    manager.free(c);snap('free C, X retained')
    assert manager.can_allocate(b,0,4,0);manager.allocate(b,0);freed=manager.get_freed_mm_hashes();snap('allocate B/Y, evict X')
    assert freed==['image-X']
    pos=PlaceholderRange(2,5,torch.tensor([False,True,False,True,True]))
    positions={'offset':pos.offset,'length':pos.length,'is_embed':pos.is_embed.tolist(),'num_embeds':pos.get_num_embeds(),'relative_range':[3,5],'embedding_slice':list(pos.get_embeds_indices_in_range(3,5))}
    assert positions['num_embeds']==3 and positions['embedding_slice']==[1,3]
    files={}
    for cls,rel in [(PlaceholderRange,'vllm/multimodal/inputs.py'),(EncoderCacheManager,'vllm/v1/core/encoder_cache_manager.py')]:
        p=ROOT/'repos/vllm'/rel;assert Path(inspect.getsourcefile(cls)).read_bytes()==p.read_bytes();files[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
    out=ROOT/'drafts/Book8-vLLM源码剖析/evidence/mm-contract';out.mkdir(exist_ok=True)
    (out/'trace.json').write_text(json.dumps({'kind':'CPU actual Request/PlaceholderRange/EncoderCacheManager; synthetic hashes and IDs; no media processing, encoder tensors or model execution','probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':files,'positions':positions,'states':states,'freed_hashes':freed},ensure_ascii=False,indent=2)+'\n');print('CPU MM positions, sharing, reference release and eviction verified')

if __name__=='__main__':main()
