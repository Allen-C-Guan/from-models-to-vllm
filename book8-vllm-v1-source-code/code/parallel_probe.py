"""可由spawn子进程导入的只读模型观测函数。"""
def model_view(model):
    from vllm.distributed.parallel_state import get_tp_group,get_pp_group
    import os
    trunk=model.model;layer=trunk.layers[trunk.start_layer]
    tp=get_tp_group();pp=get_pp_group()
    return {'pid':os.getpid(),'tp':{'rank':tp.rank_in_group,'ranks':tp.ranks,'size':tp.world_size},
        'pp':{'rank':pp.rank_in_group,'ranks':pp.ranks,'size':pp.world_size},
        'layer_range':[trunk.start_layer,trunk.end_layer],
        'layer_class':type(layer).__name__,'model_class':type(model).__name__,
        'qkv_weight':list(layer.self_attn.qkv_proj.weight.shape),
        'o_weight':list(layer.self_attn.o_proj.weight.shape),
        'num_heads':layer.self_attn.num_heads,'num_kv_heads':layer.self_attn.num_kv_heads,
        'embedding_class':type(trunk.embed_tokens).__name__,'lm_head_class':type(model.lm_head).__name__}

