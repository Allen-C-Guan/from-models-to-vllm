"""双 NPU 的 TP/PP 对照：读取每个rank的模型元数据并生成同一A/B。
在启动前由外部选择两张空闲卡；不用本脚本测吞吐。
"""
from pathlib import Path
import argparse,hashlib,importlib.metadata,json,os
from observe_requests import MODEL,ROOT,snapshot


from parallel_probe import model_view


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['tp','pp'],required=True);p.add_argument('--run-name');args=p.parse_args()
    # 本机离线探针传递由本文件定义的函数；不启动对外服务。
    os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION']='1'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='1'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn';os.environ['VLLM_USE_V2_MODEL_RUNNER']='1'
    from vllm import LLM,SamplingParams
    out=ROOT/'log/book8-source-rewrite'/(args.run_name or f'{args.mode}2');out.mkdir(exist_ok=False)
    settings=dict(model=str(MODEL),tokenizer=str(MODEL),dtype='bfloat16',max_model_len=128,
        max_num_seqs=4,max_num_batched_tokens=32,enforce_eager=True,async_scheduling=False,
        enable_prefix_caching=False,enable_chunked_prefill=True,gpu_memory_utilization=.15,
        kv_cache_memory_bytes=64*1024*1024,seed=0,disable_log_stats=True,
        distributed_executor_backend='mp',tensor_parallel_size=2 if args.mode=='tp' else 1,
        pipeline_parallel_size=2 if args.mode=='pp' else 1)
    llm=LLM(**settings)
    try:
        views=llm.apply_model(model_view)
        outputs=llm.generate(['The capital of France is','The largest ocean on Earth is'],
            [SamplingParams(temperature=0,max_tokens=n,ignore_eos=True,seed=0) for n in [2,3]],use_tqdm=False)
        result={'ranks':views,'outputs':[snapshot(o) for o in outputs]}
        assert [list(o.outputs[0].token_ids) for o in outputs]==[[12095,13],[279,16462,21575]]
        assert len(views)==2
        if args.mode=='tp':
            assert all(v['qkv_weight']==[2048,1024] and v['num_heads']==8 and v['num_kv_heads']==4 for v in views)
        else:assert sorted(v['layer_range'] for v in views)==[[0,14],[14,28]]
        manifest={'settings':settings,'packages':{n:importlib.metadata.version(n) for n in ['vllm','vllm-ascend','torch','torch-npu']},
            'visible_devices':os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),'model_revision':MODEL.name,
            'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'model_probe_sha256':hashlib.sha256(Path(__file__).with_name('parallel_probe.py').read_bytes()).hexdigest(),
            'boundary':'worker model metadata and final outputs only; no collective timing or captured communication tensors'}
        (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(result,ensure_ascii=False))
    finally:llm.llm_engine.engine_core.shutdown()


if __name__=='__main__':main()
