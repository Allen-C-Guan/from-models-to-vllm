"""复用已验证的 A/B 驱动，额外观测 Runner 边界；不改原脚本或依赖。"""
from pathlib import Path
import hashlib
import json
import sys
import observe_requests as base

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'log/book8-source-rewrite/runner-inproc-traced'
original_instrument=base.instrument
REC=None


def instrument(engine,rec):
    global REC
    REC=rec
    original_instrument(engine,rec)
    worker=engine.engine_core.engine_core.model_executor.driver_worker.worker
    runner=worker.model_runner
    rec.wrap(runner,'add_requests',after=lambda a,k,r:rec.emit('runner.registered',req_id_to_index=runner.req_states.req_id_to_index))
    rec.wrap(runner,'prepare_inputs',after=lambda a,k,r:rec.emit('runner.inputs',batch=base.pick(r,[
        'req_ids','num_reqs','num_tokens','num_tokens_after_padding','idx_mapping','idx_mapping_np',
        'num_scheduled_tokens','input_ids','positions','query_start_loc','seq_lens','logits_indices',
        'num_computed_tokens_np','prefill_len_np'])))
    rec.wrap(runner,'prepare_attn',after=lambda a,k,r:rec.emit('runner.attn',block_tables=r[0],slot_mappings=r[1]))
    rec.wrap(runner,'sample',before=lambda a,k:rec.emit('runner.hidden',hidden=base.snapshot(a[0])))
    rec.wrap(runner.model,'compute_logits',after=lambda a,k,r:rec.emit('model.logits',logits=base.snapshot(r)))


def main():
    base.instrument=instrument
    sys.argv=[sys.argv[0],'--case','baseline','--deployment','inproc','--out',str(OUT)]
    base.main()
    (OUT/'runner-provenance.json').write_text(json.dumps({'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'driver_sha256':hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),'methods':REC.methods,'observation':'NPU synchronous snapshot; not timing evidence'},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
