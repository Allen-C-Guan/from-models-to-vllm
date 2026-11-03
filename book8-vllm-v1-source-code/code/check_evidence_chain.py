"""核对全册观测的探针、固定源码与派生输入哈希；不启动模型。"""
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'drafts/Book8-vLLM源码剖析/evidence'
REPORT = ROOT / 'research/Book8-源码重写/full/evidence-chain-validation.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    probe_digests = {sha(p): p.name for p in CODE.glob('*.py')}
    checks = []

    def verify(path, expected, origin):
        assert path.is_file(), (origin, path)
        assert sha(path) == expected, (origin, path, 'hash mismatch')
        checks.append({'origin': origin, 'file': str(path.relative_to(ROOT)), 'sha256': expected})

    def walk(value, record, location=''):
        if isinstance(value, list):
            for i, child in enumerate(value):
                walk(child, record, f'{location}/{i}')
        elif isinstance(value, dict):
            origin = str(record.relative_to(EVIDENCE)) + location
            for key in ['probe_sha256', 'driver_sha256', 'model_probe_sha256']:
                if key in value:
                    expected = value[key]
                    assert expected in probe_digests, (origin, key)
                    verify(CODE / probe_digests[expected], expected, origin + '/' + key)
            if 'source_sha256' in value:
                sources = value['source_sha256']
                if isinstance(sources, str):
                    scalar_sources = {
                        'dp-contract': 'vllm/v1/engine/core_client.py',
                        'expert-map': 'vllm/model_executor/layers/fused_moe/expert_map_manager.py',
                        'ngram-contract': 'vllm/v1/spec_decode/ngram_proposer.py',
                    }
                    sources = {scalar_sources[record.parent.name]: sources}
                for name, expected in sources.items():
                    path = ROOT / name if name.startswith('repos/') else ROOT / 'repos/vllm' / name
                    verify(path, expected, origin + '/source')
            for name, expected in value.get('input_sha256', {}).items():
                verify(ROOT / name, expected, origin + '/input')
            if 'events_sha256' in value:
                verify(record.parent / 'events.jsonl', value['events_sha256'], origin + '/events')
            if {'file', 'sha256'} <= value.keys():
                installed = Path(value['file'])
                verify(installed, value['sha256'], origin + '/installed')
                relative = str(installed).split('/site-packages/', 1)[-1]
                if relative.startswith(('vllm/', 'vllm_ascend/')):
                    repo = 'vllm-ascend' if relative.startswith('vllm_ascend/') else 'vllm'
                    verify(ROOT / 'repos' / repo / relative, value['sha256'], origin + '/fixed')
            if {'repo', 'source', 'sha256'} <= value.keys():
                verify(ROOT / 'repos' / value['repo'] / value['source'], value['sha256'], origin)
            for key, child in value.items():
                walk(child, record, location + '/' + key)

    records = sorted(EVIDENCE.rglob('*.json'))
    for record in records:
        walk(json.loads(record.read_text()), record)
    metadata = json.loads((EVIDENCE / 'model-assets/metadata.json').read_text())
    model = ROOT / 'log/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots' / metadata['model_revision']
    verify(model / 'config.json', metadata['config_sha256'], 'model-assets/config')
    with (model / 'model.safetensors').open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        assert hashlib.sha256(stream.read(length)).hexdigest() == metadata['header_sha256']
    expected_probe = 'c7cba56b7077d517ff8acdecd7e18080cb62b91e3da972514c3e5794857e7c25'
    verify(CODE / 'observe_requests.py', expected_probe, 'protected-original-probe')
    result = {'records': len(records), 'hash_checks': len(checks), 'checks': checks,
              'boundary': '文件完整性与来源链检查；不代替图文语义审校或硬件复测'}
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
