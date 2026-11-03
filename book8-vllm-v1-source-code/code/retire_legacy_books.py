"""将已被新书替代的两个正式旧册完整移入drafts归档；默认只列计划。"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from assemble_book8 import ROOT, DEFAULT, validate, sha

NAMES = ['book7-vllm-v1-source-code-vol1', 'book8-vllm-v1-source-code-vol2']
ARCHIVE = ROOT / 'drafts/Book8-旧正式源码册归档/2026-11-02'


def inventory(root):
    result = {}
    for path in sorted(root.rglob('*')):
        assert not path.is_symlink(), ('归档不跟随符号链接', path)
        if path.is_file():
            result[str(path.relative_to(root))] = sha(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    validate(DEFAULT)
    assert not ARCHIVE.exists(), '归档已经存在，不覆盖历史'
    book = ROOT / 'from-models-to-vllm'
    before = {name: inventory(book / name) for name in NAMES}
    assert all(before.values()), '两个旧目录都必须完整在位'
    status = subprocess.check_output(['git', '-C', str(book), 'status', '--porcelain', '--', *NAMES], text=True)
    report = {'old_paths': NAMES, 'archive': str(ARCHIVE.relative_to(ROOT)), 'files': before,
              'git_status_before': status, 'replacement': str(DEFAULT.relative_to(ROOT)),
              'method': '完整目录重命名到drafts；不删除文件内容，包含独有和未提交文件'}
    if not args.apply:
        print(json.dumps({'planned_files': {name: len(files) for name, files in before.items()},
                          'archive': str(ARCHIVE), 'git_status': status}, ensure_ascii=False))
        return
    ARCHIVE.mkdir(parents=True)
    (ARCHIVE / 'inventory.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    for name in NAMES:
        # 同工作区重命名，所有文件（包括未跟踪资产）一起保留。
        os.rename(book / name, ARCHIVE / name)
        assert inventory(ARCHIVE / name) == before[name], name
    report['verified_after_move'] = True
    (ARCHIVE / 'inventory.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'archived': str(ARCHIVE), 'files': sum(map(len, before.values())),
                      'old_formal_directories_removed': all(not (book / name).exists() for name in NAMES)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
