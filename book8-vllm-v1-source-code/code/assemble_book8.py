"""安全组装：--stage log/新目录；核验后 --sync-from 临时目录。

按清单复制，不删除目标目录；存在未经管理或已修改的同名文件时拒绝覆盖。
旧正式上下册的归档独立执行，不是本脚本隐含副作用。
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt
import build_reader
from check_sample import Reader

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
DRAFT = ROOT / 'drafts/Book8-vLLM源码剖析'
RESEARCH = ROOT / 'research/Book8-源码重写'
DEFAULT = ROOT / 'from-models-to-vllm/book8-vllm-v1-source-code'
MANIFEST = '_assembly-manifest.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def destination_path(source):
    if source.is_relative_to(DRAFT):
        return source.relative_to(DRAFT)
    if source.is_relative_to(CODE):
        return Path('code') / source.relative_to(CODE)
    if source.is_relative_to(ROOT / 'research') or source.is_relative_to(ROOT / 'plan'):
        return Path('provenance') / source.relative_to(ROOT)
    raise ValueError(f'未授权的组装来源: {source}')


def validate(root):
    root = root.resolve()
    assert len(list(root.glob('ch*.md'))) == 28
    assert len(list(root.glob('appendix-*.md'))) == 4
    md = MarkdownIt('commonmark').enable('table')
    links = 0
    for path in root.rglob('*.md'):
        for token in md.parse(path.read_text()):
            for child in token.children or []:
                if child.type not in ['image', 'link_open']:
                    continue
                href = child.attrGet('src' if child.type == 'image' else 'href')
                url = urlsplit(unquote(href))
                if url.scheme or not url.path:
                    continue
                target = (path.parent / url.path).resolve()
                assert target.is_relative_to(root) and target.exists(), (path, href)
                if re.fullmatch(r'L\d+', url.fragment):
                    assert int(url.fragment[1:]) <= len(target.read_text().splitlines()), (path, href)
                links += 1
    for path in (root / 'code').glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    for name, count in [('全书连续阅读.html', 33), ('三章连续阅读.html', 4)]:
        reader = Reader()
        reader.feed((root / name).read_text())
        assert len(reader.sources) == count
        for name, digest in reader.sources.items():
            assert sha(root / name) == digest, name
        for href in reader.links:
            url = urlsplit(unquote(href))
            if url.scheme:
                continue
            if url.path:
                target = (root / url.path).resolve()
                assert target.is_relative_to(root) and target.exists(), href
            elif url.fragment:
                assert url.fragment in reader.ids, href
    return {'chapters': 28, 'appendices': 4, 'local_links': links, 'reader_documents': 33}


def stage(target):
    target = target.resolve()
    assert target.is_relative_to(ROOT / 'log'), '临时输出必须位于工作区log'
    if target.exists():
        raise FileExistsError('使用新的临时目录；不删除既有输出')
    report = json.loads((RESEARCH / 'full/validation.json').read_text())
    assert len(report['chapters']) == 28 and len(report['appendices']) == 4
    assert report['images'] == 87
    target.mkdir(parents=True)
    queue = sorted(DRAFT.glob('ch*.md')) + sorted(DRAFT.glob('appendix-*.md'))
    queue += [DRAFT / 'README.md', DRAFT / '实践与证据.md', CODE / 'README.md']
    queue += sorted(CODE.glob('*.py'))
    queue += [p for folder in ['figures', 'evidence'] for p in sorted((DRAFT / folder).rglob('*')) if p.is_file()]
    queue += [RESEARCH / 'full' / name for name in ['validation.json', 'evidence-index.json', 'evidence-chain-validation.json', '全册审校.md']]
    sources = {}
    seen = set()
    while queue:
        source = queue.pop(0).resolve()
        if source in seen:
            continue
        seen.add(source)
        assert source.is_file() and not source.is_symlink(), source
        rel = destination_path(source)
        output = target / rel
        output.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == '.md':
            text = source.read_text()
            if source == DRAFT / 'README.md':
                text = text.split('## 编写过程')[0]
                text += '核验过程见[全册审校](provenance/research/Book8-源码重写/full/全册审校.md)。\n'
            links = set()
            for token in MarkdownIt('commonmark').enable('table').parse(text):
                for child in token.children or []:
                    if child.type in ['image', 'link_open']:
                        links.add(unquote(child.attrGet('src' if child.type == 'image' else 'href')))
            def rewrite(match):
                href = match[1]
                # 代码围栏中的下标/调用也可能长得像 ](...)，只处理Markdown解析出的链接。
                if unquote(href) not in links:
                    return match[0]
                url = urlsplit(unquote(href))
                if url.scheme or not url.path:
                    return match[0]
                # README最后一条是组装输出自身的链接。
                if href.startswith('provenance/'):
                    return match[0]
                linked = (source.parent / url.path).resolve()
                assert linked.exists(), (source, href)
                if linked.suffix == '.html' and linked.parent == DRAFT:
                    mapped = linked.name
                else:
                    mapped = str(destination_path(linked))
                    if linked.is_file():
                        queue.append(linked)
                    else:
                        queue.extend(p for p in linked.rglob('*') if p.is_file())
                relative = os.path.relpath(target / mapped, output.parent)
                suffix = ('?' + url.query if url.query else '') + ('#' + url.fragment if url.fragment else '')
                return '](' + relative + suffix + ')'
            text = re.sub(r'\]\(([^)]+)\)', rewrite, text)
            output.write_text(text)
        else:
            shutil.copy2(source, output)
        sources[str(source.relative_to(ROOT))] = sha(source)
    original = build_reader.DRAFT
    try:
        build_reader.DRAFT = target
        build_reader.main('book')
        build_reader.main('sample')
    finally:
        build_reader.DRAFT = original
    result = validate(target)
    files = {str(p.relative_to(target)): sha(p) for p in sorted(target.rglob('*')) if p.is_file()}
    manifest = {'format': 1, 'book': 'Book8 vLLM V1 source code', 'files': files, 'sources': sources, 'validation': result}
    (target / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'stage': str(target), 'files': len(files), **result}, ensure_ascii=False))


def sync(source, destination):
    source, destination = source.resolve(), destination.resolve()
    assert source.is_relative_to(ROOT / 'log')
    assert destination == DEFAULT or destination.is_relative_to(ROOT / 'log')
    manifest = json.loads((source / MANIFEST).read_text())
    previous = json.loads((destination / MANIFEST).read_text())['files'] if (destination / MANIFEST).exists() else {}
    # 全量预检先于首次写入；任何冲突均保留用户文件。
    for rel, expected in manifest['files'].items():
        assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
        item, target = source / rel, destination / rel
        assert not item.is_symlink() and sha(item) == expected, rel
        assert not target.is_symlink() and target.resolve().is_relative_to(destination), rel
        if target.exists() and sha(target) != expected:
            if rel not in previous or sha(target) != previous[rel]:
                raise RuntimeError(f'保留用户内容，拒绝覆盖: {target}')
    for rel, expected in manifest['sources'].items():
        assert sha(ROOT / rel) == expected, ('源文件已更新，需要重新组装', rel)
    validate(source)
    for rel, expected in manifest['files'].items():
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or sha(target) != expected:
            shutil.copy2(source / rel, target)
    manifest['retained_previous_files'] = sorted(set(previous) - set(manifest['files']))
    (destination / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'destination': str(destination), 'files': len(manifest['files']), 'deleted': 0, **validate(destination)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--stage', type=Path)
    group.add_argument('--sync-from', type=Path)
    parser.add_argument('--destination', type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage:
        stage(args.stage)
    else:
        sync(args.sync_from, args.destination)


if __name__ == '__main__':
    main()
