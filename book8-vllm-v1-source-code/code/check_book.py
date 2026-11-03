"""续写章节的证据、源码注释、链接和独立阅读版检查；不启动推理。"""
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import tokenize
from urllib.parse import unquote,urlsplit
import xml.etree.ElementTree as ET
from markdown_it import MarkdownIt
from PIL import Image
from check_sample import Reader,sha
from build_chapters import projection
from build_reader import document_paths

ROOT=Path(__file__).resolve().parents[2]
DRAFT=ROOT/'drafts/Book8-vLLM源码剖析'
OUT=ROOT/'research/Book8-源码重写/full'


def main():
    docs=document_paths();blocks={};images=set();links=0
    md=MarkdownIt('commonmark').enable('table')
    for doc in docs:
        s=doc.read_text();assert '{{book:' not in s,doc
        if doc.name.startswith('ch'):
            ch=int(doc.name[2:4]);mapname=f'figures/fig-{ch}-0-map.png'
            assert mapname in s.split('\n## ',1)[0],doc
            assert '```text' in s,('missing call path',doc)
        for key,body in re.findall(r'<!-- book:(\S+):start -->(.*?)<!-- book:\1:end -->',s,re.S):
            assert key not in blocks;blocks[key]=body
        for token in md.parse(s):
            for child in token.children or []:
                if child.type not in ['image','link_open']:continue
                url=urlsplit(unquote(child.attrGet('src' if child.type=='image' else 'href')))
                if not url.scheme and url.path:
                    p=(doc.parent/url.path).resolve();assert p.exists(),(doc,p);links+=1
                    if child.type=='image':images.add(p)
                if url.netloc=='github.com' and '/blob/' in url.path:
                    match=re.fullmatch(r'/vllm-project/(vllm(?:-ascend)?)/blob/([^/]+)/(.*)',url.path)
                    if match:
                        repo,commit,path=match.groups();p=ROOT/'repos'/repo/path
                        assert p.exists(),path
                        assert all(int(n)<=len(p.read_text().splitlines()) for n in re.findall(r'L(\d+)',url.fragment)),path
    index=json.loads((OUT/'evidence-index.json').read_text());assert blocks.keys()==index.keys()
    for key,e in index.items():
        if e['kind']=='source':
            p=ROOT/'repos'/e['repo']/e['path'];assert sha(p)==e['file_sha256']
            original='\n'.join(p.read_text().splitlines()[e['start']-1:e['end']])
            teaching=re.search(r'```python\n(.*?)\n```',blocks[key],re.S)[1]
            assert '\n'.join(l for l in teaching.splitlines() if not l.lstrip().startswith('# 【解读】'))==original,key
            # 片段可能从嵌套块中部起止；放回完整源码上下文识别注释，避免孤立dedent误报。
            full=p.read_text().splitlines()
            context='\n'.join(full[:e['start']-1]+teaching.splitlines()+full[e['end']:])+'\n'
            notes=[t.string.removeprefix('# 【解读】') for t in tokenize.generate_tokens(io.StringIO(context).readline) if t.type==tokenize.COMMENT and t.string.startswith('# 【解读】')]
            assert notes==list(e['notes'].values()),key
        else:
            p=ROOT/e['file'];assert sha(p)==e['file_sha256']
            value=projection(json.loads(p.read_text()),e['paths'])
            assert value==e['value']==json.loads(re.search(r'```json\n(.*?)\n```',blocks[key],re.S)[1]),key
    for p in images:
        with Image.open(p) as im:im.verify()
        ET.parse(p.with_suffix('.svg'))
    html=(DRAFT/'全书连续阅读.html').read_text();reader=Reader();reader.feed(html)
    assert len(reader.sources)==len(docs)
    for name,digest in reader.sources.items():assert sha(DRAFT/name)==digest,name
    assert all(a[1:] in reader.ids for a in reader.links if a.startswith('#'))
    assert {hashlib.sha256(base64.b64decode(s.split(',',1)[1])).hexdigest() for s in reader.images}=={sha(p) for p in images}
    assert '{{book:' not in html and '&lt;!-- book:' not in html
    result={'chapters':[p.name for p in docs if p.name.startswith('ch')],'appendices':[p.name for p in docs if p.name.startswith('appendix-')],'new_evidence_blocks':len(blocks),'images':len(images),'local_links':links,'reader':'source hashes, embedded images and anchors passed','scope':'结构与证据自动检查；教学、关系语义及视觉检查另记逐章审校'}
    (OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
