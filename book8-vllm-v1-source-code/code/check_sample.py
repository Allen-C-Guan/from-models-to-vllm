"""样稿证据、数据连续性、阅读产物和既有工作保护检查；不启动推理。"""
import base64
from collections import defaultdict
import hashlib
from html.parser import HTMLParser
import importlib.metadata as metadata
import json
import io
import tokenize
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
from markdown_it import MarkdownIt
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
DRAFT=ROOT/'drafts/Book8-vLLM源码剖析'
RESEARCH=ROOT/'research/Book8-源码重写/2026-11-02'
LOG=ROOT/'log/book8-source-rewrite'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def events(run):return [json.loads(s) for s in (DRAFT/'evidence'/run/'events.jsonl').read_text().splitlines()]


class Reader(HTMLParser):
    def __init__(self):
        super().__init__();self.ids=set();self.links=[];self.images=[];self.sources={}
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:
            assert a['id'] not in self.ids
            self.ids.add(a['id'])
        if tag=='a':self.links.append(a.get('href',''))
        if tag=='img':self.images.append(a['src'])
        if tag=='section':self.sources[a['data-file']]=a['data-source-sha256']


def main():
    result={};docs=sorted(p for p in DRAFT.glob('ch*.md') if p.name[2:4] in ['03','04','05']);assert len(docs)==3
    index=read(RESEARCH/'source-index.json')
    for e in index:
        assert sha(ROOT/'repos'/e['project']/e['path'])==e['sha256'],e['path']
        if 'installed_sha256' in e:
            installed=metadata.distribution(e['project']).locate_file(e['path'])
            assert sha(installed)==e['sha256'] and e['installed_matches'],e['path']
    result['source_files_verified']=len(index)

    blocks={}
    for doc in docs:
        s=doc.read_text();assert '{{' not in s
        for key,body in re.findall(r'<!-- evidence:(\S+):start -->(.*?)<!-- evidence:\1:end -->',s,re.S):
            assert key not in blocks;blocks[key]=body
    evidence=read(RESEARCH/'evidence-index.json');assert blocks.keys()==evidence.keys()
    for key,e in evidence.items():
        if key.startswith('source:'):
            path=ROOT/'repos'/e['repo']/e['path']
            source='\n'.join(path.read_text().splitlines()[e['start']-1:e['end']])
            teaching=re.search(r'```python\n(.*?)\n```',blocks[key],re.S)[1]
            original='\n'.join(s for s in teaching.splitlines() if not s.lstrip().startswith('# 【解读】'))
            assert original==source,key
            notes=[s.strip().removeprefix('# 【解读】') for s in teaching.splitlines() if s.lstrip().startswith('# 【解读】')]
            assert notes==list(e['teaching_notes'].values()) and notes,key
            comments=[t.string.removeprefix('# 【解读】') for t in tokenize.generate_tokens(io.StringIO(teaching).readline)
                      if t.type==tokenize.COMMENT and t.string.startswith('# 【解读】')]
            assert comments==notes,('teaching note inside string',key)
            assert sha(path)==e['file_sha256']
            assert hashlib.sha256(source.encode()).hexdigest()==e['snippet_sha256']
        elif key.startswith('data:'):
            assert json.loads(re.search(r'```json\n(.*?)\n```',blocks[key],re.S)[1])==e['projection'],key
            rows={r['seq']:r for r in events(e['run'])}
            assert all(n in rows for n in e['seqs']),key
    result['evidence_blocks_verified']=len(blocks)

    runs=['baseline-inproc-traced','chunked-inproc-traced','baseline-mp-traced','baseline-inproc-plain']
    results={}
    for run in runs:
        for name in ['events.jsonl','manifest.json','result.json']:
            assert (DRAFT/'evidence'/run/name).read_bytes()==(LOG/run/name).read_bytes(),(run,name)
        m=read(DRAFT/'evidence'/run/'manifest.json')
        assert m['probe_sha256']==sha(ROOT/'code/Book8-vLLM源码剖析/observe_requests.py'),run
        rows=events(run);assert [r['seq'] for r in rows]==list(range(1,len(rows)+1))
        assert len({r['pid'] for r in rows})==1
        assert all(a['monotonic_ns']<=b['monotonic_ns'] for a,b in zip(rows,rows[1:]))
        results[run]=read(DRAFT/'evidence'/run/'result.json')
        assert rows[-1]['final']==results[run]['final']
        if run not in runs[:2]:continue
        assert m['effective']['balance_enabled'] is False
        assert m['effective']['use_v2_model_runner'] is True
        aliases={r['internal_id']:r['external_id'] for r in rows if r['event']=='user.accepted'}
        generated=defaultdict(list)
        plans=[r for r in rows if r['event']=='schedule.after']
        def select(event,step):return next(r for r in rows if r['event']==event and r['front_step']==step)
        for p in plans:
            plan=p['plan'];step=p['front_step'];counts=plan['num_scheduled_tokens']
            assert 0<sum(counts.values())==plan['total_num_scheduled_tokens']<=m['effective']['max_num_batched_tokens']
            before=select('schedule.before',step)['scheduler']['requests']
            after=p['scheduler']['requests']
            updated=select('update.after',step)['scheduler']['requests']
            for rid,n in counts.items():
                assert n>0
                assert after[rid]['num_computed_tokens']-before[rid]['num_computed_tokens']==n
                assert after[rid]['num_in_flight_tokens']==n
            output=select('update.before',step)['model_output']
            for rid,tokens in zip(output['req_ids'],output['sampled_token_ids']):generated[rid].extend(tokens)
            for rid,state in updated.items():
                assert state['num_in_flight_tokens']==0
                assert state['output_token_ids']==generated[rid]
        for rid,tokens in generated.items():
            assert tokens==results[run]['final'][aliases[rid]]['outputs'][0]['token_ids']
        assert rows[-1]['output_states']=={} and rows[-1]['scheduler']['requests']=={}
        result[run]={'rounds':len(plans),'events':len(rows),'budget':m['effective']['max_num_batched_tokens']}
    for run in runs[2:]:assert results[run]['final']==results[runs[0]]['final']
    for rid in ['A','B']:assert results[runs[1]]['final'][rid]==results[runs[0]]['final'][rid]
    result['four_run_output_comparison']='passed'
    for p in (DRAFT/'evidence').glob('*.log'):assert p.read_bytes()==(LOG/p.name).read_bytes()

    md=MarkdownIt('commonmark').enable('table');links=0;github_links=0
    for doc in list(DRAFT.glob('*.md'))+list(RESEARCH.rglob('*.md'))+[ROOT/'plan/Book8-源码重写.md']:
        for token in md.parse(doc.read_text()):
            for child in token.children or []:
                if child.type not in ['link_open','image']:continue
                href=child.attrGet('href') if child.type=='link_open' else child.attrGet('src')
                u=urlsplit(unquote(href))
                if not u.scheme and u.path:
                    target=(doc.parent/u.path).resolve()
                    # 本次成功检查才生成此报告；允许首次运行时的自引用。
                    assert target.exists() or target==RESEARCH/'validation.json',(doc,href)
                    links+=1
                elif u.netloc=='github.com' and '/blob/' in u.path:
                    match=re.fullmatch(r'/vllm-project/(vllm(?:-ascend)?)/blob/([^/]+)/(.*)',u.path)
                    if match:
                        repo,commit,path=match.groups();p=ROOT/'repos'/repo/path
                        assert p.exists(),href
                        head=subprocess.check_output(['git','-C',str(ROOT/'repos'/repo),'rev-parse','HEAD'],text=True).strip()
                        assert head==commit,href
                        if u.fragment:
                            nums=[int(n) for n in re.findall(r'L(\d+)',u.fragment)]
                            assert nums and all(1<=n<=len(p.read_text().splitlines()) for n in nums),href
                        github_links+=1
    result['local_links']=links;result['fixed_commit_links']=github_links
    pngs=sorted({DRAFT/path for doc in docs for path in re.findall(r'!\[[^\]]*\]\((figures/[^)]+\.png)\)',doc.read_text())});svgs=[p.with_suffix('.svg') for p in pngs]
    assert len(pngs)==len(svgs)==14
    for p in pngs:
        with Image.open(p) as im:im.verify()
    for p in svgs:ET.parse(p)
    reader=Reader();html=(DRAFT/'三章连续阅读.html').read_text();reader.feed(html)
    assert len(reader.images)==14 and len(reader.sources)==4
    assert all(h[1:] in reader.ids for h in reader.links if h.startswith('#'))
    assert {hashlib.sha256(base64.b64decode(s.split(',',1)[1])).hexdigest() for s in reader.images}=={sha(p) for p in pngs}
    for name,digest in reader.sources.items():assert sha(DRAFT/name)==digest,name
    assert '<!-- evidence:' not in html and '{{' not in html
    result['reader']='4 documents, 14 embedded images, source hashes and anchors verified'

    baseline=read(RESEARCH/'preexisting-files.json')
    for name,digest in baseline.items():
        b=(ROOT/name).read_bytes()
        if name=='CLAUDE.md':
            b=re.sub(rb'> \*\*Book8 [^\n]+\n\n',b'',b,count=1)
        assert hashlib.sha256(b).hexdigest()==digest,('preexisting changed',name)
    result['preexisting_files_preserved']=len(baseline)
    for label,repo in [('workspace',ROOT),('book_repository',ROOT/'from-models-to-vllm')]:
        check=subprocess.run(['git','-C',str(repo),'diff','--check'],text=True,capture_output=True)
        if check.returncode:
            # 只容许已按开工哈希确认未改动的旧文件中的空白问题。
            files=set(re.findall(r'^(.+?):\d+: trailing whitespace\.$',check.stdout,re.M))
            assert check.returncode==2 and files and not check.stderr,check.stderr
            headers=[s for s in check.stdout.splitlines() if not s.startswith('+')]
            assert len(headers)==len(re.findall(r'^.+?:\d+: trailing whitespace\.$',check.stdout,re.M)),headers
            for name in files:
                full=(repo/name).relative_to(ROOT).as_posix()
                assert full in baseline and full!='CLAUDE.md' and sha(repo/name)==baseline[full],full
            result[label+'_diff_check']={'preexisting_whitespace_only':sorted(files)}
        else:result[label+'_diff_check']='passed'
    subprocess.run(['git','-C',str(ROOT),'diff','--check','--','CLAUDE.md'],check=True)
    for folder in [DRAFT,ROOT/'code/Book8-vLLM源码剖析',RESEARCH]:
        for p in folder.rglob('*'):
            if p.suffix in ['.md','.py','.svg']:
                assert all(s==s.rstrip() for s in p.read_text().splitlines()),p
    result['new_text_whitespace']='passed'
    for name in ['vllm','vllm-ascend']:
        assert not subprocess.check_output(['git','-C',str(ROOT/'repos'/name),'status','--porcelain']).strip(),name
    result['upstream_repositories']='clean'
    (RESEARCH/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
