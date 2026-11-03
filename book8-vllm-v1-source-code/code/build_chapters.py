"""把续写注册的固定源码与真实观测投影填入章节，保存可还原证据。"""
from pathlib import Path
import hashlib
import json
import os
import re
from chapter_evidence import SOURCES,DATA,DESCRIPTIONS

ROOT=Path(__file__).resolve().parents[2]
DRAFT=ROOT/'drafts/Book8-vLLM源码剖析'
OUT=ROOT/'research/Book8-源码重写/full'
COMMITS={'vllm':'ced6857afa0ea7b2e3f0846a62e1394e90f15607','vllm-ascend':'c09d3e20b9c45c80b34ae3781440d90904a44023'}


def projection(obj,paths):
    output={}
    for path in paths:
        keys=path.split('.');src=obj;dst=output
        for k in keys[:-1]:src=src[k];dst=dst.setdefault(k,{})
        dst[keys[-1]]=src[keys[-1]]
    return output


def main():
    blocks={};index={}
    for key,(repo,path,start,end,notes) in SOURCES.items():
        f=ROOT/'repos'/repo/path;lines=f.read_text().splitlines()[start-1:end];out=[]
        assert all(start<=n<=end for n in notes)
        for n,line in enumerate(lines,start):
            if n in notes:out.append(line[:len(line)-len(line.lstrip())]+'# 【解读】'+notes[n])
            out.append(line)
        original='\n'.join(lines);link=f'https://github.com/vllm-project/{repo}/blob/{COMMITS[repo]}/{path}#L{start}-L{end}'
        blocks[key]=f'源码教学注释版，{repo} v0.30.0：[原文件 L{start}–L{end}]({link})。仅新增 `# 【解读】`，其他内容保留原文。\n\n```python\n'+'\n'.join(out)+'\n```'
        index[key]={'kind':'source','repo':repo,'path':path,'start':start,'end':end,'notes':notes,
            'file_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'original_sha256':hashlib.sha256(original.encode()).hexdigest()}
    for key,(file,paths) in DATA.items():
        f=ROOT/file;value=projection(json.loads(f.read_text()),paths)
        link=os.path.relpath(f,DRAFT)
        label=DESCRIPTIONS.get(key,'真实运行字段投影，保留字段名与值，省略无关项。')
        blocks[key]=f'{label}来自[原始记录]({link})。\n\n```json\n'+json.dumps(value,ensure_ascii=False,indent=2)+'\n```'
        index[key]={'kind':'observation','file':file,'paths':paths,'file_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'value':value}
    used=[]
    for doc in sorted(DRAFT.glob('ch*.md')):
        s=doc.read_text()
        def fill(m):
            key=m[1];used.append(key)
            return f'<!-- book:{key}:start -->\n{blocks[key]}\n<!-- book:{key}:end -->'
        s=re.sub(r'<!-- book:(\S+):start -->.*?<!-- book:\1:end -->',fill,s,flags=re.S)
        s=re.sub(r'\{\{book:([^}]+)\}\}',fill,s)
        if s!=doc.read_text():doc.write_text(s)
    assert set(used)==set(blocks),(set(used)^set(blocks))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'evidence-index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
    print(f'{len(SOURCES)} source blocks; {len(DATA)} observed projections')


if __name__=='__main__':main()
