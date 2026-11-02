"""生成十二章和四附录的独立 HTML 阅读文档，图片内嵌。"""
import base64
import hashlib
from html import escape
from pathlib import Path
from urllib.parse import unquote

from markdown_it import MarkdownIt
from build_reader import STYLE

ROOT=Path(__file__).resolve().parents[2]
DRAFT=ROOT/'drafts/Book7-vLLM架构剖析'

def main():
    docs=sorted(DRAFT.glob('ch*.md'))+sorted(DRAFT.glob('appendix-*.md'))
    assert len(docs)==16
    md=MarkdownIt('commonmark',{'html':False}).enable('table')
    anchors={p.name:f'doc{i}' for i,p in enumerate(docs,1)}
    sections=[];contents=[]
    for p in docs:
        tokens=md.parse(p.read_text().replace('<!-- NAVIGATION -->',''));key=anchors[p.name];headings=[];n=0
        for pos,token in enumerate(tokens):
            if token.type=='heading_open':
                n+=1;sid=f'{key}-s{n}';token.attrSet('id',sid)
                if token.tag=='h1':title=tokens[pos+1].content
                elif token.tag=='h2':headings.append(f'<a href="#{sid}">{escape(tokens[pos+1].content)}</a>')
            for child in token.children or []:
                if child.type=='image':
                    data=(p.parent/child.attrGet('src')).read_bytes()
                    child.attrSet('src','data:image/png;base64,'+base64.b64encode(data).decode())
                elif child.type=='link_open':
                    href=unquote(child.attrGet('href'))
                    if href in anchors:child.attrSet('href','#'+anchors[href])
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        sections.append(f'<section id="{key}" data-file="{escape(p.name)}" data-source-sha256="{digest}">'+md.renderer.render(tokens,md.options,{})+'<p><a href="#contents">返回目录</a></p></section>')
        contents.append(f'<details><summary><a href="#{key}">{escape(title)}</a></summary><div class="toc">'+''.join(headings)+'</div></details>')
    extras='details{padding:9px 0;border-bottom:1px solid #e9edf2}summary{font:17px/1.7 sans-serif;cursor:pointer}.toc{padding-left:18px}pre{white-space:pre-wrap}section h1{color:#183c60}'
    html='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>vLLM V1 架构剖析：全景、协作与设计原则</title><style>'+STYLE+extras+'</style></head><body><main><header><h1>vLLM V1 架构剖析</h1><p>全景、协作与设计原则</p><p>面向具备模型架构基础、读过 Book6 的读者。正文先建立全景与机制，源码导航位于章末。</p><p>12 章 · 4 附录 · 36 张中文软件架构图。阅读锚 v0.30.0；本次重写未做硬件性能实验。</p></header><nav id="contents">'+''.join(contents)+'</nav>'+''.join(sections)+'<footer><p>图片已内嵌；附录的历史实践源码位于随书 practice/。来源与版本限制见各章图注和附录 C。</p></footer></main></body></html>'
    target=DRAFT/'全书连续阅读.html';target.write_text(html)
    print(f'{len(docs)} documents, {len(html.encode()):,} bytes: {target}')

if __name__=='__main__':main()
