"""组装样稿或全书离线阅读版；图片内嵌，公式以本地 MathText 渲染。"""
import base64
import hashlib
import io
from html import escape
from pathlib import Path
import re
from urllib.parse import unquote
from markdown_it import MarkdownIt

ROOT=Path(__file__).resolve().parents[2]
DRAFT=ROOT/'drafts/Book8-vLLM源码剖析'
STYLE='''
:root{color-scheme:light;--ink:#263548;--muted:#637286;--line:#dbe3ec;--accent:#235e8e}
*{box-sizing:border-box}body{margin:0;background:#f1f4f7;color:var(--ink);font:18px/1.95 system-ui,"Noto Sans CJK SC",sans-serif}
main{max-width:1080px;margin:auto;padding:48px 58px;background:#fff}header{padding:20px 0 38px;border-bottom:3px solid var(--accent)}
h1{font-size:31px;line-height:1.5;letter-spacing:.01em}h2{font-size:24px;line-height:1.6;margin:2.2em 0 .8em;color:#234c70}
h3{font-size:20px}p{margin:1.15em 0}a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:3px}
nav{padding:26px 0}details{padding:10px 0;border-bottom:1px solid var(--line)}summary{cursor:pointer;font-size:19px}.toc{font-size:15px;padding:8px 0 4px 24px}.toc a{display:block}
section{padding-top:35px;margin-top:20px;border-top:1px solid var(--line)}section>h1{color:#193f61}
img{display:block;width:100%;height:auto;margin:28px auto 12px}table{width:100%;border-collapse:collapse;font-size:15px;line-height:1.8;margin:24px 0}th,td{padding:10px 12px;border:1px solid var(--line);vertical-align:top}th{background:#edf4fa;text-align:left}
code{font: .89em/1.7 ui-monospace,SFMono-Regular,Consolas,"Liberation Mono",monospace;background:#f0f3f7;border-radius:3px;padding:2px 4px}pre{overflow:auto;background:#f5f7fa;border:1px solid var(--line);border-left:4px solid #7e9db8;border-radius:5px;padding:18px 20px;font-size:14px;line-height:1.7;tab-size:4}pre code{padding:0;background:none;white-space:pre;font-size:inherit}
blockquote{margin:24px 0;padding:1px 22px;border-left:4px solid #b6cbdc;background:#f5f9fc}.note{font-size:15px;color:var(--muted)}footer{padding-top:30px;font-size:14px;color:var(--muted)}
@media(max-width:720px){main{padding:22px 20px}body{font-size:17px}h1{font-size:26px}h2{font-size:22px}table{display:block;overflow:auto}pre{font-size:12px}}
@media print{body{background:white}main{max-width:none;padding:0}nav{display:none}section{break-before:page}img,pre{break-inside:avoid}a{color:inherit}}
'''


def document_paths(scope="book"):
    chapters=sorted(p for p in DRAFT.glob('ch*.md') if scope=='book' or p.name[2:4] in ['03','04','05'])
    return chapters+(sorted(DRAFT.glob('appendix-*.md')) if scope=='book' else [])+[DRAFT/'实践与证据.md']


def main(scope="sample"):
    docs=document_paths(scope)
    anchors={p.name:f'doc{i}' for i,p in enumerate(docs,1)}
    md=MarkdownIt('commonmark',{'html':False}).enable('table')
    def math_rule(state,silent):
        start=state.pos
        if state.src[start:start+1]!='$':return False
        delimiter='$$' if state.src[start:start+2]=='$$' else '$'
        end=state.src.find(delimiter,start+len(delimiter))
        if end<0:return False
        if not silent:
            token=state.push('math_local','',0)
            token.content=state.src[start+len(delimiter):end]
            token.meta={'display':len(delimiter)==2}
        state.pos=end+len(delimiter)
        return True
    def render_math(tokens,idx,options,env):
        from matplotlib.mathtext import math_to_image
        from matplotlib.font_manager import FontProperties
        token=tokens[idx];buf=io.BytesIO()
        depth=math_to_image('$'+token.content+'$',buf,prop=FontProperties(size=18),format='svg')
        svg=buf.getvalue().decode();svg=svg[svg.index('<svg'):]
        width=float(re.search(r'width="([\d.]+)pt"',svg)[1])/18
        height=float(re.search(r'height="([\d.]+)pt"',svg)[1])/18
        prefix='math'+hashlib.sha256(token.content.encode()).hexdigest()[:12]+'-'
        svg=re.sub(r'\bid="([^"]+)"',lambda m:'id="'+prefix+m[1]+'"',svg)
        svg=re.sub(r'(?:xlink:)?href="#([^"]+)"',lambda m:'href="#'+prefix+m[1]+'"',svg)
        svg=svg.replace('<svg ',f'<svg role="img" aria-label="{escape(token.content,quote=True)}" style="width:{width:.3f}em;height:{height:.3f}em;vertical-align:-{depth/18:.3f}em" ',1)
        return ('<span style="display:block;text-align:center;margin:1em 0">'+svg+'</span>') if token.meta['display'] else svg
    md.inline.ruler.before('escape','math_local',math_rule)
    md.renderer.rules['math_local']=render_math
    def teaching_fence(tokens,idx,options,env):
        token=tokens[idx]
        if token.info.strip()!='python':return md.renderer.fence(tokens,idx,options,env)
        lines=[]
        for line in token.content.splitlines(keepends=True):
            content=escape(line)
            if line.lstrip().startswith('# 【解读】'):
                content='<span style="color:#17674e;font-weight:600">'+content.rstrip('\n')+'</span>\n'
            lines.append(content)
        return '<pre><code class="language-python">'+''.join(lines)+'</code></pre>\n'
    md.renderer.rules['fence']=teaching_fence
    contents=[];sections=[]
    for p in docs:
        source=re.sub(r'<!-- (?:evidence|book):[^\n]+ -->','',p.read_text())
        tokens=md.parse(source);heading=0;toc=[];key=anchors[p.name]
        for i,token in enumerate(tokens):
            if token.type=='heading_open':
                heading+=1;sid=f'{key}-s{heading}';token.attrSet('id',sid)
                title=tokens[i+1].content
                if token.tag=='h1':doc_title=title
                elif token.tag=='h2':toc.append(f'<a href="#{sid}">{escape(title)}</a>')
            for child in token.children or []:
                if child.type=='image':
                    data=(p.parent/unquote(child.attrGet('src'))).read_bytes()
                    child.attrSet('src','data:image/png;base64,'+base64.b64encode(data).decode())
                elif child.type=='link_open':
                    href=unquote(child.attrGet('href'))
                    if href in anchors:child.attrSet('href','#'+anchors[href])
        sections.append(f'<section id="{key}" data-file="{escape(p.name)}" data-source-sha256="{hashlib.sha256(p.read_bytes()).hexdigest()}">'+
            md.renderer.render(tokens,md.options,{})+'<p class="note"><a href="#contents">返回目录</a></p></section>')
        contents.append(f'<details><summary><a href="#{key}">{escape(doc_title)}</a></summary><div class="toc">'+''.join(toc)+'</div></details>')
    html='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>vLLM V1 源码剖析 · 第 3–5 章连续样稿</title><style>'+STYLE+'</style></head><body><main>'
    html+='<header><p class="note">从模型到 vLLM · Book8 · 连续样稿 · 第一次修订</p><h1>vLLM V1 源码剖析</h1><p>从对象协作到运行现场</p><p>接过 Book7 的架构地图，沿两条真实请求，读懂身份、迭代与调度。</p><p class="note">第 3–5 章 · 上游 v0.30.0 · 全局地图／UML／调用路径 · 14 张图</p></header>'
    html+='<nav id="contents">'+''.join(contents)+'</nav>'+''.join(sections)
    html+='<footer>本文图片已经内嵌，可离线阅读。原始记录保存在相邻 evidence/ 目录；复现代码与源码在线链接见实践说明。样稿完成后等待作者评阅，尚未展开全书。</footer></main></body></html>'
    if scope=='book':
        html=html.replace('接过 Book7 的架构地图，沿两条真实请求，读懂身份、迭代与调度。','接过 Book7 的架构地图，从请求、状态与执行，读到并行、扩展和诊断；每个边界都有可检查的源码与产物。').replace('第 3–5 章连续样稿','全书阅读版').replace('连续样稿 · 第一次修订','全书阅读版').replace('第 3–5 章 · 上游 v0.30.0 · 全局地图／UML／调用路径 · 14 张图','28 章 · 4 篇附录 · 上游 v0.30.0 · 87 张图').replace('样稿完成后等待作者评阅，尚未展开全书。','全书按已通过的样稿标准完成；证据层次、复现入口与未验证范围见附录。')
    else:
        html=html.replace('样稿完成后等待作者评阅，尚未展开全书。','本样稿已获作者认可；作为表达标准独立保留，完整内容见全书阅读版。')
    target=DRAFT/('全书连续阅读.html' if scope=='book' else '三章连续阅读.html');target.write_text(html)
    print(f'{len(docs)} documents; {len(html.encode())} bytes; {target}')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=['sample','book'],default='sample')
    main(parser.parse_args().scope)
