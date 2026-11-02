"""将两章样稿生成可携带的静态阅读文档；不组装、不覆盖正式书仓。

运行：source ./env.sh 后 python code/Book7-vLLM架构剖析/build_reader.py
依赖工作区已有 markdown-it-py。图片内嵌，阅读不依赖本机路径或网络脚本。
"""

import base64
import hashlib
from html import escape
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "drafts/Book7-vLLM架构剖析"
CHAPTERS = [DRAFT / "ch01-vLLM全景.md", DRAFT / "ch02-请求的进入与返回.md"]

STYLE = """
:root{color-scheme:light;font-family:'Noto Serif CJK SC','Songti SC',serif;color:#243246;background:#f3f5f7}
body{margin:0}main{max-width:900px;margin:32px auto;background:white;padding:48px 56px}
header{border-bottom:1px solid #dbe2ea;padding-bottom:24px;margin-bottom:40px}
header p{color:#53647a}nav{line-height:1.9}nav a{margin-right:20px}
h1,h2,h3{font-family:'Noto Sans CJK SC','PingFang SC',sans-serif;line-height:1.5;scroll-margin-top:20px}
h1{font-size:28px;margin:1.3em 0 .8em}h2{font-size:23px;margin:2.2em 0 .8em}h3{font-size:19px;margin:1.8em 0 .7em}
p,li{font-size:18px;line-height:1.95}p{margin:1.1em 0}a{color:#245e95;text-decoration-thickness:1px;text-underline-offset:3px}
img{display:block;max-width:100%;height:auto;margin:28px auto 18px;border:1px solid #e9edf2}
table{border-collapse:collapse;width:100%;margin:22px 0;font-size:15px;line-height:1.8}
th,td{padding:10px 12px;border:1px solid #dbe2ea;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#f1f5f9}
code{font-size:.87em;background:#f1f4f8;border-radius:3px;padding:.1em .3em;overflow-wrap:anywhere}
section+section{border-top:2px solid #dbe2ea;margin-top:70px;padding-top:35px}.toc{columns:2;font-size:15px;line-height:1.9;margin:18px 0}
.toc a{display:block;break-inside:avoid}footer{border-top:1px solid #dbe2ea;margin-top:40px;color:#53647a;font-size:14px}
@media(max-width:720px){main{padding:24px 18px;margin:0}p,li{font-size:17px}.toc{columns:1}table{font-size:13px}th,td{padding:7px}}
@media print{body{background:white}main{max-width:none;margin:0;padding:0}section+section{break-before:page}h2,h3{break-after:avoid}img{break-inside:avoid}p{font-size:11pt;line-height:1.8}table{font-size:9pt}nav,.toc{display:none}}
"""


def main():
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    sections, toc = [], []
    chapters_by_name = {p.name: f"ch{i}" for i, p in enumerate(CHAPTERS, 1)}
    for i, path in enumerate(CHAPTERS, 1):
        tokens = md.parse(path.read_text())
        heading_num = 0
        for pos, token in enumerate(tokens):
            if token.type == "heading_open":
                heading_num += 1
                anchor = f"ch{i}-s{heading_num}"
                token.attrSet("id", anchor)
                if token.tag in {"h1", "h2"}:
                    title = tokens[pos + 1].content
                    toc.append(f'<a href="#{anchor}">{escape(title)}</a>')
            for child in token.children or []:
                if child.type == "image":
                    image = path.parent / child.attrGet("src")
                    child.attrSet("src", "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode())
                if child.type == "link_open":
                    href = child.attrGet("href")
                    if href in chapters_by_name:
                        child.attrSet("href", "#" + chapters_by_name[href])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sections.append(f'<section id="ch{i}" data-source-sha256="{digest}">' + md.renderer.render(tokens, md.options, {}) + "</section>")
    html = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Book7 架构剖析 · 第1、2章连续样稿</title><style>' + STYLE + '</style><body><main>'
            '<header><h1>vLLM V1 架构剖析</h1><p>全景、协作与设计原则 · 第 1、2 章连续样稿</p>'
            '<p>阅读锚 v0.30.0。先理解职责与协作，再选择章末源码导航。</p>'
            '<nav><a href="#ch1">第 1 章　全景</a><a href="#ch2">第 2 章　进入与返回</a></nav>'
            '<div class="toc">' + ''.join(toc) + '</div></header>' + ''.join(sections) +
            '<footer><p>2026-11-02 样稿。示例数字为教学构造，插图按固定源码与设计资料绘制；本轮未做硬件性能实验。</p></footer>'
            '</main></body></html>')
    target = DRAFT / "两章连续阅读.html"
    target.write_text(html)
    print(f"generated: {target} ({len(html.encode()):,} bytes)")


if __name__ == "__main__":
    main()
