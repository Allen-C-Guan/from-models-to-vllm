"""工作区与独立书仓都可运行；模型实验不依赖编辑工具。"""
from pathlib import Path
CODE=Path(__file__).resolve().parent
if (CODE.parent/'ch01-先定义一台翻译机.md').is_file():
    DRAFT=CODE.parent
    RESEARCH=DRAFT/'log'/'examples'
else:
    WORKSPACE=CODE.parents[2]
    DRAFT=WORKSPACE/'drafts/Book1-5-系统重写/book1'
    RESEARCH=WORKSPACE/'research/Book1-5-系统重写/book1'
RESEARCH.mkdir(exist_ok=True)
