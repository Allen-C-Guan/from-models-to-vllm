# Book1《Transformer 原典：2017 原架构完全拆解》

> 《从模型到 vLLM》丛书第一册 · 定稿于 2026-10-02 · 论文锚定 arXiv:1706.03762 v7

## 目录

- 导览：我们要一起造一台 2017 年的翻译机
- 第 1 章 前史动机：seq2seq 瓶颈与 Bahdanau 注意力
- 第 2 章 缩放点积注意力
- 第 3 章 多头注意力
- 第 4 章 三种注意力角色与双塔协作
- 第 5 章 正弦位置编码
- 第 6 章 FFN、残差与 post-LN：一条 token 的数据流
- 第 7 章 整机组装与 mask 全景
- 第 8 章 原论文训练配方：点火前的最后准备
- 第 9 章 从零实现与小型翻译实验
- 第 10 章 效果实证：趋势级复现与引用对照
- 第 11 章 三大遗产与欠账：全系列地图

## 本册结构

- `chNN-*.md`——各章书稿（ch00 为导览，ch11 含全书术语表与八册欠账地图）
- `figures/`——全部插图（300dpi，自绘重制/自产，图注含来源与证据等级）
- `code/chNN/`——**随书代码快照**（每章交付的组件/实验脚本；正文中已升级为行级链接，可直接点击跳转到对应函数/类）

## 运行环境

代码在写作与实验期运行于：Apple Silicon（MPS）+ CPU 回退，Python 3.12 / PyTorch 2.13。
随书复现：把本目录放入配套工作区后 `source env.sh`（或自行安装 torch/datasets/sacrebleu/sentencepiece/matplotlib），
按各章末「动手验证」的命令运行；全部实验数字在普通笔记本 ≤10 分钟量级可复现。

## 代码索引

- `code/ch01/`：__init__.py、bottleneck_demo.py、make_figures.py
- `code/ch02/`：__init__.py、scaled_dot_product.py
- `code/ch03/`：__init__.py、figures.py、mha.py
- `code/ch04/`：__init__.py、figures.py、roles_demo.py
- `code/ch05/`：__init__.py、pe.py
- `code/ch06/`：__init__.py、blocks.py、make_fig_6_1.py
- `code/ch07/`：__init__.py、make_fig_7_2.py、masks.py、transformer.py
- `code/ch08/`：make_fig_8_1.py、recipe.py
- `code/ch09/`：__init__.py、data.py、decode.py、eval.py、gru_baseline.py、make_figures.py、model.py、train.py
- `code/ch10/`：__init__.py、ablation.py、make_figures.py
- `code/ch11/`：make_fig_11_1.py

## 说明

- 零魔改纪律：正文时间冻结在 2017；后世概念仅在章末欠账框与第 11 章地图中以「最小披露」出现。
- 证据等级：全书引用数字均带【证据等级】标注；本书实验数字全部为真实运行所得（M5 Pro，2026-10）。
