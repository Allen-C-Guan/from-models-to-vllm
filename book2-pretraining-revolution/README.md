# Book2《预训练革命：BERT、GPT 与 Scaling Laws》

> 《从模型到 vLLM》丛书第二册 · 定稿于 2026-10-03 · 前置：Book1《Transformer 原典》
> 论文锚定见各章证据等级标注（GPT-1/2 以 OpenAI PDF 为 canonical；BERT=arXiv v2；T5=v4 JMLR；Kaplan/Chinchilla 仅 v1；Wei v2/Schaeffer v2 为涌现之争白名单）

## 目录

- 导览：把互联网灌进这台机器
- 第 1 章 范式转移：监督数据不够了
- 第 2 章 分词器：BPE 从零讲透
- 第 3 章 BERT：双向理解与微调生态
- 第 4 章 GPT-1/2：单塔、因果与架构差异清单
- 第 5 章 T5：text-to-text 与系统性消融
- 第 6 章 RoBERTa：配方 ≥ 架构
- 第 7 章 GPT-3 与涌现之争
- 第 8 章 Scaling Laws：从幂律到等比
- 第 9 章 预训练工程账本：显存、算力与不稳定性
- 第 10 章 大项目 I：从零实现 GPT-2 124M
- 第 11 章 大项目 II：数据、权重与损失-算力曲线
- 第 12 章 收束：decoder-only 的胜出与本册欠账

## 本册结构

- `chNN-*.md`——各章书稿（ch00 为导览，ch12 为收束章含册末术语表与 F/N 伏笔总决算）
- `figures/`——全部插图（300dpi，自绘重制/自产，图注含来源与证据等级）
- `code/chNN/`——**随书代码快照**（正文中已升级为行级链接，可直接点击跳转到对应函数/类）；`code/00-feasibility/` 为调研期探针底座（ch10/ch11 引用）

## 运行环境

代码在写作与实验期运行于：Apple Silicon（MPS，bf16 autocast）+ CPU 数值对拍，Python 3.12 / PyTorch 2.13 / transformers 5.18 / sentencepiece / tiktoken / datasets。
随书复现：把本目录放入配套工作区后 `source env.sh`（缓存红线：一切数据/权重产物落 log/，书仓零数据零权重），按各章末「动手验证」的命令运行。
实验档位（M5 Pro 实测）：秒级（ch2 BPE/ch9 探针）～ 分钟级（ch3 mini-GLUE fast/ch10 热身）～ 小时级（ch8 族 47min/ch11 iso-FLOP 216min/124M 79min——过夜档，均有 fast 档可缩）。

## 代码索引

- `code/ch00/`：make_fig_0_1.py
- `code/ch01/`：make_figures.py
- `code/ch02/`：bpe.py、make_figs.py
- `code/ch03/`：glue_finetune.py、make_figs.py
- `code/ch04/`：make_figures.py、surgery.py、warmup_ablation.py
- `code/ch05/`：make_figures.py
- `code/ch06/`：make_figures.py
- `code/ch07/`：handcalc.py、make_figures.py
- `code/ch08/`：data_prep.py、fit_scaling.py、make_figures.py、run_family.py
- `code/ch09/`：make_figures.py、memory_probe.py
- `code/ch10/`：make_figures.py、model.py、shakespeare.py、train.py
- `code/ch11/`：data.py、iso_flop.py、official_gpt2.py、sample.py、scaling_curve.py、train_124m.py
- `code/ch12/`：make_figures.py

## 说明

- 纪律换血：正文知识冻结在 2022（GPT-3/Chinchilla 时代）；后册架构概念只在章末欠账框与 ch12 地图中以「最小披露」出现。
- 本册全部关键实验数字为真实运行所得（M5 Pro，2026-10）：停药实验、Kaplan 五点族（α=0.0413）、iso-FLOP 15 点网格、GPT-2 124M 从零训练与官方权重逐位对拍（max|Δlogits|=0.0，6/7 prompt 逐位一致）。
- 诚实条款：本机 124M 训练为 compute-limited 演示口径（≈6 TFLOPs 有效算力）；scaling 结论由 1M-20M 自训族与 iso-FLOP 网格承担，且一律带区间与不确定度（γ∈[0.85,1.01] band）。
