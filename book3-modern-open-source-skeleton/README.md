# Book3《现代开源骨架：LLaMA 与组件化时代》

> 《从模型到 vLLM》丛书第三册 · 定稿于 2026-10-04 · 前置：Book1《Transformer 原典》+ Book2《预训练革命》
> 论文锚定见各章证据等级标注（RMSNorm=1910.07467/GQA=2305.13245/YaRN=2309.00071/Code Llama=2308.12950——四处调研期勘误；RoPE=2104.09864 v5；NTK-aware=社区配方级）

## 目录

- 导览：把这台机器升级到今天
- 第 1 章 开源的最优配方：更少参数、更多数据
- 第 2 章 pre-RMSNorm：归一化零件的演化
- 第 3 章 SwiGLU：前馈插槽的门控化
- 第 4 章 RoPE 深剖：把位置转进注意力里
- 第 5 章 untied 与词表：出入口的账
- 第 6 章 KV cache 正式登场：从 MQA 到 GQA
- 第 7 章 长上下文工程线：外推的三代配方
- 第 8 章 LLaMA1/2/3：组装与代际
- 第 9 章 无论文模型怎么读：Llama 4 与线终局
- 第 10 章 大项目：单轨整机整合（设计文档版）
- 第 11 章 收束：组件化时代的判断力

## 本册结构

- `chNN-*.md`——各章书稿（ch0 为导览，ch11 为收束章含册末术语表与 F/N 十四行总决算）
- `figures/`——全部插图（300dpi，自绘重制/自产，图注含来源与证据等级）
- `code/chNN/`——**随书代码快照**（正文已升级为行级链接，可直接点击跳转）；`code/00-feasibility/` 为调研期探针底座；`code/ch10/DESIGN-215M.md` 为大项目设计文档/重启手册（**凭它可在高算力机器直接重启 215M 档全量训练**——作者决策：全量长训暂缓，本册交付整合代码+冒烟+整合短训+设计文档）

## 运行环境与依赖

代码在写作与实验期运行于：Apple Silicon（MPS，bf16 autocast）+ CPU 数值对拍，Python 3.12 / PyTorch 2.13 / transformers 5.18 / sentencepiece / numpy。
随书复现要点：
1. 最简路径——在**本丛书书仓根**（from-models-to-vllm/，含 book2）或配套工作区内运行：`source env.sh`（缓存红线：一切数据/权重产物落 log/，书仓零数据零权重）；
2. ch02/ch03/ch04/ch06 的消融脚本 import **Book2 ch10 的 GPT-2 三插槽底座**（`from model import GPT2, GPT2Config`）——脚本内 bootstrap 对工作区与书仓两种布局自适应（书仓布局解析到 `../book2-pretraining-revolution/code/ch10`）；
3. 语料复用 Book2 遗产（log/book2-ch08/tokens.bin 等）；ch10 需 sp-32k 重切（`code/ch10/prep_sp32k.py`，分钟级）；
4. 实验档位（M5 Pro 实测）：秒级（对拍/性质验证）～ 分钟级（模块消融 fast 1000 步≈5 min/臂）～ 晚间档（full 5000 步≈27 min/臂；外推全案≈40 min；ch10 整合短训 1000 步≈12-17 min）。

## 说明

- 纪律换血：正文知识冻结在 2024-07（LLaMA3 时代）；Llama 4（2025+）仅 ch9 以 config 逆向案例身份出现；Book4+ 概念只在六处白名单位置最小披露。
- 本册全部关键实验数字为真实运行所得（M5 Pro，2026-10）：四插槽模块消融（判据预注册）、KV 账本手算=实测 0.00%、外推四干预对照（直接外推 +1.14 nat 被 YaRN-lite 压回 +0.24）、207M 整机（参数逐位 207,119,360；HF 结构对拍 max|Δlogits|=6.9e-6；TinyLlama v1.1 实权重对拍 max|Δ|=4.9e-5；整合短训 1000 步 loss 10.56→5.72）。
- 大项目处置（作者决策 2026-10-03）：215M 档全量长训暂缓（本机 Chinchilla 配方需 10.34 天、15T 口径 98.8 年——量化论证见 ch10）；重启手册=code/ch10/DESIGN-215M.md（4×A100 半天/8×H100 一小时内档位估算含内）。

## 代码索引

- `code/00-feasibility/`：01_probe_a_base.py、02_probe_b_module.py、03_probe_c_215m.py、04_probe_d_longwin.py、05_probe_e_parity.py、06_probe_f_kv.py、07_probe_g_extrap.py、llama_slots.py
- `code/ch00/`：make_fig_0_1.py
- `code/ch01/`：handcalc.py、make_fig_1_1.py
- `code/ch02/`：ablation_norm.py、make_figures.py、preregistered.md、rmsnorm.py
- `code/ch03/`：ablation_ffn.py、make_figures.py、preregistered.md、swiglu.py
- `code/ch04/`：ablation_pe.py、make_figures.py、preregistered.md、rope.py
- `code/ch05/`：config_infer.py、make_figures.py、preregistered.md、vocab_scan.py
- `code/ch06/`：ablation_gqa.py、gqa.py、kv_probe.py、make_figures.py、preregistered.md
- `code/ch07/`：extrapolate.py、make_figures.py、passkey.py、preregistered.md
- `code/ch08/`：make_figures.py
- `code/ch09/`：config_reverse.py
- `code/ch10/`：DESIGN-215M.md、llama215.py、make_figures.py、prep_sp32k.py、train_215.py
- `code/ch11/`：make_figures.py
