# Book4《规模与效率：MoE、MLA 与旗舰世代》

> 《从模型到 vLLM》丛书第四册 · 定稿于 2026-10-04 · 前置：Book1《Transformer 原典》+ Book2《预训练革命》+ Book3《现代开源骨架》
> 论文锚定见各章证据等级标注（调研期 arXiv ID 勘误四处：FP8 格式=2209.05433、QK-Norm=2010.04245、FP8-LM=2310.18313、ViT-22B=2302.05442；MLA 记号以 DSV2 v5 正字 d_c/d_c'/d_h^R）

## 目录

- 导览：在好骨架上动两刀
- 第 1 章 效率悖论——三本账分开记
- 第 2 章 MoE 基本机构：路由器、top-k 与容量
- 第 3 章 MoE 训练难题：负载、不稳定与辅助损失家族
- 第 4 章 Mixtral：开源 MoE 组装术的实证
- 第 5 章 DeepSeekMoE：细粒度、共享专家与免损失的均衡
- 第 6 章 MLA：把 KV 装进小箱子
- 第 7 章 训练侧工程：低精度处方与不 spike 的组件
- 第 8 章 MTP：让模型多想一步棋
- 第 9 章 DSV2→DSV3 整机拆解与两刀整合（大项目·设计文档版）
- 第 10 章 旗舰整机拆解：gpt-oss 精拆与速览表
- 第 11 章 LatentMoE 支线：把专家搬进更小的空间
- 第 12 章 收束：训练侧效率的极限与瓶颈转移

## 本册结构

- `chNN-*.md`——各章书稿（ch0 为导览，ch12 为收束章含册末术语表与 F/N/B3 十钩总决算+B4-1~7 新钩地图）
- `figures/`——全部插图（300dpi，自绘重制/自产，图注含来源与证据等级）
- `code/chNN/`——**随书代码快照**（正文已升级为行级链接，可直接点击跳转）；`code/00-feasibility/` 为调研期探针底座（含 moe_mla_slots 组件库正身与 parity_all 聚合对拍）；`code/ch09/DESIGN-409M.md` 为大项目设计文档/重启手册（**凭它可在高算力机器直接重启两刀机全量训练**——作者决策：全量长训暂缓，本册交付整合代码+冒烟+整合短训+设计文档）

## 运行环境与依赖

代码在写作与实验期运行于：Apple Silicon（MPS，bf16 autocast）+ CPU 数值对拍，Python 3.12 / PyTorch 2.13 / transformers 5.18 / numpy / sentencepiece。
随书复现要点：
1. 最简路径——在**本丛书书仓根**（from-models-to-vllm/，含 book3）或配套工作区内运行：`source env.sh`（缓存红线：一切数据/权重产物落 log/，书仓零数据零权重）；
2. ch02/ch05/ch06 为插槽教学件（与 00-feasibility 正身逐位一致）；**ch09 两刀整机 import Book3 四插槽（ch02 RMSNorm/ch03 SwiGLU/ch04 RoPE）+ 本册 ch05 DeepSeekMoE + ch06 MLA**——脚本内 bootstrap 对工作区与书仓两种布局自适应（书仓布局解析到 `../book3-modern-open-source-skeleton/code/…`）；
3. 语料复用前册遗产：模块实验 `log/book2-ch08/tokens.bin`（sp-8k）；整机档 `log/book3-ch10/tokens32k.bin`（sp-32k，56.3M token——`code/Book3…/ch10/prep_sp32k.py` 可重切）；
4. 实验档位（M5 Pro 实测）：秒级（对拍/性质验证）～ 分钟级（模块消融 fast 300-1000 步）～ 晚间档（full 2000-5000 步/臂；ch9 整合短训 540/512 步双窗）。

## 说明

- 纪律换血：正文知识冻结在旗舰世代 2025（DeepSeek 谱系+gpt-oss+Qwen3/Gemma3/K2）；2026 对象（K3/DSV4 等）只在边界注记与欠账框出现（白名单七处）；SWA/sink 只作 gpt-oss/Gemma 的 config 事实。
- 本册全部关键实验数字为真实运行所得（M5 Pro，2026-10）：ch5 四臂消融（细粒度+共享最优、noaux 以 <0.05 nat 换掉损失项、signflip 反向对照）、ch6 MLA 显式 vs 吸收等价 9.3e-11 与多几何 KV 账本、ch8 挂 MTP 主支路反优 0.129 nat、ch9 两刀机（参数逐位 409,115,648/207,064,064；HF 对拍 7.6e-06；冒烟 10.5910≈ln 32000+d·σ²/2；整合短训双臂曲线）、ch10 gpt-oss 三方逐位 20,914,757,184+十机型速览表。
- 大项目处置（作者决策 2026-10-04）：两刀机（FFN→MoE+GQA→MLA）全量长训暂缓；重启手册=code/ch09/DESIGN-409M.md（A100/H100 MFU 档位估算含内）；Book6 mini 引擎默认服务对象仍为 Book3 终态稠密+GQA 小模型（两刀机登记为账本算例与选做扩展）。

## 代码索引

- `code/00-feasibility/`：01_probe_a_base.py、02_probe_b_moe.py、03_probe_c_mla.py、04_probe_d_dense_vmoe.py、05_probe_e_fp8sim.py、06_probe_f_config.py、07_probe_g_parity.py、moe_mla_slots.py、parity_all.py
- `code/ch00/`：make_fig0.py
- `code/ch01/`：account.py、make_figures.py
- `code/ch02/`：ablation_moe.py、fig_ch02.py、moe.py
- `code/ch03/`：ablation_balance.py、make_figures.py
- `code/ch04/`：fig_ch04.py、mixtral_infer.py、route_analysis.py
- `code/ch05/`：ablation_granularity.py、deepseek_moe.py、fig_ch05.py
- `code/ch06/`：fig_ch06.py、kv_probe_mla.py、mla.py
- `code/ch07/`：fp8_sim.py、make_figures.py
- `code/ch08/`：fig_ch08.py、mtp.py、smoke_mtp.py
- `code/ch09/`：DESIGN-409M.md、fig_ch09.py、llama409.py、train_409.py
- `code/ch10/`：gptoss_reverse.py
- `code/ch11/`：fig_ch11.py
- `code/ch12/`：make_figures.py
