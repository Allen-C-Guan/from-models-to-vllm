# Book5《前沿混合架构：稀疏、线性注意力与 2026 旗舰》

> 《从模型到 vLLM》丛书第五册 · 定稿于 2026-10 · 前置：Book1《Transformer 原典》+ Book2《预训练革命》+ Book3《现代开源骨架》+ Book4《规模与效率》
> 论文锚定见各章证据等级标注（调研期 arXiv ID 勘误六组：lost-in-the-middle=2307.03172、FWP=2102.11174、LLaVA=2304.08485、NoPE=2305.19466、Kimi K3=2607.24653（双锚+本地 PDF 页码）、GLM-5 有论文 2602.15763；NSA 记号正字 cmp/slc/win）

## 目录

- 导览：注意力躺在手术台上
- 第 1 章 满注意力仍是唯一解吗——三笔旧账、三条路线与一本新账
- 第 2 章 路线一：滑窗与汇聚——SWA 与 attention sink
- 第 3 章 路线二：可学习稀疏——NSA 到 DSA
- 第 4 章 路线二续：DSV4 与 GLM-5 的 1M 上下文
- 第 5 章 路线三：线性注意力——kernel、delta rule 与 Gated DeltaNet
- 第 6 章 Kimi K3 报告精读·上：注意力侧
- 第 7 章 Kimi K3 报告精读·下：Stable LatentMoE、MXFP4 QAT 与原生多模态
- 第 8 章 长上下文收束论：外推与状态压缩两条哲学
- 第 9 章 2026 混合配方谱系与三刀机
- 第 10 章 能力扩展专题：多模态与推理范式
- 第 11 章 方法论：如何读厂商自报与预览版时代的模型
- 第 12 章 收束：满注意力不再是唯一解

## 本册结构

- `chNN-*.md`——各章书稿（ch0 为导览，ch12 为收束章含册末术语表与十一钩+次级六钩总决算+B5-1~8 新钩地图）
- `figures/`——全部插图（300dpi，自绘重制/自产，图注含来源与证据等级；K3 报告与 Kimi Linear 图源 CC BY-NC-ND 禁用，机制主图全部从零原创）
- `code/chNN/`——**随书代码快照**（正文已升级为行级链接，可直接点击跳转）；`code/00-feasibility/` 为调研期探针底座；`code/ch09/DESIGN-3TO1.md` 为三刀机（3:1 选做整机）设计文档/重启手册（**凭它可在高算力机器直接重启全量训练**——作者决策：全量长训暂缓，本册交付整合代码+冒烟+整合短训+设计文档）

## 运行环境与依赖

代码在写作与实验期运行于：Apple Silicon（MPS，bf16 autocast）+ CPU 数值对拍，Python 3.12 / PyTorch 2.13 / transformers 5.18 / numpy / sentencepiece / matplotlib。
随书复现要点：
1. 最简路径——在**本丛书书仓根**（from-models-to-vllm/，含 book3/book4 兄弟册）或配套工作区内运行：`source env.sh`（缓存红线：一切数据/权重产物落 log/，书仓零数据零权重零图像）；
2. 跨册依赖=三候选 bootstrap（REPO_ROOT 三级上溯：工作区=workspace 根、书仓=书仓根）：ch05 gdn 件、ch06 K3 玩具、ch09 三刀机 import Book3 四插槽+Book4 两刀件+本册新件——「三刀改装史写在 import 语句里」；
3. 语料复用前册遗产：模块实验 `log/book2-ch08/tokens.bin`（sp-8k）；整机档 `log/book3-ch10/tokens32k.bin`（sp-32k）；needle/recall 任务为固定种子合成语料；
4. 实验档位（M5 Pro 实测）：秒级（对拍/性质验证）～ 分钟级（模块消融）～ 晚间档（recall 三臂 6000-10000 步/三刀机整合短训降档 100 步一窗）。

## 说明

- 纪律换血：正文知识冻结在 2026 旗舰世代（K3/DSV4/GLM-5/Qwen3.5 进正文主线）；推理系统侧概念（Book6+）只在七处白名单最小披露。
- 本册全部关键实验数字为真实运行所得（M5 Pro，2026-10）：ch2 needle 四档（full 六位全 1.0 vs 滑窗臂二跳中继界 cliff）、ch3 块稀疏对拍 5.4e-07+账本公式本机复现、ch5 GDN 三方互证 <1e-6+scaling 斜率 1.00 vs 1.85、ch6 K3 玩具 43.9M（KDA 对拍 2.98e-08/门粒度双臂/g 分布分化实测）、ch7 三处方 ablation（QB 治负载塌缩 642→2.7 强复现）+QAT 双臂、ch9 三刀机 409,000,736 逐位 assert+等参 recall 三臂负结论如实报、ch10 LLaVA mini 340 图微调。
- 大项目处置（作者决策 2026-10-05）：三刀机（两刀机 9 层注意力插槽换 GDN）全量长训暂缓；重启手册=code/ch09/DESIGN-3TO1.md；Book6 mini 引擎默认服务对象仍为 Book3 终态稠密+GQA 小模型（三刀机登记为 Book7-8 混合 KV 布局账本算例与选做扩展）。

## 代码索引

- `code/00-feasibility/`：01_probe_base.py、02_probe_swa_sink.py、03_probe_nsa.py、04_probe_gdn.py、05_probe_hybrid.py、06_probe_k3toy.py、07_probe_recall.py、08_probe_parity.py
- `code/ch00/`：make_figures.py
- `code/ch01/`：context_account.py、make_figures.py
- `code/ch02/`：fig_ch02.py、gptoss_sinks_dump.py、needle_probe.py、swa_sink.py
- `code/ch03/`：make_figures.py、nsa_block.py
- `code/ch04/`：dsv4_glm5_reverse.py
- `code/ch05/`：fig_ch05.py、gdn.py、qwen3next_mini.py
- `code/ch06/`：k3_toy.py、kda.py、make_figures.py、train_k3toy.py
- `code/ch07/`：fig_ch07.py、latentmoe_parts.py、qat_sim.py
- `code/ch08/`：make_figures.py
- `code/ch09/`：DESIGN-3TO1.md、fig_ch09.py、hybrid409.py、hybrid_recall.py、qwen35_consume.py、train_hybrid.py
- `code/ch10/`：fig_ch10.py、llava_mini.py
- `code/ch11/`：fig_ch11.py、params_audit.py
- `code/ch12/`：make_figures.py
