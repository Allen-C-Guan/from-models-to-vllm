# DESIGN-215M —— LLaMA 式 207M 预训练模型：设计文档 / 重启手册

> 版本：v1.1（2026-10-04；v1.0 同日随 Book3 第 10 章定稿。v1.1 批三审校：补 §1 依赖清单、§7/§9 全量命令显式化、§4 倍数勘正）
> 性质：**重启即可执行**的工程文档——不依赖原书上下文，所有决策自带依据与出处；持有算力的人凭本文件可直接重启本项目的设计与开发。
> 代码正身：同目录 `llama215.py`（模型）+ `train_215.py`（训练 CLI）+ `prep_sp32k.py`（数据）——三者与 §1 依赖清单所列兄弟章文件同仓部署后方可运行。
> 证据等级制度（全文随行）：官方论文 > 官方技术报告 > config 逆向 > 第三方复测 > 社区配方 > 厂商自报；本书实验数字均来自实际运行并附命令。

---

## 1. 目标与范围

从零预训练一台 LLaMA 式 207M decoder-only 语言模型（结构对齐 LLaMA1/2 出厂口径，非任何官方权重的衍生），语料为公开的 OpenWebText 镜像，词表 32000（SentencePiece BPE 自训）。交付物：

1. 整合模型代码（四插槽组件 import 组装，参数逐项账 assert）；
2. 训练 CLI（冒烟 / 整合短训 / 全量训练三档，断点续跑，运行元数据 JSON）；
3. 全量训练产出的 checkpoint 与训练曲线（**暂缓，见 §2**）。

不在范围内：微调/对齐工艺（后续册内容）、多 GPU 并行实现（预算表按 MFU 估算，工程实现留给重启现场）、推理引擎（消费方见 §10）。

**依赖与文件清单**（重启第一步照单核对；「重启即可执行」的物理前提）：

- **目录结构**：本文件与 `llama215.py`/`train_215.py`/`prep_sp32k.py` 同住 `ch10/`；同仓还需兄弟章交付 `ch02/rmsnorm.py`、`ch03/swiglu.py`、`ch04/rope.py`、`ch06/gqa.py`（四插槽正件）与 `00-feasibility/llama_slots.py`（`LLaMAConfig` 鸭子类型正身）。`llama215.py` 启动时把上述五个目录插入 `sys.path` 再 import——只拷三个文件到空目录会 ImportError。
- **Python 依赖**：torch ≥2.x（自测/训练必需）；transformers 5.18.0（仅 `--parity`/`--tinyllama` 两档对拍需要，结构对拍锚）；numpy + sentencepiece（`prep_sp32k.py` 训分词器与切流需要）。
- **数据依赖**：`prep_sp32k.py` 读两份输入——LM 主语料 `owt_docs.jsonl`（OpenWebText 公开镜像逐篇 jsonl，本书调研期下载）与分词器独立分片 `owt_tokenizer_corpus.txt`（自 OWT 另抽 6000 篇，与主语料互斥）。新机器不必找回同一份文件：从 OpenWebText 公开镜像（或按 §4 配比换 CommonCrawl 系/The Pile 类集合）重新获取等量语料，重跑 `prep_sp32k.py` 重训分词器并过 §4 七项校验即可。

## 2. 暂缓声明（作者决策原文）

> 本大项目全量训练暂时略过，待获得高算力机器后可凭此文档直接重启设计与开发。（2026-10-03）

量化依据（本机实测，2026-10）：训练档 $B{=}8$、$T{=}1024$（8,192 token/步）下，MacBook MPS bf16 实测 1.702 s/步（探针 C，12 步稳态均值；本章整合短训复测 1.21-1.93 s/步两窗——同机机器状态波动，结论对量级不变）。按 Chinchilla 配方（约 21 token/参数 ≈ 4.3B token）需 524,902 步 ≈ **10.34 天连续运算**；按 Llama 3 旗舰口径（15T token）需 18.3 亿步 ≈ **98.8 年**——前者超出单机可维护性，后者物理不成立。故全量长训不在本机执行。

## 3. 架构与 config 逐字段依据

整机：token 查表 $(B,n)\to(B,n,1024)$ → 12 层 [pre-RMSNorm → GQA（内旋 RoPE）| pre-RMSNorm → SwiGLU] 残差块（恒宽 1024）→ 终态 RMSNorm → untied 线性出口 $(B,n,1024)\to(B,n,32000)$ → 交叉熵（目标=下一 token，fp32）。无位置查表（位置=注意力内旋转角）、无任何 bias。

| 字段 | 值 | 依据 | 证据等级 |
|---|---|---|---|
| `vocab_size` V | 32000 | LLaMA1/2 出厂词表（sentencepiece BPE、数字逐位、byte fallback）；白纸黑字首见 LLaMA2 §2.2 | 官方论文 2307.09288 §2.2 |
| `hidden_size` d | 1024 | 目标带 205-225M 内的宽度档（探针 C 三候选比较） | 本书设计决策 |
| `intermediate_size` f | 2816 | $\lceil 8d/3\rceil_{256}=11\times256$：SwiGLU 同参数预算规则（Shazeer 2020 §3；LLaMA 官方代码 multiple_of=256 注释） | 官方论文 2002.05202 + 官方代码 |
| `num_hidden_layers` L | 12 | 与 GPT-2 124M 同层数（本丛书单轨底座），宽度 768→1024 放大 | 本书设计决策 |
| `num_attention_heads` h | 16 | d/64=16，head_dim=64 与 LLaMA1 小档一致 | config 逆向（LLaMA1-7B） |
| `num_key_value_heads` h_kv | 8 | GQA-2（组内 2 查询头）：KV 账本 24 KiB/token；8 组与 LLaMA2-34B/70B 及 Llama 3 全系同数（注意 LLaMA2-7B/13B 是 MHA）。组数扫描：8→4→1 付 0.109/0.158 nat 质量税、KV 字节 8:4:1——小档欠训练时砍狠了真疼，故不取 4 或 1 | 官方论文 2305.13245 + 本书实验（2026-10） |
| `max_position_embeddings` | 8192 | 训练窗设定（RoPE 无查表上限，越窗行为另议——外推工程不在本项目范围） | 本书设计决策 |
| `rope_theta` θ | 10000 | LLaMA1/2 出厂值；8192 窗内够用（Llama 3 的 5e5 属更长窗出厂设定） | config 逆向 |
| `rms_norm_eps` | 1e-5 | LLaMA 65B/LLaMA2/3 全系出厂口径（HF 默认 1e-6 是隐形开关，从 config 读） | config 逆向 |
| tie_word_embeddings | false（untied） | 对齐 LLaMA 出厂口径；代价 = 词表账翻倍至 65,536,000（占全机 31.6%）；收益 = 出口几何独立 + HF 键位 `lm_head.weight` 独立存在（结构对拍前提）。三篇 LLaMA 论文对 untying 均无论证（grep 负结论）——按出厂口径执行 | config 逆向 + 负结论登记 |
| bias（全部） | 无 | LLaMA 出厂选择（RMSNorm 论文本有 bias，LLaMA 实现砍掉） | config 逆向 + 官方代码 |
| 初始化 | $N(0, 0.02^2)$ | 初始 logit 方差 $\sigma^2=ds^2=0.4096$，初始 loss $=\ln V+\sigma^2/2=10.578$（实测 10.558，untied 无自泄漏、$E[z_t]\approx0$ 实测归零）；$s$ 变大初始 loss 失控（Book2 复演过 $s{=}1$ 时 loss≈480 的「复读机」反例） | 本书实验（2026-10） |

**参数逐项账**（assert 锁死，`llama215.py` 自测逐位验证）：

| 项 | 算式 | 参数量 |
|---|---|---|
| 注意力/层 | $2d^2+2d\,h_{kv}d_k=2(1024^2)+2(1024)(512)$ | 3,145,728 |
| SwiGLU/层 | $3df=3(1024)(2816)$ | 8,650,752 |
| 两枚 RMSNorm/层 | $2d$ | 2,048 |
| 每层小计 ×12 | $11{,}798{,}528\times12$ | 141,582,336 |
| `embed_tokens` | $Vd$ | 32,768,000 |
| `lm_head`（untied） | $Vd$ | 32,768,000 |
| 终态 RMSNorm | $d$ | 1,024 |
| **总计** | | **207,119,360** |

通用公式（任意 LLaMA 式 config 的手算账）：$N=Vd(2-\mathbb{1}_{\text{tied}})+L(2d^2+2d\,h_{kv}d_k+3df+2d)+d$。

**数值口径五约定**（与 HF transformers 5.18.0 `LlamaForCausalLM` eager 路径逐位对齐的充分条件；结构对拍实测 max|Δlogits|=6.9e-06）：RMSNorm 方差在 fp32 计算；softmax 在 fp32；RoPE 用 rotate_half 两半式约定（与论文的相邻配对数学等价、排列不同——跨实现搬权重需 permute 重排）；GQA 用 repeat_kv 扩展视图；无 bias、eps 与 tie 从 config 读。

## 4. 数据配方

- **语料**：OpenWebText 镜像 `owt_docs.jsonl`（jsonl 逐篇文本）。本机实测切流：51,976 篇 → 57,484,791 token（uint16 裸流，doc 间插 EOS）；训练池 56,284,791（流首部），eval 池 1,200,000（尾切，与训练池无重叠）。压缩率 4.5382 字节/token。
- **分词器**：`prep_sp32k.py` 训练 sp-BPE（vocab_size=32000、byte_fallback、character_coverage=1.0）。**分词器训练语料与 LM 语料必须分离**（本配方便用独立分片 `owt_tokenizer_corpus.txt`，6000 篇/29MB，与主语料互斥）。特殊 id：unk=0/bos=1/eos=2；无 pad（pad_token_id 与 unk 撞 0 的 HF 口径差异，读 config 时注意）。
- **切流协议**：流首部逐篇编码（`add_eos=True`），第 k 个训练步消费 $[(k-1)\times8192,(k+1)\times8192)$——x=前 8192、y=后 8192，顺序一次通过、不重复。
- **七项参数带校验**（重切后必须全过）：piece_size==32000；id∈[0,32000)；train≥42,000,000；eval==1,200,000；磁盘字节==2×token 数；EOS 计数==编码篇数；首 3 篇回读逐 id 一致。
- **全量训练的语料量**：Chinchilla 档需 4.3B token ≈ 76× 当前池——重启时以更大公开语料（CommonCrawl 系清洗管线或 The Pile 类开源集合）按同协议重切；数据配比可参照 LLaMA1 Table 1（CC 67%/C4 15%/GitHub 4.5%/Wikipedia 4.5%/Books 4.5%/ArXiv 2.5% 等，官方论文 2302.13971 §2.1）。
- **eval 协议**（跨机可比，勿改）：eval 池内固定种子（20261002）抽 49 窗 $(8,1024)$（401,408 token），fp32 前向，报平均交叉熵。

## 5. 训练配方（五件套，Book2 定版沿用）

| 项 | 值 |
|---|---|
| 优化器 | AdamW，$\beta_1{=}0.9$、$\beta_2{=}0.95$，weight decay 0.1 |
| 梯度裁剪 | 全局范数 1.0 |
| lr 日程 | cosine：峰值 1e-3 → 终值 1e-4（峰值的 10%），warmup 100 步线性 |
| 批量 | $B{=}8\times T{=}1024$ = 8,192 token/步（全量档建议加大批量：LLaMA1 用 4M token/批——重启时按算力与 token 预算重定，lr 随批量重调） |
| 精度 | 训练 bf16 autocast / eval fp32（数值口径五约定不随精度档放宽） |
| 种子 | 20261002（全书统一；重启可换，须写入元数据） |
| 断点 | 每 eval 边界存 ckpt（模型+优化器+步号+流位置）；中断重跑同命令自动续 |

参考锚：LLaMA1 §2.3 的旗舰配方（AdamW 0.9/0.95、wd 0.1、cosine 终 10%、warmup 2000、batch 4M token）与本五件套同宗——差异集中在 warmup/batch 两个量纲，是 token 规模的工程后果（官方论文 2302.13971 §2.3）。

## 6. 预算表（6ND 口径：FLOPs = 6 × 207,119,360 × tokens）

| 档 | tokens | FLOPs | 机器（口径） | wall-clock | 证据等级 |
|---|---|---|---|---|---|
| Chinchilla | 4.3B | 5.34e18 | 本机 MPS bf16（实测 1.702 s/步；短训两窗复测 1.21-1.93，见 §2） | 10.34 天 | 本书实验外推 |
| 旗舰口径 | 15T | 1.86e22 | 本机 MPS bf16 | 98.8 年 | 本书实验外推 |
| 本机参考 | — | — | 本机 MPS fp32（9.80 s/步）/ CPU fp32（10.40 s/步） | — | 本书实验（探针 C） |
| Chinchilla | 4.3B | 5.34e18 | 1×A100-80GB（312 TFLOPS 峰值 × 40% MFU 假设） | ≈11.9 小时 | 本书估算 |
| Chinchilla | 4.3B | 5.34e18 | 1×H100（989 TFLOPS × 40% MFU 假设） | ≈3.8 小时 | 本书估算 |
| 旗舰口径 | 15T | 1.86e22 | 8×A100 @40% | ≈216 天 | 本书估算 |
| 旗舰口径 | 15T | 1.86e22 | 8×H100 @40% | ≈68 天 | 本书估算 |
| 旗舰口径 | 15T | 1.86e22 | 64×H100 @40% | ≈8.5 天 | 本书估算 |
| 竞品对照 | 未披露 | — | llama2.c 110M：4×A100×24 h | — | 厂商自报（README） |
| 竞品对照 | 未披露 | — | happy-llm 215M：8×4090×46 h | — | 厂商自报（5.3 节） |

估算口径声明：A100/H100 行按 bf16 dense 峰值 × MFU=40% 假设外推；207M 小模型在同 $B/T$ 下实际 MFU 可能低于 40%（matmul 规模小、嵌入与 softmax 占比高），估时偏乐观、量级可靠；竞品行 token 数与 MFU 均未披露，只作存在性参照，不可折算。显存门槛：AdamW fp32 全套状态 ≈16 B/参数 ≈ 3.31 GiB + 激活（B8/T1024 数 GiB）——单张 24GB 消费卡即可重启 Chinchilla 档，**门槛是时间不是显存**。

## 7. 验收标准（可机器判定）

1. **参数逐位**：`python llama215.py` 自测——手算=实测=207,119,360。
2. **结构对拍**：`python llama215.py --parity`——同 config 随机权重 strict 直搬 HF `LlamaForCausalLM`（transformers 5.18.0），CPU fp32：max|Δlogits|<1e-4 且 argmax/top-5 一致率 100% 且未搬运对照组显著不同（实测 6.9e-06 / 100% / 100% / 4.86）。
3. **冒烟健康**：`python train_215.py --task smoke`——初始 loss ∈ ln 32000 + [0, 0.5]（精确预言 ln V+ds²/2=10.578，实测 10.558；untied 无自泄漏，E[z_t]≈0）；全程有限；20 步有限下降（实测 →8.27）。
4. **整合短训**：`python train_215.py --task integrated`——1000 步（8.2M token）eval 曲线下降（实测 →5.84，单调；本口径=管道验收，非效果训练）。
5. **（重启后）全量训练曲线**：`python train_215.py --task full --out-name f1`——默认 524,902 步＝Chinchilla 4.3B token（`--steps` 可覆盖；旗舰口径按 §5 加大批量后重定步数）。前置：按 §4 重切 ≥4.3B token 语料（当前 56M 池会被训练池断言拒跑）。判据：训练至 token 预算，eval 收敛、无发散；与本章短训曲线接续（同 eval 协议）。
6. **（条件性）实权重对拍**：`python llama215.py --tinyllama`——TinyLlama_v1.1（Apache-2.0）真权重 strict 装机 + logits 对拍（实测 4.9e-05 / 100%）。meta-llama 为 gated 仓库，不碰。

## 8. 风险与预案

| 风险 | 表现 | 预案 |
|---|---|---|
| 初始 loss 异常（≈480 或数百） | 初始化 std 被框架默认覆盖（N(0,1)） | 核对 `initializer_range=0.02`；式 $\ln V+\tfrac12 ds^2$ 定位 |
| bf16 数值病（NaN/loss 尖峰） | 方差/softmax 精度不足 | 五约定不放宽（fp32 统计）；仍复发则降 bf16→fp32 求因，勿盲目调 lr |
| 长训中断（机器睡眠/断电） | ckpt 边界前进度丢失 | 每 eval 边界存 ckpt；同命令自动续跑；nohup/独立会话启动 |
| 数据不足 4.3B | OWT 池只有 56M token | 换更大公开语料按 §4 协议重切（七项校验必须重过） |
| 过拟合（小池多 epoch） | eval 回升 | 全量档语料按 token 预算配足（每 token 至多一遍铁律）；Book2 短训档实测约 16 遍起记忆化 |
| MFU 不及预期（重启机上） | 实测 s/步高于预算表 | 先测 100 步实速再定档；加大 $B/T$ 或启用融合算子（属工程优化，不改数值口径） |
| 结构对拍失败 | max\|Δ\|>1e-4 或 strict 报错 | 按五约定逐项排查（最常见：eps 从默认 1e-6 读入、rotate_half 约定错装） |

## 9. 重启 checklist（十步）

1. 环境与文件：按 §1「依赖与文件清单」备齐五个兄弟目录的插槽正件与依赖库（torch ≥2.x；对拍另需 transformers 5.18.0、数据另需 sentencepiece/numpy）；单卡 ≥24GB（A100/H100 更佳）；`python llama215.py` 自测通过（参数逐位）。
2. 数据：`prep_sp32k.py` 重切（或按 §4 换等量/更大语料）；七项参数带校验全过。
3. 验收①：参数 assert 207,119,360。
4. 验收②：`--parity` 结构对拍（max|Δlogits|<1e-4）。
5. 验收③：`--task smoke`（初始 loss≈10.58、有限、下降）。
6. 验收④：`--task integrated`（1000 步 eval 下降，eval 协议勿改）。
7. 全量训练：按 §6 定档后执行 `python train_215.py --task full --out-name f1`（默认 524,902 步＝Chinchilla 4.3B；语料先按 §4 重切至 ≥4.3B，否则被训练池断言拒跑；批量与 lr 随机器重定并写入元数据；旗舰口径 15T 为完整档）。
8. eval 协议保持 49 窗 fp32——与本书短训曲线跨机可比。
9. 曲线入档：与本书 1000 步短训曲线（eval →5.84）的接续关系记入运行报告。
10. 消费线登记：产物权重接 Book6 mini 引擎（§10）。

## 10. 消费线衔接声明

本项目的全量长训暂缓（§2），**Book6 mini 推理引擎的默认服务对象因此登记为已训的同构小模型**：Book3 第 6 章模块消融产物（20-50M 档 GQA 同构）与第 10 章整合短训 checkpoint（207M、1000 步、四插槽同构、HF 键位可直搬）。重启完成后，全量权重沿同一登记交接。规格提醒：该 207M 的 KV 账为 2·L·h_kv·d_k = 24 KiB/token（2048 窗单序列 48 MiB），引擎侧容量规划以此为准。

## 11. 参考与文献锚

- 组件：RMSNorm arXiv:1910.07467（NeurIPS 2019）；SwiGLU/GLU 变体 arXiv:2002.05202（Table 1 全数字）；RoPE arXiv:2104.09864 v5；MQA arXiv:1911.02150；GQA arXiv:2305.13245（EMNLP 2023，含 LLaMA2 采用的官方消融 A.2.1）。
- 整机谱系：LLaMA1 arXiv:2302.13971（§2.2 三改/§2.3 配方/Table 1 数据配比）；LLaMA2 arXiv:2307.09288（GQA 落点=34B/70B，7B/13B 仍 MHA）；Llama 3 arXiv:2407.21783（15T/15.6T 双口径）。
- 配方谱系：Chinchilla arXiv:2203.15556（20 token/参数处方）；GPT-2（五件套配方的社区标准源头）。
- 对照实现：HF transformers 5.18.0 `modeling_llama.py`（结构对拍锚）；llama2.c（MIT，结构对照——注意其教学版为静默绑定嵌入，与官方 untied 有偏差）。
- 权重对拍物：TinyLlama_v1.1（Apache-2.0；d=2048/L=22/h=32/h_kv=4/f=5632/V=32000 untied）。

---

*本文档所有「实测」数字来自 2026-10 本书工作区的实际运行（设备/精度/种子/产物 JSON 路径均随行标注）；引用他人数字处已标证据等级。重启时请以自己机器的实测数字刷新预算表，勿直接抄用。*
