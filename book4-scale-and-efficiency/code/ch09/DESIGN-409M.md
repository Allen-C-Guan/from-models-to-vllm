# DESIGN-409M —— 两刀整机（207M 骨架 FFN→MoE + GQA→MLA）：设计文档 / 重启手册

> 版本：v1.0（2026-10-04，随 Book4 第 9 章定稿）
> 性质：**重启即可执行**的工程文档——不依赖原书上下文，所有决策自带依据与出处；持有算力的人凭本文件可直接重启本项目的设计与开发。
> 代码正身：同目录 `llama409.py`（模型）+ `train_409.py`（训练 CLI）——与 §1 依赖清单所列兄弟章文件同仓部署后方可运行。
> 证据等级制度（全文随行）：官方论文 > 官方技术报告 > config 逆向 > 第三方复测 > 社区配方 > 厂商自报；本书实验数字均来自实际运行并附命令。
> 形态先例：Book3 `ch10/DESIGN-215M.md`（207M 稠密四插槽整机）；本项目=同一骨架上「动两刀」的续篇——不动件与语料管线全部继承，动刀依据见 §3。

---

## 1. 目标与范围

在 Book3 的 207M LLaMA 式骨架（d=1024/L=12/h=16/V=32000 untied）上完成两刀改装并整合训练：**第一刀 FFN→MoE**（DeepSeekMoE 式：8 个路由专家 top-2 + 1 个共享专家，DSV2 键位）、**第二刀 GQA→MLA**（DSV2 式低秩 KV 联合压缩 + 解耦 RoPE + 查询压缩）。交付物（作者决策形态，2026-10-04）：

1. 整合模型代码（四插槽 import 组装，参数逐项账 assert，总参/激活参双口径）；
2. 训练 CLI（冒烟 / 整合短训两档，双臂 cand2/cand3，断点续跑，运行元数据 JSON）；
3. 一次 ≤30 分钟的整合短训双臂（两刀协同无冲突的真凭据）——**全量训练暂缓，见 §2**；
4. 本设计文档 / 重启手册。

不在范围内：全量长训（§2/§6 预算）、多 GPU 并行实现（预算表按 MFU 估算，工程留给重启现场）、推理引擎消费（§10 登记交接）、后训练工艺。

**依赖与文件清单**（重启第一步照单核对；「重启即可执行」的物理前提）：

- **目录结构**：本文件与 `llama409.py`/`train_409.py` 同住 `ch09/`。`llama409.py` 启动时做**跨册双候选 bootstrap**（工作区/书仓两布局自适应，存在即插入）：
  - Book3 兄弟目录（RMSNorm/稠密件/RoPE 供货方）：`code/Book3-现代开源骨架/{ch02,ch03,ch04,00-feasibility}` 或书仓 `from-models-to-vllm/book3-modern-open-source-skeleton/code/{...}`；
  - 本册兄弟目录（两刀供货方）：`code/Book4-规模与效率/{ch02,ch05,ch06,00-feasibility}`。
  - 只拷 `ch09/` 三个文件到空目录会 ImportError——五件 import 的物理落点必须齐。
- **五件 import（改装史即供货清单）**：RMSNorm=Book3 `ch02/rmsnorm.py`；稠密前馈件=Book3 `ch03/swiglu.py::SwiGLUMLP`（首层稠密机构用）；RoPE=Book3 `ch04/rope.py::build_rope_cache`；MLA=本册 `ch06/mla.py::MultiHeadLatentAttention`（forward(x, cos, sin) 插槽契约，`path='explicit'|'absorbed'` 双路径，支持 q_lora_rank=None）；MoE=本册 `ch05/deepseek_moe.py::DeepSeekMoE`（DSV2 键位：`mlp.gate.weight`+`mlp.experts.{gate_up,down}_proj`+`mlp.shared_experts.*`，noaux 可选）。
- **Python 依赖**：torch ≥2.x（自测/训练必需）；transformers 5.18.0（仅 `--parity` 对拍需要）；numpy + sentencepiece（语料校验需要）。
- **数据依赖**：`log/book3-ch10/tokens32k.bin`（56,284,791 token 训练池）+ `eval32k.bin`（1,200,000 token 独立 eval 池）+ `sp32000_owt.model` + `slice_meta_32k.json`（七件套校验对账单）。新机器按 §4 协议重切等量或更大语料即可，不必找回同一份文件。

## 2. 暂缓声明（作者决策原文）

> 本大项目全量训练暂时略过，待获得高算力机器后可凭此文档直接重启设计与开发。（2026-10-04）

量化依据（本机实测，2026-10）：训练档 B=8、T=1024（8,192 token/步）下，cand2 主臂 MPS bf16 实测 3.007 s/步（探针 F，10 步稳态均值；冒烟复测稳态 3.146、整合窗实测 4.318——后者为共享机器口径：窗内后台有权重预取下载与并行写作负载，损失曲线不受影响，预算外推一律以独窗值为锚）。按 Chinchilla 配方（约 20 token/参数 ≈ 8.2B token）需 999,012 步 ≈ **34.8 天连续运算**（复测口径 36.4 天）；cand3 对照臂 4.1B token ≈ 505,371 步 ≈ **20.6 天**。超出单机可维护性，故全量长训不在本机执行；≤30 分钟的整合短训（§7 验收④）作为「两刀协同无冲突」的管道验收凭据。

## 3. 架构与 config 逐字段依据

整机：token 查表 $(B,n)\to(B,n,1024)$ → 12 层 [pre-RMSNorm → MLA（解耦 RoPE 内旋）| pre-RMSNorm → MoE] 残差块（恒宽 1024）→ 终态 RMSNorm → untied 线性出口 $(B,n,1024)\to(B,n,32000)$ → 交叉熵（目标=下一 token，fp32）。无位置查表、无任何 bias。**两刀只动 FFN 与注意力两格；RMSNorm 与 RoPE 两格不动（承 Book3）——「改装史写在 import 语句里」。**

| 字段 | 值 | 依据 | 证据等级 |
|---|---|---|---|
| `vocab_size` V | 32000 | Book3 207M 底座原样（sp-32k 自训词表，语料与 Book3 短训同流跨章可比） | 本书设计决策（Book3 ch10 续） |
| `hidden_size` d | 1024 | Book3 207M 骨架不动——两刀只换插槽不改残差河宽度 | 本书设计决策 |
| `num_hidden_layers` L | 12 | 同上 | 同上 |
| `num_attention_heads` h | 16 | 同上（MLA 头数=查询头数；KV 份数字段退役——MLA 全头共享潜向量） | 同上 |
| `n_expert` E / `top_k` | 8 / 2 | E8/top-2=Mixtral 档最小完整 MoE（Book4 ch2 教学件同几何）；256 选 8 的旗舰档对本机内存不友好 | 本书设计决策（探针 F 定版） |
| `n_shared`（共享专家） | 1 | DeepSeekMoE 共享专家隔离（arXiv:2401.06066 §3.2）；1 份≈DSV3 的 1 共享（Lite 为 2） | 官方论文 + 本书设计决策 |
| `w_expert`（专家宽） | cand2=938 / cand3=329 | 等激活 `w=⌊d_ff/(k+n_shared)⌋=⌊2816/3⌋`（每层激活 FFN 宽 3×938=2814≈dense 2816——「每 token 计算量与稠密基线持平」）；等总参 `3·d·d_ff=3·d·(E+n_shared)·w+E·d → 解 w`（扫描解 329，总参最逼近 207,119,360） | 本书复算（探针 F 锁定） |
| `scoring` / `topk_method` / `routed_scaling_factor` | softmax / greedy / 1.0 | DSV2-Lite 式路由（HF `DeepseekV2` 类仅实现 greedy/group_limited_greedy——strict 对拍正身；Lite 实配 rsf=1.0）。**注意 V2-236B 是 rsf=16.0、组选择 8/3**——键位同族、参数异档，勿串 | config 逆向（deepseek-ai/DeepSeek-V2-Lite） |
| `first_k_dense_replace` | **0**（cand2/cand3 定版） | 机构由层循环实现（mini 对拍臂取 1 演练混排，层账分册）；定版取 0=12/12 层 MoE，与探针 F 锁定参数账 409,115,648/207,064,064 逐位一致。**登记**：卷级大纲行文曾作「首层稠密（first_k_dense_replace=1，DSV2-Lite 同款）」；若照此置 1，总参降为 391,824,384（cand2 口径）、cand3 需重扫 w——为保与已交付正文（ch0/ch1/ch6 引用 409M/207M 数）逐位一致，定版取 0，首层稠密留作重启变体（§8 风险表附反事实账） | 本书设计决策 + 负结论登记 |
| MLA `kv_lora_rank` $d_c$ | 256（cand2/cand3）/ 512（cand1 教学点） | 轻档=207M 尺度诚实（ch6：MLA「参数也省」要在 $h\cdot d_h\gg d_c$ 才成立；重档 512 在本尺度注意力槽反贵 +3,735,552/层）；cand1 只留参数账不作训练臂 | 本书复算 + Book4 ch6 |
| MLA `q_lora_rank` $d_c'$ | 256 / 512（同上两档） | 查询压缩只省训练激活不省 KV（DSV2 §2.1.2 原句）；轻档与 $d_c$ 同宽 | 官方论文 2405.04434 v5 §2.1.2 |
| MLA `qk_nope/rope/v_head` | 轻档 64/64/64；重档 128/64/128 | 轻档头维减半（槽参数近持平）；重档=DSV2 旗舰几何等比（128/64/128） | 本书设计决策 |
| `rope_theta` θ / `rms_norm_eps` | 10000 / 1e-5 | Book3 底座不动（8192 窗内够用；潜向量 RMSNorm 的 eps 另按 HF 类默认 1e-6 固定——见下「数值口径」） | Book3 ch10 续 |
| `tie_word_embeddings` | false（untied） | Book3 口径延续（出口几何独立 + HF 键位 strict 对拍前提） | config 逆向（LLaMA 出厂） |
| `max_position_embeddings` | 8192 | 训练窗设定（外推工程不在本项目范围；YaRN 施加于解耦键 $k^R$ 的事实见 Book4 ch6 边栏） | 本书设计决策 |
| 初始化 | $N(0, 0.02^2)$ | 初始 loss 预言 $\ln V+d\sigma^2/2=10.578$（实测 10.584-10.591，千分位咬合；untied 无自泄漏） | 本书实验（2026-10） |

**数值口径约定**（与 HF transformers 5.18.0 `DeepseekV2ForCausalLM` eager 路径逐位对齐的充分条件；结构对拍实测见 §7 验收②）：softmax/方差统计在 fp32；RMSNorm 方差 fp32；潜向量 RMSNorm（`q_a_layernorm`/`kv_a_layernorm`）eps 固定 1e-6（HF 类默认，**不随 config.rms_norm_eps**——层内 LN 才用 config 值）；RoPE 解耦通道用复数约定（HF DSV2 的 view_as_complex；与 rotate_half 数学等价但共享权重时点积不同，跨实现对拍必须同约定）；路由打分强制 fp32 线性（HF 同构）；无 bias。

**参数逐项账**（`llama409.py` 自测 assert 逐位；公式为任意两刀 config 的手算通式）：

cand2（等激活·轻 MLA，主臂——「409M 两刀机」）：

| 项 | 算式 | 参数量 |
|---|---|---|
| MLA 投影/层 | $d r_q + r_q h(d_h{+}d_h^R) + d(d_c{+}d_h^R) + d_c h(d_h{+}d_v) + h d_v d$ | 2,686,976 |
| 潜向量 RMSNorm×2/层 | $d_c + r_q$ | 512 |
| 层内 RMSNorm×2/层 | $2d$ | 2,048 |
| MoE/层 | 路由门 $Ed$ + 路由专家 $E\cdot 3dw$ + 共享 $3dw$ | 8,192 + 23,052,288 + 2,881,536 |
| 每层小计 ×12 | $28{,}631{,}552\times12$ | 343,578,624 |
| `embed_tokens` + `lm_head`（untied）+ 终态 norm | $2Vd + d$ | 65,537,024 |
| **总计** | | **409,115,648** |
| 激活参（本册口径=含 lm_head、不含输入 embedding lookup、含路由门） | | 168,877,056 |
| 激活参（non-embedding） | | 136,109,056 |
| KV/token | $L(d_c+d_h^R)=12\times320$ | 3,840 元素（vs 退役 GQA $2Lh_{kv}d_k$=12,288，**−68.75%**；bf16 7.5 KiB/token） |

cand3（等总参·轻 MLA，对照臂——「207M 等参两刀机」）：同 MLA，$w=329$ → MoE/层 9,104,384，每层 11,793,920，总计 **207,064,064**（与 Book3 207M 差 55,296=0.027%——同预算两刀 vs 稠密的最强对照）；激活参（本册口径）101,526,528 / non-emb 68,758,528；KV/token 同 3,840。

cand1（等激活·重 MLA，参数账教学点，不设训练档）：$d_c{=}d_c'{=}512$、头维 128/128 → MLA 槽 6,881,280/层（比 GQA 槽 3,145,728 **反贵 +3,735,552**）；总计 459,453,440；KV/token 6,912（−43.75%）。「旗舰叙事『参数也省』要在 $h\cdot d_h\gg d_c$ 时才成立」（DSV2 档 $h d_h/d_c=32$ 大赢；本档=4 反贵）——小尺度诚实，账根在 Book4 ch6。

## 4. 数据配方

- **语料**：与 Book3 ch10 整合短训**同一份** `tokens32k.bin`（OpenWebText 镜像 51,976 篇、sp-32000 逐篇编码、doc 间 EOS；训练池 56,284,791 token（流首部）+ eval 池 1,200,000 token（尾切独立，无重叠））。两刀机与 Book3 207M 短训曲线因此**同协议跨章可比**（cand3 等参对照臂的读数意义正在于此）。
- **切流协议**：第 k 步消费 $[(k-1)\times8192,(k+1)\times8192)$——x=前 8192、y=后 8192，顺序一次通过、不重复。
- **七件套参数带校验**（`train_409.py` 启动时复检，重切后必须全过）：piece_size==32000；id∈[0,32000)；train≥42,000,000；eval==1,200,000；两池磁盘字节==2×token 数；EOS 计数==编码篇数（51976）；流首部与首 3 篇重编码逐 id 一致。
- **eval 协议**（跨机可比，勿改）：eval 池内固定种子（20261002）抽 49 窗 $(8,1024)$（401,408 token），fp32 前向，报平均交叉熵。
- **全量训练的语料量**：Chinchilla 档需 8.2B token ≈ 146× 当前池——重启时以更大公开语料按同协议重切（配比参照 LLaMA1 Table 1，官方论文 2302.13971 §2.1）；56M 池最多支撑 6,871 步，会被步数断言拒跑。

## 5. 训练配方（五件套，Book2 定版沿用）

| 项 | 值 |
|---|---|
| 优化器 | AdamW，$\beta_1{=}0.9$、$\beta_2{=}0.95$，weight decay 0.1 |
| 梯度裁剪 | 全局范数 1.0 |
| lr 日程 | cosine：峰值 1e-3 → 终值 1e-4（峰值 10%），warmup 100 步线性 |
| 批量 | $B{=}8\times T{=}1024$ = 8,192 token/步（全量档建议加大批量并随批量重调 lr——LLaMA1 用 4M token/批） |
| 精度 | 训练 bf16 autocast / eval fp32（数值口径约定不随精度档放宽；MPS 训练逐步 sync 计时） |
| 种子 | 20261002（全书统一；重启可换，须写入元数据） |
| 断点 | 每 eval 边界存 ckpt（模型+优化器+步号）；中断重跑同命令自动续（超窗补齐，判据不降） |
| 均衡件 | **无**（softmax greedy 路由，不带 aux loss/noaux）——DSV2-Lite 同款裸配置。短训窗口内负载不均不构成验收项；全量重启时的升级选项见 §8 |

## 6. 预算表（6ND 口径：FLOPs = 6 × 激活参（含 lm_head）× tokens；cand2 激活参 168,877,056）

| 档 | tokens | FLOPs | 机器（口径） | wall-clock | 证据等级 |
|---|---|---|---|---|---|
| Chinchilla（20 tok/param） | 8.2B | 8.29e18 | 本机 MPS bf16（实测 3.007 s/步；复测 3.146） | **34.8 天**（36.4） | 本书实验外推 |
| 整合短训（已执行） | 4.4M | — | 本机 MPS bf16（540 步；共享机器口径 wall 50 min，独窗预算 ≈28 min） | eval 7.54→6.09 | 本书实验 |
| Chinchilla | 8.2B | 8.29e18 | 1×A100-80GB（312 TFLOPS × 40% MFU 假设） | ≈18.5 小时 | 本书估算 |
| Chinchilla | 8.2B | 8.29e18 | 1×H100（989 TFLOPS × 40%） | ≈5.8 小时 | 本书估算 |
| Chinchilla | 8.2B | 8.29e18 | 8×H100 @40% | ≈44 分钟 | 本书估算 |
| 旗舰口径（参照） | 15T | 1.52e22 | 8×H100 @40% | ≈55.6 天 | 本书估算 |
| 旗舰口径（参照） | 15T | 1.52e22 | 64×H100 @40% | ≈7.0 天 | 本书估算 |
| cand3 对照臂 Chinchilla | 4.1B | 2.52e18 | 本机 MPS bf16（实测 3.514 s/步；整合窗 2.751——共享机器口径） | 20.6 天 | 本书实验外推 |

估算口径声明：GPU 行按 bf16 dense 峰值 × MFU=40% 假设外推；409M 小模型实际 MFU 可能低于 40%（matmul 规模小、逐专家 Python 循环 launch-bound——本机口径 MoE 派发开销 1.44×，见 Book4 ch1），估时偏乐观、量级可靠。显存门槛：AdamW fp32 全套状态 ≈16 B/参数（按总参 409M ≈ 6.5 GiB）+ 激活（B8/T1024 数 GiB，本机 MPS 峰值 8.55 GB）——单张 24GB 消费卡可重启 Chinchilla 档，**门槛是时间不是显存**。

## 7. 验收标准（可机器判定）

1. **参数逐位**：`python llama409.py` 自测——cand2=409,115,648、cand3=207,064,064、cand1=459,453,440（手算=实测=探针 F 锁定数）。
2. **结构对拍**：`python llama409.py --parity`——两臂 vs HF `DeepseekV2ForCausalLM` 同 config 随机权重 strict 直搬，CPU fp32：max|Δlogits|<1e-4 且 argmax/top-5 一致率 100% 且未搬运对照显著不同。mini 臂（探针 G 路径复跑）历史基线 9.54e-07/100%；cand2 全机 409M 臂为本书最大 strict 直搬。
3. **冒烟健康**：`python train_409.py --task smoke --arm cand2`——初始 loss ∈ ln 32000+[0, 0.5]（精确预言 ln V+ds²/2=10.578；实测 10.5910）；全程有限；20 步有限下降（实测 →8.267，eval 8.106）。cand3 同判据（实测 10.5843→8.298，eval 8.128）。
4. **整合短训**：`python train_409.py --task integrated --arm cand2 --out-name run1`（540 步）与 `--arm cand3`（512 步）——判据：两刀协同无 NaN、eval 曲线下降（管道验收口径，非效果训练；单种子+JSON 存档+趋势判据——MPS 非逐位确定）。已执行读数（2026-10）：cand2 eval 7.5428→**6.0898**（稳态 4.318 s/step，共享机器口径）、cand3 7.6512→**6.1389**（2.751）；cand3 第 512 步对 Book3 207M 稠密第 500 步 6.2015——同预算两刀不劣（单种子趋势级）。
5. **（重启后）全量训练**：语料按 §4 重切 ≥8.2B token，`--task integrated --steps 999012 --out-name full1`（ckpt 续跑承载长训；eval 边界建议放宽到 1000 步并写入元数据）。判据：训练至 token 预算，eval 收敛、无发散；与本章短训曲线同 eval 协议可接续。
6. **（条件性）V2-Lite 真权重核验**：`python llama409.py --v2lite`——权重完整落地（≈31.4 GiB）则 safe_open 惰性读逐层形状账本；未落地降级不阻塞。

## 8. 风险与预案

| 风险 | 表现 | 预案 |
|---|---|---|
| 初始 loss 异常（数百） | 初始化 std 被框架默认覆盖 | 核对 `initializer_range=0.02`；式 $\ln V+\tfrac12ds^2$ 定位 |
| bf16 数值病（NaN/尖峰） | 方差/softmax 精度不足 | 数值口径约定不放宽（fp32 统计）；复发则降 fp32 求因；稳定化组件升级件=QK-Clip（MLA 适配版）已有社区先例（K2 §2.1），QK-Norm 不适用 MLA（K 不物化） |
| MoE 负载崩塌（全量长训期） | 专家垄断/死专家（短训窗口内不构成验收项；Book4 ch3 本机现场：无干预 200 步 max/min 可至 327.8） | 升级均衡件（组件已备、零结构改动）：①aux loss 前向钩子（Book4 ch3 方法）；②noaux bias（`DeepSeekMoE(use_bias=True)` + `step_bias`，Book4 ch5 件） |
| 长训中断 | ckpt 边界前进度丢失 | 每 eval 边界存 ckpt；同命令自动续跑；nohup/独立会话启动 |
| 数据不足 8.2B | 56M 池被步数断言拒跑 | 换更大公开语料按 §4 协议重切（七件套校验必须重过） |
| 过拟合（小池多 epoch） | eval 回升 | 全量档语料按 token 预算配足（每 token 至多一遍铁律） |
| MFU 不及预期 | 实测 s/步高于预算表 | 先测 100 步实速再定档；加大 B/T 或融合专家算子（工程优化，不改数值口径）；朴素逐专家循环在本机口径的派发开销 1.44×（Book4 ch1） |
| 结构对拍失败 | max\|Δ\|>1e-4 或 strict 报错 | 按数值口径约定逐项排查（最常见：潜向量 LN eps 读成 config 值、RoPE 约定错装 half/complex、路由未走 fp32） |
| 首层稠密变体（重启选项） | 置 `first_k_dense_replace=1` | 反事实账（cand2 口径）：层 0 FFN 换稠密 2816，总参 391,824,384；cand3 需重扫 w。与已锁定账不一致属预期——重定版须同步刷新 §3 表与验收①预期值 |

## 9. 重启 checklist（十步）

1. 环境与文件：按 §1 依赖清单备齐 Book3/Book4 兄弟目录五件 import 落点与依赖库；`python llama409.py` 自测通过（参数逐位三臂）。
2. 数据：直用 `log/book3-ch10/` 三件套（七件套校验全过），或按 §4 换更大语料重切。
3. 验收①：参数 assert 409,115,648 / 207,064,064。
4. 验收②：`--parity` 两臂 strict 直搬（max|Δlogits|<1e-4）。
5. 验收③：`--task smoke` 双臂（初始 loss≈10.578、有限、下降）。
6. 验收④：`--task integrated` 双臂（无 NaN、eval 下降；cand2 540 / cand3 512）。
7. 决策点：是否启用均衡件（§8 风险表——建议全量档预登记 noaux 或 aux 钩子，单变量对照）。
8. 全量训练：按 §6 定档后 `--task integrated --steps 999012 --out-name full1`（语料先按 §4 重切至 ≥8.2B；批量/lr/eval 边界随机器重定并写入元数据）。
9. 曲线入档：与本书 540/512 步短训曲线（同 eval 协议）的接续关系记入运行报告。
10. 消费线登记：产物权重按 §10 交接。

## 10. 消费线衔接声明

本项目的全量长训暂缓（§2）。**Book6 mini 推理引擎的默认服务对象因此维持登记为 Book3 终态稠密+GQA 小模型**（Book3 ch6 模块消融产物与 ch10 整合短训 checkpoint，207M、四插槽同构、KV 账 24 KiB/token——单轨三形态口径不变，防断链）。本项目两刀整机在消费线上的身份是**账本算例与选做扩展**：Book6 第 2-3 章的显存/算力账本以两刀机为 MoE 激活参数账与 MLA latent KV 账（3,840 元素/token）的算例；mini 引擎接入两刀机属选做扩展（MLA 吸收形态的引擎落地归 Book7 MLA 后端——本册只登记接口事实：缓存布局 $(B,1,n,d_c)$+$(B,1,n,d_h^R)$ 单头对）。重启完成后，全量权重沿同一登记交接。

## 11. 参考与文献锚

- 两刀组件：DeepSeekMoE（细粒度+共享专家）arXiv:2401.06066 v1；noaux 免辅助损失 arXiv:2408.15664 v1；DSV2（MLA 正身，§2.1 Eq 9-19 + App C 吸收推导）arXiv:2405.04434 v5；DSV3（同构续篇）arXiv:2412.19437 v2。
- 不动件与底座：RMSNorm arXiv:1910.07467；SwiGLU arXiv:2002.05202；RoPE arXiv:2104.09864 v5；GQA arXiv:2305.13245（退役件的账本仍以它为锚）；Book3 `DESIGN-215M.md`（底座设计先例）。
- 配方谱系：Chinchilla arXiv:2203.15556（20 token/参数处方）；GPT-2（五件套源头）；LLaMA1 arXiv:2302.13971（§2.3 配方/Table 1 数据配比）。
- 对照实现：HF transformers 5.18.0 `modeling_deepseek_v2.py`（结构对拍锚；DSV2 router 仅 softmax greedy/group_limited_greedy 两路）。
- 实权重参照：DeepSeek-V2-Lite（DeepSeek License v1.0；同族大表哥——MLA+细粒度 MoE+共享专家+softmax greedy；q_lora_rank=null 分支）。
- 本机实测底数：`log/book4-feasibility/probeF_config_run1.json`（三候选账+秒/步+内存）、`log/book4-ch09/`（自测/对拍/短训产物，随行 JSON）。

---

*本文档所有「实测」数字来自 2026-10 本书工作区的实际运行（设备/精度/种子/产物 JSON 路径随行标注）；引用他人数字处已标证据等级。重启时请以自己机器的实测数字刷新预算表，勿直接抄用。*
