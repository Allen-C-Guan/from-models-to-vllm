# DESIGN-3TO1 —— 三刀整机（两刀机 9 层注意力插槽换 GDN → 3:1 混合，等总参 ≈409M）：设计文档 / 重启手册

> 版本：v1.0（2026-10-08，随 Book5 第 9 章成稿）
> 性质：**重启即可执行**的工程文档——不依赖原书上下文，所有决策自带依据与出处；持有算力的人凭本文件可直接重启本项目的设计与开发。
> 代码正身：同目录 `hybrid409.py`（模型+自测）+ `train_hybrid.py`（训练 CLI）+ `hybrid_recall.py`（混合比例消融）——与 §1 依赖清单所列兄弟章文件同仓部署后方可运行。
> 证据等级制度（全文随行）：官方论文 > 官方技术报告 > config 逆向 > 第三方复测 > 社区配方 > 厂商自报；本书实验数字均来自实际运行并附命令。
> 形态先例：Book3 `ch10/DESIGN-215M.md`（207M 稠密四插槽）→ Book4 `ch09/DESIGN-409M.md`（两刀机）——本项目=同一骨架上「动第三刀」的续篇：**Book3 稠密 207M → Book4 两刀 409M → 本册三刀 409M 等参三形态对照链**收官。

---

## 1. 目标与范围

在 Book4 cand2 两刀机（409,115,648 参；d=1024/L=12/MLA_LITE/E8 top2 s1 w=938）上完成第三刀改装并整合训练：**注意力插槽 9 层换 Gated DeltaNet**（层模式 `[gdn,gdn,gdn,mla]×3`，config 0 起 MLA@3/7/11＝人读第 4/8/12 层，末层全局承 K3「末层必全局」惯例）、**w_expert 938→810 等总参补偿**（式 (9.1)，详见 §3）。交付物（作者决策形态，2026-10-05）：

1. 整合模型代码（三刀整机 import 组装，参数逐项账 assert，总参/激活参双口径 + 分类型 KV/状态账）；
2. 训练 CLI（冒烟 / 整合短训两档，断点续跑，运行元数据 JSON）；
3. 一次 ≤30 分钟的整合短训（三刀协同无冲突的真凭据；大纲预算 600-900 步经实测降档为 100 步，留痕见 §2/§6）；
4. 本设计文档 / 重启手册。

另附（同章交付、独立于整机）：`hybrid_recall.py` 混合比例等参三臂消融（全全局 vs 3:1 vs 全线性，§7 验收⑥）与 `qwen35_consume.py` 真权重消费点（Qwen3.5-4B 加载对账，属第 9 章正文实验、非本机项目验收件）。

不在范围内：全量长训（§2/§6 预算）、多 GPU 并行实现（预算表按 MFU 估算）、推理引擎消费（§10 登记交接）、后训练工艺。

**依赖与文件清单**（重启第一步照单核对；「重启即可执行」的物理前提）：

- **目录结构**：本文件与 `hybrid409.py`/`train_hybrid.py`/`hybrid_recall.py` 同住 `ch09/`。`hybrid409.py` 启动时做**跨册三候选 bootstrap**（工作区 / 书仓嵌套 `from-models-to-vllm/...` / 书仓裸布局三档路径 `if isdir(p)` 依次尝试——`_B4_CH09_CANDS` 与 `_B5_CH05_CANDS`，llama409 `_B3_CANDS` 已验证模式）：
  - Book4 兄弟目录 `code/Book4-规模与效率/ch09/`（两刀机正身 `llama409.py`+`train_409.py`——import 时自挂 Book3 四件与 Book4 ch05/ch06）；
  - 本册兄弟目录 `code/Book5-前沿混合架构/ch05/`（`gdn.py::GatedDeltaNet`）。
  - 只拷 `ch09/` 四个文件到空目录会 ImportError——两代兄弟目录的物理落点必须齐。
- **import 血统（三刀改装史即供货清单）**：RMSNorm=Book3 `ch02/rmsnorm.py`；稠密件=Book3 `ch03/swiglu.py::SwiGLUMLP`（定版不启用，`first_k_dense_replace=0`）；RoPE=Book3 `ch04/rope.py::build_rope_cache`（插槽契约参数——MLA 解耦通道自建阶梯、GDN 结构级 NoPE，两者都不消费此表）；MLA=Book4 `ch06/mla.py::MultiHeadLatentAttention`；MoE=Book4 `ch05/deepseek_moe.py::DeepSeekMoE`；GDN=本册 `ch05/gdn.py::GatedDeltaNet(d, h_k, h_v, d_k, d_v)`（recurrent/chunkwise 双 path，chunk=64 默认）。`train_hybrid.py` 另跨册 import Book4 `train_409.py` 的训练五件套（TrainStream/build_eval_batches/lr_at 等，Book2 血统）与 `hybrid_recall.py` import Book4 `00-feasibility/moe_mla_slots.py` + 本册 `00-feasibility` 探针件 02/05。
- **Python 依赖**：torch ≥2.x（自测/训练必需）；transformers 5.18.0（仅 `qwen35_consume.py` 需要）；numpy（recall 消融）。
- **数据依赖**：`log/book3-ch10/tokens32k.bin`（56,284,791 token 训练池）+ `eval32k.bin`（1,200,000 token 独立 eval 池）+ 七件套校验对账单（`train_hybrid.py` 启动复检）。与 Book4 两刀机同一语料与 eval 协议——跨章可比是三形态对照链的读数前提。新机器按 §4 协议重切等量或更大语料即可。

## 2. 暂缓声明（作者决策，节录自卷级大纲 v1.1 决策①）

> 本大项目暂缓形态：单轨选做「注意力换 3:1 混合」整机按四件套交付（整合代码+冒烟+≤30min 短训+DESIGN 重启手册），全量长训 pending。（2026-10-05）

量化依据（本机实测，2026-10-08）：训练档 B=8、T=1024（8,192 token/步）下，三刀机 MPS bf16 实测稳态 11.715 s/步（smoke）/12.824 s/步（integrated 窗，步内漂移 10.1→15.3 热节流）——**大纲推算「2-3 s/step→600-900 步/窗」作废**：归因实测（`log/book5-ch09/chunk_bench_s1.json`）单 GDN 层 fwd+bwd 0.677 s（chunk=64 块循环、MPS 无融合 kernel——ch5 已实录 GDN 教学实现慢 ~25×；chunk 加宽 128/256 收益 <20%，排程等价性 max|Δ|=5.4e-06，正身件默认 chunk=64 不动），9 层共 6.1 s + MoE/MLA/嵌入头/优化器本体 ~5.6 s。按 Chinchilla 配方（约 20 token/参数 ≈ 8.18B token）需 998,537 步 ≈ **148 天连续运算**（12.824 s/步口径）；超出单机可维护性，全量长训不在本机执行。一窗 30 min 降档定版 **100 步**（§7 验收③）作为「三刀协同无冲突」的管道验收凭据；超窗 ckpt 断点续跑补齐、判据不降。

## 3. 架构与 config 逐字段依据

整机：token 查表 $(B,n)\to(B,n,1024)$ → 12 层 [pre-RMSNorm → GDN | pre-RMSNorm → MLA（解耦 RoPE 内旋）] + MoE 残差块（恒宽 1024）→ 终态 RMSNorm → untied 线性出口 $(B,n,1024)\to(B,n,32000)$ → 交叉熵（目标=下一 token，fp32）。无位置查表、无任何 bias。**第一/二刀（MoE、MLA）与不动件（RMSNorm/RoPE/稠密机构）全部承 Book4 设计文档 §3，此处只列第三刀的增量字段；「三刀改装史写在 import 语句里」。**

| 字段 | 值 | 依据 | 证据等级 |
|---|---|---|---|
| `layer_types` | `[gdn,gdn,gdn,mla]×3`（0 起） | 周期约束 interval=4 + 末层全局惯例（K3 第 93 层/Qwen3.5 第 32 层同款会合）；字段名对齐 2026 机型 config 惯例（Qwen3.5/gpt-oss 同名） | 本书设计决策 + config 逆向（真机惯例） |
| GDN `h_k/h_v` | 16 / 16 | 教学几何（对称头）；**真机为 $h_v{=}2h_k$ 的 GQA 比**（Qwen3.5 32V/16QK）——偏离已声明，两种几何 gdn 件均支持且对拍过 $h_v\ne h_k$ 分支 | 本书设计决策（偏离声明） |
| GDN `d_k/d_v` | 128 / 64 | ch5 正身件默认；层参 7,393,312 与 MLA 槽量级相当，等总参补偿余量充足 | 本书设计决策 |
| GDN `conv_kernel`/`chunk` | 4 / 64 | ch5 定版（对拍 HF 双 kernel 的默认档）；chunk 是计算排程参数、不进架构账 | 官方实现对照 + 本书复验 |
| `w_expert` | **810**（938→810） | 等总参补偿（式 9.1）：$12\times 9\times 3d\times 128 = 42{,}467{,}328$ 抵注意力增量 $9\times 4{,}705{,}824=42{,}352{,}416$，净残差 −114,912（−0.028%） | 本书复算（assert 正身） |
| 其余全部字段 | 同 Book4 cand2 | d=1024/L=12/h=16/V=32000 untied/E8 top2 s1/MLA_LITE/theta 1e4/eps 1e-5/init 0.02/`first_k_dense_replace`=0 | Book4 DESIGN-409M §3 继承 |

**勘误登记（两处，随文已勘正、此处留档）**：①卷级大纲 v1.1 的 Δ 式按 MLA 层参 2,687,232 计（漏计一枚潜向量 RMSNorm 256）——正身 **2,687,488**（$q_a$/$kv_a$ 两枚各 256），故 Δattn=42,352,416（非 42,354,720）；大纲已预留「精确值以代码 assert 为正身」口径。②大纲 9.6「全全局臂以加宽 FFN 补偿」方向反——本几何 FullAttn 插槽 393,216 > GDN 插槽 332,808，全全局臂是**减** FFN（704→645）对齐（见 `hybrid_recall.py` §等参口径）。

**数值口径约定**（承 Book4 §3 全部条款）：softmax/方差统计 fp32；RMSNorm 方差 fp32；潜向量 RMSNorm eps 固定 1e-6；RoPE 复数约定（MLA 内部）；路由打分 fp32 线性；无 bias。GDN 件数值细节（l2norm 先于 $q\cdot d_k^{-0.5}$、门方向先忘后写、UT 三角求解、chunk 衰减三处）承 ch5 对拍定版，不重列。

**参数逐项账**（`hybrid409.py::hand_account` 与模块枚举双向逐位；手算通式）：

| 项 | 算式 | 参数量 |
|---|---|---|
| GDN 插槽/层 | in_qkvz $d\times(2h_kd_k{+}2h_vd_v)$ + in_ba $2dh_v$ + conv $4\times\mathrm{conv\_dim}$ + A_log/dt_bias $2h_v$ + out_proj $h_vd_vd$ | 6,291,456+32,768+20,480+32+1,048,576 = **7,393,312** ×9 |
| MLA 插槽/层 | 承 Book4（含两枚潜向量 RMSNorm 512） | **2,687,488** ×3 |
| MoE/层（w=810） | 路由门 $Ed$ + 路由专家 $E\cdot 3dw$ + 共享 $3dw$ | 8,192+19,906,560+2,488,320 = **22,403,072** ×12 |
| 层内 RMSNorm×2/层 | $2d$ | 2,048 ×12 |
| 终态 norm | $d$ | 1,024 |
| `embed_tokens` + `lm_head`（untied） | $2Vd$ | 65,536,000 |
| **总计** | | **409,000,736**（枚举=手算=定版常数三方一致） |
| 激活参（本册口径=含 lm_head、不含输入 embedding、含路由门） | GDN 层 14,868,512×9 + MLA 层 10,162,688×3 + lm_head 32,768,000 + 终态 norm | **197,073,696** |
| 激活参（non-embedding） | | 164,305,696 |
| 分类型状态账 | MLA 3 层 $960$ 元素/token（无界）+ GDN 9 层固定 $146{,}432\times 9=1{,}317{,}888$ | 交叉 $n^*=1{,}372.8$ |

对照锚：vs cand2 两刀机 409,115,648（差 −114,912=−0.028%）；vs Book3 稠密 207,119,360（等参三形态链第二、三环）。mini 复核臂（d=256/L=4/`[gdn,mla,gdn,mla]`）2,598,920 参枚举=手算逐位——层循环与账本函数的独立复核。

**激活参勘误登记（2026-10-09 三镜头审查）**：本表初版 GDN 激活层曾误写 16,868,512（14→6 位误写，恰差 2,000,000），级联总激活 215,073,696/non-emb 182,305,696 与 §6 FLOPs/GPU 三行随错——正身按同表 MLA 行同款方法复算为 14,868,512（=插槽 7,393,312+激活 MoE 7,464,960+路由门 8,192+双 RMSNorm 2,048），已回填并在 `hybrid409.py::activated_account` 立常数 assert（197,073,696/164,305,696，勾稽：总参−输入嵌入−激活参=闲置路由专家 179,159,040 逐位），notes/08 §8 同步。

**对拍口径声明（大纲钉死）**：三刀机**无整机正身可 strict 直搬**（不存在对应 HF 建模类）。正确性三层：①GDN 件层级已与 HF `Qwen3NextGatedDeltaNet` 双 kernel 对拍（ch5 gdn.py：recurrent vs HF 1.72e-08、chunk vs HF 2.24e-08、双形态互拍 1.17e-07（状态 4.47e-07）——正身 `log/book5-ch05/gdn_run1.json`）；②线性层机构与 Qwen3-Next/Kimi Linear 文献互证（同族 GDN 插槽+短卷积+输出门控 norm）；③MLA/MoE 件承 Book4 对拍资产（vs `DeepseekV2ForCausalLM` strict 直搬）。

## 4. 数据配方

- **语料**：与 Book3 ch10/Book4 ch9 **同一份** `tokens32k.bin`（OpenWebText 镜像 51,976 篇、sp-32000 逐篇编码、doc 间 EOS；训练池 56,284,791 token + eval 池 1,200,000 token 尾切独立）。三形态（207M/409M/409M）短训曲线同协议跨章可比——等参三形态对照链的读数前提。
- **切流协议/七件套校验/eval 协议**：全盘承 Book4 DESIGN-409M §4（`train_hybrid.py` 启动复检；eval=49 窗 $(8,1024)$ fp32 固定种子 20261002）。
- **全量训练的语料量**：Chinchilla 档需 8.18B token ≈ 145× 当前池——重启时以更大公开语料按同协议重切；56M 池最多支撑 6,872 步（步数断言拒跑超限）。
- **recall 消融语料**：合成 MQAR needle（免语料；探针 02 v4 定版管道，$T{=}96$/$B{=}32$/五位置 eval，seed 20261002）。

## 5. 训练配方（五件套，Book2 定版、train_409 同宗）

| 项 | 值 |
|---|---|
| 优化器 | AdamW，$\beta_1{=}0.9$、$\beta_2{=}0.95$，weight decay 0.1 |
| 梯度裁剪 | 全局范数 1.0 |
| lr 日程 | cosine：峰值 1e-3 → 终值 1e-4（峰值 10%），warmup 100 步线性 |
| 批量 | $B{=}8\times T{=}1024$ = 8,192 token/步 |
| 精度 | 训练 bf16 autocast / eval fp32；MPS 逐步 sync 计时 |
| 种子 | 20261002（全书统一） |
| 断点 | 每 eval 边界存 ckpt（模型+优化器+步号）；中断重跑同命令自动续 |
| 均衡件 | 无（softmax greedy 路由裸配置，承 cand2；升级选项见 §8） |
| recall 臂配方 | AdamW 同款+lr 3e-3+warmup 50+cosine 至 0（P18 管道原样升格） |

## 6. 预算表（6ND 口径：FLOPs = 6 × 激活参 197,073,696 × tokens）

| 档 | tokens | FLOPs | 机器（口径） | wall-clock | 证据等级 |
|---|---|---|---|---|---|
| Chinchilla（20 tok/param） | 8.18B | 9.7e18 | 本机 MPS bf16（实测 12.824 s/步） | **≈148 天** | 本书实验外推 |
| 整合短训（已执行） | 0.82M | — | 本机 MPS bf16（100 步独占窗） | wall 28.6 min，eval 7.80→7.38 | 本书实验 |
| Chinchilla | 8.18B | 9.7e18 | 1×A100-80GB（312 TFLOPS × 40% MFU 假设） | ≈21.5 小时 | 本书估算 |
| Chinchilla | 8.18B | 9.7e18 | 1×H100（989 TFLOPS × 40%） | ≈6.8 小时 | 本书估算 |
| Chinchilla | 8.18B | 9.7e18 | 8×H100 @40% | ≈51 分钟 | 本书估算 |

估算口径声明：GPU 行按 bf16 dense 峰值 × MFU=40% 假设外推；三刀机在小尺度实际 MFU 可能显著低于 40%——GDN chunkwise 需 FLA 类融合 kernel（MPS 无；GPU 上有、但本书未实测）、逐专家 Python 循环 launch-bound（Book4 ch1 实测派发开销 1.44×），估时偏乐观、量级可靠。显存门槛：AdamW fp32 全套状态 ≈16 B/参数 ≈ 6.5 GiB + 激活（本机 MPS 峰值 7.77 GB 含激活与 autocast 缓存）——单张 24GB 消费卡可重启 Chinchilla 档，**门槛是时间不是显存**。

## 7. 验收标准（可机器判定；①-④ 已执行）

1. **参数逐位**：`python hybrid409.py` 自测——总参 409,000,736（枚举=手算=定版三方一致）；GDN 层 7,393,312×9、MLA 层 2,687,488×3、层模式逐层核（isinstance 对 `layer_types`）；mini 臂 2,598,920。已执行 PASS（8.2 s CPU，`hybrid409_selftest_run1.json`）。
2. **初始 loss 数学自检**：随机 token 下实测 10.6151 对预言 $\ln V+d\sigma^2/2=10.5783$（|Δ|=0.037 < 0.06 判据带内；logit 方差 0.4097 对预言 0.4096）。
3. **冒烟健康**：`--task smoke` 20 步——初始 10.5916（带内）、无 NaN、有限下降（train→8.7265、eval@20 8.5287）。已执行 PASS（11.715 s/步）。
4. **整合短训**：`--task integrated` 100 步（一窗降档定版、留痕）——无 NaN、eval 下降（@50 7.7997→@100 **7.3838**）、wall 28.6 min ≤30 min、峰值 7.77 GB。已执行 PASS。
5. **（重启后）全量训练**：语料按 §4 重切 ≥8.18B token，`--task integrated --steps 998537 --out-name full1`（ckpt 续跑承载；eval 边界建议放宽至 1000 步）。
6. **（附）recall 等参三臂**：`python hybrid_recall.py --out-name run1`——判据 J1-J3/分支 B1-B3 预注册（文件头全文）；已执行两轮（6000 正档+10000 补档）读数：全全局 loss 2.061/远端 0.539、3:1 与全线性钉值先验平台 ln 64=4.159（负结论+规模/预算绑定声明——正确表述边界见第 9 章正文 9.6）。
7. **（附）真权重消费点**：`python qwen35_consume.py`——Qwen3.5-4B 32 层 isinstance 对账全中、分类型账 n*=804、两条生成样例。已执行 PASS（权重在盘 6.2 s）。

## 8. 风险与预案

| 风险 | 表现 | 预案 |
|---|---|---|
| 初始 loss 异常 | 偏离 ln V+ds²/2 预言 | 核对 initializer_range=0.02；GDN 件 conv/A_log/dt_bias 保留构造器初始化（自测②已含） |
| bf16 数值病（NaN/尖峰） | GDN 块内 exp/三角求解精度 | 数值口径不放宽（fp32 统计）；g 的 fp32 计算（ch5 K6 坑）；复发降 fp32 求因 |
| 步时热漂移 | 持续负载 s/步上漂（本窗 10.1→15.3） | 独占窗口计时；预算以独窗稳态为锚；漂移超 20% 换窗重测 |
| MoE 负载崩塌（全量期） | 专家垄断/死专家 | 升级均衡件（Book4 备件零结构改动）：aux loss 钩子或 noaux bias |
| GDN 训不动/平台 | recall 类任务钉边缘分布 | 已知地界（值先验平台，§7⑥）——区分「管道失败」与「预算/规模绑定」，预注册分支判读 |
| 长训中断 | 进度丢失 | ckpt 边界自动续跑；nohup detached 启动 |
| 数据不足 8.18B | 步数断言拒跑 | 更大语料按 §4 重切（七件套校验必须重过） |
| 结构对拍需求 | 无整机正身 | 按 §3 对拍口径三层证据走；勿尝试给三刀机找不存在的 strict 正身 |
| 头几何换档（重启选项） | 改 $h_v{=}2h_k$ 对齐真机 GQA 比 | 参数账重算（GDN 层参随 conv_dim/out_proj 变）、$w$ 重扫；与本定版账不一致属预期，须同步刷新 §3/§7① |

## 9. 重启 checklist（十步）

1. 环境与文件：按 §1 依赖清单备齐 Book4 ch09 / 本册 ch05 / Book4 00-feasibility / 本册 00-feasibility 兄弟目录与依赖库；`python hybrid409.py` 自测通过（409,000,736 逐位）。
2. 数据：直用 `log/book3-ch10/` 三件套（校验全过），或按 §4 换更大语料重切。
3. 验收①②：自测 PASS（参数+初始 loss 带内）。
4. 验收③：`--task smoke`（初始 ≈10.58、有限、下降）。
5. 验收④：`--task integrated` 100 步（无 NaN、eval 下降）——与已执行读数同 eval 协议可接续。
6. 决策点：头几何换档（$h_v{=}2h_k$ 真机比）与均衡件启用（§8）——单变量对照、逐项刷新账。
7. 全量训练：按 §6 定档 `--task integrated --steps 998537 --out-name full1`（语料先重切 ≥8.18B；批量/lr/eval 边界随机器重定并写入元数据）。
8. 曲线入档：与 207M（Book3）/两刀 409M（Book4）短训曲线同表收束——等参三形态链的完整读数。
9. （附）recall 消融升档：更长预算 `--steps <N>` 自 ckpt 续跑；多种子仅作者特批。
10. 消费线登记：产物权重按 §10 交接。

## 10. 消费线衔接声明（防断链）

本项目的全量长训暂缓（§2）。**消费线默认对象不变**：后续推理册 mini 引擎的默认服务对象仍为 Book3 终态稠密+GQA 小模型（Book3 DESIGN §10 登记，单轨三形态口径不变）。三刀整机在消费线上的身份是**账本算例与选做扩展**：分类型 KV/状态账（960·n 无界 + 1,317,888 固定、$n^*\approx1{,}373$）与等参三形态链（207M/409M/409M——无界斜率 12,288→3,840→960）作为后续两册混合缓存布局与显存账的现成算例；mini 引擎接入三刀机属选做扩展（混合层「无界账本+定长记忆」同池共管的机制展开归后续册，本册只登记接口事实：MLA 缓存布局 $(B,1,n,d_c)$+$(B,1,n,d_h^R)$ 单头对承 Book4、GDN 状态 $(B,h_v,d_k,d_v)$ 每请求定长一份 + 卷积滞留 $(B,\mathrm{conv\_dim},k{-}1)$）。重启完成后，全量权重沿同一登记交接。

**B5-8 文档锚（本文档即锚）**：三刀机的分类型状态账与层模式表，供 Book7「混合 KV 布局」与 Book8「服务侧缓存联合管理」各章直接引用为本册算例（正文零占位——第 9 章不重复展开）。

## 11. 参考与文献锚

- 第三刀组件：Gated DeltaNet arXiv:2412.06464 v3（官方仓 NC 禁抄——实现参照白名单 transformers/FLA/MIT 系）；Qwen3-Next README（3:1 布局直引，Apache-2.0）；Kimi Linear arXiv:2510.26692 v2（3:1 等参消融 Table 1、层式优于头式 §4）。
- 底座与不动件：Book4 `DESIGN-409M.md`（两刀先例——本文件 §3 增量字段的母本）；DeepSeekMoE arXiv:2401.06066；DSV2（MLA）arXiv:2405.04434 v5；RMSNorm/SwiGLU/RoPE 锚承 Book4 §11。
- 3:1 证据链：DeltaNet 并行化 arXiv:2406.06484 v6（hybrid abstract 句）；Mamba-2-Hybrid arXiv:2406.07887（6:1 净胜）；Based arXiv:2402.18668 v2（状态×召回权衡）；Repeat After Me arXiv:2402.01032 v2（固定状态定理——保留全局层的根据）。
- 真机锚：Kimi K3 官方技术报告 arXiv:2607.24653（本地 PDF）；Qwen3.5 config（Apache-2.0，snapshot 851bf6e8）；GLM-5 arXiv:2602.15763 v2（§2.1.2 变体消融——选择性对照并排标注）。
- 对照实现：HF transformers 5.18.0 `models/qwen3_next/`（GDN 对拍正身）、`models/qwen3_5/`（真权重消费点）。
- 本机实测底数：`log/book5-ch09/`（hybrid409_selftest / trainhyb_{smoke,integrated} / chunk_bench / recall3_* / qwen35_consume 全套 JSON，随行种子与设备口径）。

---

*本文档所有「实测」数字来自 2026-10 本书工作区的实际运行（设备/精度/种子/产物 JSON 路径随行标注）；引用他人数字处已标证据等级。重启时请以自己机器的实测数字刷新预算表，勿直接抄用。*
