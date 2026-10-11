# 从模型到 vLLM（From Models to vLLM）—— 系列教学丛书

这是一套八册的中文教学丛书，走一条完整的旅程：从 2017 年的 Transformer 原论文起步——零魔改、忠于原文，把原典拆到读者能凭书稿重画整机的程度；此后每一册都以上一册的已知架构为底座，用「补什么新知识、改什么架构」的方式层层递进——先看架构不动、训练范式分化的预训练革命，再把 GPT-2 逐插槽改装成 LLaMA 式现代骨架，然后换上混合专家（MoE）与多头潜在注意力（MLA）进入旗舰世代，又在注意力本体上动刀迎来 2026 年的混合架构大收敛；模型侧走完后切换系统视角，讲引擎无关的推理物理学，最后两册扎进 vLLM V1 源码，沿一条请求的生命周期把引擎读穿。每个论断都双锚定：先对论文原文（golden truth），再对本地浅克隆仓库的真实代码。一条单轨代码主线贯穿八册——Book2 从零造出 GPT-2 124M，Book3 把它改装成 LLaMA 式 215M，Book4 换上 MoE 与 MLA、Book5 选做换成 3:1 混合（演化线）；Book6 的 mini 推理引擎把 Book3 终态的稠密小模型当默认服务对象（手写引擎不承载 MoE 与混合 KV），Book7 的选做大作业再把单轨自训模型注册进 vLLM，闭环合拢。

## 这套书写什么

本丛书的主线不是论文合集，而是「构造的逻辑」：要造出 2017 年那台翻译机需要什么？要把它升级到今天需要什么？要让它便宜、快速地服务亿级请求又需要什么？八册正好回答这三个层次——

```
2014 ─ 2017 ─── 2018-2022 ─── 2023 ─────── 2023-2025 ─────── 2025-2026 ──── 系统视角 ──── 工程落地
seq2seq  Transformer  预训练范式    现代骨架(组件化)   规模与效率(MoE/MLA)  混合架构大收敛   推理学理论    vLLM V1
 瓶颈     原典        (BERT/GPT/T5)  LLaMA插槽化       旗舰世代           2026旗舰主线    (引擎无关)   源码级
  │         │            │              │                 │                  │              │           │
 Book1    Book1        Book2          Book3            Book4              Book5          Book6     Book7-8
                    (GPT-2单轨起点) (GPT-2改装LLaMA)  (FFN→MoE,GQA→MLA)  (注意力本体手术)  (mini引擎) (主链路+生产)
```

调研结论：中英文市场均无同类成书。本丛书的差异化在四处——

1. **全链条**：「Transformer → 架构谱系 → 推理理论 → vLLM V1 源码级」全链条中文系统性成书；
2. **双锚定**：论文 golden truth + 本地源码。vLLM 两册的阅读锚冻结在 v0.30.0（commit ced6857），实验观察落在运行栈 vllm 0.27.1 + vllm-ascend 0.27.1rc1，差异以版本框注处理、互证不混引；
3. **V1/MRV2 心智**：市面资料仍停留在 V0/nanovllm，本丛书以 V1 不变量为纲，并把 Model Runner V2（MRV2）重架构纳入正文；
4. **统一叙事 + 单轨代码**：每项技术按「问题→方案→落地→实证」的层次推进——隐式缝合于行文，不作标签；代码主线 GPT-2 → LLaMA 式 → MoE/MLA → 3:1 混合（选做）步步演化，Book3 终态的稠密小模型由 mini 引擎默认服务，单轨自训模型再经 Book7 选做大作业注册进 vLLM，读者全程只维护一套心智模型。

硬件口径：Book1-5 的实验读数产自 Mac（Apple Silicon / MPS / vllm-metal），为定稿历史事实、存稿不改；Book6 起实验主线切换为昇腾 NPU（Atlas 910B3，torch-npu / vllm-ascend / CANN 9.1.0）。

## 为谁而写

读者画像：具备 AI 基础的计算机本科生——线性代数与矩阵运算、高等数学、反向传播、会用 PyTorch、计算机基础。不需要任何数学前导册：每章自含「最小背景知识」栏，论证依赖的前置概念一定先搭到「架构完整可见、机制可想象」的程度才使用。每册 10-12 章、约一学期课程，八册合计约八个学期的完整旅程。

阅读顺序：Book1→2→3 是递进主干（同一台机器被逐册改装）；Book4 建立在 Book3 的骨架上，Book5 建立在 Book3 的外推线与 Book4 的 MoE/MLA 上；Book6 冻结架构、切换系统视角，前置 Book2-5；Book7 硬依赖 Book6、软依赖 Book4-5，Book8 在 Book7 之上续讲，另需 Book6 的分布式与 PD 分离理论与 Book4-5 的混合架构/MTP/MoE。只想读推理与引擎部分的读者可以从 Book6 进入，但会错过前五册攒下的架构直觉——Book7 的 attention 后端与混合 KV 布局会直接用到它们。

## 八册总览

| # | 书名 | 一句话定位 | 状态 | 入口 |
|---|------|-----------|------|------|
| 1 | 《Transformer 原典：2017 原架构完全拆解》 | 只讲原论文，零魔改，立叙事模板 | 定稿 2026-10-02 | [进入](book1-transformer-original/README.md) |
| 2 | 《预训练革命：BERT、GPT 与 Scaling Laws》 | 架构几乎不动，训练范式革命 | 定稿 2026-10-03 | [进入](book2-pretraining-revolution/README.md) |
| 3 | 《现代开源骨架：LLaMA 与组件化时代》 | 组件学与插槽化骨架 | 定稿 2026-10-04 | [进入](book3-modern-open-source-skeleton/README.md) |
| 4 | 《规模与效率：MoE、MLA 与旗舰世代》 | 效率悖论：总参数涨、单 token 成本降 | 定稿 2026-10-05 | [进入](book4-scale-and-efficiency/README.md) |
| 5 | 《前沿混合架构：稀疏、线性注意力与 2026 旗舰》 | 注意力本体动手术；2026 旗舰进正文主线 | 定稿 2026-10-09 | [进入](book5-frontier-hybrid-architectures/README.md) |
| 6 | 《LLM 推理系统导论：从自回归数学到服务算法》 | 推理的物理学（引擎无关） | 定稿 2026-10-09 | [进入](book6-inference-systems-intro/README.md) |
| 7 | 《vLLM V1 源码剖析（上）：引擎主链路》 | 沿一条请求的生命周期精读源码 | 定稿 2026-10-11 | [进入](book7-vllm-v1-source-code-vol1/README.md) |
| 8 | 《vLLM V1 源码剖析（下）：生产进阶与服务化》 | 专题横切 + 生产化 + 毕业设计 | 撰写中 | — |

Book8 规划：混合架构服务（hybrid KV cache manager 机制深讲）、投机解码工程、结构化输出、PD 分离与 KV 传输、可观测性等专题横切，加平台层（vllm-ascend）与引擎选型，毕业设计让一条请求穿越八册知识全链回顾。

## 编年史（全系列 · 七册合并去重）

本表收录 Book1-7 七册定稿稿中实质性讲解过的对象，按提出时间从早到晚排列（时间口径：论文取 arXiv v1 提交月；无 arXiv 者取会议/官方发布月；模型/系统取官方首次发布月；带 ※ 者见表下注；括注为先行时点或口径说明；同一对象多册共讲合并为一行，「册」列链至各册 README）。贡献列字数按汉字计（拉丁字母/数字/标点/空格不计），单条不逾 40 字。Book8 定稿后条目增补至此。

| 时间 | 名称 | 贡献 | 册 |
|---|---|---|---|
| 1962-04 | 操作系统分页（Atlas 一级存储系统，虚拟内存起源） | 把地址空间切成定长页、页表换帧号，根除变长分配的外部碎片——六十年后 KV 分页照搬的原型药方 | [6](book6-inference-systems-intro/README.md) |
| 1976※ | 束搜索（beam search，含 GNMT 长度惩罚式） | 同步扩展 k 条候选、按累积对数概率加长度惩罚择优，治自回归解码偏爱短句的偏置 | [1](book1-transformer-original/README.md) |
| 2009※ | MessagePack（msgpack 二进制序列化） | 以类型字节打头的紧凑二进制替换 JSON 文本，V1 线协议跨进程消息的中间一环 | [7](book7-vllm-v1-source-code-vol1/README.md) |
| 2009-04（Berkeley 报告 2008-10 在先） | Roofline 模型（屋顶线模型，Williams-Waterman-Patterson） | 以算力屋顶与带宽斜坡的单图模型预判负载快慢，「卡在算力还是访存」一图可判 | [6](book6-inference-systems-intro/README.md) |
| 2010※ | ZeroMQ（ZMQ） | 把 socket 语义抽象成免 broker 的可组合消息模式库，组出零中间件的进程拓扑 | [7](book7-vllm-v1-source-code-vol1/README.md) |
| 2012-03（正身 IEEE ICASSP） | WordPiece（似然增益子词分词，Schuster & Nakajima） | 把子词合并准则从频率换成似然增益，后成 BERT 词表地基，失败整词回退 UNK | [2](book2-pretraining-revolution/README.md) |
| 2013-01 | word2vec（skip-gram/CBOW，Mikolov et al.） | 用看窗口猜词的自造监督从无标注语料学词向量，16 亿词语料上语义类比精度过半 | [2](book2-pretraining-revolution/README.md) |
| 2013-10 | word2vec 负采样与子采样（Mikolov et al.） | 把全词表 softmax 偷换成 k 道是非题并丢弃高频词，训练再快数倍、低频词表示更准 | [2](book2-pretraining-revolution/README.md) |
| 2014※ | Dropout（随机失活，Srivastava et al.） | 训练态随机置零部分激活并逆缩放保期望，每步等效训练一个随机子网，扛住过拟合 | [1](book1-transformer-original/README.md) |
| 2014-06 | RNN 编码器-解码器与 GRU（Cho et al.） | 独立给出编码器-解码器概率框架并发明 GRU，可学阀门让循环网络长句存得住信息 | [1](book1-transformer-original/README.md) |
| 2014-09 | seq2seq（序列到序列学习，Sutskever et al.） | 两塔 LSTM 把变长翻译改写成逐词抽签，整句压进固定维向量箱子，长句必崩 | [1](book1-transformer-original/README.md) |
| 2014-09 | Bahdanau 注意力（加性注意力/软对齐） | 解码器每写一词按对齐权重回望全句自取信息，软寻址拆掉两塔独木桥，长句不再崩 | [1](book1-transformer-original/README.md) |
| 2014-09 | 固定向量瓶颈实证（Cho et al. 2014b） | 用 BLEU 随句长的下降曲线实证：短句尚可、长句急剧崩坏，瓶颈从直觉变定量证据 | [1](book1-transformer-original/README.md) |
| 2014-10 | GloVe（全局词向量，Pennington et al.） | 直接拟合全局共现统计而非逐窗采样，单机数十分钟吃下 60 亿词、先画出规模曲线 | [2](book2-pretraining-revolution/README.md) |
| 2014-12 | Adam 优化器（Kingma & Ba） | 以梯度一二阶矩指数滑动平均逐参数调步长，成 Transformer 点火标配 | [1](book1-transformer-original/README.md) |
| 2015-08 | Luong 注意力打分函数族（dot/general/concat 与 global/local） | 把打分函数当设计空间系统消融，零参数 dot 内积打赢带参数形式，为缩放点积扫清前置疑问 | [1](book1-transformer-original/README.md) |
| 2015-08 | BPE 子词切分（字节对编码，Sennrich et al.） | 把字节对合并压缩改造成子词切分，高频整存低频切碎，小词表压低未登录率与参数账单 | [1](book1-transformer-original/README.md)[2](book2-pretraining-revolution/README.md) |
| 2015-12 | ResNet（残差学习，He et al.） | 恒等捷径让深层网络只学增量修正，152 层可训不再退化，残差成为深度的许可证 | [1](book1-transformer-original/README.md) |
| 2015-12 | label smoothing（标签平滑，Szegedy et al.） | 对错误答案均匀留 ε 概率再算交叉熵，伤困惑度换 BLEU 与泛化 | [1](book1-transformer-original/README.md) |
| 2016-04 | 梯度检查点（Training Deep Nets with Sublinear Memory Cost，Chen 等） | 只存块边界、反向重算块内，激活显存压到平方根量级，以约三成时间换空间 | [2](book2-pretraining-revolution/README.md) |
| 2016-07 | LayerNorm（层归一化，Ba et al.） | 把归一化统计量从 batch 维搬到单样本层内，训练推理行为一致，天然适配变长序列 | [1](book1-transformer-original/README.md)[3](book3-modern-open-source-skeleton/README.md) |
| 2016-08（正式发表 EACL 2017） | 输入输出嵌入绑定（weight tying，Press & Wolf） | 发现输出投影矩阵本身即有效词嵌入，主张绑定输入输出——原典三处共享省 37.9M 参数 | [1](book1-transformer-original/README.md) |
| 2016-09 | GNMT（Google 神经机器翻译系统，Wu et al.） | 残差 LSTM 堆到 96 卡推到循环路线天花板，解码 GPU 反比 CPU 慢、串行墙暴露 | [1](book1-transformer-original/README.md) |
| 2016-12 | GLU（门控线性单元，Dauphin 等） | 用数据自学的 sigmoid 门逐通道调制特征、乘性跳连稳梯度，四分之一硬件追平 LSTM | [3](book3-modern-open-source-skeleton/README.md) |
| 2017-01 | 稀疏门控 MoE（Sparsely-Gated Mixture-of-Experts，Shazeer） | 稀疏门控让每 token 只点亮少数专家，容量涨千倍计算仅微损，MoE 层做到 137B | [4](book4-scale-and-efficiency/README.md) |
| 2017-05 | ConvS2S（卷积序列到序列，Gehring et al.） | 以卷积换循环拿下层内并行，K40 上快 9.3 倍，但有效感受野约 25 词——并行与远距两难 | [1](book1-transformer-original/README.md) |
| 2017-06 | Transformer（《Attention Is All You Need》，Vaswani et al.） | 自注意力拆掉循环先验，任意两词一步可达、层内串行 O(1)，三分之一算力多 2 BLEU | [1](book1-transformer-original/README.md)[5](book5-frontier-hybrid-architectures/README.md) |
| 2017-06（命名出自 t2t 代码） | Noam 学习率调度（Transformer lr 公式，warmup 调度） | 学习率先线性升温再按步数平方根倒数衰减，驯服 post-LN 初期梯度失衡 | [1](book1-transformer-original/README.md) |
| 2017-06 | Tensor2Tensor（t2t） | 官方复现库把论文配方固化成代码，最早一档与论文逐点一致，后续版本漂移成复现课教材 | [1](book1-transformer-original/README.md) |
| 2017-10 | 混合精度训练（Mixed Precision Training，Micikevicius 等） | fp16 算前向反向、fp32 主权重与损失缩放护航，同等精度下训练提速数倍 | [2](book2-pretraining-revolution/README.md) |
| 2018-01 | ULMFiT（通用语言模型微调，Howard & Ruder） | 给 LSTM 交付分层微调配方四件套，百例标注追上从零百倍数据，配方与架构平起平坐 | [2](book2-pretraining-revolution/README.md) |
| 2018-02 | ELMo（深度上下文相关词表示，Peters 等） | 双向 LSTM 拼出上下文相关词表示，多义词按场合变形、六任务全线提升 | [2](book2-pretraining-revolution/README.md) |
| 2018-04※ | The Annotated Transformer | 逐行注释实现成社区复现母本，却为简洁把 post-LN 换 pre-norm——照抄学原架构第一行就错 | [1](book1-transformer-original/README.md) |
| 2018-04 | 子词正则化与 Unigram LM（Subword Regularization，Kudo） | 从大种子词表按似然损失剪枝出 Unigram 分词，切分多样性成为训练噪声、翻译再进半分 | [2](book2-pretraining-revolution/README.md) |
| 2018-05（k=40 为 GPT-2 取值） | top-k 采样（Fan et al.，截断归一采样） | 生成时只保留概率最高的 k 个候选再归一采样，截掉长尾防退化，成采样管线默认档 | [6](book6-inference-systems-intro/README.md) |
| 2018-06（无 arXiv，日期转述级） | GPT-1（生成式预训练 Transformer，Radford 等） | 以语言模型预训练解码器半塔、判别式微调加辅助损失，十二个数据集九项 SOTA | [2](book2-pretraining-revolution/README.md) |
| 2018-07（博客级来源） | 「NLP 的 ImageNet 时刻」（2018-07-12，Ruder 博客） | 预言一年内从业者将下载预训练语言模型而非词向量，口号事后命中 | [2](book2-pretraining-revolution/README.md) |
| 2018-08 | SentencePiece（语言无关无损子词分词器，Kudo & Richardson） | 把空格转写成 ▁ 元符号实现语言无关的无损分词，免预分词直接从原始句子训练 | [2](book2-pretraining-revolution/README.md) |
| 2018-09（多源二级定月） | CUDA Graph（图执行 capture/replay） | 整串 kernel 录一次后整体回放，静态约束换发射税归零（NPUGraph 实测 11.4×） | [6](book6-inference-systems-intro/README.md)[7](book7-vllm-v1-source-code-vol1/README.md) |
| 2018-10 | BERT（Devlin 等） | 以遮词填空加句对判断在编码器半塔学双向表示，预训练一次微调 N 次、十一项任务刷新 | [2](book2-pretraining-revolution/README.md) |
| 2018-11（首版 PyPI 月） | HF Transformers 库（前称 pytorch-pretrained-bert） | 把各家权重与 config 收进一个库一套 API，成官方对拍与微调生态的地基 | [2](book2-pretraining-revolution/README.md)[7](book7-vllm-v1-source-code-vol1/README.md) |
| 2019-02 | GPT-2（Language Models are Unsupervised Multitask Learners，Radford 等） | 改装修配 40GB WebText 训到 1.5B，把任务折叠进序列押注零样本接口 | [2](book2-pretraining-revolution/README.md)[3](book3-modern-open-source-skeleton/README.md) |
| 2019-02 | byte-level BPE（字节级 BPE，GPT-2 §2.2） | 把基础词表压到 256 字节再限片段内合并，任意字符串可编码、UNK 归零 | [2](book2-pretraining-revolution/README.md) |
| 2019-04（以仓创建日锚定） | OpenWebText（WebText 开源复刻语料） | 社区按 Reddit 三赞口径复刻 WebText 管道开源，未发布的 40GB 语料得以复现 | [2](book2-pretraining-revolution/README.md) |
| 2019-04 | 核采样（top-p / nucleus sampling，Holtzman et al.） | 按累计概率质量 p 取最小候选集、名额随分布自适应，贪心复读率 73.66%→0.36% | [6](book6-inference-systems-intro/README.md) |
| 2019-07 | RoBERTa（Liu 等） | 架构不动只拧配方——动态掩码、去 NSP、十倍数据，追平此后模型并翻案 NSP | [2](book2-pretraining-revolution/README.md) |
| 2019-09 | 重复惩罚（repetition penalty，CTRL 论文 Keskar et al.） | 正 logit 除罚、负 logit 乘罚的乘式历史惩罚，温和压制复读、沿用最广 | [6](book6-inference-systems-intro/README.md) |
| 2019-10 | Transformers without Tears（pre-Norm+小初始化免 warm-up） | 论证 pre-norm 残差配小初始化可免 warmup，塔顶补归一层即 ln_f 出处 | [2](book2-pretraining-revolution/README.md) |
| 2019-10 | ZeRO（零冗余优化器，Rajbhandari 等） | 把训练显存算成每参数 16 字节的地板并多卡切分，1.5B 模型装进 24GB | [2](book2-pretraining-revolution/README.md) |
| 2019-10 | T5（Text-to-Text Transfer Transformer，Raffel 等） | 把一切任务写成文本进文本出、任务前缀统一接口，逐格消融审遍目标与架构 | [2](book2-pretraining-revolution/README.md) |
| 2019-10 | RMSNorm（均方根层归一化） | 砍掉重中心化只留重缩放，计算税减半质量不掉，经 LLaMA 定型为默认零件 | [3](book3-modern-open-source-skeleton/README.md) |
| 2019-11 | MQA（多查询注意力，Shazeer） | 全部查询头共享单个 KV 头，推理显存与带宽随头数直除，换数倍解码增速 | [3](book3-modern-open-source-skeleton/README.md) |
| 2019-11 | KV cache（键值缓存；首见 Shazeer 1911.02150 §2.4） | 利用因果性冻结已算键值、每步只算新 token，生成计算从平方降回线性 | [3](book3-modern-open-source-skeleton/README.md) |
| 2020-01 | Kaplan 缩放定律（Scaling Laws for Neural Language Models，Kaplan 等） | 跨六个数量级拟合出损失幂律，预算下参数数据按 0.73/0.27 分配、大模型优先 | [2](book2-pretraining-revolution/README.md) |
| 2020-02 | post-LN/pre-LN 梯度分析（On Layer Normalization in the Transformer Architecture，Xiong 等） | 证明 post-LN 初始梯度随深度增长、pre-LN 不增，warmup 依赖的病根定位 | [2](book2-pretraining-revolution/README.md) |
| 2020-02 | SwiGLU（GLU 变体改进 Transformer，Shazeer） | 把激活函数货架搬进门控支路逐一比选，SwiGLU 胜出并配 8/3·d 缩维 | [3](book3-modern-open-source-skeleton/README.md) |
| 2020-05 | GPT-3（Language Models are Few-Shot Learners，Brown 等） | 冻结架构只拧规模到 175B，few-shot 提示即任务接口，上下文学习成范式级现象 | [2](book2-pretraining-revolution/README.md)[3](book3-modern-open-source-skeleton/README.md) |
| 2020-06 | GShard（条件计算规模化 MoE） | 把专家群隔层搬进 Transformer 换 FFN，立容量因子与辅助损失原型 | [4](book4-scale-and-efficiency/README.md) |
| 2020-06 | 线性注意力（Linear Attention，Katharopoulos 等） | 用 elu+1 核换 softmax、以结合律先结算成固定状态，免物化 n×n 打分表 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2020-10 | QK-Norm（查询键归一化，Henry 等） | 沿头维归一化 Q/K 使点积变有界余弦，压住 logits 防 softmax 饱和、防 spike | [4](book4-scale-and-efficiency/README.md) |
| 2020-10 | ViT（Vision Transformer，Dosovitskiy 等） | 把图切成 16×16 patch 当词元喂标准编码器，图像变回序列问题、比肩卷积更省算力 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2021-01 | Switch Transformers（Top-1 稀疏 MoE） | 砍到 top-1 路由、路由器锁 FP32 稳训练，同 FLOPs 参数扩一个量级到 1.571T | [4](book4-scale-and-efficiency/README.md) |
| 2021-02（博客先行论文 28 天） | RoPE 博客三部曲（苏剑林，kexue.fm） | 先于论文以中文博客草绘旋转编码，泰勒展开透视正弦编码的相对性内核 | [3](book3-modern-open-source-skeleton/README.md) |
| 2021-02 | 快速权重编程器（Fast Weight Programmers，delta rule 正身） | 把记忆更新写成先读后写的 delta 规则，键值可覆写可删除，治纯加法记忆混叠 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2021-02（v1 提交 02-26、滚入 3 月编号；模型 2021-01 先行） | CLIP（对比图文预训练） | 四亿图文对对比配对预训练、双塔共享嵌入空间，换零样本迁移 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2021-04 | RoPE（旋转位置编码，RoFormer） | 把位置变成 Q/K 上的旋转角、内积只含相对距离，拆掉查表硬墙奠定长上下文地基 | [3](book3-modern-open-source-skeleton/README.md) |
| 2021-08 | ALiBi（线性偏置注意力，Train Short Test Long） | 不学位置向量、在打分上按距离加线性偏置，1k 训练免微调外推更长上下文 | [3](book3-modern-open-source-skeleton/README.md) |
| 2021-10（v1 提交 10-31、滚入 11 月编号） | S4（结构化状态空间模型） | 把控制论状态方程结构化离散沿序列演化，HiPPO 初始化驯服长程依赖 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2022-02 | ST-MoE（稳定可迁移稀疏专家，router z-loss 首发） | 以 router z-loss 惩罚打分量级治 MoE 低精度不稳，稳定率 3/6 提到 3/3 | [4](book4-scale-and-efficiency/README.md) |
| 2022-03 | Chinchilla（Training Compute-Optimal LLMs，Hoffmann 等） | 三种独立方法判 GPT-3 每参数 token 太少、N:D 等比增长，同算力小模型多数据反超 | [2](book2-pretraining-revolution/README.md)[3](book3-modern-open-source-skeleton/README.md) |
| 2022-03 | InstructGPT（基于人类反馈的指令对齐） | 人类偏好数据加 RLHF 对齐基础模型，会续写的机器变成会听话的助手 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2022-03 | Xia et al. 投机解码（原名 Generalized Aggressive Decoding） | 以大模型并行校验小模型草稿，最早撞自回归串行墙——三次独立发明里最早一家 | [6](book6-inference-systems-intro/README.md) |
| 2022-04 | PaLM（Scaling Language Modeling with Pathways，Chowdhery 等） | 540B 训练实录约 20 次损失尖峰且裁剪在场，归因特定数据批遇特定参数状态 | [2](book2-pretraining-revolution/README.md) |
| 2022-04 | 架构×预训练目标受控对照（Wang 等，BigScience） | 同数据同算力跑架构×目标受控对照：免微调 decoder-only 胜、指令微调双塔复活 | [2](book2-pretraining-revolution/README.md) |
| 2022-05 | OPT-175B（Open Pre-trained Transformer，含逐日训练 Logbook） | 175B 连 logbook 一起公开：35 次手动重启、缩放系数崩零病程全录 | [2](book2-pretraining-revolution/README.md) |
| 2022-05 | FlashAttention（FA1，IO 感知精确注意力） | online softmax 分块驻留片上，n² 中间量不进 HBM，精确注意力 IO 降一个量级 | [6](book6-inference-systems-intro/README.md) |
| 2022-06 | 涌现能力（Emergent Abilities，Wei 等） | 给「小模型没有、大模型突然有」的能力立定义列清单，百亿级阈值叙事成立 | [2](book2-pretraining-revolution/README.md) |
| 2022-07（无 arXiv，正身 USENIX OSDI） | Orca（迭代级调度，OSDI 2022） | 把调度粒度从批降到迭代步、异长可拼批，时隙浪费归零，吞吐 36.9× | [6](book6-inference-systems-intro/README.md) |
| 2022-09 | FP8 格式（E4M3/E5M2，Micikevicius 等） | 联合定义 E4M3/E5M2 分管前向与梯度，8 位浮点进训练循环，验证到 175B | [4](book4-scale-and-efficiency/README.md) |
| 2022-10 | GPTQ（训练后量化，Frantar et al.） | 沿 Hessian 逆把量化误差摊给不敏感权重对消，175B 四 GPU 小时压完 | [6](book6-inference-systems-intro/README.md) |
| 2022-11 | Pope et al. 大模型高效推理（Efficiently Scaling Transformer Inference） | 先立算力/访存/通信三本解析账再谈并行策略，2N 速算式与 MFU 定义的正身 | [6](book6-inference-systems-intro/README.md) |
| 2022-11 | 投机解码验证器（speculative sampling，Leviathan et al.） | 拒绝采样验证器保证输出分布逐位不变，T5-XXL 2-3× 无损加速 | [6](book6-inference-systems-intro/README.md) |
| 2022-11 | SmoothQuant（量化难度迁移，Xiao et al.） | 用等效缩放把激活离群搬进权重侧，W8A8 全整数量化近无损、INT8 提速近 2× | [6](book6-inference-systems-intro/README.md) |
| 2023-02 | LLaMA 1（Meta 开源基座第一代） | 以公开数据重选型、旧零件组装公开权重模型，约四分之一算力多数任务胜 GPT-3 | [3](book3-modern-open-source-skeleton/README.md) |
| 2023-03（口径=仓库创建） | llama.cpp（GGUF 量化生态） | 以纯 C/C++ 重写推理自成一系量化档位，划出单机运行态与服务引擎的生态分层 | [7](book7-vllm-v1-source-code-vol1/README.md) |
| 2023-03 | OpenAI Chat Completions API | 把对话定型为 messages 进、delta 流出的 HTTP 协议并成事实标准 | [7](book7-vllm-v1-source-code-vol1/README.md) |
| 2023-03（口径=PyTorch 2.0 GA） | torch.compile（PyTorch 2.0 编译栈） | Dynamo 追踪加 Inductor 编译，把逐算子 Python 开销变编译期决策 | [7](book7-vllm-v1-source-code-vol1/README.md) |
| 2023-04 | 涌现是幻象吗（Are Emergent Abilities a Mirage?，Schaeffer 等） | 指出涌现或是不连续指标对连续量的整流伪影，换连续指标神话消失 | [2](book2-pretraining-revolution/README.md) |
| 2023-04 | LLaVA（视觉指令微调） | 冻结视觉语言双塔、只训一层线性投影缝合两个世界，换视觉指令对话 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2023-05 | GQA（分组查询注意力，Ainslie 等） | 把 MQA 单 KV 头放宽为分组共享、约 5% 数据续训迁移，近 MHA 质量配近 MQA 速度 | [3](book3-modern-open-source-skeleton/README.md) |
| 2023-05 | DPO（直接偏好优化） | 把奖励模型与 RL 目标折叠成偏好对上的直接损失，免采样免奖励模型 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2023-05 | NoPE（无位置编码） | 去掉显式位置编码只留因果掩码，证明注意力能隐式学出次序 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2023-06 | PI（位置插值，Position Interpolation） | 把越窗位置线性压回训练窗内再微调，外推误差比直接越窗小约两个数量级 | [3](book3-modern-open-source-skeleton/README.md) |
| 2023-06（无论文，经 YaRN 转述） | NTK-aware 缩放（bloc97 社区配方） | 社区配方只改 RoPE 底数、不压频率，免微调保近邻分辨把窗租到 8k 以上 | [3](book3-modern-open-source-skeleton/README.md) |
| 2023-06 | 连续批处理（continuous batching 命名，Anyscale 博客） | 为 iteration 级调度起产业通名，让 Orca 机制经 vLLM 落地成行业标配心智 | [6](book6-inference-systems-intro/README.md) |
| 2023-06 | H2O（Heavy-Hitter Oracle，重击者淘汰） | 在线维护累计注意力的重击者集合、预算内淘汰轻 token，20% 预算 COPA 48→84 | [6](book6-inference-systems-intro/README.md) |
| 2023-06 | AWQ（激活感知权重量化） | 观察到 1% 显著通道支配量化误差，per-channel 缩放保离群，近无损 W4A16 | [6](book6-inference-systems-intro/README.md) |
| 2023-06（开源首发；论文 v1 2023-09） | vLLM（PagedAttention） | 把操作系统分页思想搬进 KV 管理，逻辑相邻物理离散仍算注意力，显存利用升两成、吞吐 2-4× | [6](book6-inference-systems-intro/README.md)[7](book7-vllm-v1-source-code-vol1/README.md) |
| 2023-07 | Llama 2（Meta 开源基座第二代） | 骨架不动堆数据到 2T 并转商用许可，大档引入 GQA-8，代际翻新靠配方 | [3](book3-modern-open-source-skeleton/README.md) |
| 2023-07 | llama2.c（Karpathy 教学复刻） | 单文件数百行复刻 LLaMA，静默绑定偏离官方 untied，成 config 对拍反面教材 | [3](book3-modern-open-source-skeleton/README.md) |
| 2023-07 | Lost in the Middle（长上下文中部迷失） | 系统揭示首尾好中间盲的 U 形曲线，「放进上下文≠被用到」从此可测 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2023-07 | FlashAttention-2（FA2） | 减非 matmul FLOPs、序列维并行加 warp 分工，A100 达 72-73% MFU | [6](book6-inference-systems-intro/README.md) |
| 2023-08（按 v1 提交月） | YaRN（Yet another RoPE extensioN） | 按波长把频率谱分段、慢针压快针不动，配注意力温度旋钮，微调数据省一个量级 | [3](book3-modern-open-source-skeleton/README.md) |
| 2023-08 | SARATHI（chunked prefill） | 实测两段成本差 200×，把长 prefill 切块与 decode 拼成饱和混合批消干扰 | [6](book6-inference-systems-intro/README.md) |
| 2023-09 | StreamingLLM（注意力汇聚元） | 发现初始 token 被训成汇聚元、留几个 sink 即可无限流式，窗口外丢弃不崩、加速 22 倍 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2023-09（官宣博客；GitHub 首 tag 10 月） | TensorRT-LLM（TRT-LLM） | NVIDIA 官方推理库，in-flight batching 与连续批处理同思想异名词 | [7](book7-vllm-v1-source-code-vol1/README.md) |
| 2023-10（OCP 规范 2023-09 先行） | Microscaling 数据格式（MX/MXFP4，OCP 联盟） | 以 32 元素块共享 E8M0 指数救活 E2M1，4.25 bit 块浮点让 MoE 权重原生压缩 | [4](book4-scale-and-efficiency/README.md) |
| 2023-12 | Mamba（选择性状态空间模型） | 让状态方程参数随输入变化、选择性记忘，自回归吞吐 5 倍；时不变破功逼出双形态 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2023-12 | Zoology（MQAR 多查询关联召回基准） | 立 MQAR 基准把「状态大小换检索能力」变成可度量权衡 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2023-12（官方首发；论文 2024-01） | Mixtral 8x7B（开源稀疏 MoE） | 只换 FFN 为 8 专家 top-2、余件承袭 Mistral 7B，成首台全开源旗舰 MoE | [4](book4-scale-and-efficiency/README.md) |
| 2024-01 | DeepSeekMoE（细粒度专家分割+共享专家） | 切细专家让组合数涨七个量级、共享专家去公共冗余，2B 档追平 1.5 倍参数 GShard | [4](book4-scale-and-efficiency/README.md) |
| 2024-01 | Medusa（多头投机解码） | 在主干头上挂 K 个解码头并行猜多 token、树注意力验证，加速 2.3-3.6× | [6](book6-inference-systems-intro/README.md) |
| 2024-01 | EAGLE（特征级投机解码） | 在特征层而非 token 层做 draft 自回归，轻猜手加速 2.7-3.5× | [6](book6-inference-systems-intro/README.md) |
| 2024-01 | DistServe（PD 分离） | 把 prefill 与 decode 分池部署、KV 组间流转，7.4× 请求量或更紧 SLO | [6](book6-inference-systems-intro/README.md)[7](book7-vllm-v1-source-code-vol1/README.md) |
| 2024-01（官方发布；论文 v1 2023-12） | SGLang（RadixAttention） | 以 RadixAttention 基数树组织前缀复用、调度与计算重叠先走一步 | [6](book6-inference-systems-intro/README.md)[7](book7-vllm-v1-source-code-vol1/README.md) |
| 2024-02 | Repeat After Me（固定状态模型复制能力负结论） | 电路复杂度证明固定状态模型抄不动指数长串，划出线性路线检索天花板 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2024-02 | DeepSeekMath（GRPO 组相对策略优化） | 价值网络换成组内相对优势、同问多次采样互为标尺，RL 显存减半 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2024-02 | KIVI（2bit 非对称 KV 量化） | 发现 K 离群钉通道、V 散 token，非对称 2bit 各按离群轴取缩放，压 KV 显存 2.6× | [6](book6-inference-systems-intro/README.md) |
| 2024-03 | Sarathi-Serve（无停顿调度） | 把分块 prefill 升为 stall-free 服务调度、token 预算节流，容量 2.6-5.6× | [6](book6-inference-systems-intro/README.md) |
| 2024-04（模型首发为准；论文 2024-07） | Llama 3（The Llama 3 Herd of Models） | 架构只动 GQA 转正与 θ 两格旋钮，翻代全押 15T 数据、128k 词表与后训练 | [3](book3-modern-open-source-skeleton/README.md)[4](book4-scale-and-efficiency/README.md) |
| 2024-04 | MTP（多 token 预测，Gloeckle 等） | n 个独立头共享主干同押未来 n 个 token，7B 档代码任务 +15% | [4](book4-scale-and-efficiency/README.md) |
| 2024-04 | SnapKV（观察窗 KV 预选） | 用 prefill 末观察窗的注意力分布预告生成期关注点，池化选块一次定去留 | [6](book6-inference-systems-intro/README.md) |
| 2024-05 | MLA（多头潜在注意力，DeepSeek-V2） | 细粒度专家群+MLA 组出 236B/21B 激活，训练成本省 42.5%、KV 降 93.3% | [4](book4-scale-and-efficiency/README.md) |
| 2024-05 | Mamba-2（SSD 结构半可分对偶） | 把 SSM 与注意力收进结构半可分矩阵对偶框架，chunk 并行有了统一语言 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2024-06 | DeltaNet 并行化（WY 表示，序列维并行） | 秩一修正连乘打包成 WY 表示、块内一次解出，串行训练瓶颈打通、1.3B 超 Mamba | [5](book5-frontier-hybrid-architectures/README.md) |
| 2024-06 | NVIDIA Mamba 混合语言模型实证（8B） | 同预算 8B 实测约 6:1 混合 12 项基准全胜、纯 SSM 败于复制检索 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2024-08 | 免辅助损失负载均衡（auxiliary-loss-free，Wang 等） | 每专家一枚只改派单的偏置逐步压负载，免辅助损失梯度税，1B 档 9.50 对 9.56 更优 | [4](book4-scale-and-efficiency/README.md) |
| 2024-08（模型权重 2024-06 先行） | Gemma 2（滑窗与全局注意力 1:1 交替） | 滑窗与全局 1:1 逐层交错压 KV、softcap 稳训练，窗口路线做成开源定式 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2024-08（kernel 2024-01 先行） | Marlin（W4A16 混合精度 GEMM kernel） | 为 W4A16 重排权重的专用 GEMM kernel，内存带宽区小批量保约 3.9× | [6](book6-inference-systems-intro/README.md) |
| 2024-09 | Llama 3.2（轻量与多模态系列） | 1B 小机身面对 128k 词表把 tie 拨回绑定，config 投票示范词表账的决策 | [3](book3-modern-open-source-skeleton/README.md) |
| 2024-12 | DeepSeek-V3（671B MoE） | 训练侧全面换代：noaux+FP8+MTP 进 671B 旗舰，2.788M 卡时训完 14.8T | [4](book4-scale-and-efficiency/README.md) |
| 2024-12（社区配方，博客单源） | Muon 优化器（Newton-Schulz 正交化，Keller Jordan 等） | 对矩阵动量做 Newton-Schulz 正交化、平谱更新取代逐元素偏斜，后进旗舰配方 | [4](book4-scale-and-efficiency/README.md) |
| 2024-12 | GDN（Gated DeltaNet，门控 Delta 网络） | 给 delta 记忆装数据依赖的遗忘门，双形态等价换线性训练与常数状态 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2025-01（Nature 正刊 2025-09） | DeepSeek-R1（RLVR 纯强化学习激发推理） | 纯规则奖励加 GRPO 大规模强化，长思考自发涌现，RLVR 范式确立 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2025-01 | FlashInfer（可定制 attention 引擎） | 把多形态注意力 kernel 收进可组合引擎库，一套 kernel 伺候全解码形态 | [7](book7-vllm-v1-source-code-vol1/README.md) |
| 2025-01（口径=V1 alpha 官告） | vLLM V1 架构 | 推倒五个核心系统（调度器/KV 管理/worker/采样器/API server），编译默认开 | [7](book7-vllm-v1-source-code-vol1/README.md) |
| 2025-01（官方 org 首发口径） | vllm-ascend（vLLM 昇腾 NPU 后端插件） | 以平台插件把 vLLM 长到昇腾：17 个特化架构注册、patch 层伺候上游漂移 | [7](book7-vllm-v1-source-code-vol1/README.md) |
| 2025-02 | Moonlight（Muon 规模化验证，Moonshot） | 给 Muon 补权重衰减与 RMS 对齐，compute-optimal 下自报对 AdamW 约 2× 效率 | [4](book4-scale-and-efficiency/README.md) |
| 2025-02 | NSA（原生稀疏注意力） | 三分支门控稀疏把「看谁」端到端学出来，打分复用压缩分支免辅助损失 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2025-03 | Gemma 3（5:1 滑窗混合架构） | 5:1 局部全局配比、缩放只打全局层，KV 占比从六成压到一成五，128K 档站住 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2025-04（日期锚=官方许可生效日） | Llama 4（Scout/Maverick） | 只交付 config 与权重、无论文无博客，插槽只动前馈（专家群）与位置（3:1 交错）两格 | [3](book3-modern-open-source-skeleton/README.md) |
| 2025-06（取 vllm-ascend 引入月） | ACLGraph（vllm-ascend 昇腾 NPU 图执行） | 昇腾对 CUDA Graph 的等价捕放，同一语义面下差异全在执行器与守卫层、对拍 6.15× | [7](book7-vllm-v1-source-code-vol1/README.md) |
| 2025-07 | Kimi K2（MuonClip/QK-Clip） | 1T MoE 旗舰示范 MuonClip：超阈重缩放 Q/K 权重，15.5T token 零 spike | [4](book4-scale-and-efficiency/README.md) |
| 2025-08 | gpt-oss（20b/120b，OpenAI 开源推理模型） | 无论文只交 config 与源码：交错滑窗+sink+原生 MXFP4 专家，参数账三方逐位对平 | [4](book4-scale-and-efficiency/README.md)[5](book5-frontier-hybrid-architectures/README.md) |
| 2025-09（无论文，三方拼证） | Qwen3-Next-80B-A3B（GDN 3:1 混合架构首发） | 36:12 的 GDN:满注意力排班配超稀疏 MoE，1/10 成本换 10× 吞吐（自报） | [5](book5-frontier-hybrid-architectures/README.md) |
| 2025-10 | Kimi Linear（KDA，Kimi Delta Attention） | 遗忘门细到每头每通道成 KDA、3:1 混排 MLA，48B 上 NoPE 长文反超 RoPE 版 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2025-12 | LatentMoE（Nemotron 3 潜空间专家路由） | 把专家搬进 ℓ=d/4 潜空间、门与共享专家留全宽，非线性预算白赚 3.7× | [4](book4-scale-and-efficiency/README.md) |
| 2025-12（机型 2025-09 先行） | DSA（DeepSeek Sparse Attention，DeepSeek 稀疏注意力） | KL 蒸馏轻索引器选 token、核心注意力 O(Lk)，两阶段续训送上 V3.2 生产 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2026-02 | GLM-5（智谱 MoE 旗舰，MLA+DSA） | MLA 配 DSA 全层稀疏索引、仅 20B token 续训适配，检索近半价 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2026-02（家族首发月） | Qwen3.5（阿里混合注意力家族，GDN 3:1） | 3:1 骨架延续并瘦身专家群、原生多模态与 MTP 齐上，262K 原生与 1M 托管口径分立 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2026-03（模型卡名 IndexShare） | IndexCache（GLM-5.2 IndexShare 机制正身，索引缓存复用） | 测得相邻层 top-k 重叠 0.7-1.0，索引进缓存每四层复用，1M 档索引开销回落至逐层方案的零头 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2026-03 | AttnRes（Attention Residuals，注意力残差） | 每层配一次对前层输出的检索式残差，伪查询 softmax 加权替代固定捷径 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2026-03（v0.29.0 起默认） | Model Runner V2（MRV2，vLLM 模型执行器二代） | 把 7,664 行单体 runner 重写为模块化异步优先二代并转默认；NPU 侧仍 MRV1 | [7](book7-vllm-v1-source-code-vol1/README.md) |
| 2026-04（预览版 v1 提交；编号 2606 特例） | DeepSeek-V4（Flash/Pro，CSA+HCA 预览版） | CSA 压摘要卡再选条目、HCA 看粗账，1M 档 FLOPs 与 KV 压到 V3.2 约一成 | [5](book5-frontier-hybrid-architectures/README.md) |
| 2026-06（无论文模型） | GLM-5.2（IndexShare 索引共享，1M 上下文版） | 索引器跨层共享、57 层不带索引器参数（权重级证据），1M 档计算再降（自报 2.9×） | [5](book5-frontier-hybrid-architectures/README.md) |
| 2026-07 | Kimi K3（月之暗面旗舰，KDA+Gated MLA） | 2.8T 参数 69 层 KDA 配 24 层 Gated MLA、全机 NoPE 直推 1M | [5](book5-frontier-hybrid-architectures/README.md) |
| 2026-09（+0800 口径 09-22） | vLLM v0.30.0（本丛书 Book7/8 源码阅读锚） | Book7/8 行级阅读锚：不冻结则行号随版本漂没；与运行栈 0.27.1 双轨互证 | [7](book7-vllm-v1-source-code-vol1/README.md) |

表下注（※ 条目）：

- **1976**：束搜索起源月级不可考（仅见 1976-03 孤证著录），只标年；丛书精讲的是 GNMT 式长度惩罚束搜索（2016-09）。
- **2009 / 2010**：MessagePack 月不可考只标年；ZeroMQ 时点三说并存（2007 项目起点 / 2009 仓创建 / 2010 名号期首发），取名号期首发年。
- **2014**：Dropout 无 arXiv，JMLR 官方页月级不可考，只标年。
- **2018-04※ / 2019-11**：The Annotated Transformer 官网页无日期，取书内 ch08 考据框著录 2018-04-03（第三方源码直读单源，置信 C）；KV cache 在 2017 原论文未描述、更早工程实现无文献可考，取首见 Shazeer 1911.02150 §2.4（2019-11，首次正式描述而非首创）。

表中带「自报」字样者为厂商自报口径；预览版、社区配方级条目均已括注。
