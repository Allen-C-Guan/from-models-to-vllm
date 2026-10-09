# 第 8 章 注意力 kernel：从铺表到 online softmax（数学最重章）

> **最小背景栏**：本章的数学只需 softmax 的定义（Book1 第 2 章教过：指数化+归一化）与矩阵分块乘法（本科线代）。硬件侧只需「HBM 慢、片上快」的两级存储直觉——第 1 章 roofline 的余温。Book5 第 5 章的 GDN chunkwise（chunk 内并行/块间递推）在 8.6 回望，不重教。

## 8.1 Book1 那张表，工业界重写了算法

Book1 第 2 章手写缩放点积注意力时，我们把 $(n, d_k) \times (d_k, n)$ 的打分矩阵整个算出来、整个存下来——「一张 $n \times n$ 的表」。当时留了一句：工业界后来的推理引擎为这张表重写了算法（不再整个铺开）——Book1 第 11 章把它登记为 F17（手写 matmul+softmax 的工业界命运）。欠账到期。

为什么要重写？先算中间量的账。$(B, h, n, n)$ 的打分矩阵，bf16 下每元素 2 字节：$n{=}1024$、$h{=}16$ 时是 32 MiB；$n{=}8192$ 时 2 GiB——**注意力的中间量比权重还大**。更糟的是访存模式：naive 实现要把这张表写进 HBM、再读回来做 softmax、再读回来加权 V——一来一回，HBM 带宽被中间量吃掉，而这些中间量的**算术强度低到可怜**（softmax 与加权是逐元素/归约操作，FLOPs 少、bytes 多——第 1 章 roofline 斜坡的最底部）。kernel 的任务于是明确：**让 $n^2$ 的中间量不进 HBM**——在片上（SRAM）算完一块、只留最终结果。

## 8.2 online softmax：分块算 softmax 的数学难关

分块的 matmul 好办（块乘块累加），难点在 softmax——它是**全局归一化**：每一行的分母是整行的和，块着算时「整行」还没见全。FlashAttention（FA1，arXiv:2205.14135，NeurIPS 2022）的贡献就是解开这个结：**online softmax**——两状态在线修正。

设某查询行对键 $1..n$ 的打分为 $s_1, \dots, s_n$。定义两个**可增量维护**的状态：$m$ = 至今所见最大值（数值安全锚）、$l$ = 以 $m$ 为底的归一系数。已处理键 $1..j$ 时：$$m_j = \max_{i \leq j} s_i, \qquad l_j = \sum_{i \leq j} e^{s_i - m_j}$$用人话说：指数先减去当前最大值（防溢出——$e^{80}$ 就上溢了），分母也随之变成「减 $m$ 之后的和」。真正的 softmax 分母是 $\sum e^{s_i}$（不减任何东西），但注意恒等式：$\frac{e^{s_i}}{\sum_k e^{s_k}} = \frac{e^{s_i - m}}{\sum_k e^{s_k - m}}$——分子分母同乘 $e^{-m}$，比值不变。**所以用哪个 $m$ 都行，只要同一行统一**——这就给了「边走边换 $m$」的自由。

新增一块键 $j{+}1..j{+}T$ 时怎么合并？新块自己算 $m_{new}$ 与 $\sum_{i \in blk} e^{s_i - m_{new}}$；老状态的 $m_{old}, l_{old}$ 要迁到新底——乘一个修正因子 $\alpha = e^{m_{old} - m_{new}}$：$$l_{j+T} = \alpha \cdot l_{old} + \sum_{i \in blk} e^{s_i - m_{new}}$$（加权累积 acc 同理乘 $\alpha$ 修正。）合并式的直觉还可以再压一句：alpha=e^(m_old−m_new) 就是「旧账本换新币」的汇率——旧底数下记的 l 与 acc，乘上汇率就是新底数下的等价账目。整个过程没有任何一步需要「回头重算」——这正是 online 的含义：**每来一块，旧账按汇率折一次、新账直接记，账本永远只存当前底数**。这就是全部数学：**减 max 保安全、同乘 e^(−m) 保等价、换底乘 alpha 保合并**。

**例 8.1（构造例：6 元素分 2 块的 online softmax 手算）**：打分 $s = [1, 3, 2, \; 5, 1, 2]$（前 3 为块 A、后 3 为块 B）。**块 A**：$m_A = 3$，未归一权重 $p_A = [e^{-2}, e^0, e^{-1}] = [0.135, 1, 0.368]$，$l_A = 1.503$，部分和 $acc_A = 0.135 v_1 + 1 v_2 + 0.368 v_3$。**块 B**：$m_B = 5$，$p_B = [1, e^{-4}, e^{-3}] = [1, 0.018, 0.050]$，$l_B = 1.068$。**合并**：$m_{new} = 5$，$\alpha = e^{3-5} = 0.135$——$l = 0.135 \times 1.503 + 1.068 = 1.270$；$acc = 0.135 \times acc_A + (p_B \text{ 加权 } v)$。最后除：$\mathrm{out} = acc / l$——与一次算全六个的结果逐位相同（你自己可以拿这六个数验一遍）。分块没改数学，只改了「中间量以什么底数暂存」。

还有一个小而关键的机制叫 **bonus token**：若 gamma 个候选全部被接受，target 的同一次前向顺带算出了「再下一个位置」的分布——白送一个 token 的采样机会。E 公式里 alpha^(gamma+1) 的 +1 就是它：全中时实得 gamma+1 个 token。投机的彩头机制。

复杂度账：块在片上算完即弃，HBM 只流过 $Q, K, V$ 与输出——每元素常数次，**HBM 访问从 $O(n^2)$（铺表）降到 $O(n)$**；FLOPs 一点没省（精确注意力，非近似——FA1 标题里的 exact 正是此意）。 roofline 的语言：把 softmax 与加权这两段低强度操作的输入（中间表）从 HBM 搬进片上——斜坡上的活搬到了不花 HBM 带宽的地方。

## 8.3 从零实现与对拍

`fa1.py` 给出双实现：`naive_attention`（铺表——Book1 的做法原样）与 `flash_attn1`（分块+online 修正，tile 可调）。数值对拍三方（CPU fp32）：flash1 vs naive max|Δ|=7.15e-07、flash1 vs `torch.sdpa` 4.02e-07——教学实现与工业实现、与全量重算三方一致。

NPU bf16 时延扫描（Atlas 910B3，warmup 后 3 次均值）是本章的主实测：

**实物 8.1（三种实现的时延，出处：本机实测，(1,16,n,64) 形状）**

```text
n=1024:  naive 0.553 ms | flash1(教学版) 57.656 ms | sdpa 0.311 ms
n=2048:  naive 2.234 ms | flash1(教学版) 115.834 ms | sdpa 0.267 ms
n=4096:  naive 9.576 ms | flash1(教学版) 237.693 ms | sdpa 0.692 ms
```

三行读数的形状流转先摆一遍：naive 每步 materialize (1,16,n,n) 的打分表（n=4096 时 512 MiB bf16 每次）；sdpa 内部按 tile 在片上算完即弃；教学版 flash1 按查询块循环（块打分 (16,128,n)）——三种中间量驻留策略，三种时延曲线。三行三个故事。**naive 的平方律**：0.55→2.23→9.58——n 翻两倍、时延翻四倍多，中间量的 HBM 往返主导（4096 时打分矩阵 256 MiB/次）。**sdpa 的平坦**：0.31→0.27→0.69——工业 kernel 近乎线性，4096 档对 naive 已是 **13.8×**；「不铺表」的收益在自家卡上兑现。**教学版 flash1 的「反常」**：它慢于 naive——因为教学版只分块了查询维、块中间量仍走 HBM，还背上了 python 逐块循环的税；**它示范的是算法结构（online softmax 的数学与形状流转），不是性能**——真正的 FA 收益来自「块驻留片上」的深度工程（寄存器/SRAM 分块、算子融合、并行布局），那是 kernel 工程的专门手艺（Book7 的地界，本册只隔着玻璃看）。教学诚实账：结构对、对拍齐、性能留给工业版——三层对拍在 kernel 章的读法。

## 8.4 sdpa 与本机两事实

`F.scaled_dot_product_attention`（sdpa）是 PyTorch 的统一入口，底下分发后端。本机（torch 2.10 + NPU）实测两条事实（`03_probe_sdpa.py`，notes/04 P1）：其一，npu/bf16 裸调全组合跑通（与 CPU fp32 参照差在 bf16 量级内）——8.3 的 sdpa 列正用它；其二，**`sdpa_kernel` 的后端开关在 NPU 上逐位失效**——强制 FLASH/EFFICIENT/MATH/CUDNN 四种后端，输出与默认分发逐位一致（连 NPU 上不存在的后端也「成功」）——torch_npu 注册的 kernel 不消费 CUDA 导向的后端标志。给写代码的读者的实用结论：**NPU 上不能靠后端开关选 kernel**——版本框注：torch 2.10/torch-npu 2.10.0.post4 实测口径。

## 8.5 谱系一览

FA1（2205.14135，NeurIPS 2022）立下 online softmax 与 IO 感知；谱系的演进逻辑值得点破：FA1 解决「中间量进不进 HBM」（算法层）；FA2 解决「块内的计算怎么排」（工程层）——减少非 matmul 运算（softmax 的指数与归并占了大头）、重排线程块的任务划分让并行单元吃满（A100 达峰值的 50-73%、比 FA1 快约 2 倍【官方论文摘要口径】）；**FA2**（2307.08691，ICLR 2024）减非 matmul 运算、重排并行（A100 达峰值 50-73%、比 FA1 快约 2×【官方论文摘要】）；**FA3**（2407.08608，NeurIPS 2024）异步与 FP8（H100）。谱系一行总结：**FA1 定算法、FA2 定工程、FA3 上低精度、FlashInfer 做集成、flex_attention 走编译**——五代接力，每一代把上一代的一个「静态」变成「可编程」。**FlashInfer** 把各家 kernel 汇成推理向库——kernel 生态从各家自研走向库化供给：这条演化线的终点（kernel 即插件）正是 Book7 attention registry 章的入场券。**FA4** 占位说明：它绑定 torch ≥ 2.13——本运行栈 torch 2.10 无此路径【证据等级：待核——见警示清单】，NPU 侧的 fast attention 以 vllm-ascend 内置实现为准（待核验口径）。torch 生态内还有 flex_attention（可编程注意力编译路线）与 _fa3 后端（repos/pytorch 的 `torch/nn/attention/` 目录可查）——谱系表留给读者当地图。

## 8.6 回望：Book5 的 chunkwise 是同一个思想

现在回看 Book5 第 5 章的 GDN chunkwise（chunk 内并行、块间递推）与第 6 章的 KDA——它们的「块」与 FA 的「tile」是同一个思想在不同算子上的投影：**把不可并行的全局依赖（softmax 归一/delta 递推）拆成「块内可并行+块间可合并」的两层结构**。同构的对照表列出来更有力：FA 的 m（running max）对 GDN 的门控衰减（都是「旧信息按权重折旧」）；FA 的 l（running sum）对 GDN 的状态累积（都是「增量入账、存量折算」）；FA 的换底修正 alpha 对 GDN 的逐通道遗忘门（都是「合并新块时的汇率」）。把 Book5 第 5 章的 chunkwise 递推式与本章的合并式并排，两套公式几乎可以逐项对读——**分块+状态修正是超越具体算子的通用形态**。FA 用 m/l 两状态合并块、GDN 用状态矩阵跨块递推——数学对象不同（归一系数对状态矩阵）、结构同构。线性注意力层没有 $n^2$ 中间量，kernel 压力天然小——架构侧（Book5）与系统侧（本册）在「访存友好」上殊途同归。

## 8.7 本章小结

F1 的 kernel 半句（$O(n^2)$ 的显存与 kernel 还账）与 F17（那张表的铺法革命）在此合销。带走的心智：**online softmax 的三句话——减 max 保安全、同乘 $e^{-m}$ 保等价、换底乘 $\alpha$ 保合并；分块不改数学、只改中间量住在哪**——而「住在哪」正是 HBM 带宽的全部账单。教学版给了结构与对拍、工业版给了 13.8× 的实测差距——kernel 工程的最后一公里在玻璃橱窗的另一侧（Book7）。下一章离开算子层，回到序列的产出口：采样。

## 动手验证

- 对拍与时延：`source env.sh && ASCEND_RT_VISIBLE_DEVICES=<卡> python code/ch08/fa1.py --bench-npu`（产物 `log/book6-ch08/fa1_base.json`）
- sdpa 后端门控实验：`python code/00-feasibility/03_probe_sdpa.py`（notes/04 P1）
- FA1/FA2 论文：arXiv:2205.14135 / 2307.08691（notes/03 §0 已核）
