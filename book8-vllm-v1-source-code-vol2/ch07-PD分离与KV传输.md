# 第 7 章 PD 分离与 KV 传输：connector 生态

> 站段坐标：站 8-9（P/D 各自的调度与 KV）+站 11-12（KV 跨机搬运）+全拓扑。B7-8（PD 分离工程生态——B6-7 侧账）与 Book7 ch1 的存照句在此开箱（那一章只立存照「把拆开部署做成可运维的工程是另一回事」，本章开箱）。记号契约首现：**P 实例（prefill node，预填充节点）/D 实例（decode node，解码节点）**是实例角色、不是 prefill/decode 阶段（一词两义每次消歧）；**connector（连接器）专指 KV 传输器**——与 ch3 的 backend（文法引擎）、ch4 的 adapter（LoRA 外挂件）构成三个「外挂」的同名消歧谱。

## 7.1 为什么要把一台机器拆成两台

Book6 的 DistServe 理论账指回（7.4×/12.6×{厂商自报}）：prefill 与 decode 是两种**互相干扰的计算形态**——prefill 是算力密集型（长 prompt 一次大前向，吃满 compute）、decode 是访存密集型（逐 token 小步，吃满带宽）；混在一台机器上，prefill 的长前向会把 decode 的下一个 token 卡住（ITL 尖刺）、decode 的小步会抢走 prefill 的大批槽位（TTFT 上升）。**分离的动机不是「拆开更高级」，是两种 SLO 在一台机器上互相亏待**。具体画面（Book6 的图重放）：decode 阶段每 token 一步（几十 ms），用户盯着流式输出感受 ITL；突然一个 2000-token 的 prefill 批插进来（200ms+ 的大前向），decode 的下一步被卡住——用户看到输出停了一跳。反过来：prefill 想攒大批摊开销，decode 的小步把批槽位占走了——prefill 的 TTFT 上升。**同一批槽位、两种吃法，零和**——分离是解开零和的结构性方案。K3 报告的「机群级亲和调度」是在**不分离**的前提下用调度缓解同一问题（把 prefill 重请求和 decode 轻请求分到不同机器）——**软分离（调度层）与硬分离（拓扑层）是同一谱系的两个档位**，本章讲硬分离后的那条传输带。K3 报告的机群级应对（亲和调度+预算式准入）是同一个问题在更大尺度的答案——本章讲的是把两相拆到不同机器后，**中间那条 KV 传输带**怎么建。

一句话预览全章：**PD 分离的工程难点不在拆（拓扑是配置文件的事），在缝（那条 KV 传输带怎么建、搬多少、怎么对账）**——connector 生态的十七家就是十七条缝法，本章按「协议→生态→NPU 面→一致性→探针」的顺序把它们读完——阅读面为主、实证为辅（三级降级定谳后本章的增量在源码与生态的读法）。读完你应该能回答三个问题：什么负载适合分离（临界比）、传输带选什么（四家的栈位差异）、NPU 的边界在哪（三级降级的实证与原因链）。

三段定位：专题=PD 分离与 KV 传输；站段=8-9+11-12+全拓扑；模块=`vllm/distributed/kv_transfer/`（37362 行地界、注册 17 连接器）+vllm-ascend 侧注册覆盖。本章产出：拓扑与 SLO 张力账+临界比公式（7.2）、connector 协议正身与四家逐走（7.3）、NPU 面与三级降级的报错原因链实证（7.4）、缓存一致性的三机制骨架（7.5）、配置面探针三连+关路实验方法学（7.6）。

> **最小背景栏**：DistServe 论文的分离收益数学（Book6 ch12 的式与图指回）；KV 块的字节账是 ch1 的账 2 续用。本章不讲「该不该分」（Book6 已答），讲「分了之后那条带子」——以及它的 NPU 边界在哪里。

## 7.2 xPyD 拓扑：角色、实例与传输

xPyD（**x Prefill y Decode**——x 台 P 加 y 台 D 的通配记法，如 4P4D）拓扑三件套（图 7.1 的文字版——P 列算 KV、中间箭头逐层搬、D 列逐 token 出）：P 实例跑 prefill（吃长 prompt、出 KV）、D 实例跑 decode（吃 KV+短增量 prompt、逐 token 出）、中间一条 KV 传输带（P 算完的 KV 块搬到 D）。x 与 y 的比例由负载决定：长 prompt 短输出的场景（文档摘要）x 多 y 少、短 prompt 长输出的场景（对话续写、代码补全）x 少 y 多——**比例是容量规划（ch11）的输入参数而非固定推荐**。PD 与 TP 可以叠（P 实例本身 TP=2、D 实例 TP=4——两侧各自内部再切，ch6 的刀法逐侧用）——真实部署的组合空间是三维的（P 数 × D 数 × 各侧 TP 度），组合的选型最终回到 ch11 的容量规划合账。请求路由：负载均衡器把请求发给 P、P 做完 prefill 把 KV 搬给 D、D 接着 decode——**请求的「一半」留在 P（prefill 的隐状态）、「另一半」去 D（decode 的增量），中间靠 KV 缝合**。路由层还要解决「同一会话的后续请求去哪」——通常粘到同一台 D（它有这个会话的 KV，APC 命中率高——**PD 分离把 APC 的亲和从单机哈希变成跨机器路由表**，会话亲和是连接层的新课）。

先立一个生活直觉：PD 分离像**出版社（P）与印刷厂（D）**——出版社把排版好的原稿（KV）送到印刷厂，印刷厂拿着原稿开始印（decode）；原稿只需要送一次、印刷可以印很多份（多轮对话从缓存的 KV 直接续印）——**送原稿（96MiB）比重新排版（重算 2000-token prefill）便宜**，这是分家能成立的全部理由。下面的账就是把这个直觉数字化。

**例 7.1（KV 传输账——文件交接式 PD 的直觉）** toy：2000 token prompt、块 128、Qwen3-0.6B 的每 token 每层 KV 2048B（ch1 账 2 同源）。P 算完：$2000\times24\times2048\text{B}=96\text{MiB}$ 的 KV；传输带 25Gb（约 3GB/s）：$96\text{MiB}/3\text{GB/s}\approx33\text{ms}$——比重算这笔 prefill（2000 token 约 200ms+）便宜 6 倍以上。**例 7.2（多轮续用的摊销）** 上例的 96MiB 只算首轮——多轮对话第二轮起，D 侧已缓存全部前缀 KV，**零传输零重算**（APC 命中直接续 decode）——PD 分离的最大收益在多轮场景：首轮付一次传输、后续 N 轮全免——**传输成本是一次性的、会话越长摊得越薄**（与 ch4 的 mm 多轮复合收益同一结构）。

**传输还是重算的临界比**：传输时间 = KV 字节 ÷ 带宽 < 重算时间 = token 数 ÷ prefill 吞吐率——块大小、层数、带宽三参定临界点。**这个临界比是 KV 一切搬运决策的通用公式**：PD 分离搬不搬（本章）、KV 卸载到 CPU 值不值（ch9）、甚至「预填充结果要不要存盘复用」（LMCache 的 L3 层）——同一个公式换个介质带宽与重算速度再用一次——**记住临界比，比记住任何一家的 connector 名字都值钱**。

## 7.3 connector：十七家厂商的算盘

传输带的标准件是 **KVConnectorBase_V1**（`vllm/distributed/kv_transfer/kv_connector/v1/base.py`）——基类定义了 P/D 两侧的挂点协议，分两组各三个——**调度侧三抽象**（connector 与调度器的接口）：`get_num_new_matched_tokens`（D 侧问「新请求的前缀有多少 KV 可以从缓存池复用」——跨机器的 APC 命中查询）、`update_state_after_alloc`（调度器分配完 KV 块后通知 connector 更新状态）、`build_connector_output`（本步结束时 connector 把要传的块打包成输出）；**角色方法各侧一组**：P 侧的 `save_kv_layer`（每层前向完后存 KV——**逐层**保存而不是等整个序列算完，这是传输与计算重叠的关键：第 1 层存好时第 2 层还在算）、D 侧的 `start_load_kv`/`wait_for_layer_load`（按需装载——D 在 decode 到某层前把该层 KV 从远端拿到手）。**协议的时间线本质是「逐层流水线」**：P 侧存一层→传一层，D 侧收一层→用一层——不是「全部算完再全传」，是**把 KV 传输切碎到层的粒度与计算重叠**（ch5 的重叠思想在跨机尺度的重演——同一条件式、更大的介质——这条思想谱系值得追认：ch4 的 mm encoder 异步→ch5 的 async scheduling→本章的逐层 KV 流水线，全是「把等待藏在计算里」的同一家族）。这个设计有一个下游推论：**D 侧的 decoder 第一层开工的门槛只是第一层 KV 到达**，不是全部 24 层都到——P 算第 N 层时 D 已经在 decode 第 N-5 层（逐层追上）——时间线上的「追赶窗口」是 PD 分离的延迟红利。

注册的 17 家连接器按「PD 传输/卸载」两支分（notes/02 地界的分节法）——传输支是本章正菜，四家逐个走过——**NIXL**：UCX（Unified Communication X）的统一传输抽象——一层 API 遮住多种网络硬件（TCP/IB/RDMA），connector 不用关心底下走什么线——**传输层的「平台插件」**（与 vLLM 的 platform 插件同一设计思想在不同层的投影）；**Mooncake**：KV 缓存优先的传输引擎（国内 MooncakeAI/知乎系主力）——不只搬数据，还管「搬哪段」的缓存决策（局部 KV 传输、全局缓存池），vllm-ascend 的 MooncakeConnector 是 NPU 面的正做；**LMCache**：KV 分层缓存（论文 2510.09665——L1 GPU→L2 CPU→L3 磁盘的 KV 存储层级）——它的「跨实例 KV 复用」面与 PD 分离天然耦合（P 算好的 KV 缓存池化、D 直接取——PD 的传输带在 LMCache 视角里就是「L0 远端缓存层」）；**Simple**：教学样板 119 行——协议的「裸奔实现」（没有缓存优化、没有异步流水、只有接口的直译：`save_kv_layer` 直接送张量、`get_num_new_matched_tokens` 直接返回 0——「不缓存任何东西、每次都重传」的诚实现），gated 模型换 Qwen3-0.6B 即可跑——**读它不为学优化、为学协议**（先看清「什么都没做」的基线，再看其他十六家各加了什么）；这四家不是竞争关系而是**栈位互补**：NIXL 是传输原语层（其他家可以架在它上面）、Mooncake/LMCache 是缓存决策层（决定搬什么不搬什么）、Simple 是裸协议参考——**生态的实际形态是「原语层 × 决策层」的组合**，正如 AI 栈的「硬件 × 框架 × 应用」分层。卸载支（offload 到 CPU/分层）归 ch9。另有一小支**权重传输**（weight_transfer——PD 分离的衍生需求：D 实例启动时从 P 侧或共享存储拉模型权重而非各自下载，v0.30.0 增 sharded_rdt 四件——RDMA 加速的分布式权重拉取；vllm-ascend 有 HCCL 引擎版 `hccl_engine.py`+NPU IPC 版——**PD 的传输带不止搬 KV，还搬权重**，启动加速是它的账）。每家一枚定位详 notes/02 地界表。

> **【版本差异框注】#17 两本账**——讲接口用 v0.30.0：基类比 0.27.1 多 8 法少 2 法（`get_transfer_results`/`finish_forward`/`bind_kv_cache_manager` 等新增；cross-layer 对删除）；**讲 NPU 实例用 0.27.1**：vllm-ascend 的 MooncakeConnector/AscendStoreConnector 实做全挂本轨。两本账必须分列——「协议长什么样」与「NPU 上跑的是什么」是两个版本的同一件事。

基类的一个角色方法签名（P 侧逐层保存——协议时间线的锚点之一）：

```python
# vllm/distributed/kv_transfer/kv_connector/v1/base.py::KVConnectorBase_V1.save_kv_layer
# （骨架摘录·P 侧）【源码：阅读锚 v0.30.0】（Args 段以 … 略）
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """
        Start saving a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.
        ...
```

指名看什么：docstring 的第二句——**「从 attention 层内部调用、为了在执行期间异步拷贝」**——这就是逐层流水线的正身（KV 在 attention 层算完的那一刻就开始出村、不等整个前向收尾）；`kv_layer: torch.Tensor` 一个参数就是一层 KV 块张量（按 ch1 的块账口径）。

值得单独一提的还有 **ExampleConnector**（119 行教学样板，`kv_connector/v1/example.py`）：最小可用的 connector 骨架——`save_kv_layer` 怎么存、`get_kv` 怎么传——**是读 connector 源码的入口件**——十七家的公共协议先在这里裸奔（没有优化、没有缓存、只有接口的直译），看懂它就看懂了协议本身；其他十六家的差异全在「在裸协议上加了什么缓存策略与传输优化」。

## 7.4 NPU 面：三级降级的实证

本章的实验面**按预注册三级降级定谳**（`log/book8-ch01/pd_probe/VERDICT.md`——批一末资产项提前跑的探针）：

**表 7.1 PD 配置面探针三连**（Qwen3-0.6B，卡 0，serve+kv-transfer-config；底档 `log/book8-ch01/pd_probe/VERDICT.md`+三份日志原文——每行的报错逐字摘自实跑）

| 探针 | 配置 | 结果 | 报错原文 |
|---|---|---|---|
| ① | `kv_role=kv_producer`（无 connector 名） | 🔴 配置校验拒 | `Connector name is not set in KVTransferConfig` |
| ② | `+OffloadingConnector`（8G cpu_bytes） | 🔴 上游硬拒 | `CPU Offloading is currently only supported on CUDA-alike and XPU GPUs` |
| ③ | `+RecomputeConnector` | 🔴 上游 factory 拒 | `Unsupported connector type` |

三级降级的落位：一级（本机双实例）**不可行**（上游 kv_offload 明文 CUDA/XPU-only——不是 vllm-ascend 不支持，是**上游 spec.py 的设备检查拒绝 NPU**）；二级（云 GPU）作者已裁定选做不做；**落三级：ascend_store/NIXL/mooncake 源码精读+配置面走查**——本章按此口径写。要强调的是三级不是「退而求其次」而是**本机栈的诚实边界**：上游 kv_offload 的设备检查是引擎级设计决策（CUDA 假设深入到了 KV offload 的 spec 层），不是 vllm-ascend 落后——**读「为什么不支持」比读「支持」更能看清一个引擎的假设结构**，三行报错是一份压缩的设计文档。mooncake 侧有一线希望：`sources/01 §五` 的探针定谳注册表 24 名（比 notes/03 记载的 16 名多 8——运行时 import 探针实测），且 xPyD 主路仅依赖 mooncake 主包（非三禁包）；glibc ≥2.35 前置本机 2.35 **恰好达标**（edge 版系统踩线过——差一个 patchlevel 就翻车的部署边界，如实记录）；mooncake 安装检查单（dry-run 守卫+装后双实例冒烟）保留为 ch8 开工前的可选升级路径，但装了也只是把 xPyD 从「不可行」升「可试」（A2 无 RDMA over Converged Ethernet 组网、双实例最多走到单机共享内存档——**网络硬件是 PD 分离的物理前提**，没有高带宽低延迟的互联，传输带的临界比就翻负——软件再先进也搬不动没有路的货）。

vllm-ascend 侧的真做是两件（配置面在册、实做挂 0.27.1 轨接口）：**MooncakeConnector**（`vllm_ascend/distributed/kv_transfer/kv_connector/mooncake_connector.py#L1605`——挂 0.27.1 轨接口、覆写上游的 MultiConnector/OffloadingConnector/SimpleCPUOffload 三个注册名——**覆写的意思是不让上游的这几件先跑**：上游实现假设 CUDA、覆写是让 NPU 版先被命中；但覆写管不住上游 spec.py 里的设备检查——表 7.1 第②行的报错就是覆写漏防的那层）与**AscendStoreConnector**（`ascend_store_connector.py#L77`——华为自研的 KV 存储，yuanrong（圆融）后端存在但公开文档缺——「存在但不可考」如实登记，等待上游文档补全）。注册清单全貌：运行时 import 探针实测生效 24 名（notes/17 新读数——上游 16+ascend 侧 8），比 notes/03 记载多 8——**vllm-ascend 的注册覆盖面比差异表大**，成书按探针口径（差异表漏登的部分不改变「实做挂 0.27.1 轨」的结论，但数量口径要准）。xPyD 官方模板 `tutorials/pd/kv_pool.md#L181-330` 是部署面的参考起点（yaml+脚本全套：P/D 双实例的启动配置、共享存储卷挂载、健康检查——**阅读面口径**；模板的存在说明官方认为这条路径对某类硬件可行，但「模板在」≠「所有硬件都能走通」——表 7.1 的探针就是 A2 这类硬件的对照实测）。

## 7.5 缓存一致性：KV 的版本账

P 算的 KV 与 D 缓存的 KV 怎么对版本？这是分布式系统的不变命题——**缓存一致性**——在 KV 传输场景的具体化：单机时 APC 的哈希链只有一本账（ch1 的哈希链）、PD 后变成两本（P 侧算的新 KV、D 侧缓存池的旧 KV），中间还要过一个传输带——**账本从一变三**（P 算的、传输带在途的、D 缓存的），一致性问题由此产生。三个机制配合：①**kv_events**（事件流——每次 KV 保存/加载/驱逐发事件；v0.30.0 增 `ownership`（这段 KV 的所有权归 P 还是 D 还是共享池）与 `session_id`（跨实例追踪同一会话的 KV 谱系）两字段——两个字段是分布式 KV 的身份系统，**没有身份的缓存没法失效**，这是版本对账的前提）；②**哈希链复用**（ch1 的哈希链在 PD 里跨机器——P 存的 KV 块的 hash 与 D 侧 APC 索引可比对，同前缀免搬）；③**kv_load_failure_policy**（加载失败的处置策略——重传还是重算）。一致性问题的实质：**P 侧的「最新前缀」与 D 侧的「缓存版本」可能错位**（P 改了系统 prompt、D 还缓存着旧的——用户看到的回复会突然变风格，因为 decode 的上下文在两版 KV 里各取了一段）。三机制配合的答案骨架：事件流给身份（ownership/session_id）、哈希链给快速比对（同前缀免搬——**不比对就全搬，比对省下的就是临界比的余量**）、失败策略给兜底（重传 vs 重算——遇到不一致时退到重算的安全侧）。分布式 KV 的一致性协议（多写者、并发驱逐、跨池 GC）是 ch9 kv_offload 生态的进阶面——本章只立「一致性是 PD 的账本」这个意识。另记一笔运维面的一致性风险：**KV 缓存池的热更新**——系统 prompt 变更后 P 侧新 KV 与 D 侧旧缓存的过渡期里、两种版本的请求可能并行命中不同段——生产做法通常是「双版本并存+逐步淘汰」（缓存条目带版本号、旧版等 TTL 自然过期），与数据库的蓝绿发布同一思想——**一致性运维的答案是共存与过渡，不是原子切换**。

## 7.6 实验：配置面探针三连（读数即 7.4 表）

【本机实测（Atlas 910B3，CANN 9.1.0，运行栈 0.27.1+vllm-ascend 0.27.1rc1）：底档 `log/book8-ch01/pd_probe/{producer,offload2,recompute}.log`+VERDICT.md】——三发探针各 ~1 分钟（serve 起到配置校验即拒或 EngineCore 初始化到设备检查即崩——失败快是配置面探针的优点：**一分钟内拿到三层报错的原因链**），判读已进 7.4 的表 7.1。方法学上这三发是「**降级路径的正式定谳**」：预注册的三级降级在设计书里就写好了（本机双卡→云 GPU→源码精读），探针只是把第一级的「不可行」钉上三行报错的铁证——**实验不总是为了验证「能」，也常常为了正式地关掉一条路**；关路实验的结果（三行原文）与开路实验的结果同样是一手数据。本章按降级口径以源码精读为主体，但配置面的实证不可跳过——不实测就不知道「不支持」在三层中的哪层（参数面修复就能过的≠真的不支持）。工程学教训一笔：**三行报错原文比「不支持」三个字值钱**。逐层拆：第一行（`Connector name is not set`）是**配置校验层**——参数面缺件、给参数就能过；第二行（`CPU Offloading is currently only supported on CUDA-alike and XPU GPUs`）是**设备检查层**——spec.py#L179-180 的运行时硬门、参数给了也过不了（**vllm-ascend 管不到这层——上游代码自己检查设备家族**，平台插件再深也代理不了已经写死的 `is_cuda` 分支——这是比 ch6 的 EPLB 代理更硬的边界：代理能翻转标志位、翻不了 if 分支）；第三行（`Unsupported connector type`）是**factory 注册层**——上游 factory 的注册表不认这个名字（vllm-ascend 的注册覆盖走的是自己的工厂入口，serve 的上游校验先拦了）。**引擎栈的「不支持」分三层：配置校验/设备检查/factory 注册——读报错原文=拿到原因链+知道修哪层**。

## 7.7 本章小结

回扣开篇的 SLO 张力：PD 分离把两种计算形态各归各的机器，中间的 KV 传输带是 17 家 connector 的战场；接口协议在 v0.30.0（基类 ±10 法的两本账）、NPU 实做挂 0.27.1 轨（Mooncake/AscendStore 两件）；本机三级降级落底（上游 CUDA-only 硬拒的三行原文=边界实物），传输账的临界比（传输 vs 重算）是部署决策的心算工具。

**带走的心智**（五条——本章与 ch6/ch5 构成三部曲，心智也成谱）：其一，**分离的动机是 SLO 不兼容**——两种计算形态在同一机器上互相亏待，拆开是解不是炫技；拆开的代价是引入了传输带——一条新的延迟源、一致性面、运维复杂度，**分离的收益必须大于传输带的全部代价**（临界比+一致性开销+运维成本三笔一起算）——这是「该不该上 PD」的完整决策式；其二，**传输还是重算的临界比**是 KV 一切搬运决策的公式（带宽×时间 < 重算时间就搬）——从 PD 传输到 ch9 的 KV 卸载到 CPU，同一个临界比换个存储介质再用一次；其三，**生态的十七家算盘=同一个基类的十七种实现**——读 connector 先读 ExampleConnector 的 119 行裸奔，再看各家加了什么缓存什么优化——「接口的极简实现」永远是读生态的地图——从裸奔到十七家，看到的是同一接口在不同约束下的演化谱；其四，**逐层流水线是跨机重叠的武器**——save_kv_layer 在 attention 层内部触发异步拷贝——传输与计算的重叠不是批量级的（算完再传）而是层粒度的（边算边传），D 侧的追赶窗口由此打开；其五，**关路实验也是一手数据**——三行报错的原文钉死三级降级的第一级，比「不支持」三个字的信息量大一个数量级——**实验方法学里，「正式关掉一条路」与「验证一条路」同等重要**。

下一章进入算子层：量化生产化——权重从进门（检查点白名单六法）到出门（执行格式八种）的两本账，B7-9（MoE 量化与 expert 权重加载）在 ch6 预热的装载链上长出量化形态——量化与分布式的交叉（W4A16 权重的切分装载）正是 ch6/ch8 的接缝。本章的 KV 传输带在 ch9 还要回来（kv_offload/tiering 是「PD 分离的单机版」——同一笔搬运账、同一临界比、不同介质），到时 PD 与 offload 的生态地图会拼成一张完整的「KV 搬运宇宙」。

> **【欠账】** ①mooncake 安装+双实例冒烟（可选升级路径——glibc 2.35 恰好达标+dry-run 守卫检查单在档）→ 第 8 章批前定夺；②kv_events 的 ownership/session_id 语义展开与 ch9 kv_offload 生态合章（「KV 搬运宇宙」的另一半地图）。

**动手验证**（两件；探针复现 ~3 分钟）：
① `vllm serve Qwen/Qwen3-0.6B --kv-transfer-config '{"kv_role":"kv_producer"}'`——应见 7.4 表第①行报错（配置面证据——一分钟内即可复现，最快的「源码验证」之一）；
② `grep -rn "CUDA-alike" env/lib/python3.11/site-packages/vllm/v1/kv_offload/cpu/spec.py`——第②行报错的正身（设备检查的源码位置，找到那行代码就看到了「不支持」的物理层在哪一行写着）；
③ Mooncake 源码阅读入口：`repos/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_connector/mooncake_connector.py`#L1605 起——NPU 面正做的接口实貌（对照 ExampleConnector 读：哪些方法被覆写了=mooncake 的优化面）。
