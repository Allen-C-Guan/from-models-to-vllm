# 附录 B 源码地图：从问题选择入口

这份地图集中第 3–12 章的主要入口。主仓源码锚为 v0.30.0／ced6857，Ascend 使用独立观察点 3b31886；两者不构成已验证运行组合。第 1、2 章的入口保留在各章末。先阅读相关章的职责与图，再按问题选择源码；定位到交接后即可停止。

## 第 3 章

先读：[本章正文](ch03-组件的运行位置.md)。

| 要回答的问题 | 固定版本入口 | 读到哪里即可停止 |
|---|---|---|
| 部署形态怎样选择？ | [EngineCoreClient.make_client](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/engine/core_client.py#L91) | 区分同进程与多进程客户端后停止 |
| 谁持续推进核心？ | [EngineCoreProc.run_busy_loop](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/engine/core.py#L1403) | 看输入处理与引擎步的交接 |
| 消息何时变成核心事件？ | [EngineCoreProc._process_input_queue](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/engine/core.py#L1429) | 看等待与取队列，不深入套接字工具 |
| 等待、抢占、终止怎样区分？ | [RequestStatus](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/request.py#L364) | 对照本章简化状态图即可 |

## 第 4 章

先读：[本章正文](ch04-引擎的启动与就绪.md)。

| 要回答的问题 | 固定版本入口 | 读到哪里即可停止 |
|---|---|---|
| 配置怎样组织跨组件约束？ | [VllmConfig](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/config/vllm.py#L358) | 先识别配置组及初始化检查，不逐参数背诵 |
| 架构怎样找到模型实现？ | [_ModelRegistry.resolve_model_cls](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/models/registry.py#L1371) | 识别候选与解析结果即可 |
| 文件权重交给谁？ | [DefaultModelLoader.load_weights](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/model_loader/default_loader.py#L415) | 看加载器与模型权重接口的交接 |
| 资源怎样成为缓存容量？ | [EngineCore._initialize_kv_caches](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/engine/core.py#L254) | 走完规格、容量、配置、初始化与预热 |

## 第 5 章

先读：[本章正文](ch05-调度的决策与约束.md)。

| 要回答的问题 | 固定版本入口 | 读到哪里即可停止 |
|---|---|---|
| 一步计划怎样形成？ | [Scheduler.schedule](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/core/sched/scheduler.py#L561) | 先读普通请求的预算与分配分支 |
| 抢占究竟保留和清除什么？ | [Scheduler._preempt_request](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/core/sched/scheduler.py#L1477) | 看引用释放、进度与重入队列 |
| 进度何时记账？ | [Scheduler._update_after_schedule](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/core/sched/scheduler.py#L1520) | 区分安排后的推进与完成后的校正 |
| 结果怎样影响下一轮？ | [Scheduler.update_from_output](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/core/sched/scheduler.py#L1874) | 找到请求更新与终止处理即可 |

## 第 6 章

先读：[本章正文](ch06-KV状态的管理与复用.md)。

| 要回答的问题 | 固定版本入口 | 读到哪里即可停止 |
|---|---|---|
| 哪些历史可以命中？ | [KVCacheManager.get_computed_blocks](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/core/kv_cache_manager.py#L265) | 看查找条件、命中长度和返回值 |
| 新增工作如何取得块？ | [KVCacheManager.allocate_slots](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/core/kv_cache_manager.py#L370) | 辨认需求、可行性与分配结果 |
| 块被重新使用时发生什么？ | [BlockPool.get_new_blocks](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/core/block_pool.py#L664) | 看空闲队列和旧缓存身份处理 |
| 释放为何不等于擦除？ | [BlockPool.free_blocks](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/core/block_pool.py#L770) | 看引用变化与回空闲队列 |

## 第 7 章

先读：[本章正文](ch07-计划如何变成模型计算.md)。

| 要回答的问题 | 固定版本入口 | 读到哪里即可停止 |
|---|---|---|
| 核心交给执行层什么？ | [SchedulerOutput](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/core/sched/output.py#L225) | 区分新请求、更新信息与本步安排 |
| 核心依赖什么执行契约？ | [Executor](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/executor/abstract.py#L38) | 识别初始化、执行、采样和结果接口 |
| MRV2 如何消费计划？ | [GPUModelRunner.execute_model](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/gpu/model_runner.py#L1631) | 沿状态更新到输入与执行主干，跳过专题分支 |
| 前向之后如何得到请求结果？ | [GPUModelRunner.sample_tokens](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/gpu/model_runner.py#L1997) | 看采样与结果包装，不进入各采样内核 |

## 第 8 章

先读：[本章正文](ch08-模型与硬件的适配边界.md)。

| 要回答的问题 | 固定版本入口 | 读到哪里即可停止 |
|---|---|---|
| 后端怎样声明能力？ | [AttentionBackend](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/attention/backend.py#L59) | 找到布局／dtype／头维及实现选择接口 |
| 执行需要哪些解释信息？ | [CommonAttentionMetadata](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/attention/backend.py#L385) | 对照分段、长度、块表与槽位 |
| NPU 平台接入哪些配置？ | [NPUPlatform.check_and_update_config](https://github.com/vllm-project/vllm-ascend/blob/3b31886237ed65c435e0965b10e8002ff5769766/vllm_ascend/platform.py#L442) | 找到 Worker、后端及执行模式相关选择 |
| 普通 NPU 后端声明何种缓存形状？ | [AscendAttentionBackend.get_kv_cache_shape](https://github.com/vllm-project/vllm-ascend/blob/3b31886237ed65c435e0965b10e8002ff5769766/vllm_ascend/attention/attention_v1.py#L110) | 看该接口返回值，不推广到全部后端 |

## 第 9 章

先读：[本章正文](ch09-性能优化的架构落点.md)。

| 要回答的问题 | 固定版本入口 | 读到哪里即可停止 |
|---|---|---|
| 核心怎样处理在途批次？ | [EngineCore.step_with_batch_queue](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/engine/core.py#L630) | 区分提交、队列与结果消费 |
| 状态更新与模式选择在哪里？ | [GPUModelRunner.execute_model](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/gpu/model_runner.py#L1631) | 找到状态差量及图分派位置 |
| NPU 图封装承担什么？ | [ACLGraphWrapper](https://github.com/vllm-project/vllm-ascend/blob/3b31886237ed65c435e0965b10e8002ff5769766/vllm_ascend/compilation/acl_graph.py#L60) | 辨认捕获与重放，不推断 CUDA 等价性 |

## 第 10 章

先读：[本章正文](ch10-从单设备到多设备.md)。

| 要回答的问题 | 固定版本入口 | 读到哪里即可停止 |
|---|---|---|
| 多 Worker 怎样组织？ | [MultiprocExecutor](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/executor/multiproc_executor.py#L111) | 先看初始化与工作进程持有关系 |
| 同一计划怎样分发？ | [MultiprocExecutor.execute_model](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/executor/multiproc_executor.py#L340) | 识别 RPC 与结果路径后停止 |
| TP、PP 等通信组怎样表达？ | [initialize_model_parallel](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/distributed/parallel_state.py#L1977) | 对照 rank 分组，不展开通信算子实现 |

## 第 11 章

先读：[本章正文](ch11-扩展功能的接入地图.md)。

| 要回答的问题 | 固定版本入口 | 读到哪里即可停止 |
|---|---|---|
| 候选验证位于哪里？ | [RejectionSampler](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/sample/rejection_sampler.py#L45) | 识别候选与验证结果接口 |
| 语法约束怎样进入采样？ | [StructuredOutputManager.grammar_bitmask](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/structured_output/__init__.py#L314) | 看请求状态到掩码的转换 |
| 不同层的状态需求如何表达？ | [KVCacheSpec](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/kv_cache_interface.py#L156) | 看规格接口，再选一类派生规格 |
| 远端状态怎样跨两侧协作？ | [KVConnectorBase_V1](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L185) | 区分调度侧与执行侧接口 |

## 第 12 章

先读：[本章正文](ch12-带着地图阅读源码.md)。

| 要回答的问题 | 固定版本入口 | 读到哪里即可停止 |
|---|---|---|
| 前端怎样消费内部结果？ | [LLMEngine.step](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/engine/llm_engine.py#L304) | 找到核心客户端、输出处理与终止反馈 |
| 一次核心计算怎样闭环？ | [EngineCore.step](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/engine/core.py#L589) | 识别调度、执行与状态更新 |
| 内部 token 怎样成为用户结果？ | [OutputProcessor.process_outputs](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/engine/output_processor.py#L629) | 识别身份、文本、停止与交付边界 |
