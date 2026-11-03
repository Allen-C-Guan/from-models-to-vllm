# 附录 A 术语、状态归属与源码地图

这份索引用于读过正文后的定位。先辨认自己面对的是请求状态、执行计划、设备输入还是结果，再进入对应对象；不要从一个名字相似的类推断它就是本书讨论的那一份状态。

## A.1 容易混淆的词，按作用域重新区分

| 术语 | 本书中的含义 | 不应混同的对象或现象 |
|---|---|---|
| 前端（frontend） | 负责输入接纳和结果交付的引擎侧组件；在线时还连接 HTTP 层 | 不专指浏览器，也不等于 EngineCore |
| 核心（EngineCore） | 协调调度、执行和结果提交的对象；部署可在独立进程 | 不是一个 GPU/NPU kernel |
| 请求描述（EngineCoreRequest） | 跨核心入口传递的请求信息 | 不是所有组件共同修改的全局 Request |
| 核心请求（Request） | 调度器维护的身份、token 历史、进度与状态 | 输出侧 RequestState、Runner 的 RequestState 是其他投影 |
| 调度计划（SchedulerOutput） | 本轮请求、数量、块信息及增量的交接产物 | 计划中的进度不是设备完成证明 |
| 已缓存请求（CachedRequestData） | 执行侧已认识的请求，适合传递增量 | 不表示命中前缀缓存 |
| 前缀缓存（prefix cache） | 复用符合身份和状态条件的已计算前缀 | 不表示相似语义的文本可以共享 KV |
| 引用计数（reference count） | 当前仍有多少使用者依赖某块 | 计数为零不等于块内容已经被覆盖 |
| 块表（block table） | 逻辑块到物理块的映射 | 不是 token ID 列表 |
| 槽映射（slot mapping） | 当步 token 对应的物理写入位置 | 不是请求的批次行号 |
| 持久行／当步行 | Runner 长期请求存储索引／一次批次中的索引 | B 留在持久行 2 时，当步可以变成行 0 |
| 异步（asynchronous） | 必须说明是用户协程、消息收发、核心调度还是设备操作异步 | 流式返回、非阻塞调用与异步调度不是同一个开关 |
| Future | 某个工作将产生结果的交接对象 | 拿到 Future 不表示设备已经完成 |
| TP／PP／DP／EP | 参数张量／模型层／模型副本／专家的分布维度 | rank 数字只有放回具体进程组才有含义 |
| 后端（backend） | 对某类设备计算提供元数据与实现的接口体系 | Attention Backend 不能直接当作整个平台插件 |
| 量化方法（quantization method） | 约束参数创建、装载后处理及计算的层级契约 | dtype 名字不足以说明 scale 和内核布局 |
| Connector 的 role | 调度侧／执行侧协议职责 | 与 KV producer／consumer 的数据方向是两组分类 |
| 状态提交 | 将已确认事实写回后续决策依赖的状态 | 已计划、已发送、已到达、已可读要分开判断 |
| TTFT／ITL／TPOT | 首 token 延迟／输出事件间隔／按定义计算的每 token 时间 | 不自动等于单个设备 kernel 耗时或客户端首字节时间 |

英文名称用于与代码检索对齐。实际类名相同而模块不同，是最值得先停下检查的情况：`GPUModelRunner` 有两代实现，`RequestState` 在前端、执行层和卸载子系统也可能代表不同抽象。

## A.2 沿状态所有者找入口

下面的路径均相对上游仓根目录，版本固定为 v0.30.0。源码在线链接与精确行号在各章片段旁；这里保留目录层级，让源码文件发生小幅重排后仍能从职责重新定位。

| 想回答的问题 | 源码入口 | 先读到哪里即可停止 | 章节 |
|---|---|---|---|
| 引擎如何装配 | `vllm/engine/arg_utils.py`、`vllm/v1/engine/llm_engine.py`、`vllm/v1/engine/core.py` | 配置如何交给客户端、核心与执行器，资源准备何时完成 | 2 |
| 请求由谁登记 | `vllm/v1/engine/input_processor.py`、`vllm/v1/request.py`、`vllm/v1/engine/output_processor.py` | 核心 Request 与输出 RequestState 分别在哪建立 | 3 |
| 一轮何时提交结果 | `vllm/v1/engine/core.py` | `step` 与 `step_with_batch_queue` 的结果提交边界 | 4、13 |
| 下一步计算什么 | `vllm/v1/core/sched/scheduler.py`、`output.py` | 预算、分配与 SchedulerOutput；先不展开 Worker | 5 |
| 块归谁、何时释放 | `vllm/v1/core/kv_cache_manager.py`、`kv_cache_coordinator.py`、`block_pool.py` | 申请、引用、命中与释放的一次闭环 | 6、7 |
| 单设备和多 Worker 怎样替换 | `vllm/v1/executor/abstract.py`、`uniproc_executor.py`、`multiproc_executor.py` | 执行与采样返回契约、负责返回结果的 rank | 8、16 |
| 请求怎样变成输入张量 | `vllm/v1/worker/gpu/model_runner.py`、`states.py`、`input_batch.py` | 持久状态、当步行、位置与块映射 | 9 |
| 权重怎样进入模型层 | `vllm/model_executor/models/qwen3.py`、`model_loader/`、`layers/linear.py` | 参数名字、分片规则与一次 forward 形状 | 10、16 |
| Attention 如何读写 KV | `vllm/model_executor/layers/attention/attention.py`、`vllm/v1/attention/backend.py` | Backend、Builder、Impl 与输入元数据的契约 | 11 |
| token 怎样变成用户文本 | `vllm/v1/worker/gpu/sample/`、`vllm/v1/engine/output_processor.py` | 模型输出、核心输出、反分词与停止条件的分工 | 12 |
| 编译与图执行在哪生效 | `vllm/compilation/`、`vllm/v1/worker/gpu/cudagraph_utils.py` | 执行模式选择、捕获条件、回退边界 | 14 |
| 谁处理流式与取消 | `vllm/v1/engine/async_llm.py`、`core_client.py`、`output_processor.py` | 逐请求 Collector 与 abort 如何连接核心 | 15 |
| 请求如何路由到副本 | `vllm/v1/engine/core_client.py`、`core.py` | DPLBAsyncMPClient 的路由与取消；dense/MoE 核心选择条件 | 17 |
| 专家归谁、如何合并 | `vllm/model_executor/layers/fused_moe/` | expert map、MoERunner 和 prepare/finalize 的责任 | 18 |
| NPU 分支接在哪里 | 上游 `vllm/platforms/`；Ascend `vllm_ascend/platform.py`、`worker/`、`attention/` | 平台入口、实际类、方法来源和补丁边界 | 19 |
| 候选何时成为确认 token | `vllm/v1/worker/gpu/spec_decode/`、`vllm/v1/core/sched/scheduler.py` | 提议、验证、接受计数与进度修正 | 20 |
| 语法如何限制采样 | `vllm/v1/structured_output/` | Future、mask 行、接受与回滚的生命周期 | 21 |
| 媒体特征归谁管理 | `vllm/multimodal/`、`vllm/v1/core/encoder_cache_manager.py` | 占位区间、encoder 预算与共享缓存引用 | 22 |
| adapter 身份怎样映射 | `vllm/lora/`、`vllm/v1/worker/gpu/lora_utils.py` | 逻辑 ID、常驻槽位和每 token 映射 | 23 |
| 量化层何时改变权重表示 | `vllm/model_executor/layers/quantization/` | 方法选择、参数创建、装载后处理、apply | 24 |
| 混合状态能复用到哪里 | `vllm/v1/kv_cache_interface.py`、`core/kv_cache_utils.py`、`core/single_type_kv_cache_manager.py` | spec、group、不同状态的共同恢复位置 | 25 |
| 外部 KV 何时可读 | `vllm/distributed/kv_transfer/kv_connector/v1/` | 查询、分配、传输、完成、失败与释放的闭环 | 26 |
| 等待与异常如何定位 | `vllm/v1/metrics/`、`vllm/v1/engine/output_processor.py` | 事件到统计、异常到等待者的路径 | 27 |

“读到哪里停止”是一次阅读任务的边界，不是永远跳过其余代码。例如先确认 `execute_model` 的交接协议，再进入第9章查看输入布局，理解负担就能保持在一个层次。

## A.3 把一次新阅读写成可复查记录

建议留下一页记录：源码提交与配置、实际实现类、持有状态、输入产物、输出产物、结束和失败条件。然后画一张局部类图，列一条注明条件的调用树，再保存一份真实对象快照。

如果还不能解释一次状态修改由谁授权，就继续向装配与所有权方向读；如果对象关系已经清楚，却不知道结果怎样返回，就沿动态路径读；如果二者都清楚而数字对不上，先检查时点、行映射和数据来源。三种问题需要不同阅读方向。
