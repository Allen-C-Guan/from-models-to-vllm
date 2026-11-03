# 附录 B 全册实验与证据复现

正文已经给出足够的对象现场，不要求读者先跑完实验。需要验证某个机制时，按本附录选择最小实验；不要为了查看一张块表而启动整套多卡推理。

## B.1 先选证据层次，再选脚本

基础四次运行的环境、参数与完整事件格式见[实践与证据](实践与证据.md)。续写增加的记录如下。表中“CPU”包含直接调用真实对象的方法，以及注明边界的固定源码函数体执行；这两种都不是完整设备推理。

| 章节／问题 | 脚本 | 留存记录 | 实际验证与边界 |
|---|---|---|---|
| 3–5 请求、迭代、分块 | `observe_requests.py` | `baseline-inproc-traced`、`chunked-inproc-traced`、`baseline-mp-traced`、`baseline-inproc-plain` | NPU 生成；多进程只观测前端，另有无包装输出对照 |
| 6–7 块分配、引用与命中 | `observe_kv_metadata.py` | [kv-metadata/trace.json](evidence/kv-metadata/trace.json) | CPU 真实 KV 管理对象；没有 KV 张量计算 |
| 9–12 输入、位置与张量 | `observe_runner.py` | [runner-inproc-traced/projection.json](evidence/runner-inproc-traced/projection.json) | NPU 三轮真实输入、块/槽、hidden/logits 形状；复制观测会扰动时间 |
| 10 模型配置与权重形状 | `inspect_model_assets.py` | [model-assets/metadata.json](evidence/model-assets/metadata.json) | 读取固定 config 与 safetensors 头；不把文件头当作模型前向 |
| 13–14 异步状态与模式条件 | `observe_metadata_contracts.py` | [metadata-contracts/trace.json](evidence/metadata-contracts/trace.json) | CPU 真实异步状态方法及固定模式判断；没有图捕获/重放实测 |
| 15 流式、Collector 与取消 | `observe_async_frontend.py`、`prepare_async_evidence.py` | [async-frontend/result.json](evidence/async-frontend/result.json) | NPU AsyncLLM；核心异步调度关闭；CPU Collector 合并另行注明；没有 HTTP 测试 |
| 16 TP、PP | `observe_parallel.py`、`parallel_probe.py` | [tp2-final/result.json](evidence/tp2-final/result.json)、[pp2/result.json](evidence/pp2/result.json) | 两卡各自运行，核对分片/层范围及最终 A/B；没有 TP×PP 联合与通信性能测试 |
| 17 副本路由与取消分组 | `observe_dp_contract.py` | [dp-contract/trace.json](evidence/dp-contract/trace.json) | CPU 真实客户端方法，构造负载；没有多副本服务 |
| 18 专家归属 | `observe_expert_map.py` | [expert-map/trace.json](evidence/expert-map/trace.json) | 固定函数体生成 linear/round_robin expert map；没有 MoE 模型计算 |
| 19 平台选择与类来源 | `inspect_platform.py` | [platform/metadata.json](evidence/platform/metadata.json) | 读取实际 entry point、类与继承信息；不等于新跑一次模型 |
| 20 n-gram 辅助例 | `observe_ngram_contract.py` | [ngram-contract/trace.json](evidence/ngram-contract/trace.json) | CPU 固定函数体，去除 Numba 装饰器；属于 MRV1 辅助分支，不是 MRV2 MTP 实测 |
| 21 语法状态 | `observe_grammar.py` | [grammar-cpu/trace.json](evidence/grammar-cpu/trace.json) | 真实 XGrammar 对象、小词表、mask/接受/回滚/Future；没有模型 logits masking |
| 22 占位与特征缓存 | `observe_mm_contract.py` | [mm-contract/trace.json](evidence/mm-contract/trace.json) | 真实 PlaceholderRange、EncoderCacheManager；没有图片编码器 |
| 23 LoRA 映射 | `observe_lora_mapping.py` | [lora-mapping/trace.json](evidence/lora-mapping/trace.json) | 真实 LoraState 与固定映射函数体；传输辅助函数改用 CPU 张量，未装载 adapter 权重 |
| 24 FP8 编码 | `observe_fp8_encoding.py` | [fp8-encoding/trace.json](evidence/fp8-encoding/trace.json) | CPU E4M3 编码、scale、重建与误差；没有 FP8 GEMM/模型精度实测 |
| 25 混合状态 | `observe_hybrid_state.py` | [hybrid-state/trace.json](evidence/hybrid-state/trace.json) | 真实 spec、页统一、滑窗释放和 Mamba 模式；没有混合模型前向 |
| 26 Connector | `observe_connector_roundtrip.py` | [connector-cpu/trace.json](evidence/connector-cpu/trace.json) | CPU 真实 ExampleConnector 张量存取、完成计数合并；没有 PD 网络服务 |
| 27 指标与异常 | `observe_metrics_contract.py` | [metrics-contract/trace.json](evidence/metrics-contract/trace.json) | 真实统计/Collector 方法，时间与异常由探针构造；不是性能或故障注入测试 |
| 28 闭合全书 | `prepare_closing_evidence.py` | [closing/trace.json](evidence/closing/trace.json) | 从基础同一次运行提取三轮；未启动新推理 |

每份记录保留来源说明；多数探针还记录自身与上游文件的 SHA-256。派生投影保留原始输入哈希。哈希能检查文件是否变化，不能单独证明结论正确，因此正文还核对对象含义、形状、状态和作用域。

## B.2 在原写作工作区复现

脚本按 `code/Book8-vLLM源码剖析/` 所在层级定位 `repos/`、`drafts/` 和 `log/`。随书代码是这套工作区的快照，**不是在任意目录执行即可自动安装和下载的工具包**。复现需要匹配运行环境、只读的固定源码、相同模型快照，以及已配置缓存路径的 `env.sh`。

无需 NPU 的契约实验，可以在独立复现副本中这样运行：

```bash
source ./env.sh
python code/Book8-vLLM源码剖析/observe_kv_metadata.py
```

表中的 CPU 脚本均无命令行参数；以对应文件名替换即可。它们会写入固定的 `drafts/Book8-vLLM源码剖析/evidence/` 子目录，因此不要直接覆盖随书保存的原始证据。先复制工作区到实验副本，或明确改写实验副本中的输出位置；改过脚本后，其 SHA 与原实验自然不同，应另存为新记录。

运行 NPU 案例前，先执行 `npu-smi info` 确认空闲设备。下面单卡 0、双卡 0/1 只是命令形态示例，不保证这些卡在复现时空闲。每次独立 shell 都重新指定设备、激活环境，保留 spawn 和脚本主入口守卫：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
source ./env.sh
python code/Book8-vLLM源码剖析/observe_requests.py --case baseline --out log/book8-source-rewrite/my-baseline
```

`--case chunked` 加入预算分块和 C；`--deployment mp` 切换前端与核心的部署边界；`--plain` 关闭内部方法包装。基础脚本拒绝覆盖已有输出目录，复现时使用新 `--out`。

Runner 与 AsyncLLM 脚本分别固定写入 `runner-inproc-traced`、`async-frontend`，没有 `--out` 参数；在实验副本中修改输出常量，或保留原位置而使用全新目录结构。不要给脚本传入不存在的参数。`prepare_runner_evidence.py` 和 `prepare_async_evidence.py` 用于保留原件、核对投影与输出，不是第二次模型运行。

双卡命令分别为：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
source ./env.sh
python code/Book8-vLLM源码剖析/observe_parallel.py --mode tp --run-name my-tp2
```

另开 shell、重新核对设备后，将 `--mode tp --run-name my-tp2` 改为 `--mode pp --run-name my-pp2`。多卡编号严格升序。该本机离线探针为了把 `parallel_probe.model_view` 传给 Worker，显式启用 callable 序列化；它仅用于本地受控观察，不是在线服务配置。保持 `parallel_probe.py` 可导入，不能把观察函数临时搬入 stdin 或不可导入的 `__main__`。

## B.3 失败和关闭日志也属于证据

TP 观察首次因 callable 序列化限制失败，第二次因 `__main__` 函数定位失败；调整为可导入模块后成功。两次失败日志保存在 `evidence/parallel-failed/`，没有与成功数据混写。

部分多进程、AsyncLLM 与双卡运行完成了预期输出，但关闭阶段出现超时强制结束或共享内存相关警告。正文只据此验证生成、交付和分片的观测范围，不把退出码零扩大成“所有资源清理正常”。完整控制台记录随 evidence 保存。

语法探针最初在已终结 matcher 上继续填 mask，暴露了生命周期前提；修正为终结后不再填入。混合状态与平台检查也保留了初次路径/配置问题的失败日志。失败记录用于解释探针修正，不作为最终机制结论的数据来源。

## B.4 怎样增加一个有用的观测点

先写出可被反驳的问题，例如“B 的持久行是否在 A 结束后仍为 2”，再选持久状态与当步批次之间的边界。快照立即复制必要字段，保存请求 ID、轮次、进程和 rank；不要等整个运行结束再序列化早先保存的可变对象引用。

若读取设备张量，记录复制是否会同步。正确性观测完成后，性能实验应去掉这些探针，并单独定义预热、计时区间和请求分布。本书没有把短案例、构造时钟或带探针耗时作为性能基准。
