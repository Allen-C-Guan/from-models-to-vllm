# 配套代码与复现入口

这些文件是原写作工作区的源码快照。探针按 `code/Book8-vLLM源码剖析/` 层级定位工作区根，依赖固定 `repos/`、`env.sh`、本地模型和 `log/`；随书目录并非自包含安装包。完整使用边界与输出目录说明见书末附录 B。制图、组装与检查代码也保留供审阅，静态阅读不需要运行它们。

## 观测与投影

| 脚本 | 行级入口 | 作用 |
|---|---|---|
| inspect_model_assets.py | [main](inspect_model_assets.py#L7) | 读取固定模型快照配置和 safetensors 头部，不加载权重张量。 |
| inspect_platform.py | [main](inspect_platform.py#L20) | 只检查平台插件与类定义来源；不实例化Worker、不加载模型。 |
| observe_async_frontend.py | [main](observe_async_frontend.py#L67) | AsyncLLM 真实 NPU 观测：并发 A/B 的 DELTA 输出、关闭 C 生成器后取消。 |
| observe_connector_roundtrip.py | [main](observe_connector_roundtrip.py#L10) | ExampleConnector真实CPU文件/槽拷贝；构造对象，非引擎PD部署。 |
| observe_dp_contract.py | [main](observe_dp_contract.py#L30) | CPU驱动真实DP路由方法；构造负载，不启动任何引擎/网络。 |
| observe_expert_map.py | [main](observe_expert_map.py#L9) | 从固定源码AST执行专家布局纯函数；不运行MoE模型或设备内核。 |
| observe_fp8_encoding.py | [main](observe_fp8_encoding.py#L7) | CPU执行固定源码FP8标量缩放函数；不是设备FP8矩阵乘。 |
| observe_grammar.py | [main](observe_grammar.py#L12) | 实际XGrammar/vLLM包装的CPU状态实验；小词表，不运行生成模型。 |
| observe_hybrid_state.py | [main](observe_hybrid_state.py#L12) | 真实缓存规格与滑窗元数据，CPU构造状态，无设备缓存。 |
| observe_kv_metadata.py | [main](observe_kv_metadata.py#L37) | 执行上游 KV 元数据对象；没有模型前向、设备 KV 或 NPU 性能结论。 |
| observe_lora_mapping.py | [main](observe_lora_mapping.py#L11) | MRV2真实LoraState与固定槽位转换函数；CPU构造状态，无adapter权重。 |
| observe_metadata_contracts.py | [main](observe_metadata_contracts.py#L13) | CPU 契约驱动：调用真实回填方法与固定源码的纯匹配函数；不启动推理。 |
| observe_metrics_contract.py | [main](observe_metrics_contract.py#L25) | 用构造时钟驱动真实指标对象，并验证请求收集器错误唤醒。 |
| observe_mm_contract.py | [main](observe_mm_contract.py#L11) | CPU多模态位置和encoder缓存管理实验；无视觉模型前向。 |
| observe_ngram_contract.py | [main](observe_ngram_contract.py#L7) | 执行固定源码n-gram纯函数体；无模型、无NPU、非MRV2 proposer。 |
| observe_parallel.py | [main](observe_parallel.py#L12) | 双 NPU 的 TP/PP 对照：读取每个rank的模型元数据并生成同一A/B。 |
| observe_requests.py | [main](observe_requests.py#L171) | 第 3–5 章真实请求观测；只包装本次创建的对象，不修改安装包或上游仓。 |
| observe_runner.py | [main](observe_runner.py#L30) | 复用已验证的 A/B 驱动，额外观测 Runner 边界；不改原脚本或依赖。 |
| parallel_probe.py | [model_view](parallel_probe.py#L2) | 可由spawn子进程导入的只读模型观测函数。 |
| prepare_async_evidence.py | [main](prepare_async_evidence.py#L10) | 保存 AsyncLLM 原始记录，验证增量拼接，并执行真实输出收集器的CPU合并。 |
| prepare_closing_evidence.py | [main](prepare_closing_evidence.py#L6) | 从同一次既有NPU记录抽取闭环；不启动模型。 |
| prepare_parallel_evidence.py | [main](prepare_parallel_evidence.py#L13) | 保存双NPU观测及失败历史；核对探针、分片与最终token。 |
| prepare_runner_evidence.py | [main](prepare_runner_evidence.py#L9) | 保留 Runner 观测原件，生成可追溯投影并验证输入／块／输出一致性。 |

## 制图、阅读版与校验

`render_*.py` 生成可编辑 SVG 和 PNG；`source_notes.py`、`chapter_evidence.py` 保存教学注释与字段投影定义；`build_materials.py` 和 `build_chapters.py` 从固定源码与证据生成正文实物。`build_reader.py` 生成内嵌图片的阅读版。

`check_sample.py` 保护已认可样稿的证据与原有文件；`check_architecture.py` 核对样稿关键关系与调用位置；`check_book.py` 检查全书代码还原、数据投影、图、链接及阅读版。最终证据链与组装由独立脚本核验。检查文稿不需要重跑 NPU。

实验脚本保持原始版本，以便与 manifest 中哈希对应；不要为迁移目录而直接改写归档副本。复制到复现实验目录后再适配路径，并给新运行单独命名。
