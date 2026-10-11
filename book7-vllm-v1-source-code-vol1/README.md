# Book7《vLLM V1 源码剖析（上）：引擎主链路》

沿一条请求的生命周期逐站精读 vLLM V1 主链路：进程拓扑与 ZMQ 桥、调度器、分页 KV 与前缀缓存、执行层与双 model runner、编译与图执行（ACLGraph）、attention 后端、权重加载与模型注册四门、输出链与 OpenAI 兼容层、实验方法学。大项目：自训 DESIGN-207M 注册进 vllm-ascend（C+A 双档，与 mini 引擎三方对照）。

- 双轨口径：源码引用锚 vllm v0.30.0（ced6857）；实验观察=运行栈 vllm 0.27.1 + vllm-ascend 0.27.1rc1
- 实验环境：Atlas 910B3（CANN 9.1.0 / torch 2.10 + torch-npu）
- 代码：`code/ch02`-`code/ch13`（各章实验件+制图）+ `code/00-feasibility`（大项目注册四件）
- 图：`figures/`（fig-N-M-slug.png，N=章号）
- 数字底单与引用复核：写作仓 `research/Book7-vLLM源码剖析上/`；实测底档 `log/book7-*`（两仓外部，按正文地址回查）
