# Book6《LLM 推理系统导论：从自回归数学到服务算法》

引擎无关的推理物理学：服务指标与算术体系、KV 压缩三线、批处理五部曲、投机解码、量化推理、注意力 kernel、图执行与并行拓扑。大项目：mini 推理引擎四版进化（naive→连续批处理→分页 KV→前缀缓存），默认服务 Book3 终态 207M（NPU 重训版）。

- 实验环境：Atlas 910B3（CANN 9.1.0 / torch 2.10 + torch-npu / vllm 0.27.1 + vllm-ascend 0.27.1rc1）
- 代码：`code/ch01`-`code/ch12`（各章件）+ `code/00-feasibility`（探针底座：roofline/sdpa/serving 冒烟/训练启动器）
- 图：`figures/`（fig-N-M-slug.png，N=章号）
- 数字底单与引用复核：写作仓 `research/Book6-推理系统导论/`（四件套）
