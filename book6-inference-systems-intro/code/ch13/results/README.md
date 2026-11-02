# 第13章随书实测记录

本目录保存2026-10-30在单机8张Atlas 910B2上生成的小型结果，运行栈vLLM 0.27.1＋vllm-ascend 0.27.1rc1。没有将理论值填入测量字段。

- `environment.json`、`deepseek-v2-lite-config.json`、`model-revision.json`、`hardware.txt`、`topology.txt`：版本、固定模型配置及启动前设备盘点。
- `capacity_model.json`：固定官方revision的权重头统计与逻辑MLA缓存容量。
- `mechanisms_verified.json`：CPU FP32同构计算检查；不是完整模型的精度评测。
- `collectives_hccl{2,4,8}.json`：HCCL数值验证与同步墙钟微基准，包含Python调用开销。
- `serving_matrix1_*.json`：实际HTTP请求、token ID、到达时间和测量摘要。fast档三类合成负载，非生产SLO容量测试。
- `serving_probes_*.json`：补充固定前缀teacher-forced概率检查。概率与生成差异按原样保留，不以模型能生成等同数值完全一致。
- `serving_repeat_*.json`：TP1与TP2×PP2独立重启后的重复实验，新增4096-token前缀概率与后续生成检查。
- `serving_sync_*.json`：两配置均显式关闭异步调度的控制实验。在本例中消除了PP长输入相邻token重复，长输入32-token生成均与同轮单卡参考匹配；不代表完整任务精度认证。
- `status_*.json`：实际服务启动参数与请求完成状态；`summary.json`还保留关键启动日志原句。
- `summary.json`：由`summarize.py`读取上述文件、启动日志和HBM采样生成；包含每配置状态、SP实际日志、KV容量、驱动采样峰值及原始证据SHA-256。
- `measurements.md`：自动生成的fast档对照表。

状态`pass`表示服务启动、健康检查与规定请求完成，不代表任务级精度认证。`not_completed`或`failed`不可作为性能结果。

主矩阵默认调度的`pp2`/`tp2pp2`长输入出现重复输出，其较低TPOT不作为正确推理的收益。原始读数保留并在表中标†，同期配置不被后来的同步控制数据替换。64-token前缀与4096-token前缀检查的覆盖范围不同；主矩阵最早TP1/TP2的短前缀检查由`probes`补充，长前缀检查从`repeat`起加入。所有性能场景均独立暖机后计时。

大型权重不随书发布。完整服务器日志、逐次驱动HBM采样、原始硬件拓扑保存在写作工作区`log/book6-ch13/`；汇总保留其SHA-256用于溯源。模型revision为`604d5664dddd88a0433dbae533b7fe9472482de0`。日期按本次任务记录；指标定义与局限见正文§13.14。
