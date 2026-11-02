# 第13章实验入口

所有脚本都从写作工作区根目录运行。涉及NPU实验时，先检查 `npu-smi info` 与 `npu-smi info -t topo`，选择当前空闲卡，再设置升序的 `ASCEND_RT_VISIBLE_DEVICES`，最后 `source ./env.sh`。纯CPU算法走查不需要空闲NPU。书仓中的 `code/ch13/` 是这些脚本的快照；在独立书仓使用时自行准备正文列出的环境，并相应缩短脚本路径。脚本不安装或升级推理组件。

固定模型为 `deepseek-ai/DeepSeek-V2-Lite`，revision `604d5664dddd88a0433dbae533b7fe9472482de0`。模型下载命令见正文 §13.14.3。以下 `$MODEL_PATH` 指向该快照目录。每次测量使用新的 `--out-name`，结果写入工作目录下的 `log/book6-ch13/`。

| 脚本 | 所需资源 | 作用 |
|---|---|---|
| `capacity.py` | 已下载权重；不需NPU | 从权重头统计参数、dtype与逻辑KV |
| `check_mechanisms.py` | CPU、PyTorch | TP、SP、EP、分片attention数值检查 |
| `collective_walkthrough.py` | CPU、Python标准库 | 树、Ring、递归折半/倍增、All-to-All与屏障的整数逐轮走查 |
| `fig_collective_algorithms.py` | CPU、matplotlib | 从走查状态绘制图13.4—13.10，不产生性能实测 |
| `collectives.py` | 2/4/8张空闲NPU、HCCL | 四种collective正确性与通信微基准 |
| `run_matrix.py` | 按case提供1–8张空闲NPU | 依次启动服务、测量并关闭自己创建的服务 |
| `bench_serving.py` | 已启动且匹配参数的服务 | 固定前缀概率检查与流式HTTP测量 |
| `summarize.py` | 已完成的日志 | 生成随书结果快照、校验和与对照表 |
| `plot_results.py` | 汇总JSON、matplotlib与CJK字体 | 绘制实测TP扩展曲线 |

正文第13.4—13.5节的通信算法可以先在CPU核验：

```bash
source ./env.sh
python code/Book6-推理系统导论/ch13/collective_walkthrough.py
python code/Book6-推理系统导论/ch13/fig_collective_algorithms.py
```

走查用整数精确比较：四卡树形广播/分发/收集/归约；四卡、八卡Ring RS/AG、递归折半与倍增、直接All-to-All与二进制距离转发；3/4/5/8 rank的传播式屏障。结果默认写入 `log/book6-ch13/collective-algorithms/walkthrough.json`；随书保存同一轮的 `results/collective_algorithms.json`。该JSON是构造例核验，不是HCCL日志、时延或算法自动选择的证据。图在写作工作区输出到草稿figures目录；书仓快照默认输出到本册figures，亦可用 `--out-dir` 指定。

完整fast矩阵命令如下；需要八张空闲卡。它依次执行，只有当前case占用设备。两卡或四卡环境请删去所需设备数更大的case，不能假定存在通用的自动降级。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_HOST_IP=127.0.0.1
source ./env.sh
python code/Book6-推理系统导论/ch13/run_matrix.py \
  --model-path "$MODEL_PATH" --mode fast --out-name my-matrix \
  --cases tp1 tp2 tp4 tp8 pp2 tp2pp2 replicas2 \
          dp2ep2 dp2tp2ep4 dp2tp2ep4sp tp2dcp2 tp4dcp4
```

`smoke`只有128输入、8输出、2个串行请求；`fast`是正文实测的三类负载；`full`增加长度和样本数，未作为本章已执行结果。三档均不代表真实生产任务精度或SLO容量测试。服务默认只监听本机，端口8130；独立双副本另用8131。启动器拒绝使用已有监听服务的端口。失败记录保留在各case的 `status.json` 和 `server-*.log`，不自动修改模型或并行配置重试。

对于自己的结果，可以直接读取服务JSON；`summarize.py`是本章归档工具，要求同一轮包含完整配置以及 `hardware.txt`、`topology.txt`、容量与机制记录。当前随书快照的生成入口为：

```bash
python code/Book6-推理系统导论/ch13/summarize.py --run matrix1 --probes probes
python code/Book6-推理系统导论/ch13/plot_results.py
```

主矩阵最早的TP1/TP2运行没有保存固定前缀logprobs，故另用 `probes` 做补充；新运行的客户端已统一保存该检查。`repeat`是为复核TP2×PP2长输入读数而进行的独立重复，不能混入主表假装扩大了全部配置的样本量。

复核发现默认调度PP长输入相邻token重复。另以 `--cases tp1 tp2pp2 --scheduling sync --out-name sync` 做控制，两配置都关闭异步调度。其长输入输出在本例中与单卡参考一致，数据独立保存在 `serving_sync_*.json`。后续fast/full客户端会同时保存64-token与4096-token固定前缀检查；smoke保留短前缀检查。不要把默认PP的异常低时延当作加速结果。

请结合[结果说明](results/README.md)阅读状态字段：`pass`只表示规定请求完成。正确性、时延、吞吐和容量是分别检查的维度。
