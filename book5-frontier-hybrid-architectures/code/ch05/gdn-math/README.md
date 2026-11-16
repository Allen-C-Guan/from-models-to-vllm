# GDN 数学练习

`gdn_math.py` 采用正文键行、值列约定。recurrent 返回逐步输出、末状态与轨迹；block 返回一个块的 A/U/W/R 等中间量；chunkwise 负责块间交接。

从书目录运行：

```bash
python code/ch05/gdn-math/verify.py --out-name run1
```

程序在 CPU 上以高精度检查输出、状态、梯度、因果性与分段续算，结果写到工作区的 `log/gdn-math/run1/`（独立书仓运行时用当前目录的 `log/`）；已有目录会拒绝覆盖，重跑请换 `--out-name`。其中 `case.json` 包含逐步状态及所有分块中间量，`verification.json` 保存检查结果、实际库版本和源码哈希。依赖 PyTorch、SymPy 与 transformers 5.14.1 的纯 PyTorch 参考函数。程序不加载模型权重，不测生产内核速度。历史完整层与 Qwen 缩小整机实践分别在 `../gdn.py`、`../qwen3next_mini.py`，依赖与测量条件见正文5.15。
