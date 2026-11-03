# GDN数学练习

`gdn_math.py`采用正文键行、值列约定。recurrent返回逐步输出、末状态与轨迹；block返回一个块的A/U/W/R等中间量；chunkwise负责块间交接。

从书目录运行：

```bash
python code/ch05/gdn-math/verify.py
```

CPU高精度检查输出、状态、梯度、因果性与分段续算，结果写到log/gdn-math。程序不加载模型权重，不测生产内核速度。历史完整层与Qwen缩小整机实践分别在`../gdn.py`、`../qwen3next_mini.py`，依赖与测量条件见正文5.15。
