# 配套代码与复现入口

`ch00`—`ch12` 与 `00-feasibility` 保留原书实践：分词、BERT微调、GPT参数工厂、模型族、显存探针、GPT-2实现、权重映射和训练网格。仅归一化书内路径，并修正一处dropout行和注释；不偷偷改变旧训练协议。

在既有环境中，从本书目录运行：

```bash
python code/verification/verify_full_math.py
python code/verification/verify_math.py --chapter all
```

前两项运行CPU数学练习，输出在 `log/book2-polish-verification/`，不需要训练或下载权重。

原历史训练及绘图脚本的入口保持不变，运行前查看 `--help` 与文件头。它们可能需要MPS、语料、权重、旧版本依赖与已缺失的日志，不代表当前NPU机器可以原命令直接重跑。旧 `iso_flop.py` 中的拟合输出和 `scaling_curve.py` 的置信带解释是历史分析资产；正文已明确这些拟合结果的条件与限制，不应再次用旧图覆盖校订图。

历史脚本不作为通用安装程序。书仓不打包语料、模型权重或环境；先在工作区根目录激活既有环境，再进入本书运行相对命令。
