# 随书核验说明

本次重写只运行CPU纸面算例，没有NPU模型实测。以下脚本只需Python标准库；输出在本目录results中。

```bash
python verification/check_paging_cases.py
python verification/check_theory.py
```

分页脚本包含28项检查；其余专题43项，包括对三词、四分之一概率网格上225对分布的穷举。检查结果不替代数学证明或文章教学审读。原并行专题的历史设备结果仍在code/ch13/results，版本身份不变。

完整研究、覆盖及审校记录在写作仓research/Book6-理论重写。随书附来源索引、逐章审校摘要与图示清单；图的生成源在figures-sources，PNG可直接阅读，新制SVG可编辑。旧第19章图片保持原始PNG和原绘图脚本，未伪造新SVG来源。

全册正式同步采用临时目录、受管文件清单及哈希冲突检查。不会整目录删除，也不会自动运行历史实验。

离线读本使用KaTeX 0.16.11静态预渲染结果与内嵌字体，许可见[KaTeX许可](KaTeX-LICENSE.txt)。数学解析、图片和内部链接已核验；没有将可选浏览器逐页视觉检查标记为完成。
