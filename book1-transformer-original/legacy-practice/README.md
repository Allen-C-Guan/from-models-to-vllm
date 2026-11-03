# 历史实践资产（保留原文件）

这里逐字保留重写前Book1随书代码，未作为本轮运行结果核验。旧README中的MPS、软件版本、性能数字和命令路径均为历史记录，不能直接视为当前环境保证。新正文的CPU实验入口在上一级code/，操作方式见附录A。

完整自然语言项目从[旧工程说明](code/ch09/README.md)进入：data.py、model.py、train.py、gru_baseline.py、decode.py与eval.py；消融脚本在code/ch10/。旧工作区路径code/Book1-Transformer原典仍在原工作区可用；这里因迁入legacy-practice，图输出、工作目录、导入和数据路径须先检查后适配，不承诺原命令在新目录不改就运行。

保留这些文件是为了保存实现与实验方法，不代表认可其中全部历史结论。特别是GRU尾部padding问题，按长度排序只能缓解，不是严格修复；应以有效句长/packed sequence获得末态后再比较。

新稿已将理论机制重新推导、历史结果与当前测量分开。旧正式正文和图的完整归档位于工作区drafts/Book1-5-系统重写/book1-history/，不会混入新正文目录。
