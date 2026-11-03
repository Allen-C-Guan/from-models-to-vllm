# 附录C　来源、版本与实现边界

本册以2017年的Transformer设计为对象，以arXiv固定版本v7为主要阅读文本。历史年份、论文修订时间、软件版本与本次实验时间是四个不同口径。推导以正文给出的假设为准，论文结果以原研究设置为准；下面的资料表帮助读者回到证据，而不要求用原论文补齐正文的推理。

## C.1 主要资料与阅读位置

|资料|本册使用位置|阅读时关注什么|
|---|---|---|
|[Attention Is All You Need，1706.03762v7](https://arxiv.org/abs/1706.03762v7)|全册，重点§3—6、Table1—4|设计、训练、任务结果是不同层次的证据|
|[Sequence to Sequence Learning with Neural Networks，1409.3215v3](https://arxiv.org/abs/1409.3215v3)|第1—2章|条件概率、两套LSTM、源顺序逆转及长句观察|
|[Neural Machine Translation by Jointly Learning to Align and Translate，1409.0473v7](https://arxiv.org/abs/1409.0473v7)|第2—3章|保留源位置，按旧目标状态生成上下文|
|[Learning Phrase Representations using RNN Encoder–Decoder，1406.1078v3](https://arxiv.org/abs/1406.1078v3)|第2章|GRU门与候选；保留门的约定不能混用|
|[Learning Precise Timing with LSTM Recurrent Networks，JMLR2002](https://www.jmlr.org/papers/v3/gers02a.html)|第2章|遗忘门、记忆通道与窥孔连接；本书显式采用省略窥孔的常见形式|
|[Effective Approaches to Attention-based Neural Machine Translation，D15-1166](https://aclanthology.org/D15-1166/)|第3章|global/local、打分函数与input feeding|
|[Layer Normalization，1607.06450v1](https://arxiv.org/abs/1607.06450v1)|第11章|统计轴；正文补齐含epsilon的微分|
|[Adam，1412.6980v9](https://arxiv.org/abs/1412.6980v9)|第13章|算法1、指数矩与零初始化偏差修正|
|[BLEU，P02-1040](https://aclanthology.org/P02-1040/)|第14章|截断计数、语料聚合、长度惩罚|
|[GNMT，1609.08144v2](https://arxiv.org/abs/1609.08144v2)|第14、16章|深层循环、量化推理与搜索目标|
|[Convolutional Sequence to Sequence Learning，1705.03122v3](https://arxiv.org/abs/1705.03122v3)|第16章|卷积、GLU、位置与多层注意力|
|[Rethinking the Inception Architecture，1512.00567v3](https://arxiv.org/abs/1512.00567v3)|第12章|§7给出含正确类的均匀平滑；不能误引1512以外的无关论文|
|[Neural Machine Translation of Rare Words with Subword Units，P16-1162](https://aclanthology.org/P16-1162/)|第13章、实践附录|分词规则从训练数据得到，减少未登录词问题|
|[Using the Output Embedding to Improve Language Models，E17-2025](https://aclanthology.org/E17-2025/)|第12章|输入/输出嵌入绑定与三方共享的条件|
|[Dropout，JMLR2014](https://jmlr.org/papers/v15/srivastava14a.html)|第13章|随机子网与近似平均的动机，不把完整非线性期望说成相等|
|[Deep Residual Learning，1512.03385](https://arxiv.org/abs/1512.03385)|第11章|学习相对输入的修正；视觉研究不等于所有序列模型的稳定性定理|

PDF及文本核验件、URL和SHA256保存在工作区调研目录；随书提供上面的参考文献表。下载成功只证明资料已取得，不代表逐字核验了所有附录。1997年LSTM作者镜像本次因证书过期未取得，未把它列为已经通读的原件；第2章没有将后来加入遗忘门的公式伪称为1997原式。

## C.2 同一篇论文也有不同文本

本轮重新取得v1—v7与NeurIPS会议版，核对与教学相关的版本差异。下表是英法big模型三个位置的BLEU；它们是论文报告，不是本书测量。

|版本|摘要|§6.1正文|Table2|
|---|---:|---:|---:|
|v1|41.0|41.0|41.0|
|v2|41.2|41.17|41.0|
|v3|41.0|41.17|41.0|
|v4|41.0|41.0|41.0|
|v5、v6、v7|41.8|41.0|41.8|
|NeurIPS会议版|41.0|41.0|41.0|

因此，写“论文是41.8”仍不完整；至少需要指出版本与位置。正文引用v7 Table2，同时保留正文41.0的差异，未自行解释哪次训练导致变化。版本对照的原始读数见上述版本对照表，各版链接见来源清单。

其他影响复现与阅读的差异也应保留：

|内容|核对结果|对本书的影响|
|---|---|---|
|base的FFN宽度|v1 Table3列1024，v2起列2048|正文按v7的2048核算，不混用早期表格|
|多头输出投影与缩放论证|v2在公式中补明输出投影，并增加缩放方差脚注|第6—7章完整推导，不照抄缺项的早期截图|
|学习式位置对照|v2起出现Table3(E)及对应讨论|“效果接近”是该开发集实验，不证明长度外推|
|Attention Dropout小节|v1—v4有，v5起删除；v7仍留有“三种正则化”等措辞|段落删除不能反推所有历史训练都没用过注意力dropout；本文只明确所据文本与当前代码|
|FFN作为参数读取的附录|v2/v3有，v4起撤去|不把早期探索当成v7标准层的替代定义|
|英法big的特定配置|v2/v3记过不共享源目标嵌入及FFN8192，v4起删去，只保留dropout0.1|不把已撤条款自动继承为v7配置|
|句法分析与注意力可视化|arXiv v7包含§6.3/Table4及可视化；会议版不含这些部分|引用这些实验时明确使用arXiv版本|

v2/v3的FFN附录提出一种解释：将第一层权重列看成固定的参数键，ReLU响应看成对这些键的非归一化读取系数，再以第二层权重行作为值做组合。第10章的神经元展开保留了这一数学联系；它不意味着ReLU已经变成softmax，也不意味着FFN能够访问其他token。早期论文还尝试用参数注意力替代FFN，但该实验在后续文本撤去，本册不将它计为原版架构的固定部件。

## C.3 论文、教学代码和软件库如何对照

**名字相同不等于计算图相同。** 比较实现先核对输入输出、LN位置、残差、激活、偏置、dropout位置、权重绑定与最后是否额外归一化，再看相同超参数。

[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)是有价值的教学实现，其SublayerConnection使用先规范化再调用子层的写法，并在堆叠末尾再规范化；它不能直接充当2017 post-LN计算图的逐式证据。PyTorch的MultiheadAttention也只是本册拆头数值对照对象，不能由该局部对照宣称整个训练配方都与论文一致。

本册小模型保留原版post-LN和ReLU，缩小为两层、32宽；源嵌入独立、目标输入输出绑定，使用固定学习率。完整模型与原论文各6层、512宽、预热调度等区别在第15章列明。短例不包含所有历史实现的工程优化。

Tensor2Tensor中常见“noam”命名与论文公式的关系是实现命名关系，不是另一个学习率定理。本册不根据名称推断作者归属，也不把某库今日默认配置当作2017实跑配置。原项目的相关对照代码仍在历史实践中，复用时必须重新核对版本。

后来关于LN位置的研究提供了额外解释。[Xiong等，ICML2020](https://proceedings.mlr.press/v119/xiong20b.html)在其简化模型和初始化假设下分析梯度尺度，并在报告的任务中研究pre-LN与去除预热。它支持“改变LN位置会影响训练条件”，不能证明所有post-LN模型都必须预热4000步。本册第11、13章给出通用链式法则与原论文配方，将后来的训练稳定性比较留到Book3。

## C.4 哪些旧说法需要改写

|容易误读的说法|本册保留的准确结论|
|---|---|
|固定向量必然装不下长句|有限精度、任务需求与学习条件影响能力；仅固定维度本身不是不可能性证明|
|softmax饱和就意味着所有损失梯度为零|区分softmax雅可比与完整交叉熵的logits梯度|
|多个头不增加任何成本|总投影宽度固定时主乘法量近似保持，权重表与执行开销仍可能变化|
|正弦位置自然获得可靠外推|线性位移结构可以证明，任务能力必须另外验证|
|残差彻底解决梯度消失|恒等项仍会与子层导数和后续LN共同作用|
|论文删掉一段就证明实跑没有该操作|文本证据、代码证据与训练记录分别成立，不能互相替代|
|参数数目、训练质量与吞吐可用单一数字公平比较|须分别固定词表、共享、预算、数据和设备条件|
|原稿MPS结果可改名为当前测量|只作为历史报告保留，不进入本轮通过计数|

## C.5 图、公式与实验的来源边界

本册插图均为重新制作的教学图，不是对原论文图的截图改字。结构图依据上述论文与正文公式重绘；几何图依据本书显式推导；第15章曲线和热图读取当前CPU日志。每图保留SVG图源和PNG静态版。

公式采用本书统一记号，有编号的主推导留在正文。CPU float64检查验证指定输入下的数值与梯度，CPU float32训练验证指定合成任务；两者不证明生产内核性能。外部论文链接需要网络；正文推导与本地插图可独立阅读。
