# 配套代码与复现入口

随书实践覆盖分词、BERT微调、GPT参数工厂、模型族、显存探针、GPT-2实现、权重映射和预算网格。第4章的参数化GPT工厂供第8、9、11章实验使用；第10章的三插槽实现供本章训练及第11章官方权重对拍使用。两者不能仅凭参数总量相同就认定训练、初始化和评估协议完全一致。

## 路径与环境

在已有环境中运行，不由实验脚本自动安装依赖。若使用本项目外层工作区，每次独立终端调用都在工作区根目录激活环境，并把书目录作为脚本路径；数据与产物保存在当前工作目录的log中：

```bash
source ./env.sh
BOOK_DIR="$PWD/from-models-to-vllm/book2-pretraining-revolution"
python "$BOOK_DIR/code/math-examples/verify_full_math.py"
python "$BOOK_DIR/code/math-examples/verify_math.py" --chapter all
```

单独取得书仓的读者可在本书目录运行正文所列的`python code/...`命令；此时log属于本书目录。先选择一种工作目录并保持一致，不能在两个目录间切换后仍期待相对数据路径相同。数学检查默认使用CPU，不需要训练或下载模型；完整检查输出到log/book2-math-verification，第四、八章检查输出到log/book2-polish-verification。完整检查拒绝覆盖既有报告，可用--out-dir指定新的log子目录。

## 主要入口

| 章节 | 文件 | 用途与前提 |
|---|---|---|
| 第2章 | ch02/bpe.py | 最小BPE、工具对照与覆盖实验；部分阶段需要指定语料/分词器 |
| 第3章 | ch03/glue_finetune.py | BERT微调与基线；需要数据、权重，先查模式及预算 |
| 第4章 | ch04/surgery.py、warmup_ablation.py、grad_ln_position_probe.py | 结构账、参数化工厂与归一化/梯度实验 |
| 第7章 | ch07/handcalc.py | 参数与计算量手算核对 |
| 第8章 | ch08/data_prep.py、run_family.py、fit_scaling.py | 准备SP训练流、按预算训练及条件拟合；数据/断点须匹配 |
| 第9章 | ch09/memory_probe.py | 理论账与显存探针；MPS测量需对应设备，current与driver分别记录 |
| 第10章 | ch10/model.py、train.py、shakespeare.py | GPT-2结构、HF对拍、训练及小语料准备 |
| 第11章 | ch11/data.py、official_gpt2.py、sample.py | 块化与同范围计数、官方权重映射、加载检查点采样；WT2为可选依赖 |
| 第11章 | ch11/iso_flop.py、train_124m.py、scaling_curve.py | 预算网格、124M指定预算训练、已有结果汇编；smoke不能用于预算定律结论 |
| 第12章 | ch12/make_figures.py | 阅读地图与代码依赖示意；输出到log，不改正文图 |

表中路径相对于code目录，完整调用方式与参数见各文件--help及正文对应节。00-feasibility保留早期可行性探针，除第11章说明的数据获取修正外，按历史版本阅读；它们不替代当前正式实验入口。获取脚本的旧--target-mb/--tok-mb参数实际按百万Unicode字符停止，净文本字节、JSONL字节另行记录。

## 历史记录与新运行

书中的Mac/MPS长训练、词法样例和计时保留原配置与日期。当前脚本修正了初始化、评估边界、断点身份、非有限值检查、块化计数等问题，新运行应保留自己的版本与结果，不能继续引用旧数字当成新实现的验收常量。绘图脚本把新结果写入log；正文图只有在核对图源后才更新。

第11章旧20%跨文档比例、1800.6/1304.1计数PPL和两词表混用的byte成绩存在已知口径或算法问题，正文保留勘误，等待真实数据重算。新的局部构造例证明修正机制在所测输入上成立，不替代缺失历史日志或训练。采样默认读取已保留的自训检查点，只有显式给出--train-steps才另行训练。

正式书仓提供代码及教学插图，不打包语料、权重或环境。长训练耗时为历史参考，不保证当前CPU、MPS或其他设备可原样重现；先核对数据、依赖与设备，再选择所需实验。
