# Book1 ch9 从零实现与小型翻译实验（完整教学项目）

第 9 章配套代码：手写 mini 原版 Transformer（保留论文主要结构，缩小配置并注明实现选择）在 Multi30K en→de 上完整执行一遍
2017 年的工作流——数据管道 → 整机训练 → GRU 双 baseline → beam 解码评测 → 外推实测。

本目录保留完整项目。表中MPS数字为2026年10月历史结果；当前代码已修复GRU有效长度处理、beam停止条件与交叉查询探针，尚未重跑长训练。不能把旧模型的分数和图说成修正后的新测量结果。完整的数据处理与评测需要datasets与sacrebleu，运行核心模型的前向计算不要求安装这两个包。

## 文件一览

| 文件 | 一句话用途 |
|---|---|
| `data.py` | Multi30K 加载（bentrevett 镜像 29000/1014/1000）；sentencepiece 自训 shared BPE-8k（en+de 合并）；动态填充批处理（本实现先补PAD再建张量，非连续切片用`.reshape`）；外推样本的具体组装见`eval.py` |
| `model.py` | 手写 mini 原版 Transformer：post-LN + 正弦 PE + 双塔三注意力 + ReLU FFN + 三处权重共享；N=4/d=256/h=8/d_ff=1024（9.42M，非嵌入 7.37M）；仅 matmul+softmax，不用 nn.MultiheadAttention/SDPA |
| `train.py` | Transformer 训练循环：Noam 调度（warmup 4000、factor 1.0）+ label smoothing 0.1 + Adam(β2=0.98, eps=1e-9)；label smoothing / NLL 双损失；checkpoint与逐轮JSON记录落 `log/Book1-ch09/` |
| `decode.py` | 贪心 + 批量 beam search（beam=4、α=0.6，GNMT eq(14) 长度惩罚 lp=(5+长度)^α/6^α）；模型统一 encode/decode_step 接口（GRU 同路径复用）；与逐句版使用相同的候选与停止协议 |
| `eval.py` | sacreBLEU(13a) 主指标 + tokenize=none/手写朴素 BLEU 三口径对拍自证；按参考句长分桶（≤7/8-13/≥14 词）；外推实测（band=[24,32]=1.5L~2L）；三方对照表 |
| `gru_baseline.py` | GRU 双 baseline：无 attention（c=最后隐状态逐步注入，ch1 瓶颈的实际示例）与 Bahdanau 加性注意力（缓解瓶颈的实际示例）；公平协议=同 BPE 词表/同数据同 batch 同 epochs（同 token 预算）/同优化器调度/非嵌入参数同量级（8.84M、8.12M vs Transformer 7.37M）/同 beam 推理 |
| （产物） | 数据与模型产物（BPE、checkpoint、逐轮记录、评测JSON）落 `log/Book1-ch09/`，不随书分发；使用前核对数据许可 |

## 运行命令（先激活Python环境，在本册目录运行）

```bash
# 1) 数据冒烟：加载或自训BPE + 长度分布 + 过滤率 + 一批形状（耗时以实际环境为准）
python "code/ch09/data.py"

# 2) 参考训练30轮（历史MPS约3.7分钟；修复版当前耗时另测）
python "code/ch09/train.py" --device mps --epochs 30

# 3) GRU 双 baseline（30 epochs 各，MPS 约 4.2 + 9.0 min）
python "code/ch09/gru_baseline.py" --variant both --device mps --epochs 30

# 4) 解码冒烟（贪心 vs beam 前 N 句对照）
python "code/ch09/decode.py" --ckpt transformer --n 5

# 5) 评测：三方对照（beam=4/α=0.6 全 test 1000 句）+ 贪心对照
python "code/ch09/eval.py" --compare --greedy

# 6) 外推实测（训练 max_len L=16 的 1.5~2 倍 band=[24,32] 上评 loss/BLEU）
python "code/ch09/eval.py" --ckpt transformer --extrap
python "code/ch09/eval.py" --ckpt gru_noattn --extrap

# 7) 组批对照（9.6 组批税实录：混批 vs 按源句长排序组批；结果入 eval_<tag>.json 的 sorted_by_len 块）
python "code/ch09/eval.py" --ckpt gru_noattn --sort-by-len
```

## 参考数字（M5 Pro / 48 GB，torch 2.13，2026-10 实测；复现以 `log/Book1-ch09/*.json` 为准）

- 数据：train 29000 → L=16 piece 双侧过滤后 17555（60.5%）；val 1014 / test 1000；早期shared BPE-8k探针训练约0.2秒，
  并报告约3.7字符/piece、UNK约1.9%（`Nähe→▁/N/ähe`、`Büsche→▁Bü/sche`）。字符数排除普通空格，UNK数除以piece数，训练侧每10句抽1句并统计英德两侧；该探针的特殊ID与输入打乱设置不同于正式项目，不将其统计直接当作当前正式分词模型的测量值。
- 训练（batch 64，30 epochs = 8250 步，Noam warmup 4000 峰值 lr ≈ 9.9e-4，含逐 epoch 验证）：
  - Transformer（9.42M，非嵌入 7.37M）7.5 s/epoch，总 **224.6 s**（MPS）；val LS-loss 20.87→4.34，
    PPL(piece) 最优 ~26.9（epoch 19，之后轻度过拟合，末 epoch 30.0）。
  - GRU 无 attention（10.89M，非嵌入 8.84M）8.4 s/epoch，总 **255.4 s**；val PPL 末 217.9（过拟合明显）。
  - GRU+Bahdanau（10.17M，非嵌入 8.12M）17.9 s/epoch，总 **537.4 s**；val PPL 末 46.7。
- 评测（beam=4/α=0.6，全 test 1000 句，13a）：

  | 模型 | BLEU(13a) | 桶 ≤7 词 | 桶 8-13 词 | 桶 ≥14 词 | beam 解码 | 贪心 BLEU |
  |---|---|---|---|---|---|---|
  | Transformer | 21.87 | 23.73 | 24.38 | 14.79 | 25.7 s | 20.63（+1.24 增益） |
  | GRU 无 attention | 6.52† | 7.75† | 6.89† | 4.34† | 14.9 s | 6.58（≈0） |
  | GRU+Bahdanau | **23.56** | 23.92 | 26.71 | 15.40 | 31.0 s | 22.22（+1.34） |

  † 混批口径：无注意力版编码器的末态是在处理完批内尾部 PAD 后取得的（旧实现未按有效长度停止递推；当前已改为打包序列），按源句长排序组批的**较少填充口径 18.03**
  （组批税 11.5 BLEU、990/1000 句译文发生变化、译长 6.39→9.94 词，正文 9.6 节；复现 `eval.py --ckpt gru_noattn
  --sort-by-len`，数字落 `eval_gru_noattn.json` 的 `sorted_by_len` 块，总表 `batching_protocol_check.json`）。

  - 手写朴素 BLEU 与 sacreBLEU(tokenize=none) 三模型历史记录在显示精度下差0.0；当前对拍显式统一无平滑、固定四阶，不能以旧语料恰好一致推断所有参数设置等价；13a比none分别高1.19、1.06、1.50 BLEU（标点切分口径教训）。小数据短句域 GRU+attention 反超 mini Transformer（正文9.5–9.6；一次分差不足以归因于架构本身）。
- 外推实测（band=[24,32] piece = 1.5L~2L，L=16；286 拼接对 + 43 天然长句，均值 27.6 piece）：

  | 模型 | test_natural PPL | 外推拼接 PPL | 外推拼接 BLEU | 外推天然长句 BLEU |
  |---|---|---|---|---|
  | Transformer | 25.4 | 97.0 | 9.03 | 6.62 |
  | GRU 无 attention | 155.0 | 462.6 | 2.73 | 0.83 |
  | GRU+Bahdanau | 39.5 | 191.1 | 7.46 | 7.19 |

  正弦 PE 固定、零新参数即可前向 2 倍长度（ch5 F6 兑现）：Transformer 拼接外推 BLEU 腰斩（21.87→9.03）但译文仍成句；
  GRU 无 attention 在长输入上塌缩为短而混乱的译文（历史PAD问题、优化及分布变化也可能参与）。学习式 PE 同协议补测（ch10 网格 learned PE 行、
  18 位查表超长截位复用末位向量，`log/Book1-ch10/learnedpe_extrap.json`）：拼接 8.92 / 天然 7.73，PPL 100.0/172.9
  ——截位复用是额外因素，但尚未单独隔离其影响；在这一长度带上，两者都明显下降（图 9.3 黄点线）。

## 已知口径与取舍

- 训练长度上限 L=16 piece 是**双侧内容过滤**（非截断）——外推实验的「训练 max_len」即此 L；默认band=[1.5L,2L]=[24,32]。若修改L，须同步修改`EXTRAP_BAND`和相关输出标签。
- 主表 checkpoint 取**末个 epoch**（与Table3不做checkpoint平均这一点一致，不由此推定其他checkpoint选择细节相同）。
- PPL按有效目标词元计NLL再取指数；同为piece口径仍须核对分词器、语料和特殊词元计数，不能直接与原论文PPL横比。
- 外推拼接组由测试句不重复两两配对，按源侧内容长度[24,32]筛选；另列43句天然长句。两句拼接不保证语篇连贯，两组也可能共享原始测试句。分位数p99不能证明不存在更长句子。
- 解码不做 K/V 缓存（K/V 缓存账详见 Book3 第 6 章），源侧编码复用、目标前缀整段重算（本书演示策略，不把它当作2017数学结构要求）；无效束槽不参选，top-2k 保 EOS 定稿席位。
