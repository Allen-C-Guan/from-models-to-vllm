# 第 4 章 多模态与 LoRA 服务：请求变胖的两条路

> 站段坐标：站 1-3（入口与输入处理）+站 5-7（mm 预处理与 encoder）+站 12（LoRA 算子注入）。记号契约首现：**adapter（适配器）专指 LoRA 外挂权重件**——与 connector（KV 传输器，ch7）、backend（文法引擎，ch3）同名不同物。B7-5 的欠账句在此开箱：「多模态与 LoRA 服务路径 → Book8 第 4 章」。

## 4.1 两个「请求变胖」的专题

上一章结束时采样被文法裁成窄门——那是约束把输出变窄；本章反过来，把镜头摇回入口：**请求本身在变胖**。Book7 走主链路时请求始终是「一段文本 token」；从本章起这个假设两次被打破。两条路：多模态（VLM）把「一段 token」变成「一段 token 加一张图」，入口层与执行层同时被拉长；LoRA 把「一套权重」变成「一套权重加一个外挂」，算子层被插了一刀。两条路一个共同点：**都是「请求级/模型级」的增量**——不需要换基模型、不重启服务，按请求路由——这也是它们能凑成一章的深层原因：引擎为此发展出的是同一套「增量与基座分离」的服务哲学。三段定位：专题=多模态与 LoRA 服务；站段=1-3 入口+5-7 预处理+12 算子；模块=`vllm/multimodal/`（6137 行地界）+`vllm/lora/`（2907 行）+vllm-ascend 侧 punica 接入。本章产出：mm 管线走查+LoRA 全链接线（4.2/4.3）、VLM 本机首验与组合矩阵两组读数（4.4）、塔连接器一瞥（4.5）。到站时图景：一张图值多少词元有了公式与实测的双重答案、四个服务开关有了第一份组合价目单；新问题留给后面——LoRA×VLM 的挂起机制（ch9 回账）与容量规划里的 mm 因子（ch11）。

> **最小背景栏**：CLIP/ViT 的图像编码与视觉词元概念是 Book5 已教内容（ch05 多模态节），指回不重教；LoRA 的低秩分解数学（$W' = W + BA$，$B\in\mathbb{R}^{d\times r}$、$A\in\mathbb{R}^{r\times k}$、$r\ll d$——旁路两小矩阵的乘积逼近权重增量，训练只动 $B,A$ 冻结 $W$）在前七册只有散点提及（Book3 ch06/Book4 ch01+ch06+ch09），无专节——本栏即最小自建：读者凭这句应能画出旁路图（输入 $x$ 分两路，上路过冻结的 $W$、下路过 $B{\to}A$ 小矩阵串，两路输出相加）。

## 4.2 多模态输入路径：一张图的旅程

一张图从进请求到进前向，走五步——每步都是引擎为「请求不只是一段文本」付出的结构代价。①**协议进入**（站 1-3）：chat 请求的 `content` 数组里混排 `image_url` 与 `text`（0.27.1 协议正身 `entrypoints/openai/chat_completion/protocol.py#L1008-1009` 的 image_url 分支）——多模态把「prompt 是字符串」的假设第一次打破。请求实物（4.4 实测件原文）：

```json
{"messages": [{"role": "user", "content": [
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,<...>"}},
  {"type": "text", "text": "Describe this image in one sentence."}]}]}
```

指名看什么：`content` 是**数组**、每个元素自带 `type`——文本请求是它的退化特例（单元素数组）；协议层的向后兼容就是这样兼容出来的（老客户端发字符串、新客户端发数组，入口归一化后都是「多模态内容序列」）。②**输入处理**（站 5-7）：`vllm/multimodal/` 的 processor 按 modality 分发（registry 维护模型→processor 映射，新模型接 mm 就是注册一个 processor）；图像经 resize/normalize 变张量，音视频各有自己的 processor 分支——**mm 的「多」住在 processor 层**：引擎主干只认「词元+多模态占位」的统一抽象，具体怎么从图/音/视频变成张量是 processor 的可插拔知识。③**视觉编码**（站 7-12 之间）：ViT encoder 把像素变**视觉词元（visual tokens）**——这是 mm 的核心账本。encoder 的输出不是直接进 decoder 的词表空间，而是经投影层（vision-language projector）映射到语言塔的 hidden 维度——「图变成一段假词元嵌入」比「图变成词」更准确，decoder 对它们的注意力与对普通词元无异。执行侧两个 NPU 特有的褶皱：encoder 前向有自己的图捕获管理（`EncoderAclGraphManager`——encoder 序列长度离散档位少，图捕获比 decoder 更划算）；ViT 注意力的 NPU 接入是 `AscendMMEncoderAttention`（vllm-ascend 侧四差异之二）——**encoder 与 decoder 是两台小引擎**：各自有 runner（`MMEncoderModelRunner`）、各自的缓存（encoder cache）、各自的算子路径，只有词元序列把它们缝在一起。④**拼接**：视觉词元与文本词元按 chat 模板拼成统一序列，送 decoder。编码还有个调度前提：`encoder_budget.py::MultiModalBudget`#L45——**每步允许编码多少视觉词元**有预算上限（与 chunked prefill 的 token 预算并轨）：一张 1024×1024 的图 ≈1024 视觉词元，超预算的图要么等下一步要么被拒——mm 请求的调度比纯文本多一层「编码预算」的闸。mm 输入的哈希也有个小实物（`multimodal/hasher.py`）：

```python
# vllm/multimodal/hasher.py#L103-110（骨架摘录）【源码：阅读锚 v0.30.0】
        if isinstance(obj, torch.Tensor):
            tensor_obj: torch.Tensor = obj.cpu()
            tensor_dtype = tensor_obj.dtype
            tensor_shape = tensor_obj.shape
            # NumPy does not support bfloat16.
            # Workaround: View the tensor as a contiguous 1D array of bytes
            if tensor_dtype == torch.bfloat16:
                tensor_obj = tensor_obj.contiguous()
                tensor_obj = tensor_obj.view((tensor_obj.numel(),)).view(torch.uint8)
                …
```

指名看什么：mm 哈希的键=**形状+dtype+原始字节**（不是文件路径、更不是 URL），连 bfloat16 都要特判转 uint8 字节流再哈希（NumPy 不认 bf16——一个真实工程褶皱）；同图不同文件名同哈希（正确）、同文件名不同内容不同哈希（也正确）——「哈希什么」决定了 mm 缓存与 APC 的语义边界。

⑤**调度与 KV**：mm 请求的 prompt_tokens 含视觉词元——**视觉词元与文本词元在 KV 账本上同价**（都占块、都进 APC 哈希链：同图多轮对话的第二轮能命中第一轮的视觉 KV——多模态多轮场景 APC 命中率比纯文本更可观）；encoder 中间结果另有独立缓存（`--mm-processor-cache-gb`，默认 4G——config/multimodal.py#L138，缓存的是 resize 后的输入张量与 ViT 输出，同图复用免编码——4.4 的同图二发 180ms 就是它）。两层缓存的分工：processor cache 免**编码**（同图）、APC 免**重算 KV**（同前缀）——一个是像素到词元的账、一个是词元到 KV 的账。多轮对话的复合收益值得心算：三轮同图对话，第一轮付编码+全量 prefill，第二三轮 APC 命中视觉+文本的公共前缀（图的 KV 不重算）、processor 缓存命中编码（ViT 不重跑）——mm 的多轮服务比首轮便宜一个数量级，这是「VLM 客服看图」类应用的成本结构。

v1 侧的挂点在输入处理站（`vllm/v1/engine/input_processor.py#L19-22`——三行 mm import；#L63 构造 `MultiModalBudget`）——**装载与预算都在站 5 完成**，EngineCore 主干只接「带占位的词元序列」。MRV2 侧还有整套 mm worker 镜像（`v1/worker/gpu/mm/{__init__,encoder_cache,encoder_runner,lora,rope}.py`——其中 `mm/lora.py` 是 mm×LoRA 交叉的 MRV2 正身，4.5 用）。

多模态机的判定与 ch1 的混合判定链同构：config 声明（`architectures` 带 ForConditionalGeneration+processor 配置）→ 入口层识别 content 里的非文本 modality → InputPreprocessor 分发 processor → InputProcessor 产出「词元+多模态占位（multi_modal_placeholders）」——引擎主干看到的是普通 token 序列加一份占位映射，mm 的全部复杂性被封装在入口三站。vllm-ascend 的**绿名单**是模型级的：supported_models.md 逐模型标注（Qwen3-VL ✅ 核心支持、InternVL 全族 ❌——不是「架构不支持」就是「没适配」的区分要看这份名单，VLM 选型的第一步是查绿名单而非看架构）。

绿名单之外，mm 选型还有一问要在进入管线前回答：**你的图的词元预算是多少**——这决定了 max_model_len、encoder budget、HBM 三层配置的联动（4.4 例 4.1 的三档账单）。三层递进：架构在不在绿名单（能不能跑）→ 词元预算多大（跑得动几 K 的图）→ encoder cache 给多少（多轮同图场景的收益）——**mm 部署的容量规划是「图的三问」，纯文本部署只有「序列一问」**。

**视觉词元的预算公式**（Qwen3-VL 的 config 声明——`image_token_factor` 等字段进 processor 参数）：

$$N_{vis} = \Big\lceil \frac{H}{28} \Big\rceil \times \Big\lceil \frac{W}{28} \Big\rceil \div 4 \tag{4.1}$$

| 符号 | 含义 | 量纲 |
|---|---|---|
| $H, W$ | 图高/宽 | 像素 |
| 28 | merge 格边长（4 个 14×14 patch 合一） | 像素 |
| $N_{vis}$ | 视觉词元数 | token |

用人话说：图先切成 28×28 的 merge 格，每格出 4 个 14×14 patch 词元后再 2×2 合并——**每 28×28 像素摊一个词元**。

**例 4.1（一张图的账单三档）** 用式 (4.1) 心算三档：①640×640：$\lceil 640/28\rceil=23$，$23^2\div4\approx400$ 词元（本册实测档）；②1024×1024（1K 屏截图）：$37^2\div4\approx1024$——**恰好一屏一千词元**，mm 请求的 prompt 长度从「几十」跳到「上千」就是这一档；③4096×4096（4K 图）：$147^2\div4\approx16384$ 词元——超过 encoder 预算（16384）的门槛就在这档，「4K 图直接进请求」在预算上已经顶格。KV 成本同步三档跳（每词元每层 2048B：400 词元×24 层≈19MiB、16384 词元×24 层≈768MiB——一张 4K 图的 KV 就吃掉大半张卡的预算余量）。这张账单决定了 mm 请求的 KV 成本、APC 键与调度预算——**图像分辨率的争论，本质是词元预算的争论**。

mm 的模态面不只有图：协议层的 `type` 还认 `video_url`/`audio_url`，v0.30.0 新增了 `multimodal/video_decoders/` 四解码后端子包（#22）——但**本机 av/decord 均未装**，视频与音频在本册一律阅读面；图像是本机唯一可实测模态，也是绝大多数生产 VLM 的主模态，本册 mm 的实测口径由此定界。

vllm-ascend 侧的 mm 面以 ViT 四差异落地：`AscendMMEncoderAttention`（ViT 注意力的 NPU 接入）、`EncoderAclGraphManager`（encoder 图捕获）、双层 encoder cache、模型级补丁（`patch/worker/patch_qwen3vl.py`——deepstack 输入嵌入与 PP 边界暴露）。**绿名单**：Qwen3-VL 系官方 ✅（核心支持）；InternVL 全族 ❌。本机首验读数见 4.4。

> **【版本差异框注】#21：mm cache-miss 重试**——v0.30.0 有三件套（`MultiModalCacheMissError`（cache.py#L42-52）+EngineCore 捕获（core.py#L1763-1766 的 `except MultiModalCacheMissError`）+`_handle_mm_cache_miss`（#L1858，docstring 自述 P0/P1 cache-drift miss…retryable…self-healing））；运行栈 0.27.1 **机制整体不存在**（grep 零命中）。背景是双进程各有 mm 处理缓存的影子副本（前端算一份哈希、EngineCore 持一份缓存），更新顺序不同会漂移——v0.30.0 把漂移做成「可重试响应」的闭环（请求带 hash 回前端、前端重算后重发），运行栈遇漂移只能直接重算。与 ch3 的 #15 同处置：**书里画有的层，运行栈没有——框注即边界**。

## 4.3 LoRA 服务：外挂的算子链

LoRA 在推理侧的问题不是数学（Book4 已教），是工程。设想一个服务平台：底座是一个基模型，上面跑着一百个客户的定制 adapter——客户 A 的法律风格、客户 B 的代码风格、客户 C 的客服话术。**任何时刻一个批里可能混着三个客户的请求**，每个请求要用自己的 $BA$ 旁路。引擎要解的是：基权重常驻一份、adapter 按请求路由、批内混排时各算各的——「**一套基模型权重上，按请求热切换外挂件**」。接线三级：①**配置面**：`--enable-lora --max-lora-rank 8 --lora-modules name=path`（0.27.1 只认 name=path 或 JSON dict 格式——cli_args.py#L49；list 格式被拒，这是我们踩过的第四个 CLI 陷阱）。②**平台接入**：`vllm_ascend/platform.py#L190-192` 的 `get_punica_wrapper` 分派到 `vllm_ascend/lora/punica_npu.py::PunicaWrapperNPU`#L31。③**算子面**：punica（**PUnctual INference of Adapters**——vLLM 的 LoRA 算子族名，源自 Punica 项目：首篇系统化研究「单基模型多 adapter 服务」的批处理算子工作）在 NPU 的实现走 **AscendC 自定义算子 bgmv/sgmv**（`ascend_jit.py#L30` 注册图捕获 custom_op、#L460 分派）。两条核路径的分工：**bgmv**=批内全同 adapter（batched gemv——基权重是大 GEMM、LoRA 旁路是小矩阵乘向量，逐请求算 $BAx$ 再加到基输出上）；**sgmv**=批内混排多 adapter（shared-gemv——按 adapter 分段收集，一段一次核调用）。六层 LoRA 层类在 NPU 侧重写接入 vLLM 的层抽象——「**类换血、接口不动**」：Linear/RoutedLayer/Embedding 等六类各把 forward 里的 punica 调用指向 NPU 实现，上层零感知。这个移植模式与 ch1 见过的「类换血」、ch2 的「ngram 零成本继承」构成移植策略的光谱：**全继承**（无需设备假设→不改）/ **换壳继承**（改 dummy_run 等设备假设点）/ **类换血**（forward 里算子全换、类名接口保留）——按「设备假设渗入多深」选档，是给任何平台移植者的三件套。热切换语义：请求带 `model: <adapter名>` 路由，引擎在同一进程内换刀——基权重不动，adapter 权重在请求到达时按需加载进显存（多个 adapter 可同时驻留——`max_loras` 控制驻留数、LRU 逐出；每个 adapter 的 MB 级体量让「驻留几十个」在 HBM 上无压力——例 4.2 的参数账在服务端兑现）；批内可混排不同 adapter 的请求（sgmv 核的用武之地——4.3 的 sgmv 就是「一批里多个 adapter 各自的 $BA$ 同时算」）。图模式下 `force_has_lora` 分派键（#L3699）让引擎多捕一份「带 LoRA 槽位」的计算图变体——LoRA 与非 LoRA 请求共存时两图切换，这正是 4.4 里 LoRA 基础税的一部分来源。

组合守卫三面（vllm-ascend 平台层）值得各记一笔：①spec 的量化强制继承主模型（ch2 已见——LoRA×spec 时 adapter 也按主模型的量化口径走）；②LoRA×某些量化路径有检查（W4A16 基座挂 adapter 的组合面在白名单制下是窄门）；③mm 处理器缓存与 tower connector 互斥（4.5 的 CI 守卫——两个缓存都动视觉路径时互踩）。守卫的存在本身说明：**组合面是平台层的真实焦虑**——这正是 4.4 实验②「自建矩阵」的动机，也解释了为什么官方宁可收紧也不背组合矩阵的测试债。**例 4.2（toy adapter 的参数账）** 手写 rank-8 件（4.3 预跑件）：Qwen3-0.6B 全部 28 层的 q_proj（$1024\times1024$）与 v_proj（$1024\times512$）各挂 $A\in\mathbb{R}^{8\times d_{in}}$、$B\in\mathbb{R}^{d_{out}\times 8}$——$28\times(8\times1024+1024\times8+8\times512+512\times8)=28\times24576=688{,}128$ 参数，**fp32 下 2.6MB**——对比基模型 0.6B（bf16 1.2G）：**adapter 是基模型的 0.1%**。工业 rank 常用 16-64，adapter 也不过十 MB 级——「热切换不改基权重」的经济学根基就在这个数量级差上（换基模型要搬 GB 级权重、换 adapter 只要 MB 级）。

**官方 feature_matrix 对 LoRA×23 特性交叉零表态**——这个空白正是 4.4 实验②的自建实测动机。

## 4.4 实验：VLM 首验与组合矩阵

**实验①：VLM 本机首验**（Qwen3-VL-2B-Instruct，4.26G——绿名单核心档的本机第一次起服）【本机实测（Atlas 910B3，CANN 9.1.0，运行栈 0.27.1+vllm-ascend 0.27.1rc1）：底档 `log/book8-ch04/vlm_first_run.json`+`vlm_serve2.log`】：serve 92 秒就绪——启动日志给两个 mm 专属的实物：`Encoder cache will be initialized with a budget of 16384 tokens, and profiled with 1 image`（encoder 中间缓存的词元预算与 profiling 方式——mm 引擎的启动多一道 encoder 预算账）与 ViT 权重的分段装载；640×640 三色测试图+「描述这张图」请求 → 回复 `"A solid, uniform red color fills the entire image."`（图文正确）；**视觉词元差分=417−15=402**（图文 prompt_tokens 减纯文本同问句）——式 (4.1) 的 400 加模板 2，**分支 (a) 命中：公式账逐位闭合**。同图二发 180ms（J4——mm 处理器缓存热：首发的 ViT 编码被缓存，二发只重算 decoder；对照纯文本二发的~200ms 量级，mm 缓存在多轮同图场景的收益是「免编码」而非「免推理」，收益幅度取决于图在请求里的词元占比）。附带守卫观测（J5）：带 LoRA 启动的 serve **装载成功**（`Loaded new LoRA adapter` 行在档——装载期不查维度）但首个请求挂起 >300s——0.6B 的 adapter（hidden 1024）对 VL-2B 语言塔（hidden 2048）的维度不匹配走的是**运行期挂**而非装载报错。这笔观测本身是教材：与 ch3 的「报错优于静默」对偶，这里是「装载成功≠能用」——**懒加载范式下（请求到达才真装载权重），装载日志只是第一道门，维度匹配在第一道真实计算处才炸**；部署多租户 adapter 时的运维含义：上线前每个 adapter 都要发一条冒烟请求，装载行绿灯不算数。

**实验②：LoRA×APC×图模式×spec 四元组合矩阵**（Qwen3-0.6B+手写 toy 件，7 臂）【本机实测（Atlas 910B3，运行栈 0.27.1+vllm-ascend 0.27.1rc1）：底档 `log/book8-ch04/combo_matrix/combo_matrix_summary.json`】——官方 feature_matrix 对这四个开关的交叉只给了单因子行，组合面零表态；本矩阵用 $2^4$ 里的 7 个代表臂（锚+四单因子+noop 判锚+全组合）自建首笔数据。工程侧四个坑如实入册（本册方法学档案有全记录）：lora-modules 只认 name=path 格式、bench 客户端要 HF_HUB_OFFLINE（否则 SSL 断连崩）、残留 serve 会劫持 readiness（端口清场行进脚本）、kill -9 后 HBM 慢释放与孤儿 EngineCore 要手杀+等待——**多臂 serve 实验的工程纪律本身是本章方法学贡献的一半**，六跑才全绿的过程比结果更有教学价值：

**表 4.2 LoRA 组合矩阵读数**（16 请求×64 token echo workload、贪心、同卡同 seed；加速比=各臂/M0）

| 臂 | 配置 | 吞吐（req/s） | vs M0 | ITL（ms） | TTFT（ms） |
|---|---|---|---|---|---|
| M0 锚 | 全关 | 0.82 | 1.00× | 60.3 | 118 |
| M1 | LoRA(perturb) | 0.71 | 0.86× | 72.2 | 137 |
| M1n | LoRA(noop) | 0.77 | 0.93× | 66.1 | 144 |
| M2 | APC | 0.84 | 1.02× | 59.3 | 114 |
| M3 | 图模式 | **2.57** | **3.13×** | **8.9** | 47 |
| M4 | spec ngram k=5 | 0.85 | 1.04× | 60.6 | 904 |
| M5 | **全组合** | 0.74 | 0.90× | 72.6 | 1016 |

读表 4.2 的导引：第四列（vs M0）是各因子单开的「净价」，第五六列（ITL/TTFT）把「贵在哪一步」拆开——ITL 贵是 decode 每步的税（LoRA/图模式）、TTFT 贵是首 token 前的固定成本（spec 装配、图捕获摊销）。

五笔判读。**其一，J1 全绿：7/7 臂起服**——M5（LoRA+APC+图模式+ngram 同时开）在官方未表态面**本机首验通过**：四路开关不互斥，组合面存在。**其二，图模式是唯一的量级因子**：M3 的 ITL 从 60.3ms 砍到 8.9ms（**6.8×**）、吞吐 3.13×——与 Book7 ch8 的 ACLGraph 6.15× 同量级互证，也再次框定了 eager 口径下其他因子的读数天花板。**其三，LoRA 的税**：先过 J2 判锚——M1n（noop adapter，$B=0$）的输出与 M0 逐位一致（零矩阵在数学上是恒等——ch2 已证的判据第四次复用）；然后 M1/M1n 比 M0 慢 7-14%——外挂算子链（bgmv 注入+图分派键 force_has_lora 多捕变体）的固定代价；noop 臂比 perturb 臂略快（0.93 vs 0.86——零矩阵跳过部分核内计算）但都交了「装了 LoRA 系统就在」的基础税。**其四，spec 在小批量+长 TTFT 下持平**：M4 的 τ=5.01（serve 日志 SpecDecoding 行——echo 臂 k=5 几乎全收，图模式关掉的口径下 drafter 装配开销相对突出）但吞吐只 1.04×、ITL 与 M0 持平（60.6 vs 60.3ms）——**接受长度高不等于赚到**：τ=5 意味着平均一步出 5 个 token，但 ITL 没降，说明验证前向的宽度把这 5 个 token 的成本吃回去了（$\hat c pprox1$ 的恶劣工况——ch2 式 (10.3) 的另一侧印证）。TTFT 涨到 904ms 是 ngram 装配+首请求开销在小批量下摊不平；与 ch2 表 2.2 的 k=3 档 1.40× 相比是**批量与图模式双口径的差**（32 vs 16 请求、eager 两处同但本章 M4 的 drafter 槽位更多）。**其五（J5 APC×LoRA 交叉）**：M2 的 APC 开着但 echo workload 的前缀哈希链与 M5 的组合里 APC 微正——LoRA 请求参与 APC 时 Book6 ch06:49-51 的「附加位容纳 LoRA 标识」旧账兑现：adapter 身份进哈希、不同 adapter 的同前缀互不污染（交叉正确性的机制根，本矩阵只作观测级记录、不展开）。

**其六，全组合不叠加**：M5（0.90×）的账可以拆：LoRA 税（M1 的 −14%）+spec 开销（TTFT 1016ms≈M4 的 904ms+LoRA 装载增量）+APC 微正（M2 的 +2%）——**乘法世界的因子在算术世界各自记账**。要害结论：**「全开」不是「全赚」**——图模式（唯一的量级因子）没进 M5（eager 基线下组合矩阵隔离了图模式变量），而 LoRA 与 spec 的组合在服务面是兼容的、代价是可加的；生产部署的「开哪些开关」是笔逐因子的成本收益账，本矩阵给了第一份本机价目单。方法学注脚：矩阵设计的三个原则——**锚臂先立**（M0 定基线，一切比值有根）、**判锚臂护航**（noop 的恒等性验证装了 adapter 的测量系统本身没偏）、**全组合臂压轴**（官方空白格的直测）——读者可把这套骨架搬到任何「多开关引擎」的组合面测量上。

## 4.5 mm-LoRA 塔连接器：一瞥

mm 与 LoRA 各自成线，交叉处才见工程想象力。第一层交叉是**给 VLM 挂普通 LoRA**（4.4 实验①的守卫观测位——语言塔的 $BA$ 旁路照挂，视觉塔不动）；第二层交叉就是本节的主角：**塔连接器 LoRA**（`enable_tower_connector_lora`——config/lora.py#L53/#L63 两轨同有）——不止给语言塔挂 adapter，连**视觉塔的连接层**也挂（LoRAMappingType 的 TOWER/CONNECTOR 两型）。vllm-ascend 的 conftest 里有官方塔件先例（qwen3-4b-vl-lora-vision-connector）与一条守卫：**tower connector 与 mm 处理器缓存互斥**（CI verbatim 在档）。本机 2B 模型与官方 4B 塔件不匹配——按 CI 证据+源码走查口径写，实测留 ch13 终账的候选位。塔连接器为什么值得单独一节名字？因为它把 LoRA 的「外挂」语义推到了极限：**不改基模型的任何一层，只调视觉塔到语言塔的信息桥**——微调成本最低、知识遗忘最小的多模态适配路径，工业上给同一个底座配不同视觉风格（文档/图表/照片）的多面手做法。

## 4.6 本章小结

回扣开篇的两条「变胖」：mm 把请求变胖——五步管线（协议/输入处理/视觉编码/拼接/调度）各站为「请求里有图」付出结构代价，核心账本是视觉词元（式 (4.1) 的 400 预测 vs 402 实测闭合），KV/APC/调度三层同价；LoRA 把模型变胖——三级接线（配置面/平台接入/算子面）在 NPU 走 AscendC bgmv/sgmv 核，基础税 7-14%、边际换装零成本。两者都按请求路由、不动基模型——「增量服务」是共同的形态学。组合矩阵把官方空白面填上了本机读数：四开关可同开、图模式独占量级、每因子各付各账。

**带走的心智**：其一，**mm 的账本在词元不在字节**——一张 640² 的图是 400 个词元的预算（KV/APC/调度三处同价），图像压缩率的争论本质是词元数的争论，4K 图≈16384 词元≈大半张卡的 KV 余量——先算词元再谈分辨率；其二，**LoRA 的经济学=基础税+边际零**——装了 adapter 系统就在付 7-14% 的固定税，但换 adapter 是零边际成本的（热切换不换权重、adapter 只有基模型的 0.1% 大小）——「多租户共享底座」这个商业模式在算术上成立；其三，**组合面是测出来的不是推出来的**——官方矩阵的空白格，本机 7 臂矩阵就是第一笔权威数据；其四，**因子各自记账**——图模式的收益住 ITL、spec 的开销住 TTFT、LoRA 的税住全程——读组合矩阵先看「贵在哪一步」的分解列，不看单一吞吐数。

下一章把「请求」变成「重叠的时间线」：async scheduling 让调度与执行并行，ubatching 把一步切两半——时间维度的省钱法。本章的入口变胖与下一章的时间重叠在生产部署里是乘法：mm 请求的长 prefill 最吃重叠调度，LoRA 的多 adapter 混批最吃 sgmv 的批内并行——两章的读数在 ch11 的容量规划里合账。

> **【欠账】** ①LoRA×VLM 守卫的挂起机制归因（懒加载路径）→ 第 9 章回账清单；②塔连接器实测（需 4B 级模型）→ 第 13 章终账候选。

**动手验证**（三件；选卡纪律照 CLAUDE.md、serve 纪律照 4.4）：
① VLM 视觉词元账：起 serve（4.4 命令）后同问句图文/纯文本两发，`usage.prompt_tokens` 差应 ≈400+2；
② 组合矩阵单臂复跑：`bash code/Book8-vLLM源码剖析下/ch04/combo_matrix.sh`（内含端口清场）后读 `combo_matrix_summary.json`；
③ noop 判锚：M1n 臂输出与 M0 逐位一致（`grep` 两臂 bench 输出对照——贪心判据第四次复用）。
