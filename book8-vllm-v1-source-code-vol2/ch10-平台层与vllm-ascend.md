# 第 10 章 平台层与 vllm-ascend：非 CUDA 后端的对照阅读

> 站段坐标：全站地基（平台是引擎与硬件之间的层——所有专题的地板）。本章是「平台收紧家族」的收拢章——ch3/ch5/ch6/ch7 四处埋的伏笔在此合账。

## 10.1 平台是什么：引擎的地基

把 vLLM 想成一栋楼：调度器/执行层/attention 后端是楼层，**平台层是地基**——它替整栋楼回答「我站在什么硬件上、这块地能承多少重、哪些施工方案在这块地上行不通」。具体职责四件：**设备初始化**（NPU/GRAPU 的发现与配置）、**能力声明**（这台设备支持什么——attention 格式/图模式/量化格式）、**参数守卫**（把上游的「默认值」修正为本设备的「安全值」——ch5 的 ubatch 禁用是典型）、**算子接入**（把硬件特有算子注入 vLLM 的抽象接口——punica/attention/quantization 的 NPU 实现在这里挂线）。三段定位：专题=平台层；站段=全站；模块=`vllm/platforms/`（上游抽象）+`vllm_ascend/platform.py`（NPU 实现——vllm-ascend 最大的单文件）。

> **最小背景栏**：Book7 ch2.4 已讲过三级链的骨架（entry_points→NPUPlatform→NPUWorker）；本章在七章专题的背景下重读——每个专题在平台层的投影（收紧/代理/重写）是本章的叙事线。

## 10.2 三级链与七平台

`vllm/platforms/` 仅 7 文件（v0.30.0——metal/TPU 已 OOT 化为外置插件）：`platform.py`（抽象基类 Platform）、`cuda.py`（NVIDIA 默认）、`cpu.py`、`rocm.py`（AMD）、`hpu.py`（Intel）、`tpu.py`/`metal.py`（外置壳——实现在独立包）。vllm-ascend 的三级链（Book7 ch2.4 复验+补全）：`setup.py#L516-529` 的 entry_points 注册 `ascend -> vllm_ascend:register` → `vllm_ascend/platform.py::NPUPlatform#L87-97`（类属性正身——设备名/能力矩阵/守卫集）→ `NPUWorker#L114`（每卡一 Worker 的设备管家）。v0.30.0 增 v2 镜像（`vllm_ascend/worker/v2/model_runner.py`——MRV2 路线的 NPU 接入，继承上游 MRV2 而非重写）。

**表 10.1 七平台对照**（v0.30.0 的 platforms/ 一览+OOT 补充）

| 平台 | 状态 | 特点 |
|---|---|---|
| CUDA | 默认 | 最全功能——其他平台的对齐基准 |
| CPU | 内置 | 无图模式/无 spec——最简实现 |
| ROCm | 内置 | AMD——CUDA 近亲，大部分功能可移植 |
| HPU | 内置 | Intel Gaudi——独立的图模式语义 |
| Ascend | **vllm-ascend 包** | 本章正身——重写面最大的平台 |
| Metal | OOT 外置 | Apple Silicon——独立仓库 |
| TPU | OOT 外置 | Google——PyTorch/XLA 路线 |

## 10.3 平台收紧家族：全谱收拢

前七章散见的「平台收紧上游自由度」案例在此合账——**四个成员+一个代际**：

1. **图模式默认档**（Book7 ch8）：NPU 平台把图捕获默认档设为 FULL_AND_PIECEWISE（上游的 optimization_level O2 推导）——**不收功能、收默认值**（用户显式改高档也行，但默认从安全侧起）。
2. **结构化输出单后端**（ch3 例 3.5）：同引擎混用后端在 NPU 上直接 `VLLMValidationError`——**收的是「引擎内中途换」这个自由度**（四个后端各自仍可用，只要不混）。
3. **ubatching 禁用**（ch5）：`platform.py#L856-867` 强制置 0+告警——**收的是整个功能**（上游有实现但 NPU 上不安全——多流语义与 CUDA stream 假设不匹配）。
4. **OffloadingConnector CUDA-only**（ch7）：不是 vllm-ascend 收的——**上游自己的 `spec.py` 拒了 NPU**（vllm-ascend 无法代理这层）——「上游假设」是最硬的边界。

四案的谱系：从「收默认值」（1）到「收组合」（2）到「收功能」（3）到「上游硬拒」（4）——**收紧的力度递增、可干预性递减**；力度 1-3 是 vllm-ascend 的**主动选择**（以收紧换稳定），力度 4 是**被动承受**（上游假设 CUDA）。给未来平台接入者的启示：力度 1-3 是你的工具箱，力度 4 需要上游改（或如 ch6 的 EPLB 代理——善意谎言翻转标志位）。

## 10.4 NPU 等价路径方法论

八章下来积累的「CUDA 专属→NPU 等价」模式在此收拢——**五个替代口径**的选档表：

1. **继承复用**：上游代码本就设备无关→不改（ch2 的 ngram proposer——CPU numba，与设备无关）。
2. **换壳继承**：改设备假设点（ch2 的 `dummy_run`→no-op——只有图捕获期的设备假设需要改）。
3. **类换血**：类名接口不变、forward 里的算子全换（ch4 的 punica 六类——`bgmv/sgmv` 替换 CUDA punica）。
4. **整件重写**：从头写（ch1 的 `AscendHybridKVCacheCoordinator`——654 行 NPU 版，上游的不动点假设不适用）。
5. **代理绕行**：不改上游、翻转标志位（ch6 的 `_CudaAlikeEplbPlatformProxy`——让上游以为在 CUDA 上）。

选档原则：**优先浅档**（继承>换壳>换血>重写），**代理是最后手段**（善意谎言在接口面会坏——上游新版本可能改那个标志位的语义）。这与 ch4 末尾的「三件套」是同一谱系的两面——三件套讲移植深度、五档表讲替代策略。

## 10.5 patch 体系：vllm-ascend 的暗线

vllm-ascend 对上游的修改不走 fork 路线（不是改上游源码再维护分支），走 **patch 路线**——`vllm_ascend/patch/` 目录下按功能模块化的 monkey-patch 族。这个设计决策的深层含义：**上游可以独立演进**（vllm 发新版不阻塞 vllm-ascend）、**修改是声明式的**（每个 patch 声明改哪个类的哪个方法、原方法存为 `_vllm_ascend_original_*` 前缀属性）、**修改可审计**（patch/__init__.py 的索引是全部修改的注册表——CI 可校验）。patch 的粒度从方法级（`patch_structured_output.py` 的两个方法）到类级（`AscendHybridKVCacheCoordinator` 整件替换）到模块级（`patch_eplb.py` 的平台代理）不等——粒度越大越接近重写、越难跟上游。

**patch 的风险面**：上游改了被 patch 的方法的签名或语义→patch 静默失效或崩溃——这就是 ch3 的 hasher verbatim 与 notes/14 给的版本不一致的根源（**notes 管理员看到的可能是上游版本而引擎跑的是 patched 版本**）。读 vllm-ascend 的 patch 族的方法论：先看 `patch/__init__.py` 的索引（改了什么）→ 再逐个看 patch 文件的 docstring（为什么改）→ 最后看 `_vllm_ascend_original_*` 属性（保留了什么原版）。

## 10.6 本章小结

回扣开篇的地基比喻：平台层替整栋楼承重——设备初始化/能力声明/参数守卫/算子接入四件套；七平台的对照表（v0.30.0 的内置+OOT 分野）给了生态全景；收紧家族四案（收默认值→收组合→收功能→上游硬拒）与等价路径五档（继承→换壳→换血→重写→代理）合成本章的两张速查表——**前七章的每个「NPU 不支持 XX」都能在这两张表里找到位置**。

**带走的心智**：其一，**平台是引擎的合同层**——它与硬件签合同（什么能做什么不能做）、替楼层履约（把「不能」翻译成「怎么绕」）——读平台层是理解引擎可移植性的钥匙；——它与硬件签合同（什么能做什么不能做）、替楼层履约（把「不能」翻译成「怎么绕」）——读平台层是理解引擎可移植性的钥匙；其二，**收紧不是落后是工程决策**——四案收紧的力度谱系里，1-3 档是 vllm-ascend 团队的主动选择（稳定性>自由度），与 CUDA 世界的「全部开放」不是差距而是路线差异。

**表 10.2 vllm-ascend patch 族索引**（按功能域分组——`patch/__init__.py` 索引的骨架）

| 域 | 代表 patch | 改法 | 讲述章 |
|---|---|---|---|
| 调度 | patch_speculative_config | 方法包裹（MTP draft 重映射） | ch2 |
| KV | patch_kv_cache_coordinator | 整类替换（654 行 Hybrid 协调器） | ch1 |
| KV | patch_kv_cache_utils | 函数替换（块大小解析） | ch1 |
| LoRA | punica_npu | 类换血（六类算子接入） | ch4 |
| mm | patch_qwen3vl | 方法包裹（deepstack+PP 边界） | ch4 |
| 结构化输出 | patch_structured_output | 方法包裹（单后端强制） | ch3 |
| 图模式 | 编译器接口 | 模块替换（AscendCompiler） | Book7 ch8 |
| EPLB | _CudaAlikeEplbPlatformProxy | 代理翻转（标志位） | ch6 |
| 量化 | quantization/ 子包 | 模块级（算子族接入） | ch8 |

读法：**patch 索引是 vllm-ascend 与上游的全部差异面**——每行一个「上游这样、NPU 需要那样」的声明；行的密度分布告诉你 NPU 与 CUDA 的差距集中在哪（KV/算子/图模式三个域最密）。

给读者的实操建议一条：判断某个 vLLM 功能在你的 NPU 环境能不能用，不要只查官方 feature_matrix（那说的是 CUDA 面的可用性），要查三层——①vllm-ascend 的 `supported_features.md`（NPU 面的声明）、②`platform.py` 的守卫代码（实际的收紧/禁用/代理——本章的四案谱系）、③一次快速冒烟（起 serve 后发一条请求——一分钟定谳）。三层各回答不同精度的问题：文档说「应该」、代码说「确实」、冒烟说「现在」——**从「应该」到「现在」的三级验证是部署前的标准动作**。这个方法论不只适用于 vLLM——适用于任何「上游声明+平台适配+运行时验证」的分层系统。


### 10.5 平台层的接口面：NPUPlatform 的能力声明

NPUPlatform 的类属性是**能力声明**的源码正身——每个属性对应一个「这台设备能做什么」的回答。关键属性走一遍（`vllm_ascend/platform.py` 的类属性段）：`_platform_name = "ascend"`（身份声明——引擎在启动时据此分派平台相关逻辑）；`_device_name = "npu"`（设备标识——tensor 的 .device 检查用）；`device_type = "npu"`（torch 层的设备类型——与 cuda/cpu/xpu 同级的枚举值）；`supported_quantization`（ch8 的白名单六法——此属性的 list 就是进门检查的全部依据）；`availability_message`（设备不可用时的提示信息——用户看到的第一行「为什么不支持」）；还有 `device_interface`/`communication_backend` 等与分布式相关的属性（ch6 的 HCCL 就是这里声明的）。**这些属性是引擎的「简历」**——上游代码在运行时逐个查这些字段来决定走哪条分支：`if platform.is_cuda: 走 CUDA 核`/`elif platform.is_npu: 走 NPU 核`——**引擎的全部平台相关行为都从这些类属性的分派开始**。

### 10.6 守卫的实现模式

四个收紧案例的代码模式值得统一走读——它们的实现结构是同一个模板的三种变体。**模板**：在平台层的 `check_and_update_config`（或类似的生命周期钩子）中，检查某个配置值是否与本设备兼容，不兼容则修改+告警。**变体一：改默认值**（图模式）——不禁止功能、只改起点（`optimization_level` 的 O2→FULL_AND_PIECEWISE 映射）；**变体二：禁用功能**（ubatching）——把开关强制关（`enable_dbo = False`+`ubatch_size = 0`），用户显式开也不行；**变体三：拒绝组合**（结构化输出单后端）——不禁止单开、禁止同引擎混开（首请求定后端后，后续不一致直接抛错）。三变体的代码量递增（一行改值→三行改值+告警→方法包裹+状态跟踪），**复杂度来自「要跟踪的状态量」**——改值不需要记忆、禁用需要记住「已禁」、拒绝组合需要记住「已选了谁」。

### 10.7 vllm-ascend 的 patch 体系详走

`vllm_ascend/patch/` 目录是 vllm-ascend 对上游的全部修改的**注册表**——设计哲学是「**不改上游源码、运行时打补丁**」。这个决策的得与失值得分析。

**得**：上游可以独立发版（vllm 发新版不阻塞 vllm-ascend 适配）；修改是声明式的（每个 patch 一个文件、入口在 `__init__.py` 索引——`grep` 一遍就知道改了什么）；修改可逆（`_vllm_ascend_original_*` 前缀保存了原版方法——可随时取消 patch 恢复原行为）。

**失**：patch 与上游版本紧耦合——上游改了被 patch 的方法签名或语义，patch 可能静默失效（不报错但行为变了）或显式崩溃（AttributeError）；**调试困难**——traceback 显示的是 patched 方法名，开发者需要知道 patch 链才能理解调用流；**文档与代码的漂移**——上游文档描述的是原版行为、NPU 用户遇到的是 patched 行为（`doc-code drift`——ch4 的 hasher verbatim 勘误正是这个问题的实例）。

**patch 索引的全貌**（`patch/__init__.py`）按功能域分组——每个域的 patch 数量告诉你 NPU 与 CUDA 的差距集中在哪：**KV 管理**最密（patch_kv_cache_coordinator 654 行 + patch_kv_cache_utils + related）——KV 是引擎最核心的子系统、也是平台差异最大的面（块大小/混合组/Hybrid 协调器的行为全不同）；**LoRA 算子**次之（punica_npu 类换血六件——CUDA punica 与 NPU AscendC bgmv/sgmv 的算子级差异）；**调度/图模式/量化**各有几件。**patch 密度分布=平台差距分布的地图**——新平台接入者看这张地图就知道要在哪些域花最多力气。

### 10.8 「善意谎言」代理的伦理

`_CudaAlikeEplbPlatformProxy` 值得单独一节讨论——它把 `is_cuda` 标志翻转给 EPLB 子系统看，让上游的 CUDA 路径在 NPU 上走起来。这个模式的**适用条件**很窄：上游代码只是检查标志位做分派、实际调用的算子经平台抽象层已经会落到 NPU 版本。**如果上游的 CUDA 路径里有直接的 CUDA API 调用**（如 `torch.cuda.synchronize`），代理就翻不动了——**代理能翻标志位、翻不了 if 分支里的硬编码**。所以代理只是「从禁用到能用」的桥，不是「从能用到跑好」的终点——跑好还是需要原生的 NPU 路径（正道是给上游提 PR 让它支持 NPU——这也是 vllm-ascend 团队在做的事）。

### 10.9 本章的选型含义

给读者的实操建议一条：判断某个 vLLM 功能在你的 NPU 环境能不能用，不要只查官方 feature_matrix（那说的是 CUDA 面的可用性），要查三层——①vllm-ascend 的 `supported_features.md`（NPU 面的声明）、②`platform.py` 的守卫代码（实际的收紧/禁用/代理——本章四案谱系）、③一次快速冒烟（起 serve 后发一条请求——一分钟定谳）。三层各回答不同精度的问题：**文档说「应该」、代码说「确实」、冒烟说「现在」**——从「应该」到「现在」的三级验证是部署前的标准动作。这个方法论不只适用于 vLLM——适用于任何「上游声明+平台适配+运行时验证」的分层系统。
### 10.9 vllm-ascend 的 patch 体系详走

`vllm_ascend/patch/` 目录是 vllm-ascend 对上游的全部修改的注册表——设计哲学是「不改上游源码、运行时打补丁」。这个决策的得与失值得分析——**得**：上游可以独立发版（vllm 发新版不阻塞 vllm-ascend 适配）；修改是声明式的（每个 patch 一个文件、入口在 `__init__.py` 索引——grep 一遍就知道改了什么）；修改可逆（`_vllm_ascend_original_*` 前缀保存了原版方法——可随时取消 patch 恢复原行为）。**失**：patch 与上游版本紧耦合——上游改了被 patch 的方法签名或语义，patch 可能静默失效（不报错但行为变了）或显式崩溃（AttributeError）；调试困难——traceback 显示的是 patched 方法名，开发者需要知道 patch 链才能理解调用流；文档与代码的漂移——上游文档描述的是原版行为、NPU 用户遇到的是 patched 行为（ch4 的 hasher verbatim 勘误正是这个问题的实例）。

patch 索引的全貌按功能域分组——**KV 管理最密**（patch_kv_cache_coordinator 654 行 + patch_kv_cache_utils + related）——KV 是引擎最核心的子系统、也是平台差异最大的面（块大小/混合组/Hybrid 协调器的行为全不同）；**LoRA 算子次之**（punica_npu 类换血六件——CUDA punica 与 NPU AscendC bgmv/sgmv 的算子级差异）；调度/图模式/量化各有几件。**patch 密度分布=平台差距分布的地图**——新平台接入者看这张地图就知道要在哪些域花最多力气。

### 10.10 「善意谎言」代理的伦理

`_CudaAlikeEplbPlatformProxy` 把 `is_cuda` 标志翻转给 EPLB 子系统看——让上游的 CUDA 路径在 NPU 上走起来。适用条件很窄：**上游代码只是检查标志位做分派、实际调用的算子经平台抽象层已经会落到 NPU 版本**。如果上游的 CUDA 路径里有直接的 CUDA API 调用（如 `torch.cuda.synchronize`），代理就翻不动了——代理能翻标志位、翻不了 if 分支里的硬编码。所以代理只是「从禁用到能用」的桥，不是「从能用到跑好」的终点——跑好还是需要原生的 NPU 路径（正道是给上游提 PR 让它支持 NPU——这也是 vllm-ascend 团队在做的事）。

### 10.11 本章的运维手册

给读者的实操建议：判断某个 vLLM 功能在你的 NPU 环境能不能用，不要只查官方 feature_matrix（那说的是 CUDA 面的可用性），要查三层——①vllm-ascend 的 `supported_features.md`（NPU 面的声明）、②`platform.py` 的守卫代码（实际的收紧/禁用/代理——本章四案谱系）、③一次快速冒烟（起 serve 后发一条请求——一分钟定谳）。三层各回答不同精度的问题：**文档说「应该」、代码说「确实」、冒烟说「现在」**——从「应该」到「现在」的三级验证是部署前的标准动作。这个方法论不只适用于 vLLM——适用于任何「上游声明+平台适配+运行时验证」的分层系统。

### 10.12 平台层的接口面详走

NPUPlatform 的类属性是**能力声明的正身**——每个属性回答一个「这台设备能做什么」的问题。走一遍关键属性（`vllm_ascend/platform.py`）：`_platform_name = "ascend"`——身份声明，引擎启动时据此分派平台相关逻辑；`_device_name = "npu"`——设备标识，tensor 的 .device 检查用；`supported_quantization`——ch8 的白名单六法，此属性的 list 就是进门检查的全部依据；`device_interface`/`communication_backend`——与分布式相关（ch6 的 HCCL 声明在此）。这些属性是引擎的**简历**——上游代码在运行时逐个查这些字段来决定走哪条分支。理解了「引擎的全部平台相关行为都从这些类属性的分派开始」，就理解了平台层的接口面设计。

### 10.13 守卫的代码模板

四个收紧案例的实现结构是同一个模板的三种变体。**模板**：在平台层的 `check_and_update_config` 生命周期钩子中，检查某个配置值是否与本设备兼容，不兼容则修改+告警。**变体一改默认值**（图模式）：不禁止功能、只改起点——一行代码。**变体二禁用功能**（ubatching）：把开关强制关+告警——三行。**变体三拒绝组合**（结构化输出单后端）：方法包裹+状态跟踪——几十行。三变体的代码量递增——**复杂度来自「要跟踪的状态量」**：改值不需要记忆、禁用需要记住已禁、拒绝组合需要记住已选了谁。这个模板是给新平台接入者的直接参考——你的平台需要哪种收紧，就选对应的变体。

### 10.14 给新平台接入者的完整路径图

读完本章你应该能画出一张「新平台从零接入 vLLM」的路径图——**第一站**：fork vllm/platforms/ 模板写你的 Platform 类（身份声明+能力矩阵+设备初始化）；**第二站**：对照 CUDA 实现逐个填能力（哪些 attention 格式支持、图模式什么档位、量化白名单列哪些格式）；**第三站**：写守卫（本章四案谱系的变体选择——你的设备有什么限制就写什么收紧）；**第四站**：接算子（ch4 的类换血模式——把你的硬件算子接到 vLLM 的抽象接口）；**第五站**：建 patch 体系（10.9 的模式——不改上游、运行时补丁）；**第六站**：验冒烟（起 serve+发请求——从「应该能跑」到「确实能跑」）。六站走完你的平台就活了——**vllm-ascend 走过的正是这六站**（只是每站都走得更深）。

给新平台接入者的最后一句心法：**vllm-ascend 走过的六站路径（Platform 类→能力矩阵→守卫→算子接入→patch 体系→冒烟验证）是一张通用地图**——不只适用于 NPU，适用于任何想接入 vLLM 的新硬件（或想接入任何分层引擎的新平台）。六站的核心挑战各不同：第一站是**身份**（你是谁）、第二站是**能力**（你能做什么）、第三站是**边界**（什么不能做要怎么收紧）、第四站是**算子**（底层计算怎么接）、第五站是**修改**（上游的 CUDA 假设怎么绕）、第六站是**验证**（从应该能跑到确实能跑）。六站走完，你的平台就活了——然后开始长专题（vllm-ascend 在四年间从零到九个专题深做，走的就是这条路的完整版）。

### 10.15 平台层的测试面

vllm-ascend 的 CI 体系值得一看——它解决了「patch 改了上游行为，怎么保证不 break」的问题。CI 分三层：**单元测试**（每个 patch 配套的 UT——ch3 的 patch_structured_output 有 7 个 UT 验证单后端强制的行为）；**集成测试**（conftest 里的 fixture 环境跑端到端冒烟——起 serve+发请求+断言输出）；**功能矩阵测试**（supported_features.md 的自动化核对——声明的功能逐个跑通）。三层的自动化保证了 patch 的质量底线。**给读者的启示**：如果你在给 vLLM 写一个 NPU patch，请同时写它的 UT——否则上游一发版你的 patch 可能静默坏了都没人知道。

**本章的最终定位**——它不是讲「vllm-ascend 怎么用」（那是使用手册的事）也不是讲「vLLM 的平台抽象多优雅」（那是设计文档的事），而是讲**「一个工业引擎怎么在异构硬件的现实里活下来」**——vllm-ascend 的每一行 patch、每一个守卫、每一次收紧，都是在回答这个问题。读者读完本章应该获得的不是 vllm-ascend 的知识细节，而是**「在任何分层系统里，适配层是怎么工作的」的通用判断力**——这个判断力在换引擎、换硬件、换框架时原样可用。

回到开篇的比喻——平台层是引擎的地基。读完本章你应该能回答三个问题：**这台引擎站的地基是什么**（NPUPlatform 的能力声明与守卫体系）；**地基跟楼层的接口在哪**（三级链+patch 索引+算子接入）；**如果地基换了（新硬件接入）需要改什么**（六站路径图）。三个答案合在一起就是「平台层的完整心智」——引擎的可移植性不是魔法，是工程——每个平台层决策（收紧/代理/重写）都是在特定约束下的理性选择。

平台层是引擎的翻译官——它把硬件的方言（NPU 的算子接口/图模式语义/通信协议）翻译成引擎的普通话（统一的 Platform 抽象），让楼层的每个模块不需要知道底下站的是什么硬件。**翻译的质量决定了引擎的可移植性**——好的翻译让新硬件接入只需要改平台层（六站路径）、差的翻译让每个模块都要改（不可持续）。vllm-ascend 的实践给了一个好的范本。下一章从这个地基升到天空——引擎选型，俯瞰四台引擎的全景并学习怎么为自己的负载做选择。的最后一站：引擎选型与服务设计——四台引擎各适合谁、SLO 怎么换算成卡。

> **【欠账】** ①OOT 平台的独立仓库版（metal/TPU）——远期对照阅读位；②v2 镜像的完整对照（MRV2 NPU 路线的功能面）→ 终校差异附录。

**动手验证**（两件）：
① `python -c "from vllm_ascend.platform import NPUPlatform; print(NPUPlatform._platform_name)"`——平台正身的一眼验证；
② `grep -n "def.*seal\|platform_name\|device_name" repos/vllm-ascend/vllm_ascend/platform.py | head -8`——三级链中间层的类属性正身。
