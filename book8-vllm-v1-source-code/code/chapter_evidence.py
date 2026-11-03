"""全量续写的证据注册；原三章样稿证据保持独立。"""
SOURCES={
 'ch01-client':('vllm','vllm/v1/engine/core_client.py',121,129,{
   121:'上下文：异步多进程分支已在上文处理；这里检查同步多进程配置。',
   122:'构造 SyncMPClient，将配置与执行器类交给它；不是返回 EngineCore 本体。',
   129:'剩余有效配置走 InprocClient；同一个客户端契约可以由不同部署实现。'}),
 'ch02-config':('vllm','vllm/v1/engine/llm_engine.py',175,191,{
   176:'把入口参数解析成有效系统配置；模型、调度、并行等约束在这里形成关联。',
   177:'按已经解析的配置选出执行器类，暂时还不是一个已创建的执行器实例。',
   179:'环境变量可以启用前端／核心多进程；不能只看调用参数默认值判断部署。',
   184:'用有效配置与所选类构造 LLMEngine，后续协作者在构造过程中装配。',
   190:'此标志决定客户端形态；离线入口仍可能采用多进程。'}),
 'ch02-core':('vllm','vllm/v1/engine/core.py',133,149,{
   134:'实例化执行器；本例的 UniProcExecutor 会继续创建 Worker、初始化设备并加载模型。',
   136:'若调用者提供执行失败回调，将其登记到执行器，准备传播运行故障。',
   139:'容量初值 -1 表示尚未确定，不能拿它参与调度。',
   141:'弹性扩容有专门准备路径；本例不开启。',
   145:'模型已经存在，才能取得 KV 需求并准备实际资源；返回的是调度侧 KV 配置。',
   146:'创建结构化输出管理对象，不意味着每个请求都启用语法约束。',
   149:'从配置选择 Scheduler 类；其实例还要使用刚准备的 KV 配置。'}),
 'ch02-worker':('vllm','vllm/v1/executor/uniproc_executor.py',68,75,{
   68:'通过 WorkerWrapperBase 装配实际 Worker 类型；平台会影响这个实现选择。',
   69:'先建立设备与相关执行环境，再加载模型。',
   71:'弹性 EP 扩容另有调用方式；普通启动进入 else 分支。',
   74:'委托 Worker 加载模型；权重加载机制在第 10 章打开。',
   75:'所选平台与后端还可能约束块大小；用户请求值不是最终有效值的全部来源。'}),
 'ch02-layout':('vllm','vllm/v1/engine/core.py',319,333,{
   319:'结合模型的 KV spec 与各 Worker 可用字节，求出各执行侧的 KV 配置。',
   322:'为生成的配置附上已经确定的物理布局。',
   328:'容量推导可能反过来影响最大模型长度，所以比较推导前后的值。',
   329:'仅当自动适配改变长度时同步 Worker，防止核心与执行侧约束不一致。',
   332:'由执行侧配置生成调度侧资源视图，两者用途不同。',
   333:'把有效块数量回写配置，后续观察应读取这个已解析的值。'}),
 'ch02-ready':('vllm','vllm/v1/engine/core.py',352,356,{
   352:'校验最终块大小，尽早拒绝不满足约束的组合。',
   354:'让执行侧据配置真正准备 KV 等资源，而不是只有一份描述。',
   355:'普通启动继续编译或预热；eager 关闭图执行，不代表完全没有预热。'}),
}

BASE='drafts/Book8-vLLM源码剖析/evidence/baseline-inproc-traced/manifest.json'
DATA={
 'ch01-types':(BASE,['effective.client','effective.scheduler','effective.executor','effective.worker','effective.runner','effective.balance_enabled']),
 'ch02-effective':(BASE,['requested_settings.max_model_len','requested_settings.max_num_seqs','requested_settings.max_num_batched_tokens','requested_settings.kv_cache_memory_bytes','requested_settings.enforce_eager','effective.async_scheduling','effective.use_v2_model_runner','effective.block_size','effective.num_gpu_blocks']),
}

SOURCES.update({
 'ch06-pool':('vllm','vllm/v1/core/block_pool.py',177,194,{
  178:'为每个编号创建 CPU 元数据对象；这里没有创建 K/V 张量。',184:'空闲队列保存可取用的块，开启缓存时也容纳可淘汰候选。',187:'哈希索引与空闲队列回答不同问题：内容在哪里，以及哪里可以再分配。',193:'取走编号 0 作为特殊占位块，不能继续按普通空闲块分配。'}),
 'ch06-capacity':('vllm','vllm/v1/core/kv_cache_manager.py',546,563,{
  546:'协调器按各组策略合计本次需要消耗的可用块，包含需要保留的命中块。',559:'扣除给其他在途序列预留的块；总空闲量不一定就是本请求可用量。',560:'接纳水位也进入约束，不能仅比较新增 token 数。',561:'资源不足以 None 返回，让调度器决定等待或抢占。'}),
 'ch06-grow':('vllm','vllm/v1/core/single_type_kv_cache_manager.py',393,403,{
  393:'上下文：前面的可选部分命中 CoW 已处理；读取该请求已有的组内块表。',394:'把需要覆盖的 token 总位置数向上取整为块数。',395:'已有块继续使用，只补差额。',396:'没有差额仍可能返回前面产生的 CoW 块；普通无 CoW 路径为空列表。',399:'从物理编号池取新块，并把它们接到长期块表尾部。',401:'需要执行侧清零时，另记新编号供其消费。',403:'返回此次新增块，不返回整个请求的块表。'}),
 'ch06-free':('vllm','vllm/v1/core/block_pool.py',778,800,{
  779:'把优先复用与最后复用的块分开收集。',783:'若块属于其他池，转交所属池处理，避免修改错误的空闲账本。',786:'释放一份引用；只有最后一份普通引用消失时才进入空闲队列。',788:'无哈希或关闭缓存的块优先复用。',792:'带缓存身份的块放到后部，给之后的前缀命中保留机会。',796:'队首加入优先候选。',798:'队尾加入缓存候选，释放并不在这里删除缓存身份。'}),
 'ch07-hash':('vllm','vllm/v1/core/kv_cache_utils.py',659,665,{
  659:'首块没有父块时使用初始化的 NONE_HASH。',662:'把本块 token 固定为元组，形成稳定的输入结构。',663:'哈希输入同时包含父链、本块 token 和额外身份，不能只比较这一小段文本。'}),
 'ch07-hit':('vllm','vllm/v1/core/kv_cache_manager.py',286,300,{
  286:'请求或配置禁用读取缓存时，返回各组空块和零命中。',295:'最多复用到最后一个 token 之前，仍需一次计算取得后续生成所需的 logits。',296:'让协调器寻找所有必要状态组能够共同承认的前缀；本章只展开单组全注意力。'}),
 'ch07-touch':('vllm','vllm/v1/core/block_pool.py',757,764,{
  757:'逐个处理本次复用的块；调用者正在把它们挂到新请求。',760:'零引用块原本可被回收，现在命中后先移出空闲队列。',762:'增加引用，使新请求的使用关系进入账本。'}),
 'ch07-preempt':('vllm','vllm/v1/core/sched/scheduler.py',1490,1518,{
  1490:'调用者已把请求移出 running；这里要求它此前确实处于 RUNNING。',1493:'释放请求的 KV 引用与 encoder 缓存，并撤销在途 prefill 登记。',1496:'请求仍然存在，但计算进度重置；已生成 token 历史没有在此清空。',1498:'草稿 token 不是已确认历史，抢占时丢弃。',1507:'异步路径额外标记旧输出：不能用旧返回再次推进重置后的计数。',1510:'记录仍在途的旧输出数量，后续回填负责消解。',1512:'留下累计抢占次数供统计和恢复逻辑使用。',1517:'插回等待队列，并通知执行侧清理旧请求状态。'}),
})
KV='drafts/Book8-vLLM源码剖析/evidence/kv-metadata/trace.json'
DATA.update({
 'ch06-blocks':(KV,['allocation.initial.num_free_blocks','allocation.initial.free_order','allocation.allocate_A.returned_block_ids','allocation.allocate_B.req_to_blocks','allocation.free_A.free_order','allocation.allocate_C.req_to_blocks']),
 'ch07-cache':(KV,['prefix.A_released.blocks','prefix.A_released.free_order','prefix.lookup_A_again.hit_block_ids','prefix.lookup_A_again.num_hit_tokens','prefix.attach_hit_and_allocate_tail.returned_block_ids','prefix.attach_hit_and_allocate_tail.req_to_blocks','prefix.attach_hit_and_allocate_tail.free_order']),
})

SOURCES.update({
 'ch08-call':('vllm','vllm/v1/executor/uniproc_executor.py',119,132,{
  119:'单进程执行器实现统一执行契约，参数是本轮 SchedulerOutput。',122:'把方法名和参数交给执行器自己的分发接口；single_value 避免返回一元素列表。',129:'若返回的 Future 已完成，立即取结果以提前暴露异常。'}),
 'ch08-future':('vllm','vllm/v1/executor/uniproc_executor.py',102,117,{
  102:'同步接口直接取得结果；异步输出包装需要在此解析。',108:'non_block 分支仍先在当前调用中运行 Worker 方法，不在这里创建后台线程。',110:'若执行侧返回延迟输出，用专门 Future 延后解析。',112:'普通返回值放进已完成 Future，统一上层消费方式。',114:'异常也进入 Future，调用者取得结果时可重新抛出。'}),
 'ch08-worker':('vllm','vllm/v1/worker/gpu_worker.py',1209,1222,{
  1209:'本节展示上游 GPU Worker；前面的 PP 接收准备已经完成。',1210:'把计划与可选中间张量交给 Runner，Worker 不重新决定调度顺序。',1213:'MRV2 pooling 模型有独立收尾；A/B 是生成模型，不进入此分支。',1219:'返回生成结果、延迟结果或 None；其他输出在下文按 PP 中间张量处理。'}),
 'ch09-row':('vllm','vllm/v1/worker/gpu/states.py',99,115,{
  99:'活跃请求必须先取得一个持久槽位，槽位耗尽不能继续登记。',100:'从空闲行尾部取号，所以教学配置中最先取得的不是必然为零。',101:'维护 ID 到行号及反向映射。',104:'在这行写入请求的长度约束。',106:'prefill_len 包括本次交给 Runner 的历史，恢复请求时可以长于原 prompt。',111:'大状态通过暂存写入描述更新；实际应用发生在 apply_staged_writes。',113:'保存计算进度，CPU 与设备侧状态分别更新。'}),
 'ch09-gather':('vllm','vllm/v1/worker/gpu/model_runner.py',1234,1249,{
  1234:'上下文：普通真实批次，已取得计划的请求数与工作量。',1236:'先决定本轮请求顺序，不把持久行号直接当 batch 顺序。',1240:'按相同顺序收集每条请求的工作量。',1243:'将本轮 req_id 映射回持久状态行，这是两套索引之间的桥。',1245:'使用映射收集 prefill 状态，避免读取同下标但属于其他请求的数据。'}),
 'ch09-offset':('vllm','vllm/v1/worker/gpu/model_runner.py',1342,1352,{
  1344:'当步描述可能要求补齐请求数；eager 教学路径不依赖图补齐。',1345:'预先准备边界数组，首元素为零。',1347:'每条请求的本轮 token 数做前缀和，得到拼接批次的分段边界。',1350:'填充尾部使边界非递减，不能留下任意旧值给后端。',1352:'复制到执行输入缓冲区；真正有效范围在后文裁出。'}),
 'ch09-delta':('vllm','vllm/v1/worker/gpu/model_runner.py',1185,1193,{
  1185:'旧请求更新按 ID、计算进度、新块编号并行遍历。',1188:'先通过 ID 找持久行，不使用这次循环下标当行号。',1189:'更新该请求的 CPU 进度镜像。',1190:'已有完整块表覆盖更新时不再追加；普通增量只追加新编号。',1192:'overwrite=False 保留此前块表。'}),
})
RUN='drafts/Book8-vLLM源码剖析/evidence/runner-inproc-traced/projection.json'
DATA.update({
 'ch08-contract':(RUN,['steps.1.execute.plan.num_scheduled_tokens','steps.1.execute.plan.total_num_scheduled_tokens','steps.1.execute.non_block','steps.1.returned.returned_type']),
 'ch09-index':(RUN,['steps.1.registered.req_id_to_index','steps.1.inputs.batch.req_ids','steps.1.inputs.batch.idx_mapping.values','steps.3.registered.req_id_to_index','steps.3.inputs.batch.req_ids','steps.3.inputs.batch.idx_mapping.values']),
 'ch09-input':(RUN,['steps.1.inputs.batch.input_ids','steps.1.inputs.batch.positions','steps.1.inputs.batch.query_start_loc','steps.1.inputs.batch.seq_lens','steps.1.inputs.batch.logits_indices']),
 'ch09-decode':(RUN,['steps.2.inputs.batch.input_ids.values','steps.2.inputs.batch.positions.values','steps.2.inputs.batch.query_start_loc.values','steps.2.inputs.batch.seq_lens.values']),
})

SOURCES.update({
 'ch10-create':('vllm','vllm/model_executor/model_loader/utils.py',46,61,{
  46:'未显式给模型配置时，使用系统配置中的模型部分。',48:'从模型架构描述解析实现类，和读取权重文件是不同步骤。',51:'量化配置先与模型类适配，普通 BF16 案例不进入此分支。',56:'现代模型构造约定接收 vllm_config 与层名前缀。',58:'在当前配置上下文中创建模块，并记录后续重载所需元数据。'}),
 'ch10-map':('vllm','vllm/model_executor/models/qwen2.py',324,333,{
  324:'Qwen3Model 继承并使用这份权重映射。',327:'q/k/v 三份 checkpoint 权重写入同一 qkv 参数的不同 shard。',330:'gate/up 同样进入融合参数，以 shard 0/1 区分。'}),
 'ch10-layer':('vllm','vllm/model_executor/models/qwen3.py',232,246,{
  233:'第一层尚无独立 residual 时先保存输入；后续层使用双输入归一化。',237:'融合残差与归一化返回两份状态，残差没有被省略。',238:'Attention 子模块取得位置与归一化后的隐藏状态。',244:'MLP 前再次处理残差与归一化，随后执行前馈网络。',246:'把主分支与 residual 一起交给下一层。'}),
 'ch10-qkv':('vllm','vllm/model_executor/models/qwen3.py',158,170,{
  158:'一次融合投影生成 Q/K/V，再按 query 与 KV 各自宽度拆开。',161:'按 head_dim 组织 Q，使归一化作用于每个头的最后一维。',164:'K 同样按其头数归一化；Q 与 K 的总宽度不必相同。',167:'按各 token 的序列位置应用旋转位置编码。',168:'通用 Attention 接口消费 Q/K/V，后端实现留下一章。',169:'输出投影把 query 头总宽度映射回 hidden_size。'}),
 'ch11-impl':('vllm','vllm/model_executor/layers/attention/attention.py',416,429,{
  416:'Backend 提供实现类；不是在这里启动 Attention 计算。',417:'按模型头数、维度、KV类型等构造实现对象，保存到 Attention.impl。'}),
 'ch11-forward':('vllm','vllm/model_executor/layers/attention/attention.py',770,787,{
  770:'显式依赖用于保持 KV 写入与读取的编译顺序；不是数学计算输入。',774:'用层名取当前前向上下文，补齐调用签名没有显式传入的状态。',777:'委托所选实现，将 Q/K/V、真实 KV 张量、元数据和输出缓冲交给同一契约。'}),
 'ch11-write':('vllm','vllm/v1/attention/backends/flash_attn.py',1451,1472,{
  1451:'本段是上游 FlashAttention 更新路径，不是本机 Ascend 布局实测。',1454:'按此后端布局取得 K/V 缓存视图，不能复制此布局假设到所有平台。',1463:'调用缓存写入算子，slot_mapping 决定本轮 token 对应的写入位置。',1468:'写入位置来自当步元数据，块管理器只提供编号，不在这里代算 K/V。'}),
})
ASSETS='drafts/Book8-vLLM源码剖析/evidence/model-assets/metadata.json'
DATA.update({
 'ch10-config':(ASSETS,['model_revision','config.architectures','config.hidden_size','config.head_dim','config.num_attention_heads','config.num_key_value_heads','config.num_hidden_layers','config.intermediate_size','config.vocab_size']),
 'ch10-hidden':(RUN,['steps.1.hidden.hidden','steps.2.hidden.hidden']),
 'ch11-slot':(RUN,['steps.1.attn.block_tables','steps.1.attn.slot_mappings','steps.2.attn.slot_mappings']),
})
DESCRIPTIONS={'ch10-config':'固定模型快照的配置字段投影；这是文件检查，不是执行期间张量观测。'}

SOURCES.update({
 'ch12-logits':('vllm','vllm/v1/worker/gpu/model_runner.py',1533,1545,{
  1533:'普通未分片 logits 分支；上文另有分片路径。',1534:'按每条请求需要采样的位置抽取隐藏行，本例 [11,1024] → [2,1024]。',1535:'语言模型输出头把选中隐藏行投影到词表，本例得到 [2,151936]。',1537:'若存在结构化输出约束，先在 logits 上施加语法掩码；本案例不进入。'}),
 'ch12-materialize':('vllm','vllm/v1/worker/gpu/async_utils.py',167,178,{
  168:'读取 CPU 结果前等待复制事件，异步发起不等于数据已经可读。',174:'把已复制的 token 张量转换为兼容现有接口的 Python 列表。',175:'每条请求有效输出数可能不同；不能按填充宽度全部返回。',177:'去掉无效尾部，再填入最终交给核心的 ModelRunnerOutput。'}),
 'ch12-append':('vllm','vllm/v1/core/sched/scheduler.py',2347,2364,{
  2354:'初始假设尚未停止；分块 prefill 的输出列表可以为空。',2355:'逐个接纳本轮新 token；投机路径一轮可能不止一个。',2356:'先进入请求的确认历史，再用更新后的历史检查终止条件。',2360:'停止检查同时修改请求状态，后续核心输出据此填写结束原因。',2362:'若已停止，裁掉同轮剩余 token，避免把结束之后的内容继续交付。'}),
 'ch12-stop':('vllm','vllm/v1/core/sched/utils.py',100,115,{
  100:'当前请求已至少追加一个新 token，读取最新 token 判定。',101:'先检查有效 EOS；ignore_eos 的影响在参数准备阶段体现。',105:'显式 stop_token_ids 也可终止，并记录实际触发 token。',110:'模型总长度或请求生成长度达到上限时，以长度原因结束。'}),
 'ch12-text':('vllm','vllm/v1/engine/output_processor.py',702,712,{
  703:'前端按请求保存的增量反分词状态消费新 token，同时检查文本停止串。',706:'文本匹配可以补充核心尚未识别的停止；后文负责向核心发起终止。',712:'如果用户需要 logprobs，另一个处理器整理对应输出；不从文本反推概率。'}),
})
DATA.update({
 'ch12-shapes':(RUN,['steps.1.inputs.batch.logits_indices.values','steps.1.hidden.hidden','steps.1.logits.logits']),
 'ch12-sampled':(RUN,['steps.1.sample.output.req_ids','steps.1.sample.output.sampled_token_ids','steps.2.sample.output.sampled_token_ids','steps.3.sample.output.req_ids','steps.3.sample.output.sampled_token_ids']),
 'ch12-final':('drafts/Book8-vLLM源码剖析/evidence/runner-inproc-traced/result.json',['final.A.request_id','final.A.outputs','final.A.finished']),
})

SOURCES.update({
 'ch13-queue':('vllm','vllm/v1/engine/core.py',684,692,{
  684:'不需要等待前轮语法推进时，将这轮结果句柄和原计划一起登记。',686:'左侧入队，后文从右侧取最早一轮，保证回填能找到对应计划。',687:'容量尚有余量且仍有工作时，先返回控制权，让后续调用继续提交。',692:'没有用户结果不表示没有执行；这里返回本次是否提交了模型工作的标志。'}),
 'ch13-fill':('vllm','vllm/v1/core/sched/async_scheduler.py',51,70,{
  54:'保留回填前状态，避免把已抢占请求误当成仍活跃的缓存拥有者。',55:'复用普通Scheduler的确认token与停止检查，异步类只补时间相关状态。',61:'抢占已清零占位，旧结果不能再扣一次。',62:'正常结果兑现相应占位；计数不允许为负。',66:'只有回填前仍RUNNING才推进缓存登记，并扣除尚未兑现的占位。'}),
 'ch13-buffer':('vllm','vllm/v1/worker/gpu/buffer_utils.py',85,97,{
  87:'轮转到另一缓冲槽；调用者必须保证再次复用前旧读者已经结束。',90:'容量不足时增长为可覆盖输入的大小，不覆盖仍被使用的旧对象。',95:'把当前CPU状态复制到独立传输槽，而不是让设备一直读可变主账本。',97:'返回本轮有效范围的设备可访问视图；平台不支持UVA时有显式复制回退。'}),
 'ch13-staged':('vllm','vllm/v1/worker/gpu/buffer_utils.py',177,187,{
  180:'先检查目标行与起点有效，空写入直接跳过。',184:'记录写到哪一行、从哪一列开始。',186:'把不同长度的更新内容串联，并保存累计边界；此时还没有执行设备写入。'}),
 'ch14-desc':('vllm','vllm/v1/worker/gpu/cudagraph_utils.py',63,76,{
  67:'运行描述同时约束模式与补齐后的尺寸，不能只用batch size做键。',69:'某些路径不要求按请求数补齐，因此允许None。',70:'每请求一致的query长度、最大query长度用于区分执行兼容性。',74:'LoRA与微批次分割也进入匹配契约；形状相同不保证可互换。'}),
 'ch14-dispatch':('vllm','vllm/v1/worker/gpu/cudagraph_utils.py',507,527,{
  507:'把实际LoRA数映射到预备的有效捕获档位。',509:'只有已经捕获且存在候选时，才逐个检查兼容描述。',520:'返回匹配描述，后续输入准备按它的补齐要求执行。',521:'没有可用匹配时返回NONE与真实尺寸，不能勉强使用不兼容图。'}),
 'ch14-replay':('vllm','vllm/v1/worker/gpu/cudagraph_utils.py',529,542,{
  531:'FULL重放入口要求模式一致，且字典中实际存在该图。',541:'先确保旧预取不再与静态缓冲冲突；省掉Python发射不等于省掉同步依赖。',542:'重放已捕获图；输入内容必须已经写入捕获关联的缓冲区。'}),
 'ch14-pw':('vllm','vllm/v1/worker/gpu/cudagraph_utils.py',550,555,{
  551:'默认piecewise路径通过模型调用进入已编译的分段图。',554:'启用breakable路径时使用专门包装器，说明分段图不应一概等同torch.compile。'}),
})
CONTRACT='drafts/Book8-vLLM源码剖析/evidence/metadata-contracts/trace.json'
DATA.update({
 'ch13-contract':(CONTRACT,['async.initial','async.first_result','async.second_result']),
 'ch13-stale':(CONTRACT,['async.stale_delivery']),
 'ch14-cases':(CONTRACT,['graph.descriptor','graph.cases.two_decode','graph.cases.one_prefill_two_tokens']),
})
for key in ['ch13-contract','ch13-stale']:
 DESCRIPTIONS[key]='CPU 方法契约实验：使用真实回填方法，初始状态手工构造，KV 依赖只记录调用；不是异步推理轨迹。'
DESCRIPTIONS['ch14-cases']='CPU 执行固定源码纯兼容函数的结果；候选描述手工构造，没有捕获或重放设备图。'
SOURCES.update({
 'ch14-compile':('vllm','vllm/model_executor/models/qwen2.py',313,323,{
  313:'Qwen3复用的模型骨架声明可参与编译；装饰器支持不等于本次运行已启用。',315:'input_ids的第0轴是随当轮工作量变化的token轴。',318:'positions按最后一轴声明动态；普通文本是一维，多轴位置有不同布局。',319:'PP中间张量与直接embedding输入也声明对应动态轴。'}),
})
SOURCES.update({
 'ch15-register':('vllm','vllm/v1/engine/async_llm.py',515,524,{
  515:'新请求先通过接纳检查，避免超过前端允许的资源约束。',521:'在首次await之前建立输出侧状态，使并发任务能立即看见本地登记。',524:'再异步发送核心请求；此await不代表等待整段生成完成。'}),
 'ch15-yield':('vllm','vllm/v1/engine/async_llm.py',700,713,{
  702:'这个循环只消费该请求的结果，不亲自驱动模型前向。',706:'有现成结果就立即取，否则挂起当前协程等待；事件循环可处理其他请求。',711:'以输出契约中的finished控制生成器结束，不能用文本非空代替。',713:'交给调用者的仍是RequestOutput，HTTP协议适配在更外层。'}),
 'ch15-merge':('vllm','vllm/v1/engine/output_processor.py',68,78,{
  70:'空槽或异常直接设置结果并唤醒消费者。',73:'如果已有未消费结果，不再无条件堆一条队列节点。',78:'通过RequestOutput.add按候选索引合并；DELTA累加内容，累计模式保留相应最新结果。'}),
 'ch15-cancel':('vllm','vllm/v1/engine/async_llm.py',715,723,{
  718:'协程取消或生成器关闭都要进入请求清理；正常长度结束不经这个异常分支。',720:'使用收集器持有的内部请求ID，避免把用户标识误当核心身份。',723:'清理后继续传播取消语义，不把它变成一次成功完成的文本响应。'}),
 'ch15-abort':('vllm','vllm/v1/engine/async_llm.py',848,857,{
  853:'统一单个ID和ID集合的输入形式。',856:'前端移除输出状态并解析实际内部ID，可能涉及一个用户ID下的多个请求。',857:'向核心发送终止消息，使计算与资源生命周期也结束。'}),
})
ASYNC='drafts/Book8-vLLM源码剖析/evidence/async-frontend/'
DATA.update({
 'ch15-delta':(ASYNC+'projection.json',['deltas.A-online']),
 'ch15-merge-data':(ASYNC+'projection.json',['collector_cpu']),
 'ch15-cancel-data':(ASYNC+'result.json',['cancel_first.request_id','cancel_first.outputs','cancel_first.finished','aborts','idle']),
 'ch15-effective':(ASYNC+'manifest.json',['effective']),
})
DESCRIPTIONS['ch15-merge-data']='CPU 收集器实验：重放已保存的真实 A 输出块，连续 put 后才 get；不是网络背压实测。'
SOURCES.update({
 'ch16-heads':('vllm','vllm/model_executor/layers/linear.py',1047,1054,{
 1047:'取得本层参与的TP规模；禁用TP时按单份处理。',1048:'Q头按TP划分，divide会要求合法整除。',1049:'KV头少于或等于TP规模时，每rank至少保留一个KV头。',1051:'记录复制次数；不能把KV头无限切成分数头。',1053:'否则按TP切KV头，本例8/2=4。'}),
 'ch16-row':('vllm','vllm/model_executor/layers/linear.py',1761,1783,{
 1761:'调用者已给局部输入时直接使用；否则在这里切最后一维。',1767:'按本rank选择局部输入，保证连续存储。',1772:'偏置至多加一次，避免后续求和把偏置重复计算。',1773:'调用本层配置的计算方法，得到局部输出贡献。',1775:'需要合并且TP大于1时求和；相同形状的局部贡献变为完整结果。',1780:'返回值是否附偏置由层契约控制，不能假设永远只有一个张量。'}),
 'ch16-layers':('vllm','vllm/model_executor/models/utils.py',875,887,{
 875:'根据当前PP位置求本地层段，end为开区间上界。',879:'ModuleList仍保留全局层下标。',880:'前面不属于本rank的层放占位。',882:'只有本地层段真正调用层工厂；前缀保持全局编号。',884:'尾部不属于本rank的层同样占位。',887:'调用者同时取得边界和模块表，前向只遍历本地范围。'}),
 'ch16-pp-input':('vllm','vllm/model_executor/models/qwen2.py',408,417,{
 408:'PP首段负责把用户输入变为hidden states。',409:'已有embedding输入时无需再次查词表。',413:'首段尚无分离的残差状态。',415:'后续阶段必须收到前段状态，缺失就是契约错误。',416:'取主分支与残差；它们共同构成后续层的输入。'}),
 'ch16-rpc':('vllm','vllm/v1/executor/multiproc_executor.py',416,424,{
 416:'普通执行使用方法名；本地探针也可使用可序列化函数。',420:'广播的是控制命令、参数与回复选择，不是模型层中的all-reduce。',422:'默认收集各rank回复；指定output_rank时只等该通道。'}),
 'ch16-send':('vllm','vllm/v1/worker/gpu_worker.py',1231,1239,{
 1234:'上游Worker通过PP组发送模型返回的中间张量。',1236:'TP组信息供传输实现选择配合方式，不能推断一份张量总是完整重复发送。',1239:'保存设备发送句柄，下一轮复用存储前等待；发送返回不等于数据不再在途。'}),
})
DATA.update({
 'ch16-tp-data':('drafts/Book8-vLLM源码剖析/evidence/tp2-final/result.json',['ranks']),
 'ch16-pp-data':('drafts/Book8-vLLM源码剖析/evidence/pp2/result.json',['ranks']),
})
SOURCES.update({
 'ch17-score':('vllm','vllm/v1/engine/core_client.py',1570,1582,{
 1570:'本客户端知道自己交给该引擎的在途数，不必等待下一次远端统计。',1571:'取远端队列总量与本地在途估计的较大值，防止旧快照抹掉刚提交的工作。',1572:'只有存在等待请求时才增加KV压力惩罚。',1579:'KV占用高于50%才增加这一项；它是负载启发式，不是精确耗时预测。',1580:'严格小于才更新，平分时由扫描顺序决定。'}),
 'ch17-owner':('vllm','vllm/v1/engine/core_client.py',1591,1597,{
 1591:'未显式指定目标的负载选择结束后旋转扫描起点，减少固定同分偏向。',1593:'把下标转换为真实引擎通信身份。',1595:'保存请求归属，后续取消必须发回这个核心。',1596:'立即增加本客户端的精确在途计数。'}),
 'ch17-send':('vllm','vllm/v1/engine/core_client.py',1486,1501,{
 1487:'先确保统计更新任务存在。',1489:'消息同时带当前wave与原前端编号。',1492:'选择并登记目标，再把ADD发给该核心。',1494:'跨核心计算处于暂停时，额外通知协调器唤醒伙伴。',1499:'等待本次发送操作完成，不等待整段生成。',1501:'确保输出接收任务运行，结果还要回到原前端。'}),
 'ch17-finish':('vllm','vllm/v1/engine/core_client.py',1614,1617,{
 1614:'核心通知请求终结后，再移除路由状态。',1616:'pop带默认值，重复终结不会再次扣减计数。',1617:'只有确实移除归属记录时才减本客户端计数。'}),
 'ch17-cancel':('vllm','vllm/v1/engine/core_client.py',1675,1685,{
 1675:'批量取消先按既有目标分组。',1677:'只处理仍有归属记录的请求，不为取消重新做负载选择。',1679:'每组向自己的核心发送一次ABORT消息。',1685:'这里是消息提交；核心处理与资源清理仍有后续生命周期。'}),
 'ch17-dense':('vllm','vllm/v1/engine/core.py',1318,1328,{
 1318:'保留服务侧DP身份。',1319:'只有DP且模型是MoE，才使用需要跨核心计算协调的实现。',1324:'稠密DP内部计算相互独立，重新配置为各自的模型执行副本。',1328:'保留engine_index供消息路由识别，不丢失外层服务身份。'}),
})
DP='drafts/Book8-vLLM源码剖析/evidence/dp-contract/trace.json'
DATA.update({'ch17-burst':(DP,['burst']),'ch17-pressure':(DP,['pressure','explicit_rank']),'ch17-aborts':(DP,['cancel_messages'])})
for key in ['ch17-burst','ch17-pressure','ch17-aborts']:
 DESCRIPTIONS[key]='CPU真实路由方法的构造状态实验：E0/E1是教学引擎身份，网络发送替换为记录器；没有运行DP模型。'
SOURCES.update({
 'ch18-factory':('vllm','vllm/model_executor/layers/fused_moe/layer.py',405,417,{
 405:'允许平台或调用者替换Runner类；默认装配上游MoERunner。',408:'工厂产物是Runner，不是一个名为FusedMoEFactory的实例。',411:'路由策略与专家参数分别作为协作者传入。',414:'gate交给Runner，使路由线性层可以在合适的流同步之后执行。'}),
 'ch18-gate':('vllm','vllm/model_executor/layers/fused_moe/runner/moe_runner.py',900,905,{
 900:'Runner持有gate时，这里才把hidden变成真正的路由logits。',901:'同时存在可融合的共享专家gate时可合并计算。',905:'普通分支返回路由分数；此前名为router_logits的参数可能只是原hidden。'}),
 'ch18-route':('vllm','vllm/model_executor/layers/fused_moe/runner/moe_runner.py',617,639,{
 617:'整体内核可把路由融合进去，因此直接传logits。',624:'模块化路径先显式取得所选专家ID与权重。',626:'Router封装选择规则，输出形状通常为[token, top_k]。',633:'RoutedExperts消费选择结果和hidden，负责进入参数/计算方法路径。'}),
 'ch18-map':('vllm','vllm/model_executor/layers/fused_moe/expert_map_manager.py',66,87,{
 67:'按EP规模分配专家个数；余数优先分给低rank。',72:'全局专家轴先填-1，表示该专家不在本rank。',75:'连续放置时，本rank得到一段全局编号。',77:'值是本地参数槽位，不是另一份全局专家ID。',80:'轮转放置把间隔编号映射到连续本地槽位；完整配置支持还要通过上层检查。'}),
 'ch18-format':('vllm','vllm/model_executor/layers/fused_moe/modular_kernel.py',1681,1685,{
 1681:'装配之后再解决通信准备侧与专家计算侧的关联配置。',1682:'两侧必须承认相同激活布局；抽象接口存在不代表任意实现能混搭。'}),
 'ch18-dispatch':('vllm','vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py',158,168,{
 158:'代表性的naive DP/EP实现，把激活与路由信息一起交给EP组分发。',162:'序列并行条件进入通信契约。',163:'量化scale、LoRA映射等附加信息也可随token移动，避免元数据失配。',166:'没有附加字段时解包三个返回值；它们都已按分发后的布局组织。'}),
})
DATA['ch18-map-data']=('drafts/Book8-vLLM源码剖析/evidence/expert-map/trace.json',['layouts.linear','layouts.round_robin'])
DESCRIPTIONS['ch18-map-data']='CPU执行固定源码determine_expert_map函数；4专家/2rank为构造配置，没有运行MoE模型，也未经过完整配置支持校验。'
SOURCES.update({
 'ch19-register':('vllm-ascend','vllm_ascend/__init__.py',72,75,{
 72:'entry point调用此函数发现平台；此处不创建Worker或模型。',75:'返回限定类名，上游稍后解析并实例化平台对象。'}),
 'ch19-select':('vllm','vllm/platforms/__init__.py',263,272,{
 263:'从已激活候选中取出树外平台插件。',265:'多个树外平台同时激活时拒绝模糊选择。',270:'只有一个树外平台候选时使用其返回的限定名。'}),
 'ch19-runner':('vllm-ascend','vllm_ascend/worker/worker.py',510,516,{
 510:'读取有效MRV2开关；NPUWorker在设备准备后才创建Runner。',512:'与MRV1同名的类位于不同模块，别只按短类名搜索。',514:'本书基础记录进入该分支。',516:'未启用MRV2时走独立NPUModelRunner实现。'}),
 'ch19-super':('vllm-ascend','vllm_ascend/worker/v2/model_runner.py',635,645,{
 635:'平台包装保留失败状态，供后续清理逻辑使用。',636:'在平台决定的DP协调上下文中执行。',638:'调用上游MRV2主执行逻辑，说明平台实现没有把所有调度和批次逻辑重新写一遍。',639:'传递同一SchedulerOutput及中间张量契约。'}),
 'ch19-ops':('vllm','vllm/model_executor/custom_op.py',119,128,{
 119:'未登记树外实现时创建原类。',122:'有登记时选择替代类，选择发生在实例创建阶段。',128:'返回选定类的实例；源码中的构造名字与type(obj)可能不同。'}),
 'ch19-patch':('vllm-ascend','vllm_ascend/patch/platform/patch_balance_schedule.py',892,897,{
 897:'直接修改上游模块中的Scheduler名称绑定；这是运行期补丁，不是一个普通工厂参数。'}),
 'ch19-fallback':('vllm-ascend','vllm_ascend/patch/platform/patch_balance_schedule.py',163,165,{
 163:'实际对象虽是BalanceScheduler，具体行为仍取决于有效配置。',164:'均衡未启用时，委托原Scheduler的schedule。'}),
})
DATA['ch19-runtime']=(BASE,['effective.scheduler','effective.scheduler_mro','effective.balance_enabled','effective.worker','effective.runner','effective.runner_execute.symbol'])
DATA['ch19-platform']=('drafts/Book8-vLLM源码剖析/evidence/platform/metadata.json',['entry_points','platform.class','worker.mro','runner.mro'])
DESCRIPTIONS['ch19-platform']='已安装包的entry point与类自省记录，来源文件逐字匹配固定源码；没有创建Worker实例或运行模型。'
SOURCES.update({
 'ch20-sampler':('vllm','vllm/v1/worker/gpu/model_runner.py',1552,1564,{
 1552:'当前批次没有草稿，或未装配拒绝采样器，沿普通采样路径。',1556:'有草稿时由验证采样器决定哪些候选成为确认结果。',1560:'target logits来自本轮目标模型前向。',1563:'概率验证还要知道提议分布，不能只比较两个模型的最大值。'}),
 'ch20-propose':('vllm','vllm/v1/worker/gpu/model_runner.py',2142,2158,{
 2142:'此处已经得到本轮验证结果，接下来为后续验证准备新草稿。',2144:'批次、Attention元数据和slot映射共同约束提议模型的输入位置。',2147:'目标模型的隐藏状态可作为提议模型输入，具体模型负责适配。',2149:'接受与拒绝的数量决定有效前缀，避免继续沿被否定的历史提议。',2158:'按持久请求行保存候选；不是追加到用户已确认输出。'}),
 'ch20-decode':('vllm','vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py',376,392,{
 376:'进入多步草稿生成阶段；子类可实现相关钩子。',378:'选择融合或普通多步执行方式，二者遵守同一个提议结果契约。',383:'剩余步数的执行受本轮批次、图模式和DP协作约束。',390:'阶段结束后恢复子类临时状态。',392:'返回本批次请求的候选矩阵，每行对应一个请求。'}),
 'ch20-accept':('vllm','vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py',584,595,{
 584:'synthetic模式人为指定接受率，不能用它说明正常解码正确性。',588:'正常贪心路径要求候选与目标argmax完全一致。',589:'负数占位候选无效，必须拒绝。',590:'首次拒绝后停止验证后续候选，后缀依赖的条件已失效。',592:'接受时写候选；拒绝点写目标选择作为纠正token。'}),
 'ch20-rollback':('vllm','vllm/v1/core/sched/scheduler.py',1983,1995,{
 1983:'以本轮实际调度的草稿数量为准，不用proposer最初想提议的上限。',1984:'普通自回归路径每步额外有一个目标采样结果。',1985:'从返回长度扣除这个结果，得到被接受的草稿数量。',1986:'未被接受的整个后缀都计入拒绝。',1991:'抢占后的旧结果不再回退进度，避免重复记账。',1993:'修正有效计算前缀；不要求擦除设备中所有旧字节。',1995:'异步路径还要同步修正占位计数。'}),
 'ch20-ngram':('vllm','vllm/v1/spec_decode/ngram_proposer.py',283,293,{
 283:'找不到足够长的历史匹配时，不勉强制造候选。',291:'把反向匹配位置变回原历史，候选从匹配片段之后开始。',292:'候选最多取现有历史中可用的后续部分。',293:'返回的是历史token片段；目标模型仍必须验证。'}),
})
DATA['ch20-ngram-data']=('drafts/Book8-vLLM源码剖析/evidence/ngram-contract/trace.json',['cases'])
DESCRIPTIONS['ch20-ngram-data']='CPU执行固定源码函数体，去掉Numba装饰器；token ID为构造值。属于MRV1辅助路径，不是MRV2或NPU投机推理实测。'
SOURCES.update({
 'ch21-future':('vllm','vllm/v1/structured_output/request.py',44,53,{
 44:'访问请求语法时，先检查编译Future是否已经完成。',48:'短暂等待仍未完成，就把请求留在阻塞状态。',51:'编译异常保存在该请求上，交由调度器定向处理。',53:'这里的完成包含成功和失败，不等于语法可用于采样。'}),
 'ch21-promote':('vllm','vllm/v1/core/sched/scheduler.py',2979,2987,{
 2979:'这个等待原因与KV容量不足不同。',2981:'没有语法结果时继续等待。',2983:'有结果但为异常时，登记请求级失败。',2986:'只有得到可用结果，才回到普通等待队列语义。'}),
 'ch21-matcher':('vllm','vllm/v1/structured_output/backend_xgrammar.py',123,131,{
 123:'将编译产物包装成请求级Grammar。',124:'为本请求创建独立Matcher游标，不共享可变解析进度。',126:'停止token也进入契约，语法未完成时不能随意提前结束。',127:'投机检查需要回退能力，保留足够的历史窗口。'}),
 'ch21-validate':('vllm','vllm/v1/structured_output/backend_xgrammar.py',187,201,{
 187:'语法已终结时，不再接受新的候选前缀。',191:'按候选顺序临时推进Matcher。',194:'到达终结点便停止试走。',197:'第一个不合法token截断整个后缀。',200:'撤销本次试走，不能把候选当成已经提交的输出。',201:'交出合法前缀，Grammar自身仍停在原确认位置。'}),
 'ch21-map':('vllm','vllm/v1/worker/gpu/structured_outputs.py',74,85,{
 74:'语法行与目标logits行可能顺序不同，必须建立显式映射。',79:'当步请求顺序提供执行侧索引。',80:'mask列表自己的请求顺序可能只包含有约束的子集。',82:'投机路径还需逐候选位置的mask，不能每请求只取一行。'}),
 'ch21-mask':('vllm','vllm/v1/worker/gpu/structured_outputs.py',152,162,{
 153:'压缩位图中0代表禁止，先展开为每个词表位置的布尔条件。',156:'保留允许项的原logit，只改被禁止且当前有效的位置。',160:'负无穷使该token在后续采样中不再有概率质量。'}),
 'ch21-commit':('vllm','vllm/v1/core/sched/scheduler.py',2054,2065,{
 2054:'验证后的确认token才正式推进请求语法。',2057:'若语法拒绝这批结果，说明约束执行出现不一致。',2063:'将对应请求标为错误结束，不继续输出未经约束的后续内容。'}),
})
GRAM='drafts/Book8-vLLM源码剖析/evidence/grammar-cpu/trace.json'
DATA.update({'ch21-states':(GRAM,['vocab','states']),'ch21-future-data':(GRAM,['future'])})
for key in ['ch21-states','ch21-future-data']:DESCRIPTIONS[key]='CPU实际XGrammar与vLLM请求包装实验；小词表、语法和Future完成结果由脚本构造，没有运行模型或设备mask内核。'
SOURCES.update({
 'ch22-place':('vllm','vllm/model_executor/models/qwen2_vl.py',1150,1158,{
 1150:'视觉特征经过空间合并，展开长度由合并粒度决定。',1153:'取当前媒体项的处理产物，不用整批图片数量替代。',1154:'grid_thw描述该项的时间与空间网格。',1157:'把网格大小换算为语言序列需要的特征位置数。',1158:'占位token为后续embedding替换预留位置。'}),
 'ch22-budget':('vllm','vllm/v1/core/sched/scheduler.py',1778,1797,{
 1778:'decoder能算多少，还受encoder预算和特征缓存容量约束。',1785:'媒体项之前仍有文本时，本轮可以先推进这些文本。',1788:'将decoder计划截在尚未准备好的媒体输入之前。',1792:'已处于该媒体范围而特征不可用时，不能越过它计算。'}),
 'ch22-evict':('vllm','vllm/v1/core/encoder_cache_manager.py',174,190,{
 174:'空闲容量足够即可；这里只检查和维护CPU账本。',179:'可回收容量也不足则失败，不能驱逐仍被请求引用的项。',185:'从最早成为可回收的项开始驱逐。',188:'记录hash，稍后通过调度产物通知执行端删除对应张量。'}),
 'ch22-reference':('vllm','vllm/v1/core/encoder_cache_manager.py',251,260,{
 251:'同一请求可能多次引用同一图片，仍有其他位置需要时不能释放请求引用。',256:'最后一个对应位置释放后，才移除该请求的引用。',257:'没有任何请求使用时变为可回收，而不是立即删除张量。'}),
 'ch22-window':('vllm','vllm/v1/worker/gpu/mm/encoder_runner.py',245,264,{
 246:'先定位与当前query区间有交集的媒体项。',255:'把全序列窗口转换成此媒体占位段内部的相对区间。',258:'包含文本分隔位时，占位相对位置还要再映射成真正的embedding行。',263:'本段没有特征行则跳过，避免把分隔token误当成encoder输出。'}),
 'ch22-vision':('vllm','vllm/model_executor/models/qwen2_vl.py',1390,1396,{
 1391:'普通本地视觉分支执行视觉模型，得到合并后的特征。',1394:'按同一空间合并规则计算每张图片的特征行数。',1396:'将拼接输出分回媒体项；顺序必须对应输入项。'}),
 'ch22-merge':('vllm','vllm/model_executor/models/utils.py',702,710,{
 702:'纯文本或无媒体特征时保留原embedding。',705:'将多个媒体项按既定顺序整理成特征行。',706:'融合后的dtype遵守语言输入embedding的契约。',710:'按明确的多模态位置覆盖，形状与顺序必须匹配；这是原地更新。'}),
})
MM='drafts/Book8-vLLM源码剖析/evidence/mm-contract/trace.json'
DATA.update({'ch22-position-data':(MM,['positions']),'ch22-cache-data':(MM,['states','freed_hashes'])})
for key in ['ch22-position-data','ch22-cache-data']:DESCRIPTIONS[key]='CPU实际位置对象与EncoderCacheManager实验，媒体hash和token为构造值；未处理图片、未分配视觉特征张量、未运行视觉模型。'
SOURCES.update({
 'ch23-expand':('vllm','vllm/v1/worker/gpu/lora_utils.py',95,105,{
 101:'先按当步idx_mapping从持久行取出adapter ID，避免请求重排后串用。',102:'请求级映射保持每个当步请求一项。',103:'按各请求本轮token数量展开，为每个计算行指定adapter身份。',104:'去重得到当步需要激活的adapter集合。'}),
 'ch23-limit':('vllm','vllm/v1/core/sched/scheduler.py',897,910,{
 899:'本轮已达到不同adapter数量上限时，检查新请求是否引入新身份。',904:'使用已在本轮集合中的adapter不增加种类数。',908:'暂时跳过该请求，让其他可兼容请求仍有机会入批。'}),
 'ch23-load':('vllm','vllm/lora/worker_manager.py',298,314,{
 302:'先完成加载与有效性检查，避免新adapter无效时先破坏已有缓存。',306:'处理同ID替换；只有上面加载成功才到这里。',310:'新增将超CPU缓存容量时，驱逐旧项。',314:'将加载好的权重对象登记，设备激活是后续步骤。'}),
 'ch23-slot':('vllm','vllm/lora/model_manager.py',338,353,{
 338:'index是设备槽位，lora_model.id是逻辑adapter ID，两者不相等也正常。',340:'按模型层逐一安装此adapter的参数。',343:'adapter没有覆盖某层时清理该槽旧参数，避免上一个adapter残留。',349:'将该层A/B放进所选槽，供后续批次映射引用。'}),
 'ch23-convert':('vllm','vllm/lora/punica_wrapper/utils.py',94,111,{
 94:'反转槽位表，建立逻辑adapter ID到本地槽的查找。',101:'基础模型身份0转为-1，表示不应用LoRA增量。',104:'同样转换每个计算token的映射。',111:'得到内核实际使用的槽位索引，不能把逻辑ID直接用作数组下标。'}),
 'ch23-cachekey':('vllm','vllm/lora/model_manager.py',1183,1191,{
 1187:'缓存键必须包含当前槽布局，adapter可能在批次外被换入换出。',1188:'逻辑批次不变但槽位改变，也必须重建内核元数据。',1190:'更新两份依赖快照，后续才可安全复用。'}),
 'ch23-apply':('vllm','vllm/lora/layers/base_linear.py',208,212,{
 211:'先用基础层当前计算方法取得共享权重的结果，方法也可能是量化实现。',212:'再按行对应的adapter，将低秩增量加到结果上。'}),
 'ch23-scale':('vllm','vllm/lora/lora_weights.py',36,42,{
 38:'已折入缩放时不重复处理。',40:'把缩放吸收到B中，后续内核可用乘数1。',41:'更新状态，防止下次optimize再次乘同一因子。'}),
})
DATA['ch23-mapping-data']=('drafts/Book8-vLLM源码剖析/evidence/lora-mapping/trace.json',['batch','layouts'])
DESCRIPTIONS['ch23-mapping-data']='CPU实际LoraState与固定convert_mapping函数实验；异步传输helper替换为CPU张量创建，adapter路径为未访问的构造路径，没有加载权重或执行LoRA内核。'
SOURCES.update({
 'ch24-select':('vllm','vllm/model_executor/layers/quantization/fp8.py',182,201,{
 182:'按真实层类型与名称决定实现，不是全模型统一替换一个forward。',189:'跳过列表中的层仍用未量化方法。',190:'checkpoint格式决定走在线转换还是已量化装载路径。',195:'v0.30中，未预先量化的权重由独立在线方法处理。',199:'本章主线为已经序列化FP8的权重。'}),
 'ch24-create':('vllm','vllm/model_executor/layers/quantization/fp8.py',329,343,{
 329:'创建该分片的权重参数壳，并保留装载器契约。',332:'参数属于层对象，计算方法负责其表示与使用。',335:'本片段只展开非block的scale；融合层可有多个逻辑分片。',343:'scale是权重语义的一部分，必须随正确分片一起装载。'}),
 'ch24-post':('vllm','vllm/model_executor/layers/quantization/fp8.py',412,427,{
 414:'将融合层多个逻辑分片的scale整理成执行策略要求的表示。',417:'logical_widths说明分片边界，不能只看整个矩阵shape。',420:'静态激活缩放也要归一为内核所需形式。',423:'本分支转为运行期[K,N]布局。',426:'替换参数对象；持有旧引用的代码不能假定它仍是当前权重。'}),
 'ch24-kernel':('vllm','vllm/model_executor/kernels/linear/__init__.py',678,698,{
 678:'先取当前平台的候选集合。',680:'显式linear backend选项进一步约束候选。',686:'依次验证硬件支持及此层配置是否可实现。',691:'首个通过检查的候选成为本层内核。',695:'没有可用实现时给出各候选失败原因，不凭量化名称强行执行。'}),
 'ch24-encoding':('vllm','vllm/model_executor/layers/quantization/utils/fp8_utils.py',68,79,{
 73:'CPU实验显式传入e4m3fn，不依赖当前平台默认格式。',76:'用绝对最大值确定整张量范围；下界保护全零输入。',77:'scale把最大值映射到目标格式的有限范围。',79:'返回低精度编码及反向缩放因子，重建时二者相乘。'}),
})
DATA['ch24-encoding-data']=('drafts/Book8-vLLM源码剖析/evidence/fp8-encoding/trace.json',['cases'])
DESCRIPTIONS['ch24-encoding-data']='CPU执行固定input_to_float8函数体；显式e4m3fn、构造张量。未调用设备FP8矩阵乘、未运行量化模型。'
SOURCES.update({
 'ch25-page':('vllm','vllm/v1/core/kv_cache_utils.py',1399,1426,{
 1399:'Mamba状态尺寸来自shape，增大block_size不会增加状态字节。',1406:'为状态页补齐物理空间，语义shape保持不变。',1411:'attention页可整除时，扩大每块覆盖的token数量。',1415:'部分非MLA不整除情况允许物理padding，MLA须保留自身行对齐约束。',1420:'无法满足布局契约时明确失败，调用方再决定是否有受支持的fallback。'}),
 'ch25-remove':('vllm','vllm/v1/core/single_type_kv_cache_manager.py',670,679,{
 670:'空洞仍保留在原逻辑列，不能把后面的块向左压紧。',672:'从后向前释放，已处理过的null前缀可停止。',676:'先记录要归还的实体，再用null替代请求引用。',679:'实际引用计数与空闲队列由BlockPool维护。'}),
 'ch25-window':('vllm','vllm/v1/core/single_type_kv_cache_manager.py',1178,1181,{
 1178:'窗口包含将要计算的当前token，因此只需W-1个历史位置；额外保留用于特定投机分支。'}),
 'ch25-reconcile':('vllm','vllm/v1/core/kv_cache_coordinator.py',937,953,{
 939:'候选边界缩短后，先前EAGLE验证状态不能直接沿用。',942:'更新当前共同候选及本组实际命中产物。',949:'所有组未再缩短，才到达可共同接受的边界。',951:'否则用更短候选重新检查各组。',952:'完整注意力+单一其他类型的简单组合有专门提前结束条件。'}),
 'ch25-modes':('vllm','vllm/v1/kv_cache_interface.py',1034,1045,{
 1035:'all模式按多个历史边界保存状态，不能概括成固定一个块。',1040:'align保留运行/对齐状态，并加上配置要求的额外检查点与投机空间。',1045:'none通常只需当前状态，加上必要投机块；状态字节由shapes决定。'}),
 'ch25-same-step':('vllm','vllm/v1/core/single_type_kv_cache_manager.py',1724,1732,{
 1724:'同轮刚产生的状态尚不能作为另一请求的稳定起点。',1732:'用超过池容量的需求阻止本轮接纳，等待后续安全时点。'}),
})
HYBRID='drafts/Book8-vLLM源码剖析/evidence/hybrid-state/trace.json'
DATA.update({'ch25-pages-data':(HYBRID,['pages']),'ch25-window-data':(HYBRID,['window']),'ch25-modes-data':(HYBRID,['mamba_modes'])})
for key in ['ch25-pages-data','ch25-window-data','ch25-modes-data']:DESCRIPTIONS[key]='CPU真实规格与管理器实验，手工构造shape、模式与块表；没有设备状态张量或混合模型前向。'
SOURCES.update({
 'ch26-meta':('vllm','vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py',50,65,{
 50:'本debug实现只传末token之前的完整块，具体对齐helper见正文。',52:'本地物理块号来自当前引擎分配器。',55:'将每块展开成token槽地址；两端块号无需一致。',59:'有效token长度裁剪掉多余预留槽。',63:'方向是本次任务要存还是要读，不是Python对象的进程角色。'}),
 'ch26-pre':('vllm','vllm/v1/worker/gpu/kv_connector.py',70,80,{
 70:'调度侧只交接当前轮元数据，执行侧持有实际缓存视图。',73:'绑定当前轮产物，供层内钩子和load使用。',76:'同步命中必须先装好，当前前向才能读取它。',79:'异步请求尚未参与计算，可以把提交推迟到前向后。'}),
 'ch26-save':('vllm','vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py',243,253,{
 243:'从当前轮元数据中取要保存的请求。',247:'以token身份与层名定位文件，这是debug实现的存储方式。',250:'按源引擎本地slot抽取有效内容。',251:'转为CPU张量后实际写safetensors；不是生产网络零拷贝。'}),
 'ch26-output':('vllm','vllm/v1/worker/gpu/kv_connector.py',110,122,{
 110:'是否等待保存由本次调用语境决定。',112:'统一收集完成结果，可能只是某些请求完成。',115:'接收失败与已完成接收不同，消费者必须分别处理。',116:'出错块用于修正有效前缀，不能把未写入数据当命中。',122:'轮次结束清掉绑定，避免下一轮误用旧元数据。'}),
 'ch26-full-hit':('vllm','vllm/v1/core/sched/scheduler.py',2950,2958,{
 2951:'成功接收后才提交可用缓存身份。',2956:'完整prompt的KV存在，并不自带下一token的logits。',2958:'回退一token重新执行末位置以采样；不是重新算完整prompt。'}),
 'ch26-free':('vllm','vllm/v1/core/sched/scheduler.py',2580,2588,{
 2582:'请求生成结束的信息可以先对外报告。',2584:'connector仍需要源块时，延迟归还资源。',2585:'只有不存在延迟依赖时才立即释放。'}),
 'ch26-job':('vllm','vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py',1857,1867,{
 1857:'job ID关联此前的传输计划，不靠日志出现顺序猜归属。',1858:'扣除本次汇报完成的worker数量。',1859:'仍有worker未完成，就不能宣布整个状态可用。',1861:'计数不能超额，重复完成上报会破坏协议。',1864:'全部完成后再提交存储或加载完成状态。'}),
})
CONN='drafts/Book8-vLLM源码剖析/evidence/connector-cpu/trace.json'
DATA.update({'ch26-roundtrip':(CONN,['roundtrip']),'ch26-counts':(CONN,['completion_counts'])})
for key in ['ch26-roundtrip','ch26-counts']:DESCRIPTIONS[key]='CPU真实ExampleConnector文件/槽拷贝与OffloadingWorkerMetadata聚合，手工初始化对象及构造张量；没有启动两引擎、网络传输或NPU模型。'
SOURCES.update({
 'ch27-clocks':('vllm','vllm/v1/metrics/stats.py',224,234,{
 224:'前端接收起点使用墙钟；同域结束点才能与它相减。',227:'核心事件使用单调时钟，计核心内部阶段间隔。',234:'首token延迟另存，不能用核心时间直接减arrival_time。'}),
 'ch27-events':('vllm','vllm/v1/metrics/stats.py',518,529,{
 518:'事件按请求关联，不按整个批次只记一份状态。',522:'首次进入调度的时间用于请求级阶段划分。',523:'抢占后重新调度不覆盖最初scheduled时间。',526:'抢占同时累计本轮和该请求的计数。'}),
 'ch27-intervals':('vllm','vllm/v1/metrics/stats.py',540,562,{
 540:'E2E从前端到前端计算，包含前端观察到的完整等待。',543:'队列区间使用两个核心时间。',547:'这里是首调度到首token事件的经过时间，不是设备kernel净执行时间。',551:'decode包含首token之后直到末token的停顿与抢占。',558:'平均每输出token时间排除prefill产出的第一个token。',561:'只有一个生成token时没有后续间隔，返回0。'}),
 'ch27-socket-error':('vllm','vllm/v1/engine/core_client.py',1158,1161,{
 1158:'后台接收协程失败，必须把异常交给等待输出的消费者。',1160:'接收任务被取消也通知engine不可用，避免消费者永久等待。'}),
 'ch27-propagate':('vllm','vllm/v1/engine/output_processor.py',493,498,{
 493:'共享输出处理失败时，通知所有尚存请求的收集器。',496:'每条generate任务等待自己的queue，必须逐一唤醒。',498:'异常作为队列载荷交付；get时会重新抛出。'}),
 'ch27-get':('vllm','vllm/v1/engine/output_processor.py',84,92,{
 86:'无数据时等待事件；put(error)也会设置此事件。',88:'取走当前载荷后清空槽与事件，为后续更新准备。',90:'把异常重新抛给原请求任务，不把它伪装成空输出。'}),
})
MET='drafts/Book8-vLLM源码剖析/evidence/metrics-contract/trace.json'
DATA.update({'ch27-metrics-data':(MET,['updates','finished']),'ch27-error-data':(MET,['error_delivery'])})
for key in ['ch27-metrics-data','ch27-error-data']:DESCRIPTIONS[key]='CPU执行真实指标/错误交付方法；时间、事件与异常均由探针构造，不是性能实测或真实故障注入到运行引擎。'
SOURCES['ch28-loop']=('vllm','vllm/v1/engine/core.py',598,619,{
 598:'是否仍有需要推进的状态由调度器判断，核心不重新维护一套请求队列。',600:'产物描述本轮允许做的工作，不携带整个引擎的所有内部状态。',601:'执行器封装单设备或多Worker组织；核心只依赖执行契约。',602:'约束准备与设备任务可以在职责分离后安排重叠。',607:'取得实际执行结果；Future本身不等于结果已就绪。',608:'是否另行采样取决于执行返回协议，不能无条件多采一次。',613:'模型在算时可能收到取消，提交结果前处理它。',614:'把执行事实交回拥有请求状态的调度器。',619:'对外返回核心输出与是否执行模型；外层继续负责交付和循环。'})
CLOSE='drafts/Book8-vLLM源码剖析/evidence/closing/trace.json'
DATA.update({'ch28-rounds':(CLOSE,['rounds']),'ch28-final':(CLOSE,['final'])})
for key in ['ch28-rounds','ch28-final']:DESCRIPTIONS[key]='同一次既有NPU基础运行的字段投影，保留原事件seq与pid；没有拼入其他运行的张量或时间。'
