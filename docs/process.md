# 全部流程

从环境到最终数字,按发生顺序。每一步写清做了什么、为什么、怎么验证。数字全部在 `docs/results.md`;读侧和原文那部分的逐步实验在 `docs/accuracy-plan.md`;外部系统的做法在 `docs/memory-systems-survey.md`;提速在 `docs/speed-optimization.md`。仓库根目录的 `selective-memory-for-long-conversation-agents.md` 是 09-10 的原始方案,和这里不一致的地方以这里为准。

## 1. 系统是什么

一个带存储预算的对话记忆系统,在 LongMemEval-S 上评测。每道题给 47 场历史对话(约 10.3 万 token),系统逐场"写入"记忆,最后只凭记忆回答一个问题。

**写入**(每场对话一次):
1. 抽取:本地 Qwen3-8B 把对话变成情节句和事实三元组(实体/属性/值,带时间),受限解码保证 JSON 格式;结果按会话缓存。
2. 向量化:bge-m3,按文本缓存。
3. 写入策略:SieveStreaming 按"覆盖增益 / token"决定存不存,预算满了按 swap(置换)或 fifo/lru 淘汰;事实按属性形成有效期链(知识更新时旧值标记 valid_to)。
4. 原话:每轮原文作为二级条目,事实优先,原话用剩余预算(最终版加的)。

**读取**(每题一次):解析时间约束 → 查询改写(本地 Qwen)→ BM25 ⊕ 向量 RRF 检索 → 有效期链解析 → 实体二跳 → 预算贪心打包 2k token 事实 → 按来源轮和直接检索取 6k token 原话 → gpt-4.1-mini 作答 → GPT-4o 判分。

## 2. 时间线

**09-12 至 09-13:跑通。** 装环境、起 vLLM、接 OpenAI 与 TokenRouter 网关、加 Claude 后端。抽取格式错误 5.67% → 截断(max_output_tokens 4096)、缺自定义属性名、重复循环三处修到 0.85%。网关免费档大量空响应和 503,加重试层、空响应不缓存、多次仍空则抛错。判分改回 GPT-4o 直连(官方口径)。

**09-13:写入策略校准。** sieve 门槛用极短事实做归一化基准,导致存储只填到 32%、淘汰从不发生;加成本下限 20,门槛校准到 3e-4。多值属性(偏好、过敏等)不再被当作"更新"覆盖。评测 ablation 从十几组砍到 evict 一组(swap / 去 Hawkes / fifo)。

**09-14 上午:提速。** ingest 结果落盘缓存(按题 × 写侧配置),读侧改动不再重新 ingest;模型进程内只加载一次;合并步骤关闭(97% 时间换 1 条记录);读者从 glm 换到 gpt-4.1-mini;三分片并行。100 题参照 run 从 5-12 小时到 14 分钟。写了 `docs/speed-optimization.md`。

**09-14 下午:读者不是瓶颈。** gpt-4.1 / gpt-4o / glm / gpt-4.1-mini 全在 48-57;两版 prompt 无净收益。结论:损失在上游。

**09-14 晚:归因和方案。** oracle 上限 84;40 道失败题逐题归到阶段;发现证据会话 100% 都进了上下文,问题是改写丢细节。调研五篇 80%+ 系统的做法(`docs/memory-systems-survey.md`),共同点是读者看原文、事实只做键。读侧实验验证:事实 + 原文 4k 从 58 到 71。

**09-14 深夜至 09-15:落地。** 原话进存储、读侧按来源轮扩展(71);原文轮做检索单元(73);6k 预算(77)。prompt 两版、8k、权重减半、会话分散都试过,没采用。

**09-15:原文计入预算。** 三个设计依次否决:免门槛(B=0.015 只剩 15)、当覆盖目标(短废话自覆盖、归一化基准被抬到 140)。最终:二级条目,事实选择与只事实时逐条一致(验证过集合相同),原话用剩余预算。同时发现 `configs/raw_turns.yaml` 的 `extends` 之前不生效、部分字段静默回落默认值(向量模型 hash、判分 gpt-4.1-mini),实现真正的配置继承后重跑。预算曲线 34 / 46 / 49 / 60 / 72 / 76。

**09-15 至 09-16:held-out。** test 400 题抽取要 20 小时,改为分层抽 200 题:抽取 9.5 小时、预热 1.2 小时、评测 1 小时。0.82。

## 3. 每次改动怎么验证

- 单元测试 107 个,覆盖抽取解析、有效期链、写入/淘汰、ingest 缓存往返、原话二级条目、配置继承。
- 任何读侧改动:dev100 一轮 15 分钟;差 4 题以上才算数。
- 任何写侧改动:先用真实向量在一道题上冒烟(存了多少事实、多少原话、证据轮在不在),再跑 dev100。
- 关键不变量用集合相等验证:开原话后事实集合必须和只事实时完全一致。
- 最终只在 test200 上跑一次,不再挑选。

## 4. 复现

```bash
# 抽取(GPU,一次性)
smem-extract --split dev --config configs/default.yaml --workers 16 --ablate extract.prompt_version=v3-detail
# 向量预热(GPU,几分钟到一小时)
python scripts/prewarm_embeddings.py --split dev --prompt-version v3-detail
python scripts/prewarm_embeddings.py --split dev --turns --device cuda --batch 8
# 评测(3 片并行;ingest 缓存自动建立)
smem-eval --split dev --config configs/raw_turns.yaml --backend llm --judge llm --ids "<逗号分隔的题 id>" --out evals/results_x/shard0
# 预算档
smem-eval ... --store-budget 0.03 --ablate evict.policy=fifo
```

test200:`--split test --ids "$(python -c 'import json;print(",".join(json.load(open("data/test200_seed0.json"))["question_ids"]))')"`。

## 5. 运维教训

- 后台任务 `setsid … & disown`;会话重启后先盘点进程,别起重复的写入者。
- 杀进程按 /proc 扫 cmdline 并排除自身;`pkill -f` 会自杀。
- `HF_HUB_OFFLINE=1`,否则模型加载时去 Hub 查更新会挂死。
- 改正在运行的 bash 脚本无效,要重启;改源码里的常量会静默影响正在跑的评测,可变项要做成配置。
- 六个分片并行撞 OpenAI TPM 限额,429 单独给 20 轮退避。
- GPU 与 vLLM 共享时,向量模型输入封顶 2048 token、小批次;NLI 放 CPU。
- 配置文件的字段不写就是代码默认值,加继承前 `extends` 只是个被忽略的键。

## 6. 还没做的

- B=0.015 的 34:写入价值函数偏"覆盖"不偏"针尖",一半的题证据在写入时就丢了。方向:用户自述的具体事实加权、知识更新链上的最新值加权、重复提及的价值递减。
- multi-session 跨会话计数和 preference 两类在 dev、test 上都弱。
- 合并模块需要"用摘要替换成员"的语义才有意义。
- 工作区未提交;`evals/results_*` 已加入 .gitignore。
