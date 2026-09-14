# 评测流水线提速记录

2026-09-13 ~ 09-14。目标:让一次 100 题的 dev 评测从"几小时、要人盯"变成"分钟级、无人值守"。

## 起点:一道题要多久,时间花在哪

存储不限、swap 策略、glm 读者,一道题的 ingest 逐段计时(`after_session` = 合并步骤):

```
ingest 总计   2652 s
  合并        2647 s   (97%)
    摘要生成   2386 s   230 次调用:41 次真打 vLLM,189 次磁盘缓存命中
    NLI 蕴含    261 s   1718 对(CPU)
  其余          5 s    嵌入(缓存命中)、覆盖计算、写入策略
读取 + 回答      ~10 s  glm 思考 5-10 s;判分 1-2 s
```

也就是说慢的不是检索、不是打包、不是模型加载——**是合并**。而合并的产出:230 个摘要里 229 个被蕴含检查拒绝,整题只写入 1 条摘要事实。97% 的时间换来一条记录。

外层还有两个和算法无关的时间黑洞:

- **每道题重新加载模型**:`build_system` 逐题调用,bge-m3(2 GB)+ NLI 每题从磁盘重载,约 15 s/题。
- **每次 run 都重新 ingest**:换读者、改检索参数、换判分模型,存储内容完全一样,47 场对话照样重新过一遍。

## 做了什么

### 1. ingest 缓存(最大的一项)

每道题 ingest 完成后,把整个状态落盘;之后任何只改**读侧**的 run 直接加载。

- 存储(sqlite `:memory:`)用 `db.backup()` 复制成文件(`store.dump_to` / `restore_from`,恢复时 `_load()` 重建向量索引、BM25、实体索引、链指针);
- 写入策略(sieve、覆盖状态、候选、统计)、Hawkes、合并器记账、系统层的证据映射用 pickle(`SelectiveMemory.save_state` / `load_state`);模型和网络客户端不进缓存。
- 缓存键 = `(question_id, 写侧配置)`。`SystemConfig.ingest_identity()` 从配置身份里剔除 `read.*`、`anthropic.*`、`budget.read_tokens`、回答/判分模型等读侧字段——所以换读者、调 `k`、换判分,键不变、直接命中;改 `evict.policy`、`write.*`、抽取版本,键变、重新 ingest。
- 开关:`models.ingest_cache_dir`(默认 `.cache/ingest`,`null` 关闭;属于基础设施字段,不进 run 身份)。

实测(一道题,不淘汰):ingest 18 s → save 0.1 s → load 0.1 s;加载后读结果与统计和原始对象**逐项一致**(有往返测试)。代价:每题约 40 MB(覆盖状态里的向量矩阵),100 题 × 一份写侧配置 ≈ 4 GB。

一个坑:`sqlite3.Connection.backup()` 在源连接有未提交事务时会拿到 BUSY,然后每 0.25 s 无限重试——`MemoryStore` 只 `execute` 不 `commit`,第一版直接把测试挂死。`dump_to` 里先 `commit()`。

### 2. 合并步骤关闭(参照对比期间)

方向修正后(`consolidation.nli_direction = summary_supported`)接受率从 1/230 升到 93/218,但写入策略把摘要当作**新候选**和仍在存储里的成员比覆盖增益,增益≈0 被拒,实际只进 2 条。合并需要"用摘要替换成员"的语义(按 `gain_without(成员)` 评估),这是另一块工作;在此之前用 `consolidation.policy=no_consolidation`。这一项把 ingest 从 10 分钟级降到 20 秒级。

### 3. 模型进程内只加载一次

`get_embedder` / `get_nli` 按 `(模型名, 设备)` 缓存实例;`build_system` 每题复用。省 ~15 s/题。

### 4. 读者换成不思考的模型

glm-5.3-free 走网关:每题思考 200-500 token、9 s,免费档 11-17 点大量返回**零 token 的空响应**(缓存按小时统计空答案率最高 63%),还会半死连接挂住整题。gpt-4.1-mini 直连:0.9 s,不限流。准确率持平(0.51 vs 0.52),弃答减半,空答案归零。

配套的健壮性改动:空响应视为临时故障重试(`finish=length` 时上限翻倍),多次仍空则**抛错**而不是记一个空答案(空答案会被判成答错,和真错分不清);空响应永不写缓存;`request_timeout` 120 s(本地 vLLM 另设 `local_request_timeout` 900 s);判分 `max_tokens` 10 → 512(推理模型 10 个 token 全花在思考上,返回空)。

### 5. 分片并行

runner 外面切:100 道题的 id 轮流分 3 份,三个 `smem-eval` 进程各带 `--ids`、各写自己的 `--out` 目录,共享抽取/向量/ingest 三个缓存,结果拼三个 `records.jsonl`。绝不能让两个进程写同一个 records 文件——发生过一次,靠 `question_id` 去重发现的。

### 6. run 身份只含影响结果的字段

`config_hash` 曾把超时、重试次数、设备、缓存目录都算进去,改一个 timeout 半跑完的 run 就找不到自己的记录、从头重来。`SystemConfig.INFRA_FIELDS` 把这些剔出身份;续跑必须用和原 run **完全一致**的科学参数(多一个等价的 override 也会变身份,这一点还可以再改进:按解析后的配置算而不是按 override 字面量)。

### 7. 抽取 prompt 版本做成配置项

`extract.prompt_version`(`v3` / `v3-detail` / `v3-detail1`)是抽取缓存键的一部分,新旧版本在缓存里共存,按 run 选择。教训:直接改源码里的 `PROMPT_VERSION` 常量,正在跑的评测启动时读到新版本,缓存全部未命中,**静默地在评测里重抽**——三个分片 33 分钟 0 条记录才发现。

### 8. 向量缓存预热(`scripts/prewarm_embeddings.py`)

换抽取版本后条目文本全变,向量缓存全部未命中;分片在 CPU 上算(GPU 被 vLLM 占着),100 题多花 1.5 小时。正确顺序是重抽结束后先用**一个** GPU 进程批量把新条目的向量算进缓存(几分钟),再起分片。`CachedEmbedder` 加了 `has()` 和 30 s 的 sqlite 锁等待。

## 效果

| | 之前 | 之后 |
|---|---|---|
| 100 题参照 run(不淘汰) | 5-12 小时,需人盯限流/挂起 | **14 分钟**(缓存热) |
| 100 题 B=0.015 | 3-5 小时 | **~35 分钟** |
| 只改读侧参数的重跑 | 同上 | 分钟级(ingest 全命中) |
| 换抽取版本后的首轮 | — | +1.5 h(CPU 算向量);用预热脚本可降到几分钟 |
| 全量重抽(4564 场) | 4.9 h | 6.7 h(新 prompt 输出更长;32 路并发会把 KV cache 挤到 99%,16 路反而更快) |

## 换抽取版本的标准流程

```bash
# 1. 重抽(GPU,~5-7 h),16 路
smem-extract --split dev --config configs/default.yaml --workers 16 --ablate extract.prompt_version=<v>
# 2. 预热向量缓存(GPU,几分钟)
python scripts/prewarm_embeddings.py --split dev --prompt-version <v>
# 3. 分片评测(每片一个 --out 目录),ingest 缓存自动建立
smem-eval ... --ids "$(cat shard0.ids)" --out evals/results_<v>/shard0 --ablate extract.prompt_version=<v> --ablate consolidation.policy=no_consolidation
```

## 运维上的几条

- 后台任务用 `setsid ... & disown`,能活过会话重启;但**会话重启后先盘点还在跑的启动脚本**——丢失上下文里写的脚本会继续拉起进程,出现过重复写入。
- 杀进程按 /proc 扫 cmdline 并排除自身,不要 `pkill -f`——模式会匹配到自己的 shell,自杀过三次。
- 看门狗判"挂起"要以**进程自身启动时间**为基准;按文件时间戳判,刚续跑的进程会被误杀。
- `HF_HUB_OFFLINE=1`:模型都在本地,否则每次加载都去 Hub 查更新,Hub 抽风时整题卡死且无超时。
- 编辑正在运行的 bash 脚本无效(bash 持有旧 inode),要重启脚本。

## 还没做的

- 合并模块的替换语义(见上),之后重新打开合并要再评估一次。
- ingest 缓存压缩(float16 / 压缩 npz),40 MB/题可以降到 1/4。
- runner 原生 `--shard k/n`,替代命令行上的一长串 id。
- run 身份按解析后的配置计算。
