# 本地 Langfuse 使用与评测流程

本项目把 Langfuse 用作调查运行的可视化、prompt 版本管理和人工评测平台。
它不负责执行故障修复，也不自动训练或校准模型。

## 登录与数据位置

- 页面：<http://127.0.0.1:3000>
- 项目：**TraceMind → TraceMind Evaluation**。
- 项目入口：<http://127.0.0.1:3000/project/tracemind-evaluation>
- 当前自托管版本：v4.77.0；无需 Langfuse Cloud 账号。

本地 Compose 中，PostgreSQL 保存账户、组织、项目和配置等数据；
ClickHouse 保存观测和评分；MinIO 保存事件及文件；Redis 负责队列和缓存。
这些都是本机 Docker 服务，使用本地 Docker volumes。
OTel Demo 的 Jaeger、OpenSearch、Prometheus 是另一套被调查系统的遥测后端。
DeepSeek 推理仍通过远程 API 进行；“Langfuse 本地”不代表模型也在本地。

登录使用自己的本地账号。账号密码不要写进仓库、prompt 或评测数据。
不要为了登录再次执行 `setup_langfuse.py`：它生成部署密钥，不是重置账号工具。

## 第一次看调查

1. 打开左上角 **Open menu → Tracing**。
2. 时间范围选择 **Past 7 days**；近期没有结果时先检查日期范围和项目。
3. 打开一次 `diagnosis`，查看输入故障、输出报告以及调用树。
4. 展开 `DeepSeekChat` 查看模型、token、推理 token、费用及耗时。
5. 展开 `search_logs` 等工具查看查询及返回；展开 `diagnosis.verification`
   查看本轮候选报告、证据包和审核意见。
6. **Sessions** 以 TraceMind run ID 聚合。恢复执行可能产生多个 root spans，
   因此 span 数量不等于独立调查数量。

Langfuse 记录的是 **Agent 如何调查**；查询到的 trace/log 是 **业务系统发生了什么**。
两者不能混作同一种 trace。

## 运行仪表盘

进入 **Dashboards → TraceMind initial triage v1**。已配置：

| 指标 | 解释 |
|---|---|
| Run count | 已终结的逻辑调查数；稳定 score ID 避免恢复/重发重复计数 |
| First triage P50/P95 | 首份可读初查出现时间；同时看初查覆盖率 |
| Final P50/P95 | 从运行开始到结束的墙钟时间，包含等待和电脑休眠 |
| Execution failure rate | 执行失败/超时等；审核异议单独统计 |
| Review completion rate | 最后一轮审核是否完成；首轮完成不代表复核完成 |
| First/Final review objection rate | 仅在相应审核完成的样本中统计异议 |
| Reported tokens / Usage coverage | 提供商实际报告的 token 与覆盖率 |
| Known model cost estimate / Cost coverage | 已知用量的费用估算及可计价调用覆盖率 |
| Unpriced model calls | 用量、缓存拆分或价格缺失的调用数 |
| Verifier/Revision P95 | 审核阶段与被拒后修订阶段的墙钟耗时 |

默认通过 `metadata.run_kind=investigation` 筛选完整调查。
独立验证器验收应标为 `verification_only`，不可混入完整调查的耗时比较。
模型、思考强度、prompt 版本和运行状态写入 score metadata，可按这些维度筛选。
历史记录没有的模型配置保持未知，不从当前 `.env` 猜测。

原始失败尝试与重跑都保留。只有几个样本时，P95 只是该小样本的分位数，
不能当成生产性能指标；应先查看单次记录、样本数和失败原因。

### 费用口径

价格快照：`deepseek-peak-list-usd-2026-11-12`。
按 [DeepSeek 公布的高峰标价](https://api-docs.deepseek.com/quick_start/pricing/)
计算 USD 估算；不计算时段/节假日折扣，不冒充供应商账单，不包含基础设施费用。
输入缓存命中和未命中分别计价；输出总量已包含 reasoning，不额外重复相加。

原生 generation 的模型价格规则只作用于新的计价记录。
历史 generation 没有价格时，Langfuse 某些总计会显示 `$0`，这不代表免费。
本项目的历史估算另写入 run scores；完整费用有缺失时保持 unknown，
只展示已知部分和覆盖率。不能把未知部分当零。

## Prompt 版本

进入 **Prompts**，当前管理两个文本 prompt：

- `tracemind/investigator`：单 Agent 调查与人工 follow-up 共用。
- `tracemind/verifier`：在线审核及独立 verifier 实验。

新建版本后先用测试标签试验，通过固定评测集再把 `production` 指向新版本。
运行配置是 `TRACEMIND_LANGFUSE_PROMPT_LABEL=production`。
一次运行开始时获取该标签，保存具体版本、完整渲染内容及 SHA256。
本次运行的修订和恢复不会跟随标签移动。模型 generation 会关联实际版本。
prompt 的 `config.contract_version` 与代码契约配套，不要在 UI 随意删除或改写。

工具权限、输出 schema 和预算仍由代码执行。
本批不接管 multi-agent coordinator/子任务 prompt，也不改变离线 Judge 的独立 rubric。
旧运行没有快照时不会伪称原 prompt 已固定；继续按原有恢复规则处理。
代码契约不兼容时要求新建运行，不复用旧审核。

启用远程标签后，首次获取失败会使新运行失败，不悄悄切换到不同 prompt。
已保存快照的运行恢复不需要重新获取。
需要使用随代码打包的 prompt 时，将该配置设为 `none`，之后启动的新进程生效。

## 人工标注与验证器校准

进入 **Human annotation → TraceMind verifier calibration v1 → Process queue**。
每个队列项对应一个具体的 `diagnosis.verification`，不是整个 run 的所有修订轮。

先独立阅读 input 中的候选条目和证据，再给以下维度评分：

| 维度 | 选项 |
|---|---|
| human.verdict | supported / insufficient / contradicted / unknown |
| human.severity | none / minor / material / unknown |
| human.usefulness | useful / limited / misleading / unknown |
| human.verifier | reasonable / too_strict / missed / unknown |

在评论中写出具体 item ID（如 F4）、相关证据和理由，然后点 **Complete + next**。
报告有局部事实错误但主要定位仍有用时，可以分别标为 insufficient 和 useful；
“有用”不能把错误引用变成 supported。
模型已有意见可以作为被评价对象，但不要直接照抄；不确定就选择 unknown。

**Scores → Analytics** 可比较 `tracemind.human.verdict` 与
`tracemind.verifier.verdict`，查看同一 observation 上的判定差异。
人工标签、模型评分和初查有用度各自保留，不能混同为根因正确率。

当前队列中的历史案例是待人工评分材料。自动导出只接受已完成队列中、
有真实作者的人工标签；缺失、unknown、冲突或自动预填均不进入确定标签集。
不会自动多数投票或自动发布修改后的 prompt。

### CLI 工作流

以下命令在仓库根目录、已配置 `.env` 下运行，使用现有虚拟环境，无新增依赖。

```sh
# 初始化：已有同名资源保留，不覆盖人工修改或重复创建。
PYTHONPATH=src .venv/bin/python scripts/manage_langfuse.py setup \
  --output artifacts/langfuse/setup.json

# 从一个已保存运行发布指标；不调用模型。
PYTHONPATH=src .venv/bin/python scripts/manage_langfuse.py metrics \
  --state artifacts/example/state.json --publish \
  --output artifacts/example/metrics.json

# 将可验证的审核轮次加入队列。QUEUE_ID 来自 setup.json。
PYTHONPATH=src .venv/bin/python scripts/manage_langfuse.py queue \
  --state artifacts/example/state.json --evidence artifacts/example/evidence.json \
  --queue-id QUEUE_ID --family cart-failure --split development \
  --output artifacts/example/annotation-manifest.json

# 人工在 UI 完成后导出。可重复传 --manifest；同故障家族不得跨 split。
PYTHONPATH=src .venv/bin/python scripts/manage_langfuse.py export-annotations \
  --manifest artifacts/example/annotation-manifest.json \
  --output-directory artifacts/labelled-verifier

# 检查 coverage.json；数据非空时才运行真实模型实验。
PYTHONPATH=src .venv/bin/python scripts/run_experiment.py \
  --task verification --data artifacts/labelled-verifier/holdout.json \
  --name verifier-heldout --max-model-calls 20
```

`development.json` 用于修改 rubric；`holdout.json` 用于未参与调优的验收。
同一次故障、重跑和修订属于同一 family，不能分散进两个集合。
导出保留原始证据/候选指纹，标签仅放在 expected_output/metadata，模型输入不含答案。
coverage.json 报告尚未完成人审、未知和冲突数量；重新导出会清空已失效的 split 文件。
历史材料无法验证指纹时会跳过，不能悄悄用最终报告替代旧轮候选。

误放率仅在“人工要求拒绝且模型审核完成”的样本计算；误拒率仅在
“人工要求接受且模型审核完成”的样本计算。审核失败另计完成率，
不能作为零错误稀释分母。没有适用样本时不输出该错误率。

## 接入边界

Langfuse 提供记录、标注、比较和版本发布能力。人工数据积累及独立评测完成前，
不能声称验证器已经校准，也不能把模型自报 confidence 当作正确概率。
本批不实现自动模型升级、自动调 prompt 或自动微调。
本地部署及 API 接口随版本可能变化，dashboard/widget API 目前属于 unstable。
