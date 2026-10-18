# TraceMind

[中文](#中文) | [English](#english)

TraceMind 使用指标、链路追踪和日志调查微服务故障，输出包含证据、假设和明确缺口的结构化诊断。项目使用 OpenTelemetry Demo 中可复现的故障场景进行验证。

默认采用受预算约束的 LangGraph 单 Agent。可选的多 Agent 工作流包含协调者、最多两个并发调查任务，以及有限轮次的验证反馈。两种模式共用权限控制、模型与工具调用预算、保留证据和持久化运行状态。遥测数据由独立的只读 HTTP MCP 服务提供。

### 已实现能力

- 固定调查时间窗口、证据引用与来源校验。
- 上下文限制、有界重试、可选备用模型与受控收尾。
- 持久化任务队列、worker 租约、checkpoint、取消与恢复。
- 持久化 SSE 进度事件与事件后续追问。
- 默认开启的完成验证，明确区分部分结果和失败。
- 基于快照的 RAG 与反馈记忆，默认关闭。
- Langfuse 追踪与评估、脚本化回归回放和真实故障用例。

当前支持本地单 worker 部署，并行调查任务在该 worker 内执行。分布式 worker、共享服务鉴权、自动告警接入和故障修复工具不在当前范围内。恢复机制不保证外部模型或工具调用恰好执行一次。

### 环境要求

- Python 3.12+、`uv`，以及已配置的诊断与验证模型所需的 API 凭据。
- Docker 和 Compose，用于 Demo、遥测 MCP 及可选配套服务。
- 同级目录中的 OpenTelemetry Demo，固定到提交 `dedc0178918e260823323b8d95005a8cb924b007`（版本 3.1.0）。

以下命令均在仓库根目录执行。Demo 获取方式与故障场景见 [Demo 与评估](docs/evaluation.md)。

### 快速开始

#### 1. 安装与配置

```bash
uv sync --locked --dev
cp .env.example .env
```

示例配置使用 DeepSeek 进行诊断与验证，并配置 DeepSeek Flash 作为备用模型。在本地 `.env` 中填写对应密钥：

```dotenv
TRACEMIND_LLM_PROVIDER=deepseek
TRACEMIND_LLM_MODEL=deepseek-v4-pro
TRACEMIND_DEEPSEEK_API_KEY=your-deepseek-key
TRACEMIND_AGENT_FALLBACK_PROVIDER=deepseek
TRACEMIND_AGENT_FALLBACK_MODEL=deepseek-flash
TRACEMIND_VERIFIER_ENABLED=true
TRACEMIND_VERIFIER_PROVIDER=deepseek
TRACEMIND_VERIFIER_MODEL=deepseek-v4-pro
```

诊断支持 `google`、`openai` 和 `deepseek`，分别配置对应模型与 `TRACEMIND_GOOGLE_API_KEY`、`TRACEMIND_OPENAI_API_KEY` 或 `TRACEMIND_DEEPSEEK_API_KEY`。切换供应商时，还需检查独立配置的备用模型与验证模型及其密钥；停用备用模型时需同时移除其 provider 和 model 配置。上述模型名来自仓库配置示例。

单 Agent 模式可通过 `TRACEMIND_VERIFIER_ENABLED=false` 显式关闭验证；多 Agent 模式始终要求验证。

[.env.example](.env.example) 列出了配置项与诊断预算。填写了凭据的环境文件应保留在本地。修改配置后重启 API 与 worker；已接收的任务继续使用其保存的限制和策略。

#### 2. 启动 Demo 与遥测服务

将固定版本的 Demo 放在 `../opentelemetry-demo`，并启动 Docker：

```bash
uv run python scripts/demo_scenario.py prepare
uv run python scripts/demo_scenario.py start
docker compose -f deploy/compose.telemetry-mcp.yaml up -d --build
uv run python scripts/check_mcp.py
```

默认场景启用 `paymentFailure=100%`。等待负载生成器产生结账失败后再开始调查。生成的配置位于已忽略的 `artifacts/` 下。遥测 MCP 默认地址为 `http://127.0.0.1:8012/mcp`；其后端配置见独立的[遥测服务文档](services/otel-telemetry-mcp/README.md)。

#### 3. 启动 API 与 worker

在使用相同环境配置的两个终端中分别运行：

```bash
uv run uvicorn tracemind.api.app:app --reload
```

```bash
uv run python scripts/run_worker.py
```

`GET /health` 不需要模型密钥。交互式 API 文档位于 `http://127.0.0.1:8000/docs`。队列中的任务需要 worker 执行。

#### 4. 调查与追问

```bash
uv run python scripts/demo_investigation.py --lookback-minutes 10
```

该命令提交调查、调用真实模型、流式显示进度并输出诊断。保留 investigation ID，以便继续追问：

```bash
uv run python scripts/demo_investigation.py --investigation-id <id> \
  --purpose explain --message "Why is payment the root cause?"
```

解释模式读取已保留的证据和知识。使用 `--purpose investigate` 可以查询新的遥测数据；每一轮拥有独立的运行记录与预算。

### 可选能力

**多 Agent 调查：** 设置 `TRACEMIND_AGENT_MODE=multi`。`graph-v7` 工作流每批最多委派两个任务，执行有限轮次的汇总与验证。部分任务结果和原始观测会在可恢复中断后保留。预算准入、上下文限制和恢复边界见[多 Agent 架构](docs/multi-agent.md)。

每个任务接收冻结的输入，其中包含验收标准、明确的证据引用和收窄后的工具权限。`TRACEMIND_AGENT_MAX_TOTAL_TOKENS` 可选地限制跨角色、重试与验证的累计模型 token 准入量，独立于单次请求的上下文限制。完成验证同时检查原始任务标准是否满足，以及结论是否有证据支持。

**知识检索：** 安装 `rag` 额外依赖，准备并发布快照，再配置知识工具权限与快照参数。RAG 结合图、向量与 BM25 候选、排序融合和重排。配置了 `TRACEMIND_RAG_SERVICE_URL` 时，先启动独立服务：

```bash
uv run --extra rag python scripts/run_rag_service.py
curl --fail http://127.0.0.1:8011/ready
```

`TRACEMIND_RAG_SNAPSHOT` 必须与 `TRACEMIND_AGENT_KNOWLEDGE_SNAPSHOT` 一致。启用知识检索时，API 与 worker 的启动命令也需要添加 `--extra rag`。Qdrant 配置位于 `deploy/compose.rag.yaml`。详见 [RAG 配置](docs/rag.md)与[服务运行说明](docs/rag-online-integration.md#independent-query-service)。

**反馈记忆：** 单 Agent 模式可以保留限定作用范围、经过人工确认的事件经验和持久指导。启用时必须明确环境与部署范围。多 Agent 模式当前不允许开启反馈记忆。配置与保留规则见[运行指南](docs/operations.md)。

### API 概览

| 接口 | 用途 |
| --- | --- |
| `POST /api/v1/diagnoses` | 等待同步诊断结果 |
| `POST /api/v1/runs` | 将持久化任务加入队列，返回 202 |
| `GET /api/v1/runs/{id}` | 查询运行状态 |
| `GET /api/v1/runs/{id}/result` | 获取诊断与证据 |
| `POST /api/v1/runs/{id}/cancel` 或 `/resume` | 取消任务或重新入队 |
| `GET /api/v1/runs/{id}/events` | 订阅持久化 SSE 进度事件 |
| `POST /api/v1/investigations` | 创建事件及首个运行任务 |
| `POST /api/v1/investigations/{id}/messages` | 提交后续问题 |

创建请求需要指定带时区的调查时间窗口。同步响应包含 `X-TraceMind-Run-ID` 和 `X-TraceMind-Stop-Reason`。请求示例与生命周期规则见[运行指南](docs/operations.md)。

### 测试与评估

```bash
uv run pytest
uv run --extra rag pytest --require-rag
```

基础安装跳过可选 RAG 模块；第二条命令要求完整 RAG 依赖。脚本化测试与回放验证工作流行为，不能代替模型诊断质量评估。真实评估需要单独控制的故障场景、模型凭据和明确的调用预算。

配置本地 Langfuse 后，可运行带评分的 payment-failure 实验：

```bash
uv run python scripts/run_experiment.py --task agent --name payment-baseline \
  --case evals/cases/payment_failure.json --lookback-minutes 10 \
  --env-file .env --env-file deploy/.env.langfuse --judge --max-model-calls 20
```

完成验证器根据保留证据检查编号断言。单 Agent 模式下，验证拒绝会保留有证据支持的部分结论，不会自动重新调查；多 Agent 模式可以根据有界反馈修订或继续调查。验证失败不能批准未经审查的草稿。机制尚不明确时，已有证据支持的服务定位仍可保留。

模型质量结论需要来自受控评估，并保留失败、结论不足和超时的尝试。评估方法及其适用范围见[评估文档](docs/evaluation.md)。

### 本地数据与 Git

`.gitignore` 排除本地环境配置、运行与评估产物、checkpoint 数据库、知识语料、RAG 快照与模型缓存，以及日志、测试缓存和系统文件。默认运行数据位于 `artifacts/`，本地知识文件位于 `knowledge/`；自定义输出目录也应在提交前检查。

源码、测试、`evals/` 下的可复现用例、依赖锁文件和通用配置示例保留在版本控制中。忽略规则不会取消已跟踪文件的跟踪，也不会清理 Git 历史。

### 文档导航

以下详细文档目前以英文为主。

| 文档 | 内容 |
| --- | --- |
| [设计](DESIGN.md) | 架构、契约与限制 |
| [运行指南](docs/operations.md) | API、预算、持久化、进度与记忆 |
| [多 Agent 诊断](docs/multi-agent.md) | 协调、预算与恢复 |
| [Demo 与评估](docs/evaluation.md) | 故障场景、真实测试与评分 |
| [RAG 配置](docs/rag.md) | 导入、快照与检索评估 |
| [在线 RAG 集成](docs/rag-online-integration.md) | 服务与生命周期契约 |
| [遥测 MCP 服务](services/otel-telemetry-mcp/README.md) | 部署与协议 |
| [更新记录](CHANGELOG.md) | 版本与功能变更 |

Python 分发包和导入包均为 `tracemind`，应用配置前缀为 `TRACEMIND_`。旧部署需要更新环境变量名、导入路径及响应头使用方；Compose 项目和数据卷改名不会自动迁移现有本地服务数据。

---

TraceMind investigates microservice incidents using metrics, traces and logs,
then produces a structured diagnosis with supporting evidence, hypotheses and
explicit gaps. It runs against reproducible faults in the OpenTelemetry Demo.

A bounded LangGraph agent is the default. An opt-in multi-agent workflow adds a
coordinator, up to two concurrent investigators and bounded verifier feedback.
Both modes share permissions, model/tool budgets, retained evidence and durable
run state. Telemetry comes from an independent read-only HTTP MCP service.

### What is implemented

- Fixed investigation windows, evidence references and source validation.
- Context limits, bounded retries, optional fallback and controlled finalization.
- Persistent queued runs, worker leases, checkpoints, cancellation and recovery.
- Durable SSE progress and incident follow-up questions.
- Completion verification, enabled by default, with explicit partial/failure outcomes.
- Optional snapshot-based RAG and feedback memory, disabled by default.
- Langfuse tracing and evaluation, scripted regression replays and live fault cases.

The supported deployment is local with one worker. Parallel investigation tasks
run inside that worker; distributed workers, shared-service authentication,
automatic alert ingestion and remediation tools are outside the current scope.
Recovery does not guarantee exactly-once external model or tool execution.

### Requirements

- Python 3.12+, `uv`, and credentials for the configured diagnosis and review models.
- Docker with Compose for the Demo, telemetry MCP and optional supporting services.
- A sibling OpenTelemetry Demo checkout pinned to
  `dedc0178918e260823323b8d95005a8cb924b007` (version 3.1.0).

Commands below run from the repository root. See
[Demo setup](docs/evaluation.md) for checkout and fault scenario instructions.

### Quick start

#### 1. Install and configure

```bash
uv sync --locked --dev
cp .env.example .env
```

The example configuration uses DeepSeek for diagnosis and verification, with
DeepSeek Flash as the fallback. Set the matching key in your local `.env`:

```dotenv
TRACEMIND_LLM_PROVIDER=deepseek
TRACEMIND_LLM_MODEL=deepseek-v4-pro
TRACEMIND_DEEPSEEK_API_KEY=your-deepseek-key
TRACEMIND_AGENT_FALLBACK_PROVIDER=deepseek
TRACEMIND_AGENT_FALLBACK_MODEL=deepseek-flash
TRACEMIND_VERIFIER_ENABLED=true
TRACEMIND_VERIFIER_PROVIDER=deepseek
TRACEMIND_VERIFIER_MODEL=deepseek-v4-pro
```

Diagnosis supports `google`, `openai` and `deepseek`; set the matching model and
`TRACEMIND_GOOGLE_API_KEY`, `TRACEMIND_OPENAI_API_KEY` or
`TRACEMIND_DEEPSEEK_API_KEY`. Fallback and review have independent provider/model settings and require their
matching keys when switching providers. Remove both fallback provider and model
settings to disable fallback. Model names above come from repository configuration
examples.
To explicitly disable review in single-agent mode, set
`TRACEMIND_VERIFIER_ENABLED=false`; multi-agent mode requires review.

[.env.example](.env.example) documents settings and diagnostic budgets. Keep populated
environment files local. Restart API and worker after configuration changes;
admitted runs retain their saved limits and policies.

#### 2. Start the Demo and telemetry service

With the pinned Demo at `../opentelemetry-demo` and Docker running:

```bash
uv run python scripts/demo_scenario.py prepare
uv run python scripts/demo_scenario.py start
docker compose -f deploy/compose.telemetry-mcp.yaml up -d --build
uv run python scripts/check_mcp.py
```

The default scenario enables `paymentFailure=100%`. Allow the load generator to
produce checkout failures before investigating. Generated configuration lives
under ignored `artifacts/`. The telemetry MCP endpoint defaults to
`http://127.0.0.1:8012/mcp`; its backend configuration belongs to the independent
[telemetry service](services/otel-telemetry-mcp/README.md).

#### 3. Start API and worker

Run these in separate terminals with the same environment:

```bash
uv run uvicorn tracemind.api.app:app --reload
```

```bash
uv run python scripts/run_worker.py
```

`GET /health` requires no model key. Interactive API schemas are available at
`http://127.0.0.1:8000/docs`. Queued runs require the worker to execute.

#### 4. Investigate and follow up

```bash
uv run python scripts/demo_investigation.py --lookback-minutes 10
```

This submits an investigation, makes real model calls, streams progress and
prints a diagnosis. Keep the investigation ID for follow-up questions:

```bash
uv run python scripts/demo_investigation.py --investigation-id <id> \
  --purpose explain --message "Why is payment the root cause?"
```

Explanation uses retained evidence and knowledge. `--purpose investigate`
permits fresh telemetry queries; each round has its own run and budgets.

### Optional capabilities

**Multi-agent investigation:** set `TRACEMIND_AGENT_MODE=multi`. The `graph-v7`
workflow delegates at most two tasks per batch and performs bounded synthesis
and verification rounds. Partial task results and original observations survive
recoverable interruptions. See [multi-agent architecture](docs/multi-agent.md)
for admission requirements, context limits and recovery boundaries.

Each task receives a frozen input with acceptance criteria, explicit evidence
references and narrowed tool grants. `TRACEMIND_AGENT_MAX_TOTAL_TOKENS`
optionally caps cumulative model-token admission across roles, retries and
review; it is separate from per-request context limits. Completion review
checks whether the original task criteria were met as well as claim support.

**Knowledge retrieval:** install the `rag` extra, prepare and publish a snapshot,
then configure knowledge permissions and snapshot settings. RAG combines graph,
vector and BM25 candidates, rank fusion and reranking. With
`TRACEMIND_RAG_SERVICE_URL` configured, start the independent service first:

```bash
uv run --extra rag python scripts/run_rag_service.py
curl --fail http://127.0.0.1:8011/ready
```

`TRACEMIND_RAG_SNAPSHOT` must match `TRACEMIND_AGENT_KNOWLEDGE_SNAPSHOT`.
API and worker also need `--extra rag` when knowledge is enabled. Qdrant is
configured in `deploy/compose.rag.yaml`. See [RAG setup](docs/rag.md) and
[service operation](docs/rag-online-integration.md#independent-query-service).

**Feedback memory:** single-agent mode can retain scoped, human-confirmed
incidents and durable guidance. It requires explicit environment/deployment
scope. Multi-agent mode currently rejects enabled feedback memory. See
[operations](docs/operations.md) for configuration and retention rules.

### API overview

| Endpoint | Purpose |
| --- | --- |
| `POST /api/v1/diagnoses` | Wait for a synchronous diagnosis |
| `POST /api/v1/runs` | Queue a persistent run; returns 202 |
| `GET /api/v1/runs/{id}` | Read run status |
| `GET /api/v1/runs/{id}/result` | Read diagnosis and evidence |
| `POST /api/v1/runs/{id}/cancel` or `/resume` | Cancel or requeue a run |
| `GET /api/v1/runs/{id}/events` | Subscribe to durable SSE progress |
| `POST /api/v1/investigations` | Create an incident and its first run |
| `POST /api/v1/investigations/{id}/messages` | Ask a follow-up question |

Creation requires a timezone-aware investigation window. Synchronous responses
expose `X-TraceMind-Run-ID` and `X-TraceMind-Stop-Reason`. Request examples and
lifecycle rules are in [operations](docs/operations.md).

### Tests and evaluation

```bash
uv run pytest
uv run --extra rag pytest --require-rag
```

Base installations skip optional RAG modules. The second command requires all
RAG dependencies. Scripted tests and replays check workflow behavior, not model
diagnosis quality. Live evaluations require separate controlled faults, model
credentials and explicit call budgets.

After configuring local Langfuse, a scored payment-failure experiment can run as:

```bash
uv run python scripts/run_experiment.py --task agent --name payment-baseline \
  --case evals/cases/payment_failure.json --lookback-minutes 10 \
  --env-file .env --env-file deploy/.env.langfuse --judge --max-model-calls 20
```

The completion verifier reviews numbered assertions against retained evidence.
In single-agent mode, rejection retains supported partial findings without an
automatic investigation revision; multi-agent mode can use bounded feedback to
revise or investigate again. Review failures cannot approve an unreviewed draft.
Unknown mechanisms do not erase evidence-backed service localization.

Model quality claims require controlled evaluations that retain failed,
inconclusive and timeout attempts. See [evaluation](docs/evaluation.md) for
methods and their scope.

### Local data and Git

`.gitignore` excludes local environment configuration, run and evaluation output,
checkpoint databases, knowledge corpora, RAG snapshots and model caches, logs,
test caches and OS files. Default runtime data lives under `artifacts/` and local
knowledge files under `knowledge/`; check custom output directories before committing.

Source code, tests, reproducible cases under `evals/`, dependency lockfiles and
reusable configuration templates remain versioned. Ignore rules do not
untrack existing files or remove Git history.

### Documentation

The detailed documents below are primarily in English.

| Document | Contents |
| --- | --- |
| [Design](DESIGN.md) | Architecture, contracts and limitations |
| [Operations](docs/operations.md) | API, budgets, persistence, progress and memory |
| [Multi-agent diagnosis](docs/multi-agent.md) | Coordination, budgets and recovery |
| [Demo and evaluation](docs/evaluation.md) | Fault scenarios, live tests and scoring |
| [RAG setup](docs/rag.md) | Ingestion, snapshots and retrieval evaluation |
| [Online RAG integration](docs/rag-online-integration.md) | Service and lifetime contracts |
| [Telemetry MCP service](services/otel-telemetry-mcp/README.md) | Deployment and protocol |
| [Changelog](CHANGELOG.md) | Version and feature changes |

The Python distribution and import package are both `tracemind`; application
settings use `TRACEMIND_`. Existing local deployments must update their environment
keys, import paths and response-header consumers. Renamed Compose projects and
named volumes do not automatically migrate existing local service data.
