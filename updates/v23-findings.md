# v23 Findings — 关于 Bayesian 边界、巨石文件与 framework 重组

> 2026-04-25  
> 背景：v23 已经把 `spot_trader` 推进到 run-scoped / constraint-active / Bayesian-style refinement 原型，但这也暴露出新的框架层组织问题。  
> 本文聚焦三个问题：  
> 1. 为什么 Bayesian 代码现在在 `domains/spot_trader/` 里  
> 2. 哪些大文件已经到了必须拆分的程度  
> 3. framework 目录结构下一步应该如何重组

---

## 1. 一句话结论

> **当前 v23 的 Bayesian 代码放在 `domains/spot_trader/` 里，作为一次 domain-driven spike 是可以解释的，但从长期架构上看已经越界；与此同时，`framework/analyze.py`、`framework/core/db.py` 等文件已经明确进入“巨石化”状态，下一阶段应先完成 framework 的结构化拆分，再把 Bayesian loop 从 domain 中抽回 framework。**

更直接地说：

1. **现在的放置方式并不优雅，但不是完全错误。**
2. **下一阶段必须开始结构重组，否则功能继续增加后，维护成本会急剧恶化。**
3. **最先要拆的不是所有 domain 文件，而是 framework 的几个“总控型巨石”。**

---

## 2. 问题 1：为什么 Bayesian 代码现在在 `spot_trader` domain 里

## 2.1 当前实现为什么会落在 domain 内

这次 v23 新增的：

```text
domains/spot_trader/bayesian_refine.py
```

从工程边界看，确实会让人立刻不舒服，因为“Bayesian frontier refinement”听上去明显像 framework 能力，而不是某个 domain 的私有能力。

但它之所以会先出现在 `spot_trader` 里，是因为这次实现实际上混在了一起做了 **三层东西**：

1. **domain-specific candidate universe**
   - treasury scenario
   - sell / buy currency
   - route template
   - sell amount mode
   - quote scenario

2. **domain-specific evaluation**
   - `evaluate_route()`
   - `MockQuoteProvider`
   - liquidity floor / headroom / uplift / spread / lag

3. **generic-ish Bayesian loop**
   - seed observations
   - surrogate fitting
   - uncertainty-guided acquisition
   - batch resampling
   - random baseline 对比

也就是说，当前这个文件并不是“纯 Bayesian engine”，而是：

> **FX candidate universe + FX evaluator + FX benchmark harness + 一个初版 BO loop 的混合体。**

因此，它先放在 domain 中，在 v23 阶段作为研究原型，是可以理解的：

- 因为最初目标是验证 `spot_trader` 这个 domain 能不能承载 constrained Bayesian refinement；
- 不是先做一个抽象完美、但尚未被真实 domain 证明可用的泛化引擎。

换句话说，**它是一个 spike，先让研究闭环成立，而不是先做架构终局。**

## 2.2 但从长期架构看，它已经越界了

虽然这种落点在 v23 可以接受，但如果继续沿着这个方向堆功能，就会出现三个问题：

1. **framework 能力被 domain 吸走**
   - surrogate fitting
   - acquisition scoring
   - replay benchmark
   - frontier coverage 评价
   - knee targeting
   这些都不该永久停留在 `domains/spot_trader/`。

2. **下一 domain 无法复用**
   - 如果 v24/v25 进入新 domain，而 Bayesian loop 仍绑死在 `spot_trader`，那就会重复造轮子。

3. **domain 与 framework 的职责边界变脏**
   - domain 本应回答：
     - 候选如何生成
     - 特征如何编码
     - 单点评估如何执行
     - 指标如何解释
   - framework 本应回答：
     - 如何做 Pareto / knee
     - 如何做 surrogate / acquisition
     - 如何组织 benchmark / replay
     - 如何输出研究 artifact

所以，结论不是：

> “v23 把 Bayesian 放在 spot_trader 里是错得离谱的。”

而是：

> **“v23 作为 spike 可以先放在 `spot_trader`，但下一步应该把 generic Bayesian machinery 抽回 framework，把 `spot_trader` 保留为 adapter / evaluator / candidate provider。”**

## 2.3 推荐的正确拆分

推荐把它拆成：

### framework 层

```text
framework/services/bayes/
  interfaces.py
  surrogate.py
  acquisition.py
  loop.py
  benchmark.py
```

职责：

1. `interfaces.py`
   - 定义 domain adapter 协议：
     - `enumerate_candidates()`
     - `encode_candidates()`
     - `evaluate_candidate()`
     - `point_from_result()`
     - `default_objectives()`

2. `surrogate.py`
   - surrogate fitting
   - posterior mean / uncertainty

3. `acquisition.py`
   - feasibility-aware acquisition
   - frontier-gap bonus
   - knee bonus
   - uncertainty bonus

4. `loop.py`
   - seed -> fit -> score -> evaluate -> repeat

5. `benchmark.py`
   - random / heuristic / BO baseline comparison
   - budget curve
   - frontier hit / recall / infeasible rate

### domain 层

```text
domains/spot_trader/
  bayes_adapter.py
  train.py
  route_eval.py
  mock_provider.py
```

职责：

1. `bayes_adapter.py`
   - 生成 FX candidate universe
   - 编码 mixed features
   - 调用 `evaluate_route`
   - 解释 FX-specific point labels / plot defaults

2. `train.py`
   - 维持单点执行入口

也就是说，**Bayesian loop 应该回 framework；FX 只保留 adapter。**

## 2.4 推荐迁移策略

不要下一步直接删除 `domains/spot_trader/bayesian_refine.py`。

正确方式是：

1. 先把 generic 逻辑抽到 `framework/services/bayes/`
2. 再让 `domains/spot_trader/bayesian_refine.py` 变成一个薄 facade：
   - 解析 CLI
   - 注入 `FxSpotBayesAdapter`
   - 调用 framework 的通用 loop

这样可以：

1. 不打断 v23 已经存在的命令和文档
2. 逐步完成抽象
3. 给下一 domain 复用 BO loop 留出稳定接口

---

## 3. 问题 2：现在有哪些文件已经到了必须拆分的程度

## 3.1 本轮扫描到的最大 Python 文件

按行数统计，当前最大的 Python 文件包括：

| 文件 | 行数 | 判断 |
|---|---:|---|
| `framework/analyze.py` | 2700 | **必须拆，P0** |
| `domains/gomoku/train.py` | 2664 | 很大，但更偏 domain 内部；不是当前第一优先级 |
| `framework/core/db.py` | 2234 | **必须拆，P0** |
| `framework/sweep.py` | 827 | **应该拆，P1** |
| `domains/gomoku/prepare.py` | 764 | 应拆，但优先级低于 framework 巨石 |
| `framework/branch.py` | 666 | **应该拆，P1** |
| `framework/selector.py` | 591 | **应该拆，P1** |
| `domains/spot_trader/bayesian_refine.py` | 561 | **应该拆，且涉及边界修正，P0/P1** |
| `framework/promote.py` | 447 | 应拆 |
| `framework/pareto_plot.py` | 407 | 开始偏大，应拆为 frontier plotting/export |

这里最重要的不是“行数大就一定该拆”，而是看是否已经同时承担了 **多类职责**。

真正危险的是这类文件：

1. **既有 CLI，又有业务逻辑**
2. **既有 schema/migration，又有大量 repository CRUD**
3. **既有 domain 逻辑，又有 generic framework 能力**
4. **既有数据查询，又有格式化输出，又有 plot/export**

这几类一旦同时出现，就会迅速进入“改一个点、全文件都容易受影响”的状态。

## 3.2 `framework/analyze.py`：已经是“命令总控 + 查询总控 + 报告总控 + Pareto 总控”

`framework/analyze.py` 当前已经混在一起承担了：

1. CLI argument parser
2. runs / checkpoints / compare / report 命令
3. campaign summary / stage summary / promotion log
4. branch / trajectory report
5. recommendation / acquisition report
6. Pareto / knee / plot / frontier snapshot
7. JSON/Markdown 输出格式化

也就是说，它不是一个单一模块，而是一个：

> **“把整个 research observability 面都塞进去的总控脚本。”**

这类文件的问题不是“看起来长”，而是：

1. command surface 太多
2. import dependencies 太多
3. report logic / pareto logic / formatting logic 缠在一起
4. 很难局部修改而不触碰其他命令

### 推荐拆法

```text
framework/facade/analyze_cli.py
framework/services/reporting/run_reports.py
framework/services/reporting/campaign_reports.py
framework/services/reporting/recommendation_reports.py
framework/services/frontier/pareto.py
framework/services/frontier/knee.py
framework/services/frontier/snapshots.py
framework/services/frontier/exports.py
```

保留：

```text
framework/analyze.py
```

但让它只做兼容 facade：

1. parse args
2. dispatch 到 service

不要再把所有算法和报表细节继续堆进这个文件。

## 3.3 `framework/core/db.py`：已经不是 db helper，而是整个系统的数据库世界

`framework/core/db.py` 现在同时承担：

1. SQLite connection
2. `_SCHEMA_SQL`
3. migration DDL
4. hardware info
5. runs CRUD
6. checkpoints CRUD
7. recordings CRUD
8. opponents CRUD
9. campaign / search-space / stage / promotion / recommendation / surrogate CRUD
10. run_metrics / frontier_snapshots / fx quote tables / experiment_runs

换句话说，它已经不是：

> “一个数据库 helper”

而是：

> **“整个项目的数据库 schema + migration + repository + utility 全家桶。”**

这会导致两个问题：

1. 每加一个新功能，就更容易继续往这一个文件里堆
2. 任何 schema / repo 改动都要进入一个 2000+ 行的总文件

### 推荐拆法

最关键的一步，是把：

```text
framework/core/db.py
```

改成 package，而不是单文件。

推荐目标：

```text
framework/core/db/
  __init__.py
  connection.py
  migrations.py
  schema_base.py
  schema_campaigns.py
  schema_recommendations.py
  schema_fx.py
  repo_runs.py
  repo_campaigns.py
  repo_recommendations.py
  repo_fx.py
  repo_frontier.py
```

其中：

1. `__init__.py`
   - 继续 re-export 现有常用 API，保证旧 import 暂时不崩：
     ```python
     from core.db import init_db, create_run, finish_run
     ```

2. `connection.py`
   - `_connect`
   - `init_db`
   - PRAGMA / connection handling

3. `migrations.py`
   - 所有 `ALTER TABLE ... ADD COLUMN`
   - versionless migration steps

4. `schema_*.py`
   - 按领域分 DDL：
     - base runs/checkpoints
     - campaigns/promotion
     - recommendations/surrogates
     - fx evidence

5. `repo_*.py`
   - 按聚合根拆 CRUD

### 为什么先拆 db

因为 `core.db` 现在被大量地方直接 import，包括：

- framework
- domains/gomoku
- domains/spot_trader
- tests

它已经是最核心的依赖节点之一。

如果这个点不拆，后面继续长功能，所有目录重组都会被它反向牵制。

## 3.4 `framework/sweep.py`：CLI、矩阵生成、workspace、执行 orchestration 混在一起

`sweep.py` 当前至少混了：

1. CLI parsing
2. dynamic axis parsing
3. search-space validation
4. campaign / objective profile 绑定
5. fx workspace setup
6. subprocess execution
7. recommendation execution
8. auto Pareto handoff

这说明它已经不只是“扫参数”，而是：

> **一个 execution orchestrator。**

### 推荐拆法

```text
framework/facade/sweep_cli.py
framework/services/execution/matrix.py
framework/services/execution/workspace.py
framework/services/execution/sweep_runner.py
framework/services/execution/recommendation_execution.py
```

保留 `framework/sweep.py` 为兼容入口即可。

## 3.5 `framework/branch.py`、`selector.py`、`promote.py`

这几个文件单个没有 `analyze.py` / `db.py` 那么夸张，但它们已经在走同一条路：

1. policy load
2. DB access
3. business rule
4. CLI / output

推荐统一重组到：

```text
framework/services/research/
  branch_service.py
  selector_service.py
  promotion_service.py
  recommendation_service.py
```

配合：

```text
framework/facade/
  branch_cli.py
  recommend_cli.py
  promote_cli.py
```

## 3.6 `domains/gomoku/train.py` 很大，但不是当前第一刀

`domains/gomoku/train.py` 2664 行，当然也是巨石。

但它和 `analyze.py` / `db.py` 不一样：

1. 它主要还是 domain 内部训练逻辑
2. 它不是 framework 的全局控制点
3. 现在优先拆它，不会像拆 `analyze.py` / `db.py` 那样立刻改善整个项目的组织结构

所以我会把它归类为：

> **“应拆，但在 framework 重组之后再拆。”**

推荐未来拆成：

```text
domains/gomoku/
  model.py
  self_play.py
  replay_buffer.py
  evaluation.py
  runtime.py
  train.py   # 只保留 facade
```

## 3.7 `domains/spot_trader/bayesian_refine.py` 是“边界错误型大文件”

它的 561 行不是最大的问题，最大的问题是：

1. 既包含 generic BO loop
2. 又包含 FX-specific candidate generation
3. 又包含 replay benchmark persistence
4. 又包含 plot/export handoff

因此它比一些更大的 domain 文件更值得优先处理。

它的优先级并不是因为“太长”，而是因为：

> **它刚好站在 framework/domain 的交界线上。**

---

## 4. 问题 3：framework 目录下一步应该怎么重组

## 4.1 现在的主要结构问题

当前 `framework/` 下的情况是：

1. 平铺的顶层脚本越来越多
2. `core/` 作为唯一“抽象区”，开始承担过多含义
3. `core/` 里面其实也并不是真正统一的 infra taxonomy
4. 新功能一旦落不到现成位置，就很容易继续平铺

比如当前：

```text
framework/
  analyze.py
  sweep.py
  branch.py
  promote.py
  selector.py
  acquisition.py
  pareto_plot.py
  objective_profile.py
  search_space.py
  stage_policy.py
  branch_policy.py
  selector_policy.py
  core/
```

这在项目早期是高效的，但当命令面、服务面、研究面都变大后，就会开始失控。

## 4.2 我赞成新增 `services/` 与 `facade/`

我同意你提出的方向，而且不只是“为了目录漂亮”。

它们在这个项目里有真实职责：

### `facade/`

职责不是放业务逻辑，而是放：

1. CLI entry
2. compatibility wrappers
3. human-facing orchestration entrypoints

也就是说，`facade/` 是：

> **外部看到的稳定门面层。**

### `services/`

职责是：

1. frontier service
2. campaign execution service
3. recommendation service
4. promotion service
5. bayesian refinement service
6. reporting service

也就是说，`services/` 是：

> **内部真正的业务/研究编排层。**

### `core/`

`core/` 应该收缩回真正底层：

1. db connection / migrations
2. native bindings
3. low-level infra utilities

不要再让 `core/` 继续变成“所有抽象都往里塞”的黑洞。

## 4.3 推荐的目录目标

推荐目标结构：

```text
framework/
  facade/
    analyze_cli.py
    sweep_cli.py
    branch_cli.py
    promote_cli.py
    recommend_cli.py

  services/
    frontier/
      pareto.py
      knee.py
      snapshots.py
      plotting.py
      exports.py

    execution/
      workspace.py
      matrix.py
      sweep_runner.py
      recommendation_execution.py

    research/
      selector_service.py
      branch_service.py
      promotion_service.py
      recommendation_service.py

    bayes/
      interfaces.py
      surrogate.py
      acquisition.py
      loop.py
      benchmark.py

    reporting/
      run_reports.py
      campaign_reports.py
      recommendation_reports.py
      trajectory_reports.py

  core/
    db/
      __init__.py
      connection.py
      migrations.py
      schema_base.py
      schema_campaigns.py
      schema_recommendations.py
      schema_fx.py
      repo_runs.py
      repo_campaigns.py
      repo_recommendations.py
      repo_fx.py
    native/
      mcts.py
      mcts_native.py
    tui/
      tui.py

  analyze.py
  sweep.py
  branch.py
  promote.py
  selector.py
```

注意最后那几个顶层文件 **不必立刻删除**。

它们可以保留为兼容 shim：

1. 保持当前命令路径不变
2. 保持测试和文档暂时不变
3. 内部再逐步转调到 `facade/`

这比一次性移动全部文件更稳。

## 4.4 为什么我不建议“大爆炸重排”

因为当前项目里有大量地方直接依赖这些模块路径：

- `from core.db import ...`
- `from search_space import ...`
- `from objective_profile import ...`
- `framework/analyze.py`
- `framework/sweep.py`

如果下一步直接：

1. 把文件全搬走
2. 改所有 import
3. 改所有测试
4. 改所有脚本

那会把“结构优化”变成一次高风险行为变化。

因此正确策略应是：

> **先抽出新目录和新模块，再让旧入口做兼容 re-export / dispatch，最后再逐步清理旧路径。**

---

## 5. 推荐的执行路径

## Phase A — 先建立新骨架，不改外部命令

先创建：

```text
framework/facade/
framework/services/
framework/core/db/
```

但先不大规模改调用方。

目标：

1. 新的组织骨架先出现
2. 后续迁移有落点
3. 外部命令仍然稳定

## Phase B — 先拆 `core/db.py`

这是第一优先级。

顺序：

1. 把 schema / migrations / repos 拆出去
2. 在 `framework/core/db/__init__.py` 里继续 export 老 API
3. 让旧 import 暂时不变

为什么先拆它：

1. 它是依赖中心
2. 它影响所有 command / domain / tests
3. 先拆开后，后续所有服务层更容易组织

## Phase C — 再拆 `analyze.py`

拆成：

1. CLI facade
2. reporting services
3. frontier services
4. format/export services

重点：

1. 不改变现有 `uv run python framework/analyze.py ...` 的命令入口
2. 先内部拆，再外部保持兼容

## Phase D — 拆 `sweep.py` + `branch.py` + `selector.py` + `promote.py`

这一步是把“研究 orchestration”从扁平脚本变成 service 层。

重点：

1. workspace setup
2. matrix generation
3. execution runner
4. selector / promotion / branch service

## Phase E — 把 Bayesian 从 `spot_trader` 抽回 framework

等到：

1. `services/`
2. `facade/`
3. `core/db/`

都出现之后，再做这一刀最合理。

因为那时才有地方放：

```text
framework/services/bayes/
```

此时再让：

```text
domains/spot_trader/bayesian_refine.py
```

退化为薄 facade / adapter，就顺理成章了。

## Phase F — 最后再处理 domain 内部巨石

例如：

- `domains/gomoku/train.py`
- `domains/gomoku/prepare.py`
- `domains/spot_trader/train.py`

这些应该拆，但不应抢在 framework 结构调整之前。

---

## 6. 我的推荐判断

如果只问我一句话建议，我会这样回答：

> **是的，Bayesian 代码最终应该属于 framework，而不是永久留在 `spot_trader`；同时，`analyze.py` 和 `core/db.py` 已经到了必须拆分的程度；而 framework 目录也确实应该进入 `facade/ + services/ + 收缩后的 core/` 的下一轮重组。**

但执行上，我不建议一步到位大重排。

我建议的顺序是：

1. **先拆 `core/db.py`**
2. **再拆 `analyze.py`**
3. **再拆 execution / research orchestration**
4. **然后把 Bayesian 抽回 framework**
5. **最后再拆 domain 内部巨石**

这是成本最低、风险最可控、同时也最利于下一阶段扩展的路径。

---

## 7. 最终结论

当前 v23 已经证明了：

1. `spot_trader` 可以承载 constrained Bayesian-style frontier refinement
2. 但这次实现也证明了：
   - framework 边界需要重整
   - 大文件需要拆
   - 目录组织需要升级

所以，v23 的新 findings 不再只是关于 FX front 本身，而是关于 **autoresearch 框架已经进入第二次结构重构窗口**。

更准确地说，当前最重要的结论是：

> **下一阶段不应只继续加功能，而应先把 framework 的边界、目录和巨石文件整理好；否则后续再增加 domain、Bayesian 能力、可视化和 adapter，系统会在组织层先失控。**
