# myautoresearch

> 一个 **基于 benchmark、由 agent 驱动** 的研究框架。
> 框架是 domain-agnostic 的；五子棋（Gomoku）只是当前的第一个 domain 实例。

---

## 这是什么

**myautoresearch** 是一个开放的实验框架，目标是让一个 agent（大模型）能够：

1. **读懂** 自己跑出来的实验数据（tracker.db / cycle_metrics / checkpoints）
2. **形成假设**：基于 Pareto 前沿、晋升门禁、训练健康指标做下一步决策
3. **修改 domain 代码**（hyperparameters / 架构 / 训练循环）
4. **跑实验** 验证假设
5. **保留或回退**，然后进入下一轮

整个 loop 是有数据库支撑、可追溯的研究循环。框架本身是 domain-agnostic 的——它不知道五子棋是什么，只知道 `(成本轴, 真理指标)` 这样的抽象结构。

当前有两个 domain：**`domains/gomoku/`** 是第一个实证域，一个 15×15 五子棋的 AlphaZero 风格训练系统；**`domains/spot_trader/`** 是 v22 新增的第二个 domain，用 mock quote provider 演绎当前报价窗口内的外汇头寸 route / quote-surface 搜索。未来其他 domain 也通过相同协议挂载到框架上。

---

## 致谢与来源

这个项目从两个完全不同的方向汇流而来，必须明确致谢：

- **[Code Bullet](https://www.youtube.com/@CodeBullet)** — 五子棋 domain 的最初动机和视觉语言。原本目标是做一个像 Code Bullet 风格的 "AI 学下棋" 视频；`domains/gomoku/replay.py` 的成长蒙太奇回放、TUI 渲染都保留了这个基因。"做一个能讲故事的训练实验" 是这个 domain 永远不变的主题。
- **[autoresearch](https://github.com/humansandais/autoresearch)** — 框架名称与核心 loop 设计的灵感来源。"agent 主导的研究循环" 这个范式不是我们发明的，是从 autoresearch 这套思路里继承下来的。本项目把它具象化到了 "benchmark-constrained + Pareto-aware" 的方向上。

> 命名变更：项目早期叫 **mag-gomoku**（Multi-Agent Gomoku），随着框架层从 domain 解耦完成（v14-v15），重命名为 **myautoresearch**。Gomoku 现在是 `domains/gomoku/` 下的一个 domain，不再是项目身份。

---

## 架构

```
myautoresearch/
├── framework/                      ═══ 框架层（domain-agnostic）═══
│   ├── index.py                    统一 CLI 入口（analyze / sweep / promote / branch）
│   ├── core/
│   │   ├── db.py                   tracker.db schema + migrations + campaign / trajectory / recommendation / acquisition / run_metrics ledger
│   │   ├── tui.py                  ASCII sparkline / progress bar
│   │   ├── mcts.py                 通用 MCTS 算法（Python 参考实现）
│   │   ├── mcts_c.c                C 原生 MCTS 树操作（v14 落地）
│   │   ├── mcts_native.py          mcts_c 的 ctypes wrapper
│   │   └── build_native.sh         一行编译脚本
│   ├── facade/                     CLI facade / 参数解析
│   ├── services/                   frontier / reporting / research / execution / bayes
│   ├── policies/                   stage / branch / selector / acquisition policy
│   ├── profiles/                   search-space / objective-profile loader
│   └── templates/                  domain train / prepare template
│
├── domains/                        ═══ 可替换的 domain 层 ═══
│   ├── gomoku/                     —— 五子棋 domain ——
│       ├── train.py                训练入口（agent 修改的主目标文件）
│       ├── game.py                 棋盘引擎 (Board, BatchBoards, Renderer)
│       ├── prepare.py              minimax 对手 L0-L3 + evaluate_win_rate
│       ├── manifest/               domain policies / profiles
│       │   ├── search_space.json       v20.1 search-space profile
│       │   ├── stage_policy.json       v20.2 multi-fidelity stage policy
│       │   ├── branch_policy.json      v20.3 continuation / branch policy
│       │   ├── selector_policy.json    v21 recommendation policy
│       │   └── acquisition_policy.json v21.1 acquisition policy
│       ├── minimax_c.c             C 原生 minimax + pattern scorer (v15)
│       ├── minimax_native.py       minimax_c 的 ctypes wrapper
│       ├── play.py                 CLI 人机对弈 (pygame)
│       ├── play_service.py         共享对弈服务
│       ├── replay.py               棋局回放 / 成长蒙太奇导出
│       ├── web/                    FastAPI + 浏览器 UI
│       └── build_native.sh         编译 minimax_c
│   └── spot_trader/                —— spot FX quote-surface mock domain ——
│       ├── train.py                当前报价窗口内的 mock route evaluation 入口
│       ├── portfolio.py            当前头寸 / liquidity floor / headroom
│       ├── mock_provider.py        deterministic local quote provider
│       ├── route_eval.py           本地 mock execution + run_metrics
│       ├── airwallex_provider.py   quote-only adapter boundary（默认不执行 conversion）
│       ├── manifest/
│       │   ├── search_space.json       v22 dynamic route search-space
│       │   └── objective_profile.json  liquidity-first spot FX objectives
│       └── bayesian_refine.py      受约束 BO / frontier benchmark smoke
│
├── output/                         ═══ 训练产物（gitignore）═══
│   ├── tracker.db                  框架的核心数据库
│   ├── opponents/<alias>/          注册的对手权重
│   └── <run_uuid>/                 每次 run 的输出目录
│       ├── checkpoints/            .safetensors 模型权重
│       └── recordings/games/       .json 棋局录像
│
├── tests/                          单元 + 集成测试
└── updates/                        每个 version 的 update / findings 文档
```

**框架与 domain 的契约** 是两个东西：

1. **tracker.db schema**（数据协议）：Gomoku legacy 仍写 `final_win_rate / wall_time_s / total_games / total_steps / num_params`；v22 之后的新 domain 优先写 `run_metrics`，并用 `objective_profiles` 声明目标、约束、显示格式和 knee point 规则。
2. **CLI subprocess 协议**：`framework/index.py sweep` / `framework/index.py branch` 通过 `python domains/<name>/train.py --time-budget N --sweep-tag X ...` 启动 domain；v22 的 generic domain 还可以接收 `--candidate-json`。退出码 0 = 成功。

这两个协议合在一起就是 "如何接入新 domain"。详见下文 [Domain 接入指南](#domain-接入指南)。

而在 v20 之后，这个契约又向上扩了一层：  
除了训练脚本本身，domain 还需要提供自己的 `search_space / stage_policy / branch_policy / selector_policy / acquisition_policy`，框架才知道**什么点可比、什么点该晋升、什么 continuation 合法、什么 recommendation 值得执行**。

---

## 系统层次（5 层）

来自 [`updates/pareto-frontier.md`](updates/pareto-frontier.md) 的设计：

```
┌─────────────────────────────────────────────────────────────┐
│  Autoresearch Layer    高层判断：读报告、提假设、修代码、决策保留/回退 │
├─────────────────────────────────────────────────────────────┤
│  Local Exploration     局部探索：sweep / 多 seed / 阈值扫描 │
│   (index.py sweep)                                           │
├─────────────────────────────────────────────────────────────┤
│  Benchmark Layer       标准化真理：固定预算 / 固定对手 / 固定评估  │
│  (domain train.py + db.py)                                   │
├─────────────────────────────────────────────────────────────┤
│  Pareto Layer          Trade-off 组织：成本轴 vs 真理轴的非支配前沿 │
│  (index.py analyze --pareto)                                 │
├─────────────────────────────────────────────────────────────┤
│  Human Decision Layer  人类选点：在 Pareto 边界上做最终决定        │
└─────────────────────────────────────────────────────────────┘
```

**红线（不可被稀释）**：

- 真理优先：domain truth（如 Gomoku 的 WR）永远高于任何 Pareto 解释
- agent 主体：所有新层只能增强 agent 的感知和行动，不能替代 agent 做决策
- 人类不过早接管：人类在最后选 Pareto 点，但不指挥研究循环

> 今天的代码已经在这 5 层之上，继续长出了 **campaign / promotion / trajectory / recommendation / acquisition** 这条执行骨架。  
> 它们不是在替代原叙事，而是在把这个叙事真正做成一个可运行的研究系统。

---

## 快速开始

### 0. 环境

```bash
git clone <repo>
cd myautoresearch
uv sync   # 安装依赖（mlx, numpy, fastapi, pygame）

# 编译两个 C 扩展（必需）
cd framework/core && bash build_native.sh && cd ../..
cd domains/gomoku && bash build_native.sh && cd ../..
```

需要：

- macOS + Apple Silicon（MLX 必需 GPU）— 跑训练 / 推理
- Linux（无 MLX）— 只能跑 analyze / replay / 测试

### 1. 5 分钟 smoke training（vs L0 random）

```bash
uv run python domains/gomoku/train.py \
  --mcts-sims 800 --parallel-games 16 --mcts-batch 32 \
  --num-blocks 8 --num-filters 128 \
  --learning-rate 3e-4 --steps-per-cycle 50 --buffer-size 100000 \
  --time-budget 300 \
  --eval-level 0 \
  --eval-interval 5 --probe-games 80 \
  --full-eval-games 200 --eval-openings 16 \
  --seed 42
```

启动后 TUI 会显示训练进度、loss 曲线、health row、async eval 状态。Ctrl+C 随时安全中断。

### 2. 阅读训练报告

```bash
uv run python framework/index.py analyze --runs                 # 列出所有 run
uv run python framework/index.py analyze --report               # 实验报告（中文）
uv run python framework/index.py analyze --pareto               # Pareto 前沿
uv run python framework/index.py analyze --promotion-chain      # v15 新增
uv run python framework/index.py analyze --opening-breakdown <run_uuid>  # v15 新增
uv run python framework/index.py analyze --stability <run_uuid>
```

### 3. 浏览器对弈

```bash
uv run python domains/gomoku/web/web_app.py
# 浏览器打开 http://127.0.0.1:8000
```

前端会自动列出已注册的 NN 对手 + 内置 minimax L0-L3。

### 4. 注册一个 checkpoint 为对手

```bash
# 手动方式
uv run python domains/gomoku/train.py \
  --register-opponent S2 \
  --from-run <run_uuid> --from-tag wr090_c0246 \
  --description "8x128 vs L2, 90% WR"

# 自动方式（v15 新增）：训练结束时根据晋升门禁自动注册
uv run python domains/gomoku/train.py ... --auto-promote-to S2
```

---

## 当前阶段（v22）的核心能力

下面按“Gomoku domain 能力”和“framework research loop 能力”分开写。前者仍然建立在 v14-v15 的训练体系上，后者已经推进到 v22 的 multi-domain 泛化。

### Gomoku 训练侧（v14-v15 奠定的基础）

- **Native C MCTS**（v14）：800 sims × 16 boards 的批量树搜索在 C 里完成，比 Python 快 10-20×
- **Native C minimax**（v15）：L1/L2/L3 对手都在 C 里跑，L2 单次 minimax 从 ~2200ms → ~50ms（**~40× 加速**），L3 从不可用 → 可用
- **Async eval**（v15）：probe / full eval 在后台线程运行，主训练线程不阻塞。3h 训练里 wall-clock 效率从 ~33% → ~95%+
- **Stochastic minimax 对手**（v14）：L1/L2/L3 在根节点从 top-k 用 softmax 采样，避免确定性评估的统计坍缩
- **开局多样化评估**（v14）：每局 eval 从 16 条不同开局之一开始，配合 `unique_trajectories` 指纹保证 "200 局 eval" 不再退化成 "2 局复制"
- **policy / value loss 拆分**（v14）：训练时 P-loss 和 V-loss 单独入库，诊断"模型在学什么"
- **MLX allocator 周期性 clear_cache**（v14.1）：训练循环 RAM 不再涨到 115 GB

### Gomoku 训练完成后

- **晋升门禁** `can_promote()`（v15 E1）：4 条判据（WR 阈值 / unique_openings ≥ 16 / avg_length ∈ [12,60] / 最近 5 个 smoothed WR 稳定）一致通过才能被晋升
- **Auto-promote**（v15 E5）：`--auto-promote-to S2` 训练结束时自动注册，含 `prev_alias` 晋升链
- **Cross-level resume 修复**（v15 A1）：从 vs L1 的 100% checkpoint resume 到 vs L2 训练时，checkpoint 阈值链会正确重置
- **Per-opening WR 分解**（v15 E3）：可以查看模型在哪条开局上输了
- **TUI health row**（v15 E4）：5 个健康指标实时显示（learning / diverse / plateau / value / collapse）

### Gomoku reporting

- **`--pareto`**：成本 vs WR 的非支配前沿（v14 已有）
- **`--promotion-chain`**：S0 → S1 → S2 → ... 的对手进化链可视化（v15）
- **`--opening-breakdown`**：单个 run 所有 checkpoint 的 per-opening WR 表（v15）
- **`--report`**：完整中文实验报告（v14 已有）
- **`--stability`** / **`--stagnation`** / **`--matrix`** / **`--frontier`** / **`--lineage`** / **`--compare`** / **`--compare-by-steps`**

### Framework research loop（v20 → v21.1）

- **Campaign / search-space governance**（v20.1）
  - `search_spaces` / `campaigns` / `campaign_runs`
  - protocol drift guard
  - `index.py analyze --campaign-summary`

- **Multi-fidelity promotion**（v20.2）
  - `campaign_stages` / `promotion_decisions`
  - `index.py promote`
  - `framework.policies.stage_policy`
  - `index.py analyze --stage-summary / --promotion-log`

- **Continuation / trajectory**（v20.3）
  - `run_branches`
  - `index.py branch`
  - `framework.policies.branch_policy`
  - `index.py analyze --branch-tree / --trajectory-report / --compare-parent-child`

- **Recommendation / selector**（v21）
  - `framework.services.research.selector_service`
  - `framework.policies.selector_policy`
  - `recommendation_batches` / `recommendations` / `recommendation_outcomes`
  - `index.py analyze --recommend-next / --recommendation-log / --recommendation-outcomes`

- **Acquisition / execution closure**（v21.1）
  - `framework.services.research.acquisition_service`
  - `framework.policies.acquisition_policy`
  - `surrogate_snapshots`
  - accepted recommendation 可直接由 `framework/index.py sweep` / `framework/index.py branch` 执行
  - `index.py analyze --acquisition-summary`
  - `scripts/v21_1_replay_benchmark.py`

- **Multi-domain metric / objective layer**（v22）
  - `objective_profiles` / `run_metrics`
  - campaign 可绑定 objective profile
  - `index.py analyze --pareto --metric-source run_metrics --objective-profile ...`
  - generic Pareto 支持 hard constraints、infeasible 分组和 knee point
  - `framework.services.frontier.plotting` 支持 profile-driven axis label / formatter / knee marker

- **Spot FX mock domain**（v22）
  - `domains/spot_trader/`
  - 当前头寸 + liquidity floor + mock quote window
  - direct / USD-bridge route candidate
  - local mock execution 写入 `quote_windows` / `fx_quotes` / `fx_route_legs` / `run_metrics`
  - Airwallex adapter 只定义 quote-only 边界，默认不执行 live conversion

---

## CLI 速查表

### `domains/gomoku/train.py`

| 参数 | 说明 |
|------|------|
| `--time-budget N` | 训练总时长（秒） |
| `--target-win-rate F` | 达到该 WR 提前停止 |
| `--target-games N` | 达到该 game 数提前停止 |
| `--num-blocks N` | ResNet 块数（默认 6） |
| `--num-filters N` | ResNet 卷积通道数（默认 64） |
| `--learning-rate F` | LR（默认 5e-4） |
| `--steps-per-cycle N` | 每 cycle 训练步数（默认 30） |
| `--buffer-size N` | replay buffer 大小 |
| `--parallel-games N` | 同时进行的自对弈盘数 |
| `--mcts-sims N` | 每步 MCTS 模拟次数 |
| `--mcts-batch N` | sims_per_round（C 批量大小） |
| `--c-puct F` | MCTS 探索常数 |
| `--dirichlet-alpha F` | MCTS 根节点 noise alpha |
| `--eval-level N` | 评估对手等级（0-3） |
| `--eval-opponent ALIAS` | 用注册的 NN 对手做评估 |
| `--eval-interval N` | 每 N cycle 跑一次 probe |
| `--probe-games N` | probe eval 的对局数 |
| `--full-eval-games N` | 越过阈值时的 full eval 对局数 |
| `--eval-openings N` | 强制使用的 distinct opening 数 |
| `--auto-stop-stagnation` | WR 平台期自动停止 |
| `--stagnation-window N` | 停滞判断窗口 |
| `--resume UUID` | 从历史 run 的最后 checkpoint 续训 |
| `--initial-opponent ALIAS` | **v15 F2** — 从注册对手的权重起步（区别于 resume：不继承 cycle/optimizer state） |
| `--train-opponent ALIAS` | 训练时混入对该 NN 对手的对弈 |
| `--opponent-mix F` | 对弈混入比例（**1.0 = 纯对弈，0 self-play**，v15 F1 修复） |
| `--auto-promote-to ALIAS` | **v15 E5** — 训练结束时自动按晋升门禁注册新对手 |
| `--register-opponent ALIAS --from-run UUID --from-tag TAG` | 手动注册对手 |
| `--seed N` | 随机种子 |

### `framework/index.py analyze`

| 子命令 | 说明 |
|--------|------|
| `--runs` | 列出所有训练 run |
| `--best` | 每个 run 的最佳 checkpoint |
| `--frontier` | WR 前沿 |
| `--pareto` | Pareto 非支配排序（legacy: 成本 vs WR；v22: 可用 `--metric-source run_metrics --objective-profile ...`） |
| `--compare RUN_A RUN_B` | 两个 run 并排对比 |
| `--compare-by-steps RUN_A RUN_B` | 按训练步数对齐对比（用于 MCTS vs non-MCTS） |
| `--lineage RUN_ID` | 追溯 resume 链 |
| `--stability RUN_ID` | 训练稳定性报告 |
| `--stagnation RUN_ID` | 停滞检测 |
| `--matrix TAG_PREFIX` | sweep 结果矩阵 |
| `--campaign-summary CAMPAIGN` | campaign 汇总 |
| `--stage-summary CAMPAIGN` | stage 汇总 |
| `--promotion-log CAMPAIGN` | promotion 决策日志 |
| `--branch-tree CAMPAIGN` | continuation / branch 树 |
| `--trajectory-report CAMPAIGN` | trajectory 汇总报告 |
| `--compare-parent-child BRANCH_ID` | 某条 branch 的 parent/child 对比 |
| `--opponents` | 列出所有注册对手 |
| `--promotion-chain` | **v15 E7** — S0 → S1 → S2 ... 晋升链 |
| `--opening-breakdown RUN_ID` | **v15 E3** — per-opening WR 分解 |
| `--recommend-next CAMPAIGN` | 生成下一步 recommendation |
| `--recommendation-log CAMPAIGN` | recommendation 历史 |
| `--recommendation-outcomes CAMPAIGN` | recommendation 执行结果 |
| `--acquisition-summary CAMPAIGN` | v21.1 acquisition lineage 汇总 |
| `--report` | 完整实验报告（中文 markdown） |

---

## Domain 接入指南

myautoresearch 框架对一个新 domain 的最小要求：

1. **目录结构**：`domains/<name>/`
2. **入口脚本** `domains/<name>/train.py`：
   - 接受 `--time-budget`、`--seed`、`--sweep-tag` 等标准参数
   - v22 generic domain 建议接受 `--candidate-json`
   - 跑完后写入 `tracker.db`：legacy domain 至少写 `runs.final_win_rate`（domain 真理）、`runs.wall_time_s`、`runs.total_games`、`runs.total_steps`、`runs.num_params`；generic domain 优先写 `run_metrics`
   - 退出码 0 = 成功
3. **policy 文件**：
   - `manifest/search_space.json`
   - `manifest/objective_profile.json`（v22 generic Pareto / constraints）
   - `manifest/stage_policy.json`
   - `manifest/branch_policy.json`
   - `manifest/selector_policy.json`
   - `manifest/acquisition_policy.json`
4. **(可选)** 评估模块 `domains/<name>/prepare.py`：定义对手 / 评估函数
5. **(可选)** 注册自定义列：domain 可以往 `runs` 表写自己的列（框架不会解读，但 analyze 会展示）

接入完成后，**framework 层自动复用**：

- `index.py sweep` 可以 sweep 你的超参矩阵或 dynamic JSON candidate
- `index.py promote` 可以做多阶段晋升
- `index.py branch` 可以做 continuation / trajectory
- `index.py analyze --pareto` 可以画你的成本 vs 真理前沿，或按 objective profile 读取 `run_metrics`
- `index.py analyze --recommend-next` 可以生成 recommendation
- `index.py analyze --acquisition-summary` 可以追踪 acquisition evidence
- tracker.db 自动追踪所有数据

参考实现：`domains/gomoku/`（完整训练 domain，含 game engine + minimax + train loop + replay + web UI）与 `domains/spot_trader/`（轻量 mock domain，展示 v22 generic metrics / objective profile / quote evidence）。

---

## 路线图

| 版本 | 主题 | 关键产出 |
|------|------|---------|
| v11-v15 | Gomoku 训练基座 | MCTS / minimax / async eval / promotion gate / reporting |
| **v20** | Point frontier observation | Pareto / frontier snapshot / sweep auto-pareto |
| **v20.1** | Search-space schema + campaign ledger | `search_spaces` / `campaigns` / `campaign_runs` |
| **v20.2** | Multi-fidelity promotion | `campaign_stages` / `promotion_decisions` / `index.py promote` |
| **v20.3** | Continuation / trajectory | `run_branches` / `index.py branch` / trajectory report |
| **v21** | Selector / recommendation | next-point / next-branch recommendation ledger |
| **v21.1** | Acquisition / execution closure | accepted recommendation execution / surrogate snapshots / replay benchmark |
| **v22** | Multi-domain / spot FX quote-surface | `run_metrics` / `objective_profiles` / dynamic candidate / `spot_trader` mock domain |

每个版本的细节在 `updates/v{N}-update.md`（计划）和 `updates/v{N}-findings.md`（实测）。

---

## 项目纪律（写给未来的 maintainer）

- **真理 > Pareto > 人类偏好**：domain truth 永远先于 Pareto 边界，agent 永远先于人类决策（pareto-frontier §11 三条红线）
- **`output/tracker.db` 是协议**：framework 和 domain 之间不通过别的方式通信
- **C 扩展是性能优化，不是架构边界**：所有 C 化都必须保留 Python fallback
- **测试先行**：`tests/test_*.py` 覆盖每一次涉及 db schema / 训练流程 / minimax 算法的改动
- **Findings 必须等数据**：v{N}-update.md 是计划，v{N}-findings.md 是实测后的复盘——计划里的"预期收益"和实测数字可以不同，但必须如实记录差异

---

## 依赖致谢

- **[MLX](https://github.com/ml-explore/mlx)** — Apple Silicon 上的高性能 ML 框架（model + autograd）
- **[NumPy](https://numpy.org/)** — 数值后端
- **[SQLite](https://www.sqlite.org/)** — `tracker.db` 的存储
- **[FastAPI](https://fastapi.tiangolo.com/)** — `domains/gomoku/web/` 的后端
- **[pygame](https://www.pygame.org/)** — `domains/gomoku/play.py` 的图形界面（可选）
- **[uv](https://github.com/astral-sh/uv)** — Python 包管理与脚本运行

以及前文 [致谢与来源](#致谢与来源) 中的 Code Bullet 和 autoresearch。

---

## License

MIT
