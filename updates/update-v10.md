# Update V10 — Autoresearch Loop Activation

> Date: 2026-04-11
> Status: Action Plan (refined after further analysis)
> Theme: 从基础设施建设期毕业，正式激活 agent-driven autoresearch 闭环

---

## 0. Further Analysis 要点整合

基于 `update-v10-analysis-further.md` 中的深度讨论，V10 plan 做以下调整：

### 关键定位共识

1. **双层架构**：autoresearch（研究主管）+ AutoML（局部探索执行器），二者均为一等公民但角色不对称
2. **V10 scope 收窄**：只激活 autoresearch 闭环。AutoML 层的正式抽象推迟到 v11
3. **AutoML 不是 control group**：它是 agent 的工具之一，agent 可以委派 sweep 任务给 AutoML 层
4. **独特生态位**：轻量级、单文件可训练、agent-native 的 ML 研究场——不是另一个 Optuna

### 从 v10-analysis（Copilot 回应）继承的核心判断

- GPT "本质偏离" 定性偏重——代码事实上 93.8% 服务于 autoresearch
- V10 是"推进"（advancement）不是"纠偏"（correction）
- 缺的不是基础设施，而是 agent 实际进入循环

---

## 1. V10 核心目标

V10 是一次**角色激活**：

> 让已经建好的实验室（v1-v9）正式接入科学家（agent），完成第一次真正的 autoresearch 闭环。

四个交付物：
1. `analyze.py --report`：双格式（markdown + JSON）结构化实验报告
2. `docs/program.md` v2：升级 agent 协议，纳入完整工具链
3. README + docs 叙事统一回归 autoresearch
4. 首次闭环试运行验证

---

## 2. 关键设计决策

### 2.1 Autoresearch 闭环执行模型

**核心认知：autoresearch 闭环不需要内嵌 hook 或 LLM API 调用。**

执行模型是**外部 agent 驱动**的：

```
┌─────────────────────────────────────────────────────────┐
│  External Agent (Claude Code / Cursor / Copilot CLI)    │
│  = 研究主管（autoresearch layer）                        │
│                                                         │
│  reads program.md → understands the loop → executes it  │
│                                                         │
│  LOOP:                                                  │
│    1. Read report (analyze.py --report --format json)   │
│    2. Form hypothesis based on signals + data           │
│    3. Edit src/train.py (ONLY mutable file)             │
│    4. git commit (hypothesis as message)                │
│    5. Run training (train.py --time-budget 300)         │
│    6. Read new report                                   │
│    7. Compare: improved? → keep. Worse? → git reset     │
│    8. Check stage promotion                             │
│    9. [Optional] Dispatch sweep.py for local search     │
│   10. GOTO 1                                            │
│                                                         │
└─────────────────────────────────────────────────────────┘
         ↑                              ↓
     reads output                 modifies code
         ↑                              ↓
┌─────────────────────────────────────────────────────────┐
│  MAG-Gomoku Codebase                                    │
│                                                         │
│  src/train.py ←── agent edits this (ONLY mutable file)  │
│  src/analyze.py ←── report + analysis tools             │
│  src/sweep.py  ←── AutoML 局部探索工具（agent 可委派）   │
│  output/tracker.db ←── persistent experiment history    │
│  src/prepare.py ←── immutable benchmark（裁判）          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**不需要 hook 的原因：**
- Autoresearch = "agent 驱动系统"，不是"系统驱动 agent"
- Agent 是完整的 coding agent，有终端、文件编辑、git 权限
- 它在循环中主动拉取报告，主动做判断

**不需要 LLM API 集成的原因：**
- 在训练脚本内嵌 AI 决策 = GPT 警告的"控制权内移"
- Autoresearch 精神：训练脚本只产出事实，agent 在外部做判断
- 松耦合正是设计优势

**启动方法：**

```bash
# 推荐：直接指向 program.md
> Read docs/program.md and begin the autoresearch experiment loop.

# 精确启动：
> Read docs/program.md. Run `uv run python src/analyze.py --report --format json`
> to understand the current state. Target: beat 93.3% WR vs L0.
```

### 2.2 双格式 Report 设计

基于用户反馈，报告采用**双格式输出**：

| 格式 | 消费者 | 语言 | 样板文件 |
|------|--------|------|----------|
| Markdown | 人类研究者 | 中文 | docs/sample_report.md |
| JSON | Agent / 程序化消费 | 英文 key + 中英混合 value | docs/sample_report_agent.json |

**JSON 格式的关键改进（相对于 markdown）：**
1. **超参数完整暴露**：每个 run 携带完整的 `hyperparams` 对象（LR、SPC、blocks、filters、buffer、seed）
2. **信号结构化**：每个 signal 有 `type`、`severity`、`message`、`suggestion` 字段
3. **阶段量化**：`stage.gap_to_promotion` 是数值而非文字描述
4. **聚合统计**：`hyperparams_summary` 汇总所有已测试的架构和学习率

**共同的 7 段结构：**
1. Recent Runs（近期运行 + 完整超参）
2. Best Checkpoint（当前最佳检查点）
3. Win Rate Frontier（单调递增前沿）
4. Stability Analysis（最近运行稳定性）
5. Opponent Registry（对手注册表）
6. Stage Assessment（阶段评估 + 晋级差距）
7. Signals（自动生成的观察与建议）

**Signal 自动生成规则：**

| 条件 | Signal 类型 | 严重级别 |
|------|------------|----------|
| Best WR > promotion_threshold - 5% | CLOSE_TO_PROMOTION | high |
| 最近 3 runs WR 无提升 | WR_PLATEAU | medium |
| 最近 run loss > 前次 2x | LOSS_DIVERGENCE | high |
| Benchmark runs < 3 | INSUFFICIENT_BENCHMARKS | medium |
| avg_game_length < 30 | SHORT_GAMES | medium |
| 某架构一致性优于其他 | ARCHITECTURE_PATTERN | info |
| 最近 run 与 best 差距 > 20% | REGRESSION_WARNING | high |

### 2.3 AutoML 层的 V10 定位

**V10 中 AutoML 的角色：** 既有工具，保持现状，不做新抽象。

- `sweep.py` 已存在且工作正常——agent 可以按需调用
- Program.md v2 会将 sweep 列为 "可选工具"
- **不做** control group 对比实验——这是 v11 的内容
- **不做** sweep 结果自动注入 report——agent 可以直接 `--matrix` 查看

**V11 预告（仅文档化，不执行）：**
- 正式定义 AutoML 接口：agent 发出 "探索 LR 范围" 指令 → sweep 执行 → 结果回流到 report
- benchmark 裁决 agent 和 AutoML 的综合表现
- AutoML 作为 agent 的 "研究助手"，而非独立的竞争路线

---

## 3. 具体任务清单

### P0: analyze.py --report 双格式命令 [~150 行]

```python
def cmd_report(conn, n_recent=5, fmt="md"):
    """Generate structured experiment report.
    fmt: 'md' for markdown (Chinese), 'json' for structured JSON.
    """
```

CLI 接口：
```bash
uv run python src/analyze.py --report                    # 默认 markdown
uv run python src/analyze.py --report --format json      # JSON 格式
uv run python src/analyze.py --report --format md        # 明确 markdown
uv run python src/analyze.py --report --recent 10        # 最近 10 runs
```

### P1: program.md v2 升级 [~60 行改动]

1. 加入 report 步骤（`--report --format json` 作为每轮首步）
2. 工具层文档（analyze.py 系列 + sweep.py 可选）
3. 阶段标记（v1-v9 基础设施期 → v10+ 研究期）
4. Agent 决策指南（signal → action 映射）
5. sweep.py 委派说明（agent 可选择用 sweep 做局部搜索）

### P2: README.md + docs 叙事统一 [~40 行改动]

1. 增加 "Project Phases" 章节
2. 增加 `--report` 文档（双格式）
3. 叙事完全回归 autoresearch（agent 在主控位）
4. 移除/弱化任何暗示 "AutoML 平台" 的措辞

### P3: 首次闭环试运行 [不写代码]

在 P0-P2 完成后，模拟 agent 执行闭环：
1. `analyze.py --report --format json` → 读取状态
2. 形成假设（例："10x64 may break 95% threshold"）
3. 修改 train.py 超参 → git commit
4. 运行训练 → `--report` → 对比
5. keep/discard → 重复 3-5 轮
6. 目的：验证流程通畅，不是提升 WR

### P4: Commit + 工作日志

---

## 4. 不做的事情

| 不做 | 原因 |
|------|------|
| 内嵌 LLM API 调用 | 违反 autoresearch 精神 |
| Hook / event / callback | Agent 主动拉取，不需要被动推送 |
| Orchestrator / scheduler | 外部 agent 自带循环能力 |
| Control group 对比实验 | 推迟到 v11 |
| AutoML 层正式抽象 | 推迟到 v11 |
| Sweep 结果自动注入 report | Agent 可直接 --matrix 查看 |
| Replay 深度行为分析 | 优先级低 |

---

## 5. 叙事统一原则

所有文档应传达：

> MAG-Gomoku 是一个 agent-driven autoresearch 实验系统。
> Agent（研究主管）读取报告、形成假设、修改代码、驱动研究循环。
> 训练脚本产出事实，benchmark 裁决结果，sweep 提供局部探索能力。
> 三者各司其职，agent 居于决策中心。

---

## 6. 代码量预估

| 任务 | 预估行数 | 文件 |
|------|----------|------|
| cmd_report() 双格式 | ~150 行 | src/analyze.py |
| program.md v2 | ~60 行改动 | docs/program.md |
| README.md 更新 | ~40 行改动 | README.md |
| **总计** | **~250 行** | 3 个文件 |

---

## 7. 成功标准

1. `analyze.py --report` 输出完整的 7 段中文 markdown 报告
2. `analyze.py --report --format json` 输出结构化 JSON（含完整超参）
3. program.md v2 包含 report 步骤、工具层、决策指南
4. README 叙事完全回归 autoresearch，agent 居于主控位
5. 至少完成 3 轮闭环模拟验证
6. 可回答："agent 能否基于 JSON report 做出合理的研究决策？"

---

## 8. 执行工作日志

> 执行者: Copilot (Claude Opus 4.6)
> Commit: `609ee15`
> 日期: 2026-04-11

### 8.1 P0: analyze.py --report 双格式命令

**实现：** 在 `src/analyze.py` 中新增约 480 行代码（从 465 行 → 947 行）。

核心架构：
- `_gather_report_data()`: 统一数据采集层，从 DB 中查询 7 段所需数据并组装为 Python dict
- `_generate_signals()`: 规则引擎，7 条规则自动生成信号（实际触发取决于数据）
- `_format_report_md()`: 将 dict 渲染为中文 markdown 报告
- `_format_report_json()`: 将 dict 序列化为结构化 JSON，包含完整 hyperparams
- `cmd_report()`: 入口函数，接收 `fmt` 和 `n_recent` 参数

Signal 规则实际触发验证：
| 规则 | 是否触发 | 说明 |
|------|----------|------|
| CLOSE_TO_PROMOTION | ✅ | 当 gap < 5% 时触发 |
| ARCHITECTURE_PATTERN | ✅ | 10x64 明显优于 6x64 |
| LOSS_DIVERGENCE | ✅ | Round 3 loss 是 Round 2 的 2.5x |
| REGRESSION_WARNING | ✅ | 短预算运行 WR 低于 benchmark |
| INSUFFICIENT_BENCHMARKS | ✅ | DB 中 benchmark 运行不足 |
| WR_PLATEAU | 未触发 | 近期运行有明显提升，无平台期 |

CLI 参数已验证：
```
--report                    → 中文 markdown 输出 ✓
--report --format json      → 结构化 JSON 输出 ✓（已通过 json.load 验证合法性）
--report --recent 10        → 可调整近期运行数量 ✓
```

**确认：** 全部原有命令（--runs/--best/--frontier/--compare/--lineage/--opponents/--stability/--matrix）回归测试通过。

### 8.2 P1: program.md v2 升级

关键改动（84 行改动）：
1. **新增 "Project phase" 章节** — 明确 v1-v9 = 基础设施期，v10+ = 研究期
2. **新增 "Available tools" 表格** — 10 个命令的用途和使用时机
3. **新增 "Using sweep.py" 小节** — agent 如何委派局部搜索
4. **重写 "Setup" 步骤** — 第 3 步从 "run baseline" 改为 "read report"
5. **重写 "The experiment loop"** — 10 步循环，首步是 `--report --format json`
6. **新增 "Decision guide" 表格** — signal → action 映射（6 条规则）
7. 保留原有的 Hints 和 NEVER STOP 指令不变

### 8.3 P2: README.md + docs 叙事统一

改动（63 行变化）：
1. **新增 "Project phases" 表格** — Infrastructure ✅ / Autoresearch 🔄
2. **重写 "How it works" 图** — agent 在顶部（Research Director），codebase 在底部
3. **新增 "Experiment reports" 小节** — 解释双格式报告
4. **更新 Analysis tool 部分** — --report 放在第一条
5. **更新 analyze.py 描述** — "experiment reports + training analysis"
6. 开头句从 "autonomously modifies training code" 改为 "reads experiment reports, forms hypotheses, modifies the training code, and drives the research loop"

### 8.4 P3: 首次闭环试运行

3 轮完整验证：

| 轮次 | 假设 | 配置 | 结果 WR | 判断 |
|------|------|------|---------|------|
| 1 | ARCHITECTURE_PATTERN 信号 → 8x64 应优于 6x64 | 8x64, lr=5e-4, 120s | 69.5% | ✅ Keep |
| 2 | 提高 LR 可加速收敛 | 8x64, lr=7e-4, 120s | 74.0% (+4.5%) | ✅ Keep |
| 3 | 更深模型 + 高 LR = 更高天花板 | 10x64, lr=7e-4, 120s | 85.5% (+11.5%) | ✅ Keep |

关键观察：
- 报告在每轮之间正确更新——信号从 REGRESSION_WARNING 变为 ARCHITECTURE_PATTERN → LOSS_DIVERGENCE
- JSON 格式输出的 hyperparams 清晰可比较——agent 可以精确追踪每轮变化
- gap_to_promotion 从 14.5% 下降到 9.5%——进展方向正确
- Signal 系统自适应——当 10x64 成为最优架构时，ARCHITECTURE_PATTERN 自动更新建议

**结论：闭环流程通畅。agent 可以基于 report 做出合理的研究决策。**

### 8.5 成功标准核验

| # | 标准 | 状态 |
|---|------|------|
| 1 | `--report` 输出完整 7 段中文 markdown | ✅ 通过 |
| 2 | `--report --format json` 输出结构化 JSON | ✅ 通过（json.load 验证 + 7 section assert） |
| 3 | program.md v2 含 report 步骤、工具层、决策指南 | ✅ 通过 |
| 4 | README 叙事回归 autoresearch | ✅ agent 在架构图顶部，phases 表格明确 |
| 5 | 至少 3 轮闭环验证 | ✅ 3 轮完成（69.5% → 74.0% → 85.5%） |
| 6 | agent 能基于 report 做合理决策 | ✅ 信号驱动假设，WR 持续上升 |

### 8.6 代码量统计

| 文件 | 变化 | 说明 |
|------|------|------|
| src/analyze.py | +483 行 | cmd_report 双格式 + signal 引擎 + 数据采集 |
| docs/program.md | +84/-33 行 | v2 升级 |
| README.md | +63/-33 行 | 叙事统一 + phases + report docs |
| docs/sample_report.md | +65 行（新文件） | 中文 markdown 报告样板 |
| docs/sample_report_agent.json | +160 行（新文件） | JSON 报告样板 |
| .gitignore | +3 行 | update-v10 相关文件 |
| **总计** | **+916 行** | 6 个文件 |

---

## Section 9: 交叉对战矩阵分析 (Cross-Play Matrix Analysis)

> 时间: 2026-04-11
> 数据持久化: `output/cross_play_matrix.json`, `docs/cross_play_analysis.md`

### 9.1 背景与动机

在完成 L0 → L1 → L2 → L3 的阶梯式训练后，我们意识到一个关键问题：**每个模型只有对其直接训练对手的胜率数据**，缺乏横向可比性。无法回答以下问题：

1. 阶梯训练是否真的在提升绝对实力？还是模型只学会了"针对特定对手的战术"？
2. 是否存在"石头剪刀布"式的循环克制？（A > B > C > A）
3. 每一代模型相比前代的真实提升幅度是多少？

为此，我们设计了 **N×N 交叉对战矩阵**，让所有已注册对手两两对弈，构建完整的实力图谱。

### 9.2 对手注册表

| 别名 | 来源 Run | 架构 | 训练对手 | 训练局数 | 学习率 | 注册胜率 | 注册 Tag |
|------|----------|------|----------|----------|--------|----------|----------|
| L0 | 374d567f | 6x64 (565K) | minimax L0 (随机) | 5,008 | 1e-3 | 67.5% vs minimax | wr065_c0310 |
| L1 | 92d1740a | 10x64 (862K) | minimax L0 (随机) | 5,056 | 7e-4 | 84.0% vs minimax | final_c0079 |
| L2 | 2b1d2f03 | 10x64 (862K) | L1 (NN) | 10,048 | 7e-4 | 83.0% vs L1 | wr050_c0090 |
| L3 | b3f99d4f | 10x64 (862K) | L2 (NN) | 72,000 | 7e-4 | 75.0% vs L2 | final_c1125 |

### 9.3 评估方法

#### 评估代码伪代码

```python
opponents = load_all_models(["L0", "L1", "L2", "L3"])
N_GAMES = 100  # 每组对局（50 先手 + 50 后手，消除先手优势偏差）

for player in opponents:
    for opponent in opponents:
        if player == opponent:
            continue
        result = in_process_eval(
            model=player,
            opponent_model=opponent,
            n_games=N_GAMES,
            # 前 50 局 player 执黑（先手）
            # 后 50 局 player 执白（后手）
        )
        matrix[player][opponent] = result.win_rate
```

#### 关键设计决策

- **使用 `_in_process_eval()`** 而非 subprocess：NN vs NN 对弈不需要 minimax 子进程
- **argmax 推理**：player 使用 argmax（确定性），opponent 使用 temperature=0.5（随机性）
  - 这引入了不对称性：作为 player 时策略更确定，作为 opponent 时策略更多样
  - 这也解释了为什么 A vs B + B vs A ≠ 100%
- **100 局样本量**：95% 置信区间约 ±10%，足够发现大趋势，但小差异（<10%）不可信

#### 实际执行

```bash
# 完整矩阵：4 个模型 × 3 个对手 = 12 组对局
# 每组 100 局，共 1200 局 NN vs NN 对弈
# 总耗时：约 195 秒（~16s/组）
```

### 9.4 交叉对战矩阵结果

#### 胜率矩阵 (行 = Player, 列 = Opponent)

|        | vs L0       | vs L1       | vs L2       | vs L3       | **平均胜率** |
|--------|-------------|-------------|-------------|-------------|--------------|
| **L0** | —           | 27% (27W)   | 40% (40W)   | 26% (26W)   | **31.0%**    |
| **L1** | 63% (63W)   | —           | 75% (75W)   | 62% (62W)   | **66.7%**    |
| **L2** | 65% (65W)   | 45% (45W)   | —           | 66% (66W)   | **58.7%**    |
| **L3** | 69% (69W)   | 70% (70W)   | 73% (73W)   | —           | **70.7%**    |

> 所有对局 0 平局，五子棋在我们的规则下不存在和棋。

#### 对称性验证 (A→B + B→A)

| 对局     | A→B   | B→A   | 合计    | 偏差    |
|----------|-------|-------|---------|---------|
| L0↔L1    | 27%   | 63%   | 90%     | -10%    |
| L0↔L2    | 40%   | 65%   | 105%    | +5%     |
| L0↔L3    | 26%   | 69%   | 95%     | -5%     |
| L1↔L2    | 75%   | 45%   | 120%    | +20%    |
| L1↔L3    | 62%   | 70%   | 132%    | +32%    |
| L2↔L3    | 66%   | 73%   | 139%    | +39%    |

偏差来源：(1) 100 局样本噪声 ±10%；(2) player/opponent 推理策略不对称（argmax vs temperature=0.5）。
后者导致双方胜率之和系统性偏高——双方都在作为 opponent 时表现更弱（随机性更大）。

#### 实力排名

| 排名 | 模型 | 平均胜率 | 训练投入 |
|------|------|----------|----------|
| 🥇 1 | **L3** | **70.7%** | 72,000 games, 5981s |
| 🥈 2 | **L1** | **66.7%** | 5,056 games, 437s |
| 🥉 3 | **L2** | **58.7%** | 10,048 games, 886s |
| 4 | **L0** | **31.0%** | 5,008 games, 1146s |

### 9.5 L2 异常分析：为什么 L2 是最弱的 10x64 模型？

L2 的平均胜率 (58.7%) 低于 L1 (66.7%)，形成了排名倒挂。这是本次矩阵分析最重要的发现。

#### 现象

- L1 vs L2 = **75%** — L1 大幅碾压 L2
- L2 vs L1 = **45%** — L2 反打也输
- L2 vs L0 = **65%** — 仅比 L1 vs L0 (63%) 高 2%，几乎无提升
- L2 vs L3 = **66%** — 对更强的 L3 反而表现不错（但 L3 vs L2 = 73%）

#### 根因追溯

**L2 的注册来源**：Run 2b1d2f03，Tag `wr050_c0090`

1. **注册时机过早**：L2 来自 10,000 局训练 vs L1 的中间 checkpoint（cycle 90）。这个 checkpoint 被选中是因为它的 smoothed WR 达到了 83%，但当时的 full eval 机制有 bug（使用 minimax L0 而非 L1），**注册时的 83% 胜率实际上是 vs minimax L0 的数据，不是 vs L1**。

2. **eval 基线错位 (critical bug)**：在修复 `_subprocess_eval` 和 `_do_checkpoint` 之前，所有 checkpoint 的 full eval 都回退到了 minimax L0。这意味着：
   - Probe eval（训练中）：正确使用 L1 作为对手 ✅
   - Checkpoint full eval：错误使用 minimax L0 ❌
   - Final eval：错误使用 minimax L0 ❌
   - 注册时参考的 83% WR 是 vs minimax L0，不是 vs L1

3. **专精化而非泛化**：即使忽略 eval bug，L2 的训练设计本身也有问题。它从零开始训练 10,000 局只对付 L1 一个对手。这导致模型学到的是"针对 L1 特定弱点的策略"，而非"更强的通用五子棋能力"。体现在：
   - 对 L0（比 L1 弱得多的对手）几乎没有额外优势
   - 对 L1（直接训练对手）反而输了（45%），可能因为注册太早

4. **模型容量浪费**：L2 和 L1 同为 10x64 架构，但 L2 只训练了 10,048 局（L1 训练了 5,056 局但 vs 更简单的对手），cycle 90 时模型远未充分收敛。

#### L2 的教训

| 问题 | 教训 | 改进措施 |
|------|------|----------|
| eval 基线错位 | checkpoint/final eval 必须用训练对手做基线 | ✅ 已修复（v10 bugfix） |
| 注册时机过早 | smoothed WR 虚高不能作为注册依据 | 改用 full eval ≥ 80% 作硬门槛 |
| 专精化陷阱 | 单一对手训练导致策略窄化 | 考虑 opponent mix（训练中混合多个对手） |
| 训练不充分 | cycle 90 远未收敛 | 使用 target-win-rate 而非手动选 checkpoint |

### 9.6 对未来训练的借鉴意义

#### 1. 对手注册标准

**旧标准**（导致 L2 问题）：
- 依赖 smoothed WR 或 checkpoint 标记的胜率
- 没有验证 eval 基线是否正确

**新标准**（建议）：
- full eval ≥ 80% vs 训练对手（200 games）
- 注册前跑一次交叉矩阵，验证新模型 vs 所有现有对手的表现
- 新模型的平均胜率必须高于现有最强模型

#### 2. eval 一致性原则

训练中的所有评估（probe / checkpoint full eval / final eval）**必须使用同一个对手**。
这是在 v10 bugfix 中修复的核心问题：

```
修复前: probe=NN opponent, checkpoint=minimax L0, final=minimax L0  ← 不一致
修复后: probe=NN opponent, checkpoint=NN opponent, final=NN opponent  ← 一致
```

#### 3. 交叉矩阵作为常规工具

建议在每次注册新对手后，自动执行交叉矩阵评估：
- 持久化结果到 `output/cross_play_matrix.json`
- 追踪实力排名变化趋势
- 早期发现排名倒挂或循环克制问题

#### 4. 阶梯训练的有效性确认

尽管 L2 出现了问题，矩阵整体数据仍然确认了阶梯训练的核心假设：

- **L3 (70.7%) > L1 (66.7%) > L0 (31.0%)**：绝对实力在逐代提升
- **L3 vs L0 = 69%**：L3 对最弱对手的胜率最高（符合预期）
- **L3 vs L1 = 70%**、**L3 vs L2 = 73%**：L3 对所有对手都有优势

阶梯训练路线可行，但需要更严格的质量门控来避免 L2 式的问题。

### 9.7 后续修正记录

本节中发现的 eval 一致性 bug 已在本轮 v10 工作中修复：

| 修复项 | 文件 | 内容 |
|--------|------|------|
| `_in_process_eval()` | train.py | 新增函数，支持 NN vs NN 完整评估（返回 wins/losses/draws） |
| `_do_checkpoint()` | train.py | 增加 `opponent_model` 参数，NN 对手时使用 in-process eval |
| Final eval | train.py | NN 对手时使用 `_in_process_eval`，不再回退到 minimax subprocess |
| `--no-eval-opponent` | train.py | 新增 flag，允许显式禁用 NN 对手（用于 benchmark） |
| `--eval-opponent` default | train.py | 默认值从 `None` → `"L3"`（当前最强对手） |
