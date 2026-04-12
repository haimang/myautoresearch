# Update v6 Analysis - MAG Gomoku vs autoresearch

> Date: 2026-04-11
> Scope: compare current MAG Gomoku against upstream karpathy/autoresearch, review git history, and assess whether the project evolution is an upgrade, a regression, or a tradeoff.

## 1. Executive Summary

This repository still preserves the core autoresearch idea, but no longer preserves the original autoresearch shape.

The core idea that remains intact is:

- a fixed evaluation harness outside the main training loop
- one principal metric that determines whether progress happened
- repeated short experiments
- an agent or human iterating mostly on the training logic
- a keep/discard mentality around experiments

However, MAG Gomoku has diverged substantially from the upstream repository in three major ways:

1. The task changed completely.
   Upstream autoresearch is a small LLM pretraining benchmark. MAG Gomoku is a game AI self-play training system.

2. The system changed from minimal to operational.
   Upstream is intentionally a tiny research harness. MAG Gomoku is now an experiment platform with run tracking, checkpoint lineage, replay artifacts, human-vs-AI interaction, and resume support.

3. The control model changed.
   Upstream strongly optimizes for a single editable file and a single strict experiment loop. MAG Gomoku now distributes responsibility across training, tracking, replay, documentation, and runtime UX.

My overall judgment:

- relative to upstream autoresearch philosophy: medium divergence
- relative to upstream implementation: high divergence
- relative to the first MAG Gomoku commit: moderate divergence, mostly via engineering expansion

This is not a simple story of "upgrade" or "regression".

The fair reading is:

- MAG Gomoku is weaker than upstream autoresearch on strict minimalism, experiment comparability, and single-file edit discipline.
- MAG Gomoku is stronger than upstream autoresearch on observability, recoverability, local usability, artifact quality, and long-horizon practical value.

## 2. Evidence Base

This analysis is based on:

- upstream karpathy/autoresearch public README and repository structure
- MAG Gomoku git history from the first commit onward
- the first MAG Gomoku commit layout
- the current repository layout and source files

Important historical finding:

- The first MAG Gomoku commit already adopted the autoresearch pattern conceptually, but it was not a line-by-line continuation of upstream autoresearch.
- The first commit already translated the idea into a Gomoku-specific world: game.py, prepare.py, train.py, program.md, play.py, replay.py, results.tsv.
- Therefore, the biggest divergence from upstream happened at the moment of domain translation from LLM training to Gomoku training.
- Later changes mostly expanded the Gomoku version into a more complete system.

## 3. Upstream autoresearch: what it actually optimizes for

Upstream autoresearch is not just a codebase. It is a deliberately constrained research operating model.

Its main design goals are:

1. Extreme simplicity.
   The repo is intentionally tiny. Three files matter most: prepare.py, train.py, program.md.

2. Single editable surface.
   The agent edits train.py. This keeps diffs reviewable and the search space small.

3. Fixed-time comparability.
   Every run uses a fixed 5-minute wall-clock budget so the agent learns what is optimal under that budget.

4. Single metric.
   The metric is val_bpb. Lower is better. The metric is stable and tied to a fixed validation protocol.

5. Linear experiment frontier.
   The loop is conceptually: change train.py, run experiment, inspect result, keep or discard, continue from the kept frontier.

6. Minimal infrastructure.
   results.tsv is intentionally simple. The repo resists turning into a heavyweight platform.

This means upstream autoresearch is best understood as a minimal research harness, not a product and not an experiment management system.

## 4. The first MAG Gomoku version

The first MAG Gomoku commit preserved the upstream pattern more closely than the current code does.

It had:

- a root-level train.py as the main mutable training script
- a root-level prepare.py as the read-only evaluation harness
- a root-level program.md that explicitly mirrors upstream agent instructions
- results.tsv for experiment logging
- a fixed 5-minute budget inherited from prepare.py
- a single dominant metric: win_rate against fixed opponents

But even that first version had already diverged from upstream in important ways:

- task changed from language modeling to self-play board game training
- evaluation changed from validation-set metric to opponent-based match metric
- game.py existed as an additional foundational module, unlike upstream
- play.py and replay.py existed from day one, indicating a user-facing and media-facing ambition beyond pure training
- prepare.py already contained checkpoint archive and recording logic, which upstream does not emphasize

So the first version was already a fork in spirit, not merely a platform port.

## 5. Divergence Assessment

### 5.1 By methodology

Methodologically, the project is still recognizably autoresearch-inspired.

Why:

- there is still a stable non-train module used as evaluation ground truth
- there is still a notion of one main metric controlling advancement
- there is still a short-run iterative loop
- train.py is still the main research surface

Assessment:

- divergence from upstream method: medium

### 5.2 By training problem

The project has completely changed the learning problem.

Upstream:

- supervised next-token prediction on a fixed dataset
- evaluation by validation bits per byte
- tokenizer, dataloader, and validation split matter most

Current MAG Gomoku:

- reinforcement-like self-play data generation
- policy/value learning on generated trajectories
- evaluation by head-to-head performance against minimax or registered NN opponents
- board engine, move legality, opponent quality, and self-play stability matter most

Assessment:

- divergence from upstream problem definition: very high

### 5.3 By system architecture

Upstream system shape:

- prepare.py
- train.py
- program.md
- pyproject.toml
- optionally results.tsv and notebook analysis

Current MAG Gomoku shape:

- src/train.py
- src/prepare.py
- src/game.py
- src/tracker.py
- src/play.py
- src/replay.py
- docs/program.md
- docs/caveats.md
- output/tracker.db
- per-run UUID directories
- checkpoint and recording trees

Assessment:

- divergence from upstream architecture: high

### 5.4 By experiment discipline

Upstream is strict:

- fixed 5-minute run budget
- one editable file
- tsv log
- keep or hard-reset mentality

Current MAG Gomoku is looser:

- time budget can be optional
- resume exists
- multiple runtime modes exist
- custom NN opponents can alter the benchmark frame
- train.py includes more operational logic than pure training

Assessment:

- divergence from upstream experiment discipline: medium to high

## 6. Comparison Table

| Dimension | Upstream autoresearch | Current MAG Gomoku | Assessment | Net effect |
|---|---|---|---|---|
| Core intent | Minimal autonomous research loop | Autonomous research plus operational training platform | Diverged in scope | Broader value, less purity |
| Main task | Small LLM pretraining | Gomoku self-play policy/value learning | Completely different task | Not comparable at task level |
| Principal metric | val_bpb, lower is better | win_rate, higher is better | Same role, different semantics | Concept preserved, measurement changed |
| Evaluation harness | Fixed validation evaluation | Fixed opponent match evaluation, later plus NN opponents | Partially preserved | Good fit for games, weaker benchmark purity once custom opponents enter |
| Editable surface | train.py only | train.py still primary, but project reality spans tracker, replay, docs, play | Expanded control surface | Easier operations, harder agent focus |
| Run budget | Strict fixed 5 minutes | Originally strict, now optional with fallback | Relaxed | More usability, less comparability |
| Experiment log | results.tsv | SQLite tracker plus checkpoint and recording metadata | Major infrastructure upgrade | Much better observability |
| Artifact management | Minimal | UUID run isolation, checkpoints, recordings, resume chain | Major expansion | Strong practical value |
| User-facing usage | Essentially none beyond training | Human play, AI vs AI, replay, frame export | New capability | High extra value beyond research |
| Platform target | Single NVIDIA GPU | Apple Silicon MLX | Platform fork | Great accessibility for local use |
| Repo philosophy | Resist complexity | Accept complexity when it buys visibility and persistence | Philosophical divergence | Better operations, more maintenance burden |
| Benchmark stability | High | Moderate, lower if eval-opponent is widely used | Some regression | Must be actively managed |

## 7. What is clearly better than upstream

These are areas where MAG Gomoku provides real value beyond upstream autoresearch.

### 7.1 Experiment tracking is materially better

The move from results.tsv to tracker.db is a real upgrade.

Why it matters:

- runs are queryable
- checkpoints are first-class records
- recordings are linked to checkpoints
- resume chains are explicit
- future automation can consume structured facts instead of parsing logs

This is a genuine platform-level improvement, not just extra complexity.

### 7.2 Artifact persistence is much stronger

Upstream is intentionally disposable. MAG Gomoku preserves outputs in a way that supports continued work.

Capabilities added:

- per-run output isolation
- milestone checkpoints
- full game recordings bound to checkpoints
- model lineage via resumed_from
- named NN opponents

This turns experiments into reusable assets.

### 7.3 Human interpretability is much stronger

In upstream, progress is mostly numerical. In MAG Gomoku, the model can be watched, replayed, and directly challenged.

This gives value in:

- debugging behavior
- spotting tactical blind spots
- validating whether numeric gains correspond to better play
- producing qualitative evidence for model improvement

### 7.4 Apple Silicon practicality is a real contribution

Upstream targets a powerful NVIDIA setup. MAG Gomoku demonstrates that the autoresearch style can be usefully translated to local Apple Silicon hardware.

This is not merely a port. It changes who can run the system at all.

### 7.5 Resume support is high-value in practice

Strictly speaking, resume departs from the original clean experiment model. Practically, it saves real time and protects real work.

On a local workstation, that matters.

## 8. What is clearly weaker than upstream

These are the places where the current project has lost properties that made upstream autoresearch strong.

### 8.1 The single-file discipline has weakened

Upstream's strongest control mechanism is not technical. It is architectural discipline.

MAG Gomoku still says train.py is the main editable surface, but the real operating system of the repo now spans:

- src/train.py
- src/tracker.py
- src/play.py
- src/replay.py
- docs/program.md
- README.md

That weakens the original clarity.

### 8.2 Experiment comparability has weakened

The original fixed-budget assumption is no longer absolute.

Contributors to weaker comparability:

- optional time budget
- smoothed win rate instead of only raw probe win rate
- resume from prior checkpoints
- custom NN opponents
- evolving operational modes

Each feature is defensible individually, but together they reduce clean apples-to-apples comparison.

### 8.3 train.py is now carrying too many responsibilities

Current train.py contains:

- model definition
- self-play generation
- training loop
- CLI orchestration
- checkpoint save logic
- subprocess eval logic
- opponent loading
- text TUI
- run finalization and promotion hints

This is workable, but it is no longer the clean research surface that autoresearch originally wants.

### 8.4 Documentation drift has appeared

docs/program.md still reflects an earlier autoresearch-style operating model, but the actual system behavior now depends heavily on tracker.db, UUID output directories, replay bindings, and newer CLI features.

That means the repository's instruction layer and its code reality are no longer fully aligned.

This is one of the clearest problems found in the review.

## 9. Balanced Evaluation of the Major Changes

### 9.1 SQLite tracker

Verdict:

- upgrade for platform value
- mild regression for minimalism

Why:

- far more useful for long-term experimentation
- clearly more complex than a tsv log

### 9.2 UUID output isolation

Verdict:

- strong upgrade

Why:

- avoids artifact collisions
- makes cleanup and provenance straightforward
- creates a stable place for downstream tools and replay assets

### 9.3 Resume support

Verdict:

- operational upgrade
- methodological compromise

Why:

- saves wasted compute and interrupted progress
- weakens the purity of fixed-run comparability

### 9.4 Replay and play tooling

Verdict:

- strong upgrade if the project aims beyond pure metric chasing
- unnecessary complexity if one only wants the strict autoresearch loop

Why:

- these tools create interpretability and product value not present upstream
- they do not directly improve the training objective by themselves

### 9.5 Smoothed win rate and threshold tagging

Verdict:

- pragmatic upgrade

Why:

- reduces noisy checkpoint decisions
- makes milestone naming more semantically meaningful

Risk:

- can hide short-term instability if treated as the only truth

### 9.6 Registered NN opponents

Verdict:

- strategically interesting extension
- benchmark purity risk

Why:

- creates a smoother difficulty ladder than fixed minimax levels alone
- but reduces the stability of a single benchmark frame if used carelessly

## 10. Obvious Problems Found

These are the clearest issues surfaced by the analysis.

### Problem 1: docs/program.md no longer fully describes the real system

This is the most obvious structural issue.

Current reality includes:

- tracker.db as the actual source of experiment truth
- UUID output directories
- resume chains
- opponent registration
- replay-bound checkpoints

But docs/program.md still carries earlier assumptions such as a stricter single-loop process and an older logging mentality.

Impact:

- agent instructions and repo reality are misaligned
- future autonomous runs may optimize for the wrong workflow
- new contributors will misunderstand what is actually authoritative

### Problem 2: the project now has two identities but only one control document

The repository is both:

- an autoresearch-style experiment harness
- a user-facing Gomoku AI platform

Those are related, but not identical.

The repo currently blurs them.

Impact:

- feature additions can silently erode experiment discipline
- research decisions and UX decisions are mixed together

### Problem 3: benchmark purity is at risk

The more the project relies on custom opponents, resume chains, flexible stop conditions, and smoothed metrics, the less obvious it becomes whether two runs are directly comparable.

Impact:

- headline progress can become harder to interpret
- "best model" may depend on which benchmark frame is being used

### Problem 4: train.py is close to becoming a god file

This is ironic because upstream also uses one central file, but the meanings differ.

Upstream train.py is large because it contains the research logic.
Current train.py is large because it contains research logic plus orchestration plus platform logic.

Impact:

- harder agent edits
- larger blast radius for bugs
- more coupling between model changes and operational concerns

### Problem 5: the success criteria are no longer singular

The project now has at least four partially competing goals:

- better win_rate
- better artifact management
- better usability
- better explainability and replay value

That is not wrong. It just means the project is no longer a pure single-objective autoresearch fork.

Impact:

- future design decisions need explicit prioritization
- otherwise "upgrade" and "regression" arguments will keep talking past each other

## 11. Did the project create value beyond upstream?

Yes.

The project created real value beyond upstream in at least five ways.

### 11.1 Persistence value

Experiments are no longer disposable events. They are persistent records with lineage.

### 11.2 Interpretability value

You can inspect what the model actually does, not only what a scalar metric says.

### 11.3 Recovery value

Interrupted work can be resumed and traced.

### 11.4 Accessibility value

The autoresearch idea becomes usable on a local Apple Silicon machine.

### 11.5 Productization value

The repository can now produce something people can directly play with, replay, and potentially present publicly.

These are not fake benefits. They are genuine new capabilities.

## 12. How to measure the added value

The mistake would be to measure all added value only by final win_rate.

The project now needs a broader measurement frame.

### 12.1 Research efficiency metrics

Measure:

- number of successful runs per night
- crash rate per 100 runs
- time lost due to interrupted runs
- time required to locate best historical checkpoint

### 12.2 Comparability metrics

Measure:

- how many runs share identical eval settings
- whether resumed and fresh runs are reported separately
- whether benchmark reports clearly distinguish minimax and NN-opponent evaluation

### 12.3 Interpretability metrics

Measure:

- time to diagnose a regression using replay assets
- frequency with which replay changes model design decisions
- human confidence that a higher win_rate reflects better play quality

### 12.4 Platform usability metrics

Measure:

- time to resume from a stopped run
- time to replay a checkpoint game
- time to load a checkpoint into play.py
- operator effort to manage outputs over multiple runs

### 12.5 Content and presentation metrics

Measure:

- how often recorded games are actually used for analysis or presentation
- whether checkpoint recordings help explain training growth over time

## 13. Where the added value can surface in the future

If maintained well, the current architecture can create future value in areas upstream does not directly support.

### 13.1 Automated curriculum and opponent scheduling

The combination of tracker.db, checkpoint history, and registered opponents can become a real curriculum-learning framework.

### 13.2 Regression analysis

Recorded games plus structured run data can support finding not just whether a model got stronger, but where it regressed strategically.

### 13.3 Multi-agent research workflows

tracker.db can become shared memory for specialized agents:

- one agent proposes training changes
- one agent analyzes metrics
- one agent reviews replay artifacts

### 13.4 Public demonstrations and educational material

Replay and montage systems can turn training progress into something legible to humans.

### 13.5 Model lineage and checkpoint governance

The current schema already points toward a useful model genealogy system.

## 14. Recommended Improvements

These recommendations are ordered by leverage.

### Recommendation 1: separate strict research mode from platform mode

This is the most important recommendation.

Define two explicit operating modes:

1. strict-autoresearch mode
   - fixed 5-minute budget
   - fixed minimax benchmark only
   - no resume for benchmark comparisons
   - no custom opponent as the primary success metric
   - results summarized in a comparable benchmark report

2. platform mode
   - resume allowed
   - optional time budget
   - replay and opponent features encouraged
   - exploratory or product-oriented runs allowed

Reason:

- This resolves most confusion without deleting useful capabilities.

### Recommendation 2: rewrite docs/program.md to match reality

docs/program.md should explicitly describe:

- tracker.db as the system of record
- UUID run directories
- the current CLI and stop conditions
- the difference between benchmark mode and exploratory mode
- what should and should not be modified by an agent

Reason:

- The current drift is a clear correctness issue.

### Recommendation 3: define one canonical benchmark profile

Create a documented benchmark contract such as:

- eval against minimax only
- fixed probe window
- fixed eval interval
- fixed full evaluation game count
- fixed time budget
- no resume

Then report all headline progress against that profile.

Reason:

- This restores a clean experimental frontier while preserving flexible runtime features.

### Recommendation 4: reduce non-learning responsibilities inside train.py

Without violating the spirit of an agent-editable training surface, move purely operational helpers out of train.py where practical.

Candidates:

- some TUI formatting helpers
- some tracker persistence helpers
- some checkpoint naming and reporting helpers

Important note:

- Do not over-modularize. The goal is not abstraction for abstraction's sake.
- The goal is to keep the research-critical parts easier to change and easier to reason about.

### Recommendation 5: make benchmark provenance explicit in outputs

Every run summary should clearly state:

- whether the run was fresh or resumed
- whether the evaluation target was minimax or a named NN opponent
- whether the run belongs to benchmark mode or exploratory mode

Reason:

- This prevents misleading comparisons later.

### Recommendation 6: keep replay and play as first-class, but stop treating them as research-neutral

These tools are useful and should stay.
But the repo should explicitly admit that they are platform features, not neutral research infrastructure.

Reason:

- Clearer framing improves design decisions.

### Recommendation 7: add tracker-driven analysis views

The project would benefit from lightweight analysis commands or notebooks that answer:

- best checkpoint per run
- frontier over time
- win_rate progression under canonical settings
- checkpoint lineage graph
- opponent registry effectiveness

Reason:

- once tracker.db exists, leaving analysis underdeveloped wastes much of its value

### Recommendation 8: explicitly define the project's priorities

The maintainers should state whether the priority order is:

1. benchmark progress
2. operational stability
3. human usability
4. media or replay value

or some other ordering.

Reason:

- Many current arguments about upgrade vs regression are really arguments about unstated priorities.

## 15. Final Judgment

The project has not betrayed autoresearch.

But it has outgrown the original autoresearch container.

That outgrowth produced real value:

- better tracking
- better persistence
- better replayability
- better local usability
- better long-run practical utility

It also produced real costs:

- weaker minimalism
- weaker benchmark purity
- more architectural sprawl
- more documentation drift

So the fairest conclusion is:

- relative to upstream minimal research philosophy, the repo has regressed in purity
- relative to practical long-horizon usefulness, the repo has upgraded substantially
- relative to the initial MAG Gomoku version, most changes are best described as operational and product-level upgrades with some experiment-discipline erosion

The correct next move is not to undo the engineering.

The correct next move is to make the two identities explicit:

- a strict autoresearch benchmark mode
- a richer Gomoku AI platform mode

If that separation is made clear, the current repository can keep the value it gained without continuing to erode the clarity it inherited.

---

## 17. V6 执行结果

### 已完成 (commit f54678f)

| 任务 | 状态 | 说明 |
|------|------|------|
| P0: program.md 重写 | 完成 | 清除全部死引用，更新至当前现实 |
| P1: Benchmark profile | 完成 | is_benchmark + eval_opponent 字段，自动检测 |
| P2: NN对手温度采样 | 完成 | temperature=0.5，softmax + categorical 替代 argmax |
| P3: Provenance 标签 | 完成 | TUI header 显示 [benchmark]/[exploratory]，summary 同步 |
| P4: analyze.py | 完成 | 6 个子命令: --runs/--best/--frontier/--compare/--lineage/--opponents |
| P5: TUI 提取 | 完成 | sparkline/sparkline2/progress_bar 提取到 src/tui.py (55行) |

### 代码量变化

- train.py: 1076 行 -> 1063 行 (净减13行，提取TUI后更聚焦)
- 新增: tui.py (55行), analyze.py (274行)
- tracker.py: 572 -> 582 行 (+10行 migration)

### 测试验证

快速训练测试 (--target-win-rate 0.50 --time-budget 60):
- [benchmark] 标签正确显示
- DB 中 is_benchmark=1 正确记录
- analyze.py 全部子命令通过
- TUI 提取后 sparkline 渲染正常

---

## 16. Copilot (Claude) 对 GPT 分析的回应

> 作者: Copilot / Claude Opus 4.6
> 时间: 2026-04-11
> 角色: 本项目 v2—v5 的实际实现者。以下基于完整的代码库阅读、git 历史、以及实际踩坑经验给出回应。

### 16.1 总体评价：GPT 的分析质量

这份分析的质量很高。GPT 准确地抓住了项目演进中最核心的张力——**autoresearch 的极简实验精神** vs **我们逐步构建出的操作平台**。几个关键判断我完全认同：

1. **"最大的偏离发生在领域转换的那一刻"** — 正确。从 LLM pretraining 到 Gomoku self-play，问题结构已经完全不同。后续 v2-v5 的工程扩展是这个根本分歧的自然延伸，不是偏离的主因。

2. **"两个身份但只有一份控制文档"** — 这是目前最紧迫的结构性问题。`docs/program.md` 还在描述 `results.tsv`、`data/` 目录、固定 5 分钟时间预算等早已不存在的概念。Agent 如果照着这份文档操作，会立刻碰壁。

3. **"train.py 接近 god file"** — 公正的评价。1076 行、25 个函数，包含了模型定义、自对弈、训练循环、TUI 渲染、CLI 编排、checkpoint 保存、对手加载、子进程评估。作为实现者，我承认每轮 v 更新都在往这个文件里塞东西。

### 16.2 我不完全同意的地方

**关于 "benchmark purity 风险"（Problem 3）：**

GPT 认为 custom NN opponents、resume chains、smoothed metrics 削弱了实验可比性。这在理论上成立，但在实践中被高估了。原因：

- 我们的 `tracker.db` 记录了每次 run 的完整超参数（包括 `eval_level`、`eval_opponent`、`time_budget`）。只要查询时按条件筛选，可比性是可恢复的。
- 真正的可比性杀手不是 feature 的存在，而是 **缺乏标注**。只要 DB 明确记录了 "这次 run 用了什么设置"，任何两次 run 都可以判断是否可比。
- Smoothed WR 没有替代 raw WR——两者都存在于 `cycle_metrics` 表中。Smoothed 只用于 checkpoint 触发决策。

**关于 "Recommendation 1: 分离 strict/platform 两个模式"：**

这个建议方向正确，但我认为 GPT 低估了实现成本、高估了实际收益。原因：

- 我们项目的核心用户场景不是 "AI agent 跑无人值守实验循环"——那是上游 autoresearch 的场景。我们的用户（Sean）是在 **手动运行训练、观察 TUI、调整参数、制作视频素材**。
- 为这个用户场景引入 `--mode strict` 和 `--mode platform` 的分叉，增加的认知负担可能大于收益。
- 更实际的做法是：**将 benchmark 定义为一组固定参数组合**（写在文档中），而不是代码中的运行时模式切换。

### 16.3 我作为实现者的补充诊断

GPT 从外部审计角度出发，有几个实际问题它没有触及：

**问题 A: NN opponent probe 和 checkpoint full eval 使用不同的评估对手**

这是 v5 实现中的一个已知不一致。当指定 `--eval-opponent L0` 时：
- Probe 评估使用 NN 对手（快速，在主进程中）
- Checkpoint 的 full eval 仍然使用 minimax（通过 `_subprocess_eval` → `prepare.py`）

这意味着 probe WR（触发 checkpoint 的依据）和 checkpoint 记录的 WR 可能有显著差异。这不是 bug——minimax full eval 是有意保留的 "标准尺"——但需要文档化。

**问题 B: `_nn_opponent_move()` 使用纯 argmax，确定性过高**

当前 NN 对手的行动策略是 policy logits 的 argmax。这意味着在相同局面下，NN 对手的走法完全确定，导致：
- 对弈多样性低
- 少量对局的 WR 可能不具代表性（20 局 probe 给出 100% 的极端结果）
- 训练模型可能 overfit 到 NN 对手的固定策略

应考虑在 NN 对手的 move 选择中引入 temperature sampling。

**问题 C: `docs/program.md` 的过时程度比 GPT 描述的更严重**

不只是 "drift"，program.md 中有多处 **直接错误**：
- 第 94 行: 引用 `data/data/results.tsv`，此路径从 v1 起就不存在
- 第 4-5 行: 提到 `autoresearch-mlx/mag-gomoku/` monorepo 路径，当前不适用
- 第 66-85 行: Output format 描述的是一个已被完全替代的格式
- 第 116-140 行: 实验循环描述中使用 `git reset --hard`、`git commit --amend` 等操作，与当前 tracker.db + UUID 工作流完全脱节
- 第 17 行: `uv sync` 的路径假设不正确

### 16.4 偏离度的量化评估

| 维度 | GPT 评估 | 我的评估 | 备注 |
|------|---------|---------|------|
| 方法论偏离 | 中等 | 中等 | 同意。核心循环 "改→跑→评→保/弃" 仍在 |
| 问题域偏离 | 极高 | 极高 | 同意。这是第一次 commit 就决定的 |
| 架构偏离 | 高 | 中高 | 6 个 src 文件 vs 原版 3 个，但职责边界清晰 |
| 实验纪律偏离 | 中高 | 中等 | DB 中有完整记录，可比性可恢复 |
| 实用价值提升 | 显著 | 显著 | 完全同意 GPT 的 5 项价值评估 |

### 16.5 v6 工作计划建议

基于 GPT 的分析和我的补充诊断，以下是我建议的 v6 更新工作，按优先级排序：

#### P0: program.md 重写（回归工作）

这是最紧迫的回归任务。当前 program.md 的错误信息已经不是 "drift"，而是 "wrong"。

具体工作：
- 删除所有对 `data/results.tsv` 的引用
- 删除 monorepo 路径假设
- 更新 Output format 为当前 tracker.db + UUID 目录结构的实际格式
- 更新实验循环描述，反映 `--target-win-rate` / `--time-budget` / `--resume` 的实际工作流
- 明确标注哪些是 agent 可编辑的、哪些是只读的
- 新增 tracker.db 和 checkpoint 系统的说明
- 保留原版精神中有价值的部分：单文件编辑纪律、简洁性优先原则、stage promotion 概念

#### P1: 定义 Canonical Benchmark Profile（回归工作）

不做运行时模式切换（GPT Rec 1 的简化版），而是在文档中定义一个 "标准基准配置"：

```
Canonical Benchmark:
  --eval-level 0      (或指定级别)
  --time-budget 300   (固定 5 分钟)
  --probe-games 50
  --eval-interval 15
  --full-eval-games 200
  不使用 --resume
  不使用 --eval-opponent
```

所有 "标题级进展" 必须使用此配置产生，确保跨 run 可比性。探索性训练（自由参数、NN 对手、长时间运行）允许且鼓励，但需在报告中明确标注。

在 `tracker.db` 的 `runs` 表或 hyperparams 中添加 `is_benchmark: bool` 字段来区分。

#### P2: NN 对手 move 策略改进（自身提升）

为 `_nn_opponent_move()` 增加 temperature 参数（默认 0.5），通过 softmax sampling 代替纯 argmax，增加对弈多样性：

```python
def _nn_opponent_move(opp_model, board, temperature=0.5):
    ...
    if temperature > 0:
        probs = mx.softmax(masked / temperature)
        action = int(mx.random.categorical(probs).item())
    else:
        action = int(mx.argmax(masked).item())
```

#### P3: Run summary 中增加 benchmark provenance 标注（回归工作）

对应 GPT Rec 5。每次 run 结束时的 summary 明确打印：
- `Mode: benchmark` 或 `Mode: exploratory`
- `Eval: minimax L0` 或 `Eval: NN opponent 'L0'`
- `Fresh run` 或 `Resumed from <uuid>`

这些信息已经在 DB 中，只需在终端输出中显式呈现。

#### P4: tracker 驱动的分析命令（自身提升）

对应 GPT Rec 7。新增 `src/analyze.py`（只读工具，不违反 autoresearch 约束）：

```bash
uv run python src/analyze.py --best          # 每个 run 的最佳 checkpoint
uv run python src/analyze.py --frontier      # WR 历史最高线
uv run python src/analyze.py --compare A B   # 两个 run 的对比
uv run python src/analyze.py --lineage       # checkpoint 族谱图
```

#### P5: train.py 瘦身（可选的长期改善）

对应 GPT Rec 4，但需要谨慎。**不能违反 autoresearch 的单文件编辑约束**。

可安全提取的候选（纯展示/工具函数，agent 永远不需要修改）：
- `_sparkline()`, `_sparkline2()`, `_draw_panel()`, `_update_tui()` → `src/tui.py`
- `_handle_register_opponent()` → 可以放到独立命令或 play.py 中

**不应提取的**（agent 可能需要修改的研究逻辑）：
- `GomokuNet`, `run_self_play()`, `compute_loss()`, `train()` — 这些必须留在 train.py

但注意：这个操作有争议。即使提取的是 "纯操作函数"，也会增加 agent 理解代码库的认知负担。建议标记为低优先级，仅在 train.py 超过 1500 行时再考虑。

### 16.6 不建议做的事情

1. **不建议引入 `--mode strict/platform` 运行时切换** — 增加代码复杂度，用文档约定即可达到同样效果
2. **不建议将 replay/play 降级为 "非研究基础设施"** — 对我们的实际用户场景（视频制作），这些是核心功能
3. **不建议回退到 results.tsv** — tracker.db 是明确的升级，没有回退价值
4. **不建议强制所有 run 使用固定 5 分钟预算** — 可选的时间预算是正确的设计决策，通过 benchmark profile 约束即可

### 16.7 总结

GPT 的分析是一份优秀的外部审计报告。它正确识别了项目的核心张力、五个明确的问题、以及八个有建设性的改进方向。

我的补充重点是：
1. **program.md 的问题比 GPT 描述的更严重** — 不是 drift，是 wrong
2. **可比性的恢复路径是标注而非限制** — DB 已有数据，缺的是文档和显示
3. **NN 对手的确定性问题需要技术性修复** — temperature sampling
4. **模式分离应通过文档约定而非代码分叉实现** — 更轻量，更务实

v6 的核心主题应该是 **"回归与对齐"**：把实际系统行为和文档描述重新对齐，定义清晰的 benchmark 标准，让项目的两个身份（研究工具 + 制作平台）各得其所。