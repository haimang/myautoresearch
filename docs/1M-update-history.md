# 1M Update History

> 日期: 2026-04-11
> 模型代号: 1M
> 定义: 12x64 架构，约 1.01M 参数

---

## Stage 1: 1M 首次训练

### 训练身份

| 项目 | 值 |
|---|---|
| Run ID | `8b9486f4` |
| 完整 Run ID | `8b9486f4-6b42-4523-8fbe-fa8568108e0b` |
| 架构 | `12x64` |
| 参数量 | `1,010,675` |
| 输出目录 | `output/8b9486f4-6b42-4523-8fbe-fa8568108e0b` |

### Stage 1 训练命令

```bash
uv run python src/train.py --num-blocks 12 --num-filters 64 --learning-rate 7e-4 --target-win-rate 0.80 --eval-opponent L0 --parallel-games 80 --probe-games 120 --probe-window 7 --full-eval-games 200 --seed 42
```

### Stage 1 参数快照

| 参数 | 值 |
|---|---|
| num-blocks | `12` |
| num-filters | `64` |
| learning-rate | `7e-4` |
| target-win-rate | `0.80` |
| eval-opponent | `L0` |
| parallel-games | `80` |
| probe-games | `120` |
| probe-window | `7` |
| full-eval-games | `200` |
| replay-buffer-size | `50000` |
| steps-per-cycle | `30` |
| seed | `42` |

### Stage 1 结果

| 指标 | 值 |
|---|---|
| 状态 | `completed` |
| 总周期 | `675` |
| 总对局 | `54,000` |
| 总训练步数 | `20,250` |
| 总耗时 | `4,243.6s` |
| final loss | `0.3733` |
| final win rate | `78.0%` |

### Stage 1 Checkpoint 摘要

| 类型 | Tag | Cycle | Full Eval | 详情 |
|---|---|---:|---:|---|
| 最新 checkpoint | `wr080_c0675` | `675` | `85.0%` | `170W / 30L / 0D`, `200` games |
| final checkpoint | `final_c0675` | `675` | `78.0%` | `156W / 44L / 0D`, `200` games |

### Stage 1 评估 Summary

1. 1M 首次训练已经证明 12x64 可以在 L0 标尺下稳定长起来。
2. 最新 checkpoint `wr080_c0675` 的 full eval 达到 `85.0%`，说明模型在训练末段已经明显越过 `target-win-rate 0.80` 对应门槛。
3. 但最终 `final_c0675` 只有 `78.0%`，说明末尾状态存在一定回落，最佳点应以 `wr080_c0675` 为准，而不是 final model。
4. 最近运行稳定性统计显示：`44` 次 probe，均值 `77.6%`，标准差 `4.1%`，范围 `65.8%–85.0%`；Loss 均值 `0.893`，整体下降约 `81%`。
5. 结论上，这次训练是成功的“1M 冷启动基线建立”，并且给出了一个适合继续续训的末段 checkpoint。

### Stage 1 当前结论

1. `12x64 (~1.01M)` 是可训练的，并没有出现此前 `15x64` 直接对强 NN 评估时那种坍塌现象。
2. `7e-4` 在 L0 阶段仍然有效，能够较快把 1M 模型推到可用强度。
3. 继续训练时应从 `wr080_c0675` 这一末段最好 checkpoint 出发，而不是重新从零开始。

---

## Stage 1 之后的多轮尝试总结

> 说明: 多次手动中断的续训在 tracker 里仍显示为 `running`。这不是结果缺失，而是当前中断路径没有把这些 run 正常 finalize 到 completed / interrupted 状态。

### 续训尝试总表

| Run | 主要设定 | 观察到的结果 | 结论 |
|---|---|---|---|
| `a644f563` | `resume 8b9486f4`, `lr=6e-4`, `eval=L1`, 目标 `85%`，并尝试保留低阶基础盘 | 最近记录 probe 长期在 `37.5%–47.5%` 区间徘徊，未出现接近过线的抬升 | 能继续优化 loss，但无法把能力真正迁移到更高评估目标 |
| `f4b7be19` | `resume 8b9486f4`, `lr=6e-4`, `eval=L4`, 目标 `85%` | 最近记录 probe 只有 `4.2%–14.2%`，基本属于被 L4 全面压制 | 直接把 1M 拉到 L4 标尺时，现有训练信号几乎失效 |
| `743db53d` | `resume 8b9486f4`, `lr=5e-4`, `eval=L4`, 目标 `85%` | 最近记录 probe 在 `17.5%–34.2%` 之间波动，仍然远低于可用区间 | 降低学习率并未解决根因，只是让坍塌形式稍缓 |
| `a456c6cf` | `resume 8b9486f4`, `lr=5e-4`, `eval=L0`, `target=95%`, `probe=150`, `full-eval=250` | 最近记录 probe 在 `68.7%–82.0%` 之间波动，最高到 `82%`，但无法持续逼近 `95%` | 即使回到熟悉对手，自博弈后期仍明显进入低收益平台期 |

### 刚刚结束的一次训练结果

最近一轮结束的 1M 续训，是以下命令对应的 run：

```bash
uv run python src/train.py --resume 8b9486f4 --num-blocks 12 --num-filters 64 --learning-rate 5e-4 --target-win-rate 0.95 --eval-opponent L0 --parallel-games 80 --probe-games 150 --probe-window 7 --full-eval-games 250 --seed 42
```

这轮的核心现象不是“训练直接崩坏”，而是：

1. loss 已经稳定压到 `0.26–0.31` 一带
2. probe 胜率多次达到 `80%–82%`
3. 但长期无法继续向 `95%` 逼近
4. 说明模型优化器还在工作，但训练目标产出的新增信息已经不足以驱动后续成长

这轮结果很关键，因为它排除了“只有在高难对手下才失败”的解释。即使把评估重新放回 `L0`，只要训练仍主要依赖当前这套自博弈信号，1M 模型在后期依旧会进入明显平台期。

---

## 当前诊断: 瓶颈更像信号质量，而不是超参数

### 结论先行

到目前为止，已有足够证据支持下面这个判断：

> 1M 当前的主要瓶颈，不再是单个超参数，而是训练信号本身过于稀疏、过于自我复制、缺少高质量纠错目标。

### 为什么不是超参数问题

我们已经实际尝试过多个方向：

1. 学习率从 `7e-4` 下调到 `6e-4`、`5e-4`
2. 评估目标从 `L0` 切到 `L1`、`L4`
3. 目标胜率从 `0.80` 提到 `0.85`、`0.95`
4. probe / full-eval 预算增加
5. 继续从同一个 1M checkpoint 续训，而不是反复从零启动

如果问题主要是超参数，通常我们应该看到下面其中一种模式：

1. 某个学习率明显有效，WR 开始持续上升
2. 某个目标对手切换后，训练突然稳定
3. 降低学习率后，后期可以继续逼近更高阈值

但实际观察不是这样。相反，出现的是一种高度一致的模式：

1. loss 可以继续下降
2. probe 胜率却经常停滞，甚至在强对手下接近无效
3. 同一模型在较弱目标下还能维持可用表现，但很难从自博弈里继续长出新能力

这说明优化器没有坏，模型容量也不是完全不够；真正卡住的是“训练数据还能不能继续提供有效的新信息”。

### 代码层面的根因

当前训练信号的核心结构非常简单：

1. **value target 只有终局标签**
	- 赢 = `1.0`
	- 输 = `-1.0`
	- 和 = `0.0`

2. **policy target 直接来自当时自己采样出来的分布**

3. **loss 只是 policy cross-entropy + value MSE**

这套设计在早期足够工作，但在 1M 的后期会暴露三个问题：

#### 1. 信号过于稀疏

所有中间质量差异，最终都被压缩成终局输赢标签。  
关键坏手、普通坏手、被强对手精准惩罚的坏手，在监督里经常只体现为同一个 `-1.0`。

#### 2. 信号过于自我复制

policy 头学到的不是“更强策略”，而是“把自己刚刚的采样分布再拟合一遍”。  
没有 MCTS、没有 teacher、没有强对手生成的显式策略监督时，后期训练就会越来越像自我模仿，而不是自我提升。

#### 3. 缺少高质量纠错目标

即使引入 `eval-opponent`，只要没有把高价值失败样本有效灌回训练，强对手仍然只是在“打分”，不是在“教学”。  
所以现象就会变成：

1. 分数知道自己不够强
2. 但训练过程并没有持续、密集地看到那些真正该纠正的局面

这就是为什么我们会看到：

1. 对 L4 的评估很差
2. 但 loss 仍然下降
3. 对 L0 又还能维持一定胜率

这不是普通意义上的训练崩溃，而是典型的**信号密度不足**。

---

## 对 1M 训练机制改动的预想

基于上面的诊断，后续修复方向不应优先落在“再微调一次超参数”，而应直接提升训练信号质量。

### 预想方向 A: 强化高价值失败样本

当前 replay 主要是 recency-weighted，并不会优先学习那些真正说明“模型哪里不行”的样本。  
更合理的做法是提高下面这些样本的训练权重：

1. 对强对手输掉的样本
2. opponent-play 中的样本
3. 自博弈中最终落败一侧的样本

这样做的目标不是手工改 reward，而是让训练更频繁地看到“真正需要被修正的局面”。

### 预想方向 B: 提高 opponent 样本密度

如果目标是超越更强的外部对手，那么训练中不能让这些对手只存在于 probe / full eval。  
应当让强对手参与生成更高信息密度的数据，并确保这些数据不会被普通自博弈样本迅速淹没。

### 预想方向 C: 在 batch 内保留纠错配额

仅靠全局概率加权还不够稳。更进一步的机制，是在每个训练 batch 中保留一部分配额给：

1. opponent-play 样本
2. 失败样本
3. 关键纠错样本

这样可避免“高价值样本在大 replay buffer 里被稀释”。

### 为什么我不优先做 reward shaping

手工正向奖励 / 反向惩罚虽然直觉上很诱人，但在棋类问题里很容易把模型带向 proxy。  
例如奖励更快进攻、更多连子、占中心，都可能让模型学会讨好指标，而不是更会赢棋。

因此更稳妥的路线是：

1. 保持终局胜负仍然是核心 truth
2. 先改“哪些样本更值得学”
3. 再考虑是否需要更复杂的 reward shaping

## Stage 2: 计划中的续训方案

### 目标

从 Stage 1 的最新 checkpoint 继续训练，开始切换到更强目标与混合训练分布。

### 计划参数

| 参数 | 值 |
|---|---|
| resume source | `8b9486f4` |
| learning-rate | `6e-4` |
| target-win-rate | `0.85` |
| eval-opponent | `L1` |
| train-opponent | `L0` |
| opponent-mix | `0.2` |
| parallel-games | `80` |
| probe-games | `120` |
| probe-window | `7` |
| full-eval-games | `200` |
| seed | `42` |

### Stage 2 测试命令

```bash
uv run python src/train.py --resume 8b9486f4 --num-blocks 12 --num-filters 64 --learning-rate 6e-4 --target-win-rate 0.85 --eval-opponent L1 --train-opponent L0 --opponent-mix 0.2 --parallel-games 80 --probe-games 120 --probe-window 7 --full-eval-games 200 --seed 42
```

### Stage 2 预期用途

1. 在不丢失 1M 已有基础能力的前提下，开始从 `L0` 升级到 `L1` 标尺。
2. 通过 `20%` 的 `L0` 混合对局，降低切换阶段时完全忘掉低阶基本盘的风险。
3. 观察 `6e-4` 相对 Stage 1 的 `7e-4` 是否能提供更稳的中期推进。

### Stage 2 结果回填区

| 指标 | 值 |
|---|---|
| 状态 | 待回填 |
| 总周期 | 待回填 |
| 总对局 | 待回填 |
| 总训练步数 | 待回填 |
| 总耗时 | 待回填 |
| final loss | 待回填 |
| final win rate | 待回填 |
| 最佳 checkpoint | 待回填 |
| 最佳 full eval | 待回填 |

### Stage 2 评估回填区

待回填。

---

## Stage 3: 预定方向

在 Stage 2 完成后，下一步计划为：

1. `learning-rate = 5e-4`
2. `eval-opponent = L3`
3. `train-opponent = L1`
4. 保持 1M 架构继续向更强 NN 对手推进

具体命令与结果，待 Stage 2 完成后再补充。

---

## 2026-04-12: 信号质量修复记录

> 基线 commit: `39bbdd8`  
> 目标: 不再把问题继续归因于超参数微调，而是直接提高 1M 的训练信号密度与纠错强度。

### 本轮代码修改摘要

本轮没有引入手工 reward shaping，而是保持“终局胜负仍然是 truth”，只修改 replay 采样机制，让训练更频繁地学到真正有价值的样本。

#### 修改 1: 为 replay sample 增加来源与优先级元数据

在 `src/train.py` 中新增了 `ReplaySample` 结构，样本不再只是：

1. board
2. policy
3. value

而是额外记录：

1. `source` — 样本来自 `self` 还是 `opponent`
2. `priority` — 该样本在 replay 中的抽样优先级

这一步的意义是：从现在开始，训练系统可以区分“普通自博弈样本”和“真正暴露弱点的样本”。

#### 修改 2: 引入 opponent / loss priority boosting

新增了以下优先级常量：

1. `OPPONENT_SAMPLE_BOOST = 4.0`
2. `LOSS_SAMPLE_BOOST = 2.5`
3. `WIN_OPPONENT_SAMPLE_BOOST = 1.5`

含义如下：

1. 对手对局产生的样本，默认更常被抽到
2. 最终落败一侧的样本，也会被更高频地学习
3. 对强对手赢下来的样本，也会被适度保留，避免只学防守失败而不学有效反制

这不是“改 reward”，而是“让模型更常学习高信息样本”。

#### 修改 3: 在每个 batch 内保留纠错样本配额

新增 `FOCUSED_SAMPLE_RATIO = 0.25`，并实现 `_sample_replay_indices()`：

1. 每个 batch 默认预留约 `25%` 配额给纠错样本
2. 纠错样本定义为：
	- `opponent-play` 样本
	- `value < 0` 的失败样本
3. 剩余 `75%` 再从全体 replay 中按 recency × priority 抽样

这一步非常关键，因为它解决的是“高价值样本在大 buffer 中被淹没”的问题。

#### 修改 4: 将 avglen 以低风险方式接入 replay priority

这一步没有把 avglen 直接写进 reward / value target，而是只影响样本优先级。

关键原则是：

1. **使用每盘棋自己的 game length**，而不是 TUI 上按 cycle 汇总的平均 `AvgLen`
2. **只做轻微倍率调整**，避免破坏当前 value 头的 `[-1, 1]` 语义
3. **保持终局胜负仍然是唯一 truth**，avglen 只影响“哪些样本更值得反复学习”

本轮采用的低风险规则是：

1. 赢棋且 `game_length < 85`：轻微上调优先级
2. 赢棋且 `game_length < 75`：进一步上调优先级
3. 赢棋且 `game_length < 60`：较明显上调优先级
4. 赢棋且 `game_length > 85`：轻微下调优先级
5. 赢棋且 `game_length > 90`：进一步下调优先级
6. 输棋且 `game_length < 60`：额外上调优先级，用于强化“被快速击穿”的纠错价值

为什么这是低风险版本：

1. 不改变训练目标的数值边界
2. 不会把 value target 推到 `1.2 / 1.5 / 2.0` 之类不可表示区间
3. 不会把整套系统从“赢棋 truth”改成“追求短局 truth”
4. 只是让训练更频繁地回看那些“赢得很高效”或“输得很惨”的对局

### 本轮修改的预期提升

这批改动对 1M 的预期提升，不是“突然多拿 10% 胜率”，而是更基础也更重要的三件事：

1. **让后期训练不再只有低信息自博弈样本**
	- 失败样本会被系统性拉高权重
	- 后期平台期将不再完全依赖随机碰运气突破

2. **让 train-opponent 真正成为教学源，而不只是混入背景噪声**
	- 以前就算设置了 `train-opponent`，这些样本也可能被普通 self-play 快速淹没
	- 现在 opponent-play 样本会被显式抬权

3. **让 1M 在更强对手面前更容易形成“纠错循环”**
	- 不是只知道自己输了
	- 而是更频繁地训练那些导致失败的具体局面

4. **让 1M 更快固化高效率胜局，而不是平均对所有赢局一视同仁**
	- 快速、高质量赢下来的对局会被更频繁地重复学习
	- 冗长、低效但侥幸获胜的样本会适度降权

### 为什么这比继续调 learning rate 更值得优先做

原因很直接：

1. `7e-4 → 6e-4 → 5e-4` 我们已经试过多轮
2. 结果显示 loss 还能下降，但强度迁移并没有随之持续改善
3. 这说明优化器不是主矛盾，样本价值分布才是主矛盾

因此，这一轮的正确动作不是再找一个新的学习率，而是让训练更集中地处理“模型为什么输”。

### 最新推荐 CLI

为了真正利用这次“signal-density replay + avglen-aware priority”改动，下一轮不再建议 `eval=L1` 但 `train=L0`。  
如果继续这么做，提升后的 opponent 样本优先级大部分还是会浪费在低级对手上。

当前更推荐的下一轮，是直接让 `L1` 同时成为评估对手和训练中的一部分教学对手：

```bash
uv run python src/train.py --resume 8b9486f4 --num-blocks 12 --num-filters 64 --learning-rate 6e-4 --target-win-rate 0.85 --eval-opponent L1 --train-opponent L1 --opponent-mix 0.2 --parallel-games 80 --probe-games 120 --probe-window 7 --full-eval-games 200 --seed 42
```

### 对这条新命令的预期

1. `80%` 自博弈仍然保留通用能力生成
2. `20%` 的 `L1` 对局现在不再是弱信号，而会被 replay 优先采样机制放大
3. 每个 batch 都更有机会看到“为什么打不过 L1”的失败局面
4. avglen 现在也会对优先级产生轻量作用，使高效率胜局更容易被巩固
5. 这将比单纯继续 `eval=L1` 但 `train=L0` 更有机会形成真实提升

### 本轮修改日志

| 文件 | 修改点 |
|---|---|
| `src/train.py` | 新增 `ReplaySample`、样本优先级、focused batch sampling、avglen-aware priority |
| `docs/1M-update-history.md` | 补充失败尝试总结、瓶颈分析、信号质量修复记录、avglen 低风险接入与最新 CLI |

当前状态：这批“信号质量修复”代码已经完成，并通过语法编译与 CLI 启动级别验证。下一步应直接运行上面的新命令，观察 1M 在 `L1` 上是否开始出现比旧机制更清晰的持续改进。