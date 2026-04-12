# Update v10 Analysis - 回归 autoresearch 路线的必要性、路线规划与执行建议

> 日期：2026-04-11  
> 范围：围绕 MAG Gomoku 当前已经出现的 AutoML 倾向，重新从源头审视项目定位，说明为何必须纠偏、为何现在具备回归 autoresearch 路线的现实条件，并给出 v10+ 的总体升级路线、v10 的具体执行建议，以及本阶段应确立的 statement 与 verdict。

---

## 1. 执行摘要

我们当前面临的不是一个普通的工程分叉问题，而是一个**项目身份问题**。

MAG Gomoku 起步时的核心不是“做一个能自动回归超参的训练框架”，而是：

> **把 autoresearch 的外层研究闭环迁移到五子棋场景中，让 agent 成为真正的研究主体，而训练脚本成为被研究、被修改、被反复实验的对象。**

但随着版本推进，系统逐步加强了：

- 训练脚本内部的自动化
- benchmark / exploratory 分类
- checkpoint 与 tracker 体系
- 对手注册与混合训练
- 参数化与快筛思路

这些能力本身并不是错误，但它们共同带来了一个根本变化：

> **研究控制权正在从外层 agent 手中，转移到训练脚本内部的固定规则与指标流程中。**

一旦这种趋势继续下去，项目就会从“autoresearch in Gomoku”逐步变成“一个自对弈训练 + benchmark 驱动的实验平台”，进一步滑向 AutoML 范式。

而这恰恰是当前最值得警惕的地方。

因为如果目标只是做 AutoML：

- 已经有大量成熟框架可用
- 现成的超参搜索、Bayesian optimization、population based training、sweeper、scheduler 都不缺
- 从 autoresearch 出发就不再有足够必要性

MAG Gomoku 的真正独特价值不在于“也可以做超参回归”，而在于：

1. 它可以把强 agent 真正接入研究闭环
2. 它已经拥有比原版 autoresearch 更丰富的观测基础设施
3. 它有机会让 agent 看到的不只是单个 summary，而是结构化的长期实验史、稳定性、frontier、lineage、对手生态与 replay 数据

因此，本轮分析的核心结论是：

1. **当前的 AutoML 倾向是本质偏离，而不是轻微风格漂移。**
2. **这个偏离必须纠正，而且越早越好。**
3. **我们现在具备比原版 autoresearch 更好的 agent 接入条件，因此不仅可以回归，而且应该以“增强版 autoresearch”作为下一阶段目标。**
4. **v10 的任务，不是继续优化训练参数，而是重建外层 agent 研究闭环的架构方向。**

---

## 2. 必须从源头重新理解项目初心

### 2.1 autoresearch 的原点是什么

autoresearch 的真正核心，不是某一个指标，也不是某一个训练脚本，而是下面这个研究秩序：

1. 外层 agent 观察结果
2. 外层 agent 提出假设
3. 外层 agent 修改训练代码
4. 训练脚本跑出新实验结果
5. 固定 benchmark 负责裁决真假
6. 外层 agent 决定保留、回退或继续推进

也就是说：

- **agent 是研究者**
- **benchmark 是裁判**
- **训练脚本是实验对象**

这三者的角色分工，才是 autoresearch 的灵魂。

### 2.2 当前担忧为什么是合理的

你当前的担忧不是抽象层面的哲学不安，而是对项目结构变化的准确识别。

现在的问题并不是“win rate 不够纯粹”，而是：

1. 单次训练脚本内部承担了越来越多自动决策逻辑
2. 项目叙事开始围绕快筛、收敛、proxy/target、自动回归展开
3. 外层 agent 目前并不是默认运行路径里的主角
4. 跨 run 的研究判断并没有真正被制度化实现

因此，项目正在发生的事情不是“autoresearch 的工程增强”，而是：

> **autoresearch 外壳下的研究控制权内移。**

这就是本质偏离。

---

## 3. 为什么这是“本质偏离”而不是“正常演化”

### 3.1 偏离点不在 benchmark，而在控制中心

很多时候，人们会误以为只要 benchmark 还在、win rate 还在，项目就仍然属于 autoresearch。

这种理解不完整。

benchmark 的存在，只能证明项目还保留了一部分实验裁决纪律；但它不能证明项目仍然是由 agent 主导的研究系统。

真正的问题在于：

- **谁在提出下一轮实验假设？**
- **谁在比较跨 run 结果？**
- **谁在决定下一轮该改哪里？**

如果这些权力逐渐被固定规则、参数化入口、内部 stop 条件与自动回归逻辑所取代，那么即使 benchmark 还在，项目也已经偏离原本的 autoresearch 结构。

### 3.2 当前路径更像什么

当前系统越来越像：

1. 一个结构化 experiment runner
2. 一个 benchmark 驱动的训练平台
3. 一个正在向超参筛选系统靠近的 AutoML 雏形

这类系统当然可以有价值，但它们的价值逻辑已经与 autoresearch 不同。

autoresearch 的价值在于：

- 利用 agent 的开放式分析能力
- 利用 agent 的跨实验记忆与抽象能力
- 让 agent 在固定 benchmark 下进行真正的研究而不是单纯搜索

而 AutoML 的价值在于：

- 系统性枚举或采样搜索空间
- 通过固定算法做参数优化
- 尽可能减少开放式研究判断

这两者不是一回事。

---

## 4. 为什么我们现在不仅可以回归，而且应该回归

### 4.1 今天的 agent 能力已经和最初不同了

这次回归之所以现实，不是因为我们怀旧，而是因为现在的 agent 能力已经明显升级。

当前一代 agent 相比项目最初设想时，有几个关键变化：

1. **更强的长时执行能力**
   agent 可以持续运行更长时间，不再只是短回合辅助。

2. **更强的多步任务能力**
   agent 不只是改一两个参数，而是可以执行“读报告 → 做判断 → 改代码 → 跑实验 → 总结结果”的完整链路。

3. **更强的代码理解与工具使用能力**
   agent 可以利用仓库内现有工具，而不是只能看一段终端日志。

4. **更强的跨实验综合分析能力**
   agent 更适合阅读结构化报告、稳定性分析、frontier 变化、checkpoint 记录，而不只是看单个分数。

因此，今天再谈“回归 autoresearch”，不是退回一个更原始、更弱的方案，而是在更强 agent 基础上，重建更完整的外层研究闭环。

### 4.2 当前仓库已经具备比上游更好的接入条件

从实际基础设施看，MAG Gomoku 已经具备一组非常适合 agent 接入的能力：

1. tracker.db 存在长期实验史
2. analyze.py 已经具备 runs / compare / frontier / stability / lineage 等分析入口
3. checkpoint 体系完整
4. benchmark 与 exploratory 语义已经分层
5. opponent registry 已经形成训练生态入口
6. replay 与 recording 让行为级证据具备扩展空间

这些能力意味着：

> 我们不是“要不要用 agent 的起点太低”，而是“已经有足够好的观测层，却还没有把 agent 正式接回主控位”。

这就是为什么现在不仅可以回归，而且应该回归。

---

## 5. 为什么不能继续滑向 AutoML

### 5.1 因为那会消解项目独特性

如果最终目标只是：

- 快筛超参
- 比较容量
- 测 buffer
- 做 proxy / target 双轨优化

那项目的独特性会快速下降。

因为这些工作虽然重要，但它们属于广义实验工程，不足以构成“为什么必须是 autoresearch-inspired 项目”的回答。

### 5.2 因为 AutoML 会鼓励错误的重心转移

一旦项目重心转向 AutoML，系统会自然鼓励：

1. 把越来越多内容做成固定可调参数
2. 把研究问题压缩成搜索问题
3. 把 agent 的价值降成一个执行器或批处理控制器
4. 让“谁来提假设”这个问题越来越不重要

这与我们真正想探索的方向是冲突的。

### 5.3 因为 agent 的真正价值不是替代网格搜索

强 agent 最有价值的地方，不是帮我们跑更复杂的 sweeps，而是：

- 它能读更复杂的上下文
- 它能形成更高层的实验假设
- 它能在行为证据与指标证据之间做解释性联结
- 它能发现“不是一个单参就能描述”的结构性问题

如果把 agent 只用来代替一层自动调参器，那实际上是在低估它。

---

## 6. v10 之后应回归怎样的 autoresearch 路线

### 6.1 新的总原则

从 v10 开始，项目应明确恢复下面这条原则：

> **训练脚本负责产出事实，agent 负责研究判断，benchmark 负责最终裁决。**

这句话应成为 v10+ 阶段的总原则。

### 6.2 角色重新分工

#### 训练层

训练层只负责：

- 训练
- 评估
- 记录
- 导出 checkpoint / recording / tracker 数据

训练层不应继续吸收更多“跨实验研究决策权”。

#### 报告层

报告层负责把原始运行数据转成 agent 可以消费的结构化实验报告。

报告层不是简单日志，而是“面向研究判断的报告放大器”。

#### Agent 层

Agent 层负责：

- 读取报告
- 形成假设
- 选择下一步修改方向
- 决定 keep / discard / rollback / continue

#### Benchmark 层

Benchmark 层继续保持相对稳定，负责作为外部事实裁判。

---

## 7. v10+ 的总体路线规划

### 阶段 A：纠偏与重新定性

目标：

- 明确项目不是在走向 AutoML
- 明确后续版本以“恢复外层 agent 主控权”为方向
- 把已有实验基础设施重新解释为 agent 工具层

这一阶段的核心不是写很多代码，而是统一方向、统一术语、统一架构边界。

### 阶段 B：建立 Hook + Report 机制

目标：

- 在训练脚本之外建立 agent 可消费的事件触发点
- 在 hook 触发后输出结构化实验报告
- 让 agent 能够看到比原版 autoresearch 更丰富的事实面

建议的 hook 触发点：

1. run-end hook
2. checkpoint full-eval hook
3. frontier-review hook

### 阶段 C：建立 Agent Research Loop

目标：

- 让 agent 真正接管跨实验决策
- 让项目重新拥有“看结果 → 改代码 → 跑实验 → keep / discard”的完整闭环

这一步才是真正的“回归 autoresearch”。

### 阶段 D：扩展为增强版 autoresearch

目标：

- 不只是复刻上游，而是超越上游
- 让 agent 能看到 tracker、stability、lineage、replay、opponent 生态摘要
- 让 agent 在更强事实基础上做研究判断

这一步意味着我们不是简单后退，而是沿着 autoresearch 路线继续推进。

---

## 8. v10 update 的具体执行内容推荐

v10 不应是一个大而散的版本，而应聚焦在“回归路线的架构准备”上。

### 8.1 推荐事项一：正式改写项目阶段定义

内容：

- 在文档层明确指出当前出现的 AutoML 倾向
- 明确后续版本的核心不是继续内部自动回归，而是恢复外层 agent 闭环

原因：

- 如果阶段定义不改，后续所有工程动作仍会沿着 AutoML 惯性滑行

### 8.2 推荐事项二：定义 Hook 架构边界

内容：

- 约定训练脚本只在关键事件产出事实，不直接内嵌 LLM 决策
- 约定 hook 位于训练脚本之外或边界处
- 约定 hook 只在低噪声节点触发

推荐只保留三类关键节点：

1. run-end
2. checkpoint full-eval 完成后
3. 多 run 汇总 review 后

原因：

- 防止 LLM 被高频噪声驱动
- 保持 benchmark 的边界清晰

### 8.3 推荐事项三：定义 Agent Report 的最小可用集合

内容：

v10 应明确，提供给 agent 的第一版报告至少包含：

1. 最近 N 个 run 摘要
2. 当前 run 的稳定性摘要
3. frontier 变化
4. best checkpoints
5. opponent registry 摘要
6. 如果可行，加入 replay 统计摘要

原因：

- agent 的价值来自“看得比单个 summary 更多”
- 没有报告层，hook 只是空壳

### 8.4 推荐事项四：明确 keep / discard 的主决策权回到 agent

内容：

- 恢复跨实验层的研究裁决逻辑
- 不再让内部自动停止和自动筛选承担越来越多研究判断职责

原因：

- autoresearch 的核心不是自动执行，而是 agent 主导研究推进

### 8.5 推荐事项五：把快筛体系降级为 agent 工具，而不是项目身份

内容：

- proxy / target
- quick screen
- stability analysis
- benchmark / exploratory

这些内容仍然保留，但在 v10+ 中应被定义为：

> 供 agent 使用的研究工具，而不是项目主路线本身。

原因：

- 这些方法本身有价值，但它们不能继续抢占 autoresearch 的主身份

---

## 9. v10+ 阶段建议提供给 LLM 的内部工具能力

如果项目要真正回归 autoresearch 路线，必须认真设计 agent 的“感官系统”。

建议至少提供下面几类内部工具能力：

### 9.1 运行总览能力

让 agent 读取：

- 最近 runs
- benchmark / exploratory 区分
- 模型规模
- 最终 WR / loss / wall time
- lineage

### 9.2 稳定性分析能力

让 agent 读取：

- probe 波动
- 最大 swing
- loss 下降轨迹
- 不同 run 的稳定性对比

### 9.3 frontier 能力

让 agent 知道哪些实验真正推进了边界，而不是只看单点最好值。

### 9.4 对手生态能力

让 agent 知道：

- 当前有哪些 opponent
- 它们来源于哪些 run / checkpoint
- 当前生态是否过窄或过旧

### 9.5 replay 摘要能力

不一定让 agent 直接读整盘棋 JSON，但至少应提供：

- 典型失利样本摘要
- 平均步数变化
- 黑白边分布
- 常见失败模式

### 9.6 变更历史能力

让 agent 明确知道：

- 上一轮改了什么
- 为什么改
- 结果如何
- 最终是否保留

这是构建真正研究记忆的关键。

---

## 10. v10+ 路线中必须避免的误区

### 误区 1：把 LLM 直接嵌进训练中途

这会导致：

- 高噪声驱动
- benchmark 边界变脏
- 单次 run 不再清晰可比

正确做法是：

- 在关键完成态之后触发 hook
- 让 LLM 做跨实验判断，而不是在线 steering 单个 run

### 误区 2：用 LLM 替代 benchmark

LLM 不是裁判，benchmark 仍然必须保留为外部事实标准。

### 误区 3：把 AutoML 工具继续当主路线

快筛、双轨、参数化都可以保留，但它们必须回到“agent 工具层”而不是“项目身份层”。

### 误区 4：只回归形式，不回归控制权

如果只是接了一个 LLM 调用，但真正的研究方向仍然由固定内部规则决定，那并不是真回归。

真正的回归是：

> **把跨实验研究控制权重新交给 agent。**

---

## 11. v10+ 升级 Statement

建议从 v10 开始，项目正式确立如下 statement：

> **MAG Gomoku 不再以“自动化超参回归平台”为目标叙事，而重新明确为一个 agent-driven 的 autoresearch 实验系统。训练脚本负责产出事实，benchmark 负责裁决结果，agent 负责读取放大后的实验报告并推进跨实验研究。我们保留现有实验基础设施，但将其重新定位为 agent 的工具层，而不是 AutoML 的替代实现。**

这个 statement 的价值在于，它明确回答了三个问题：

1. 我们不准备去做什么：
   不把项目主身份做成 AutoML 框架。

2. 我们准备去做什么：
   恢复 agent 作为研究主体的外层闭环。

3. 我们为什么有资格这么做：
   因为当前仓库已经有足够强的观测和分析基础设施，可以让 agent 比原版看到更多事实。

---

## 12. v10+ 希望实现价值的 Verdict

本轮的最终 verdict 如下：

1. **当前的 AutoML 倾向是实质性偏离。**
   它不是简单的实现细节变化，而是研究控制中心从外层 agent 向训练脚本内部迁移。

2. **这种偏离必须被纠正。**
   如果不纠偏，项目会逐步失去从 autoresearch 出发的根本必要性与独特性。

3. **现在正是回归的最好时机。**
   因为当前 agent 能力已经更强，而仓库内部的 tracker、analyze、checkpoint、replay、opponent 体系也已经成熟到足以支撑更强的外层研究闭环。

4. **v10 的正确任务不是继续做训练细节优化，而是重新定义架构边界。**
   重点应放在：
   - 重新定性项目方向
   - 设计 hook
   - 定义 agent report
   - 恢复 agent 的跨实验主控权

5. **v10+ 的真正目标，不是回到一个更原始的 autoresearch，而是走向一个增强版 autoresearch。**
   一个能让 agent 看到长期实验史、稳定性、frontier、lineage、对手生态与行为摘要的研究系统，反而有可能比上游更接近“真正自动研究”的方向。

一句话总结：

> **如果说 v1-v9 主要是在把系统做出来，那么 v10+ 要做的，就是把“谁在研究”这件事重新摆正。MAG Gomoku 不应继续滑向 AutoML，而应借助更强的 agent 与更丰富的实验观测层，回到并推进 autoresearch 的真正路线。**

---

## 13. 简版结论

如果只保留最重要的结论，v10+ 应该做的是：

1. 正式承认当前 AutoML 倾向是一种本质偏离
2. 重新把项目身份定义为 agent-driven autoresearch system
3. 保留 tracker / analyze / checkpoint / replay，但把它们重新定义为 agent 工具层
4. 建立 run-end / checkpoint-end / frontier-review 的 hook 机制
5. 让 agent 基于放大后的结构化报告做跨实验决策
6. 让 benchmark 继续担任裁判，而不是让内部回归逻辑取代研究闭环

这就是 v10 应确立的方向，也是 v10+ 之后应坚持的路线。

---

> 额外的计划：也许可以保留 automl 的成分作为 control group。用来对比自回归实现与有 agent 参与实现之间的差值
> 看看 agent 参与了之后，是不是真的可以提高回归速度。或者说，可以做成benchmark，来对比不同 agent 在固定实验环境下，agent 的闭环研究能力


这个想法不是在动摇 v10，而是在给 v10 增加一个极强的验证层。只要你把 AutoML 放在 control group 而不是主路线，它不但不会稀释 autoresearch，反而会让“回归 autoresearch”从理念变成可以被实证检验的路线。

- 保留 AutoML 成分作为 control group，这个价值是真实存在的。
- 它最大的意义，不是替代 autoresearch，而是验证 autoresearch 路线是否真的带来研究效率提升。
- 它还可以进一步演化为一种“闭环研究 benchmark”，用来比较不同 agent 在真实实验系统中的研究能力。
- 但它不应被表述为“绝对理论推理性能 benchmark”，更准确的说法是：
- agent-driven closed-loop research performance benchmark。

最关键的边界是：
- AutoML 是对照组，不是项目身份。
- Agent autoresearch 是主路线，AutoML control 负责给这条主路线提供证据和约束。

---

## Section 14: Copilot 执行者分析 — 基于代码事实的全面复盘与 V10 实施规划

> 身份：v1–v9 全部代码的实际编写者与执行者
> 方法：基于完整代码审计（4342 行 Python，16 次 git commit，9 个版本迭代），对 GPT 的分析进行逐项辩证评估

---

### 14.1 总体评价：GPT 的方向感知正确，但严重程度被高估

GPT 的 v10 文档提出了一个尖锐的核心论断：

> "当前的 AutoML 倾向是本质偏离，而不是轻微风格漂移。"

作为每一行代码的实际编写者，我的评价是：**方向感知正确，但"本质偏离"这个定性过重，不符合代码层面的事实**。

#### 为什么说方向感知正确？

GPT 准确识别了一个真实的结构性问题：**外层 agent 目前不在默认运行路径中**。v1–v9 的全部工作都聚焦于"训练层基础设施建设"——我们建了 TUI、tracker、checkpoint、opponent registry、analyze、sweep，但我们从未真正让一个 agent 执行过哪怕一次完整的 autoresearch 闭环。这是事实。

#### 为什么说严重程度被高估？

因为代码层面的量化分析完全不支持"本质偏离"这个结论：

**代码组成审计结果：**

| 模块 | 行数 | 角色 | 服务于 |
|------|------|------|--------|
| train.py | 1295 | Agent 唯一可编辑文件 | autoresearch |
| prepare.py | 620 | 固定评估裁判 | autoresearch |
| tracker.py | 602 | 实验历史记录 | 两者共享 |
| game.py | 542 | 棋盘引擎 (read-only) | autoresearch |
| analyze.py | 464 | 分析工具 | 两者共享 |
| play.py | 288 | 人机对弈 | 独立 |
| **sweep.py** | **268** | **批量超参扫描** | **AutoML** |
| replay.py | 208 | 回放工具 | 独立 |
| tui.py | 55 | 渲染工具 | 独立 |
| **总计** | **4342** | — | — |

唯一可以被定性为"AutoML 功能"的是 `sweep.py`（268 行），占总代码量的 **6.2%**。而这个工具在 v9 中被明确定义为"agent 的研究工具"，agent 可以用它来批量验证假设——这与 AutoML 自动搜索有根本区别。

**train.py 内部决策逻辑审计：**

| 自动决策 | 行数 | 性质 |
|----------|------|------|
| 时间预算停止 | 1 行 | 运行安全边界，不是研究决策 |
| 目标胜率停止 | 3 行 | 可选 CLI 参数，不影响 agent 循环 |
| Checkpoint 阈值触发 | 12 行 | 记录快照，不改变训练方向 |
| 对手选择 | 18 行 | CLI 传入，agent 全权决定 |
| 参数自适应 | **0 行** | **训练中无任何自适应逻辑** |

train.py 在训练过程中 **没有任何参数自适应、自动架构搜索、贝叶斯优化、population-based training 或任何 AutoML 核心特征**。所有超参数都是 agent 在运行前通过 CLI 或代码编辑设定的，运行中完全不变。

这意味着：**系统的控制权从未离开过 agent 的手——只是 agent 还没有被正式接入。**

---

### 14.2 对 GPT "本质偏离" 论断的辩证分析

GPT 的论证链条是：

1. 训练脚本内部自动化越来越多 → 
2. 研究控制权从 agent 转移到训练脚本 → 
3. 项目从 autoresearch 滑向 AutoML

**我认为第 1 步的前提就有问题。**

v1–v9 增加的"自动化"本质上是**观测基础设施**，而不是**决策自动化**：

| GPT 列举的"AutoML 倾向" | 实际性质 | 正确分类 |
|--------------------------|----------|----------|
| checkpoint 与 tracker 体系 | 实验记录工具 | agent 工具层 |
| 对手注册与混合训练 | 训练环境参数 | agent 决定何时使用 |
| 参数化与快筛思路 | CLI 参数暴露 | 降低 agent 修改代码的门槛 |
| benchmark / exploratory 分类 | 实验元数据标注 | agent 参考信息 |
| sweep.py | 批量验证工具 | agent 可选使用的工具 |

这些功能没有一个在"替代 agent 做研究判断"。它们全部是**让 agent 看得更多、做得更快**的工具。

GPT 混淆了两个概念：
- **"训练脚本功能越来越多"**（事实，但这是基础设施成熟的正常过程）
- **"研究控制权转移到训练脚本内部"**（不是事实——没有任何跨实验决策逻辑被写入训练脚本）

一个类比：给实验室配备了更好的显微镜、色谱仪、数据记录器，不等于"研究控制权从科学家转移到了仪器"。仪器越好，科学家的研究能力反而越强。

---

### 14.3 真正的问题是什么

GPT 的方向感知之所以正确，不是因为我们偏向了 AutoML，而是因为：

> **我们花了 9 个版本建造了一间设备齐全的实验室，但还没有让科学家（agent）进来工作。**

这是一个**工程顺序问题**，不是**路线偏离问题**。

回顾 v1–v9 的演进路径：

```
v1: 基础引擎 + 训练循环          ← 没有实验室
v2: checkpoint + 追踪系统        ← 开始建实验室
v3: UUID 隔离 + 断点续训         ← 实验可重现
v4: 纯文本 TUI + 平滑 WR        ← 实时观测面板
v5: NN 对手 + 对手注册           ← 训练环境扩展
v6: Benchmark profile + analyze  ← 分析工具上线
v7: 信号质量 (D4, 动态, 稳定性)  ← 训练信号可靠
v8: 模型容量 + 混合训练          ← 实验空间扩大
v9: CLI 超参 + sweep + matrix    ← 批量验证能力
```

这条路径的逻辑是清晰的：**先让单次实验可靠（v1-v4），再让实验空间丰富（v5-v8），最后让批量验证可行（v9）**。

现在，v10 的正确任务确实是"让 agent 进入实验室"。但这不是"纠偏"——这是**按计划推进到下一阶段**。

---

### 14.4 与原版 autoresearch 的精神对齐度评估

让我们对照 Karpathy autoresearch 的六个核心要素，逐项评估 MAG-Gomoku 的当前状态：

| 核心要素 | 原版 autoresearch | MAG-Gomoku 现状 | 对齐度 |
|----------|-------------------|------------------|--------|
| Agent 是研究者 | ✓ agent 读结果、改代码 | ✓ program.md 完整定义了 agent 循环 | ✅ 完全对齐 |
| Benchmark 是裁判 | ✓ 固定评估 | ✓ minimax L0-L3 + 200 局 full-eval | ✅ 完全对齐 |
| 训练脚本是实验对象 | ✓ 只修改一个文件 | ✓ 只修改 train.py (1295 行) | ✅ 完全对齐 |
| Keep/Discard 循环 | ✓ git commit/reset | ✓ program.md 明确定义 | ✅ 完全对齐 |
| 无限运行 | ✓ 不暂停 | ✓ "NEVER STOP" 写在 program.md | ✅ 完全对齐 |
| Agent 实际运行过 | ✓ 这是上游的核心 | ❌ **从未执行过一次完整闭环** | ❌ 缺失 |

结论：**MAG-Gomoku 在架构设计上与 autoresearch 完全对齐（5/6），唯一缺失的是"agent 实际运行"这一关键环节。**

这不是偏离——这是"建好了赛道但还没开赛"。

---

### 14.5 GPT 建议的逐项审查

#### 8.1 正式改写项目阶段定义 → ⚠️ 部分同意

GPT 建议"明确指出当前出现的 AutoML 倾向"。我认为这个定性不准确，不应该写入项目文档。

但我同意**明确宣告项目进入 autoresearch 闭环阶段**。我们不需要"纠偏声明"，需要的是"阶段推进声明"。

**我的替代方案：** 在 README.md 和 program.md 中增加阶段标记：
- v1-v9：Infrastructure Phase（基础设施建设期）
- v10+：Autoresearch Phase（agent 闭环研究期）

#### 8.2 定义 Hook 架构边界 → ✅ 同意，但需要极简实现

GPT 建议三个 hook：run-end、checkpoint full-eval、frontier-review。

**代码事实：** train.py 已经有 `_log_event()` 机制（第 691 行），stdout 已经输出结构化的 final summary（第 942-982 行）。agent 在运行结束后已经可以读取结果。

**真正缺失的不是 hook，而是"报告生成器"——** 一个把分散在 tracker.db 中的数据组装成 agent 可直接消费的结构化文本报告的工具。

**我的实施方案：** 在 analyze.py 中新增 `--report` 命令，输出一份面向 agent 的完整实验报告（最近 N 次 run 摘要 + 最佳 checkpoint + frontier 变化 + 稳定性概要 + 对手生态）。这比建 hook 系统更实际——agent 每次实验后运行 `analyze.py --report` 即可。

#### 8.3 定义 Agent Report 最小可用集合 → ✅ 完全同意

这是 v10 最高优先级的工作。GPT 列举的 6 项内容（runs 摘要、稳定性、frontier、best checkpoints、opponent registry、replay 统计）在 analyze.py 中**已经全部实现为独立命令**——我们只需要一个 `--report` 入口把它们组装起来。

**现有命令覆盖度：**

| GPT 要求 | 已有命令 | 状态 |
|----------|----------|------|
| 最近 N 个 run 摘要 | `--runs` | ✅ 已实现 |
| 当前 run 稳定性 | `--stability RUN_ID` | ✅ 已实现 |
| frontier 变化 | `--frontier` | ✅ 已实现 |
| best checkpoints | `--best` | ✅ 已实现 |
| opponent registry | `--opponents` | ✅ 已实现 |
| replay 统计摘要 | 无 | ⚠️ 需新增 |

5/6 已经存在。v10 的工作量比 GPT 预估的要小得多。

#### 8.4 keep/discard 主决策权回到 agent → ✅ 同意，但这本来就是设计

这个决策权**从未离开过 agent**。program.md 第 139-152 行明确定义了 keep/discard 逻辑由 agent 执行（git commit vs git reset）。train.py 内部没有任何跨实验的 keep/discard 逻辑。

GPT 可能把 `--target-win-rate` 自动停止（单次运行内的终止条件）混淆为了"跨实验研究决策"。这两者完全不同。

#### 8.5 快筛体系降级为 agent 工具 → ✅ 同意

sweep.py 从创建之日起就被设计为 agent 工具，不是项目主路线。这一点不需要"降级"，只需要在文档中更明确地标注。

#### 9.1-9.6 Agent 内部工具能力 → ✅ 同意，核心交付物

GPT 列举的 6 类能力（运行总览、稳定性、frontier、对手生态、replay 摘要、变更历史）正是 `--report` 命令需要组装的内容。

#### 额外：AutoML 作为 control group → ✅ 强烈同意

这是整份文档中最有价值的洞察之一。保留 sweep.py 作为 AutoML control group，用它的结果与 agent-driven 实验进行对比——这不仅验证了 autoresearch 路线的价值，还可以进化为"agent research benchmark"。

---

### 14.6 V10 具体实施规划

基于以上分析，V10 的工作可以分为三个优先级：

#### P0：Agent 报告系统 (analyze.py --report)

**目标：** 让 agent 每次实验后获得一份结构化的、可直接作为上下文输入的完整实验报告。

**输出格式：** 纯文本，分段落，面向 LLM 上下文窗口优化。

**包含内容：**

```
=== MAG-Gomoku Experiment Report ===

## Recent Runs (last 5)
[run_id] [status] [model] [cycles] [WR] [loss] [wall_time] [benchmark/exploratory]

## Current Best
[best checkpoint info, WR, model architecture]

## Win Rate Frontier
[monotonically improving WR records across all runs]

## Stability Summary (latest run)
[WR mean/std, loss trend, max swing]

## Opponent Registry
[alias, source, WR, description]

## Recommendations
[auto-generated: if WR plateaued → suggest architecture change; if loss diverged → suggest LR adjustment]
```

**实现方式：** 在 analyze.py 新增 `cmd_report()` 函数，复用现有 5 个 cmd_* 函数的查询逻辑，组装为单一文本输出。预计 ~120 行。

#### P1：program.md 升级 — Agent 闭环协议 v2

**目标：** 把 program.md 从 v1 的简单循环升级为利用完整工具链的研究协议。

**关键改动：**

1. 在循环中加入 `analyze.py --report` 步骤：
   ```
   # 实验前：读取报告
   uv run python src/analyze.py --report > /tmp/experiment_report.txt
   # 读取报告内容，形成假设
   # 修改 src/train.py
   # 运行实验
   # 实验后：再次读取报告，对比前后变化
   ```

2. 明确标注工具层（sweep.py、analyze.py）是 agent 可调用的研究工具

3. 增加"研究策略提示"：基于报告中的模式给出建议方向

4. 增加阶段声明：Infrastructure Phase (v1-v9) → Autoresearch Phase (v10+)

#### P2：首次 Autoresearch 闭环试运行

**目标：** 用当前系统进行第一次真正的 agent-driven 实验循环（3-5 轮），验证闭环可行性。

**方式：**
1. 以当前 train.py 作为基线
2. Agent 读取 `--report` 输出
3. Agent 提出假设并修改 train.py
4. 运行 5 分钟 benchmark（`--time-budget 300 --eval-level 0`）
5. 读取新报告，决定 keep/discard
6. 重复 3-5 轮

**价值：** 这是项目首次真正执行 autoresearch 闭环。它的意义不在于训练结果，而在于验证整个系统是否可以支撑 agent 闭环研究。

#### P3：文档与定位更新

**目标：** 更新 README 和项目文档，明确 v10 阶段定位。

**改动：**
1. README.md 增加 "Project Phases" 章节
2. README.md 增加 analyze.py --report 文档
3. 不写"纠偏声明"——写"阶段推进声明"

---

### 14.7 关于不做什么的清单

以下是 GPT 文档中提到但我认为 **v10 不应该做的事情**：

| 不做的事 | 原因 |
|----------|------|
| 建立正式的 hook/event 系统 | 过度工程。`--report` 命令足以替代，agent 主动拉取比被动推送更可靠 |
| 写"纠偏声明"或"AutoML 倾向承认" | 定性不准确。我们没有偏离，只是还没到 agent 上场的阶段 |
| 建立变更历史追踪系统 | Git log 本身就是变更历史。agent 可以直接读 git log |
| replay 深度分析（失利模式、常见失败） | 优先级低，agent 初期不需要棋局级别的行为分析 |
| 把 sweep.py 重新定性或重构 | 它已经是工具层，不需要改 |
| 建立 "retention/deletion mechanism" | v9 已经说过——过度形式化 |

---

### 14.8 对 GPT "四阶段路线规划" 的修正

GPT 提出了 A → B → C → D 四个阶段。我的修正如下：

| GPT 阶段 | GPT 定义 | 我的修正 |
|----------|----------|----------|
| A：纠偏与重新定性 | 承认 AutoML 偏离 | → **阶段推进声明**（不是纠偏） |
| B：建立 Hook + Report | 事件触发 + 报告系统 | → **只建 Report**（`--report` 命令，不建 hook） |
| C：建立 Agent Research Loop | agent 接管跨实验决策 | → **首次闭环试运行**（验证而非"建立"） |
| D：增强版 autoresearch | 超越上游 | → 同意方向，但这是 v11+ 的事 |

关键区别：GPT 把这当作一个大型架构重构项目，我认为实际工作量非常小——因为 **90% 的基础设施已经存在**。

---

### 14.9 代码量预估

| 任务 | 预估行数 | 涉及文件 |
|------|----------|----------|
| `cmd_report()` 函数 | ~120 行 | src/analyze.py |
| program.md v2 | ~40 行改动 | docs/program.md |
| README 阶段文档 | ~30 行改动 | README.md |
| 试运行记录 | 0 行代码 | 运行结果 |
| **总计** | **~190 行** | 3 个文件 |

V9 是 280 行。V10 预计更小——因为它的核心不是写代码，而是**让已有代码发挥设计时的作用**。

---

### 14.10 Final Verdict

1. **GPT 的方向判断正确**：agent 确实需要被正式接入系统。这是 v10 的核心任务。

2. **GPT 的严重程度评估不准确**："本质偏离"这个定性不符合代码事实。4342 行代码中只有 268 行（6.2%）服务于 AutoML，且被设计为 agent 工具。训练脚本内部没有任何跨实验决策自动化。

3. **正确的叙事不是"纠偏"而是"推进"**：v1-v9 是基础设施建设期，v10 是 agent 上场期。这是正常的工程序列，不是路线偏离后的修正。

4. **v10 的实际工作量很小（~190 行）**：因为 analyze.py 已经具备了 report 所需的全部查询能力（5/6 已实现），只需要一个组装入口。

5. **AutoML control group 是全文档最有价值的洞察**：保留 sweep.py 作为对照组，让 autoresearch 路线可以被实证检验——这比任何"纠偏声明"都更有说服力。

6. **v10 的成功标志不是代码量，而是第一次真正的 agent 闭环运行**。如果我们能完成 3-5 轮 agent-driven 实验循环（读报告→形成假设→改代码→跑实验→keep/discard），那 v10 就已经完成了它最核心的使命。

一句话总结：

> **GPT 看到了正确的下一步（接入 agent），但用了错误的叙事框架（纠偏）来描述它。v10 不是一次方向修正——它是 9 个版本基础设施建设后的自然毕业典礼。实验室已经建好，是时候让科学家进来工作了。**