# Update v11 Analysis - 将 Gomoku 同时作为真实目标与框架验证场的桥接版本设计

> 日期：2026-04-11  
> 范围：基于 v10、further analysis 与 Pareto Frontier 讨论，正式定义 v11 的角色：既要继续推进 Code Bullet 精神下的高水准 Gomoku 自我研究与可视化产出，也要让 Gomoku 成为 Local Exploration Layer、Pareto Layer、Autoresearch Layer 协同工作的第一个真实验证场。本文件同时明确 v11 的 in-scope、out-of-scope，以及这些工作如何推动整个框架向前演进。

> 背景文件: update-pareto-frontier.md [重要，务必阅读]
> 背景文件：update-v10-analysis-further.md [重要，请回顾]

---

## 1. 执行摘要

到当前阶段，我们已经形成两个看似分叉、但实际上必须同时保留的目标：

1. **Gomoku 目标**
   继续沿着 Code Bullet 精神，把系统推向一个较高水准的自我研究型五子棋 AI，并保留 replay、checkpoint、成长记录等可展示材料。

2. **框架目标**
   让当前的实验体系不只是“训练出一个模型”，而是积累真实研究事实、真实边界数据、真实 Pareto 结构，为后续更一般化的 agent-driven research framework 演进提供证据。

这两个目标不能二选一。

如果只保留第一个目标，项目会重新收缩为单一应用项目。  
如果只保留第二个目标，Gomoku 会退化成抽象框架的测试木偶，失去 Code Bullet 精神与项目原初张力。

因此，v11 的任务不是在两者之间选边，而是正式承认：

> **Gomoku 在 v11 中既是目标本身，也是框架验证场。**

但这里必须把边界说清楚：

v11 不是要把项目直接扩展成通用平台，也不是要立刻实现 webhook / 外汇等泛化场景的完整能力。  
v11 的正确定位应当是：

> **用 Gomoku 这个真实、可运行、可追踪、可回放的实验域，把 autoresearch、Local Exploration Layer、Pareto 分析、红线纪律这些关键构件先在一个具体场景中验证扎实。**

换句话说，v11 是一个**桥接版本**：

- 向下连接 v10：把 autoresearch loop activation 变成真正可持续运转的研究闭环
- 向上连接未来版本：为更一般化的 framework 演进提供足够真实的实验材料和结构事实

本轮分析的核心结论如下：

1. **“Gomoku 已退化成 Pareto Frontier 验证工具”这个判断只说对了一半。**
   更准确地说，它正在从“单一目标项目”升级成“真实目标 + 框架验证场”的双重角色。

2. **v11 必须明确保留 Gomoku 作为真实作品目标。**
   也就是继续追求较高水准模型、成长过程、replay 材料、可视化产出。

3. **v11 也必须明确让 Gomoku 承担框架验证任务。**
   包括：Pareto Frontier、threshold 分析、局部探索结果表达、agent 消费局部边界信息等。

4. **v11 的 in-scope 必须严格克制。**
   它应只做能直接在 Gomoku 场景中产生真实证据的工作，不提前铺开过多跨领域实现。

5. **v11 的 out-of-scope 也必须明确。**
   泛化框架、通用平台、多领域产品化、人类决策面板等内容，应该继续放在后续版本。

---

## 2. 为什么需要一个 v11 桥接版本

### 2.1 因为 v10 只解决了“agent 回到实验室”

按当前设计，v10 的主要任务是：

- 激活 autoresearch 闭环
- 建立 `analyze.py --report`
- 升级 `program.md`
- 让 agent 真正开始消费实验历史并驱动下一轮研究

这是必要的，但它并不足以回答下面这些更高阶问题：

1. agent 是否真的能利用 Local Exploration 的局部结论？
2. Pareto Frontier 是否真的能帮助 agent 做更好的研究判断？
3. Gomoku 作为一个真实目标，是否会因为框架化而被稀释？
4. 我们能否同时产出“更强的 Gomoku AI”和“更成熟的研究结构证据”？

这些问题就是 v11 的任务。

### 2.2 因为我们现在已经进入双目标阶段

当前项目不再只有“把 WR 做高”这一个目标。

它至少同时承担：

1. **作品目标**
   做出一个真正有看点的、具有自我研究过程的 Gomoku AI

2. **方法目标**
   验证 agent-driven research system 在 Local Exploration 与 Pareto 分析加持下，是否真的更强、更清晰、更可扩展

一旦项目进入双目标阶段，就需要一个专门版本把两条线收束到同一个执行面里。

v11 就是这个版本。

---

## 3. 对当前局面的重新定性

### 3.1 Gomoku 没有“失去意义”，而是获得了第二层意义

你说“现在的五子棋游戏已经退化成为了帕累托前沿的验证工具”，这句话有现实依据，因为当前很多讨论已经不是单纯围绕“五子棋本身怎么更强”，而是在围绕：

- 如何表达 trade-off
- 如何让局部探索层产出边界信息
- 如何让 agent 消费这些边界信息
- 如何为框架演进积累真实案例

但如果把 Gomoku 只定性成“验证工具”，那会产生一个严重副作用：

> 它会把 Gomoku 从真实目标降级成纯测试样本。

这会削弱项目原本最有生命力的部分：

- Code Bullet 风格的自我训练叙事
- 模型成长过程的可观看性
- replay / montage / checkpoints 的展示价值
- “让 AI 自己变强”的核心吸引力

因此更准确的定性应该是：

> **Gomoku 在 v11 中不只是 Pareto 验证工具，而是第一个既能产生真实作品价值、又能产生真实框架证据的验证域。**

### 3.2 v11 应把这两层意义同时写进版本定义

所以，v11 必须同时承认：

1. Gomoku 是真正要做强、做漂亮、做出材料的目标对象
2. Gomoku 也是用来验证 Pareto Frontier、Local Exploration Layer、Autoresearch Loop 是否能真正协同工作的第一个真实实验场

这两者不是冲突关系，而是彼此增益：

- 因为它是真实目标，所以框架验证有真实价值
- 因为它是验证场，所以 Gomoku 实验不再只是“分数变化”，而会变成下一阶段框架设计的证据库

---

## 4. v11 的总目标

v11 应明确追求下面三个总目标：

### 4.1 目标一：继续追求高水准 Gomoku 模型

这条线不能丢。

v11 仍应继续追求：

1. 更高质量的 benchmark WR
2. 更强、更稳、更有说服力的模型推进路径
3. 更有观赏性和可展示性的 replay / checkpoint / montage 材料
4. 更接近 Code Bullet 精神的“AI 自己研究自己变强”的叙事张力

### 4.2 目标二：让实验结果具备 Pareto 可解释性

这条线对应框架验证。

v11 需要回答：

1. 不同容量与不同超参是否在 WR、参数量、时间、推进速度之间形成边界点？
2. threshold-based Pareto 是否能更好表达“谁推进得更快、更省”？
3. Local Exploration 结果是否能被压缩成 agent 可消费的边界结论？

### 4.3 目标三：为未来框架演进积累真实研究材料

v11 的所有分析和工具，不应只为当前版本服务，而应主动产出：

1. 真实的 frontier 数据
2. 真实的 Pareto 结构
3. 真实的局部探索结果
4. 真实的 agent 决策样本
5. 真实的 replay / run history / stage promotion 记录

这些内容将直接成为后续版本继续演进 framework 的材料基础。

---

## 5. v11 的 in-scope 工作

v11 的 in-scope 必须遵循一个原则：

> **只做那些能够直接在 Gomoku 场景里产生真实研究证据，并同时推动 Gomoku 目标与框架目标前进的工作。**

### 5.1 in-scope A：Gomoku 内的 Pareto Frontier 分析正式化

这是 v11 的核心之一。

应优先正式化的分析包括：

1. **Benchmark Run Pareto**
   在固定 benchmark 条件下，比较：
   - WR vs num_params
   - WR vs wall_time_s
   - WR vs total_cycles / total_games / total_steps

2. **Threshold Pareto**
   比较不同配置达到固定 WR 阈值的代价：
   - 到达 0.65 / 0.80 / 0.90 WR 的最早 checkpoint
   - train_elapsed_s vs num_params
   - cycle vs num_params

3. **Grouped Sweep Pareto**
   在 `sweep_tag` 与 `seed` 维度上汇总：
   - mean WR
   - WR std
   - throughput
   - params

这些分析直接服务两条线：

- 对 Gomoku：帮助找更强、更省、更快的推进路径
- 对框架：验证 Pareto layer 是否真的有实用价值

### 5.2 in-scope B：Gomoku 内的 Local Exploration 结果表达规范

v11 不需要立刻把 Local Exploration 抽象成通用平台，但至少要在 Gomoku 场景里把它表达清楚。

建议内容包括：

1. 对 sweep 结果进行结构化汇总
2. 让 agent 能看到哪些点是局部 Pareto 前沿
3. 让 agent 区分：
   - 局部最优点
   - 被支配点
   - 值得继续扩展的边界点

这一步是 future hybrid mode 的必要前置，但仍然在 Gomoku 内完成。

### 5.3 in-scope C：保持 Code Bullet 精神下的作品产出

v11 不能只做分析系统，必须继续产出“作品层”的结果。

建议包括：

1. 保持 checkpoint 归档与命名清晰
2. 保持 replay / recording 可追踪
3. 形成一条更清晰的成长叙事：
   - 从哪个基线出发
   - 经过哪些关键跃迁
   - 哪些实验形成了关键前沿点

也就是说，v11 仍然应服务“一个可以看、可以讲、可以展示的 Gomoku 自我研究故事”。

### 5.4 in-scope D：agent 消费 Pareto 结果的最小闭环验证

v11 不一定要把整个 hybrid mode 全部做完，但至少要验证下面这件事：

> **agent 是否能基于 Pareto frontier 和 grouped sweep 结果，做出比单看 WR 更成熟的研究判断。**

这一步非常关键，因为它直接决定 Pareto layer 不是漂亮分析，而是真正有用。

### 5.5 in-scope E：红线纪律内化到 v11 的验收逻辑中

v11 必须显式继承三条绝对红线：

1. 单一裁决纪律不能被稀释
2. agent 主控权不能被慢慢稀释
3. 人类选择不能过早上位

这意味着 v11 的所有新增能力都必须先回答：

1. 有没有削弱 WR 作为 Gomoku benchmark truth 的地位？
2. 有没有让 Local Exploration layer 替代 agent 做高层判断？
3. 有没有让人类过早根据 Pareto 点直接主导方向？

---

## 6. v11 的 out-of-scope 工作

为了让 v11 保持清晰，下面这些工作应明确列为 out-of-scope。

### 6.1 out-of-scope A：跨领域实现

以下内容不应在 v11 中真正落地：

1. webhook 数据分析系统的正式接入
2. 外汇头寸策略系统的正式实现
3. 跨场景 benchmark harness 的产品化抽象

原因：

v11 的任务是用 Gomoku 这个单一真实域，先把框架方向验证扎实，而不是立刻铺到多领域。

### 6.2 out-of-scope B：通用平台化抽象

以下内容也应延后：

1. 通用 Local Exploration API
2. 通用 Benchmark Layer SDK
3. 通用 Human Decision UI
4. 多项目共享的 orchestrator / scheduler

原因：

现在还缺真实证据来证明这些抽象边界应该怎么切。

### 6.3 out-of-scope C：人类决策平面的产品化

虽然 Pareto 点最终会服务人类选择，但 v11 不应进入：

1. 完整可视化决策面板
2. 高交互策略选点界面
3. 面向业务用户的产品层设计

原因：

这会太早把系统中心从“研究闭环”拉向“人工决策平台”。

### 6.4 out-of-scope D：让 Gomoku 彻底退化成框架测试样本

这是一条特别重要的 out-of-scope：

> v11 不允许把 Gomoku 降级成“只是用来验证 Pareto layer 是否工作”的空壳案例。

原因：

一旦这么做，项目会同时失去：

1. Code Bullet 风格的作品生命力
2. 真实模型推进带来的研究难度
3. replay / checkpoint / montage 的展示价值
4. 作为首个真实验证域的说服力

---

## 7. 这些工作如何推动框架向前演进

v11 的价值不只在于把某几个命令做出来，而在于它将为后续版本沉淀一批不可替代的研究事实。

### 7.1 事实一：Pareto 在真实场景里是否真的有决策价值

如果 v11 完成后，我们看到：

- agent 能利用 Pareto frontier 形成更成熟的选择
- grouped sweep 能被压缩成局部边界结论
- threshold Pareto 能帮助判断谁推进更快

那就意味着 Pareto layer 在 future framework 中是有资格被正式保留的。

### 7.2 事实二：Local Exploration Layer 是否真正有用

如果 v11 显示：

- 局部搜索结果能显著帮助 agent
- 但又没有取代 agent 的高层判断

那就说明 Local Exploration Layer 在未来版本里具有稳定生态位。

### 7.3 事实三：Gomoku 是否足够作为首个真实验证域

如果 v11 证明：

- Gomoku 可以同时提供作品价值和框架验证价值
- WR、replay、checkpoint、Pareto、局部探索这几层可以协同运转

那就说明以后扩展到 webhook / 外汇等场景是有根的，而不是空想。

### 7.4 事实四：红线纪律是否能在功能增加时仍然守住

v11 会成为第一轮真正的压力测试：

- 功能在增加
- 解释层在增加
- 局部探索在增加

如果在这种情况下仍然能守住：

1. 单一裁决纪律
2. agent 主控权
3. 人类不过早上位

那说明框架演进方向是可持续的。

---

## 8. v11 的建议版本声明

建议将 v11 的版本声明定义为：

> **v11 是 MAG Gomoku 从“autresearch loop activation”走向“Pareto-aware research system”的桥接版本。它不放弃 Gomoku 作为真实目标，也不把 Gomoku 降级成纯测试样本；相反，它把 Gomoku 作为第一个真实验证域，用来同时推进高水准模型、可展示 replay 材料，以及 Local Exploration、Pareto Frontier、agent 决策消费这些框架能力的现实验证。**

---

## 9. 最终 Verdict

本轮的最终 verdict 如下：

1. **v11 不应把 Gomoku 视为被动验证工具，而应把它视为“真实目标 + 验证场”的双重对象。**

2. **v11 必须同时服务两个目标：**
   - 做出更强、更有看点的 Gomoku AI 及其 replay 材料
   - 为未来框架演进积累真实 Pareto、真实局部探索、真实 agent 决策样本

3. **v11 的 in-scope 应严格限制在 Gomoku 域内。**
   所有工作都应优先在这个单一真实域内形成扎实证据，而不是过早跨域扩张。

4. **v11 的 out-of-scope 也必须明确。**
   通用平台化、跨领域产品化、人类决策界面化，都不应在这个版本过早展开。

5. **v11 的真正历史任务，不是把框架做完，而是把框架“证明到足够值得继续演进”。**

一句话总结：

> **v11 应把 Gomoku 从“单一训练对象”提升为“第一个真实验证域”：一方面继续追求 Code Bullet 精神下更强、更好看的五子棋自我研究成果，另一方面让每一次实验都成为框架演进的研究材料和结构事实。只有这样，项目才会既保留作品生命力，又真正推动 framework 向前生长。**