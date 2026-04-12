# Update Pareto Frontier Analysis - 从单一分数判断走向 Pareto 驱动的演进分析

> 日期：2026-04-11  
> 范围：基于我们最新的讨论，重新定义本项目中 Pareto Frontier 的角色、价值和适用范围；同时对原先被称为 AutoML 的组件重新命名，并把 webhook 数据分析与外汇头寸策略两个案例纳入同一套分析框架，用来代表项目从 MAG Gomoku 出发、逐步走向更一般化 benchmark-constrained research system 的最新推演。

---

## 1. 执行摘要

这份文档当前最需要被强调的，不是 Pareto Frontier 本身有多有用，而是：

> **Pareto Frontier 只有在三条绝对红线不被突破的前提下，才有资格进入项目的主分析层。**

这三条红线是：

1. **单一裁决纪律不能被稀释**
2. **agent 主控权不能被慢慢稀释**
3. **人类选择不能过早上位**

它们不是普通 warning，而是未来所有更新都必须优先服从的结构性红线。

在这个前提下，这份文档的核心结论有五条：

1. **Pareto Frontier 不应再被理解为只服务 Gomoku 训练的附属分析。**
   它正在成为整个项目演进方向中的关键解释层。

2. **我们应当放弃 `AutoML` 这个名字。**
   这个名字在项目还只聚焦模型训练时尚可勉强成立，但在我们已经讨论 webhook 系统优化、外汇头寸策略探索这类更广义问题之后，它已经不再准确。

3. **新的名称建议是：`Local Exploration Layer`。**
   中文可以称为：**局部探索层**。
   这个名字保留了该组件最重要的能力内核：
   - 在受限空间内做局部搜索
   - 形成结构化局部结论
   - 作为上层 autoresearch 决策的证据来源
   同时避免了 `AutoML` 这个名称带来的机器学习语义限制。

4. **Pareto Frontier 是把“真理裁决”和“现实 trade-off”放在同一张图上的关键层。**
   在 Gomoku 里，它围绕 WR 展开；
   在 webhook 系统里，它围绕 correctness + throughput 展开；
   在外汇头寸策略里，它围绕 liquidity safety + cost/risk trade-off 展开。

5. **Pareto Frontier 的地位只能是“真理之后的解释层”，不能升级成“替代真理的新裁判”。**

因此，Pareto Frontier 不只是一个分析技巧，而是在我们当前设想的系统里承担下面这个角色：

> **把基准真理、局部探索结果、工程成本和人类选择空间，组织成可解释、可比较、可决策的边界结构；但前提永远是，真理层先于 Pareto，agent 主体先于工具层，研究闭环先于人类偏好。**

### 1.1 最高优先级提醒

为了避免整份文档被误读，这里先明确一条总原则：

> **任何未来更新，只要削弱了 domain truth 的最终裁决、削弱了 agent 的研究主体地位、或让人类过早接管研究循环，那么无论 Pareto 分析多漂亮、局部探索多高效、案例多丰富，都应被视为严重偏离。**

---

## 2. 为什么这份文档需要被全面翻新

原始版本的分析，主要基于 MAG Gomoku 当前语境展开：

- WR 是唯一质量真理
- 模型参数量、wall time、cycles 等是围绕 WR 的解释轴
- Pareto Frontier 用于增强 agent 对“以什么代价达到这个 WR”的理解

这个判断本身仍然成立，但它已经不够覆盖我们最新讨论得到的更大图景。

因为现在我们已经明确看到：

1. 这套框架可能不只适用于模型训练
2. 它可以扩展到 webhook 数据处理系统这类工程代码优化问题
3. 它可以扩展到外汇头寸配置这类策略研究问题
4. 它未来更像一个 **benchmark-constrained, agent-driven research framework**

一旦项目进入这个阶段，Pareto Frontier 的地位就会变化。

它不再只是：

- WR 的副报告

而是逐渐变成：

- 不同领域 benchmark 真理之上的 trade-off 表达层

所以，这份文档必须从“只讨论 WR”的局部分析，升级成“讨论整个项目如何围绕 Pareto 形成新的演进方向”的总分析。

---

## 3. 重新命名：从 AutoML 到 Local Exploration Layer

### 3.1 为什么必须改名

`AutoML` 这个名字最大的问题，不是它完全错误，而是它已经越来越局限。

在当前项目早期阶段，它主要对应：

- 调超参
- 看 WR
- 比较模型容量
- 做 sweep 和局部搜索

在这个阶段，它还能勉强成立。

但一旦我们讨论的对象扩展到：

- webhook 处理代码的重构与实现路线比较
- TypeScript / Python / 不同包 / 不同实现架构的系统级探索
- 外汇头寸策略表、交易阈值、流动性安全线、风险预算分配

那么 `AutoML` 这个名字就明显不准确了。

因为这些问题不再是 machine learning。

### 3.2 为什么 `Local Exploration Layer` 更准确

新的名字建议：

> **Local Exploration Layer**

中文建议：

> **局部探索层**

这个名字的优点有四个：

1. 不绑定机器学习语义
2. 保留“在受限空间内自动探索”的能力核心
3. 可以同时覆盖：
   - 超参探索
   - 参数扫描
   - 局部实现对比
   - 局部策略变体验证
4. 很自然地形成与 autoresearch 的分工：
   - autoresearch：做高层判断
   - Local Exploration Layer：做低层局部勘探

### 3.3 新的角色定义

在最新构型下，角色应重新定义为：

1. **Autoresearch Layer**
   - 理解问题
   - 读取报告
   - 提出假设
   - 修改代码或策略表
   - 决定是否调用局部探索

2. **Local Exploration Layer**
   - 在受限空间内做局部搜索
   - 进行多 seed / 多阈值 / 多局部变体比较
   - 输出局部 Pareto 边界或局部结论摘要

3. **Benchmark Layer**
   - 提供外部事实裁决
   - 给出 domain-specific truth

4. **Pareto Layer**
   - 把 benchmark truth 与现实 trade-off 显式组织成边界图景
   - 为 agent 和人类提供不同粒度的决策视图

---

## 4. Pareto Frontier 在最新项目演进中的新位置

### 4.1 它不是新的真理，而是“真理外的决策平面”

无论在哪个场景里，Pareto Frontier 都不应被理解成“替代 benchmark 真理”。

更准确的说法是：

> **benchmark 给出真假与优劣的核心事实，Pareto 给出在满足这一事实前提下，成本、效率、复杂度与风险的边界结构。**

因此，Pareto Frontier 的位置不是在 benchmark 之前，也不是在 benchmark 之上，而是在 benchmark 之后。

### 4.2 它在整个系统里的作用

Pareto Frontier 解决的问题是：

1. 不让系统只输出一个单一分数
2. 不让系统只能做“单轴冠军比较”
3. 不让 agent 或人类只能看到“谁最高”，而看不到“谁更省”“谁更快”“谁更稳”“谁更容易部署”

所以它承担的是：

> **从单点最优，转向边界最优。**

这是项目下一阶段非常关键的演进方向。

---

## 5. 与当前代码事实的关系：这个方向不是空想

虽然项目未来可能泛化到更多场景，但从当前 MAG Gomoku 仓库来看，Pareto 分析依然有非常扎实的现实基础。

### 5.1 当前已经具备 run 级成本与结果指标

当前 tracker 已能记录：

- `final_win_rate`
- `num_params`
- `wall_time_s`
- `total_cycles`
- `total_games`
- `total_steps`
- `learning_rate`
- `train_steps_per_cycle`
- `replay_buffer_size`
- `seed`
- `sweep_tag`

这意味着第一版 Pareto 分析在当前仓库中完全可做。

### 5.2 checkpoint 数据天然支持 threshold Pareto

当前 checkpoint 记录还有：

- `win_rate`
- `cycle`
- `step`
- `train_elapsed_s`
- `num_params`

这让我们可以分析：

- 某个配置达到 65%、80%、90% WR 的最早时间
- 不同容量配置在相同阈值下的推进速度
- 相同阈值下哪些点构成真实前沿

### 5.3 analyze.py 已经有自然前身

当前 `src/analyze.py` 已经有：

1. 单轴 `frontier`
2. `matrix` 聚合
3. `compare`
4. `stability`

Pareto Frontier 不需要推翻这些工具，而是自然生长于它们之上。

也就是说：

> 当前仓库已经为 Pareto 分析搭好了数据地基，只是还没有把“边界结构”正式提炼出来。

---

## 6. 在 Gomoku 训练场景中，Pareto Frontier 仍然成立

虽然项目语义在扩展，但 MAG Gomoku 仍然是当前最现实的主战场。

在这个场景中，Pareto 应继续这样定义：

### 6.1 benchmark 真理

- WR 仍然是质量真理
- 对固定 opponent、固定 budget、固定评估条件下的 WR 负责最终裁决

这也是 Gomoku 场景中的**绝对试验真理指标**：

> **在固定 benchmark 条件下得到的 WR，是 Pareto 分析之前、之上、且不可被替代的主裁判。**

### 6.2 Pareto 解释轴

围绕 WR，最自然的解释轴包括：

1. `num_params`
2. `wall_time_s`
3. `total_cycles`
4. `train_elapsed_s`
5. `games/s` 或近似 throughput

### 6.3 最合理的 Pareto 问题

在 Gomoku 里，Pareto 最值得回答的问题不是：

- 谁 WR 最高？

而是：

1. 谁用更小模型达到接近 WR？
2. 谁更快达到 80% / 90% WR？
3. 谁的推进路径更稳而不是偶然冲高？
4. 哪些配置只是把代价堆高，却没有形成新的边界点？

因此，Pareto 在 Gomoku 里仍然非常有价值，只是它已不再是项目唯一的解释场景。

### 6.4 Gomoku 场景对三条红线的触碰与试探

在 Gomoku 场景里，三条红线的试探方式最清楚，也最容易被忽略：

1. **对红线一的试探**
   如果未来把 `num_params`、`wall_time_s`、`games/s` 与 WR 放成完全并列的目标集合，而不再承认固定 benchmark WR 是唯一质量裁判，那么就属于直接踩线。

2. **对红线二的试探**
   如果局部探索层开始自动判断“哪条研究路线最值得继续”，而 agent 只是读取矩阵摘要并被动接受，那么 agent 主控权会被无声稀释。

3. **对红线三的试探**
   如果人类因为看到一条更省参数或更省时间的 Pareto 点，就绕过 benchmark 保留逻辑，直接强行选点并改变路线，那么系统会从研究闭环退回人工偏好驱动。

因此，Gomoku 场景虽然最接近当前仓库，但也恰恰最适合作为检验红线意识是否被守住的第一块试金石。

---

## 7. 案例思考一：webhook 数据分析系统

### 7.1 这个场景为什么重要

我们讨论过一个“旧的数据分析系统”场景：

- 系统消费 EDM 的 webhook 数据
- 目标是在一定时间内尽可能安全地消费更多回调
- 最终稳定产出一份固定 JSON 报告
- 关注点主要是吞吐量，以及在吞吐量之外的正确性与稳定性

这个场景很重要，因为它说明：

> 这套框架可以从“训练模型”扩展到“优化真实工程代码”。

### 7.2 在这个场景里，benchmark 真理怎么定义

这里不能再简单说 WR 是唯一真理。

在 webhook 场景里，更合理的真理结构是：

1. **硬约束真理**
   - JSON 输出正确
   - 数据不丢失
   - 逻辑安全
   - 消费过程稳定

2. **主优化目标**
   - 吞吐量

因此，这里的 benchmark 真理不是一个分数，而是一组门槛加一个主性能目标。

更明确地说，webhook 场景中的**绝对试验真理指标**应定义为：

1. JSON 输出正确
2. 数据不丢失、不重复破坏语义
3. 安全与稳定性门槛通过
4. 在通过上述硬门槛后，再比较吞吐量

也就是说：

> **throughput 不是独立真理，而是通过 correctness / safety gate 之后才有资格进入 Pareto 平面的主优化轴。**

### 7.3 Pareto Frontier 在这里怎么定义

一旦候选实现都满足正确性和安全性门槛，就可以把它们投射到 Pareto 平面上，例如：

1. 吞吐量 vs 内存占用
2. 吞吐量 vs P95 延迟
3. 吞吐量 vs CPU 成本
4. 吞吐量 vs 改动规模 / 实现复杂度

这时 human decision maker 看到的就不是：

- 谁最快

而是：

- 哪些实现点构成不同工程 trade-off 的有效边界

### 7.4 Local Exploration Layer 在这个场景里做什么

在 webhook 场景下，局部探索层不必局限于“调参数”，它可以做：

1. 并发参数扫描
2. buffer / batch / flush 策略试验
3. 局部实现变体验证
4. 某一条重构路线附近的局部搜索

而 autoresearch layer 负责：

1. 决定是否切换技术路线
2. 决定是否用 Python 替换某块逻辑
3. 决定是否用另一种包或另一种实现架构
4. 消费 Pareto 边界结果后，继续推进或回退

这个案例说明：

> Pareto Frontier 能把“真实工程实现路线探索”变成可审查、可比较、可决策的边界问题。

### 7.5 webhook 场景对三条红线的触碰与试探

这个场景比 Gomoku 更容易真实踩线，因为它天然会诱惑系统追求“更快”。

1. **对红线一的试探**
   如果系统开始用“吞吐量更高但偶尔出错也能接受”的思路替代 correctness / safety gate，那就是直接破坏真理秩序。

2. **对红线二的试探**
   如果局部探索层通过一组基准测试就替 agent 决定“应不应该切语言、切框架、切技术路线”，那高层研究判断就已经内移了。

3. **对红线三的试探**
   如果人类因为看到一条“吞吐量很漂亮的 Pareto 点”就过早指定技术路线，而 agent 还没有完成更广泛的结构探索，那么系统就会退化成可视化性能选型工具。

因此，webhook 场景最大的纪律要求是：

> **先守 correctness / safety，再谈 throughput trade-off；先让 agent 研究，再让人类在合格边界上选点。**

---

## 8. 案例思考二：外汇头寸策略与虚拟自动交易

### 8.1 这个场景为什么关键

我们进一步讨论过一个更接近企业决策的问题：

- 外贸企业需要动态管理不同货币头寸
- 既不能在某一时刻无钱可用
- 又希望尽可能减少汇率波动带来的损失
- 可以通过沙箱交易 API 构建虚拟自动交易策略
- 希望让 agent 自动探索策略，并把结果投影到 Pareto 平面上供人类决策

这个场景的意义在于：

> 它把项目进一步从“代码优化”推广到“策略研究”。

### 8.2 在这个场景里，benchmark 真理怎么定义

在外汇头寸策略场景中，最合理的 truth structure 应该是：

1. **硬约束**
   - 不允许出现流动性断裂
   - 不允许违反风控或合规边界
   - 不允许策略执行逻辑不可靠

2. **主业务目标**
   - 在满足流动性与安全前提下，降低风险暴露与汇兑损失

因此，这里的真理也不是单一分数，而是：

- 先过门槛
- 再看 trade-off

更明确地说，外汇头寸策略场景中的**绝对试验真理指标**应定义为：

1. 不出现流动性断裂
2. 不违反合规与风险边界
3. 不允许执行逻辑不可靠
4. 只有在上述硬门槛下，才比较风险暴露、汇兑损失、交易成本与资金占用

### 8.3 Pareto Frontier 在这里为什么尤其自然

外汇头寸问题天生就是 Pareto 问题。

因为人类真正关心的是：

1. 更高美元持有比例，更安全，但占用更大
2. 更低持有比例，更省，但风险更高
3. 更积极换汇，可能降低某类暴露，但提高交易成本
4. 更保守策略，可能牺牲收益但换来流动性安全边际

这些问题没有唯一单点答案。

所以最终系统最有价值的输出，不应只是：

- 这是最佳策略

而应是：

- 这是当前条件下的一条 Pareto 前沿
- 你作为人类决策者，可以根据现金流压力、风险偏好、当下市场状态选择合适的点

### 8.4 这个案例对项目演进方向的启发

这个案例说明，项目未来的真正抽象，也许不再是：

- train model and compare WR

而更接近：

> 在 benchmark 和硬约束存在的前提下，让 agent 主动探索实现或策略空间，再通过 Pareto Frontier 把结果组织成可供人类选择的 trade-off 面。

这就是一个比传统 ML 训练框架更一般化、也更有现实业务价值的方向。

### 8.5 外汇策略场景对三条红线的触碰与试探

这是三个场景里最危险的一个，因为人类业务偏好天然很强。

1. **对红线一的试探**
   如果系统把“某条 Pareto 点更省成本”视为可以压过 liquidity safety / compliance 的理由，那么就已经彻底踩线。

2. **对红线二的试探**
   如果局部探索层开始根据回测结果自动固定主要策略哲学，而 agent 不再主动研究不同风险结构与再平衡逻辑，那么 agent 的研究主体地位会迅速被掏空。

3. **对红线三的试探**
   如果人类过早频繁基于当下风险偏好直接挑选策略点、反复改动方向，而 agent 与 benchmark 还没完成系统探索，那么整套系统会退化成“有 agent 加持的决策面板”。

因此，外汇策略场景里最需要被反复提醒的是：

> **人类可以最终选点，但不能提前主导研究闭环；Pareto 可以展示边界，但不能凌驾于流动性安全和合规真理之上。**

---

## 9. 这对整个项目演进意味着什么

### 9.1 项目正在从“模型训练平台”走向更一般化框架

如果只看最早阶段，项目像一个 Gomoku 训练实验系统。

但按最新讨论推演，它正在逐步走向：

1. benchmark-constrained research system
2. agent-driven code / strategy evolution system
3. Pareto-guided decision support system

这个演化方向是连贯的。

因为在所有场景里，真正稳定不变的抽象不是“训练神经网络”，而是：

1. 有一个外部 truth / benchmark / hard constraints
2. 有一个 agent 在外层提出假设和修改方案
3. 有一个局部探索层在受限空间内做勘探
4. 有一个 Pareto 层把 trade-off 显式化
5. 最终由 agent 或人类做下一轮决策

### 9.2 Pareto Frontier 是这次泛化里的关键桥梁

如果没有 Pareto Frontier，系统很容易再次退回：

- 只看单一分数
- 只选唯一冠军
- 只做单轴比较

一旦这样，项目对现实问题的表达能力就会急剧下降。

Pareto Frontier 之所以关键，是因为它让系统可以从“求一个最优值”转向“呈现一条边界结构”。

而现实世界中，很多真正重要的问题都是边界问题，而不是单点问题。

---

## 10. 当前最需要守住的边界

这一章的所有表述，都应被理解为高优先级约束，而不是普通设计建议。

### 边界 1：Pareto 不能替代 domain truth

无论在哪个场景里，Pareto 都不是新的真理标准。

在 Gomoku 里，WR 仍然是质量真理。  
在 webhook 里，correctness / safety 是硬真理。  
在外汇策略里，liquidity safety / compliance 是硬真理。

Pareto 的位置始终是：

- 在真理之后，组织 trade-off

### 边界 2：Local Exploration Layer 不能反客为主

局部探索层很重要，但它不应替代 autoresearch 的高层判断。

否则系统会再次滑向“只会搜索，不会研究”。

### 边界 3：人类最终选择在某些场景中仍然不可替代

尤其在工程实现和金融策略场景里，系统很适合做：

- 候选生成
- 验证
- Pareto 映射

但最终选点往往仍应由人类完成。

这不是系统不够自动化，而是这些问题本身就包含现实业务偏好与责任边界。

---

## 11. 独立风险评审：广义 autoresearch 演进中的三条致命红线

前面章节讨论了 Pareto Frontier、Local Exploration Layer、跨场景泛化和人类选点界面，这些内容都具有真实价值。

但如果只讨论潜力，而不把风险提升到结构纪律层面，这条路线会非常危险。

这里必须明确：

> **当前项目未来最大的风险，不是代码复杂，不是领域切换，也不是名字变化，而是项目在精神层面失去 autoresearch 的核心秩序。**

而这种失序，集中体现为三条风险。这三条风险不是普通注意事项，而是：

> **完全、彻底、致命的红线。**

未来任何版本更新，只要踩中其中一条，都应被视为严重偏离，必须优先修正，而不能被解释成“正常演进”。

### 11.1 红线一：单一裁决纪律被稀释

这是第一条，也是最根本的一条。

项目无论怎么扩展、怎么泛化、怎么引入 Pareto，都必须保留这样一个核心秩序：

- 在 Gomoku 中，质量真理由固定 benchmark 下的 WR 裁决
- 在 webhook 场景中，真理由 correctness / safety gate 与明确主性能目标裁决
- 在外汇策略场景中，真理由 liquidity safety / compliance / hard constraints 裁决

也就是说：

> **领域真理必须始终先于 Pareto、先于局部探索、先于人类偏好。**

一旦未来更新出现下面这些倾向，就属于踩线：

1. 把 Pareto 维度误当成与真理层完全并列的多目标集合
2. 用“边界上可选”来替代“是否真的有效”
3. 让系统越来越难回答“这个实验到底算不算进步”
4. 用过多解释性指标，稀释 benchmark 的最终裁决地位

为什么这是致命风险：

因为一旦单一裁决纪律消失，系统就会迅速从“研究系统”变成“结果解释系统”。

研究系统的核心是：

- 有清楚的保留
- 有清楚的淘汰
- 有清楚的推进

如果什么都能解释、什么都能放在边界上、什么都能继续讨论，那系统将失去研究推进所需的残酷性。

这条红线必须写成未来更新的硬原则：

> **Pareto 只负责组织 trade-off，永远不能替代 domain truth 的最终裁决。**

### 11.2 红线二：agent 主控权被慢慢稀释

这是第二条，也是最容易在“功能越来越丰富”的过程中被无声踩中的红线。

当前我们讨论了很多层：

- Benchmark Layer
- Local Exploration Layer
- Pareto Layer
- Human Decision Layer
- Report Layer

这些层本身都不是问题。

真正的问题是：

> **如果它们越来越强，最终把 agent 从研究主体压缩成“摘要阅读器”或“结果解说员”，那就构成了对 autoresearch 精神的根本掏空。**

未来如果出现下面这些倾向，就属于踩线：

1. 局部探索层开始替 agent 决定主要研究方向
2. 报告层越来越像“预做结论”，agent 只是在接受结论
3. keep / discard 的研究判断越来越被固定规则接管
4. 系统越来越依赖自动流程推进，而不是 agent 主动提出假设与改动

为什么这是致命风险：

因为 autoresearch 的灵魂不是“自动运行”，而是：

- **agent 在外层研究**
- **代码或系统在内层被研究**

一旦 agent 不再是研究主体，项目就会重新滑回工具主导、流程主导、搜索主导的系统。

这条红线必须写成未来更新的硬原则：

> **任何新层的加入，都只能增强 agent 的感知与行动能力，不能替代 agent 作为研究主体的地位。**

### 11.3 红线三：人类选择过早上位，研究闭环退化成决策支持平台

这是第三条，也是最容易在“真实业务场景”里被合理化的一条风险。

我们已经讨论过：

- webhook 工程系统里，人类最终可能需要选 Pareto 点
- 外汇头寸策略里，人类最终几乎必然需要选 Pareto 点

这本身是合理的。

但合理不等于没有危险。

真正的风险是：

> **如果人类过早、过频繁、过深地介入，把 agent 和 benchmark 还没完成的研究闭环提前接管，那么项目会从 autoresearch 退化成“agent 辅助的决策分析平台”。**

未来如果出现下面这些倾向，就属于踩线：

1. 人类频繁跳过 benchmark / hard constraints，直接按直觉选点
2. agent 还没完成探索，人类就不断手动改研究方向
3. Pareto 面板变成主要价值，而 agent 研究闭环退居次要
4. 系统的中心从“研究新边界”变成“给人类做可视化备选菜单”

为什么这是致命风险：

因为这会把整个系统从：

- agent-driven research system

退化成：

- human-led decision dashboard with agent assistance

这两者不是一回事。

后者当然也可能有价值，但它已经不再是当前项目想守住的主精神。

这条红线必须写成未来更新的硬原则：

> **人类在某些场景下可以做最终选点，但不能过早接管研究主循环；否则项目会从“自动研究”退化为“人工主导、agent 辅助”的分析平台。**

### 11.4 为什么这三条风险必须被视为绝对红线

这三条风险之所以不是普通 warning，而是绝对红线，是因为它们分别对应三个核心支柱：

1. **真理秩序**
   如果裁决纪律被稀释，系统失去研究的客观推进标准。

2. **主体秩序**
   如果 agent 主控权被稀释，系统失去 autoresearch 的研究主体。

3. **责任秩序**
   如果人类过早上位，系统失去闭环研究的自治性。

这三个支柱缺一不可。

因此，未来所有更新都应先问这三个问题：

1. 这个更新是否削弱了 domain truth 的最终裁决地位？
2. 这个更新是否稀释了 agent 作为研究主体的主控权？
3. 这个更新是否让人类过早从“最终选点者”变成“研究过程主导者”？

只要有一个答案是“是”，这个更新就必须被重新审查。

### 11.5 风险评审的最终结论

所以，本章节的最终结论不是“我们需要注意风险”，而是：

> **未来所有版本更新，都必须把这三条风险当作绝对红线。**

它们分别是：

1. **单一裁决纪律被稀释**
2. **agent 主控权被慢慢稀释**
3. **人类选择过早上位**

这三条一旦被突破，项目就不再只是“有些偏离”，而是会在精神层面失去当前路线最核心的自我定义。

---

## 12. 推荐的下一阶段实现思路

如果以当前仓库为起点，Pareto Frontier 的第一阶段最合理的实现路径是：

### 12.1 先在 Gomoku 内完成最小可用版本

优先做：

1. benchmark run Pareto
2. threshold Pareto
3. grouped sweep Pareto

这一步直接建立在当前 tracker 和 analyze 数据结构上，风险最低。

### 12.2 再把 Pareto 层从“Gomoku 专用”抽象出来

第二阶段再明确：

- Pareto layer 不绑定 WR
- Pareto layer 接收 domain-specific truth 和 cost axes

这样它才能扩展到 webhook / treasury 这类场景。

### 12.3 最后把它正式纳入项目分层

未来更完整的系统可以写成：

1. Autoresearch Layer
2. Local Exploration Layer
3. Benchmark Layer
4. Pareto Layer
5. Human Decision Layer

这比“训练脚本 + 分数输出”要完整得多，也更接近我们最新讨论出来的项目未来形态。

---

## 13. 最终 Verdict

本轮翻新后的最终 verdict 如下：

1. **Pareto Frontier 已经不应再被视为仅服务 WR 的附属分析，而应被视为整个项目演进中的核心解释层。**

2. **`AutoML` 这个名字应正式退场，新的名称建议是 `Local Exploration Layer`。**
   这个名字更准确地表达了它在当前和未来系统中的真实角色。

3. **Pareto Frontier 在当前代码事实下是完全合理的演进方向。**
   因为 runs、checkpoints、matrix、seed、sweep_tag 等结构已经提供了足够的数据基础。

4. **Pareto Frontier 不只适用于 Gomoku，也适用于 webhook 数据处理系统与外汇头寸策略系统。**
   这说明项目正在从单一 ML 训练平台，走向更一般化的 agent-driven research framework。

5. **这个方向在生态位上有真实价值。**
   它让系统不再只输出“单点最优”，而开始输出“在真实约束下可供选择的边界结构”。

6. **最重要的边界必须始终保留：Pareto 负责组织 trade-off，不负责替代 benchmark 真理。**

7. **未来所有更新都必须把三条红线提升到最高 awareness 层次。**
   也就是：
   - 单一裁决纪律不能被稀释
   - agent 主控权不能被慢慢稀释
   - 人类选择不能过早上位
   这三条不是附属风险，而是整个系统是否还能自称为当前这条路线的根本判据。

一句话总结：

> **本项目下一阶段真正值得追求的，不只是让 agent 找到更高分，而是让系统能够在绝对红线不被突破的前提下，围绕 domain truth、局部探索与现实成本，持续构造一条可解释、可比较、可由人类选点的 Pareto 前沿。只有这样，项目才会从“自动试验”升级成“真正可用于研究与决策的演化系统”。**
---


## 14. autoresearch 框架的 Pareto 能力实证：基于 Gomoku 201,728 盘棋局

> 2026-04-12 | 基于 tracker.db 全部训练数据

本章站在 **autoresearch 框架自身** 的视角，回答两个核心问题：

1. **autoresearch 是否有能力发现 Pareto 前沿？**
2. **autoresearch 是否有能力加速前沿曲线的发现和建立？**

分析的边界非常明确：**框架不关心用户定义的算法"好不好"、"够不够强"。它只关心，在用户给定的变量空间内，框架自身的机制能否找到、并加速找到 Pareto 最优曲线。** 至于用户是否要换算法、是否要加 MCTS，那是用户的决策，不是框架的职责。

### 14.1 数据基础

截至目前，Gomoku domain 积累的训练数据：

| 指标 | 值 |
|------|-----|
| 有效训练 run 数 | 27 |
| 总 checkpoint 数 | 73 |
| 总训练棋局 | 201,728 |
| 总训练墙钟时间 | ~4.9 小时 |
| 涉及的架构 | 6x32, 6x64, 6x96, 8x64, 10x64, 12x64, 15x64, 16x64 |
| 学习率范围 | 3e-4 ~ 1e-3 |
| steps-per-cycle 范围 | 20 ~ 50 |
| parallel-games 范围 | 16 ~ 100 |

用户定义的变量空间由 `train.py` 顶部的超参常量构成：`NUM_RES_BLOCKS`, `NUM_FILTERS`, `LEARNING_RATE`, `BATCH_SIZE`, `PARALLEL_GAMES`, `TRAIN_STEPS_PER_CYCLE` 等。用户定义的真理指标是 WR（win rate vs 指定对手）。成本轴包括墙钟时间、训练局数、参数量。

框架的任务就是：**在这个变量空间 × 这些成本轴上，画出 WR 的 Pareto 前沿。**

---

### 14.2 问题一：autoresearch 是否有能力发现 Pareto 前沿？

#### 14.2.1 实证：前沿确实存在且可被发现

在 `is_benchmark = 1` 的 run 中（统一 120s 时间预算、minimax-L0 对手），不同超参配置的 WR 如下：

| 架构 | LR | spc | pg | 训练局 | WR | 墙钟 |
|------|-----|-----|-----|--------|-----|------|
| 10x64 | 7e-4 | 30 | 64 | 1,280 | **85.5%** | 120s |
| 8x64 | 5e-4 | 30 | 16 | 624 | **80.5%** | 180s |
| 8x64 | 7e-4 | 30 | 64 | 1,280 | 74.0% | 120s |
| 6x64 | 5e-4 | 30 | 16 | 160 | 71.0% | 36s |
| 8x64 | 5e-4 | 30 | 64 | 1,600 | 69.5% | 120s |
| 6x64 | 3e-4 | 20 | 16 | 224 | 62.0% | 20s |
| 6x32 | 5e-4 | 30 | 16 | 240 | 53.0% | 34s |
| 6x64 | 5e-4 | 30 | 16 | 192 | 49.0% | 30s |
| 6x64 | 5e-4 | 30 | 16 | 192 | 42.0% | 30s |
| 6x64 | 5e-4 | 30 | 16 | 176 | 37.0% | 20s |

从这张表中，框架可以直接通过非支配排序提取 Pareto 前沿（WR vs 墙钟）：

```
6x64 / LR=5e-4 / 36s  → 71.0%    (最快达到 70%+ 的配置)
8x64 / LR=5e-4 / 180s → 80.5%    (中等投入的稳定选择)
10x64 / LR=7e-4 / 120s → 85.5%   (固定预算下的最强配置)
```

被支配的点全部可以被自动淘汰：
- `6x32 / 53%`：被 `6x64 / 71%` 支配（时间接近但 WR 远低）
- `8x64 / LR=7e-4 / 74%`：被 `10x64 / 85.5%` 支配（同样 120s 但 WR 更低）
- `6x64 / 42%, 49%`：被同配置的 `6x64 / 71%` 支配

**结论：可以。** 只要有标准化的 benchmark 条件（统一时间预算 + 统一对手），框架通过 `analyze.py` 已有的查询能力，可以在 `(WR, wall_time, params)` 空间上直接执行非支配排序，产出 Pareto 前沿。这不需要框架理解 Gomoku 是什么，只需要它能读取 tracker.db 中的数值列。

#### 14.2.2 实证：架构维度的 Pareto 边界

切换到参数量维度（params vs WR），同样清晰：

| 架构 | 参数量 | 最佳 WR（vs mm-L0） | 墙钟 | 训练局 |
|------|--------|---------------------|------|--------|
| 6x32 | 229K | 56.0% | 22s | 240 |
| 6x64 | 564K | 71.0% | 36s | 160 |
| 8x64 | 713K | 85.0% | 18s | 192 |
| 10x64 | 862K | 85.5% | 120s | 1,280 |
| 6x96 | 1,120K | 20.0% | 14s | 128 |

Pareto 前沿：`6x32 → 6x64 → 8x64`。之后 10x64 参数量 +21% 但 WR 仅 +0.5%（边际收益急剧递减），6x96 参数量翻倍但 WR 崩溃到 20%。

框架完全可以自动发现这条曲线。它不需要知道"为什么 6x96 不行"——它只需要看到 6x96 在 `(params, WR)` 空间被 8x64 严格支配，然后将其标记为被支配点即可。

#### 14.2.3 实证：学习率维度的前沿

在 6x64 架构、同一对手的 9 个 run 中：

| 学习率 | run 数 | 最佳 WR | 最差 WR | 方差 |
|--------|--------|---------|---------|------|
| 3e-4 | 1 | 62.0% | 62.0% | — |
| 5e-4 | 6 | 71.0% | 37.0% | **极高** |
| 1e-3 | 2 | 58.0% | 54.5% | 低 |

补充：8x64 架构中，LR=7e-4 在 120s benchmark 达 74.0%，LR=5e-4 只有 69.5%。

框架可以发现：LR=5e-4~7e-4 是当前最佳区间。但更重要的是，框架可以发现 **LR=5e-4 的方差极大**（37%~71%），如果框架记录了足够多的重复 run，它就能把"高方差"这个信息附加到 Pareto 分析中——标记该配置为"不稳定前沿点"。

#### 14.2.4 实证：样本效率维度的前沿

| run | 架构 | 训练局 | WR | WR / 千局 |
|-----|------|--------|-----|-----------|
| 608fd768 | 6x64 | 128 | 70.0% | **5.47** |
| 6bb72701 | 8x64 | 192 | 85.0% | **4.43** |
| b4f8e1bc | 8x64 | 640 | 93.3% | **1.46** |
| 92d1740a | 10x64 | 5,056 | 84.0% | 0.17 |
| b3f99d4f | 10x64 | 72,000 | 75.0% | **0.01** |

框架可以发现：**前 200-600 盘是高效区，之后效率急剧衰减。** 192 盘达到 85% WR 的效率是 72,000 盘达到 75% WR 的 400 倍。这是一条 `(训练局数, WR)` 上的 Pareto 曲线，框架只需要对比不同 run 的数值就能看到拐点。

#### 14.2.5 小结：发现能力

| 维度 | 框架能否发现 Pareto？ | 依赖什么 |
|------|---------------------|---------|
| 架构（params vs WR） | ✅ 是 | tracker.db 中的 num_res_blocks, num_filters, eval_wr |
| 学习率（LR vs WR） | ✅ 是 | tracker.db 中的 learning_rate, eval_wr |
| 训练预算（games vs WR） | ✅ 是 | tracker.db 中的 total_games, eval_wr |
| 墙钟效率（time vs WR） | ✅ 是 | tracker.db 中的 wall_seconds, eval_wr |
| 方差可靠性（LR vs variance） | ⚠️ 需要多次重复 | 同配置多 run 的 WR 分布 |
| 训练停滞检测 | ⚠️ 需要新机制 | cycle_metrics 表中连续 N 个 checkpoint 的 WR 趋势 |

**框架的 Pareto 发现能力完全取决于 tracker.db 中记录了哪些数值列。** 只要用户的变量空间（超参）和真理指标（WR）都被记录到数据库，框架就可以在任意维度组合上执行非支配排序。这是纯数值操作，与 domain 无关。

---

### 14.3 问题二：autoresearch 是否有能力加速 Pareto 前沿的建立？

"发现"是事后分析——27 个 run 跑完了，回头看数据能不能画出前沿。"加速"是在线策略——在跑下一个 run 之前，框架能否利用已有数据做出更聪明的选择，减少探索浪费？

#### 14.3.1 当前的加速机制：sweep.py

`sweep.py` 是框架目前唯一的主动探索工具。它的机制是：

1. 用户定义一个超参矩阵（架构 × LR × spc × pg 等）
2. `sweep.py` 对矩阵中的每个组合启动一个 `train.py` 子进程
3. 每个 run 有统一的时间预算（如 120s）
4. 所有 run 的结果写入 tracker.db
5. 用户用 `analyze.py` 查看结果

这是一个**穷举式探索**。它的加速效果来自两点：

- **统一短时间预算**：每个配置只给 120s，避免像 b3f99d4f 那样浪费 72,000 盘（100 分钟）在已经停滞的配置上
- **批量执行**：一次性跑完整个矩阵，而不是人工一个个试

#### 14.3.2 实证：穷举 sweep 的效率分析

以实际数据估算。如果用户想在以下变量空间中找 Pareto：

- 架构：6x32, 6x64, 8x64, 10x64（4 种）
- LR：3e-4, 5e-4, 7e-4, 1e-3（4 种）
- spc：20, 30, 50（3 种）

总组合数 = 4 × 4 × 3 = **48 个 run**。每个 120s，总墙钟 = 48 × 120s = **96 分钟**。

实际上，当前 27 个 run 中只有约 10 个是 benchmark run（统一条件），其余都是非标准化的长训练或不同对手。如果一开始就用 sweep 做标准化探索，**48 个 run × 120s = 96 分钟** 就能覆盖整个变量空间，而目前 4.9 小时（294 分钟）只覆盖了其中一个子集。

**加速倍率：约 3x**（96 分钟 vs 294 分钟），且覆盖更全面。

#### 14.3.3 可以实现但尚未实现的加速机制

除了穷举 sweep，框架还有以下加速策略的空间，全部在框架能力范围内：

**（1）早停（Early Stopping）**

当前问题：b3f99d4f 从 cycle 225 之后 WR 停滞在 ~75%，但一直跑到 cycle 1125（72,000 盘），浪费了约 58,000 盘训练量和 ~70 分钟墙钟。

框架可以做什么：在 `cycle_metrics` 表中检测连续 N 个 checkpoint 的 WR 无趋势性改善（±2%），自动触发停止。**这不需要理解 Gomoku，只需要对时间序列数值做斜率检测。**

预估节省：在 27 个 run 中，至少有 5 个 run 在到达其最终 WR 的 50% 训练量时就已经停滞。早停可以节省约 40% 的总训练墙钟。

**（2）自适应采样（Adaptive Sampling）**

当前问题：6x64/LR=5e-4 跑了 6 个 run，WR 从 37% 到 71%。如果框架在第 3 个 run 已经看到方差极大，它可以自动决定"这个配置需要更多样本才能确定其 Pareto 位置"，同时对方差已经很低的配置（如 LR=1e-3）停止追加样本。

框架可以做什么：对每个配置维护 WR 的均值和标准差。当标准差超过阈值时，自动追加 run；当标准差足够小时，标记该配置的 Pareto 位置为"已确定"。**这是纯统计操作，domain-agnostic。**

**（3）逐步精化（Progressive Refinement）**

当前问题：48 个组合的全矩阵 sweep 中，很多组合（如 6x32 + LR=1e-3）在前几个 cycle 就能看出表现很差，不需要跑完全部 120s。

框架可以做什么：
- 第一轮：所有 48 个配置各跑 30s
- 第二轮：淘汰 WR 排名后 50% 的配置，剩下 24 个跑到 60s
- 第三轮：淘汰后 50%，剩下 12 个跑到 120s
- 第四轮：对前沿附近的 6 个配置追加 run 以确认方差

总墙钟：48×30 + 24×30 + 12×60 + 6×120 = 1440 + 720 + 720 + 720 = **3,600s = 60 分钟**，比全矩阵 96 分钟快 **37%**，且信息损失极小（淘汰的都是被支配点）。

**（4）停滞检测 + 自动切换对手**

当前问题：模型对某个对手的 WR 到达天花板后继续训练是浪费。

框架可以做什么：检测到 WR 停滞后，不是停止训练，而是自动将该 checkpoint 注册为新的更强对手，然后开始用新对手继续下一轮训练。**这是 prepare.py 中对手注册机制已经支持的操作。**

#### 14.3.4 小结：加速能力

| 加速机制 | 当前状态 | 框架能力 | 预估加速 |
|----------|---------|---------|---------|
| 标准化 sweep（穷举） | ✅ `sweep.py` 已实现 | 完全具备 | ~3x vs 非标准化探索 |
| 早停 | ❌ 未实现 | 可实现（时间序列斜率检测） | ~40% 墙钟节省 |
| 自适应采样 | ❌ 未实现 | 可实现（均值/标准差追踪） | 减少低信息 run |
| 逐步精化 | ❌ 未实现 | 可实现（多轮淘汰） | ~37% vs 全矩阵 |
| 停滞 → 注册新对手 | ❌ 未实现 | 可实现（prepare.py 接口已存在） | 消除手动干预延迟 |

**结论：可以。** 框架有能力加速 Pareto 前沿的建立。当前仅通过穷举 sweep 就已实现约 3x 加速。如果加入早停、自适应采样、逐步精化三个机制，预估总加速可达 5-8x。**这些机制全部是 domain-agnostic 的纯数值操作，不需要框架理解 Gomoku 的任何知识。**

---

### 14.4 框架的边界在哪里

框架能做的事情清单：

1. ✅ 在用户定义的变量空间内执行非支配排序，输出 Pareto 前沿
2. ✅ 通过标准化 sweep 系统性覆盖变量空间
3. ✅ 通过早停避免在已停滞的配置上浪费资源
4. ✅ 通过方差追踪标记不可靠的前沿点
5. ✅ 通过逐步精化减少低价值探索
6. ✅ 呈现 `(truth_metric, cost_axis_1, cost_axis_2, ...)` 的多维 Pareto 图供人类选点

框架 **不做** 的事情清单：

1. ❌ 不告诉用户应该用什么算法（MCTS 还是纯 policy）
2. ❌ 不评判用户定义的真理指标"好不好"（WR 是否反映真实棋力是用户的判断）
3. ❌ 不越界修改变量空间（agent 只在 `train.py` 的超参常量范围内操作）
4. ❌ 不解释为什么某个配置表现好/差（归因是用户的工作）

这个边界是 **autoresearch 作为工具框架的根本定位**。框架就像一个高效的实验室自动化系统：它能快速跑完实验矩阵、画出结果曲线、标出最优点——但它不会告诉化学家该换什么反应路线。

---

### 14.5 从 Gomoku 数据验证的关键框架特性

以下是从 201,728 盘实际数据中，可以验证的框架设计要点：

#### 特性一：小片段 benchmark 是 Pareto 发现的核心基础

120s benchmark run 是最高效的 Pareto 数据源：
- 条件完全标准化（统一时间预算、统一对手）
- 每个只需 2 分钟，可以大量执行
- 数据质量远高于非标准化长训练

实际数据证明：仅用 10 个标准化 benchmark run（累计 ~20 分钟），就足以确定 `6x32 → 6x64 → 8x64` 的 Pareto 前沿。而非标准化的 17 个长 run 累计花了 ~4.5 小时，对 Pareto 发现的贡献反而更小。

**框架设计启示：sweep 应当首先投资小片段标准化探索。**

#### 特性二：方差是 Pareto 分析的一等公民

6x64/LR=5e-4 的 6 个 run WR 从 37% 到 71%。如果框架只报告"最佳 WR=71%"，这是误导——用户选了这个配置，下次跑出 37% 的概率也不低。

**框架设计启示：每个 Pareto 点必须附带置信区间。高方差点应标记为"需要更多样本"。**

#### 特性三：停滞检测是止损的关键

b3f99d4f：72,000 盘，WR 在 cycle 225 (~14,000 盘) 后停滞，浪费了 ~58,000 盘。
8b9486f4：54,000 盘，WR 从峰值 85% 反降到 78%。

**框架设计启示：如果 cycle_metrics 中连续 N 个 checkpoint 的 eval_wr 在 ±2% 内无趋势改善，框架应发出停滞警告或自动停止。这是 domain-agnostic 的时间序列操作。**

#### 特性四：12x64 的"更大就更好"被数据否定

12x64（1.01M 参数）vs L0 训练 54,000 盘，峰值 WR 85.0%，最终反降到 78.0%。
8x64（713K 参数）vs L0 仅 640 盘就达到 93.3% WR。

12x64 花了 84 倍训练量、51 倍墙钟，WR 反而更低。**框架不需要解释为什么——它只需要在 Pareto 图上标注 12x64 为被支配点。** 用户看到这个数据后，自然会得出"在当前变量空间内，更大不等于更好"的结论。

---

### 14.6 总结

> **autoresearch 框架完全有能力在用户定义的变量空间内发现 Pareto 前沿。从 Gomoku 的 201,728 盘数据中，框架可以自动提取架构、学习率、训练预算、墙钟效率四个维度上的非支配曲线。这些操作是纯数值的，与 domain 无关。**
>
> **autoresearch 框架同样有能力加速前沿的建立。当前已实现的 sweep 机制提供约 3x 加速；如果加入早停、自适应采样、逐步精化三个 domain-agnostic 机制，预估可达 5-8x。**
>
> **框架不越界：它不评判用户的算法选择，不建议改变变量空间，不解释因果。它只在给定的实验空间内，用最少的资源，画出最准确的 Pareto 曲线，然后交给人类选点。**

---

## 15. 解耦方案：从 Gomoku 单体到 autoresearch 框架 + 可替换 Domain

### 15.1 解耦的动机

当前仓库的全部代码都在一个扁平的 `src/` 目录下，Gomoku 是唯一的 domain。这在项目初期完全合理——先让系统跑起来，再考虑结构。

但根据第 14 章的分析，框架的核心能力（Pareto 发现、sweep 探索、tracker 记录、analyze 报告）是 **domain-agnostic** 的。它们不依赖 Gomoku 的任何知识。将它们从 Gomoku 中分离出来，意味着：

1. 框架可以服务于新的 domain（如 webhook 优化、FX 策略探索），而不需要重写
2. 每个 domain 自带自己的训练脚本、评估逻辑、对手定义
3. 框架层的改进（早停、自适应采样等）自动惠及所有 domain

### 15.2 当前代码的耦合分析

通过对 `src/` 下所有 11 个 `.py` 文件的 import 依赖分析，当前代码清晰地分为三类：

#### 框架层（零内部依赖，domain-agnostic）

| 文件 | 行数 | 职责 | 内部 import |
|------|------|------|-------------|
| `tracker.py` | 602 | SQLite 实验追踪（runs, checkpoints, cycle_metrics） | 无 |
| `analyze.py` | 947 | 查询 tracker.db，生成报告/前沿/对比 | 无 |
| `sweep.py` | 268 | 超参矩阵 sweep，通过 subprocess 启动 train.py | 无 |
| `tui.py` | 141 | TUI 渲染工具（sparkline, progress_bar 等） | 无 |

这四个文件 **没有任何内部 import**。它们只依赖标准库和 SQLite。它们通过两个接口与 domain 通信：
- **SQLite**：tracker.db 的 schema 是协议
- **subprocess**：sweep.py 通过 `subprocess` 调用 `train.py`

**这意味着框架层已经事实上解耦了。**

#### Domain 层（Gomoku 专属）

| 文件 | 行数 | 职责 | 内部 import |
|------|------|------|-------------|
| `game.py` | 542 | 棋盘引擎（Board, BatchBoards, BOARD_SIZE, BLACK/WHITE/EMPTY） | 无 |
| `prepare.py` | 620 | minimax 对手（L0-L3）、评估函数、对手注册 | `game` |
| `replay.py` | 208 | 棋局回放与帧导出 | `game` |

这三个文件构成 Gomoku 的核心领域代码。`game.py` 是基础，`prepare.py` 和 `replay.py` 依赖它。

#### 混合层（需要拆分）

| 文件 | 行数 | 职责 | 内部 import |
|------|------|------|-------------|
| `train.py` | 1,562 | NN 定义 + 训练循环 + 自对弈 + TUI 输出 | `game`, `tui`, `prepare` |
| `play.py` | 226 | CLI 人机对弈 | `game`, `play_service`, `prepare`, `tracker` |
| `play_service.py` | 168 | 共享对弈服务（加载模型、创建 player） | `game`, `prepare`, `tracker` |
| `web_app.py` | 230 | FastAPI 浏览器 UI | `game`, `play_service` |

`train.py` 是最重要的混合文件——它包含了 Gomoku 的 NN 架构定义（GomokuNet）、训练循环、自对弈逻辑，同时也是 agent 唯一修改的文件。其他三个是对弈界面，依赖 domain 和 framework 的组合。

### 15.3 解耦后的目标目录结构

```text
mag-gomoku/                           # 项目根目录
│
├── pyproject.toml                    # 项目元数据 & 依赖
├── uv.lock                          # 锁文件
├── README.md                        # 项目说明
├── .gitignore
│
├── framework/                        # ═══ autoresearch 框架层 ═══
│   │
│   ├── __init__.py
│   ├── tracker.py                   # [602行] 实验追踪 — SQLite CRUD
│   │                                #   runs / checkpoints / cycle_metrics / opponents
│   │                                #   接口：TrackerDB class
│   │                                #   协议：tracker.db schema（见 15.4）
│   │
│   ├── analyze.py                   # [947行] 只读分析 — 查询 tracker.db
│   │                                #   报告生成、Pareto 前沿提取、矩阵对比
│   │                                #   输入：tracker.db 路径
│   │                                #   输出：stdout 表格/图表
│   │
│   ├── sweep.py                     # [268行] 批量实验 — 超参矩阵 sweep
│   │                                #   对每个超参组合 subprocess 调用 domain 的 train.py
│   │                                #   输入：sweep 配置（YAML/dict）
│   │                                #   输出：多个 run 的 tracker.db 记录
│   │
│   └── tui.py                       # [141行] TUI 渲染工具
│                                    #   sparkline, progress_bar 等纯函数
│                                    #   无状态，无 I/O
│
├── domains/                          # ═══ 可替换的执行域 ═══
│   │
│   └── gomoku/                      # --- Gomoku domain ---
│       │
│       ├── __init__.py
│       ├── game.py                  # [542行] 棋盘引擎
│       │                            #   Board, BatchBoards, GameRecord
│       │                            #   常量：BOARD_SIZE=15, BLACK, WHITE, EMPTY
│       │                            #   纯逻辑，无 ML 依赖
│       │
│       ├── prepare.py               # [620行] 对手 & 评估
│       │                            #   minimax 对手 L0-L3（不同搜索深度）
│       │                            #   evaluate_win_rate() 评估函数
│       │                            #   OPPONENTS 注册表
│       │                            #   TIME_BUDGET 常量
│       │                            #   依赖：game.py
│       │
│       ├── train.py                 # [1562行] 训练脚本 ★ agent 唯一修改的文件 ★
│       │                            #   GomokuNet: ResBlock CNN (policy+value heads)
│       │                            #   超参常量区（NUM_RES_BLOCKS, LR, BATCH_SIZE 等）
│       │                            #   self_play() → replay_buffer → train_step() 循环
│       │                            #   写 tracker.db、保存 checkpoint
│       │                            #   依赖：game.py, prepare.py, framework/tui.py
│       │
│       ├── replay.py                # [208行] 棋局回放
│       │                            #   读取 JSON 录像，渲染棋盘帧
│       │                            #   可用于视频导出
│       │                            #   依赖：game.py
│       │
│       ├── play_service.py          # [168行] 共享对弈服务
│       │                            #   load_nn_player(), create_player()
│       │                            #   get_frontend_opponents()
│       │                            #   依赖：game.py, prepare.py, framework/tracker.py
│       │
│       ├── play.py                  # [226行] CLI 对弈界面
│       │                            #   pygame 实时对弈（人 vs AI / AI vs AI）
│       │                            #   依赖：game.py, play_service.py, prepare.py
│       │
│       └── web/                     # 浏览器对弈界面
│           ├── web_app.py           # [230行] FastAPI 后端
│           │                        #   依赖：game.py, play_service.py
│           ├── index.html           # HTML 入口
│           ├── app.js               # 前端逻辑
│           └── styles.css           # 样式
│
├── output/                           # ═══ 运行产物（gitignore 大部分） ═══
│   ├── tracker.db                   # SQLite 数据库（框架协议）
│   └── {run_uuid}/                  # 每个 run 的输出
│       ├── checkpoints/             # .safetensors 权重文件
│       └── recordings/
│           └── games/               # .json 棋局录像
│
└── docs/                             # ═══ 文档 ═══
    ├── program.md                   # 项目规则（"train.py 是唯一可修改文件"）
    ├── caveats.md                   # 已知限制
    ├── performance_baseline.md      # 性能基线
    └── ...
```

### 15.4 框架与 Domain 的接口协议

解耦后，框架层与 domain 层之间通过 **两个接口** 通信：

#### 接口一：tracker.db Schema（数据协议）

这是最核心的协议。任何 domain 的 `train.py` 只需要往 tracker.db 写入以下标准字段：

```sql
-- runs 表：每个训练 run 一行
runs (
    run_id TEXT PRIMARY KEY,          -- UUID
    -- 框架标准字段（所有 domain 必须写）
    start_time TEXT,                  -- ISO 8601
    wall_seconds REAL,                -- 总墙钟
    total_games INTEGER,              -- 总训练样本数（Gomoku 中=棋局数）
    total_steps INTEGER,              -- 总梯度步数
    eval_wr REAL,                     -- 真理指标（Gomoku 中=WR）
    is_benchmark INTEGER,             -- 是否标准化 benchmark run
    -- domain 自定义字段（框架不解读，只存储和展示）
    num_res_blocks INTEGER,           -- Gomoku: 架构深度
    num_filters INTEGER,              -- Gomoku: 架构宽度
    learning_rate REAL,               -- Gomoku: LR
    ...                               -- domain 可以自由添加列
)

-- checkpoints 表：每个 checkpoint 一行
checkpoints (
    ckpt_id INTEGER PRIMARY KEY,
    run_id TEXT REFERENCES runs,
    cycle INTEGER,
    eval_wr REAL,                     -- 该 checkpoint 的评估 WR
    path TEXT,                        -- 权重文件相对路径
    ...
)
```

**关键设计：** 框架只读取 `eval_wr`（真理指标）和若干成本列（`wall_seconds`, `total_games`, `total_steps`）。其余所有列对框架来说是"自定义维度"——它可以展示它们、在 Pareto 图上用它们作轴，但不解读其含义。

#### 接口二：subprocess 启动协议

`sweep.py` 通过命令行启动 domain 的训练脚本：

```bash
python domains/gomoku/train.py --time-budget 120 --tag sweep_exp_001
```

协议要求：
- domain 的 `train.py` 接受 `--time-budget`（秒）和 `--tag`（标签）参数
- `train.py` 自行写入 tracker.db
- `train.py` 退出码 0 = 成功

**这意味着新增 domain 不需要修改 `sweep.py`——只需要提供一个符合上述协议的 `train.py`。**

### 15.5 解耦的具体步骤

解耦不需要重写任何逻辑，只需要移动文件和调整 import 路径：

| 步骤 | 操作 | 影响范围 |
|------|------|---------|
| 1 | 创建 `framework/` 和 `domains/gomoku/` 目录 | 无代码变更 |
| 2 | 移动 `tracker.py`, `analyze.py`, `sweep.py`, `tui.py` → `framework/` | 这四个文件无内部 import，移动后零修改 |
| 3 | 移动 `game.py`, `prepare.py`, `replay.py` → `domains/gomoku/` | 内部 import 不变（都在同一目录） |
| 4 | 移动 `train.py` → `domains/gomoku/train.py` | 修改 import：`from framework.tui import ...` |
| 5 | 移动 `play.py`, `play_service.py` → `domains/gomoku/` | 修改 import：`from framework.tracker import ...` |
| 6 | 移动 `web_app.py` + `web/` → `domains/gomoku/web/` | 修改 import 路径 |
| 7 | 更新 `sweep.py` 中 train.py 的路径 | 一行改动 |
| 8 | 更新 `pyproject.toml` 的包结构 | 调整 packages 配置 |
| 9 | 更新 `docs/program.md`（"单一规则"中的路径） | 文档更新 |

**总修改量估算：约 20 行 import 路径调整 + 2 个配置文件更新。** 没有逻辑变更。

### 15.6 新 Domain 的接入模板

解耦完成后，接入新 domain 只需要：

```text
domains/
  new_domain/
    __init__.py
    train.py              # 必须：符合 subprocess 协议 + 写 tracker.db
    evaluate.py           # 可选：domain 专属评估逻辑
    ...                   # 其余文件由 domain 自行组织
```

`train.py` 的最小要求：
1. 接受 `--time-budget` 和 `--tag` 命令行参数
2. 训练过程中写 tracker.db（使用 `framework.tracker`）
3. 写入标准字段：`run_id`, `wall_seconds`, `total_games`, `total_steps`, `eval_wr`
4. 退出码 0 表示成功

**框架层不需要任何修改即可对新 domain 执行 sweep → track → analyze → Pareto。**

### 15.7 解耦优先级

根据当前项目阶段，解耦的优先级：

| 优先级 | 行动 | 理由 |
|--------|------|------|
| P0 | 确认 tracker.db schema 作为正式接口协议 | 这是框架与 domain 的唯一数据通道 |
| P1 | 移动 4 个框架文件到 `framework/` | 零修改移动，立即可用 |
| P2 | 移动 domain 文件到 `domains/gomoku/` | 需要调整 ~20 行 import |
| P3 | 编写 domain 接入文档（`docs/domain-guide.md`） | 为新 domain 提供模板 |

**解耦不意味着立刻执行。** 当前只有一个 domain，过早解耦会增加路径复杂度。建议在以下任一条件满足时执行：
- 第二个 domain 开始接入
- 框架层需要重大升级（如加入早停、自适应采样）
- agent 循环需要同时服务多个 domain
