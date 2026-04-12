# Update v10 Further Analysis - 关于 autoresearch 与 AutoML 双线路共存的进一步讨论

> 日期：2026-04-11  
> 范围：记录围绕 MAG Gomoku 后续路线的进一步讨论，重点分析在当前项目中同时保留 autoresearch 与 AutoML 两种模式的可行性、角色分工、生态位价值，以及为什么这不是简单的 control group 设计，而是一个可能形成独特价值闭环的系统结构。

---

## 1. 执行摘要

在前一轮讨论中，我们已经明确：

- 项目不能继续无意识地滑向纯 AutoML
- autoresearch 应重新成为项目的主身份与主路线
- agent 应重新回到跨实验研究决策的主控位置

但进一步思考之后，我们得到一个更强、也更适合当前仓库实际的判断：

> **autoresearch 与 AutoML 不一定必须二选一。真正重要的不是是否共存，而是如何共存、谁主谁辅、谁消费谁的产出。**

这一轮讨论得出的关键结论是：

1. **autoresearch 和 AutoML 都可以在当前项目里成为一等公民。**
2. **但二者的一等公民地位，并不意味着二者角色对称。**
3. **autoresearch 负责研究判断与跨实验决策。**
4. **AutoML 负责接受约束后的局部探索、超参调整、重复验证与局部结论提取。**
5. **AutoML 并不是一个独立并行的 control group，而是整个研究生态里具有真实用途的子系统。**
6. **只有在这种主从分层关系下，项目才能形成一个明显不同于纯 autoresearch、也不同于纯 AutoML 的独特价值闭环。**

换句话说，本轮的核心不是“我们要不要保留 AutoML”，而是：

> **我们是否应该把 AutoML 从潜在偏离对象，重定义为 agent 可调度、可审查、可消费的研究工具层。**

本轮分析的答案是：**应该，而且这很可能是当前项目最有前景的结构升级方向之一。**

---

## 2. 为什么这个问题值得单独讨论

### 2.1 因为它不是路线摇摆，而是结构升级问题

如果把问题理解成：

- 要么纯 autoresearch
- 要么纯 AutoML

那么讨论会很快陷入简单的路线对立。

但当前仓库的实际情况并不是这样。

MAG Gomoku 已经具备两类能力：

1. **autoresearch 所需要的外层研究闭环基础**
   - benchmark 语义
   - experiment tracking
   - run / checkpoint / lineage / frontier 历史
   - agent 可修改训练代码的工作方式

2. **AutoML 所需要的局部探索基础**
   - 参数化训练入口
   - sweep tag / seed 等元数据跟踪
   - 对超参组合进行批量试验的现实可能性
   - 稳定性与对比分析能力

这说明当前仓库的真实问题不再是“有没有能力做二者之一”，而是：

> **如何给这两类能力安排不冲突、而且互相增益的位置。**

### 2.2 因为当前仓库已经处在一个天然适合分层的状态

从当前项目结构看，已经出现了多层天然边界：

1. 训练执行层
2. tracker / analyze 报告层
3. benchmark 裁决层
4. 潜在的外层 agent 决策层

在这种情况下，引入 AutoML 不一定会破坏系统，反而可能恰好填补“训练执行层”和“agent 研究层”之间的局部探索空白。

---

## 3. 新的核心判断：双线路可以共存，但角色绝不对称

### 3.1 autoresearch 与 AutoML 都是一等公民

说二者都应是一等公民，含义不是“同权同责”，而是：

1. 二者都不是临时补丁
2. 二者都不是边角附属功能
3. 二者都应被正式设计进系统结构与项目叙事中

autoresearch 作为一等公民，意味着它继续定义项目的研究身份。

AutoML 作为一等公民，意味着它不再只是一个被动的 control baseline，也不是一个仅用于反证的对照组，而是对整个研究生态真实有用的组成部分。

### 3.2 但二者的一等公民地位不代表对称关系

真正关键的地方在这里：

> **二者都重要，但重要的方式不同。**

在我们现在形成的定义中：

- **autoresearch 负责做决定**
- **AutoML 负责做局部探索并形成可供消费的结论**

也就是说，二者不是并列竞争关系，而是分层协作关系。

---

## 4. 明确角色分工

### 4.1 autoresearch 的角色

autoresearch 的职责应明确为：

1. 读取长期实验上下文
2. 解释 benchmark、stability、frontier、lineage 等信号
3. 判断当前问题是局部超参问题，还是结构性研究问题
4. 决定何时需要修改代码、何时需要发起局部搜索
5. 消费 AutoML 的产出并决定是否推进下一阶段
6. 保留、回退、升级或冻结某个方向

换句话说，autoresearch 是：

> **研究主管与方向裁决者。**

### 4.2 AutoML 的角色

AutoML 的职责则应明确为：

1. 接受 autoresearch 给出的局部搜索任务
2. 在受限空间内做超参探索
3. 做多 seed 重复试验或局部扫描
4. 汇总探索结果，形成结构化结论
5. 把局部最优、边际收益、稳定性差异返回给 autoresearch

换句话说，AutoML 是：

> **局部探索执行器与局部证据生产者。**

### 4.3 benchmark 的角色不变

在这个结构中，benchmark 的角色仍然不应改变。

benchmark 依然负责：

1. 作为外部事实标准
2. 裁决某个方向是否真的更优
3. 防止 autoresearch 和 AutoML 共同滑入自说自话

因此，最终结构不是二元，而是三元：

1. autoresearch：做决策
2. AutoML：做局部探索
3. benchmark：做裁决

---

## 5. 为什么 AutoML 不应只被定义为 control group

### 5.1 control group 的定义太弱

如果我们只把 AutoML 当作 control group，那么它的功能就会被收缩为：

1. 偶尔做对照实验
2. 用来证明 agent 是否真的更强
3. 在项目叙事中处于被动、防御、次要的位置

这种定位太弱，不足以体现它在当前生态中的真实用途。

### 5.2 当前项目里，AutoML 有真实的工具价值

在当前框架下，AutoML 的真实用途至少包括：

1. 做局部参数面探索
2. 帮助验证某个方向是否只是噪声
3. 对同一假设进行多 seed 重复
4. 快速给出某个局部区域的边际收益轮廓
5. 为 autoresearch 提供更高密度的局部证据

这说明它不是单纯的对照物，而是对研究链路本身有用。

### 5.3 把 AutoML 只当 control group 会浪费它的结构价值

如果 AutoML 已经能被做成：

- 一个可调度的局部探索器
- 一个稳定性验证器
- 一个超参边际收益分析器

那把它只定义成对照组，实际上是在浪费当前框架的潜力。

因此更准确的说法不是：

> 我们保留一点 AutoML 作为参考。

而是：

> **我们把 AutoML 正式纳入研究生态，作为被 autoresearch 调度与消费的局部探索系统。**

---

## 6. 当前框架里，这种结构为什么有生存空间

### 6.1 因为当前代码已经具备实验分层基础

当前仓库已经具备支撑这种结构的若干前提：

1. 训练运行本身已具备明确输入参数面
2. tracker 已经能记录 run 级元数据
3. analyze 已经能做对比、frontier、stability、lineage 等分析
4. benchmark / exploratory 已经是现成语义
5. checkpoint 与 opponent registry 已经构成长期实验生态的一部分

这些前提说明：

双线路共存不是要从零发明，而是在已有结构上完成角色重定义。

### 6.2 因为当前框架适合形成“分层消费”关系

这个结构里，信息流可以自然组织成：

1. 训练脚本产出原始实验结果
2. AutoML 在局部空间做探索并汇总结论
3. tracker / analyze / report 层放大这些结论
4. autoresearch agent 读取这些结果并做下一轮决策

这种信息流在当前项目中是自然的，并不违和。

### 6.3 因为当前仓库最缺的不是能力，而是角色秩序

从技术上讲，当前项目最缺的已经不是再多几个功能，而是：

> **如何明确“谁负责提出方向、谁负责局部搜索、谁负责裁决结果”。**

双线路结构的价值，恰恰就在于它能建立这套秩序。

---

## 7. 这种双线路结构真正独特的价值闭环是什么

### 7.1 不是“两个模式并排摆着”

真正有价值的，不是仓库支持两个命令模式，而是形成下面这个闭环：

1. autoresearch 识别当前研究问题
2. autoresearch 判断该问题是否需要局部 AutoML 探索
3. AutoML 在受限空间内完成局部扫描、重复验证与结果汇总
4. AutoML 产出结构化结论
5. autoresearch 消费该结论，并决定：
   - 是否修改训练代码
   - 是否冻结当前方向
   - 是否扩大或缩小搜索空间
   - 是否切换到新的研究问题
6. benchmark 对最终推进方向做裁决

这才是真正不同于纯 autoresearch 和纯 AutoML 的价值闭环。

### 7.2 这个闭环为什么有明显不同的价值

因为它同时利用了三种能力：

1. **agent 的开放式研究能力**
2. **AutoML 的局部搜索效率**
3. **benchmark 的外部事实约束**

三者组合后，系统获得的是一种新的研究机制：

- 不是盲目的网格搜索
- 不是完全依赖 agent 单步试错
- 也不是人类手工做所有高层策划

而是：

> **agent 负责高层研究方向，AutoML 负责低层局部勘探，benchmark 负责判定真实进展。**

这就是它的独特价值。

---

## 8. 这种结构的真实生态位价值

### 8.1 它解决了纯 autoresearch 的一个现实弱点

纯 autoresearch 虽然研究意味很强，但在局部参数面扫描上往往效率不高。

如果所有微调都由 agent 亲自一轮轮试，成本会偏高，噪声也更难抑制。

把 AutoML 作为局部探索子系统后，autoresearch 可以把机械性的局部搜索外包出去，而自己保留高层研究判断。

### 8.2 它解决了纯 AutoML 的一个根本弱点

纯 AutoML 的问题在于：

- 它不擅长形成结构性研究假设
- 它不擅长在行为证据与指标证据之间建立解释
- 它不擅长判断“是否该换问题本身”

而 autoresearch 恰好可以承担这些上层任务。

因此，这种结构让 AutoML 不再假装自己能做全部研究，而是把它放回最擅长的位置。

### 8.3 它能形成新的比较维度

如果以后演化成熟，这个系统甚至可以比较：

1. 纯 AutoML 推进 frontier 的效率
2. 纯 autoresearch 推进 frontier 的效率
3. autoresearch + AutoML 混合体系的推进效率

这会让项目不只是一个训练平台，而变成一个“研究系统结构”的实验场。

---

## 9. 双线路结构下需要守住的边界

### 9.1 不能让 AutoML 反客为主

最需要防止的事情是：

- AutoML 开始决定主要研究方向
- agent 退化成结果解说员
- 所有问题都被压缩成参数搜索

一旦这样发生，系统就会重新滑回 AutoML 主导。

### 9.2 不能让 autoresearch 失去消费 AutoML 产出的能力

如果 AutoML 只是生成一堆数字，而 agent 无法有效读取与消费这些结论，那么双线路会沦为两套割裂系统。

因此，report 层和结果组织方式会非常关键。

### 9.3 不能让 benchmark 退位

无论 AutoML 多强、agent 多强，最终仍然必须由相对稳定的 benchmark 负责裁决真实进展。

---

## 10. 对 v10+ 路线的进一步修正建议

基于这次讨论，对 v10+ 路线应进一步做如下修正：

### 10.1 不再把 AutoML 只定义为防偏离对照组

新的定义应是：

> AutoML 是研究生态中的局部探索层，而不是只用于证明 agent 是否有效的被动对照组。

### 10.2 重新定义系统结构

建议未来正式定义三种系统层级：

1. **autoresearch layer**：研究判断与方向决策
2. **automl layer**：局部搜索、重复验证、结论提取
3. **benchmark layer**：外部裁决

### 10.3 重新定义“模式”

未来如果做模式区分，不应只写成：

- autoresearch mode
- automl mode

而应明确第三种真正关键的模式：

- **hybrid mode**：autoresearch 调度并消费 AutoML 产出

因为真正最有价值的，并不是二者分立，而是二者之间的消费关系。

---

## 11. Statement

基于本轮进一步讨论，建议确立如下 statement：

> **在 MAG Gomoku 中，autoresearch 与 AutoML 都应成为系统的一等公民，但它们承担不同角色。autoresearch 负责读取放大的实验事实、理解问题、做跨实验决策，并消费 AutoML 的探索结论；AutoML 负责接受约束后的局部超参探索、重复验证与结论提取，作为 autoresearch 后续决策的证据来源。AutoML 不再只是并行 control group，而是研究生态中具有真实用途的局部探索子系统。只有在这种主从分层与结果消费关系下，项目才能形成明显不同于纯 autoresearch 或纯 AutoML 的独特价值闭环。**

---

## 12. Verdict

本轮进一步讨论的最终 verdict 如下：

1. **当前项目完全有空间容纳 autoresearch 与 AutoML 双线路共存。**
2. **二者都应是一等公民，但绝不能把“一等公民”误解为对称关系。**
3. **autoresearch 必须继续担任研究决策中心。**
4. **AutoML 应被提升为真实有用的局部探索层，而不是只作为 control group 存在。**
5. **真正的系统价值不在于“双模式并存”，而在于形成“autoresearch 做决定，AutoML 产出局部证据，benchmark 做裁决”的闭环。**
6. **只有这样，MAG Gomoku 才能形成一个明显不同于现成 AutoML 框架、也不同于原版极简 autoresearch 的独特生态位。**

一句话总结：

> **不是让 AutoML 和 autoresearch 并排共存，而是让 autoresearch 消费 AutoML；不是让 AutoML 作为防偏离的附属对照，而是让它成为研究生态里真正有用的局部探索器。只有这样，项目才会形成一个清晰、完整、具有明显差异化价值的闭环。**

---

## 13. 关于 v10 与 v11 的预定工作

基于当前已经完成的 `update-v10.md` 设计稿，以及本份 further analysis 后形成的新认识，有必要把 v10 与 v11 的职责边界重新切清。

核心原则应当是：

> **v10 负责激活 autoresearch 主循环并做好承接准备，v11 再正式固化 autoresearch layer / AutoML layer / hybrid mode 的结构分工。**

换句话说，v10 是“让 agent 重新开始研究”，v11 才是“让 AutoML 正式成为被 autoresearch 消费的局部探索层”。

### 13.1 我对当前 v10 方案的总体评价

当前 `update-v10.md` 的大方向是正确的，而且与本份 further analysis 的主旨并不冲突。

它最正确的地方有三个：

1. **明确了 autoresearch 闭环的激活优先级。**
   v10 把重点放在 `analyze.py --report`、`program.md` v2、首次闭环试运行，这一点非常关键。因为在当前阶段，最优先的问题仍然是让 agent 真正回到主控位，而不是立刻把整个双层系统一次性实现完。

2. **明确拒绝把 hook / LLM API 直接内嵌进训练脚本。**
   这与 further analysis 的边界完全一致。训练脚本继续负责产出事实，agent 在外层做判断，这是必须保留的结构纪律。

3. **把报告层放在了 v10 的中心位置。**
   这也是正确的，因为无论后面要不要引入 AutoML layer，都必须先让 agent 能稳定消费结构化实验事实。

因此，v10 的总方向不用推翻，反而应被保留为：

> **回归 autoresearch 主路线的第一次正式启动。**

### 13.2 我对当前 v10 方案的主要修正建议

虽然 v10 主方向是正确的，但基于 further analysis，有几处需要重新定性或降级处理。

#### 第一，P4 中的 Control Group 叙事应降级或改写

当前 v10 设计里，AutoML 仍较明显地被放在“control group comparison”的位置。

这在 earlier reasoning 中是合理的，但在 further analysis 之后，这个表述已经不够准确。

更合适的处理方式是：

1. **v10 不再把 AutoML 主要定义为 control group。**
2. **v10 最多只保留 AutoML 的“预备入口”和“未来角色说明”。**
3. **真正的 AutoML layer 正式固化，放到 v11。**

也就是说，v10 可以承认未来会有 AutoML，但不必在本版本把它作为完整对照组叙事展开。

#### 第二，`sweep.py` 不应在 v10 中被拔得过高

如果 v10 立刻把 `sweep.py`、局部搜索矩阵、AutoML benchmark 全部做成正式主组件，就会有两个风险：

1. 重新稀释掉刚刚恢复的 autoresearch 主线
2. 让 v10 从“激活主循环”膨胀成“同时搭双层系统”

因此，我建议：

- **v10 可以为 `sweep_tag` / `seed` / report 兼容性等元数据做准备**
- **但不要在 v10 中把 AutoML layer 做成主叙事或主交付物**

#### 第三，v10 的成功标准应更聚焦

v10 的成功，不应定义为：

- 同时跑通 autoresearch 与 AutoML
- 同时完成 control benchmark
- 同时验证混合系统的研究效率

v10 更合理的成功标准应是：

1. agent 能稳定读取 `--report`
2. agent 能按 `program.md` 持续做 3-5 轮闭环实验
3. 我们能确认 autoresearch 主控制权已经恢复

只要这三点成立，v10 就完成了它的历史任务。

### 13.3 v10 预定工作：建议保留在本版本 scope 内的内容

基于上面的判断，我建议 v10 严格保留以下工作内容：

1. **`analyze.py --report`**
   这是 v10 最核心的技术交付。它既服务当前 autoresearch 激活，也会成为 future AutoML layer 被消费的统一报告面。

2. **`docs/program.md` v2 升级**
   让 agent 明确使用报告、明确 keep/discard 逻辑、明确 benchmark 纪律、明确本版本开始进入真正 autoresearch phase。

3. **README 叙事统一**
   对外明确项目已经从基础设施阶段转入 agent-driven autoresearch phase。

4. **首次闭环试运行**
   先验证外层 agent loop 是否顺畅，而不是急于验证双层体系是否成熟。

5. **对 future AutoML layer 的轻量预留**
   可以保留：
   - `seed`
   - `sweep_tag`
   - report 中对 grouped runs 的兼容
   - 对 future `sweep.py` 的文档预告

这些内容可以作为 v11 的前置准备，但不应在 v10 中喧宾夺主。

### 13.4 v10 不建议在本版本强推的内容

以下内容我建议明确延后到 v11：

1. **正式的 AutoML layer 定义与实现**
2. **hybrid mode 的完整协议**
3. **autoresearch 消费 AutoML 结论的正式工作流**
4. **control benchmark / mixed benchmark 的制度化运行方式**
5. **大规模 `sweep.py` 叙事和矩阵实验主线化**

原因很简单：

> v10 的首要目标是把“外层 agent 主控权”重新接回系统；在这一步尚未稳定之前，过早正式引入 AutoML layer，会重新模糊项目重心。

### 13.5 v11 预定工作：进一步固化 further analysis 的内容

如果 v10 顺利完成，那么 v11 就应成为真正承接本份 further analysis 的版本。

v11 建议正式承担的任务包括：

1. **定义 autoresearch layer / AutoML layer / benchmark layer**
   把三层结构从概念写成正式系统边界。

2. **定义 hybrid mode**
   明确：
   - 何时由 autoresearch 发起 AutoML 局部探索
   - AutoML 返回什么样的结构化结论
   - agent 如何消费这些结论并做后续决策

3. **定义 AutoML 的真实角色**
   不再只是 control group，而是：
   - 局部参数扫描器
   - 重复验证器
   - 稳定性证据生产者
   - 局部边际收益分析器

4. **扩展 report 层**
   让报告不仅能服务 agent 单步闭环，也能服务 autoresearch 消费 grouped sweep / local search 结果。

5. **定义比较框架**
   到 v11，再去正式讨论：
   - 纯 autoresearch
   - 纯 AutoML
   - hybrid system
   三者之间的效率比较与生态位区别。

### 13.6 对 v10 的一句话建议

如果要把我对 v10 的建议压成一句话，那就是：

> **v10 不要急着把双层体系全部做完；v10 先把 autoresearch 主循环激活并把报告层打通，v11 再把 AutoML 正式提升为被 autoresearch 消费的局部探索层。**

### 13.7 最终承接 Verdict

基于当前 `update-v10.md` 与本份 further analysis 的综合判断，最终建议如下：

1. **v10 方向总体正确，应继续执行。**
2. **v10 应保持克制，聚焦 autoresearch loop activation。**
3. **AutoML 在 v10 中只做预留，不做全面主线化。**
4. **v11 才是固化双线路分工与 hybrid mode 的合适版本。**
5. **只有按这个节奏推进，项目才能既不失去 v10 的清晰性，也不失去 further analysis 所指出的更高阶价值。**

一句话总结：

> **v10 负责把 agent 真正带回实验室，v11 负责把 AutoML 变成 agent 可以调度和消费的工具层；前者解决“谁来研究”，后者解决“研究如何借助局部自动探索变得更强”。**
