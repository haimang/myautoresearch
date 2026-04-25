# 训练一个真正能下棋的五子棋模型

> 2026-04-12 | 根因分析 · 改造方案 · 实施路线

---

## 1. 当前模型为什么不能下棋

L0 到 L4，包括正在训练的 L5，全部共享同一个核心缺陷：**模型没有学到五子棋的基本战术逻辑**。

具体表现：

1. 不会堵对方连四
2. 不会抓自己的四连胜
3. 不会做活三、不会堵活三
4. 不会做出任何有目的性的形状构建

这不是超参数的问题，不是训练时间不够的问题，也不是模型容量不够的问题。**是训练信号本身有结构性缺陷**。

---

## 2. 根因分析

当前训练流水线有三个致命问题。

### 2.1 策略目标 = 模型自己的采样分布

在 `run_self_play()` 中（train.py:323-418），每一步的 policy target 是这样生成的：

```python
scaled = game_logits / temp
probs = softmax(scaled)
action = np.random.choice(225, p=probs)
policy_dist = probs  # ← 这就是 policy target
```

然后在 `compute_loss()` 中：

```python
policy_loss = -mean(sum(batch_policies * log_probs, axis=-1))
```

**这意味着模型在学习复制自己的行为分布**。如果模型当前认为某个烂位置有 30% 概率，那 policy target 就会告诉它：这个烂位置应该有 30% 概率。

这是一个自我强化的反馈回路。模型产出噪声 → 噪声变成训练目标 → 模型学会产出更稳定的噪声 → loss 下降但棋力没涨。

### 2.2 价值信号 = 仅终局胜负

在 `_collect_game_with_policies()` 中（train.py:514-566）：

```python
if winner == player:
    value_target = 1.0
elif winner == -1:
    value_target = 0.0
else:
    value_target = -1.0
```

**整盘棋所有位置的 value target 都是同一个数**。第 5 步和第 50 步如果都输了，value target 都是 -1.0。

这种信号的问题：

- 一盘棋 50 步里最多只有 2-3 步是真正的关键转折点
- 但模型无法区分哪一步导致了输赢
- 信用分配问题（credit assignment）几乎完全没解决
- 模型的 value head 学到的只是"这局最后谁赢了"的统计相关性，不是"这个局面有多好"

### 2.3 自对弈的数据分布 = 两个弱鸡互打

自对弈生成的棋谱质量完全取决于模型本身的水平。两个不会下棋的模型打出来的棋谱里：

- 没有正确的进攻样本（模型不会做活三，也就不会出现"做出活三后赢了"的数据）
- 没有正确的防守样本（模型不会堵连四，也就不会出现"堵住之后赢了"的数据）
- 随机乱下 → 随机获胜/失败 → 模型学到的"策略"就是随机的

即使加了 opponent play（vs L4），L4 本身也是同样的弱鸡。模型在对弈中看不到任何有章法的棋，自然学不会有章法的下法。

### 2.4 评估指标的虚假膨胀

对手使用 temperature=0.5 采样（train.py:1289）：

```python
def _nn_opponent_move(opp_model, board, temperature: float = 0.5):
```

这意味着 L4 在评估时也在随机采样，不是全力出招。WR 数字包含大量对手随机犯错带来的"赠分"。80% WR 并不代表能赢一个会下棋的对手 80%。

---

## 3. AlphaGo/AlphaZero 做对了什么

AlphaZero 解决了上述所有问题，靠的是 **MCTS（蒙特卡洛树搜索）**：

| 问题 | 当前做法 | AlphaZero 做法 |
|------|----------|----------------|
| Policy target | 模型自己的采样分布 | MCTS 搜索后的访问计数分布 |
| Value target | 终局胜负（全局一个值） | MCTS 回传的搜索值（每步不同） |
| 数据质量 | 靠当前弱模型随机采样 | MCTS 搜索产生强度远超原始网络的走法 |
| 信用分配 | 完全没有 | MCTS 天然为每步提供局部评估 |

**核心洞察：MCTS 是一个"计算换质量"的放大器。网络给出粗糙的策略和评估 → MCTS 通过搜索把它放大成高质量的走棋和评估 → 高质量数据反过来训练网络。**

没有 MCTS，网络只能从自己的噪声中学习。有了 MCTS，网络能从远强于自己的搜索结果中学习。这是自对弈能成功的根本前提。

---

## 4. 改造方案

### 方案 A：加入 MCTS（推荐，正道）

这是 AlphaZero 原版路线，也是目前已知的唯一能让纯自对弈产生强棋力的方法。

#### 4.1 MCTS 核心流程

每一步走棋时，执行 N 次 MCTS 模拟（N=200~800），每次模拟：

1. **选择（Select）**：从根节点沿树走到叶子节点，每一步选择使 PUCT 分数最高的子节点
2. **扩展（Expand）**：遇到未展开的叶子节点，用神经网络评估该位置，得到 (policy, value)
3. **回传（Backup）**：把 value 沿搜索路径回传，更新每个节点的统计值

PUCT 公式（AlphaZero 版本）：

$$U(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$

- $Q(s, a)$：动作 $a$ 的平均回传值
- $P(s, a)$：网络给出的先验概率
- $N(s)$：父节点访问次数
- $N(s, a)$：动作 $a$ 的访问次数
- $c_{\text{puct}}$：探索常数（通常 1.0~2.5）

模拟结束后：

- **Policy target** = 根节点各动作的访问计数分布：$\pi(a) = N(root, a)^{1/\tau} / \sum_b N(root, b)^{1/\tau}$
- **Value target** = 终局胜负（不变，但因为棋谱质量高得多，信号密度天然更高）
- **实际走棋** = 从 $\pi$ 中采样（训练时）或取最大访问次数（评估时）

#### 4.2 训练信号的本质变化

| 信号 | 之前 | 加入 MCTS 后 |
|------|------|-------------|
| Policy target | 网络自身 softmax 输出 | 200+ 次模拟的搜索结果 |
| 信号质量 | 等于当前网络水平 | 远强于当前网络水平 |
| 策略改进方向 | 模仿自己 → 原地打转 | 模仿搜索 → 持续提升 |
| 一步必杀/必堵 | 网络不知道 → 不学 | MCTS 搜索到 → policy target 集中在该点 → 学到 |

#### 4.3 实现代价

MCTS 的代价是**速度**。每步需要 N 次前向传播（= N 次网络调用）。

对当前 M3 Max 硬件的估计：

| 参数 | 值 |
|------|-----|
| 每次前向传播（单棋盘） | ~0.1ms |
| MCTS 模拟次数 / 步 | 200 |
| 每步搜索时间 | ~20ms |
| 每盘棋 ~50 步 | ~1s / 盘 |
| 64 盘并行 | ~64s / cycle（vs 当前 ~2s） |

训练会慢 **30-50 倍**。但棋谱质量会高几个数量级。**少量高质量棋谱 >> 大量垃圾棋谱**。

#### 4.4 批量 MCTS 优化

Apple Silicon 的 MLX 对批量推理很高效。可以通过以下方式大幅缓解速度问题：

1. **虚拟损失（Virtual Loss）**：允许同时展开多条搜索路径，把待评估的叶子节点凑成 batch
2. **叶子节点批量评估**：攒 16-64 个待评估节点一起推理，摊薄 GPU 调用开销
3. **多棋盘叶子合并**：多盘同时搜索，叶子节点跨棋盘合并成大 batch

实际期望加速比：10-20x。即搜索代价从 30x 降到 2-3x。

#### 4.5 最小可行 MCTS 实现

初版不需要实现 AlphaZero 论文中的所有优化。最小版本：

```
MCTSNode:
    state: Board
    parent: MCTSNode?
    children: dict[action, MCTSNode]
    visit_count: int
    value_sum: float
    prior: float

def mcts_search(root_board, model, num_simulations):
    root = MCTSNode(state=root_board)
    
    for _ in range(num_simulations):
        node = root
        # 1. Select
        while node.is_expanded and not node.state.is_terminal():
            node = node.select_child()  # PUCT
        
        # 2. Expand + Evaluate
        if not node.state.is_terminal():
            policy, value = model(node.state.encode())
            node.expand(policy)  # create children with priors
        else:
            value = terminal_value(node.state)
        
        # 3. Backup
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # flip for opponent
            node = node.parent
    
    # Return visit count distribution as policy target
    visits = {a: child.visit_count for a, child in root.children.items()}
    return visits
```

#### 4.6 实施步骤

1. 在 train.py 中实现 `MCTSNode` 和 `mcts_search()`
2. 修改 `run_self_play()` 将每步的 policy target 从 softmax 采样改为 MCTS 搜索分布
3. 添加 `--mcts-sims` 参数控制模拟次数（默认 200）
4. 保留 `MCTS_SIMULATIONS = 0` 作为退化到旧行为的开关
5. 先跑小规模验证（`--mcts-sims 50 --parallel-games 8`），确认搜索结果合理
6. 确认后上量（`--mcts-sims 200 --parallel-games 32`）

### 方案 B：战术教师注入（快速启动，过渡方案）

在不实现 MCTS 的情况下，通过在数据生成阶段注入有确定性正确答案的战术局面来部分解决数据质量问题。

#### 4.7 战术教师的概念

在自对弈中，如果当前局面存在以下情况之一：

1. **一步必胜**（己方有连五机会）
2. **一步必堵**（对方有连五机会必须堵）
3. **做活四**（己方可以形成不可阻挡的双四/活四）
4. **堵活四**（对方有活四威胁必须处理）

那么 policy target 不用模型自己的采样分布，而是直接设置为正确答案的 one-hot 或小范围集中分布。

```python
def get_tactical_target(board, player):
    """如果当前是战术必然点，返回正确的 policy target；否则返回 None。"""
    # 一步必胜
    winning_move = find_winning_move(board, player)
    if winning_move is not None:
        target = np.zeros(225)
        target[winning_move[0] * 15 + winning_move[1]] = 1.0
        return target
    
    # 一步必堵
    opponent = other_player(player)
    threat_move = find_winning_move(board, opponent)
    if threat_move is not None:
        target = np.zeros(225)
        target[threat_move[0] * 15 + threat_move[1]] = 1.0
        return target
    
    # 非战术局面，退回模型自己的分布
    return None
```

#### 4.8 战术教师能解决什么

| 问题 | 能否解决 |
|------|---------|
| 一步必胜/必堵 | **完全解决** — 直接给正确答案 |
| 活三/冲四构建 | **部分解决** — 需要扩展搜索深度 |
| 中盘战略布局 | **不解决** — 非战术局面仍依赖自对弈噪声 |
| credit assignment | **不解决** — value target 不变 |

#### 4.9 为什么战术教师不够

战术教师只修复了"知道正确答案时不告诉模型"的问题，但没有解决：

1. 非战术局面（占 80%+ 的走棋）的 policy target 仍然是垃圾
2. value signal 仍然太稀疏
3. 模型仍然不知道如何主动构建进攻形状

**战术教师是止血措施，不是治本方案。** 但它可以在几小时内实现，让模型至少不会犯一步棋就能看出来的低级错误。

---

## 5. 推荐路径

分两阶段推进，先止血、后治本。

### Phase 1：战术教师 + 评估基准（1-2 天）

**目标**：让模型能通过基本战术测试，提供量化基准。

1. 实现战术扫描函数（复用 prepare.py 的 pattern 检测逻辑）
2. 在 `run_self_play()` 中注入战术教师 policy target
3. 在 `run_opponent_play()` 中同样注入
4. 新增 `--tactical-teacher` CLI flag（默认开启）
5. 新增战术基准测试：
   - 50 个"一步必胜"局面 → 模型选对的比例
   - 50 个"一步必堵"局面 → 模型选对的比例
   - 50 个"活三构建"局面 → 模型选对的比例
6. 在 probe eval 中加入战术通过率指标

**预期效果**：

- 训练后模型应能 100% 识别一步必胜和一步必堵
- WR vs 旧 L4 应明显上升，因为旧 L4 不会堵

### Phase 2：MCTS 自对弈（1-2 周）

**目标**：实现真正的策略提升引擎。

1. 实现基础 MCTS（单线程，无批量优化）
2. 验证搜索结果合理性（搜索策略应明显强于原始网络）
3. 接入训练循环，用 MCTS policy target 替代采样分布
4. 实现批量叶子评估优化
5. 调节 `c_puct`、模拟次数、温度衰减
6. 在战术基准上验证 MCTS 训练的模型表现
7. 完整训练：从随机初始化开始，跑到能稳定赢 minimax L3
8. 继续训练：对手升级到自身之前的 checkpoint，形成 AlphaZero 式自我进化链

**预期效果**：

- MCTS 200 sims 训练的模型应在 2000-5000 盘后达到 minimax L3 水平
- 继续训练应能远超 minimax L3（minimax 深度搜索有限，MCTS 无此限制）
- 在浏览器中对弈时，模型应表现出明确的战术意识和连贯进攻

---

## 6. 训练参数建议

### 6.1 Phase 1（战术教师）

基本沿用现有参数，改动最小：

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| num_blocks | 6 | 6 | 容量够用 |
| num_filters | 64 | 64 | 不变 |
| learning_rate | 5e-4 | 5e-4 | 不变 |
| steps_per_cycle | 30 | 30 | 不变 |
| parallel_games | 64 | 64 | 不变 |
| tactical_teacher | N/A | **true** | 新增 |
| temperature | 1.0 | 1.0 | 不变 |

### 6.2 Phase 2（MCTS 训练）

MCTS 改变了数据生成的性质，需要调整：

| 参数 | 建议值 | 理由 |
|------|--------|------|
| mcts_sims | 200 | 搜索质量与速度的平衡点 |
| num_blocks | 6 | 先保持，后续可增加 |
| num_filters | 64 | 先保持 |
| learning_rate | 2e-3 → 2e-4 | 余弦退火；MCTS target 更干净，可容忍更大初始 LR |
| steps_per_cycle | 100 | 每 cycle 棋谱少但质量高，训练步数应增加 |
| parallel_games | 16 | MCTS 单盘耗时更长，并行数适当减少 |
| replay_buffer | 100000 | MCTS 棋谱复用价值更高，buffer 可放大 |
| temperature | 1.0 → 0.3 | 前 30 步高温探索，之后低温 |
| c_puct | 1.5 | 标准值 |
| dirichlet_alpha | 0.03 | 根节点噪声，增加探索多样性 |
| dirichlet_frac | 0.25 | 标准值 |

### 6.3 评估参数

不论 Phase 1 还是 Phase 2：

| 参数 | 建议值 | 理由 |
|------|--------|------|
| eval 时 temperature | **0** | 评估必须 argmax，不随机 |
| eval 时 MCTS sims | **0** | 评估纯网络；或另设（如 100）看搜索加持后表现 |
| eval 对手 temperature | **0** | NN 对手评估时也应 argmax |
| 战术基准 | 每次 checkpoint | 量化基本功 |

> **重要**：当前评估中 NN 对手的 temperature=0.5 必须改为 0。否则 WR 永远被对手的随机失误虚假膨胀。

---

## 7. 自我对弈进化链设计

Phase 2 成熟后，训练路线应改为 AlphaZero 风格的自我进化：

```text
随机初始化
  → MCTS 自对弈训练 2000 盘
  → checkpoint A（能赢 minimax L2）
  → 注册为对手 "S1"

继续训练 vs S1
  → MCTS 自＋对手混合训练
  → checkpoint B（能赢 S1 80%+）
  → 注册为对手 "S2"

继续训练 vs S2
  → ...
  → checkpoint C
  → 注册为对手 "S3"
```

与当前 L0-L4 链的区别：

| 维度 | 当前链 | MCTS 链 |
|------|--------|---------|
| 数据质量 | 垃圾 | MCTS 搜索后的高质量走法 |
| 对手强度 | 弱鸡 vs 弱鸡 | 逐级增强 |
| 策略进步幅度 | 几乎没有 | 每代可测量的提升 |
| 战术能力 | 不存在 | 搜索天然发现战术 |

---

## 8. 预期里程碑

| 里程碑 | 可行判据 | 预估训练量 |
|--------|---------|-----------|
| 能 100% 识别一步必胜/必堵 | 战术基准通过率 >95% | Phase 1 + 5000 盘 |
| 能赢 minimax L2 | WR >80% (temp=0) | Phase 2 + 2000 盘 |
| 能赢 minimax L3 | WR >70% (temp=0) | Phase 2 + 5000 盘 |
| 能与人类初学者正常对弈 | 不犯低级错误，有进攻意图 | Phase 2 + 10000 盘 |
| 能赢大多数业余棋手 | 具备多步策略能力 | Phase 2 + 50000+ 盘 |

---

## 9. 需要改动的文件

### 9.1 Phase 1

| 文件 | 改动 |
|------|------|
| `src/train.py` | 新增 `get_tactical_target()` 函数；修改 `run_self_play()` 和 `run_opponent_play()` 注入战术 target；新增 `--tactical-teacher` CLI flag |
| `src/train.py` | 新增战术基准评估函数 `evaluate_tactical_accuracy()` |
| `src/train.py` | 修改 `_nn_opponent_move()` 评估时 temperature 默认改为 0 |
| `src/train.py` | 修改 `_in_process_eval()` 评估时 NN 对手 temperature=0 |

### 9.2 Phase 2

| 文件 | 改动 |
|------|------|
| `src/train.py` | 新增 `MCTSNode` 类和 `mcts_search()` 函数 |
| `src/train.py` | 修改 `run_self_play()` 用 MCTS 替代温度采样 |
| `src/train.py` | 新增 `--mcts-sims`、`--c-puct`、`--dirichlet-alpha` CLI flags |
| `src/train.py` | 新增批量叶子评估优化 |

---

## 10. 与现有设施的关系

| 现有设施 | Phase 1 | Phase 2 |
|---------|---------|---------|
| 自对弈循环 | 保留，注入战术 target | 保留，MCTS 替代温度采样 |
| Replay buffer | 不变 | 增大到 100K |
| D4 对称增广 | 不变 | 不变 |
| Signal-density 采样 | 不变 | 保留但重要性降低 |
| NN opponent 体系 | 不变 | 对手逐级进化 |
| 浏览器前端 | 不变 | 不变，可选开启 MCTS 推理 |
| CLI 对弈 | 不变 | 不变 |
| Tracker / checkpoint | 不变 | 不变 |

---

## 11. 关于当前 L4 和 L5

**L4 是什么**：10x64 模型，vs L3 达到 83.6% WR。听起来不错，但 L3 本身也是弱鸡，而且评估时对手用 temperature=0.5 随机采样。L4 在浏览器里表现就是：随机落子，不堵不攻。

**正在训练的 L5（16x64）是什么**：更大的模型 vs L4，目前 ~22% WR。即使最终训练到 85% WR，也只是在赢一个不会下棋的对手。棋力不会有质的提升。

**结论**：只要训练信号不改，L6、L7、L100 都不会比 L4 强到哪去。模型大小、学习率、训练时长都是二阶因素。一阶因素是：**模型在学什么**。

当前模型在学复制自己的随机行为。需要改成：学搜索发现的正确行为。

---

## 12. 总结

一句话：**不上 MCTS，不可能训出会下棋的模型。**

战术教师可以在短期内让模型通过最低限度的战术测试，但中盘、布局、多步进攻这些能力只有 MCTS 能教。

推荐现在立刻做 Phase 1（战术教师），然后启动 Phase 2（MCTS）。Phase 1 的代码量很小（~100 行），Phase 2 的 MCTS 核心实现也不大（~200 行），但它是自对弈训练从"原地打转"到"真正进化"的分水岭。
