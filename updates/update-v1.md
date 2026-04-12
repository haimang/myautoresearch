# Action Plan: 五子棋 autoresearch

用 autoresearch 范式训练一个五子棋 AI，在 M3 Max 128GB 上运行。

---

## 1. 核心思路

把 autoresearch 的三文件架构直接映射到五子棋：

| 原版 (LLM) | 五子棋版 | 职责 |
|---|---|---|
| `prepare.py` (不可修改) | `prepare.py` | 棋盘引擎、固定对手、评估函数、数据接口 |
| `train.py` (AI 可修改) | `train.py` | 神经网络架构、自对弈逻辑、训练循环 |
| `program.md` | `program.md` | 自主实验协议 |

**唯一指标**: `win_rate`（对固定对手的胜率），等价于原版的 `val_bpb`。

---

## 2. 项目结构

```
autoresearch-mlx/
├── gomoku/
│   ├── game.py             # [Phase 0] 五子棋游戏本体（pygame 渲染 + 棋盘逻辑）
│   ├── prepare.py          # [不可修改] 对手 AI + 评估 + 录制系统
│   ├── train.py            # [可修改]   NN 架构 + 训练循环
│   ├── program.md          # 实验协议
│   ├── play.py             # 人机对弈入口（加载 checkpoint 对战）
│   ├── replay.py           # 回放录像 + 渲染视频帧
│   ├── results.tsv         # 实验日志
│   ├── pyproject.toml      # 依赖 (mlx, numpy, pygame)
│   ├── checkpoints/        # 里程碑 checkpoint 存档（不被 git reset 影响）
│   │   ├── stage0_baseline.safetensors
│   │   ├── stage1_beat_random.safetensors
│   │   ├── stage2_beat_minimax2.safetensors
│   │   ├── ...
│   │   └── manifest.json   # checkpoint 元数据索引
│   └── recordings/         # 训练过程录制数据（视频素材原料）
│       ├── games/          # 棋局记录（JSON 格式的完整对局）
│       ├── metrics/        # 训练指标时间线（CSV）
│       └── frames/         # 关键帧截图（PNG）
├── prepare.py              # 原版 LLM 训练（保持不动）
├── train.py
├── program.md
└── ...
```

---

## 3. prepare.py 设计（不可修改部分）

### 3.1 棋盘引擎

```python
BOARD_SIZE = 15         # 标准五子棋棋盘
WIN_LENGTH = 5          # 五子连珠

class Board:
    """棋盘状态管理"""
    def __init__(self)
    def place(self, row, col, player) -> bool  # 落子
    def is_legal(self, row, col) -> bool       # 合法性检查
    def check_win(self) -> int                 # 0=未结束, 1=黑赢, 2=白赢
    def get_legal_moves(self) -> list          # 所有合法位置
    def encode(self) -> ndarray                # 编码为 NN 输入 [3, 15, 15]
```

### 3.2 编码方式

```
通道 0: 当前玩家的棋子 (1=有子, 0=无子)
通道 1: 对手的棋子
通道 2: 全 1 或全 0 (表示当前轮到谁)
```

输入 shape: `[batch, 3, 15, 15]`

### 3.3 固定对手（写在 prepare.py 中，不可修改）

| 等级 | 对手类型 | 实现方式 | 搜索深度 | 每步耗时 |
|---|---|---|---|---|
| L0 | 随机 | 随机选合法位置 | - | < 0.01ms |
| L1 | 弱 minimax | alpha-beta + 仅搜相邻空位 | depth=2 | ~1ms |
| L2 | 中 minimax | alpha-beta + 威胁检测 | depth=4 | ~20ms |
| L3 | 强 minimax | alpha-beta + 棋型评估 + 杀棋搜索 | depth=6 | ~200ms |

**关键优化**：minimax 只搜索已有棋子周围 2 格范围内的空位（candidate moves pruning），这使得 15x15 棋盘的搜索和 9x9 棋盘复杂度相当。

### 3.4 评估函数

```python
# 固定时间预算: 5 分钟训练
TIME_BUDGET = 300

def evaluate_win_rate(model_path: str, level: int = 2, n_games: int = 200) -> dict:
    """
    加载模型，对阵指定等级的 minimax 对手下 n_games 盘。
    黑白各半（消除先手优势）。
    
    返回:
        {
            "win_rate": 0.73,       # 主指标（agent 用此判断 keep/revert）
            "wins": 146,
            "losses": 38,
            "draws": 16,
            "avg_game_length": 47.2,
            "level": 2
        }
    """
```

**评估耗时估算（Level 2，200 盘）**：
- 每盘平均 50 步，其中 NN 走 25 步（< 1ms/步），minimax 走 25 步（~20ms/步）
- 200 盘 × 25 步 × 20ms = 100 秒 ≈ 1.5 分钟
- 加上 NN 推理和开销 → 总评估时间约 **2 分钟**
- 单次实验总耗时 = 5 min 训练 + 2 min 评估 + 0.5 min 启动 = **~7.5 分钟**

### 3.5 输出格式

```
---
win_rate:         0.7300
eval_level:       2
wins:             146
losses:           38
draws:            16
avg_game_length:  47.2
training_seconds: 300.0
total_seconds:    452.3
peak_vram_mb:     1842.0
num_params_K:     876.5
```

---

## 4. M3 Max GPU 利用率：批量并行策略

### 4.1 问题

五子棋 NN 很小（~1-2M 参数），单次推理 < 0.1ms。如果逐个对局做自对弈，GPU 利用率极低（< 1%）。

### 4.2 解决方案：批量自对弈（Batched Self-Play）

同时开 N 盘棋局，每步棋做一次批量推理：

```python
# 核心循环伪代码
parallel_games = 256  # 同时进行 256 盘棋

boards = [Board() for _ in range(parallel_games)]

while not all_finished:
    # 1. 把所有棋盘编码成一个 batch
    batch = stack([b.encode() for b in boards])  # [256, 3, 15, 15]
    
    # 2. 一次批量推理
    policies, values = model(batch)  # GPU 并行处理 256 个位置
    
    # 3. 每盘棋各自落子
    for i, board in enumerate(boards):
        move = sample_move(policies[i], temperature=1.0)
        board.place(move)
    
    # 4. 已结束的棋局 → 收集数据 → 开新棋局
    for i, board in enumerate(boards):
        if board.is_terminal():
            replay_buffer.add(board.history)
            boards[i] = Board()  # 重置，立即开始新一盘
```

### 4.3 M3 Max 吞吐量估算

| 组件 | 配置 | 吞吐量 |
|---|---|---|
| 批量推理 | batch=256, model ~1M params | ~50,000 positions/sec |
| 批量训练 | batch=512, SGD step | ~20,000 positions/sec |
| 自对弈产出 | 256 并行 × 50 步/局 | ~2,500 完整棋局/min |

**5 分钟内的训练量**：

```
自对弈产出:   ~12,000 盘棋局 → ~300,000 个 (棋盘, 策略, 胜负) 样本
训练消耗:     300,000 样本 × 3 epoch = 900,000 梯度更新样本
实际训练步:   900,000 / 512 (batch) ≈ 1,750 gradient steps
```

这已经足够训练一个小型网络。GPU 利用率通过大 batch 拉满。

### 4.4 自对弈与训练的交替策略

5 分钟内不是先自对弈完再训练。而是交替进行（类似 AlphaZero）：

```
Phase 1 (0-30s):   用随机初始化的 NN 自对弈 → 攒数据到 buffer
Phase 2 (30s-5m):  循环 {
                      自对弈 256 盘 → 加入 buffer（~5 秒）
                      训练 100 步 on buffer samples（~3 秒）
                    }  ← 约 35 个 cycle
```

agent 可以调的参数：
- `parallel_games`: 并行棋局数（128 / 256 / 512）
- `train_steps_per_cycle`: 每轮训练步数
- `replay_buffer_size`: 数据缓冲区大小
- `temperature`: 探索温度（自对弈时的随机性）
- `mcts_simulations`: MCTS 模拟次数（0 = 纯策略网络，无搜索）

---

## 5. 神经网络初始架构（train.py 默认值）

Agent 从这个 baseline 开始修改：

```python
# === 超参数 (agent 可调) ===
NUM_RES_BLOCKS = 6        # 残差块数量
NUM_FILTERS = 64          # 每层卷积通道数
LEARNING_RATE = 0.001     # Adam 学习率
WEIGHT_DECAY = 1e-4       # L2 正则化
BATCH_SIZE = 512          # 训练 batch size
PARALLEL_GAMES = 256      # 并行自对弈棋局数
MCTS_SIMULATIONS = 0      # 起步不用 MCTS，纯策略网络
TEMPERATURE = 1.0         # 自对弈温度
REPLAY_BUFFER_SIZE = 50000
TRAIN_STEPS_PER_CYCLE = 100

# === 网络结构 ===
class GomokuNet(nn.Module):
    """
    输入:  [batch, 3, 15, 15]
    主体:  Conv 3x3 → N 个残差块 (Conv-BN-ReLU-Conv-BN + skip)
    输出:  Policy head [batch, 225]  (每个位置的落子概率)
           Value head  [batch, 1]    (当前局面胜率 [-1, 1])
    参数量: ~876K (6 blocks × 64 filters)
    """
```

**Agent 可探索的修改方向**：
- 更深/更浅的网络（blocks 数量）
- 更宽/更窄的网络（filters 数量）
- 不同的激活函数（ReLU → SiLU / GELU）
- 加入 Squeeze-and-Excitation 或 Attention
- 尝试 MCTS（加入搜索，牺牲自对弈速度换取数据质量）
- 不同的训练策略（课程学习、优先级采样）
- 不同的 loss 权重（policy loss vs value loss 的比例）

---

## 6. 自动评估与自动 Checkpoint

### 6.1 评估触发流程

```
train.py 运行完毕
    ↓
保存 model.safetensors 到固定路径
    ↓
打印训练统计
    ↓
调用 prepare.evaluate_win_rate("model.safetensors", level=当前阶段)
    ↓
打印评估结果（格式同 3.5 节）
    ↓
Agent 读取 win_rate → 决定 keep/revert
```

### 6.2 双轨 Checkpoint 策略

checkpoint 管理分两层：**git 层（autoresearch 循环用）** 和 **存档层（你体验用）**。

**第一层：git keep/revert（autoresearch 循环的工作 checkpoint）**

```
实验成功 (win_rate 提升):
  → model.safetensors 留在工作目录
  → git commit --amend 把 results.tsv 更新进去
  → 此时 HEAD 就是最新最好的模型

实验失败 (win_rate 没提升):
  → git reset --hard <上一个 keep 的 commit>
  → model.safetensors 自动恢复为上一个最好的版本
```

**第二层：里程碑存档（你手动体验不同阶段的 AI）**

`checkpoints/` 目录存在 `.gitignore` 之外，**不受 git reset 影响**。每次阶段晋升或重大突破时，prepare.py 自动归档：

```python
# prepare.py 中的存档函数
CHECKPOINT_DIR = "~/.cache/gomoku/checkpoints"  # 在用户 home 目录，完全不受 git 影响

def archive_checkpoint(model_path: str, tag: str, metadata: dict):
    """
    存档一个里程碑 checkpoint。
    
    调用时机（写在 train.py 的评估逻辑中）：
      1. 阶段晋升时（beat_random, beat_minimax2, ...）
      2. 每 10 次 keep 实验自动快照
      3. win_rate 创新高时
    
    存档内容：
      ~/.cache/gomoku/checkpoints/
      ├── stage1_beat_random_exp012_wr0.97.safetensors
      ├── stage2_beat_minimax2_exp038_wr0.82.safetensors
      ├── periodic_exp050_wr0.65.safetensors
      ├── best_exp089_wr0.71.safetensors
      └── manifest.json   ← 所有 checkpoint 的索引
    """
```

**manifest.json 示例**：

```json
[
  {
    "file": "stage0_baseline_exp001_wr0.42.safetensors",
    "tag": "stage0_baseline",
    "experiment": 1,
    "win_rate": 0.42,
    "eval_level": 0,
    "timestamp": "2026-04-11T02:30:00",
    "description": "初始 baseline，随机乱下",
    "train_params": {"blocks": 6, "filters": 64, "mcts": 0}
  },
  {
    "file": "stage1_beat_random_exp012_wr0.97.safetensors",
    "tag": "stage1_beat_random",
    "experiment": 12,
    "win_rate": 0.97,
    "eval_level": 0,
    "timestamp": "2026-04-11T04:00:00",
    "description": "首次稳定击败随机对手"
  }
]
```

**你可以随时加载任意阶段的 AI 来体验**：

```bash
# 和最弱的 AI（刚学会下棋）对弈
uv run play.py --checkpoint stage0_baseline

# 和刚学会赢随机的 AI 对弈
uv run play.py --checkpoint stage1_beat_random

# 和最强的 AI 对弈
uv run play.py --checkpoint best

# 列出所有可用 checkpoint
uv run play.py --list-checkpoints
```

### 6.3 阶段晋升机制

当 agent 在某一级别的胜率稳定超过阈值时，program.md 指导它切换到更难的评估对手：

```
Level 0 (随机):      win_rate 稳定 > 0.95 → 晋升到 Level 1 → 自动存档 checkpoint
Level 1 (minimax-2): win_rate 稳定 > 0.80 → 晋升到 Level 2 → 自动存档 checkpoint
Level 2 (minimax-4): win_rate 稳定 > 0.60 → 晋升到 Level 3 → 自动存档 checkpoint
Level 3 (minimax-6): 持续优化，无上限
```

晋升由 agent 自主判断，不需要人工干预。晋升后 results.tsv 记录新的 baseline。每次晋升自动触发 `archive_checkpoint()`。

---

## 7. 迭代轮次估算

### 7.1 每轮实验的时间构成

```
MLX 编译 (首次):       ~30 秒
训练 (固定预算):        300 秒
评估 (200 盘 vs L2):   ~120 秒
Agent 分析 + 改代码:    ~60 秒
───────────────────────────
总计:                   ~8 分钟/轮
```

每小时约 **7-8 次实验**。

### 7.2 各阶段迭代估算

| 阶段 | 目标 | 估计实验次数 | 估计耗时 | 说明 |
|---|---|---|---|---|
| **Stage 0** | 跑通 baseline | 1 次 | 8 min | 确认代码能运行，建立初始 win_rate |
| **Stage 1** | 胜率 > 95% vs 随机 | 3-8 次 | 0.5-1 h | 几乎任何能学习的网络都能做到 |
| **Stage 2** | 胜率 > 80% vs minimax-2 | 10-25 次 | 1.5-3.5 h | 需要学会基本攻防（活三、冲四） |
| **Stage 3** | 胜率 > 60% vs minimax-4 | 25-60 次 | 3.5-8 h | 需要策略深度，可能引入 MCTS |
| **Stage 4** | 胜率 > 50% vs minimax-6 | 50-120 次 | 7-16 h | 深度策略，适合 overnight run |

**总计**：从零到能击败 depth-6 minimax，大约 **90-200 次实验**，**12-28 小时**。

可以分 2-3 个 overnight run 完成：
- 第一晚：Stage 0-2（从零到击败 minimax-2）
- 第二晚：Stage 3（击败 minimax-4）
- 第三晚：Stage 4（挑战 minimax-6）

### 7.3 M3 Max 128GB 资源使用预估

| 指标 | 预估值 |
|---|---|
| 模型参数 | 0.5M - 3M |
| 模型权重文件 | 2-12 MB |
| 训练 peak VRAM | 1-3 GB |
| 统一内存占用 | < 5 GB |
| GPU 利用率（批量自对弈时） | 60-80% |
| 128GB 内存使用占比 | < 4% |

**结论**：M3 Max 128GB 对这个任务来说性能远超需求。瓶颈不在硬件，在于 agent 的创意和评估的稳定性。

---

## 8. results.tsv 格式

```
commit	win_rate	eval_level	memory_gb	status	description
a1b2c3d	0.4200	0	1.2	keep	baseline: 6 res blocks, 64 filters, no MCTS
e5f6g7h	0.6800	0	1.2	keep	increase parallel games to 512
i9j0k1l	0.5100	0	1.3	discard	deeper network 12 blocks (overfit, slower training)
m2n3o4p	0.9700	0	1.2	keep	add temperature annealing during self-play
q5r6s7t	0.8200	1	1.2	keep	promoted to L1, new baseline
```

---

## 9. game.py：Code Bullet 风格——先做游戏

Code Bullet 的视频总是从"我先自己写了这个游戏"开始。我们也一样：**game.py 是整个项目的第一个文件**，也是贯穿始终的核心。

### 9.1 game.py 的角色

game.py 不是训练完才写的附属品，而是项目的地基：

```
game.py 被谁使用:

  1. 你自己       → play.py 导入 game.py，你和 AI 下棋
  2. prepare.py   → 评估时调用 game.py 的棋盘引擎跑对局
  3. train.py     → 自对弈时调用 game.py 的棋盘逻辑（不渲染画面）
  4. replay.py    → 回放录像时用 game.py 渲染画面
  5. 视频录制      → game.py 支持逐帧导出 PNG，用于合成视频
```

### 9.2 game.py 的两种模式

```python
class GomokuGame:
    def __init__(self, render=True, record=True):
        """
        render=True:  打开 pygame 窗口，实时绘制棋盘（人类对弈 / 视频录制）
        render=False: 纯逻辑模式，无窗口（训练时的自对弈，最大速度）
        record=True:  记录每一步到 GameRecord 对象（用于回放和视频）
        """
```

### 9.3 视觉设计

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│   ┌────────────────────────┐  ┌───────────────────────┐  │
│   │                        │  │ Experiment #47         │  │
│   │     15×15 棋盘          │  │ Stage: 2 (vs minimax4)│  │
│   │     木纹背景            │  │ Win Rate: 63.5%       │  │
│   │     黑白棋子 + 阴影     │  │ Network: 8 blocks     │  │
│   │     最后一手标记         │  │ Params: 1.2M          │  │
│   │     候选落子热力图       │  │ ────────────────────  │  │
│   │     (显示 policy head   │  │ [win rate 折线图]      │  │
│   │      的概率分布)        │  │                       │  │
│   │                        │  │ [当前对局: 第 23 手]    │  │
│   └────────────────────────┘  │ [Black ●: NN (63.5%)] │  │
│                               │ [White ○: Minimax L2]  │  │
│                               └───────────────────────┘  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

关键视觉元素：
- **棋盘主体**：经典木纹底色，圆润棋子带阴影
- **落子热力图**：半透明覆盖层，显示 NN 对每个位置的评估概率（红=高概率，蓝=低概率）
- **信息面板**：当前实验编号、阶段、胜率、网络参数
- **胜率曲线**：实时更新的 win_rate 折线图

### 9.4 play.py：加载 checkpoint 对弈

```bash
# 列出所有里程碑 checkpoint
uv run play.py --list
# 输出:
#   [0] stage0_baseline       - wr: 0.42  exp#001  "随机乱下"
#   [1] stage1_beat_random    - wr: 0.97  exp#012  "首次击败随机"
#   [2] stage2_beat_minimax2  - wr: 0.82  exp#038  "击败 minimax-2"
#   [3] periodic_exp050       - wr: 0.65  exp#050  "定期快照"
#   [4] best                  - wr: 0.71  exp#089  "当前最强"

# 和初学者 AI 下（体验它有多菜）
uv run play.py --checkpoint stage0_baseline

# 和最强 AI 下
uv run play.py --checkpoint best

# 让两个不同阶段的 AI 互下（观赏成长）
uv run play.py --black stage1_beat_random --white best

# 增强 AI 强度（加 MCTS 搜索）
uv run play.py --checkpoint best --mcts 400
```

MCTS 次数是推理时的参数，不需要重新训练。同一个模型可以通过调 MCTS 次数来调整难度：

```
纯策略网络:      直接用 policy head 选位置（最快，~1ms/步）
MCTS 100 次:    中等强度（~50ms/步）
MCTS 800 次:    高强度（~400ms/步）
```

---

## 10. 训练过程录制系统（视频素材）

Code Bullet 风格的视频需要大量素材。我们在训练过程中**自动录制一切**，后期剪辑时再挑选。

### 10.1 录制的三类数据

#### A. 棋局记录（games/）

每一盘评估对局都保存为 JSON：

```json
{
  "experiment": 47,
  "eval_level": 2,
  "result": "black_win",
  "black": "nn",
  "white": "minimax_l2",
  "moves": [
    {"step": 1, "row": 7, "col": 7, "player": "black", "policy_entropy": 4.2,
     "value": 0.01, "top3": [[7,7,0.12], [7,8,0.08], [8,7,0.07]]},
    {"step": 2, "row": 7, "col": 8, "player": "white"},
    {"step": 3, "row": 8, "col": 7, "player": "black", "policy_entropy": 3.1,
     "value": 0.15, "top3": [[8,7,0.22], [6,6,0.11], [8,8,0.09]]}
  ],
  "total_moves": 47,
  "timestamp": "2026-04-11T03:45:12"
}
```

**什么时候录**：
- 每次评估的 200 盘中，保存**前 5 盘**的完整记录（含 policy/value 数据）
- 阶段晋升时，保存**全部评估对局**
- 每 10 次 keep 实验，保存 5 盘
- 存储量极小：每盘 ~5KB，1000 盘 = ~5MB

#### B. 训练指标时间线（metrics/）

一个持续追加的 CSV 文件：

```csv
experiment,timestamp,win_rate,eval_level,loss_policy,loss_value,num_params,blocks,filters,mcts_sims,games_played,status,description
1,2026-04-11T02:30:00,0.42,0,2.34,0.89,876544,6,64,0,12000,keep,baseline
2,2026-04-11T02:38:00,0.38,0,2.51,0.92,1203200,12,64,0,8000,discard,deeper network
3,2026-04-11T02:46:00,0.55,0,1.89,0.71,876544,6,64,0,12000,keep,lower learning rate
```

这个 CSV 是制作**训练进度图表**的数据源（win_rate 曲线、loss 曲线、参数量变化等）。

#### C. 关键帧截图（frames/）

prepare.py 中的评估函数在以下时刻自动截取棋盘帧（调用 game.py 渲染）：

```python
# 自动截帧时机
FRAME_TRIGGERS = {
    "first_win":        "NN 第一次赢棋的最终局面",
    "stage_promotion":  "晋升时的一盘代表性胜局（每步一帧）",
    "best_move":        "NN 做出高 value 评估落子的瞬间",
    "blunder":          "NN 输棋前的最后几手（反面教材）",
    "periodic":         "每 10 次实验的一盘棋的关键帧",
}
```

截帧存储为 PNG，命名规则 `exp047_game003_step023.png`。

### 10.2 replay.py：回放与视频渲染

```bash
# 回放一盘棋（pygame 窗口，可暂停/快进）
uv run replay.py games/exp047_game003.json

# 批量渲染为 PNG 序列（用于视频剪辑）
uv run replay.py games/exp047_game003.json --render-frames --output frames/

# 渲染 "AI 成长蒙太奇"：每个阶段的代表性对局拼接
uv run replay.py --montage --stages all --output montage/

# 渲染 "新旧 AI 对决"
uv run replay.py --versus stage1_beat_random best --output versus/
```

### 10.3 视频素材清单（Code Bullet 叙事线）

录制系统产出的素材，对应视频的各个段落：

| 视频段落 | 素材来源 | 内容 |
|---|---|---|
| "我做了个五子棋游戏" | game.py 的 pygame 画面 | 展示棋盘、人类对弈演示 |
| "AI 一开始有多蠢" | `stage0_baseline` 的棋局录像 | NN 随机乱下，毫无章法 |
| "让 AI 自己练一晚上" | terminal 录屏 autoresearch 循环 | agent 改代码 → 跑实验 → 数字跳动 |
| "它学会了什么" | `stage1` → `stage2` 的对局录像 | 从乱下到有攻防意识 |
| "训练曲线" | metrics CSV → matplotlib 图表 | win_rate 从 0.4 爬到 0.8 的曲线 |
| "AI 有了意想不到的策略" | 关键帧截图 + policy 热力图 | agent 发现的反直觉架构变化 |
| "新旧 AI 对决" | `--versus` 回放 | 早期 AI vs 最终 AI，直观对比 |
| "我能赢它吗" | play.py 的人机对弈 | 你自己和最终 AI 下 |
| "results.tsv 分析" | results.tsv 数据 | 哪些实验 keep 了，哪些 discard 了 |

---

## 11. 实施步骤

### Phase 0: 做游戏（Code Bullet 起手式）

像 Code Bullet 一样，第一步是**把游戏本身做出来**。

- [ ] 实现 `game.py`：
  - 棋盘逻辑（落子、胜负判定、合法位置）
  - pygame 渲染（木纹棋盘、棋子、最后一手标记）
  - 双人模式：两个人可以在同一台电脑上下
  - 录制模式：每步记录到 GameRecord 对象
  - 无渲染模式：纯逻辑，用于训练
  - 帧导出：支持把当前画面存为 PNG
- [ ] 手动玩几盘确认游戏本身好用

此时你已经有了一个可以玩的五子棋游戏。**这就是视频的开头素材。**

### Phase 1: 搭建训练基础设施

- [ ] 实现 `prepare.py`：
  - minimax 对手 (L0-L3)，调用 game.py 的棋盘引擎
  - 评估函数 `evaluate_win_rate()`
  - 自动录制评估对局到 `recordings/games/`
  - 里程碑 checkpoint 存档到 `~/.cache/gomoku/checkpoints/`
  - 指标追加到 `recordings/metrics/training_log.csv`
  - 关键帧自动截取到 `recordings/frames/`
- [ ] 实现 `train.py` baseline：初始 NN + 批量自对弈 + 训练循环
- [ ] 编写 `program.md`：五子棋版实验协议
- [ ] 配置 `pyproject.toml`
- [ ] 手动运行一次 `train.py` 确认全流程跑通
- [ ] 实现 `play.py`：加载 checkpoint 对弈（`--checkpoint`, `--list`, `--versus`）
- [ ] 实现 `replay.py`：回放 + 帧导出

### Phase 2: 启动自主实验循环

```bash
git checkout -b autoresearch/gomoku-v1
cd gomoku
uv run train.py   # 建立 baseline → 自动存档 stage0_baseline checkpoint
# 然后让 Claude Code 读 program.md 开始自主循环
```

- [ ] 运行 baseline，记录初始 win_rate
- [ ] Agent 自主循环：Stage 0 → 1 → 2（预计 3-5 小时）
- [ ] 检查 results.tsv，确认 agent 在有效探索
- [ ] 检查 `~/.cache/gomoku/checkpoints/manifest.json`，确认 checkpoint 在自动存档
- [ ] 检查 `recordings/` 目录，确认对局和指标在录制

### Phase 3: overnight 深度优化

- [ ] Stage 3: 对阵 minimax-4（一晚）
- [ ] Stage 4: 对阵 minimax-6（一晚）
- [ ] 查看实验日志，提取有趣的发现

### Phase 4: 体验与出片

- [ ] 用 `play.py --list` 查看所有阶段 checkpoint
- [ ] 依次和 stage0 → stage1 → stage2 → best 的 AI 对弈，亲身体验成长
- [ ] 用 `play.py --versus stage0 best` 看新旧 AI 对决
- [ ] 用 `replay.py --montage` 生成成长蒙太奇帧序列
- [ ] 用 metrics CSV 制作训练曲线图表
- [ ] 剪辑视频

---

## 12. 风险与缓解

| 风险 | 概率 | 缓解方案 |
|---|---|---|
| 评估胜率方差大，agent 误判 keep/revert | 中 | 增加评估对局数到 300-500；使用 Wilson score 置信区间 |
| 自对弈训练不稳定（loss 爆炸/坍缩） | 中 | prepare.py 中加 loss 数值检查；agent 可调 lr/clip |
| minimax 对手太弱或太强，阶梯不平滑 | 低 | 4 级对手覆盖够宽；agent 可自主调整当前评估级别 |
| Agent 陷入局部最优（找到一招鲜策略） | 中 | program.md 提示 agent 关注 avg_game_length 变化 |
| 5 分钟内训练数据不够 | 低 | 批量自对弈保证产出；M3 Max 性能足够 |

---

## 13. 与原版 autoresearch 的关键差异

| 维度 | 原版 (LLM) | 五子棋版 |
|---|---|---|
| 指标 | val_bpb (越低越好) | win_rate (越高越好) |
| 数据 | 外部数据集 (ClimbMix) | 自对弈生成（无需下载） |
| 评估 | 固定验证集上计算 BPB | 对固定对手下 N 盘 |
| 噪声 | 几乎无（确定性计算） | 有（棋局有随机性），需要更多样本 |
| 阶段性 | 单一指标一路优化 | 多级对手，阶梯式晋升 |
| 部署 | 无直接使用场景 | 可直接做成游戏 |
| 观赏性 | 看数字变化 | 可以直观看到 AI 下棋 |

---

## 14. 开发日志

### 2026-04-10: Phase 0 + Phase 1 完成（初始开发）

**环境**：Linux 开发机（无 MLX / 无 GPU），代码编写 + 逻辑验证。

**完成内容**：

1. **项目结构创建**
   - `mag-gomoku/` 目录，含 `recordings/{games,metrics,frames}/`
   - `pyproject.toml`：依赖 mlx>=0.22.0, numpy>=1.26.0, pygame>=2.5.0
   - `.gitignore`：排除 `model.safetensors`、录制文件、缓存

2. **game.py — 游戏引擎** (Phase 0)
   - `Board` 类：15x15 棋盘、落子、胜负判定、合法位置、编码（3通道 NCHW）
   - `BatchBoards` 类：N 盘并行游戏管理（纯 numpy，无 pygame），用于训练自对弈
   - `GameRecord` / `MoveRecord`：棋局录制、JSON 序列化/反序列化
   - `Renderer` 类：pygame 渲染（木纹棋盘、棋子阴影、最后一手标记、policy 热力图、信息面板）
   - 单元测试全部通过：Board、BatchBoards、encoding、legal mask、candidate moves、GameRecord

3. **prepare.py — 评估基础设施** (Phase 1, READ-ONLY)
   - 棋型评估函数 `evaluate_position()`：扫描 4 方向，评分五连/活四/冲四/活三/眠三/活二 等棋型
   - Minimax + alpha-beta 剪枝：`_minimax()` 带即时胜负检测、候选位剪枝（radius=2）
   - 4 级对手：L0 随机、L1 depth=2、L2 depth=4 + 启发式排序、L3 depth=6 + killer 启发
   - `evaluate_win_rate()`：加载模型 → 下 N 盘（黑白各半）→ 返回胜率 + 录制前 5 盘
   - `archive_checkpoint()`：复制到 `~/.cache/mag-gomoku/checkpoints/` + 更新 manifest.json
   - `log_metrics()`：追加到 `recordings/metrics/training_log.csv`
   - `capture_key_frames()`：headless 安全的 pygame 截帧
   - Smoke test 通过：L1 vs L0，L1 在 19 步内获胜，耗时 0.68s

4. **train.py — 训练脚本** (Phase 1, MUTABLE)
   - `GomokuNet`：ResNet CNN (Conv3x3 → 6 ResBlocks × 64 filters → Policy head [225] + Value head [1])
   - NCHW → NHWC 转换（MLX Conv2d 使用 channels-last）
   - 批量自对弈 `run_self_play()`：64 并行棋局，temperature + softmax 采样，policy 分布记录
   - 训练循环：交替自对弈/训练，AdamW 优化器，replay buffer，5 分钟固定预算
   - 评估集成：训练结束自动调用 `evaluate_win_rate(MODEL_PATH, level=EVAL_LEVEL)`
   - **Bug fix**：修复 `run_self_play` 中的无限循环问题（reset 导致永远不会 all-finished），改为所有游戏跑完一轮后返回

5. **play.py — 人机对弈**
   - `--checkpoint` / `--list` / `--black` / `--white` / `--level` / `--swap` / `--mcts`
   - 支持：人 vs NN、人 vs minimax、NN vs NN、checkpoint 选择
   - 自动查找 checkpoint（本地 model.safetensors / 存档目录 / fuzzy match）

6. **replay.py — 回放与帧导出**
   - 单局回放（pygame 窗口 + 可选 PNG 帧导出）
   - 蒙太奇模式：自动收集各阶段代表性胜局并连续播放
   - `--export` / `--output` / `--speed` / `--montage`

7. **program.md — 自主实验协议**
   - 完整的 autoresearch 循环规则，适配五子棋场景
   - 阶段晋升机制 (L0→L1→L2→L3) + checkpoint 归档流程
   - Agent 探索建议（架构、训练策略、需要关注的指标）

**已验证**（在 Linux 开发机上）：
- 所有 Python 文件语法检查通过
- game.py 单元测试全部通过（Board、BatchBoards、GameRecord）
- prepare.py smoke test 通过（minimax L1 vs L0 对局正确）
- prepare.py 单元测试通过（评估函数、胜负检测、对手函数、metrics logging）

**待验证**（需在 macOS M3 Max 上）：
- MLX 训练循环能否正常运行（train.py）
- pygame 渲染是否正常（game.py、play.py、replay.py）
- 5 分钟训练预算内能完成多少自对弈 + 梯度步
- 评估函数的实际耗时
- checkpoint 归档 + manifest.json 写入

**默认测试参数**（适合首次在 M3 Max 上试跑）：
```
PARALLEL_GAMES = 64       # 保守起步，后续可提到 256
BATCH_SIZE = 256           # 训练 batch
NUM_RES_BLOCKS = 6         # ~876K 参数
NUM_FILTERS = 64
EVAL_LEVEL = 0             # 先对阵随机
TIME_BUDGET = 300          # 5 分钟
```

---

## 环境搭建踩坑记录 (2026-04-10)

### Bug 1: pyproject.toml build-backend 错误

**现象**: `uv sync` 安装项目时报错 `No module named 'setuptools.backends'`

**原因**: `build-backend` 配置错误，使用了不存在的 `setuptools.backends._legacy:_Backend`

**修复**: 改为标准的 `setuptools.build_meta`:
```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
```

### Bug 2: Metal GPU 资源限制 — 评估崩溃

**现象**: 训练正常完成，但 `evaluate_win_rate()` 在运行约 100+ 局后崩溃:
```
RuntimeError: [metal::malloc] Resource limit (499000) exceeded.
```

**根因分析**:
1. `prepare.py` 的 `evaluate_win_rate()` 加载模型后 **没有调用 `model.eval()`**
2. MLX 的 `BatchNorm` 在 train 模式下，每次前向传播都会执行 `self.running_var = (1 - mu) * self.running_var + mu * var`
3. 这行代码每次都会分配一个新的 Metal buffer（MLX 的 immutable array 语义）
4. 200 局评估 × ~50 步/局 = ~10,000 次前向传播 → 超过 macOS Metal 的 499,000 buffer 上限

**测试验证**:
- 20 局评估: ✅ 正常通过
- 200 局评估 (无 model.eval): ❌ 崩溃，即使在全新进程中
- 200 局评估 (有 model.eval): ✅ 正常通过

**修复**:
1. 在 `src/prepare.py` 的 `evaluate_win_rate()` 中加载模型后调用 `model.eval()` — **根本修复**
2. 每 20 局调用一次 `mx.clear_cache()` — **防御性清理**
3. 训练后的评估改为子进程执行 — **隔离训练期间积累的 Metal buffer**

**教训**: MLX 的 BatchNorm 在 train 模式下会持续分配新 buffer，eval 模式下使用固定的 running stats 不分配新 buffer。**所有推理场景必须调用 `model.eval()`**。

### Bug 3: Python 输出缓冲

**现象**: `uv run train.py > run.log 2>&1` 执行时 run.log 长时间为空（0 字节），训练结束后才一次性输出。

**原因**: Python 检测到 stdout 被重定向到文件/管道时，会启用全缓冲（block buffering），不会实时写出。

**修复**: 使用 `PYTHONUNBUFFERED=1` 环境变量禁用缓冲:
```bash
PYTHONUNBUFFERED=1 uv run python src/train.py > output/run.log 2>&1
```

### Baseline 性能结果

| 指标 | 数值 |
|---|---|
| win_rate | 0.6800 (vs L0 随机) |
| 模型参数 | 564.5K |
| 训练时间 | 300s |
| 自对弈局数 | 5,568 |
| 训练步数 | 4,290 |
| 最终 loss | 0.33 |

### 目录重组 (2026-04-10)

将原来的扁平结构重组为清晰的分层结构:

```
mag-gomoku/
├── src/          # 所有 Python 源码
├── docs/         # 文档 (program.md, action-plan.md)
├── data/         # 实验跟踪数据 (results.tsv, git tracked)
├── output/       # 生成的训练产物 (gitignored)
│   ├── model.safetensors
│   ├── run.log
│   └── recordings/{games,metrics,frames}
├── .gitignore
├── README.md
├── pyproject.toml
└── uv.lock
```

运行命令统一为 `uv run python src/<script>.py`，从项目根目录执行。
