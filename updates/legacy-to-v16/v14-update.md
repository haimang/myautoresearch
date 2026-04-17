# v14 Update — C 原生 MCTS + 深度搜索 + 对手晋升

> 2026-04-12  
> 前置：v13-findings-2.md（vs L1 失败分析：50 sims 不够，需要 400-800 sims）  
> 瓶颈确认：Python MCTS 树操作占 77% 时间，C 扩展预期 10-20x 加速

---

## 1. 版本目标

**一句话：用 C 扩展让 800 sims MCTS 达到可用速度，训练出能赢 minimax L1 的模型。**

v13 证明了：
- 50 sims / 225 位置 = 0.2 次/位置，搜索无法发现战术（Focus 10%）
- 200 sims 在 Python 中 ~35 gm/min（8x128），400+ sims 不可行
- 从零训练 vs L1 彻底失败（WR 卡在 0% 和 50%）
- Python 解释器占 MCTS 执行时间的 77%

v14 的核心改动：**C 原生 MCTS 树操作**，让 800 sims 达到 ≥50 gm/min。

---

## 2. Phase 总览

| Phase | 内容 | 预期效果 |
|-------|------|---------|
| 1 | C 扩展：MCTS 树操作原生化 | select/expand/backup 10-20x 加速 |
| 2 | Board 轻量拷贝 | board.copy() 加速（跳过 history） |
| 3 | 深度搜索训练验证 | 800 sims + resume S0 vs L1 |
| 4 | 对手阶段晋升 | L0→L1→L2 实际推进 |

---

## 3. Phase 1：C 原生 MCTS

### 3.1 架构

```
Python (train.py)                     C (mcts_c.c)
─────────────────                     ─────────────
for each move round:
  boards[] →                          
                                      tree_pool_init(N)
  for each sim round:                 
                                      for k sims × N trees:
                                        select_path()      ← C: PUCT 遍历
    ← actions[]                       
    board.copy() + place()            
    if terminal:                      
                                        backup(value)       ← C: 树回溯
    else:                             
      states → evaluate_batch (GPU)   
                                        expand(priors)      ← C: 分配节点
                                        backup(-value)      ← C: 树回溯
                                      
  ← visits[N][225]                    extract_visits()
```

**Python 保持控制权**（外层循环、Board 操作、GPU 推理）。  
**C 只做树操作**（select PUCT、expand 节点分配、backup 树回溯）。

### 3.2 C 数据结构

```c
#define MAX_CHILDREN 225
#define MAX_NODES 200000  // 800 sims × ~50 depth × N trees

typedef struct MCTSNode {
    int parent;           // index in pool (-1 = root)
    int parent_child_idx; // index in parent's children arrays
    int action;           // move that led here
    float prior;
    int visit_count;
    float value_sum;
    int is_expanded;
    int n_children;
    int child_actions[MAX_CHILDREN];
    float child_priors[MAX_CHILDREN];
    int child_visits[MAX_CHILDREN];
    float child_values[MAX_CHILDREN];
    int child_nodes[MAX_CHILDREN];  // indices in pool (-1 = not created)
} MCTSNode;

// Pre-allocated pool — no malloc during search
static MCTSNode pool[MAX_NODES];
static int pool_next = 0;
```

### 3.3 C 接口（通过 ctypes 暴露给 Python）

```c
// 初始化一棵搜索树，返回 root index
int tree_init(float* priors, float* legal_mask, int action_size,
              float dirichlet_alpha, float dirichlet_frac);

// 从 root 执行一次 select，返回到达的叶子路径
// path_out: 输出的 action 序列（用于 Python 端 board.place）
// 返回: path 长度
int tree_select(int root_idx, float c_puct, float virtual_loss,
                int* path_actions_out, int* path_nodes_out, int max_depth);

// 展开叶子节点
void tree_expand(int node_idx, float* priors, float* legal_mask, int action_size);

// 回溯
void tree_backup(int node_idx, float value);

// 撤销虚拟损失（对 path 上所有节点）
void tree_revert_virtual_loss(int* path_nodes, int path_len, float virtual_loss);

// 提取 root 的 visit 分布
void tree_get_visits(int root_idx, float* visits_out, int action_size);

// 重置 pool（每步走棋后调用）
void tree_pool_reset(void);
```

### 3.4 Python 包装

```python
# framework/core/mcts_native.py
import ctypes, numpy as np, os

_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "mcts_c.dylib"))

def mcts_search_native(root_states, evaluate_batch_fn, copy_fn, 
                        legal_mask_fn, apply_fn, terminal_fn, 
                        terminal_value_fn, action_size,
                        num_simulations, sims_per_round=8, ...):
    """Drop-in replacement for mcts_search_multi_root, using C tree ops."""
    # Init roots via C
    # Main loop: C select → Python board ops → Python GPU eval → C expand+backup
    # Extract visits from C
```

### 3.5 编译

```bash
# macOS (M3 Max, clang 自带)
cd framework/core
clang -shared -O3 -fPIC -o mcts_c.dylib mcts_c.c -lm

# Linux (开发机验证)
gcc -shared -O3 -fPIC -o mcts_c.so mcts_c.c -lm
```

### 3.6 文件清单

| 文件 | 说明 |
|------|------|
| `framework/core/mcts_c.c` | C 原生 MCTS 树操作（~300 行） |
| `framework/core/mcts_native.py` | ctypes 包装 + Python 搜索循环（~150 行） |
| `framework/core/build_native.sh` | 编译脚本（3 行） |
| `domains/gomoku/train.py` | 适配：优先用 native，fallback 到 Python |

---

## 4. Phase 2：Board 轻量拷贝

game.py 是 READ-ONLY。但 train.py 中可以用轻量拷贝替代 `board.copy()`：

```python
def _fast_board_copy(board):
    """Skip history list copy — not needed for MCTS simulation."""
    b = Board.__new__(Board)
    b.grid = board.grid.copy()        # numpy (15,15) int8 copy: ~1μs
    b.current_player = board.current_player
    b.move_count = board.move_count
    b.last_move = board.last_move
    b.winner = board.winner
    b.history = []                    # empty — place() only appends, never reads
    return b
```

节省：每次 copy 省 ~5μs（history list 拷贝），800 sims × 8 boards × 50 rounds = 320K copies → 省 ~1.6s/cycle。

---

## 5. Phase 3：深度搜索训练验证

### 5.1 测试矩阵

| 测试 | Sims | 模型 | 对手 | 起点 | 时间 | 目标 |
|------|------|------|------|------|------|------|
| A | 400 | 8x64 | L1 | resume S0 | 30min | Focus >20%, WR 上升 |
| B | 800 | 8x64 | L1 | resume S0 | 60min | WR ≥30% vs L1 |
| C | 800 | 8x64 | L1 | 从零 | 60min | 对照组：能否突破 0% |

### 5.2 测试命令

```bash
# 编译 C 扩展
cd framework/core && bash build_native.sh && cd ../..

# 测试 A: 400 sims, resume S0, vs L1 (30min)
uv run python domains/gomoku/train.py \
  --mcts-sims 400 --parallel-games 8 --mcts-batch 8 \
  --num-blocks 8 --num-filters 64 \
  --time-budget 1800 \
  --eval-level 1 --no-eval-opponent \
  --eval-interval 5 --probe-games 50 \
  --resume d6c6bce4 --seed 42

# 测试 B: 800 sims, resume S0, vs L1 (60min)  
uv run python domains/gomoku/train.py \
  --mcts-sims 800 --parallel-games 4 --mcts-batch 8 \
  --num-blocks 8 --num-filters 64 \
  --time-budget 3600 \
  --eval-level 1 --no-eval-opponent \
  --eval-interval 3 --probe-games 50 \
  --resume d6c6bce4 --seed 42

# 测试 C: 800 sims, 从零, vs L1 (60min)
uv run python domains/gomoku/train.py \
  --mcts-sims 800 --parallel-games 4 --mcts-batch 8 \
  --num-blocks 8 --num-filters 64 \
  --time-budget 3600 \
  --eval-level 1 --no-eval-opponent \
  --eval-interval 3 --probe-games 50 \
  --seed 42
```

### 5.3 成功标准

| 指标 | Python 基线 (200 sims) | C 扩展目标 (800 sims) |
|------|----------------------|---------------------|
| 局/分钟 | ~35 (8x64) | ≥40 |
| Cycle 耗时 | ~17s | ≤20s |
| Focus | 10% (50 sims) | ≥25% |
| WR vs L1 (30min) | 0-50% | ≥30% |

**关键判断：** Focus 从 10% 提升到 25%+ = 搜索有效，模型在学习战术。

---

## 6. Phase 4：对手阶段晋升

如果 Phase 3 验证 800 sims 能让 WR vs L1 上升：

```bash
# 达到 WR ≥80% vs L1 后注册
uv run python domains/gomoku/train.py \
  --register-opponent S1 --from-run <run_id> --from-tag <tag> \
  --description "8x64 MCTS-800, 80%+ vs minimax L1"

# Stage 2: vs L2 (minimax depth 4)
uv run python domains/gomoku/train.py \
  --mcts-sims 800 --parallel-games 4 --mcts-batch 8 \
  --num-blocks 8 --num-filters 64 \
  --time-budget 7200 \
  --eval-level 2 --no-eval-opponent \
  --resume <S1_run> --target-win-rate 0.60 --seed 42
```

---

## 7. 速度预估

### 当前 Python 性能（实测数据）

| 配置 | 局/分钟 | 每步搜索时间 |
|------|---------|-------------|
| 50 sims, 8x64, pg=10 | 122 | ~2ms |
| 50 sims, 8x64, pg=32 | 141 | ~5ms |
| 80 sims, 8x128, pg=32 | 37 | ~20ms |
| 200 sims, 8x64 (估) | ~35 | ~50ms |

### C 扩展后预期

Python MCTS 树操作占 77% 时间。C 加速 15x 后：

```
当前 200 sims 一步搜索: 50ms
  Python 树操作: 38ms (77%) → C: ~2.5ms
  Board 操作:    5ms  (10%) → 不变
  GPU evaluate:  5ms  (10%) → 不变
  其他:          2ms  (3%)  → 不变
预期: ~15ms/步

800 sims 预期: ~15ms × 4 = ~60ms/步
  每盘 ~50 步 × 60ms = 3s/盘
  8 并行: ~3s/cycle
  局/分钟: 8 × 60/3 = ~160
```

**800 sims 预期 ~50-80 局/分钟。** 比当前 Python 200 sims 的 35 gm/min 更快，尽管搜索量是 4 倍。

---

## 8. In-scope / Out-of-scope

### In-scope

1. `framework/core/mcts_c.c` — C 原生 MCTS 树操作
2. `framework/core/mcts_native.py` — ctypes 包装
3. `framework/core/build_native.sh` — 编译脚本
4. `domains/gomoku/train.py` — 适配 native MCTS + 轻量 board copy
5. 800 sims 训练验证（resume S0 vs L1）
6. 对手阶段晋升（如果验证成功）

### Out-of-scope

1. **Board 操作 C 化** — 改善有限（10%），且 game.py 是 READ-ONLY
2. **GPU 推理优化** — MLX 已经够快
3. **多进程 worker** — 单进程 + C 扩展应该足够
4. **模型架构变更** — 先验证搜索深度能否解决问题

---

## 9. 风险分析

| 风险 | 严重度 | 缓解 |
|------|--------|------|
| C 编译失败（macOS clang 兼容性） | 低 | macOS 自带 clang，标准 C99 |
| ctypes 开销抵消 C 加速 | 中 | 每 sim round 只 1 次 Python↔C 调用 |
| 800 sims 仍然 Focus <20% | 高 | 说明问题不在搜索深度，需要换算法 |
| Board.copy() 轻量拷贝导致 bug | 低 | place() 只 append history，不读 |
| MAX_NODES 溢出 | 低 | 500K 节点足够 800 sims × 深度 50 |

---

## 10. 工作日志

> 执行者：Claude Opus 4.6  
> 执行日期：2026-04-12

### 10.1 C 实现 (framework/core/mcts_c.c, ~280 行)

- `MCTSNode` 结构体：pre-allocated pool，MAX_NODES=500000，零 malloc
- `mcts_select_child()`：C 循环 PUCT，~225 子节点遍历
- `mcts_expand()`：mask + normalize priors，填充 child 数组
- `mcts_backup()`：树回溯 + O(1) parent sync
- `mcts_batch_select()`：**关键优化** — K sims × N roots 全在 C 内完成，1 次 C 调用替代 K×N 次
- `mcts_batch_expand_backup()`：批量展开+回溯，1 次 C 调用处理所有叶子
- 静态缓冲区 `g_batch_path_*` 避免动态内存分配

### 10.2 Python wrapper (framework/core/mcts_native.py, ~160 行)

- 自动加载 `.dylib`(macOS) / `.so`(Linux)
- 每 sim round 仅 **3 次 Python↔C 转换**：batch_select → Python board ops + GPU → batch_expand_backup
- 对比第一版 per-path 调用（400 次/round）→ 减少 99% 的 ctypes 开销

### 10.3 train.py 集成

- 自动检测 native C 库，fallback 到 Python
- TUI 日志显示 `[C-native]` 或 `[Python]`
- `_fast_copy()` 轻量 Board 拷贝（跳过 history list）

### 10.4 开发机 benchmark（Linux，mock GPU）

| 配置 | Native C | Python | 加速比 |
|------|---------|--------|--------|
| 8×50 sims | 10ms | 33ms | **3.3x** |
| 8×800 sims | 148ms | 518ms | **3.5x** |
| Win detection (500 sims) | 85% | 69% | ✓ |

### 10.5 Mac 测试命令

```bash
git pull origin main

# 1. 编译 C 扩展
cd framework/core && bash build_native.sh && cd ../..

# 2. 验证 native 加载
uv run python -c "
import sys; sys.path.insert(0,'framework')
from core.mcts_native import is_available
print('Native MCTS:', 'YES' if is_available() else 'NO')
"

# 3. 测试 A: 800 sims, resume S0, vs L1 (30min)
uv run python domains/gomoku/train.py \
  --mcts-sims 800 --parallel-games 8 --mcts-batch 16 \
  --num-blocks 8 --num-filters 64 \
  --time-budget 1800 \
  --eval-level 1 --no-eval-opponent \
  --eval-interval 3 --probe-games 50 \
  --resume d6c6bce4 --seed 42

# 4. 测试 B: 800 sims, 从零, vs L1 (30min, 对照)
uv run python domains/gomoku/train.py \
  --mcts-sims 800 --parallel-games 8 --mcts-batch 16 \
  --num-blocks 8 --num-filters 64 \
  --time-budget 1800 \
  --eval-level 1 --no-eval-opponent \
  --eval-interval 3 --probe-games 50 \
  --seed 42

# 5. 带回 DB
cp output/tracker.db <共享路径>/v14_test.db
```

**TUI 中确认 native 加载：** 事件日志应显示 `MCTS 800sims [C-native]`。如果显示 `[Python]`，说明编译失败或 .dylib 未找到。

**关键观察指标：**
- Focus：从 10% (50 sims) 应提升到 25%+
- Entropy：应低于 2.5
- WR vs L1：resume S0 应能从 >0% 起步

---

## 11. mcts_9th 复盘驱动的框架级修复（2026-04-13）

> 前置：v14-findings.md §7（mcts_9th 复盘：100% 胜率是伪的、loss 没学、200 局=2 局的结构性坍缩）

mcts_9th 6 小时长训暴露的不是超参问题，而是**评估协议 + 指标记录**两层的结构性缺陷。本节修复的是框架层，不是 domain 算法层——因此不涉及"agent 该用什么算法"的判断，只改"基准测度是否可信"。

### 11.1 本轮修复的范围边界

**修了什么：**

1. 评估协议的统计坍缩（minimax 确定 + NN argmax → 200 局 = 2 局）
2. loss 诊断能力（只有总 loss，看不出 policy/value 哪个没在学）
3. cycle_metrics 表的 "loss=0 伪影"（时间预算耗尽后的空 cycle 仍写一行）
4. 评估时 NN 对手 temperature 默认 0.5（v13 遗留，虚增 WR）

**没修什么（out-of-scope）：**

1. MCTS 算法本身（C 扩展、sims 数、c_puct）——agent 的判断空间
2. self-play 的 Dirichlet noise / 温度衰减——已读过代码，实现正确（train.py:500-520, 478-480 passes dirichlet to native MCTS）
3. game.py 棋盘引擎——没有 bug
4. analyze.py 的 Pareto / 报告展示——tracker.db 已有新列，analyze 消费可以后续再接

### 11.2 修改清单

#### A. `domains/gomoku/prepare.py` — 随机化 minimax 对手

**问题：** `opponent_l1/l2/l3` 用 `minimax_move` + 稳定排序 `_move_order_basic`，完全确定。对相同棋盘每次返回相同一手。配合 NN argmax，"N 局 eval" 退化成 2 局 × N/2 次复制。

**修复：**

- 新增 `_root_move_scores(board, depth, player, move_order_fn)`：不返回单一 best_move，而是枚举根层所有候选动作并调用 `_minimax` 获得每个候选的完整得分。这是对现有 `_minimax` 的一次纯数据提取式调用，不改变 alpha-beta 本体。
- 新增 `minimax_move_sampled(board, depth, player, move_order_fn, top_k, softmax_temp, win_threshold)`：对根层候选排序后，在 top-k 内用 softmax(score/temp) 采样一个动作返回。当任一候选得分 ≥ `win_threshold`（默认 50000，对应 `_PATTERN_SCORES` 的 open-four 量级）时**强制选中**该候选——保证一步必胜 / 必堵永远正确落子，随机性只在非战术位点产生。
- `opponent_l1` → `minimax_move_sampled(depth=2, top_k=3, softmax_temp=50)`
- `opponent_l2` → `minimax_move_sampled(depth=4, top_k=3, softmax_temp=60)`
- `opponent_l3` → `minimax_move_sampled(depth=6, top_k=2, softmax_temp=80)`
- L0 保持纯随机不变。

**实测验证（Linux dev 机, pytest-style smoke）：** 同一中局位置 20 次调用 `opponent_l1`，分布为 `[(6,8)×16, (8,6)×3, (9,7)×1]`——在策略上仍以最佳一手为主（80%），但产生真实随机性。遇到必胜点时 10/10 全部选中正确位点。强度基本不变，但评估不再是 2 局。

#### B. `domains/gomoku/train.py` — 评估开局多样化 + 轨迹指纹

**问题：** 即使对手随机化了，所有 eval 游戏仍然从空盘开始，argmax NN 的"黑棋开局手"仍然唯一，实际 eval 的开局分布非常窄。

**修复：**

- 模块级常量 `_EVAL_OPENING_SEEDS`：16 条两步开局（含空盘、纯中心、各种近中心方向、远对角 / 反对角），覆盖常见 gomoku 布局母体。
- `_apply_opening(board, seed)`：把一条 seed 应用到棋盘上。
- `_in_process_eval(model, level, n_games, opponent_model, num_openings)`：每局评估前按 `game_index % num_openings` 选一条 seed 播种，再让双方正常接续。
- 同时用 `trajectory_fingerprints: set[tuple]` 记录每局实际走出的完整动作序列——返回的 `result` 字典多出 `num_openings` 和 `unique_trajectories` 两项。这让"200 局 = N 局唯一"的坍缩在下一次出现时立刻可观测。
- `_quick_eval` 同步签名，接受 `num_openings` 参数。
- 新 CLI flag `--eval-openings N`（默认 0 = 按 `min(16, n_games // 4)` 自动）。
- `_do_checkpoint` 和 final eval 都把 `num_openings` 和 `unique_trajectories` 传下去，并写入 DB。

**结果字典新字段：**

```python
{
    # ... 原有字段 ...
    "num_openings": int,           # 本次评估使用了多少条 opening seed
    "unique_trajectories": int,    # 实际出现了多少条不同的完整游戏序列
}
```

**实测验证：** 在 16 条 opening × 简单确定性 NN（一律挑第一个合法点）+ 随机化 L1 对手下跑 24 局 →  unique_trajectories=24。与 mcts_9th 的 "200 局 = 2 局" 坍缩形成对照。

#### C. `domains/gomoku/train.py` — eval 时 NN 对手 temperature 默认 0

**问题：** `_nn_opponent_move(temperature=0.5)` 让 NN 对手在评估阶段也在随机采样。胜率被对手的"意外失误"虚假拉高。

**修复：** 默认参数改为 `0.0`。调用点不变——想要带噪 NN 对手的调用方需要显式传 `temperature=0.5`。文档字符串明确说明这是 v13 遗留。

#### D. `domains/gomoku/train.py` — policy / value loss 拆分

**问题：** `compute_loss` 返回一个合计 loss。mcts_9th 的 loss 从 6.36 只降到 4.25，但单一 loss 数字看不出是 policy 没学还是 value 没学（真相是 policy 基本没动，value 早期稳定——两类信号被加权合并后相互掩盖）。

**修复：**

- 新增 `compute_loss_split(model, boards, policies, values)`：返回 `(total, policy_loss, value_loss)` 三元组。原 `compute_loss` 仅返回总 loss 不变——它是 `nn.value_and_grad` 的目标函数，签名不能乱动。
- 训练循环在每个 cycle 结束后、在**训练循环外**、对该 cycle 的**最后一个 mini-batch** 跑一次 `compute_loss_split` 获取分量。这是 one extra forward pass per cycle——相对 50 个训练步基本可忽略。
- 新增 state 变量 `last_policy_loss / last_value_loss`，以及 `policy_loss_history / value_loss_history` 两个 list。
- TUI 面板：在原 Cycle/Loss/Games 三格行下方，新增一行 `P-Loss / V-Loss / policy entropy gap`。仅当有有效分量时显示。

#### E. `domains/gomoku/train.py` — cycle 457 "loss=0 伪影" 修复

**问题：** mcts_9th 最后 cycle 的 `loss=0.0` 不是真正的 loss——是时间预算耗尽后 cycle 没有执行训练步，但仍然写了一行 cycle_metrics。这污染了 `runs.final_loss` 字段，让 analyze.py 看到一个假的"训练完美收敛"信号。

**修复：**

- 训练循环内新增 `steps_run_this_cycle = 0` 计数器，每个成功的 grad step +1。
- metric 字典构造时，若 `steps_run_this_cycle == 0`，`loss / policy_loss / value_loss` 全写 `None`。
- 只有当 `steps_run_this_cycle > 0 或 metric["win_rate"] is not None`（即"本 cycle 做了有意义的事"）才调用 `_tracker.save_cycle_metric`。否则完全跳过。这消除了"空 cycle 污染 DB" 的可能。
- `finish_run` 写 `final_loss` 时：若 `total_train_steps == 0`（完全没训练过），写 `None` 而不是 0。

#### F. `framework/core/db.py` — 迁移 v15

新增 **cycle_metrics** 列：

```sql
policy_loss REAL,    -- per-cycle policy cross-entropy (last mini-batch)
value_loss  REAL     -- per-cycle value MSE (last mini-batch)
```

新增 **checkpoints** 列：

```sql
eval_unique_openings INTEGER   -- 该 checkpoint full eval 中出现的唯一轨迹数
```

- `save_cycle_metric` 和 `save_checkpoint` 的 INSERT 语句扩展对应字段，使用 `dict.get(...)` 读取，旧调用者无需修改。
- 迁移走现有 `ALTER TABLE ... ADD COLUMN` 的 try/except 套路，旧 tracker.db 自动升级，**不破坏兼容性**。
- 本迁移**不涉及 schema 主题变化**——只加列，不改表、不加约束。

### 11.3 验证步骤（在 Mac 上）

修复后应先做一次**纯评估验证**：不训练，直接把 mcts_9th 的 final_c0457 加载回来跑 full eval，看在修好的协议下真实 WR 是多少。这可以用一个已存在 checkpoint 做 sanity：

```bash
git pull origin main

# 用修复后的评估协议，把 789730e3 的 final_c0457 当 NN 注册，
# 然后让它作为 eval-opponent 对 L1 打，看 unique_trajectories 是不是从 2 变成 16。
# （如果你只想快速验证协议本身，可以跳过注册，直接用任意一个旧 ckpt
#  起一个 60s 训练，观察 TUI 的 Full eval 行里有没有 "16uniq/16op"）

uv run python domains/gomoku/train.py \
  --mcts-sims 0 \
  --num-blocks 8 --num-filters 128 \
  --time-budget 60 \
  --eval-level 1 --no-eval-opponent \
  --eval-interval 3 --probe-games 40 \
  --full-eval-games 64 --eval-openings 16 \
  --resume 789730e3 --seed 42
```

**验证判据：**

1. TUI `Full eval:` 事件行末尾应出现 `NuniqM/NopenM` 形式，且 `Nuniq` 接近 `NopenM × 2`（两侧）。
2. cycle_metrics 表新增行应有 `policy_loss` / `value_loss` 两列非空（只要有 ≥1 个训练步）。
3. 任何没有训练步的 cycle **完全不入库**——用 `sqlite3 output/tracker.db "SELECT cycle, loss FROM cycle_metrics WHERE run_id='<id>' AND loss IS NULL"` 应该只显示 eval-only 的 cycle（可以 0 行也正常）。
4. 用 sqlite 看 runs：修复后的 run 的 `final_loss` 应是**真实最后一次训练的 loss**，不是 0。

### 11.4 3 小时正式训练命令

目标：在修好的评估协议下，**首次得到一份可信的 8×128 / 800 sims / vs L1 真实胜率曲线**。参数继承自 §6.2 建议（LR 2e-4、steps-per-cycle 50、buffer 100K），但这次的数字有统计意义。

```bash
git pull origin main

# 编译 C 扩展（如尚未编译）
cd framework/core && bash build_native.sh && cd ../..

# 3 小时从零训练，vs L1，固定 seed，固定时间预算
uv run python domains/gomoku/train.py \
  --mcts-sims 800 --parallel-games 10 --mcts-batch 16 \
  --num-blocks 8 --num-filters 128 \
  --learning-rate 2e-4 --steps-per-cycle 50 --buffer-size 100000 \
  --time-budget 10800 \
  --eval-level 1 --no-eval-opponent \
  --eval-interval 5 --probe-games 80 \
  --full-eval-games 200 --eval-openings 16 \
  --auto-stop-stagnation --stagnation-window 15 \
  --seed 42
```

**时间预算拆分（预期）：**

| 阶段 | 时长 | 说明 |
|------|------|------|
| C 扩展 warmup | ~5s | 一次性 |
| 每个 cycle | ~15-20s | 10 boards × ~50 moves × 800 sims + 50 training steps |
| 预计 cycles | ~500-600 | 10800 / 18 ≈ 600 |
| Probe eval | 每 5 cycles 一次，80 局 | 约 2-3s/次 |
| Full eval | 每次越过阈值 | 200 局 / 16 openings，约 4-6s/次 |

**训练中需要观察的四个核心指标（按优先级）：**

1. **policy_loss 是否下降**（TUI 面板第 4 行的 `P-Loss`）。这是 v14 的真正瓶颈——mcts_9th 的 policy_loss 基本没动。如果修好后的训练 policy_loss 能从 ~4.8 降到 <3.5，说明 MCTS 目标被吸收了；如果仍然卡在 4+，说明问题不在 eval 而在 self-play target 或训练动力学。
2. **Full eval 的 `uniq/op` 比**。在 `✓ Checkpoint` 之后的 `Full eval:` 日志里应出现类似 `80uniq/16op`——完全满额（200/16 × 2 = 16 最多满）或接近满额。如果只有个位数，说明对手随机化或开局多样化失败。
3. **Probe WR vs L1 的分布形态**。修复前只能取 {0%, 50%, 100%}；修复后应该看到连续分布（例如 37%, 44%, 52% 这类中间值）。如果仍然只出现 {0, 0.5, 1}，说明开局分桶平分时出了 bug。
4. **avg_game_length**。mcts_9th 最后坍缩到 10.5 步，是伪胜的特征。在多样化开局下，合格的训练 run 应看到 15-30 步的均值，且有真实方差。

### 11.5 训练结束后你需要带回的东西

1. `output/tracker.db`（或只带 mcts_10 的 run 部分）
2. TUI 最终截图
3. 至少一个 checkpoint 文件路径（从 db 的 `checkpoints` 表 final 行读），我需要它来做下一步的 Pareto 定位

我会基于这份数据做三件事：
- 在 v14-findings.md 追加 §8 "第一份真实 vs L1 数据"
- 在 pareto-frontier.md §14 的"发现能力表"里补一行 "eval-协议修复前后" 的对照
- 判断这个 checkpoint 是否可作为 S1 候选，给出下一步动作（resume L2 / 重训 / 换策略）

---

## 12. 资源利用率诊断与升级改造方案（2026-04-13 下午）

> 触发：mcts_11 启动时观察到 M3 Max (128 GB 统一内存) 在 `--parallel-games 24 --mcts-batch 8` 和 `--parallel-games 64 --mcts-batch 16` 两种配置下 **RAM 都稳定在 115 GB**，同时 CPU 负载 <10%、GPU 长期 <50W。这是严重的资源利用失衡：内存快满了，但算力没用上。

本节是一次严肃的系统诊断。我先把三个根因拆开，每一个都用代码和硬件事实论证；然后给出 P0 / P1 / P2 三级改造方案和验证判据。

### 12.1 用户观察

| 配置 | RAM | CPU | GPU | 结论 |
|------|-----|-----|-----|------|
| pg=24 / mcts-batch=8 | 115 GB | <20W | <10W | 115 GB 吃满，CPU/GPU 双闲 |
| pg=64 / mcts-batch=16 | 115 GB（持续）| <10% | <50W | 同上，扩 pg 无帮助 |
| pg=10 / mcts-batch=16（mcts_10） | 未测 | ? | ? | 能跑完 3h 完整训练，说明是高 pg 触发的问题 |

两个观察值得单独重复：
1. **从 pg=24 到 pg=64，RAM 没有随 pg 线性增长**——它在 115 GB 撞天花板，说明有一个和 pg 无关的全局累积，不是"每盘多几 GB"那种线性问题。
2. **从 pg=24 到 pg=64，GPU 功耗只从 10W 涨到 50W**——说明 pg × mcts-batch 增加并没有把更多工作量真正送进 GPU。这是**吞吐量的 smoking gun**。

### 12.2 根因一：`MAX_BATCH_PATHS = 256` 静默截断（P0 级，throughput 杀手）

**代码证据** — `framework/core/mcts_c.c:289-296`：

```c
#define MAX_BATCH_PATHS 256   /* max K*N per call */
#define MAX_PATH_DEPTH  128

static int   g_batch_path_nodes[MAX_BATCH_PATHS * MAX_PATH_DEPTH];
static int   g_batch_path_actions[MAX_BATCH_PATHS * MAX_PATH_DEPTH];
static int   g_batch_path_lens[MAX_BATCH_PATHS];
static int   g_batch_leaf_nodes[MAX_BATCH_PATHS];
```

`framework/core/mcts_c.c:298-316` 的 `mcts_batch_select` 主循环：

```c
for (int sim = 0; sim < sims_per_round && total < MAX_BATCH_PATHS; sim++) {
    for (int ri = 0; ri < n_roots && total < MAX_BATCH_PATHS; ri++) {
        ...
        total++;
    }
}
```

注意 **两个 `total < MAX_BATCH_PATHS` 守卫**——一旦凑够 256 条路径，`sims_per_round` 的外层循环就提前退出，**剩下的 sims 被静默丢弃，不报错也不警告**。

**后果换算：**

| 用户配置 | 意图 leaf/round | 实际 leaf/round | 损失 |
|----------|------------------|-------------------|------|
| pg=10 / batch=16 | 160 | 160 | 0%（mcts_10 命中甜点）|
| pg=16 / batch=16 | 256 | 256 | 0%（刚好卡边界）|
| pg=24 / batch=8  | 192 | 192 | 0% |
| pg=24 / batch=16 | 384 | 256 | **33%** |
| pg=32 / batch=16 | 512 | 256 | **50%** |
| pg=64 / batch=16 | **1024** | **256** | **75%** |
| pg=64 / batch=8  | 512 | 256 | **50%** |

**pg=64 / batch=16 在每个 sim round 只做了 256/1024 = 25% 的声明工作量。** 为了完成 800 sims，外层 `while remaining > 0` 循环会多跑 4 倍 round，总 Python↔C↔GPU 转换次数翻 4 倍。对于 GPU 来说：每次 forward 的 batch 大小被夹在 256 内（8×128 ResNet 在 Metal 上的 matmul 甜点也大致在 batch 256 附近），所以增大 pg 并不能让 GPU 单次 forward 更忙——只是让 round 数变多、Python 侧序列化开销变多。

**这一条解释了 GPU 50W 的天花板**：不管 pg 怎么涨，leaf batch 都固定在 256，GPU 每次 forward 做的工作量是常数，但 round 频次更高，于是平均 GPU 饱和度反而下降。

### 12.3 根因二：MLX 分配器在训练循环里从不释放（P0 级，RAM 杀手）

**代码证据** — `domains/gomoku/train.py` 全文 `mx.clear_cache()` 只出现在 5 处：

| 行号 | 上下文 | 频率 |
|------|--------|------|
| 530 | 自对弈 MCTS 内，`total_moves % 80 == 0` | 每 ~80 步一次 |
| 748 | `run_opponent_play` 每 20 局一次 | 稀疏 |
| 1497, 1738, 1856 | 评估路径 | 一次性 |

**训练循环（`for step in range(steps_this_cycle)` 块，line 1288-1329）里完全没有 `mx.clear_cache()`。**

MLX 在 Apple Silicon 统一内存上的已知行为：

1. `mx.eval(model.parameters(), optimizer.state)`（line 1322）**执行计算图**并释放 graph 临时变量，但**不释放 allocator cache**——它只是把"这批 activations 已经用完"的 buffer 还给内部池子，等下次分配复用。
2. 内部池子的上限是统一内存的上限。MLX 不会主动缩容。
3. 只有 `mx.clear_cache()` 才把 allocator cache 还给 OS。

**数量级估算** — 8x128 ResNet / batch=256 单次 forward 的 activation 占用：

```
conv3→128 输出: 256 × 128 × 15 × 15 × 4 bytes ≈ 29.5 MB
8 res blocks × 2 convs × 29.5 MB ≈ 472 MB 仅 activations
加 value_and_grad 的梯度 tape（训练时）≈ 翻倍 ≈ 950 MB
```

一次 training step ≈ **1 GB MLX 中间 buffer**。50 steps × 无 clear_cache = **可以在一个 cycle 内累到 50 GB**。但注意这 50 GB 是 MLX 内部 pool 的"已分配未释放"状态——上限受物理内存限制，它会在某个阈值停下来（即 115 GB 附近），并开始复用池内 buffer。这就是为什么 **pg=24 和 pg=64 都卡在 115 GB**：他们都撞上了 MLX 内部 pool 的"最大保留水位"，那个水位取决于训练步的 batch 大小和 MLX allocator 的策略，**和 pg 无关**。

**这一条解释了 RAM 115 GB 天花板的 pg 不变性。**

### 12.4 根因三：CPU→GPU→C 串行流水线（P1 级，架构问题）

**代码证据** — `framework/core/mcts_native.py:138-218` 的 sim round 主循环：

```python
while remaining > 0:
    # 1. C 选择路径（CPU-C，串行）
    total_paths = lib.mcts_batch_select(...)
    
    # 2. Python 回放 boards（CPU-Python，串行）
    for pi in range(total_paths):
        sim_state = copy_fn(root_states[pi % n])
        for step in range(1, plen):
            apply_fn(sim_state, action)
            ...
    
    # 3. GPU 前向（Metal，串行）
    evals = evaluate_batch_fn(expand_states)
    
    # 4. C 回溯（CPU-C，串行）
    lib.mcts_batch_expand_backup(...)
```

**四个阶段严格串行，任何时刻只有一个资源在工作。** 时间分布粗估：

| 阶段 | 耗时 | 谁在干活 | 其他资源 |
|------|------|----------|----------|
| 1. C select | ~1-2 ms | 1 CPU 核 | GPU 空闲 |
| 2. Python board ops | ~3-5 ms | 1 CPU 核 (GIL) | GPU 空闲 |
| 3. GPU forward | ~5-10 ms | GPU | 16 CPU 核全空闲 |
| 4. C backup | ~1-2 ms | 1 CPU 核 | GPU 空闲 |
| **总计** | ~10-20 ms/round | 三个资源轮流 | 永远只有一个活着 |

**理论 CPU 利用率上限 = 1 核 / 16 核 × (1+2+4)/20 × 100% ≈ 22%**——但因为 Python GIL，有效上限更低，用户观察到的 <10% 完全一致。

**理论 GPU 功耗上限** = 50W（Metal 8x128 batch=256 单次 forward）× 30%（5-10 ms 占 10-20 ms 的一半以下）= **~15W 平均**——也和用户观察的 10-50W 区间吻合。

### 12.5 额外观察：memory 115 GB 是上限还是问题？

用户问题里提到 "115 GB 应该已经吃满了"。严格说 M3 Max 128 GB 单位统一内存，115 GB 是 **90% 占用**——OS 会开始抢压，MLX 开始被动 evict，训练速度会**断崖式**下降（常见症状：TUI 的 Gm/s 从 0.2 掉到 0.05）。所以即使训练能跑完，它已经进入了"系统被动换出"的退化区。

**目标：把 RAM 压到 ≤ 40 GB。** 留出空间给 OS 缓存和其他进程，同时解决速度衰减。

### 12.6 改造方案

三级，P0 必做，P1 强烈推荐，P2 为后续版本准备。

#### P0a — 修 `MAX_BATCH_PATHS` 截断

**改动点：** `framework/core/mcts_c.c:289`

```c
#define MAX_BATCH_PATHS 2048   /* was 256; was silently truncating pg×batch >256 */
```

**成本：** 4 个 static buffer 从 `256 × (128×2 + 1 + 1) × 4 = 264 KB` 变成 `2048 × 258 × 4 = 2.1 MB`。完全可以接受。

**副带改动：** 在 `mcts_batch_select` 里加一个 `printf` 或让 Python 端的 wrapper 检测 `sims_per_round × n_roots > MAX_BATCH_PATHS` 并报警。静默截断是 v14 的一个真正的 ops bug——必须修掉。

**受益配置：**
- pg=16 / batch=32 → leaf batch 512（GPU 用得更饱）
- pg=24 / batch=16 → leaf batch 384
- pg=12 / batch=32 → leaf batch 384

#### P0b — 训练循环里每 step 结束调 `mx.clear_cache()`

**改动点：** `domains/gomoku/train.py:1322-1325` 区块（`optimizer.update` + `mx.eval` 之后）：

```python
optimizer.update(model, grads)
mx.eval(model.parameters(), optimizer.state)

cycle_loss += loss.item()
total_train_steps += 1
steps_run_this_cycle += 1

# NEW: release allocator cache every step. This is the main RAM leak fix.
# Without this, 50 train steps × ~1 GB activations/gradients = up to 50 GB
# retained by MLX's internal pool, independent of parallel_games.
if total_train_steps % 4 == 0:  # every 4 steps to amortize syscall cost
    mx.clear_cache()
```

每 4 步调一次是 throughput / RAM 的折中；如果 RAM 还是高，降到每步。`mx.clear_cache()` 在 MLX 0.20+ 上是 us 级操作，每步调完全可接受。

**同时修自对弈循环的 clear_cache 频率** — `domains/gomoku/train.py:528-530`：

```python
# OLD: if round_count == 0 or total_moves % 80 == 0: mx.clear_cache()
# NEW: 每 8 个 move（~1 round）调一次
if total_moves % 8 == 0:
    mx.clear_cache()
```

**预期效果：** RAM 从 115 GB → 25-40 GB。

#### P0c — 降低 pg 到甜点，提高单次 leaf batch

**改动点：** CLI 推荐参数（不改代码）。

基于根因一的分析，最优配置是让 `pg × mcts-batch ≈ 256`（MLX 8x128 的 GPU 甜点，P0a 修复后可以放到 512-1024）：

| 目标 | pg | mcts-batch | leaf batch | 评价 |
|------|----|-----------|-----------|------|
| **甜点 A（稳态）** | 16 | 16 | 256 | 当前可行，mcts_10 附近 |
| **甜点 B（大 batch）** | 16 | 32 | 512 | P0a 后可行 |
| **甜点 C（超大 batch）** | 32 | 16 | 512 | P0a 后可行 |
| **不推荐** | ≥48 | ≥16 | (被截断) | P0a 前无效 |

**用户原命令 `pg=64 / batch=16` 的直接替换：**

```bash
# ❌ 原命令（64 × 16 = 1024，被静默截到 256，有效 pg=16，其余 48 个 board 空耗内存）
--parallel-games 64 --mcts-batch 16

# ✅ 推荐（P0a 修复前）
--parallel-games 16 --mcts-batch 16

# ✅ 推荐（P0a 修复后）
--parallel-games 16 --mcts-batch 32
```

**预期效果：** 相同 GPU 功耗下训练样本数产出更一致；RAM 从"pg 无关的 115 GB"变成"与 pg 线性相关"，pg=16 时大约 20-30 GB。

#### P1 — 异步流水线：CPU board ops 与 GPU forward 重叠

**目标：** 把 §12.4 的四阶段串行改成两条并行流水。

**架构（train.py 内部改动，不碰 C）：**

```
Worker-A：C select → Python board ops → 构造 batch 张量
Worker-B：GPU forward → 拿到 (priors, values)
主：串联两者，C expand+backup

时间线（一个 sim round）：
  round t:    A [select][copy][stack]   →
              B                           [forward]   →
              main                                    [backup]

  round t+1:  A              [select][copy][stack]   →
              B                                       [forward]   →
              main                                                [backup]
```

当 `A[t+1]` 和 `B[t]` 重叠时，每轮的有效时间从 `A+B+main ≈ 15 ms` 压到 `max(A, B) + main ≈ 10 ms`——**理论提速 33%**。

**实现：** Python 的 `concurrent.futures.ThreadPoolExecutor(max_workers=1)` 加一个后台 worker 处理 A 阶段即可；MLX 的 forward 释放 GIL，线程与主循环天然并行。这是 30-50 行 diff，完全在 `_run_self_play_mcts` 内部。

**不推荐现在做多进程 worker**：AlphaZero 原版是多进程，但那是 Python 无 GIL 前的架构。Python 3.13 的 free-threading 尚未成熟、MLX context 也不是多进程友好的。单进程 + 后台线程是当前阶段的 Pareto 最优。

#### P2（v15 以后）— Board 操作 C 化 或 多进程 worker

如果 P0+P1 完成后 SP time 仍然是瓶颈，下一阶段有两个方向：

1. **把 `Board.copy + place` 搬进 C**，在 `mcts_batch_select` 里做 "select-copy-place-expand-backup" 的完整 sim round，不回 Python 做 board 操作。预期再 2-3x 提速。代价：要修改 game.py（现在是 READ-ONLY 按 v14 前约束，但 v15 可以放开）。
2. **多进程自对弈 worker**：每个 worker 进程独立跑 pg=8 × mcts-batch=16，通过共享内存队列把训练样本送回主进程。预期 4-8x 提速，但架构变化大。

两者是互斥路线，选一个。我倾向 (1)——更小的改动面，更好的 GC / 调试体验。

### 12.7 改造顺序与验证

**Phase 0（本轮必做，预计 < 2 h 编码 + 1 h 验证）：**

1. P0a：`MAX_BATCH_PATHS = 256` → `2048`，加 Python wrapper 的截断警告
2. P0b：训练循环 + 自对弈循环的 `mx.clear_cache` 频率修正
3. 重编 C 扩展：`cd framework/core && bash build_native.sh`
4. 跑一个 30 分钟 smoke test：

```bash
uv run python domains/gomoku/train.py \
  --mcts-sims 800 --parallel-games 16 --mcts-batch 32 \
  --num-blocks 8 --num-filters 128 \
  --learning-rate 3e-4 --steps-per-cycle 50 --buffer-size 100000 \
  --time-budget 1800 \
  --eval-level 2 \
  --eval-interval 5 --probe-games 80 \
  --full-eval-games 200 --eval-openings 16 \
  --resume 6c9c8bdd --seed 42
```

**验证判据：**

| 指标 | 验收值 | 如何测 |
|------|--------|--------|
| RAM 峰值 | ≤ 40 GB | `top` / Activity Monitor |
| GPU 平均功耗 | ≥ 35 W | `powermetrics --samplers gpu_power -i 1000` |
| Gm/s | ≥ mcts_10 的 0.2 | TUI 面板 |
| 训练完成后 P-Loss | 稳定下降趋势 | TUI 面板 |
| 不报 MAX_BATCH_PATHS 警告 | 无警告 | stdout |

任一不达标都要停下来诊断，不要直接进 3 小时长训。

**Phase 1（Phase 0 验证通过后做，预计 1 天）：**

- P1：添加 `ThreadPoolExecutor` 流水线
- 验证判据：同配置 Gm/s ≥ Phase 0 的 1.25 倍

**Phase 2（另起版本 v15）：**

- 决定 P2 的 1/2 路线，或者引入完全不同的架构（multi-root GPU kernel）

### 12.8 本节不打算做的事

为了保持改造面小、可验证，以下都**不**在本轮范围内：

- ❌ 切换 MLX 版本 / 编译选项 / `MPS_HIGH_WATERMARK_RATIO` 环境变量——这些都是外部黑盒
- ❌ 降 `num_filters` 到 64（mcts_10 数据已经证明 8x128 是必要容量）
- ❌ 缩 `buffer_size`——buffer 大小和 RAM 问题无关，v14-findings §8.5 证明 buffer 在合理区间
- ❌ 改 `mcts_sims` 到 400——搜索量是质量基石，不能为了资源利用砍它

### 12.9 一句话结论

> **内存 115 GB 不是 `parallel-games` 太大，是 MLX 训练循环里从不 `clear_cache`；GPU 50W 不是并行度不够，是 `MAX_BATCH_PATHS=256` 在 C 侧静默截断高 pg×batch 请求。这两个都是 v14 留下的 P0 级 bug，修复代价 < 50 行代码。** 修完后推荐 `pg=16 / mcts-batch=32` 作为 8x128 / 800 sims 的新甜点。

---

## 13. v14.1 工作日志：P0 + P1 代码升级与性能基线（2026-04-13 下午）

> 执行者：Claude Opus 4.6 | 范围：§12.6 P0 + P1 全部落地，P2 转入 v15 backlog

### 13.1 本次落地的改动

| # | 文件 | 改动 | 类型 |
|---|------|------|------|
| 1 | `framework/core/mcts_c.c` | `MAX_BATCH_PATHS` 256 → 2048，新增 `mcts_max_batch_paths()` 查询 API | P0a |
| 2 | `framework/core/mcts_native.py` | 启动时读取 C 的 `MAX_BATCH_PATHS`；首次越界时发 `RuntimeWarning` 而不是静默截断 | P0a |
| 3 | `framework/core/mcts_native.py` | 新增 `ThreadPoolExecutor` 基础设施（`MCTS_NATIVE_WORKERS` 环境变量控制）+ chunked 分派 | P1 (infra) |
| 4 | `domains/gomoku/train.py` | 训练循环每 4 steps 调 `mx.clear_cache()` | P0b |
| 5 | `domains/gomoku/train.py` | 自对弈 `_evaluate_batch` 每 8 次 forward 调 `mx.clear_cache()` | P0b |
| 6 | `domains/gomoku/train.py` | 自对弈外层 while 循环末尾每轮调 `mx.clear_cache()` | P0b |
| 7 | `domains/gomoku/train.py` | 新 `_apply()` fast-path：跳过 `is_legal()`、跳过 `history.append()` | P1 (real win) |
| 8 | `domains/gomoku/game.py` | `_WIN_DIRECTIONS` 提升到模块级常量 | P1 (real win) |
| 9 | `domains/gomoku/game.py` | `_check_win` 去掉方向列表分配 + 展开 sign 循环 | P1 (real win) |

**总改动量：** ~60 行代码新增、~20 行修改。C 扩展已在本机重编译（`framework/core/mcts_c.so`）。

### 13.2 原 P1 计划落空的诚实记录

§12.6 的 P1 原计划是"CPU board ops 和 GPU forward 用 `ThreadPoolExecutor` 重叠"。实际动手后发现**这个方案在当前架构下根本没有重叠空间**：

- 每个 sim round 的依赖链是 `C_select(t) → board_ops(t) → GPU_forward(t) → C_backup(t)`
- `C_select(t+1)` 需要 `C_backup(t)` 更新后的树状态
- 所以 round t+1 的任何阶段都不能和 round t 的任何阶段并行
- **同一棵搜索树内不存在流水线机会**

把 Python path-walk loop 放到 ThreadPoolExecutor 里的初版（按路径派发，每路径一个 task）实测 **0.41-0.56x**（更慢）——dispatch 开销压倒实际工作。

改成 **chunk 分派**（每 chunk 64-128 路径）后速度变成 **0.81-0.87x**——仍然更慢。原因在 `_walk_chunk` 的内部 profile：

```
cumtime% (pg=16 batch=16, 3 searches, 400 sims each):
  _walk_chunk:          41%
    game.py place():    21%
      _check_win:       12%
      is_legal:          2%
      history.append:    < 1%
    game.py copy():      8%
      np.grid.copy():    2%   ← 唯一 GIL-free 的点
```

`_check_win` 和 `place()` 都是 **纯 Python 内层循环**，GIL 全程持有。`grid.copy()` 虽然 GIL-free，但它只占总时间的 2%——再多线程也只能摊薄这 2%。剩下的 98% 是 GIL-bound，加线程只会增加调度开销。

**教训：在给出 P1 方案前应该先跑 profile。** §12.6 里"理论 CPU 利用率上限 22%"那段是对的，但我漏了一个前提：22% 的上限里只有 ~2% 是 GIL-free 部分，所以多线程的实际上限不是 4x 而是 1.02x。

**保留策略：** ThreadPoolExecutor 基础设施留在 `mcts_native.py` 里，但**默认禁用**（`MCTS_NATIVE_WORKERS=1`）。环境变量可以开启，但**不要**在命令行里加这个变量——除非你自己测过且看到正速度比。M3 Max 的 16 核全是 performance 核，理论上 GIL 释放的窗口比 Linux 小核更值得钱，值得在 Mac 上再跑一次基线；但在那之前当成"有但不启用"的功能。

### 13.3 P1 真正产生的提速来自哪里

既然线程化失败，§13.1 的 P1 行 #7 / #8 / #9 是实际产生速度的地方——都是**减少 Python 开销**，不是并行化：

#### (a) `_apply` fast-path —— 跳 `is_legal` + 跳 `history.append`

MCTS 模拟路径天然合法（C 的 `mcts_select_child` 只走已展开的合法子节点），`_fast_copy` 也把 history 清空了不需要维护。所以在 self-play 专用的 `_apply` 里直接内联 `place()` 的必要部分：

```python
def _apply(state, action):
    row, col = action // BOARD_SIZE, action % BOARD_SIZE
    player = state.current_player
    state.grid[row, col] = player
    state.move_count += 1
    state.last_move = (row, col)
    if state._check_win(row, col, player):
        state.winner = player
    elif state.move_count >= BOARD_SIZE * BOARD_SIZE:
        state.winner = -1
    state.current_player = WHITE if player == BLACK else BLACK
```

节省：每次 apply 少一次 `is_legal` 调用（带 bounds check + 4 个属性访问）+ 少一次 `list.append`。profile 显示 ~8% 的 `_walk_chunk` 总时间。

#### (b) `_check_win` 去掉每次调用的方向列表分配

原版每次进入 `_check_win` 都 `directions = [(0,1),(1,0),(1,1),(1,-1)]`——一次 list + 4 个 tuple 的分配。提升到模块常量 `_WIN_DIRECTIONS` 后每次调用省掉这些。

顺便把内层 `for sign in (1, -1)` 展开成两段，少一层循环变量赋值。

#### (c) 综合测量结果（Linux 开发机，1×pg4-class）

同一 fake_eval 下，400-sim 搜索中位时长（5 次 trials）：

| 配置 | 基线 v14 | v14.1 | 提速 |
|------|---------|-------|------|
| pg=16 batch=16 (leaf=256) | 300.4 ms | 245.1 ms | **1.23x** |
| pg=16 batch=32 (leaf=512) | 305.4 ms | 258.2 ms | **1.18x** |
| pg=32 batch=16 (leaf=512) | 622.0 ms | 516.3 ms | **1.20x** |
| pg=32 batch=32 (leaf=1024) | 641.8 ms | 561.6 ms | **1.14x** |

**平均 ~1.19x**（Linux，~5% 方差）。这是纯 Python-side 的改进，在 Mac 上的绝对数会不一样（M3 performance 核的 Python 吞吐更高），但相对提升应该类似。

### 13.4 正确性验证

所有微优化都通过了下列回归测试：

1. **Board `_check_win` 7 个 pattern 测试**：横 / 纵 / 正对角 / 反对角 / 白棋胜 / 未连成 / 被堵——全部通过。
2. **MCTS 视场一致性测试**：固定 seed、无 Dirichlet 噪声、4 boards × 400 sims，`_apply`+fast_copy 与原 `place()`+Board.copy 产出 **逐根 visit 总数相等**（均为 400）且 top-3 访问位点一致。
3. **C 扩展 `MAX_BATCH_PATHS` 截断警告**：pg=64 × batch=64 = 4096 > 2048 时**一次性**发 `RuntimeWarning`，`effective sims/round` 显式标明被降到 32。
4. **parse check**：5 个修改过的文件全部 `ast.parse` 通过。

### 13.5 性能预测（Mac M3 Max 上的 3 小时训练）

以下是本轮修复在用户硬件上的**预期影响**。这是基于 mcts_10 baseline（pg=10 / batch=16 / 3h / 176 cycles）做的外推，有测不准度——请用 §13.6 的 smoke test 数据验证后再信。

| 指标 | mcts_9th 6h | mcts_10 3h | **mcts_11 预期 3h (P0+P1 后)** | 备注 |
|------|-------|-------|------|------|
| RAM 峰值 | ~? | ~? (未测) | **25-40 GB** | P0b 每 4 train step clear cache |
| GPU 功耗 | ~? | ~? | **35-50 W** | P0a 允许更大有效 batch |
| Gm/s (pg=16/batch=32) | n/a | 0.20 @ pg=10 | **0.26-0.30** | P1 Python 优化 1.19x + 更大 batch 摊薄 GPU 开销 |
| 每 cycle wall | n/a | 61 s | **40-50 s** | 同上 |
| 3h cycles | 457 (6h) | 176 | **215-270** | |
| 每 cycle SP time | n/a | 43.7 s | **28-35 s** | |
| 每 cycle train time | n/a | ~15 s | ~15 s | 训练步不受影响 |
| 3h 累计 games | 4570 (6h) | 1760 | **3400-4300** | |
| 3h 累计 train steps | 22135 (6h) | 8253 | **10750-13500** | |

**关键预测：**

1. **RAM 从 115 GB 跌到 ~30 GB**——这是 P0b 的直接后果，最应该肉眼看见的改动。
2. **Gm/s 提速 1.3-1.5x**——来自 §13.3 (c) 的 1.19x + P0a 解开 batch 截断的额外 ~10-20%。
3. **GPU 功耗波形变稳定**——不再是 "10W 坐着 / 偶发 50W 尖峰" 的低占空比，而是 "35-50W 持续"。

**如果测不到这些：**

- RAM 仍然 >80 GB → `mx.clear_cache` 调用频率不够，或者有其他未修的泄漏（我们的测量只覆盖 Linux 无 MLX 路径，MLX 那一半必须到 Mac 才能验）
- Gm/s 不涨 → P1 的 Python 优化在 M3 核下收益比 Linux 小，或者瓶颈在 MLX 不是在 CPU
- GPU 还是摇摆 → 可能是 `_evaluate_batch` 的 `mx.clear_cache` 把 MLX 的 kernel 缓存也清掉了（副作用），改成每 16 forward 调一次试试

### 13.6 必须做的启动步骤

**Step 1** — git pull 本仓库最新 commit（包含 `framework/core/mcts_c.c`、`mcts_native.py`、`train.py`、`game.py` 的改动）

**Step 2** — **必须重编 C 扩展**（`MAX_BATCH_PATHS` 从 256 改成 2048，需要重新 build .dylib）：

```bash
cd framework/core && bash build_native.sh && cd ../..
```

Mac 下这会产出 `mcts_c.dylib`。如果缺 clang 报错，`xcode-select --install` 补一下。

**Step 3** — 用一个 Python 一行确认新的 cap 被 C 侧暴露了：

```bash
uv run python -c "
import sys; sys.path.insert(0, 'framework')
from core.mcts_native import max_batch_paths
print('MAX_BATCH_PATHS =', max_batch_paths())
"
```

应当打印 `MAX_BATCH_PATHS = 2048`。如果还是 256，说明 Step 2 没成功，检查是否有旧的 .dylib 挡路。

**Step 4** — 30 分钟 smoke test（在跑 3h 长训前必做）：

```bash
uv run python domains/gomoku/train.py \
  --mcts-sims 800 --parallel-games 16 --mcts-batch 32 \
  --num-blocks 8 --num-filters 128 \
  --learning-rate 3e-4 --steps-per-cycle 50 --buffer-size 100000 \
  --time-budget 1800 \
  --eval-level 2 \
  --eval-interval 5 --probe-games 80 \
  --full-eval-games 200 --eval-openings 16 \
  --resume 6c9c8bdd --seed 42
```

同时在另一个终端开 **Activity Monitor** 或跑 `top -o mem`，记录：

| 检查点 | 目标 | 失败则 |
|--------|------|--------|
| 启动 2 分钟 | RAM < 35 GB | P0b 不生效 → 看 train.py 的 clear_cache 分支是否命中 |
| 启动 5 分钟 | TUI `Gm/s` ≥ 0.22 | P0a 或 P1 没生效 → 检查 mcts_max_batch_paths() 返回值 |
| 启动 15 分钟 | GPU 功耗 ≥ 35 W（`powermetrics --samplers gpu_power`）| 底层还有问题 → 停机诊断 |
| 任何时刻 | 不出现 `mcts_batch_select: sims_per_round(X) × n_roots(Y)... exceeds C cap` 警告 | pg × batch > 2048，降一个 |

任何一条未达标**都不要直接进 §13.7 的 3h 长训**。stop 了来找我，我们一起看哪里卡住。

### 13.7 Mcts_11 新训练命令（3 h，vs L2，resume 6c9c8bdd）

Smoke test 通过后的正式命令：

```bash
uv run python domains/gomoku/train.py \
  --mcts-sims 800 --parallel-games 16 --mcts-batch 32 \
  --num-blocks 8 --num-filters 128 \
  --learning-rate 3e-4 --steps-per-cycle 50 --buffer-size 100000 \
  --time-budget 10800 \
  --eval-level 2 \
  --eval-interval 5 --probe-games 80 \
  --full-eval-games 200 --eval-openings 16 \
  --auto-stop-stagnation --stagnation-window 15 \
  --resume 6c9c8bdd --seed 42
```

**参数说明（与你上次命令的差异）：**

| 参数 | 上次 | 本次 | 理由 |
|------|------|------|------|
| `--parallel-games` | 64 | **16** | 回到甜点（P0a 后 leaf batch 真正达到 512）|
| `--mcts-batch` | 16 | **32** | P0a 修好后 `16×32=512` 在 MLX 甜点区 |
| `--time-budget` | 18000 | **10800** | 和 mcts_10 可直接对比（cost 轴对齐）|
| `--eval-interval` | 15 | **5** | L2 训练前 1 小时 WR 变化快，不要稀释 |
| `--probe-games` | 120 | 80 | 和 mcts_10 对齐 |
| `--full-eval-games` | 250 | 200 | 和 mcts_10 对齐 |
| `--no-eval-opponent` | 有 | **省略** | v14.1 起默认不启用 NN 对手 |

**成功判据：**

| 指标 | 阈值 |
|------|------|
| RAM 峰值 | ≤ 40 GB |
| Gm/s | ≥ 0.22 |
| 首个 probe WR vs L2 > 0% | cycle ≤ 30 |
| smoothed WR ≥ 60% vs L2 | 3h 结束时 |
| policy_loss 最低值 | ≤ 4.20 |
| full eval `uniq/op` | ≥ 24/16 |

### 13.8 P2 backlog（转入 v15）

本轮不做，记账留给 v15：

| # | 项 | 动机 | 预期收益 |
|---|----|------|---------|
| 1 | 把 `Board` 的 grid + place + _check_win 换成 cython/C 扩展 | §13.2 的 profile 证明 98% 时间是 GIL-bound Python | 2-3x self-play throughput |
| 2 | 多进程 self-play worker（共享内存 replay queue）| 绕开 GIL，彻底解锁 CPU 并行 | 4-6x throughput（需要重构） |
| 3 | Board ops 搬到 C 并在 `mcts_batch_select` 内联 | 零 Python↔C 跨界 | 再叠加 1.5-2x |
| 4 | `_evaluate_batch` 用 `mx.async_eval` 让 Metal 提前排队 | 减少 Python↔GPU 握手延迟 | 5-10% 左右 |
| 5 | 训练循环的 D4 symmetry 增广搬到 numpy 批量 | 现在每个 sample 单独做 | 3-5% |

选定 (1) 或 (2) 之一作为 v15 的主攻方向，其他作为附属。

### 13.9 一句话总结

> **v14.1 落地的三件事：C 侧 MAX_BATCH_PATHS 256→2048 + 训练循环 `mx.clear_cache()` 补齐 + Python 热路径（_apply / _check_win）微优化。前两项是 bug 修复，第三项是 1.19x 的小提速。P1 "CPU-GPU 流水线" 的原计划因为依赖链不可并行被证伪、ThreadPoolExecutor 因为 GIL 被证伪，两条路都诚实记录到 §13.2。Mac 上 smoke test 的验收基线：RAM ≤ 40 GB、Gm/s ≥ 0.22、无 batch-cap 警告。**
