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
