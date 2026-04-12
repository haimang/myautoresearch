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
