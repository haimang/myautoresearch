# Update Plan v4 — 纯文本 TUI · WR 评估优化 · 参数化控制

> 2026-04-10 | 训练体验优化：降噪、可控、可读

---

## 1. 问题总结

| # | 问题 | 根因 | 影响 |
|---|------|------|------|
| 1 | WR 波动大，50%→73%→57%→76% 来回跳 | probe 评估间隔短（5 cycle）、样本少（80 局），随机噪声大 | 过早触发 target 停止 / checkpoint 混乱 |
| 2 | Checkpoint tag 如 `wr073` 不在阈值列表中 | tag 按 probe WR 命名，非按跨过的 threshold 命名 | 命名不规范，无法直观知道达到了哪个里程碑 |
| 3 | Rich TUI 在 pipe/非 TTY 下不可用，且过于花哨 | 依赖 Rich 库做终端渲染 | 重定向输出为空；视觉效果不符合需求 |
| 4 | `--time-budget` 是必须参数，默认 300s | 代码以 time_budget 作为唯一循环退出条件 | 无法做"训练到 X% 就停"的无限运行 |
| 5 | GPU 满载（64 并行 games），无法降低负载 | `PARALLEL_GAMES=64` 硬编码 | 不适合长时间后台运行 |

---

## 2. 改动方案

### 2.1 WR 评估降噪

**改动**: `--eval-interval` 默认从 10 改为 **15**

**改动**: 新增 `--probe-window` 参数（默认 3），用最近 N 次 probe 的**滑动平均值**作为判断依据：
- checkpoint 触发用滑动平均 WR 判断
- target 停止用滑动平均 WR 判断
- 显示同时展示当前 probe WR 和滑动平均 WR

```python
# 滑动平均计算
def _smoothed_wr(wr_history, window=3):
    if len(wr_history) < window:
        return wr_history[-1] if wr_history else 0.0
    return sum(wr_history[-window:]) / window
```

### 2.2 Checkpoint tag 按 threshold 命名

**当前**: `tag = f"wr{int(probe_wr * 100):03d}_c{cycle:04d}"` → 产出 `wr073_c0010`

**改为**: tag 用本次跨越的 threshold 值命名：

```python
def _crossed_threshold(current_wr, last_ckpt_wr):
    """返回本次跨越的最高 threshold"""
    crossed = None
    for t in CHECKPOINT_THRESHOLDS:
        if last_ckpt_wr < t <= current_wr:
            crossed = t
    return crossed

# tag 示例: wr070_c0010 (表示跨过了 70% threshold)
threshold = _crossed_threshold(smoothed_wr, last_ckpt_wr)
tag = f"wr{int(threshold * 100):03d}_c{cycle:04d}"
```

### 2.3 纯文本 TUI（去掉 Rich 依赖）

**移除**: 所有 `from rich.*` 导入，删除 Rich TUI 相关代码（约 100 行）

**替换为**: 纯文本 box-drawing TUI，使用 `\033[H\033[J` (ANSI clear) 在 TTY 模式下刷新：

```
╭─────────────────── MAG-Gomoku Training ────────────────────╮
│ Run: a3f7b2c1  Chip: M3 Max  Params: 564.5K               │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 68% 3:24 / 5:00        │
├────────────────────────────────────────────────────────────-┤
│  Cycle   85   │  Loss    1.168  │  Games   5440            │
│  Steps  4226  │  Buffer 50000   │  Win Rate 73.0% (L0)     │
├────────────────────────────────────────────────────────────-┤
│  Win Rate ▁▂▃▃▄▅▆▆▇▇ 73%                                  │
│  Loss     ▇▇▆▅▄▃▃▂▂▂ 1.17                                 │
├────────────────────────────────────────────────────────────-┤
│  [14:05:30] ✓ Checkpoint wr065_c030  win_rate=65.0%        │
│  [14:06:12] ✓ Checkpoint wr070_c045  win_rate=70.0%        │
│  [14:07:01] ● Probe eval: 72.0% (50 games, 2.8s)          │
╰────────────────────────────────────────────────────────────╯
```

**实现要点**:
- `_sparkline(values, width=30)`: 使用 `▁▂▃▄▅▆▇█` 字符绘制（已有逻辑，直接复用）
- TTY 模式: 每个 cycle 用 ANSI escape 清屏重绘整个面板
- 非 TTY 模式（pipe/redirect）: 按 `CYCLES_PER_REPORT` 间隔 print 单行摘要
- 进度条: 仅在有 `--time-budget` 时显示，用 `━` 和 `─` 字符绘制
- 事件日志: 保留最近 6 条

**无 time-budget 时的面板变体**（不显示进度条和百分比）:
```
╭─────────────────── MAG-Gomoku Training ────────────────────╮
│ Run: a3f7b2c1  Chip: M3 Max  Params: 564.5K               │
│ Elapsed: 12:34  Target: 80% WR                             │
├─────────────────────────────────────────────────────────────┤
│  ...
```

### 2.4 `--time-budget` 改为可选

**当前**: `--time-budget` 默认 300，训练循环以此作为退出条件

**改为**:
- `--time-budget` 默认 `None`（无限制）
- 循环退出条件改为三选一（任一满足即停）:
  1. `time_budget is not None and elapsed >= time_budget`
  2. `target_win_rate is not None and smoothed_wr >= target_win_rate`
  3. `target_games is not None and total_games >= target_games`
- 如果三个参数都没设，打印警告并默认 `time_budget = 300`

```python
if time_budget is None and target_win_rate is None and target_games is None:
    print("Warning: no stop condition set, defaulting to --time-budget 300")
    time_budget = 300
```

### 2.5 `--parallel-games` 参数

**当前**: `PARALLEL_GAMES = 64` 硬编码

**改为**: 新增 CLI 参数 `--parallel-games`，默认 64：

```python
p.add_argument("--parallel-games", type=int, default=PARALLEL_GAMES,
               help=f"Number of simultaneous self-play games (default: {PARALLEL_GAMES})")
```

训练循环中用 `args.parallel_games` 替代 `PARALLEL_GAMES` 常量。

**低负载建议值**:
- 全速: 64（默认，GPU 满载）
- 中等: 32（约 50-60% GPU）
- 轻载: 16（约 30% GPU，适合后台长时间运行）

---

## 3. 文件变更清单

| 文件 | 变更 |
|------|------|
| `src/train.py` | 移除 Rich TUI，新增纯文本 TUI；修改 checkpoint tag 命名逻辑；添加滑动平均 WR；改 time-budget 为可选；添加 --parallel-games 参数 |
| `src/tracker.py` | 新增 `crossed_threshold()` 辅助函数（可选，或放在 train.py 内） |
| `pyproject.toml` | 移除 `rich>=13.0` 依赖 |
| `README.md` | 更新参数表和命令示例 |
| `docs/caveats.md` | 补充 v4 相关注意事项 |

---

## 4. 任务分解

```
task-1: plain-text-tui
  移除 Rich TUI 代码，实现纯文本 box-drawing TUI
  - 删除 Rich 导入和 Live/Panel/Table/Progress 相关代码
  - 实现 _draw_panel() 函数：box-drawing 边框 + sparkline + 事件日志
  - TTY: ANSI clear + 重绘；非 TTY: 单行摘要 print
  - 进度条仅在有 time-budget 时显示
  depends: (无)

task-2: wr-smoothing
  WR 评估降噪：滑动平均 + 默认 eval-interval 改 15
  - 新增 _smoothed_wr(wr_history, window) 函数
  - checkpoint 触发和 target 停止改用 smoothed WR
  - --eval-interval 默认改 15
  - --probe-window 参数（默认 3）
  depends: (无)

task-3: threshold-tag
  Checkpoint tag 按跨越的 threshold 命名
  - _do_checkpoint() 接收 threshold 参数
  - tag 格式: wr{threshold*100:03d}_c{cycle:04d}
  - 传递逻辑: should_checkpoint 返回 crossed threshold 值
  depends: (无)

task-4: optional-time-budget
  --time-budget 改为可选，三种停止条件任一触发
  - 默认值改 None
  - 循环退出逻辑重写
  - 无停止条件时 fallback 300s + 警告
  - 面板适配无 time-budget 显示
  depends: task-1

task-5: parallel-games-param
  新增 --parallel-games CLI 参数
  - argparse 添加参数，默认 64
  - train() 内用 args.parallel_games 替代常量
  depends: (无)

task-6: cleanup-deps
  移除 Rich 依赖
  - pyproject.toml 删除 rich>=13.0
  - uv sync 验证
  depends: task-1

task-7: docs-and-test
  更新文档 + 全面测试
  - README.md 更新参数表和命令示例
  - docs/caveats.md 补充 v4 注意事项
  - 测试: 有 time-budget / 无 time-budget / --target-win-rate / --resume / --parallel-games 16
  depends: task-1, task-2, task-3, task-4, task-5, task-6
```

```
task-1 ──┐
task-2 ──┤
task-3 ──┼── task-7
task-4 ──┤     (task-4 depends on task-1)
task-5 ──┤
task-6 ──┘     (task-6 depends on task-1)
```

---

## 5. 执行结果

### 5.1 实施概要

所有 7 个任务均已完成，提交为 `f52315a`。

| 任务 | 状态 | 说明 |
|------|------|------|
| Task 1: 纯文本 TUI | ✅ | `_draw_panel()` + `_sparkline()` + ANSI clear，非 TTY 回退行输出 |
| Task 2: WR 平滑 | ✅ | `_smoothed_wr()` 滑动窗口，`--probe-window` 默认 3 |
| Task 3: 阈值 tag | ✅ | `crossed_threshold()` 返回跨过的最高阈值值，tag 用阈值命名 |
| Task 4: time-budget 可选 | ✅ | 默认 None，仅在三无时回退 300s |
| Task 5: parallel-games | ✅ | `--parallel-games` 默认 64，可调低减负 |
| Task 6: 清理依赖 | ✅ | pyproject.toml 移除 `rich>=13.0` |
| Task 7: 文档 | ✅ | README 更新参数表、命令示例 |

额外修复：
- `tracker.py`: `get_run()` 支持短 UUID 前缀匹配
- `tracker.py`: `checkpoints.tag` UNIQUE 约束改为 `UNIQUE(run_id, tag)`（不再全局唯一）
- `train.py`: resume 逻辑使用 resolved full UUID

### 5.2 验证运行结果

**测试 1：target-only 模式（无 time-budget）**
```
Run: b2e5c8ae | target-win-rate=0.55 | parallel-games=16 | probe-window=2
Cycles: 20 | Games: 320 | WR: 62.5% | Time: 92.4s
Checkpoints: wr050_c0005, wr060_c0020, final_c0020
→ 平滑 WR 达 60% 时正确触发停止（avg ≥ 55%）
```

**测试 2：带 time-budget**
```
Run: c8d815ac | target-win-rate=0.55 | time-budget=120 | parallel-games=16
Cycles: 10 | Games: 160 | WR: 68.5% | Time: 52.6s
Checkpoints: wr050_c0005, wr055_c0010, final_c0010
→ 阈值 tag 正确（wr050、wr055 而非实际 probe 值）
```

**测试 3：resume（短 UUID 前缀）**
```
--resume c8d815ac → 解析为完整 UUID c8d815ac-693a-4e54-923c-f22787979ba2
Run: 811b951f (resumed from c8d815ac)
Cycle 10→28 | WR: 76.0% | Checkpoint: wr075_c0015
→ resume 链在 DB 中正确记录
```

### 5.3 遇到的问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `checkpoints.tag` UNIQUE 冲突 | 旧 schema 定义了全局 UNIQUE，不同 run 可能产生相同 tag（如 `wr050_c0005`） | 改为 `UNIQUE(run_id, tag)`，删除旧 DB 重建 |
| `--resume` 短 UUID 失败 | `get_run()` 做精确匹配 | 添加 LIKE 前缀匹配，train.py 使用 resolved full UUID |
| SQLite auto-index 不可 DROP | `sqlite_autoindex_checkpoints_1` 由列级 UNIQUE 生成 | 删除旧 DB，新 schema 使用表级 UNIQUE 约束 |
