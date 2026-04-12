# Update Plan v2 — Checkpoint 追踪 · Rich TUI · 录制管线

> 2026-04-10 | MAG-Gomoku 训练基础设施升级

---

## 1. 目标

将当前"跑一次、存一个 model.safetensors"的粗放模式，升级为**完整的实验追踪 + 可视化 + 视频素材生成管线**，支持：

- 每次训练用 UUID 标识，所有产物可溯源
- 训练过程中按胜率阈值自动导出 checkpoint，附带完整元数据
- 每个 checkpoint 绑定 200 局完整对局录制（Code Bullet 视频素材）
- SQLite 单一数据源，字段完备，支持后续数据分析
- Rich TUI 实时面板（进度条 + 指标 + 胜率曲线 + 事件日志）
- 灵活的 CLI 参数（目标胜率、训练局数、评估级别等）

---

## 2. 架构设计

### 2.1 目录结构（output/ 下）

```
output/
├── tracker.db                          # SQLite 数据库（单一数据源）
├── checkpoints/                        # 模型快照
│   ├── {run_uuid}_wr065_c030.safetensors
│   ├── {run_uuid}_wr070_c045.safetensors
│   └── ...
├── recordings/
│   ├── games/                          # JSON 对局记录
│   │   ├── {run_uuid}_wr065_c030_game000.json
│   │   ├── {run_uuid}_wr065_c030_game001.json
│   │   └── ...
│   ├── frames/                         # 关键帧截图 (PNG)
│   └── metrics/                        # (废弃，迁移到 SQLite)
├── model.safetensors                   # 最终模型（训练结束覆盖写入）
└── run.log                             # 可选日志文件
```

### 2.2 SQLite Schema (output/tracker.db)

```sql
-- 训练运行（一次 train.py 执行 = 一个 run）
CREATE TABLE runs (
    id            TEXT PRIMARY KEY,        -- UUID v4
    started_at    TEXT NOT NULL,           -- ISO 8601
    finished_at   TEXT,                    -- ISO 8601, NULL if running
    status        TEXT DEFAULT 'running',  -- running | completed | failed | interrupted
    
    -- 硬件环境
    chip          TEXT,                    -- e.g. "Apple M3 Max"
    cpu_cores     INTEGER,
    gpu_cores     INTEGER,
    memory_gb     INTEGER,
    mlx_version   TEXT,
    
    -- 超参数快照（训练开始时冻结）
    num_res_blocks   INTEGER,
    num_filters      INTEGER,
    learning_rate    REAL,
    weight_decay     REAL,
    batch_size       INTEGER,
    parallel_games   INTEGER,
    mcts_simulations INTEGER,
    temperature      REAL,
    temp_threshold   INTEGER,
    replay_buffer_size INTEGER,
    train_steps_per_cycle INTEGER,
    time_budget      INTEGER,             -- seconds
    target_win_rate  REAL,                 -- NULL = no early stop
    target_games     INTEGER,             -- NULL = use time budget
    eval_level       INTEGER,
    
    -- 最终摘要（训练结束时填充）
    total_cycles     INTEGER,
    total_games      INTEGER,
    total_steps      INTEGER,
    final_loss       REAL,
    final_win_rate   REAL,
    num_params       INTEGER,
    num_checkpoints  INTEGER,
    wall_time_s      REAL,
    peak_memory_mb   REAL
);

-- 每 cycle 的指标快照（用于绘图 + 数据分析）
CREATE TABLE cycle_metrics (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES runs(id),
    cycle         INTEGER NOT NULL,
    timestamp_s   REAL NOT NULL,           -- seconds since run start
    loss          REAL,
    total_games   INTEGER,
    total_steps   INTEGER,
    buffer_size   INTEGER,
    win_rate      REAL,                    -- NULL if this cycle didn't eval
    eval_type     TEXT,                    -- NULL | 'probe' | 'full'
    eval_games    INTEGER,                 -- number of eval games played
    eval_level    INTEGER
);

-- Checkpoint 快照（胜率阈值触发）
CREATE TABLE checkpoints (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES runs(id),
    tag           TEXT NOT NULL UNIQUE,    -- e.g. "{run_uuid_short}_wr065_c030"
    cycle         INTEGER NOT NULL,
    step          INTEGER NOT NULL,
    loss          REAL NOT NULL,
    
    -- 评估结果
    win_rate      REAL NOT NULL,
    eval_level    INTEGER NOT NULL,
    eval_games    INTEGER NOT NULL,
    wins          INTEGER,
    losses        INTEGER,
    draws         INTEGER,
    avg_game_length REAL,
    
    -- 模型信息
    num_params    INTEGER,
    model_path    TEXT NOT NULL,           -- 相对于项目根目录 e.g. "output/checkpoints/xxx.safetensors"
    model_size_bytes INTEGER,
    
    -- 时间信息
    created_at    TEXT NOT NULL,           -- ISO 8601
    train_elapsed_s REAL,                 -- seconds since run start
    eval_elapsed_s  REAL                  -- evaluation duration
);

-- 对局录制（与 checkpoint 绑定）
CREATE TABLE recordings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id INTEGER NOT NULL REFERENCES checkpoints(id),
    run_id        TEXT NOT NULL REFERENCES runs(id),
    game_index    INTEGER NOT NULL,        -- 0-based within this checkpoint eval
    game_file     TEXT NOT NULL,           -- 相对路径 e.g. "output/recordings/games/xxx.json"
    result        TEXT NOT NULL,           -- "black_win" | "white_win" | "draw"
    total_moves   INTEGER NOT NULL,
    black         TEXT NOT NULL,           -- "nn" or "minimax_L{n}"
    white         TEXT NOT NULL,
    nn_side       TEXT NOT NULL,           -- "black" | "white"
    nn_won        INTEGER NOT NULL         -- 1 = NN won, 0 = NN lost/draw
);

CREATE INDEX idx_cycle_metrics_run ON cycle_metrics(run_id, cycle);
CREATE INDEX idx_checkpoints_run ON checkpoints(run_id);
CREATE INDEX idx_recordings_checkpoint ON recordings(checkpoint_id);
CREATE INDEX idx_recordings_run ON recordings(run_id);
```

### 2.3 Checkpoint 阈值逻辑

```python
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75,   # <80%: 每 5%
              0.80, 0.82, 0.84, 0.86, 0.88,           # 80-90%: 每 2%
              0.90, 0.91, 0.92, 0.93, 0.94, 0.95,     # >90%: 每 1%
              0.96, 0.97, 0.98, 0.99, 1.00]

def should_checkpoint(current_wr, last_checkpoint_wr) -> bool:
    """当 current_wr 跨越了下一个阈值时返回 True"""
    for t in THRESHOLDS:
        if last_checkpoint_wr < t <= current_wr:
            return True
    return False
```

### 2.4 混合评估策略

```
训练循环:
  每 1 cycle:   记录 cycle_metrics (loss, games, steps)
  每 10 cycles: 轻量探测评估 (50 局, in-process) → 记录 win_rate
  触达阈值时:   完整评估 (200 局, subprocess) + 全量录制 + 导出 checkpoint
```

### 2.5 CLI 参数

```bash
uv run python src/train.py \
  --time-budget 300 \           # 训练时间预算（秒）, 默认 300
  --target-win-rate 0.95 \      # 达到此胜率提前结束, 默认 None (跑满时间)
  --target-games 10000 \        # 达到此自对弈局数提前结束, 默认 None
  --eval-level 0 \              # 评估对手等级 0-3, 默认 0
  --eval-interval 10 \          # 每 N cycles 探测评估一次, 默认 10
  --probe-games 50 \            # 探测评估局数, 默认 50
  --full-eval-games 200 \       # 完整评估局数, 默认 200
  --resume output/model.safetensors  # 从已有模型继续训练, 默认 None
```

### 2.6 Rich TUI 面板布局

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

---

## 3. 文件变更清单

| 文件 | 操作 | 说明 |
|---|---|---|
| `src/tracker.py` | **新建** | SQLite 数据库管理：schema 初始化、CRUD、阈值逻辑、查询工具 |
| `src/train.py` | **重写** | CLI 参数解析 (argparse)、Rich TUI 面板、checkpoint 导出逻辑、cycle_metrics 写入 |
| `src/prepare.py` | **修改** | evaluate_win_rate() 支持 tag 命名录制、全量录制、返回文件路径列表；删除旧 manifest/CSV 代码 |
| `pyproject.toml` | **修改** | 新增 `rich>=13.0` 依赖 |
| `data/results.tsv` | **删除** | 被 SQLite 取代 |
| `.gitignore` | **修改** | 确保 `output/` 整体 gitignore (已包含 tracker.db) |
| `README.md` | **更新** | 新增 CLI 参数说明、查询 tracker.db 的示例 |
| `docs/action-plan.md` | **更新** | 记录 v2 升级变更 |

---

## 4. 任务分解

### Task 1: `tracker` — SQLite 追踪模块
**新建 `src/tracker.py`**

- `init_db(db_path)` — 创建/迁移 schema
- `create_run(run_id, hyperparams, hardware_info)` → 写入 runs 表
- `finish_run(run_id, summary)` → 更新 runs.finished_at + 摘要字段
- `save_cycle_metric(run_id, metric_dict)` → 写入 cycle_metrics
- `save_checkpoint(run_id, checkpoint_dict)` → 写入 checkpoints，返回 checkpoint_id
- `save_recording(checkpoint_id, run_id, recording_dict)` → 写入 recordings
- `should_checkpoint(current_wr, last_ckpt_wr)` → 阈值判断
- `get_run_summary(run_id)` → 查询单次训练摘要
- `get_checkpoints(run_id)` → 列出所有 checkpoint
- `get_recordings(checkpoint_id)` → 列出关联录制
- `collect_hardware_info()` → 采集芯片、核心数、内存、MLX 版本

**依赖**: 无

### Task 2: `deps-cleanup` — 依赖更新与清理
- `pyproject.toml` 添加 `rich>=13.0`
- `uv sync` 安装
- 删除 `data/results.tsv` 和 `data/` 目录
- 从 `prepare.py` 中删除 `archive_checkpoint()`, `list_checkpoints()`, `log_metrics()` 等旧代码
- 删除 `prepare.py` 中 `CHECKPOINT_DIR`, `_METRIC_COLUMNS` 等废弃常量
- 更新 `.gitignore`（如需要）

**依赖**: tracker

### Task 3: `recording-bind` — 录制与 Checkpoint 绑定
**修改 `src/prepare.py`**

- `evaluate_win_rate()` 签名变更:
  ```python
  def evaluate_win_rate(
      model_path: str,
      level: int = 0,
      n_games: int = 200,
      record_games: int = 0,    # 改为默认不录制
      tag: str = "",            # 新增：用于文件命名
      run_id: str = "",         # 新增：用于 tracker DB
  ) -> dict:
  ```
- 录制文件命名: `{tag}_game{i:03d}.json` (三位数补零)
- 返回值新增 `"recorded_files": [list of paths]`
- 返回值新增 `"eval_elapsed_s": float`

**依赖**: tracker

### Task 4: `checkpoint-logic` — Checkpoint 导出逻辑
**修改 `src/train.py`**

核心流程改造:
```python
# 训练主循环
run_id = str(uuid.uuid4())
tracker.create_run(run_id, ...)

for cycle in ...:
    # 自对弈 + 训练 (保持现有逻辑)
    ...
    
    # 记录 cycle metrics
    tracker.save_cycle_metric(run_id, {...})
    
    # 每 eval_interval cycles: 轻量探测
    if cycle % eval_interval == 0:
        probe_wr = quick_eval(model, probe_games)  # in-process, 50 局
        
        # 检查阈值
        if tracker.should_checkpoint(probe_wr, last_ckpt_wr):
            # 保存 checkpoint 文件
            tag = f"{run_id[:8]}_wr{int(probe_wr*100):03d}_c{cycle:04d}"
            ckpt_path = f"output/checkpoints/{tag}.safetensors"
            save_model(model, ckpt_path)
            
            # 完整评估 + 全量录制 (subprocess)
            result = full_eval(ckpt_path, full_eval_games, tag, run_id)
            
            # 写入 DB
            ckpt_id = tracker.save_checkpoint(run_id, {...})
            for rec in result["recorded_files"]:
                tracker.save_recording(ckpt_id, run_id, {...})
            
            last_ckpt_wr = probe_wr
    
    # 提前结束检查
    if target_win_rate and probe_wr >= target_win_rate:
        break
    if target_games and total_games >= target_games:
        break

tracker.finish_run(run_id, {...})
```

**依赖**: tracker, recording-bind

### Task 5: `train-tui` — Rich TUI 面板
**修改 `src/train.py`**

- 用 `rich.live.Live` + `rich.layout.Layout` 构建面板
- 组件:
  - `rich.progress.Progress` — 时间进度条
  - `rich.table.Table` — 实时指标表
  - 自定义 sparkline — 胜率/loss 历史曲线 (用 Unicode block chars ▁▂▃▄▅▆▇█)
  - `rich.text.Text` — 事件日志 (滚动，保留最近 10 条)
- 每 cycle 更新面板 (通过 `live.update()`)
- checkpoint 事件用 `[bold green]✓[/]` 高亮
- 非 TTY 时 fallback 为普通 print 输出（保证 subprocess/pipe 兼容）

**依赖**: tracker, deps-cleanup

### Task 6: `cli-args` — CLI 参数解析
**修改 `src/train.py`**

- 使用 `argparse` 替代硬编码常量
- 参数覆盖顺序: CLI args > train.py 常量 > prepare.py 默认值
- 参数列表见 2.5 节
- `--resume` 支持从已有模型文件继续训练

**依赖**: 无 (可与 tracker 并行)

### Task 7: `docs-update` — 文档更新
- `README.md`: 新增 CLI 参数说明、tracker.db 查询示例、录制回放说明
- `docs/action-plan.md`: 记录 v2 升级

**依赖**: 所有实现任务完成后

### Task 8: `verify-run` — 端到端验证
- 运行 5min 训练，验证:
  - ✅ Rich TUI 正常渲染
  - ✅ Checkpoint 按胜率阈值导出
  - ✅ SQLite 数据完整（runs, cycle_metrics, checkpoints, recordings）
  - ✅ 录制文件与 DB 记录一一对应
  - ✅ output/ 目录结构符合预期
  - ✅ replay.py 可回放录制的对局
  - ✅ CLI 参数 (--target-win-rate, --time-budget) 工作正常

**依赖**: 所有实现任务

---

## 5. 依赖图

```
tracker ─────→ deps-cleanup ────→ train-tui ──→ verify-run
  │                                               ↑
  ├──→ recording-bind ──→ checkpoint-logic ────────┤
  │                                                │
cli-args ──────────────────────────────────────────┤
                                                   │
docs-update ───────────────────────────────────────┘
```

## 6. 废弃清单

| 废弃项 | 替代 |
|---|---|
| `data/results.tsv` | `output/tracker.db` → `runs` + `checkpoints` 表 |
| `~/.cache/mag-gomoku/checkpoints/` + manifest.json | `output/checkpoints/` + `checkpoints` 表 |
| `output/recordings/metrics/training_log.csv` | `output/tracker.db` → `cycle_metrics` 表 |
| `prepare.py` 中 `archive_checkpoint()` | `tracker.save_checkpoint()` |
| `prepare.py` 中 `list_checkpoints()` | `tracker.get_checkpoints()` |
| `prepare.py` 中 `log_metrics()` | `tracker.save_cycle_metric()` |
| `prepare.py` 中 `CHECKPOINT_DIR` 常量 | 删除 |
| `prepare.py` 中 `_METRIC_COLUMNS` 常量 | 删除 |
| `train.py` 末尾的 print summary | Rich TUI 面板 |

## 7. 后续查询示例

```sql
-- 查看某次训练的所有 checkpoint
SELECT tag, cycle, win_rate, loss, model_path 
FROM checkpoints WHERE run_id = ? ORDER BY cycle;

-- 查看胜率曲线数据
SELECT cycle, timestamp_s, win_rate 
FROM cycle_metrics WHERE run_id = ? AND win_rate IS NOT NULL;

-- 查看某个 checkpoint 的所有录制对局
SELECT game_file, result, total_moves, nn_side, nn_won
FROM recordings WHERE checkpoint_id = ?;

-- 跨训练对比
SELECT r.id, r.started_at, r.num_filters, r.final_win_rate, r.wall_time_s
FROM runs r ORDER BY r.started_at DESC;

-- 导出为 CSV (命令行)
sqlite3 -header -csv output/tracker.db "SELECT * FROM checkpoints" > checkpoints.csv
```

---

## 8. 执行结果（回填）

> 实施日期: 2026-04-10 | Commit: `5562dc4`

### 8.1 实施概要

所有 8 个任务全部完成并通过验证。

| 任务 | 状态 | 关键交付物 |
|---|---|---|
| Task 1: tracker | ✅ | `src/tracker.py` (452 行) — 4 表 schema, CRUD, 硬件检测 |
| Task 2: deps-cleanup | ✅ | `pyproject.toml` +rich, 删除 `data/results.tsv`, 清理 prepare.py 旧代码 |
| Task 3: recording-bind | ✅ | `evaluate_win_rate()` 重写: tag 命名, 全量录制, game_details |
| Task 4: cli-args | ✅ | argparse 8 个参数: --time-budget, --eval-level, --probe-games 等 |
| Task 5: checkpoint-logic | ✅ | UUID run_id, probe/full 混合评估, 阈值导出, subprocess eval |
| Task 6: train-tui | ✅ | Rich Live 面板: 进度条 + 指标表 + sparkline + 事件日志 |
| Task 7: docs-update | ✅ | README.md 全面更新 |
| Task 8: verify-run | ✅ | 120s 验证运行通过 |

### 8.2 验证运行结果 (120s, --eval-level 0)

```
- 2 个 checkpoint 自动导出 (wr060 + final)
- 100 个对局录制 (50 per checkpoint)
- SQLite 数据完整: 1 run, 6 probe metrics, 2 checkpoints, 100 recordings
- Rich TUI 面板正常渲染, 非 TTY 回退到 print
- CLI 参数全部生效
```

### 8.3 实施中遇到的问题及修复

1. **cpu_cores 解析**:  macOS 26 下 `system_profiler` 返回 `"proc 16:12:4"` 格式 → 按 `:` 分割取首段
2. **GPU 核心数推断**: plist 无 `platform_number_gpu_cores` 字段 → 从 chip 名称推断 (M3 Max → 40)
3. **play.py 遗留 bug**: 仍引用 prepare.py 中已删除的 `CHECKPOINT_DIR` 和 `list_checkpoints()` → 未修复 (v3 计划)

### 8.4 已知遗留问题

- `play.py` 因 broken imports 无法运行 (ImportError)
- 所有训练产出共用 `output/checkpoints/` 和 `output/recordings/` — 不同 run 互相污染
- `--resume` 仅支持文件路径，不支持 UUID 续训
- 以上问题全部在 v3 中解决
