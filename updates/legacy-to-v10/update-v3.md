# Update Plan v3 — Output UUID 隔离 · 断点续训 · play.py 修复

> 2026-04-10 | MAG-Gomoku 训练产出隔离与断点续训

---

## 1. 目标

解决 v2 遗留的三个结构性问题：

1. **产出物混杂**: 所有训练 run 的 checkpoints/recordings 共用同一目录，互相污染，无法按 run 整理/清理
2. **无法断点续训**: 训练中断后只能从头开始，浪费已有的训练进度
3. **play.py 无法运行**: 引用了 prepare.py 中已删除的 `CHECKPOINT_DIR` 和 `list_checkpoints()`

## 2. 架构变更

### 2.1 新 output/ 目录结构

```
output/
├── tracker.db                              # 全局数据库（不动）
├── <uuid-1>/                               # Run 1 的所有产出
│   ├── model.safetensors                   # 该 run 的工作模型
│   ├── checkpoints/
│   │   ├── wr060_c0005.safetensors         # tag 不再带 run_id 前缀
│   │   ├── wr070_c0015.safetensors
│   │   └── final_c0030.safetensors
│   └── recordings/
│       └── games/
│           ├── wr060_c0005_game000.json
│           ├── wr060_c0005_game001.json
│           └── ...
├── <uuid-2>/                               # Run 2
│   └── ...
```

**关键变化**：
- `output/model.safetensors` → `output/<uuid>/model.safetensors`
- `output/checkpoints/{run_short}_xxx.safetensors` → `output/<uuid>/checkpoints/xxx.safetensors`
- `output/recordings/games/{run_short}_xxx.json` → `output/<uuid>/recordings/games/xxx.json`
- Tag 格式: `wr060_c0005` (不再包含 run_id_short，因为目录已经隔离)
- `tracker.db` 不动 — 它是跨 run 的全局索引

### 2.2 DB Schema 迁移

`runs` 表新增两列（ALTER TABLE，向后兼容）：
```sql
ALTER TABLE runs ADD COLUMN resumed_from TEXT;   -- 续训来源 run UUID
ALTER TABLE runs ADD COLUMN output_dir TEXT;      -- e.g. "output/<uuid>"
```

新增查询函数：
- `get_latest_checkpoint(conn, run_id)` → 该 run 最新的 checkpoint（按 cycle DESC）
- `list_all_checkpoints(conn, limit)` → 跨 run 列出所有 checkpoint

### 2.3 断点续训流程

```
uv run python src/train.py --resume <uuid>
```

1. 用 `<uuid>` 查询 DB → 获取该 run 的最新 checkpoint
2. 加载 checkpoint 模型权重
3. 恢复: cycle 起始编号, last_ckpt_wr
4. 创建**新 run** (新 UUID)，设 `resumed_from = <old_uuid>`
5. 新 run 的产出写入 `output/<new_uuid>/`
6. **不恢复**: replay buffer (内存态), optimizer state (重新预热)

### 2.4 play.py 修复方案

- 删除 `from prepare import CHECKPOINT_DIR, list_checkpoints`
- 改用 `tracker.py` 查询 DB：
  - `list_all_checkpoints()` 替代 `list_checkpoints()`
  - `resolve_checkpoint()` 先查 DB tag → 再回退到文件路径
  - 不再依赖 `CHECKPOINT_DIR` 常量

---

## 3. 文件变更清单

| 文件 | 操作 | 说明 |
|---|---|---|
| `src/tracker.py` | **修改** | ALTER TABLE 迁移 + 新增 get_latest_checkpoint / list_all_checkpoints |
| `src/train.py` | **修改** | 动态 output 路径 (output/{run_id}/) + tag 简化 + recording_dir 传递 + --resume UUID |
| `src/prepare.py` | **修改** | evaluate_win_rate() 新增 recording_dir 参数 |
| `src/play.py` | **修改** | 移除 broken imports, 改用 tracker.py |
| `.gitignore` | **修改** | 新增 update-v3.md |
| `README.md` | **更新** | 新目录结构, --resume 文档 |

---

## 4. 任务分解

### Task 1: schema-migration
**文件**: `src/tracker.py`

变更：
- 在 `init_db()` 后追加 ALTER TABLE 迁移（用 try/except 处理列已存在的情况）
- 新增 `get_latest_checkpoint(conn, run_id) -> dict|None`
  - `SELECT * FROM checkpoints WHERE run_id = ? ORDER BY cycle DESC LIMIT 1`
- 新增 `list_all_checkpoints(conn, limit=50) -> list[dict]`
  - `SELECT c.*, r.chip, r.eval_level as run_eval_level FROM checkpoints c JOIN runs r ON c.run_id = r.id ORDER BY c.created_at DESC LIMIT ?`
- 新增 `get_run_output_dir(conn, run_id) -> str|None`
  - 从 runs.output_dir 获取

**依赖**: 无

### Task 2: output-uuid-dirs
**文件**: `src/train.py`

变更：
- 删除模块级 `MODEL_PATH = "output/model.safetensors"` 常量
- 在 `train()` 中计算动态路径:
  ```python
  output_dir = f"output/{run_id}"
  model_path = f"{output_dir}/model.safetensors"
  ckpt_dir = f"{output_dir}/checkpoints"
  recording_dir = f"{output_dir}/recordings"
  os.makedirs(ckpt_dir, exist_ok=True)
  ```
- 更新 `create_run()` 调用，传入 output_dir
- Tag 格式简化: `wr{wr:03d}_c{cycle:04d}` (不再带 run_id_short)
- 更新 `_do_checkpoint()`: 接收 ckpt_dir 和 recording_dir 参数
- 更新 `_subprocess_eval()`: 传递 recording_dir
- 更新 final model save 和 final checkpoint
- 更新 summary print 中的路径信息

**依赖**: schema-migration

### Task 3: prepare-recording-dir
**文件**: `src/prepare.py`

变更：
- `evaluate_win_rate()` 新增 `recording_dir: str = ""` 参数
- 若 recording_dir 为空，回退到 `RECORDING_DIR` 常量
- `games_dir = os.path.join(recording_dir or RECORDING_DIR, "games")`
- 其余逻辑不变

**依赖**: 无（可与 Task 1 并行）

### Task 4: resume-checkpoint
**文件**: `src/train.py`

变更：
- `--resume` 参数: 接受 UUID 字符串（而非文件路径）
- 在 `train()` 开头新增恢复逻辑:
  ```python
  resumed_from = None
  initial_cycle = 0
  initial_ckpt_wr = 0.0
  
  if args.resume:
      # args.resume is a UUID
      db_conn_tmp = _tracker.init_db()
      old_run = _tracker.get_run(db_conn_tmp, args.resume)
      if not old_run:
          print(f"Error: run {args.resume} not found in tracker.db")
          sys.exit(1)
      latest_ckpt = _tracker.get_latest_checkpoint(db_conn_tmp, args.resume)
      if not latest_ckpt:
          print(f"Error: no checkpoint found for run {args.resume}")
          sys.exit(1)
      resumed_from = args.resume
      resume_model_path = latest_ckpt["model_path"]
      initial_cycle = latest_ckpt["cycle"]
      initial_ckpt_wr = latest_ckpt["win_rate"]
      db_conn_tmp.close()
  ```
- 模型初始化: 若 resumed_from, 用 `load_model(resume_model_path)` 替代 `GomokuNet()`
- 设置 `cycle = initial_cycle`, `last_ckpt_wr = initial_ckpt_wr`
- `create_run()` 传入 resumed_from
- `_tracker.create_run()` 需要接受 resumed_from 参数 → 写入 DB

**依赖**: schema-migration, output-uuid-dirs

### Task 5: fix-play-imports
**文件**: `src/play.py`

变更：
- 删除 `from prepare import OPPONENTS, CHECKPOINT_DIR, list_checkpoints`
- 改为:
  ```python
  from prepare import OPPONENTS
  import tracker
  ```
- `print_checkpoints()` 重写: 调用 `tracker.init_db()` + `tracker.list_all_checkpoints()`
- `resolve_checkpoint()` 重写:
  - 先检查是否为文件路径
  - 再查 DB by tag
  - 再查 DB by run_id (部分匹配)
  - 最后尝试 "latest" → 最近一个 checkpoint
- `CHECKPOINT_DIR` 的使用全部去掉（从 DB 获取 model_path）

**依赖**: schema-migration

### Task 6: verify-and-test
- 运行 60s 训练 → 确认 output/<uuid>/ 目录结构
- 查询 DB 确认 output_dir 字段
- 用第一次的 UUID 执行 --resume → 确认续训
- 运行 play.py --list → 确认 checkpoint 列表
- 检查旧的 output/checkpoints/ 和 output/recordings/ 不再写入

**依赖**: 所有实现任务

### Task 7: docs-update
- README.md: 更新目录结构树、--resume 文档、play.py 说明
- 清理 output/ 下的旧目录（如果存在）

**依赖**: verify-and-test

---

## 5. 依赖图

```
schema-migration ──┬──→ output-uuid-dirs ──→ resume-checkpoint ──┐
                   │                                              ├→ verify → docs
                   └──→ fix-play-imports ────────────────────────┘
                                                                  │
prepare-recording-dir ────────────────────────────────────────────┘
```

---

## 6. 执行结果

### 6.1 实施概要

所有 7 个任务全部完成并验证通过。

| 任务 | 状态 | 改动文件 | 关键变更 |
|------|------|----------|----------|
| schema-migration | ✅ Done | tracker.py | ALTER TABLE 添加 `resumed_from`, `output_dir`; 新增 `get_latest_checkpoint()`, `list_all_checkpoints()`, `find_checkpoint_by_tag()` |
| output-uuid-dirs | ✅ Done | train.py | 移除 `MODEL_PATH` 常量，所有路径改为 `output/{run_id}/` 动态生成 |
| prepare-recording-dir | ✅ Done | prepare.py | `evaluate_win_rate()` 新增 `recording_dir` 参数 |
| resume-checkpoint | ✅ Done | train.py | `--resume <uuid>` 从上次最新 checkpoint 恢复 cycle/wr |
| fix-play-imports | ✅ Done | play.py | 用 `import tracker` 替代已删除的 prepare.py 导出 |
| verify-and-test | ✅ Done | — | 三项测试全部通过 |
| docs-update | ✅ Done | README.md | 更新目录结构树、resume 文档、查询示例 |

### 6.2 验证运行结果

**测试 1: 新 run（60s 训练）**
```
Run:     c117fa23 (fresh)
Cycles:  15
Output:  output/c117fa23-12f1-40f6-ba95-e70d187b6fcf/
├── model.safetensors
├── checkpoints/wr056_c0005.safetensors
├── checkpoints/final_c0015.safetensors
└── recordings/games/ (100 files)
DB:      output_dir = "output/c117fa23-...", resumed_from = NULL ✓
```

**测试 2: --resume（30s 续训）**
```
Resume from: c117fa23 → New run: 9f22b5f8
Starting cycle: 15 (inherited from parent)
Ending cycle: 23
Output:  output/9f22b5f8-5c74-41df-84a7-ba95c6a73eef/
DB:      resumed_from = "c117fa23-..." ✓
Checkpoints: wr056_c0020, final_c0023 ✓
```

**测试 3: play.py --list**
```
#  Tag                    Run       WR    Cycle
0  final_c0023            9f22b5f8  50.0%   23
1  wr056_c0020            9f22b5f8  53.3%   20
2  final_c0015            c117fa23  52.0%   15
3  wr056_c0005            c117fa23  54.0%    5
4  05bb6792_final_c0030   05bb6792  48.0%   30  ← v2 遗留
5  05bb6792_wr060_c0005   05bb6792  46.0%    5  ← v2 遗留
```

### 6.3 遇到的问题

1. **v2 遗留的孤立目录**: `output/checkpoints/`, `output/recordings/`, `output/model.safetensors` 在 v3 迁移后成为孤儿文件。已手动清理。DB 中对应的 v2 记录（run `05bb6792`）保留但路径已失效。
2. **无其他问题**: schema migration 的 `ALTER TABLE ... ADD COLUMN` 幂等设计运行顺畅，旧数据自动获得 NULL 默认值。
