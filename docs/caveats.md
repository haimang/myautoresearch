# MAG-Gomoku 踩坑记录 & 注意事项

> 汇总自 v1/v2/v3 三轮迭代中遇到的所有已知陷阱、注意事项和解决方案。

---

## 1. MLX / Metal GPU

### 1.1 BatchNorm 导致 Metal Buffer 耗尽 ⚠️ 关键

**现象**: `evaluate_win_rate()` 跑到约 100 局后崩溃：
```
RuntimeError: [metal::malloc] Resource limit (499000) exceeded.
```

**根因**: MLX 的 BatchNorm 在 train 模式下，每次 forward 都会执行 `running_var = (1-mu)*running_var + mu*var`。由于 MLX 数组不可变语义，每次更新都分配一个 **新的 Metal buffer**。200 局评估 × ~50 步/局 = ~10,000 次 forward → 超出 macOS Metal 的 499,000 buffer 上限。

**解决方案**:
1. **必须**: 加载模型后立即调用 `model.eval()` 关闭 running stats 更新
2. **防御性清理**: 每 20 局调用 `mx.clear_cache()`
3. **进程隔离**: 训练后评估在子进程中运行，隔离 Metal buffer 累积

### 1.2 `mx.metal.clear_cache()` 已弃用

MLX ≥ 0.31.1 中 `mx.metal.clear_cache()` 已移除，需改用 `mx.clear_cache()`。

### 1.3 GPU 核心数无法直接检测

macOS `system_profiler` 的 plist 数据中不包含 `platform_number_gpu_cores` 字段。

**解决方案**: 从芯片名推断 GPU 核心数（如 M3 Max → 40 cores）。

---

## 2. Python 环境

### 2.1 stdout 重定向时日志不实时写入

**现象**: `uv run python src/train.py > run.log 2>&1` 执行期间日志文件为空，训练结束后才一次性写出。

**根因**: Python 检测到 stdout 重定向到文件后，自动启用 block buffering。

**解决方案**:
```bash
PYTHONUNBUFFERED=1 uv run python src/train.py > output/run.log 2>&1
```

### 2.2 pyproject.toml build-backend 配置错误

**现象**: `uv sync` 报 `No module named 'setuptools.backends'`

**根因**: build-backend 误写为不存在的路径 `setuptools.backends._legacy:_Backend`

**解决方案**:
```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
```

### 2.3 Rich TUI 在非 TTY 环境下失败

**现象**: Rich 面板在 stdout 被 pipe 或子进程中时无法渲染。

**解决方案**: 检测 TTY，非终端环境回退到 `print()` 输出。

### 2.4 macOS 26 的 CPU 核心数格式变更

**现象**: `system_profiler` 返回 `"proc 16:12:4"` 而非纯数字。

**解决方案**: 按 `:` 分割取第一段作为总核心数。

---

## 3. 训练逻辑

### 3.1 self-play 无限循环

**现象**: `run_self_play()` 进入死循环不返回。

**根因**: 游戏完成后重置并继续，终止条件未正确判断所有游戏已完成一轮。

**解决方案**: 重构终止逻辑，所有游戏完成一轮后立即返回。

### 3.2 `--resume` 不恢复 replay buffer 和 optimizer 状态

**现象**: 断点续训后，前几个 cycle 的 loss 可能偏高。

**根因**: replay buffer 是内存状态、optimizer state 刻意不保存（re-warmup 是可接受的）。只恢复模型权重、cycle 计数和 win rate。

**注意**: 这是设计取舍，不是 bug。续训的 run 会在数个 cycle 后恢复正常。

### 3.3 `--resume` 创建新 UUID，不复用旧 UUID

每次 `--resume <uuid>` 都会创建 **全新的 run**，通过 DB 中的 `resumed_from` 字段链接到父 run。不要期望续训数据写入旧目录。

---

## 4. 数据库 & 存储

### 4.1 Schema 迁移必须幂等

**现象**: 重复运行 `ALTER TABLE ... ADD COLUMN` 报 "column already exists"。

**解决方案**: 所有迁移语句用 `try/except` 包裹，忽略 "duplicate column" 错误。

### 4.2 旧 run 的 `output_dir` 和 `resumed_from` 为 NULL

v3 新增的两个字段对 v2 历史记录默认为 NULL。查询时需处理 NULL 情况。

### 4.3 tracker.db 是全局单文件

`output/tracker.db` 跨所有 run 共享。**不支持并发训练**。如需并发，需启用 SQLite WAL 模式或加锁。

### 4.4 v2 遗留 checkpoint 路径失效

v3 迁移后 v2 的 `output/checkpoints/` 和 `output/recordings/` 如已清理，DB 中旧记录的 `model_path` 指向不存在的文件。旧记录保留供参考，但实际文件不可用。

### 4.5 Checkpoint model_path 使用相对路径

DB 中 `checkpoints.model_path` 存储的是相对于项目根目录的路径（如 `output/<uuid>/checkpoints/xxx.safetensors`）。**必须从项目根目录运行**，否则路径解析失败。

---

## 5. 目录结构

### 5.1 output/ 下的 UUID 目录必须动态创建

训练启动时 UUID 目录不存在，代码中必须在写入前调用:
```python
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(recording_dir, exist_ok=True)
```

### 5.2 Tag 格式 v2 → v3 不兼容

| 版本 | 格式 | 示例 |
|------|------|------|
| v2 | `{run_short}_{metric}_{cycle}` | `05bb6792_wr060_c0005` |
| v3 | `{metric}_{cycle}` | `wr060_c0005` |

两种格式在 DB 中共存。`play.py --list` 和查询工具已兼容两种格式。

---

## 6. Git & 版本管理

### 6.1 `output/` 目录已 gitignore

所有训练产出（模型、checkpoint、录像、DB）都在 `output/` 下，不进入版本控制。备份需手动操作。

### 6.2 update-v*.md 不跟踪

`update-v1.md`、`update-v2.md`、`update-v3.md` 均在 `.gitignore` 中。这些是本地开发日志。

### 6.3 Autoresearch 约束: 只能修改 train.py

`docs/program.md` 约定 AI agent 只能修改 `src/train.py`。`game.py` 和 `prepare.py` 对 agent 是只读的。不要将 train.py 的逻辑拆分到其他模块。
