# Update Plan v5 — NN 对手注册 · TUI 双行图表 · 对齐修复

> 2026-04-10 | 自定义对手体系 + 终端显示优化

---

## 1. 背景讨论总结

### 1.1 对手等级与自动升级

**原版 autoresearch 的设计：**
- 原版是语言模型预训练任务，度量指标是 `val_bpb`，没有"对手"概念
- 单次运行、单一指标、固定时间预算，极简设计
- L0-L3 对手升级体系完全是我们 mag-gomoku 自行设计的

**讨论结论 —— 不做训练内自动升级：**
- 保持原版精神：单次运行、单一指标、单一对手
- WR 可比性：一次 run 内所有 WR 数据点对同一对手，曲线有意义
- 阶段切换应该是跨 run 的：跑完后手动 `--eval-level 1 --resume <uuid>` 开新 run

### 1.2 NN Checkpoint 作为自定义对手（本次重点）

**动机：** minimax L0-L3 强度跳跃大，缺乏中间梯度。用训练产生的 checkpoint 注册为对手，可以：
1. 基线可追溯 —— DB 记录对手来源，WR 有清晰参考系
2. 难度连续可调 —— 不像 minimax 跳跃式
3. 自我进化链 —— 第一代模型当对手 → 训练更强的 → 再提取 → 形成链条

**实现路线选择 A（保持 prepare.py 只读）：**
- 在 train.py 中实现 `_eval_vs_nn()` 函数
- 不修改 prepare.py 的 `evaluate_win_rate()`
- minimax 评估基线不受污染

### 1.3 WR 计算策略

**结论 —— 维持现状：**
- WR 通过 probe 评估获得（每 N cycle 打 50 局）
- 自对弈是模型打自己，WR 恒为 ~50%，不可用
- DB 中 WR 采样率 = probe 频率，不会更高
- 不做改变

### 1.4 终端图表改进

**要做的改进：**
1. sparkline 从单行改为双行高度
2. WR 和 Loss 分成两行显示，中间有隔断
3. 新增统计指标：games/sec、avg game length
4. 修复所有对齐错位问题（数值占位、变长字符串）

**不做的：**
- 不用 Rich TUI（保持纯文本）
- 不改变 WR 计算方式

---

## 2. 改动方案

### Task 1: NN 对手注册体系

**2.1.1 存储结构**
```
output/
  opponents/                    # 注册的 NN 对手
    alpha/model.safetensors     # alias = "alpha"
    bravo/model.safetensors
  tracker.db                    # 新增 opponents 表
```

**2.1.2 DB schema — opponents 表**
```sql
CREATE TABLE IF NOT EXISTS opponents (
    alias        TEXT PRIMARY KEY,
    source_run   TEXT,
    source_tag   TEXT,
    model_path   TEXT NOT NULL,
    win_rate     REAL,
    eval_level   INTEGER,
    description  TEXT,
    created_at   TEXT NOT NULL
);
```

**2.1.3 CLI 接口**

注册对手（从已有 checkpoint）：
```bash
# 基本用法
uv run python src/train.py --register-opponent alpha --from-run b2e5c8ae --from-tag wr075_c0030

# 附加描述
uv run python src/train.py --register-opponent alpha --from-run b2e5c8ae --from-tag wr075_c0030 --description "First 75% model"
```

对打 NN 对手训练：
```bash
uv run python src/train.py --eval-opponent alpha --target-win-rate 0.80
```

列出对手：
```bash
uv run python src/play.py --list-opponents
```

**2.1.4 评估逻辑**
- `--eval-level` 和 `--eval-opponent` 互斥
- 新增 `_nn_opponent_move(model, board)` 函数：加载对手模型，argmax 策略出招
- `_quick_eval()` 支持传入 opponent_model 代替 opponent_fn
- `_subprocess_eval()` / `evaluate_win_rate()` 不改（prepare.py 只读），checkpoint 完整评估也在 train.py 内实现
- DB 中 cycle_metrics / checkpoints 增加 `eval_opponent TEXT` 字段

**2.1.5 run 结束时升级建议**

当 WR 达到高阈值时自动提示：
```
🎯 Win rate 96.5% vs L0 — consider promoting:
   uv run python src/train.py --eval-level 1 --resume b2e5c8ae
   
💡 Or register this model as opponent:
   uv run python src/train.py --register-opponent <alias> --from-run b2e5c8ae --from-tag wr095_c0080
```

### Task 2: TUI 双行图表

**2.2.1 双行 sparkline 实现**

用上下两行字符叠加。上行显示高于中位的部分，下行显示低于中位的部分：
```
上行: " " 或 "▄▅▆▇█"（值 > 中位时）
下行: "▁▂▃▄" 或 "█"（值 ≤ 中位时的高度）
```

效果：
```
│  Win Rate                                           73%  │
│            ░░▁▂▃▃▅▅▆▇▇██████████████████                │
│            ██████████████████████▇▆▅▅▃▂▁                │
├──────────────────────────────────────────────────────────┤
│  Loss                                              1.17  │
│            █████▇▇▆▅▄▃▃▂▂▁▁░░░░░░░░░░░░                │
│            ▃▃▂▂▁▁░░░░░░░░░░░░░░░░░░░░░░                │
╰──────────────────────────────────────────────────────────╯
```

### Task 3: TUI 对齐修复

**当前问题分析：**

r2 行（Steps/Buffer/WR）与 r1 行（Cycle/Loss/Games）的 `│` 分隔符未对齐：
- r1: `  Cycle {cycle:5d}  │  Loss {loss:8.4f}  │  Games {games:7d}`  → 固定 49 字符 ✓
- r2: `  Steps {steps:5d}  │  Buffer {buf:5d}  │  WR {wr}{sm} (L{level})`  → **变长** ✗
  - Buffer 只有 5 位，r1 的 Loss 有 8 位 → 列宽不匹配
  - WR 字符串长度剧烈变化：`—` vs `73.0% avg:72.0%`
  - 加上 `(L0)` 后可能超出面板宽度

**修复方案：** 两行使用统一的列宽模板

```
│  Cycle    85  │  Loss   1.2470  │  Games     5440  │
│  Steps  4226  │  Buffer  50000  │  WR 73.0% (L0)   │
```

规则：
- 列 1 (label+val): 14 字符 (`  Cycle    85` / `  Steps  4226`)
- 分隔符: `  │  ` (5 字符)
- 列 2 (label+val): 14 字符 (`Loss   1.2470` / `Buffer  50000`)
- 分隔符: `  │  ` (5 字符)
- 列 3 (label+val): 剩余宽度，右填充

WR 固定格式：
- 无 WR: `WR   —         `
- 有 WR: `WR 73.0%  L0   `
- 有平滑: `WR 73.0% ⌀72%  ` （⌀ 表示 avg，或直接用 avg）

### Task 4: 新增统计指标

在 stats 区域增加第三行或重新安排：

```
│  Cycle    85  │  Loss   1.2470  │  Games     5440  │
│  Steps  4226  │  Buffer  50000  │  WR 73.0%  (L0)  │
│  Gm/s    3.2  │  AvgLen    107  │  Ckpts        3  │
```

新指标来源：
- **Gm/s** (games/sec): `total_games / elapsed` — 吞吐量
- **AvgLen**: 自对弈平均棋局长度（从 self-play 返回数据计算）— 模型效率
- **Ckpts**: 已保存的 checkpoint 数量

---

## 3. 不做的事项

| 项目 | 原因 |
|------|------|
| 训练内自动对手升级 | 破坏 WR 曲线连续性，不符合 autoresearch 单一指标精神 |
| WR 从 DB 计算 | DB 中 WR 采样率 = probe 频率，不会更高；自对弈 WR 无意义 |
| 修改 prepare.py | 保持只读契约，NN 对手评估在 train.py 内实现 |
| Rich TUI | 已在 v4 移除，继续使用纯文本 |

---

## 4. 执行计划

| # | 任务 | 依赖 | 文件 |
|---|------|------|------|
| 1 | opponents 表 + 注册 CLI | 无 | tracker.py, train.py |
| 2 | `_nn_opponent_move()` + `_eval_vs_nn()` | T1 | train.py |
| 3 | `--eval-opponent` 集成到训练循环 | T2 | train.py |
| 4 | run 结束升级建议 | T3 | train.py |
| 5 | 双行 sparkline `_sparkline2()` | 无 | train.py |
| 6 | TUI 对齐修复（统一列宽模板） | 无 | train.py |
| 7 | 新增 Gm/s、AvgLen、Ckpts 指标 | T6 | train.py |
| 8 | play.py `--list-opponents` | T1 | play.py |
| 9 | 测试 + 文档更新 | T1-T8 | README.md, docs/ |

---

## 5. 执行结果（待回填）

### 5.1 实施概要
（待填）

### 5.2 验证运行结果
（待填）

### 5.3 遇到的问题
（待填）
