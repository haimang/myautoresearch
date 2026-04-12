"""
autoresearch 评估模板 — 对手体系与基准测试。

这是评估模块的框架根副本，定义了 autoresearch 评估系统的标准结构：

    对手（分级难度） → 评估框架 → 指标与归档

创建新领域：
    1. 复制本文件到 domains/<name>/prepare.py
    2. 替换所有 <<<DOMAIN>>> 标记为领域特定实现
    3. 实现你的对手层次（如 随机 → 启发式 → 搜索）
    4. 适配 evaluate_win_rate() 到你的领域的对局循环

== autoresearch 契约 ==
本模块是只读评估代码 — 不修改模型权重。
评估框架必须保持不变以确保实验结果可比较。只有 train.py 是可变的。
"""

import csv
import json
import os
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# <<<DOMAIN>>> — 替换为你的领域游戏引擎导入:
#   from game import BOARD_SIZE, WIN_LENGTH, BLACK, WHITE, EMPTY, Board, GameRecord
#
# 领域必须提供的概念:
#   BOARD_SIZE   — int, 棋盘维度
#   Board        — 单盘对局用于顺序博弈
#     .grid              — 当前棋盘状态的 numpy 数组
#     .current_player    — 当前执棋方
#     .is_terminal()     -> bool
#     .get_legal_moves() -> (row, col) 列表
#     .get_legal_mask()  -> [action_space] 二值数组
#     .place(row, col)   — 执行落子
#     .winner            — int 结果
#     .move_count        — 已下步数
#     .history           — (row, col, player) 元组列表
#     .encode()          -> [C, H, W] 用于神经网络输入的 numpy 数组
#   GameRecord   — 可序列化的对局记录
#     .add_move(step, row, col, player)
#     .save(path)

BOARD_SIZE = 15     # <<<DOMAIN>>>
WIN_LENGTH = 5      # <<<DOMAIN>>>
BLACK = 1           # <<<DOMAIN>>>
WHITE = 2           # <<<DOMAIN>>>
EMPTY = 0           # <<<DOMAIN>>>

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

TIME_BUDGET = 300   # 5 分钟训练墙钟时间
EVAL_GAMES = 200    # 每次评估的对局数
CHECKPOINT_DIR = os.path.expanduser("~/.cache/mag-gomoku/checkpoints")  # <<<DOMAIN>>>
RECORDING_DIR = "recordings"


# ---------------------------------------------------------------------------
# 棋盘评估启发函数 — <<<DOMAIN>>> 整体替换
# ---------------------------------------------------------------------------

# 模式得分: (连子数, 开放端数) -> 分数
# <<<DOMAIN>>> 此启发函数适用于连子类游戏（五子棋、四子棋等）
# 替换为你的领域的局面评估函数。
_PATTERN_SCORES = {
    (5, 0): 100000, (5, 1): 100000, (5, 2): 100000,
    (4, 2): 10000,  (4, 1): 1000,
    (3, 2): 1000,   (3, 1): 100,
    (2, 2): 100,    (2, 1): 10,
    (1, 2): 10,     (1, 1): 1,
}


def evaluate_position(grid: np.ndarray, player: int) -> int:
    """
    从 `player` 角度对棋盘进行静态评估。

    <<<DOMAIN>>> 替换为你的领域启发式评估。
    必须返回数值分数：正数 = 对 player 有利，负数 = 不利。
    """
    # 占位符 — 实现领域特定的评估
    return 0


# ---------------------------------------------------------------------------
# 带 alpha-beta 剪枝的 Minimax 搜索
# ---------------------------------------------------------------------------

def _minimax(grid: np.ndarray, depth: int, alpha: float, beta: float,
             maximizing: bool, player: int, opponent: int,
             candidate_fn, move_order_fn=None):
    """
    Alpha-beta minimax — 领域无关的树搜索。

    游戏特定部分通过以下参数注入:
    - candidate_fn(grid) -> 候选落子 (row, col) 列表
    - move_order_fn(moves, grid) -> 排序后的落子（可选，用于剪枝）
    - evaluate_position(grid, player) -> 数值分数（叶节点评估）
    """
    if depth == 0:
        return float(evaluate_position(grid, player)), None

    moves = candidate_fn(grid)
    if not moves:
        return float(evaluate_position(grid, player)), None

    if move_order_fn is not None:
        moves = move_order_fn(moves, grid)

    best_move = moves[0]

    if maximizing:
        max_eval = -float("inf")
        for (r, c) in moves:
            grid[r, c] = player
            # <<<DOMAIN>>> 检查是否立即终局:
            #   if _check_win(grid, r, c, player): ...
            val, _ = _minimax(grid, depth - 1, alpha, beta,
                              False, player, opponent, candidate_fn, move_order_fn)
            grid[r, c] = EMPTY
            if val > max_eval:
                max_eval = val
                best_move = (r, c)
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float("inf")
        for (r, c) in moves:
            grid[r, c] = opponent
            val, _ = _minimax(grid, depth - 1, alpha, beta,
                              True, player, opponent, candidate_fn, move_order_fn)
            grid[r, c] = EMPTY
            if val < min_eval:
                min_eval = val
                best_move = (r, c)
            beta = min(beta, val)
            if beta <= alpha:
                break
        return min_eval, best_move


def _make_candidate_fn(radius: int):
    """
    返回一个候选落子生成器，搜索现有棋子附近的位置。
    <<<DOMAIN>>> 适配为你的领域的落子生成策略。
    """
    def fn(grid):
        occupied = np.argwhere(grid != EMPTY)
        if len(occupied) == 0:
            return [(BOARD_SIZE // 2, BOARD_SIZE // 2)]
        candidates = set()
        for r, c in occupied:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE
                            and grid[nr, nc] == EMPTY):
                        candidates.add((int(nr), int(nc)))
        return list(candidates)
    return fn


def _move_order_basic(moves, grid):
    """基础落子排序: 优先中心。"""
    center = BOARD_SIZE // 2
    return sorted(moves, key=lambda m: abs(m[0] - center) + abs(m[1] - center))


def _move_order_heuristic(moves, grid):
    """启发式落子排序: 按邻接度和中心度评分。"""
    center = BOARD_SIZE // 2
    scored = []
    for (r, c) in moves:
        adj = 0
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE
                        and grid[nr, nc] != EMPTY):
                    adj += 1
        center_bonus = max(0, 7 - abs(r - center) - abs(c - center))
        scored.append((-adj * 10 - center_bonus, r, c))
    scored.sort()
    return [(r, c) for (_, r, c) in scored]


def minimax_move(board, depth: int, player: int, move_order_fn=None):
    """
    使用 alpha-beta 剪枝的 minimax 搜索找到最佳落子。

    <<<DOMAIN>>> `board` 必须拥有 .grid 属性（numpy 数组）。
    """
    grid = board.grid.copy()
    opponent = WHITE if player == BLACK else BLACK
    candidate_fn = _make_candidate_fn(radius=2)

    _, best_move = _minimax(
        grid, depth, -float("inf"), float("inf"),
        True, player, opponent, candidate_fn, move_order_fn,
    )

    if best_move is None:
        legal = board.get_legal_moves()
        if legal:
            return legal[0]
        return (BOARD_SIZE // 2, BOARD_SIZE // 2)
    return best_move


# ---------------------------------------------------------------------------
# 对手层次 — <<<DOMAIN>>> 定义你的难度等级
# ---------------------------------------------------------------------------

def opponent_l0(board) -> tuple[int, int]:
    """等级0: 随机合法落子。"""
    legal = board.get_legal_moves()
    return random.choice(legal)


def opponent_l1(board) -> tuple[int, int]:
    """等级1: minimax 深度2。"""
    return minimax_move(board, depth=2, player=board.current_player,
                        move_order_fn=_move_order_basic)


def opponent_l2(board) -> tuple[int, int]:
    """等级2: minimax 深度4 + 启发式排序。"""
    return minimax_move(board, depth=4, player=board.current_player,
                        move_order_fn=_move_order_heuristic)


def opponent_l3(board) -> tuple[int, int]:
    """等级3: minimax 深度6 + 启发式排序。"""
    return minimax_move(board, depth=6, player=board.current_player,
                        move_order_fn=_move_order_heuristic)


OPPONENTS = {0: opponent_l0, 1: opponent_l1, 2: opponent_l2, 3: opponent_l3}


# ---------------------------------------------------------------------------
# 评估框架: 神经网络 vs 对手
# ---------------------------------------------------------------------------

def evaluate_win_rate(
    model_path: str,
    level: int = 2,
    n_games: int = EVAL_GAMES,
    record_games: int = 5,
    experiment_id: int = 0,
) -> dict:
    """
    让训练好的神经网络模型与对手进行 n_games 局对弈。
    神经网络前半局执黑，后半局执白。

    <<<DOMAIN>>> 适配对局循环和导入到你的领域。

    返回字典: win_rate, wins, losses, draws, avg_game_length, level。
    """
    import mlx.core as mx
    # <<<DOMAIN>>> 导入你的领域的模型和棋盘:
    #   from train import PolicyValueNet, load_model
    #   from game import Board, GameRecord, BLACK, WHITE, BOARD_SIZE

    # model = load_model(model_path)
    # opponent_fn = OPPONENTS[level]
    # ... 对局循环 ...

    raise NotImplementedError(
        "evaluate_win_rate() 必须为你的领域实现。"
        "参考 domains/gomoku/prepare.py 的示例实现。"
    )


# ---------------------------------------------------------------------------
# 检查点归档
# ---------------------------------------------------------------------------

def archive_checkpoint(model_path: str, tag: str, metadata: dict):
    """将模型检查点复制到归档目录，附带可读标签。"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    dest = os.path.join(CHECKPOINT_DIR, f"{tag}.safetensors")
    shutil.copy2(model_path, dest)

    manifest_path = os.path.join(CHECKPOINT_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = []

    manifest.append({
        "tag": tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_path": os.path.abspath(model_path),
        "archived_path": os.path.abspath(dest),
        **metadata,
    })
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def list_checkpoints() -> list[dict]:
    """读取并返回检查点清单中的所有条目。"""
    manifest_path = os.path.join(CHECKPOINT_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 指标记录
# ---------------------------------------------------------------------------

_METRIC_COLUMNS = [
    "experiment", "timestamp", "win_rate", "eval_level",
    "num_params", "status", "description",
]


def log_metrics(experiment_id: int, metrics: dict):
    """向训练日志 CSV 追加一行指标记录。"""
    metrics_dir = os.path.join(RECORDING_DIR, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    csv_path = os.path.join(metrics_dir, "training_log.csv")

    row = {"experiment": experiment_id,
           "timestamp": datetime.now(timezone.utc).isoformat()}
    for col in _METRIC_COLUMNS[2:]:
        row[col] = metrics.get(col, "")
    for k, v in metrics.items():
        if k not in row:
            row[k] = v

    all_columns = list(row.keys())
    file_exists = os.path.exists(csv_path)
    if file_exists:
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
        if existing_header:
            new_cols = [c for c in all_columns if c not in existing_header]
            all_columns = existing_header + new_cols
        else:
            file_exists = False

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# 入口: 烟雾测试 — <<<DOMAIN>>> 替换为你的领域的测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("这是框架模板。复制到 domains/<name>/prepare.py")
    print("并填充领域特定实现后再运行。")
    print()
    print("需要实现的接口:")
    print("  - evaluate_position(grid, player) -> int")
    print("  - evaluate_win_rate(model_path, level, ...) -> dict")
    print("  - 对手函数 (opponent_l0 .. opponent_l3)")
