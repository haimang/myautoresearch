"""
autoresearch 训练模板 — MLX on Apple Silicon。

这是训练脚本的框架根副本，定义了 autoresearch 训练循环的标准结构：

    神经网络 → 自对弈 → 经验池 → 梯度更新 → 评估

创建新领域：
    1. 复制本文件到 domains/<name>/train.py
    2. 替换所有 <<<DOMAIN>>> 标记为领域特定实现
    3. 导入你的领域游戏引擎代替占位导入
    4. 调整神经网络输入/输出维度以适配你的领域

框架基础设施（tracker, tui, analyze, sweep）与任何遵循此接口的领域兼容。

== autoresearch 契约 ==
本文件是可变文件。agent 在实验间修改超参、架构、自对弈策略和训练调度。
只要评估接口保持兼容，所有内容均可修改。
"""

import argparse
import random
import time as _time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# <<<DOMAIN>>> — 替换为你的领域游戏引擎导入:
#   from game import BatchBoards, BOARD_SIZE, Board, BLACK, WHITE, EMPTY
#
# 领域必须提供的概念:
#   BOARD_SIZE   — int, 棋盘边长
#   BLACK, WHITE, EMPTY — int, 格子状态常量
#   BatchBoards  — 并行对局管理器
#     .encode_all()       -> [N, C, H, W] numpy 数组
#     .get_legal_masks()  -> [N, action_space] 二值 numpy 数组
#     .get_finished_mask()-> [N] bool numpy 数组
#     .step(actions)      — 每盘执行一步
#     .winners            — [N] 数组 (0=进行中, player=胜者, -1=和棋)
#     .histories          — 每盘的落子历史 (encoded, action, player)
#   Board        — 单盘游戏用于评估
#     .encode()           -> [C, H, W] numpy 数组
#     .get_legal_mask()   -> [action_space] 二值数组
#     .is_terminal()      -> bool
#     .place(row, col)    — 执行落子
#     .winner             — int 结果
#     .current_player     — 当前执棋方

BOARD_SIZE = 15          # <<<DOMAIN>>> 棋盘边长
ACTION_SPACE = BOARD_SIZE * BOARD_SIZE
INPUT_CHANNELS = 3       # <<<DOMAIN>>> 编码通道数

# <<<DOMAIN>>> — 替换为你的领域评估导入:
#   from prepare import evaluate_win_rate, TIME_BUDGET
# NOTE: framework 评估模板已重命名为 prepare_template.py（v15.4），
# 避免与 domains/<name>/prepare.py 同名冲突。
TIME_BUDGET = 300  # 5 分钟默认


# === 超参（autoresearch agent 在实验间修改） ===
NUM_RES_BLOCKS = 6
NUM_FILTERS = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
PARALLEL_GAMES = 64
TEMPERATURE = 1.0
TEMP_THRESHOLD = 15      # 切换到利用温度前的步数
REPLAY_BUFFER_SIZE = 50000
TRAIN_STEPS_PER_CYCLE = 50
CYCLES_PER_REPORT = 5
POLICY_LOSS_WEIGHT = 1.0
VALUE_LOSS_WEIGHT = 1.0
EVAL_LEVEL = 0           # 评估对手等级

MODEL_PATH = "model.safetensors"


# ---------------------------------------------------------------------------
# 神经网络 — 带策略/价值头的 ResNet
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """标准残差块: conv → BN → ReLU → conv → BN → skip。"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(channels)

    def __call__(self, x):
        residual = x
        x = nn.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = nn.relu(x + residual)
        return x


class PolicyValueNet(nn.Module):
    """
    领域无关的策略-价值网络。

    输入:  [B, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE]
    输出: (policy_logits [B, ACTION_SPACE], value [B, 1])

    <<<DOMAIN>>> 调整输入通道和输出维度以适配你的领域。
    """
    def __init__(self, num_blocks=NUM_RES_BLOCKS, num_filters=NUM_FILTERS):
        super().__init__()
        self.input_conv = nn.Conv2d(INPUT_CHANNELS, num_filters, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm(num_filters)
        self.res_blocks = [ResBlock(num_filters) for _ in range(num_blocks)]

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm(2)
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, ACTION_SPACE)

        # Value head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm(1)
        self.value_fc1 = nn.Linear(1 * BOARD_SIZE * BOARD_SIZE, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def __call__(self, x):
        # x: [B, C, H, W] (NCHW) — MLX Conv2d 需要 NHWC 格式
        x = mx.transpose(x, (0, 2, 3, 1))

        x = nn.relu(self.input_bn(self.input_conv(x)))
        for block in self.res_blocks:
            x = block(x)

        # 策略头
        p = nn.relu(self.policy_bn(self.policy_conv(x)))
        p = mx.reshape(p, (p.shape[0], -1))
        p = self.policy_fc(p)

        # 价值头
        v = nn.relu(self.value_bn(self.value_conv(x)))
        v = mx.reshape(v, (v.shape[0], -1))
        v = nn.relu(self.value_fc1(v))
        v = mx.tanh(self.value_fc2(v))

        return p, v


# ---------------------------------------------------------------------------
# 模型保存/加载
# ---------------------------------------------------------------------------

def save_model(model, path: str):
    model.save_weights(path)


def load_model(path: str, num_blocks=NUM_RES_BLOCKS, num_filters=NUM_FILTERS):
    model = PolicyValueNet(num_blocks=num_blocks, num_filters=num_filters)
    model.load_weights(path)
    return model


def count_parameters(model) -> int:
    leaves = nn.utils.tree_flatten(model.parameters())
    return sum(v.size for _, v in leaves)


# ---------------------------------------------------------------------------
# 自对弈 — 批量并行对局
# ---------------------------------------------------------------------------

def run_self_play(model, batch_boards_cls, num_games=PARALLEL_GAMES, temperature=TEMPERATURE):
    """
    批量自对弈。

    <<<DOMAIN>>> batch_boards_cls 应为你的领域的 BatchBoards 类。
    返回 (training_data, games_completed)。
    """
    model.eval()
    batch = batch_boards_cls(num_games)
    move_policies = [[] for _ in range(num_games)]

    while not batch.get_finished_mask().all():
        encoded = batch.encode_all()
        legal_masks = batch.get_legal_masks()

        encoded_mx = mx.array(encoded)
        logits, values = model(encoded_mx)
        mx.eval(logits, values)
        logits_np = np.array(logits)

        logits_np[legal_masks == 0] = -1e9
        actions = np.zeros(num_games, dtype=np.int32)

        for i in range(num_games):
            if batch.winners[i] != 0:
                actions[i] = 0
                continue

            game_logits = logits_np[i]
            move_num = batch.move_counts[i]
            temp = temperature if move_num < TEMP_THRESHOLD else 0.1

            if temp < 0.01:
                action = int(np.argmax(game_logits))
                policy_dist = np.zeros(ACTION_SPACE, dtype=np.float32)
                policy_dist[action] = 1.0
            else:
                scaled = game_logits / temp
                scaled -= np.max(scaled)
                probs = np.exp(scaled)
                probs_sum = probs.sum()
                if probs_sum > 0:
                    probs /= probs_sum
                else:
                    probs = legal_masks[i].copy().astype(np.float32)
                    s = probs.sum()
                    if s > 0:
                        probs /= s
                    else:
                        actions[i] = 0
                        continue
                action = np.random.choice(ACTION_SPACE, p=probs)
                policy_dist = probs

            actions[i] = action
            move_policies[i].append(policy_dist.copy())

        batch.step(actions)

    all_training_data = []
    for i in range(num_games):
        game_data = _collect_game_data(batch, i, move_policies[i])
        all_training_data.extend(game_data)

    return all_training_data, num_games


def _collect_game_data(batch, game_idx, policies):
    """将每个局面与其策略目标和结果价值配对。"""
    winner = batch.winners[game_idx]
    history = batch.histories[game_idx]
    data = []
    for move_idx, (encoded, action, player) in enumerate(history):
        if winner == -1:
            value_target = 0.0
        elif winner == player:
            value_target = 1.0
        else:
            value_target = -1.0

        if move_idx < len(policies):
            policy_target = policies[move_idx]
        else:
            policy_target = np.zeros(ACTION_SPACE, dtype=np.float32)
            policy_target[action] = 1.0

        data.append((encoded, policy_target, value_target))
    return data


# ---------------------------------------------------------------------------
# 损失函数
# ---------------------------------------------------------------------------

def compute_loss(model, batch_boards, batch_policies, batch_values):
    pred_policies, pred_values = model(batch_boards)

    # 策略损失: 软目标交叉熵
    log_probs = mx.log(mx.softmax(pred_policies, axis=-1) + 1e-8)
    policy_loss = -mx.mean(mx.sum(batch_policies * log_probs, axis=-1))

    # 价值损失: MSE
    value_loss = mx.mean((pred_values.squeeze() - batch_values) ** 2)

    return POLICY_LOSS_WEIGHT * policy_loss + VALUE_LOSS_WEIGHT * value_loss


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------

def train(batch_boards_cls, evaluate_fn=None):
    """
    主训练循环（固定时间预算）。

    参数
    ------
    batch_boards_cls : class
        <<<DOMAIN>>> 你的领域的 BatchBoards 类，用于并行自对弈。
    evaluate_fn : callable, 可选
        <<<DOMAIN>>> 你的领域的 evaluate_win_rate(model_path, level=...) 函数。
    """
    start_time = _time.time()

    model = PolicyValueNet()
    dummy = mx.zeros((1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE))
    _ = model(dummy)
    mx.eval(model.parameters())
    num_params = count_parameters(model)
    print(f"Parameters: {num_params / 1000:.1f}K")

    optimizer = optim.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    replay_buffer = []
    total_games = 0
    total_train_steps = 0
    cycle = 0
    last_loss = 0.0
    loss_and_grad = nn.value_and_grad(model, compute_loss)

    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Starting training loop...")
    print()

    while True:
        elapsed = _time.time() - start_time
        if elapsed >= TIME_BUDGET:
            break
        cycle += 1

        # === 自对弈 ===
        model.eval()
        data, games_done = run_self_play(model, batch_boards_cls,
                                         num_games=PARALLEL_GAMES,
                                         temperature=TEMPERATURE)
        total_games += games_done
        for item in data:
            if len(replay_buffer) >= REPLAY_BUFFER_SIZE:
                idx = random.randint(0, len(replay_buffer) - 1)
                replay_buffer[idx] = item
            else:
                replay_buffer.append(item)

        # === 训练 ===
        if len(replay_buffer) >= BATCH_SIZE:
            model.train()
            cycle_loss = 0.0
            steps = min(TRAIN_STEPS_PER_CYCLE, len(replay_buffer) // BATCH_SIZE)
            steps = max(steps, 1)

            for step in range(steps):
                if _time.time() - start_time >= TIME_BUDGET:
                    break
                indices = random.sample(range(len(replay_buffer)), BATCH_SIZE)
                b_boards = mx.array(np.stack([replay_buffer[i][0] for i in indices]))
                b_policies = mx.array(np.stack([replay_buffer[i][1] for i in indices]))
                b_values = mx.array(np.array([replay_buffer[i][2] for i in indices],
                                             dtype=np.float32))

                loss, grads = loss_and_grad(model, b_boards, b_policies, b_values)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                cycle_loss += loss.item()
                total_train_steps += 1

            last_loss = cycle_loss / steps if steps > 0 else 0.0

        # === 定期报告 ===
        if cycle % CYCLES_PER_REPORT == 0:
            elapsed = _time.time() - start_time
            print(f"Cycle {cycle:4d} | "
                  f"Buffer: {len(replay_buffer):6d} | "
                  f"Loss: {last_loss:.4f} | "
                  f"Games: {total_games:6d} | "
                  f"Steps: {total_train_steps:6d} | "
                  f"Time: {elapsed:.1f}s")

    # === 保存与汇总 ===
    total_elapsed = _time.time() - start_time
    print(f"\nTraining complete. Saving model to {MODEL_PATH}")
    save_model(model, MODEL_PATH)

    param_bytes = num_params * 4
    estimated_vram_mb = param_bytes / (1024 * 1024) * 10

    print(f"\n---")
    print(f"training_seconds: {total_elapsed:.1f}")
    print(f"total_seconds:    {total_elapsed:.1f}")
    print(f"peak_vram_mb:     {estimated_vram_mb:.1f}")
    print(f"num_params_K:     {num_params / 1000:.1f}")
    print(f"total_games:      {total_games}")
    print(f"total_train_steps: {total_train_steps}")
    print(f"final_loss:       {last_loss:.4f}")
    print()

    # === 评估 ===
    if evaluate_fn is not None:
        print(f"Running evaluation vs L{EVAL_LEVEL}...")
        try:
            evaluate_fn(MODEL_PATH, level=EVAL_LEVEL)
        except Exception as e:
            print(f"Evaluation error: {e}")
    else:
        print("(未提供评估函数 — 跳过评估)")


# ---------------------------------------------------------------------------
# 入口 — <<<DOMAIN>>> 替换为你的领域导入并运行
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # <<<DOMAIN>>> 示例用法:
    #   from game import BatchBoards
    #   from prepare import evaluate_win_rate
    #   train(batch_boards_cls=BatchBoards, evaluate_fn=evaluate_win_rate)
    print("这是框架模板。复制到 domains/<name>/train.py")
    print("并填充领域特定导入后再运行。")
