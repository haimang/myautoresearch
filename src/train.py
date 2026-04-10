"""
Gomoku training script — MLX on Apple Silicon.

This is the MUTABLE training script for the autoresearch loop.
The agent modifies hyperparameters and architecture between runs.
"""

import os
import random
import time as _time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from game import BatchBoards, BOARD_SIZE, BLACK, WHITE, EMPTY

# Try to import TIME_BUDGET and evaluate_win_rate from prepare.py.
# If prepare.py doesn't exist yet, use defaults.
try:
    from prepare import TIME_BUDGET
except ImportError:
    TIME_BUDGET = 300  # 5 minutes default

try:
    from prepare import evaluate_win_rate
except ImportError:
    evaluate_win_rate = None

# === Hyperparameters (autoresearch agent modifies these) ===
NUM_RES_BLOCKS = 6
NUM_FILTERS = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256          # training batch size
PARALLEL_GAMES = 64       # number of simultaneous self-play games
MCTS_SIMULATIONS = 0      # 0 = pure policy network (no MCTS)
TEMPERATURE = 1.0         # self-play exploration temperature
TEMP_THRESHOLD = 15       # after this many moves, use temp=0.1 (exploit)
REPLAY_BUFFER_SIZE = 50000
TRAIN_STEPS_PER_CYCLE = 50
CYCLES_PER_REPORT = 5     # print stats every N cycles
POLICY_LOSS_WEIGHT = 1.0
VALUE_LOSS_WEIGHT = 1.0
EVAL_LEVEL = 0            # opponent level for evaluation (0=random, 1=minimax2, 2=minimax4, 3=minimax6)

MODEL_PATH = "output/model.safetensors"

# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
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


class GomokuNet(nn.Module):
    def __init__(self, num_blocks=NUM_RES_BLOCKS, num_filters=NUM_FILTERS):
        super().__init__()
        # Input conv: 3 channels -> num_filters
        self.input_conv = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm(num_filters)

        # Residual tower
        self.res_blocks = [ResBlock(num_filters) for _ in range(num_blocks)]

        # Policy head: conv 1x1 -> 2 filters -> flatten -> FC -> 225
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm(2)
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

        # Value head: conv 1x1 -> 1 filter -> flatten -> FC -> 64 -> FC -> 1
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm(1)
        self.value_fc1 = nn.Linear(1 * BOARD_SIZE * BOARD_SIZE, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def __call__(self, x):
        # x: [B, 3, 15, 15] (NCHW from numpy encoding)
        # MLX Conv2d expects NHWC format, so transpose
        x = mx.transpose(x, (0, 2, 3, 1))  # [B, 15, 15, 3]

        # Input block
        x = nn.relu(self.input_bn(self.input_conv(x)))

        # Residual tower
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        p = nn.relu(self.policy_bn(self.policy_conv(x)))
        p = mx.reshape(p, (p.shape[0], -1))  # [B, 2*15*15]
        p = self.policy_fc(p)  # [B, 225]

        # Value head
        v = nn.relu(self.value_bn(self.value_conv(x)))
        v = mx.reshape(v, (v.shape[0], -1))  # [B, 1*15*15]
        v = nn.relu(self.value_fc1(v))
        v = mx.tanh(self.value_fc2(v))  # [B, 1], range [-1, 1]

        return p, v


# ---------------------------------------------------------------------------
# Model save / load
# ---------------------------------------------------------------------------

def save_model(model: GomokuNet, path: str):
    """Save model weights to safetensors format."""
    model.save_weights(path)


def load_model(path: str, num_blocks=NUM_RES_BLOCKS, num_filters=NUM_FILTERS) -> GomokuNet:
    """Load model weights from safetensors."""
    model = GomokuNet(num_blocks=num_blocks, num_filters=num_filters)
    model.load_weights(path)
    return model


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(model: GomokuNet) -> int:
    """Count total trainable parameters using tree_flatten."""
    params = model.parameters()
    leaves = nn.utils.tree_flatten(params)
    total = 0
    for _, v in leaves:
        total += v.size
    return total


# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------

def run_self_play(model, num_games=PARALLEL_GAMES, temperature=TEMPERATURE):
    """
    Run batched self-play games using the model.
    Uses game.BatchBoards for parallel game management.

    Plays all num_games to completion (no resetting — each slot plays exactly
    one game). With 64 parallel games at ~50 moves each, this needs ~50
    batched forward passes and finishes in seconds.

    Returns (training_data, games_completed):
      training_data: list of (board_encoding, policy_target, value_target)
      games_completed: int
    """
    model.eval()

    batch = BatchBoards(num_games)
    move_policies: list[list[np.ndarray]] = [[] for _ in range(num_games)]

    # Play all games to completion
    while not batch.get_finished_mask().all():
        encoded = batch.encode_all()  # [N, 3, 15, 15]
        legal_masks = batch.get_legal_masks()  # [N, 225]

        # Forward pass through model
        encoded_mx = mx.array(encoded)
        logits, values = model(encoded_mx)
        mx.eval(logits, values)
        logits_np = np.array(logits)  # [N, 225]

        # Mask illegal moves: set to -inf before softmax
        logits_np[legal_masks == 0] = -1e9

        # Compute actions for each game
        actions = np.zeros(num_games, dtype=np.int32)
        for i in range(num_games):
            if batch.winners[i] != 0:
                actions[i] = 0  # dummy, won't be applied
                continue

            game_logits = logits_np[i]

            # Determine temperature based on move count
            move_num = batch.move_counts[i]
            temp = temperature if move_num < TEMP_THRESHOLD else 0.1

            if temp < 0.01:
                # Greedy
                action = int(np.argmax(game_logits))
                policy_dist = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
                policy_dist[action] = 1.0
            else:
                # Apply temperature and softmax
                scaled = game_logits / temp
                scaled -= np.max(scaled)  # numerical stability
                probs = np.exp(scaled)
                probs_sum = probs.sum()
                if probs_sum > 0:
                    probs /= probs_sum
                else:
                    # Fallback: uniform over legal moves
                    probs = legal_masks[i].copy()
                    probs_sum = probs.sum()
                    if probs_sum > 0:
                        probs /= probs_sum
                    else:
                        # No legal moves — game should be terminal
                        actions[i] = 0
                        continue

                # Sample from distribution
                action = np.random.choice(BOARD_SIZE * BOARD_SIZE, p=probs)
                policy_dist = probs

            actions[i] = action
            move_policies[i].append(policy_dist.copy())

        # Apply actions to all games
        batch.step(actions)

    # Collect training data from all completed games
    all_training_data = []
    for i in range(num_games):
        game_data = _collect_game_with_policies(batch, i, move_policies[i])
        all_training_data.extend(game_data)

    return all_training_data, num_games


def _collect_game_with_policies(batch, game_idx, policies):
    """
    Collect training data for a finished game, pairing each move's
    board encoding with its policy distribution and value target.

    Returns list of (board_encoding [3,15,15], policy_target [225], value_target float).
    """
    winner = batch.winners[game_idx]
    history = batch.histories[game_idx]
    data = []

    for move_idx, (encoded, action, player) in enumerate(history):
        # Value target from this player's perspective
        if winner == -1:
            value_target = 0.0
        elif winner == player:
            value_target = 1.0
        else:
            value_target = -1.0

        # Policy target: use the actual distribution that was sampled from
        if move_idx < len(policies):
            policy_target = policies[move_idx]
        else:
            # Fallback: one-hot on the action taken
            policy_target = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
            policy_target[action] = 1.0

        data.append((encoded, policy_target, value_target))

    return data


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def compute_loss(model, batch_boards, batch_policies, batch_values):
    """
    Compute combined policy + value loss.

    batch_boards:   [B, 3, 15, 15]
    batch_policies: [B, 225] probability targets
    batch_values:   [B] float targets in [-1, 1]
    """
    pred_policies, pred_values = model(batch_boards)

    # Policy loss: cross-entropy with soft targets
    log_probs = mx.log(mx.softmax(pred_policies, axis=-1) + 1e-8)
    policy_loss = -mx.mean(mx.sum(batch_policies * log_probs, axis=-1))

    # Value loss: MSE
    value_loss = mx.mean((pred_values.squeeze() - batch_values) ** 2)

    return POLICY_LOSS_WEIGHT * policy_loss + VALUE_LOSS_WEIGHT * value_loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    """Main training function with fixed time budget."""
    start_time = _time.time()

    # Initialize model
    model = GomokuNet()
    # Warm up model parameters by doing a dummy forward pass
    dummy = mx.zeros((1, 3, BOARD_SIZE, BOARD_SIZE))
    _ = model(dummy)
    mx.eval(model.parameters())

    num_params = count_parameters(model)
    print(f"Parameters: {num_params / 1000:.1f}K")

    # Optimizer
    optimizer = optim.AdamW(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Replay buffer: list of (board_encoding, policy_target, value_target)
    replay_buffer: list[tuple[np.ndarray, np.ndarray, float]] = []

    # Training state
    total_games = 0
    total_train_steps = 0
    cycle = 0
    last_loss = 0.0

    # Loss and grad function
    loss_and_grad = nn.value_and_grad(model, compute_loss)

    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Starting training loop...")
    print()

    while True:
        elapsed = _time.time() - start_time
        if elapsed >= TIME_BUDGET:
            break

        cycle += 1

        # ----- Self-play cycle -----
        model.eval()
        data, games_done = run_self_play(
            model, num_games=PARALLEL_GAMES, temperature=TEMPERATURE
        )
        total_games += games_done

        # Add data to replay buffer
        for item in data:
            if len(replay_buffer) >= REPLAY_BUFFER_SIZE:
                # Replace a random entry
                idx = random.randint(0, len(replay_buffer) - 1)
                replay_buffer[idx] = item
            else:
                replay_buffer.append(item)

        # ----- Training cycle -----
        if len(replay_buffer) >= BATCH_SIZE:
            model.train()
            cycle_loss = 0.0
            steps_this_cycle = min(
                TRAIN_STEPS_PER_CYCLE,
                len(replay_buffer) // BATCH_SIZE
            )
            steps_this_cycle = max(steps_this_cycle, 1)

            for step in range(steps_this_cycle):
                # Check time budget during training
                if _time.time() - start_time >= TIME_BUDGET:
                    break

                # Sample a batch
                indices = random.sample(range(len(replay_buffer)), BATCH_SIZE)
                batch_boards_np = np.stack([replay_buffer[i][0] for i in indices])
                batch_policies_np = np.stack([replay_buffer[i][1] for i in indices])
                batch_values_np = np.array(
                    [replay_buffer[i][2] for i in indices], dtype=np.float32
                )

                # Convert to MLX arrays
                batch_boards_mx = mx.array(batch_boards_np)
                batch_policies_mx = mx.array(batch_policies_np)
                batch_values_mx = mx.array(batch_values_np)

                # Forward + backward + update
                loss, grads = loss_and_grad(
                    model, batch_boards_mx, batch_policies_mx, batch_values_mx
                )
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)

                cycle_loss += loss.item()
                total_train_steps += 1

            last_loss = cycle_loss / steps_this_cycle if steps_this_cycle > 0 else 0.0

        # ----- Periodic reporting -----
        if cycle % CYCLES_PER_REPORT == 0:
            elapsed = _time.time() - start_time
            print(
                f"Cycle {cycle:4d} | "
                f"Buffer: {len(replay_buffer):6d} | "
                f"Loss: {last_loss:.4f} | "
                f"Games: {total_games:6d} | "
                f"Steps: {total_train_steps:6d} | "
                f"Time: {elapsed:.1f}s"
            )

    # ----- Save model -----
    total_elapsed = _time.time() - start_time
    print()
    print(f"Training complete. Saving model to {MODEL_PATH}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    save_model(model, MODEL_PATH)

    # ----- Estimate peak VRAM -----
    # MLX doesn't have a direct peak VRAM query; estimate from param size.
    # On Apple Silicon, this is unified memory usage.
    param_bytes = num_params * 4  # float32
    estimated_vram_mb = param_bytes / (1024 * 1024) * 10  # rough 10x multiplier for activations + optimizer

    # ----- Print summary -----
    print()
    print("---")
    print(f"training_seconds: {total_elapsed:.1f}")
    print(f"total_seconds:    {total_elapsed:.1f}")
    print(f"peak_vram_mb:     {estimated_vram_mb:.1f}")
    print(f"num_params_K:     {num_params / 1000:.1f}")
    print(f"total_games:      {total_games}")
    print(f"total_train_steps: {total_train_steps}")
    print(f"final_loss:       {last_loss:.4f}")
    print()

    # ----- Evaluation -----
    # Run evaluation in a subprocess to get a clean Metal GPU state.
    # The training process accumulates Metal buffers; a fresh process avoids
    # contention with any residual allocations.
    if evaluate_win_rate is not None:
        print(f"Running evaluation vs L{EVAL_LEVEL}...")
        import subprocess, sys
        src_dir = os.path.dirname(os.path.abspath(__file__))
        env = {**os.environ, 'PYTHONPATH': src_dir}
        eval_cmd = [
            sys.executable, "-c",
            f"from prepare import evaluate_win_rate; evaluate_win_rate('{MODEL_PATH}', level={EVAL_LEVEL})"
        ]
        proc = subprocess.run(eval_cmd, capture_output=True, text=True, env=env)
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.returncode != 0:
            print(f"Evaluation error (exit {proc.returncode}):")
            if proc.stderr:
                print(proc.stderr.strip())
    else:
        print("(prepare.py not found — skipping evaluation)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Training with {PARALLEL_GAMES} parallel games, "
          f"{NUM_RES_BLOCKS} res blocks, {NUM_FILTERS} filters")
    print(f"Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}")

    # Count and display params
    model = GomokuNet()
    dummy = mx.zeros((1, 3, BOARD_SIZE, BOARD_SIZE))
    _ = model(dummy)
    mx.eval(model.parameters())
    num_params = count_parameters(model)
    print(f"Parameters: {num_params / 1000:.1f}K")
    del model

    train()
