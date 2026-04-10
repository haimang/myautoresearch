"""
Gomoku training script — MLX on Apple Silicon.

This is the MUTABLE training script for the autoresearch loop.
The agent modifies hyperparameters and architecture between runs.
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time as _time
import uuid

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from game import BatchBoards, BOARD_SIZE, BLACK, WHITE, EMPTY

# Try to import TIME_BUDGET and evaluate_win_rate from prepare.py.
try:
    from prepare import TIME_BUDGET
except ImportError:
    TIME_BUDGET = 300

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

def train(args):
    """Main training function with checkpoint tracking and text TUI."""
    import tracker as _tracker

    run_id = str(uuid.uuid4())
    run_id_short = run_id[:8]
    start_time = _time.time()

    # Resolve effective parameters (CLI args override module constants)
    time_budget = args.time_budget
    eval_level = args.eval_level
    eval_interval = args.eval_interval
    probe_games = args.probe_games
    full_eval_games = args.full_eval_games
    target_win_rate = args.target_win_rate
    target_games = args.target_games
    parallel_games = args.parallel_games
    probe_window = args.probe_window

    # Ensure at least one stop condition
    if time_budget is None and target_win_rate is None and target_games is None:
        print("Warning: no stop condition set, defaulting to --time-budget 300")
        time_budget = 300

    # --- Output paths (per-run UUID directory) ---
    output_dir = f"output/{run_id}"
    model_path = f"{output_dir}/model.safetensors"
    ckpt_dir = f"{output_dir}/checkpoints"
    recording_dir = f"{output_dir}/recordings"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(recording_dir, "games"), exist_ok=True)

    # --- Resume support ---
    resumed_from = None
    initial_cycle = 0
    initial_ckpt_wr = 0.0
    resume_model_path = None

    if args.resume:
        db_tmp = _tracker.init_db()
        old_run = _tracker.get_run(db_tmp, args.resume)
        if not old_run:
            print(f"Error: run '{args.resume}' not found in tracker.db")
            sys.exit(1)
        resolved_id = old_run["id"]
        latest_ckpt = _tracker.get_latest_checkpoint(db_tmp, resolved_id)
        if not latest_ckpt:
            print(f"Error: no checkpoint found for run '{resolved_id[:8]}'")
            sys.exit(1)
        resumed_from = resolved_id
        resume_model_path = latest_ckpt["model_path"]
        initial_cycle = latest_ckpt["cycle"]
        initial_ckpt_wr = latest_ckpt["win_rate"]
        print(f"Resuming from run {resolved_id[:8]}  "
              f"cycle={initial_cycle}  wr={initial_ckpt_wr:.1%}  "
              f"model={resume_model_path}")
        db_tmp.close()

    # Initialize tracker DB
    db_conn = _tracker.init_db()
    hw_info = _tracker.collect_hardware_info()

    hyperparams = {
        "num_res_blocks": NUM_RES_BLOCKS,
        "num_filters": NUM_FILTERS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "parallel_games": parallel_games,
        "mcts_simulations": MCTS_SIMULATIONS,
        "temperature": TEMPERATURE,
        "temp_threshold": TEMP_THRESHOLD,
        "replay_buffer_size": REPLAY_BUFFER_SIZE,
        "train_steps_per_cycle": TRAIN_STEPS_PER_CYCLE,
        "time_budget": time_budget,
        "target_win_rate": target_win_rate,
        "target_games": target_games,
        "eval_level": eval_level,
    }
    _tracker.create_run(db_conn, run_id, hyperparams, hw_info,
                        resumed_from=resumed_from, output_dir=output_dir)

    # Initialize model
    if resume_model_path and os.path.exists(resume_model_path):
        model = load_model(resume_model_path)
    else:
        model = GomokuNet()
    dummy = mx.zeros((1, 3, BOARD_SIZE, BOARD_SIZE))
    _ = model(dummy)
    mx.eval(model.parameters())

    num_params = count_parameters(model)

    # Optimizer
    optimizer = optim.AdamW(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Replay buffer
    replay_buffer: list[tuple[np.ndarray, np.ndarray, float]] = []

    # Training state
    total_games = 0
    total_train_steps = 0
    cycle = initial_cycle
    last_loss = 0.0
    last_probe_wr: float | None = None
    last_ckpt_wr: float = initial_ckpt_wr
    num_checkpoints = 0
    stop_reason = "time_budget"

    # History for sparklines
    loss_history: list[float] = []
    wr_history: list[float] = []

    # Loss and grad function
    loss_and_grad = nn.value_and_grad(model, compute_loss)

    # --- Plain text TUI ---
    use_tui = sys.stdout.isatty()
    events: list[str] = []

    def _sparkline(values: list[float], width: int = 30) -> str:
        if not values:
            return ""
        chars = "▁▂▃▄▅▆▇█"
        recent = values[-width:]
        lo, hi = min(recent), max(recent)
        span = hi - lo if hi > lo else 1.0
        return "".join(chars[min(int((v - lo) / span * 7), 7)] for v in recent)

    def _smoothed_wr(window: int = probe_window) -> float | None:
        """Sliding average of recent probe WRs for stable decisions."""
        if not wr_history:
            return None
        w = min(window, len(wr_history))
        return sum(wr_history[-w:]) / w

    def _progress_bar(elapsed: float, budget: float | None, width: int = 34) -> str:
        if budget is None or budget <= 0:
            return ""
        frac = min(elapsed / budget, 1.0)
        filled = int(frac * width)
        bar = "━" * filled + "─" * (width - filled)
        em, es = divmod(int(elapsed), 60)
        bm, bs = divmod(int(budget), 60)
        return f"{bar} {frac:3.0%} {em}:{es:02d} / {bm}:{bs:02d}"

    def _draw_panel():
        """Render the full text TUI panel."""
        W = 62  # inner width
        chip = hw_info.get("chip", "")
        elapsed = _time.time() - start_time

        lines = []
        lines.append("╭" + "─" * W + "╮")
        hdr = f" Run: {run_id_short}  {chip}  {num_params/1000:.1f}K params"
        lines.append("│" + hdr.ljust(W) + "│")
        if time_budget is not None:
            pbar = " " + _progress_bar(elapsed, time_budget)
            lines.append("│" + pbar.ljust(W) + "│")
        else:
            tgt = f"  Target: {target_win_rate:.0%} WR" if target_win_rate else ""
            em, es = divmod(int(elapsed), 60)
            info = f" Elapsed: {em}:{es:02d}{tgt}"
            lines.append("│" + info.ljust(W) + "│")
        lines.append("├" + "─" * W + "┤")
        wr_str = f"{last_probe_wr:.1%}" if last_probe_wr is not None else "—"
        sm_wr = _smoothed_wr()
        sm_str = f" avg:{sm_wr:.1%}" if sm_wr is not None and len(wr_history) > 1 else ""
        r1 = f"  Cycle {cycle:5d}  │  Loss {last_loss:8.4f}  │  Games {total_games:7d}"
        lines.append("│" + r1.ljust(W) + "│")
        r2 = f"  Steps {total_train_steps:5d}  │  Buffer {len(replay_buffer):5d}  │  WR {wr_str}{sm_str} (L{eval_level})"
        lines.append("│" + r2.ljust(W) + "│")
        if loss_history or wr_history:
            lines.append("├" + "─" * W + "┤")
            if wr_history:
                spark = _sparkline(wr_history)
                wr_last = wr_history[-1]
                sl = f"  Win Rate {spark} {wr_last:.0%}"
                lines.append("│" + sl.ljust(W) + "│")
            if loss_history:
                spark = _sparkline(loss_history)
                sl = f"  Loss     {spark} {last_loss:.2f}"
                lines.append("│" + sl.ljust(W) + "│")
        if events:
            lines.append("├" + "─" * W + "┤")
            for e in events[-6:]:
                lines.append("│" + f"  {e}".ljust(W) + "│")
        lines.append("╰" + "─" * W + "╯")
        return "\n".join(lines)

    def _update_tui():
        if use_tui:
            sys.stdout.write("\033[H\033[J")
            sys.stdout.write(_draw_panel() + "\n")
            sys.stdout.flush()

    def _log_event(msg: str):
        ts = _time.strftime("%H:%M:%S")
        events.append(f"[{ts}] {msg}")
        if not use_tui:
            print(f"[{ts}] {msg}")

    _log_event(f"Started run {run_id_short} | {num_params/1000:.1f}K params"
               + (f" | budget {time_budget}s" if time_budget else ""))

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    try:
        while True:
            elapsed = _time.time() - start_time
            if time_budget is not None and elapsed >= time_budget:
                stop_reason = "time_budget"
                break
            if target_games and total_games >= target_games:
                stop_reason = "target_games"
                break

            cycle += 1

            # ----- Self-play -----
            model.eval()
            data, games_done = run_self_play(
                model, num_games=parallel_games, temperature=TEMPERATURE
            )
            total_games += games_done

            for item in data:
                if len(replay_buffer) >= REPLAY_BUFFER_SIZE:
                    idx = random.randint(0, len(replay_buffer) - 1)
                    replay_buffer[idx] = item
                else:
                    replay_buffer.append(item)

            # ----- Training -----
            if len(replay_buffer) >= BATCH_SIZE:
                model.train()
                cycle_loss = 0.0
                steps_this_cycle = min(
                    TRAIN_STEPS_PER_CYCLE,
                    len(replay_buffer) // BATCH_SIZE
                )
                steps_this_cycle = max(steps_this_cycle, 1)

                for step in range(steps_this_cycle):
                    if time_budget is not None and _time.time() - start_time >= time_budget:
                        break

                    indices = random.sample(range(len(replay_buffer)), BATCH_SIZE)
                    batch_boards_np = np.stack([replay_buffer[i][0] for i in indices])
                    batch_policies_np = np.stack([replay_buffer[i][1] for i in indices])
                    batch_values_np = np.array(
                        [replay_buffer[i][2] for i in indices], dtype=np.float32
                    )

                    batch_boards_mx = mx.array(batch_boards_np)
                    batch_policies_mx = mx.array(batch_policies_np)
                    batch_values_mx = mx.array(batch_values_np)

                    loss, grads = loss_and_grad(
                        model, batch_boards_mx, batch_policies_mx, batch_values_mx
                    )
                    optimizer.update(model, grads)
                    mx.eval(model.parameters(), optimizer.state)

                    cycle_loss += loss.item()
                    total_train_steps += 1

                last_loss = cycle_loss / steps_this_cycle if steps_this_cycle > 0 else 0.0

            # Record cycle metrics
            metric = {
                "cycle": cycle,
                "timestamp_s": _time.time() - start_time,
                "loss": last_loss,
                "total_games": total_games,
                "total_steps": total_train_steps,
                "buffer_size": len(replay_buffer),
            }

            # ----- Probe evaluation -----
            if cycle % eval_interval == 0 and evaluate_win_rate is not None:
                model.eval()
                probe_wr = _quick_eval(model, eval_level, probe_games)
                last_probe_wr = probe_wr
                wr_history.append(probe_wr)
                metric["win_rate"] = probe_wr
                metric["eval_type"] = "probe"
                metric["eval_games"] = probe_games
                metric["eval_level"] = eval_level

                sm_wr = _smoothed_wr()
                sm_str = f" (avg:{sm_wr:.1%})" if sm_wr is not None and len(wr_history) > 1 else ""
                _log_event(f"Probe: {probe_wr:.1%}{sm_str} ({probe_games} games vs L{eval_level})")

                # Check checkpoint threshold using smoothed WR
                effective_wr = sm_wr if sm_wr is not None else probe_wr
                crossed = _tracker.crossed_threshold(effective_wr, last_ckpt_wr)
                if crossed is not None:
                    _do_checkpoint(
                        model, db_conn, run_id, run_id_short, cycle,
                        total_train_steps, last_loss, effective_wr,
                        eval_level, full_eval_games, num_params,
                        start_time, events, _log_event,
                        ckpt_dir, recording_dir,
                        threshold=crossed,
                    )
                    last_ckpt_wr = effective_wr
                    num_checkpoints += 1

                # Early stop on target win rate (using smoothed WR)
                if target_win_rate and effective_wr >= target_win_rate:
                    _log_event(f"🎯 Target win rate {target_win_rate:.0%} reached! (avg:{effective_wr:.1%})")
                    stop_reason = "target_win_rate"
                    _update_tui()
                    break

            if last_loss > 0:
                loss_history.append(last_loss)

            _tracker.save_cycle_metric(db_conn, run_id, metric)

            # Update TUI or print
            if use_tui:
                _update_tui()
            elif cycle % CYCLES_PER_REPORT == 0:
                elapsed = _time.time() - start_time
                wr_str = f" | WR: {last_probe_wr:.1%}" if last_probe_wr is not None else ""
                print(
                    f"Cycle {cycle:4d} | "
                    f"Loss: {last_loss:.4f} | "
                    f"Games: {total_games:6d} | "
                    f"Steps: {total_train_steps:6d} | "
                    f"Buffer: {len(replay_buffer):6d}{wr_str} | "
                    f"Time: {elapsed:.1f}s"
                )

    except KeyboardInterrupt:
        stop_reason = "interrupted"
        _log_event("Training interrupted by user")
    finally:
        if use_tui:
            # Print final state after clearing
            sys.stdout.write("\033[H\033[J")
            sys.stdout.write(_draw_panel() + "\n")
            sys.stdout.flush()

    # ----- Save final model -----
    total_elapsed = _time.time() - start_time
    save_model(model, model_path)
    _log_event(f"Model saved to {model_path}")

    # ----- Final evaluation (subprocess) -----
    final_wr = last_probe_wr
    if evaluate_win_rate is not None:
        print(f"\nRunning final evaluation vs L{eval_level} ({full_eval_games} games)...")
        tag = f"final_c{cycle:04d}"
        result = _subprocess_eval(
            model_path, eval_level, full_eval_games, tag, run_id,
            recording_dir=recording_dir,
        )
        if result:
            final_wr = result.get("win_rate", final_wr)
            print(f"Final win_rate: {final_wr:.1%}")

            # Save as checkpoint
            ckpt_path = f"{ckpt_dir}/{tag}.safetensors"
            shutil.copy2(model_path, ckpt_path)

            ckpt_id = _tracker.save_checkpoint(db_conn, run_id, {
                "tag": tag,
                "cycle": cycle,
                "step": total_train_steps,
                "loss": last_loss,
                "win_rate": final_wr,
                "eval_level": eval_level,
                "eval_games": full_eval_games,
                "wins": result.get("wins"),
                "losses": result.get("losses"),
                "draws": result.get("draws"),
                "avg_game_length": result.get("avg_game_length"),
                "num_params": num_params,
                "model_path": ckpt_path,
                "model_size_bytes": os.path.getsize(ckpt_path),
                "train_elapsed_s": total_elapsed,
                "eval_elapsed_s": result.get("eval_elapsed_s"),
            })
            # Save recording metadata
            for gd in result.get("game_details", []):
                if "game_file" in gd:
                    _tracker.save_recording(db_conn, ckpt_id, run_id, gd)
            db_conn.commit()
            num_checkpoints += 1

    # ----- Finalize run -----
    _tracker.finish_run(db_conn, run_id, {
        "status": "completed" if stop_reason != "interrupted" else "interrupted",
        "total_cycles": cycle - initial_cycle,
        "total_games": total_games,
        "total_steps": total_train_steps,
        "final_loss": last_loss,
        "final_win_rate": final_wr,
        "num_params": num_params,
        "num_checkpoints": num_checkpoints,
        "wall_time_s": total_elapsed,
        "peak_memory_mb": None,
    })
    db_conn.close()

    # ----- Print summary -----
    print()
    print("=" * 60)
    resumed_str = f"  (resumed from {resumed_from[:8]})" if resumed_from else ""
    print(f"Run:        {run_id_short}{resumed_str} ({stop_reason})")
    print(f"Cycles:     {cycle - initial_cycle} (total cycle #{cycle})")
    print(f"Games:      {total_games}")
    print(f"Steps:      {total_train_steps}")
    print(f"Final loss: {last_loss:.4f}")
    if final_wr is not None:
        print(f"Win rate:   {final_wr:.1%} (vs L{eval_level})")
    print(f"Checkpoints:{num_checkpoints}")
    print(f"Wall time:  {total_elapsed:.1f}s")
    print(f"Output:     {output_dir}/")
    print(f"Tracker:    output/tracker.db")
    print("=" * 60)


def _quick_eval(model, level: int, n_games: int) -> float:
    """In-process lightweight evaluation. Returns win_rate."""
    from prepare import OPPONENTS
    opponent_fn = OPPONENTS[level]

    wins = 0
    for game_i in range(n_games):
        nn_is_black = game_i < n_games // 2
        nn_player = BLACK if nn_is_black else WHITE

        from game import Board
        board = Board()
        while not board.is_terminal():
            if board.current_player == nn_player:
                encoded = board.encode()
                x = mx.array(encoded[np.newaxis, ...])
                policy_logits, _ = model(x)
                policy = policy_logits[0]
                legal_mask = mx.array(board.get_legal_mask())
                masked = mx.where(legal_mask > 0, policy, mx.array(float("-inf")))
                action = int(mx.argmax(masked).item())
                row, col = divmod(action, BOARD_SIZE)
            else:
                row, col = opponent_fn(board)
            board.place(row, col)

        if board.winner == nn_player:
            wins += 1

        if game_i % 20 == 0:
            mx.clear_cache()

    return wins / n_games if n_games > 0 else 0.0


def _subprocess_eval(model_path: str, level: int, n_games: int,
                     tag: str, run_id: str,
                     recording_dir: str = "") -> dict | None:
    """Run full evaluation in subprocess, return parsed result dict."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    env = {**os.environ, "PYTHONPATH": src_dir, "PYTHONUNBUFFERED": "1"}
    rec_arg = f", recording_dir='{recording_dir}'" if recording_dir else ""
    code = (
        f"import json; from prepare import evaluate_win_rate; "
        f"r = evaluate_win_rate('{model_path}', level={level}, "
        f"n_games={n_games}, record_games={n_games}, "
        f"tag='{tag}', run_id='{run_id}'{rec_arg}); "
        f"print('JSON_RESULT:' + json.dumps(r, default=str))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0:
        print(f"Evaluation error (exit {proc.returncode}):")
        if proc.stderr:
            print(proc.stderr.strip()[:500])
        return None

    # Parse JSON result from stdout
    for line in proc.stdout.splitlines():
        if line.startswith("JSON_RESULT:"):
            return json.loads(line[len("JSON_RESULT:"):])

    # Fallback: print raw output
    if proc.stdout:
        print(proc.stdout)
    return None


def _do_checkpoint(model, db_conn, run_id, run_id_short, cycle,
                   total_steps, loss, win_rate, eval_level,
                   full_eval_games, num_params, start_time,
                   events, log_event_fn, ckpt_dir, recording_dir,
                   threshold=None):
    """Save checkpoint, run full eval in subprocess, record to DB."""
    import tracker as _tracker

    # Tag uses the crossed threshold (not raw probe WR)
    wr_pct = int((threshold if threshold is not None else win_rate) * 100)
    tag = f"wr{wr_pct:03d}_c{cycle:04d}"
    ckpt_path = f"{ckpt_dir}/{tag}.safetensors"
    save_model(model, ckpt_path)

    log_event_fn(f"✓ Checkpoint {tag}  wr={win_rate:.1%}")

    # Full eval in subprocess
    elapsed = _time.time() - start_time
    result = _subprocess_eval(
        ckpt_path, eval_level, full_eval_games, tag, run_id,
        recording_dir=recording_dir,
    )

    if result:
        full_wr = result.get("win_rate", win_rate)
        log_event_fn(
            f"  Full eval: {full_wr:.1%} "
            f"({result.get('wins', '?')}W/{result.get('losses', '?')}L "
            f"in {result.get('eval_elapsed_s', 0):.1f}s)"
        )

        ckpt_id = _tracker.save_checkpoint(db_conn, run_id, {
            "tag": tag,
            "cycle": cycle,
            "step": total_steps,
            "loss": loss,
            "win_rate": full_wr,
            "eval_level": eval_level,
            "eval_games": full_eval_games,
            "wins": result.get("wins"),
            "losses": result.get("losses"),
            "draws": result.get("draws"),
            "avg_game_length": result.get("avg_game_length"),
            "num_params": num_params,
            "model_path": ckpt_path,
            "model_size_bytes": os.path.getsize(ckpt_path),
            "train_elapsed_s": elapsed,
            "eval_elapsed_s": result.get("eval_elapsed_s"),
        })
        # Save recording metadata
        for gd in result.get("game_details", []):
            if "game_file" in gd:
                _tracker.save_recording(db_conn, ckpt_id, run_id, gd)
        db_conn.commit()
    else:
        # Eval failed — save checkpoint without eval details
        _tracker.save_checkpoint(db_conn, run_id, {
            "tag": tag,
            "cycle": cycle,
            "step": total_steps,
            "loss": loss,
            "win_rate": win_rate,
            "eval_level": eval_level,
            "eval_games": 0,
            "num_params": num_params,
            "model_path": ckpt_path,
            "model_size_bytes": os.path.getsize(ckpt_path),
            "train_elapsed_s": elapsed,
        })
        db_conn.commit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAG-Gomoku Training")
    p.add_argument("--time-budget", type=int, default=None,
                   help="Training time budget in seconds (default: unlimited)")
    p.add_argument("--target-win-rate", type=float, default=None,
                   help="Stop when smoothed win rate reaches this value")
    p.add_argument("--target-games", type=int, default=None,
                   help="Stop after this many self-play games")
    p.add_argument("--eval-level", type=int, default=EVAL_LEVEL,
                   help=f"Evaluation opponent level 0-3 (default: {EVAL_LEVEL})")
    p.add_argument("--eval-interval", type=int, default=15,
                   help="Probe evaluation every N cycles (default: 15)")
    p.add_argument("--probe-games", type=int, default=50,
                   help="Games per probe evaluation (default: 50)")
    p.add_argument("--probe-window", type=int, default=3,
                   help="Sliding window size for smoothed win rate (default: 3)")
    p.add_argument("--full-eval-games", type=int, default=200,
                   help="Games per full evaluation at checkpoint (default: 200)")
    p.add_argument("--parallel-games", type=int, default=PARALLEL_GAMES,
                   help=f"Number of simultaneous self-play games (default: {PARALLEL_GAMES})")
    p.add_argument("--resume", type=str, default=None,
                   help="UUID of a previous run to resume from its last checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
