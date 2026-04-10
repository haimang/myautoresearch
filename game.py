"""
Gomoku (五子棋) game engine with optional pygame rendering.

This is the foundation of the entire project — used by:
  - play.py:    human vs AI gameplay
  - prepare.py: evaluation matches (headless)
  - train.py:   batched self-play (headless)
  - replay.py:  game replay with rendering
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOARD_SIZE = 15
WIN_LENGTH = 5
EMPTY = 0
BLACK = 1
WHITE = 2

# ---------------------------------------------------------------------------
# Game record (for replay / video)
# ---------------------------------------------------------------------------

@dataclass
class MoveRecord:
    step: int
    row: int
    col: int
    player: int  # BLACK or WHITE
    # Optional NN metadata (filled during evaluation, not during raw self-play)
    policy_entropy: Optional[float] = None
    value: Optional[float] = None
    top3: Optional[list] = None


@dataclass
class GameRecord:
    moves: list[MoveRecord] = field(default_factory=list)
    result: int = 0  # 0=ongoing, BLACK=black wins, WHITE=white wins, -1=draw
    black_name: str = "black"
    white_name: str = "white"
    metadata: dict = field(default_factory=dict)

    def add_move(self, step, row, col, player, **kwargs):
        self.moves.append(MoveRecord(step=step, row=row, col=col, player=player, **kwargs))

    def to_dict(self):
        moves = []
        for m in self.moves:
            d = {"step": m.step, "row": m.row, "col": m.col, "player": m.player}
            if m.policy_entropy is not None:
                d["policy_entropy"] = round(m.policy_entropy, 4)
            if m.value is not None:
                d["value"] = round(m.value, 4)
            if m.top3 is not None:
                d["top3"] = m.top3
            moves.append(d)
        result_map = {0: "ongoing", BLACK: "black_win", WHITE: "white_win", -1: "draw"}
        return {
            "black": self.black_name,
            "white": self.white_name,
            "result": result_map.get(self.result, "unknown"),
            "total_moves": len(self.moves),
            "moves": moves,
            **self.metadata,
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GameRecord":
        with open(path) as f:
            data = json.load(f)
        rec = cls(
            black_name=data.get("black", "black"),
            white_name=data.get("white", "white"),
        )
        result_map = {"ongoing": 0, "black_win": BLACK, "white_win": WHITE, "draw": -1}
        rec.result = result_map.get(data.get("result", "ongoing"), 0)
        for m in data.get("moves", []):
            rec.add_move(
                step=m["step"], row=m["row"], col=m["col"], player=m["player"],
                policy_entropy=m.get("policy_entropy"),
                value=m.get("value"),
                top3=m.get("top3"),
            )
        rec.metadata = {k: v for k, v in data.items()
                        if k not in ("black", "white", "result", "total_moves", "moves")}
        return rec


# ---------------------------------------------------------------------------
# Board logic
# ---------------------------------------------------------------------------

class Board:
    """Pure game logic — no rendering, maximum speed."""

    def __init__(self):
        self.grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = BLACK
        self.move_count = 0
        self.last_move: Optional[tuple[int, int]] = None
        self.winner = 0  # 0=ongoing, BLACK/WHITE=winner, -1=draw
        self.history: list[tuple[int, int, int]] = []  # (row, col, player)

    def copy(self) -> "Board":
        b = Board()
        b.grid = self.grid.copy()
        b.current_player = self.current_player
        b.move_count = self.move_count
        b.last_move = self.last_move
        b.winner = self.winner
        b.history = list(self.history)
        return b

    def is_legal(self, row: int, col: int) -> bool:
        return (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE
                and self.grid[row, col] == EMPTY and self.winner == 0)

    def get_legal_moves(self) -> list[tuple[int, int]]:
        if self.winner != 0:
            return []
        rows, cols = np.where(self.grid == EMPTY)
        return list(zip(rows.tolist(), cols.tolist()))

    def get_legal_mask(self) -> np.ndarray:
        """Return a flat [225] boolean mask of legal moves."""
        return (self.grid == EMPTY).flatten().astype(np.float32)

    def place(self, row: int, col: int) -> bool:
        """Place a stone for current_player. Returns True if valid."""
        if not self.is_legal(row, col):
            return False
        player = self.current_player
        self.grid[row, col] = player
        self.move_count += 1
        self.last_move = (row, col)
        self.history.append((row, col, player))

        if self._check_win(row, col, player):
            self.winner = player
        elif self.move_count >= BOARD_SIZE * BOARD_SIZE:
            self.winner = -1  # draw

        self.current_player = WHITE if player == BLACK else BLACK
        return True

    def _check_win(self, row: int, col: int, player: int) -> bool:
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                r, c = row + dr * sign, col + dc * sign
                while (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE
                       and self.grid[r, c] == player):
                    count += 1
                    r += dr * sign
                    c += dc * sign
            if count >= WIN_LENGTH:
                return True
        return False

    def is_terminal(self) -> bool:
        return self.winner != 0

    def encode(self) -> np.ndarray:
        """
        Encode board state for neural network input.

        Returns: [3, 15, 15] float32 array
          Channel 0: current player's stones
          Channel 1: opponent's stones
          Channel 2: all ones if current player is BLACK, all zeros if WHITE
        """
        current = self.current_player
        opponent = WHITE if current == BLACK else BLACK
        planes = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        planes[0] = (self.grid == current).astype(np.float32)
        planes[1] = (self.grid == opponent).astype(np.float32)
        if current == BLACK:
            planes[2] = 1.0
        return planes

    def encode_for_player(self, player: int) -> np.ndarray:
        """Encode from a specific player's perspective (used for training data)."""
        opponent = WHITE if player == BLACK else BLACK
        planes = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        planes[0] = (self.grid == player).astype(np.float32)
        planes[1] = (self.grid == opponent).astype(np.float32)
        if player == BLACK:
            planes[2] = 1.0
        return planes

    def get_candidate_moves(self, radius: int = 2) -> list[tuple[int, int]]:
        """
        Get empty positions within `radius` of any existing stone.
        Used by minimax to prune the search space dramatically.
        If the board is empty, returns center.
        """
        if self.move_count == 0:
            return [(BOARD_SIZE // 2, BOARD_SIZE // 2)]

        candidates = set()
        occupied = np.argwhere(self.grid != EMPTY)
        for r, c in occupied:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE
                            and self.grid[nr, nc] == EMPTY):
                        candidates.add((nr, nc))
        return list(candidates)


# ---------------------------------------------------------------------------
# Pygame renderer
# ---------------------------------------------------------------------------

# Colors
BG_COLOR = (220, 179, 92)       # wood-tone background
LINE_COLOR = (50, 40, 30)       # dark brown grid lines
BLACK_STONE = (20, 20, 20)
WHITE_STONE = (240, 240, 240)
LAST_MOVE_MARKER = (200, 50, 50)
HOVER_COLOR = (100, 100, 100, 128)
STAR_COLOR = (50, 40, 30)

# Layout
CELL_SIZE = 40
MARGIN = 40
BOARD_PX = CELL_SIZE * (BOARD_SIZE - 1) + MARGIN * 2
INFO_WIDTH = 260
WINDOW_WIDTH = BOARD_PX + INFO_WIDTH
WINDOW_HEIGHT = BOARD_PX
STONE_RADIUS = CELL_SIZE // 2 - 2


def _grid_to_pixel(row: int, col: int) -> tuple[int, int]:
    return MARGIN + col * CELL_SIZE, MARGIN + row * CELL_SIZE


def _pixel_to_grid(x: int, y: int) -> Optional[tuple[int, int]]:
    col = round((x - MARGIN) / CELL_SIZE)
    row = round((y - MARGIN) / CELL_SIZE)
    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
        return row, col
    return None


class Renderer:
    """Pygame-based board renderer. Can be used live or for frame export."""

    def __init__(self, title: str = "MAG Gomoku"):
        import pygame
        self.pygame = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 22, bold=True)
        self.clock = pygame.time.Clock()

    def draw_board(self, board: Board, info: Optional[dict] = None,
                   policy_map: Optional[np.ndarray] = None,
                   hover: Optional[tuple[int, int]] = None):
        pg = self.pygame
        screen = self.screen

        # Background
        screen.fill(BG_COLOR)

        # Grid lines
        for i in range(BOARD_SIZE):
            x0, y0 = _grid_to_pixel(i, 0)
            x1, y1 = _grid_to_pixel(i, BOARD_SIZE - 1)
            pg.draw.line(screen, LINE_COLOR, (x0, y0), (x1, y1), 1)
            x0, y0 = _grid_to_pixel(0, i)
            x1, y1 = _grid_to_pixel(BOARD_SIZE - 1, i)
            pg.draw.line(screen, LINE_COLOR, (x0, y0), (x1, y1), 1)

        # Star points (天元 + 四星)
        star_points = [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]
        for r, c in star_points:
            px, py = _grid_to_pixel(r, c)
            pg.draw.circle(screen, STAR_COLOR, (px, py), 4)

        # Policy heatmap overlay
        if policy_map is not None:
            max_p = policy_map.max()
            if max_p > 0:
                surf = pg.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pg.SRCALPHA)
                for r in range(BOARD_SIZE):
                    for c in range(BOARD_SIZE):
                        if board.grid[r, c] == EMPTY:
                            p = policy_map[r, c] / max_p
                            if p > 0.05:
                                px, py = _grid_to_pixel(r, c)
                                alpha = int(p * 160)
                                red = int(p * 255)
                                blue = int((1 - p) * 180)
                                pg.draw.circle(surf, (red, 40, blue, alpha),
                                               (px, py), STONE_RADIUS - 2)
                screen.blit(surf, (0, 0))

        # Stones
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board.grid[r, c] != EMPTY:
                    px, py = _grid_to_pixel(r, c)
                    color = BLACK_STONE if board.grid[r, c] == BLACK else WHITE_STONE
                    # Shadow
                    pg.draw.circle(screen, (0, 0, 0, 60), (px + 2, py + 2), STONE_RADIUS)
                    # Stone
                    pg.draw.circle(screen, color, (px, py), STONE_RADIUS)
                    if board.grid[r, c] == WHITE:
                        pg.draw.circle(screen, (180, 180, 180), (px, py), STONE_RADIUS, 1)

        # Last move marker
        if board.last_move:
            lr, lc = board.last_move
            px, py = _grid_to_pixel(lr, lc)
            pg.draw.circle(screen, LAST_MOVE_MARKER, (px, py), 5)

        # Hover indicator
        if hover and board.grid[hover[0], hover[1]] == EMPTY:
            px, py = _grid_to_pixel(hover[0], hover[1])
            surf = pg.Surface((STONE_RADIUS * 2, STONE_RADIUS * 2), pg.SRCALPHA)
            pg.draw.circle(surf, (*HOVER_COLOR[:3], 100),
                           (STONE_RADIUS, STONE_RADIUS), STONE_RADIUS)
            screen.blit(surf, (px - STONE_RADIUS, py - STONE_RADIUS))

        # Info panel
        info_x = BOARD_PX + 15
        if info:
            y = 20
            for key, val in info.items():
                if key == "_title":
                    text = self.font_large.render(str(val), True, (30, 30, 30))
                else:
                    text = self.font.render(f"{key}: {val}", True, (50, 50, 50))
                screen.blit(text, (info_x, y))
                y += 28

        # Winner banner
        if board.winner != 0:
            if board.winner == -1:
                msg = "DRAW"
            else:
                name = "BLACK" if board.winner == BLACK else "WHITE"
                msg = f"{name} WINS"
            text = self.font_large.render(msg, True, (200, 30, 30))
            rect = text.get_rect(center=(BOARD_PX // 2, WINDOW_HEIGHT // 2))
            bg_rect = rect.inflate(40, 20)
            pg.draw.rect(screen, (255, 255, 255), bg_rect)
            pg.draw.rect(screen, (200, 30, 30), bg_rect, 3)
            screen.blit(text, rect)

        pg.display.flip()

    def save_frame(self, path: str):
        self.pygame.image.save(self.screen, path)

    def get_human_move(self, board: Board) -> Optional[tuple[int, int]]:
        """Wait for a human click and return (row, col), or None if window closed."""
        pg = self.pygame
        hover = None
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return None
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    pos = _pixel_to_grid(*event.pos)
                    if pos and board.is_legal(*pos):
                        return pos
                if event.type == pg.MOUSEMOTION:
                    hover = _pixel_to_grid(*event.pos)

            self.draw_board(board, hover=hover)
            self.clock.tick(30)

    def close(self):
        self.pygame.quit()


# ---------------------------------------------------------------------------
# Headless batch game runner (for training self-play)
# ---------------------------------------------------------------------------

class BatchBoards:
    """
    Manage N parallel games for batched self-play.
    Pure numpy, no pygame — maximum speed.
    """

    def __init__(self, n: int):
        self.n = n
        self.grids = np.zeros((n, BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_players = np.full(n, BLACK, dtype=np.int8)
        self.move_counts = np.zeros(n, dtype=np.int32)
        self.winners = np.zeros(n, dtype=np.int8)  # 0=ongoing, 1=BLACK, 2=WHITE, -1=draw
        # Per-game history for training data collection
        self.histories: list[list[tuple[np.ndarray, int, int]]] = [[] for _ in range(n)]
        # (encoded_board, action_index, player)

    def get_legal_masks(self) -> np.ndarray:
        """Return [N, 225] float32 mask of legal moves for all games."""
        masks = (self.grids.reshape(self.n, -1) == EMPTY).astype(np.float32)
        # Zero out masks for finished games
        finished = self.winners != 0
        masks[finished] = 0.0
        return masks

    def encode_all(self) -> np.ndarray:
        """Encode all boards for NN input. Returns [N, 3, 15, 15]."""
        out = np.zeros((self.n, 3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for i in range(self.n):
            cur = self.current_players[i]
            opp = WHITE if cur == BLACK else BLACK
            out[i, 0] = (self.grids[i] == cur).astype(np.float32)
            out[i, 1] = (self.grids[i] == opp).astype(np.float32)
            if cur == BLACK:
                out[i, 2] = 1.0
        return out

    def step(self, actions: np.ndarray):
        """
        Apply one move per game.
        actions: [N] int array, each in [0, 225), representing row*15+col.
        Records history for training data.
        """
        for i in range(self.n):
            if self.winners[i] != 0:
                continue
            action = int(actions[i])
            row, col = divmod(action, BOARD_SIZE)
            player = self.current_players[i]

            # Record state before the move
            encoded = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
            cur = player
            opp = WHITE if cur == BLACK else BLACK
            encoded[0] = (self.grids[i] == cur).astype(np.float32)
            encoded[1] = (self.grids[i] == opp).astype(np.float32)
            if cur == BLACK:
                encoded[2] = 1.0
            self.histories[i].append((encoded, action, player))

            # Place stone
            self.grids[i, row, col] = player
            self.move_counts[i] += 1

            # Check win
            if self._check_win_single(i, row, col, player):
                self.winners[i] = player
            elif self.move_counts[i] >= BOARD_SIZE * BOARD_SIZE:
                self.winners[i] = -1

            # Switch player
            self.current_players[i] = WHITE if player == BLACK else BLACK

    def _check_win_single(self, idx: int, row: int, col: int, player: int) -> bool:
        grid = self.grids[idx]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                r, c = row + dr * sign, col + dc * sign
                while (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE
                       and grid[r, c] == player):
                    count += 1
                    r += dr * sign
                    c += dc * sign
            if count >= WIN_LENGTH:
                return True
        return False

    def get_finished_mask(self) -> np.ndarray:
        """Return [N] bool mask of finished games."""
        return self.winners != 0

    def collect_training_data(self, game_idx: int) -> list[tuple[np.ndarray, int, float]]:
        """
        Collect (board_encoding, action, outcome) for a finished game.
        outcome: +1 if the player who made this move won, -1 if lost, 0 if draw.
        """
        winner = self.winners[game_idx]
        data = []
        for encoded, action, player in self.histories[game_idx]:
            if winner == -1:
                outcome = 0.0
            elif winner == player:
                outcome = 1.0
            else:
                outcome = -1.0
            data.append((encoded, action, outcome))
        return data

    def reset_game(self, idx: int):
        """Reset a single finished game to start a new one."""
        self.grids[idx] = 0
        self.current_players[idx] = BLACK
        self.move_counts[idx] = 0
        self.winners[idx] = 0
        self.histories[idx] = []


if __name__ == "__main__":
    # Quick test: two-player human game
    board = Board()
    renderer = Renderer(title="MAG Gomoku - Two Player")

    info = {"_title": "MAG Gomoku", "Mode": "Two Player", "Turn": "BLACK"}

    while not board.is_terminal():
        renderer.draw_board(board, info=info)
        move = renderer.get_human_move(board)
        if move is None:
            break
        board.place(*move)
        info["Turn"] = "BLACK" if board.current_player == BLACK else "WHITE"
        info["Move"] = board.move_count

    if board.winner:
        renderer.draw_board(board, info=info)
        import pygame
        pygame.time.wait(3000)

    renderer.close()
