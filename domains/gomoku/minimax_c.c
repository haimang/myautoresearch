/*
 * gomoku/minimax_c.c — native Gomoku minimax (pattern scoring + alpha-beta)
 *
 * Replaces the pure-Python minimax in domains/gomoku/prepare.py. Single
 * entry point `gomoku_root_scores` returns a (row, col, score) array for
 * all root candidates; Python does the top-k softmax sampling on top.
 *
 * Design decisions:
 *   - Single file, domain-specific. Future domains (chess, go) would get
 *     their own *_minimax_c.c. Alpha-beta is short enough to duplicate;
 *     the expensive part (pattern scoring) is domain-specific anyway.
 *   - No transposition table / iterative deepening in v15 — v16 material.
 *   - All state on the C stack / fixed buffers; zero malloc per call.
 *
 * Build:
 *   macOS:  clang -shared -O3 -fPIC -o minimax_c.dylib minimax_c.c -lm
 *   Linux:  gcc   -shared -O3 -fPIC -o minimax_c.so    minimax_c.c -lm
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define BOARD_SIZE     15
#define BOARD_CELLS    (BOARD_SIZE * BOARD_SIZE)
#define WIN_LENGTH     5
#define EMPTY_CELL     0
#define BLACK_STONE    1
#define WHITE_STONE    2
#define MAX_CANDIDATES 225
#define MAX_DEPTH      8

#define SCORE_WIN      100000.0f

/* ── Pattern scores — mirror prepare.py::_PATTERN_SCORES ───────────────── */

static int pattern_score(int count, int open_ends) {
    if (count >= 5) return 100000;
    if (open_ends == 0) return 0;
    switch (count) {
        case 4: return (open_ends == 2) ? 10000 : 1000;
        case 3: return (open_ends == 2) ? 1000 : 100;
        case 2: return (open_ends == 2) ? 100 : 10;
        case 1: return (open_ends == 2) ? 10 : 1;
        default: return 0;
    }
}

/* ── Segment scoring (hot inner loop of evaluate_position) ─────────────── */

static inline int in_bounds(int r, int c) {
    return r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE;
}

static int score_segment(const int8_t *grid, int row, int col,
                         int dr, int dc, int player) {
    int count = 1;
    int r, c;

    r = row + dr; c = col + dc;
    while (in_bounds(r, c) && grid[r * BOARD_SIZE + c] == player) {
        count++; r += dr; c += dc;
    }
    int open_pos = in_bounds(r, c) && grid[r * BOARD_SIZE + c] == EMPTY_CELL;

    r = row - dr; c = col - dc;
    while (in_bounds(r, c) && grid[r * BOARD_SIZE + c] == player) {
        count++; r -= dr; c -= dc;
    }
    int open_neg = in_bounds(r, c) && grid[r * BOARD_SIZE + c] == EMPTY_CELL;

    return pattern_score(count, open_pos + open_neg);
}

/* ── Full-board static evaluation ───────────────────────────────────────── */

static const int DIRECTIONS[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};

static int evaluate_position(const int8_t *grid, int player) {
    int opponent = (player == BLACK_STONE) ? WHITE_STONE : BLACK_STONE;
    int player_score = 0, opp_score = 0;

    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            int stone = grid[r * BOARD_SIZE + c];
            if (stone == EMPTY_CELL) continue;

            for (int d = 0; d < 4; d++) {
                int dr = DIRECTIONS[d][0], dc = DIRECTIONS[d][1];
                int pr = r - dr, pc = c - dc;
                /* Only score from the start of a run */
                if (in_bounds(pr, pc) && grid[pr * BOARD_SIZE + pc] == stone)
                    continue;

                int s = score_segment(grid, r, c, dr, dc, stone);
                if (stone == player) player_score += s;
                else                 opp_score += s;
            }
        }
    }
    return player_score - opp_score;
}

/* ── Win detection at the just-placed stone ─────────────────────────────── */

static int check_win_fast(const int8_t *grid, int row, int col, int player) {
    for (int d = 0; d < 4; d++) {
        int dr = DIRECTIONS[d][0], dc = DIRECTIONS[d][1];
        int count = 1;
        int r = row + dr, c = col + dc;
        while (in_bounds(r, c) && grid[r * BOARD_SIZE + c] == player) {
            count++; r += dr; c += dc;
        }
        r = row - dr; c = col - dc;
        while (in_bounds(r, c) && grid[r * BOARD_SIZE + c] == player) {
            count++; r -= dr; c -= dc;
        }
        if (count >= WIN_LENGTH) return 1;
    }
    return 0;
}

/* ── Candidate generation (radius = 2 around occupied cells) ────────────── */

static int make_candidates(const int8_t *grid, int radius,
                           int *out_rows, int *out_cols, int max_out) {
    /* 1D bitmap on the board to dedupe */
    int8_t seen[BOARD_CELLS];
    memset(seen, 0, BOARD_CELLS);

    int n = 0;
    int any_stone = 0;
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            if (grid[r * BOARD_SIZE + c] == EMPTY_CELL) continue;
            any_stone = 1;
            for (int dr = -radius; dr <= radius; dr++) {
                for (int dc = -radius; dc <= radius; dc++) {
                    int nr = r + dr, nc = c + dc;
                    if (!in_bounds(nr, nc)) continue;
                    int idx = nr * BOARD_SIZE + nc;
                    if (grid[idx] != EMPTY_CELL) continue;
                    if (seen[idx]) continue;
                    seen[idx] = 1;
                    if (n < max_out) {
                        out_rows[n] = nr;
                        out_cols[n] = nc;
                        n++;
                    }
                }
            }
        }
    }
    if (!any_stone) {
        /* Empty board: only the center */
        out_rows[0] = BOARD_SIZE / 2;
        out_cols[0] = BOARD_SIZE / 2;
        return 1;
    }
    return n;
}

/* ── Move ordering ──────────────────────────────────────────────────────── */

/* order=0 : center-distance (stable). Used by L1. */
/* order=1 : adjacency heuristic — count occupied neighbors + center bonus. L2. */
/* order=2 : killer-move (not implemented in v15; falls back to heuristic). L3. */

typedef struct { int row, col; int key; } OrderCand;

static int order_cmp_key_asc(const void *a, const void *b) {
    const OrderCand *x = (const OrderCand *)a;
    const OrderCand *y = (const OrderCand *)b;
    return x->key - y->key;
}

static void order_by_center(int *rows, int *cols, int n) {
    OrderCand buf[MAX_CANDIDATES];
    for (int i = 0; i < n; i++) {
        int dr = rows[i] - BOARD_SIZE / 2;
        int dc = cols[i] - BOARD_SIZE / 2;
        if (dr < 0) dr = -dr;
        if (dc < 0) dc = -dc;
        buf[i].row = rows[i];
        buf[i].col = cols[i];
        buf[i].key = dr + dc;  /* Manhattan distance */
    }
    qsort(buf, n, sizeof(OrderCand), order_cmp_key_asc);
    for (int i = 0; i < n; i++) {
        rows[i] = buf[i].row;
        cols[i] = buf[i].col;
    }
}

static void order_by_heuristic(const int8_t *grid, int *rows, int *cols, int n) {
    OrderCand buf[MAX_CANDIDATES];
    for (int i = 0; i < n; i++) {
        int r = rows[i], c = cols[i];
        int adj = 0;
        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                if (dr == 0 && dc == 0) continue;
                int nr = r + dr, nc = c + dc;
                if (in_bounds(nr, nc) && grid[nr * BOARD_SIZE + nc] != EMPTY_CELL)
                    adj++;
            }
        }
        int center_bonus = 7 - ((r - 7 < 0 ? 7 - r : r - 7) + (c - 7 < 0 ? 7 - c : c - 7));
        if (center_bonus < 0) center_bonus = 0;
        buf[i].row = r;
        buf[i].col = c;
        buf[i].key = -(adj * 10 + center_bonus);  /* Negate so lower = better */
    }
    qsort(buf, n, sizeof(OrderCand), order_cmp_key_asc);
    for (int i = 0; i < n; i++) {
        rows[i] = buf[i].row;
        cols[i] = buf[i].col;
    }
}

/* v15 C7 threat-aware move ordering.
 *
 * For each candidate (r, c), evaluate the POTENTIAL threat magnitude if
 * `player` played there AND the opponent threat being blocked. This is the
 * single biggest alpha-beta pruning improvement for Gomoku — moves that
 * create/block fours and threes bubble to the front, causing massive cutoffs
 * at deep plies. Adds ~n×constant overhead per call but reduces total
 * branching factor by 3-5×, which at depth 6 is the difference between
 * "unusable" and "usable".
 */
static int local_threat_score(const int8_t *grid, int row, int col, int player) {
    int opponent = (player == BLACK_STONE) ? WHITE_STONE : BLACK_STONE;

    /* Score all four directions through (row, col) as if `player` had just
     * played there. The grid is NOT mutated — we do a virtual scan.
     * For each direction, count consecutive stones of `player` through
     * (row, col) including the virtual stone, plus check open ends.
     */
    int max_own = 0;
    int max_block = 0;
    for (int d = 0; d < 4; d++) {
        int dr = DIRECTIONS[d][0], dc = DIRECTIONS[d][1];

        /* Own direction: pretend stone is `player`. */
        int count = 1;
        int r = row + dr, c = col + dc;
        while (in_bounds(r, c) && grid[r * BOARD_SIZE + c] == player) {
            count++; r += dr; c += dc;
        }
        int open_pos = in_bounds(r, c) && grid[r * BOARD_SIZE + c] == EMPTY_CELL;
        r = row - dr; c = col - dc;
        while (in_bounds(r, c) && grid[r * BOARD_SIZE + c] == player) {
            count++; r -= dr; c -= dc;
        }
        int open_neg = in_bounds(r, c) && grid[r * BOARD_SIZE + c] == EMPTY_CELL;
        int own_s = pattern_score(count, open_pos + open_neg);
        if (own_s > max_own) max_own = own_s;

        /* Block direction: what if opponent played here? Count opponent runs. */
        count = 1;
        r = row + dr; c = col + dc;
        while (in_bounds(r, c) && grid[r * BOARD_SIZE + c] == opponent) {
            count++; r += dr; c += dc;
        }
        int op_pos = in_bounds(r, c) && grid[r * BOARD_SIZE + c] == EMPTY_CELL;
        r = row - dr; c = col - dc;
        while (in_bounds(r, c) && grid[r * BOARD_SIZE + c] == opponent) {
            count++; r -= dr; c -= dc;
        }
        int op_neg = in_bounds(r, c) && grid[r * BOARD_SIZE + c] == EMPTY_CELL;
        int blk_s = pattern_score(count, op_pos + op_neg);
        if (blk_s > max_block) max_block = blk_s;
    }

    /* Return combined priority — own offense + slightly less opponent defense. */
    return max_own * 2 + max_block;
}

static void order_by_threat(const int8_t *grid, int *rows, int *cols,
                             int n, int player) {
    OrderCand buf[MAX_CANDIDATES];
    for (int i = 0; i < n; i++) {
        int r = rows[i], c = cols[i];
        int s = local_threat_score(grid, r, c, player);
        buf[i].row = r;
        buf[i].col = c;
        buf[i].key = -s;  /* qsort ascending → negate so higher threat goes first */
    }
    qsort(buf, n, sizeof(OrderCand), order_cmp_key_asc);
    for (int i = 0; i < n; i++) {
        rows[i] = buf[i].row;
        cols[i] = buf[i].col;
    }
}

/* ── Alpha-beta recursive core ──────────────────────────────────────────── */

/* Per-depth candidate cap. Gomoku has a heavy branching factor (~40-60
 * candidates with radius=2). For depths ≥ 4, exploring only the top-N
 * threat-ordered candidates drops the tree from 50^6 to manageable levels
 * while keeping strong tactical play, because threat ordering bubbles the
 * truly relevant moves to the front. */
static int candidate_cap_for_depth(int depth) {
    if (depth >= 5) return 8;   /* L3 tail plies */
    if (depth >= 3) return 12;  /* L2 mid plies, L3 near root */
    if (depth >= 2) return 16;
    return MAX_CANDIDATES;
}

static float minimax_ab(int8_t *grid, int depth, float alpha, float beta,
                        int maximizing, int player, int opponent,
                        int move_order) {
    if (depth == 0) {
        return (float)evaluate_position(grid, player);
    }

    int rows[MAX_CANDIDATES], cols[MAX_CANDIDATES];
    int n = make_candidates(grid, 2, rows, cols, MAX_CANDIDATES);
    if (n == 0) {
        return (float)evaluate_position(grid, player);
    }

    /* Threat-aware ordering for depth >= 2 (where it matters). For depth=1
     * (leaf ply) we skip ordering because all children are evaluated anyway.
     * For move_order==0 (L1 basic), keep center ordering for parity with
     * the Python baseline. */
    int current_side = maximizing ? player : opponent;
    if (move_order >= 1 && depth >= 2) {
        order_by_threat(grid, rows, cols, n, current_side);
    } else {
        order_by_center(rows, cols, n);
    }

    /* Candidate cap: keep only the top-N threat-ordered moves at deep plies */
    int cap = candidate_cap_for_depth(depth);
    if (n > cap) n = cap;

    if (maximizing) {
        float max_eval = -INFINITY;
        for (int i = 0; i < n; i++) {
            int r = rows[i], c = cols[i];
            int idx = r * BOARD_SIZE + c;
            grid[idx] = (int8_t)player;
            if (check_win_fast(grid, r, c, player)) {
                grid[idx] = EMPTY_CELL;
                return SCORE_WIN + depth;
            }
            float v = minimax_ab(grid, depth - 1, alpha, beta,
                                 0, player, opponent, move_order);
            grid[idx] = EMPTY_CELL;
            if (v > max_eval) max_eval = v;
            if (v > alpha) alpha = v;
            if (beta <= alpha) break;
        }
        return max_eval;
    } else {
        float min_eval = INFINITY;
        for (int i = 0; i < n; i++) {
            int r = rows[i], c = cols[i];
            int idx = r * BOARD_SIZE + c;
            grid[idx] = (int8_t)opponent;
            if (check_win_fast(grid, r, c, opponent)) {
                grid[idx] = EMPTY_CELL;
                return -SCORE_WIN - depth;
            }
            float v = minimax_ab(grid, depth - 1, alpha, beta,
                                 1, player, opponent, move_order);
            grid[idx] = EMPTY_CELL;
            if (v < min_eval) min_eval = v;
            if (v < beta) beta = v;
            if (beta <= alpha) break;
        }
        return min_eval;
    }
}

/* ── Root-level scoring: the Python-facing entry point ─────────────────── */

/* Output layout: flat [n_moves * 3] of (row, col, score_as_int_x1000).
 * We write int-coded scores because ctypes.c_float arrays are fine too, but
 * this one is simpler — Python converts back with np.frombuffer + reshape.
 *
 * Actually: simpler still — write rows, cols, scores into three separate
 * pre-allocated buffers. Python passes them in.
 */
int gomoku_root_scores(const int8_t *grid_in,
                        int player,
                        int depth,
                        int move_order,
                        int *out_rows,
                        int *out_cols,
                        float *out_scores,
                        int max_out) {
    /* Copy grid to local mutable buffer (we modify during search) */
    int8_t grid[BOARD_CELLS];
    memcpy(grid, grid_in, BOARD_CELLS);

    int opponent = (player == BLACK_STONE) ? WHITE_STONE : BLACK_STONE;

    int rows[MAX_CANDIDATES], cols[MAX_CANDIDATES];
    int n = make_candidates(grid, 2, rows, cols, MAX_CANDIDATES);
    if (n == 0) return 0;

    /* Root ordering: use threat-aware if move_order >= 1. The root scans
     * every candidate regardless of cap — Python wants full score array
     * for top-k sampling. */
    if (move_order >= 1) {
        order_by_threat(grid, rows, cols, n, player);
    } else {
        order_by_center(rows, cols, n);
    }

    int written = 0;
    for (int i = 0; i < n && written < max_out; i++) {
        int r = rows[i], c = cols[i];
        int idx = r * BOARD_SIZE + c;
        grid[idx] = (int8_t)player;

        /* Immediate-win shortcut: assign massive score so Python's
         * win-threshold check forces this move. */
        if (check_win_fast(grid, r, c, player)) {
            grid[idx] = EMPTY_CELL;
            out_rows[written] = r;
            out_cols[written] = c;
            out_scores[written] = 1e9f;
            written++;
            continue;
        }
        /* Regular: one-ply shallower minimax at the opponent's turn */
        float v = minimax_ab(grid, depth - 1,
                             -INFINITY, INFINITY,
                             0, player, opponent, move_order);
        grid[idx] = EMPTY_CELL;
        out_rows[written] = r;
        out_cols[written] = c;
        out_scores[written] = v;
        written++;
    }
    return written;
}

/* ── Exported helper: just a single static eval (for testing) ──────────── */

int gomoku_evaluate_grid(const int8_t *grid, int player) {
    return evaluate_position(grid, player);
}

int gomoku_check_win_at(const int8_t *grid, int row, int col, int player) {
    return check_win_fast(grid, row, col, player);
}
