/*
 * mcts_c.c — Native C MCTS tree operations for autoresearch.
 *
 * Pre-allocated node pool, vectorized PUCT select, zero-malloc search.
 * Python calls via ctypes; board ops + GPU eval stay in Python.
 *
 * Build:
 *   macOS:  clang -shared -O3 -fPIC -o mcts_c.dylib mcts_c.c -lm
 *   Linux:  gcc   -shared -O3 -fPIC -o mcts_c.so    mcts_c.c -lm
 */

#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ── Constants ────────────────────────────────────────────────────── */

#define MAX_ACTIONS   225       /* 15x15 Gomoku board */
#define MAX_NODES     500000    /* enough for 800 sims × N trees × depth */
#define NULL_NODE     (-1)

/* ── Node structure ───────────────────────────────────────────────── */

typedef struct {
    int   parent;              /* index in pool, -1 = root */
    int   parent_child_idx;    /* index in parent's child arrays */
    int   action;              /* move that led here */
    float prior;               /* P(s,a) from network */
    int   visit_count;
    float value_sum;
    int   is_expanded;
    int   n_children;
    int   child_actions[MAX_ACTIONS];
    float child_priors[MAX_ACTIONS];
    int   child_visits[MAX_ACTIONS];
    float child_values[MAX_ACTIONS];
    int   child_nodes[MAX_ACTIONS];  /* pool index, -1 = not yet created */
} MCTSNode;

/* ── Pool ─────────────────────────────────────────────────────────── */

static MCTSNode g_pool[MAX_NODES];
static int      g_pool_next = 0;

static int pool_alloc(void) {
    if (g_pool_next >= MAX_NODES) return NULL_NODE;
    int idx = g_pool_next++;
    MCTSNode *n = &g_pool[idx];
    n->parent = NULL_NODE;
    n->parent_child_idx = -1;
    n->action = -1;
    n->prior = 0.0f;
    n->visit_count = 0;
    n->value_sum = 0.0f;
    n->is_expanded = 0;
    n->n_children = 0;
    return idx;
}

/* ── Exported: reset pool ─────────────────────────────────────────── */

void mcts_pool_reset(void) {
    g_pool_next = 0;
}

int mcts_pool_usage(void) {
    return g_pool_next;
}

/* ── Exported: create root, expand with priors ────────────────────── */

int mcts_create_root(void) {
    return pool_alloc();
}

void mcts_expand(int node_idx,
                 const float *priors,     /* [action_size] */
                 const float *legal_mask, /* [action_size] */
                 int action_size)
{
    if (node_idx < 0 || node_idx >= g_pool_next) return;
    MCTSNode *node = &g_pool[node_idx];
    if (node->is_expanded) return;

    /* Mask and normalize priors */
    float masked[MAX_ACTIONS];
    float total = 0.0f;
    for (int i = 0; i < action_size && i < MAX_ACTIONS; i++) {
        masked[i] = priors[i] * legal_mask[i];
        total += masked[i];
    }
    if (total > 1e-8f) {
        float inv = 1.0f / total;
        for (int i = 0; i < action_size; i++) masked[i] *= inv;
    } else {
        /* Fallback: uniform over legal */
        total = 0.0f;
        for (int i = 0; i < action_size; i++) {
            masked[i] = legal_mask[i];
            total += masked[i];
        }
        if (total > 1e-8f) {
            float inv = 1.0f / total;
            for (int i = 0; i < action_size; i++) masked[i] *= inv;
        }
    }

    /* Build child arrays — only legal actions */
    int nc = 0;
    for (int i = 0; i < action_size && i < MAX_ACTIONS; i++) {
        if (legal_mask[i] > 0.5f) {
            node->child_actions[nc] = i;
            node->child_priors[nc] = masked[i];
            node->child_visits[nc] = 0;
            node->child_values[nc] = 0.0f;
            node->child_nodes[nc]  = NULL_NODE;
            nc++;
        }
    }
    node->n_children = nc;
    node->is_expanded = 1;
}

/* ── Exported: add Dirichlet noise to root ────────────────────────── */

void mcts_add_dirichlet(int node_idx, const float *noise, int n,
                        float frac)
{
    if (node_idx < 0 || node_idx >= g_pool_next) return;
    MCTSNode *node = &g_pool[node_idx];
    int nc = node->n_children;
    if (nc <= 0 || n < nc) return;
    for (int i = 0; i < nc; i++) {
        node->child_priors[i] = (1.0f - frac) * node->child_priors[i]
                                + frac * noise[i];
    }
}

void mcts_set_root_value(int node_idx, float value) {
    if (node_idx < 0 || node_idx >= g_pool_next) return;
    g_pool[node_idx].visit_count = 1;
    g_pool[node_idx].value_sum = value;
}

/* ── Exported: select child (vectorized PUCT) ─────────────────────── */

int mcts_select_child(int node_idx, float c_puct) {
    if (node_idx < 0 || node_idx >= g_pool_next) return NULL_NODE;
    MCTSNode *node = &g_pool[node_idx];
    int nc = node->n_children;
    if (nc <= 0) return NULL_NODE;

    float sqrt_parent = sqrtf((float)node->visit_count);
    float best_score = -1e30f;
    int best_idx = 0;

    for (int i = 0; i < nc; i++) {
        float q = 0.0f;
        if (node->child_visits[i] > 0)
            q = node->child_values[i] / (float)node->child_visits[i];
        float exploration = c_puct * node->child_priors[i] * sqrt_parent
                            / (1.0f + (float)node->child_visits[i]);
        float score = q + exploration;
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    /* Lazily create child node if needed */
    if (node->child_nodes[best_idx] == NULL_NODE) {
        int child = pool_alloc();
        if (child == NULL_NODE) return NULL_NODE;
        g_pool[child].parent = node_idx;
        g_pool[child].parent_child_idx = best_idx;
        g_pool[child].action = node->child_actions[best_idx];
        g_pool[child].prior = node->child_priors[best_idx];
        node->child_nodes[best_idx] = child;
    }
    return node->child_nodes[best_idx];
}

/* ── Internal: sync node stats to parent's arrays ─────────────────── */

static void sync_to_parent(int node_idx) {
    MCTSNode *node = &g_pool[node_idx];
    if (node->parent != NULL_NODE && node->parent_child_idx >= 0) {
        MCTSNode *par = &g_pool[node->parent];
        par->child_visits[node->parent_child_idx] = node->visit_count;
        par->child_values[node->parent_child_idx] = node->value_sum;
    }
}

/* ── Exported: backup ─────────────────────────────────────────────── */

void mcts_backup(int node_idx, float value) {
    float v = value;
    int cur = node_idx;
    while (cur != NULL_NODE) {
        g_pool[cur].visit_count += 1;
        g_pool[cur].value_sum += v;
        sync_to_parent(cur);
        v = -v;
        cur = g_pool[cur].parent;
    }
}

/* ── Exported: virtual loss ───────────────────────────────────────── */

void mcts_apply_virtual_loss(int node_idx, float vl) {
    if (node_idx < 0 || node_idx >= g_pool_next) return;
    g_pool[node_idx].visit_count += 1;
    g_pool[node_idx].value_sum -= vl;
    sync_to_parent(node_idx);
}

void mcts_revert_virtual_loss(int node_idx, float vl) {
    if (node_idx < 0 || node_idx >= g_pool_next) return;
    g_pool[node_idx].visit_count -= 1;
    g_pool[node_idx].value_sum += vl;
    sync_to_parent(node_idx);
}

/* ── Exported: get action of a node ───────────────────────────────── */

int mcts_node_action(int node_idx) {
    if (node_idx < 0 || node_idx >= g_pool_next) return -1;
    return g_pool[node_idx].action;
}

int mcts_node_is_expanded(int node_idx) {
    if (node_idx < 0 || node_idx >= g_pool_next) return 0;
    return g_pool[node_idx].is_expanded;
}

/* ── Exported: extract visit distribution from root ───────────────── */

void mcts_get_visits(int node_idx, float *visits_out, int action_size) {
    memset(visits_out, 0, sizeof(float) * action_size);
    if (node_idx < 0 || node_idx >= g_pool_next) return;
    MCTSNode *node = &g_pool[node_idx];
    for (int i = 0; i < node->n_children; i++) {
        int a = node->child_actions[i];
        if (a >= 0 && a < action_size)
            visits_out[a] = (float)node->child_visits[i];
    }
}

/* ── Exported: select full path from root to leaf ─────────────────── */

int mcts_select_path(int root_idx, float c_puct, float virtual_loss,
                     int *path_nodes_out, int *path_actions_out,
                     int max_depth)
{
    int cur = root_idx;
    int depth = 0;
    path_nodes_out[0] = cur;
    path_actions_out[0] = -1;
    depth = 1;

    while (depth < max_depth && cur != NULL_NODE && g_pool[cur].is_expanded) {
        mcts_apply_virtual_loss(cur, virtual_loss);
        int child = mcts_select_child(cur, c_puct);
        if (child == NULL_NODE) break;
        path_nodes_out[depth] = child;
        path_actions_out[depth] = g_pool[child].action;
        cur = child;
        depth++;
    }
    if (cur != NULL_NODE && depth > 0)
        mcts_apply_virtual_loss(cur, virtual_loss);
    return depth;
}

void mcts_revert_path_virtual_loss(const int *path_nodes, int path_len,
                                   float virtual_loss)
{
    for (int i = 0; i < path_len; i++)
        mcts_revert_virtual_loss(path_nodes[i], virtual_loss);
}

/* ── Exported: BATCH select — K sims × N roots in ONE C call ──────── *
 *
 * Runs K*N select paths without returning to Python between them.
 * Output: flat arrays of all paths, plus per-path metadata.
 *
 * Returns total number of paths written.
 */
/*
 * Path batch buffers.
 *
 * IMPORTANT: MAX_BATCH_PATHS is the hard cap on (sims_per_round * n_roots)
 * per call. Before v14.1 this was 256, which SILENTLY TRUNCATED any larger
 * request — the outer loop guard `total < MAX_BATCH_PATHS` just stopped
 * iterating without reporting. At pg=64 / batch=16 that was 1024 requested
 * but only 256 served (75% work loss), which completely hid GPU saturation.
 *
 * New value 2048 covers realistic configurations:
 *   pg=32  / batch=64 = 2048 ✓
 *   pg=64  / batch=32 = 2048 ✓
 *   pg=128 / batch=16 = 2048 ✓
 *
 * Memory cost: 2048 * (128*2 + 1 + 1) * 4 bytes ≈ 2.1 MB static — negligible.
 *
 * A caller requesting more than MAX_BATCH_PATHS is still truncated, but the
 * Python wrapper in mcts_native.py now detects and warns on that case.
 */
#define MAX_BATCH_PATHS 2048  /* was 256 — see comment above */
#define MAX_PATH_DEPTH  128

/* Flat output buffers (caller reads after call) */
static int   g_batch_path_nodes[MAX_BATCH_PATHS * MAX_PATH_DEPTH];
static int   g_batch_path_actions[MAX_BATCH_PATHS * MAX_PATH_DEPTH];
static int   g_batch_path_lens[MAX_BATCH_PATHS];
static int   g_batch_leaf_nodes[MAX_BATCH_PATHS];   /* leaf node index */

/* Exported so Python can query the compile-time cap at runtime. */
int mcts_max_batch_paths(void) { return MAX_BATCH_PATHS; }

int mcts_batch_select(const int *root_indices, int n_roots,
                      int k_sims, float c_puct, float virtual_loss)
{
    int total = 0;
    for (int sim = 0; sim < k_sims; sim++) {
        for (int ri = 0; ri < n_roots && total < MAX_BATCH_PATHS; ri++) {
            int *pn = &g_batch_path_nodes[total * MAX_PATH_DEPTH];
            int *pa = &g_batch_path_actions[total * MAX_PATH_DEPTH];

            int len = mcts_select_path(root_indices[ri], c_puct, virtual_loss,
                                       pn, pa, MAX_PATH_DEPTH);
            g_batch_path_lens[total] = len;
            g_batch_leaf_nodes[total] = pn[len - 1];
            total++;
        }
    }
    return total;
}

/* Accessor functions for batch results (called from Python) */
int* mcts_batch_get_path_actions(void) { return g_batch_path_actions; }
int* mcts_batch_get_path_nodes(void)   { return g_batch_path_nodes; }
int* mcts_batch_get_path_lens(void)    { return g_batch_path_lens; }
int* mcts_batch_get_leaf_nodes(void)   { return g_batch_leaf_nodes; }

/* ── Exported: BATCH expand+backup — process all leaves in ONE call ── */

void mcts_batch_expand_backup(int n_paths,
                              const int *leaf_indices,   /* which paths to expand (indices into batch) */
                              int n_expand,
                              const float *all_priors,   /* [n_expand][action_size] flat */
                              const float *all_masks,    /* [n_expand][action_size] flat */
                              const float *all_values,   /* [n_expand] values (already negated) */
                              int action_size,
                              const int *term_indices,   /* which paths are terminal */
                              int n_terminal,
                              const float *term_values,  /* terminal values */
                              float virtual_loss)
{
    /* Expand + backup non-terminal leaves */
    for (int i = 0; i < n_expand; i++) {
        int pi = leaf_indices[i];  /* path index in batch */
        int leaf = g_batch_leaf_nodes[pi];
        int path_len = g_batch_path_lens[pi];
        int *pn = &g_batch_path_nodes[pi * MAX_PATH_DEPTH];

        const float *priors = &all_priors[i * action_size];
        const float *mask = &all_masks[i * action_size];

        mcts_expand(leaf, priors, mask, action_size);

        /* Revert virtual loss */
        for (int j = 0; j < path_len; j++)
            mcts_revert_virtual_loss(pn[j], virtual_loss);

        mcts_backup(leaf, all_values[i]);
    }

    /* Backup terminal paths */
    for (int i = 0; i < n_terminal; i++) {
        int pi = term_indices[i];
        int leaf = g_batch_leaf_nodes[pi];
        int path_len = g_batch_path_lens[pi];
        int *pn = &g_batch_path_nodes[pi * MAX_PATH_DEPTH];

        for (int j = 0; j < path_len; j++)
            mcts_revert_virtual_loss(pn[j], virtual_loss);

        mcts_backup(leaf, term_values[i]);
    }
}
