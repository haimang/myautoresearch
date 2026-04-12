#!/bin/bash
# Build native C MCTS library for autoresearch.
# Usage: cd framework/core && bash build_native.sh

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$DIR/mcts_c.c"

if [[ "$(uname)" == "Darwin" ]]; then
    OUT="$DIR/mcts_c.dylib"
    clang -shared -O3 -fPIC -o "$OUT" "$SRC" -lm
else
    OUT="$DIR/mcts_c.so"
    gcc -shared -O3 -fPIC -o "$OUT" "$SRC" -lm
fi

echo "Built: $OUT"
