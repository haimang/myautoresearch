#!/bin/bash
# Build native C extensions for the Gomoku domain.
# Usage: cd domains/gomoku && bash build_native.sh

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$DIR/minimax_c.c"

if [[ "$(uname)" == "Darwin" ]]; then
    OUT="$DIR/minimax_c.dylib"
    clang -shared -O3 -fPIC -march=native -o "$OUT" "$SRC" -lm
else
    OUT="$DIR/minimax_c.so"
    gcc -shared -O3 -fPIC -march=native -o "$OUT" "$SRC" -lm
fi

echo "Built: $OUT"
