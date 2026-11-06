#!/usr/bin/env bash
# Fixed-work GPU-funnel benchmark: builds a GPU-only binary that runs ITERS
# pipeline iterations (2^28 seeds each) from START, prints the stage table once, exits.
# usage: tools/bench.sh [ITERS] [START] [extra make vars...]
set -euo pipefail
ITERS=${1:-64}; START=${2:-1000000000000}; shift $(( $# < 2 ? $# : 2 ))
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
make -B -s NO_NET=1 MAX_ITERS="$ITERS" PRINT_INTERVAL="$ITERS" "$@" >/dev/null
cp "$(ls -t main-* | head -1)" "$tmp/bench-bin"
"$tmp/bench-bin" --pipelines 1 --start "$START" --output /tmp/commission-bench-out.txt | grep -v -E '^(Initializing|Running)'
