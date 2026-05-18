#!/bin/bash
# tlfix-sweep wrapper.
# Usage: bash run.sh <tilelang_mlir_ascend_dir> <out_dir> [extra args passed to sweep.py]
set -e
TLPATH="${1:?usage: bash run.sh <tilelang_mlir_ascend_dir> <out_dir> [extra args]}"
OUTDIR="${2:?missing out_dir}"
shift 2
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/sweep.py" "$TLPATH" "$OUTDIR" "$@"
