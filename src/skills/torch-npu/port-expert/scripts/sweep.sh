#!/usr/bin/env bash
# Sweep driver for torch-npu/port-expert Mode B (drift scan).
#
# Runs the full 4-scanner sequence against a baseline->target torch tag
# pair. Emits a summary and a per-scanner output file.
#
# Usage:
#   sweep.sh --baseline v2.11.0 --target v2.12.0-rc3 \\
#            --pt-repo <community-pytorch> \\
#            --torch-npu-path <torch-npu-checkout> \\
#            [--out-dir <tmp>]
#
# Exit codes:
#   0 = nothing actionable (all clean or verified)
#   1 = one or more scanners reported potentially-breaking drifts
#   2 = usage / harness error
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASELINE=""
TARGET=""
PT_REPO=""
TORCH_NPU_PATH=""
OUT_DIR="/tmp/torch_npu_sweep_$(date +%s)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline)        BASELINE="$2"; shift 2 ;;
    --target)          TARGET="$2"; shift 2 ;;
    --pt-repo)         PT_REPO="$2"; shift 2 ;;
    --torch-npu-path)  TORCH_NPU_PATH="$2"; shift 2 ;;
    --out-dir)         OUT_DIR="$2"; shift 2 ;;
    -h|--help)         grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$BASELINE" ]]       || { echo "--baseline required" >&2; exit 2; }
[[ -n "$TARGET" ]]         || { echo "--target required" >&2; exit 2; }
[[ -n "$PT_REPO" ]]        || { echo "--pt-repo required" >&2; exit 2; }
[[ -n "$TORCH_NPU_PATH" ]] || { echo "--torch-npu-path required" >&2; exit 2; }
[[ -d "$PT_REPO" ]]        || { echo "pt-repo not found: $PT_REPO" >&2; exit 2; }

mkdir -p "$OUT_DIR"
echo "# sweep output -> $OUT_DIR" >&2
echo "# baseline: $BASELINE" >&2
echo "# target:   $TARGET" >&2
echo

# Step 1: extract imports
echo "=== Step 1: extract torch_npu imports ==="
python3 "$SCRIPT_DIR/extract_imports.py" \
  --root "$TORCH_NPU_PATH/torch_npu" \
  > "$OUT_DIR/imports.txt"
PAIRS=$(grep -c "^  " "$OUT_DIR/imports.txt" || echo 0)
echo "  pairs: $PAIRS"
echo

# Step 2: F1/F2-path-move
echo "=== Step 2: F1/F2-path-move (check_drift.py) ==="
set +e
python3 "$SCRIPT_DIR/check_drift.py" \
  --pt-repo "$PT_REPO" \
  --pairs-file "$OUT_DIR/imports.txt" \
  --baseline-tag "$BASELINE" \
  --target-tag "$TARGET" \
  --out "$OUT_DIR/f1_f2.json" 2>&1 | tail -10
F1_RC=$?
set -e
echo

# Step 3: F3 sig drift
echo "=== Step 3: F3 signature drift (check_sig_drift.py) ==="
set +e
python3 "$SCRIPT_DIR/check_sig_drift.py" \
  --pt-repo "$PT_REPO" \
  --pairs-file "$OUT_DIR/imports.txt" \
  --baseline-tag "$BASELINE" \
  --target-tag "$TARGET" \
  --out "$OUT_DIR/f3.json" 2>&1 | tail -10
F3_RC=$?
set -e
echo

# Step 4: F7/F8 class-API additions
echo "=== Step 4: F7/F8 class-API additions (check_f7_f8.py) ==="
set +e
python3 "$SCRIPT_DIR/check_f7_f8.py" \
  --pt-repo "$PT_REPO" \
  --torch-npu-path "$TORCH_NPU_PATH" \
  --baseline-tag "$BASELINE" \
  --target-tag "$TARGET" \
  --out "$OUT_DIR/f78.json" 2>&1 | tail -15
F78_RC=$?
set -e
echo

echo "=== Summary ==="
echo "F1/F2-path-move exit: $F1_RC"
echo "F3              exit: $F3_RC"
echo "F7/F8           exit: $F78_RC"
echo "Outputs in: $OUT_DIR"
echo

if [[ "$F1_RC" -ne 0 ]] || [[ "$F3_RC" -ne 0 ]] || [[ "$F78_RC" -ne 0 ]]; then
  echo "One or more scanners reported findings. Review per-step output above."
  exit 1
fi
echo "All clean."
exit 0
