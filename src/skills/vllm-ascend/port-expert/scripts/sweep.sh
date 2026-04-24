#!/usr/bin/env bash
# Sweep driver for vllm-ascend port-expert Mode Sweep.
#
# Runs kb_drive_test.py across every commit in a range, de-duplicates
# by (kind, symbol), cross-checks against the KB case registry, and
# reports the novel findings that need work.
#
# Usage:
#   sweep.sh --commit-range <A..B> \
#            --vllm-path <community-vllm> \
#            --vllm-ascend-path <vllm-ascend> \
#            [--kb-dir <references/>] \
#            [--out-dir <tmp>]
#
# Exit codes:
#   0 = no novel findings (all drifts already in KB)
#   1 = novel findings exist (see report)
#   2 = usage / harness error
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KB_DRIVE="$SCRIPT_DIR/kb_drive_test.py"
SKILL_REFS="$(cd "$SCRIPT_DIR/.." && pwd)/references"

COMMIT_RANGE=""
VLLM_PATH=""
VLLM_ASCEND_PATH=""
KB_DIR="$SKILL_REFS"
OUT_DIR="/tmp/vllm_ascend_sweep_$(date +%s)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit-range)      COMMIT_RANGE="$2"; shift 2 ;;
    --vllm-path)         VLLM_PATH="$2"; shift 2 ;;
    --vllm-ascend-path)  VLLM_ASCEND_PATH="$2"; shift 2 ;;
    --kb-dir)            KB_DIR="$2"; shift 2 ;;
    --out-dir)           OUT_DIR="$2"; shift 2 ;;
    -h|--help)           grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$COMMIT_RANGE" ]]  || { echo "--commit-range required" >&2; exit 2; }
[[ -n "$VLLM_PATH" ]]     || { echo "--vllm-path required" >&2; exit 2; }
[[ -n "$VLLM_ASCEND_PATH" ]] || { echo "--vllm-ascend-path required" >&2; exit 2; }
[[ -d "$VLLM_PATH" ]]     || { echo "vllm path not found: $VLLM_PATH" >&2; exit 2; }

mkdir -p "$OUT_DIR"
echo "# sweep output -> $OUT_DIR" >&2

SHAS=$(cd "$VLLM_PATH" && git log --format="%h" "$COMMIT_RANGE")
N=$(echo "$SHAS" | wc -l)
echo "# sweeping $N commits" >&2

HIT_FILE="$OUT_DIR/impactful.tsv"
: > "$HIT_FILE"

for sha in $SHAS; do
  SCAN_OUT="$OUT_DIR/scans/$sha"
  python3 "$KB_DRIVE" \
    --vllm-ref "$sha" \
    --vllm-path "$VLLM_PATH" \
    --vllm-ascend-path "$VLLM_ASCEND_PATH" \
    --kb-dir "$KB_DIR" \
    --out "$SCAN_OUT" >/dev/null 2>&1 || true
  SUMMARY="$SCAN_OUT/summary.json"
  [[ -f "$SUMMARY" ]] || continue
  IMPACT=$(python3 -c "import json; print(json.load(open('$SUMMARY'))['impact_ascend'])")
  if [[ "$IMPACT" -gt 0 ]]; then
    python3 -c "
import json
with open('$SUMMARY') as f: d = json.load(f)
for dr in d['drifts']:
  print('\t'.join(['$sha', dr['kind'], dr['symbol'], dr['family'], str(dr['ascend_sites'])]))
" >> "$HIT_FILE"
  fi
done

TOTAL_HITS=$(wc -l < "$HIT_FILE")
UNIQUE_SYMS=$(cut -f3 "$HIT_FILE" | sort -u | wc -l)

KB_INDEX="$KB_DIR/KB_INDEX.md"
NOVEL_FILE="$OUT_DIR/novel.tsv"
: > "$NOVEL_FILE"

cut -f3 "$HIT_FILE" | sort -u | while read -r sym; do
  if [[ -z "$sym" ]]; then continue; fi
  if ! grep -q "\`$sym\`" "$KB_INDEX"; then
    echo "$sym" >> "$NOVEL_FILE"
  fi
done

NOVEL_COUNT=$(wc -l < "$NOVEL_FILE")

echo
echo "=== sweep $COMMIT_RANGE — F1/F2-rename/F3/F5-suspect per-commit ==="
echo "commits scanned:     $N"
echo "impactful rows:      $TOTAL_HITS"
echo "unique symbols:      $UNIQUE_SYMS"
echo "novel (not in KB):   $NOVEL_COUNT"
echo

# Also run tag-range scanners (F4, F7/F8) — extract baseline..target from range
BASE_REF=$(echo "$COMMIT_RANGE" | sed 's/\.\..*//')
TARGET_REF=$(echo "$COMMIT_RANGE" | sed 's/^.*\.\.//')

echo "=== F4 (return-type migration) — check_f4.py ${BASE_REF} -> ${TARGET_REF} ==="
python3 "$SCRIPT_DIR/check_f4.py" \
  --vllm-path "$VLLM_PATH" \
  --vllm-ascend-path "$VLLM_ASCEND_PATH" \
  --baseline-tag "$BASE_REF" \
  --target-tag "$TARGET_REF" \
  --out "$OUT_DIR/f4.json" 2>&1 | tail -5 || true

echo
echo "=== F7/F8 (class-API additions) — check_f7_f8.py ${BASE_REF} -> ${TARGET_REF} ==="
python3 "$SCRIPT_DIR/check_f7_f8.py" \
  --vllm-path "$VLLM_PATH" \
  --vllm-ascend-path "$VLLM_ASCEND_PATH" \
  --baseline-tag "$BASE_REF" \
  --target-tag "$TARGET_REF" \
  --out "$OUT_DIR/f78.json" 2>&1 | tail -15 || true

echo

if [[ "$NOVEL_COUNT" -gt 0 ]]; then
  echo "## Novel symbols (need work, from F1/F2-rename/F3/F5)"
  while read -r sym; do
    first=$(grep -P "^\S+\t\S+\t${sym}\t" "$HIT_FILE" | head -1)
    echo "- $sym   first seen in: $first"
  done < "$NOVEL_FILE"
  echo
  echo "Full hit list:    $HIT_FILE"
  echo "Per-commit scans: $OUT_DIR/scans/"
  echo "F4 output:        $OUT_DIR/f4.json"
  echo "F7/F8 output:     $OUT_DIR/f78.json"
  exit 1
else
  echo "All impactful F1/F2-rename/F3/F5 drifts already registered in KB."
  echo "Check F4/F7/F8 outputs above for class-API additions (may need verification)."
  echo
  echo "F4 output:    $OUT_DIR/f4.json"
  echo "F7/F8 output: $OUT_DIR/f78.json"
  exit 0
fi
