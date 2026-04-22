#!/usr/bin/env bash
# Run a smoke rung on A3 and assert its numerical baseline.
# Called by easyr1-port-worker Phase D per rung.
#
# Usage:
#   smoke_validate.sh --rung <RUNG> --image-tag <IMG> [options]
#
# Required:
#   --rung <V1.1|V1.3|V1.4|V1.5|V2.1|V2.2>
#   --image-tag <TAG>     e.g. easyr1-npu:round3-20260422-xxxx
#
# Optional:
#   --image-family <v1|v2>  default: v1 (which smoke baseline table to use)
#   --chips <LIST>          default auto per rung (V1.4/V2.1: "0,1"; V1.5/V2.2: "0,1,2,3")
#   --a3-host/port/user     as in deploy_to_a3.sh
#   --npu-user <USER>       default: z00637938
#   --log-dir <DIR>         default: /tmp/${NPU_USER}/easyr1-logs/
#   --timeout-min <N>       default: 15
#
# Workflow:
#   1. Check chip availability on A3 (OL-05)
#   2. ssh to A3, run run-npu-container.sh with the smoke script for the rung
#   3. Tee log to {log-dir}/{rung-tag}-{session}.log
#   4. On completion, grep entropy_loss (or equivalent marker per rung)
#   5. Assert numeric in baseline band from references/SMOKE_BASELINE.md
#
# Exit codes:
#   0   PASS (numeric in band)
#   1   FAIL (numeric out of band, or smoke errored)
#   2   usage error
#   5   infra (ssh/docker failed, chip busy)
#   10  log file not found

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

A3_HOST="${A3_HOST:-115.190.166.102}"
A3_PORT="${A3_PORT:-443}"
A3_USER="${A3_USER:-root}"
NPU_USER="${NPU_USER:-z00637938}"
RUNG=""
IMAGE_TAG=""
IMAGE_FAMILY="v1"
CHIPS=""
LOG_DIR=""
TIMEOUT_MIN=15

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rung)         RUNG="$2"; shift 2 ;;
    --image-tag)    IMAGE_TAG="$2"; shift 2 ;;
    --image-family) IMAGE_FAMILY="$2"; shift 2 ;;
    --chips)        CHIPS="$2"; shift 2 ;;
    --a3-host)      A3_HOST="$2"; shift 2 ;;
    --a3-port)      A3_PORT="$2"; shift 2 ;;
    --a3-user)      A3_USER="$2"; shift 2 ;;
    --npu-user)     NPU_USER="$2"; shift 2 ;;
    --log-dir)      LOG_DIR="$2"; shift 2 ;;
    --timeout-min)  TIMEOUT_MIN="$2"; shift 2 ;;
    -h|--help)      grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *)              echo "ERROR: unknown arg $1" >&2; exit 2 ;;
  esac
done

[[ -z "$RUNG" ]]      && { echo "ERROR: --rung required" >&2; exit 2; }
[[ -z "$IMAGE_TAG" ]] && { echo "ERROR: --image-tag required" >&2; exit 2; }
LOG_DIR="${LOG_DIR:-/tmp/${NPU_USER}/easyr1-logs}"

# --- rung config ---
case "$RUNG" in
  V1.1)
    SMOKE_CMD="python3 /opt/easyr1/scripts/smoke_v11_device.py"
    DEFAULT_CHIPS="0"        # OL-05b: 1 chip sufficient for device smoke
    EXPECTED_MARKER="ALL SMOKE CHECKS PASSED"
    ASSERT_TYPE="marker"
    ;;
  V1.3)
    SMOKE_CMD="python3 /opt/easyr1/scripts/smoke_v13_rollout.py"
    DEFAULT_CHIPS="0"        # OL-05b: 1 chip sufficient for single-prompt rollout
    EXPECTED_MARKER="V1.3 ROLLOUT SMOKE PASSED"
    ASSERT_TYPE="marker"
    ;;
  V1.4)
    SMOKE_CMD="bash /opt/easyr1/examples/qwen2_0_5b_math_grpo_npu_smoke.sh"
    DEFAULT_CHIPS="0,1"
    ASSERT_TYPE="entropy_loss"
    ;;
  V1.5)
    SMOKE_CMD="bash /opt/easyr1/examples/qwen2_0_5b_math_grpo_npu_smoke_4chip.sh"
    DEFAULT_CHIPS="0,1,2,3"
    ASSERT_TYPE="entropy_loss"
    ;;
  V2.1)
    SMOKE_CMD="bash /opt/easyr1/examples/qwen2_0_5b_math_grpo_npu_smoke_v2_1_padfree.sh"
    DEFAULT_CHIPS="0,1"
    ASSERT_TYPE="entropy_loss"
    ;;
  V2.2)
    SMOKE_CMD="bash /opt/easyr1/examples/qwen2_0_5b_math_grpo_npu_smoke_v2_2_ulysses.sh"
    DEFAULT_CHIPS="0,1,2,3"
    ASSERT_TYPE="entropy_loss"
    ;;
  *)
    echo "ERROR: unknown rung '$RUNG'" >&2; exit 2 ;;
esac

CHIPS="${CHIPS:-$DEFAULT_CHIPS}"

# Baseline step1 band per rung + image family (from references/SMOKE_BASELINE.md)
# Format: "<lower>:<upper>"
case "${IMAGE_FAMILY}:${RUNG}" in
  v1:V1.4) EXPECTED_BAND="0.94:1.04" ;;
  v1:V1.5) EXPECTED_BAND="1.07:1.18" ;;   # ±5% of 1.127
  v1:V2.1) EXPECTED_BAND="0.94:1.04" ;;   # V1.4-equivalent
  v1:V2.2) EXPECTED_BAND="0.94:1.04" ;;
  v2:V1.4) EXPECTED_BAND="1.21:1.34" ;;
  v2:V2.2) EXPECTED_BAND="1.36:1.51" ;;
  *)       EXPECTED_BAND="" ;;            # V1.1/V1.3 marker-based
esac

SSH_CMD="ssh -p $A3_PORT $A3_USER@$A3_HOST"
TS=$(date +%Y%m%d-%H%M%S)
LOG_FILE="${LOG_DIR}/${RUNG}-${IMAGE_TAG//:/-}-${TS}.log"

echo "=== smoke_validate ${RUNG} ==="
echo "  image:    $IMAGE_TAG"
echo "  chips:    $CHIPS"
echo "  log:      $LOG_FILE"
echo "  band:     ${EXPECTED_BAND:-(marker-based)}"
echo

# --- 1. chip availability (OL-05) ---
echo "--- chip occupancy check ---"
IFS=',' read -ra CHIP_ARR <<< "$CHIPS"
for c in "${CHIP_ARR[@]}"; do
  npu_id=$((c / 2))
  chip_id=$((c % 2))
  info=$($SSH_CMD "npu-smi info -t proc-mem -i $npu_id" 2>/dev/null | awk -v cid="$chip_id" '
    /^[[:space:]]*Chip ID[[:space:]]*:[[:space:]]*[0-9]+/ { match($0,/[0-9]+$/); capture=(substr($0,RSTART,RLENGTH)+0==cid); next }
    capture { print }
  ')
  if echo "$info" | grep -qE "Process id[[:space:]]*:[[:space:]]*[0-9]+"; then
    echo "ERROR: chip $c (NPU $npu_id chip $chip_id) is occupied (OL-05 violation)." >&2
    echo "$info" | grep -E "Process id|Process name" | head >&2
    exit 5
  fi
done
echo "  all chips idle ✓"

# --- 2. mkdir log + run smoke ---
$SSH_CMD "mkdir -p $LOG_DIR"

# Use run-npu-container.sh from the expert-agnostic repo-level scripts (preserved)
RUNNER_SCRIPT="/home/$NPU_USER/workspace/easyr1-npu/scripts/run-npu-container.sh"

echo "--- launching smoke (timeout ${TIMEOUT_MIN}min) ---"
TIMEOUT_SEC=$((TIMEOUT_MIN * 60))
$SSH_CMD "timeout ${TIMEOUT_SEC} bash $RUNNER_SCRIPT --chips $CHIPS --user $NPU_USER --image $IMAGE_TAG --live-source /home/$NPU_USER/workspace/easyr1-npu/upstream/EasyR1 -- $SMOKE_CMD 2>&1 | tee $LOG_FILE" || true

# --- 3. assert ---
echo
echo "--- assertion ---"
$SSH_CMD "[ -s $LOG_FILE ]" || { echo "ERROR: log file empty or missing at $LOG_FILE" >&2; exit 10; }

case "$ASSERT_TYPE" in
  marker)
    if $SSH_CMD "grep -q '$EXPECTED_MARKER' $LOG_FILE"; then
      echo "✅ ${RUNG} PASS (marker '$EXPECTED_MARKER' found)"
      echo "  log: $LOG_FILE"
      exit 0
    else
      echo "❌ ${RUNG} FAIL: marker '$EXPECTED_MARKER' not in log"
      echo "  log tail (last 20 lines):"
      $SSH_CMD "tail -20 $LOG_FILE" | sed 's/^/    /'
      exit 1
    fi
    ;;
  entropy_loss)
    # Extract first entropy_loss (step 1)
    VALUE=$($SSH_CMD "grep -m1 -oE 'entropy_loss:[[:space:]]*[0-9.]+' $LOG_FILE | head -1 | awk -F: '{print \$2}' | tr -d ' '" 2>/dev/null || echo "")
    if [[ -z "$VALUE" ]]; then
      echo "❌ ${RUNG} FAIL: no 'entropy_loss:' marker in log — smoke probably errored before step 1"
      echo "  log tail (last 30 lines):"
      $SSH_CMD "tail -30 $LOG_FILE" | sed 's/^/    /'
      exit 1
    fi
    IFS=':' read -r LOW HIGH <<< "$EXPECTED_BAND"
    # use awk for numeric comparison (bash can't do float)
    IN_BAND=$(awk -v v="$VALUE" -v l="$LOW" -v h="$HIGH" 'BEGIN { print (v >= l && v <= h) ? 1 : 0 }')
    if [[ "$IN_BAND" == "1" ]]; then
      echo "✅ ${RUNG} PASS: entropy_loss=$VALUE in band [$LOW, $HIGH]"
      echo "  log: $LOG_FILE"
      exit 0
    else
      echo "❌ ${RUNG} FAIL: entropy_loss=$VALUE OUT OF BAND [$LOW, $HIGH]"
      echo "  log: $LOG_FILE"
      exit 1
    fi
    ;;
esac
