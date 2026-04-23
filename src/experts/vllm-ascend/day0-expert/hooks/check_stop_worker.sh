#!/usr/bin/env bash
# Stop hook for vllm-ascend-day0-worker.
#
# Enforces G2 (V1.3 smoke marker) / G3 (outcome artifacts) /
# OL-04b (cleanup) / OL-09 (provenance fields in PROGRESS).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$EXPERT_ROOT/../../../.." && pwd)}"

AGENT="${CLAUDE_AGENT_NAME:-}"
if [[ -n "$AGENT" && "$AGENT" != "vllm-ascend-day0-worker" ]]; then
  exit 0
fi

err() { echo "[check_stop_vllm_ascend_day0] $*" >&2; }

WS_ROOT="$PROJECT_DIR/workspace"
WS=""
if [[ -d "$WS_ROOT" ]]; then
  WS=$(find "$WS_ROOT" -maxdepth 2 -type f -name PROGRESS.md \
      \( -path '*/vllm-ascend-day0-analysis-*/*' -o -path '*/vllm-ascend-day0-deploy-*/*' \) \
      -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -1 | awk '{print $2}' | xargs -I{} dirname {} 2>/dev/null || true)
fi

if [[ -z "$WS" || ! -d "$WS" ]]; then
  err "no vllm-ascend-day0-* workspace with PROGRESS.md found under $WS_ROOT"
  exit 4
fi

PROGRESS="$WS/PROGRESS.md"
echo "[check_stop_vllm_ascend_day0] workspace: $WS"

# OL-09: provenance fields
for field in "TARGET_DELTA:" "BASE_IMAGE:" "SESSION_TAG:"; do
  if ! grep -q "$field" "$PROGRESS" 2>/dev/null; then
    err "OL-09 violation: PROGRESS.md missing '$field'"
    exit 4
  fi
done

# OL-04b: cleanup
CLEANUP_LINE=$(grep -E '^Cleanup:' "$PROGRESS" 2>/dev/null | head -1 || true)
if [[ -z "$CLEANUP_LINE" ]]; then
  err "OL-04b: PROGRESS.md missing 'Cleanup:' field"
  exit 4
fi

# Detect outcome
OUTCOME=$(grep -iE '^Outcome:|^Phase 4 outcome' "$PROGRESS" 2>/dev/null | head -1 | \
    sed -E 's/.*[Oo]utcome[:[:space:]]*//;s/[[:space:]].*$//' || true)
echo "[check_stop_vllm_ascend_day0] outcome: ${OUTCOME:-unknown}"

# G2: V1.3 smoke marker for A / B / C-patch
case "$OUTCOME" in
  A|B|C-patch)
    SMOKE_LOG=$(find "$WS_ROOT" -maxdepth 3 -type f -name 'smoke*.log' \
                  -path '*/vllm-ascend-day0-deploy-*/*' \
                  -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | awk '{print $2}' || true)
    if [[ -z "$SMOKE_LOG" || ! -f "$SMOKE_LOG" ]]; then
      err "G2: no smoke log found under workspace/vllm-ascend-day0-deploy-*/"
      exit 2
    fi
    if ! grep -q 'V1.3 ROLLOUT SMOKE PASSED' "$SMOKE_LOG"; then
      err "G2: smoke log $SMOKE_LOG missing 'V1.3 ROLLOUT SMOKE PASSED' marker"
      exit 2
    fi
    ;;
  C-report)
    BLOCKER=$(find "$WS_ROOT" -maxdepth 3 -type f -name 'blocker-report.md' \
                -path '*/vllm-ascend-day0-*' 2>/dev/null | head -1 || true)
    if [[ -z "$BLOCKER" ]]; then
      err "G3: C-report requires blocker-report.md"
      exit 3
    fi
    ;;
esac

# G3: deploy artifacts for C-patch
if [[ "$OUTCOME" == "C-patch" ]]; then
  DEPLOY_DIR=$(find "$WS_ROOT" -maxdepth 2 -type d -name 'vllm-ascend-day0-deploy-*' \
                 -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | awk '{print $2}' || true)
  for f in ONBOARDING.md PR_MATERIAL.md deploy_vllm_ascend_*.sh; do
    if ! ls "$DEPLOY_DIR"/$f 1>/dev/null 2>&1; then
      err "G3: C-patch requires deploy artifact '$f' in $DEPLOY_DIR"
      exit 3
    fi
  done
fi

echo "[check_stop_vllm_ascend_day0] OK — invariants satisfied"
exit 0
