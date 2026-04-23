#!/usr/bin/env bash
# Stop hook for torch-day0-worker.
#
# Enforces G2 (runtime smoke PASS inside overlay) / G3 (outcome evidence
# match) / OL-04b (cleanup field) / OL-09 (PROGRESS provenance fields).
#
# Exit codes:
#   0  all invariants OK
#   2  BLOCKING — G2 runtime smoke marker missing
#   3  BLOCKING — G3 outcome claim without matching evidence
#   4  BLOCKING — PROGRESS.md missing required provenance fields (OL-09)
#   10 hook internal error

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$EXPERT_ROOT/../../../.." && pwd)}"

# --- gate: only run for our worker agent ---
AGENT="${CLAUDE_AGENT_NAME:-}"
if [[ -n "$AGENT" && "$AGENT" != "torch-day0-worker" ]]; then
  exit 0
fi

err() { echo "[check_stop_torch_day0] $*" >&2; }

# --- locate the agent's workspace ---
WS_ROOT="$PROJECT_DIR/workspace"
WS=""
if [[ -d "$WS_ROOT" ]]; then
  # newest torch-day0-{analysis,manual,deploy}-* dir with a PROGRESS.md
  WS=$(find "$WS_ROOT" -maxdepth 2 -type f -name PROGRESS.md \
      \( -path '*/torch-day0-manual-*/*' -o -path '*/torch-day0-analysis-*/*' \) \
      -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -1 | awk '{print $2}' | xargs -I{} dirname {} 2>/dev/null || true)
fi

if [[ -z "$WS" || ! -d "$WS" ]]; then
  err "no torch-day0-{analysis,manual}-* workspace with PROGRESS.md found under $WS_ROOT"
  err "worker must create workspace/torch-day0-manual-{SESSION_TAG}/PROGRESS.md in P2"
  exit 4
fi

PROGRESS="$WS/PROGRESS.md"
echo "[check_stop_torch_day0] workspace: $WS"

# --- OL-09: PROGRESS.md provenance fields (torch-day0 specific set) ---
for field in "TARGET_TORCH_VERSION:" "TARGET_TORCH_NPU_VERSION:" "BASE_IMAGE:" "SESSION_TAG:"; do
  if ! grep -q "$field" "$PROGRESS" 2>/dev/null; then
    err "OL-09 violation: PROGRESS.md missing required field '$field'"
    exit 4
  fi
done

# --- OL-04b: cleanup status must be present ---
CLEANUP_LINE=$(grep -E '^Cleanup:' "$PROGRESS" 2>/dev/null | head -1 || true)
if [[ -z "$CLEANUP_LINE" ]]; then
  err "OL-04b violation: PROGRESS.md missing 'Cleanup:' field. Run scripts/cleanup_session.sh."
  exit 4
fi
case "$CLEANUP_LINE" in
  "Cleanup: clean"*|"Cleanup: partial"*) ;;
  "Cleanup: skipped "?*) ;;
  "Cleanup: skipped"|"Cleanup: skipped "*)
    err "OL-04b violation: 'Cleanup: skipped' requires a concrete reason after 'skipped'."
    exit 4
    ;;
  *)
    err "OL-04b violation: 'Cleanup:' value must be 'clean', 'partial', or 'skipped <reason>'. Got: $CLEANUP_LINE"
    exit 4
    ;;
esac

# --- Detect outcome from PROGRESS ---
OUTCOME=$(grep -iE '^Phase 2 outcome|^Outcome:|^outcome:' "$PROGRESS" 2>/dev/null | head -1 | \
    sed -E 's/.*[Oo]utcome[:[:space:]]*//;s/[[:space:]].*$//' || true)
echo "[check_stop_torch_day0] detected outcome: ${OUTCOME:-unknown}"

# --- G2: runtime smoke marker ---
# A / A-with-note / C-patch outcomes require a smoke log with marker.
# Look at workspace/torch-day0-manual-*/ or workspace/torch-day0-deploy-*/ for smoke.log.
case "$OUTCOME" in
  A|A-with-note|C-patch|PASS)
    SMOKE_LOG_PATTERN="*/torch-day0-*-*/smoke*.log"
    SMOKE_LOG=$(find "$WS_ROOT" -maxdepth 3 -type f -name 'smoke*.log' \
                  \( -path '*/torch-day0-manual-*/*' -o -path '*/torch-day0-deploy-*/*' \) \
                  -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | awk '{print $2}' || true)
    if [[ -z "$SMOKE_LOG" || ! -f "$SMOKE_LOG" ]]; then
      err "G2 BLOCKING: outcome=$OUTCOME requires smoke log, none found under workspace/torch-day0-*/"
      exit 2
    fi
    if ! grep -q 'ALL SMOKE STEPS PASSED' "$SMOKE_LOG"; then
      err "G2 BLOCKING: smoke log $SMOKE_LOG missing 'ALL SMOKE STEPS PASSED' marker"
      exit 2
    fi
    echo "[check_stop_torch_day0] G2 OK: smoke marker found in $SMOKE_LOG"
    ;;
  C-report|deferred-target)
    # C-report doesn't require a passing smoke; require blocker-report.md exists
    BLOCKER=$(find "$WS_ROOT" -maxdepth 3 -type f -name 'blocker-report.md' \
                -path '*/torch-day0-*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | awk '{print $2}' || true)
    if [[ -z "$BLOCKER" || ! -f "$BLOCKER" ]]; then
      err "G3 BLOCKING: outcome=$OUTCOME requires blocker-report.md, none found"
      exit 3
    fi
    echo "[check_stop_torch_day0] G3 OK: blocker report found at $BLOCKER"
    ;;
  "")
    err "G3 WARNING: no outcome line in PROGRESS.md; worker must set 'Phase 2 outcome:' or 'Outcome:'"
    ;;
esac

# --- G3: deploy artifacts dir populated for A / A-with-note / C-patch ---
case "$OUTCOME" in
  A|A-with-note|C-patch)
    DEPLOY_DIR=$(find "$WS_ROOT" -maxdepth 2 -type d -name 'torch-day0-deploy-*' \
                   -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | awk '{print $2}' || true)
    if [[ -z "$DEPLOY_DIR" || ! -d "$DEPLOY_DIR" ]]; then
      err "G3 BLOCKING: outcome=$OUTCOME requires Phase 2.5 deploy artifacts dir (workspace/torch-day0-deploy-*/)"
      exit 3
    fi
    for f in Dockerfile.overlay-torch* smoke_torch*.sh deploy_torch*.sh ONBOARDING.md; do
      if ! ls "$DEPLOY_DIR"/$f 1>/dev/null 2>&1; then
        err "G3 BLOCKING: deploy artifacts dir $DEPLOY_DIR missing '$f'"
        exit 3
      fi
    done
    echo "[check_stop_torch_day0] G3 OK: deploy artifacts present at $DEPLOY_DIR"
    ;;
esac

echo "[check_stop_torch_day0] OK — all invariants satisfied"
exit 0
