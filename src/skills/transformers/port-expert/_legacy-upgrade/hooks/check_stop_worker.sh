#!/usr/bin/env bash
# Generic Stop hook for NPU-port worker agents.
#
# REFERENCE COPY. Each expert forks this into its own hooks/ and fills in:
#   - CLAUDE_AGENT_NAME expected value (the expert's worker name)
#   - WS_ROOT workspace-directory-pattern (expert-specific prefix)
#   - caller-package — the thing dry-import validates
#
# Enforces the universal G2 / G3 / OL-04b / OL-09 invariants as documented
# in _shared/references/ALWAYS_LOADED_UNIVERSAL.md.
#
# Inputs (from Claude Code hook env):
#   CLAUDE_AGENT_NAME        — must equal 'transformers-upgrade-worker' or hook is a no-op
#   CLAUDE_PROJECT_DIR       — repo root
#   CLAUDE_SESSION_ID        — used to derive workspace dir, if known
#   CLAUDE_TRANSCRIPT_PATH   — JSONL of the agent run (for edit extraction)
#
# Fallback: if env vars missing, walk recent sessions under
# workspace/transformers-upgrade-*/ and find the one with latest PROGRESS.md mtime.
#
# Exit codes:
#   0  all invariants OK
#   2  BLOCKING — G2 static_check failed (py_compile / dry-import)
#   3  BLOCKING — G3 PASS claim without log evidence
#   4  BLOCKING — PROGRESS.md missing required provenance fields (OL-09)
#   10 hook internal error (bad env, missing tooling) — treat as BLOCKING too

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$EXPERT_ROOT/../../../.." && pwd)}"

# --- gate: only run for the worker agent ---
AGENT="${CLAUDE_AGENT_NAME:-}"
if [[ -n "$AGENT" && "$AGENT" != "transformers-upgrade-worker" ]]; then
  exit 0
fi

err() { echo "[check_transformers_upgrade_worker] $*" >&2; }

# --- locate the agent's workspace ---
WS_ROOT="$PROJECT_DIR/workspace"
WS=""
if [[ -d "$WS_ROOT" ]]; then
  # newest transformers-upgrade-* dir with a PROGRESS.md
  WS=$(find "$WS_ROOT" -maxdepth 2 -type f -name PROGRESS.md -path '*/transformers-upgrade-*/*' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -1 | awk '{print $2}' | xargs -I{} dirname {} 2>/dev/null || true)
fi

if [[ -z "$WS" || ! -d "$WS" ]]; then
  err "no transformers-upgrade-* workspace with PROGRESS.md found under $WS_ROOT"
  err "worker must create workspace/transformers-upgrade-{SESSION_TAG}/PROGRESS.md in P0"
  exit 4
fi

PROGRESS="$WS/PROGRESS.md"
echo "[check_transformers_upgrade_worker] workspace: $WS"

# --- OL-09: PROGRESS.md provenance fields (transformers-upgrade-specific set) ---
for field in "MODE:" "SOURCE_IMAGE:" "TARGET_IMAGE:" "UPSTREAM_REF:" "Handoff:"; do
  if ! grep -q "^$field" "$PROGRESS" 2>/dev/null; then
    err "OL-09 violation: PROGRESS.md missing required field '$field'"
    exit 4
  fi
done

# --- OL-04b: cleanup field must be present and non-empty ---
# Accept: "Cleanup: clean", "Cleanup: partial", "Cleanup: skipped <reason>".
# Reject: missing line, or bare "Cleanup: skipped" without a reason.
CLEANUP_LINE=$(grep -E '^Cleanup:' "$PROGRESS" 2>/dev/null | head -1 || true)
if [[ -z "$CLEANUP_LINE" ]]; then
  err "OL-04b violation: PROGRESS.md missing 'Cleanup:' field. Run scripts/cleanup_session.sh."
  exit 4
fi
case "$CLEANUP_LINE" in
  "Cleanup: clean"*|"Cleanup: partial"*) ;;
  "Cleanup: skipped "?*) ;;  # skipped WITH reason
  "Cleanup: skipped"|"Cleanup: skipped "*)
    err "OL-04b violation: 'Cleanup: skipped' requires a concrete reason after 'skipped'."
    exit 4
    ;;
  *)
    err "OL-04b violation: 'Cleanup:' value must be 'clean', 'partial', or 'skipped <reason>'. Got: $CLEANUP_LINE"
    exit 4
    ;;
esac

# --- G2: static_check on edited files ---
# Extract edited files from transcript if available; else scan upstream consumer tree
EDITED_FILES=""
TRANSCRIPT="${CLAUDE_TRANSCRIPT_PATH:-}"
if [[ -n "$TRANSCRIPT" && -f "$TRANSCRIPT" ]]; then
  # Pull file_path from any Edit/Write/MultiEdit tool_use lines
  EDITED_FILES=$(python3 -c "
import json, sys
seen = set()
try:
    with open('$TRANSCRIPT') as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            msg = rec.get('message') or {}
            for blk in (msg.get('content') or []):
                if not isinstance(blk, dict): continue
                if blk.get('type') != 'tool_use': continue
                if blk.get('name') not in ('Edit','Write','MultiEdit'): continue
                fp = (blk.get('input') or {}).get('file_path')
                if fp and fp.endswith('.py'):
                    seen.add(fp)
    for p in sorted(seen):
        print(p)
except Exception as e:
    sys.stderr.write(f'transcript-parse-failed: {e}\n')
" 2>/dev/null || true)
fi

# Fallback: scan upg branch's consumer tree (default: upstream/EasyR1)
CONSUMER_DIR="$PROJECT_DIR/../upstream/${UPSTREAM_CONSUMER:-EasyR1}"
[[ -d "$CONSUMER_DIR" ]] || CONSUMER_DIR="$HOME/workspace/easyr1-npu/upstream/${UPSTREAM_CONSUMER:-EasyR1}"
if [[ -z "$EDITED_FILES" && -d "$CONSUMER_DIR" ]]; then
  # diff vs the baseline ref this upgrade session started from
  BASE_REF="${UPSTREAM_REF:-main}"
  EDITED_FILES=$(cd "$CONSUMER_DIR" && git diff --name-only "$BASE_REF..." 2>/dev/null | grep '\.py$' | xargs -I{} echo "$CONSUMER_DIR/{}" | head -50 || true)
fi

STATIC_CHECK="$EXPERT_ROOT/scripts/static_check.py"
if [[ ! -x "$STATIC_CHECK" && ! -f "$STATIC_CHECK" ]]; then
  err "static_check.py not found at $STATIC_CHECK"
  exit 10
fi

if [[ -n "$EDITED_FILES" ]]; then
  echo "[check_transformers_upgrade_worker] G2: static_check on $(echo "$EDITED_FILES" | wc -l) files"
  # shellcheck disable=SC2086
  # --import-package defaults to consumer package verl; override via env if different consumer
  IMPORT_PKG="${UPSTREAM_CONSUMER_PKG:-verl}"
  if ! python3 "$STATIC_CHECK" --files $EDITED_FILES --import-package "$IMPORT_PKG" 2>&1 | tee "$WS/static-check-report.txt"; then
    err "G2 BLOCKING: static_check failed. See $WS/static-check-report.txt"
    exit 2
  fi
else
  err "WARNING: no edited .py files detected — G2 not runnable, proceeding with caution"
fi

# --- G3: PASS claim must cite log with entropy_loss in band ---
if grep -qiE '\b(PASS|green|closed|end.to.end|complete)\b' "$PROGRESS"; then
  echo "[check_transformers_upgrade_worker] G3: PASS-style claim found in PROGRESS.md, verifying evidence"
  # Find referenced log paths (lines like "log: /path/…log" or "/tmp/…/easyr1-logs/*.log")
  LOG_REFS=$(grep -oE '(/tmp/[^[:space:]]+\.log|/home/[^[:space:]]+\.log)' "$PROGRESS" | sort -u)
  if [[ -z "$LOG_REFS" ]]; then
    err "G3 BLOCKING: PASS claim in PROGRESS.md without any log file path reference"
    exit 3
  fi
  ANY_VALID=0
  for logref in $LOG_REFS; do
    # logref is an A3-side path typically. Skip local check if not present locally;
    # treat bare presence of a well-formed path + PROGRESS.md mentioning 'entropy_loss:' as OK here.
    # (Full validation happens in smoke_validate.sh which runs on A3.)
    if grep -qE 'entropy_loss[:=][[:space:]]*[0-9.]+' "$PROGRESS"; then
      ANY_VALID=1
      break
    fi
  done
  if [[ $ANY_VALID -eq 0 ]]; then
    err "G3 BLOCKING: PASS claim with log path but no entropy_loss numeric cited in PROGRESS.md"
    err "      (smoke_validate.sh output must be pasted or its numeric echoed into PROGRESS)"
    exit 3
  fi
fi

echo "[check_transformers_upgrade_worker] OK — G2 + G3 + OL-09 all satisfied"
exit 0
