#!/usr/bin/env bash
# Generic PreToolUse hook for Edit/Write/MultiEdit.
#
# REFERENCE COPY. Each expert forks this into its own hooks/ and changes:
#   - CLAUDE_AGENT_NAME expected value (their worker's name)
#   - the protected path globs (what the expert alone can edit)
#
# Enforces G1 (orchestrator cannot edit expert-controlled paths) as
# documented in _shared/references/ALWAYS_LOADED_UNIVERSAL.md.
#
# Reads the tool input from stdin (JSON: {"tool_input": {"file_path": ...}}).
# Exits non-zero to block the tool call.

set -uo pipefail

AGENT="${CLAUDE_AGENT_NAME:-orchestrator}"

# transformers-upgrade-worker is the only agent allowed in the consumer
# repo's Dockerfile + shim files (for this expert's scope).
if [[ "$AGENT" == "transformers-upgrade-worker" ]]; then
  exit 0
fi

# Read tool input JSON from stdin
INPUT=$(cat 2>/dev/null || echo '{}')
FILE_PATH=$(python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1] or '{}')
    print((d.get('tool_input') or {}).get('file_path') or '')
except Exception:
    print('')
" "$INPUT" 2>/dev/null || echo "")

# No file_path → nothing to check
[[ -z "$FILE_PATH" ]] && exit 0

# Protected paths for this expert:
#   - Dockerfile.npu-* in any upstream consumer tree (this expert's territory)
#   - The two known backcompat-shim files (EC-02 + EC-03)
#   - requirements*.txt and examples/* if edited as part of upgrade
# (Other verl/*.py files are easyr1-expert's territory; this hook doesn't
#  block them here — easyr1-expert's own check_edit_scope.sh protects them.)
case "$FILE_PATH" in
  *upstream/*/Dockerfile.npu-*|*/Dockerfile.npu-*|*upstream/*/verl/workers/fsdp_workers.py|*upstream/*/verl/workers/rollout/vllm_rollout_spmd.py)
    echo "[check_edit_scope] BLOCKING: G1 invariant — transformers-upgrade code edits must go through transformers-upgrade-worker agent, not $AGENT" >&2
    echo "  target: $FILE_PATH" >&2
    echo "  see: state_machine.yaml G1, NPU-OPS-010" >&2
    exit 1
    ;;
esac

exit 0
