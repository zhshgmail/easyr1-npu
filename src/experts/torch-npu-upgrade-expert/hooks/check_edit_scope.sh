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

# torch-npu-upgrade-worker is the only agent allowed to touch
# Dockerfile.npu-torch-* for this expert's scope.
if [[ "$AGENT" == "torch-npu-upgrade-worker" ]]; then
  exit 0
fi

INPUT=$(cat 2>/dev/null || echo '{}')
FILE_PATH=$(python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1] or '{}')
    print((d.get('tool_input') or {}).get('file_path') or '')
except Exception:
    print('')
" "$INPUT" 2>/dev/null || echo "")

[[ -z "$FILE_PATH" ]] && exit 0

# Protected paths for this expert: ONLY Dockerfile.npu-torch-*.
# Other Dockerfiles + verl/*.py belong to sibling experts; their hooks guard
# those.
case "$FILE_PATH" in
  *upstream/*/Dockerfile.npu-torch-*|*/Dockerfile.npu-torch-*)
    echo "[check_edit_scope] BLOCKING: G1 invariant — torch-stack Dockerfile edits must go through torch-npu-upgrade-worker agent, not $AGENT" >&2
    echo "  target: $FILE_PATH" >&2
    echo "  see: torch-npu-upgrade-expert/state_machine.yaml G1" >&2
    exit 1
    ;;
esac

exit 0
