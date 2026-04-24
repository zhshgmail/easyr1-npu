#!/usr/bin/env bash
# PreToolUse hook for Edit/Write/MultiEdit.
# Enforces G1: orchestrator MUST NOT directly edit port-code paths —
# those edits belong to easyr1-port-worker.
#
# Reads the tool input from stdin (JSON: {"tool_input": {"file_path": ...}}).
# Exits non-zero to block the tool call.

set -uo pipefail

AGENT="${CLAUDE_AGENT_NAME:-orchestrator}"

# Worker is allowed anywhere in port-code tree.
if [[ "$AGENT" == "easyr1-port-worker" ]]; then
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

# Protected paths: upstream/EasyR1/verl/**, examples/**, Dockerfile*
case "$FILE_PATH" in
  *upstream/EasyR1/verl/*|*upstream/EasyR1/examples/*|*upstream/EasyR1/Dockerfile*|*/EasyR1/verl/*|*/EasyR1/examples/*|*/EasyR1/Dockerfile*)
    echo "[check_edit_scope] BLOCKING: G1 invariant — port code edits must go through easyr1-port-worker agent, not $AGENT" >&2
    echo "  target: $FILE_PATH" >&2
    echo "  see: state_machine.yaml G1, NPU-OPS-010" >&2
    exit 1
    ;;
esac

exit 0
