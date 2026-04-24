#!/usr/bin/env bash
# PreToolUse hook for dep-analysis-worker.
#
# This expert is PURE READ-ONLY outside its own session workspace.
# Any Edit/Write/MultiEdit targeting a path outside
#   workspace/dep-analysis-<SESSION_TAG>/
# is G1 / OL-08 violation and gets blocked.
#
# Reads the tool input from stdin (JSON: {"tool_input": {"file_path": ...}}).
# Exits non-zero to block the tool call.

set -uo pipefail

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

# The ONLY allowed write target: workspace/dep-analysis-<SESSION_TAG>/**
case "$FILE_PATH" in
  */workspace/dep-analysis-*/*)
    exit 0 ;;  # allowed
  *)
    echo "[check_edit_scope] BLOCKING: dep-analysis is read-only outside its session workspace" >&2
    echo "  target: $FILE_PATH" >&2
    echo "  allowed prefix: */workspace/dep-analysis-\${SESSION_TAG}/" >&2
    echo "  see: references/ALWAYS_LOADED_RULES.md §OL-08" >&2
    exit 1 ;;
esac
