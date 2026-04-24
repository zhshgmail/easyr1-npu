#!/usr/bin/env bash
# PreToolUse hook for vllm-ascend-day0 expert.
#
# Enforces G1. Reads tool input from stdin, exits non-zero to block.

set -uo pipefail

AGENT="${CLAUDE_AGENT_NAME:-orchestrator}"

if [[ "$AGENT" == "vllm-ascend-day0-worker" ]]; then
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

# Protected paths for vllm-ascend-day0 (only vllm-ascend-day0-worker may touch):
# - upstream/vllm-ascend/**/*.py on ascend-day0-<delta>-<SESSION> branch
# - $WORKSPACE/vllm-ascend-day0-*/
# - Dockerfile.overlay-vllm-ascend-*
case "$FILE_PATH" in
  *upstream/vllm-ascend/*|\
  *workspace/vllm-ascend-day0-*/*|\
  *Dockerfile.overlay-vllm-ascend-*|\
  */Dockerfile.overlay-vllm-ascend-*)
    echo "[check_edit_scope] BLOCKING: G1 — vllm-ascend-day0 files must go through vllm-ascend-day0-worker, not $AGENT" >&2
    echo "  target: $FILE_PATH" >&2
    echo "  see: vllm-ascend/port-expert/state_machine.yaml G1" >&2
    exit 1
    ;;
esac

exit 0
