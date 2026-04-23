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

# vllm-day0-worker is the only agent allowed to touch
# Dockerfile.overlay-vllm* and the 3 vllm-adjacent shim files.
if [[ "$AGENT" == "vllm-day0-worker" ]]; then
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

# Protected paths for vllm-day0:
# - Dockerfile.overlay-vllm* anywhere in upstream consumer tree
# - The 3 vllm-adjacent shim files (outcome B consumer-side shim)
# - $WORKSPACE/vllm-day0-*/ (forward-port notes etc.)
# Other Dockerfiles + verl/*.py belong to sibling experts.
case "$FILE_PATH" in
  *upstream/*/Dockerfile.overlay-vllm*|*/Dockerfile.overlay-vllm*|\
  *upstream/*/verl/utils/vllm_utils.py|\
  *upstream/*/verl/workers/rollout/vllm_rollout_spmd.py|\
  *upstream/*/verl/workers/sharding_manager/fsdp_vllm.py|\
  *workspace/vllm-day0-*/*)
    echo "[check_edit_scope] BLOCKING: G1 invariant — vllm-day0 files must go through vllm-day0-worker, not $AGENT" >&2
    echo "  target: $FILE_PATH" >&2
    echo "  see: vllm-day0-expert/state_machine.yaml G1" >&2
    exit 1
    ;;
esac

exit 0
