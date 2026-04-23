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

# vllm-upgrade-worker is the only agent allowed to touch the vllm-adjacent
# shim files + any Dockerfile.npu-vllm-*.
if [[ "$AGENT" == "vllm-upgrade-worker" ]]; then
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

[[ -z "$FILE_PATH" ]] && exit 0

# Protected paths for this expert:
# - verl/utils/vllm_utils.py          (CP-002 + VLLMHijack)
# - verl/workers/rollout/vllm_rollout_spmd.py (EC-03)
# - verl/workers/sharding_manager/fsdp_vllm.py (CP-004)
# - Dockerfile.npu-vllm-* (if any)
# Other verl/*.py files are NOT this expert's domain; rely on easyr1-expert's
# own hook for those.
case "$FILE_PATH" in
  *upstream/*/Dockerfile.npu-vllm-*|*/Dockerfile.npu-vllm-*|*upstream/*/verl/utils/vllm_utils.py|*upstream/*/verl/workers/rollout/vllm_rollout_spmd.py|*upstream/*/verl/workers/sharding_manager/fsdp_vllm.py)
    echo "[check_edit_scope] BLOCKING: G1 invariant — vllm-upgrade shim edits must go through vllm-upgrade-worker agent, not $AGENT" >&2
    echo "  target: $FILE_PATH" >&2
    echo "  see: vllm/upgrade-expert/state_machine.yaml G1" >&2
    exit 1
    ;;
esac

exit 0
