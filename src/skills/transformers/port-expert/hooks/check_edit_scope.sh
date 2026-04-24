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

# transformers-day0-worker is the only agent allowed to touch
# Dockerfile.overlay-trans* and $WORKSPACE/patches/npu_flash_attention.py.
if [[ "$AGENT" == "transformers-day0-worker" ]]; then
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

# Protected paths for this expert (only transformers-day0-worker may touch):
# - Dockerfile.overlay-trans* anywhere in upstream consumer tree
# - $WORKSPACE/transformers-day0-*/ (workspace + patches/)
# - verl/workers/fsdp_workers.py (EC-02 shim; widened per 2026-04-23 vllm-day0
#   wet-run finding — fixture=master case needs the shim applied by this
#   expert, not orchestrator)
# - scripts/smoke_v{11,13}*.py + examples/qwen2*.sh (harness; fixture=master
#   doesn't have them)
# Other verl/*.py belong to sibling experts (easyr1-expert, etc.).
case "$FILE_PATH" in
  *upstream/*/Dockerfile.overlay-trans*|*/Dockerfile.overlay-trans*|\
  *workspace/transformers-day0-*/*|\
  *upstream/*/verl/workers/fsdp_workers.py|\
  *upstream/*/scripts/smoke_v11_device.py|\
  *upstream/*/scripts/smoke_v13_rollout.py|\
  *upstream/*/examples/qwen2_0_5b_math_grpo_npu_smoke.sh|\
  *upstream/transformers/src/transformers/integrations/npu_*)
    echo "[check_edit_scope] BLOCKING: G1 invariant — transformers-day0 files must go through transformers-day0-worker, not $AGENT" >&2
    echo "  target: $FILE_PATH" >&2
    echo "  see: transformers/port-expert/state_machine.yaml G1" >&2
    exit 1
    ;;
esac

exit 0
