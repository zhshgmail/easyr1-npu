#!/usr/bin/env bash
# PreToolUse hook for torch-day0 expert.
#
# Enforces G1 (orchestrator cannot edit torch-day0-controlled paths).
# Reads tool input from stdin, exits non-zero to block.

set -uo pipefail

AGENT="${CLAUDE_AGENT_NAME:-orchestrator}"

# torch-day0-worker is the only agent allowed to touch the paths below.
if [[ "$AGENT" == "torch-day0-worker" ]]; then
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

# Protected paths for torch-day0 (only torch-day0-worker may touch):
# - Dockerfile.overlay-torch* anywhere in consumer/workspace tree
# - $WORKSPACE/torch-day0-*/ (analysis, manual, deploy dirs)
# - upstream/torch-npu/**/*.py + csrc on ascend-day0-torch<M><m>-<SESSION> branch
#   (C-patch scope: Huawei-owned torch_npu, day0 expert's legitimate edit target)
# - upstream/torch-npu/torch_npu/** for python-layer integration patches
# Community PyTorch (upstream/pytorch/**) is NOT in scope (C-report only).
case "$FILE_PATH" in
  *upstream/*/Dockerfile.overlay-torch*|*/Dockerfile.overlay-torch*|\
  *workspace/torch-day0-*/*|\
  *upstream/torch-npu/*)
    echo "[check_edit_scope] BLOCKING: G1 invariant — torch-day0 files must go through torch-day0-worker, not $AGENT" >&2
    echo "  target: $FILE_PATH" >&2
    echo "  see: torch-npu/port-expert/state_machine.yaml G1" >&2
    exit 1
    ;;
esac

exit 0
