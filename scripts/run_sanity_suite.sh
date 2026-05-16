#!/bin/bash
# Sanity test suite for easyr1-npu (P0h.0/.1, T31.4).
#
# Runs pytest tests under tests/ — all 100% mechanical (no NPU, no LLM).
# Designed to run in < 2s before every commit.
#
# Usage:
#   bash scripts/run_sanity_suite.sh           # quiet
#   bash scripts/run_sanity_suite.sh -v        # verbose
#   bash scripts/run_sanity_suite.sh --tb=long # full traceback

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1

echo "== Sanity test suite (easyr1-npu) =="
echo "Repo: $(pwd)"
if git rev-parse --git-dir >/dev/null 2>&1; then
    HEAD_SHORT=$(git rev-parse --short HEAD)
    HEAD_SUBJECT=$(git log -1 --format=%s | head -c 60)
    echo "HEAD: $HEAD_SHORT ($HEAD_SUBJECT)"
fi
echo

RC=0
python3 -m pytest tests/ "$@" --tb=short 2>&1 || RC=$?

echo
if [[ $RC -eq 0 ]]; then
    echo "== SANITY SUITE: PASS =="
else
    echo "== SANITY SUITE: FAIL (exit $RC) =="
fi
exit $RC
