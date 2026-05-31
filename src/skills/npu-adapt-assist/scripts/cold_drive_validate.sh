#!/usr/bin/env bash
# Cold-drive validation for /npu-adapt-assist retrieval.
# Verifies that 3 known-case inputs produce the expected top-1 KB id.
# Run from anywhere; resolves paths relative to this script.

set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTS="${SKILL_DIR}/tests"
RETRIEVE="${SKILL_DIR}/scripts/retrieve.py"

expect() {
    local case_name="$1"
    local input_file="$2"
    local expected_id="$3"

    echo "[case ${case_name}] input: ${input_file}"
    local json
    json="$(python3 "${RETRIEVE}" --trace-file "${input_file}" --json)"
    local got
    got="$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d['top'][0]['id'] if d['matched'] else 'NO_MATCH')" "${json}")"
    if [[ "${got}" == "${expected_id}" ]]; then
        echo "  PASS — top-1 = ${got}"
        return 0
    else
        echo "  FAIL — expected ${expected_id}, got ${got}"
        echo "  full result:"
        echo "${json}" | sed 's/^/    /'
        return 1
    fi
}

fails=0
expect A "${TESTS}/case_a_rmsnorm.txt" "sglang-002" || fails=$((fails+1))
expect B "${TESTS}/case_b_syspath.txt" "cross-layer-008" || fails=$((fails+1))
expect C "${TESTS}/case_c_sparse_mla.txt" "bishengir-001" || fails=$((fails+1))

if [[ "${fails}" -eq 0 ]]; then
    echo
    echo "Cold-drive PASS — 3/3 cases produce expected top-1"
    exit 0
else
    echo
    echo "Cold-drive FAIL — ${fails}/3 case(s) mismatched"
    exit 1
fi
