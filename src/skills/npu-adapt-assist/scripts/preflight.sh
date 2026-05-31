#!/usr/bin/env bash
# /npu-adapt-assist preflight — fail loud and early.
#
# Validates that the skill's prerequisites are in place before retrieve.py runs.
# Modeled on a5_ops's three-tier preflight (see workspace/a5_ops_audit_2026_05_31/
# FINDINGS.md §1.1).
#
# Verdicts:
#   CLEAN  (exit 0) — all checks pass, proceed
#   WARN   (exit 1) — non-blocking issues, retrieve.py may proceed but quality
#                     may be degraded
#   ABORT  (exit 2) — blocking issue, retrieve.py MUST NOT proceed
#
# Output: human-readable ASCII summary on stderr; nothing on stdout (so
# retrieve.py can be chained without parsing noise).
#
# Usage:
#   preflight.sh           # full check
#   preflight.sh --quiet   # only output on WARN/ABORT
#   preflight.sh --json    # machine-readable verdict

set -uo pipefail

QUIET=0
JSON=0
for arg in "$@"; do
    case "$arg" in
        --quiet) QUIET=1 ;;
        --json)  JSON=1 ;;
        *) ;;
    esac
done

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RETRIEVE="${SKILL_DIR}/scripts/retrieve.py"
TESTS_DIR="${SKILL_DIR}/tests"

# Find repo root (walk up for docs/_meta/kb/porting_lessons)
REPO_ROOT=""
cur="${SKILL_DIR}"
while [[ "${cur}" != "/" ]]; do
    if [[ -d "${cur}/docs/_meta/kb/porting_lessons" ]]; then
        REPO_ROOT="${cur}"
        break
    fi
    cur="$(dirname "${cur}")"
done

KB_DIR="${REPO_ROOT}/docs/_meta/kb/porting_lessons"

errors=()
warnings=()

# ---- mandatory checks (any failure → ABORT) ----

# M1: KB dir reachable
if [[ -z "${REPO_ROOT}" || ! -d "${KB_DIR}" ]]; then
    errors+=("M1: KB dir docs/_meta/kb/porting_lessons not found by walking up from ${SKILL_DIR}")
fi

# M2: retrieve.py executable
if [[ ! -f "${RETRIEVE}" ]]; then
    errors+=("M2: retrieve.py missing at ${RETRIEVE}")
elif [[ ! -x "${RETRIEVE}" ]]; then
    warnings+=("M2: retrieve.py exists but not executable; chmod +x recommended")
fi

# M3: python3 available
if ! command -v python3 >/dev/null 2>&1; then
    errors+=("M3: python3 not in PATH")
fi

# M4: KB has at least 1 schema-conformant entry
if [[ -d "${KB_DIR}" ]]; then
    entry_count=0
    while IFS= read -r -d '' f; do
        # skip index.md and _schema.md
        case "$(basename "$f")" in
            index.md|_schema.md) continue ;;
        esac
        if head -1 "$f" 2>/dev/null | grep -q '^---$'; then
            entry_count=$((entry_count + 1))
        fi
    done < <(find "${KB_DIR}" -maxdepth 1 -name '*.md' -print0 2>/dev/null)
    if [[ "${entry_count}" -eq 0 ]]; then
        errors+=("M4: no schema-conformant KB entries in ${KB_DIR}")
    fi
fi

# ---- soft checks (failure → WARN) ----

# S1: schema file present (cookbook authors need it)
if [[ -d "${KB_DIR}" && ! -f "${KB_DIR}/_schema.md" ]]; then
    warnings+=("S1: ${KB_DIR}/_schema.md missing — new-cookbook authors lose template")
fi

# S2: index.md present
if [[ -d "${KB_DIR}" && ! -f "${KB_DIR}/index.md" ]]; then
    warnings+=("S2: ${KB_DIR}/index.md missing — humans lose navigation")
fi

# S3: cold-drive tests dir + 3 known cases
if [[ -d "${TESTS_DIR}" ]]; then
    for f in case_a_rmsnorm.txt case_b_syspath.txt case_c_sparse_mla.txt; do
        if [[ ! -f "${TESTS_DIR}/${f}" ]]; then
            warnings+=("S3: cold-drive test ${f} missing — validator cannot run")
        fi
    done
else
    warnings+=("S3: tests dir missing at ${TESTS_DIR} — cold-drive validator unrunnable")
fi

# S4: python3 stdlib modules retrieve.py needs
if command -v python3 >/dev/null 2>&1; then
    if ! python3 -c "import re, json, pathlib, argparse, sys" 2>/dev/null; then
        warnings+=("S4: python3 stdlib check failed — re/json/pathlib/argparse/sys missing")
    fi
fi

# ---- verdict ----

if [[ "${#errors[@]}" -gt 0 ]]; then
    verdict="ABORT"
    exit_code=2
elif [[ "${#warnings[@]}" -gt 0 ]]; then
    verdict="WARN"
    exit_code=1
else
    verdict="CLEAN"
    exit_code=0
fi

if [[ "${JSON}" -eq 1 ]]; then
    # emit JSON to stdout
    printf '{"verdict":"%s","errors":[' "${verdict}"
    sep=""
    for e in "${errors[@]:-}"; do
        [[ -z "${e}" ]] && continue
        printf '%s"%s"' "${sep}" "${e//\"/\\\"}"
        sep=","
    done
    printf '],"warnings":['
    sep=""
    for w in "${warnings[@]:-}"; do
        [[ -z "${w}" ]] && continue
        printf '%s"%s"' "${sep}" "${w//\"/\\\"}"
        sep=","
    done
    printf '],"kb_dir":"%s","kb_entries":%d}\n' "${KB_DIR}" "${entry_count:-0}"
    exit "${exit_code}"
fi

# human-readable output to stderr
if [[ "${verdict}" == "CLEAN" && "${QUIET}" -eq 1 ]]; then
    exit 0
fi

{
    echo "=== /npu-adapt-assist preflight ==="
    echo "  repo root: ${REPO_ROOT:-<not found>}"
    echo "  KB dir:    ${KB_DIR}"
    echo "  KB entries: ${entry_count:-0}"
    echo
    if [[ "${#errors[@]}" -gt 0 ]]; then
        echo "  ERRORS (blocking):"
        for e in "${errors[@]}"; do
            echo "    - ${e}"
        done
    fi
    if [[ "${#warnings[@]}" -gt 0 ]]; then
        echo "  WARNINGS (non-blocking):"
        for w in "${warnings[@]}"; do
            echo "    - ${w}"
        done
    fi
    echo
    echo "  Verdict: ${verdict}"
    case "${verdict}" in
        CLEAN)
            echo "  Next: run retrieve.py with your error trace"
            ;;
        WARN)
            echo "  Next: retrieve.py may proceed but address warnings when possible"
            ;;
        ABORT)
            echo "  Next: FIX ERRORS above before invoking retrieve.py"
            ;;
    esac
} >&2

exit "${exit_code}"
