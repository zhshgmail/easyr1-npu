#!/bin/bash
# critic-on-significant-commit.sh
# Hook script triggered after every Bash `git commit *`.
#
# Two responsibilities:
#   1. Inspect commit message + diff; emit `[CRITIC-HOOK]` marker on
#      "significant" commits (feat/fix/perf, or SKILL.md/KB/port-expert
#      touches) so the assistant considers running /porting-self-challenge.
#   2. Run the sanity test suite (scripts/run_sanity_suite.sh) and emit
#      its PASS/FAIL summary. Non-blocking — never exits non-zero.

set -u

cd "${CLAUDE_PROJECT_DIR:-$(pwd)}" || exit 0
MSG="$(git log -1 --pretty=%B 2>/dev/null)" || exit 0
SUBJECT="$(printf '%s\n' "$MSG" | head -1)"

# --- 1. Significance heuristic ---------------------------------------------
significant=0
case "$SUBJECT" in
  feat:*|feat\(*|fix:*|fix\(*|perf:*|perf\(*) significant=1 ;;
esac

if git show --stat HEAD 2>/dev/null \
   | grep -qE 'SKILL\.md|porting_lessons/|challenge_patterns/|references/KB_INDEX|src/skills/.*/port-expert'; then
  significant=1
fi

if [ "$significant" -eq 1 ]; then
  cat <<EOF
[CRITIC-HOOK] Commit "${SUBJECT}" looks significant. Consider running /porting-self-challenge before declaring this work delivered.
EOF
fi

# --- 2. Sanity suite (P0h.1) ------------------------------------------------
# Run only if the script exists; emit a one-line summary to the
# transcript regardless of outcome. Failures do NOT block the commit
# (it already happened) — but they surface so the assistant can react.
SANITY="scripts/run_sanity_suite.sh"
if [ -x "$SANITY" ]; then
  if SANITY_OUT="$(bash "$SANITY" 2>&1)"; then
    SUMMARY="$(printf '%s\n' "$SANITY_OUT" | grep -E '^=+ SANITY SUITE: ' || echo 'sanity suite: ran (no summary)')"
    cat <<EOF
[SANITY-HOOK] ${SUMMARY}
EOF
  else
    # Failed sanity — surface the test summary + last failures.
    SUMMARY="$(printf '%s\n' "$SANITY_OUT" | grep -E '^=+ SANITY SUITE: |^FAILED ' | head -10)"
    cat <<EOF
[SANITY-HOOK] FAILED — review and fix:
${SUMMARY}
EOF
  fi
fi

exit 0
