#!/bin/bash
# critic-on-significant-commit.sh
# Hook script triggered after every Bash `git commit *`.
# Inspects the commit message; if it looks like a "significant" commit
# (feat/fix that touches user-facing or upstream-port code), emits a
# stdout marker that surfaces in Claude's transcript so the assistant
# can decide to invoke /porting-self-challenge.
#
# Non-blocking: never exits non-zero.

set -u

cd "${CLAUDE_PROJECT_DIR:-$(pwd)}" || exit 0
MSG="$(git log -1 --pretty=%B 2>/dev/null)" || exit 0
SUBJECT="$(printf '%s\n' "$MSG" | head -1)"

# Significant if subject starts with feat/fix/perf/refactor on a non-trivial scope,
# or touches a port-expert SKILL.md / KB / fork branch.
significant=0
case "$SUBJECT" in
  feat:*|feat\(*|fix:*|fix\(*|perf:*|perf\(*) significant=1 ;;
esac

# Or if the commit touched any SKILL.md / KB / port-expert reference
if git show --stat HEAD 2>/dev/null \
   | grep -qE 'SKILL\.md|porting_lessons/|challenge_patterns/|references/KB_INDEX|src/skills/.*/port-expert'; then
  significant=1
fi

if [ "$significant" -eq 1 ]; then
  cat <<EOF
[CRITIC-HOOK] Commit "${SUBJECT}" looks significant. Consider running /porting-self-challenge before declaring this work delivered.
EOF
fi

exit 0
