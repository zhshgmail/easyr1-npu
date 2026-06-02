#!/usr/bin/env python3
"""Ship-claim audit — PreToolUse hook on outbound Discord messages (easyr1-npu).

Adapted from a5_ops `src/scripts/workflow/ship_claim_audit.py` (the safety-net the
owner pointed to 2026-06-02). Purpose: a MECHANICAL gate on the only surface that
reaches the human (Discord reply/edit), so the main-session agent cannot ship
"win"-style or substitution claims without an evidence anchor. Memories are passive
(they only help if recalled); this hook fires unconditionally before the message goes
out.

Why this exists (this session's evidence)
==========================================
2026-06-02 the owner caught, in one sitting: (a) "real delta drives RL" overclaim
(mechanics-not-meaning), (b) dropped PR links + the only merged row in a rewrite,
(c) "#251 landed" (comment-posted ≠ resolved), (d) all results were V4-Flash while the
user asked for V4-Pro — an unconsented target substitution, same family as V3.2→V4.
Every one surfaced only because the user asked. The agent lacked an automated gate. This
is it.

Decision rule
=============
1. tool not in MONITORED_TOOLS → exit 0 (allow).
2. empty text → exit 0.
3. TARGET-SUBSTITUTION check: if text makes a result/claim about a DeepSeek-V4 variant
   (mentions V4/DSV4/DeepseekV4 + a win token) it must NOT be silent about which variant.
   Require an explicit variant word (Flash / Pro) OR an explicit "Flash≠Pro" / variant
   disclaimer token. Otherwise block (the V4-Flash-vs-Pro lesson). Fail-closed.
4. WIN-CLAIM check: if any FORBIDDEN_WIN_TOKEN present, require an EVIDENCE ANCHOR in the
   same message — one of: a commit SHA reachable from origin/main, a log/file path, an
   "N/N" with a method word, or an explicit honest-boundary token (诚实/honest/未验证/
   N/A/待补/spec-matched/plumbing-only). Otherwise block.
5. else exit 0.

Fail-open only on hook-internal errors (malformed stdin / missing git) so a hook bug
never blocks legitimate work; fail-CLOSED on win/substitution claims (a falsely-allowed
claim to the human is worse than a falsely-blocked one — rewrite the message).
"""
import json
import os
import re
import subprocess
import sys

MONITORED_TOOLS = {
    "mcp__plugin_discord_discord__reply",
    "mcp__plugin_discord_discord__edit_message",
}

# "win"-style language that, alone, has burned the owner this session/before.
FORBIDDEN_WIN_TOKENS = [
    "✅", "PASS", "passed", "verified", "已验证", "跑通", "搞定", "done", "DONE",
    "merged", "已合入", "landed", "wired", "已接入", "bit-exact", "逐位等价",
    "end-to-end", "e2e 跑通", "闭环", "training-specific", "REAL", "real-run",
    "production-ready", "verified-run",
]

# Evidence anchors that justify a win token. Any ONE present → allow.
#  - a verified commit SHA (checked against origin/main below)
#  - a log / file path the claim points to
#  - an N/N with a method word (precision/perf evidence)
#  - an explicit honesty-boundary token (the claim is already hedged)
EVIDENCE_PATH_RE = re.compile(r"(workspace/|docs/|output/|src/|\.py|\.log|\.txt|\.md|\.json|\.pt)")
EVIDENCE_NN_RE = re.compile(r"\b\d+\s*/\s*\d+\b")
HONESTY_TOKENS = [
    "诚实", "honest", "未验证", "未捕获", "未接入", "N/A", "待补", "spec-matched",
    "coverage-confirmed", "plumbing-only", "plumbing 闭合", "撤回", "retract",
    "不是", "并非", "不能", "FAIL", "BLOCKED", "阻塞", "负收益", "min 0.", "shape-dependent",
    "减层", "reduced", "stand-in", "替身", "占位", "synth",
]

# Target-substitution guard: V4 family mentions that carry a win token must name the variant.
V4_MENTION_RE = re.compile(r"(DeepseekV4|DSV4|DSv4|V4[- ]?Flash|V4[- ]?Pro|V4\b|DeepSeek-V4)", re.IGNORECASE)
VARIANT_OK_TOKENS = ["Flash", "Pro", "Flash≠Pro", "Flash/Pro", "变体", "并非 Pro", "不是 Pro", "非 Pro"]

SHA_RE = re.compile(r"\b[0-9a-f]{7,40}\b")


def _is_ancestor(sha: str, project_dir: str) -> bool:
    try:
        r = subprocess.run(
            ["git", "-C", project_dir, "merge-base", "--is-ancestor", sha, "origin/main"],
            capture_output=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def _has_verified_sha(text: str, project_dir: str) -> bool:
    for cand in SHA_RE.findall(text):
        if _is_ancestor(cand, project_dir):
            return True
    return False


def _block(msg: str) -> int:
    sys.stderr.write("SHIP-CLAIM AUDIT — message BLOCKED\n" + msg + "\n")
    return 2


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # fail-open on hook bug

    tool = payload.get("tool_name", "")
    if tool not in MONITORED_TOOLS:
        return 0
    text = (payload.get("tool_input", {}) or {}).get("text", "") or ""
    if not text.strip():
        return 0

    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", "")

    # --- Check 1: target-substitution (V4 variant) ---
    win_present = any(tok in text for tok in FORBIDDEN_WIN_TOKENS)
    if win_present and V4_MENTION_RE.search(text):
        if not any(v in text for v in VARIANT_OK_TOKENS):
            return _block(
                "A V4-family result/claim with win-language must NAME THE VARIANT "
                "(Flash vs Pro) or disclaim it. Silent 'V4/DSV4' = the V4-Flash-vs-Pro "
                "substitution trap (user asked Pro, prior work was Flash). Add 'Flash' / "
                "'Pro' / a variant disclaimer, then resend."
            )

    # --- Check 2: win-claim needs an evidence anchor ---
    if win_present:
        has_anchor = (
            _has_verified_sha(text, project_dir)
            or bool(EVIDENCE_PATH_RE.search(text))
            or bool(EVIDENCE_NN_RE.search(text))
            or any(tok in text for tok in HONESTY_TOKENS)
        )
        if not has_anchor:
            hits = [t for t in FORBIDDEN_WIN_TOKENS if t in text]
            return _block(
                f"win-language {hits[:5]} present with NO evidence anchor. Add ONE of: "
                f"a commit SHA on origin/main · a file/log path the claim points to · an "
                f"N/N count · an explicit honesty boundary (诚实/未验证/N/A/spec-matched/…). "
                f"This is the mechanics-vs-meaning / overclaim guard."
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
