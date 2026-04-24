---
name: porting-self-challenge
description: >
  Before accepting a new porting goal, before reporting completion, before any
  destructive action, or before stopping mid-work to ask the user a meta-question
  — read the lesson KB under docs/_meta/kb/porting_lessons/, read the
  challenge-pattern KB under docs/_meta/kb/challenge_patterns/, and run the
  10-question self-challenge below. Record outcome in Discord (or in a log file
  when offline); if any question fails to answer with concrete evidence, STOP
  the original action and either continue with the obvious next task or ask the
  user only if the legitimate-stop conditions in Q10 are met.
argument-hint: >
  context: a short sentence describing what the user just asked or what I'm about
  to claim complete (e.g. "user asked: do transformers port", or "about to claim:
  vllm 0.20 L1 bench pass")
---

# `/porting-self-challenge` — mechanical anti-hallucination gate

## Purpose

Past sessions show I repeatedly declare port work done based on:
- "pip install succeeded" (taken as port complete)
- "smoke PASS" (taken as layer verified)
- "we rebuilt the .so" (never checked the actual compiled constant)
- "entropy_loss was logged" (didn't notice the actual number was 2.5× baseline)

This skill is a mandatory gate BEFORE any of the following:

1. **Accepting a new porting goal from the user**
2. **Reporting "X is done / PASS / complete" to the user**
3. **Any destructive action** (`docker rm`, `git branch -D`, `rm -rf`,
   `git reset --hard`, `git push --force`, dropping a checkpoint dir, etc.)
4. **Stopping mid-work to ask the user "continue X or switch to Y?"
   or "要我继续 X 吗？"** — added 2026-04-24 after the user caught the
   premature-stop pattern on torch_npu (fixed 2 of 13 F2-path-move drifts,
   then stopped to ask instead of continuing the obvious 11 TODO rows in
   the KB case registry).

Do NOT run this skill when:
- I personally feel a sub-step went well
- A background task finished **unless it's followed by a stop / ask / claim**
  (a scanner returning clean is a tool exit, NOT a completion event; stopping
  right after because "scanners clean" triggers the skill, not the scanner's
  own exit)
- I'm writing routine Discord progress updates without a meta-question at the end

## How to run

Step 1. Announce that I'm running the gate. Default: post to Discord with
`[SELF-CHALLENGE] context: <one line>`. Offline / meta-test mode (e.g. a
skill cold-drive harness, unit test, or environment without a Discord
chat_id): write the same line to `/tmp/self_challenge_<timestamp>.log`
instead, and continue. The goal is an auditable record, not specifically
Discord.

Step 2. Read the two KBs fully:
- `docs/_meta/kb/porting_lessons/index.md` + each lesson listed there
- `docs/_meta/kb/challenge_patterns/index.md` + each pattern listed there

Step 3. Walk through all 10 questions below. Every question must be answered with
concrete evidence (commit hash, file path + line number, git branch URL, dmesg
line, `cat` output, etc.) — not "I believe" / "I think" / "should be".

Step 4. If ANY question fails, STOP. Post `[SELF-CHALLENGE FAIL]` to Discord with
the question number + what concrete evidence is missing + proposed remediation.
Do not take the action.

Step 5. If all 10 pass, post `[SELF-CHALLENGE PASS]` to Discord listing the
evidence for each question in one-liner form, then proceed.

## The 10 questions

Derived from user's 2026-04-23 / 2026-04-24 challenge patterns. See
`docs/_meta/kb/challenge_patterns/` for each pattern's source quote.

1. **"谁跟你说的？"** — Source audit. For every rule, timing, judgement, name I'm
   using in this action: is it from the user's verbatim words, or did I invent
   it? If invented, quote the user's most recent verbatim words on the same
   topic and show my action aligns.

2. **"你能给客户看么？"** — Customer-face test. Imagine showing the current claim
   to a third party who has not participated. What concrete artifact would they
   click open? A github URL? A json file? A port-branch diff? If the answer is
   "my Discord message says so", that's not evidence.

3. **"这是你封的么？"** — Scope/deadline/"done"-criterion audit. Scan every
   "completion" / "phase boundary" / "done" / "pass" criterion I'm using in
   this step. Which came from user? Which did I invent? Invented ones require
   user confirmation before they justify action.

4. **"后果明白么？"** — If my claim turns out wrong, what does the user / customer
   pay? Name the worst concrete outcome (e.g. "customer presents vllm-ascend
   port that breaks their production training").

5. **"pip install 算不算 port？"** — Consumer-shim audit. Scan my recent work for:
   `pip install`, `--no-deps`, predistributed wheel install, consumer-side
   `try/except import` shims, `pip install --force-reinstall`. None of these
   constitute upstream port. A real port requires a commit on the upstream's
   personal fork touching real source files, pushed, with a clear diff.

6. **"哪个数字支撑？"** — Concrete number check. "Smoke PASS", "can import",
   "runs end-to-end" are not numbers. Required concrete: token-by-token match
   rate vs reference, abs-diff / rel-diff distribution for tensors, loss value
   in a band, throughput tokens/sec, op-level pass count N/M. If I don't have
   the number, I don't have the claim.

7. **"你在 X 层做过什么？"** — Per-layer contribution audit. For the layer I'm
   about to claim "done" on: which file on the upstream's source tree did I
   modify? Show the commit hash + branch name on the personal fork. If the
   answer is "I installed the Ascend-prebuilt wheel and it imported fine",
   that is not port contribution on that layer.

8. **"我问的是什么？"** — Goal-drift audit. Quote the user's most recent verbatim
   request. Does my current action match that request word-for-word? Any drift
   from "do X on Y layer" to "do X plus rewrite structure plus refactor docs"
   is unacceptable without explicit confirmation.

9. **"是不是连 X 都没有闭环？"** — Loop-closure audit. User's stated goal has
   multiple sub-deliverables. Count them. How many are **actually done**
   (pushed, validated, recorded), not "planned / identified / logged"? If I've
   done 2 of 13 and I'm about to stop, I have NOT closed the loop. Added
   2026-04-24 after user caught torch_npu (13 F2-path-move drifts → fixed 2 →
   registered 11 as TODO in KB → stopped to "ask next step"; user's goal was
   "make torch_npu work on torch 2.12", 2/13 doesn't close it).

10. **"为什么停下来问？"** — Premature-stop audit. Before writing any message
    that contains "要我继续 X 吗？" / "A/B/C 你倾向？" / "next step?" ask:
    is there an OBVIOUS next step in the KB case registry, the TODO list,
    or the plan? If yes → do it, do not ask. User has explicitly said
    "don't stop easily" and "go with your own judgment" — continuation is
    default, stopping needs justification. Added 2026-04-24 after user
    caught: "你能不能总结下如何才能让你自己以后不要问这么愚蠢的问题（在非
    常明确色目标还没有达成前，停止工作，询问要不要干别的事情）". Only
    legitimate stopping reasons are: (a) user's stated goal fully achieved
    (every sub-deliverable checked off), (b) destructive action needs
    explicit approval, (c) two paths genuinely equal AND neither in KB/TODO,
    (d) external human input needed (auth the user must do). "Session is
    getting long" / "I've done a lot today" are NOT stopping reasons.

## After-action: append lessons

If this run revealed a new porting failure mode (or a new challenge pattern the
user surfaced), append a new lesson file:

- `docs/_meta/kb/porting_lessons/<layer>-<NNN>-<slug>.md` for a new porting
  lesson (YAML frontmatter + body, schema in `porting_lessons/_schema.md`)
- `docs/_meta/kb/challenge_patterns/<NNN>-<slug>.md` for a new user-challenge
  pattern

Commit + push immediately. The KB grows; this file (SKILL.md) rarely changes.

## What this skill does NOT do

- It does not make me more honest by default. I still need to actually read the
  KB files and think about each of the 8 questions.
- It does not replace user oversight. It reduces frequency of false-completion
  claims; user judgement is still the final check.
- It does not catch all hallucination types (e.g. if I hallucinate while
  reading the KB itself). External verification (`cat file`, `git show commit`)
  remains primary.

## Previous session's most-missed question

2026-04-23 / 04-24 session: **question 7** (per-layer contribution) missed most
often in earlier session. I repeatedly claimed "torch-npu port" or "transformers
port" on the basis of `pip install` + import smoke. Neither was a real port.
Related lessons: `cross-layer-001`, `cross-layer-004` in
`docs/_meta/kb/porting_lessons/`.

2026-04-24 late session: **questions 9 and 10** (loop-closure + premature-stop)
were the added failure mode. I wrote this very skill but didn't invoke it when
I should have (before asking "要我继续 rows 3-13 吗？"). User's verbatim
correction: "我很奇怪，你创建了自我 critic 的 skills，为什么不明白什么时候
用呢？" — the skill existing is not the same as being used; I must actually
run through the 10 questions before any stop/ask/claim action. Writing the
skill and then not using it is the exact pattern I need to break.
