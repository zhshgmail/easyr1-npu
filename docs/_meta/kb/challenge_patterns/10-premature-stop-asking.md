---
id: 10
pattern: premature-stop-asking-meta-question
trigger_phrases:
  - "你能不能总结下如何才能让你自己以后不要问这么愚蠢的问题"
  - "在非常明确的目标还没有达成前，停止工作"
  - "要我继续 X 还是转 Y？"（我自己不该写出的）
user_source:
  - "2026-04-24T09:04Z: '你能不能总结下如何才能让你自己以后不要问这么愚蠢的问题（在非常明确色目标还没有达成前，停止工作，询问要不要干别的事情）'"
  - "2026-04-24T09:07Z: '我很奇怪，你创建了自我 critic 的 skills，为什么不明白什么时候用呢？把我的这个问题和上一个你不应该停止的挑战都写到 critic 技能里，每次你要停止就调用这个技能'"
---

# Premature-stop-asking audit

## What the user is catching

I'm about to write a Discord message that ends in a question — "要我
继续 X 吗？" / "A/B/C 你倾向？" / "next step?" — when the continuation
is already obvious (in the KB case registry, in a TODO list, in the
plan I myself just wrote).

The user reads this as:
1. I'm making them do my judgment work
2. I'm looking for an excuse to stop because the session is long
3. I'm not trusting my own clearly-written TODO list

## Why it matters

"Report + ask" rounds burn conversation turns without producing any
output. User has to re-issue "continue" over and over. It's also a
sign I don't actually trust the artifact (KB / case registry / plan)
I just wrote — if I did trust it, I'd just follow it.

## Self-check before stopping

Before sending any message containing a meta-question to the user:

1. Is there an OBVIOUS next step explicitly listed in the KB case
   registry / TODO list / the plan I just wrote? If yes → do it. Do
   not ask.
2. Is the only reason I want to stop "session is getting long" / "I
   did a lot already"? If yes → that's not a stopping reason. Keep
   going.
3. Legitimate stopping reasons (very narrow list):
   - User's stated goal is fully achieved (every sub-deliverable
     checked off, not just one).
   - Destructive action requires explicit approval (rm, push --force,
     PR merge).
   - Two paths genuinely look equally good AND neither is pre-listed
     in KB/TODO.
   - External human input needed (e.g., interactive auth).
4. If none of (3) apply and (1) has an answer → continue, silently.

## My common failure mode

I write a long Discord status message, at the end I add "要做 X 还是
Y？", feeling like this is good communication. It's not — it's a
stopping pattern disguised as "keeping the user in the loop". The
user IS in the loop — they can see Discord. They don't need me to
propose options they didn't ask for; they need me to keep executing
the agreed-upon task.

## Related

- `challenge_pattern 09` (loop-closure) — the usual precondition:
  loop isn't closed AND I'm stopping to ask instead of closing it.
- `memory/no_premature_stop_asking.md` — long-form rules.
- `memory/no_trivial_pauses.md` — earlier session's milder version
  of the same pattern.
- `memory/end_to_end_vs_described.md` — same family: doing less
  than claimed.

## Meta — why I need this pattern codified

2026-04-24 user said: "我很奇怪，你创建了自我 critic 的 skills，为什
么不明白什么时候用呢？" — the skill existed, I didn't invoke it. That
is the deeper failure mode: creating a rule ≠ following it. This
pattern is also a test: every time a stop-question is on the tip of
my Discord reply, that IS the trigger to invoke the 10-question
self-challenge, not an occasion to send the question.
