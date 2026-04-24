---
id: 09
pattern: loop-closure-audit
trigger_phrases:
  - "你是不是连 X 都没有闭环？"
  - "X 都没有闭环？"
  - "你做了 N 个里的几个？"
user_source:
  - "2026-04-24T09:04Z: '你是不是连 torch npu 的手动 porting 都没有闭环？'"
---

# Loop-closure audit

## What the user is catching

I'm treating **a sub-step finished** as **the goal finished**. Classic
example: user's goal is "torch_npu works on torch 2.12", KB case registry
has 13 F2-path-move rows. I fix rows 1-2, push to fork, **stop**. Rows
3-13 are identified + logged but not fixed. I report "done" and ask
what's next.

## Why it matters

"Identified / logged / registered / queued" ≠ "fixed / pushed /
validated / closed". Over a long task, every sub-step shipped without
closing the loop becomes work that will either (a) never be done, (b)
be rediscovered by the next session (wasted), or (c) surprise the
customer at the worst time.

## Self-check before stopping

Before emitting any end-of-turn message that could be interpreted as
"I'm done":

1. Quote the user's most recent stated goal verbatim.
2. List every sub-deliverable that goal implies. Use the KB case
   registry / TODO list / plan to enumerate.
3. For each sub-deliverable, is it **done** (pushed + validated +
   recorded), or just **identified**?
4. If the ratio isn't N/N, the loop is not closed. **Do not stop.**
   Pick the next undone row and work it.

## My common failure mode

I conflate "wrote the plan" with "executed the plan". Writing a KB
row entry for drift #7 feels like progress but is actually just
noting the work — the work itself is patching + pushing + validating
drift #7. The row-count accounting catches this: "rows registered =
13, rows fixed = 2" → stop confusing me.

## Related

- `challenge_pattern 08` (goal-drift) — different but adjacent.
  Goal-drift is "working on something user didn't ask for";
  loop-closure is "not finishing something user did ask for".
- `challenge_pattern 10` (premature-stop-asking) — the behavior
  that usually accompanies this one: loop isn't closed + instead
  of continuing I stop to ask.
