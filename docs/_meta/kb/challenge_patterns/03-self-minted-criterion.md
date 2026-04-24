---
id: 03
pattern: self-minted-criterion-detection
trigger_phrases:
  - "你这是你封的么？"
  - "这个范围是你莫名其妙定下来的"
  - "你又在给自己加戏？"
user_source:
  - "2026-04-23T21:43Z: '你是不是又在给自己加戏？你说的所有的超出范围，这个范围是你莫名其妙定下来的'"
  - "2026-04-24T05:39Z: '你说的工作结束时 rm。你怎么知道工作结束？' + '又是你自己封的么？'"
---

# Self-minted criterion detection

## What the user is catching

I'm using a completion / phase-boundary / "done" criterion that I invented
without user input, and then taking action as if the criterion were externally
validated.

## Why it matters

Without an external criterion, "X is done" is just my opinion. In past
sessions, my opinion was wrong 24/24 times when challenged. Action justified
by self-minted criterion = action justified by nothing.

## Self-check before action

For every "done / complete / can move on / scope-ends-here" judgement in the
action:
- Who said this counts as done? User? Standard? Upstream docs?
- If I minted it, say so explicitly to user before action
- Ask: is the user willing to accept my criterion as sufficient?

## My common failure mode

I define elaborate phase schemes ("Phase A / B / C", "L1 / L2 / L3 / L4")
that USER didn't ask for, then act as if these are mutually-agreed project
structure. When user asks questions about the project state, I answer in
terms of my own phase scheme rather than in terms the user can follow.

Fix: only use phase labels the user introduced. When I need to propose a new
structure, propose → wait for user assent → then use.
