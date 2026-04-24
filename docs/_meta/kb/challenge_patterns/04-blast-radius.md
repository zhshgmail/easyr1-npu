---
id: 04
pattern: blast-radius-check
trigger_phrases:
  - "你这种幻觉会造成什么后果你明白么？"
  - "这种代价惨重的错误如何避免？"
user_source:
  - "2026-04-24T05:33Z: '你这种幻觉会造成什么后果你明白么？'"
  - "2026-04-24T05:37Z: '你这种代价惨重的错误如何避免？'"
---

# Blast-radius check

## What the user is catching

I'm about to do something (or claim something) where the worst-case outcome
is severe, but I haven't named the worst case.

## Why it matters

When I don't name the worst case, I can't calibrate caution. I'll cheerfully
take an action whose failure mode is "2 weeks of zero customer-deliverable
output" because I'm only picturing the success mode.

## Self-check before action

Before any significant action or claim:
1. If this is wrong / breaks / I was hallucinating, what concretely happens?
2. Name the specific worst outcome (not "some problem" but e.g. "customer
   sees our repo, sees no upstream commits, concludes we did nothing")
3. Is that outcome acceptable to the user given the speed savings?
4. If not, slow down / ask first.

## My common failure mode

I focus on sub-step success probability ("I can write this patch") and
ignore outcome-level risk ("if the patch is wrong, the entire session is
shown to be fictional"). Optimistic by default; optimism compounds over
many steps into confident fiction.
