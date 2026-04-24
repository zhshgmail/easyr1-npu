---
id: 06
pattern: concrete-number-requirement
trigger_phrases:
  - "哪个数字支撑？"
  - "就像 torch 和 Transformer 应该定义他们那个级别的 benchmark 一样"
user_source:
  - "2026-04-24T04:04Z: '你先停一下，是否应该定义一个 inference 方面的 benchmark，而不是训练的？'"
  - "2026-04-24T04:05Z: '就像 torch 和 Transformer 应该定义他们那个级别的 benchmark 一样。最后这些依赖都做完，再到 rl 那层去看那层的 benchmark'"
---

# Concrete-number requirement

## What the user is catching

I'm saying "PASS / works / can import / runs" without a number. User knows
these words can hide failures, while a concrete number forces honesty.

## Why it matters

"Smoke PASS" means different things to me in different contexts (sometimes
"marker matched", sometimes "non-empty output", sometimes "exit 0"). A
concrete number (e.g. `21/21 ops PASS`, `entropy_loss=1.275 in band [1.21, 1.34]`,
`99% token match vs ref`) has one meaning.

## Self-check before action

For any "PASS / works" claim:
1. What exact number supports the claim? (not a description)
2. What's the reference baseline the number is compared to?
3. What's the tolerance / band? Is my number inside it?
4. If I have no number, I have no claim.

## My common failure mode

I accept "no error, rollout produced some tokens" as PASS without looking at
what tokens came out. Later I discover the tokens were nonsense ( iter 15
bit-exact failure hidden under V1.3 "PASS marker"). Lesson: always compute
a number that would change if the thing I'm checking broke.
