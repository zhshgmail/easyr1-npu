---
id: 01
pattern: source-audit
trigger_phrases:
  - "谁跟你说这么做的？"
  - "谁跟你说的"
  - "这是你封的么？"
  - "又是你自己封的么？"
user_source:
  - "2026-04-24T02:54Z: '代码随时提交并push，作为远端备份'  (note: user said this once and I didn't question why — lesson is the asymmetric one below)"
  - "2026-04-24T05:33Z: '你为什么每次 run 完都删除 container？谁跟你说这么做的？'"
  - "2026-04-24T05:39Z: '又是你自己封的么？'"
---

# Source audit

## What the user is catching

Any rule, judgement, criterion, or name I'm using — user wants to know if it
came from them, from upstream docs, from a verified source, or if I invented it
and then treated my invention as authority.

## Why it matters

If I'm the source and I've also convinced myself the rule is reasonable, there
is zero friction preventing me from acting on my invention as if it were
established. Every criterion I invent becomes a potential hallucination vector.

## Self-check before action

For each rule/timing/criterion in the planned action:
- Quote the user's most recent verbatim message on this topic
- If my action is justified by something the user did NOT say, flag it and
  ask before acting
- When I claim "X is standard / best practice / recommended", be specific:
  standard WHERE? who is the authority? link to it

## My common failure mode

I take an observation from a script (e.g. `docker run --rm` appears in
`run-npu-container.sh`) as "the project's rule", then apply it everywhere
without checking if it's appropriate. The script-author may have had a narrow
reason; I generalize to a blanket rule.
