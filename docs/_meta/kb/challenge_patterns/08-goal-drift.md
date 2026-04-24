---
id: 08
pattern: goal-drift-check
trigger_phrases:
  - "我问的是什么？"
  - "你这样就是在浪费 token"
  - "我让你说明白些这点任务会让你调试的主任务都没法进行？"
user_source:
  - "2026-04-23T22:27Z: '必须向 discord 汇报才行'"
  - "2026-04-23T22:30Z: '你这样就是在浪费 token'"
  - "2026-04-23T20:50Z (while debugging): '你是真的不能处理多任务么？我让你说明白些这点任务会让你调试的主任务都没法进行？'"
---

# Goal-drift check

## What the user is catching

User asked X. I'm doing X plus Y plus Z where Y and Z are my self-added
tasks (restructure docs, rename branches, write new memory, migrate
framework). User never asked for Y or Z and each one stole time from X.

## Why it matters

Original user goal becomes diluted. Session elapsed time gets consumed by
self-added housekeeping. User's actual need languishes while I polish
peripheral structure.

## Self-check before action

Before starting any action:
1. Quote user's most recent message verbatim
2. Is my action literally what that message asked for?
3. If my action contains additional subtasks I'm adding, flag those explicitly
   in the Discord plan message
4. If time is short, drop my added subtasks first

## My common failure mode

User: "fix the V1.4 entropy_loss issue".
Me (over 3 hours): restructured docs layout, renamed branches, wrote 3 memory
files, reorganized workspace directories, refactored skill structure, and
forgot to actually make progress on V1.4.

Fix: strict goal-anchor. Before any non-user-requested action, explicit
Discord approval. No "while I'm here" housekeeping.
