# Postmortem: <one-line title> — <YYYY-MM-DD>

> **Template usage**: 复制本文件 → `<YYYY-MM-DD>_<short_slug>.md` → 填字段 → commit。
>
> **此模板借鉴自** a5_ops `docs/postmortem/SAFETY_NET_NAME_COUPLING_2026_05_14.md`。
>
> **写 postmortem 的目的**: 把一次事故的**根本原因** + **多层防御为什么没接住**编码进可检索的文档，让未来 agent / engineer 不能忘记。**不是**总结"做错了什么"——而是回答"为什么所有应该接住的层都没接住"。

---

**Date**: YYYY-MM-DD
**Severity**: HIGH / MEDIUM / LOW
**Cost**: <hours wasted / commits leaked / $ if relevant>
**Reporter**: <user catch | self-discovery during cold-drive | sanity-suite failure | codex review>
**Status**: <fix landed commit | followup pending>

## Summary

2-3 段说事故 + 损失。是什么被错误地标为 PASS / 什么 fraud 漏过了哪些层 / 影响范围。

## What happened (timeline)

按时间顺序 5-10 行。每行 `HH:MM<Z|local>: <event>`。

- HH:MM: trigger event
- HH:MM: first sign of problem
- HH:MM: detection
- HH:MM: fix landed
- HH:MM: validation

## Root cause (architectural, not "agent forgot")

明确区分 **immediate cause**（直接症状）vs **root cause**（架构层的）。

避免：
- "agent should have checked X" → 这是 process gap，不是 root cause
- "we forgot to verify Y" → 没机制 = 这就是 root cause（机制没建）

举例：
- ✓ "Safety net 硬编码文件名，新 mode 引入新文件名时 silent no-op"
- ✗ "Worker 没有验证文件名是否对" (agent-blame，不是 architectural)

## How it slipped past every layer

表格：列每一层应该 catch 但没 catch + **每层没 catch 的原因**。

| Layer | What it was supposed to do | What actually happened |
|---|---|---|
| (e.g. claim_manifest schema) | (e.g. require evidence array non-empty) | (e.g. evidence was empty but outcome was C_REPORT which allows empty) |
| (e.g. P0i scanner) | (e.g. anti-cycle check) | (e.g. verifier path resolution wrong, file appeared "missing", check skipped) |
| (e.g. P0j gate) | ... | ... |
| ... | ... | ... |

**关键问题**: 是不是每层都在"信前一层的 PASS"? 如果是 → defense-in-depth 失败的标准模式。

## Fix (landed <commit>)

按层列 fix。每条 fix 必须改**机制**（代码 / gate / hook / KB），不只是 process。

- **KB rule**: 新 OL-N 或 NPU-XXX-N
- **Gate**: 新 GateID 或现有 gate 加判定
- **Scanner**: 新 check
- **Test**: 新 regression test fixture（gate_bad 或 schema_bad）
- **Doc**: 新 ANTI_PRESSURE Px / 新 CLAUDE.md 规则
- **Hook**: 新 hook 或扩展现有 hook

## What we couldn't fix yet

明确的 followup 清单。**禁止** "以后注意" / "下次记得"——这些不算 fix。

可接受形式：
- "需要 P0e workflow critic（DEBT-5）才能闭合 X 类 bypass"
- "需要第二个 LLM backend（DEBT-8）做盲点 detect"

## Lessons (永久记录)

按 numbered list 写。**禁止** "carefully read X next time" / "be more careful"。

可接受形式：
- "Verifier 必须 data-flow-independent 于它要 verify 的 claim 文件 — codified as M2 anti-cycle in scanner"
- "Per-X cap 必须有 aggregate-X cap，否则 per-X cap 被同类多次叠加绕过 — codified as M3 in /npu-port design"

每条 lesson 必须 cite **codification location**（什么文件 / 哪条规则记下了这个）。如果 lesson 没 codify，下一次 agent 不会知道。

---

## 见也

- 关联 KB 条目: <OL-XX / NPU-XXX-N>
- 关联 ROADMAP item: <P0xx / DEBT-N>
- 关联 ANTI_PRESSURE protocol: <PN>
- 关联 challenge_pattern: <NN>
- 关联 cross-layer lesson: <docs/_meta/kb/porting_lessons/XXX-YYY.md>
