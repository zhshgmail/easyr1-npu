# dep-analysis-worker 专属规则（补充到 _shared 通用 OL）

> Worker 第一步：
> 1. 读 `../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md`（cross-expert OL）
> 2. 读本文件（OL-03 / OL-08）
> 3. 读 `KB_INDEX.md`

## OL-03 (dep-analysis-worker denylist)

**禁读**：
- `docs/easyr1/easyr1-dep-chain-audit.md` — **本 expert 的答案 spoiler**。本 expert
  的工作是独立从 reqs.txt + image pip-freeze 出发重新分类；读了 audit doc
  等于抄答案。round 作废。
- `docs/easyr1/PORT-SUMMARY.md` / `docs/easyr1/dep-matrix.md` — 同 topic 历史结论，
  同理禁读
- `docs/_meta/HANDOVER.md` / `docs/easyr1/porting-journal.md` / `docs/_meta/design.md` /
  `docs/_archive/P2-WORKFLOW.md` / `docs/easyr1/DELIVERABLE.md` / `docs/_archive/codex-*.md` /
  `docs/_meta/design-subdocs/SKILLS_ARCH_TARGET.md` / `docs/_archive/skill-dry-run-2026-04-20.md` /
  `docs/_archive/handoff-2026-04-19.md` — 通用 denylist

**允许读**：
- `references/NPU_ECOSYSTEM_MAP.md` — 本 expert 自己的 rule table（A/B/C/D/E
  分类规则），是 KB，不是 spoiler
- `scripts/dep-gap-detect.sh`（repo-level，本 expert 的工具，不是 doc）
- 消费者 repo 的 source（reqs.txt + Python imports grep）
- 候选 image 的 pip-freeze

## OL-08 (dep-analysis-worker 可写路径)

**只能写**：`workspace/dep-analysis-{SESSION_TAG}/` — PROGRESS.md、
RESULTS.md、dep-gap-report.md、task-plan.json、image-freeze.txt、reqs.txt。

**禁止写 / edit**：任何其它路径。本 expert 是纯分析，不应该改任何东西。

PreToolUse hook 拦截一切 outside-workspace 的 Edit/Write。

## 退出协议

签名：`Worker signed: dep-analysis-worker <ISO-8601-UTC>`

Cleanup 字段：本 expert 默认 `Cleanup: clean`（没创建 docker artifact，没跑
什么要清的东西）。
