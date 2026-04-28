# `docs/_archive/` — 历史归档

已完成 / 过时 / 已被取代的文档。**不要从这里 load 新工作**——这里只保留项目演进的历史痕迹，避免误把过时信息当成现状。

## 归档批次

| 子目录 / 文件 | 归档日期 | 来源 | 备注 |
|---|---|---|---|
| `obsolete_2026-04-28/` | 2026-04-28 | T26 文档整理 | 内部 dev 流程文档（HANDOVER / WORKLOG / NEXT-SESSION-STARTER 等）+ 早期 design / dep-matrix / port-summary 等被 ARCHITECTURE.md 取代的文档 |
| `obsolete_2026-04-25/` | 2026-04-25 | T15-T18 重构 | 旧 `src/skills/vllm/` 被并入 `src/skills/vllm-ascend/port-expert/` |
| `codex-review-skills-audit.md`, `codex-signoff*.md` | 2026-04-19~20 | session 期 codex 审查记录 | 单次 review 产出，不是长期文档 |
| `handoff-2026-04-19.md` | 2026-04-19 | 早期 session handover | 已被后续 HANDOVER.md 多轮迭代取代，HANDOVER.md 自身于 2026-04-28 也归档 |
| `P2-WORKFLOW.md`, `skill-dry-run-2026-04-20.md` | 2026-04-19~20 | 早期 skill 设计 dry-run | 当前 skill 已稳定，dry-run 历史只供回看 |

## 当前活跃文档入口

- 客户面：[`README.md`](../../README.md)、[`ONBOARDING.md`](../../ONBOARDING.md)、[`docs/easyr1/PORT-GUIDE.md`](../easyr1/PORT-GUIDE.md)、[`docs/easyr1/PORT-GUIDE-v2-integrated.md`](../easyr1/PORT-GUIDE-v2-integrated.md)
- 上游维护者面：[`docs/{vllm-ascend,torch-npu,transformers}/PORTING-GUIDE.md`](../)、[`docs/transformers/PR_MATERIAL_v5.4_outcome_A.md`](../transformers/PR_MATERIAL_v5.4_outcome_A.md)
- Skill 使用：[`docs/_meta/SKILLS-USAGE.md`](../_meta/SKILLS-USAGE.md)
- 架构与流程：[`docs/_meta/ARCHITECTURE.md`](../_meta/ARCHITECTURE.md)
- 上游 fork ledger：[`docs/_meta/UPSTREAM_FORKS.md`](../_meta/UPSTREAM_FORKS.md)
- 术语：[`docs/_meta/GLOSSARY.md`](../_meta/GLOSSARY.md)
- KB：[`knowledge/npu-patterns.md`](../../knowledge/npu-patterns.md)、[`docs/_meta/kb/porting_lessons/`](../_meta/kb/porting_lessons/)、[`docs/_meta/kb/challenge_patterns/`](../_meta/kb/challenge_patterns/)
