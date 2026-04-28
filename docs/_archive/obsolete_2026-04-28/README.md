# `obsolete_2026-04-28/` — 2026-04-28 T26 文档整理归档

T26 整理时把以下两类文件搬过来：

## 1. 内部 dev 流程文档（不是给客户的）

- `HANDOVER.md` — session 间交接说明，含大量未对外的工作进度
- `WORKLOG.md` — session 内时序工作日志
- `NEXT-SESSION-STARTER.md` — 下一 session cold-start checklist
- `MODULE-PORT-STATUS.md` — 已被 [`UPSTREAM_FORKS.md`](../../_meta/UPSTREAM_FORKS.md) 取代
- `cold-start-pass-criteria.md`, `e2e-validation-spec.md`, `BENCHMARK-LAYERS.md` — 内部验证规范（属于"我们怎么 review 自己"，不是产出物）

## 2. 已被 ARCHITECTURE.md / 当前 KB 取代的早期产物

- `design.md` — 早期设计文档（2026-04-17 写）；当前架构在 [`docs/_meta/ARCHITECTURE.md`](../../_meta/ARCHITECTURE.md)
- `skills-design.md` — 显式标注 "⚠️ SUPERSEDED 2026-04-23"
- `design-subdocs/SKILLS_ARCH_TARGET.md` — 同上的 supersede target，已 fold 入新 ARCHITECTURE
- `SKILLS-GUIDE.md` (370 行) — 与 [`docs/_meta/SKILLS-USAGE.md`](../../_meta/SKILLS-USAGE.md) (57 行) 重叠；保留后者
- `RL_INTEGRATION_PLAN.md` — T22 集成执行 log（dated）；架构内容并入 ARCHITECTURE.md
- `code-path-sweep-EasyR1.md` — 2026-04-21 一次性扫描产物
- `dep-matrix.md`, `easyr1-dep-chain-audit.md`, `npu-gap-plan.md`, `npu-adaptation-tasks.md` — 早期依赖审计与 gap 列表，已被 fork branch 实际产出取代
- `PORT-SUMMARY.md`, `DELIVERABLE.md`, `porting-journal.md` — 早期 EasyR1 port 总结/交付/journal；当前总结见 [`docs/easyr1/PORT-GUIDE.md`](../../easyr1/PORT-GUIDE.md) 与 [`docs/easyr1/PORT-GUIDE-v2-integrated.md`](../../easyr1/PORT-GUIDE-v2-integrated.md)
- `UPGRADE-DRILL-STATUS.md`, `transformers-upgrade-drill.md` — vllm 0.18 时代的 image upgrade drill 演练记录
- `WORK-PLAN-2026-04-25.md` — triton-ascend 一次性 work plan

## 为什么归档而不是删除

留作历史回看：当后续有人想知道"之前是怎么得出 X 结论的"时，可以翻这些文件。但不希望出现在客户文档树里造成"这是不是当前要看的"的歧义。
