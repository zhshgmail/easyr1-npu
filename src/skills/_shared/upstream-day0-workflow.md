# Shared upstream day-0 workflow (vllm-ascend / torch-npu / transformers)

> **Shared by 3 day-0 skills** (vllm-ascend / torch-npu / transformers).
> Each skill's `SKILL.md` is a thin wrapper that points here for the
> shared workflow + invariants + outcome classification, and adds only
> upstream-specific phase details.
>
> The 4th NPU upstream — **triton-ascend** — uses a different model
> (fork-merge, not plugin-day-0); it has its own SKILL.md and does NOT
> reference this file.

## Phase skeleton (P0..P6)

| Phase | What | Notes |
|---|---|---|
| P0 | Parse args + pre-probe target | Inputs: target version / SHA / delta. Pre-probe = check the upstream's Huawei fork hasn't already adapted (would make this run pointless — pivot target). |
| P1 | Analyze drift / scan / classify | Run upstream-specific scanners (per-skill scripts). Classify each finding into F1–F8 family per [`_shared/patterns/F-family-taxonomy.md`](patterns/F-family-taxonomy.md). |
| P2 | Probe / reproduce / minimum failure | Find the smallest reproducer for each drift before you patch. Classify-too-early = match the wrong family. |
| P3 | Design fix at minimum invasive level | F-family fix templates from the shared taxonomy. Forward-compat shims preferred (one tree against both old + new upstream). |
| P4 | Apply patch on `ascend-port/<target-version-slug>` branch + smoke test | Branch convention is **mandatory**, see [`docs/_meta/UPSTREAM_FORKS.md`](../../docs/_meta/UPSTREAM_FORKS.md). Validate via `/drift-port-validate` (F1/F2) or upstream-specific runtime smoke. |
| P5 | Phase 2.5 deploy artifacts | Per [`_shared/references/patterns/domains/day0-deploy-artifacts.md`](references/patterns/domains/day0-deploy-artifacts.md) (overlay image, etc.) — only if applicable to your upstream. |
| P6 | Handoff: PR_MATERIAL.md + branch push | Push `ascend-port/<slug>` to personal fork; commit `PR_MATERIAL.md` at branch root. Update [`docs/_meta/UPSTREAM_FORKS.md`](../../docs/_meta/UPSTREAM_FORKS.md). |

## Invariants (apply to all 3 day-0 skills)

- **G1 — worker scope**: Orchestrator never edits upstream source.
  Worker edits only on `ascend-port/<target-version-slug>` branch on
  the personal fork. Never on `main`/`master` of the upstream.
- **G2 — runtime validation mandatory**: A claim of outcome A or
  C-patch requires running the patched code (or unmodified code, for
  outcome A) end-to-end and observing the documented PASS criterion.
  py_compile / import / build success alone is **not** validation.
- **G3 — PR_MATERIAL.md is the deliverable**: Every C-patch outcome
  ships a `PR_MATERIAL.md` at branch root: per-fix rationale, diff,
  reproducer, validation done, known limitations. The fork branch
  diff alone is not a hand-off; it must be readable to the upstream
  maintainer in one open.

## Outcome classification (shared)

| Outcome | Meaning | Action |
|---|---|---|
| **A** | Runtime smoke PASS without patches; upstream version is drop-in compatible | Ship notes + overlay/marker; update KB; close as outcome A. |
| **A-with-note** | Smoke PASS but dep tree has a known gap not exercised by smoke | Ship + note "known broken" section in ONBOARDING / PR_MATERIAL. |
| **B** | Smoke fails on consumer-side config only (env var / CLI flag / pin loosen — no upstream code change) | Document workaround in ONBOARDING; update KB. |
| **C-patch** | Upstream-side source change needed; we have authority to make it | Patch on `ascend-port/<slug>` branch; smoke PASS; ship `PR_MATERIAL.md`. |
| **C-report** | Fix belongs to community upstream (vllm core / pytorch core / transformers core) — outside our scope | Emit blocker report with reproducer + suggested fix; session ends. |

**Default goal is C-patch + smoke PASS.** Project value is producing
validated, PR-ready patches for the corresponding upstream
maintainer.

## Phase 0 constraints (shared)

- Always pre-probe the upstream's Ascend-team fork (e.g.
  `gitcode.com/Ascend/<upstream>` or the equivalent) BEFORE picking a
  target version. If they've already adapted, this run produces no
  value — pivot to a newer delta where community is ahead and Ascend
  isn't.
- Always check the upstream's CI image for what versions are pinned.
  Project Makefile / Dockerfile pins are executable truth; doc /
  installation guide pins often lag.

## Branch convention (mandatory)

```
ascend-port/<target-version-slug>
```

- See [`docs/_meta/UPSTREAM_FORKS.md`](../../docs/_meta/UPSTREAM_FORKS.md)
  for the authoritative table of (upstream, fork URL, active branch).
- Never commit to `main`/`master` of the upstream personal fork.
- One slug = one branch; new target version = new branch (don't
  rebase / overwrite history).

## Cross-references

- F-family taxonomy: [`_shared/patterns/F-family-taxonomy.md`](patterns/F-family-taxonomy.md)
- NPU container runner: [`_shared/npu-container-runner/SKILL.md`](npu-container-runner/SKILL.md)
- Drift port validate (sub-skill): [`_shared/drift-port-validate/SKILL.md`](drift-port-validate/SKILL.md)
- Self-challenge (mandatory before completion claims): [`_shared/porting-self-challenge/SKILL.md`](porting-self-challenge/SKILL.md)
- Code/branch hygiene: [`_shared/upstream-branch-hygiene/SKILL.md`](upstream-branch-hygiene/SKILL.md)
- Lessons KB: [`docs/_meta/kb/porting_lessons/`](../../docs/_meta/kb/porting_lessons/)
- Critic patterns: [`docs/_meta/kb/challenge_patterns/`](../../docs/_meta/kb/challenge_patterns/)

## What this file is NOT

- Not a replacement for the per-skill SKILL.md. The per-skill file
  must still document upstream-specific phase content (e.g. "Phase 2
  for torch-npu builds an overlay image; vllm-ascend's Phase 2 is
  failure-minimization").
- Not a replacement for the F-family taxonomy. The taxonomy describes
  *what to fix*; this file describes *the workflow that finds, fixes,
  and ships*.
- Not used by `triton-ascend` skill — that skill uses fork-merge
  model (not plugin day-0); see its own SKILL.md.
