---
name: vllm-ascend-day0
description: >
  Day-0 NPU probe for vllm-ascend against a deeper upstream move (e.g.
  new torch that vllm-ascend's C++ extension was not rebuilt for, or
  new community vllm that dropped a symbol vllm-ascend imports). Runs
  ON TOP of a deployed torch-day0 / transformers-day0 base image.
  Produces validated patches to vllm-ascend on an ascend-day0 branch.

  Usage: /vllm-ascend-day0 --target-delta <symbol-or-desc>
                           --base-image <torch-day0-deployed-tag>
argument-hint: >
  target-delta: what moved (e.g. "torch==2.11", "vllm==0.20.0", "C++ ABI drift")
  base-image: torch-day0 or transformers-day0 deployed image
context: inline
---

# /vllm-ascend-day0 — patch vllm-ascend for new upstream ABI/API

## Your role (orchestrator)

Spawn `vllm-ascend-day0-worker`, wait, read Handoff, propagate {outcome,
patched branch, overlay image tag, PR material} back to caller. This
skill is specifically for **patching vllm-ascend itself** on a
Huawei-owned upstream; it's invoked AFTER torch-day0 / transformers-day0
has produced a usable base image.

## Workflow

```
P0  parse args (target delta, base image from previous Day-0 session)
P1  analysis: reproduce failure minimally, classify root cause
    (API-level drift vs C++ ABI drift vs schema mismatch vs other);
    identify which call sites in vllm-ascend are affected
P2  probe upstream vllm-ascend main: has the fix already landed?
    If yes, switch target or reproduce on an explicit commit
P3  design fix at minimum-invasive level: python-side workaround if
    possible, env-var guard, source patch if needed. Prefer strategies
    that "route through existing safe paths" (like VLLM_BATCH_INVARIANT)
    over introducing new APIs
P4  apply patch on `ascend-day0-<delta>-<SESSION>` branch of
    upstream/vllm-ascend/; smoke test it to PASS
P5  Phase 2.5 deploy artifacts per shared pattern
P6  handoff: patched branch + overlay image + ONBOARDING + PR material
```

## Stage 0 constraints

- **Edit scope strictly vllm-ascend only** (Huawei open source). Never
  edit community vllm, community torch, community transformers. Those
  are C-report territory.
- Env-var / python-level workarounds preferred over C++ changes. C++
  changes typically trigger rebuilds of the .so which is a slow
  development loop. Many "Day-0 on new torch" bugs can be fixed by
  detecting the ABI mismatch in Python and routing around it.
- **The `vllm_is_batch_invariant()` gate** (vllm's own mechanism) is
  an invaluable escape hatch because vllm-ascend already gates many
  call sites on it for OTHER reasons. Setting `VLLM_BATCH_INVARIANT=1`
  before vllm's module-import-time cache can bypass every gated call
  site at once.
- Patches land on a branch in `upstream/vllm-ascend/`, not on image
  files. Overlay image consumes the patch via git clone or COPY.

## Invariants

- G1: orchestrator never edits vllm-ascend or any upstream. Worker only
  on `ascend-day0-<delta>-<SESSION>` branch.
- G2: patch must be validated end-to-end via V1.3 rollout smoke marker
  match AFTER applying the patch + rebuilding overlay.
- G3: PR material must include a minimal reproducer + before/after
  behavior table.

## Outcome classification

| Outcome | Meaning | Action |
|---|---|---|
| A | No patch needed — existing vllm-ascend handles the delta already | Ship notes + overlay image; update KB |
| B | Workaround in consumer config (env var, CLI flag) without code patch | Document in ONBOARDING + add to KB |
| C-patch | vllm-ascend source change needed; we can make it | Patch on ascend-day0-<delta>-<SESSION> branch, smoke PASS, PR material |
| C-report | Fix belongs to community vllm / community torch / etc. | Blocker report |

**Goal is C-patch + smoke PASS**. Our value is producing validated,
PR-ready patches for the vllm-ascend team.

## See also

- `README.md`, `agent.md`, `state_machine.yaml`
- `references/ALWAYS_LOADED_RULES.md` — OL-03 / OL-08 specific to this expert
- `references/KB_INDEX.md` — known vllm-ascend Day-0 patterns + 2026-04-23 wet-run findings
- `references/patterns/domains/vllm-ascend-probe.md` — reproducer minimization, call-site location, fix-level selection
- `../torch-day0-expert/` — sibling whose deploy output is this expert's base image
- `../../_shared/references/patterns/domains/day0-deploy-artifacts.md` — Phase E deliverables
