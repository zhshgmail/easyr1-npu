---
name: transformers-port
description: >
  Day-0 NPU support for a new community transformers release
  (version not shipped in any NPU base image yet). Probe the new version
  against existing NPU bindings via pip-overlay; emit one of
  A (works-as-is), B (forward-port needed + feasible), C (blocked with
  precise diagnosis). Run V1.1 / V1.3 / V1.4 smokes; for A/B, record
  fresh baseline; for C, pinpoint which NPU ecosystem piece (torch_npu
  kernel / vllm-ascend / transformers.integrations.npu_flash_attention)
  needs the upstream fix.

  Usage: /transformers-day0 --target-transformers-version <V>
                             --base-image <TAG> [--upstream-ref <sha>]
argument-hint: >
  target-transformers-version: the community release to probe (e.g. 5.6.0)
  base-image: starting NPU image to overlay on (default: v2)
  upstream-ref: consumer ref whose transformers pin has been loosened (fixture)
context: inline
---

# /transformers-day0 — new-community-transformers probe + adapt

## Stage 0 — fast-path decision tree (read this FIRST)

Before the full P0-P4 workflow, check if the fast-path applies. For
transformers, 90% of probes resolve to outcome **A (works-as-is)**
because the NPU integration surface is tiny (1 file, 143 lines). The
following static checks reach outcome A without A3/docker:

1. **File byte-match**: does
   `<target-ref>:src/transformers/integrations/npu_flash_attention.py`
   match the KB-recorded known-good baseline (see KB_INDEX §"baseline
   snapshot") byte-for-byte? `git show` is enough.
2. **Import chain**: are the imports `is_torch_npu_available` +
   `from torch_npu import npu_fusion_attention` unchanged?
3. **Upstream consumption**: does `modeling_flash_attention_utils.py`
   still import the NPU handlers on the `is_torch_npu_available()`
   branch?
4. **ALL_ATTENTION_FUNCTIONS**: are the keys still a superset of
   what the KB recorded, or has a new default been added that
   consumer models now route to?

If 1-3 all YES and 4 shows only additive entries (no new NPU-routing
default), **outcome A with note** — skip the full workflow, emit the
classification, write the note about newly-added unhandled keys in
ONBOARDING, done.

If ANY of 1-3 NO, or a default changed to a new NPU-unhandled key in
4, fall through to the full P0-P4 workflow below.

**This fast-path added 2026-04-24 after cold-drive showed 90% of the
full workflow was unnecessary for the transformers outcome-A case.**

## Your role (orchestrator)

Spawn `transformers-day0-worker`, wait, read handoff, propagate outcome
classification {A/B/C} back to caller.

## Workflow

```
P0   parse args (target version, base image, fixture consumer ref)
P0.5 firewalled-host check: if the machine running this skill cannot
     reach PyPI / docker hub (common for developer laptops behind a
     corporate proxy), SKIP P1 and use the static byte-match path in
     Stage 0 above. For outcome A this suffices. For B/C, the wet-run
     steps MUST happen on A3 host (or another net-reachable box); emit
     "pending A3 execution" marker instead of attempting them locally.
P1   pip-overlay probe: inside base image, pip install target version,
     measure drift vs base image's transformers
P2   consumer-fixture loosening — MANDATORY even on outcome A if the
     consumer (verl / EasyR1 / user repo) has a hard `transformers<X`
     pin and X <= target. Branch from consumer ref, loosen the pin,
     push; provenance = "orchestrator-fixture". Without this step, the
     downstream V1.1/V1.3/V1.4 smokes will fail at consumer pip install
     regardless of NPU adapter status.
P3   spawn transformers-day0-worker:
    - API-diff analysis (ALL_ATTENTION_FUNCTIONS, npu_flash_attention sig,
      modeling_utils hooks)
    - Decision: A/B/C
    - If A: build overlay image by baking the pip install, verify V1.1/V1.3/V1.4
    - If B: cherry-pick NPU FA adaptation to target sig, build wheel,
      rebuild overlay, verify smokes
    - If C: emit blocker report with specific ecosystem piece naming
P4   return outcome + image tag (A/B) or blocker doc (C)
```

### SKILL vs agent.md / state_machine.yaml scoping

- **SKILL.md (this file)** = decision tree + classification KB + Stage 0
  fast-path. Reading this + KB_INDEX is sufficient for dry classification.
- **agent.md / state_machine.yaml** = wet-run runbook (A3 docker commands,
  smoke harness integration, deploy artifact generation). Only needed when
  running P3 / P4 end-to-end on A3. A cold-driving LLM doing dry-classify
  can skip these.

## Stage 0 constraints for this expert

- ONE transformers version per session. No multi-hop.
- "Overlay" path preferred over full rebuild: `docker build -f Dockerfile.overlay`
  that takes an existing NPU image as base and just `pip install
  transformers==VERSION`. Fast, reversible, doesn't disturb other deps.
- If overlay fails at pip install: diagnose dep tree conflict; the
  orchestrator may need to chain this with vllm-day0 or torch-day0.
  Don't try to resolve multi-dep conflicts inside this expert.
- **Fresh baseline protocol**: if V1.4 is the first time this target
  transformers version has been measured, the assertion is "2 steps
  complete + checkpoint saves + jsonl has a numeric entropy_loss" —
  not a band match. The number is RECORDED as the new baseline for
  future runs (and added to SMOKE_BASELINE.md with evidence tag).

## Invariants + outcome classification

Shared across all 3 day-0 skills (vllm-ascend / torch-npu /
transformers): see [`_shared/upstream-day0-workflow.md`](../../_shared/upstream-day0-workflow.md)
§"Invariants" + §"Outcome classification".

transformers specifics layered on top:

- G1 narrowing: Day-0 expert may edit `npu_flash_attention` adapter
  IF forward-porting — narrow OL-08 scope.
- G2 specialization: container dry-import MANDATORY (the whole point
  is runtime-level compatibility).
- G3 specialization: smoke claims backed by log paths + jsonl
  numerics; for "fresh baseline" outcomes, explicitly note baseline
  is NEW vs historical band.
- Branch convention: `ascend-port/transformers-<target-version-slug>`
  (e.g. `ascend-port/transformers-v5.4`).

## See also

- **Shared workflow + invariants + outcome classification**:
  [`_shared/upstream-day0-workflow.md`](../../_shared/upstream-day0-workflow.md)
- **F1–F8 + F2-path-move drift family taxonomy**:
  [`_shared/patterns/F-family-taxonomy.md`](../../_shared/patterns/F-family-taxonomy.md)
- **Fork branch ledger**: [`docs/_meta/UPSTREAM_FORKS.md`](../../../../docs/_meta/UPSTREAM_FORKS.md)
- transformers-specific: `references/ALWAYS_LOADED_RULES.md` (OL-03/08), `references/patterns/domains/api-drift-scan.md` (Phase A scan protocol), `references/patterns/domains/overlay-image.md` (Dockerfile template)
