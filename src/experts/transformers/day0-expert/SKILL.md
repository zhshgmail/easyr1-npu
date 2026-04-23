---
name: transformers-day0
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

## Your role (orchestrator)

Spawn `transformers-day0-worker`, wait, read handoff, propagate outcome
classification {A/B/C} back to caller.

## Workflow

```
P0  parse args (target version, base image, fixture consumer ref)
P1  pip-overlay probe: inside base image, pip install target version,
    measure drift vs base image's transformers
P2  upstream branch fork from fixture consumer ref (transformers pin
    loosened)
P3  spawn transformers-day0-worker:
    - API-diff analysis (ALL_ATTENTION_FUNCTIONS, npu_flash_attention sig,
      modeling_utils hooks)
    - Decision: A/B/C
    - If A: build overlay image by baking the pip install, verify V1.1/V1.3/V1.4
    - If B: cherry-pick NPU FA adaptation to target sig, build wheel,
      rebuild overlay, verify smokes
    - If C: emit blocker report with specific ecosystem piece naming
P4  return outcome + image tag (A/B) or blocker doc (C)
```

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

## Invariants

- G1: orchestrator never edits consumer source. Day-0 expert may edit
  npu_flash_attention adapter IF forward-porting — narrow OL-08 scope.
- G2: static_check all py edits. Container dry-import MANDATORY for
  this expert (the whole point is runtime-level compatibility).
- G3: smoke claims backed by log paths + jsonl numerics. For "fresh
  baseline" outcomes, explicitly note the baseline is NEW and not
  compared against historical band.

## See also

- `README.md`, `agent.md`, `state_machine.yaml`
- `references/ALWAYS_LOADED_RULES.md` (OL-03/08)
- `references/patterns/domains/api-drift-scan.md` — how to do the
  API-diff across transformers minors
- `references/patterns/domains/overlay-image.md` — Dockerfile.overlay
  template
