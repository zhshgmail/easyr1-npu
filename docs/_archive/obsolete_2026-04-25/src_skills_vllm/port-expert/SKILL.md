---
name: vllm-port
description: >
  Day-0 NPU probe for a community vllm release with NO matching vllm-ascend
  build yet. Pip-overlay target vllm onto existing NPU image (ships older
  vllm-ascend), verify runtime compatibility, emit A/B/C outcome. V1.3
  rollout is the primary smoke (stresses vllm-ascend end-to-end); V1.4
  catches training-path regressions.

  Usage: /vllm-day0 --target-vllm-version <V> --base-image <TAG>
                    [--upstream-ref <sha>]
argument-hint: >
  target-vllm-version: community vllm release under test (e.g. 0.19.1)
  base-image: NPU image to overlay on (default: v2)
  upstream-ref: consumer ref with loosened vllm pin (fixture if needed)
context: inline
---

# /vllm-day0 — community vllm probe on existing vllm-ascend

## Your role (orchestrator)

Spawn `vllm-day0-worker`, wait, read Handoff, propagate {A/B/C, overlay
image, smoke results} back to caller.

## Workflow

```
P0  parse args
P1  pip overlay probe inside base image: new vllm + existing vllm-ascend
    plugin load, API surface diff (lora.lora_model, parallel_state,
    SamplingParams props, LLM.generate sig, VLLMHijack targets,
    platform_plugins group registration)
P2  decide A/B/C based on drift + consumer-source impact grep
P3  build overlay image (outcome A or B), apply shim .py files in
    consumer if needed (B forward-port is a consumer-side pin loosen,
    not a vllm-ascend source edit; deeper patches go to outcome C)
P4  V1.3 (primary) + V1.4 (secondary) smoke; fresh baseline protocol
    for V1.4 (target vllm has no known entropy_loss band)
P5  report + handoff, preserve overlay image
```

## Stage 0 constraints

- ONE vllm version per session. Don't bundle with transformers / torch.
- **Overlay path preferred**: `FROM base-image + pip install vllm==VERSION
  --no-deps` — fast, reversible, no rebuild of vllm-ascend needed.
- vllm-ascend source edits are OUT OF SCOPE for this expert's outcome B.
  If the new vllm needs vllm-ascend internals patched (e.g. a removed
  symbol), that's outcome C → report the specific vllm-ascend file +
  line for the NPU team to patch upstream.
- **Fresh baseline for V1.4**: new vllm version has no known-good band.
  Assertion is "2 steps complete + jsonl has entropy_loss"; measured
  value becomes the new baseline.

## Invariants

- G1: orchestrator never edits consumer source or vllm-ascend source
  directly.
- G2: container dry-import of `vllm` inside overlay image MANDATORY
  (Day-0 is runtime compat).
- G3: outcome A/B need V1.3 marker + V1.4 numeric; C needs blocker
  report.

## See also

- `README.md`, `agent.md`, `state_machine.yaml`
- `references/patterns/domains/vllm-overlay-probe.md` — API surface
  scan protocol specific to vllm
- `references/patterns/domains/overlay-image.md` — pip-overlay Dockerfile
  template
- `../transformers/port-expert/` — sibling Day-0 expert; same scaffolding
