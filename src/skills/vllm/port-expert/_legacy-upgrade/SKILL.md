---
name: vllm-upgrade
description: >
  Bump vllm / vllm-ascend on NPU within a fixed base-image family
  (e.g. vllm-ascend 0.13 → 0.14 on CANN 8.5.0). Diff source vs target
  image pip-freeze, apply consumer-side shims for rename/move classes
  (NPU-CP-002 lora.models→lora.lora_model, NPU-CP-004 TP group,
  EC-03 SamplingParams read-only properties), verify existing hijack
  points still resolve on the new version, build image (or reuse
  pre-built), V1.3 rollout smoke + V1.4 training smoke must both PASS.

  Usage: /vllm-upgrade --target-image <TAG> --source-image <TAG>
                       [--target-vllm-version <v>] [--upstream-ref <sha>]
argument-hint: >
  target-image: NPU base image shipping the target vllm (required)
  source-image: current working image (default v1)
  target-vllm-version: advisory canonical version (default derived from target-image pip-freeze)
  upstream-ref: baseline-working consumer ref (default: last known-green on source-image)
context: inline
---

# /vllm-upgrade — NPU vllm/vllm-ascend bump (single-dep)

**Your role**: spawn `vllm-upgrade-worker`, read Handoff, return
{image_tag, branch, V1.3 marker, V1.4 numerics} to caller.

## Canonical workflow

See `state_machine.yaml`. Summary:
```
P0  parse args
P1  image inspect (pip-freeze diff focused on vllm+ascend)
P2  upstream branch (fork baseline-working ref)
P3  spawn vllm-upgrade-worker (Phase A analyze, B shim, C build/reuse, D smoke)
P4  validation — V1.3 rollout (vllm-ascend stress) + V1.4 training
P5  report + handoff
```

## Stage 0 constraints

- **Single-driver-only**: don't bundle a transformers jump and a vllm
  jump in one session. If both need to move, chain two calls
  (transformers-upgrade first, then vllm-upgrade, or vice versa —
  orchestrator decides).
- **vllm upstream never built from source**. We consume pre-built
  `vllm-ascend` wheels.
- **Backcompat with SOURCE image MANDATORY**: post-shim, re-run V1.3 on
  the source image; must still produce coherent output.

## Invariants (G1/G2/G3 from universal)

- G1: orchestrator never edits `upstream/<consumer>/` directly.
- G2: edited .py files pass `static_check.py`. Container dry-import
  against the target vllm image is an **opt-in bonus** (see shared
  `--container-import-image`) since vllm import is heavy.
- G3: any PASS claim ties to log path + V1.3 marker / V1.4 entropy_loss
  numeric in the target image's band, cited via jsonl.

## Return payload

See `README.md §Deliverable`.

## See also

- `README.md`, `agent.md`, `state_machine.yaml`
- `references/ALWAYS_LOADED_RULES.md` — OL-03 (this expert's denylist)
  + OL-08 (this expert's edit scope: vllm-adjacent files only)
- `references/patterns/domains/vllm-rename-catalog.md` — the
  rename-per-version table (CP-002, CP-004, EC-03, future entries)
