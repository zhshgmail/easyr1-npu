---
name: torch-npu-upgrade
description: >
  Bump the NPU torch stack (torch / torch_npu / triton-ascend /
  compressed_tensors / torchdata) within one base-image family or to a new
  base image where transformers + vllm stay put. Pip-freeze diff, re-apply
  NPU-BUG-001 (triton-ascend partial install) and NPU-BUG-004 (upstream
  triton backend prune) per new stack, verify `import torch_npu` clean
  in container, V1.1 / V1.3 / V1.4 smokes all PASS in target-image band.

  Usage: /torch-npu-upgrade --target-image <TAG> --source-image <TAG>
                             [--target-torch-npu-version <v>]
                             [--upstream-ref <sha>]
argument-hint: >
  target-image: NPU base image shipping the target torch stack (required)
  source-image: current working image (default v1)
  target-torch-npu-version: advisory canonical version (e.g. 2.9.0)
  upstream-ref: baseline-working consumer ref
context: inline
---

# /torch-npu-upgrade — NPU torch stack bump (single-dep-family)

**Your role**: spawn `torch-npu-upgrade-worker`, read Handoff, return
{image_tag, branch, V1.1/V1.3/V1.4 results, platform-bug workarounds
applied} to caller.

## Workflow overview

```
P0  parse args
P1  image inspect (pip-freeze diff focused on torch stack; check for
    NPU-BUG-001 triton-ascend integrity on target; check for NPU-BUG-004
    upstream triton backend pollution on target)
P2  upstream branch
P3  spawn torch-npu-upgrade-worker (analyze, Dockerfile adjust, build)
P4  container `import torch_npu` verification + V1.1 + V1.3 + V1.4
    validation; G3 band discipline
P5  report + handoff
```

See `state_machine.yaml` for authoritative invariants.

## Stage 0 constraints

- Heaviest of the 3 upgrade experts: the torch stack drags along
  triton-ascend, NPU-BUG-001/003/004 recur per image, and FSDP internals
  can shift. Budget accordingly (docker build ~15min on fresh base).
- **V1.1 is the primary health gate** — any torch_npu / triton-ascend /
  inductor regression shows up here loudly. Don't run V1.3 or V1.4 until
  V1.1 passes cleanly.
- **Backcompat with SOURCE image**: re-run V1.1 on source image post any
  consumer-side edits. Must still PASS. (Typically this expert writes
  zero consumer .py edits — all fixes are in Dockerfile.)

## Invariants

- G1: orchestrator doesn't edit Dockerfile or consumer tree.
- G2: static_check.py on any consumer .py touched (usually empty — this
  expert mostly edits Dockerfile). Container dry-import of consumer
  package inside target image is **strongly recommended** — this is
  exactly the class of regression container-dry-import catches.
- G3: V1.1/V1.3/V1.4 claims cite logs; V1.4 numeric in target band.

## Return payload

See `README.md §Deliverable`.

## See also

- `README.md`, `agent.md`, `state_machine.yaml`
- `references/patterns/domains/torch-stack-migration.md` — pip-freeze
  delta handling + Dockerfile NPU-BUG-001/004 workaround templates
  per torch-stack version
