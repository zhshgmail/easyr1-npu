---
name: torch-day0
description: >
  Day-0 NPU probe for a community PyTorch release with NO matching
  torch_npu stable yet. Pip-overlay target torch + torch_npu rc onto an
  existing NPU base image (preserving CANN from base), validate import +
  device visibility + basic NPU tensor ops, emit A/B/C outcome. Produces
  a deployable overlay image whose next-layer downstream (vllm-ascend,
  triton-ascend, user RL framework) can port against.

  Usage: /torch-day0 --target-torch-version <V> --target-torch-npu-version <V>
                     --base-image <TAG>
argument-hint: >
  target-torch-version: community torch release under test (e.g. 2.11.0)
  target-torch-npu-version: matching torch_npu, often rc1 for Day-0 (e.g. 2.11.0rc1)
  base-image: NPU image to overlay on (default: v2)
context: inline
---

# /torch-day0 — community PyTorch probe on existing NPU base

## Your role (orchestrator)

Spawn `torch-day0-worker`, wait, read Handoff, propagate {A/B/C, overlay
image tag, validated combos table, patches} back to caller.

## Workflow

```
P0  parse args + pre-probe upstream Ascend/pytorch for target ref
P1  analyze PyTorch 2.x → 2.x+1 delta (native_functions.yaml, DispatchKey,
    distributed) vs torch_npu rc coverage; classify gap into
    python-fallback / needs-native-kernel / ABI-blocker
P2  build overlay: FROM base-image + pip install torch + torch_npu + rc
    wheels; build-time VERSION ONLY (no `import torch` — torch 2.11's
    _import_device_backends auto-loads torch_npu which needs CANN)
P3  runtime smoke: 6 steps (metadata / import torch / import torch_npu /
    device count / basic NPU op / API-presence checks). ALL must pass
    for outcome A
P4  if A-with-note: document the note and still ship; if C-patch: apply
    minimal patch to torch_npu / upstream CANN-side integration on
    ascend-day0-torch<M><m> branch; if C-report: emit blocker to
    Ascend/pytorch maintainers
P5  Phase 2.5 deploy artifacts per
    `_shared/references/patterns/domains/day0-deploy-artifacts.md`
P6  handoff: overlay image + ONBOARDING + PR material (if C-patch)
```

## Stage 0 constraints

- ONE torch version per session. Don't bundle vllm or transformers deltas.
- **Overlay path preferred**: `FROM base-image + pip install --no-deps
  torch==X + torch_npu==X.rc1 + torchvision + torchaudio` — all from
  `download.pytorch.org/whl/cpu/` extra index.
- torch_npu source edits are this expert's C-patch scope (it's
  Huawei-owned). Community torch source edits are OUT OF SCOPE
  (community decision → C-report).
- **No runtime imports at build time**: PyTorch 2.11 broke
  previously-safe `import torch` in docker build containers via its
  `_import_device_backends()` auto-load mechanism. Use py_compile /
  AST checks for build-time syntax validation.

## Invariants

- G1: orchestrator never edits torch/torch_npu source directly. Worker
  edits `upstream/torch-npu/**` only on ascend-day0-torch<stamp> branch.
- G2: container runtime import of `torch` and `torch_npu` inside overlay
  image MANDATORY (Day-0 is a runtime-compat claim).
- G3: outcome A needs 6/6 smoke PASS + ONBOARDING `Validated
  combinations` row; C-patch needs patch + re-verified smoke; C-report
  needs blocker with reproducer + suggested fix.

## Outcome classification (same as vllm-day0 / transformers-day0)

| Outcome | Meaning | Action |
|---|---|---|
| A | Runtime smoke 6/6 PASS without patches | Ship overlay + ONBOARDING |
| A-with-note | Smoke PASS but dep tree has a known gap not hit by smoke | Ship overlay + note in ONBOARDING "known broken" section |
| B | Smoke fails on consumer-side pin loosen only (no upstream patch) | Patch requirements + smoke PASS |
| C-patch | Needs torch_npu / Huawei-owned-integration fix | Open branch, patch, rebuild overlay, smoke PASS, PR material |
| C-report | Needs community torch change | Blocker report with reproducer + suggested fix; session ends at C-report |

## Pre-probe discipline (required before target selection)

Before committing to `--target-torch-version X`, verify:

1. `upstream/torch-npu` main tip has NOT already adapted to X. If it
   has, the session produces no value — switch to a newer target.
2. CANN bundle matrix (hiascend.com/developer/download/community)
   confirms there's no stable torch_npu X yet. rc wheels on PyPI are
   the Day-0 target; stable isn't.
3. `git log origin/main -S '<target-specific-symbol>'` — probe the
   torch-npu repo for any mention of the target version.

## See also

- `README.md`, `agent.md`, `state_machine.yaml`
- `references/ALWAYS_LOADED_RULES.md` — OL rules specific to this expert
- `references/KB_INDEX.md` — symptom → outcome table + pre-probe results
- `references/patterns/domains/overlay-image.md` — Dockerfile template
- `../vllm/day0-expert/`, `../transformers/day0-expert/` — sibling
  Day-0 experts; same scaffolding
- `../_shared/references/patterns/domains/day0-deploy-artifacts.md` —
  Phase 2.5 deploy artifacts pattern (mandatory for A/C-patch outcomes)
