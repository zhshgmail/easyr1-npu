---
name: torch-npu-day0
description: >
  Day-0 NPU probe for a community PyTorch release with NO matching
  torch_npu stable yet. Pip-overlay target torch + torch_npu rc onto an
  existing NPU base image (preserving CANN from base), validate import +
  device visibility + basic NPU tensor ops, emit A/B/C outcome. Produces
  a deployable overlay image whose next-layer downstream (vllm-ascend,
  triton-ascend, user RL framework) can port against.

  Usage: /torch-npu-day0 --target-torch-version <V> --target-torch-npu-version <V>
                         --base-image <TAG>
argument-hint: >
  target-torch-version: community torch release under test (e.g. 2.11.0)
  target-torch-npu-version: matching torch_npu, often rc1 for Day-0 (e.g. 2.11.0rc1)
  base-image: NPU image to overlay on (default: v2)
context: inline
---

# /torch-npu-day0 — community PyTorch probe on existing NPU base

## Two modes — decide before Phase 0

**Mode A — Overlay Day-0 probe** (this document's original scope):
there is no torch_npu stable for the target torch version yet. Build
an overlay image, install an rc wheel, run 6-step runtime smoke,
classify A/B/C. Phases below (P0..P6) describe this mode.

**Mode B — Inductor/Dynamo private-API drift scan** (added 2026-04-24
after cold-drive caught the gap): torch_npu source already has a
branch for the target torch version, but we need to patch the
`from torch._inductor...` / `from torch._dynamo...` imports that
broke between torch N.x and N.y. Skip P2-P4 entirely.

Prereq — clone community pytorch with tags:
```bash
mkdir -p ~/workspace/easyr1-npu/upstream
cd ~/workspace/easyr1-npu/upstream
git clone --filter=blob:none https://github.com/pytorch/pytorch.git pytorch
git -C pytorch fetch --tags origin
git -C pytorch tag --list 'v2.11*' 'v2.12*'   # verify baseline + target tags exist
```

Then run:

```bash
# One-command sweep (F1 + F2-path-move + F3 + F7 + F8):
bash src/skills/torch-npu/port-expert/scripts/sweep.sh \
  --baseline v2.11.0 --target v2.12.0-rc3 \
  --pt-repo ~/workspace/easyr1-npu/upstream/pytorch \
  --torch-npu-path ~/workspace/easyr1-npu/upstream/torch-npu
```

`sweep.sh` runs `extract_imports` → `check_drift` → `check_sig_drift`
→ `check_f7_f8` in sequence and emits a summary with per-family exit
codes. You can also call each scanner individually (useful when
iterating on a single family); see `docs/torch-npu/PORTING-GUIDE.md`
for the per-step breakdown.

Classify each finding per family, apply shims at
`torch_npu/compat/<module>.py` (or inline for tiny cases), validate
with `/drift-port-validate`. Commit to fork branch
`ascend-port/<target-torch-version-slug>` (e.g.
`ascend-port/torch-2.12-rc3`); see [`docs/_meta/UPSTREAM_FORKS.md`](../../../../docs/_meta/UPSTREAM_FORKS.md)
for the authoritative naming convention. (Older sessions used
`<target-torch-version>_auto_porting` — that naming is deprecated.)

**If uncertain which mode**: look at the user request. "Build overlay
for torch X" → Mode A. "Port torch_npu to torch Y's API drift" /
"find what broke between 2.N and 2.N+1" → Mode B.

## Your role (orchestrator)

Spawn `torch-npu-day0-worker`, wait, read Handoff, propagate {A/B/C, overlay
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

## Invariants + outcome classification

Shared across all 3 day-0 skills (vllm-ascend / torch-npu /
transformers): see [`_shared/upstream-day0-workflow.md`](../../_shared/upstream-day0-workflow.md)
§"Invariants" + §"Outcome classification".

torch-npu specifics layered on top:

- G2 specialization: Day-0 is a runtime-compat claim → container
  runtime `import torch; import torch_npu` inside overlay image is
  MANDATORY (build-only / py_compile is not enough).
- Branch convention: `ascend-port/torch-<target-version-slug>`
  (e.g. `ascend-port/torch-2.12-rc3`).

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

- **Shared workflow + invariants + outcome classification**:
  [`_shared/upstream-day0-workflow.md`](../../_shared/upstream-day0-workflow.md)
- **F1–F8 + F2-path-move drift family taxonomy**:
  [`_shared/patterns/F-family-taxonomy.md`](../../_shared/patterns/F-family-taxonomy.md)
- **Fork branch ledger**: [`docs/_meta/UPSTREAM_FORKS.md`](../../../../docs/_meta/UPSTREAM_FORKS.md)
- torch-npu-specific: `references/ALWAYS_LOADED_RULES.md`, `references/KB_INDEX.md` (cases), `references/patterns/domains/overlay-image.md` (Dockerfile template), `references/patterns/domains/torch-api-drift.md`, `references/patterns/domains/torch-overlay-probe.md`
- `../vllm/port-expert/`, `../transformers/port-expert/` — sibling Day-0 experts; same scaffolding
- `../_shared/references/patterns/domains/day0-deploy-artifacts.md` — Phase 2.5 deploy artifacts pattern
