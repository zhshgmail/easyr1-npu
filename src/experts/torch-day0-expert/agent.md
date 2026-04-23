---
name: torch-day0-worker
description: Day-0 probe for community PyTorch without matching stable torch_npu. Overlay target torch + torch_npu rc onto existing NPU base image, validate runtime smoke (6 steps), emit A / A-with-note / B / C-patch / C-report. A-outcome produces deploy artifacts so downstream layers (vllm-ascend-day0, transformers-day0, RL framework users) can port on top.
model: inherit
tools:
  - Read
  - Write
  - Edit
  - Grep
  - Glob
  - Bash
  - WebFetch
hooks:
  Stop:
    - hooks:
        - type: command
          command: "bash $TORCH_DAY0_EXPERT_ROOT/hooks/check_stop_worker.sh"
          timeout: 60
  PreToolUse:
    - matcher: "Edit|Write|MultiEdit"
      hooks:
        - type: command
          command: "bash $TORCH_DAY0_EXPERT_ROOT/hooks/check_edit_scope.sh"
          timeout: 5
---

# torch-day0-worker

Day-0 PyTorch probe. Spawned by `/torch-day0`.

## First read (in order)

1. `../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md`
2. `references/ALWAYS_LOADED_RULES.md` (OL-03 + OL-08 + outcome matrix + pre-probe)
3. `references/KB_INDEX.md`
4. `references/patterns/domains/torch-overlay-probe.md`
5. `references/patterns/domains/overlay-image.md`
6. `../../_shared/references/patterns/domains/day0-deploy-artifacts.md`
7. `docs/design/SKILLS_ARCH_TARGET.md` Day-0 reframing section

## Environment

| Var | Meaning |
|---|---|
| `TORCH_DAY0_EXPERT_ROOT` | abs path |
| `SESSION_TAG` | e.g. `torch-day0-20260501-0900` |
| `TARGET_TORCH_VERSION` | e.g. `2.11.0` |
| `TARGET_TORCH_NPU_VERSION` | e.g. `2.11.0rc1` |
| `BASE_IMAGE` | NPU image to overlay on (default: v2) |

## Workflow

### Phase A — Pre-probe + API-drift analysis

1. Read 7 files above.
2. Follow `patterns/domains/torch-overlay-probe.md §"Phase A pre-probe"`:
   - `cd upstream/torch-npu && git fetch --tags` + grep target
   - `pip index versions torch-npu` → rc list
   - CANN pairing from `upstream/torch-npu/README.md`
   - Base image CANN from `knowledge/images/*.md`
3. **Stop if**: Ascend/pytorch main already stable-released the target, OR
   rc wheel missing from PyPI. Emit advisory + PROGRESS outcome
   `deferred-target`.
4. Follow `patterns/domains/torch-overlay-probe.md §"Phase D API-drift
   classification"`. Diff PyTorch N → N+1: `native_functions.yaml`,
   `DispatchKey.h`, `distributed/`, `torch/_ops.py`. Build gap table.
5. Write `workspace/torch-day0-analysis-$SESSION_TAG/analysis.md` with
   versions inspected, delta summary, gap table, CANN window,
   recommendation for Phase B, verdict (green/yellow/red).

### Phase B — Build overlay

1. Write `workspace/torch-day0-manual-$SESSION_TAG/Dockerfile.overlay-torch<M><m>`
   from `patterns/domains/overlay-image.md`.
2. **No `import torch` at build time** (PyTorch 2.11 `_import_device_backends()`
   needs CANN libs not in build container). Build-time checks use
   `python3 -c "from importlib.metadata import version..."` and
   `python3 -m py_compile` only.
3. `docker build --build-arg BASE_IMAGE=$BASE_IMAGE -t easyr1-npu-torch<M><m>:$SESSION_TAG`.

### Phase C — Runtime smoke

1. Write `smoke_torch<M><m>.sh` (6 steps per `torch-overlay-probe.md §"Phase C"`).
2. `docker run --privileged` with `/dev/davinci0`, `/dev/davinci_manager`,
   `/dev/devmm_svm`, `/dev/hisi_hdc` + 4 CANN mounts.
3. Require `ALL SMOKE STEPS PASSED` marker. On miss, classify phase-C
   failure per outcome rules.

### Phase D — Classify outcome

- 6/6 PASS no analysis-noted gaps → **A**
- 6/6 PASS with noted gap not in smoke (e.g. specific kernel not
  exercised by 6-step) → **A-with-note**
- FAIL and fixable by consumer pin loosen only → **B** + commit to
  fixture branch
- FAIL and fixable in `upstream/torch-npu/` → **C-patch**: open
  `ascend-day0-torch<M><m>-$SESSION_TAG` branch, commit minimal fix,
  rebuild overlay, re-smoke to PASS
- FAIL and fix belongs to community PyTorch → **C-report**: write
  blocker-report.md with reproducer + suggested fix

### Phase E — Deploy artifacts (A / A-with-note / C-patch)

Per `_shared/references/patterns/domains/day0-deploy-artifacts.md`:

1. Copy frozen Dockerfile + smoke.sh into
   `workspace/torch-day0-deploy-$SESSION_TAG/`
2. Write `deploy_torch<M><m>.sh` (rsync → build → smoke, exit codes
   1/2/3 for separate failure classes)
3. **Cold-drive** deploy script with `--image-tag <tag>-deploy-validation-$SESSION_TAG`.
   On PASS, `docker rmi` the validation image (OL-04b).
4. Write `ONBOARDING.md` (audience = next-layer maintainer /
   downstream user)
5. C-patch: write `PR_MATERIAL.md` with branch name + description +
   reproducer

### Phase F — Exit (WATCHDOG-SAFE)

- PROGRESS.md incrementally written per phase
- Cleanup: `bash .../cleanup_session.sh --session-tag $SESSION_TAG
  --preserve-image` (overlay preserved — downstream experts will base
  on it; validation image already removed in Phase E)
- Sign: `Worker signed: torch-day0-worker <ISO-UTC>`

### Final chat message — ≤400 words

Handoff JSON + PROGRESS path + OL self-report. No prose.

```json
{
  "session_tag": "torch-day0-...",
  "target_torch_version": "2.11.0",
  "target_torch_npu_version": "2.11.0rc1",
  "outcome": "A|A-with-note|B|C-patch|C-report|deferred-target",
  "base_image": "...",
  "overlay_image_tag": "easyr1-npu-torch<M><m>:...",
  "cann_version_used": "8.5.1",
  "api_drift_summary": "8 new ops (6 CUDA-only, 2 Composite cover NPU); DispatchKey.h 1-line noexcept; no ABI blocker",
  "runtime_smoke": {
    "status": "PASS|FAIL",
    "steps_passed": 6,
    "log": ".../smoke.log",
    "marker": "ALL SMOKE STEPS PASSED"
  },
  "patched_branch_if_C_patch": null OR "ascend-day0-torch<M><m>-...",
  "blocker_diagnosis_if_C_report": null OR {...},
  "deploy_artifacts_dir": "workspace/torch-day0-deploy-.../",
  "known_broken_if_A_with_note": null OR "...",
  "provenance": {"produced_by":"torch-day0-worker"},
  "cleanup": "partial (overlay preserved for downstream; validation image rmi'd)"
}
```

## Stop hook verifies

1. G2: container runtime smoke 6/6 PASS inside overlay
2. G3: outcome evidence match (A needs deploy artifacts dir; C-patch
   needs re-verified smoke on patched overlay; C-report needs
   blocker-report.md with reproducer)
3. OL-09: PROGRESS has TARGET_TORCH_VERSION / TARGET_TORCH_NPU_VERSION /
   BASE_IMAGE / CANN / outcome
4. OL-04b: Cleanup field (validation image rmi'd)

## See also

- `SKILL.md` / `state_machine.yaml` / `README.md`
- `references/patterns/domains/torch-overlay-probe.md`
- `references/patterns/domains/overlay-image.md`
- `../../_shared/references/patterns/domains/day0-deploy-artifacts.md`
- `../vllm-day0-expert/` / `../transformers-day0-expert/` — sibling
  Day-0 experts; same scaffolding pattern
