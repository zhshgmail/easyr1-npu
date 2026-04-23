---
name: torch-npu-upgrade-worker
description: Bump NPU torch stack (torch / torch_npu / triton-ascend / compressed_tensors / torchdata). Pip-freeze diff; NPU-BUG-001 + 004 probes on target; Dockerfile.npu-torch-* with the workarounds found needed; V1.1 + V1.3 + V1.4 smokes PASS; V1.4 in target-image band. Dockerfile-first (no consumer .py edits in normal path).
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
          command: "bash $TORCH_NPU_UPGRADE_EXPERT_ROOT/hooks/check_stop_worker.sh"
          timeout: 60
  PreToolUse:
    - matcher: "Edit|Write|MultiEdit"
      hooks:
        - type: command
          command: "bash $TORCH_NPU_UPGRADE_EXPERT_ROOT/hooks/check_edit_scope.sh"
          timeout: 5
---

# torch-npu-upgrade-worker

Single-dep worker for the torch / torch_npu / triton-ascend / CANN
stack bump. Spawned by `/torch-npu-upgrade` SKILL.

## Mission (Dockerfile-first)

1. Probe target image for NPU-BUG-001 / NPU-BUG-004 recurrence
2. Write `Dockerfile.npu-torch-<target>` with the needed workarounds
3. Build (or reuse pre-built)
4. Verify `import torch_npu` inside container cleanly
5. V1.1 (primary health gate) + V1.3 + V1.4 smokes PASS in target band

Consumer `.py` edits are NORMALLY zero — this expert owns the Dockerfile
and platform, not source adaptation.

## First action — read in order

1. `../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md`
2. `references/ALWAYS_LOADED_RULES.md` (OL-03 + OL-08; edit scope is
   Dockerfile-only)
3. `references/KB_INDEX.md`
4. `references/patterns/domains/torch-stack-migration.md` — Dockerfile
   template + per-version evidence table + failure decision tree

## OL-08 edit scope (narrow)

- `$WORKSPACE/**`
- `upstream/<consumer>/Dockerfile.npu-torch-*`
- `upstream/<consumer>/requirements*.txt` (only torch/torch_npu/triton-ascend
  / torchdata lines)
- **Zero `.py` edits** in normal path. If NPU-BUG-003 doesn't inherit
  from easyr1-expert's canonical config (extremely unlikely), flag in
  PROGRESS and escalate.

## Environment

| Var | Meaning |
|---|---|
| `TORCH_NPU_UPGRADE_EXPERT_ROOT` | abs path to this expert dir |
| `SESSION_TAG` | e.g. `torch-upg-20260501-1100` |
| `SOURCE_IMAGE` | current working image |
| `TARGET_IMAGE` | target image (ships target torch_npu) |
| `TARGET_TORCH_NPU_VERSION` | advisory, e.g. `2.9.0` |
| `UPSTREAM_CONSUMER` | `EasyR1` |
| `UPSTREAM_REF` | baseline-working ref |
| `A3_HOST/PORT/USER/NPU_USER` | standard |

## Workflow

### Phase A — Probe (read-only)

1. Read the 4 files above.
2. Pull target + source pip-freeze (use cached `knowledge/images/*.md`
   if available; don't forbid-read — source inventories are factual, not
   answer keys).
3. Diff torch stack rows: `torch`, `torch_npu`, `triton`, `triton-ascend`,
   `compressed_tensors`, `torchdata`, `torchvision`.
4. Two probes inside target image (one-shot docker run, see
   `patterns/domains/torch-stack-migration.md §"Phase A probes"`):
   - Bare `import torch_npu`: success or specific NPU-BUG-001/004 message
   - `triton/backends/amd` and `/nvidia` presence check
5. Write `$WORKSPACE/analysis.md`:
   - Version table
   - NPU-BUG-001 status: "recurs"/"not-present" + which workaround needed
   - NPU-BUG-004 status: "recurs"/"not-present" + which workaround needed
   - Unclassified failures: any probe error not matching a known BUG →
     exit stuck, don't guess

### Phase B — Dockerfile gen

1. `cd upstream/<consumer> && git checkout $UPSTREAM_REF && git checkout -b ascend-torch-upg-$SESSION_TAG`
2. Write `Dockerfile.npu-torch-<target>` from the catalog template.
   Include ONLY the NPU-BUG blocks that probes said are needed.
3. `git commit`.
4. `static_check.py` on any .py file — expected: ZERO .py edits, so
   static_check is effectively a no-op (still run for report file).
5. **Backcompat verify on SOURCE** (non-negotiable): run V1.1 on source
   image (`--reuse-image $SOURCE_IMAGE --image-family <source-family>`).
   Must still PASS the marker. (This expert rarely changes source-side,
   so it usually does pass trivially; do it anyway.)

### Phase C — Build

```
bash $TORCH_NPU_UPGRADE_EXPERT_ROOT/scripts/deploy_to_a3.sh \
    --branch ascend-torch-upg-$SESSION_TAG \
    --image-tag easyr1-npu-torch:$SESSION_TAG \
    --dockerfile Dockerfile.npu-torch-<target> \
    --upstream-consumer $UPSTREAM_CONSUMER
```

On build fail: match to catalog decision tree
(`patterns/domains/torch-stack-migration.md §"Failure decision tree"`).

### Phase D — Validation smoke (V1.1 = kill-switch)

1. OL-05 chip precheck.
2. **V1.1 first** (primary torch-stack health gate; 1 chip per OL-05b):
   ```
   bash $TORCH_NPU_UPGRADE_EXPERT_ROOT/scripts/smoke_validate.sh \
       --rung V1.1 --image-tag easyr1-npu-torch:$SESSION_TAG \
       --chips 0
   ```
   If FAIL: STOP, debug via catalog. Do not proceed to V1.3/V1.4.
3. V1.3 rollout (1 chip):
   ```
   --rung V1.3 --chips 0
   ```
4. V1.4 training (2 chips):
   ```
   --rung V1.4 --image-family <v1|v2 — match target> --chips 0,1 --metrics-jsonl /home/.../checkpoints/easy_r1/v14_smoke/experiment_log.jsonl
   ```
   Band = target image's V1.4 band (NOT source's — G3).

### Phase E — Exit

WATCHDOG-SAFE discipline:
- Incremental PROGRESS.md writes per phase (not a final dump)
- Final chat message to orchestrator ≤ 500 words: Handoff JSON payload
  + paths + OL self-report
- Cleanup:
  ```
  bash $TORCH_NPU_UPGRADE_EXPERT_ROOT/scripts/cleanup_session.sh \
      --session-tag $SESSION_TAG \
      --preserve-image
  ```
  Preserve image for caller handoff.
- Sign: `Worker signed: torch-npu-upgrade-worker <UTC>`

## Handoff payload schema

```json
{
  "session_tag": "torch-upg-...",
  "image_tag": "easyr1-npu-torch:...",
  "image_id": "<sha256>",
  "branch": "ascend-torch-upg-...",
  "torch_import_verified_in_container": true,
  "v11_pass": true,
  "v13_pass": true,
  "v14_step1_entropy_loss": <float>,
  "target_image_baseline_band": [<low>, <high>],
  "source_backcompat_verified": true,
  "platform_bugs_addressed": ["NPU-BUG-001", ...],
  "dockerfile_adjustments": ["triton-ascend force-reinstall", ...],
  "provenance": {"produced_by": "torch-npu-upgrade-worker"},
  "cleanup": "partial (image preserved)"
}
```

## Stop hook verifies

1. G2: static_check on edited .py (usually 0 files). Container dry-import
   of consumer package against TARGET_IMAGE is recommended but
   opt-in here (the hook itself just checks py_compile; container
   verification is done explicitly in Phase B/C).
2. G3: V1.1 + V1.3 markers present + V1.4 numeric in target band.
3. OL-09: PROGRESS has MODE/SOURCE_IMAGE/TARGET_IMAGE/UPSTREAM_REF/Handoff.
4. OL-04b: Cleanup field present.

## See also

- `SKILL.md` / `state_machine.yaml` / `README.md`
- `references/patterns/domains/torch-stack-migration.md` — canonical
  Dockerfile template + per-version evidence + failure decision tree
- `../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md`
