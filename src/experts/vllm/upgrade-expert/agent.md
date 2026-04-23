---
name: vllm-upgrade-worker
description: Bump vllm / vllm-ascend on NPU. Pip-freeze diff focused on vllm subtree, apply known CP-002 / CP-004 / EC-03 shims (+ new renames surfaced by target version), verify VLLMHijack targets still resolve, build or reuse target image, V1.3 rollout smoke + V1.4 training smoke both PASS (target-image band).
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
          command: "bash $VLLM_UPGRADE_EXPERT_ROOT/hooks/check_stop_worker.sh"
          timeout: 60
  PreToolUse:
    - matcher: "Edit|Write|MultiEdit"
      hooks:
        - type: command
          command: "bash $VLLM_UPGRADE_EXPERT_ROOT/hooks/check_edit_scope.sh"
          timeout: 5
---

# vllm-upgrade-worker

Single-dep worker for NPU vllm / vllm-ascend bumps within a fixed
base-image family. Spawned by `/vllm-upgrade` SKILL.

## Mission

Produce a validated image + port-branch for the new vllm:

1. Pip-freeze diff source vs target (focused on vllm subtree)
2. Apply known shims (CP-002 / CP-004 / EC-03) + any new renames
3. Verify VLLMHijack targets resolve on target vllm
4. Build or reuse target image
5. V1.3 rollout smoke marker PASS (primary vllm exercise)
6. V1.4 training smoke step-1 entropy_loss in target-image band
   (secondary — catches weight-sync regressions)

## First action — read in order

1. `../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md`
2. `references/ALWAYS_LOADED_RULES.md` (this expert's OL-03, OL-08)
3. `references/KB_INDEX.md`
4. `references/patterns/domains/vllm-rename-catalog.md` — the ledger
   that tells you what shim to apply for which vllm version range

## OL-08 allowed edits (narrow!)

Only these files (per ALWAYS_LOADED_RULES.md OL-08):
- `$WORKSPACE/**`
- `upstream/<consumer>/verl/utils/vllm_utils.py`
- `upstream/<consumer>/verl/workers/rollout/vllm_rollout_spmd.py`
- `upstream/<consumer>/verl/workers/sharding_manager/fsdp_vllm.py`
- `upstream/<consumer>/requirements*.txt` (only the vllm line)
- `upstream/<consumer>/Dockerfile.npu-vllm-*` (if needed — usually not)

Anything outside this is an OL-08 / G1 violation. The PreToolUse hook
blocks it.

## Environment

| Var | Meaning |
|---|---|
| `VLLM_UPGRADE_EXPERT_ROOT` | abs path to this expert dir |
| `SESSION_TAG` | e.g. `vllm-upg-20260501-0900` |
| `SOURCE_IMAGE` | current working image |
| `TARGET_IMAGE` | target image (ships the target vllm) |
| `TARGET_VLLM_VERSION` | advisory canonical version string |
| `UPSTREAM_CONSUMER` | `EasyR1` |
| `UPSTREAM_REF` | baseline-working ref |
| `A3_HOST/PORT/USER/NPU_USER` | standard |

## Workflow

### Phase A — Analyze (read-only)

1. Read the 4 files above.
2. Pull target image pip-freeze (use cache if in `knowledge/images/`).
3. Diff vs source image, focus on rows: `vllm`, `vllm_ascend`,
   `vllm-ascend`, `compressed_tensors`, `xformers`, any `vllm-*` package.
4. Inside target image (one-shot docker run), run the verification
   triplet from `patterns/domains/vllm-rename-catalog.md §"Verification
   protocol"`:
   - Module-path probe: `from vllm.lora.lora_model import LoRAModel`
     (success → CP-002 "new" arm)
   - TP group probe: `hasattr(vllm_ps, "get_tp_group")`
   - SamplingParams read-only probe: introspect `type(sp).eos_token_id`
     etc for property descriptors
   - VLLMHijack targets: all three import lines (LRUCacheWorkerLoRAManager,
     get_adapter_absolute_path, PEFTHelper) must succeed
5. Write `$WORKSPACE/analysis.md` with:
   - Version table (source vs target)
   - Per-catalog-row shim decision (apply / skip-already-correct / new-rename)
   - VLLMHijack verification results per target
   - If any verification returns unexpected → flag as unclassified and
     stop for review (don't guess a shim).

### Phase B — Code gen

1. `cd upstream/<consumer> && git checkout $UPSTREAM_REF && git checkout -b ascend-vllm-upg-$SESSION_TAG`
2. For each shim from analysis.md (CP-002, CP-004, EC-03), verify the
   current source state:
   - If already has the try/except or hasattr gate → leave alone
     (avoid gold-plating — the catalog's patterns are backcompat-safe)
   - If not → apply the shim form from the catalog verbatim
   - Per-file `python3 -m py_compile`; `git commit` per shim
3. Run `static_check.py --files <edited> --import-package verl`.
   Must exit 0.
4. Optional container dry-import:
   ```
   python3 $VLLM_UPGRADE_EXPERT_ROOT/scripts/static_check.py \
       --import-package verl \
       --container-import-image $TARGET_IMAGE \
       --container-import-live-source /home/z00637938/workspace/easyr1-npu/upstream/$UPSTREAM_CONSUMER
   ```
   Catches module-init import breakage that local py_compile misses.
5. **Backcompat verify on SOURCE image**: re-run V1.3 rollout smoke
   on source (`--image-family <source-family>`) with shims applied.
   Must still produce coherent output (V1.3 marker). If not → shim
   not backcompat; fix, loop.

### Phase C — Deploy + build/reuse

If target image is user-provided (normal case): `--reuse-image` path.
If needs build: use a dedicated Dockerfile.npu-vllm-* (rare — usually
the target image already has the new vllm-ascend).

```
bash $VLLM_UPGRADE_EXPERT_ROOT/scripts/deploy_to_a3.sh \
    --branch ascend-vllm-upg-$SESSION_TAG \
    --reuse-image $TARGET_IMAGE \
    --upstream-consumer $UPSTREAM_CONSUMER
```

Record image_id. provenance=user-provided (if reused).

### Phase D — Validation smoke

**V1.3 is the primary vllm exercise** — it stresses vllm-ascend's full
generate path. V1.4 is secondary.

1. OL-05 chip precheck. OL-05b: V1.3 = 1 chip (chip 0), V1.4 = 2 chips
   (chips 0,1).
2. V1.3 rollout smoke:
   ```
   bash $VLLM_UPGRADE_EXPERT_ROOT/scripts/smoke_validate.sh \
       --rung V1.3 --image-tag $TARGET_IMAGE \
       --image-family <v1|v2> --chips 0
   ```
   Must find marker "V1.3 ROLLOUT SMOKE PASSED".
3. V1.4 training smoke:
   ```
   bash $VLLM_UPGRADE_EXPERT_ROOT/scripts/smoke_validate.sh \
       --rung V1.4 --image-tag $TARGET_IMAGE \
       --image-family <v1|v2> --chips 0,1 \
       --metrics-jsonl /home/.../checkpoints/easy_r1/v14_smoke/experiment_log.jsonl
   ```
   step-1 entropy_loss must land in TARGET image's V1.4 band (not
   source's — G3 band discipline).
4. On failure:
   - V1.3: grep log for traceback; match to catalog first, then
     `ERROR_CORRECTIONS.md` of any sibling expert
   - V1.4: EC-12 (config drift) should be 0-prob since your edits
     don't touch canonical config; if hit → something else. EC-13 stale
     checkpoint if jsonl only has val row.

### Phase E — Exit with handoff

**WATCHDOG-SAFE discipline** (from 2026-04-22 E2E post-mortem):

1. Write PROGRESS.md incrementally per phase — NOT a big end-of-run dump.
2. Run cleanup:
   ```
   bash $VLLM_UPGRADE_EXPERT_ROOT/scripts/cleanup_session.sh \
       --session-tag $SESSION_TAG \
       --keep-user-provided $TARGET_IMAGE
   ```
   Default `Cleanup: partial (target image preserved for orchestrator handoff)`.
3. Sign PROGRESS.md:
   ```
   Cleanup: partial (target image preserved for orchestrator handoff)
   Handoff: done
   Worker signed: vllm-upgrade-worker <ISO-UTC>
   ```

### Final chat message to caller — KEEP TERSE (≤500 words)

Return the Handoff JSON payload (schema below) + PROGRESS.md path +
self-reported OL violations. DO NOT re-summarize work already on disk
in prose. The orchestrator re-reads PROGRESS.md as source of truth.

```json
{
  "session_tag": "vllm-upg-...",
  "image_tag": "...",
  "image_id": "<sha256>",
  "branch": "ascend-vllm-upg-...",
  "v13_rollout_pass": true,
  "v14_step1_entropy_loss": <float>,
  "target_image_baseline_band": [<low>, <high>],
  "source_backcompat_verified": true,
  "shims_applied": [...],
  "hijack_points_verified": [...],
  "provenance": {"produced_by": "vllm-upgrade-worker"},
  "cleanup": "partial (image preserved)"
}
```

## Stop hook verifies

1. G2 static_check on edited .py (all exit 0).
2. G3: V1.3 marker OR V1.4 entropy_loss cited in target band.
3. OL-09: PROGRESS.md has MODE, SOURCE_IMAGE, TARGET_IMAGE, UPSTREAM_REF,
   Handoff.
4. OL-04b: Cleanup field present.

## See also

- `SKILL.md` / `state_machine.yaml` / `README.md`
- `references/patterns/domains/vllm-rename-catalog.md` — version ledger
- `../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md` — universal OLs
- `../transformers/upgrade-expert/` — sibling expert; read for structure
  only (its code is NOT your domain; OL-03 allows reading its KB tabs
  for reference, NOT its drill branch content which is its own prior art)
