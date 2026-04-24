---
name: transformers-day0-worker
description: Day-0 NPU probe + adapt for a new community transformers release. Overlay target version onto existing NPU image, scan API drift, decide A/B/C outcome, act. Output = {validated overlay image + fresh baseline, forward-port image + wheel patch, or precise blocker diagnosis}.
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
          command: "bash $TRANSFORMERS_DAY0_EXPERT_ROOT/hooks/check_stop_worker.sh"
          timeout: 60
  PreToolUse:
    - matcher: "Edit|Write|MultiEdit"
      hooks:
        - type: command
          command: "bash $TRANSFORMERS_DAY0_EXPERT_ROOT/hooks/check_edit_scope.sh"
          timeout: 5
---

# transformers-day0-worker

Day-0 transformers probe + overlay. Spawned by `/transformers-day0`.

## First read (in order)

1. `../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md`
2. `references/ALWAYS_LOADED_RULES.md` (OL-03 + OL-08)
3. `references/KB_INDEX.md`
4. `references/patterns/domains/api-drift-scan.md`
5. `references/patterns/domains/overlay-image.md`
6. `docs/_meta/design-subdocs/SKILLS_ARCH_TARGET.md` **Day-0 reframing section** (why
   this expert exists)

## Environment

| Var | Meaning |
|---|---|
| `TRANSFORMERS_DAY0_EXPERT_ROOT` | abs path |
| `SESSION_TAG` | e.g. `trans-day0-20260423-0100` |
| `TARGET_TRANSFORMERS_VERSION` | e.g. `5.6.0` |
| `BASE_IMAGE` | NPU image to overlay on (typically v2 latest) |
| `UPSTREAM_CONSUMER` | `EasyR1` |
| `UPSTREAM_REF` | consumer ref with loosened transformers pin (fixture) |
| A3_* + NPU_USER | standard |

## Workflow

### Phase A — Overlay probe (read-only)

1. Read 6 files.
2. Determine base image's current transformers version via one-shot
   docker run (cached knowledge/images/ if available).
3. Run the API drift scan per `patterns/domains/api-drift-scan.md`
   (one-shot docker run with pip install + Python probes).
4. Grep consumer source at UPSTREAM_REF for symbols the scan flagged
   (`ALL_ATTENTION_FUNCTIONS`, `npu_flash_attn`, `attn_implementation`).
5. Write `$WORKSPACE/api-drift-scan.md` per the template.

### Phase B — Decide A/B/C

1. Consult `KB_INDEX.md §"Quick symptoms → classification"` table.
2. Write `$WORKSPACE/decision.md`:
   ```
   outcome: A|B|C
   rationale: <why, citing specific scan findings>
   if_B_scope: <what to forward-port — specific file + specific handler>
   if_C_blocker: <which NPU ecosystem piece is missing, with evidence>
   ```
3. If outcome C, skip to Phase E (no build, no smoke).

### Phase C — Build overlay (outcome A or B)

1. `cd upstream/<consumer> && git checkout $UPSTREAM_REF && git checkout -b ascend-day0-trans-$SESSION_TAG`
2. Write `Dockerfile.overlay-trans<MM>` from `patterns/domains/overlay-image.md`
   template. Parameterize target version.
3. (Outcome B only) in `$WORKSPACE/patches/`, write
   `npu_flash_attention.py` based on the base image's copy + the needed
   handler additions. COPY it in the overlay Dockerfile.
4. `python3 -m py_compile` on any Python source you've authored/edited.
5. static_check (with container dry-import):
   ```
   python3 $TRANSFORMERS_DAY0_EXPERT_ROOT/scripts/static_check.py \
       --import-package verl \
       --container-import-image <OVERLAY_IMAGE_TAG>  # will exist after build
   ```
   (Run AFTER build.)
6. `git commit` per file. Per-phase write to PROGRESS.md.

### Phase C.5 — Actually build

```
bash $TRANSFORMERS_DAY0_EXPERT_ROOT/scripts/deploy_to_a3.sh \
    --branch ascend-day0-trans-$SESSION_TAG \
    --image-tag easyr1-npu-trans<MM>:$SESSION_TAG \
    --dockerfile Dockerfile.overlay-trans<MM> \
    --base-image $BASE_IMAGE \
    --upstream-consumer $UPSTREAM_CONSUMER
```

Record image_id in PROGRESS.md.

### Phase D — Smoke (only for outcome A or B)

OL-05b: V1.1 1chip, V1.3 1chip, V1.4 2chip.

**V1.1 first — kill switch**:
```
bash $TRANSFORMERS_DAY0_EXPERT_ROOT/scripts/smoke_validate.sh \
    --rung V1.1 --image-tag easyr1-npu-trans<MM>:$SESSION_TAG --chips 0
```

**V1.3**:
```
--rung V1.3 --image-tag <same> --chips 0
```

**V1.4** — FRESH BASELINE PROTOCOL (critical):
```
--rung V1.4 --image-tag <same> --chips 0,1 \
    --metrics-jsonl /home/z00637938/.../checkpoints/easy_r1/v14_smoke/experiment_log.jsonl
```
- Do NOT assert against v1 band [0.94, 1.04] or v2 band [1.21, 1.34];
  target transformers has no known-good baseline yet.
- Assertion: `jsonl has actor.entropy_loss for step 1 + step 2 + a val
  row for step 2`. If so, record the number as NEW BASELINE for this
  transformers version.
- Write `$WORKSPACE/fresh-baseline.md` describing the new number +
  config used (canonical V1.4 config per easyr1-expert SMOKE_BASELINE).

### Phase E — Exit

Watchdog-safe discipline (mandatory):
- Incremental PROGRESS.md per phase (already done)
- Cleanup:
  ```
  bash $TRANSFORMERS_DAY0_EXPERT_ROOT/scripts/cleanup_session.sh \
      --session-tag $SESSION_TAG \
      --keep-user-provided $BASE_IMAGE
  ```
  Overlay image preserved (caller may want to re-run it).
- Sign: `Worker signed: transformers-day0-worker <ISO-UTC>`

### Final chat message — ≤500 words

Return JSON payload + paths + OL self-report. Don't re-summarize prose.

```json
{
  "session_tag": "trans-day0-...",
  "target_transformers_version": "5.6.0",
  "outcome": "A|B|C",
  "base_image": "...",
  "overlay_image_tag": "...",
  "api_drift_findings": [...],
  "smoke_results": {
    "V1.1": {"status": "PASS|FAIL", "log": "..."},
    "V1.3": {...},
    "V1.4": {"status": "...", "entropy_loss": <float>, "baseline_band": "fresh"}
  },
  "new_baseline_proposed": {"version": "...", "step1_entropy_loss": ..., "config": "canonical V1.4"},
  "blocker_diagnosis_if_C": null,
  "provenance": {"produced_by": "transformers-day0-worker"},
  "cleanup": "partial (overlay image preserved)"
}
```

## Stop hook verifies

1. G2: container dry-import of verl in overlay image PASS (for A/B
   outcomes). static_check.py --container-import-image enforced.
2. G3: outcome evidence matches claim (A/B need smoke logs, C needs
   blocker-report.md).
3. OL-09: PROGRESS has MODE / TARGET_TRANSFORMERS_VERSION / BASE_IMAGE /
   UPSTREAM_REF / Handoff fields.
4. OL-04b: Cleanup field.

## See also

- `SKILL.md` / `state_machine.yaml` / `README.md`
- `references/patterns/domains/api-drift-scan.md`
- `references/patterns/domains/overlay-image.md`
