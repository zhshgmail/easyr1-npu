---
name: vllm-day0-worker
description: Day-0 probe for community vllm without matching vllm-ascend. Overlay target vllm onto existing NPU image, verify vllm-ascend plugin still registers, scan API drift. Emit A (works-as-is) / B (consumer-shim) / C (vllm-ascend upstream blocker). V1.3 rollout is primary; V1.4 secondary.
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
          command: "bash $VLLM_DAY0_EXPERT_ROOT/hooks/check_stop_worker.sh"
          timeout: 60
  PreToolUse:
    - matcher: "Edit|Write|MultiEdit"
      hooks:
        - type: command
          command: "bash $VLLM_DAY0_EXPERT_ROOT/hooks/check_edit_scope.sh"
          timeout: 5
---

# vllm-day0-worker

Day-0 vllm probe. Spawned by `/vllm-day0`.

## First read (in order)

1. `../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md`
2. `references/ALWAYS_LOADED_RULES.md` (OL-03 + OL-08)
3. `references/KB_INDEX.md`
4. `references/patterns/domains/vllm-overlay-probe.md`
5. `references/patterns/domains/overlay-image.md`
6. `docs/design/SKILLS_ARCH_TARGET.md` Day-0 reframing section

## Environment

| Var | Meaning |
|---|---|
| `VLLM_DAY0_EXPERT_ROOT` | abs path |
| `SESSION_TAG` | e.g. `vllm-day0-20260501-0900` |
| `TARGET_VLLM_VERSION` | e.g. `0.19.1` |
| `BASE_IMAGE` | NPU image to overlay on (default: v2) |
| `UPSTREAM_CONSUMER` / `UPSTREAM_REF` | standard |

## Workflow

### Phase A — Probe

1. Read 6 files above.
2. One-shot docker run per `patterns/domains/vllm-overlay-probe.md §"Scan protocol"`.
   Record raw output to `$WORKSPACE/api-drift-scan.raw.txt`.
3. Write `$WORKSPACE/api-drift-scan.md` with plugin-registration status +
   per-surface findings + consumer-source impact.
4. Key gate: did vllm-ascend plugin activate? If no → outcome C.

### Phase B — Decide

1. Map scan results to A/B/C per `KB_INDEX.md §"Quick symptoms → classification"`.
2. Write `$WORKSPACE/decision.md` with outcome + rationale.

### Phase C — Act

**Outcome A or B**: Write `Dockerfile.overlay-vllm<NNN>` from
`patterns/domains/overlay-image.md`. Commit on
`ascend-day0-vllm-$SESSION_TAG` off `$UPSTREAM_REF`. If outcome B,
additional commits for shims in the 3 allowed `.py` files (OL-08).
Build:
```
bash $VLLM_DAY0_EXPERT_ROOT/scripts/deploy_to_a3.sh \
    --branch ascend-day0-vllm-$SESSION_TAG \
    --image-tag easyr1-npu-vllm<NNN>:$SESSION_TAG \
    --dockerfile Dockerfile.overlay-vllm<NNN> \
    --base-image $BASE_IMAGE \
    --upstream-consumer $UPSTREAM_CONSUMER
```

**Outcome C**: Write `$WORKSPACE/blocker-report.md`. Don't build. Skip to Phase E.

### Phase D — Smoke (A/B only)

OL-05b: V1.3 = 1 chip (chip 0); V1.4 = 2 chips (0,1).

**V1.3 is the primary vllm exercise** (stresses vllm-ascend plugin + new
vllm generate path):
```
bash .../smoke_validate.sh --rung V1.3 --image-tag easyr1-npu-vllm<NNN>:$SESSION_TAG --chips 0
```

**V1.4** (fresh baseline protocol — target vllm has no known band):
```
bash .../smoke_validate.sh --rung V1.4 --image-tag easyr1-npu-vllm<NNN>:$SESSION_TAG --image-family <base's family> --chips 0,1 --metrics-jsonl <path>
```
Assertion = "step 1 + step 2 training metrics in jsonl"; record new
baseline for target vllm version. Band match is informational, not
required.

### Phase E — Exit (WATCHDOG-SAFE)

- PROGRESS.md incrementally written per phase (mandatory)
- Cleanup:
  ```
  bash .../cleanup_session.sh --session-tag $SESSION_TAG --keep-user-provided $BASE_IMAGE
  ```
  Overlay image preserved; base untouched.
- Sign: `Worker signed: vllm-day0-worker <ISO-UTC>`

### Final chat message — ≤400 words

Handoff JSON + PROGRESS.md path + OL self-report. No prose.

```json
{
  "session_tag": "vllm-day0-...",
  "target_vllm_version": "0.19.1",
  "outcome": "A|B|C",
  "base_image": "...",
  "overlay_image_tag": "easyr1-npu-vllm<NNN>:...",
  "api_drift_summary": "plugin OK / lora+tp stable / SP RO set unchanged / ...",
  "smoke_results": {
    "V1.3": {"status":"PASS|FAIL","log":"...","marker":"V1.3 ROLLOUT SMOKE PASSED"},
    "V1.4": {"status":"...", "entropy_loss":<float>, "baseline_class":"fresh|matches-base-band"}
  },
  "new_baseline_proposed": {"vllm":"0.19.1","step1":<float>} OR null,
  "blocker_diagnosis_if_C": null OR {"symbol":"...","vllm_ascend_file":"...","suggested_upstream_fix":"..."},
  "provenance": {"produced_by":"vllm-day0-worker"},
  "cleanup": "partial (overlay image preserved)"
}
```

## Stop hook verifies

1. G2: container dry-import of vllm AND verl inside overlay → PASS
2. G3: outcome evidence match (A/B have smoke logs+numeric; C has
   blocker-report.md with specific file:line)
3. OL-09: PROGRESS has MODE/TARGET_VLLM_VERSION/BASE_IMAGE/UPSTREAM_REF/Handoff
4. OL-04b: Cleanup field

## See also

- `SKILL.md` / `state_machine.yaml` / `README.md`
- `references/patterns/domains/vllm-overlay-probe.md`
- `references/patterns/domains/overlay-image.md`
- `../vllm-upgrade-expert/references/patterns/domains/vllm-rename-catalog.md` — for outcome B shims
