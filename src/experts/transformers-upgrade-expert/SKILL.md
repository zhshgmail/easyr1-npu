---
name: transformers-upgrade
description: >
  Swap the NPU base image to a newer stack (e.g. verl-8.5.0-a3 → verl-8.5.2-a3
  carrying transformers 4.57→5.3, vllm 0.13→0.18, torch_npu 2.7→2.9, CANN
  8.5.0→8.5.1). Produce a validated image: adapt Dockerfile, cherry-pick or
  write backcompat shims for API renames, repair platform bugs that recur on
  the new stack, run a single V1.4-equivalent validation smoke, assert
  numerics in the new image's own baseline band. Output = working image tag
  + branch + numerics evidence, consumable by easyr1-expert via --reuse-image.

  Usage: /transformers-upgrade --target-image <TAG> --source-image <TAG>
                                [--upstream-consumer <repo-name> --upstream-ref <sha>]
argument-hint: >
  target-image: e.g. quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5
  source-image: current working baseline (default: v1, verl-8.5.0-a3)
  upstream-consumer: which consumer repo's V1.4 to use as smoke target (default EasyR1)
  upstream-ref: which commit of the consumer to validate against (default the last known-green ref for source-image)
context: inline
---

# /transformers-upgrade — NPU base-image upgrade orchestrator (Stage 0 single-dep)

**Your role**: spawn `transformers-upgrade-worker` agent, read its artifacts,
decide done/retry, hand the resulting image tag back to the caller.

## Canonical workflow spec

**The authoritative definition of phases, invariants, and prohibited actions
lives in `src/experts/transformers-upgrade-expert/state_machine.yaml`**
(machine-readable, consumed by workflow critic at runtime).

## Workflow overview

```
P0: Parse args + load env (target/source image, consumer repo + ref, A3 conn)
P1: Inspect target image (pip freeze inside, triton-ascend integrity check)
P2: Checkout consumer repo at the baseline-working ref; branch off to
    ascend-upg-{SESSION_TAG}
P3: Spawn transformers-upgrade-worker → Phase A/B/C/D
    * A: diff pip-freeze(source) vs pip-freeze(target); enumerate
         backcompat-break candidates; write analysis.md
    * B: write Dockerfile.npu-{target-short} + apply minimum shim set
         from references/patterns (no_init_weights move, SamplingParams
         read-only, triton coexist repair, etc.)
    * C: deploy_to_a3 + docker build on A3; image tag easyr1-npu:{SESSION_TAG}
    * D: run V1.4-equivalent validation smoke (2-chip GRPO) with
         smoke_validate.sh, assert step-1 entropy_loss in the target
         image's own baseline band (from SMOKE_BASELINE.md, v2 section)
P4: Return handoff payload to caller (orchestrator or user):
    {image_tag, branch, step1_entropy_loss, step2_entropy_loss, provenance}
```

Full sequence + invariants: see `state_machine.yaml`.

## Stage 0 constraints

- **Single dep per session**: don't bundle multiple upgrade drivers in one
  run. If the orchestrator needs two (e.g. transformers AND torch separately),
  invoke the upgrade expert twice.
- **Backcompat-with-source MANDATORY**: all shims must leave the source-image
  (v1) smoke PASS intact. Verified by re-running a smoke on v1 after
  committing shims (Phase B exit check).
- **Numerics assertion**: must land in the **target image's own baseline
  band**, not v1's. v2 band is different from v1 band (see SMOKE_BASELINE.md
  §v2 row). Conflating bands is a false-green trap.

## Invariants the skill enforces

- **G1** (universal): orchestrator (you) must NOT directly Edit files under
  `upstream/<consumer>/verl/` or `upstream/<consumer>/Dockerfile*`. All edits
  go through worker. Enforced by `hooks/check_edit_scope.sh`.
- **G2** (universal): code claimed ready must pass `static_check.py`
  (py_compile + dry-import of the consumer's top-level package). Enforced by
  worker's Stop hook.
- **G3** (universal): any PASS/green/validated claim must reference a log
  file path + numeric entropy_loss in the target-image's baseline band.

## Return payload to caller

```json
{
  "image_tag": "easyr1-npu:trans-upg-{SESSION_TAG}",
  "image_id": "<sha256>",
  "branch": "ascend-upg-{SESSION_TAG}",
  "step1_entropy_loss": <float>,
  "step2_entropy_loss": <float>,
  "target_image_baseline_band": [<low>, <high>],
  "v1_backcompat_verified": <bool>,
  "provenance": {
    "shims_applied": ["EC-02", "EC-03", ...],
    "platform_bugs_addressed": ["NPU-BUG-001", "NPU-BUG-004", ...],
    "produced_by": "transformers-upgrade-worker"
  },
  "cleanup": "clean | partial | skipped <reason>"
}
```

## See also

- `README.md` — product definition
- `state_machine.yaml` — authoritative workflow spec
- `agent.md` — transformers-upgrade-worker definition
- `references/ALWAYS_LOADED_RULES.md` — agent's mandatory first read
- `../_shared/references/ALWAYS_LOADED_UNIVERSAL.md` — universal OL rules
- `docs/design/SKILLS_ARCH_TARGET.md` — target architecture (multi-expert layer)
