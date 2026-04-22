# Validation smoke baseline per target image

> Load at Phase D when asserting the new image's numerics.
> Reference: easyr1-expert/references/SMOKE_BASELINE.md has the full V1.1-V2.2
> table per image. This file focuses on the **V1.4 validation row** that this
> expert uses as its single assertion.

## V1.4 per image (2-chip GRPO, canonical config)

| image family | base image | V1.4 step-1 baseline | band (±5%) | source |
|---|---|---|---|---|
| v1 | quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest | **0.991** | [0.94, 1.04] | 2026-04-17; round 3 iter3 replay 0.9922; round 4 iter7 0.9583 |
| v2 | quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5 | **1.275** | [1.21, 1.34] | 2026-04-19 transformers-upgrade drill |

## Canonical smoke config (MUST match to hit the band — EC-12)

Copy from easyr1-expert round-4's
`examples/qwen2_0_5b_math_grpo_npu_smoke.sh` verbatim. Critical parameters
(any deviation shifts numerics off-baseline):

- `algorithm.disable_kl=true`, `algorithm.use_kl_loss=false` (biggest lever)
- `data.filter_overlong_prompts=true` (round 4 iter7 evidence)
- `data.max_prompt_length=512`, `data.max_response_length=256`
- `worker.actor.model.enable_gradient_checkpointing=true`
- `worker.ref.fsdp.enable_cpu_offload=true`, `worker.ref.offload.offload_params=true`
- `worker.rollout.enforce_eager=true`, `enable_chunked_prefill=false`
- `worker.actor.use_torch_compile=false` (NPU-BUG-003 variant recurs)
- `trainer.logger=['console','file']` (EC-11 — jsonl needs 'file')
- `trainer.find_last_checkpoint=false` + rm -rf the save path at script
  start (EC-13: stale checkpoint silently skips training)

Full list: easyr1-expert `SMOKE_BASELINE.md` §"Reference smoke config for V1.4".

## Backcompat baseline (source image)

When verifying shims don't break source-image users, pin the source V1.4:

- Source is v1 → expect step-1 in [0.94, 1.04] (band above, v1 row).

If it drifts, one of the shims isn't backcompat-clean.

## When a target image is NEW (no prior baseline)

First-time targets have no known baseline band yet. Procedure:

1. Run the V1.4 smoke on the new image to establish **its** first number.
2. Record in PROGRESS.md §"unverified target numerics" — DO NOT promote
   to this file's table until verified across at least 2 sessions.
3. Until verified, assertion mode is **"no crash + 2 training steps + val
   step + jsonl has an entropy_loss number"** — not a band check. Stop
   hook G3 still requires the numeric be cited, just not within-band.
4. After a second session reproduces within 5% of the first, append to the
   table here and start enforcing the band.

## Updating this file

Only after TWO confirming sessions (different days, different SESSION_TAGs,
same target image). Single-session numerics go in workspace/*/PROGRESS.md,
not here.

**Never edit v1 V1.4 = 0.991 or v2 V1.4 = 1.275 without fresh evidence**.
Both are multi-session verified (v1: 4+ runs across 2026-04-17..22;
v2: 1 drill + 1 re-check 2026-04-19).
