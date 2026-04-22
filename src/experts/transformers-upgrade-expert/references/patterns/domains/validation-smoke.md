# Domain — Validation smoke (target-image numerics)

**Load when**: Phase D writing the validation smoke + picking the right
baseline band.

## Scope

The validation smoke proves "the new stack runs the same workload with
similar numerics". We use a **V1.4-equivalent run** (2-chip GRPO on
Qwen2-0.5B-Instruct, padding_free=False, 2 training steps) because:

- V1.1 (device check) doesn't exercise transformers / vllm at all — not
  sensitive to the upgrade.
- V1.3 (rollout only) exercises vllm_ascend + tokenization — half the upgrade.
- **V1.4 (training) exercises the full stack** — forward, backward,
  optimizer, rollout, checkpoint saver. One smoke = full coverage.
- V1.5/V2.1/V2.2 are bonus — OL-05b says default skip in Stage 0.

## Smoke script template (target-specific)

Copy `examples/qwen2_0_5b_math_grpo_npu_smoke.sh` from the consumer's
current port branch (easyr1-expert round-4's canonical config is the
reference). Parameterize:

- `save_checkpoint_path=/opt/<consumer>/checkpoints/easy_r1/v14_validate_<target-short>`
- `experiment_name=v14_validate_<target-short>`
- `rm -rf` the save path at script top (EC-13: stale checkpoint silently
  skips training on rerun)

All other config stays identical to V1.4 canonical (disable_kl=true,
use_kl_loss=false, filter_overlong_prompts=true, logger=['console','file'],
enable_gradient_checkpointing=true, ref cpu_offload=true, etc. — see
SMOKE_BASELINE.md §"canonical config").

## CRITICAL: band picking

Pass `--image-family` to `smoke_validate.sh` matching the TARGET image:

```bash
bash scripts/smoke_validate.sh \
    --rung V1.4 \
    --image-tag $SESSION_IMAGE_TAG \
    --image-family v2 \      # ← MUST match target, NOT source
    --chips 0,1 \
    --metrics-jsonl /opt/<consumer>/checkpoints/easy_r1/v14_validate_<target-short>/experiment_log.jsonl
```

Band lookup (from `SMOKE_BASELINE.md` / `smoke_validate.sh`):

| image family | target_image | V1.4 step-1 baseline | band (±5%) |
|---|---|---|---|
| v1 | verl-8.5.0-a3 | 0.991 | [0.94, 1.04] |
| v2 | verl-8.5.2-a3 | 1.275 | [1.21, 1.34] |

**G3 hard rule**: if you conflate bands (e.g. pass `--image-family v1`
on a v2 smoke), the assertion becomes meaningless. The Stop hook's G3
check watches for this: PROGRESS.md entries citing a v2 smoke with a
v1 band are REJECTED.

## Why the two bands differ

v2 uses transformers 5.3.0.dev0 + torch_npu 2.9 + vllm 0.17 — different
fused kernels and fp internals shift numerics. The drill report
(2026-04-19) measured:

- v1 V1.4 step-1: 0.991 (fused via NPU FA on torch 2.8.0+cpu)
- v2 V1.4 step-1: 1.275 (different kernel path; Qwen2-0.5B forward verified
  numerically equivalent within noise, but the absolute entropy_loss value
  is shifted)

This is not a regression — it's the new stack's fingerprint.

## Verify

```bash
# the assertion you want to see:
✅ V1.4 PASS: entropy_loss=<1.21..1.34> in band [1.21, 1.34]
  source: jsonl (.../v14_validate_852/experiment_log.jsonl)
```

and in PROGRESS.md:
```
### V1.4 validation (2026-MM-DD...)
- image: <target-image-tag>
- smoke_image: easyr1-npu:<SESSION_TAG>
- log: /tmp/.../V1.4-easyr1-npu-<SESSION_TAG>-*.log
- step1 entropy_loss: <1.21..1.34> (target_image=v2, baseline 1.275, band [1.21, 1.34])
- source: jsonl
- produced_by: transformers-upgrade-worker
```

## Backcompat verification (pre-validation)

Before the target-image smoke, verify the shims didn't break source-image
users. Re-run the same V1.4 smoke against SOURCE_IMAGE (easyr1-expert's
existing image, via `--reuse-image`):

```bash
bash scripts/smoke_validate.sh \
    --rung V1.4 \
    --image-tag <SOURCE_IMAGE_TAG_preserved_from_round4_or_similar> \
    --image-family v1 \
    --chips 0,1 \
    --metrics-jsonl /opt/easyr1/checkpoints/easy_r1/v14_smoke/experiment_log.jsonl
```

Expect: step-1 still lands in v1 band [0.94, 1.04]. If it drifts, one of
the shims isn't backcompat-clean — fix.

Only after source-image re-smoke PASSES do we advance to target-image
validation.
