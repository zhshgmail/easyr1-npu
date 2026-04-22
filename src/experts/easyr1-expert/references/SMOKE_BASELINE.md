# Smoke ladder baseline numerics (per image)

> Load when: Phase D asserts each rung's result.
> Each rung's assertion range is "baseline_value ± 5%" unless noted.

These are **empirical baselines** established by actual runs on this project
(see porting-journal 2026-04-17..22). Stage 0 Acceptance T0.3 requires agent
to reproduce V1.4 numerics within band.

---

## v1 image: `quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest`

| Rung | Config | Step 1 entropy_loss | Step 2 entropy_loss | Step 1 grad_norm | Duration |
|---|---|---|---|---|---|
| V1.1 | device smoke (no training) | n/a | n/a | n/a | <1 min |
| V1.3 | vllm_ascend rollout (3 prompts) | n/a | n/a | n/a | ~5 min |
| **V1.4** | 2-chip GRPO 2-step, padding_free=False, ulysses=1 | **0.991** (±5% = [0.94, 1.04]) | 1.263 | 1.43 | ~8 min |
| V1.5 | 4-chip GRPO 2-step | 1.127 (4-chip differs from V1.4 — HCCL gathering affects numerics) | 1.163 | — | ~10 min |
| V2.1 | 2-chip, padding_free=True, ulysses=1 | ≈ 0.991 (within V1.4 band) | 1.264 | 1.43 | ~8 min |
| V2.2 | 4-chip, padding_free=True, ulysses=2 | — | — | — | TBD |

**PASS criteria per rung**:

- V1.1: `torch_npu.npu.device_count() > 0`; `is_npu_available() == True`; tensor round-trip OK
- V1.3: 3 prompts each generate ≥1 coherent token, no crash
- **V1.4**: step 1 entropy_loss in `[0.94, 1.04]` + step 2 completes + checkpoint saved
- V1.5: HCCL collectives don't crash, 2 steps complete
- V2.1: step 1 entropy_loss in V1.4 band (padding_free should be numerically equivalent)
- V2.2: 4-chip + ulysses=2 doesn't crash; numerics still in V1.4 band

---

## v2 drill image: `quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5`

v2 uses transformers 5.3.0.dev0 + torch_npu 2.9 + vllm 0.17 — numerics diverge from v1 (different kernels, different fp internals).

| Rung | Config | Step 1 entropy_loss | Step 2 entropy_loss | Duration |
|---|---|---|---|---|
| V1.4 | 2-chip, padding_free=False | **1.275** (±5% = [1.21, 1.34]) | 0.895 (grad_norm 2.07) | ~9 min |
| V2.2 | 4-chip, padding_free=True, ulysses=2 | 1.434 (original drill config; padding_free=True changes numerics from V1.4) | 1.58 | ~9 min |

Drill V2.2 20-step trajectory: entropy_loss ∈ `[1.31, 1.83]`, grad_norm max ~3.2.

---

## Reference smoke config for V1.4 (canonical baseline config)

The 0.991 baseline is **only valid** if the smoke config matches these critical
parameters. 2026-04-22 round 3 saw step1 entropy_loss=1.276 from a variant
config — drift came from KL defaults, not the port logic.

Critical parameters that shift entropy_loss numerics:

| Parameter | Canonical value | Why it matters |
|---|---|---|
| `algorithm.disable_kl` | `true` | **Largest effect.** Master defaults KL on; KL term raises entropy_loss. |
| `algorithm.use_kl_loss` | `false` | Same reason as above. |
| `data.max_prompt_length` | `512` | Different truncation changes token mix. |
| `data.max_response_length` | `256` | Same reason. |
| `worker.rollout.n` | `2` | Smaller n = higher variance step 1. |
| `worker.rollout.gpu_memory_utilization` | `0.4` | (≤0.5 OK; memory pressure doesn't change numerics but `enforce_eager`/`chunked_prefill` do.) |
| `worker.rollout.enforce_eager` | `true` | Non-eager = vllm captures CUDA graphs, different NPU dispatch. |
| `worker.rollout.enable_chunked_prefill` | `false` | Chunked prefill routes differently on NPU. |
| `worker.actor.model.enable_gradient_checkpointing` | `true` | Slightly different fp accumulation order. |
| `worker.ref.fsdp.enable_cpu_offload` | `true` | Ref model on CPU or NPU changes which kernel runs. |
| `worker.ref.offload.offload_params` | `true` | Same reason. |
| `data.rollout_batch_size` | `4` | Batch size interacts with group advantage math. |
| `worker.actor.global_batch_size` | `4` | Same reason. |
| `trainer.logger` | `['console', 'file']` | **Required for jsonl assertion**. `['console']` alone = no experiment_log.jsonl; `['file']` alone = no stdout summary. Best to include both. |

Use `examples/qwen2_0_5b_math_grpo_npu_smoke.sh` from the `ascend-port` branch
as the ground truth. Copy it verbatim — changes require re-establishing
baseline (empirical run + journal entry).

## How to use these in assertions

`scripts/smoke_validate.sh` V1.4+ first reads `experiment_log.jsonl` (written
by EasyR1's `FileLogger`), falls back to stdout grep only if jsonl missing.
See `--metrics-jsonl` flag.

```bash
smoke_validate.sh \
  --rung V1.4 \
  --image-tag easyr1-npu:<SESSION_TAG> \
  --image-family v1
# → exits 0 if step-1 entropy_loss in [0.94, 1.04]
# → exits 1 with specific reason if out of band or no entropy_loss found
```

---

## Updating this file

Any new empirically-observed baseline (from a successful real smoke run with
known-good port) should append a row here. Unverified numbers go in
PROGRESS.md of the specific session, not here.

**Never edit the v1 V1.4 row without evidence**. That number (0.991) is the
most-verified datum in this project (runs on 2026-04-17, 2026-04-22 round 1
ascend-port copy, 2026-04-22 round 1 ascend-port direct).
