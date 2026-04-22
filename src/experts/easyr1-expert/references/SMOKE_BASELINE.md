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

## How to use these in assertions

`scripts/smoke_validate.sh` reads this file via a simple ini-like parser:

```bash
smoke_validate.sh \
  --rung V1.4 \
  --image v1 \
  --log /tmp/$USER/easyr1-logs/v14_{tag}.log
# → exits 0 if log has "entropy_loss: X.XXX" in [0.94, 1.04]
# → exits 1 with specific reason if out of band or log missing
```

---

## Updating this file

Any new empirically-observed baseline (from a successful real smoke run with
known-good port) should append a row here. Unverified numbers go in
PROGRESS.md of the specific session, not here.

**Never edit the v1 V1.4 row without evidence**. That number (0.991) is the
most-verified datum in this project (runs on 2026-04-17, 2026-04-22 round 1
ascend-port copy, 2026-04-22 round 1 ascend-port direct).
