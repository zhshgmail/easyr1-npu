# Smoke ladder — entry points

The V1.x / V2.x / drill smoke scripts themselves live in the **EasyR1 fork**
(`zhshgmail/EasyR1`, branch `ascend-port` or `ascend-port-transformers-upgrade`)
under `examples/`. They're kept there because they reference EasyR1's own
`verl/` package layout and must move in lockstep with the code changes.

This directory provides:
- `run.sh` — convenience wrapper: `cd` into the cloned EasyR1 tree, then
  invoke the right `examples/qwen2_0_5b_math_grpo_npu_smoke*.sh` inside the
  container via `scripts/run-npu-container.sh`.
- This README — the **index** of which smoke level validates what, and
  which example script implements it.

## Smoke ladder index

| Level | Config | Example script (in EasyR1 fork) | Target image |
|---|---|---|---|
| V1.1 | device accessors | `scripts/smoke_v11_device.py` (in **this** repo) | either |
| V1.2 | tensor round-trip + vendored bert_padding | same V1.1 script | either |
| V1.3 | vllm_ascend rollout | `scripts/smoke_v13_rollout.py` (in **this** repo) | either |
| V1.4 | GRPO 2-step, 2 chips, padding_free=False, ulysses=1 | `examples/qwen2_0_5b_math_grpo_npu_smoke.sh` | 8.5.0 / 8.5.2 |
| V1.5 | GRPO 2-step, 4 chips | `examples/qwen2_0_5b_math_grpo_npu_smoke_4chip.sh` | 8.5.0 / 8.5.2 |
| V2.1 | GRPO 2-step, 2 chips, padding_free=True | `examples/qwen2_0_5b_math_grpo_npu_smoke_v2_1_padfree.sh` | 8.5.0 / 8.5.2 |
| V2.2 | GRPO 2-step, 4 chips, padding_free=True, ulysses=2 | `examples/qwen2_0_5b_math_grpo_npu_smoke_v2_2_ulysses.sh` | 8.5.0 / 8.5.2 |
| drill 20-step | V2.2 with `max_steps=20`, transformers 5.x | `examples/qwen2_0_5b_math_grpo_npu_smoke_v2_2_ulysses.sh` with `max_steps=20` override | 8.5.2 only |

## Baseline numerics (for comparison)

Recorded on `quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest`,
`ascend-port` branch head `ecce71d`:

| Smoke | Step 1 entropy_loss | Step 1 grad_norm |
|---|---|---|
| V1.4 | 0.991 → 1.263 (step 2) | 1.43 (step 1) |
| V2.1 | 0.991 → 1.264 (step 2) | 1.43 (step 1) |
| V2.2 | similar band | similar band |

Recorded on `quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5`,
drill branch `ascend-port-transformers-upgrade` head `2fd9337`:

| Smoke | Step 1 entropy_loss | 20-step entropy_loss band |
|---|---|---|
| drill 2-step | 1.434 (matches V1.4) | — |
| drill 20-step | 1.434 → {1.31, …, 1.83} | `grad_norm` max ~3.2 |

If your reproduction drifts by more than ±5% at step 1, something's wrong with
the deps — stop and re-inspect via `skills/npu-image-inspect/`.

## How to run

See `docs/easyr1/PORT-GUIDE.md` §"Run the smoke ladder" for the full recipe.
Short version:

```bash
# 1. Clone the EasyR1 fork on the A3 host
git clone -b ascend-port https://github.com/zhshgmail/EasyR1.git ~/workspace/EasyR1

# 2. Download Qwen2-0.5B-Instruct to /data/$USER/models/
hf download Qwen/Qwen2-0.5B-Instruct --local-dir /data/$USER/models/Qwen2-0.5B-Instruct

# 3. Pull the image (if not already on host)
docker pull quay.nju.edu.cn/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest

# 4. Build the layered image
cd ~/workspace/EasyR1 && docker build -t easyr1-npu:ascend-port -f Dockerfile.npu .

# 5. Run V1.4 smoke (see PORT-GUIDE.md §Run the smoke ladder for full command)
bash /path/to/easyr1-npu/scripts/run-npu-container.sh \
     --chips 0,1 \
     --live-source ~/workspace/EasyR1 \
     -- bash /opt/easyr1/examples/qwen2_0_5b_math_grpo_npu_smoke.sh
```
