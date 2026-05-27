# Miles tilelang ops on Ascend A3 — end-to-end validation

## Status: ✅ all 4 ops PASS on NPU via miles' contract surfaces

Validation run on tlrescue container, Ascend A3 / 910C, 2026-05-27.

| Op | Status | Precision (vs CPU autograd fp32 ref) |
|---|---|---|
| `indexer_fwd_interface`     | ✅ PASS | max abs err 1.0e-2, rel 0.001 |
| `indexer_bwd_interface`     | ✅ PASS | dq 4.2e-4 / dw 4.5e-4 / dk 3.7e-4 |
| `sparse_mla_fwd_interface`  | ✅ PASS | max abs err 6.7e-4 |
| `sparse_mla_bwd`            | ✅ PASS | dq cosine 1.0000, max abs err 1.6e-4 |

## Where the code lives

* **NPU dispatch wrappers**: `zhshgmail/miles` fork, branch `npu-tilelang-dispatch`, commit `cd982a8`.
* **Dispatch path**: each of miles' 4 `*_interface` functions in `miles_plugins/models/glm5/ops/tilelang_*.py` carries a top-of-body `if hasattr(q, "is_npu") and q.is_npu:` guard that routes to `miles_plugins/models/glm5/ops/_npu/{indexer,sparse_mla}.py`.
* **NPU kernels themselves**: bundled inside `miles_plugins/models/glm5/ops/_npu/_*.py` (copied from `tile-ai/tilelang-mlir-ascend` PR #59 branch + the R-KA-13 E5 sparse_mla_bwd from `t33-sparse-mla-fwd-port-and-tdynamic`).

## Open NPU runtime bugs worked around at this layer

* **R-KA-13** (vsub schedule-locality silent zero) — E5 fix baked into the bwd kernel itself
* **R-KA-14** (multi-block scatter NaN) — `indexer_bwd` invoked per-seq-position (SEQ=1 grid)
* **R-KA-15** (atomic_addx4 all-zero src 6e37) — wrapper short-circuits when `grad_scores.abs().max() < 1e-30`

Upstream issues filed at AscendNPU-IR #247 / #248 / #249 and triton-ascend #277.

## How to reproduce

On A3, inside the `tlrescue` container:

```bash
ssh -p 443 root@115.190.166.102
docker exec tlrescue bash -c "
  cd /home/z00637938/workspace/miles
  PYTHONPATH=/home/z00637938/workspace/miles:/home/z00637938/workspace/tilelang-mlir-ascend \
    python -m miles_plugins.models.glm5.ops._npu._validate_npu
"
```

The harness imports the dispatch wrappers directly (bypassing miles' GPU kernel modules whose `@tilelang.jit(...)` decorators reference CUDA-only symbols at module load).

## Caveat: shape-adapter dependencies

The dispatchers auto-pick power-of-2 block sizes that divide the runtime shapes. For non-trivial miles workloads (large topk, large seq_len_kv), block selection may need tuning — the current auto-pick is "largest power-of-2 dividing N, capped at the kernel's default cap". Adjust in `_npu/indexer.py::_largest_pow2_divisor` and `_npu/sparse_mla.py` if a real glm5 workload selects sub-optimal blocks.

## End-to-end autograd smoke — ALSO PASSES (2026-05-27)

Beyond the per-op contracts above, we drive miles' actual `IndexerFunction.apply()` and `SparseMLA.apply()` from one autograd graph on NPU and check a single `loss.backward()` populates gradients on every input.

* Driver: `miles_plugins/models/glm5/ops/_npu/_e2e_autograd.py`
* Shapes: miles canonical (D_V=512, D_TAIL=64, DQK=576), H_MLA=16, H_indexer=8
* Result: ✅ PASS — all 5 input gradients finite + nonzero (`index_q` 6.4e-1, `index_k` 1.9, `weights` 3.1, `q_mla` 2.5e-5, `kv_mla` 4.6e-4)

## Mini RL train-step — ALSO PASSES (2026-05-27)

Smallest end-to-end training loop that exercises all 4 NPU kernels through miles' contract surfaces, with a mock GRPO-surrogate loss + Adam step.

* Driver: `miles_plugins/models/glm5/ops/_npu/_e2e_train_step.py`
* Module: `GLM5MiniBlock(hidden=128, h_idx=8, d_idx=32, h_mla=16)`, total 2.3 M params
* One step: forward → backward → optim.step() — result: ✅ PASS
* Loss = -0.079 (finite), grad norm = 4.93 (sensible), proj_index_q weight delta after step = 1e-3 (Adam lr at 1e-3)

## Outstanding gaps to "full miles.train RL"

These are the remaining barriers between the mini smoke and `python train.py ...` running on NPU:

1. **sglang on NPU is broken** (triton-ascend `extract_slice` / `AttrsDescriptor` ABI mismatch, [triton-ascend#277](https://github.com/triton-lang/triton-ascend/issues/277) overlapping [#234](https://github.com/triton-lang/triton-ascend/issues/234)). Real rollout cannot run until upstream publishes a `release/3.6.x` triton-ascend wheel or the image re-pins to a working pair.
2. **No Megatron on NPU**: miles defaults to `--training-backend megatron`. On Ascend that needs `Ascend/MindSpeed-LLM` (heavy install). The miles `--training-backend fsdp` path is reachable on NPU without MindSpeed.
3. **No `Megatron-LM-miles` install in tlrescue**: only the local sources are present at `/home/z00637938/workspace/Megatron-LM-miles`; not pip-installed.

## What this does NOT validate (yet)

* Full `glm5.GLM5Layer` end-to-end (requires `megatron.core` + `transformer_engine` — CUDA-only).
* Multi-step training convergence (only single-step weight motion is verified).
* `miles.train` driver end-to-end (blocked by sglang + Megatron above).
