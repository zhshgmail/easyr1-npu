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

## What this does NOT validate (yet)

* Full `glm5.GLM5Layer` forward + backward pass on NPU (one layer of the actual model). The op-level contracts match but model-level shape flow has not been driver-tested.
* `miles.train` end-to-end (blocked by sglang on NPU being non-functional — see triton-ascend #277).

Next step suggestions (in order of value):
1. Run a single-layer GLM-5 forward + backward on NPU using the dispatch path.
2. Once that passes, write a 1-iter optimizer-step driver that synthesizes inputs and runs the training-side math without sglang rollout.
