# [BUG] tilelang multi-block kernel with `T.atomic_addx4` scatter produces NaN at grid block 1+

> **FILED 2026-05-27**: <https://gitcode.com/Ascend/AscendNPU-IR/issues/248>

## Summary

When invoking a tilelang NPU kernel with `T.Kernel(N, is_npu=True)` for `N≥2` and the kernel includes an `T.atomic_addx4` scatter writing to a global tensor via an indexed lookup pattern, **only grid block 0 produces correct output. Grid blocks 1+ produce NaN values regardless of which destination indices they target.**

## Environment

Same as R-KA-15 issue.

## Reproducer

Full kernel: <https://github.com/zhshgmail/tilelang-mlir-ascend/blob/t33-sparse-mla-fwd-port-and-tdynamic/examples/deepseek_v4/example_lighting_indexer_bwd_kernel.py> commit `f6d30e2`.

Demonstration test (in `_smoke_bwd()` function of that file):

| seq_len (= N grid blocks) | Result |
|--------|--------|
| 1 | ✅ dQ/dKV/dW all match autograd to 1e-5 |
| 2, queries write DISJOINT halves of dKV | ❌ block 0's writes correct, block 1's writes NaN |
| 2, queries write SAME indices (concurrent atomic) | ❌ all atomic targets become NaN |
| 8, random topk indices | ❌ half of dKV rows NaN |

## What we ruled out

* Not concurrent-atomic race: tested with queries 0 and 1 writing to disjoint halves (no collision on dKV addresses) — block 1's writes still NaN.
* Not specific to atomic_addx4 src content: with R-KA-15 workaround applied (skip when src all-zero), still NaN.
* Not multi-buffer optimizer: `npuir.enable_auto_multi_buffer: False` doesn't help.

## Workaround

Invoke the kernel **per-batch-element** with `seq_len=1` each call (single grid block), Python-loop over the batch dimension. This matches miles framework's `batched_indexer_bwd` natural pattern, so it's the production-correct call shape anyway.

## Hypothesis

Some kernel state (suspected: `T.alloc_shared` lifetimes, `T.alloc_fragment` registers, or HIVM tensor.empty placeholder) isn't being properly initialized for grid blocks 1+. The MLIR `tensor.empty` ops at the top of each iteration may be reused across grid blocks without reset.

## Asks

1. Confirm reproduction with the linked kernel + smoke at seq_len ∈ {1, 2, 8}.
2. Identify whether `T.alloc_shared` and `T.alloc_fragment` are documented to be re-initialized between grid blocks on NPU, or if they're shared across blocks.
3. Provide a kernel-side mechanism (annotation or explicit init op) to force per-block reset.
