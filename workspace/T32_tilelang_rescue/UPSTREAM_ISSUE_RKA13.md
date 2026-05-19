# [BUG] tilelang `T.vsub(big_BHxBS, small_BHx1, big_BHxBS)` zeros result at certain pipeline positions

## Summary

`T.vsub(acc_dp, delta_frag, acc_dp)` where `acc_dp` is `[block_H, BS] = [16, 8]` (float32) and `delta_frag` is `[block_H, 1] = [16, 1]` (float32) **silently zeros out the result** at a specific position in a pipelined kernel — while the structurally identical pattern `T.vsub(acc_p, lse_frag, acc_p)` at an earlier position in the same iteration WORKS correctly.

## Environment

Same as R-KA-15 issue.

## Reproducer

The full sparse_mla_bwd kernel that triggers this is at <https://github.com/zhshgmail/tilelang-mlir-ascend/blob/t33-sparse-mla-fwd-port-and-tdynamic/examples/deepseek_v4/example_sparse_mla_bwd_kernel.py> commit `83f5ca2`.

Minimal kernel triggering only the dS computation chain hasn't been fully minimized yet — bisect on standalone 2-vsub patterns inside a pipelined block didn't reproduce (both vsubs worked). The bug requires the surrounding 7-GEMM context + acc_p chain.

## Diagnostic bisect chain

In sparse_mla_bwd kernel inner Pipelined loop:

```python
# Step 1: scores = Q @ K^T (split D + D_tail)
T.gemm(Q_shared, KV_shared, acc_p, initC=True, b_transpose=True)
T.gemm(Q_tail_shared, KV_tail_shared, acc_p, initC=False, b_transpose=True)
T.vbrc(sm_log2e, tmp_HB)
T.vmul(acc_p, tmp_HB, acc_p)
T.vsub(acc_p, lse_frag, acc_p)         # ← WORKS: acc_p [16,8] - lse_frag [16,1]
T.vexp(acc_p, acc_p)
T.vcast(acc_p, P_shared_cast, "rint")

# Step 2: dP = dO @ KV^T
T.gemm(dO_shared, KV_shared, acc_dp, initC=True, b_transpose=True)
T.vsub(acc_dp, delta_frag, acc_dp)     # ← ZEROS acc_dp; identical shape pattern as above
T.vmul(acc_dp, sm_scale_buf, acc_dp)
T.vmul(acc_dp, acc_p, acc_dp)
```

Tested workarounds, all FAIL to make the second vsub produce non-zero:
- Replace `delta_frag [BH,1]` with `delta_expanded [BH,BS]` (manual broadcast)
- Replace `T.vsub(...)` with `T.vmul(...)` against pre-negated delta + vadd
- Reorder: do `vmul(acc_dp, sm_scale_buf, acc_dp)` BEFORE the vsub

Tested controls, all WORK:
- Replace `T.vsub` with `T.vmul` (same operands): non-zero
- Replace `delta_frag` with `T.vbrc(0.0)` (broadcast zero buffer): non-zero (correct identity)

So `T.vsub(acc_dp, X, acc_dp)` works for some `X` but not for `delta_frag` at this pipeline position.

## Quantified bias when vsub is omitted

When we omit the buggy `vsub(acc_dp, delta_frag, acc_dp)`, the resulting dQ is non-zero but mathematically wrong (missing the dS=(dP-Δ)*s*P correction):

```
dQ kernel max abs:  0.0988
dQ ref    max abs:  0.0664
max abs err:        0.0988  (rel: 1.486x)
cosine similarity:  0.5255
```

This is **not training-quality** (cosine ~0.5 means ~half gradient direction is wrong). Confirms the vsub must be fixed for production training.

## Asks

1. Confirm the bisect chain reproduces (with the full kernel source on the fork branch).
2. Identify what differs in the bishengir lowering between the first vsub (works) and the second vsub (zeros). Suspects: (a) post-vexp register layout, (b) cube vs vector mode transition, (c) HIVM pipeline scheduling interaction with gemm operand reuse.
3. Fix or provide a kernel-side annotation to control the lowering.
