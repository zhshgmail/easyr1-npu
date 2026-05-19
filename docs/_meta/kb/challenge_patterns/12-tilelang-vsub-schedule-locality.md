---
id: 12
pattern: tilelang-vsub-schedule-locality
trigger_phrases:
  - "T.vsub on the output of a T.gemm in a pipelined kernel"
  - "broadcast buffer constructed once outside a pipelined loop, consumed inside"
  - "T.vbrc into a [BH, BS] buffer that feeds T.vsub after T.gemm"
  - "vsub silently zeros / produces no effect"
  - "all 4 obvious workarounds for T.vsub failed (direct, expanded, vadd-negated, reorder)"
user_source:
  - "2026-05-19: R-KA-13 originally filed as OPEN upstream bug with 4 failed workarounds. E5 worked when broadcast buf was constructed inside the same inner pipelined iter immediately before the vsub."
---

# tilelang T.vsub silent-zero on broadcast operand built outside the inner pipelined iter

## What the user / future-me is catching

Whenever a tilelang `T.vsub(acc_after_gemm, broadcast_buf, acc_after_gemm)`
is failing silently (output stays equal to the unchanged `acc_after_gemm`),
**before** filing it upstream as an OPEN bug, check the schedule-locality
of `broadcast_buf` construction. If the buffer was built via `T.vbrc(...)`,
or by a Python loop placed outside the inner pipelined iteration that
contains the vsub, the lowering may produce a no-op for the vsub.

## Why it matters

The bishengir lowering treats schedule-adjacency as register-layout
consumption order. A broadcast buffer constructed earlier in the kernel
(outside the pipelined iter) materialises with a register layout different
from what the post-gemm `acc_after_gemm` fragment has, and the vsub op
lowers to a layout-mismatched no-op rather than failing loudly.

This was originally diagnosed on 2026-05-18 in `sparse_mla_bwd`. Four
workarounds tried and FAILED:

1. `vsub(acc_dp, delta_frag, acc_dp)` direct (broadcast operand is `[BH,1]`).
2. `vsub(acc_dp, delta_expanded, acc_dp)` where `delta_expanded` was
   filled by `T.vbrc(0, ...)` + assignment earlier in the kernel.
3. Pre-negate delta + `T.vadd(acc_dp, neg_delta, acc_dp)`.
4. Reorder: `T.vmul(acc_dp, sm_scale_buf, acc_dp)` first, then `T.vsub`.

E5 (2026-05-19) WORKED:

```python
# Fill the broadcast buffer with a SCALAR Python loop INSIDE the same
# inner pipelined iter as the vsub, IMMEDIATELY before the vsub:
for h_i in T.serial(block_H):
    for bi_i in T.serial(BS):
        delta_expanded[h_i, bi_i] = delta_frag[h_i, 0]
T.vsub(acc_dp, delta_expanded, acc_dp)
```

This mirrors the construction of the working `lse_expanded` earlier in the
same kernel. Result on sparse_mla_bwd: dQ cosine 0.5255 → 0.9276 vs autograd.

## Self-check before filing upstream

Before declaring `T.vsub` silently-zeros as an OPEN upstream bug:

1. Find any OTHER `T.vsub` in the same kernel that works correctly. Note
   exactly HOW its broadcast operand is constructed: is the construction
   inside the inner pipelined iter? Is it scalar-loop filled or `T.vbrc`'d?
2. Replicate that working construction pattern for the failing vsub —
   move the broadcast-buf fill into the same inner pipelined iter,
   immediately before the failing vsub. Use the same Python `T.serial`
   loop shape.
3. If a working vsub does not exist in the kernel: write a 2-vsub
   experiment kernel to find one that works on the target backend, then
   match that construction pattern.

Only file upstream if even with the matched construction, the failing
vsub still zeros. (If you do file, include the side-by-side diff of
the working vs failing variants.)

## Generalises to

The schedule-locality sensitivity likely applies to other `T.vXxx` ops
that consume a broadcast operand following a `T.gemm`. Consider auditing
`T.vmul`, `T.vadd`, `T.vexp`-with-pre-broadcast when they show similar
silent-incorrect-output symptoms.

## Cost paid

Originally surfaced 2026-05-18 by sparse_mla_bwd having a 0.5 cosine
dQ vs autograd, blocking production use of the kernel. Took 1 full
session of bisecting + 4 failed workarounds before the E5 schedule-fix
was found 2026-05-19. ~6 hours of bisect time. With this pattern, the
next encounter should resolve in < 30 minutes.

## Related

* `workspace/T32_tilelang_rescue/KB_TILELANG_ASCEND.md` §12 R-KA-13
* `workspace/T32_tilelang_rescue/UPSTREAM_ISSUE_RKA13.md`
* commit `502c29f` on `t33-sparse-mla-fwd-port-and-tdynamic`
