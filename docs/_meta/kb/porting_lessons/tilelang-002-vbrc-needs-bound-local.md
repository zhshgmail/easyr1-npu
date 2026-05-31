---
id: tilelang-002
date: 2026-05-12
layer: tilelang
title: T.vbrc(0, buf) fails rank check on raw int literal — bind to local variable first
trigger:
  - "writing tilelang code that initializes a buffer with a constant"
  - "T.vbrc(0, buf) or T.vbrc(1.0, buf) producing a rank check error"
  - "tilelang compile error 'expected rank N got rank 0' at vbrc call site"
symptom_in_wild:
  - "tilelang compile fails with rank-mismatch error at T.vbrc usage"
  - "Same pattern T.vbrc(0, buf) works in CUDA tilelang but fails on NPU tilelang-mlir-ascend"
  - "Error message points at the literal 0 itself, not the buf"
root_cause: >
  T.vbrc is a vectorized broadcast intrinsic. Its first argument must be a
  rank-N scalar with shape info that matches the target buffer. When called
  with a Python int literal `0`, the rank inference resolves to rank-0
  (Python scalar), which mismatches buf's expected rank.

  CUDA tilelang's implementation does implicit promotion; NPU tilelang-mlir-ascend
  does strict rank checking. Easy fix: bind the literal to a typed local first.
mistake_pattern: "raw literal vs typed local — inference rules differ between backends"
correction:
  - "Always bind: `zero = T.constant(0, dtype=buf.dtype); T.vbrc(zero, buf)`"
  - "Same pattern for non-zero constants: `one = T.constant(1.0, dtype=buf.dtype); T.vbrc(one, buf)`"
  - "If buf is fp16: dtype must match — `T.constant(0, dtype='float16')`"
  - "If buf is bf16: same — `T.constant(0, dtype='bfloat16')`"
  - "Avoid mixing dtypes between the local and buf — implicit conversion is not guaranteed"
evidence:
  - "First hit 2026-05-12 in miles `_sparse_mla_fwd_kernel.py` initialization of `acc_l`"
  - "Memory: tilelang_vbrc_literal_trap.md"
  - "Same fix needed in `_lighting_indexer_bwd_kernel.py` 2026-05-14"
  - "tilelang-mlir-ascend PRs that touched vbrc: examine #80 commit history for examples"
---

# tilelang-002 — T.vbrc rank check needs bound local

## Why this matters

Small papercut, repeated cost. Every time a new tilelang kernel is written on NPU, this fires unless the author remembers. After 2-3 hits, write it down so the third author doesn't relearn.

## Standard idiom

```python
# Wrong — raw literal trips rank check
T.vbrc(0, my_buf)

# Right — bound local with matching dtype
zero = T.constant(0, dtype=my_buf.dtype)
T.vbrc(zero, my_buf)
```

## Why not fix upstream

Could request tilelang-mlir-ascend to do implicit promotion like CUDA tilelang. Counter-argument: strict rank checking catches real bugs in user code (genuinely wrong dtype mixing). The minor inconvenience of binding constants is worth keeping the strictness.

If a PR proposes promotion, it should be opt-in via a context flag, not the default.
