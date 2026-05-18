# Option A — compile-time guard for tilelang-ascend issue #996

**Date**: 2026-05-17
**Approach**: add `static_assert` to `src/tl_templates/ascend/common.h`
`copy_gm_to_ub` template, catching `dstN * sizeof(T) < 32` for 2-D copies at
compile time with a clear remediation message.
**Outcome**: VERIFIED — silent-wrong-output replaced with clear compile error;
no regression on baseline.

## The patch

```c++
template <typename T, uint32_t dstN, uint32_t dstM = 1>
CATLASS_DEVICE void
copy_gm_to_ub(LocalTensor<T> dstTensor, GlobalTensor<T> srcTensor,
              uint32_t realSrcN = 1, uint32_t maskShapeM = dstM,
              uint32_t maskShapeN = dstN, T padValue = T(0)) {
+  static_assert(dstM == 1 || dstN * sizeof(T) >= 32,
+      "tilelang-ascend issue #996: 2-D GM->UB copy requires "
+      "dstN * sizeof(T) >= 32 bytes per row (DMA alignment). "
+      "Reshape your kernel to 1-D layout (T.alloc_ub((N,), ...) with "
+      "M*N flattened) or increase block_N so per-row bytes >= 32. "
+      "Example: fp32 needs dstN >= 8, fp16 needs dstN >= 16.");
   // ... rest unchanged
```

Predicate `dstM == 1 || dstN * sizeof(T) >= 32`:
- `dstM == 1` short-circuits — 1-D copies are inherently fine (no per-row stride)
- `dstN * sizeof(T) >= 32` enforces the AscendC DMA's blockLen ≥ 32B requirement

## Verification matrix

All cases run inside `tlrescue` container (verl-8.5.2 image, chip 14) on
A3 host, with patched header copied to installed location at
`/usr/local/python3.11.14/lib/python3.11/site-packages/tilelang/src/tl_templates/ascend/common.h`.

| Case | dstM | dstN | bytes/row | Expected | Actual |
|------|------|------|-----------|----------|--------|
| Upstream baseline `--m 1024 --n 1024 --block_m 128 --block_n 256` | 64 (=128/2) | 256 | 1024 B | PASS no error | ✓ Kernel Output Match! |
| Issue #996 `--m 32 --n 32 --block-m 4 --block-n 4` | 2 | 4 | 16 B | static_assert fires | ✓ error w/ #996 message |
| Boundary OK `--m 64 --n 64 --block-m 8 --block-n 16` | 4 | 16 | 64 B | PASS no error | ✓ Match at (64,64)/(8,16) |
| Boundary bug `--m 64 --n 64 --block-m 8 --block-n 4` | 4 | 4 | 16 B | static_assert fires | ✓ error w/ #996 message |
| 1-D rescue `elementwise_add_flat.py --m 32 --n 32` | 1 | 1024 (alloc_ub size) | n/a 1-D | PASS no error | ✓ Kernel Output Match! |

## What the user sees on the bug case (after patch)

```
$ python3 repro_issue_996.py --m 32 --n 32 --block-m 4 --block-n 4
...
/usr/local/.../tilelang/src/tl_templates/ascend/common.h:201:3: error:
  static assertion failed due to requirement '2U == 1 || 4U * sizeof(float) >= 32':
  tilelang-ascend issue #996: 2-D GM->UB copy requires
  dstN * sizeof(T) >= 32 bytes per row (DMA alignment). Reshape your kernel
  to 1-D layout (T.alloc_ub((N,), ...) with M*N flattened) or increase
  block_N so per-row bytes >= 32. Example: fp32 needs dstN >= 8,
  fp16 needs dstN >= 16.
```

(vs. the pre-patch behavior: silent 49.7% wrong output with no warning)

## What this is NOT

- **Not** a full fix — small-block configs that the user might want are now
  rejected rather than supported.
- **Option B / auto-coalesce** (which would emit a 1-D copy automatically
  when per-row bytes < 32, equivalent to our manual `elementwise_add_flat.py`)
  remains the next milestone.

## Upstream PR pieces

This patch is suitable for upstream PR to tile-ai/tilelang-ascend (`ascendc_pto` branch):

- Diff: `workspace/T32_tilelang_rescue/issue_996_compile_check.patch`
- Test fixtures: `repro_issue_996.py` (shows the bug it catches),
  `elementwise_add_flat.py` (shows the documented workaround)
- Result: this REPORT.md

## Files touched

- `src/tl_templates/ascend/common.h` (in tilelang-ascend repo on A3, +9 lines)

## Files NOT touched

- No TIR pass changes (avoided the more invasive C++ pass rewrite)
- No build system changes
- No Python frontend changes
